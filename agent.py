"""Discovery-call roleplay agent, spoken in Ryan's cloned voice.

Flow:
  1. Coach greets the rep and asks for a company name + website.
  2. We scrape the site, generate a buyer persona, and hand off to that buyer.
  3. The rep runs a discovery call selling AssemblyAI to the buyer.
  4. When the rep says the call is over, the coach scores the call, reads the
     highlights aloud, and writes the full scorecard to scorecards/.

Stack: AssemblyAI Universal-3.6 Pro (STT) -> GPT-4.1 (LLM) -> AssemblyAI TTS
(sandbox000 checkpoint, cloned voice) via the forked LiveKit TTS plugin.
The post-call scorecard uses Claude, where latency doesn't matter.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

import aiohttp
import anthropic as anthropic_sdk
import openai as openai_sdk
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.plugins import assemblyai, openai, silero

from assemblyai_tts import TTS as AssemblyAITTS

load_dotenv()
logger = logging.getLogger("ryan-simulation-agent")

LIVE_LLM_MODEL = os.environ.get("LIVE_LLM_MODEL", "gpt-4.1")
SCORECARD_LLM_MODEL = os.environ.get("SCORECARD_LLM_MODEL", "claude-opus-5")
SCORECARD_DIR = Path(__file__).parent / "scorecards"

def build_tts() -> AssemblyAITTS:
    # Prefer a long-lived API key sent in the Authorization header, as the TTS
    # docs specify. A `token=` copied out of a browser URL is a short-lived
    # temporary token, so it's only a fallback for quick tests.
    if key := os.environ.get("ASSEMBLYAI_TTS_API_KEY"):
        api_key, auth = key, "header"
    elif token := os.environ.get("ASSEMBLYAI_TTS_TOKEN"):
        api_key, auth = token, "query"
    else:
        raise RuntimeError("Set ASSEMBLYAI_TTS_API_KEY in .env (an API key that sandbox000 accepts)")
    return AssemblyAITTS(
        base_url=os.environ["ASSEMBLYAI_TTS_URL"],
        voice_id=os.environ["ASSEMBLYAI_TTS_VOICE_ID"],
        model_id=os.environ["ASSEMBLYAI_TTS_MODEL_ID"],
        language="english",
        sample_rate=24000,
        word_timestamps=True,
        api_key=api_key,
        auth=auth,
    )


# --------------------------------------------------------------------------
# Website research + persona generation
# --------------------------------------------------------------------------


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "head"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self.description = ""
        self._depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            a = dict(attrs)
            if a.get("name", "").lower() == "description" or a.get("property") == "og:description":
                self.description = self.description or (a.get("content") or "")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._depth:
            self._depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._depth and data.strip():
            self.parts.append(data.strip())


def normalize_url(website: str) -> str:
    site = website.strip().lower().replace(" dot ", ".").replace(" ", "")
    return site if site.startswith("http") else f"https://{site}"


async def fetch_site_summary(website: str, max_chars: int = 6000) -> str:
    url = normalize_url(website)
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=8),
            headers={"User-Agent": "Mozilla/5.0 (discovery-roleplay research)"},
        ) as http:
            async with http.get(url, allow_redirects=True) as resp:
                html = await resp.text(errors="ignore")
    except Exception as e:
        logger.warning("could not fetch %s: %s", url, e)
        return f"(Could not load {url}; rely on general knowledge of the company.)"

    parser = _TextExtractor()
    parser.feed(html)
    body = re.sub(r"\s+", " ", " ".join(parser.parts))[:max_chars]
    return f"URL: {url}\nTitle: {parser.title.strip()}\nDescription: {parser.description}\nPage text: {body}"


PERSONA_PROMPT = """You design realistic buyer personas for sales discovery-call practice. The rep works at AssemblyAI (speech-to-text, streaming STT for voice agents, speech understanding, TTS, LLM Gateway) and is running a first discovery call.

Using the company research below, write a private persona brief for the buyer the rep is talking to. Make it specific and plausible for this company. Include:
- Name, title, and how long they've been at the company
- What the company does, in the buyer's own words
- The speech/voice use case they're evaluating (pick the most plausible one for this business), current solution or vendor, and rough volumes (hours/month, concurrent streams, languages)
- 2-3 real pains, with the business impact behind each (only revealed if the rep digs)
- Technical requirements (accuracy, latency, diarization, PII/compliance, deployment region, integrations)
- Decision process: who else is involved, who signs, budget situation, timeline, and what triggered this evaluation now
- Competitors they're also looking at, and 2 objections they will raise
- Personality: how guarded or chatty they are, and one quirk that makes them feel human
- "Hidden" facts the buyer only shares if asked a good, specific question

Return JSON with these keys:
- "brief": the full persona brief above as plain prose and short bullet lists, under 450 words
- "buyer_name": the buyer's full name
- "buyer_title": the buyer's title
- "company_summary": one sentence on what the company does
- "competitors": the competing vendors the buyer is evaluating, as short names (e.g. ["Deepgram", "Google STT"])"""


async def generate_persona(company_name: str, website: str) -> dict:
    research = await fetch_site_summary(website)
    client = openai_sdk.AsyncOpenAI()
    resp = await client.chat.completions.create(
        model=LIVE_LLM_MODEL,
        max_tokens=1500,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PERSONA_PROMPT},
            {
                "role": "user",
                "content": f"Company: {company_name}\nWebsite: {website}\n\nResearch:\n{research}",
            },
        ],
    )
    persona = json.loads(resp.choices[0].message.content or "{}")
    persona.setdefault("brief", "")
    persona.setdefault("competitors", [])
    return persona


# --------------------------------------------------------------------------
# Web UI state (participant attributes + text streams the browser listens to)
# --------------------------------------------------------------------------


async def publish_ui(**fields: object) -> None:
    """Set `app.<field>` attributes on the agent participant; no-op without a room."""
    ctx = agents.get_job_context(required=False)
    if ctx is None or not ctx.room.isconnected():
        return
    attrs = {f"app.{k}": v if isinstance(v, str) else json.dumps(v) for k, v in fields.items()}
    try:
        await ctx.room.local_participant.set_attributes(attrs)
    except Exception:
        logger.warning("failed to publish UI state", exc_info=True)


async def publish_scorecard(card: dict) -> None:
    # Too big for attributes, so it goes out as a text stream.
    ctx = agents.get_job_context(required=False)
    if ctx is None or not ctx.room.isconnected():
        return
    try:
        await ctx.room.local_participant.send_text(json.dumps(card), topic="app.scorecard", compress=False)
    except Exception:
        logger.warning("failed to publish scorecard", exc_info=True)


async def research_buyer(company_name: str, website: str) -> BuyerAgent:
    await publish_ui(phase="researching", buyer="")
    logger.info("starting roleplay: %s (%s)", company_name, website)
    persona = await generate_persona(company_name, website)
    logger.info("persona:\n%s", persona.get("brief"))
    return BuyerAgent(company_name, website, persona)


# --------------------------------------------------------------------------
# Scorecard
# --------------------------------------------------------------------------

RUBRIC = [
    ("opening", "Opening & agenda: rapport, purpose, agenda and time check, permission to ask questions"),
    ("current_state", "Current state: use case, current vendor/stack, volumes, languages, audio types, streaming vs async"),
    ("pain_impact", "Pain & impact: uncovered real pains and quantified their business impact"),
    ("technical_requirements", "Technical requirements: accuracy, latency, diarization, compliance/PII, region, integrations"),
    ("decision_process", "Decision process (MEDDPICC): metrics, economic buyer, criteria, process, paper process, champion, competition"),
    ("timeline_budget", "Timeline, budget, and the trigger/why-now"),
    ("value_positioning", "Positioning AssemblyAI: tied capabilities to stated pains instead of feature-dumping"),
    ("listening", "Active listening: open questions, follow-ups, summarizing back, letting the buyer talk"),
    ("next_steps", "Next steps: concrete, dated, mutually agreed (e.g. technical eval, POC criteria, stakeholders)"),
]

SCORECARD_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_score": {"type": "integer", "description": "0-100"},
        "headline": {"type": "string", "description": "One-sentence verdict"},
        "categories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "enum": [k for k, _ in RUBRIC]},
                    "score": {"type": "integer", "description": "1-5"},
                    "evidence": {"type": "string", "description": "What the rep did, quoting them where possible"},
                    "coaching": {"type": "string", "description": "Specific thing to do differently"},
                },
                "required": ["key", "score", "evidence", "coaching"],
                "additionalProperties": False,
            },
        },
        "strengths": {"type": "array", "items": {"type": "string"}},
        "improvements": {"type": "array", "items": {"type": "string"}},
        "missed_discovery": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Hidden persona facts the rep never uncovered, phrased as the question they should have asked",
        },
        "spoken_summary": {
            "type": "string",
            "description": "What the coach says aloud: under 90 words, plain conversational speech, no lists or markdown",
        },
    },
    "required": [
        "overall_score", "headline", "categories", "strengths",
        "improvements", "missed_discovery", "spoken_summary",
    ],
    "additionalProperties": False,
}


def talk_ratio(transcript: list[tuple[str, str]]) -> tuple[int, int, float]:
    rep = sum(len(t.split()) for role, t in transcript if role == "rep")
    buyer = sum(len(t.split()) for role, t in transcript if role == "buyer")
    return rep, buyer, (rep / (rep + buyer) if rep + buyer else 0.0)


async def generate_scorecard(company: str, persona: str, transcript: list[tuple[str, str]]) -> dict:
    rep_words, buyer_words, ratio = talk_ratio(transcript)
    convo = "\n".join(f"{'REP' if r == 'rep' else 'BUYER'}: {t}" for r, t in transcript)
    rubric = "\n".join(f"- {k}: {d}" for k, d in RUBRIC)
    client = anthropic_sdk.AsyncAnthropic()
    async with client.messages.stream(
        model=SCORECARD_LLM_MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCORECARD_SCHEMA}},
        system=(
            "You are a demanding but fair sales coach at AssemblyAI scoring a practice discovery call. "
            "The REP is the AssemblyAI seller being evaluated; the BUYER was played by an AI using the persona below. "
            "Score each rubric category 1-5 (3 = acceptable for a first call). Ground every score in the transcript. "
            "Use the persona's hidden facts to judge what discovery the rep missed. A healthy discovery call has the rep "
            "talking roughly 30-45% of the time."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Company: {company}\n\nBuyer persona (hidden from the rep):\n{persona}\n\n"
                f"Rubric:\n{rubric}\n\n"
                f"Talk ratio: rep {rep_words} words, buyer {buyer_words} words, rep share {ratio:.0%}\n\n"
                f"Transcript:\n{convo}"
            ),
        }],
    ) as stream:
        msg = await stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError("scorecard model declined the request")
    text = next(b.text for b in msg.content if b.type == "text")
    card = json.loads(text)
    card["talk_ratio"] = {"rep_words": rep_words, "buyer_words": buyer_words, "rep_share": round(ratio, 3)}
    return card


def write_scorecard(company: str, persona: str, transcript: list[tuple[str, str]], card: dict) -> Path:
    SCORECARD_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", company.lower()).strip("-") or "call"
    path = SCORECARD_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.md"
    labels = dict(RUBRIC)
    tr = card["talk_ratio"]
    lines = [
        f"# Discovery scorecard: {company}",
        "",
        f"**Overall: {card['overall_score']}/100**: {card['headline']}",
        "",
        f"Talk ratio: rep {tr['rep_share']:.0%} ({tr['rep_words']} words) / buyer {tr['buyer_words']} words",
        "",
        "| Category | Score | Evidence | Coaching |",
        "|---|---|---|---|",
    ]
    for c in card["categories"]:
        name = labels.get(c["key"], c["key"]).split(":")[0]
        esc = lambda s: s.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {name} | {c['score']}/5 | {esc(c['evidence'])} | {esc(c['coaching'])} |")
    for title, key in (("Strengths", "strengths"), ("Improvements", "improvements"), ("Missed discovery", "missed_discovery")):
        lines += ["", f"## {title}", *[f"- {x}" for x in card[key]]]
    lines += ["", "## Buyer persona", "", persona, "", "## Transcript", ""]
    lines += [f"**{'Rep' if r == 'rep' else 'Buyer'}:** {t}  " for r, t in transcript]
    path.write_text("\n".join(lines))
    (path.with_suffix(".json")).write_text(json.dumps({"company": company, "scorecard": card, "transcript": transcript}, indent=2))
    return path


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------

SPEECH_RULES = (
    "Everything you write is spoken aloud by a text-to-speech voice. Use plain conversational sentences: "
    "no markdown, bullets, emojis, or stage directions. Spell out anything awkward to read aloud."
)


class StartsRoleplay:
    """Mixin: the coach can kick off a roleplay from the greeting or after a scorecard."""

    @function_tool()
    async def start_roleplay(self, context: RunContext, company_name: str, website: str):
        """Research a company and switch into character as a buyer from it.

        Call this once the rep has given you both a company name and its website.

        Args:
            company_name: The company the buyer works at.
            website: The company's website domain, e.g. "acme.com". Convert spoken forms like "acme dot com".
        """
        return await research_buyer(company_name, website), None


class CoachAgent(StartsRoleplay, Agent):
    def __init__(self, company: str = "", website: str = "") -> None:
        # Set when the rep typed the company into the web form, skipping the spoken setup.
        self.prefill = (company, website) if company and website else None
        super().__init__(
            instructions=(
                "You are Ryan Seams's practice partner for AssemblyAI sales discovery calls, speaking in Ryan's voice. "
                "Your job right now is to collect a company name and website from the rep, then call start_roleplay. "
                "If they only give one of the two, ask for the other. If the website is ambiguous, confirm it in a few words. "
                "Right before calling start_roleplay, say one short line like 'Got it, give me a few seconds to get into character.' "
                "Keep every reply to one or two sentences. " + SPEECH_RULES
            )
        )

    async def on_enter(self) -> None:
        if self.prefill:
            company, website = self.prefill
            await publish_ui(phase="researching")
            self.session.say(f"Hey, it's Ryan. Give me a few seconds to get into character as a buyer at {company}.")
            self.session.update_agent(await research_buyer(company, website))
            return
        await publish_ui(phase="setup", buyer="")
        await self.session.say(
            "Hey, it's Ryan. Let's do a discovery call roleplay. "
            "Give me a company name and their website, and I'll play a buyer from there. "
            "When you're done with the call, just tell me the call's over and I'll score you."
        )


class BuyerAgent(Agent):
    def __init__(self, company: str, website: str, profile: dict) -> None:
        self.company = company
        self.website = website
        self.profile = profile
        self.persona = profile["brief"]
        persona = self.persona
        super().__init__(
            instructions=(
                f"You are role-playing a buyer at {company} on a first discovery call with an AssemblyAI account executive "
                "(the user). This is sales training: be a realistic, moderately busy buyer, not a pushover.\n\n"
                "How to behave:\n"
                "- Stay fully in character. Never mention that you are an AI, a roleplay, or a persona.\n"
                "- Keep replies short and natural: usually one to three sentences, like a real person on a call.\n"
                "- Answer what was asked. Don't volunteer pains, budget, decision makers, or hidden facts unless the rep asks a "
                "good, specific question or earns it with a smart follow-up. Vague questions get vague answers.\n"
                "- Ask the rep a question or raise an objection now and then, the way a real buyer would.\n"
                "- If the rep pitches features before understanding your needs, get a bit impatient.\n"
                "- Don't invent AssemblyAI product facts; ask the rep instead.\n\n"
                "Call end_roleplay when the rep says the call or roleplay is over, asks to stop, or asks for their score "
                "or feedback. Don't call it just because the rep says goodbye in character; in that case say a natural goodbye "
                "and wait.\n\n"
                f"{SPEECH_RULES}\n\nYour persona (private, never read it out):\n{persona}"
            )
        )

    async def on_enter(self) -> None:
        await publish_ui(phase="roleplay", buyer={
            "company": self.company,
            "website": normalize_url(self.website),
            "name": self.profile.get("buyer_name", ""),
            "title": self.profile.get("buyer_title", ""),
            "summary": self.profile.get("company_summary", ""),
            "competitors": self.profile.get("competitors", []),
        })
        # Hand the floor to the rep as a buyer joining the call.
        self.session.generate_reply(
            instructions="You've just joined the call. Greet the rep briefly in character, e.g. 'Hey, sorry, just jumped off another call. How's it going?'"
        )

    def transcript(self) -> list[tuple[str, str]]:
        turns: list[tuple[str, str]] = []
        for item in self.chat_ctx.items:
            if getattr(item, "type", None) != "message" or item.role not in ("user", "assistant"):
                continue
            text = (item.text_content or "").strip()
            if text:
                turns.append(("rep" if item.role == "user" else "buyer", text))
        return turns

    @function_tool()
    async def end_roleplay(self, context: RunContext):
        """End the roleplay and move to scoring. Use only when the rep says the call is over or asks for feedback."""
        return ScorecardAgent(self.company, self.persona, self.transcript()), None


class ScorecardAgent(StartsRoleplay, Agent):
    def __init__(self, company: str, persona: str, transcript: list[tuple[str, str]]) -> None:
        self.company = company
        self.persona = persona
        self.transcript = transcript
        super().__init__(
            instructions=(
                "You are Ryan Seams's sales coach, speaking in Ryan's voice, debriefing a practice discovery call. "
                "You are scoring the call now; answer follow-up questions about it briefly and specifically. "
                "If the rep wants another round, ask for a company name and website and call start_roleplay. "
                + SPEECH_RULES
            )
        )

    async def on_enter(self) -> None:
        if not any(r == "rep" for r, _ in self.transcript):
            await publish_ui(phase="setup")
            await self.session.say("We didn't really get a call in there, so there's nothing to score yet. Want to try another company?")
            return

        await publish_ui(phase="scoring")
        await self.session.say("Alright, stepping out of character. Give me about thirty seconds to score that.")
        try:
            card = await generate_scorecard(self.company, self.persona, self.transcript)
        except Exception:
            logger.exception("scorecard generation failed")
            await self.session.say("Sorry, something broke while I was scoring that. The transcript is in the logs.")
            return

        path = await asyncio.to_thread(write_scorecard, self.company, self.persona, self.transcript, card)
        logger.info("scorecard written to %s", path)
        await publish_scorecard({**card, "categories": [
            {**c, "label": dict(RUBRIC).get(c["key"], c["key"]).split(":")[0]} for c in card["categories"]
        ]})
        await publish_ui(phase="debrief")

        # Let follow-up questions draw on the full scorecard.
        await self.update_instructions(
            self.instructions
            + f"\n\nThe scorecard you produced (JSON):\n{json.dumps(card)}\n\nThe hidden buyer persona:\n{self.persona}"
        )
        await self.session.say(
            f"You scored {card['overall_score']} out of 100. {card['spoken_summary']} "
            "The full scorecard is on your screen. Ask me about any part of it, or give me another company to go again."
        )


async def entrypoint(ctx: agents.JobContext) -> None:
    await ctx.connect()

    session = AgentSession(
        stt=assemblyai.STT(
            model="universal-3-6-pro",
            mode="balanced",
            vad_threshold=0.2,
            keyterms_prompt=[
                "AssemblyAI", "Universal-3.6 Pro", "Universal", "LiveKit", "diarization",
                "speaker labels", "PII redaction", "HIPAA", "SOC 2", "LLM Gateway", "MEDDPICC", "POC",
            ],
        ),
        llm=openai.LLM(model=LIVE_LLM_MODEL),
        tts=build_tts(),
        vad=silero.VAD.load(activation_threshold=0.2),
        turn_handling={"turn_detection": "stt", "endpointing": {"min_delay": 0}},
        # Send each agent reply to U3.6 Pro as `agent_context` so the rep's next
        # turn is transcribed knowing what was just asked. With TTS word
        # boundaries on, an interrupted reply forwards only the words actually heard.
        stt_context_options={"forward_chat_context": True},
    )

    # The web app passes {"company", "website"} as dispatch metadata when the form is filled.
    try:
        meta = json.loads(ctx.job.metadata or "{}")
    except json.JSONDecodeError:
        meta = {}

    async def end_call(data) -> str:
        # The "End call & score" button: same as the rep saying the call's over.
        current = session.current_agent
        if not isinstance(current, BuyerAgent):
            return "not-in-call"
        session.interrupt()
        session.update_agent(ScorecardAgent(current.company, current.persona, current.transcript()))
        return "ok"

    ctx.room.local_participant.register_rpc_method("end_call", end_call)

    await session.start(
        room=ctx.room,
        agent=CoachAgent(meta.get("company", "").strip(), meta.get("website", "").strip()),
    )


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            # Explicit dispatch: the web app's token asks for this agent by name, so
            # other agents on the same LiveKit project don't join these rooms.
            agent_name=os.environ.get("AGENT_NAME", "ryan-simulation-agent"),
        )
    )
