# Ryan simulation agent

Voice roleplay for AssemblyAI discovery calls. Give it a company and website; it plays a
buyer from that company in Ryan's cloned voice, then scores the rep when the call ends.

- **STT:** AssemblyAI Universal-3.6 Pro (production)
- **LLM:** GPT-4.1 for the live call, Claude Opus 5 for the scorecard
- **TTS:** AssemblyAI streaming TTS (sandbox000 checkpoint, cloned voice) via
  [ryanseams/assemblyai-livekit-tts@sandbox-custom-voice](https://github.com/ryanseams/assemblyai-livekit-tts/tree/sandbox-custom-voice)
- **Transport:** LiveKit Cloud

`server.py` serves the web UI (`web/index.html`) and mints LiveKit tokens that dispatch the
agent by name; `agent.py` is the LiveKit worker. `start.sh` runs both (used by the Dockerfile).

## Run locally

```
uv sync
uv run python server.py            # http://localhost:8080
uv run python agent.py dev         # in a second terminal
uv run python agent.py console     # or: talk in the terminal, no browser
```

## Environment

| Variable | Purpose |
|---|---|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | LiveKit Cloud project |
| `ASSEMBLYAI_API_KEY` | Production STT |
| `ASSEMBLYAI_TTS_API_KEY` | Key accepted by the TTS sandbox |
| `ASSEMBLYAI_TTS_URL`, `ASSEMBLYAI_TTS_VOICE_ID`, `ASSEMBLYAI_TTS_MODEL_ID` | TTS endpoint, cloned voice, checkpoint |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | Live LLM and scorecard LLM |
| `LIVE_LLM_MODEL`, `SCORECARD_LLM_MODEL` | Optional model overrides |
| `AGENT_NAME` | Dispatch name (default `ryan-simulation-agent`) |
