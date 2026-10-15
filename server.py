"""Web front door: serves the UI and mints LiveKit tokens that dispatch the agent."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

from aiohttp import web
from dotenv import load_dotenv
from livekit import api

load_dotenv()

STATIC = Path(__file__).parent / "web"
AGENT_NAME = os.environ.get("AGENT_NAME", "ryan-simulation-agent")
# Gate token minting on a public deploy so a stranger can't run up STT/LLM/TTS spend.
PASSCODE = os.environ.get("APP_PASSCODE", "")


async def index(_: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC / "index.html")


async def config(_: web.Request) -> web.Response:
    return web.json_response({"passcodeRequired": bool(PASSCODE)})


async def token(request: web.Request) -> web.Response:
    body = await request.json() if request.can_read_body else {}
    if PASSCODE and not secrets.compare_digest(str(body.get("passcode", "")), PASSCODE):
        return web.json_response({"error": "Wrong passcode"}, status=401)

    company = str(body.get("company", "")).strip()[:120]
    website = str(body.get("website", "")).strip()[:200]
    room = f"discovery-{secrets.token_hex(4)}"
    identity = f"rep-{secrets.token_hex(3)}"

    jwt = (
        api.AccessToken()  # reads LIVEKIT_API_KEY / LIVEKIT_API_SECRET
        .with_identity(identity)
        .with_name("Rep")
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .with_room_config(
            api.RoomConfiguration(
                agents=[
                    api.RoomAgentDispatch(
                        agent_name=AGENT_NAME,
                        metadata=json.dumps({"company": company, "website": website}),
                    )
                ]
            )
        )
        .to_jwt()
    )
    return web.json_response({"serverUrl": os.environ["LIVEKIT_URL"], "participantToken": jwt})


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/api/config", config)
    app.router.add_post("/api/token", token)
    app.router.add_get("/healthz", lambda _: web.Response(text="ok"))
    return app


if __name__ == "__main__":
    web.run_app(make_app(), port=int(os.environ.get("PORT", "8080")))
