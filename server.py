"""
aiohttp web server: REST API + SSE for live frames + static UI.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

from aiohttp import web

import db
from event_tracker import EventTracker
from simulator import run_simulator
from file_player import run_file_player

logger = logging.getLogger(__name__)

PORT = int(os.environ.get("NOISEAIR_PORT", 8100))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# SSE subscribers
_sse_clients: list[web.StreamResponse] = []
_last_frame: dict = {}


async def _broadcast(event_type: str, payload: dict):
    dead = []
    for resp in _sse_clients:
        try:
            await resp.write(
                f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()
            )
        except Exception:
            dead.append(resp)
    for d in dead:
        _sse_clients.remove(d)


async def _broadcast_frame(frame: dict):
    global _last_frame
    _last_frame = frame
    await _broadcast("frame", frame)


async def broadcast_new_event(event: dict):
    await _broadcast("new_event", event)


# ── Routes ────────────────────────────────────────────────────────────────────

async def handle_live(request: web.Request) -> web.StreamResponse:
    """SSE endpoint: streams live frames to browser."""
    resp = web.StreamResponse(headers={
        "Content-Type":  "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
    await resp.prepare(request)
    _sse_clients.append(resp)
    # Send last known frame immediately
    if _last_frame:
        await resp.write(
            f"event: frame\ndata: {json.dumps(_last_frame)}\n\n".encode()
        )
    try:
        while True:
            await asyncio.sleep(30)
            await resp.write(b": keepalive\n\n")
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    finally:
        if resp in _sse_clients:
            _sse_clients.remove(resp)
    return resp


async def handle_events(request: web.Request) -> web.Response:
    """GET /api/events?date_from=&date_to=&icao=&limit="""
    p = request.rel_url.query
    events = db.get_events(
        date_from=p.get("date_from"),
        date_to=p.get("date_to"),
        icao=p.get("icao"),
        limit=int(p.get("limit", 200)),
    )
    return web.json_response(events)


async def handle_stats(request: web.Request) -> web.Response:
    return web.json_response(db.get_stats())


async def handle_index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ── App factory ───────────────────────────────────────────────────────────────

async def create_app() -> web.Application:
    db.init_db()

    tracker = EventTracker(on_event=broadcast_new_event)

    async def on_frame(frame: dict):
        tracker.push_frame(frame)
        await _broadcast_frame(frame)

    async def _audio_state(label: str, volume: float = 1.0):
        await _broadcast("audio_state", {"label": label, "volume": volume})

    # Source: "files" = real WAV files, "sim" = synthetic, "mic" = detector.py
    source = os.environ.get("NOISEAIR_SOURCE", "files")

    async def _run_source(app):
        if source == "files":
            task = asyncio.create_task(run_file_player(on_frame, _audio_state))
        else:
            audio_out = os.environ.get("NOISEAIR_AUDIO", "0") == "1"
            task = asyncio.create_task(run_simulator(on_frame, audio=audio_out))
        yield
        task.cancel()
        tracker.flush()

    app = web.Application()
    app.cleanup_ctx.append(_run_source)

    app.router.add_get("/",            handle_index)
    app.router.add_get("/api/live",    handle_live)
    app.router.add_get("/api/events",  handle_events)
    app.router.add_get("/api/stats",   handle_stats)
    app.router.add_static("/static",   STATIC_DIR)
    app.router.add_static("/audio",    os.path.join(os.path.dirname(__file__), "audio"))

    return app


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    web.run_app(create_app(), port=PORT)
