"""EasyCater LLM API — FastAPI entrypoint.

Endpoints:
  POST /v1/chat                     — SSE streaming chat
  POST /v1/chat/sync                — non-streaming variant (debug/tests)
  GET  /v1/health                   — liveness + Redis ping
  POST /v1/session/{sid}/reset      — clear conversation history
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agent.app.config import get_settings
from agent.app.graph import run_turn
from agent.app.graph.streaming import stream_turn
from agent.app.guardrails_middleware import AkiGuardrails
from agent.app.logging_setup import configure_logging, get_logger
from agent.app.rag import get_retriever
from agent.app.state import get_redis, load_history, reset_history, save_history
from agent.app.state.history import load_order_params, save_order_params

configure_logging()
log = get_logger(__name__)

guardrails = AkiGuardrails()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    # Warm up retriever (loads bge-m3 ~2GB into RAM)
    log.info("startup_begin")
    get_retriever()
    # Initialize guardrails
    await guardrails.initialize()
    # Ping Redis
    r = await get_redis()
    await r.ping()
    log.info("startup_ready")
    yield
    log.info("shutdown")


app = FastAPI(title="EasyCater LLM API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────────────────────────────────
# Schemas
# ────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=4000)
    user_id: Optional[str] = None


class ChatSyncResponse(BaseModel):
    session_id: str
    text: str


# ────────────────────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────────────────────

@app.get("/v1/health")
async def health():
    try:
        r = await get_redis()
        await r.ping()
        redis_ok = True
    except Exception as e:  # noqa: BLE001
        redis_ok = False
        log.error("redis_health_failed", extra={"err": str(e)})
    return {
        "ok": redis_ok,
        "redis": redis_ok,
        "model": get_settings().vllm_model,
        "guardrails": guardrails._initialized,
    }


@app.post("/v1/chat/sync", response_model=ChatSyncResponse)
async def chat_sync(req: ChatRequest):
    """Non-streaming chat — useful for tests and server-to-server calls."""

    # ── Guardrail input check ──────────────────────────────────
    guard = await guardrails.check_input(req.message, req.session_id)
    if guard["blocked"]:
        log.info(
            "guardrail_blocked",
            extra={"session_id": req.session_id, "violation": guard["violation"]},
        )
        return ChatSyncResponse(session_id=req.session_id, text=guard["response"])

    # ── Normal pipeline ────────────────────────────────────────
    history = await load_history(req.session_id)
    order_params = await load_order_params(req.session_id)

    text, new_history, updated_order_params = await run_turn(
        req.session_id, req.message, history, order_params
    )

    await save_history(
        req.session_id,
        history + [{"role": "user", "content": req.message}] + _without_user_dup(new_history, req.message),
    )
    await save_order_params(req.session_id, updated_order_params)

    return ChatSyncResponse(session_id=req.session_id, text=text)


@app.post("/v1/chat")
async def chat_stream(req: ChatRequest):
    """SSE streaming chat. Events:
       status, tool, token, done, error — see app/graph/streaming.py
    """
    history = await load_history(req.session_id)
    order_params = await load_order_params(req.session_id)

    async def event_gen():
        # ── Guardrail input check ──────────────────────────────
        guard = await guardrails.check_input(req.message, req.session_id)
        if guard["blocked"]:
            log.info(
                "guardrail_blocked",
                extra={"session_id": req.session_id, "violation": guard["violation"]},
            )
            yield {"event": "token", "data": json.dumps({"text": guard["response"]})}
            yield {"event": "done",  "data": json.dumps({"ok": True})}
            return

        # ── Normal pipeline ────────────────────────────────────
        persistable: list[dict] | None = None
        updated_params: dict = order_params
        try:
            async for event in stream_turn(
                req.session_id, req.message, history, order_params
            ):
                if event["type"] == "done":
                    persistable = event["messages"]
                    updated_params = event.get("order_params", {})
                    yield {"event": "done", "data": json.dumps({"ok": True})}
                else:
                    yield {"event": event["type"], "data": json.dumps(event)}

            # Persist after the stream finishes successfully
            if persistable is not None:
                new_hist = history + persistable
                await save_history(req.session_id, new_hist)
                await save_order_params(req.session_id, updated_params)

        except asyncio.CancelledError:
            log.info("client_disconnect", extra={"session_id": req.session_id})
            raise

    return EventSourceResponse(event_gen())


@app.post("/v1/session/{session_id}/reset")
async def reset_session(session_id: str):
    await reset_history(session_id)
    await save_order_params(session_id, {})  # clear order params too
    return {"ok": True, "session_id": session_id}


def _without_user_dup(new_msgs: list[dict], user_msg: str) -> list[dict]:
    """Strip a leading user message from new_msgs if it duplicates user_msg.

    `run_turn` returns messages that include the user message we just added,
    because we built initial_messages with it. We avoid double-storing.
    """
    if new_msgs and new_msgs[0].get("role") == "user" and new_msgs[0].get("content") == user_msg:
        return new_msgs[1:]
    return new_msgs