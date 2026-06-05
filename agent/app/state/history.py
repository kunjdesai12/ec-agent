"""Valkey-backed (or in-memory) conversation history + order param persistence."""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from agent.app.config import get_settings
from agent.app.logging_setup import get_logger

log = get_logger(__name__)

USE_MEMORY_STORE = get_settings().use_memory_store

_redis = None


async def get_redis():
    global _redis
    if _redis is None:
        if USE_MEMORY_STORE:
            from agent.app.state.memory_store import get_memory_store
            _redis = get_memory_store()
            log.info("using_memory_store")
        else:
            import redis.asyncio as aioredis
            s = get_settings()
            _redis = aioredis.from_url(s.redis_url, decode_responses=True)
            log.info("using_valkey", extra={"url": s.redis_url})
    return _redis


# ─────────────────────────────────────────────────────────────────────────────
# Message history
# ─────────────────────────────────────────────────────────────────────────────

async def load_history(session_id: str) -> list[dict[str, Any]]:
    r = await get_redis()
    raw = await r.get(f"history:{session_id}")
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("history_parse_error", extra={"session_id": session_id})
        return []


async def save_history(session_id: str, messages: list[dict[str, Any]]) -> None:
    r = await get_redis()
    s = get_settings()

    persistent = [
        m for m in messages
        if m.get("role") in ("user", "assistant", "tool")
    ]

    max_msgs = s.history_max_turns * 2
    if len(persistent) > max_msgs:
        persistent = persistent[-max_msgs:]

    await r.set(f"history:{session_id}", json.dumps(persistent), ex=s.history_ttl_s)


async def reset_history(session_id: str) -> None:
    r = await get_redis()
    await r.delete(f"history:{session_id}", f"order_params:{session_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Order params persistence
# ─────────────────────────────────────────────────────────────────────────────

async def load_order_params(session_id: str) -> dict[str, Any]:
    r = await get_redis()
    raw = await r.get(f"order_params:{session_id}")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


async def save_order_params(session_id: str, params: dict[str, Any]) -> None:
    r = await get_redis()
    s = get_settings()
    if not params:
        await r.delete(f"order_params:{session_id}")
    else:
        await r.set(f"order_params:{session_id}", json.dumps(params), ex=s.history_ttl_s)