"""Valkey-backed conversation history.

Key schema:
  session:{session_id}:messages  → JSON list of message dicts (TTL'd)

Each message follows OpenAI/Anthropic chat shape:
  {"role": "user"|"assistant"|"tool", "content": "...", "tool_calls": [...], "tool_call_id": "..."}
"""
from __future__ import annotations

import json
from typing import Any, Optional

from redis.asyncio import Redis, from_url

from agent.app.config import get_settings
from agent.app.logging_setup import get_logger

log = get_logger(__name__)

_redis: Optional[Redis] = None


async def get_redis() -> Redis:
    global _redis
    if _redis is None:
        s = get_settings()
        _redis = from_url(s.redis_url, encoding="utf-8", decode_responses=True)
    return _redis


def _key(session_id: str) -> str:
    return f"session:{session_id}:messages"


async def load_history(session_id: str) -> list[dict[str, Any]]:
    r = await get_redis()
    raw = await r.get(_key(session_id))
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("history_decode_failed", extra={"session_id": session_id})
        return []


async def save_history(session_id: str, messages: list[dict[str, Any]]) -> None:
    s = get_settings()
    r = await get_redis()
    # Keep only the last N turns (user+assistant pairs ≈ 2 msgs/turn, tools add more)
    # A "turn" here = one user message and everything that follows up to next user
    trimmed = _trim_history(messages, max_turns=s.history_max_turns)
    await r.set(_key(session_id), json.dumps(trimmed), ex=s.history_ttl_s)


async def reset_history(session_id: str) -> None:
    r = await get_redis()
    await r.delete(_key(session_id))


def _trim_history(messages: list[dict[str, Any]], *, max_turns: int) -> list[dict[str, Any]]:
    """Trim from the front while preserving turn boundaries.

    A turn starts at a 'user' message. We keep the most recent max_turns turns.
    System messages (if any leak in) are preserved at the front.
    """
    if not messages:
        return messages

    # Find user-message indices
    user_idxs = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(user_idxs) <= max_turns:
        return messages

    cut_from = user_idxs[-max_turns]
    return messages[cut_from:]
