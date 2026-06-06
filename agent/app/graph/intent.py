"""Intent classification and order parameter collection."""
from __future__ import annotations

import json
import re
from typing import Any

from agent.app.llm import chat_complete
from agent.app.logging_setup import get_logger
from agent.app.state.order_state import OrderParams, merge_order_params, params_complete, missing_fields_message

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Intent classification
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_SYSTEM = """\
You are an intent classifier for a food ordering assistant.
Classify the user message into exactly one of these intents:
- order_food       : user wants to place a new food order
- check_order_status : user wants to check status of an existing order
- general          : anything else (discovery, questions, greetings, complaints)

Respond with ONLY a JSON object, nothing else:
{"intent": "<intent>"}
"""


async def intent_node(state: dict[str, Any]) -> dict[str, Any]:
    """Classify user message intent."""
    user_message = state["user_message"]
    print(user_message)
    resp = await chat_complete(
        [
            {"role": "system", "content": _INTENT_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        tools=None,
        tool_choice=None,
        stream=False,
    )

    raw = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
        intent = data.get("intent", "general")
    except (json.JSONDecodeError, AttributeError):
        lower = user_message.lower()
        if any(w in lower for w in ["order", "want", "buy", "get me", "parcel"]):
            intent = "order_food"
        elif any(w in lower for w in ["status", "track", "where is", "my order"]):
            intent = "check_order_status"
        else:
            intent = "general"

    log.info("intent_classified", extra={"intent": intent, "user_msg": user_message[:80]})
    return {"intent": intent}


# ─────────────────────────────────────────────────────────────────────────────
# Parameter extraction
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """\
You are a parameter extractor for a food ordering assistant.
Extract order parameters from the user message and conversation history.

Extract:
- restaurant_name : name of the restaurant (string or null)
- items           : list of objects with {name: string, quantity: integer}
                    quantity defaults to 1 if not specified
                    can be multiple items

Return ONLY a JSON object, nothing else:
{
  "restaurant_name": "<name or null>",
  "items": [{"name": "<item>", "quantity": <int>}, ...]
}

If a field is not mentioned, use null for restaurant_name and [] for items.
Do not invent values. Only extract what the user explicitly said.
"""


async def _extract_params(user_message: str, history: list[dict]) -> dict[str, Any]:
    """Use LLM to extract order params from current message + recent history."""
    recent = history[-8:] if len(history) > 8 else history
    messages = [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        *recent,
        {"role": "user", "content": user_message},
    ]

    resp = await chat_complete(
        messages,
        tools=None,
        tool_choice=None,
        stream=False,
    )

    raw = (resp.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("param_extraction_parse_failed", extra={"raw": raw[:200]})
        return {"restaurant_name": None, "items": []}


async def collect_params_node(state: dict[str, Any]) -> dict[str, Any]:
    """Extract and accumulate order parameters."""
    user_message = state["user_message"]
    history = state.get("messages", [])

    clean_history = [
        m for m in history
        if m.get("role") in ("user", "assistant") and not m.get("tool_calls")
    ]

    extracted = await _extract_params(user_message, clean_history)

    existing: OrderParams = state.get("order_params") or {}
    merged = merge_order_params(existing, extracted)

    log.info(
        "params_collected",
        extra={
            "restaurant": merged.get("restaurant_name"),
            "item_count": len(merged.get("items", [])),
        },
    )

    if params_complete(merged):
        return {
            "order_params": merged,
            "params_complete": True,
        }
    else:
        ask_text = missing_fields_message(merged)
        log.info("params_incomplete", extra={"asking_msg": ask_text})
        return {
            "order_params": merged,
            "params_complete": False,
            "final_text": ask_text,
            "messages": [{"role": "assistant", "content": ask_text}],
        }