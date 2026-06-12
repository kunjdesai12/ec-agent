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
- order_food         : user wants to place a new food order
- check_order_status : user wants to check status of an existing order
- cancel_intent      : user wants to cancel or start over
- general            : anything else (discovery, questions, greetings, complaints)

Respond with ONLY a JSON object, nothing else:
{"intent": "<intent>"}
"""

# Signals that the user explicitly wants to switch away from current intent
_SWITCH_SIGNALS = {
    "order_food": [
        "check my order", "where is my order", "order status",
        "track my order", "cancel", "start over", "never mind",
    ],
    "check_order_status": [
        "order food", "i want to order", "place order",
        "get me", "i'll have", "cancel", "start over", "never mind",
    ],
}

# These always force a fresh classification regardless of sticky intent
_ALWAYS_RECLASSIFY = [
    "cancel", "start over", "never mind", "forget it",
    "reset", "new order", "different order",
]


def _should_reclassify(user_message: str, current_intent: str) -> bool:
    """
    Decide whether to run the LLM classifier or reuse the sticky intent.

    Returns True  → run classifier (intent may change)
    Returns False → reuse current_intent as-is
    """
    lower = user_message.lower()

    # Always reclassify on explicit reset phrases
    if any(phrase in lower for phrase in _ALWAYS_RECLASSIFY):
        return True

    # Check if user is explicitly switching away from current intent
    switch_signals = _SWITCH_SIGNALS.get(current_intent, [])
    if any(signal in lower for signal in switch_signals):
        return True

    # Otherwise keep the sticky intent
    return False


async def intent_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Classify user message intent with sticky intent support.

    If an intent is already active (stored in order_params),
    reuse it unless the user explicitly signals a switch.
    This prevents mid-order messages like "yes", "2 please",
    "14 Alkapuri" from being misclassified as 'general'.
    """
    user_message = state["user_message"]
    order_params: OrderParams = state.get("order_params") or {}

    # ── Sticky intent check ───────────────────────────────────────────────────
    current_intent = order_params.get("active_intent")

    if current_intent and not _should_reclassify(user_message, current_intent):
        log.info(
            "intent_sticky",
            extra={"intent": current_intent, "user_msg": user_message[:80]},
        )
        return {"intent": current_intent}

    # ── Fresh classification ──────────────────────────────────────────────────
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

    log.info(
        "intent_classified",
        extra={"intent": intent, "user_msg": user_message[:80]},
    )

    # ── Store active intent in order_params so it persists across turns ───────
    # Only set/update when intent is actionable (not general/cancel)
    if intent in ("order_food", "check_order_status"):
        order_params["active_intent"] = intent
    elif intent == "cancel_intent":
        # Clear sticky intent and order params on explicit cancel
        order_params["active_intent"] = None
        log.info("intent_reset", extra={"reason": "cancel_intent"})

    return {
        "intent": intent,
        "order_params": order_params,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parameter extraction
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """\
Extract:
- restaurant_name : name of the restaurant (string or null)
- items           : list of objects with:
                      - name: string
                      - quantity: integer (defaults to 1)
                      - special_instructions: string or null
                        (e.g. "extra spicy", "no onions", "less oil")
                    Only include special_instructions if the user explicitly
                    said something about how the item should be prepared.

Return ONLY a JSON object, nothing else:
{
  "restaurant_name": "<name or null>",
  "items": [
    {"name": "<item>", "quantity": <int>, "special_instructions": "<str or null>"},
    ...
  ]
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