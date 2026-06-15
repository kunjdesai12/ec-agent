"""Intent classification and order parameter collection."""
from __future__ import annotations

import json
import re
from typing import Any

from agent.app.llm import chat_complete
from agent.app.logging_setup import get_logger
from agent.app.state.order_state import (
    OrderParams, merge_order_params, params_complete,
    missing_fields_message, should_clear_items,
)

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Intent classification
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_SYSTEM = """\
You are an intent classifier for a food ordering assistant.
Classify the user message into exactly one of these intents:
- order_food         : user wants to place a new food order
- check_order_status : user wants to check status of an existing order
- cancel_order       : user wants to cancel an existing order
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
    "cancel_order": [
        "order food", "i want to order", "place order",
        "start over", "never mind", "forget it",
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

    if any(phrase in lower for phrase in _ALWAYS_RECLASSIFY):
        return True

    switch_signals = _SWITCH_SIGNALS.get(current_intent, [])
    if any(signal in lower for signal in switch_signals):
        return True

    return False


async def intent_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Classify intent with sticky support. Sticky is a FALLBACK, not an override:
    we always classify, and only fall back to the active intent when the fresh
    result is ambiguous ('general'). This keeps mid-flow filler ("yes", "2 please",
    "14 Alkapuri") in the active flow while still allowing a clear new request
    ("get me biryani from honest") to switch intents.
    """
    user_message = state["user_message"]
    order_params: OrderParams = state.get("order_params") or {}
    active_intent = order_params.get("active_intent")

    # ── Always classify fresh ─────────────────────────────────────────────
    resp = await chat_complete(
        [
            {"role": "system", "content": _INTENT_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        tools=None, tool_choice=None, stream=False,
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
        fresh = data.get("intent", "general")
    except (json.JSONDecodeError, AttributeError):
        lower = user_message.lower()
        if any(w in lower for w in ["order", "want", "buy", "get me", "parcel"]):
            fresh = "order_food"
        elif any(w in lower for w in ["status", "track", "where is", "my order"]):
            fresh = "check_order_status"
        elif any(w in lower for w in ["cancel", "cancellation", "stop order"]):
            fresh = "cancel_order"
        else:
            fresh = "general"

    # ── Sticky as fallback, not override ──────────────────────────────────
    ACTIONABLE = ("order_food", "check_order_status", "cancel_order")
    if fresh in ACTIONABLE:
        intent = fresh                              # clear intent → honor / switch
        if active_intent and fresh != active_intent:
            log.info("intent_switched", extra={
                "from": active_intent, "to": fresh, "user_msg": user_message[:80]})
            # Intent changed → abandon the previous flow's order scratch so a
            # new order doesn't inherit stale restaurant/items.
            order_params = {"active_intent": fresh}
        else:
            order_params["active_intent"] = fresh
            log.info("intent_classified", extra={
                "intent": intent, "user_msg": user_message[:80]})
        order_params["active_intent"] = intent
    elif active_intent:
        intent = active_intent                      # ambiguous → stay in flow
        log.info("intent_sticky", extra={
            "intent": intent, "user_msg": user_message[:80]})
    else:
        intent = fresh                              # general, no active flow
        log.info("intent_classified", extra={
            "intent": intent, "user_msg": user_message[:80]})

    return {"intent": intent, "order_params": order_params}


# ─────────────────────────────────────────────────────────────────────────────
# Parameter extraction
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """\
Extract order parameters from the user message.

Return ONLY a JSON object, nothing else:
{
  "restaurant_name": "<name or null>",
  "items": [
    {"name": "<item>", "quantity": <int>, "special_instructions": "<str or null>"}
  ],
  "order_type": "<delivery|pickup|dinein or null>",
  "is_cod": <true|false|null>
}

Extraction rules:
- restaurant_name : name of the restaurant, or null if not mentioned
- items           : list of food/drink items the user wants to order
                    - name: item name (string)
                    - quantity: how many (integer, defaults to 1)
                    - special_instructions: only if user explicitly said how
                      to prepare it (e.g. "extra spicy", "no onions"), else null
- order_type      : "delivery" if user wants it delivered
                    "pickup" if user says takeaway/pickup/take away/parcel/collect
                    "dinein" if user says dine in/eat here/sit down
                    null if not mentioned
- is_cod          : true if user says cash/COD/cash on delivery
                    false if user says online/card/UPI/digital payment
                    null if not mentioned

Do not invent values. Only extract what the user explicitly said.
If a field is not mentioned, use null for strings and null for booleans.
"""


async def _extract_params(user_message: str, history: list[dict]) -> dict[str, Any]:
    """Use LLM to extract order params from current message + recent history.

    history should contain only clean user/assistant turns (no system prompts,
    no tool messages, no tool_calls) — filtered by collect_params_node before
    passing in.
    """
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
        return {"restaurant_name": None, "items": [], "order_type": None, "is_cod": None}


async def collect_params_node(state: dict[str, Any]) -> dict[str, Any]:
    """Extract and accumulate order parameters.

    Extracts: restaurant_name, items, order_type, is_cod from the user message.
    order_type and is_cod are stored in order_params so llm_node can detect
    when all details are confirmed and force the place_order tool call.

    NOTE: This node never writes to state["messages"].

    Item rejection flow:
    - When retrieve_node finds the requested item isn't available, it sets
      item_rejected=True in GraphState. run_turn() persists this into
      order_params so it survives the Valkey round-trip.
    - On the next turn, we detect item_rejected in order_params and clear
      the stale items[] before merging the user's new item selection.
      This ensures "biryani" doesn't persist when the user switches to "dosa".
    - The flag is consumed (popped) immediately so it only fires once.
    """
    user_message = state["user_message"]
    history = state.get("messages", [])
    existing: OrderParams = state.get("order_params") or {}

    # ── Clear stale items if previous turn rejected the requested item ────────
    # item_rejected is stored in order_params by run_turn() so it survives
    # the Valkey round-trip. Takes priority over should_clear_items() since
    # the user may simply say "dosa" without any explicit "instead of" signal.
    item_rejected = state.get("item_rejected") or existing.get("item_rejected", False)
    if item_rejected:
        existing = dict(existing)
        existing["items"] = []
        existing.pop("item_rejected", None)   # consume — only fires once
        log.info("items_cleared_after_rejection", extra={
            "restaurant": existing.get("restaurant_name"),
        })

    # ── Clear stale items if user explicitly signals a change ─────────────────
    # Handles phrases like "instead of", "change my order", "swap" etc.
    # Only runs if item_rejected didn't already clear items above.
    elif should_clear_items(user_message):
        existing = dict(existing)
        existing["items"] = []
        log.info("items_cleared", extra={
            "reason": "change_signal_detected",
            "msg": user_message[:80],
        })

    # ── Strip duplicate user message from history tail ────────────────────────
    # build_initial_messages already appended the current user_message to
    # state["messages"]. _extract_params also appends it explicitly, so
    # remove it from the tail of clean_history to avoid sending it twice.
    clean_history = [
        m for m in history
        if m.get("role") in ("user", "assistant") and not m.get("tool_calls")
    ]
    if clean_history and clean_history[-1].get("content") == user_message:
        clean_history = clean_history[:-1]

    extracted = await _extract_params(user_message, clean_history)
    merged = merge_order_params(existing, extracted)

    # ── Persist order_type and is_cod from extraction into order_params ───────
    # Only overwrite if the extractor found a value (don't clear existing).
    if extracted.get("order_type") is not None:
        merged["order_type"] = extracted["order_type"]
    if extracted.get("is_cod") is not None:
        merged["is_cod"] = extracted["is_cod"]

    log.info(
        "params_collected",
        extra={
            "restaurant": merged.get("restaurant_name"),
            "items": merged.get("items"),
            "order_type": merged.get("order_type"),
            "is_cod": merged.get("is_cod"),
        },
    )

    if params_complete(merged):
        return {
            "order_params": merged,
            "params_complete": True,
            # No messages written — graph continues to retrieve → llm
        }
    else:
        ask_text = missing_fields_message(merged)
        log.info("params_incomplete", extra={"asking_msg": ask_text})
        return {
            "order_params": merged,
            "params_complete": False,
            "final_text": ask_text,
            # No messages written — run_turn() appends this to Valkey history
        }