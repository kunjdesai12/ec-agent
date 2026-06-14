"""Intent classification and order parameter collection."""
from __future__ import annotations

import json
import re
from typing import Any

from agent.app.llm import chat_complete
from agent.app.logging_setup import get_logger
from agent.app.state.order_state import ( OrderParams, merge_order_params, params_complete, missing_fields_message, should_clear_items)

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

    NOTE: This node never writes to state["messages"]. Intent classification
    is internal routing only and must not appear in LLM conversation history.
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
        elif any(w in lower for w in ["cancel", "cancellation", "stop order"]):
            intent = "cancel_order"
        else:
            intent = "general"

    log.info(
        "intent_classified",
        extra={"intent": intent, "user_msg": user_message[:80]},
    )

    # ── Store active intent in order_params so it persists across turns ───────
    # Only set/update when intent is actionable (not general/cancel)
    if intent in ("order_food", "check_order_status", "cancel_order"):
        order_params["active_intent"] = intent

    return {
        "intent": intent,
        "order_params": order_params,
    }


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


def _resolve_address_selection(
    user_message: str, saved_addresses: list[dict]
) -> dict | None:
    import re
    lower = user_message.lower().strip()

    # Numeric selection — "1", "2", etc.
    num_match = re.search(r'\b([1-9])\b', lower)
    if num_match:
        idx = int(num_match.group(1)) - 1
        if 0 <= idx < len(saved_addresses):
            return _format_address_for_order(saved_addresses[idx])

    # Ordinal words
    ordinals = {
        "first": 0, "1st": 0,
        "second": 1, "2nd": 1,
        "third": 2, "3rd": 2,
    }
    for word, idx in ordinals.items():
        if word in lower and idx < len(saved_addresses):
            return _format_address_for_order(saved_addresses[idx])

    # Label match — check if any address label appears in the user message
    # Also handle common aliases
    label_aliases = {
        "office": "work",
        "my home": "home",
        "house": "home",
        "at home": "home",
        "home address": "home",
        "work address": "work",
    }
    # Apply aliases first
    normalized = lower
    for alias, replacement in label_aliases.items():
        if alias in normalized:
            normalized = normalized.replace(alias, replacement)

    for addr in saved_addresses:
        label = (addr.get("label") or "").lower()
        # Check if the label appears anywhere in the (normalized) user message
        if label and label in normalized:
            return _format_address_for_order(addr)

    # If only one address exists and user said yes/ok/that one/confirm
    confirm_signals = ["yes", "ok", "okay", "that one", "this one", "confirm", "sure", "yep", "yeah"]
    if len(saved_addresses) == 1 and any(s in lower for s in confirm_signals):
        return _format_address_for_order(saved_addresses[0])

    return None


def _format_address_for_order(addr: dict) -> dict:
    """Convert a saved address dict into the shape place_order expects."""
    return {
        "address_line_1": addr.get("full_address") or addr.get("address_line_1", ""),
        "address_line_2": addr.get("house_number") or addr.get("floor_no") or "",
        "landmark":       addr.get("landmark") or "",
        "city":           addr.get("city", ""),
        "state":          addr.get("state", ""),
        "pincode":        str(addr.get("pincode", "")),
        "latitude":       float(addr.get("latitude") or 0),
        "longitude":      float(addr.get("longitude") or 0),
        # preserve label for logging
        "label":          addr.get("label", ""),
    }


async def collect_params_node(state: dict[str, Any]) -> dict[str, Any]:
    """Extract and accumulate order parameters.

    Extracts: restaurant_name, items, order_type, is_cod from the user message.
    order_type and is_cod are stored in order_params so llm_node can detect
    when all details are confirmed and force the place_order tool call.

    NOTE: This node never writes to state["messages"].
    """
    user_message = state["user_message"]
    history = state.get("messages", [])

    clean_history = [
        m for m in history
        if m.get("role") in ("user", "assistant") and not m.get("tool_calls")
    ]

        # The current user_message is already in clean_history (added by
    # build_initial_messages). Strip it from the tail before passing to
    # _extract_params, which appends it explicitly.
    if clean_history and clean_history[-1].get("content") == user_message:
        clean_history = clean_history[:-1]

    extracted = await _extract_params(user_message, clean_history)

    existing: OrderParams = state.get("order_params") or {}

    # ── Clear stale items if user is changing their selection ─────────────────
    if should_clear_items(user_message):
        existing = dict(existing)
        existing["items"] = []
        log.info("items_cleared", extra={
            "reason": "change_signal_detected",
            "msg": user_message[:80],
        })

    merged = merge_order_params(existing, extracted)

    saved_addresses = merged.get("_saved_addresses", [])
    if saved_addresses and not merged.get("delivery_address"):
        resolved = _resolve_address_selection(user_message, saved_addresses)
        if resolved:
            merged["delivery_address"] = resolved
            log.info("delivery_address_resolved", extra={
                "label": resolved.get("label"),
                "city":  resolved.get("city"),
            })

    # ── Persist order_type and is_cod from extraction into order_params ───────
    # These are not handled by merge_order_params since they're new fields.
    # Only overwrite if the extractor found a value (don't clear existing).
    if extracted.get("order_type") is not None:
        merged["order_type"] = extracted["order_type"]
    if extracted.get("is_cod") is not None:
        merged["is_cod"] = extracted["is_cod"]

    log.info(
        "params_collected",
        extra={
            "restaurant": merged.get("restaurant_name"),
            "item_count": len(merged.get("items", [])),
            "order_type":  merged.get("order_type"),
            "is_cod": merged.get("is_cod"),
        },
    )

    if params_complete(merged):
        return {
            "order_params": merged,
            "params_complete": True,
            # No messages written — graph continues to semantic_match → llm
        }
    else:
        ask_text = missing_fields_message(merged)
        log.info("params_incomplete", extra={"asking_msg": ask_text})
        return {
            "order_params": merged,
            "params_complete": False,
            "final_text": ask_text,
        }