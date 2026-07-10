"""
Pre-call reminders injected as system messages when the conversation state
suggests the model is about to skip a required tool call.

Two triggers today:
  A) User just confirmed and the cart is built -> place_order is expected.
  B) User just confirmed cancellation of an order the previous assistant
     message asked about -> cancel_order is expected.

Both cases have failure modes where the model wants to conversationally
acknowledge before acting. The reminder closes that gap.

Design note:
    Trigger detection is heuristic. A false positive costs one extra system
    message in context (cheap); a false negative means the reminder doesn't
    fire, which is recoverable — the validator still catches fabricated
    placements/cancellations post-hoc.
"""

import re


# Affirmative phrases: English + common Hindi/Gujarati romanizations Aki
# routinely encounters in Vadodara conversations.
_CONFIRM_PATTERNS = [
    r"^\s*(?:yes|yep|yeah|yup|ok|okay|sure|confirm(?:ed)?|go ahead|do it|please do)\b",
    r"^\s*(?:place (?:it|the order|order)|order it|book it|checkout)\b",
    r"^\s*(?:haan|haa|ha ji|bilkul|kar do|kardo|karo|thik hai|theek hai)\b",
    r"^\s*(?:chalo|chalega|pakka|done|confirmed)\b",
]

_CANCEL_CONFIRM_PATTERNS = [
    r"^\s*(?:yes|yeah|confirm|do it|go ahead)\b.*cancel",
    r"^\s*cancel (?:it|that|the order)\b",
    r"^\s*(?:haan|haa)\b.*(?:cancel|band)",
    r"^\s*(?:band kar do|rehne do)\b",
]

# Signals from the previous assistant TEXT message that Aki just asked the
# user to confirm cancelling a placed order.
_ASSISTANT_CANCEL_ASK_PATTERNS = [
    r"\bcancel(?:l?ing)?\s+(?:this|that|your|the)\s+order\b",
    r"\bshould i cancel\b",
    r"\bwant me to cancel\b",
    r"\bconfirm (?:the )?cancellation\b",
]

# Signals from the persisted history that place_order already succeeded in
# this session — so a "yes" now should NOT re-fire the place_order reminder.
_ORDER_PLACED_PATTERNS = [
    r"\border(?:'s| is| has been| was)?\s*(?:been\s*)?placed\b",
    r"\bplaced your order\b",
    r"\border\s*(?:id|number|#)\s*(?:is|:)?\s*\w+",
]


def _looks_like_affirmative(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in _CONFIRM_PATTERNS)


def _looks_like_cancel_confirm(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in _CANCEL_CONFIRM_PATTERNS)


def _last_assistant_text(history: list[dict]) -> str:
    """Return the most recent assistant TEXT message.

    Skips assistant turns that were purely tool_calls with no user-facing
    text — those aren't what the user is responding to.
    """
    for m in reversed(history):
        if m.get("role") != "assistant":
            continue
        content = (m.get("content") or "").strip()
        if m.get("tool_calls") and not content:
            continue
        return content
    return ""


def _assistant_asked_to_cancel(history: list[dict]) -> bool:
    text = _last_assistant_text(history)
    return any(re.search(p, text, flags=re.IGNORECASE) for p in _ASSISTANT_CANCEL_ASK_PATTERNS)


def _order_already_placed_in_history(history: list[dict]) -> bool:
    """True if a prior assistant message in this session already confirmed
    a placed order. Prevents the place_order reminder from re-firing when
    the user says 'yes' to something else after a successful placement."""
    for m in history:
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        if any(re.search(p, content, flags=re.IGNORECASE) for p in _ORDER_PLACED_PATTERNS):
            return True
    return False


def maybe_inject_reminder(
    *,
    user_message: str,
    history: list[dict],
    cart_summary: str,
) -> str | None:
    """
    Returns a short system-message string to inject before the next model call,
    or None if no reminder is needed.

    Args:
        user_message: The current turn's user text.
        history:      Persisted message history for this session (may be empty).
                      Used to detect (a) whether the previous assistant message
                      asked about a cancellation, and (b) whether an order was
                      already placed earlier in the session.
        cart_summary: Output of cart_store.format_for_prompt(session_id).
                      Truthy means the in-progress cart has items.
    """
    if not user_message:
        return None

    # Case A: cart is built and user just confirmed -> place_order expected.
    # Guard against re-firing if this session already placed an order.
    if (
        _looks_like_affirmative(user_message)
        and cart_summary
        and not _order_already_placed_in_history(history)
    ):
        return (
            "REMINDER: The user has just confirmed. The cart is built. Your "
            "next action MUST be to call place_order. Do NOT reply with a "
            "confirmation message first — call the tool, wait for its result, "
            "and only then tell the user the outcome using the order_id from "
            "the tool result. If place_order fails, apologize and offer one "
            "concrete next step. Never claim the order is placed unless "
            "place_order returned success."
        )

    # Case B: previous assistant message asked to cancel an order and the
    # user just confirmed -> cancel_order expected. The order_id lives in
    # the [ORDER STATUS] block or comes from get_active_orders — the model
    # already has access to it.
    if _looks_like_cancel_confirm(user_message) and _assistant_asked_to_cancel(history):
        return (
            "REMINDER: The user has just confirmed cancellation. Your next "
            "action MUST be to call cancel_order with the order_id (from the "
            "[ORDER STATUS] block or get_active_orders) and a reason. Do NOT "
            "reply with a cancellation message first. Only announce the "
            "cancellation after cancel_order returns success."
        )

    return None