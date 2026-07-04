"""
Pre-call reminders injected as system messages when the conversation state
suggests the model is about to skip a required tool call.

Two triggers today:
  A) User just confirmed and the cart is ready -> place_order is expected.
  B) User just confirmed cancellation of a specific placed order -> cancel_order.

Both cases have failure modes where the model wants to conversationally
acknowledge before acting. The reminder closes that gap.
"""

import re


# Affirmative phrases: English + common Hindi/Gujarati romanizations Aki
# routinely encounters in Vadodara conversations.
_CONFIRM_PATTERNS = [
    r"^\s*(?:yes|yep|yeah|yup|ok|okay|sure|confirm(?:ed)?|go ahead|do it|please do)\b",
    r"^\s*(?:place (?:it|the order)|order it|book it|checkout)\b",
    r"^\s*(?:haan|haa|ha ji|bilkul|kar do|kardo|karo|thik hai|theek hai)\b",
    r"^\s*(?:chalo|chalega|pakka|done|confirmed)\b",
]

_CANCEL_CONFIRM_PATTERNS = [
    r"^\s*(?:yes|confirm|do it|go ahead)\b.*cancel",
    r"^\s*cancel (?:it|that|the order)\b",
    r"^\s*(?:haan|haa)\b.*(?:cancel|band)",
    r"^\s*(?:band kar do|rehne do)\b",
]


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )
        return content or ""
    return ""


def _looks_like_affirmative(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in _CONFIRM_PATTERNS)


def _looks_like_cancel_confirm(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in _CANCEL_CONFIRM_PATTERNS)


def maybe_inject_reminder(
    messages: list[dict],
    session_state: dict,
) -> str | None:
    """
    Returns a short system-message string to inject before the next model call,
    or None if no reminder is needed.

    session_state (hydrated from Valkey) is expected to expose:
      - cart_items: list           # items in the in-progress cart
      - has_delivery_address: bool # user has a resolved address
      - order_placed_this_session: bool  # avoid re-firing after success
      - pending_cancel_order_id: str | None
            # set when the model just asked user to confirm cancelling
            # a specific placed order; cleared after cancel_order returns
    """
    user_text = _last_user_text(messages)
    if not user_text:
        return None

    # Case A: user just confirmed and cart is ready -> place_order is expected.
    if (
        _looks_like_affirmative(user_text)
        and session_state.get("cart_items")
        and session_state.get("has_delivery_address")
        and not session_state.get("order_placed_this_session")
    ):
        return (
            "REMINDER: The user has just confirmed. The cart is built and the "
            "delivery address is set. Your next action MUST be to call "
            "place_order. Do NOT reply with a confirmation message first — "
            "call the tool, wait for its result, and only then tell the user "
            "the outcome using the order_id from the tool result. If "
            "place_order fails, apologize and offer one concrete next step. "
            "Never claim the order is placed unless place_order returned success."
        )

    # Case B: user just confirmed cancellation of a specific placed order.
    pending_id = session_state.get("pending_cancel_order_id")
    if _looks_like_cancel_confirm(user_text) and pending_id:
        return (
            "REMINDER: The user has just confirmed cancelling order "
            f"{pending_id}. Your next action MUST be to call cancel_order "
            "with that order_id and a reason. Do NOT reply with a "
            "cancellation message first. Only announce the cancellation "
            "after cancel_order returns success."
        )

    return None