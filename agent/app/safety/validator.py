"""
Post-generation validator for Aki assistant responses.

Scans the assistant's text output for claims of actions (order placed,
cart modified, order cancelled) and verifies a matching successful tool
call occurred in the SAME turn. If a claim is unbacked, returns a
reprompt string to inject before re-running the model.

Design note:
    Patterns are deliberately broad on the trigger side. A false positive
    just costs one retry; a false negative sends a fabricated confirmation
    to a real customer. Tune against transcripts once collected.
"""

import re
from dataclasses import dataclass
from enum import Enum


class Violation(str, Enum):
    CLAIMED_PLACED = "claimed_order_placed_without_success"
    CLAIMED_CANCELLED = "claimed_cancel_without_success"
    CLAIMED_ADDED = "claimed_add_without_success"
    CLAIMED_REMOVED = "claimed_remove_without_success"
    CLAIMED_UPDATED = "claimed_update_without_success"


_PLACED_PATTERNS = [
    r"\border(?:'s| is| has been| was)?\s*(?:been\s*)?placed\b",
    r"\bplaced your order\b",
    r"\border\s*(?:is\s*)?confirmed\b",
    r"\border\s*(?:is\s*)?on (?:its|the) way\b",
    r"\border\s*(?:id|number|#)\s*(?:is|:)?\s*\w+",
    r"\byour order is in\b",
]

_CANCELLED_PATTERNS = [
    r"\b(?:order|it)\s*(?:has been|is|was)?\s*cancell?ed\b",
    r"\bcancell?ed your order\b",
    r"\bcancellation\s*(?:is\s*)?(?:done|complete|confirmed)\b",
]

_ADDED_PATTERNS = [
    r"\badded (?:it |them |that )?to your cart\b",
    r"\bin your cart (?:now|already)\b",
    r"\b(?:i've|i have) added\s+(?:the|a|an|\d)",
]

_REMOVED_PATTERNS = [
    r"\bremoved (?:it |them |that )?from your cart\b",
    r"\btaken (?:it |them |that )?off (?:your |the )?cart\b",
    r"\b(?:i've|i have) removed\b",
]

_UPDATED_PATTERNS = [
    r"\bquantity\s*(?:is|has been|updated to)\s*\d+",
    r"\bmade it (?:\d+|two|three|four|five)\b",
    r"\bnow you have \d+\s+\w+\s+in (?:your |the )?cart\b",
]

_REQUIRED_TOOL = {
    Violation.CLAIMED_PLACED: {"place_order"},
    Violation.CLAIMED_CANCELLED: {"cancel_order", "discard_current_order"},
    Violation.CLAIMED_ADDED: {"add_to_cart"},
    Violation.CLAIMED_REMOVED: {"remove_from_cart", "update_cart_item"},
    Violation.CLAIMED_UPDATED: {"update_cart_item", "add_to_cart"},
}

_PATTERN_TO_VIOLATION = [
    (_PLACED_PATTERNS, Violation.CLAIMED_PLACED),
    (_CANCELLED_PATTERNS, Violation.CLAIMED_CANCELLED),
    (_ADDED_PATTERNS, Violation.CLAIMED_ADDED),
    (_REMOVED_PATTERNS, Violation.CLAIMED_REMOVED),
    (_UPDATED_PATTERNS, Violation.CLAIMED_UPDATED),
]


@dataclass
class ValidationResult:
    ok: bool
    violations: list[Violation]
    reprompt: str | None  # System note to inject on retry, None if ok.


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def validate_assistant_turn(
    assistant_text: str,
    successful_tool_calls_this_turn: list[str],
) -> ValidationResult:
    """
    Args:
        assistant_text: The final text the model wants to send to the user.
        successful_tool_calls_this_turn: Names of tools that returned success
            in the current turn (since the last user message).

    Returns:
        ValidationResult. If ok=False, `reprompt` is a system-message string
        to append before re-running the model.
    """
    called = set(successful_tool_calls_this_turn)
    violations: list[Violation] = []

    for patterns, violation in _PATTERN_TO_VIOLATION:
        if _matches_any(assistant_text, patterns):
            required = _REQUIRED_TOOL[violation]
            if not (called & required):
                violations.append(violation)

    if not violations:
        return ValidationResult(ok=True, violations=[], reprompt=None)

    return ValidationResult(
        ok=False,
        violations=violations,
        reprompt=_build_reprompt(violations),
    )


def _build_reprompt(violations: list[Violation]) -> str:
    lines = [
        "VALIDATION FAILED. Your previous draft claimed an action that did not happen.",
        "You have NOT actually done what you said. Do not send that message. Correct it now:",
    ]
    if Violation.CLAIMED_PLACED in violations:
        lines.append(
            "- You said the order was placed, but place_order was not called "
            "successfully this turn. If the user has confirmed, call place_order "
            "NOW. Otherwise, read back the cart and ask for confirmation. "
            "Do NOT claim placement."
        )
    if Violation.CLAIMED_CANCELLED in violations:
        lines.append(
            "- You said the order was cancelled, but no cancel/discard tool "
            "succeeded this turn. Call cancel_order (for a placed order) or "
            "discard_current_order (for an in-progress cart)."
        )
    if Violation.CLAIMED_ADDED in violations:
        lines.append(
            "- You said an item was added, but add_to_cart did not succeed "
            "this turn. Call add_to_cart now using real ids from tool results."
        )
    if Violation.CLAIMED_REMOVED in violations:
        lines.append(
            "- You said an item was removed, but no remove/update tool "
            "succeeded this turn. Call the correct tool with the item_id from "
            "[ORDER STATUS]."
        )
    if Violation.CLAIMED_UPDATED in violations:
        lines.append(
            "- You said a quantity was updated, but update_cart_item did not "
            "succeed this turn. Call it now with the item_id from [ORDER STATUS]."
        )
    return "\n".join(lines)