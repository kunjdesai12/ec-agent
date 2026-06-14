"""Order parameter accumulation and validation.

OrderParams is a plain dict stored in GraphState.order_params and persisted
to Valkey as part of conversation history between turns.

Shape:
{
    "restaurant_name": str | None,
    "items": [{"name": str, "quantity": int,
               "special_instructions": str | None}, ...],
    "order_type": str | None,          # "delivery" | "pickup" | "dinein"
    "is_cod": bool | None,             # True = cash, False = online
    "delivery_address": dict | None,   # full address object from get_user_addresses
    "active_intent": str | None        # sticky intent across turns
}
"""
from __future__ import annotations

from typing import Any, TypedDict


class OrderItem(TypedDict, total=False):
    name: str
    quantity: int
    special_instructions: str | None


class OrderParams(TypedDict, total=False):
    restaurant_name: str | None
    items: list[OrderItem]
    order_type: str | None
    is_cod: bool | None
    delivery_address: dict | None
    active_intent: str | None


# ── Change-detection ──────────────────────────────────────────────────────────

_CHANGE_SIGNALS = [
    "instead of",
    "instead",
    "not that",
    "change my order",
    "change the order",
    "change item",
    "different item",
    "different food",
    "replace",
    "swap",
    "cancel that",
    "forget the",
    "forget that",
    "not anymore",
    "no more",
    "actually i want",
    "actually i'll have",
    "let me change",
    "i changed my mind",
]


def should_clear_items(user_message: str) -> bool:
    """Return True if the user is changing their item selection.

    Signals like "instead of tea, I want sandwich" or "change my order"
    indicate the existing items should be cleared before merging new ones,
    so stale items don't persist across turns.
    """
    lower = user_message.lower()
    return any(signal in lower for signal in _CHANGE_SIGNALS)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_instructions(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


# ── Core merge ────────────────────────────────────────────────────────────────

def merge_order_params(existing: OrderParams, extracted: dict[str, Any]) -> OrderParams:
    """Merge newly extracted params on top of existing accumulated params.

    Rules:
    - restaurant_name: use extracted if non-null, else keep existing
    - items: merge by name — extracted items override existing ones with the
      same name (case-insensitive). If should_clear_items() fired upstream,
      existing items will already be empty before this is called.
    - order_type, is_cod, delivery_address: preserved from existing unless
      explicitly provided in extracted
    - active_intent: always preserved from existing
    """
    merged: OrderParams = dict(existing)

    # Restaurant name
    new_restaurant = extracted.get("restaurant_name")
    if new_restaurant:
        merged["restaurant_name"] = new_restaurant
    elif "restaurant_name" not in merged:
        merged["restaurant_name"] = None

    # Items
    existing_items: list[OrderItem] = list(merged.get("items") or [])
    new_items: list[dict] = extracted.get("items") or []
    existing_by_name = {item["name"].lower(): i for i, item in enumerate(existing_items)}

    for new_item in new_items:
        name = new_item.get("name", "").strip()
        qty = int(new_item.get("quantity") or 1)
        if not name:
            continue

        instr_provided = "special_instructions" in new_item
        instr_value = _clean_instructions(new_item.get("special_instructions"))

        key = name.lower()
        if key in existing_by_name:
            idx = existing_by_name[key]
            existing_items[idx]["quantity"] = qty
            if instr_provided:
                existing_items[idx]["special_instructions"] = instr_value
        else:
            item: OrderItem = {"name": name, "quantity": qty}
            if instr_provided:
                item["special_instructions"] = instr_value
            existing_items.append(item)

    merged["items"] = existing_items

    # Preserve these fields from existing — LLM collects them after tools run
    for field in ("order_type", "is_cod", "delivery_address"):
        if field not in merged:
            merged[field] = None

    # active_intent — never overwrite with extracted
    if "active_intent" not in merged:
        merged["active_intent"] = None

    return merged


# ── Completion check ──────────────────────────────────────────────────────────

# REPLACE params_complete:
def params_complete(params: OrderParams) -> bool:
    has_restaurant = bool(params.get("restaurant_name"))
    has_items = bool(params.get("items"))
    return has_restaurant and has_items


# REPLACE missing_fields_message:
def missing_fields_message(params: OrderParams) -> str:
    missing = []

    if not params.get("restaurant_name"):
        missing.append("which restaurant you'd like to order from")

    if not params.get("items"):
        missing.append("which items you'd like (and how many of each)")

    if not missing:
        return ""

    if len(missing) == 1:
        return f"Could you let me know {missing[0]}?"

    parts = ", ".join(missing[:-1]) + f" and {missing[-1]}"
    return f"Could you let me know {parts}?"