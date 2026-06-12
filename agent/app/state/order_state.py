"""Order parameter accumulation and validation.

OrderParams is a plain dict stored in GraphState.order_params and persisted
to Valkey as part of conversation history between turns.

Shape:
{
    "restaurant_name": str | None,
    "items": [{"name": str, "quantity": int,
               "special_instructions": str | None}, ...],
    "active_intent": str | None   # sticky intent across turns
}
"""
from __future__ import annotations

from typing import Any, TypedDict


class OrderItem(TypedDict, total=False):
    name: str
    quantity: int
    special_instructions: str | None    # optional — e.g. "extra spicy", "no onions"


class OrderParams(TypedDict, total=False):
    restaurant_name: str | None
    items: list[OrderItem]
    active_intent: str | None          # sticky intent — set by intent_node


def _clean_instructions(value: Any) -> str | None:
    """Normalize special_instructions to a non-empty string or None."""
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def merge_order_params(existing: OrderParams, extracted: dict[str, Any]) -> OrderParams:
    """Merge newly extracted params on top of existing accumulated params.

    Rules:
    - restaurant_name: use extracted if non-null, else keep existing
    - items: merge by name — extracted items override existing ones with the
      same name (case-insensitive). Quantity is replaced. special_instructions
      is replaced if explicitly provided in extracted (including empty string,
      which clears it); otherwise preserved.
    - active_intent: always preserved from existing unless explicitly overwritten
    """
    merged: OrderParams = dict(existing)  # shallow copy

    # Restaurant name — only update if newly extracted
    new_restaurant = extracted.get("restaurant_name")
    if new_restaurant:
        merged["restaurant_name"] = new_restaurant
    elif "restaurant_name" not in merged:
        merged["restaurant_name"] = None

    # Items — merge by name
    existing_items: list[OrderItem] = list(merged.get("items") or [])
    new_items: list[dict] = extracted.get("items") or []

    existing_by_name = {item["name"].lower(): i for i, item in enumerate(existing_items)}

    for new_item in new_items:
        name = new_item.get("name", "").strip()
        qty = int(new_item.get("quantity") or 1)
        if not name:
            continue

        # Only treat special_instructions as "provided" if the key is present
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

    # active_intent — preserve from existing, never overwrite with extracted
    if "active_intent" not in merged:
        merged["active_intent"] = None

    return merged


def params_complete(params: OrderParams) -> bool:
    """Return True if we have enough to proceed to semantic matching."""
    has_restaurant = bool(params.get("restaurant_name"))
    has_items = bool(params.get("items"))
    return has_restaurant and has_items


def missing_fields_message(params: OrderParams) -> str:
    """Generate a single natural-language question asking for all missing params."""
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