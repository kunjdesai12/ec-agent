"""Order parameter accumulation and validation.

OrderParams is a plain dict stored in GraphState.order_params and persisted
to Valkey as part of conversation history between turns.

Shape:
{
    "restaurant_name": str | None,
    "items": [{"name": str, "quantity": int}, ...],
    "active_intent": str | None   # sticky intent across turns
}
"""
from __future__ import annotations

from typing import Any, TypedDict


class OrderItem(TypedDict):
    name: str
    quantity: int


class OrderParams(TypedDict, total=False):
    restaurant_name: str | None
    items: list[OrderItem]
    active_intent: str | None          # sticky intent — set by intent_node


def merge_order_params(existing: OrderParams, extracted: dict[str, Any]) -> OrderParams:
    """Merge newly extracted params on top of existing accumulated params.

    Rules:
    - restaurant_name: use extracted if non-null, else keep existing
    - items: merge by name — extracted items override existing ones with the
      same name (case-insensitive), new items are appended
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

    # Build lookup of existing items by lowercase name
    existing_by_name = {item["name"].lower(): i for i, item in enumerate(existing_items)}

    for new_item in new_items:
        name = new_item.get("name", "").strip()
        qty = int(new_item.get("quantity") or 1)
        if not name:
            continue
        key = name.lower()
        if key in existing_by_name:
            # Update quantity if re-specified
            existing_items[existing_by_name[key]]["quantity"] = qty
        else:
            existing_items.append({"name": name, "quantity": qty})

    merged["items"] = existing_items

    # active_intent — preserve from existing, never overwrite with extracted
    # (intent_node manages this field directly, not via merge)
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