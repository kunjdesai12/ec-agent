"""In-memory order scratchpad (the live 'order_params') for Aki.

One cart per session. EasyCater orders are single-restaurant, so adding an item
from a different restaurant than the current cart resets the cart first.

This is the structured state behind 'make it 2', 'remove the X', 'add a Y'. Each
mutation is a tool call (add_to_cart / update_cart_item / remove_from_cart), and
the current cart is injected into every turn as the [ORDER STATUS] block (see
format_for_prompt) so the model always sees the truth instead of trying to
remember it across turns.

Displayed prices are for user confirmation only — business-service recomputes the
authoritative total when the order is actually placed/checked out. Like the
message store, this is process-local: not shared across workers, cleared on
restart. Swap the _CARTS dict for a shared store if you scale out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CartLine:
    item_id: str
    name: str
    price: float
    quantity: int

    @property
    def line_total(self) -> float:
        return self.price * self.quantity


@dataclass
class Cart:
    restaurant_id: Optional[str] = None
    restaurant_name: str = ""
    # item_id -> line. Dict keying makes "is this already in the cart?" O(1),
    # which is what add/update/remove all need.
    lines: dict[str, CartLine] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return sum(l.line_total for l in self.lines.values())

    @property
    def item_count(self) -> int:
        return sum(l.quantity for l in self.lines.values())

    @property
    def is_empty(self) -> bool:
        return not self.lines


_CARTS: dict[str, Cart] = {}


# ── Accessors ───────────────────────────────────────────────────────────────

def get_cart(session_id: str) -> Cart:
    return _CARTS.setdefault(session_id, Cart())


def clear(session_id: str) -> None:
    """Empty the scratchpad — call on order placed, discarded, or 'start over'."""
    _CARTS.pop(session_id, None)


def summary(session_id: str) -> dict:
    cart = get_cart(session_id)
    return {
        "restaurant_id":   cart.restaurant_id,
        "restaurant_name": cart.restaurant_name,
        "items": [
            {
                "item_id":    l.item_id,
                "name":       l.name,
                "price":      l.price,
                "quantity":   l.quantity,
                "line_total": l.line_total,
            }
            for l in cart.lines.values()
        ],
        "total":      cart.total,
        "item_count": cart.item_count,
    }


# ── Mutations (each returns the updated summary for the tool result) ─────────

def add_item(
    session_id: str,
    *,
    restaurant_id: str,
    restaurant_name: str,
    item_id: str,
    name: str,
    price: float,
    quantity: int = 1,
) -> dict:
    """Add an item, or increment its quantity if already present.

    Single-restaurant rule: if the cart already holds items from a different
    restaurant, it is reset before adding (the result is flagged so the model can
    tell the user their previous items were cleared).
    """
    restaurant_id = str(restaurant_id)
    item_id = str(item_id)
    try:
        quantity = max(1, int(quantity))
    except (TypeError, ValueError):
        quantity = 1

    cart = get_cart(session_id)

    switched = False
    if cart.restaurant_id and cart.restaurant_id != restaurant_id:
        cart = Cart()
        _CARTS[session_id] = cart
        switched = True

    cart.restaurant_id = restaurant_id
    cart.restaurant_name = restaurant_name or cart.restaurant_name

    if item_id in cart.lines:
        cart.lines[item_id].quantity += quantity
    else:
        cart.lines[item_id] = CartLine(
            item_id=item_id, name=name, price=float(price or 0), quantity=quantity
        )

    return _result(session_id, note="restaurant_switched" if switched else None)


def set_quantity(session_id: str, item_id: str, quantity: int) -> dict:
    """Set an existing line's quantity. quantity <= 0 removes the line."""
    item_id = str(item_id)
    try:
        quantity = int(quantity)
    except (TypeError, ValueError):
        return {"success": False, "error": "quantity must be a whole number."}

    cart = get_cart(session_id)
    if item_id not in cart.lines:
        return {
            "success": False,
            "error": f"item_id {item_id} is not in the cart — nothing to update.",
        }

    if quantity <= 0:
        del cart.lines[item_id]
    else:
        cart.lines[item_id].quantity = quantity
    return _result(session_id)


def remove_item(session_id: str, item_id: str) -> dict:
    item_id = str(item_id)
    cart = get_cart(session_id)
    if item_id not in cart.lines:
        return {
            "success": False,
            "error": f"item_id {item_id} is not in the cart — nothing to remove.",
        }
    del cart.lines[item_id]
    return _result(session_id)


def _result(session_id: str, note: Optional[str] = None) -> dict:
    out = summary(session_id)
    out["success"] = True
    if note:
        out["note"] = note
    return out


# ── [ORDER STATUS] block injected into each turn ────────────────────────────

def format_for_prompt(session_id: str) -> Optional[str]:
    """Render the current cart as the [ORDER STATUS] block, or None if empty.

    The orchestrator injects this as a system message every turn so the model
    always reasons against the real cart instead of its own memory.
    """
    cart = get_cart(session_id)
    if cart.is_empty:
        return None

    lines = [
        "[ORDER STATUS] CURRENT ORDER (in progress — NOT yet placed):",
        f"Restaurant: {cart.restaurant_name} (restaurant_id={cart.restaurant_id})",
    ]
    for l in cart.lines.values():
        lines.append(
            f"  - {l.quantity}x {l.name} "
            f"(item_id={l.item_id}) @ \u20b9{l.price:.0f} = \u20b9{l.line_total:.0f}"
        )
    lines.append(f"Total: \u20b9{cart.total:.0f}")
    lines.append(
        "This is the authoritative cart. Use these item_ids and quantities for "
        "any edit; do not re-add items already listed here."
    )
    return "\n".join(lines)