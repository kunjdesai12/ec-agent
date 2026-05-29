"""Tool registry: JSON schemas sent to the model + mock handlers.

When you're ready to wire real services:
  - get_menu / search_food / get_restaurant_details -> business-service
  - place_order / get_order_status / cancel_order   -> users-service / order-service
  - get_payment_methods                              -> payout-service

Replace the body of each handler with an httpx call to the corresponding
internal endpoint. Schemas should stay stable so the model doesn't need
to be re-prompted.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


# ────────────────────────────────────────────────────────────────────────────
# Schemas (OpenAI-compatible function schema — vLLM converts to Hermes format)
# ────────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_food",
            "description": (
                "Semantic search over restaurants and menu items. Use when the user "
                "wants to discover food by cuisine, dish, dietary preference, or vibe. "
                "Returns ranked results with restaurant_id, item_id, name, price."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text query, e.g. 'spicy north indian veg' or 'late-night biryani'",
                    },
                    "cuisine": {"type": "string", "description": "Optional cuisine filter"},
                    "max_price": {"type": "number", "description": "Optional max price in INR"},
                    "dietary": {
                        "type": "string",
                        "enum": ["veg", "non-veg", "vegan", "jain"],
                        "description": "Optional dietary filter",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_menu",
            "description": "Fetch the full menu for a specific restaurant by restaurant_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "restaurant_id": {"type": "string"},
                    "category": {
                        "type": "string",
                        "description": "Optional category filter, e.g. 'starters', 'mains', 'desserts'",
                    },
                },
                "required": ["restaurant_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_restaurant_details",
            "description": "Get details about a specific restaurant: hours, rating, delivery time, address.",
            "parameters": {
                "type": "object",
                "properties": {"restaurant_id": {"type": "string"}},
                "required": ["restaurant_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": (
                "Place a confirmed food order. Call ONLY after confirming items, quantities, "
                "and delivery address with the user. Returns order_id and ETA."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "restaurant_id": {"type": "string"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "item_id": {"type": "string"},
                                "name": {"type": "string"},
                                "quantity": {"type": "integer", "minimum": 1},
                                "modifiers": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "e.g. ['no onion', 'extra spicy']",
                                },
                            },
                            "required": ["item_id", "name", "quantity"],
                        },
                    },
                    "delivery_address_id": {"type": "string"},
                    "payment_method": {
                        "type": "string",
                        "enum": ["upi", "card", "cod", "wallet"],
                    },
                },
                "required": ["restaurant_id", "items", "delivery_address_id", "payment_method"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Check the status of an existing order by order_id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel an order. Only works while order is in 'placed' or 'confirmed' state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_addresses",
            "description": "List the user's saved delivery addresses. Use before place_order if address is unknown.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ────────────────────────────────────────────────────────────────────────────
# Mock handlers — replace bodies with real microservice calls later
# ────────────────────────────────────────────────────────────────────────────

async def _search_food(args: dict[str, Any]) -> dict[str, Any]:
    query = args.get("query", "")
    return {
        "results": [
            {
                "restaurant_id": "rest_042",
                "restaurant_name": "Saffron Kitchen",
                "item_id": "itm_1021",
                "item_name": "Butter Chicken",
                "price": 320,
                "cuisine": "north-indian",
                "rating": 4.4,
                "match_reason": f"matches query '{query}'",
            },
            {
                "restaurant_id": "rest_088",
                "restaurant_name": "Tandoor Tales",
                "item_id": "itm_2210",
                "item_name": "Paneer Tikka Masala",
                "price": 280,
                "cuisine": "north-indian",
                "rating": 4.2,
            },
        ]
    }


async def _get_menu(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "restaurant_id": args["restaurant_id"],
        "menu": [
            {"item_id": "itm_1021", "name": "Butter Chicken", "price": 320, "category": "mains"},
            {"item_id": "itm_1022", "name": "Garlic Naan", "price": 60, "category": "breads"},
            {"item_id": "itm_1023", "name": "Gulab Jamun (2 pc)", "price": 90, "category": "desserts"},
        ],
    }


async def _get_restaurant_details(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "restaurant_id": args["restaurant_id"],
        "name": "Saffron Kitchen",
        "rating": 4.4,
        "hours": "11:00 – 23:00",
        "delivery_eta_minutes": 35,
        "address": "Alkapuri, Vadodara",
    }


async def _place_order(args: dict[str, Any]) -> dict[str, Any]:
    order_id = f"ord_{uuid.uuid4().hex[:10]}"
    total = sum(it.get("quantity", 1) * 280 for it in args.get("items", []))
    return {
        "order_id": order_id,
        "status": "confirmed",
        "eta_minutes": 35,
        "total_inr": total,
        "placed_at": datetime.now(timezone.utc).isoformat(),
    }


async def _get_order_status(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "order_id": args["order_id"],
        "status": "out_for_delivery",
        "eta_minutes": 8,
        "rider_name": "Ramesh",
    }


async def _cancel_order(args: dict[str, Any]) -> dict[str, Any]:
    return {"order_id": args["order_id"], "status": "cancelled", "refund_initiated": True}


async def _get_user_addresses(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "addresses": [
            {"address_id": "addr_001", "label": "Home", "line": "12 Alkapuri, Vadodara"},
            {"address_id": "addr_002", "label": "Office", "line": "Race Course Road, Vadodara"},
        ]
    }


HANDLERS: dict[str, ToolHandler] = {
    "search_food": _search_food,
    "get_menu": _get_menu,
    "get_restaurant_details": _get_restaurant_details,
    "place_order": _place_order,
    "get_order_status": _get_order_status,
    "cancel_order": _cancel_order,
    "get_user_addresses": _get_user_addresses,
}


async def execute_tool(name: str, arguments: str | dict[str, Any]) -> str:
    """Execute a tool by name. Returns JSON string suitable for tool-result message."""
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"invalid JSON arguments: {e}"})
    else:
        args = arguments

    handler = HANDLERS.get(name)
    if handler is None:
        return json.dumps({"error": f"unknown tool: {name}"})

    try:
        result = await handler(args)
        return json.dumps(result)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"tool execution failed: {e}"})
