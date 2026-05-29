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
from typing import Any, Awaitable, Callable, Optional, List, Dict
import httpx
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

BASE_URL = os.getenv("TOOL_BASE_URL")

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class CustomerDetails(BaseModel):
    name: str
    email: str
    contact: str

class DeliveryAddress(BaseModel):
    latitude: str
    longitude: str
    address_line_1: str
    city: str
    state: str
    pincode: str
    AddressID: Optional[str] = "100"
 
class PlaceOrderInput(BaseModel):
    restaurant_id: int = Field(..., description="Restaurant ID")
    menu_item_ids: List[int] = Field(..., description="Menu item IDs")
    price: float = Field(..., description="Base item price")
    order_type: str = Field(..., description="delivery / pickup / dinein")
    customer_details: CustomerDetails
    delivery_address: DeliveryAddress
    quantity: int = Field(1, description="Quantity per item")
    is_cod: bool = Field(False, description="Cash on Delivery")
    order_instructions: str = Field("", description="Special instructions")
    jwt_token: str = Field(..., description="JWT bearer token")

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
                "Create a food delivery order by creating/updating the cart, "
                "calculating totals, and generating the final order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "restaurant_id": {
                        "type": "integer",
                        "description": "Unique restaurant ID where the order should be placed."
                    },
                    "menu_item_ids": {
                        "type": "array",
                        "items": {
                            "type": "integer"
                        },
                        "description": "List of menu item IDs selected by the customer."
                    },
                    "price": {
                        "type": "number",
                        "description": "Base price of each menu item."
                    },
                    "order_type": {
                        "type": "string",
                        "enum": ["delivery", "pickup", "dinein"],
                        "description": "Type of order being placed."
                    },
                    "customer_details": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Customer full name."
                            },
                            "email": {
                                "type": "string",
                                "description": "Customer email address."
                            },
                            "phone": {
                                "type": "string",
                                "description": "Customer contact number."
                            }
                        },
                        "required": ["name", "email", "phone"]
                    },
                    "delivery_address": {
                        "type": "object",
                        "properties": {
                            "address_line_1": {
                                "type": "string",
                                "description": "Primary delivery address line."
                            },
                            "address_line_2": {
                                "type": "string",
                                "description": "Secondary delivery address line."
                            },
                            "landmark": {
                                "type": "string",
                                "description": "Nearby landmark for easy navigation."
                            },
                            "city": {
                                "type": "string",
                                "description": "City name."
                            },
                            "state": {
                                "type": "string",
                                "description": "State name."
                            },
                            "pincode": {
                                "type": "string",
                                "description": "Postal or ZIP code."
                            },
                            "latitude": {
                                "type": "number",
                                "description": "Latitude coordinate."
                            },
                            "longitude": {
                                "type": "number",
                                "description": "Longitude coordinate."
                            }
                        },
                        "required": [
                            "address_line_1",
                            "city",
                            "state",
                            "pincode",
                            "latitude",
                            "longitude"
                        ]
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Quantity applied uniformly to each menu item.",
                        "default": 1
                    },
                    "is_cod": {
                        "type": "boolean",
                        "description": "Whether payment mode is Cash on Delivery.",
                        "default": False
                    },
                    "order_instructions": {
                        "type": "string",
                        "description": "Special cooking or delivery instructions."
                    },
                    "jwt_token": {
                        "type": "string",
                        "description": "JWT access token for authenticated API requests."
                    }
                },
                "required": [
                    "restaurant_id",
                    "menu_item_ids",
                    "price",
                    "order_type",
                    "customer_details",
                    "delivery_address",
                    "jwt_token"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Retrieve complete order details and live order status "
                "including payment, delivery, tracking, and restaurant information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "Unique order ID"
                    },
                    "jwt_token": {
                        "type": "string",
                        "description": "JWT bearer token for authenticated user"
                    }
                },
                "required": ["order_id", "jwt_token"]
            }
        }
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

async def fetch_api(
    url: str,
    method: str = "GET",
    params: Dict = None,
    json: Dict = None,
    headers: Dict = None,
) -> Dict:

    async with httpx.AsyncClient(timeout=20.0) as client:

        if method.upper() == "GET":
            response = await client.get(url, params=params, headers=headers)

        elif method.upper() == "POST":
            response = await client.post(url, json=json, headers=headers)

        elif method.upper() == "DELETE":
            response = await client.request(
                "DELETE",
                url,
                json=json,
                headers=headers
            )

        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()

        return response.json()
    
    
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


async def _place_order(
    restaurant_id: int,
    menu_item_ids: List[int],
    price: float,
    order_type: str,
    customer_details: CustomerDetails,
    delivery_address: DeliveryAddress,
    quantity: int = 1,
    is_cod: bool = False,
    order_instructions: str = "",
    jwt_token: str = "",
) -> Dict[str, Any]:
    """Create a cart and place a food delivery order.
 
    Handles the full order lifecycle — cart creation, pricing, taxes,
    delivery fees, and order generation — in a single call.
 
    Args:
        restaurant_id: Restaurant ID where the order should be placed.
        menu_item_ids: List of menu item IDs to order.
        customer_details: Customer name, email, and contact number.
        delivery_address: Full delivery address including coordinates and pincode.
        quantity: Quantity applied uniformly to each menu item. Defaults to 1.
        is_cod: Whether to use Cash on Delivery. Defaults to False (online payment).
        order_instructions: Optional special instructions for the restaurant.
    """
    cart_payload = {
        "restaurants": [
            {
                "restaurant_id": restaurant_id,
                "menu_items": [
                    {"menu_item_id": item_id, "quantity": quantity, "baseprice": price, "totalprice": price * quantity, "customizations": []}
                    for item_id in menu_item_ids
                ],
            }
        ]
    }
    headers = {
    "Authorization": f"Bearer {jwt_token}"
    }
    
    cart_response = await fetch_api(
        f"{BASE_URL}/new-cart/add-update",
        method="POST",
        json=cart_payload,
        headers=headers
    )
    cart_data = cart_response.get("data", {})
    cart_details = cart_data.get("cart_details") or cart_data.get("cart") or {}

    now_utc = datetime.now(timezone.utc).isoformat()
    order_payload = {
        "cart_details": cart_details,
        "customer_details": customer_details.model_dump(),
        "delivery_address": delivery_address.model_dump(),
        "order_type": order_type,
        "apply_coupons": [],
        "order_instructions": order_instructions,
        "company_id": "",
        "order_time": now_utc,
        "order_placed_time": now_utc,
        "order_tip": 0,
        "delivery_instructions": "",
        "loyalty_points": 0,
        "dinein_booking_id": None,
        "is_cod": is_cod,
    }
    print("Order Payload:", order_payload)
    order_response = await fetch_api(f"{BASE_URL}/new-payment/create-order", method="POST", json=order_payload, headers=headers)
    print("Order Response:", order_response)
 
    return {
        "success": True,
        "cart_response": cart_response,
        "order_response": order_response,
    }


async def _get_order_status(
    order_id: int,
    jwt_token: str,
) -> Dict[str, Any]:
    """
    Retrieve complete order details and live order status using an order ID.

    Useful for:
    - order progress
    - payment status
    - delivery updates
    - ordered items
    - restaurant details
    - tracking information
    - customer order history

    Args:
        order_id: Unique order ID
        jwt_token: User JWT bearer token
    """

    url = f"{BASE_URL}/order/{order_id}"

    headers = {
        "Authorization": f"Bearer {jwt_token}"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:

        response = await client.get(
            url,
            headers=headers
        )

        response.raise_for_status()

        return response.json()
    

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
