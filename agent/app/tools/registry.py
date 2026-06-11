"""Tool registry: JSON schemas sent to the model + handlers.

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

from agent.app.config import get_settings

load_dotenv()

BASE_URL = get_settings().tool_base_url

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


# class CustomerDetails(BaseModel):
#     name: str
#     email: str
#     contact: str
class DeliveryAddress(BaseModel):
    latitude: str
    longitude: str
    address_line_1: str
    city: str
    state: str
    pincode: str
    AddressID: Optional[str] = "100"

# class PlaceOrderInput(BaseModel):
#     restaurant_id: int = Field(..., description="Restaurant ID")
#     menu_item_ids: List[int] = Field(..., description="Menu item IDs")
#     price: float = Field(..., description="Base item price")
#     order_type: str = Field(..., description="delivery / pickup / dinein")
#     customer_details: CustomerDetails
#     delivery_address: DeliveryAddress
#     quantity: int = Field(1, description="Quantity per item")
#     is_cod: bool = Field(False, description="Cash on Delivery")
#     order_instructions: str = Field("", description="Special instructions")
#     jwt_token: str = Field(..., description="JWT bearer token")

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
        "description": "Fetch the complete menu of a restaurant by restaurant_id. Returns clean list of menu items with item_id, name, price, category, description and image.",
        "parameters": {
            "type": "object",
            "properties": {
                "restaurant_id": {
                    "type": "string",
                    "description": "The unique ID of the restaurant (e.g. '30', '25')"
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
        "description": "Get professional restaurant details including name, address, rating, cuisines and operating hours.",
        "parameters": {
            "type": "object",
            "properties": {
                "restaurant_id": {
                    "type": "string",
                    "description": "The unique ID of the restaurant (e.g. '25', '30')"
                },
            },
            "required": ["restaurant_id"]
        },
    },
    },
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": (
                "Create a food delivery order by creating/updating the cart, "
                "calculating totals, and generating the final order. "
                "Always call get_user_addresses first to resolve delivery_address."
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
                    # "customer_details": {
                    #     "type": "object",
                    #     "properties": {
                    #         "name": {
                    #             "type": "string",
                    #             "description": "Customer full name."
                    #         },
                    #         "email": {
                    #             "type": "string",
                    #             "description": "Customer email address."
                    #         },
                    #         "phone": {
                    #             "type": "string",
                    #             "description": "Customer contact number."
                    #         }
                    #     },
                    #     "required": ["name", "email", "phone"]
                    # },
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
                    # "jwt_token": {
                    #     "type": "string",
                    #     "description": "JWT access token for authenticated API requests."
                    # }
                },
                "required": [
                    "restaurant_id",
                    "menu_item_ids",
                    "price",
                    "order_type",
                    "delivery_address",
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
                    # "jwt_token": {
                    #     "type": "string",
                    #     "description": "JWT bearer token for authenticated user"
                    # }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_active_orders",
            "description": (
                "Fetch the current user's active orders. "
                "Call this FIRST when user wants to cancel but hasn't provided an order_id. "
                "Returns list of active orders with order_id, status, restaurant name, "
                "items, total amount and scheduled time. "
                "If multiple orders exist, show them to user and ask which one to cancel."
            ),
            # "parameters": {
            #     "type": "object",
            #     "properties": {
            #         "jwt_token": {
            #             "type": "string",
            #             "description": "JWT bearer token for the authenticated user."
            #         }
            #     },
            #     "required": ["jwt_token"]
            # }
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": (
                "Cancel a food order by order_id. "
                "orderId goes in the URL path, reason goes in the request body. "
                "Backend handles all eligibility checks internally: "
                "  - Instant orders: cancellable within 4 hours of scheduled time. "
                "  - Catering orders: cancellable only if 48+ hours before scheduled time. "
                "  - Refund is triggered automatically via Razorpay. "
                "If cancellation is not allowed, backend returns a clear error — "
                "pass it directly to the user. "
                "Always call get_active_orders first if you don't have the order_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "The order ID to cancel. Get this from get_active_orders if unknown."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for cancellation. Ask the user if not provided."
                    },
                    # "jwt_token": {
                    #     "type": "string",
                    #     "description": "JWT bearer token for the authenticated user."
                    # }
                },
                "required": ["order_id", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_addresses",
            "description": (
                "List all the user's saved delivery addresses (e.g. Home, Work, or any custom label). "
                "Always call this before place_order if the user hasn't explicitly provided a full address. "
                "Returns address_id, label, full_address, landmark, floor_no, house_number, city, state, "
                "pincode, latitude, longitude, and is_primary — all fields needed directly for place_order."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ────────────────────────────────────────────────────────────────────────────
# Helpers
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


def _auth_headers(jwt_token: str) -> dict[str, str]:
    """Build Authorization header dict from a JWT token."""
    return {"Authorization": f"Bearer {jwt_token}"}


# ────────────────────────────────────────────────────────────────────────────
# Handlers
# ────────────────────────────────────────────────────────────────────────────

async def _search_food(args: dict[str, Any]) -> dict[str, Any]:
    query = args.get("query", "")
    cuisine = args.get("cuisine")
    max_price = args.get("max_price")

    # Parse query into fields — single field goes as menu_item
    # unless it clearly looks like a restaurant name (handled by LLM upstream)
    payload = {
        "menu_item":    query,
        "cuisine":      cuisine,
        "max_price":    max_price,
        "limit":        5,
    }

    settings = get_settings()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{settings.semantic_search_url}/search",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as e:
        return {"error": f"Semantic search unavailable: {e}"}

    top_matches = data.get("top_matches", [])

    if not top_matches:
        return {"results": [], "message": "No matches found"}

    # Shape results for the LLM
    results = [
        {
            "restaurant_id":   match.get("restaurant_id"),
            "restaurant_name": match.get("restaurant_name"),
            "item_id":         match.get("menu_item_id"),
            "item_name":       match.get("food_item"),
            "price":           match.get("price"),
            "cuisine":         match.get("cuisine_name"),
            "rating":          match.get("rating"),
            "confidence":      match.get("similarity"),
        }
        for match in top_matches
    ]

    return {"results": results}


async def _get_menu(args: dict[str, Any]) -> dict[str, Any]:
    """
    Final Production Version - Clean menu with only restaurant_id
    """
    restaurant_id = args.get("restaurant_id")

    if not restaurant_id:
        return {
            "error": "restaurant_id is required",
            "status": "failed"
        }

    params = {
        "restaurantId": restaurant_id,
        "listing_type": "instant"
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get(
                f"{BASE_URL}/new-menu-item/getMenuItemsForUserByRestaurantId",
                params=params,
            )
            
            response.raise_for_status()
            data = response.json()

            menu_data = data.get("data", {})
            raw_items = menu_data.get("menuItems", [])
            categories = menu_data.get("menuCategories", [])

            # Category mapping
            category_map = {
                cat.get("menu_category_id"): cat.get("category_name", "Uncategorized")
                for cat in categories
            }

            # Clean menu items
            cleaned_menu = []
            for item in raw_items[:100]:
                cleaned_menu.append({
                    "item_id": str(item.get("menu_item_id")),
                    "name": item.get("title", "Unknown Item"),
                    "price": float(item.get("price") or 0),
                    "category": category_map.get(item.get("menu_category_id"), "Uncategorized"),
                    "description": item.get("description", ""),
                    "image_url": item.get("image_url")
                })

            return {
                "restaurant_id": restaurant_id,
                "menu": cleaned_menu,
                "total_items": len(cleaned_menu),
                "success": True
            }

        except httpx.HTTPStatusError as e:
            return {
                "error": f"API Error: {e.response.status_code}",
                "status": "failed"
            }
        except Exception as e:
            return {
                "error": f"Failed to fetch menu: {str(e)}",
                "status": "failed"
            }


async def _get_restaurant_details(args: dict[str, Any]) -> dict[str, Any]:
    """
    Professional version: Returns real restaurant details from API.
    """
    restaurant_id = args.get("restaurant_id")
    
    if not restaurant_id:
        return {"error": "restaurant_id is required", "status": "failed"}

    params = {"restaurantId": restaurant_id}

    async with httpx.AsyncClient(timeout=12.0) as client:
        try:
            response = await client.get(
                f"{BASE_URL}/new-menu-item/getMenuItemsForUserByRestaurantId",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            restaurant = data.get("data", {}).get("restaurantDetails", {})
            operating_hours = restaurant.get("operating_hours", {})

            # Current time in IST
            from datetime import timedelta
            IST = timezone(timedelta(hours=5, minutes=30))
            now = datetime.now(IST)
            current_day = now.strftime("%A").lower()      # e.g. "saturday"
            current_time = now.strftime("%H:%M")          # e.g. "14:30"

            # Build per-day display + check if open right now
            hours_list = []
            is_currently_open = False

            for day, info in operating_hours.items():
                if not info.get("isOpen"):
                    continue

                slots = info.get("hours", [])
                if not slots:
                    continue

                # Handle ALL slots, not just first
                slot_strings = []
                for slot in slots:
                    start = slot.get("start_time", "")[:5]
                    end = slot.get("end_time", "")[:5]
                    slot_strings.append(f"{start}-{end}")

                    # Check if currently open (handles overnight like 07:30-00:30)
                    if day == current_day:
                        if end < start:
                            # Overnight slot — open if after start OR before end
                            if current_time >= start or current_time <= end:
                                is_currently_open = True
                        else:
                            if start <= current_time <= end:
                                is_currently_open = True

                hours_list.append(f"{day.capitalize()}: {' & '.join(slot_strings)}")

            hours_display = " | ".join(hours_list) if hours_list else None

            return {
                "restaurant_id": restaurant_id,
                "name": restaurant.get("restaurant_name"),
                "rating": restaurant.get("rating") or restaurant.get("catering_rating"),
                "address": restaurant.get("address"),
                "city": restaurant.get("city"),
                "state": restaurant.get("state"),
                "cuisines": data.get("data", {}).get("restaurantCuisines", []),
                "operating_hours": hours_display,
                "is_currently_open": is_currently_open,
                "status": restaurant.get("status"),
                "success": True
            }

        except httpx.HTTPStatusError as e:
            return {"error": f"API Error: {e.response.status_code}", "status": "failed"}
        except Exception as e:
            return {"error": f"Failed to fetch restaurant details: {str(e)}", "status": "failed"}


async def _place_order(args: dict[str, Any]) -> dict[str, Any]:
    """Create a cart and place a food delivery order.

    jwt_token is injected by tools_node from GraphState — never passed by the LLM.
    customer_details are derived server-side from the JWT token.
    """
    restaurant_id      = args.get("restaurant_id")
    menu_item_ids      = args.get("menu_item_ids", [])
    price              = args.get("price", 0)
    order_type         = args.get("order_type", "delivery")
    delivery_address   = args.get("delivery_address", {})
    quantity           = args.get("quantity", 1)
    is_cod             = args.get("is_cod", False)
    order_instructions = args.get("order_instructions", "")
    jwt_token          = args.get("jwt_token", "")

    if not jwt_token:
        return {"success": False, "error": "Authentication required. Please log in."}

    headers = _auth_headers(jwt_token)

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

    try:
        cart_response = await fetch_api(
            f"{BASE_URL}/new-cart/add-update",
            method="POST",
            json=cart_payload,
            headers=headers,
        )
        cart_data = cart_response.get("data", {})
        cart_details = cart_data.get("cart_details") or cart_data.get("cart") or {}

        now_utc = datetime.now(timezone.utc).isoformat()
        order_payload = {
            "cart_details":         cart_details,
            "delivery_address":     delivery_address,
            "order_type":           order_type,
            "apply_coupons":        [],
            "order_instructions":   order_instructions,
            "company_id":           "",
            "order_time":           now_utc,
            "order_placed_time":    now_utc,
            "order_tip":            0,
            "delivery_instructions": "",
            "loyalty_points":       0,
            "dinein_booking_id":    None,
            "is_cod":               is_cod,
        }

        print("Order Payload:", order_payload)

        order_response = await fetch_api(
            f"{BASE_URL}/new-payment/create-order",
            method="POST",
            json=order_payload,
            headers=headers,
        )

        print("Order Response:", order_response)

        return {
            "success":        True,
            "cart_response":  cart_response,
            "order_response": order_response,
        }

    except httpx.HTTPStatusError as e:
        try:
            error_msg = e.response.json().get("message") or e.response.text
        except Exception:
            error_msg = e.response.text
        return {"success": False, "error": f"Order failed: {error_msg}"}
    except Exception as e:
        return {"success": False, "error": f"Order failed: {str(e)}"}


async def _get_order_status(args: dict[str, Any]) -> dict[str, Any]:
    """Retrieve complete order details and live status.

    jwt_token is injected by tools_node from GraphState.
    """
    order_id  = args.get("order_id")
    jwt_token = args.get("jwt_token", "")

    if not order_id:
        return {"error": "order_id is required", "success": False}
    if not jwt_token:
        return {"error": "Authentication required. Please log in.", "success": False}

    headers = _auth_headers(jwt_token)

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                f"{BASE_URL}/order/{order_id}",
                headers=headers,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"API Error: {e.response.status_code}", "success": False}
        except Exception as e:
            return {"error": f"Failed to fetch order status: {str(e)}", "success": False}


async def _get_active_orders(args: dict[str, Any]) -> dict[str, Any]:
    """Fetch the current user's active orders.

    jwt_token is injected by tools_node from GraphState.
    """
    jwt_token = args.get("jwt_token", "")

    if not jwt_token:
        return {"error": "Authentication required. Please log in.", "success": False}

    headers = _auth_headers(jwt_token)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{BASE_URL}/order/user/active-order",
                params={"order_type": "instant"},
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        orders_raw = (
            data.get("data")
            or data.get("orders")
            or data.get("result")
            or []
        )

        if not orders_raw:
            return {
                "success": True,
                "active_orders": [],
                "count": 0,
                "message": "No active orders found."
            }

        active_orders = []
        for order in orders_raw:
            active_orders.append({
                "order_id":        order.get("order_id") or order.get("id"),
                "status":          order.get("status", "unknown"),
                "is_cod":          order.get("is_cod", False),
                "grand_total":     order.get("grand_total") or order.get("total_amount", 0),
                "scheduled_date":  order.get("order_schedule_date", ""),
                "scheduled_time":  order.get("order_schedule_time", ""),
                "restaurant_name": (
                    order.get("restaurant", {}).get("restaurant_name")
                    or order.get("restaurant_name", "Unknown Restaurant")
                ),
                "items": [
                    {
                        "name":     item.get("title") or item.get("name", ""),
                        "quantity": item.get("quantity", 1),
                        "price":    item.get("price", 0),
                    }
                    for item in (order.get("order_items") or order.get("items") or [])
                ],
                "placed_at": order.get("createdAt") or order.get("created_at", ""),
            })

        return {
            "success":       True,
            "active_orders": active_orders,
            "count":         len(active_orders)
        }

    except httpx.HTTPStatusError as e:
        return {"success": False, "error": f"API error {e.response.status_code}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to fetch active orders: {str(e)}"}


async def _cancel_order(args: dict[str, Any]) -> dict[str, Any]:
    """Cancel a food order.

    jwt_token is injected by tools_node from GraphState.
    """
    order_id  = args.get("order_id")
    reason    = args.get("reason", "").strip()
    jwt_token = args.get("jwt_token", "")

    if not order_id:
        return {"success": False, "error": "order_id is required."}
    if not reason:
        return {"success": False, "error": "reason is required. Please ask the user why they want to cancel."}
    if not jwt_token:
        return {"success": False, "error": "Authentication required. Please log in."}

    headers = _auth_headers(jwt_token)

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                f"{BASE_URL}/cancellation/orders/{order_id}/cancel",  # orderId in path
                json={"reason": reason},                               # only reason in body
                headers=headers
            )

        if response.status_code == 400:
            try:
                error_msg = response.json().get("message") or "Cancellation not allowed."
            except Exception:
                error_msg = response.text or "Cancellation not allowed."
            return {"success": False, "cancelled": False, "error": error_msg}

        if response.status_code == 404:
            return {"success": False, "cancelled": False, "error": "Order not found."}

        response.raise_for_status()

        return {
            "success":     True,
            "cancelled":   True,
            "order_id":    order_id,
            "message":     "Order cancelled successfully.",
            "refund_note": "Refund will be processed to your original payment method within 5-7 business days.",
        }

    except httpx.HTTPStatusError as e:
        try:
            error_msg = e.response.json().get("message") or str(e.response.text)
        except Exception:
            error_msg = e.response.text
        return {"success": False, "cancelled": False, "error": f"Cancellation failed: {error_msg}"}

    except Exception as e:
        return {"success": False, "cancelled": False, "error": f"Something went wrong: {str(e)}"}

async def _get_user_addresses(args: dict[str, Any]) -> dict[str, Any]:
    """Fetch user's saved delivery addresses.

    jwt_token is injected by tools_node from GraphState.
    """
    jwt_token = args.get("jwt_token", "")

    if not jwt_token:
        return {"error": "Authentication required. Please log in.", "status": "failed"}

    headers = _auth_headers(jwt_token)

    async with httpx.AsyncClient(timeout=12.0) as client:
        try:
            response = await client.get(
                f"{BASE_URL}/users/get-address",
                headers=headers
            )
            response.raise_for_status()
            data = response.json()

            raw_addresses = data.get("data", {}).get("user_address", [])

            if not raw_addresses:
                return {
                    "success": True,
                    "addresses": [],
                    "total": 0,
                    "message": "No saved addresses found for this user."
                }

            cleaned_addresses = [
                {
                    "address_id":   addr.get("address_id"),
                    "label":        addr.get("address_name"),        # Home / Work / custom
                    "full_address": addr.get("user_address"),
                    "landmark":     addr.get("landmark"),
                    "floor_no":     addr.get("floor_no"),
                    "house_number": addr.get("house_number"),
                    "city":         addr.get("user_city"),
                    "state":        addr.get("user_state"),
                    "pincode":      addr.get("user_pincode"),
                    "latitude":     addr.get("latitude"),
                    "longitude":    addr.get("longitude"),
                    "is_primary":   addr.get("primary_address"),
                }
                for addr in raw_addresses
            ]

            return {
                "success": True,
                "addresses": cleaned_addresses,
                "total": len(cleaned_addresses)
            }

        except httpx.HTTPStatusError as e:
            return {"error": f"API Error: {e.response.status_code}", "status": "failed"}
        except Exception as e:
            return {"error": f"Failed to fetch addresses: {str(e)}", "status": "failed"}


# ────────────────────────────────────────────────────────────────────────────
# Dispatch table
# ────────────────────────────────────────────────────────────────────────────

HANDLERS: dict[str, ToolHandler] = {
    "search_food": _search_food,
    "get_menu": _get_menu,
    "get_restaurant_details": _get_restaurant_details,
    "place_order": _place_order,
    "get_order_status": _get_order_status,
    "get_active_orders": _get_active_orders,
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
