from wsgiref import headers

import httpx
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from datetime import datetime, timezone
from pydantic import BaseModel, Field

#BASE_URL = "https://api.easycatering.com/v1"
BASE_URL = "http://localhost:5610/v1"

# Define Pydantic models for structured input and output data


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
    restaurant_id: int = Field(
        ...,
        description="Restaurant ID where the order should be placed"
    )

    menu_item_ids: List[int] = Field(
        ...,
        description="List of menu item IDs to order"
    )

    price: float = Field(
        ...,
        description="Base price of each menu item"
    )

    order_type: str = Field(
        ...,
        description="delivery / pickup / dinein"
    )

    customer_details: CustomerDetails = Field(
        ...,
        description="Customer name, email, and contact"
    )

    delivery_address: DeliveryAddress = Field(
        ...,
        description="Full delivery address with coordinates"
    )

    quantity: int = Field(
        default=1,
        description="Quantity applied to each menu item"
    )

    is_cod: bool = Field(
        default=False,
        description="Set True to use Cash on Delivery"
    )

    order_instructions: str = Field(
        default="",
        description="Special instructions for the restaurant"
    )

    jwt_token: str = Field(
        ...,
        description="JWT bearer token for authenticated user"
    )


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


@tool
async def search_restaurants(query: str) -> List[Dict[str, Any]]:
    """Search restaurants by name, cuisine, or keyword.

    Returns matching restaurant details including cuisines, location,
    delivery information, ratings, availability, restaurant type,
    and contact information.

    Args:
        query: Restaurant search keyword or name.
               Example: "pizza", "chinese", "McDonald's"
    """
    url = f"{BASE_URL}/search/master"

    data = await fetch_api(url, method="POST", json={"query": query})
    restaurants = data.get("data", {}).get("restaurants", [])

    return [
        {
            "restaurant_id": r.get("restaurantID"),
            "name": r.get("restaurantName"),
            "cuisines": r.get("restaurantCuisines", []),
            "food_type": r.get("foodType"),
            "restaurant_type": r.get("restaurant_type"),
            "is_open": r.get("isOpen"),
            "rating": r.get("ratings"),
            "total_ratings": r.get("totalRatings"),
            "address": r.get("address"),
            "area": r.get("area"),
            "city": r.get("city"),
            "state": r.get("state"),
            "delivery_time": r.get("deliveryInfo", {}).get("timeLabel"),
            "delivery_distance": r.get("deliveryInfo", {}).get("distanceLabel"),
            "phone": r.get("phone"),
            "latitude": r.get("latitude"),
            "longitude": r.get("longitude"),
            "image_urls": r.get("restaurantImageURLs", []),
        }
        for r in restaurants
    ]


@tool
async def search_food_items(query: str) -> List[Dict[str, Any]]:
    """Search food or menu items by name.

    Returns matching dishes with pricing, restaurant details,
    food type, spice level, and images.

    Args:
        query: Food item name to search for.
               Example: "pizza", "pani puri", "burger", "dosa"
    """
    url = f"{BASE_URL}/search/master"

    data = await fetch_api(url, method="POST", json={"query": query})
    items = data.get("data", {}).get("menuItems", [])

    return [
        {
            "menu_item_id": item.get("menu_item_id"),
            "title": item.get("title"),
            "price": float(item.get("price", 0)),
            "image_url": item.get("image_url"),
            "food_type": item.get("food_type"),
            "spice_level": item.get("spice_level"),
            "business_id": item.get("business_id"),
            "restaurant_name": item.get("restaurant", {}).get("restaurant_name"),
            "currency": "INR",
            "is_veg": item.get("food_type") == "1",
        }
        for item in items
    ]


@tool
async def get_order_status(
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

@tool(args_schema=PlaceOrderInput)
async def place_order(
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

    await fetch_api(
        f"{BASE_URL}/new-cart/delete-all",
        method="DELETE",
        headers=headers
    )
    
    cart_response = await fetch_api(
        f"{BASE_URL}/new-cart/add-update",
        method="POST",
        json=cart_payload,
        headers=headers
    )
    cart_data = cart_response.get("data", {})
    cart_details_2 = cart_data.get("cart_details") or cart_data.get("cart") or {}

    now_utc = datetime.now(timezone.utc).isoformat()
    order_payload = {
        "cart_details": cart_details_2,
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

TOOLS = [
    search_restaurants,
    search_food_items,
    get_order_status,
    place_order,
]

