"""Semantic match node — resolves fuzzy restaurant + item names to real IDs.

Separately searches:
1. restaurant_name_embedding — find best matching restaurant
2. food_item_embedding       — find best matching items within that restaurant
3. cuisine_embedding         — fallback if no restaurant match

Clear not-found messages if restaurant or items can't be matched above threshold.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer

from agent.app.config import get_settings
from agent.app.logging_setup import get_logger
from agent.app.state.order_state import OrderParams

log = get_logger(__name__)

# Minimum similarity thresholds
RESTAURANT_THRESHOLD = 0.5
ITEM_THRESHOLD = 0.4

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        s = get_settings()
        _model = SentenceTransformer(s.embedding_model)
    return _model


def _get_conn():
    s = get_settings()
    return psycopg2.connect(
        host=s.pg_host,
        dbname=s.pg_db,
        user=s.pg_user,
        password=s.pg_password,
        port=s.pg_port,
    )


def _embed(text: str) -> str:
    model = _get_model()
    vec = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
    return "[" + ",".join(f"{v:.6f}" for v in vec.tolist()) + "]"


def _match_restaurant(restaurant_name: str, top_k: int = 3) -> list[dict]:
    """Search restaurant_name_embedding for best matching restaurants."""
    emb = _embed(restaurant_name)
    sql = """
        SELECT DISTINCT ON (restaurant_id)
            restaurant_id,
            restaurant_name,
            cuisine_name,
            1 - (restaurant_name_embedding <=> %s::vector) AS similarity
        FROM restaurant_embeddings
        WHERE restaurant_name_embedding IS NOT NULL
        ORDER BY restaurant_id, restaurant_name_embedding <=> %s::vector
        LIMIT %s
    """
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (emb, emb, top_k))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def _match_items_in_restaurant(
    item_names: list[str],
    restaurant_id: int,
    top_k_per_item: int = 2,
) -> dict[str, list[dict]]:
    """Search food_item_embedding within a specific restaurant."""
    results: dict[str, list[dict]] = {}
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            for item_name in item_names:
                emb = _embed(item_name)
                sql = """
                    SELECT
                        menu_item_id,
                        food_item,
                        price,
                        cuisine_name,
                        1 - (food_item_embedding <=> %s::vector) AS similarity
                    FROM restaurant_embeddings
                    WHERE restaurant_id = %s
                      AND food_item_embedding IS NOT NULL
                    ORDER BY food_item_embedding <=> %s::vector
                    LIMIT %s
                """
                cur.execute(sql, (emb, restaurant_id, emb, top_k_per_item))
                results[item_name] = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    return results


def _match_cuisine(cuisine_name: str, top_k: int = 3) -> list[dict]:
    """Search cuisine_embedding as fallback."""
    emb = _embed(cuisine_name)
    sql = """
        SELECT DISTINCT ON (restaurant_id)
            restaurant_id,
            restaurant_name,
            cuisine_name,
            1 - (cuisine_embedding <=> %s::vector) AS similarity
        FROM restaurant_embeddings
        WHERE cuisine_embedding IS NOT NULL
        ORDER BY restaurant_id, cuisine_embedding <=> %s::vector
        LIMIT %s
    """
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (emb, emb, top_k))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


async def semantic_match_node(state: dict[str, Any]) -> dict[str, Any]:
    """Resolve order_params names to real IDs via separate pgvector searches."""
    order_params: OrderParams = state.get("order_params", {})
    restaurant_name = order_params.get("restaurant_name", "")
    items = order_params.get("items", [])

    # ── Step 1: Match restaurant ──────────────────────────────────────────
    restaurant_matches = await asyncio.to_thread(_match_restaurant, restaurant_name)

    # Filter by threshold
    good_restaurants = [
        r for r in restaurant_matches
        if float(r["similarity"]) >= RESTAURANT_THRESHOLD
    ]

    if not good_restaurants:
        msg = (
            f"I couldn't find a restaurant matching '{restaurant_name}' "
            f"on EasyCater. Could you check the name or try a different restaurant?"
        )
        log.warning("semantic_match_no_restaurant", extra={"query": restaurant_name})
        return {
            "semantic_matches": [],
            "messages": [{"role": "system", "content": msg}],
            "final_text": msg,
        }

    best_restaurant = good_restaurants[0]
    restaurant_id = int(best_restaurant["restaurant_id"])
    matched_restaurant_name = best_restaurant["restaurant_name"]
    restaurant_similarity = float(best_restaurant["similarity"])

    log.info(
        "restaurant_matched",
        extra={
            "query": restaurant_name,
            "matched": matched_restaurant_name,
            "restaurant_id": restaurant_id,
            "similarity": round(restaurant_similarity, 3),
        },
    )

    # ── Step 2: Match each item within the restaurant ─────────────────────
    item_names = [item["name"] for item in items]
    item_matches = await asyncio.to_thread(
        _match_items_in_restaurant, item_names, restaurant_id
    )

    # ── Step 3: Build context + semantic_matches ──────────────────────────
    lines = [
        "## Resolved order parameters (use these exact IDs in place_order):",
        f"Restaurant: {matched_restaurant_name} "
        f"(restaurant_id={restaurant_id}, similarity={restaurant_similarity:.2f})",
        "",
        "Matched menu items:",
    ]

    semantic_matches = []
    not_found_items = []

    for order_item in items:
        item_name = order_item["name"]
        quantity = order_item.get("quantity", 1)
        matches = item_matches.get(item_name, [])

        # Filter by threshold
        good_matches = [m for m in matches if float(m["similarity"]) >= ITEM_THRESHOLD]

        if good_matches:
            best = good_matches[0]
            lines.append(
                f"  ✓ '{item_name}' (qty {quantity}) → "
                f"{best['food_item']} "
                f"(item_id={best['menu_item_id']}, "
                f"price=₹{best['price']:.0f}, "
                f"similarity={best['similarity']:.2f})"
            )
            semantic_matches.append({
                "requested_name": item_name,
                "quantity": quantity,
                "restaurant_id": restaurant_id,
                "restaurant_name": matched_restaurant_name,
                "item_id": best["menu_item_id"],
                "item_name": best["food_item"],
                "price": float(best["price"]),
                "similarity": float(best["similarity"]),
            })
        else:
            not_found_items.append(item_name)
            lines.append(
                f"  ✗ '{item_name}' (qty {quantity}) → "
                f"NOT FOUND in {matched_restaurant_name}"
            )

    # ── Step 4: Handle not-found items ────────────────────────────────────
    if not_found_items:
        not_found_str = ", ".join(f"'{i}'" for i in not_found_items)
        lines += [
            "",
            f"NOT FOUND: {not_found_str} not available at {matched_restaurant_name}.",
            f"Tell the user clearly that {not_found_str} could not be found at "
            f"{matched_restaurant_name} and ask if they'd like to:",
            "  a) Choose a different item from this restaurant",
            "  b) Search a different restaurant",
        ]

    if semantic_matches:
        lines += [
            "",
            "Instructions:",
            "1. Use EXACT restaurant name from above — never paraphrase or invent names.",
            "2. If similarity < 0.6 for any item, flag the mismatch and ask user to confirm.",
            "3. Mention any NOT FOUND items clearly to the user.",
            "4. Use the item_id and restaurant_id values above — do not guess or invent IDs.",
        ]
    else:
        # All items not found
        msg = (
            f"I found {matched_restaurant_name} but couldn't find "
            f"{', '.join(not_found_items)} on their menu. "
            f"Would you like to see what's available there, or try a different restaurant?"
        )
        return {
            "semantic_matches": [],
            "messages": [{"role": "system", "content": msg}],
            "final_text": msg,
        }

    context = "\n".join(lines)
    log.info(
        "semantic_match_done",
        extra={
            "restaurant_id": restaurant_id,
            "restaurant_name": matched_restaurant_name,
            "items_matched": len(semantic_matches),
            "items_not_found": len(not_found_items),
            "items_requested": len(items),
        },
    )

    return {
        "semantic_matches": semantic_matches,
        "messages": [{"role": "system", "content": context}],
    }
