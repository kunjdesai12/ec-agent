"""pgvector + static-rule retriever for EasyCater Aki.

Restaurant and item resolution uses deterministic SQL rules (exact → contains →
trigram) rather than vector similarity. Vector search is reserved for pure
discovery queries where no restaurant or item name is known.

Matching hierarchy (applied in order, stops at first hit):
  1. Exact match         LOWER(col) = LOWER(input)
  2. Contains match      LOWER(col) LIKE '%input%'
  3. Trigram match       pg_trgm similarity() > threshold
  4. Not found           → tell user, do not proceed
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer

from agent.app.config import get_settings
from agent.app.logging_setup import get_logger

log = get_logger(__name__)

# Trigram similarity threshold for fuzzy name matching.
# 0.3 is permissive enough for "honest" → "Honest Restaurant" and
# typos like "honnest", but won't match totally unrelated names.
_RESTAURANT_TRGM_THRESHOLD = 0.3
_ITEM_TRGM_THRESHOLD       = 0.3

# Minimum vector similarity for pure discovery queries (no name known).
_VECTOR_SIM_THRESHOLD = 0.35


@dataclass
class MenuChunk:
    chunk_id:        str
    restaurant_id:   str
    restaurant_name: str
    cuisine:         str
    item_id:         str
    item_name:       str
    price:           float
    text:            str
    description:     str = ""
    rating:          Optional[float] = None
    latitude:        Optional[float] = None
    longitude:       Optional[float] = None
    metadata:        dict[str, Any] = field(default_factory=dict)


class Retriever:
    """Static-rule retriever with pgvector fallback for discovery."""

    def __init__(self) -> None:
        s = get_settings()
        self._pg_conn_str = (
            f"host={s.pg_host} "
            f"dbname={s.pg_db} "
            f"user={s.pg_user} "
            f"password={s.pg_password} "
            f"port={s.pg_port}"
        )
        self.top_k = s.rag_top_k

        # Embedding model — only used for discovery (no restaurant/item name)
        log.info("loading_embedding_model", extra={"model": s.embedding_model})
        self.model = SentenceTransformer(s.embedding_model)

        self._check_connection()
        log.info("retriever_ready", extra={"backend": "pgvector+static"})

    def _check_connection(self) -> None:
        try:
            conn = psycopg2.connect(self._pg_conn_str)
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM restaurant_embeddings")
                count = cur.fetchone()[0]
                # Confirm pg_trgm is available
                cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'")
                has_trgm = cur.fetchone() is not None
            conn.close()
            log.info("pgvector_connected", extra={
                "rows": count,
                "pg_trgm": has_trgm,
            })
            if not has_trgm:
                log.warning("pg_trgm_missing", extra={
                    "hint": "Run: CREATE EXTENSION pg_trgm; as superuser"
                })
        except Exception as e:
            log.error("pgvector_connection_failed", extra={"error": str(e)})
            raise

    def _get_conn(self):
        return psycopg2.connect(self._pg_conn_str)

    # ── Public API ────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        restaurant_name: Optional[str] = None,
        menu_item: Optional[str] = None,
        restaurant_id: Optional[str] = None,
        cuisine: Optional[str] = None,
        max_price: Optional[float] = None,
        user_lat: Optional[float] = None,
        user_lon: Optional[float] = None,
        max_distance_km: Optional[float] = 10.0,
    ) -> list[MenuChunk]:
        k = top_k or self.top_k

        # ── Path A: restaurant_id already known — static item lookup ──────────
        if restaurant_id and menu_item:
            return await asyncio.to_thread(
                self._find_items_in_restaurant,
                restaurant_id, menu_item, k
            )

        # ── Path A2: restaurant_id known, no item — sample the menu ──────────
        if restaurant_id and not menu_item:
            return await asyncio.to_thread(
                self._sample_restaurant_menu, restaurant_id, k
            )

        # ── Path B: restaurant name known — static restaurant match first ─────
        if restaurant_name:
            matched = await asyncio.to_thread(
                self._find_restaurant, restaurant_name
            )
            if not matched:
                return []   # caller interprets empty as not found

            rid, rname = matched

            if menu_item:
                # Scoped item search within the matched restaurant
                return await asyncio.to_thread(
                    self._find_items_in_restaurant, rid, menu_item, k
                )
            else:
                # No specific item — return a sample of the restaurant's menu
                return await asyncio.to_thread(
                    self._sample_restaurant_menu, rid, k
                )

        # ── Path C: item name known, no restaurant — find who serves it ───────
        if menu_item:
            return await asyncio.to_thread(
                self._find_item_across_restaurants, menu_item, cuisine, k
            )

        # ── Path D: pure discovery — vector search over combined_text ─────────
        return await self._vector_search(query, cuisine, max_price, k)

    # ── Tool-facing API ───────────────────────────────────────────────────────

    def find_restaurant_candidates(self, name: str, limit: int = 5) -> list[dict]:
        """Resolve a typed restaurant name to up to `limit` distinct restaurants.

        Same exact → contains → trigram hierarchy as _find_restaurant, but returns
        MULTIPLE candidates instead of stopping at the single best row — so the
        confirm_restaurant tool can disambiguate when a query like "spice" matches
        several different restaurants. Stops at the first tier that yields hits
        (a `contains` hit never falls through to trigram), mirroring the rest of
        the retriever's "first hit wins" philosophy.

        Pure SQL — does NOT touch the embedding model. Rows are grouped to one per
        restaurant_id; cuisine/rating are representative values for display only.

        Returns a list of dicts shaped for the model (restaurant_id is a STRING to
        line up with get_menu / get_restaurant_details):
            [{"restaurant_id", "name", "cuisine", "rating", "match_type"}, ...]
        Empty list means not found — caller should tell the user, not invent one.
        """
        name = (name or "").strip()
        if not name:
            return []

        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # 1. Exact
                cur.execute("""
                    SELECT restaurant_id, restaurant_name,
                           MIN(cuisine_name) AS cuisine_name,
                           MAX(rating)       AS rating
                    FROM restaurant_embeddings
                    WHERE LOWER(restaurant_name) = LOWER(%s)
                    GROUP BY restaurant_id, restaurant_name
                    ORDER BY MAX(rating) DESC NULLS LAST
                    LIMIT %s
                """, (name, limit))
                rows = cur.fetchall()
                if rows:
                    log.info("restaurant_candidates_exact", extra={
                        "input": name, "n": len(rows)
                    })
                    return self._restaurant_rows(rows, "exact")

                # 2. Contains
                cur.execute("""
                    SELECT restaurant_id, restaurant_name,
                           MIN(cuisine_name) AS cuisine_name,
                           MAX(rating)       AS rating
                    FROM restaurant_embeddings
                    WHERE LOWER(restaurant_name) LIKE LOWER(%s)
                    GROUP BY restaurant_id, restaurant_name
                    ORDER BY MAX(rating) DESC NULLS LAST
                    LIMIT %s
                """, (f"%{name}%", limit))
                rows = cur.fetchall()
                if rows:
                    log.info("restaurant_candidates_contains", extra={
                        "input": name, "n": len(rows)
                    })
                    return self._restaurant_rows(rows, "contains")

                # 3. Trigram
                cur.execute("""
                    SELECT restaurant_id, restaurant_name,
                           MIN(cuisine_name) AS cuisine_name,
                           MAX(rating)       AS rating,
                           MAX(similarity(LOWER(restaurant_name), LOWER(%s))) AS sim
                    FROM restaurant_embeddings
                    WHERE similarity(LOWER(restaurant_name), LOWER(%s)) > %s
                    GROUP BY restaurant_id, restaurant_name
                    ORDER BY sim DESC
                    LIMIT %s
                """, (name, name, _RESTAURANT_TRGM_THRESHOLD, limit))
                rows = cur.fetchall()
                if rows:
                    log.info("restaurant_candidates_trigram", extra={
                        "input": name,
                        "n": len(rows),
                        "top_sim": round(float(rows[0]["sim"]), 3),
                    })
                    return self._restaurant_rows(rows, "trigram")

                log.info("restaurant_candidates_none", extra={"input": name})
                return []
        finally:
            conn.close()

    def _restaurant_rows(self, rows: list[dict], match_type: str) -> list[dict]:
        """Project grouped restaurant rows into the model-facing dict shape."""
        out = []
        for r in rows:
            out.append({
                "restaurant_id": str(r["restaurant_id"]),
                "name":          r["restaurant_name"] or "",
                "rating":        float(r["rating"]) if r.get("rating") is not None else None,
                "match_type":    match_type,
            })
        return out

    def find_item_candidates(
        self, restaurant_id: str, item_name: str, limit: int = 5
    ) -> list[dict]:
        """Resolve a typed dish name within a KNOWN restaurant to canonical items.

        Wraps _find_items_in_restaurant (exact → contains → trigram) and projects
        the MenuChunks into the model-facing dict shape for the confirm_item tool.
        Pure SQL — no embedding model. Empty list means the dish is not on this
        restaurant's menu (caller should offer real alternatives, not invent one).

        Returns: [{"item_id", "name", "price", "cuisine", "match_type"}, ...]
        """
        restaurant_id = str(restaurant_id or "").strip()
        item_name = (item_name or "").strip()
        if not restaurant_id or not item_name:
            return []

        chunks = self._find_items_in_restaurant(restaurant_id, item_name, limit)
        return [
            {
                "item_id":    c.item_id,
                "name":       c.item_name,
                "price":      c.price,
                "cuisine":    c.cuisine,
                "match_type": c.metadata.get("match_type", "unknown"),
            }
            for c in chunks
        ]

    # ── Static SQL helpers ────────────────────────────────────────────────────

    def _find_restaurant(self, name: str) -> Optional[tuple[str, str]]:
        """Return (restaurant_id, restaurant_name) or None.

        Tries in order:
          1. Exact case-insensitive match
          2. Contains match (input is substring of stored name)
          3. Trigram similarity above threshold
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # 1. Exact
                cur.execute("""
                    SELECT DISTINCT restaurant_id, restaurant_name
                    FROM restaurant_embeddings
                    WHERE LOWER(restaurant_name) = LOWER(%s)
                    LIMIT 1
                """, (name,))
                row = cur.fetchone()
                if row:
                    log.info("restaurant_match_exact", extra={"input": name, "matched": row[1]})
                    return str(row[0]), row[1]

                # 2. Contains
                cur.execute("""
                    SELECT DISTINCT restaurant_id, restaurant_name
                    FROM restaurant_embeddings
                    WHERE LOWER(restaurant_name) LIKE LOWER(%s)
                    LIMIT 1
                """, (f"%{name}%",))
                row = cur.fetchone()
                if row:
                    log.info("restaurant_match_contains", extra={"input": name, "matched": row[1]})
                    return str(row[0]), row[1]

                # 3. Trigram
                cur.execute("""
                    SELECT DISTINCT restaurant_id, restaurant_name,
                           similarity(LOWER(restaurant_name), LOWER(%s)) AS sim
                    FROM restaurant_embeddings
                    WHERE similarity(LOWER(restaurant_name), LOWER(%s)) > %s
                    ORDER BY sim DESC
                    LIMIT 1
                """, (name, name, _RESTAURANT_TRGM_THRESHOLD))
                row = cur.fetchone()
                if row:
                    log.info("restaurant_match_trigram", extra={
                        "input": name, "matched": row[1], "similarity": round(row[2], 3)
                    })
                    return str(row[0]), row[1]

                log.info("restaurant_not_matched", extra={"input": name})
                return None
        finally:
            conn.close()

    def _find_items_in_restaurant(
        self, restaurant_id: str, item_name: str, k: int
    ) -> list[MenuChunk]:
        """Find menu items within a specific restaurant.

        Tries in order:
          1. Exact match on food_item
          2. Contains match
          3. Trigram similarity above threshold
        Returns empty list if nothing found above threshold.
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # 1. Exact
                cur.execute("""
                    SELECT * FROM restaurant_embeddings
                    WHERE restaurant_id = %s
                      AND LOWER(food_item) = LOWER(%s)
                    LIMIT %s
                """, (restaurant_id, item_name, k))
                rows = cur.fetchall()
                if rows:
                    log.info("item_match_exact", extra={
                        "restaurant_id": restaurant_id, "item": item_name, "n": len(rows)
                    })
                    return self._rows_to_chunks(rows, match_type="exact")

                # 2. Contains
                cur.execute("""
                    SELECT * FROM restaurant_embeddings
                    WHERE restaurant_id = %s
                      AND LOWER(food_item) LIKE LOWER(%s)
                    LIMIT %s
                """, (restaurant_id, f"%{item_name}%", k))
                rows = cur.fetchall()
                if rows:
                    log.info("item_match_contains", extra={
                        "restaurant_id": restaurant_id, "item": item_name, "n": len(rows)
                    })
                    return self._rows_to_chunks(rows, match_type="contains")

                # 3. Trigram
                cur.execute("""
                    SELECT *,
                           similarity(LOWER(food_item), LOWER(%s)) AS trgm_sim
                    FROM restaurant_embeddings
                    WHERE restaurant_id = %s
                      AND similarity(LOWER(food_item), LOWER(%s)) > %s
                    ORDER BY trgm_sim DESC
                    LIMIT %s
                """, (item_name, restaurant_id, item_name, _ITEM_TRGM_THRESHOLD, k))
                rows = cur.fetchall()
                if rows:
                    log.info("item_match_trigram", extra={
                        "restaurant_id": restaurant_id,
                        "item": item_name,
                        "n": len(rows),
                        "top_sim": round(float(rows[0]["trgm_sim"]), 3),
                    })
                    return self._rows_to_chunks(rows, match_type="trigram")

                log.info("item_not_found_in_restaurant", extra={
                    "restaurant_id": restaurant_id, "item": item_name
                })
                return []
        finally:
            conn.close()

    def _find_item_across_restaurants(
        self, item_name: str, cuisine: Optional[str], k: int
    ) -> list[MenuChunk]:
        """Find an item across all restaurants (no restaurant scope).

        Same exact → contains → trigram hierarchy.
        Cuisine filter applied when provided.
        Returns up to k results, one per restaurant preferred.
        """
        conn = self._get_conn()
        cuisine_clause = "AND LOWER(cuisine_name) = LOWER(%s)" if cuisine else ""

        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                params_base = [item_name]
                if cuisine:
                    params_base.append(cuisine)

                # 1. Exact
                cur.execute(f"""
                    SELECT * FROM restaurant_embeddings
                    WHERE LOWER(food_item) = LOWER(%s)
                    {cuisine_clause}
                    ORDER BY rating DESC NULLS LAST
                    LIMIT %s
                """, (*params_base, k * 3))
                rows = cur.fetchall()

                # 2. Contains
                if not rows:
                    cur.execute(f"""
                        SELECT * FROM restaurant_embeddings
                        WHERE LOWER(food_item) LIKE LOWER(%s)
                        {cuisine_clause}
                        ORDER BY rating DESC NULLS LAST
                        LIMIT %s
                    """, (f"%{item_name}%", *(params_base[1:]), k * 3))
                    rows = cur.fetchall()

                # 3. Trigram
                if not rows:
                    trgm_params = [item_name, item_name, _ITEM_TRGM_THRESHOLD]
                    if cuisine:
                        trgm_params.append(cuisine)
                    cur.execute(f"""
                        SELECT *,
                               similarity(LOWER(food_item), LOWER(%s)) AS trgm_sim
                        FROM restaurant_embeddings
                        WHERE similarity(LOWER(food_item), LOWER(%s)) > %s
                        {cuisine_clause}
                        ORDER BY trgm_sim DESC, rating DESC NULLS LAST
                        LIMIT %s
                    """, (*trgm_params, k * 3))
                    rows = cur.fetchall()

                if not rows:
                    log.info("item_not_found_anywhere", extra={"item": item_name})
                    return []

                # Deduplicate — keep highest-rated item per restaurant
                seen: set = set()
                deduped = []
                for row in rows:
                    rid = row["restaurant_id"]
                    if rid not in seen:
                        seen.add(rid)
                        deduped.append(row)
                    if len(deduped) >= k:
                        break

                log.info("item_found_across_restaurants", extra={
                    "item": item_name, "n_restaurants": len(deduped)
                })
                return self._rows_to_chunks(deduped, match_type="cross_restaurant")
        finally:
            conn.close()

    def _sample_restaurant_menu(self, restaurant_id: str, k: int) -> list[MenuChunk]:
        """Return a sample of a restaurant's menu when no item is specified."""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM restaurant_embeddings
                    WHERE restaurant_id = %s
                    ORDER BY rating DESC NULLS LAST
                    LIMIT %s
                """, (restaurant_id, k))
                rows = cur.fetchall()
                return self._rows_to_chunks(rows, match_type="menu_sample")
        finally:
            conn.close()

    # ── Vector search (discovery only) ────────────────────────────────────────

    async def _vector_search(
        self,
        query: str,
        cuisine: Optional[str],
        max_price: Optional[float],
        k: int,
    ) -> list[MenuChunk]:
        """Pure semantic search — used only when no restaurant or item name known."""
        q_emb = await asyncio.to_thread(
            self.model.encode,
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        emb_str = "[" + ",".join(f"{v:.6f}" for v in q_emb[0].tolist()) + "]"

        return await asyncio.to_thread(
            self._vector_query, emb_str, cuisine, max_price, k
        )

    def _vector_query(
        self,
        emb_str: str,
        cuisine: Optional[str],
        max_price: Optional[float],
        k: int,
    ) -> list[MenuChunk]:
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT *,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM restaurant_embeddings
                    WHERE
                        (%s IS NULL OR price <= %s)
                        AND (%s IS NULL OR LOWER(cuisine_name) = LOWER(%s))
                        AND 1 - (embedding <=> %s::vector) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (
                    emb_str,
                    max_price, max_price,
                    cuisine, cuisine,
                    emb_str, _VECTOR_SIM_THRESHOLD,
                    emb_str, k,
                ))
                rows = cur.fetchall()
                log.info("vector_search_done", extra={"n": len(rows), "query_len": len(emb_str)})
                return self._rows_to_chunks(rows, match_type="vector")
        finally:
            conn.close()

    # ── Shared row → MenuChunk converter ─────────────────────────────────────

    def _rows_to_chunks(
        self, rows: list[dict], match_type: str = "unknown"
    ) -> list[MenuChunk]:
        chunks = []
        for row in rows:
            sim = float(row.get("trgm_sim", row.get("similarity", 1.0)))
            chunks.append(MenuChunk(
                chunk_id        = str(row["menu_item_id"]),
                restaurant_id   = str(row["restaurant_id"]),
                restaurant_name = row["restaurant_name"] or "",
                cuisine         = row["cuisine_name"] or "",
                item_id         = str(row["menu_item_id"]),
                item_name       = row["food_item"] or "",
                price           = float(row["price"]) if row["price"] else 0.0,
                text            = row["combined_text"] or "",
                rating          = float(row["rating"]) if row["rating"] else None,
                latitude        = float(row["latitude"]) if row["latitude"] else None,
                longitude       = float(row["longitude"]) if row["longitude"] else None,
                metadata        = {"similarity": sim, "match_type": match_type},
            ))
        return chunks


# ── Module-level singleton ─────────────────────────────────────────────────

_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def format_chunks_for_prompt(chunks: list[MenuChunk]) -> str:
    """Render retrieved chunks as a hard constraint block for the LLM."""
    if not chunks:
        return (
            "AVAILABLE ITEMS: none found matching this request.\n"
            "Do NOT suggest or invent any menu items. Tell the user nothing "
            "was found and offer to search elsewhere."
        )

    by_restaurant: dict[str, list[MenuChunk]] = {}
    for c in chunks:
        by_restaurant.setdefault(c.restaurant_name, []).append(c)

    lines = [
        "AVAILABLE ITEMS — these are the ONLY real items you may mention or order.",
        "Do NOT mention, suggest, or order any item not listed here.",
        "If the user asked for something not in this list, say it is not available "
        "and suggest from this list.",
        "",
    ]
    for restaurant_name, items in by_restaurant.items():
        lines.append(f"Restaurant: {items[0].restaurant_name} (ID: {items[0].restaurant_id})")
        for c in items:
            rating_str = f" \u2b50{c.rating:.1f}" if c.rating else ""
            lines.append(
                f"  - {c.item_name} | item_id={c.item_id} | \u20b9{c.price:.0f}{rating_str}"
            )
        lines.append("")

    return "\n".join(lines).strip()