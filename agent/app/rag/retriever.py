"""bge-m3 retriever backed by pgvector (RDS Postgres).

Replaces the in-memory numpy corpus with a Postgres vector search.
The search_menu() SQL function handles:
  - cosine similarity via pgvector <=> operator
  - cuisine / price / geo-distance filters
  - top-k ranking

All heavy lifting stays in Postgres — no embeddings loaded into RAM.
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


@dataclass
class MenuChunk:
    chunk_id:       str
    restaurant_id:  str
    restaurant_name: str
    cuisine:        str
    item_id:        str
    item_name:      str
    price:          float
    text:           str          # combined_text from DB
    description:    str = ""
    rating:         Optional[float] = None
    latitude:       Optional[float] = None
    longitude:      Optional[float] = None
    metadata:       dict[str, Any] = field(default_factory=dict)


class Retriever:
    """pgvector-backed bge-m3 retriever."""

    def __init__(self) -> None:
        s = get_settings()
        log.info("loading_embedding_model", extra={"model": s.embedding_model})
        self.model = SentenceTransformer(s.embedding_model)
        self.top_k = s.rag_top_k
        self._pg_conn_str = (
            f"host={s.pg_host} "
            f"dbname={s.pg_db} "
            f"user={s.pg_user} "
            f"password={s.pg_password} "
            f"port={s.pg_port}"
        )
        # Verify connection at startup
        self._check_connection()
        log.info("retriever_ready", extra={"backend": "pgvector"})

    def _check_connection(self) -> None:
        try:
            conn = psycopg2.connect(self._pg_conn_str)
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM menu_embeddings")
                count = cur.fetchone()[0]
            conn.close()
            log.info("pgvector_connected", extra={"menu_embeddings_count": count})
        except Exception as e:
            log.error("pgvector_connection_failed", extra={"error": str(e)})
            raise

    def _get_conn(self):
        return psycopg2.connect(self._pg_conn_str)

    async def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        cuisine: Optional[str] = None,
        max_price: Optional[float] = None,
        user_lat: Optional[float] = None,
        user_lon: Optional[float] = None,
        max_distance_km: Optional[float] = 10.0,
    ) -> list[MenuChunk]:
        """
        Embed query and search pgvector for similar menu items.
        Filters: cuisine, max_price, geo-distance (optional).
        """
        k = top_k or self.top_k

        # Embed query — CPU bound, run in thread
        q_emb = await asyncio.to_thread(
            self.model.encode,
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Format embedding as Postgres vector string
        emb_str = "[" + ",".join(f"{v:.6f}" for v in q_emb[0].tolist()) + "]"

        # Run DB query in thread to avoid blocking event loop
        results = await asyncio.to_thread(
            self._query_pgvector,
            emb_str, cuisine, max_price,
            user_lat, user_lon, max_distance_km, k
        )
        return results

    def _query_pgvector(
        self,
        emb_str: str,
        cuisine: Optional[str],
        max_price: Optional[float],
        user_lat: Optional[float],
        user_lon: Optional[float],
        max_distance_km: Optional[float],
        k: int,
    ) -> list[MenuChunk]:
        sql = """
            SELECT
                id,
                menu_item_id,
                restaurant_id,
                restaurant_name,
                cuisine_name,
                food_item,
                description,
                price,
                rating,
                latitude,
                longitude,
                combined_text,
                1 - (embedding <=> %s::vector) AS similarity
            FROM menu_embeddings
            WHERE
                (%s IS NULL OR cuisine_name ILIKE %s)
                AND (%s IS NULL OR price <= %s)
                AND (
                    %s IS NULL OR %s IS NULL
                    OR (
                        6371 * acos(
                            LEAST(1.0,
                                cos(radians(%s)) * cos(radians(latitude)) *
                                cos(radians(longitude) - radians(%s)) +
                                sin(radians(%s)) * sin(radians(latitude))
                            )
                        ) <= %s
                    )
                )
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        params = (
            emb_str,                        # query embedding for similarity
            cuisine, cuisine,               # cuisine filter (NULL skips)
            max_price, max_price,           # price filter (NULL skips)
            user_lat, user_lon,             # NULL check for geo filter
            user_lat, user_lon, user_lat,   # haversine calculation
            max_distance_km,                # distance threshold
            emb_str,                        # query embedding for ORDER BY
            k,                              # LIMIT
        )

        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        finally:
            conn.close()

        chunks = []
        for row in rows:
            chunks.append(MenuChunk(
                chunk_id       = str(row["id"]),
                restaurant_id  = str(row["restaurant_id"]),
                restaurant_name= row["restaurant_name"],
                cuisine        = row["cuisine_name"] or "",
                item_id        = str(row["menu_item_id"]),
                item_name      = row["food_item"],
                price          = float(row["price"]),
                text           = row["combined_text"],
                description    = row["description"] or "",
                rating         = float(row["rating"]) if row["rating"] else None,
                latitude       = float(row["latitude"]) if row["latitude"] else None,
                longitude      = float(row["longitude"]) if row["longitude"] else None,
                metadata       = {"similarity": float(row["similarity"])},
            ))
        return chunks


# Module-level singleton
_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def format_chunks_for_prompt(chunks: list[MenuChunk]) -> str:
    """Render retrieved chunks as a compact context block for the LLM."""
    if not chunks:
        return "No relevant menu items found."
    lines = ["Relevant menu items:"]
    for c in chunks:
        similarity = c.metadata.get("similarity", 0)
        rating_str = f" ⭐{c.rating:.1f}" if c.rating else ""
        lines.append(
            f"- [{c.restaurant_id} / {c.item_id}] {c.item_name} @ {c.restaurant_name} "
            f"({c.cuisine}) — ₹{c.price:.0f}{rating_str} "
            f"[similarity: {similarity:.2f}]\n"
            f"  {c.description}"
        )
    return "\n".join(lines)