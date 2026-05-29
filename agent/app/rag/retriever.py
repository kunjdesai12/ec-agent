"""bge-m3 retriever for menu items / restaurants / cuisine descriptions.

Currently uses an in-memory corpus + cosine similarity for development.
Production swap-in: pgvector (you already use Postgres/PostGIS) or Qdrant.

Chunk strategy (per your design):
  one chunk per menu item, prefixed with restaurant + cuisine context.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from agent.app.config import get_settings
from agent.app.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class MenuChunk:
    chunk_id: str
    restaurant_id: str
    restaurant_name: str
    cuisine: str
    item_id: str
    item_name: str
    price: float
    text: str  # the embedded text
    metadata: dict[str, Any] = field(default_factory=dict)


# Tiny seed corpus for development — replace with a real index in prod
_SEED_CORPUS: list[MenuChunk] = [
    MenuChunk(
        chunk_id="c1", restaurant_id="rest_042", restaurant_name="Saffron Kitchen",
        cuisine="north-indian", item_id="itm_1021", item_name="Butter Chicken", price=320,
        text="Saffron Kitchen [north-indian]: Butter Chicken — tandoor-grilled chicken in a creamy tomato-butter gravy. Rich, mildly spiced, served with garlic naan.",
    ),
    MenuChunk(
        chunk_id="c2", restaurant_id="rest_042", restaurant_name="Saffron Kitchen",
        cuisine="north-indian", item_id="itm_1024", item_name="Dal Makhani", price=240,
        text="Saffron Kitchen [north-indian]: Dal Makhani — slow-cooked black lentils with cream and butter. Vegetarian, hearty.",
    ),
    MenuChunk(
        chunk_id="c3", restaurant_id="rest_088", restaurant_name="Tandoor Tales",
        cuisine="north-indian", item_id="itm_2210", item_name="Paneer Tikka Masala", price=280,
        text="Tandoor Tales [north-indian]: Paneer Tikka Masala — char-grilled cottage cheese in spiced onion-tomato gravy. Vegetarian.",
    ),
    MenuChunk(
        chunk_id="c4", restaurant_id="rest_201", restaurant_name="Coastal Curry",
        cuisine="south-indian", item_id="itm_3301", item_name="Chicken Chettinad", price=340,
        text="Coastal Curry [south-indian]: Chicken Chettinad — Tamil Nadu style chicken in roasted spice gravy. Very spicy, peppery.",
    ),
    MenuChunk(
        chunk_id="c5", restaurant_id="rest_201", restaurant_name="Coastal Curry",
        cuisine="south-indian", item_id="itm_3305", item_name="Masala Dosa", price=140,
        text="Coastal Curry [south-indian]: Masala Dosa — crisp rice-lentil crepe with spiced potato filling. Vegetarian, served with sambar and coconut chutney.",
    ),
    MenuChunk(
        chunk_id="c6", restaurant_id="rest_310", restaurant_name="Biryani Junction",
        cuisine="hyderabadi", item_id="itm_4401", item_name="Hyderabadi Chicken Biryani", price=360,
        text="Biryani Junction [hyderabadi]: Hyderabadi Chicken Biryani — basmati rice layered with marinated chicken, saffron, fried onions. Slow-cooked dum style.",
    ),
]


class Retriever:
    """bge-m3 cosine retriever. Embeddings normalized so dot product = cosine."""

    def __init__(self) -> None:
        s = get_settings()
        log.info("loading_embedding_model", extra={"model": s.embedding_model})
        # bge-m3 dims = 1024
        self.model = SentenceTransformer(s.embedding_model)
        self.top_k = s.rag_top_k
        self.corpus: list[MenuChunk] = list(_SEED_CORPUS)
        self.embeddings: np.ndarray = self._embed_corpus()
        log.info("retriever_ready", extra={"n_chunks": len(self.corpus)})

    def _embed_corpus(self) -> np.ndarray:
        texts = [c.text for c in self.corpus]
        emb = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return emb

    async def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        cuisine: Optional[str] = None,
        max_price: Optional[float] = None,
    ) -> list[MenuChunk]:
        """Embed query, score against corpus, apply metadata filters, return top-k."""
        k = top_k or self.top_k
        # encode is CPU-bound; run in thread to avoid blocking the event loop
        q_emb = await asyncio.to_thread(
            self.model.encode, [query], normalize_embeddings=True, convert_to_numpy=True
        )
        scores = (self.embeddings @ q_emb[0])  # cosine since both normalized

        # Apply filters
        candidates: list[tuple[float, MenuChunk]] = []
        for score, chunk in zip(scores, self.corpus):
            if cuisine and chunk.cuisine != cuisine:
                continue
            if max_price is not None and chunk.price > max_price:
                continue
            candidates.append((float(score), chunk))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in candidates[:k]]


# Module-level singleton — loaded once at startup
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
        lines.append(
            f"- [{c.restaurant_id} / {c.item_id}] {c.item_name} @ {c.restaurant_name} "
            f"({c.cuisine}) — ₹{c.price:.0f}"
        )
    return "\n".join(lines)
