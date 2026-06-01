"""
ETL Pipeline: MySQL (EasyCater) → pgvector (RDS Postgres)

Steps:
1. Read menu items from MySQL replica (joined with restaurants + cuisine_master)
2. Enrich thin descriptions using Claude Haiku
3. Format combined_text for embedding
4. Run bge-m3 embeddings in batches on GPU
5. Insert into menu_embeddings table in Postgres

Usage:
    pip install pymysql psycopg2-binary sentence-transformers anthropic torch
    python etl_menu_embeddings.py
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import anthropic
import numpy as np
import pymysql
import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# CONFIG — set via environment variables
# ─────────────────────────────────────────────
MYSQL_HOST      = os.environ["MYSQL_HOST"]
MYSQL_USER      = os.environ["MYSQL_USER"]
MYSQL_PASSWORD  = os.environ["MYSQL_PASSWORD"]
MYSQL_DB        = os.environ["MYSQL_DB"]

PG_HOST         = os.environ["PG_HOST"]
PG_USER         = os.environ["PG_USER"]
PG_PASSWORD     = os.environ["PG_PASSWORD"]
PG_DB           = os.environ["PG_DB"]

ANTHROPIC_KEY   = os.environ["ANTHROPIC_API_KEY"]

EMBEDDING_MODEL = "BAAI/bge-m3"
BATCH_SIZE      = 64
ENRICH_BATCH    = 20

FOOD_TYPE_MAP = {"1": "veg", "2": "non-veg", "3": "egg"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DATA MODEL
# ─────────────────────────────────────────────
@dataclass
class MenuItem:
    menu_item_id:         int
    restaurant_id:        int
    restaurant_name:      str
    cuisine_name:         str
    cuisine_preference:   str
    type_of_cuisine:      str
    item_name:            str
    description:          Optional[str]
    price:                float
    food_type:            str
    restaurant_type:      str
    is_budget_friendly:   bool
    is_street_food:       bool
    is_chef_recommended:  bool = False
    rating:               Optional[float] = None
    restaurant_rating:    Optional[float] = None
    latitude:             Optional[float] = None
    longitude:            Optional[float] = None
    spice_level:          Optional[str] = None
    is_healthy:           bool = False
    is_high_protein:      bool = False
    is_low_calorie:       bool = False
    is_gluten_free:       bool = False
    calories:             Optional[int] = None
    item_type:            str = "instant"
    enriched_description: Optional[str] = None
    combined_text:        Optional[str] = None


# ─────────────────────────────────────────────
# STEP 1 — FETCH FROM MYSQL
# ─────────────────────────────────────────────
def fetch_menu_items() -> list[MenuItem]:
    log.info("Connecting to MySQL...")
    conn = pymysql.connect(
        host=MYSQL_HOST, user=MYSQL_USER,
        password=MYSQL_PASSWORD, database=MYSQL_DB,
        cursorclass=pymysql.cursors.DictCursor
    )
    query = """
        SELECT
            mi.menu_item_id,
            mi.restaurant_id,
            mi.title          AS item_name,
            mi.description,
            mi.price,
            mi.food_type,
            mi.spice_level,
            mi.is_healthy,
            mi.is_high_protein,
            mi.is_low_calorie,
            mi.is_gluten_free,
            mi.calories,
            mi.type           AS item_type,
            r.restaurant_name,
            r.rating          AS restaurant_rating,
            r.latitude,
            r.longitude,
            r.restaurant_type,
            r.is_budget_friendly,
            r.is_street_food,
            cm.cuisines_name,
            cm.cuisine_preferences,
            cm.type_of_cuisine
        FROM new_menu_items mi
        JOIN restaurants r      ON mi.restaurant_id = r.restaurant_id
        JOIN cuisine_master cm  ON mi.cuisines_id   = cm.cuisines_id
        WHERE mi.status      = 1
          AND mi.deletedAt   IS NULL
          AND mi.review_status = 'approved'
          AND r.deletedAt    IS NULL
        ORDER BY mi.menu_item_id
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    conn.close()
    log.info(f"Fetched {len(rows)} active menu items from MySQL")

    items = []
    for r in rows:
        items.append(MenuItem(
            menu_item_id       = r["menu_item_id"],
            restaurant_id      = r["restaurant_id"],
            restaurant_name    = r["restaurant_name"],
            cuisine_name       = r["cuisines_name"] or "",
            cuisine_preference = r["cuisine_preferences"] or "Vegetarian",
            type_of_cuisine    = r["type_of_cuisine"] or "normal",
            item_name          = r["item_name"],
            description        = r["description"] if r["description"] and len(r["description"].strip()) > 10 else None,
            price              = float(r["price"]),
            food_type          = FOOD_TYPE_MAP.get(str(r["food_type"]), "veg"),
            restaurant_type    = r["restaurant_type"] or "veg",
            is_budget_friendly = bool(r["is_budget_friendly"]),
            is_street_food     = bool(r["is_street_food"]),
            spice_level        = r["spice_level"],
            is_healthy         = bool(r["is_healthy"]),
            is_high_protein    = bool(r["is_high_protein"]),
            is_low_calorie     = bool(r["is_low_calorie"]),
            is_gluten_free     = bool(r["is_gluten_free"]),
            calories           = r["calories"],
            item_type          = r["item_type"] or "instant",
            restaurant_rating  = float(r["restaurant_rating"]) if r["restaurant_rating"] else None,
            latitude           = float(r["latitude"]) if r["latitude"] else None,
            longitude          = float(r["longitude"]) if r["longitude"] else None,
        ))
    return items


# ─────────────────────────────────────────────
# STEP 2 — ENRICH DESCRIPTIONS VIA CLAUDE HAIKU
# ─────────────────────────────────────────────
def enrich_description(client: anthropic.Anthropic, item: MenuItem) -> str:
    tags = []
    if item.food_type == "veg":
        tags.append("vegetarian")
    elif item.food_type == "non-veg":
        tags.append("non-vegetarian")
    if item.is_healthy:
        tags.append("healthy")
    if item.is_high_protein:
        tags.append("high protein")
    if item.is_low_calorie:
        tags.append("low calorie")
    if item.is_gluten_free:
        tags.append("gluten free")
    if item.is_budget_friendly:
        tags.append("budget friendly")
    if item.spice_level:
        tags.append(f"{item.spice_level} spice")
    if item.item_type == "catering":
        tags.append("catering package")

    prompt = f"""Write a 1-2 sentence appetizing description for this Indian restaurant menu item.

Item: {item.item_name}
Restaurant: {item.restaurant_name}
Cuisine: {item.cuisine_name}
Tags: {', '.join(tags) if tags else 'none'}
Price: ₹{item.price:.0f}
Existing description: {item.description or 'none'}

Rules:
- Focus on ingredients, cooking style, taste, and texture
- Natural and appetizing, not marketing fluff
- Under 40 words
- No quotes, no bullet points
- If existing description has useful info, incorporate it"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


def enrich_all_descriptions(items: list[MenuItem]) -> list[MenuItem]:
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    needs_enrichment = [i for i in items if i.description is None]
    has_description  = [i for i in items if i.description is not None]

    log.info(f"Items with good descriptions: {len(has_description)}")
    log.info(f"Items needing enrichment: {len(needs_enrichment)}")

    for idx, item in enumerate(needs_enrichment):
        try:
            item.enriched_description = enrich_description(client, item)
            log.info(f"  [{idx+1}/{len(needs_enrichment)}] {item.item_name}: {item.enriched_description}")
            time.sleep(0.5)  # ~40 req/min on Haiku
        except Exception as e:
            log.warning(f"Failed to enrich item {item.menu_item_id}: {e}")
            item.enriched_description = f"{item.item_name} — {item.cuisine_name} dish at {item.restaurant_name}"

    for item in has_description:
        item.enriched_description = item.description

    return items


# ─────────────────────────────────────────────
# STEP 3 — FORMAT COMBINED TEXT
# ─────────────────────────────────────────────
def build_combined_text(item: MenuItem) -> str:
    parts = [
        f"{item.restaurant_name} [{item.cuisine_name}]: {item.item_name}",
        f"— {item.enriched_description}",
    ]

    tags = []
    if item.food_type == "veg":
        tags.append("vegetarian")
    elif item.food_type == "non-veg":
        tags.append("non-vegetarian")
    elif item.food_type == "egg":
        tags.append("contains egg")
    if item.spice_level:
        tags.append(f"{item.spice_level} spice")
    if item.is_healthy:
        tags.append("healthy")
    if item.is_high_protein:
        tags.append("high protein")
    if item.is_low_calorie:
        tags.append("low calorie")
    if item.is_gluten_free:
        tags.append("gluten free")
    if item.is_budget_friendly:
        tags.append("budget friendly")
    if item.item_type == "catering":
        tags.append("catering")
    if item.calories:
        tags.append(f"{item.calories} kcal")

    if tags:
        parts.append(f"({', '.join(tags)})")

    parts.append(f"₹{item.price:.0f}")
    return " ".join(parts)


# ─────────────────────────────────────────────
# STEP 4 — GENERATE EMBEDDINGS
# ─────────────────────────────────────────────
def generate_embeddings(items: list[MenuItem]) -> np.ndarray:
    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [item.combined_text for item in items]
    log.info(f"Generating embeddings for {len(texts)} items in batches of {BATCH_SIZE}...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    log.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


# ─────────────────────────────────────────────
# STEP 5 — INSERT INTO POSTGRES
# ─────────────────────────────────────────────
def insert_into_postgres(items: list[MenuItem], embeddings: np.ndarray) -> None:
    log.info("Connecting to Postgres...")
    conn = psycopg2.connect(
        host=PG_HOST, user=PG_USER,
        password=PG_PASSWORD, dbname=PG_DB
    )
    conn.autocommit = False

    insert_sql = """
        INSERT INTO menu_embeddings (
            menu_item_id, restaurant_id, restaurant_name,
            cuisine_name, food_item, description,
            price, rating, latitude, longitude,
            combined_text, embedding, model
        ) VALUES (
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s::vector, %s
        )
        ON CONFLICT DO NOTHING
    """

    batch = []
    for item, emb in zip(items, embeddings):
        emb_str = "[" + ",".join(f"{v:.6f}" for v in emb.tolist()) + "]"
        batch.append((
            item.menu_item_id,
            item.restaurant_id,
            item.restaurant_name,
            item.cuisine_name,
            item.item_name,
            item.enriched_description,
            item.price,
            item.rating or item.restaurant_rating,
            item.latitude,
            item.longitude,
            item.combined_text,
            emb_str,
            EMBEDDING_MODEL,
        ))

    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, insert_sql, batch, page_size=100)

    conn.commit()
    conn.close()
    log.info(f"Inserted {len(batch)} rows into menu_embeddings")


# ─────────────────────────────────────────────
# STEP 6 — REBUILD INDEX AFTER LOAD
# ─────────────────────────────────────────────
def rebuild_vector_index() -> None:
    log.info("Rebuilding ivfflat index...")
    conn = psycopg2.connect(
        host=PG_HOST, user=PG_USER,
        password=PG_PASSWORD, dbname=PG_DB
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS menu_embeddings_embedding_idx")
        cur.execute("""
            CREATE INDEX menu_embeddings_embedding_idx
            ON menu_embeddings
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
    conn.close()
    log.info("Index rebuilt successfully")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    log.info("=== EasyCater ETL: MySQL → pgvector ===")

    # 1. Fetch
    items = fetch_menu_items()
    if not items:
        log.error("No items fetched — check MySQL connection and filters")
        return

    # 2. Enrich descriptions
    items = enrich_all_descriptions(items)

    # 3. Build combined text
    for item in items:
        item.combined_text = build_combined_text(item)

    # Sample check
    log.info("Sample combined_text:")
    for item in items[:3]:
        log.info(f"  [{item.menu_item_id}] {item.combined_text}")

    # 4. Embed
    embeddings = generate_embeddings(items)

    # 5. Insert
    insert_into_postgres(items, embeddings)

    # 6. Rebuild index
    rebuild_vector_index()

    log.info("=== ETL complete ===")


if __name__ == "__main__":
    main()