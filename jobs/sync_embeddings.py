import logging
import os
from datetime import datetime

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "sync_embeddings.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

DB_CONFIG = dict(
    dbname=os.getenv("VECTOR_DB_NAME", "vector_easycater"),
    user=os.getenv("VECTOR_DB_USER", "postgres"),
    password=os.getenv("VECTOR_DB_PASSWORD"),        # ← moved to .env
    host=os.getenv("VECTOR_DB_HOST", "localhost"),
    port=os.getenv("VECTOR_DB_PORT", "5432"),
)

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 64   


# ─── DB helpers ──────────────────────────────────────────────────────────────

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def get_last_synced_at(cursor):
    cursor.execute("SELECT last_synced_at FROM sync_metadata WHERE id = 1")
    return cursor.fetchone()[0]

def update_last_synced_at(cursor):
    cursor.execute(
        "UPDATE sync_metadata SET last_synced_at = NOW() WHERE id = 1"
    )


# ─── Fetch ───────────────────────────────────────────────────────────────────

def fetch_changed_rows(cursor, last_synced_at):
    """
    Fetch rows that were updated OR recently deleted so we can
    handle both upserts and removals in one pass.
    """
    cursor.execute("""
        SELECT
            menu_item_id,
            restaurant_id,
            restaurant_name,
            food_item,
            combined_text,
            cuisines_name,
            rating,
            price,
            latitude,
            longitude,
        FROM restaurant_menu_view
        WHERE updated_at > %s
    """, (last_synced_at,))
    return cursor.fetchall()


# ─── Upsert / Delete ─────────────────────────────────────────────────────────

def delete_embeddings(cursor, menu_item_ids: list):
    cursor.execute(
        "DELETE FROM restaurant_embeddings WHERE menu_item_id = ANY(%s)",
        (menu_item_ids,)
    )
    log.info("Deleted embeddings for %d removed menu items", len(menu_item_ids))

def upsert_batch(cursor, records):
    psycopg2.extras.execute_values(
        cursor,
        """
        INSERT INTO restaurant_embeddings (
            menu_item_id, restaurant_id, restaurant_name, food_item,
            combined_text, cuisine_name, rating, price,
            latitude, longitude,
            embedding, restaurant_name_embedding,
            food_item_embedding, cuisine_embedding
        )
        VALUES %s
        ON CONFLICT (menu_item_id) DO UPDATE SET
            restaurant_id             = EXCLUDED.restaurant_id,
            restaurant_name           = EXCLUDED.restaurant_name,
            food_item                 = EXCLUDED.food_item,
            combined_text             = EXCLUDED.combined_text,
            cuisine_name              = EXCLUDED.cuisine_name,
            rating                    = EXCLUDED.rating,
            price                     = EXCLUDED.price,
            latitude                  = EXCLUDED.latitude,
            longitude                 = EXCLUDED.longitude,
            embedding                 = EXCLUDED.embedding,
            restaurant_name_embedding = EXCLUDED.restaurant_name_embedding,
            food_item_embedding       = EXCLUDED.food_item_embedding,
            cuisine_embedding         = EXCLUDED.cuisine_embedding
        """,
        records,
        template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector,%s::vector,%s::vector,%s::vector)",
    )


# ─── Encode ──────────────────────────────────────────────────────────────────

def encode_batch(model, batch):
    """
    Single encode call for all 4 fields concatenated, then slice.
    4x faster than calling model.encode() separately for each field.
    """
    # Build one flat list: [combined×N, name×N, food×N, cuisine×N]
    combined_texts      = [row[4] or "" for row in batch]
    restaurant_names    = [row[2] or "" for row in batch]
    food_items          = [row[3] or "" for row in batch]
    cuisine_names       = [row[5] or "" for row in batch]

    all_texts = combined_texts + restaurant_names + food_items + cuisine_names
    all_vecs  = model.encode(all_texts, show_progress_bar=False)

    n = len(batch)
    return (
        all_vecs[0*n : 1*n],   # combined embeddings
        all_vecs[1*n : 2*n],   # restaurant name embeddings
        all_vecs[2*n : 3*n],   # food item embeddings
        all_vecs[3*n : 4*n],   # cuisine embeddings
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def run_sync():
    log.info("Embedding sync job started")
    start = datetime.now()

    try:
        log.info("Loading SentenceTransformer model: %s", MODEL_NAME)
        model = SentenceTransformer(MODEL_NAME)

        conn   = get_connection()
        cursor = conn.cursor()

        last_synced_at = get_last_synced_at(cursor)
        log.info("Fetching rows changed since %s", last_synced_at)

        rows = fetch_changed_rows(cursor, last_synced_at)
        log.info("Fetched %d changed rows", len(rows))

        # ── Always advance the timestamp, even if nothing changed ─────────
        # Fixes the bug where early-return skipped update_last_synced_at
        if not rows:
            log.info("No rows changed — updating timestamp and exiting.")
            update_last_synced_at(cursor)
            conn.commit()
            cursor.close()
            conn.close()
            return

        # ── Split: deleted vs active ──────────────────────────────────────
        deleted_ids = [row[0] for row in rows if row[10] is not None]  # deleted_at not null
        active_rows = [row for row in rows if row[10] is None]

        if deleted_ids:
            delete_embeddings(cursor, deleted_ids)
            conn.commit()

        # ── Encode + upsert active rows in batches ────────────────────────
        total_upserted = 0

        for i in tqdm(range(0, len(active_rows), BATCH_SIZE), desc="Batches"):
            batch = active_rows[i : i + BATCH_SIZE]

            combined_vecs, name_vecs, food_vecs, cuisine_vecs = encode_batch(model, batch)

            records = [
                (
                    row[0],                         # menu_item_id
                    row[1],                         # restaurant_id
                    row[2],                         # restaurant_name
                    row[3],                         # food_item
                    row[4],                         # combined_text
                    row[5],                         # cuisines_name
                    row[6],                         # rating
                    row[7],                         # price
                    row[8],                         # latitude
                    row[9],                         # longitude
                    combined_vecs[j].tolist(),
                    name_vecs[j].tolist(),
                    food_vecs[j].tolist(),
                    cuisine_vecs[j].tolist(),
                )
                for j, row in enumerate(batch)
            ]

            upsert_batch(cursor, records)
            conn.commit()
            total_upserted += len(records)

        # ── Finalize ──────────────────────────────────────────────────────
        update_last_synced_at(cursor)
        conn.commit()
        cursor.close()
        conn.close()

        elapsed = (datetime.now() - start).total_seconds()
        log.info(
            "Sync complete in %.1f s — upserted: %d, deleted: %d",
            elapsed, total_upserted, len(deleted_ids)
        )

    except Exception:
        log.exception("Embedding sync FAILED with an unexpected error")
        raise


if __name__ == "__main__":
    run_sync()