import logging
import os
from datetime import datetime

import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sync_embeddings.log")

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
    dbname="postgres",
    user="postgres.eilbullhdgvzgflqzgnq",
    password="Agent@DB_SUPA",
    host="aws-1-ap-southeast-1.pooler.supabase.com",
    port="6543",
)

MODEL_NAME  = "all-MiniLM-L6-v2"
BATCH_SIZE  = 256


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def get_last_synced_at(cursor):
    cursor.execute("SELECT last_synced_at FROM sync_metadata WHERE id = 1")
    row = cursor.fetchone()
    return row[0]


def update_last_synced_at(cursor):
    cursor.execute("""
        UPDATE sync_metadata
        SET last_synced_at = NOW()
        WHERE id = 1
    """)


def fetch_source_rows(cursor, last_synced_at):
    cursor.execute("""
        SELECT
            menu_item_id,
            restaurant_id,
            restaurant_name,
            food_item,
            combined_text,
            cuisine_name
        FROM restaurant_menu_view
        WHERE updated_at > %s
    """, (last_synced_at,))
    return cursor.fetchall()


def upsert_batch(cursor, records):
    psycopg2.extras.execute_values(
        cursor,
        """
        INSERT INTO restaurant_embeddings
            (menu_item_id, restaurant_id, restaurant_name, food_item, combined_text, embedding, restaurant_name_embedding, food_item_embedding, cuisine_embedding)
        VALUES %s
        ON CONFLICT (menu_item_id)
        DO UPDATE SET
            restaurant_id   = EXCLUDED.restaurant_id,
            restaurant_name = EXCLUDED.restaurant_name,
            food_item       = EXCLUDED.food_item,
            combined_text   = EXCLUDED.combined_text,
            embedding       = EXCLUDED.embedding,
            restaurant_name_embedding = EXCLUDED.restaurant_name_embedding,
            food_item_embedding       = EXCLUDED.food_item_embedding,
            cuisine_embedding         = EXCLUDED.cuisine_embedding
        """,
        records,
        template="(%s, %s, %s, %s, %s, %s::vector, %s::vector, %s::vector, %s::vector)",
    )


# Main job
def run_sync():
    log.info("Embedding sync job started")
    start = datetime.now()

    try:
        # Loading model
        log.info("Loading SentenceTransformer model: %s", MODEL_NAME)
        model = SentenceTransformer(MODEL_NAME)

        # Fetching source rows
        conn   = get_connection()
        cursor = conn.cursor()

        log.info("Fetching rows from restaurants_raw ⨝ menu_items_raw …")

        last_synced_at = get_last_synced_at(cursor)
        log.info("Fetching rows changed since %s", last_synced_at)
        rows = fetch_source_rows(cursor, last_synced_at)

        log.info("Fetched %d rows", len(rows))

        if not rows:
            log.warning("No rows found — nothing to embed. Exiting.")
            cursor.close()
            conn.close()
            return

        # Encode + upsert in batches
        total_upserted = 0

        for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Batches"):
            batch = rows[i : i + BATCH_SIZE]

            combined_embeddings = model.encode(
                [row[4] for row in batch],
                show_progress_bar=False
            )

            restaurant_name_embeddings = model.encode(
                [row[2] or "" for row in batch],
                show_progress_bar=False
            )

            food_item_embeddings = model.encode(
                [row[3] or "" for row in batch],
                show_progress_bar=False
            )

            cuisine_embeddings = model.encode(
                [row[5] or "" for row in batch],
                show_progress_bar=False
            )

            records = [
                (
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    combined_embeddings[j].tolist(),
                    restaurant_name_embeddings[j].tolist(),
                    food_item_embeddings[j].tolist(),
                    cuisine_embeddings[j].tolist()
                )
                for j, row in enumerate(batch)
            ]

            upsert_batch(cursor, records)
            conn.commit()

            total_upserted += len(records)

        update_last_synced_at(cursor)
        conn.commit()

        cursor.close()
        conn.close()

        elapsed = (datetime.now() - start).total_seconds()
        log.info("Sync complete — %d rows upserted in %.1f s", total_upserted, elapsed)

    except Exception:
        log.exception("Sync job FAILED with an unexpected error")
        raise

if __name__ == "__main__":
    run_sync()