import logging
import os
import requests
import psycopg2
import psycopg2.extras
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("LOCAL_URL")

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "sync_raw_data.log")

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
    dbname=os.getenv("SYNC_DB_NAME", "postgres"),
    user=os.getenv("SYNC_DB_USER"),
    password=os.getenv("SYNC_DB_PASSWORD"),      # ← moved to .env
    host=os.getenv("SYNC_DB_HOST"),
    port=os.getenv("SYNC_DB_PORT", "6543"),
)

# API endpoints
RESTAURANTS_API = "{base_url}/restaurants/details-for-ai"
MENU_ITEMS_API  = "{base_url}/new-menu-item/get-menu-items-for-ai"
CUISINES_API    = "{base_url}/cuisines/details-for-ai"   # ← was missing

# Constants
SYNC_METADATA_ID = 2
PAGE_LIMIT        = 1000


# ─── DB helpers ──────────────────────────────────────────────────────────────

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def get_last_synced_at(cursor):
    cursor.execute(
        "SELECT last_synced_at FROM sync_metadata WHERE id = %s",
        (SYNC_METADATA_ID,)
    )
    row = cursor.fetchone()
    return row[0]

def update_last_synced_at(cursor):
    cursor.execute(
        "UPDATE sync_metadata SET last_synced_at = NOW() WHERE id = %s",
        (SYNC_METADATA_ID,)
    )


# ─── Fetch ───────────────────────────────────────────────────────────────────

def fetch_all_pages(endpoint, sync_after=None):
    all_records = []
    page = 1
    while True:
        params = {"page": page, "limit": PAGE_LIMIT}
        if sync_after:
            params["sync_after"] = sync_after.isoformat()

        log.info("Fetching page %d from %s", page, endpoint)
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        records = data.get("data", [])
        all_records.extend(records)
        log.info(
            "Page %d — received %d records (total so far: %d)",
            page, len(records), len(all_records)
        )

        if not data.get("hasMore", False):
            break
        page += 1

    return all_records


# ─── Upsert / Delete helpers ─────────────────────────────────────────────────

def sync_restaurants(cursor, records):
    """Upsert active restaurants; delete soft-deleted ones."""
    to_delete = [r["restaurant_id"] for r in records if r.get("deleted_at")]
    to_upsert = [r for r in records if not r.get("deleted_at")]

    if to_delete:
        log.info("Deleting %d soft-deleted restaurants", len(to_delete))
        cursor.execute(
            "DELETE FROM restaurants_raw WHERE restaurant_id = ANY(%s)",
            (to_delete,)
        )

    if to_upsert:
        rows = [(r["restaurant_id"], r["restaurant_name"]) for r in to_upsert]
        psycopg2.extras.execute_values(
            cursor,
            """
            INSERT INTO restaurants_raw (restaurant_id, restaurant_name, updated_at)
            VALUES %s
            ON CONFLICT (restaurant_id)
            DO UPDATE SET
                restaurant_name = EXCLUDED.restaurant_name,
                updated_at      = NOW()
            """,
            rows,
            template="(%s, %s, NOW())",
        )

    return len(to_upsert), len(to_delete)


def sync_menu_items(cursor, records):
    """Upsert active menu items; delete soft-deleted ones."""
    to_delete = [r["menu_item_id"] for r in records if r.get("deletedAt")]
    to_upsert = [r for r in records if not r.get("deletedAt")]

    if to_delete:
        log.info("Deleting %d soft-deleted menu items", len(to_delete))
        cursor.execute(
            "DELETE FROM menu_items WHERE menu_item_id = ANY(%s)",
            (to_delete,)
        )

    if to_upsert:
        rows = [
            (r["menu_item_id"], r["business_id"], r["title"], r["cuisines_id"], r.get("description"))
            for r in to_upsert
        ]
        psycopg2.extras.execute_values(
            cursor,
            """
            INSERT INTO menu_items (menu_item_id, business_id, title, cuisines_id, description, updated_at)
            VALUES %s
            ON CONFLICT (menu_item_id)
            DO UPDATE SET
                business_id = EXCLUDED.business_id,
                title       = EXCLUDED.title,
                cuisines_id = EXCLUDED.cuisines_id,
                description = EXCLUDED.description,
                updated_at  = NOW()
            """,
            rows,
            template="(%s, %s, %s, %s, %s, NOW())",
        )

    return len(to_upsert), len(to_delete)


def sync_cuisines(cursor, records):
    """Upsert active cuisines; delete soft-deleted or inactive ones."""
    to_delete = [
        r["cuisines_id"] for r in records
        if r.get("deletedAt") or r.get("status") == "inactive"
    ]
    to_upsert = [
        r for r in records
        if not r.get("deletedAt") and r.get("status") != "inactive"
    ]

    if to_delete:
        log.info("Deleting %d soft-deleted/inactive cuisines", len(to_delete))
        cursor.execute(
            "DELETE FROM cuisines WHERE cuisines_id = ANY(%s)",
            (to_delete,)
        )

    if to_upsert:
        rows = [(r["cuisines_id"], r["cuisines_name"]) for r in to_upsert]
        psycopg2.extras.execute_values(
            cursor,
            """
            INSERT INTO cuisines (cuisines_id, cuisines_name, updated_at)
            VALUES %s
            ON CONFLICT (cuisines_id)
            DO UPDATE SET
                cuisines_name = EXCLUDED.cuisines_name,
                updated_at    = NOW()
            """,
            rows,
            template="(%s, %s, NOW())",
        )

    return len(to_upsert), len(to_delete)


# ─── Main ────────────────────────────────────────────────────────────────────

def run_sync():
    log.info("Raw data sync job started")
    start = datetime.now()

    # Default counts in case fetches return nothing (fixes UnboundLocalError)
    restaurant_upserted = restaurant_deleted = 0
    menu_upserted = menu_deleted = 0
    cuisine_upserted = cuisine_deleted = 0

    try:
        conn   = get_connection()
        cursor = conn.cursor()

        last_synced_at = get_last_synced_at(cursor)
        log.info("Syncing records changed since %s", last_synced_at)

        # ── Restaurants ──────────────────────────────────────────────────────
        log.info("Syncing restaurants")
        restaurants = fetch_all_pages(
            RESTAURANTS_API.format(base_url=BASE_URL), last_synced_at
        )
        if restaurants:
            restaurant_upserted, restaurant_deleted = sync_restaurants(cursor, restaurants)
            conn.commit()
            log.info(
                "Restaurants — upserted: %d, deleted: %d",
                restaurant_upserted, restaurant_deleted
            )
        else:
            log.info("No new or modified restaurants found")

        # ── Menu Items ───────────────────────────────────────────────────────
        log.info("Syncing menu items")
        menu_items = fetch_all_pages(
            MENU_ITEMS_API.format(base_url=BASE_URL), last_synced_at
        )
        if menu_items:
            menu_upserted, menu_deleted = sync_menu_items(cursor, menu_items)
            conn.commit()
            log.info(
                "Menu items — upserted: %d, deleted: %d",
                menu_upserted, menu_deleted
            )
        else:
            log.info("No new or modified menu items found")

        # ── Cuisines ─────────────────────────────────────────────────────────
        log.info("Syncing cuisines")
        cuisines = fetch_all_pages(
            CUISINES_API.format(base_url=BASE_URL), last_synced_at  # ← now paginated with sync_after
        )
        if cuisines:
            cuisine_upserted, cuisine_deleted = sync_cuisines(cursor, cuisines)
            conn.commit()
            log.info(
                "Cuisines — upserted: %d, deleted: %d",
                cuisine_upserted, cuisine_deleted
            )
        else:
            log.info("No new or modified cuisines found")

        # ── Finalize ─────────────────────────────────────────────────────────
        update_last_synced_at(cursor)
        conn.commit()
        cursor.close()
        conn.close()

        elapsed = (datetime.now() - start).total_seconds()
        log.info(
            "Sync complete in %.1f s — "
            "restaurants(+%d -%d)  menu_items(+%d -%d)  cuisines(+%d -%d)",
            elapsed,
            restaurant_upserted, restaurant_deleted,
            menu_upserted, menu_deleted,
            cuisine_upserted, cuisine_deleted,
        )

    except requests.exceptions.ConnectionError:
        log.error("Could not connect to API — is the server running?")
        raise
    except requests.exceptions.HTTPError as e:
        log.error("API returned an error: %s", e)
        raise
    except Exception:
        log.exception("Sync job FAILED with an unexpected error")
        raise


if __name__ == "__main__":
    run_sync()