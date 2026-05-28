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
    dbname="postgres",
    user="postgres.eilbullhdgvzgflqzgnq",
    password="Agent@DB_SUPA",
    host="aws-1-ap-southeast-1.pooler.supabase.com",
    port="6543",
)

RESTAURANTS_API = "{base_url}/restaurants/details-for-ai"
MENU_ITEMS_API  = "{base_url}/new-menu-item/get-menu-items-for-ai"

# Constants
SYNC_METADATA_ID = 2        # id=1 for sync embeddings
PAGE_LIMIT       = 1000 


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


def fetch_all_pages(endpoint, sync_after):

    all_records = []
    page = 1

    while True:
        params = {
            "page":       page,
            "limit":      PAGE_LIMIT,
            "sync_after": sync_after.isoformat(),
        }

        log.info("Fetching page %d from %s", page, endpoint)
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        records = data.get("data", [])
        all_records.extend(records)

        log.info("Page %d — received %d records (total so far: %d)",
                 page, len(records), len(all_records))

        # Stop if no more pages
        if not data.get("hasMore", False):
            break

        page += 1

    return all_records


def upsert_restaurants(cursor, records):

    rows = [
        (r["restaurant_id"], r["restaurant_name"])
        for r in records
    ]

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


def upsert_menu_items(cursor, records):

    rows = [
        (r["menu_item_id"], r["business_id"], r["title"], r["cuisine_id"])
        for r in records
    ]

    psycopg2.extras.execute_values(
        cursor,
        """
        INSERT INTO menu_items (menu_item_id, business_id, title, cuisine_id, updated_at)
        VALUES %s
        ON CONFLICT (menu_item_id)
        DO UPDATE SET
            business_id = EXCLUDED.business_id,
            title       = EXCLUDED.title,
            cuisine_id  = EXCLUDED.cuisine_id,
            updated_at  = NOW()
        """,
        rows,
        template="(%s, %s, %s, %s, NOW())",
    )


# Main job
def run_sync():
    log.info("Raw data sync job started")
    start = datetime.now()

    try:
        
        conn   = get_connection()
        cursor = conn.cursor()

        last_synced_at = get_last_synced_at(cursor)
        log.info("Syncing records changed since %s", last_synced_at)

        # Sync restaurants
        log.info("Syncing restaurants")
        restaurant_endpoint = RESTAURANTS_API.format(base_url=BASE_URL)
        restaurants = fetch_all_pages(restaurant_endpoint, last_synced_at)

        if restaurants:
            log.info("Upserting %d restaurants into restaurants_raw …", len(restaurants))
            upsert_restaurants(cursor, restaurants)
            conn.commit()
            log.info("Restaurants upserted successfully")
        else:
            log.info("No new or modified restaurants found")

        # Sync menu items
        log.info("Syncing menu items")
        menu_items_endpoint = MENU_ITEMS_API.format(base_url=BASE_URL)
        menu_items = fetch_all_pages(menu_items_endpoint, last_synced_at)

        if menu_items:
            log.info("Upserting %d menu items into menu_items …", len(menu_items))
            upsert_menu_items(cursor, menu_items)
            conn.commit()
            log.info("Menu items upserted successfully")
        else:
            log.info("No new or modified menu items found")

        update_last_synced_at(cursor)
        conn.commit()

        cursor.close()
        conn.close()

        elapsed = (datetime.now() - start).total_seconds()
        log.info(
            "Sync complete — %d restaurants, %d menu items synced in %.1f s",
            len(restaurants), len(menu_items), elapsed
        )

    except requests.exceptions.ConnectionError:
        log.error("Could not connect to API — is the local server running?")
        raise

    except requests.exceptions.HTTPError as e:
        log.error("API returned an error: %s", e)
        raise

    except Exception:
        log.exception("Sync job FAILED with an unexpected error")
        raise


if __name__ == "__main__":
    run_sync()