# Google Colab script to batch-update Supabase/Postgres
# This avoids Supabase SQL editor timeout issues by updating in chunks.


import psycopg2
from psycopg2.extras import execute_batch
from tqdm import tqdm

# =========================
# SUPABASE CONNECTION INFO
# =========================
DB_NAME="postgres"
DB_USER="postgres.eilbullhdgvzgflqzgnq"
DB_PASSWORD="Agent@DB_SUPA"
DB_HOST="aws-1-ap-southeast-1.pooler.supabase.com"
DB_PORT="6543"

BATCH_SIZE = 5000

# =========================
# CONNECT
# =========================
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)

conn.autocommit = False
cur = conn.cursor()

print("Connected to Supabase!")

# =========================
# COUNT TOTAL ROWS
# =========================
count_query = """
SELECT COUNT(*)
FROM menu_items mi
JOIN cuisines c        ON c.cuisines_id   = mi.cuisines_id
JOIN restaurants_raw r ON r.restaurant_id = mi.business_id;
"""

cur.execute(count_query)
total_rows = cur.fetchone()[0]

print(f"Total rows to process: {total_rows}")

# =========================
# PROCESS IN BATCHES
# =========================
offset = 0

while offset < total_rows:

    print(f"\nProcessing batch OFFSET {offset}")

    # Fetch batch data
    fetch_query = f"""
    SELECT
        mi.title,
        c.cuisines_name,
        mi.price,
        r.rating,
        r.latitude,
        r.longitude
    FROM menu_items mi
    JOIN cuisines c        ON c.cuisines_id   = mi.cuisines_id
    JOIN restaurants_raw r ON r.restaurant_id = mi.business_id
    ORDER BY mi.menu_item_id
    LIMIT {BATCH_SIZE}
    OFFSET {offset};
    """

    cur.execute(fetch_query)
    rows = cur.fetchall()

    if not rows:
        break

    # Batch update
    update_query = """
    UPDATE restaurant_embeddings re
    SET
        cuisine_name = data.cuisine_name,
        price        = data.price,
        rating       = data.rating,
        latitude     = data.latitude,
        longitude    = data.longitude
    FROM (
        VALUES (%s, %s, %s, %s, %s, %s)
    ) AS data(
        food_item,
        cuisine_name,
        price,
        rating,
        latitude,
        longitude
    )
    WHERE re.food_item = data.food_item;
    """

    for row in tqdm(rows):

        execute_batch(
            cur,
            update_query,
            [row],
            page_size=100
        )

    conn.commit()

    offset += BATCH_SIZE

print("\nUpdate completed successfully!")

cur.close()
conn.close()