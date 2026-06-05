import psycopg2
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-m3', device='cuda')

conn = psycopg2.connect(
    host="easycater-vector-db-v2.cngqs42w0l3u.ap-south-1.rds.amazonaws.com",
    database="easycater_vectors",
    user="vectoradmin",
    password="vector123"
)
cur = conn.cursor()

cur.execute("SELECT menu_item_id, combined_text, food_item, restaurant_name, cuisine_name FROM restaurant_embeddings")
rows = cur.fetchall()
print(f"Total rows: {len(rows)}")

BATCH_SIZE = 32

for i in range(0, len(rows), BATCH_SIZE):
    batch = rows[i:i+BATCH_SIZE]
    ids            = [r[0] for r in batch]
    combined_texts = [r[1] or "" for r in batch]
    food_items     = [r[2] or "" for r in batch]
    rest_names     = [r[3] or "" for r in batch]
    cuisines       = [r[4] or "" for r in batch]

    emb_combined = model.encode(combined_texts, normalize_embeddings=True).tolist()
    emb_food     = model.encode(food_items,     normalize_embeddings=True).tolist()
    emb_name     = model.encode(rest_names,     normalize_embeddings=True).tolist()
    emb_cuisine  = model.encode(cuisines,       normalize_embeddings=True).tolist()

    for j, menu_item_id in enumerate(ids):
        cur.execute("""
            UPDATE restaurant_embeddings SET
                embedding = %s,
                food_item_embedding = %s,
                restaurant_name_embedding = %s,
                cuisine_embedding = %s,
                updated_at = now()
            WHERE menu_item_id = %s
        """, (
            emb_combined[j],
            emb_food[j],
            emb_name[j],
            emb_cuisine[j],
            menu_item_id
        ))

    conn.commit()
    print(f"Processed {min(i+BATCH_SIZE, len(rows))}/{len(rows)}")

cur.close()
conn.close()
print("Done!")