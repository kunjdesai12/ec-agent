import os
import asyncio
import random
import psycopg2
import psycopg2.extras
import anthropic
from sentence_transformers import SentenceTransformer

DB_CONFIG = {
    "host": "easycater-vector-db-v2.cngqs42w0l3u.ap-south-1.rds.amazonaws.com",
    "database": "easycater_vectors",
    "user": "vectoradmin",
    "password": ""
}
ANTHROPIC_API_KEY = ""
CONCURRENCY = 15
BATCH_SIZE = 64
EMBED_BATCH_SIZE = 128

print("Loading bge-m3 model...")
model = SentenceTransformer('BAAI/bge-m3', device='cuda')
print("Model loaded.")

SYSTEM_PROMPT = """You are a food discovery assistant. Given structured restaurant and menu data,
write a concise 2-3 sentence natural language description that captures the dish, cuisine style,
dietary suitability, and value. Be specific and natural. Return only the description, no preamble."""

async def enrich_row(client, semaphore, row):
    menu_item_id, restaurant_name, cuisine_name, food_item, price, rating = row
    prompt = (
        f"Restaurant: {restaurant_name}. "
        f"Cuisine: {cuisine_name}. "
        f"Menu Item: {food_item}. "
        f"Price: ₹{price}. "
        f"Rating: {rating}."
    )
    async with semaphore:
        for attempt in range(5):
            try:
                response = await client.messages.create(
                    model="claude-haiku-4-5",
                    max_tokens=120,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}]
                )
                return menu_item_id, response.content[0].text.strip()
            except anthropic.RateLimitError:
                wait = 2 ** attempt + random.uniform(0, 1)
                print(f"Rate limit hit for {menu_item_id}, retrying in {wait:.1f}s...")
                await asyncio.sleep(wait)
            except Exception as e:
                print(f"Error for {menu_item_id}: {e}")
                await asyncio.sleep(2)
        fallback = f"{restaurant_name} serves {food_item} ({cuisine_name}) at ₹{price}."
        return menu_item_id, fallback

async def enrich_batch(rows):
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [enrich_row(client, semaphore, row) for row in rows]
    results = {}
    total = len(tasks)
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        mid, text = await coro
        results[mid] = text
        if (i + 1) % 500 == 0:
            print(f"  Enriched {i+1}/{total}")
    return results

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Only fetch rows not yet enriched
    print("Fetching unenriched rows...")
    cur.execute("""
        SELECT menu_item_id, restaurant_name, cuisine_name,
               food_item, price, rating
        FROM restaurant_embeddings
        WHERE enriched_embedding IS NULL
        ORDER BY menu_item_id
    """)
    rows = cur.fetchall()
    print(f"Rows to process: {len(rows)}")

    total = len(rows)
    processed = 0

    # Process in chunks so we commit progress frequently
    CHUNK_SIZE = 1000
    for chunk_start in range(0, total, CHUNK_SIZE):
        chunk = rows[chunk_start:chunk_start+CHUNK_SIZE]
        print(f"\nChunk {chunk_start//CHUNK_SIZE + 1}: enriching {len(chunk)} rows...")

        # Step 1 — Enrich with Haiku async
        enriched_map = asyncio.run(enrich_batch(chunk))

        # Step 2 — Embed enriched texts in large batches (GPU loves this)
        ids            = [r[0] for r in chunk]
        enriched_texts = [enriched_map[r[0]] for r in chunk]

        print(f"  Embedding {len(enriched_texts)} texts...")
        all_embeddings = []
        for i in range(0, len(enriched_texts), EMBED_BATCH_SIZE):
            batch_embs = model.encode(
                enriched_texts[i:i+EMBED_BATCH_SIZE],
                normalize_embeddings=True,
                show_progress_bar=False
            ).tolist()
            all_embeddings.extend(batch_embs)

        # Step 3 — Bulk update
        print(f"  Writing to DB...")
        update_data = [
            (enriched_texts[i], all_embeddings[i], ids[i])
            for i in range(len(ids))
        ]
        psycopg2.extras.execute_batch(cur, """
            UPDATE restaurant_embeddings
            SET enriched_text = %s,
                enriched_embedding = %s,
                updated_at = now()
            WHERE menu_item_id = %s
        """, update_data, page_size=200)

        conn.commit()
        processed += len(chunk)
        print(f"  Progress: {processed}/{total}")

    cur.close()
    conn.close()
    print("\nAll done!")

if __name__ == "__main__":
    main()