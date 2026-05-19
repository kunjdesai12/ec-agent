import os
import json

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from llama_cpp import Llama

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

app = FastAPI(title="Delivery Agent — Semantic Search API")

# Database config (from sync_raw_data.py)
DB_CONFIG = dict(
    dbname="postgres",
    user="postgres.eilbullhdgvzgflqzgnq",
    password="Agent@DB_SUPA",
    host="aws-1-ap-southeast-1.pooler.supabase.com",
    port="6543",
)

embedder = SentenceTransformer(EMBEDDING_MODEL)

# Schemas
class UserPromptRequest(BaseModel):
    prompt: str

class ConfidenceScores(BaseModel):
    overall: float | None = None

class SearchResponse(BaseModel):
    restaurant_name: str | None
    menu_item: str | None
    cuisine: str | None
    confidence: ConfidenceScores
    top_matches: list[dict]


# Step 1 — Entity extraction
EXTRACTION_SYSTEM_PROMPT = """
You are a structured data extractor for a food delivery app.

Given a user message, extract the following fields:
- restaurant_name: the name of the restaurant the user mentions (or null)
- menu_item: the specific food or drink item the user mentions (or null)
- cuisine: the type of cuisine ONLY IF the user explicitly mentions it (or null)

Rules:
- Extract ONLY what is literally present in the user's message. Do NOT invent, substitute, or guess items not mentioned.
- Food items may be in any language or transliterated form (e.g. "pav bhaji", "dal makhani", "chole bhature"). Extract them exactly as written.
- Restaurant names may include prefixes/suffixes like “Restaurant”, “Hotel”, “Cafe”, “Dhaba”, “Kitchen”, “Food”, “Dining”, “Eatery”, “Bistro”, “House”, etc. Treat them as part of the restaurant name and preserve the exact name as written in the prompt.
- Never replace or normalize a food item with a different item from your knowledge.
- Never infer cuisine from the food item name.
- Return ONLY a valid JSON object. No explanation, no markdown, no extra text.
- If a field is not present, use null.

Example:
User: "I want to eat pav bhaji"
Output: {"restaurant_name": null, "menu_item": "pav bhaji", "cuisine": null}
"""

# Loading the Qwen model locally
LLAMA_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")

llm = Llama(
    model_path=LLAMA_MODEL_PATH,
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=99,
    temperature=0.2,
    verbose=False
)

def extract_entities(prompt: str, cache_prompt: bool = False) -> dict:

    """ Runs the user prompt and returns a dict with restaurant_name, menu_item, cuisine. """

    full_prompt = f"""{EXTRACTION_SYSTEM_PROMPT}\nUser message: {prompt}\nJSON:"""

    output = llm(full_prompt, max_tokens=256, stop=["\n\n", "\nUser message:"])

    raw_content = output["choices"][0]["text"].strip()
    
    try:
        extracted = json.loads(raw_content)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if match:
            extracted = json.loads(match.group())
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Could not parse JSON from Qwen response: {raw_content}"
            )
    return {
        "restaurant_name": extracted.get("restaurant_name"),
        "menu_item":       extracted.get("menu_item"),
        "cuisine":         extracted.get("cuisine"),
    }


# Step 2 — Embed a string locally
def embed(text: str) -> list[float]:
    """
    Generates a vector embedding for the given text using the local
    SentenceTransformer model. Returns a plain Python list for Supabase.
    """
    return embedder.encode(text, convert_to_numpy=True).tolist()


# Step 3 — Vector similarity search in pgvector
def search_embeddings(query: str, limit: 9999) -> list[dict]:
    """
    Single vector search against restaurant_embeddings using pgvector.
    Returns all rows sorted by similarity (top-k applied later).
    """
    query_vec = embed(query)
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    cursor.execute(
        """
        SELECT *,
            (embedding <#> %s::vector) AS similarity
        FROM restaurant_embeddings
        ORDER BY similarity ASC
        LIMIT %s
        """,
        (query_vec, limit)
    )
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    # Convert similarity to descending (higher is better)
    for r in results:
        r["similarity"] = 1.0 - r["similarity"]  # If using <#> (L2 distance)
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results


# Step 4 — Merge: pick the best canonical match for each extracted field
def best_canonical_match(matches: list[dict], name_key: str = "name") -> tuple[str | None, float | None]:
    """
    From a list of similarity-scored DB rows, returns the top match's
    canonical name and its confidence score. Returns (None, None) if empty.
    """
    if not matches:
        return None, None
    best = matches[0]
    return best.get(name_key), round(best.get("similarity", 0.0), 4)


# Step 5 — Determine how many top matches to return based on extraction confidence
def get_top_k(extracted: dict) -> int:
    """
    Determines how many top matches to return based on how many
    fields were extracted. More fields = higher confidence = fewer results needed.
    """
    filled = sum(1 for v in extracted.values() if v is not None)
    if filled == 3:
        return 1
    elif filled == 2:
        return 1
    elif filled == 1:
        return 3
    else:
        return 5


# Main endpoint
@app.post("/extract", response_model=SearchResponse)
def extract_and_search(request: UserPromptRequest):
    """
    Full pipeline:
    1. Extract entities from user prompt
    2. Embed each extracted field
    3. Vector search all matches in database
    4. Return canonical best match + all candidates for each field
    """
    extracted = extract_entities(request.prompt)

    restaurant_raw = extracted["restaurant_name"]
    menu_item_raw  = extracted["menu_item"]
    cuisine_raw    = extracted["cuisine"]

    query_parts = []
    if restaurant_raw: query_parts.append(f"Restaurant: {restaurant_raw}")
    if menu_item_raw:  query_parts.append(f"Menu Item: {menu_item_raw}")
    if cuisine_raw:    query_parts.append(f"Cuisine: {cuisine_raw}")

    combined_query = ". ".join(query_parts) if query_parts else request.prompt

    k = get_top_k(extracted)
    top_matches = search_embeddings(combined_query, limit=k)

    best = top_matches[0] if top_matches else {}

    overall_conf = round(best.get("similarity", 0.0), 4) if best else None

    # Strip embedding vector from matches before returning
    for match in top_matches:
        match.pop("embedding", None)

    return SearchResponse(
        restaurant_name = restaurant_raw,
        menu_item       = menu_item_raw,
        cuisine         = cuisine_raw,
        confidence = ConfidenceScores(overall=overall_conf),
        top_matches = top_matches,
    )

@app.get("/health")
async def health():
    return {"status": "ok", "embedding_model": EMBEDDING_MODEL}