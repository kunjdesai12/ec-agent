import os
import json
import re

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
- cuisine: the broad cuisine category ONLY IF the user explicitly states it as a category (e.g. "Chinese food", "Indian cuisine", "South Indian"). Return null if cuisine is not explicitly stated as a category.

Rules:
- Extract ONLY what is literally present in the user's message. Do NOT invent, substitute, or guess items not mentioned.
- Food items may be in any language or transliterated form (e.g. "pav bhaji", "dal makhani", "chole bhature"). Extract them exactly as written.
- Restaurant names may include prefixes/suffixes like “Restaurant”, “Hotel”, “Cafe”, “Dhaba”, “Kitchen”, “Food”, “Dining”, “Eatery”, “Bistro”, “House”, etc. Treat them as part of the restaurant name and preserve the exact name as written in the prompt.
- Never replace or normalize a food item with a different item from your knowledge.
- Never infer cuisine from the food item name.
- Return ONLY a valid JSON object. No explanation, no markdown, no extra text.
- If a field is not present, use null.
- "cuisine" must be a broad food category. Word fragments inside dish names (e.g. "hakka" in "hakka noodles"), words inside restaurant names (e.g. "Punjabi" in "Punjabi Dhaba"), and food descriptors (e.g. "spicy", "crispy", "sweet") are NOT cuisine values — set cuisine to null in these cases.
- If the user says "I want something spicy" or "I want spicy food", cuisine is null. If the user says "I want Kashmiri food" or "show me Chinese options", extract the cuisine.
- NEVER infer cuisine from your knowledge of what category a dish or restaurant belongs to. "Chicken tikka masala" does not make cuisine "Indian". "Burger" does not make cuisine "American". "McDonald's" does not make cuisine "Fast Food". "Veg manchurian" does not make cuisine "Chinese". "Lasagna" or "Bella Italia" does not make cuisine "Italian". Cuisine is null unless the user explicitly says the cuisine category word themselves.
- When cuisine IS present, return ONLY the bare category word or phrase. Never append extra words to it. Return "Chinese" not "Chinese cuisine". Return "South Indian" not "South Indian food". Return "Italian" not "Italian cuisine".

Example:
User: "I want to eat pav bhaji"
Output: {"restaurant_name": null, "menu_item": "pav bhaji", "cuisine": null}
"""

# Loading the Qwen model locally
LLAMA_MODEL_PATH = "./models/qwen2.5-1.5B-instruct-q4_k_m.gguf"

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
        match = re.search(r"\{.*?\}", raw_content, re.DOTALL)
        if match:
            try:
                extracted = json.loads(match.group())
            except json.JSONDecodeError:
                extracted = {}
        else:
            extracted = {}

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
def search_embeddings(extracted: dict, limit: int = 5) -> list[dict]:
    """
    Field-aware vector search. Builds a targeted query string per extracted
    field so restaurant names match against restaurant names and menu items
    match against menu items, rather than competing in a single combined search.
    """
    restaurant_raw = extracted.get("restaurant_name")
    menu_item_raw  = extracted.get("menu_item")
    cuisine_raw    = extracted.get("cuisine")

    query_parts = []
    if restaurant_raw:
        query_parts.append(f"Restaurant Name: {restaurant_raw}")
    if cuisine_raw:
        query_parts.append(f"Cuisine: {cuisine_raw}")
    if menu_item_raw:
        query_parts.append(f"Menu Item: {menu_item_raw}")

    targeted_query = ". ".join(query_parts) if query_parts else ""

    if not targeted_query:
        return []

    query_vec = embed(targeted_query)

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cursor.execute(
        """
        SELECT *,
            1.0 - (embedding <=> %s::vector) AS similarity
        FROM restaurant_embeddings
        ORDER BY similarity DESC
        LIMIT %s
        """,
        (query_vec, limit)
    )
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()

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

    k = get_top_k(extracted)
    top_matches = search_embeddings(extracted, limit=k)

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