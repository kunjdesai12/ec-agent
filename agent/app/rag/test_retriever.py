# test_retriever.py
import asyncio
from agent.app.config import get_settings
from agent.app.rag.retriever import get_retriever, format_chunks_for_prompt

async def test():
    print("=== Testing pgvector Retriever ===\n")
    
    retriever = get_retriever()

    # Test 1 — basic semantic search
    print("Test 1: 'something creamy and mild from foodies baithak'")
    chunks = await retriever.search("something creamy and mild")
    print(format_chunks_for_prompt(chunks))
    print()

    # Test 2 — cuisine filter
    print("Test 2: 'vegetrian food' with cuisine filter")
    chunks = await retriever.search("vegetaian food", cuisine="Italian")
    print(format_chunks_for_prompt(chunks))
    print()

    # Test 3 — price filter
    print("Test 3: 'quick snack' under ₹150")
    chunks = await retriever.search("quick snack", max_price=150)
    print(format_chunks_for_prompt(chunks))
    print()

    # Test 4 — geo filter (use coordinates from your data)
    print("Test 4: 'pizza' near Vadodara")
    chunks = await retriever.search(
        "pizza",
        user_lat=22.31,
        user_lon=73.16,
        max_distance_km=10.0
    )
    print(format_chunks_for_prompt(chunks))

asyncio.run(test())