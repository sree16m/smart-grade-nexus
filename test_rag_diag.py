import asyncio
from app.core.config import settings
from app.services.agents import search_knowledge_base
import google.generativeai as genai

async def test_rag():
    print("Testing RAG Retrieval...")
    subject = "Maths"
    # A query that should match something in the 564 chunks
    query = "What is number system?"
    
    print(f"Searching for: '{query}' in subject: '{subject}'")
    
    # Test 1: Simple Search
    context = await search_knowledge_base(query, subject, limit=3)
    print("\n--- Match Result (No Chapter Filter) ---")
    if context:
        print(context[:500] + "...")
    else:
        print("FAIL: No context returned.")

    # Test 2: Search with Chapter Filter
    chapter = "1"
    print(f"\nSearching for: '{query}' in subject: '{subject}', chapter: '{chapter}'")
    context_ch = await search_knowledge_base(query, subject, limit=3, filter={"chapter": chapter})
    print("--- Match Result (With Chapter Filter) ---")
    if context_ch:
        print(context_ch[:500] + "...")
    else:
        print("FAIL: No context returned with chapter filter.")

    # Test 3: Search with Case variation
    subject_var = "maths"
    print(f"\nSearching for: '{query}' in subject: '{subject_var}'")
    context_var = await search_knowledge_base(query, subject_var, limit=3)
    print("--- Match Result (Subject: 'maths') ---")
    if context_var:
        print(context_var[:500] + "...")
    else:
        print("FAIL: No context returned for variation.")

if __name__ == "__main__":
    asyncio.run(test_rag())
