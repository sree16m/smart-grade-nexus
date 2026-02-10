
import asyncio
from app.services.agents import search_knowledge_base
import json

async def debug_rag():
    query = "State Euclidean parallel postulate"
    subject = "Maths"
    
    print(f"--- Simulating RAG Search for: '{query}' [Subject: {subject}] ---")
    context = await search_knowledge_base(query, subject, limit=5)
    
    if not context.strip():
        print("RESULT: No context found!")
    else:
        print("RESULT: Context found!")
        print("-" * 30)
        # Show raw escaped text to see what's actually there
        print(repr(context[:1000]))
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(debug_rag())
