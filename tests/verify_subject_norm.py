import asyncio
from app.services.agents import search_knowledge_base

async def verify():
    print("--- Verifying Subject Normalization ---")
    # Search for 'Euclid' with subject 'Mathematics'
    # The DB has chunks with 'Maths'
    context = await search_knowledge_base("Euclid Parallel Postulate", "Mathematics")
    if context:
        print("Success: Found context even with 'Mathematics' vs 'Maths' mismatch!")
        print(f"Context snippet: {context[:200]}...")
    else:
        print("Failure: Could not find context. Subject normalization might still be failing or data is truly missing.")

if __name__ == "__main__":
    asyncio.run(verify())
