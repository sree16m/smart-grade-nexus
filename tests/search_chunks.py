import asyncio
from app.services.agents import supabase

async def search():
    # Fetch 10 chunks from the 'Maths' subject to see what they contain
    res = supabase.table('documents').select('content').eq('metadata->>subject', 'Maths').limit(50).execute()
    print(f"Found {len(res.data)} chunks for 'Maths'")
    
    for i, row in enumerate(res.data):
        content = row['content']
        if "Euclid" in content or "Parallel" in content:
            print(f"Found content at index {i}:")
            print(f"{content[:500]}...")
            return

    print("No Euclid content found in first 50 chunks of 'Maths'.")

if __name__ == "__main__":
    asyncio.run(search())
