import asyncio
from app.services.agents import supabase

async def search():
    keywords = ["Euclid", "Parallel", "Postulate", "Geometry"]
    for kw in keywords:
        res = supabase.table('documents').select('content,metadata').ilike('content', f'%{kw}%').execute()
        print(f"Keyword '{kw}': {len(res.data)} matches")
        for i, row in enumerate(res.data[:2]):
            print(f"  - [Subject: {row['metadata'].get('subject')}] {row['content'][:150]}...")

if __name__ == "__main__":
    asyncio.run(search())
