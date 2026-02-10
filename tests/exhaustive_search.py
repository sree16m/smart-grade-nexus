
import asyncio
from app.services.agents import supabase
import json

async def exhaustive_debug():
    print("--- Checking All Ingested Subjects/Books ---")
    # Get unique subjects/books from metadata
    res = supabase.table('documents').select('metadata').execute()
    metadata_list = res.data or []
    
    unique_subjects = set()
    unique_books = set()
    for m in metadata_list:
        meta = m.get('metadata', {})
        unique_subjects.add(meta.get('subject'))
        unique_books.add(meta.get('book_name'))
    
    print(f"Unique Subjects in DB: {unique_subjects}")
    print(f"Unique Books in DB: {unique_books}")
    print(f"Total Chunks in DB: {len(metadata_list)}")

    print("\n--- Searching for 'Euclid' content (Literal Match) ---")
    # Using ilike on content column
    euclid_res = supabase.table('documents').select('content,metadata').ilike('content', '%Euclid%').execute()
    print(f"Chunks containing 'Euclid': {len(euclid_res.data)}")
    for i, row in enumerate(euclid_res.data[:3]):
        print(f"Match {i+1} [Subject: {row['metadata'].get('subject')}]: {row['content'][:200]}...")

    print("\n--- Searching for 'Geometry' content (Literal Match) ---")
    geo_res = supabase.table('documents').select('content,metadata').ilike('content', '%Geometry%').execute()
    print(f"Chunks containing 'Geometry': {len(geo_res.data)}")

if __name__ == "__main__":
    asyncio.run(exhaustive_debug())
