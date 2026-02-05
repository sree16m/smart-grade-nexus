import asyncio
from app.core.config import settings
from supabase import create_client

def check_db():
    print("Connecting to Supabase...")
    try:
        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        
        # Fetch all rows
        response = supabase.table("documents").select("content, metadata").execute()
        
        rows = response.data
        if not rows:
            print("No documents found in the database.")
            return

        print(f"Total Document Chunks found: {len(rows)}")
        
        # Aggregate by book_name and show samples
        stats = {}
        samples = []
        for row in rows:
            meta = row.get('metadata', {}) or {}
            book_name = meta.get('book_name', 'Unknown/None')
            stats[book_name] = stats.get(book_name, 0) + 1
            
            # Store a sample content for this book if we haven't seen it yet
            if book_name not in [s['book'] for s in samples]:
                 samples.append({"book": book_name, "snippet": row.get('content', '')[:100].replace('\n', ' ')})
            
        print("\n--- Ingestion Report ---")
        for book, count in stats.items():
            print(f"[Book]: '{book}'")
            print(f"   - Total Paragraphs/Chunks: {count}")
            # Find sample
            sample = next((s['snippet'] for s in samples if s['book'] == book), "N/A")
            print(f"   - Sample Text: \"{sample}...\"")
            print("")
            
        print("\n--- Ingestion Report ---")
        for book, count in stats.items():
            print(f"[Book]: '{book}'")
            print(f"   - Total Paragraphs/Chunks: {count}")
            print("   - Status: likely complete" if count > 10 else "   - Status: Warning (very few chunks)")
            print("")
            
    except Exception as e:
        print(f"Error connecting to DB: {e}")

if __name__ == "__main__":
    check_db()
