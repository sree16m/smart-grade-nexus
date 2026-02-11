import os
import asyncio
from supabase import create_client, Client
from dotenv import load_dotenv
import google.generativeai as genai

# Load from .env if it exists
load_dotenv()

# Hardcoded or from env
URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_KEY)

async def check_semantic():
    if not URL or not KEY or not GOOGLE_KEY:
        print("Error: Missing env vars.")
        return

    supabase: Client = create_client(URL, KEY)
    
    query = "How many lines can be drawn through a point and parallel to a given line? Euclid Geometry Playfair Axiom"
    
    # 1. Embed
    print(f"Embedding query: {query}")
    emb = genai.embed_content(
        model="models/gemini-embedding-001",
        content=query,
        task_type="retrieval_query",
        output_dimensionality=768
    )['embedding']
    
    # 2. Match
    print(f"Listing all chunks for 'Maths Class 9 Semester 1 AI Ingest'...")
    res = supabase.table('documents').select('content,metadata').eq('metadata->>book_name', 'Maths Class 9 Semester 1 AI Ingest').execute()
    
    print(f"Found {len(res.data)} chunks in this book. Writing to book_audit.txt...")
    with open("book_audit.txt", "w", encoding="utf-8") as f:
        for i, row in enumerate(res.data):
            f.write(f"\n--- Chunk {i+1} | Metadata: {row['metadata']} ---\n")
            f.write(row['content'])
            f.write("\n")
    print("Done.")

if __name__ == "__main__":
    asyncio.run(check_semantic())
