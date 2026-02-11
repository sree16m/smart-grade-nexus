import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

def probe_tables():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    supabase = create_client(url, key)
    
    variations = ["documents", "Documents", "DOCUMENTS", "chunks", "document_chunks", "knowledge"]
    
    for table_name in variations:
        try:
            res = supabase.table(table_name).select("id", count="exact").limit(1).execute()
            print(f"Table '{table_name}': Count = {res.count}")
        except Exception as e:
            if "not found" in str(e).lower() or "not exist" in str(e).lower():
                print(f"Table '{table_name}': Does not exist.")
            else:
                print(f"Table '{table_name}': Error -> {str(e)[:100]}...")

if __name__ == "__main__":
    probe_tables()
