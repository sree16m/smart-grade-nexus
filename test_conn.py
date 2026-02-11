import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

def verify_connection():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        print("Missing URL or KEY in .env")
        return

    print(f"Connecting to: {url.split('//')[1].split('.')[0]} (Project Ref)")
    
    try:
        supabase = create_client(url, key)
        
        # Test 1: Simple select from documents
        print("Testing select from 'documents'...")
        res = supabase.table("documents").select("id", count="exact").limit(1).execute()
        print(f"Result count for 'documents': {res.count}")
        
        # Test 2: Try to find if 'match_documents' exists (check error hint)
        print("Testing RPC 'match_documents'...")
        try:
            supabase.rpc("match_documents", {"query_embedding": [0]*768, "match_threshold": 0, "match_count": 0}).execute()
            print("RPC exists.")
        except Exception as e:
            print(f"RPC Error (expected): {e}")

        # Test 3: List columns of documents if possible
        print("Testing select * from documents limit 0...")
        try:
            res_all = supabase.table("documents").select("*").limit(0).execute()
            print(f"Columns might be available? Success.")
        except Exception as e:
            print(f"Select * Error: {e}")

    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    verify_connection()
