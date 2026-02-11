import asyncio
from app.core.config import settings
from supabase import create_client

def verify_match_rpc():
    print("Connecting to Supabase...")
    try:
        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        
        # Dummy embedding (768 zeros)
        dummy_emb = [0.0] * 768
        
        print("Calling RPC 'match_documents' with dummy data...")
        response = supabase.rpc("match_documents", {
            "query_embedding": dummy_emb,
            "match_threshold": 0.5,
            "match_count": 1
        }).execute()
        
        print("RPC 'match_documents' call successful.")
        return True
    except Exception as e:
        print(f"Error calling 'match_documents': {e}")
        return False

if __name__ == "__main__":
    verify_match_rpc()
