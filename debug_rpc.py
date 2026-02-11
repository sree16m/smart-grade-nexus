import asyncio
from app.core.config import settings
from supabase import create_client

def debug_rpc():
    print("Connecting to Supabase...")
    try:
        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        
        print("Calling RPC 'get_uploaded_books'...")
        response = supabase.rpc("get_uploaded_books").execute()
        
        if hasattr(response, 'data'):
            print(f"Data returned: {response.data}")
            if not response.data:
                print("RPC returned an empty list.")
        else:
            print(f"Unexpected response format: {response}")

        # Also check if any data exists at all
        base_res = supabase.table("documents").select("id", count="exact").limit(1).execute()
        print(f"Total documents in table: {base_res.count}")

    except Exception as e:
        print(f"Error calling RPC: {e}")

if __name__ == "__main__":
    debug_rpc()
