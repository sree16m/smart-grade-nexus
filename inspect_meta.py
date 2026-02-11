import asyncio
from app.core.config import settings
from supabase import create_client

def inspect_metadata():
    print("Connecting to Supabase...")
    try:
        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        
        # Fetch a few rows with metadata
        response = supabase.table("documents").select("metadata").limit(5).execute()
        
        if response.data:
            print("Sample Metadata:")
            for i, row in enumerate(response.data):
                print(f"Row {i+1}: {row.get('metadata')}")
        else:
            print("No data found.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_metadata()
