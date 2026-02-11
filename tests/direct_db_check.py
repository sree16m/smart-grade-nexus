import os
import asyncio
from supabase import create_client, Client
from dotenv import load_dotenv

# Load from .env if it exists
load_dotenv()

# Hardcoded or from env
URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")

async def check_db():
    if not URL or not KEY:
        print("Error: SUPABASE_URL and SUPABASE_KEY must be set in env.")
        return

    supabase: Client = create_client(URL, KEY)
    
    print("Checking Database Connection...")
    try:
        res = supabase.table('documents').select('id', count='exact').limit(1).execute()
        print(f"Total Rows in 'documents': {res.count}")
    except Exception as e:
        print(f"Failed to connect or query: {e}")

if __name__ == "__main__":
    asyncio.run(check_db())
