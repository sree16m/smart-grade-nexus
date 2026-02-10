import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    # Try getting from settings if env var not set directly
    try:
        from app.core.config import settings
        api_key = settings.GOOGLE_API_KEY
    except ImportError:
        print("Could not find GOOGLE_API_KEY in env or settings.")
        exit(1)

genai.configure(api_key=api_key)

print("Listing available models...")
try:
    for m in genai.list_models():
        if 'embedContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")

print("\nTesting embedding with 'models/text-embedding-004'...")
try:
    genai.embed_content(
        model="models/text-embedding-004",
        content="Hello world",
        task_type="retrieval_document"
    )
    print("Success: models/text-embedding-004 is working.")
except Exception as e:
    print(f"Failed: {e}")

print("\nTesting embedding with 'models/embedding-001'...")
try:
    genai.embed_content(
        model="models/embedding-001",
        content="Hello world",
        task_type="retrieval_document"
    )
    print("Success: models/embedding-001 is working.")
except Exception as e:
    print(f"Failed: {e}")
