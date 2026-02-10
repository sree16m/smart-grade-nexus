import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    try:
        from app.core.config import settings
        api_key = settings.GOOGLE_API_KEY
    except:
        pass
genai.configure(api_key=api_key)

model_name = "models/gemini-embedding-001"

print(f"Testing dimensions for {model_name}...")
try:
    result = genai.embed_content(
        model=model_name,
        content="Test",
        task_type="retrieval_document"
    )
    dims = len(result['embedding'])
    print(f"Default Dimensions: {dims}")
    
    # Try requesting 768
    print("Trying to request 768 dimensions...")
    try:
        result_768 = genai.embed_content(
            model=model_name,
            content="Test",
            task_type="retrieval_document",
            output_dimensionality=768
        )
        dims_768 = len(result_768['embedding'])
        print(f"Requested 768 Dimensions: {dims_768}")
    except Exception as e:
        print(f"Failed to set output_dimensionality: {e}")

except Exception as e:
    print(f"Error: {e}")
