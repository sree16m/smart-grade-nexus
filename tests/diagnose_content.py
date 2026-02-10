
import asyncio
from app.services.agents import supabase
import re

async def diagnose():
    res = supabase.table('documents').select('content').limit(100).execute()
    chunks = res.data or []
    
    garbage_chunks = 0
    total_len = 0
    english_char_count = 0
    
    for i, row in enumerate(chunks):
        content = row['content']
        total_len += len(content)
        # Count English characters
        english_chars = len(re.findall(r'[a-zA-Z]', content))
        english_char_count += english_chars
        
        # If less than 10% is English, it might be garbage or another language
        if len(content) > 0 and (english_chars / len(content)) < 0.1:
            garbage_chunks += 1
            if garbage_chunks <= 3:
                print(f"Sample Garbage Chunk {garbage_chunks}: {repr(content[:100])}")
    
    print(f"\nTotal chunks analyzed: {len(chunks)}")
    print(f"Chunks with <10% English: {garbage_chunks}")
    if len(chunks) > 0:
        print(f"Avg English density: {english_char_count / total_len:.2%}")

if __name__ == "__main__":
    asyncio.run(diagnose())
