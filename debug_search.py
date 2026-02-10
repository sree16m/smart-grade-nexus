
import asyncio
from app.services.agents import search_knowledge_base

async def find_answer():
    question = "Euclid Parallel Postulate single line through point"
    subject = "Maths"
    
    print(f"Searching Knowledge Base for: '{question}' in Subject: '{subject}'...")
    
    # Use the existing search function in agents.py
    context = await search_knowledge_base(question, subject, limit=5)
    
    print("\n--- Retrieved Context from Vector DB ---")
    try:
        print(context)
    except UnicodeEncodeError:
        print(context.encode('ascii', errors='ignore').decode('ascii'))
    print("\n----------------------------------------")

    if not context.strip():
        print("No relevant context found in the database.")
    else:
        print("Answer can be derived from the above context.")

if __name__ == "__main__":
    asyncio.run(find_answer())
