import google.generativeai as genai
from supabase import create_client, Client
from app.core.config import settings
from typing import List, Dict, Any
import json

# Initialize Clients
genai.configure(api_key=settings.GOOGLE_API_KEY)
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
# Initialize RAG Model
embedding_model = "models/text-embedding-004"
generative_model = genai.GenerativeModel("gemini-2.0-flash")

import asyncio

# ... imports ...

async def search_knowledge_base(query: str, subject: str, limit: int = 3) -> str:
    """Retrieves relevant chunks from Vector DB (Non-blocking)."""
    
    # 1. Generate Query Embedding (Threadpool)
    def _embed():
         return genai.embed_content(
            model=embedding_model,
            content=query,
            task_type="retrieval_query"
        )['embedding']
    
    query_emb = await asyncio.to_thread(_embed)
    
    # 2. Search Supabase (Threadpool)
    def _search():
        return supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_emb,
                "match_threshold": 0.5,
                "match_count": limit,
                "filter": {"subject": subject}
            }
        ).execute()

    response = await asyncio.to_thread(_search)
    
    # 3. Format Context
    context = ""
    if response.data:
        for item in response.data:
            context += f"---\n{item['content']}\n"
    return context

class TopicAgent:
    def __init__(self, subject: str):
        self.subject = subject

    async def categorize(self, questions: List[Dict[str, str]]) -> List[Dict]:
        """Maps questions to topics based on KB."""
        results = []
        for q in questions:
            q_text = q['text']
            # RAG: Find where this concept is mentioned in the book
            context = await search_knowledge_base(q_text, self.subject, limit=2)
            
            prompt = f"""
            Task: Identify the specific Subject Topic and Chapter for the following question.
            Context from Textbook:
            {context}
            
            Question: {q_text}
            
            Output JSON only: {{ "topic_path": "Chapter > Subtopic", "confidence": 0.95 }}
            """
            
            # Non-blocking generation
            response = await asyncio.to_thread(
                lambda: generative_model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            )
            try:
                data = json.loads(response.text)
                results.append({
                    "question_id": q['id'],
                    "topic_path": data.get("topic_path", "Unknown"),
                    "confidence": data.get("confidence", 0.0)
                })
            except:
                 results.append({"question_id": q['id'], "error": "Parsing Failed"})
                 
        return results

class GradingAgent:
    def __init__(self, subject: str):
        self.subject = subject

    async def evaluate(self, question: str, answer: str, max_marks: int) -> Dict:
        """Grades answer using Ground Truth from KB."""
        # RAG: Get the official explanation
        context = await search_knowledge_base(question, self.subject, limit=3)
        
        prompt = f"""
        Role: Strict Academic Grader for {self.subject}.
        Task: Grade the student's answer based ONLY on the provided Context (Ground Truth).
        
        Question: {question}
        Student Answer: {answer}
        Max Marks: {max_marks}
        
        Context (Official Textbook):
        {context}
        
        Output JSON:
        {{
            "score": float, 
            "feedback": "Concise justification",
            "citation": "Quote from context used for grading"
        }}
        """
        
        # Non-blocking generation
        response = await asyncio.to_thread(
            lambda: generative_model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        )
        try:
            return json.loads(response.text)
        except Exception as e:
            return {"error": str(e), "score": 0}
