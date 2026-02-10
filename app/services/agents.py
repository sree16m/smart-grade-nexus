import google.generativeai as genai
from supabase import create_client, Client
from app.core.config import settings
from typing import List, Dict, Any
import json

# Initialize Clients
genai.configure(api_key=settings.GOOGLE_API_KEY)
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
# Initialize RAG Model
embedding_model = "models/gemini-embedding-001"
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
            task_type="retrieval_query",
            output_dimensionality=768
        )['embedding']
    
    query_emb = await asyncio.to_thread(_embed)
    
    # 2. Search Supabase (Threadpool)
    def _search(search_subject):
        return supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_emb,
                "match_threshold": 0.5,
                "match_count": limit,
                "filter": {"subject": search_subject}
            }
        ).execute()

    response = await asyncio.to_thread(_search, subject)
    
    # Retry logic: Handle "Physic" vs "Physics" mismatch
    if not response.data and subject == "Physics":
        response = await asyncio.to_thread(_search, "Physic")
    
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
        
        # 1. Fail fast if no context is found
        if not context.strip():
             return {
                 "score": 0,
                 "feedback": f"Evaluation skipped: No reference material found for subject '{self.subject}'. Please ensure the Knowledge Base has relevant textbooks uploaded.",
                 "citation": None
             }
        
        prompt = f"""
        Role: Strict Academic Grader and Subject Matter Expert for {self.subject}.
        
        Primary Objectives:
        1. REFERENCE: Search the provided 'Context' strictly to find the ground truth for the question.
        2. EXTRACTION: Identify specific source details (Chapter, Topic, Semester, Page No) present in the text.
        3. SOLUTION: Formulate the ideal Correct Answer based *only* on the Context.
        4. EVALUATION: Compare the Student Answer against the Correct Answer and assign a score.

        Context (Official Subject Knowledge Base):
        {context}
        
        Question: {question}
        Student Answer: {answer}
        Max Marks: {max_marks}
        
        Output JSON format strictly:
        {{
            "correct_answer_from_context": "The ideal answer derived strictly from the provided context.",
            "source_metadata": {{
                "subject": "{self.subject}",
                "semester": "Extract from context text or 'Unknown'",
                "chapter": "Extract chapter name/number from context text or 'Unknown'",
                "topic": "Extract specific topic from context text or 'Unknown'"
            }},
            "grading": {{
                "score": float, 
                "feedback": "Concise feedback explaining the gap between student answer and ground truth.",
                "citation": "Direct quote from context used for verification"
            }}
        }}
        """
        
        # Non-blocking generation
        response = await asyncio.to_thread(
            lambda: generative_model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        )
        try:
            return json.loads(response.text)
        except Exception as e:
            # 2. Return system error as feedback instead of silently swallowing it
            return {
                "score": 0, 
                "feedback": f"System Error during grading (AI parsing failed): {str(e)}", 
                "citation": None
            }
