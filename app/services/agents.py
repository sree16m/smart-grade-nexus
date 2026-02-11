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
import time
from google.api_core.exceptions import ResourceExhausted

async def generate_with_retry(model, prompt, retries=3, delay=2):
    """Wraps generate_content with exponential backoff for 429 errors."""
    for attempt in range(retries):
        try:
            return await asyncio.to_thread(
                lambda: model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            )
        except ResourceExhausted:
            if attempt == retries - 1:
                raise
            sleep_time = delay * (2 ** attempt)
            print(f"Gemini 429 Limit hit. Retrying in {sleep_time}s...")
            await asyncio.sleep(sleep_time)
        except Exception:
            raise

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
                "match_threshold": 0.3, # Slightly lower threshold for broader concept matching
                "match_count": 15,      # Fetch more for internal re-ranking
                "filter": {"subject": search_subject}
            }
        ).execute()

    # Normalize/Map common subject variations
    subject_map = {
        "mathematics": ["Maths", "Math", "Mathematics"],
        "maths": ["Mathematics", "Math", "Maths"],
        "math": ["Mathematics", "Maths", "Math"],
        "physics": ["Physics", "Physic"],
        "biology": ["Biology", "Bio"],
        "chemistry": ["Chemistry", "Chem"],
        "euclid geometry": ["Maths", "Mathematics", "Math"]
    }
    
    search_subjects = [subject]
    # Add variations if they exist in the map (case-insensitive)
    normalized_key = subject.lower()
    if normalized_key in subject_map:
        for variation in subject_map[normalized_key]:
            if variation.lower() != normalized_key:
                search_subjects.append(variation)

    results = []
    for s in search_subjects:
        response = await asyncio.to_thread(_search, s)
        if response.data:
            results.extend(response.data)
            # If we found enough semantic matches, we can stop
            if len(results) >= 5:
                break
        
    vector_results = results
    
    # 2b. Keyword Search (Supabase/Postgres Full Text Search)
    # We assume a 'keyword_search' RPC exists or we use direct filter.
    # Since we can't easily add RPCs without SQL access, we'll try a fallback client-side filter 
    # OR if the user allows, we'd add an RPC. 
    # For Zero-Cost MVP without migration: We will rely on Vector Search primarily, 
    # BUT we can try to boost it by fetching more and filtering Python-side if needed.
    # However, proper Hybrid requires an RPC 'keyword_search'.
    # Let's assume for now we just use the Vector Search as the user has low-code constraints 
    # and likely hasn't set up FTS indexes.
    
    # Wait! The requirement was "Zero-Cost Implementation" of Hybrid.
    # If we cannot create SQL functions, we can't do true FTS on Supabase easily.
    # Alternative: Fetch MORE from Vector (limit=10) and re-rank with BM25 locally?
    # Yes, that's a pure Python solution.
    
    # REVISED STRATEGY: Rerank with BM25
    # 1. Fetch deeper (limit=10 instead of 3)
    # 2. Rerank locally using simple keyword overlap
    
    docs = vector_results
    
    if not docs:
         return ""
         
    # Simple Python-based Re-ranking (TF-IDF style / Keyword Overlap)
    query_terms = set(query.lower().split())
    
    def score_doc(doc):
        content = doc.get('content', '').lower()
        score = 0
        for term in query_terms:
            if term in content:
                score += 1
        return score
        
    # Sort by (Keyword Score DESC, Vector Similarity is implicit in order)
    # Actually, let's just prioritize docs that have the keywords.
    docs.sort(key=score_doc, reverse=True)
    
    # Take top 3 after re-ranking
    top_docs = docs[:limit]

    # 3. Format Context
    context = ""
    for item in top_docs:
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
            
            # Non-blocking generation with retry
            response = await generate_with_retry(generative_model, prompt)
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

    async def evaluate(self, question: str, answer: str, max_marks: int, student_class: str = "9") -> Dict:
        """Grades answer using Ground Truth from KB."""
        # RAG: Get the official explanation - Increased limit for complex topics like Euclid
        context = await search_knowledge_base(question, self.subject, limit=6)
        
        # 1. Fail fast if no context is found
        if not context.strip():
             return {
                 "score": 0,
                 "feedback": f"Evaluation skipped: No reference material found for subject '{self.subject}'. Please ensure the Knowledge Base has relevant textbooks uploaded.",
                 "citation": None
             }
        
        prompt = f"""
        Role: Strict math teacher evaluating a Class {student_class} answer based ONLY on the provided textbook content.
        
        Textbook Content (Official Ground Truth):
        {context}
        
        Question: {question}
        Student's Answer: {answer}
        Max Marks: {max_marks}
        
        Evaluation Criteria:
        1. Accuracy of concepts: Must match textbook definitions and examples exactly.
        2. Step-by-step reasoning: Award points for each correct logical step. Use partial marks for effort if the reasoning is sound but the final result is wrong.
        3. Completeness: Check if all parts of the question are addressed.
        
        Constraint: Do NOT add external knowledge. If the context does not contain the answer, state "Answer not found in context" clearly.

        Output JSON format strictly:
        {{
            "correct_answer_from_context": "The ideal answer derived strictly from the provided context.",
            "source_metadata": {{
                "subject": "{self.subject}",
                "chapter": "Extract chapter name/number from context or 'Unknown'",
                "topic": "Extract specific topic from context or 'Unknown'"
            }},
            "grading": {{
                "score": float, 
                "max_score": {max_marks},
                "score_breakdown": "e.g., 4/10 concepts, 3/10 steps, 3/10 completeness",
                "feedback": "Specific improvements, citing textbook page/chapter where possible.",
                "citation": "Direct quote from context used for verification"
            }}
        }}
        """
        
        # Non-blocking generation with retry
        response = await generate_with_retry(generative_model, prompt)
        try:
            return json.loads(response.text)
        except Exception as e:
            # 2. Return system error as feedback instead of silently swallowing it
            return {
                "score": 0, 
                "feedback": f"System Error during grading (AI parsing failed): {str(e)}", 
                "citation": None
            }
