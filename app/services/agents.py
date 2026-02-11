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

async def search_knowledge_base(query: str, subject: str, limit: int = 5, filter: Dict[str, Any] = None) -> str:
    """Retrieves relevant chunks from Vector DB (Non-blocking) with Metadata Filtering."""
    
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
    def _search(search_filter):
        return supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_emb,
                "match_threshold": 0.3, # Slightly lower threshold for broader concept matching
                "match_count": 15,      # Fetch more for internal re-ranking
                "filter": search_filter
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
        # Prepare filter for this subject
        current_filter = {"subject": s}
        if filter:
            current_filter.update(filter)
            
        response = await asyncio.to_thread(_search, current_filter)
        if response.data:
            results.extend(response.data)
            # If we found enough semantic matches, we can stop
            if len(results) >= limit + 10: # Fetch slightly more than needed for re-ranking
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
    print(f"RAG Debug: Found {len(top_docs)} relevant chunks.")
    for i, item in enumerate(top_docs):
        sim = item.get('similarity', 0)
        print(f"CHUNK {i+1} [Sim: {sim:.3f}]: {item['content'][:150]}...")
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
            Task: Identify the specific Chapter Number and Subject Topic for the following question.
            Context from Textbook:
            {context}
            
            Question: {q_text}
            
            Output JSON only: {{ "chapter": "string", "topic": "string", "confidence": 0.95 }}
            """
            
            # Non-blocking generation with retry
            response = await generate_with_retry(generative_model, prompt)
            try:
                data = json.loads(response.text)
                results.append({
                    "question_id": q['id'],
                    "chapter": data.get("chapter", "Unknown"),
                    "topic": data.get("topic", "Unknown"),
                    "confidence": data.get("confidence", 0.0)
                })
            except:
                 results.append({"question_id": q['id'], "error": "Parsing Failed"})
                 
        return results

class GradingAgent:
    def __init__(self, subject: str):
        self.subject = subject

    async def evaluate(self, question: str, answer: str, max_marks: int, student_class: str = "9", chapter: str = None) -> Dict:
        """Grades answer using Ground Truth from KB."""
        # 1. Prepare Filter
        search_filter = {}
        if chapter and chapter != "Unknown":
            search_filter["chapter"] = chapter
            print(f"RAG Filter: Hard-filtering search scope for Chapter {chapter}")

        # RAG: Get the official explanation
        context = await search_knowledge_base(question, self.subject, limit=6, filter=search_filter)
        
        # 1. Fail fast if no context is found
        if not context.strip():
             return {
                 "score": 0,
                 "feedback": f"Evaluation skipped: No reference material found for subject '{self.subject}'. Please ensure the Knowledge Base has relevant textbooks uploaded.",
                 "citation": None
             }
        
        prompt = f"""
        Role: You are a very strict SCERT Class {student_class} Maths teacher. 
        Grade using ONLY the textbook excerpts provided below. Ignore all external knowledge.
        
        Textbook Excerpts (Official Ground Truth):
        {context}
        
        Question: {question} (Max Marks: {max_marks})
        Student's Answer: {answer}
        
        Rules for Grading:
        - Award marks ONLY for exact matches to textbook concepts or logical steps stated in the excerpts.
        - Award partial marks only for correct textbook-aligned steps.
        - Score 0 if the topic is not mentioned in the provided excerpts.
        - Penalize heavily for missing key points or including information not in the textbook.
        - Be EXTREMELY STRICT. If the student answers logically but the textbook isn't found in context, awarding 0 is acceptable.

        Output JSON format strictly:
        {{
            "score": float, 
            "feedback": "Specific improvements, citing textbook concepts or missing steps.",
            "breakdown": {{
                "concepts": "X / Y",
                "steps": "X / Y",
                "completeness": "X / Y"
            }},
            "citation": "Chapter/Page quote or 'None'"
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
