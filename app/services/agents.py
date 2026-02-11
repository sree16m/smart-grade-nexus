import google.generativeai as genai
from supabase import create_client, Client
from app.core.config import settings
from typing import List, Dict, Any
import json

# Initialize Clients
genai.configure(api_key=settings.GOOGLE_API_KEY)
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
# Initialize RAG Model
embedding_model = settings.EMBEDDING_MODEL
generative_model = genai.GenerativeModel(settings.GENERATIVE_MODEL)

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

async def search_knowledge_base(query: str, subject: str, limit: int = None, filter: Dict[str, Any] = None) -> str:
    """Retrieves relevant chunks from Vector DB (Non-blocking) with Metadata Filtering."""
    if limit is None:
        limit = settings.DEFAULT_RAG_LIMIT
    
    # 1. Generate Query Embedding (Threadpool)
    def _embed():
         return genai.embed_content(
            model=embedding_model,
            content=query,
            task_type="retrieval_query",
            output_dimensionality=settings.EMBEDDING_DIM
        )['embedding']
    
    query_emb = await asyncio.to_thread(_embed)
    
    # 2. Search Supabase (Threadpool)
    def _search(search_filter):
        return supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_emb,
                "match_threshold": settings.MATCH_THRESHOLD,
                "match_count": settings.MATCH_COUNT,
                "filter": search_filter
            }
        ).execute()

    # Normalize/Map common subject variations
    subject_map = settings.SUBJECT_MAP
    
    search_subjects = [subject]
    # Add variations if they exist in the map (case-insensitive)
    normalized_key = subject.strip().lower()
    if normalized_key in subject_map:
        for variation in subject_map[normalized_key]:
            if variation not in search_subjects:
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
            context = await search_knowledge_base(q_text, self.subject, limit=settings.CATEGORIZATION_RAG_LIMIT)
            
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

    async def evaluate(self, question: str, answer: str, max_marks: int, student_class: str = None, chapter: str = None, board: str = None) -> Dict:
        """Grades answer using Ground Truth from KB."""
        if student_class is None: student_class = settings.DEFAULT_CLASS
        if board is None: board = settings.DEFAULT_BOARD
        # 1. Prepare Filter
        search_filter = {}
        if chapter and chapter != "Unknown":
            search_filter["chapter"] = chapter
            print(f"RAG Filter: Hard-filtering search scope for Chapter {chapter}")

        # RAG: Get the official explanation
        context = await search_knowledge_base(question, self.subject, limit=settings.GRADING_RAG_LIMIT, filter=search_filter)
        
        # 1. Fail fast if no context is found
        if not context.strip():
             return {
                 "score": 0,
                 "feedback": f"Evaluation skipped: No reference material found for subject '{self.subject}'. Please ensure the Knowledge Base has relevant textbooks uploaded.",
                 "citation": None
             }
        
        # Subject-specific strategy
        subject_rule = settings.SUBJECT_RULES.get(self.subject, settings.SUBJECT_RULES.get(self.subject.capitalize(), "Be strict and follow the textbook excerpts exactly."))

        prompt = f"""
        Role: You are a very strict {board} Class {student_class} {self.subject} teacher. 
        Grade using ONLY the textbook excerpts provided below. Ignore all external knowledge.
        
        {self.subject} Grading Strategy: {subject_rule}

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
