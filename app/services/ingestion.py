import fitz  # PyMuPDF
import google.generativeai as genai
from supabase import create_client, Client
from app.core.config import settings
from typing import List, Dict, Any, AsyncGenerator
import asyncio
import re
import pytesseract
import io
from PIL import Image

# Initialize Clients
genai.configure(api_key=settings.GOOGLE_API_KEY)
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

class IngestionService:
    def __init__(self):
        self.embedding_model = "models/gemini-embedding-001"

    async def parse_pdf(self, file_content: bytes) -> AsyncGenerator[str, None]:
        """Extracts text from a PDF file as a stream of pages with OCR fallback."""
        def _get_doc():
            return fitz.open(stream=file_content, filetype="pdf")
            
        doc = await asyncio.to_thread(_get_doc)
        
        # Heuristic to decide if we need OCR: Sample start, middle, end
        sample_indices = set([0, len(doc)//2, len(doc)-1]) if len(doc) > 0 else set()
        sample_text = ""
        for i in sorted(list(sample_indices)):
             if i < len(doc):
                sample_text += doc[i].get_text()
        
        text_stripped = sample_text.strip()
        
        # 1. Check if too short
        is_too_short = len(text_stripped) < 50
        
        # 2. Check for "Garbage" or Non-English density
        english_chars = len(re.findall(r'[a-zA-Z]', text_stripped))
        garbage_chars = text_stripped.count('\ufffd') # Replacement character 
        
        density = english_chars / len(text_stripped) if len(text_stripped) > 0 else 0
        garbage_ratio = garbage_chars / len(text_stripped) if len(text_stripped) > 0 else 0
        
        is_low_density = len(text_stripped) > 0 and density < 0.3
        is_high_garbage = garbage_ratio > 0.05 # More than 5% garbage characters
        
        use_ocr = is_too_short or is_low_density or is_high_garbage
        
        if use_ocr:
            reason = "too short" if is_too_short else (f"low English density ({density:.1%})" if is_low_density else f"high garbage ratio ({garbage_ratio:.1%})")
            print(f"Extraction quality is low ({reason}). Using page-by-page OCR...")
            for i in range(len(doc)):
                def _ocr_page(page_num):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img)
                    return text
                
                page_text = await asyncio.to_thread(_ocr_page, i)
                yield page_text
        else:
            for page in doc:
                yield page.get_text()
        
        doc.close()

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Robust Recursive Chunker.
        Splits text by decreasing structural hierarchy: \n\n -> \n -> sentence -> char.
        """
        if not text:
            return []

        def _recursive_split(current_text: str, separators: List[str]) -> List[str]:
            if len(current_text) <= chunk_size:
                return [current_text]
            
            if not separators:
                # Base case: character split
                return [current_text[i : i + chunk_size] for i in range(0, len(current_text), chunk_size - overlap)]

            sep = separators[0]
            # Try splitting by current separator
            if sep == ". ":
                # Special regex for sentence split to keep delimiter
                parts = re.split(r'(?<=\. )', current_text)
            else:
                parts = current_text.split(sep)
            
            all_parts = []
            for part in parts:
                if len(part) <= chunk_size:
                    all_parts.append(part)
                else:
                    all_parts.extend(_recursive_split(part, separators[1:]))
            return all_parts

        # 1. Split hierarchically
        raw_parts = _recursive_split(text, ["\n\n", "\n", ". "])
        
        # 2. Merge parts into chunks of target size
        final_chunks = []
        current_chunk = ""
        
        for part in raw_parts:
            part = part.strip()
            if not part:
                continue
                
            if len(current_chunk) + len(part) + 2 <= chunk_size:
                current_chunk = (current_chunk + "\n\n" + part) if current_chunk else part
            else:
                if current_chunk:
                    final_chunks.append(current_chunk)
                
                # Handle Overlap: keep some of the previous chunk
                if overlap > 0 and current_chunk:
                    # Take last 'overlap' chars from previous chunk
                    current_chunk = current_chunk[-overlap:] + "\n\n" + part
                else:
                    current_chunk = part
        
        if current_chunk:
            final_chunks.append(current_chunk)
            
        return final_chunks

    async def get_embedding(self, text: str) -> List[float]:
        """Generates embedding using Gemini Free Tier (runs in threadpool)."""
        def _embed():
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document",
                output_dimensionality=768
            )
            return result['embedding']
            
        return await asyncio.to_thread(_embed)

    async def process_document(self, file_content: bytes, metadata: Dict[str, Any]):
        """Main Orchestrator (Background): Parse -> Chunk -> Embed -> Store Incrementally."""
        book_name = metadata.get("book_name")

        async def _run_ingestion():
            try:
                # 1. Idempotency: Clear existing chunks for this book
                if book_name:
                    print(f"Clearing existing entries for book: {book_name}")
                    await self.delete_book(book_name)

                batch_text = ""
                batch_page_count = 0
                total_chunks = 0
                
                async for page_text in self.parse_pdf(file_content):
                    batch_text += page_text + "\n"
                    batch_page_count += 1
                    
                    if batch_page_count >= 5: # Process every 5 pages
                        print(f"Processing batch of {batch_page_count} pages for {book_name}...")
                        chunks_added = await self._process_batch(batch_text, metadata)
                        total_chunks += chunks_added
                        batch_text = ""
                        batch_page_count = 0
                
                # Process remaining pages
                if batch_text:
                    print(f"Processing final batch for {book_name}...")
                    total_chunks += await self._process_batch(batch_text, metadata)
                
                print(f"Ingestion complete for '{book_name}'. Total chunks stored: {total_chunks}")
            except Exception as e:
                print(f"CRITICAL ERROR during ingestion for {book_name}: {str(e)}")

        # Start background task
        asyncio.create_task(_run_ingestion())
        
        return {
            "status": "processing", 
            "message": f"Ingestion started in background for '{book_name}'. Progress will appear in Supabase shortly."
        }

    async def _process_batch(self, text: str, metadata: Dict[str, Any], retries: int = 3) -> int:
        """Process a text batch: Chunk -> Embed (with retry) -> Store (with retry)."""
        if not text.strip():
            return 0
            
        chunks = self.chunk_text(text)
        if not chunks:
            return 0

        # Limited concurrency for embeddings
        sem = asyncio.Semaphore(10)
        
        async def _proc_chunk(i: int, chunk: str):
            if not chunk.strip(): return None
            
            for attempt in range(retries):
                try:
                    async with sem:
                        embedding = await self.get_embedding(chunk)
                        return {
                            "content": chunk,
                            "metadata": metadata,
                            "embedding": embedding,
                            "chunk_index": i
                        }
                except Exception as e:
                    if attempt == retries - 1:
                        print(f"Embedding failed for chunk {i} after {retries} attempts: {e}")
                        return None
                    await asyncio.sleep(2 ** attempt)

        tasks = [_proc_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        records = [r for r in results if r]

        if not records:
            return 0

        # Insert batch into Supabase with retry
        for attempt in range(retries):
            try:
                supabase.table("documents").insert(records).execute()
                return len(records)
            except Exception as e:
                if attempt == retries - 1:
                    print(f"Supabase insert failed for batch after {retries} attempts: {e}")
                    return 0
                await asyncio.sleep(2 ** attempt)
        
        return 0

    async def get_uploaded_books(self) -> List[Dict]:
        """Fetches unique books metadata via RPC."""
        # Directly call RPC so errors (like missing function) propagate to API
        response = supabase.rpc("get_uploaded_books").execute()
        return response.data

    async def delete_all_documents(self):
        """Deletes ALL documents from the Knowledge Base."""
        # delete().neq("id", 0) is a hack to delete all rows in Supabase/Postgrest
        return supabase.table("documents").delete().neq("id", -1).execute()

    async def delete_book(self, book_name: str):
        """Deletes a specific book by name."""
        # Filter where metadata->>'book_name' equals the provided name
        return supabase.table("documents").delete().eq("metadata->>book_name", book_name).execute()
