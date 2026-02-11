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
import json
import time

# Initialize Clients
genai.configure(api_key=settings.GOOGLE_API_KEY)
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

from app.services.job_registry import job_registry

class IngestionService:
    def __init__(self):
        self.embedding_model = settings.EMBEDDING_MODEL

    async def parse_pdf(self, file_content: bytes, book_name: str = None) -> AsyncGenerator[str, None]:
        """Extracts text from a PDF file as a stream of pages with OCR fallback."""
        def _get_doc():
            return fitz.open(stream=file_content, filetype="pdf")
            
        doc = await asyncio.to_thread(_get_doc)
        total_pages = len(doc)
        
        # Heuristic to decide if we need OCR: Sample start, middle, end
        sample_indices = set([0, total_pages//2, total_pages-1]) if total_pages > 0 else set()
        sample_text = ""
        for i in sorted(list(sample_indices)):
             if i < total_pages:
                sample_text += doc[i].get_text()
        
        text_stripped = sample_text.strip()
        
        # 1. Check if too short
        is_too_short = len(text_stripped) < settings.OCR_MIN_TEXT_LENGTH
        
        # 2. Check for "Garbage" or Non-English density
        english_chars = len(re.findall(r'[a-zA-Z]', text_stripped))
        garbage_chars = text_stripped.count('\ufffd') # Replacement character 
        
        density = english_chars / len(text_stripped) if len(text_stripped) > 0 else 0
        garbage_ratio = garbage_chars / len(text_stripped) if len(text_stripped) > 0 else 0
        
        is_low_density = len(text_stripped) > 0 and density < settings.OCR_MIN_ENGLISH_DENSITY
        is_high_garbage = garbage_ratio > settings.OCR_MAX_GARBAGE_RATIO # More than 5% garbage characters
        
        use_ocr = is_too_short or is_low_density or is_high_garbage
        
        if use_ocr:
            reason = "too short" if is_too_short else (f"low English density ({density:.1%})" if is_low_density else f"high garbage ratio ({garbage_ratio:.1%})")
            print(f"Extraction quality is low ({reason}). Starting page-by-page OCR for {total_pages} pages...")
            for i in range(total_pages):
                # Check for cancellation
                if book_name and job_registry.is_cancelled(book_name):
                    print(f"Ingestion CANCELLED for {book_name}")
                    break

                print(f"OCR: Processing page {i+1}/{total_pages}...")
                def _ocr_page(page_num):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img)
                    return text
                
                page_text = await asyncio.to_thread(_ocr_page, i)
                if book_name:
                    job_registry.update_progress(book_name, i + 1)
                yield page_text
        else:
            print(f"Extraction quality is Good. Streaming {total_pages} pages...")
            for i, page in enumerate(doc):
                # Check for cancellation
                if book_name and job_registry.is_cancelled(book_name):
                    print(f"Ingestion CANCELLED for {book_name}")
                    break

                if (i+1) % 20 == 0:
                    print(f"Streaming: page {i+1}/{total_pages}...")
                
                if book_name:
                    job_registry.update_progress(book_name, i + 1)
                yield page.get_text()
        
        doc.close()

    async def parse_pdf_ai(self, file_content: bytes, book_name: str = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Extracts text and enriched metadata from a PDF file using Gemini Vision."""
        def _get_doc():
            return fitz.open(stream=file_content, filetype="pdf")
            
        doc = await asyncio.to_thread(_get_doc)
        total_pages = len(doc)
        print(f"AI Ingestion: Starting Gemini Vision transcription for {total_pages} pages...")

        model = genai.GenerativeModel(settings.GENERATIVE_MODEL)
        
        for i in range(total_pages):
            # Check for cancellation
            if book_name and job_registry.is_cancelled(book_name):
                print(f"Ingestion CANCELLED for {book_name}")
                break

            print(f"AI OCR: Processing page {i+1}/{total_pages}...")
            
            def _prep_page():
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                return buf.getvalue()

            img_bytes = await asyncio.to_thread(_prep_page)
            
            prompt = """
            Transcribe this textbook page into high-quality Markdown. 
            - IMPORTANT: If the page is bilingual (e.g., Telugu and English), transcribe BOTH languages accurately.
            - PRIORITIZE ENGLISH: Ensure technical terms and definitions are clearly transcribed in English.
            - Use LaTeX for all mathematical formulas (e.g., $E=mc^2$).
            - Describe any diagrams or complex visuals in brackets [Diagram: ...].
            - Preserve table structures.
            - Do not include page numbers or headers/footers.
            
            Identify the structural context:
            - is_instructional_content: Set to false if the page is "Front Matter" (Foreword, Committee lists, National Anthem, Copyright, Translators, Preamble, Acknowledgements, etc.). Set to true if it contains actual educational lessons, theory, or exercises.
            - chapter_number: e.g., "5"
            - chapter_title: e.g., "Introduction to Euclid's Geometry"
            - subtopic: The specific section heading on this page if any.
            - lang: "en" or "te" (dominant language for content).

            Also, provide a brief summary and a list of key concepts found on this page.
            Return the result in JSON format:
            {
              "content": "the transcribed markdown text",
              "page_summary": "1-2 sentence overview",
              "key_concepts": ["concept1", "concept2"],
              "is_instructional_content": boolean,
              "structural_metadata": {
                "chapter_number": "string",
                "chapter_title": "string",
                "subtopic": "string",
                "lang": "string"
              }
            }
            """

            try:
                def _gen():
                    return model.generate_content(
                        [prompt, {"mime_type": "image/jpeg", "data": img_bytes}],
                        generation_config={"response_mime_type": "application/json"}
                    )
                
                response = await asyncio.to_thread(_gen)
                res_data = json.loads(response.text)
                if book_name:
                    job_registry.update_progress(book_name, i + 1)
                yield res_data
            except Exception as e:
                print(f"AI OCR failed for page {i+1}: {e}")
                # Fallback to empty if AI fails for one page to keep going
                yield {"content": f"[AI Error on Page {i+1}]", "page_summary": "", "key_concepts": []}

        doc.close()

    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Robust Recursive Chunker.
        Splits text by decreasing structural hierarchy: \n\n -> \n -> sentence -> char.
        """
        if chunk_size is None: chunk_size = settings.CHUNK_SIZE
        if overlap is None: overlap = settings.CHUNK_OVERLAP
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
                output_dimensionality=settings.EMBEDDING_DIM
            )
            return result['embedding']
            
        return await asyncio.to_thread(_embed)

    async def process_document(self, file_content: bytes, metadata: Dict[str, Any], background_tasks: Any):
        """Main Orchestrator (Background): Parse -> Chunk -> Embed -> Store Incrementally."""
        book_name = metadata.get("book_name") or f"unnamed_{int(time.time())}"
        ingestion_mode = metadata.get("ingestion_mode", "ai")

        async def _run_ingestion():
            try:
                # Initialize Registry
                # We need to know total pages beforehand for progress tracking
                def _get_page_count():
                    with fitz.open(stream=file_content, filetype="pdf") as doc:
                        return len(doc)
                total_pages = await asyncio.to_thread(_get_page_count)
                job_registry.start_job(book_name, total_pages)

                # 1. Idempotency: Clear existing chunks for this book
                if book_name:
                    print(f"Clearing existing entries for book: {book_name}")
                    await self.delete_book(book_name)

                batch_text = ""
                batch_page_count = 0
                total_chunks = 0
                
                if ingestion_mode == "ai":
                    # AI Mode: Enriched per-page processing
                    page_idx = 0
                    async for page_data in self.parse_pdf_ai(file_content, book_name):
                        if job_registry.is_cancelled(book_name): break
                        page_idx += 1

                        # Skip if it's Front Matter (Committee lists, forewords, etc.)
                        if not page_data.get("is_instructional_content", True):
                            print(f"Ingestion: Skipping non-instructional page {page_idx} (Front Matter)")
                            continue

                        print(f"AI Ingestion: Processing enriched page {total_chunks+1} for {book_name}...")
                        # Merge page-level metadata into chunk metadata
                        page_metadata = metadata.copy()
                        struct_meta = page_data.get("structural_metadata", {})
                        
                        page_metadata.update({
                            "page_summary": page_data.get("page_summary"),
                            "key_concepts": page_data.get("key_concepts"),
                            "chapter": struct_meta.get("chapter_number"),
                            "chapter_title": struct_meta.get("chapter_title"),
                            "subtopic": struct_meta.get("subtopic"),
                            "lang": struct_meta.get("lang"),
                            "is_content": True
                        })
                        chunks_added = await self._process_batch(page_data["content"], page_metadata)
                        total_chunks += chunks_added
                else:
                    # Standard Mode (Tesseract / Extraction)
                    async for page_text in self.parse_pdf(file_content, book_name):
                        if job_registry.is_cancelled(book_name): break

                        batch_text += page_text + "\n"
                        batch_page_count += 1
                        
                        if batch_page_count >= settings.INGESTION_BATCH_SIZE: # Process every N pages
                            print(f"Processing batch of {batch_page_count} pages for {book_name}...")
                            chunks_added = await self._process_batch(batch_text, metadata)
                            total_chunks += chunks_added
                            print(f"Successfully added {chunks_added} chunks. Total so far: {total_chunks}")
                            batch_text = ""
                            batch_page_count = 0
                    
                    # Process remaining pages
                    if batch_text and not job_registry.is_cancelled(book_name):
                        print(f"Processing final batch for {book_name}...")
                        chunks_added = await self._process_batch(batch_text, metadata)
                        total_chunks += chunks_added
                        print(f"Successfully added {chunks_added} chunks.")
                
                if job_registry.is_cancelled(book_name):
                    print(f"INGESTION STOPPED (Cancelled) for '{book_name}'.")
                else:
                    job_registry.complete_job(book_name)
                    print(f"INGESTION COMPLETE for '{book_name}'. Total stored: {total_chunks} chunks.")
            except Exception as e:
                import traceback
                traceback.print_exc()
                job_registry.fail_job(book_name, str(e))
                print(f"CRITICAL ERROR during ingestion for {book_name}: {str(e)}")

        # Start background task using FastAPI's BackgroundTasks
        background_tasks.add_task(_run_ingestion)
        
        return {
            "status": "processing", 
            "message": f"Ingestion ({ingestion_mode} mode) started in background for '{book_name}'."
        }

    async def _process_batch(self, text: str, metadata: Dict[str, Any], retries: int = 3) -> int:
        """Process a text batch: Chunk -> Embed (with retry) -> Store (with retry)."""
        if not text.strip():
            return 0
            
        chunks = self.chunk_text(text)
        if not chunks:
            return 0

        # Limited concurrency for embeddings
        sem = asyncio.Semaphore(settings.MAX_EMBEDDING_CONCURRENCY)
        
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
