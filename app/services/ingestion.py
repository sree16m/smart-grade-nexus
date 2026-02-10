import fitz  # PyMuPDF
import google.generativeai as genai
from supabase import create_client, Client
from app.core.config import settings
from typing import List, Dict, Any
import asyncio
import re
import pytesseract
from pdf2image import convert_from_bytes
import io
from PIL import Image

# Initialize Clients
genai.configure(api_key=settings.GOOGLE_API_KEY)
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

class IngestionService:
    def __init__(self):
        self.embedding_model = "models/gemini-embedding-001"

    async def parse_pdf(self, file_content: bytes) -> str:
        """Extracts text from a PDF file with OCR fallback."""
        def _parse():
            doc = fitz.open(stream=file_content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            
            # Fallback to OCR if extracted text is suspiciously short (e.g., scanned PDF)
            if len(text.strip()) < 50: # Threshold for "not enough text"
                print("Minimal text extracted via PyMuPDF. Falling back to Tesseract OCR...")
                images = convert_from_bytes(file_content)
                ocr_text = ""
                for img in images:
                    ocr_text += pytesseract.image_to_string(img)
                return ocr_text
                
            return text
            
        return await asyncio.to_thread(_parse)

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
        """Main Orchestrator: Parse -> Chunk -> Embed -> Store."""
        # 1. Parse
        raw_text = await self.parse_pdf(file_content)
        
        if not raw_text.strip():
            raise ValueError("No text extracted. PDF might be empty or corrupted even after OCR attempt.")
        
        # 2. Chunk (Run in threadpool to avoid blocking event loop)
        chunks = await asyncio.to_thread(self.chunk_text, raw_text)
        
        # 3. Embed (Parallelized with Semaphore)
        sem = asyncio.Semaphore(10) # Limit concurrent requests to avoid rate limits
        
        async def _process_chunk(i: int, chunk: str):
            if not chunk.strip():
                return None
            
            async with sem:
                embedding = await self.get_embedding(chunk)
                
            return {
                "content": chunk,
                "metadata": metadata,
                "embedding": embedding,
                "chunk_index": i
            }

        tasks = [_process_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        
        # Filter out None results (empty chunks)
        records = [r for r in results if r is not None]
            
        # 4. Store in Supabase
        if records:
            # Assumes a table 'documents' exists with 'embedding' column
            response = supabase.table("documents").insert(records).execute()
            return {"status": "success", "chunks_processed": len(records)}
        else:
             return {"status": "success", "chunks_processed": 0}

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
