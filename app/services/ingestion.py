import fitz  # PyMuPDF
import google.generativeai as genai
from supabase import create_client, Client
from app.core.config import settings
from typing import List, Dict, Any
import asyncio
import re

# Initialize Clients
genai.configure(api_key=settings.GOOGLE_API_KEY)
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

class IngestionService:
    def __init__(self):
        self.embedding_model = "models/gemini-embedding-001"

    async def parse_pdf(self, file_content: bytes) -> str:
        """Extracts text from a PDF file (runs in threadpool to avoid blocking)."""
        def _parse():
            doc = fitz.open(stream=file_content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text
            
        return await asyncio.to_thread(_parse)

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Regex-based Semantic Chunker.
        Splits by double newlines (paragraphs) and merges them to form chunks.
        """
        if not text:
            return []

        # 1. Split by Paragraphs (Double Newline or more)
        # Identify possible paragraph breaks
        splits = re.split(r'\n\s*\n', text)
        
        final_chunks = []
        current_chunk = []
        current_len = 0
        
        for split in splits:
            split = split.strip()
            if not split:
                continue
                
            split_len = len(split)
            
            # If a single paragraph is HUGE (bigger than chunk_size), we might need to sub-split it later.
            # For now, let's treat it as a block.
            
            # Check if adding this paragraph exceeds chunk size
            if current_len + split_len + 2 > chunk_size:
                # Flush current chunk
                if current_chunk:
                    text_block = "\n\n".join(current_chunk)
                    final_chunks.append(text_block)
                    
                    # Handle Overlap: Keep last few paragraphs that fit in overlap size
                    # Simplified overlap: just keep the last paragraph if it's small
                    overlap_buffer = []
                    overlap_len = 0
                    for p in reversed(current_chunk):
                        if overlap_len + len(p) < overlap:
                            overlap_buffer.insert(0, p)
                            overlap_len += len(p)
                        else:
                            break
                    current_chunk = overlap_buffer
                    current_len = overlap_len

                # If the new split itself is huge, force split it? 
                # Or just accept it as one big chunk (LLMs handle 2-3k tokens easily now)
                # Let's add it effectively.
                current_chunk.append(split)
                current_len += split_len
            else:
                # Add to current
                current_chunk.append(split)
                current_len += split_len + 2 # +2 for newline
        
        # Flush remainder
        if current_chunk:
            final_chunks.append("\n\n".join(current_chunk))
            
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
            raise ValueError("No text extracted. This PDF might be an image/scan (OCR not supported in Zero-Cost MVP).")
        
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
