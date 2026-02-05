import fitz  # PyMuPDF
import google.generativeai as genai
from supabase import create_client, Client
from app.core.config import settings
from typing import List, Dict, Any
import asyncio

# Initialize Clients
genai.configure(api_key=settings.GOOGLE_API_KEY)
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

class IngestionService:
    def __init__(self):
        self.embedding_model = "models/text-embedding-004"

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
        """Robust Recursive Chunker handling various separators."""
        if not text:
            return []
            
        separators = ["\n\n", "\n", ". ", " ", ""]
        final_chunks = []
        
        # Helper to merge small pieces into chunks
        def merge_splits(splits: List[str], separator: str) -> List[str]:
            docs = []
            current_doc = []
            total_len = 0
            separator_len = len(separator)

            for d in splits:
                l = len(d)
                if total_len + l + separator_len > chunk_size:
                    if total_len > 0:
                        docs.append(separator.join(current_doc))
                        # Keep trailing for overlap (simple approach)
                        while total_len > chunk_size - overlap and current_doc:
                            total_len -= (len(current_doc.pop(0)) + separator_len)
                        if not current_doc: # If popped everything
                             total_len = 0
                    
                    # If the single piece is still too big, it needs further splitting
                    if l > chunk_size and separator: 
                        # This piece needs strictly smaller splitting
                        pass # It will be handled by next recursion level or forced split
                        
                current_doc.append(d)
                total_len += (l + separator_len)
                
            if current_doc:
                docs.append(separator.join(current_doc))
            return docs

        # Recursive function
        def _recursive_split(text: str, separators: List[str]) -> List[str]:
            final_chunks = []
            separator = separators[-1]
            new_separators = []
            
            # Find the best separator that exists in text
            for i, sep in enumerate(separators):
                if sep == "":
                    separator = ""
                    break
                if sep in text:
                    separator = sep
                    new_separators = separators[i+1:]
                    break
            
            # Split
            if separator:
                splits = text.split(separator)
            else:
                splits = list(text) # Character split
                
            # Now merge
            docs = []
            current_doc = []
            total_len = 0
            sep_len = len(separator)
            
            for s in splits:
                if not s.strip(): continue
                
                s_len = len(s)
                if s_len < chunk_size:
                    if total_len + s_len + sep_len > chunk_size:
                        docs.append(separator.join(current_doc))
                        current_doc = [s]
                        total_len = s_len
                    else:
                        current_doc.append(s)
                        total_len += (s_len + sep_len)
                else:
                    # Current piece is huge
                    if current_doc:
                        docs.append(separator.join(current_doc))
                        current_doc = []
                        total_len = 0
                    
                    if new_separators:
                        docs.extend(_recursive_split(s, new_separators))
                    else:
                        # Force split by char matching chunk_size
                        for i in range(0, s_len, chunk_size - overlap):
                            docs.append(s[i : i + chunk_size])
            
            if current_doc:
                docs.append(separator.join(current_doc))
            
            return docs

        return _recursive_split(text, separators)

    async def get_embedding(self, text: str) -> List[float]:
        """Generates embedding using Gemini Free Tier (runs in threadpool)."""
        def _embed():
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
            
        return await asyncio.to_thread(_embed)

    async def process_document(self, file_content: bytes, metadata: Dict[str, Any]):
        """Main Orchestrator: Parse -> Chunk -> Embed -> Store."""
        # 1. Parse
        raw_text = await self.parse_pdf(file_content)
        
        if not raw_text.strip():
            raise ValueError("No text extracted. This PDF might be an image/scan (OCR not supported in Zero-Cost MVP).")
        
        # 2. Chunk
        chunks = self.chunk_text(raw_text)
        
        # 3. Embed & Prepare Records
        records = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            embedding = await self.get_embedding(chunk)
            
            record = {
                "content": chunk,
                "metadata": metadata,
                "embedding": embedding, # pgvector field
                "chunk_index": i
            }
            records.append(record)
            
        # 4. Store in Supabase
        # Assumes a table 'documents' exists with 'embedding' column
        response = supabase.table("documents").insert(records).execute()
        return {"status": "success", "chunks_processed": len(chunks)}

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
