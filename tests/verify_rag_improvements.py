
import unittest
from app.services.ingestion import IngestionService
import re

class TestRAGImprovements(unittest.TestCase):
    def test_semantic_chunking(self):
        service = IngestionService()
        text = "Paragraph 1.\n\nParagraph 2 is slightly longer and contains important details.\n\nParagraph 3."
        
        chunks = service.chunk_text(text, chunk_size=100, overlap=10)
        print(f"\nOriginal Text Length: {len(text)}")
        print(f"Chunks Created: {len(chunks)}")
        for i, c in enumerate(chunks):
            print(f"Chunk {i}: {repr(c)}")
            
        # Basic assertions
        self.assertTrue(len(chunks) > 0)
        # Check if paragraphs are preserved (not split in middle of words)
        self.assertIn("Paragraph 1.", chunks[0])

    def test_chunking_large_text(self):
        service = IngestionService()
        # Create a large text with many paragraphs
        para = "This is a test paragraph that has reasonable length. " * 5
        text = "\n\n".join([para] * 10) # 10 paragraphs
        
        chunks = service.chunk_text(text, chunk_size=200, overlap=50) # Small chunk size to force splits
        
        # The regex chunker will first split by \n\n, giving 10 paragraphs.
        # Each paragraph (len ~53) fits in 200 chunk size.
        # It will likely merge 3-4 paragraphs per chunk.
        
        print(f"\nLarge Text Chunks: {len(chunks)}")
        # Verify that we didn't just return 1 giant chunk
        self.assertTrue(len(chunks) > 1)
        # Verify that no chunk is ridiculously large (e.g., > 300)
        for c in chunks:
            self.assertTrue(len(c) <= 300)
            
    def test_keyword_reranking_logic_standalone(self):
        # Simulation of the logic added to agents.py
        docs = [
            {"content": "Standard Vector Result about Geometry"},
            {"content": "Irrelevant text about Algebra"},
            {"content": "Exact match for Euclid Parallel Postulate here"}
        ]
        query = "Euclid Parallel"
        
        query_terms = set(query.lower().split())
        
        def score_doc(doc):
            content = doc.get('content', '').lower()
            score = 0
            for term in query_terms:
                if term in content:
                    score += 1
            return score
            
        docs.sort(key=score_doc, reverse=True)
        
        print("\nRe-ranked Docs:")
        for d in docs:
            print(f"- {d['content']} (Score: {score_doc(d)})")
            
        self.assertEqual(docs[0]['content'], "Exact match for Euclid Parallel Postulate here")

if __name__ == '__main__':
    unittest.main()
