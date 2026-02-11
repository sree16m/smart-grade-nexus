import pytest
from app.services.ingestion import IngestionService
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_chunk_text():
    service = IngestionService()
    text = "Para 1.\n\nPara 2 is longer.\n\nPara 3."
    chunks = service.chunk_text(text, chunk_size=20)
    
    assert len(chunks) > 1
    assert "Para 1." in chunks[0]

@pytest.mark.asyncio
async def test_process_document(mock_gemini, mock_supabase):
    service = IngestionService()
    
    # Mock parse_pdf as an async generator
    async def mock_parse_pdf(*args, **kwargs):
        yield "Sample content page 1"
        yield "Sample content page 2"
        
    service.parse_pdf = mock_parse_pdf
    
    from fastapi import BackgroundTasks
    bg_tasks = BackgroundTasks()
    
    result = await service.process_document(
        b"dummy_pdf_bytes", 
        {"book_name": "TestBook", "subject": "Math", "ingestion_mode": "standard"},
        bg_tasks
    )
    
    # Since it runs in BackgroundTasks, we need to wait for the internal task if we were testing the side effect,
    # but here we just check if the orchestrator started correctly.
    assert result["status"] == "processing"
    assert "started" in result["message"]
