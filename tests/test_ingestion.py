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
    
    # Mock parse_pdf since we don't have a real PDF file here
    service.parse_pdf = AsyncMock(return_value="Sample text content from PDF.")
    
    result = await service.process_document(
        b"dummy_pdf_bytes", 
        {"subject": "Math"}
    )
    
    assert result["status"] == "success"
    # Check if storage was called
    mock_supabase.table.return_value.insert.return_value.execute.assert_called()
