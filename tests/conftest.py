import pytest
from unittest.mock import MagicMock, AsyncMock
import google.generativeai as genai
from app.services.ingestion import IngestionService
from app.services.agents import TopicAgent, GradingAgent
from app.main import app

@pytest.fixture
def mock_gemini(monkeypatch):
    """Mocks Google Gemini API calls."""
    mock_embed = MagicMock()
    mock_embed.return_value = {'embedding': [0.1, 0.2, 0.3]}
    monkeypatch.setattr(genai, "embed_content", mock_embed)

    mock_generate = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"topic_path": "Physics > Mechanics", "confidence": 0.99, "score": 5.0, "feedback": "Good job", "citation": "Page 10"}'
    mock_generate.return_value = mock_response
    
    # Mock the GenerativeModel class
    mock_model_cls = MagicMock()
    mock_model_instance = MagicMock()
    mock_model_instance.generate_content.return_value = mock_response
    mock_model_cls.return_value = mock_model_instance
    
    monkeypatch.setattr(genai, "GenerativeModel", mock_model_cls)
    return mock_embed, mock_model_instance

@pytest.fixture
def mock_supabase(monkeypatch):
    """Mocks Supabase Client."""
    mock_client = MagicMock()
    
    # Mock table().insert().execute()
    mock_insert = MagicMock()
    mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[])
    
    # Mock rpc().execute()
    mock_rpc = MagicMock()
    mock_client.rpc.return_value.execute.return_value = MagicMock(data=[{'content': 'Newton Law'}])
    
    monkeypatch.setattr("app.services.ingestion.supabase", mock_client)
    monkeypatch.setattr("app.services.agents.supabase", mock_client)
    return mock_client

@pytest.fixture(autouse=True)
def mock_fitz(monkeypatch):
    """Mocks PyMuPDF (fitz) to avoid needing real PDF files."""
    mock_doc = MagicMock()
    mock_page = MagicMock()
    # Return a long string to avoid triggering OCR fallback (> 50 chars)
    mock_page.get_text.return_value = "This is a long mock page text content to ensure that standard tests do not trigger the OCR fallback mechanism unnecessarily."
    mock_doc.__iter__.return_value = [mock_page]
    
    # Mock context manager
    mock_open = MagicMock()
    mock_open.return_value = mock_doc
    
    import app.services.ingestion
    monkeypatch.setattr(app.services.ingestion, "fitz", MagicMock(open=mock_open))

@pytest.fixture(autouse=True)
def mock_ocr(monkeypatch):
    """Mocks Tesseract and pdf2image for all tests."""
    mock_pytesseract = MagicMock()
    mock_pytesseract.image_to_string.return_value = "OCR extracted text"
    
    import app.services.ingestion
    monkeypatch.setattr(app.services.ingestion, "pytesseract", mock_pytesseract)
    monkeypatch.setattr(app.services.ingestion, "convert_from_bytes", MagicMock(return_value=[]))

