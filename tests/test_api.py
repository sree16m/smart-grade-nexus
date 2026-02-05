from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "SmartGrade Nexus is Online"

def test_categorize_questions(mock_gemini, mock_supabase):
    payload = {
        "subject": "Physics",
        "questions": [
            {"id": "q1", "text": "What is force?"}
        ]
    }
    response = client.post("/api/v1/intelligence/categorize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "mappings" in data
    assert len(data["mappings"]) == 1

def test_evaluate_answer(mock_gemini, mock_supabase):
    payload = {
        "subject": "Physics",
        "question_text": "Define Force",
        "student_answer_text": "Push or pull",
        "max_marks": 2
    }
    response = client.post("/api/v1/intelligence/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "score" in data

def test_ingest_knowledge(mock_gemini, mock_supabase):
    # Mock file upload
    files = {"file": ("test.pdf", b"dummy content", "application/pdf")}
    data = {
        "subject": "Math",
        "book_name": "Geometry V1"
    }
    # Test with optional parameters
    optional_data = {
        "board": "CBSE",
        "school": "DPS",
        "class": "10",
        "semister": "2"  # Note: API expects 'semister' based on alias
    }
    full_data = {**data, **optional_data}

    response = client.post("/api/v1/knowledge/ingest", data=full_data, files=files)
    
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["status"] == "success"
    
    # Verify mock was called with correct metadata
    # The IngestionService processes this, so we check if process_document was called effectively
    # by checking if Supabase insert was triggered.
    # In a real unit test for the service, we'd check the exact metadata dict.
    mock_supabase.table.return_value.insert.return_value.execute.assert_called()

def test_ingest_knowledge_defaults(mock_gemini, mock_supabase):
    """Test that ingestion works even when ALL optional fields (including subject/book) are missing."""
    files = {"file": ("test.pdf", b"dummy content", "application/pdf")}
    # Only sending file, no form data
    data = {}

    response = client.post("/api/v1/knowledge/ingest", data=data, files=files)
    
    assert response.status_code == 200

def test_delete_endpoints(mock_gemini, mock_supabase):
    # Test Clear
    response = client.delete("/api/v1/knowledge/clear")
    assert response.status_code == 200
    assert "cleared" in response.json()["message"]
    
    # Test Delete Book by Name
    response = client.delete("/api/v1/knowledge/books/Geometry")
    assert response.status_code == 200
    assert "deleted" in response.json()["message"]
    res_json = response.json()
    assert res_json["status"] == "success"
    # The metadata in the backend should have None for the missing fields, 
    # but the request should succeed.
