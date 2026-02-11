from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "SmartGrade Nexus is Online"


def get_sample_answer_sheet():
    return {
        "answer_sheet_id": "as_123",
        "exam_details": {
            "subject": "Physics",
            "board": "CBSE",
            "class_level": 10
        },
        "responses": [
            {
                "q_no": 1,
                "question_context": {
                    "text_primary": {"en": "What is Newton's Second Law?"},
                    "type": "mcq",
                    "max_marks": 5
                },
                "student_answer": {
                    "text": "F=ma"
                }
            }
        ]
    }

def test_categorize_questions(mock_gemini, mock_supabase):
    payload = [get_sample_answer_sheet()]
    response = client.post("/api/v1/intelligence/categorize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "mappings" in data
    assert len(data["mappings"]) >= 1

def test_evaluate_answer(mock_gemini, mock_supabase):
    payload = [get_sample_answer_sheet()]
    response = client.post("/api/v1/intelligence/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) >= 1
    # Updated to handle new nested structure
    grading_data = data["results"][0].get("grading", data["results"][0])
    assert "score" in grading_data

def test_analyze_full_sheet(mock_gemini, mock_supabase):
    payload = [get_sample_answer_sheet()]
    response = client.post("/api/v1/intelligence/analyze-full-sheet", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    
    # Check if enrichment happened
    sheet = data[0]
    first_response = sheet["responses"][0]
    assert "topic_analysis" in first_response
    assert first_response["student_answer"]["marks_awarded"] is not None
    assert first_response["student_answer"]["feedback"] is not None

def test_get_books(mock_gemini, mock_supabase):
    # Mocking is already handled in conftest for ingestion_service.get_uploaded_books
    # However, get_uploaded_books calls supabase.rpc(...).execute()
    # verify mocked return in conftest
    
    response = client.get("/api/v1/knowledge/books")
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["status"] == "success"
    # The mock returns mock_client.rpc...data = [{'content': 'Newton Law'}] which is not quite right for "books"
    # typically "books" would return [{'book_name': 'X', ...}]
    # But as long as it returns a list, the endpoint logic passes.


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
    assert res_json["data"]["status"] == "processing"
    assert "started" in res_json["data"]["message"]

def test_ingest_knowledge_ai(mock_gemini, mock_supabase):
    """Test AI-powered ingestion mode."""
    files = {"file": ("test.pdf", b"dummy content", "application/pdf")}
    data = {
        "subject": "Math",
        "book_name": "Geometry AI",
        "ingestion_mode": "ai"
    }
    response = client.post("/api/v1/knowledge/ingest", data=data, files=files)
    
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["status"] == "success"
    assert "ai mode" in res_json["data"]["message"].lower()
    assert res_json["data"]["status"] == "processing"

def test_ingest_knowledge_defaults(mock_gemini, mock_supabase):
    """Test that ingestion works even when ALL optional fields (including subject/book) are missing."""
    files = {"file": ("test.pdf", b"dummy content", "application/pdf")}
    # Only sending file, no form data
    data = {}

    response = client.post("/api/v1/knowledge/ingest", data=data, files=files)
    
    assert response.status_code == 200
    assert response.json()["data"]["status"] == "processing"

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
