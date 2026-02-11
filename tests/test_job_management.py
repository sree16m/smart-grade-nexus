from fastapi.testclient import TestClient
from app.main import app
import time

client = TestClient(app)

def test_job_lifecycle_standard(mock_gemini, mock_supabase):
    """Test that standard ingestion correctly creates and completes a job in the registry."""
    files = {"file": ("test.pdf", b"dummy content", "application/pdf")}
    data = {"book_name": "LifecycleTest", "ingestion_mode": "standard"}
    
    # 1. Start Ingestion
    response = client.post("/api/v1/knowledge/ingest", data=data, files=files)
    assert response.status_code == 200
    
    # 2. Check Status Immediately
    # Note: In tests, BackgroundTasks might run synchronously or very quickly
    response = client.get("/api/v1/knowledge/ingest/status/LifecycleTest")
    assert response.status_code == 200
    res_data = response.json()["data"]
    assert res_data["status"] in ["processing", "completed"]
    assert res_data["total_pages"] == 1

def test_job_cancellation(mock_gemini, mock_supabase, monkeypatch):
    """Test that a job can be cancelled via the API."""
    files = {"file": ("test.pdf", b"dummy content", "application/pdf")}
    data = {"book_name": "CancelTest"}
    
    # Start ingestion
    client.post("/api/v1/knowledge/ingest", data=data, files=files)
    
    # Cancel ingestion
    response = client.post("/api/v1/knowledge/ingest/cancel/CancelTest")
    assert response.status_code == 200
    assert "cancellation requested" in response.json()["message"]
    
    # Verify status in registry
    response = client.get("/api/v1/knowledge/ingest/status/CancelTest")
    assert response.json()["data"]["cancelled"] is True

def test_job_not_found():
    """Test 404 for non-existent jobs."""
    response = client.get("/api/v1/knowledge/ingest/status/NonExistent")
    assert response.status_code == 404
    
    response = client.post("/api/v1/knowledge/ingest/cancel/NonExistent")
    assert response.status_code == 404
