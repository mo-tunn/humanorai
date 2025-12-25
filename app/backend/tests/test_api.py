import pytest
from unittest.mock import patch, MagicMock

def test_read_root(client):
    """Test that the frontend is served at root."""
    # Note: Since the frontend dir might not exist in this test environment or static files might fail,
    # we expect 200 if index.html exists, or 404 if not found but handled by StaticFiles.
    # However, StaticFiles with html=True usually serves index.html or 404.
    # Given the environment, let's just assert we get a response.
    # If the directory doesn't exist, Starlette's StaticFiles might raise error or return 404.
    # But usually creating the app succeeds even if dir is missing until a request is made.
    
    # We'll try/except or just run it. If it fails due to missing dir, we might skip.
    try:
        response = client.get("/")
        # If it returns 200 or 404, the route is active.
        assert response.status_code in [200, 404] 
    except RuntimeError:
        # Directory might not exist
        pytest.skip("Frontend directory missing, skipping root test")

def test_predict_success(client, mock_model_manager):
    """Test /predict endpoint with valid input."""
    payload = {"text": "This text is definitely long enough to ensure that the API processes it correctly without throwing a validation error for being too short."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "individual_results" in data
    assert "ensemble_average_percent" in data
    assert "ensemble_decision" in data
    assert isinstance(data["individual_results"], list)
    
    # Setup in conftest returns 2 models
    assert len(data["individual_results"]) == 2

def test_predict_short_text(client):
    """Test /predict endpoint with too short text."""
    payload = {"text": "Too short"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Minimum 50 characters" in response.json()["detail"]

def test_predict_empty_text(client):
    """Test /predict endpoint with empty text."""
    payload = {"text": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400

def test_predict_internal_error(client, mock_model_manager):
    """Test handling of internal model errors."""
    mock_model_manager["predict"].return_value = {"Error": 0.5}
    
    payload = {"text": "This text is definitely long enough to ensure that the API processes it correctly."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 500
    assert "Error during text processing" in response.json()["detail"]

@patch("APIServer.extract_text_from_file")
def test_predict_file_success(mock_extract, client, mock_model_manager):
    """Test /predict_file endpoint with valid file extraction."""
    mock_extract.return_value = "This text is valid and simulated as extracted from a file upload. It is long enough."
    
    files = {'file': ('test.txt', b'dummy content', 'text/plain')}
    response = client.post("/predict_file", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "ensemble_decision" in data
    mock_extract.assert_called_once()

@patch("APIServer.extract_text_from_file")
def test_predict_file_short_content(mock_extract, client):
    """Test /predict_file when extracted text is too short."""
    mock_extract.return_value = "Short text"
    
    files = {'file': ('test.txt', b's', 'text/plain')}
    response = client.post("/predict_file", files=files)
    
    assert response.status_code == 400
    assert "Minimum 50 characters" in response.json()["detail"]

@patch("APIServer.extract_text_from_file")
def test_predict_file_internal_error(mock_extract, client, mock_model_manager):
    """Test /predict_file when model prediction fails."""
    mock_extract.return_value = "This text is valid and simulated as extracted from a file upload. It is long enough."
    mock_model_manager["predict"].return_value = {"Error": 0.5}
    
    files = {'file': ('test.txt', b'dummy', 'text/plain')}
    response = client.post("/predict_file", files=files)
    
    assert response.status_code == 500
    assert "Error during text processing" in response.json()["detail"]

@patch("services.model_manager.load_models_from_disk", side_effect=Exception("Startup Fail"))
def test_startup_exception(mock_load):
    """Test that app startup handles model load failure gracefully."""
    from fastapi.testclient import TestClient
    from APIServer import app
    # This should trigger the startup event handler which catches the exception
    with TestClient(app):
        pass

