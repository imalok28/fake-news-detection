import os
import sys
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import app, model_info

client = TestClient(app)

def test_health_endpoint_not_loaded():
    """Tests the health check when no model is loaded."""
    # Temporarily set model status to not loaded
    with patch("src.api.model_info", {
        "model_loaded": False,
        "model_type": None,
        "detail": "No model loaded yet."
    }):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False

def test_health_endpoint_loaded():
    """Tests the health check when a model is loaded."""
    # Temporarily set model status to loaded
    with patch("src.api.model_info", {
        "model_loaded": True,
        "model_type": "Baseline TF-IDF",
        "detail": "Baseline machine learning model loaded successfully."
    }):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_type"] == "Baseline TF-IDF"

def test_predict_endpoint_no_model():
    """Tests that predicting without a model returns 503 Service Unavailable."""
    with patch("src.api.model_info", {
        "model_loaded": False,
        "model_type": None,
        "detail": "No model loaded yet."
    }), patch("src.api.load_inference_model") as mock_load:
        response = client.post("/predict", json={"text": "This is a news article that should fail since no model is loaded."})
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]
        mock_load.assert_called_once()

def test_predict_endpoint_validation():
    """Tests API input validation for short or invalid text."""
    # Text too short
    response = client.post("/predict", json={"text": "Short"})
    assert response.status_code == 422  # Unprocessable Entity
    
    # Missing text field
    response = client.post("/predict", json={})
    assert response.status_code == 422

def test_predict_endpoint_success():
    """Tests a successful classification and explainability request by mocking the explainer."""
    mock_explainer = MagicMock()
    mock_explainer.explain.return_value = {
        "label": "REAL",
        "authenticity_score": 0.95,
        "top_influential_words": [
            {"word": "senate", "weight": 0.08, "influence": "REAL"},
            {"word": "unverified", "weight": -0.05, "influence": "FAKE"}
        ]
    }
    
    # Patch the global explainer and model_info
    with patch("src.api.explainer", mock_explainer), \
         patch("src.api.model_info", {
             "model_loaded": True,
             "model_type": "Baseline TF-IDF",
             "detail": "Baseline model loaded."
         }):
        
        test_text = "This is a real news article talking about the senate passing a bill."
        response = client.post("/predict", json={"text": test_text})
        
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "REAL"
        assert data["authenticity_score"] == 0.95
        assert data["model_used"] == "Baseline TF-IDF"
        
        words = data["top_influential_words"]
        assert len(words) == 2
        assert words[0]["word"] == "senate"
        assert words[0]["weight"] == 0.08
        assert words[0]["influence"] == "REAL"
        
        mock_explainer.explain.assert_called_once_with(test_text, num_features=10)
