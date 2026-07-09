import sys
import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# Mock the collections before importing api
from utils.database import users_collection, predictions_collection

# Set env before importing app
os.environ["ENABLE_GEMINI_VERIFICATION"] = "false"
os.environ["JWT_SECRET"] = "test"

with patch("services.prediction_service.PredictionService"):
    # We will mock the prediction service entirely
    pass

from routes.api import router, prediction_service
from fastapi import FastAPI
app = FastAPI()
app.include_router(router, prefix="/api")

client = TestClient(app)

@pytest.fixture
def override_auth():
    # Mock get_current_user
    pass

@pytest.mark.asyncio
async def test_predict_endpoint(mocker):
    # Mock current user dependency
    app.dependency_overrides = {}
    
    # Mock predict service
    mock_result = {
        "ml_confidence": 80.0,
        "fake_probability": 85.0,
        "real_probability": 15.0,
        "rule_penalty": 5,
        "final_score": 85.0,
        "confidence": 85.0,
        "prediction": "Likely Fake",
        "risk_level": "High",
        "reasons": [],
        "highlight_terms": [],
        "article_profile": {},
        "explanation": [],
        "model_name": "test_model",
        "model_warning": "",
        "gemini_enabled": False,
        "gemini_verdict": None,
        "gemini_fake_likelihood": None,
        "gemini_confidence": None,
        "gemini_claims_checked": [],
        "gemini_summary": None,
        "gemini_sources": [],
        "gemini_search_queries": [],
        "gemini_search_suggestions_html": None,
        "rule_score": 75.0,
        "weights_used": {"ml": 0.7, "rule": 0.3, "gemini": 0.0},
        "degraded_mode": False,
        "degraded_reason": None,
        "signal_conflict": False
    }
    
    mocker.patch.object(prediction_service, 'analyze_article', new_callable=AsyncMock, return_value=mock_result)
    
    # Mock token validation
    mocker.patch('routes.api.decode_access_token', return_value={"sub": "test@test.com"})
    mocker.patch('routes.api.users_collection.find_one', new_callable=AsyncMock, return_value={"email": "test@test.com"})
    mocker.patch('routes.api.predictions_collection.insert_one', new_callable=AsyncMock)
    
    response = client.post(
        "/api/predict",
        json={"text": "This is a test article."},
        headers={"Authorization": "Bearer test_token"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Likely Fake"
    assert data["gemini_enabled"] == False
    assert "weights_used" in data
