from pydantic import BaseModel, EmailStr
from typing import Any, Dict, List, Optional

class UserRegistration(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    ml_confidence: float
    fake_probability: float
    real_probability: float
    rule_penalty: int
    final_score: float
    confidence: float
    prediction: str
    risk_level: str
    reasons: List[str]
    highlight_terms: List[str]
    article_profile: Dict[str, Any]
    explanation: List[List] # [[word, weight], ...]
    model_name: str
    model_warning: str

class FeedbackRequest(BaseModel):
    prediction_id: Optional[str] = None
    text: Optional[str] = None
    expected_label: str
    comments: Optional[str] = None
