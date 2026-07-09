from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from models.schemas import FeedbackRequest, PredictionRequest, PredictionResponse, UserLogin, UserRegistration
from utils.database import feedback_collection, predictions_collection, users_collection
from utils.security import get_password_hash, verify_password, create_access_token, decode_access_token
from services.prediction_service import PredictionService
import datetime
from bson import ObjectId

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")

# Load ML service once
prediction_service = PredictionService()

async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    email = payload.get("sub")
    user = await users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user

@router.post("/register")
async def register(user: UserRegistration):
    existing_user = await users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    user_dict = {
        "username": user.username,
        "email": user.email,
        "password": hashed_password
    }
    await users_collection.insert_one(user_dict)
    return {"message": "User registered successfully"}

@router.post("/login")
async def login(user: UserLogin):
    db_user = await users_collection.find_one({"email": user.email})
    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, current_user: dict = Depends(get_current_user)):
    try:
        # Analyze article
        start_time = datetime.datetime.utcnow()
        result = await prediction_service.analyze_article(request.text)
        end_time = datetime.datetime.utcnow()
        
        # Log prediction
        prediction_log = {
            "user_id": current_user["email"],
            "text": request.text,
            "ml_confidence": result["ml_confidence"],
            "fake_probability": result["fake_probability"],
            "real_probability": result["real_probability"],
            "rule_penalty": result["rule_penalty"],
            "final_score": result["final_score"],
            "confidence": result["confidence"],
            "prediction": result["prediction"],
            "risk_level": result["risk_level"],
            "reasons": result["reasons"],
            "model_name": result["model_name"],
            "timestamp": start_time,
            "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
            "gemini_enabled": result.get("gemini_enabled"),
            "gemini_verdict": result.get("gemini_verdict"),
            "gemini_fake_likelihood": result.get("gemini_fake_likelihood"),
            "gemini_confidence": result.get("gemini_confidence"),
            "rule_score": result.get("rule_score"),
            "weights_used": result.get("weights_used"),
            "degraded_mode": result.get("degraded_mode"),
            "signal_conflict": result.get("signal_conflict")
        }
        await predictions_collection.insert_one(prediction_log)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_history(current_user: dict = Depends(get_current_user)):
    cursor = predictions_collection.find({"user_id": current_user["email"]}).sort("timestamp", -1).limit(20)
    history = await cursor.to_list(length=20)
    
    # Fix ObjectId serialization
    for item in history:
        item["_id"] = str(item["_id"])
        
    return {"history": history}

@router.get("/stats")
async def get_stats(current_user: dict = Depends(get_current_user)):
    cursor = predictions_collection.find({"user_id": current_user["email"]}).sort("timestamp", -1).limit(1000)
    records = await cursor.to_list(length=1000)

    total = len(records)
    by_prediction = {}
    for item in records:
        label = item.get("prediction", "Unknown")
        by_prediction[label] = by_prediction.get(label, 0) + 1

    avg_fake_risk = 0
    if total:
        avg_fake_risk = sum(item.get("final_score", 0) for item in records) / total

    return {
        "total_predictions": total,
        "by_prediction": by_prediction,
        "average_fake_risk": round(avg_fake_risk, 2),
    }

@router.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest, current_user: dict = Depends(get_current_user)):
    payload = feedback.dict()
    payload.update({
        "user_id": current_user["email"],
        "timestamp": datetime.datetime.utcnow(),
    })
    inserted = await feedback_collection.insert_one(payload)
    return {"message": "Feedback recorded", "feedback_id": str(inserted.inserted_id)}

@router.get("/model-info")
async def get_model_info():
    return prediction_service.model_info()
