from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.api import router
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from the project root .env file
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

app = FastAPI(
    title="Intelligent Fake News Detection System",
    description="API for classifying news articles and generating LIME explanations",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the Vercel frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Fake News Detection API is running"}
