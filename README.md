# TruthLens

> **Hybrid explainable fake-news detection** with modern frontend, Python AI backend, and interactive explanation.

---

## 🚀 What is TruthLens?

TruthLens is a futuristic investigative pipeline that blends machine learning, NLP heuristics, and explainable AI to assess whether a news article is likely to be fake or real.

This repo contains a full-stack proof-of-concept:

- **React + Vite** frontend for intuitive article analysis
- **FastAPI** backend serving prediction APIs
- **TF-IDF + custom NLP features** for document representation
- **Rule-based risk scoring** layered on top of ML confidence
- **LIME-powered explanation** for transparent decision-making
- **Model training, storage, and analytics** baked into the backend

---

## ✨ Why this project stands out

- **Hybrid intelligence**: combines statistical features with classic NLP and ML outputs
- **Explainability first**: highlights influential words using LIME
- **Interactive history**: tracks predictions and allows insight into past analysis
- **Production-ready structure**: clean separation of frontend and backend services

---

## 🌐 Architecture

```text
[ User Browser ]
      │
      │ HTTP
      ▼
[ Frontend / Vite ]
      │
      │ POST /api/predict
      ▼
[ FastAPI Backend ]
      │
      ├─ preprocess text
      ├─ TF-IDF + feature engineering
      ├─ ML probability scoring
      ├─ rule engine penalty
      ├─ LIME explanation generation
      └─ response payload
```

---

## 🎯 Features

- `POST /api/predict` → returns prediction + score + explanation
- `POST /api/register`, `POST /api/login` → demo auth flow
- `GET /api/history` → saved prediction history
- `GET /api/model-info` → model metadata and metrics
- `GET /api/stats` → aggregated analytics
- Clean, modern dark UI with polished data presentation

---

## 🧠 Core backend components

- `backend/main.py` — FastAPI app entrypoint
- `backend/routes/api.py` — REST endpoints
- `backend/services/prediction_service.py` — ML + rule engine + LIME
- `backend/preprocessing/pipeline.py` — text cleaning + feature extraction
- `backend/utils/database.py` — in-memory repo for demo storage
- `backend/saved_models/` — trained model artifacts

---

## 💻 Quick start

### Backend

```powershell
cd "c:\Users\anubh\Projects\Placement project\fake-news-detector\backend"
.\venv\Scripts\Activate.ps1
python -m uvicorn main:app --reload --port 8000
```

### Frontend

```powershell
cd "c:\Users\anubh\Projects\Placement project\fake-news-detector\frontend"
npm install
npm run dev
```

Then open:

- `http://localhost:5173` for the UI
- `http://localhost:8000/docs` for backend API docs

---

## ⚡ Pro tips

- Use the frontend text editor to paste articles and watch realtime analysis
- The backend includes an explainability layer so you can inspect which words matter most
- This project is ideal for demoing how AI + rule systems can work together in a transparency-aware pipeline

---

## 📁 Project structure

```text
fake-news-detector/
  backend/
    main.py
    routes/api.py
    services/prediction_service.py
    preprocessing/pipeline.py
    saved_models/
    utils/database.py
  frontend/
    src/App.jsx
    src/main.jsx
    index.html
  data/
    fake_or_real_news.csv
```

---

## 🔧 Notes

- This is a detection assist tool, not a proof of truth
- The ML model estimates text behavior, not factual correctness
- For production, swap the demo DB and credential flow with persistent storage and secure auth

---

## 📬 Want to contribute?

- Add new explainability visualizations
- Improve the NLP preprocessing pipeline
- Add more model evaluation metrics
- Turn the demo into a fully deployed SaaS app
