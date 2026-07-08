# TruthLens - Fake News Detection

TruthLens is a demo-ready fake-news risk analysis app with:

- React frontend
- FastAPI backend
- TF-IDF plus statistical NLP features
- Multiple model training and comparison
- Rule-based risk adjustment
- LIME explanation heatmap
- Prediction history, stats, and feedback endpoints

The app estimates whether text resembles labelled real/fake news. It does not prove factual accuracy.

## Project Layout

```text
fake-news-detector/
  data/
  backend/
    main.py
    preprocessing/
    routes/
    saved_models/
    services/
    training/
  frontend/
```

## Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Download Dataset

```bash
python training/download_dataset.py
```

The downloader creates `data/fake_or_real_news.csv` and `data/dataset_source.json`.

You can also place the larger ISOT-style files in `data/`:

```text
data/Fake.csv
data/True.csv
```

The trainer automatically prefers `Fake.csv` and `True.csv` when both are present.

## Train Models

```bash
python training/train_models.py
```

Saved outputs:

```text
backend/saved_models/model.pkl
backend/saved_models/vectorizer.pkl
backend/saved_models/scaler.pkl
backend/saved_models/metrics.json
backend/saved_models/metadata.json
backend/saved_models/classification_report.txt
```

## Run Backend

```bash
cd backend
venv\Scripts\activate
uvicorn main:app --reload
```

Backend URL:

```text
http://localhost:8000
```

## Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

## Main API Endpoints

```text
POST /api/register
POST /api/login
POST /api/predict
GET  /api/history
GET  /api/stats
POST /api/feedback
GET  /api/model-info
```
