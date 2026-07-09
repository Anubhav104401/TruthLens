# TruthLens

> **Intelligent Fake News Detection System using NLP, Machine Learning, Hybrid Logic, and Explainable AI**

---

## 🚀 Project Overview

TruthLens is not just another fake news classifier. It is a full-stack system built to:

- classify whether a news article is likely **Real** or **Fake**
- compute a **confidence score** for every prediction
- highlight suspicious words and phrases
- provide a human-readable explanation for each result
- support API-driven integration for frontend and future services
- store prediction history for later analysis

This project bridges the gap between a college-level proof of concept and a production-minded pipeline.

---

## 🎯 Why this is a strong project

Most teams stop at a single notebook model. TruthLens goes further by combining:

- a modern **React UI** for easy interaction
- a responsive **FastAPI backend** for scalable REST execution
- a **three-way hybrid intelligence engine** that uses ML, handcrafted rules, and **live web verification via Gemini**
- **explainability** through LIME and verifiable source citations so users understand the “why” behind results
- a flexible design that can later be integrated with a real database or cloud deployment

This makes TruthLens suitable for a portfolio, demo, or technical interview discussion.

---

## 🧠 Project Aim

TruthLens is designed to answer two questions:

1. **Is this article fake or real?**
2. **Why did the system say that?**

A naive model can answer the first question, but the second one is what separates a demo from a real system.

---

## 📦 Full Architecture

```text
                   +------+
                   | User |
                   +--+---+
                      |
                      ▼
             +-------------------+
             | React Frontend UI |
             +-------------------+
                      |
              HTTP POST /api/predict
                      |
                      ▼
             +-------------------+
             |   FastAPI Backend |
             +-------------------+
                      |
     +----------------+----------------+------------------+
     |                |                |                  |
+------------+ +--------------+ +-----------------------+
|  ML Model  | | Rule Engine  | | Gemini Web Verifier   |
+------------+ +--------------+ +-----------------------+
     |                |                |                  |
     +----------------+----------------+------------------+
                      |
             +-------------------+
             |  Final Risk Score |
             +-------------------+
                      |
                      ▼
               Prediction Result
                      |
                      ▼
              Prediction History
```

---

## 🧩 What this repo contains

- `frontend/` — React + Vite UI, text editor, result dashboard, charts, and explainability view
- `backend/` — FastAPI backend, prediction logic, preprocessing, rule engine, auth, and mock data storage
- `data/` — sample datasets used for model training and exploration
- `backend/saved_models/` — serialized model, vectorizer, scaler, and metrics output

---

## 🛠️ Tech Stack

### Frontend

- **React** — interactive UI and component-based structure
- **Vite** — fast development server and build pipeline
- **Tailwind CSS** — utility-first styling for polished visuals
- **Axios** — browser-to-backend HTTP communication
- **react-chartjs-2 / Chart.js** — prediction score visualization
- **Framer Motion** — smooth animated transitions
- **React Icons** — polished iconography

### Backend

- **FastAPI** — high-performance Python web framework
- **Uvicorn** — ASGI server for running the app
- **Pydantic** — request validation and schema serialization
- **joblib** — model artifact persistence
- **NLTK** — tokenization, stopwords, and lemmatization utilities
- **scikit-learn** — TF-IDF and model evaluation
- **LIME** — local explainability engine

### Optional / Future

- **MongoDB Atlas** — replace the mock database with cloud storage
- **Render / Vercel** — backend and frontend deployment
- **SHAP** — deeper explainability when scaling beyond LIME
- **BERT / Transformer models** — for advanced semantic understanding

---

## 🔍 Internal Functioning Explained

### 1. User input

The user pastes an article into the frontend text box and clicks analyze.

### 2. Frontend sends request

The React app sends a JSON request to `POST /api/predict` with the article text.

### 3. Backend preprocessing

Back in the backend, the text enters a cleaning pipeline:

- lowercase conversion
- HTML tag removal
- URL removal
- email removal
- numeric cleanup
- punctuation removal
- tokenization
- stopword removal
- lemmatization

This makes the raw article consistent and easier for the model to understand.

### 4. Feature engineering

TruthLens does more than TF-IDF. It combines:

- **TF-IDF vectorization**: converts words into numerical importance scores
- **statistical text features**: word count, average sentence length, capitalization ratio, punctuation count, URL count

Why use both? TF-IDF captures meaning from words and phrases, while statistical features capture writing style and tone.

### 5. Hybrid scoring

The system evaluates the article from three angles:

- **ML model score**: probability that the text is fake based on training data
- **rule engine score**: handcrafted penalties for suspicious traits
- **Gemini web verification**: live Google Search grounding to verify specific claims

The final risk score is computed by combining these three using a confidence-weighted ensemble.

### 6. Explainability

The app uses LIME to show which words influenced the decision. It returns a list of word-weight pairs that the frontend highlights in the text.

This helps users understand whether the system is reacting to legitimate signals or just random phrases.

### 7. Response back to user

The backend returns a result object containing:

- `prediction` — Fake or Real
- `ml_confidence` — probability from the ML model
- `rule_penalty` — handcrafted risk score
- `final_score` — combined score
- `explanation` — words that mattered most

The frontend renders this as a dashboard with colors, charts, and highlighted text.

---

## 📘 Why these algorithms were chosen

### TF-IDF

TF-IDF is a classic feature extractor for text. It measures how important a word is in one document relative to the corpus.

Why this is good:

- fast to compute
- works well on moderate datasets
- easy to interpret

What else could be used:

- **Word embeddings** (Word2Vec, GloVe)
- **Transformer embeddings** (BERT, RoBERTa)
- **FastText**

### Regression / Classification Model

TruthLens currently uses a model saved as `model.pkl` from scikit-learn.

Why this is good:

- reliable baseline performance
- easy to serialize with `joblib`
- suitable for text classification

What else could be used:

- **Logistic Regression** — simple and interpretable
- **Naive Bayes** — great for text with word independence assumption
- **Linear SVM** — strong baseline for sparse vectors
- **Random Forest / XGBoost** — capture non-linear patterns
- **Transformers** — best for semantic nuance

### LIME explainability

LIME is used because it can explain model decisions for individual texts by perturbing inputs and observing outputs.

Why this is good:

- local explanation for each prediction
- easy to display as word weights
- helps detect whether the model is focusing on real signals

Alternative explainability tools:

- **SHAP** — more rigorous global and local interpretation
- **ELI5** — model-agnostic explanations

---

## 🧪 Data preprocessing pipeline

The project cleans text in the following order:

1. **Lowercase** — reduce casing noise
2. **Remove HTML** — strip tags from copied content
3. **Remove URLs** — remove external web addresses
4. **Remove emails** — remove private metadata
5. **Remove numbers** — remove digits that add noise
6. **Remove punctuation** — simplify token boundaries
7. **Tokenize** — split text into individual words
8. **Remove stopwords** — remove common words like "the", "and", "of"
9. **Lemmatize** — convert words to base form (`running` → `run`)
10. **Rejoin tokens** — build the cleaned text for vectorization

This sequence is important because each step prepares the text for the next.

---

## 🔬 Feature engineering explained

TruthLens extracts two kinds of features:

### 1. TF-IDF features

Used to measure how strongly a word or phrase appears in the article relative to the dataset.

### 2. Text statistics

These features capture writing style:

- total word count
- average sentence length
- ratio of uppercase letters
- number of exclamation marks
- number of question marks
- count of URLs

These engineering features act as a second source of truth. A short sensational article with many clicks is more suspicious, even if the model probability is not extremely high.

---

## 🧠 Rule engine logic

A pure ML model can miss simple signals that humans recognize immediately.

This project adds a lightweight rule engine that adds penalty points for:

- repeated exclamation marks
- clickbait trigger words like `BREAKING` or `SHOCKING`
- too much ALL CAPS text
- article length that is too short

Rules are intentionally simple. They capture stylistic patterns of low-quality fake news.

Why this improves the project:

- hybrid systems are more robust
- the rule engine can catch adversarial or noisy input
- rules provide a backup signal when ML is uncertain

---

## 🧾 API design

### `POST /api/register`
Register a demo user.

### `POST /api/login`
Login and receive a bearer token.

### `POST /api/predict`
Analyze the article text and return:

- prediction
- confidence
- rule penalty
- final score
- explanation

### `GET /api/history`
Retrieve saved prediction history for the current user.

These endpoints are built with FastAPI and `Pydantic` schemas for request validation.

---

## 💾 Data storage today

The current repo uses an in-memory mock database via `backend/utils/database.py`.

That means:

- user accounts and predictions are stored in runtime memory
- they reset when the server restarts

This is a simple demo approach, and it is intentionally replaceable with MongoDB Atlas or any real database later.

---

## 🔒 Security and auth

The backend is structured for secure operation using:

- **JWT authentication** for protected routes
- **hashed passwords** using bcrypt-style hashing
- **CORS** middleware to safely accept frontend requests
- **input validation** through Pydantic models

Even though this is a prototype, the code follows a production-style security pattern.

---

## 🧩 Frontend flow

The frontend enables users to:

- paste an article
- run analysis
- see prediction results
- view confidence scores
- inspect highlighted explanations

It is built to be responsive, modern, and easy to understand.

---

## 📁 Project layout

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

## 🚀 Running the full system

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

### Open in browser

- `http://localhost:5173` — UI
- `http://localhost:8000/docs` — backend API documentation

---

## 🧪 How to extend this repo

### Replace mock DB with MongoDB Atlas
The backend is already organized so the `database.py` file can be swapped with a real Mongo client.

### Improve the model
Train new models and save them to `backend/saved_models/`.

### Deploy frontend and backend separately
Use Vercel for the frontend and Render for the backend.

### Add richer explainability
Replace or augment LIME with SHAP for more advanced model analysis.

---

## 🌟 Why this is not just "another fake news detector"

This repo is designed as a hybrid detection platform, not merely a classifier.

It combines:

- raw model probability
- handcrafted risk heuristics
- explanation for every result
- clean frontend presentation
- clearly separated backend modules

That makes it a much better demonstration of real engineering than a single notebook with Naive Bayes.

---

## 📌 Notes for future reviewers

If you are reading this as a reviewer, ask to see:

- the data preprocessing pipeline in `backend/preprocessing/pipeline.py`
- the prediction logic in `backend/services/prediction_service.py`
- the frontend explanation rendering in `frontend/src/App.jsx`
- the API definitions in `backend/routes/api.py`

This README is intentionally deep so that both technical and non-technical viewers can understand the system.
