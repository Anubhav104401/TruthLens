import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
from scipy.sparse import csr_matrix, hstack

from preprocessing.pipeline import FeatureEngineer, TextCleaner
import asyncio
from services.gemini_verification_service import GeminiVerificationService
from services.ensemble_scorer import blend, detect_conflict, rescale_rule_adjustment

class RuleEngine:
    SENSATIONAL_PHRASES = {
        "you won't believe": 5,
        "what happened next": 5,
        "shocking": 4,
        "unbelievable": 4,
        "miracle": 3,
        "exposed": 3,
        "secret": 3,
        "viral": 2,
        "urgent": 2,
        "breaking": 2,
        "leaked": 2,
    }

    CREDIBILITY_MARKERS = {
        "according to": 2,
        "officials said": 2,
        "police said": 2,
        "court documents": 2,
        "the report said": 2,
        "researchers said": 2,
        "data showed": 2,
    }

    def evaluate(self, text):
        adjustment = 0
        reasons = []
        highlight_terms = []
        words = re.findall(r"\b[\w'-]+\b", text)
        text_lower = text.lower()

        exclamation_count = text.count("!")
        if exclamation_count > 3:
            points = min(8, exclamation_count)
            adjustment += points
            reasons.append(f"Excessive exclamation marks increased fake-risk by {points}.")
            highlight_terms.append("!")

        if words:
            caps_words = sum(1 for word in words if len(word) > 2 and word.isupper())
            caps_ratio = caps_words / len(words)
            if caps_ratio > 0.25:
                adjustment += 6
                reasons.append("Many all-caps words look like sensational formatting.")

        matched_sensational = []
        for phrase, points in self.SENSATIONAL_PHRASES.items():
            if phrase in text_lower:
                adjustment += points
                matched_sensational.append(phrase)
                highlight_terms.append(phrase)

        if matched_sensational:
            reasons.append("Sensational/clickbait phrases were found: " + ", ".join(matched_sensational[:5]) + ".")

        url_count = len(re.findall(r"https?://\S+|www\.\S+", text))
        if url_count > 2:
            points = min(6, url_count * 2)
            adjustment += points
            reasons.append(f"Many URLs increased fake-risk by {points}.")

        if 0 < len(words) < 50:
            adjustment += 3
            reasons.append("The text is very short for a news article, so confidence is limited.")

        credibility_points = 0
        matched_credible = []
        for phrase, points in self.CREDIBILITY_MARKERS.items():
            if phrase in text_lower:
                credibility_points += points
                matched_credible.append(phrase)

        if credibility_points:
            reduction = min(8, credibility_points)
            adjustment -= reduction
            reasons.append("Reported-source language reduced fake-risk: " + ", ".join(matched_credible[:4]) + ".")

        adjustment = int(max(-15, min(20, adjustment)))
        return {
            "adjustment": adjustment,
            "reasons": reasons,
            "highlight_terms": sorted(set(highlight_terms)),
        }


class ArticleProfiler:
    NEWS_MARKERS = {
        "minister",
        "government",
        "police",
        "court",
        "election",
        "official",
        "officials",
        "company",
        "report",
        "research",
        "university",
        "agency",
        "department",
        "statement",
        "investigation",
        "parliament",
        "president",
        "prime minister",
        "chief minister",
        "school",
        "board",
        "exam",
        "students",
        "arrested",
    }

    NON_NEWS_MARKERS = {
        "once upon a time",
        "chapter",
        "i woke up",
        "my dream",
        "my friend",
        "dear diary",
    }

    ATTRIBUTION_PATTERNS = [
        r"\bsaid\b",
        r"\btold\b",
        r"\baccording to\b",
        r"\breported\b",
        r"\bannounced\b",
        r"\bconfirmed\b",
    ]

    def profile(self, text):
        words = re.findall(r"\b[\w'-]+\b", text)
        lower = text.lower()
        word_count = len(words)
        sentence_count = len([part for part in re.split(r"[.!?]+", text) if part.strip()])
        news_marker_count = sum(1 for marker in self.NEWS_MARKERS if marker in lower)
        non_news_marker_count = sum(1 for marker in self.NON_NEWS_MARKERS if marker in lower)
        attribution_count = sum(1 for pattern in self.ATTRIBUTION_PATTERNS if re.search(pattern, lower))
        first_person_count = sum(1 for word in words if word.lower() in {"i", "me", "my", "mine"})
        first_person_ratio = first_person_count / word_count if word_count else 0
        has_dateline = bool(re.search(r"^[A-Z][A-Za-z\s.'-]{2,40}\s[-:]", text.strip()))

        score = 0
        if word_count >= 120:
            score += 2
        elif word_count >= 60:
            score += 1
        if sentence_count >= 3:
            score += 1
        if news_marker_count:
            score += min(2, news_marker_count)
        if attribution_count:
            score += 1
        if has_dateline:
            score += 1
        if first_person_ratio > 0.06:
            score -= 1
        if non_news_marker_count:
            score -= 2

        reasons = []
        if word_count < 35:
            reasons.append("Text is too short for a dependable article-level judgement.")
        if score < 2:
            reasons.append("Text does not look enough like a reported news article.")
        if non_news_marker_count:
            reasons.append("Narrative/story-like markers were detected.")

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "news_marker_count": news_marker_count,
            "attribution_count": attribution_count,
            "article_likeness_score": score,
            "needs_review": word_count < 35 or score < 2,
            "reasons": reasons,
        }


class PredictionService:
    def __init__(self):
        model_dir = Path(__file__).resolve().parents[1] / "saved_models"
        self.model = joblib.load(model_dir / "model.pkl")
        self.vectorizer = joblib.load(model_dir / "vectorizer.pkl")
        self.scaler = joblib.load(model_dir / "scaler.pkl")
        self.metadata = self._load_metadata(model_dir / "metadata.json")

        expected_features = len(FeatureEngineer.FEATURE_COLUMNS)
        fitted_features = getattr(self.scaler, "n_features_in_", expected_features)
        if fitted_features != expected_features:
            raise RuntimeError(
                "Saved model artifacts are stale. Run: python backend/training/train_models.py"
            )

        self.cleaner = TextCleaner()
        self.feature_engineer = FeatureEngineer()
        self.rule_engine = RuleEngine()
        self.article_profiler = ArticleProfiler()
        self.explainer = LimeTextExplainer(class_names=["Real", "Fake"], random_state=42)
        self.gemini_service = GeminiVerificationService()

    def _load_metadata(self, path):
        if not path.exists():
            return {
                "model_name": "unknown",
                "warning": "Model metadata is missing. Retrain the model to generate metadata.json.",
            }
        return json.loads(path.read_text())

    def predict_pipeline_fn(self, texts):
        cleaned_texts = [self.cleaner.clean(text) for text in texts]
        features = [self.feature_engineer.extract_features(text) for text in texts]
        features_df = pd.DataFrame(features, columns=FeatureEngineer.FEATURE_COLUMNS).fillna(0)

        tfidf_matrix = self.vectorizer.transform(cleaned_texts)
        scaled_features = self.scaler.transform(features_df)
        x_matrix = hstack([tfidf_matrix, csr_matrix(scaled_features)], format="csr")

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x_matrix)

        scores = self.model.decision_function(x_matrix)
        fake_probs = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - fake_probs, fake_probs])

    def _risk_level(self, prediction, fake_score):
        if prediction == "Needs Review":
            return "Review"
        if fake_score >= 75:
            return "High"
        if fake_score >= 55:
            return "Medium"
        return "Low"

    def _label_from_score(self, fake_score, profile):
        if profile["needs_review"]:
            return "Needs Review"
        if fake_score >= 60:
            return "Likely Fake"
        if fake_score <= 40:
            return "Likely Real"
        return "Needs Review"

    def _confidence(self, prediction, fake_score):
        if prediction == "Likely Fake":
            return fake_score
        if prediction == "Likely Real":
            return 100 - fake_score
        return min(max(fake_score, 100 - fake_score), 55)

    def _explain(self, text):
        short_text = " ".join(text.split()[:500])
        if len(short_text.split()) < 5:
            return []

        try:
            exp = self.explainer.explain_instance(
                short_text,
                self.predict_pipeline_fn,
                labels=(1,),
                num_features=15,
                num_samples=500,
            )
            return [[term, float(weight)] for term, weight in exp.as_list(label=1)]
        except Exception:
            return []

    def _analyze_article_local(self, text):
        rule_result = self.rule_engine.evaluate(text)
        profile = self.article_profiler.profile(text)

        probabilities = self.predict_pipeline_fn([text])[0]
        ml_fake_probability = float(probabilities[1] * 100)

        reasons = []
        explanations = self._explain(text)
        highlight_terms = set(rule_result["highlight_terms"])
        for term, weight in explanations:
            if weight > 0:
                highlight_terms.add(term)

        return {
            "rule_result": rule_result,
            "profile": profile,
            "ml_fake_probability": ml_fake_probability,
            "reasons": reasons,
            "highlight_terms": highlight_terms,
            "explanations": explanations
        }

    async def analyze_article(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text is required for prediction.")

        local_task = asyncio.to_thread(self._analyze_article_local, text)
        gemini_task = self.gemini_service.verify_article(text)

        local_result, gemini_result = await asyncio.gather(local_task, gemini_task, return_exceptions=True)

        if isinstance(local_result, Exception):
            raise RuntimeError(f"Local ML pipeline failed: {local_result}")

        if isinstance(gemini_result, Exception):
            # Graceful degradation if gemini fails completely outside of its own try/catch
            gemini_result = self.gemini_service._create_fallback_response(str(gemini_result))

        rule_result = local_result["rule_result"]
        profile = local_result["profile"]
        ml_fake_probability = local_result["ml_fake_probability"]
        local_reasons = local_result["reasons"]
        highlight_terms = local_result["highlight_terms"]
        explanations = local_result["explanations"]

        gemini_enabled = str(os.getenv("ENABLE_GEMINI_VERIFICATION", "true")).lower() == "true"
        gemini_available = gemini_enabled and gemini_result.get("confidence", 0) > 0
        gemini_score = gemini_result.get("fake_likelihood", 50.0)
        gemini_confidence = gemini_result.get("confidence", 0.0)
        
        degraded_mode = gemini_enabled and not gemini_available
        degraded_reason = gemini_result.get("summary") if degraded_mode else None

        rule_score = rescale_rule_adjustment(rule_result["adjustment"])
        
        final_score, weights_used = blend(
            ml_score=ml_fake_probability,
            rule_score=rule_score,
            gemini_score=gemini_score,
            gemini_confidence=gemini_confidence,
            gemini_available=gemini_available
        )
        
        conflict = detect_conflict(ml_fake_probability, gemini_score, gemini_confidence)

        prediction = self._label_from_score(final_score, profile)
        confidence = self._confidence(prediction, final_score)

        if prediction == "Needs Review":
            local_reasons.extend(profile["reasons"])
            if not profile["reasons"]:
                local_reasons.append("The combined score is too close to the decision boundary.")
        elif prediction == "Likely Fake":
            local_reasons.append("The combined analysis indicates the text resembles fake news or contains false claims.")
        else:
            local_reasons.append("The combined analysis indicates the text resembles real news or contains verified claims.")
        local_reasons.extend(rule_result["reasons"])
        
        if conflict:
            local_reasons.append("Note: The local machine learning model and live web verification yielded conflicting results.")

        return {
            "ml_confidence": round(ml_fake_probability, 2),
            "fake_probability": round(final_score, 2),
            "real_probability": round(100 - final_score, 2),
            "rule_penalty": rule_result["adjustment"],
            "final_score": round(final_score, 2),
            "confidence": round(confidence, 2),
            "prediction": prediction,
            "risk_level": self._risk_level(prediction, final_score),
            "reasons": local_reasons,
            "highlight_terms": sorted(highlight_terms),
            "article_profile": profile,
            "explanation": explanations,
            "model_name": self.metadata.get("model_name", "unknown"),
            "model_warning": self.metadata.get(
                "warning",
                "This model estimates dataset similarity; it is not a fact-checker."
            ),
            "gemini_enabled": gemini_enabled,
            "gemini_verdict": gemini_result.get("verdict"),
            "gemini_fake_likelihood": gemini_result.get("fake_likelihood"),
            "gemini_confidence": gemini_result.get("confidence"),
            "gemini_claims_checked": gemini_result.get("claims_checked", []),
            "gemini_summary": gemini_result.get("summary"),
            "gemini_sources": gemini_result.get("sources", []),
            "gemini_search_queries": gemini_result.get("search_queries", []),
            "gemini_search_suggestions_html": gemini_result.get("search_suggestions_html"),
            "rule_score": round(rule_score, 2),
            "weights_used": weights_used,
            "degraded_mode": degraded_mode,
            "degraded_reason": degraded_reason,
            "signal_conflict": conflict
        }

    def model_info(self):
        return self.metadata
