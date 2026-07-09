import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocessing.pipeline import FeatureEngineer, TextCleaner


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "saved_models"
RANDOM_STATE = 42


def _combine_title_and_text(df):
    title = df["title"].fillna("") if "title" in df.columns else ""
    text = df["text"].fillna("") if "text" in df.columns else df.iloc[:, 0].fillna("")
    return (title.astype(str).str.strip() + ". " + text.astype(str).str.strip()).str.strip()


def _load_isot_pair(data_dir):
    fake_path = data_dir / "Fake.csv"
    true_path = data_dir / "True.csv"

    if not fake_path.exists() or not true_path.exists():
        return None

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 1
    true_df["label"] = 0
    fake_df["article_text"] = _combine_title_and_text(fake_df)
    true_df["article_text"] = _combine_title_and_text(true_df)

    df = pd.concat([fake_df[["article_text", "label"]], true_df[["article_text", "label"]]], ignore_index=True)
    return df, "ISOT (Fake.csv + True.csv)"


def _normalise_label(value):
    text = str(value).strip().lower()

    fake_values = {"fake", "false", "f", "unreliable", "misleading", "1"}
    real_values = {"real", "true", "t", "reliable", "0"}

    if text in fake_values:
        return 1
    if text in real_values:
        return 0

    return np.nan


def _load_single_labelled_csv(data_dir):
    preferred = [
        data_dir / "fake_or_real_news.csv",
        data_dir / "news.csv",
        data_dir / "dataset.csv",
    ]
    candidates = [path for path in preferred if path.exists()]
    
    # We just want one typical single CSV if multiple exist (excluding ISOT which is handled separately)
    # The user asked to combine *multiple sources*, not just one single CSV, so we will return the first valid one we find, 
    # but rename the function slightly so it returns a list if we wanted to support multiple. We'll just stick to one for now to avoid duplicates.
    for path in candidates:
        df = pd.read_csv(path)
        lower_columns = {column.lower(): column for column in df.columns}

        label_column = lower_columns.get("label")
        text_column = lower_columns.get("text")
        title_column = lower_columns.get("title")

        if not label_column or not text_column:
            continue

        working = pd.DataFrame()
        working["label"] = df[label_column].apply(_normalise_label)
        if title_column:
            working["article_text"] = (
                df[title_column].fillna("").astype(str).str.strip()
                + ". "
                + df[text_column].fillna("").astype(str).str.strip()
            ).str.strip()
        else:
            working["article_text"] = df[text_column].fillna("").astype(str).str.strip()

        working = working.dropna(subset=["label"])
        working["label"] = working["label"].astype(int)
        return working[["article_text", "label"]], path.name

    return None

def _load_liar(data_dir):
    liar_dir = data_dir / "liar"
    if not liar_dir.exists():
        return None
        
    dfs = []
    for split in ["train.tsv", "valid.tsv", "test.tsv"]:
        path = liar_dir / split
        if path.exists():
            df = pd.read_csv(path, sep='\t', header=None, on_bad_lines='skip')
            # Columns: 0: id, 1: label, 2: statement
            if len(df.columns) >= 3:
                # Map 6-way to binary
                # fake: pants-fire, false, barely-true
                # real: half-true, mostly-true, true (treating half-true as a judgement call, mapping to real per request)
                label_map = {
                    "pants-fire": 1, "false": 1, "barely-true": 1,
                    "half-true": 0, "mostly-true": 0, "true": 0
                }
                
                mapped_df = pd.DataFrame()
                mapped_df["label"] = df[1].map(label_map)
                mapped_df["article_text"] = df[2].astype(str)
                mapped_df = mapped_df.dropna(subset=["label"])
                mapped_df["label"] = mapped_df["label"].astype(int)
                dfs.append(mapped_df)
                
    if dfs:
        return pd.concat(dfs, ignore_index=True), "LIAR Dataset"
    return None

def load_dataset(data_dir=DATA_DIR, sample_size=None):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    sources_loaded = []
    source_names = []
    
    isot_data = _load_isot_pair(data_dir)
    if isot_data:
        sources_loaded.append(isot_data[0])
        source_names.append(isot_data[1])
        print(f"Loaded {len(isot_data[0]):,} from {isot_data[1]}")
        
    single_csv_data = _load_single_labelled_csv(data_dir)
    if single_csv_data:
        sources_loaded.append(single_csv_data[0])
        source_names.append(single_csv_data[1])
        print(f"Loaded {len(single_csv_data[0]):,} from {single_csv_data[1]}")
        
    liar_data = _load_liar(data_dir)
    if liar_data:
        sources_loaded.append(liar_data[0])
        source_names.append(liar_data[1])
        print(f"Loaded {len(liar_data[0]):,} from {liar_data[1]}")

    if not sources_loaded:
        raise FileNotFoundError(
            "No supported datasets found. Download them first."
        )

    df = pd.concat(sources_loaded, ignore_index=True)
    combined_source_name = " + ".join(source_names)
    
    df = df.dropna(subset=["article_text", "label"]).copy()
    df["article_text"] = df["article_text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["article_text"].str.len() > 80]
    df = df.drop_duplicates(subset=["article_text"]).reset_index(drop=True)

    if sample_size and sample_size < len(df):
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda group: group.sample(min(len(group), sample_size // 2), random_state=RANDOM_STATE))
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )

    counts = df["label"].value_counts().to_dict()
    print(f"Total Combined Dataset: {len(df):,} articles")
    print(f"Class balance: real={counts.get(0, 0):,}, fake={counts.get(1, 0):,}")

    return df, source_names


def build_features(train_texts, test_texts, train_cleaned, test_cleaned, feature_engineer):
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    scaler = MaxAbsScaler()

    print("Vectorizing text...")
    train_tfidf = vectorizer.fit_transform(train_cleaned)
    test_tfidf = vectorizer.transform(test_cleaned)

    print("Extracting statistical features...")
    train_stats = feature_engineer.extract_features_df(train_texts)
    test_stats = feature_engineer.extract_features_df(test_texts)

    train_stats_scaled = scaler.fit_transform(train_stats)
    test_stats_scaled = scaler.transform(test_stats)

    x_train = hstack([train_tfidf, csr_matrix(train_stats_scaled)], format="csr")
    x_test = hstack([test_tfidf, csr_matrix(test_stats_scaled)], format="csr")

    return x_train, x_test, vectorizer, scaler


def candidate_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "linear_svm": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", dual=False, max_iter=5000, random_state=RANDOM_STATE),
            cv=3,
        ),
        "naive_bayes": MultinomialNB(alpha=0.2),
        "passive_aggressive": CalibratedClassifierCV(
            PassiveAggressiveClassifier(
                max_iter=1000,
                tol=1e-3,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            cv=3,
        ),
    }


def probability_for_fake(model, x_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]

    scores = model.decision_function(x_test)
    return 1 / (1 + np.exp(-scores))


def evaluate_model(name, model, x_train, y_train, x_test, y_test):
    print(f"Training {name}...")
    started = perf_counter()
    model.fit(x_train, y_train)
    train_seconds = perf_counter() - started

    predictions = model.predict(x_test)
    fake_probabilities = probability_for_fake(model, x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, fake_probabilities),
        "train_seconds": train_seconds,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    print(
        f"{name}: accuracy={metrics['accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}, auc={metrics['roc_auc']:.4f}"
    )

    return metrics, predictions


def _json_safe(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_artifacts(model, vectorizer, scaler, metrics, metadata, y_test, y_pred):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / "model.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_safe))
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_safe))
    (MODEL_DIR / "classification_report.txt").write_text(
        classification_report(y_test, y_pred, target_names=["Real", "Fake"], zero_division=0)
    )

    print(f"Saved model artifacts to {MODEL_DIR}")


def train(data_dir=DATA_DIR, sample_size=None):
    df, source_name = load_dataset(data_dir, sample_size=sample_size)
    cleaner = TextCleaner()
    feature_engineer = FeatureEngineer()

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    print("Cleaning text...")
    train_cleaned = [cleaner.clean(text) for text in train_df["article_text"]]
    test_cleaned = [cleaner.clean(text) for text in test_df["article_text"]]

    x_train, x_test, vectorizer, scaler = build_features(
        train_df["article_text"].tolist(),
        test_df["article_text"].tolist(),
        train_cleaned,
        test_cleaned,
        feature_engineer,
    )
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    all_metrics = {}
    trained_models = {}
    predictions_by_model = {}

    for name, model in candidate_models().items():
        metrics, predictions = evaluate_model(name, model, x_train, y_train, x_test, y_test)
        all_metrics[name] = metrics
        trained_models[name] = model
        predictions_by_model[name] = predictions

    best_name = max(all_metrics, key=lambda item: (all_metrics[item]["f1"], all_metrics[item]["accuracy"]))
    best_model = trained_models[best_name]

    metadata = {
        "model_name": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_source": source_name,
        "dataset_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "label_mapping": {"Real": 0, "Fake": 1},
        "tfidf_vocabulary_size": int(len(vectorizer.vocabulary_)),
        "feature_columns": FeatureEngineer.FEATURE_COLUMNS,
        "selection_metric": "highest f1, then highest accuracy",
        "warning": "This model estimates whether text resembles labelled real/fake news. It is not a fact-checker.",
    }

    save_artifacts(
        best_model,
        vectorizer,
        scaler,
        all_metrics,
        metadata,
        y_test,
        predictions_by_model[best_name],
    )

    print(f"Best model: {best_name}")
    return best_name, all_metrics[best_name]


def main():
    parser = argparse.ArgumentParser(description="Train fake news detection models.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Directory containing training CSV files.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional balanced sample size for quick runs.")
    args = parser.parse_args()

    train(data_dir=args.data_dir, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
