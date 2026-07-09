import asyncio
import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from services.prediction_service import PredictionService
from sklearn.metrics import accuracy_score, f1_score

load_dotenv()

async def evaluate():
    service = PredictionService()
    
    csv_path = Path(__file__).resolve().parents[2] / "data" / "evaluation_set.csv"
    if not csv_path.exists():
        print(f"Evaluation set not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    results = {
        "rules_only": [],
        "rules_ml": [],
        "ensemble": []
    }
    y_true = df["label"].tolist()
    
    # Process each item
    for idx, row in df.iterrows():
        text = row["text"]
        
        # 1. Rules only
        rule_res = service.rule_engine.evaluate(text)
        # simplistic thresholding: > 0 means fake
        rules_pred = 1 if rule_res["adjustment"] > 0 else 0
        results["rules_only"].append(rules_pred)
        
        # 2. ML + Rules (force Gemini off)
        os.environ["ENABLE_GEMINI_VERIFICATION"] = "false"
        res_ml = await service.analyze_article(text)
        ml_pred = 1 if res_ml["final_score"] >= 60 else 0
        results["rules_ml"].append(ml_pred)
        
        # 3. Ensemble (Gemini on)
        os.environ["ENABLE_GEMINI_VERIFICATION"] = "true"
        res_ens = await service.analyze_article(text)
        ens_pred = 1 if res_ens["final_score"] >= 60 else 0
        results["ensemble"].append(ens_pred)
        
    # Calculate metrics
    print("--- Evaluation Results ---")
    for mode in ["rules_only", "rules_ml", "ensemble"]:
        acc = accuracy_score(y_true, results[mode])
        f1 = f1_score(y_true, results[mode], zero_division=0)
        print(f"{mode.upper()}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")

if __name__ == "__main__":
    asyncio.run(evaluate())
