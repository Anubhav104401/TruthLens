import sys
import os
import pytest
from unittest.mock import patch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from services.ensemble_scorer import blend, detect_conflict, rescale_rule_adjustment

def test_rescale_rule_adjustment():
    assert rescale_rule_adjustment(-15) == 0.0
    assert rescale_rule_adjustment(20) == 100.0
    assert rescale_rule_adjustment(2.5) == 50.0

@patch.dict('os.environ', {"ENSEMBLE_W_GEMINI": "0.50", "ENSEMBLE_W_ML": "0.35", "ENSEMBLE_W_RULE": "0.15"})
def test_blend_gemini_unavailable():
    ml_score = 80.0
    rule_score = 60.0
    # Gemini unavailable, so w_gemini = 0
    # remaining weight = 1.0
    # eff_ml = 0.35 / 0.50 = 0.70
    # eff_rule = 0.15 / 0.50 = 0.30
    # final = 0.70 * 80 + 0.30 * 60 = 56 + 18 = 74
    final, weights = blend(ml_score, rule_score, 0, 0, False)
    assert round(final, 1) == 74.0
    assert weights["gemini"] == 0.0
    assert weights["ml"] == 0.70
    assert weights["rule"] == 0.30

@patch.dict('os.environ', {"ENSEMBLE_W_GEMINI": "0.50", "ENSEMBLE_W_ML": "0.35", "ENSEMBLE_W_RULE": "0.15"})
def test_blend_gemini_confident():
    ml_score = 80.0
    rule_score = 60.0
    gemini_score = 90.0
    gemini_confidence = 100.0
    # eff_gemini = 0.50
    # eff_ml = 0.35
    # eff_rule = 0.15
    # final = 0.5 * 90 + 0.35 * 80 + 0.15 * 60 = 45 + 28 + 9 = 82
    final, weights = blend(ml_score, rule_score, gemini_score, gemini_confidence, True)
    assert round(final, 1) == 82.0
    assert weights["gemini"] == 0.50
    assert weights["ml"] == 0.35
    assert weights["rule"] == 0.15

def test_detect_conflict():
    assert detect_conflict(80, 20, 100) == True
    assert detect_conflict(20, 80, 100) == True
    assert detect_conflict(80, 80, 100) == False
    assert detect_conflict(80, 20, 50) == False # low confidence
