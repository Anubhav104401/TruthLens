import os
from typing import Tuple, Dict

def blend(
    ml_score: float, 
    rule_score: float, 
    gemini_score: float, 
    gemini_confidence: float, 
    gemini_available: bool
) -> Tuple[float, Dict[str, float]]:
    """
    Blends the ML, Rule, and Gemini scores into a single 0-100 score.
    """
    w_gemini = float(os.getenv("ENSEMBLE_W_GEMINI", "0.50"))
    w_ml = float(os.getenv("ENSEMBLE_W_ML", "0.35"))
    w_rule = float(os.getenv("ENSEMBLE_W_RULE", "0.15"))
    
    # Normalize weights just in case they don't sum to 1.0
    total = w_gemini + w_ml + w_rule
    w_gemini /= total
    w_ml /= total
    w_rule /= total

    if not gemini_available:
        eff_gemini = 0.0
    else:
        # Scale Gemini's weight by its confidence
        eff_gemini = w_gemini * (gemini_confidence / 100.0)

    remaining = 1.0 - eff_gemini
    
    # Redistribute remaining weight proportionally to ML and Rule
    ml_rule_total = w_ml + w_rule
    if ml_rule_total > 0:
        eff_ml = (w_ml / ml_rule_total) * remaining
        eff_rule = (w_rule / ml_rule_total) * remaining
    else:
        eff_ml = remaining
        eff_rule = 0.0

    final_score = (eff_gemini * gemini_score) + (eff_ml * ml_score) + (eff_rule * rule_score)
    weights_used = {
        "gemini": round(eff_gemini, 3),
        "ml": round(eff_ml, 3),
        "rule": round(eff_rule, 3)
    }
    
    return final_score, weights_used

def detect_conflict(ml_score: float, gemini_score: float, gemini_confidence: float) -> bool:
    """
    Detects if the ML score and Gemini score are in clear conflict.
    Conflict is defined as:
    - Gemini confidence is high (> 70)
    - Both scores are more than 20 points away from the midpoint (50) but on opposite sides.
    """
    if gemini_confidence <= 70:
        return False
        
    ml_is_fake = ml_score >= 70
    ml_is_real = ml_score <= 30
    
    gemini_is_fake = gemini_score >= 70
    gemini_is_real = gemini_score <= 30
    
    if (ml_is_fake and gemini_is_real) or (ml_is_real and gemini_is_fake):
        return True
        
    return False

def rescale_rule_adjustment(rule_adjustment: int) -> float:
    """
    Rescales the [-15, +20] rule adjustment linearly to [0, 100].
    """
    # -15 -> 0
    # +20 -> 100
    val = ((rule_adjustment - (-15)) / (20 - (-15))) * 100
    return max(0.0, min(100.0, float(val)))
