from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT, MODEL_DIR, TARGETS, TARGET_LABELS, LEAKAGE_COLUMNS

_LOADED_MODELS: Dict[str, Any] = {}


def load_model(target_key: str, force_reload: bool = False):
    """Load and cache a trained scikit-learn pipeline for the specified target."""
    target_key = target_key.lower()
    if target_key not in TARGETS:
        raise ValueError(f"Unknown target key '{target_key}'. Must be one of {list(TARGETS.keys())}")

    if not force_reload and target_key in _LOADED_MODELS:
        return _LOADED_MODELS[target_key]

    model_path = MODEL_DIR / f"{target_key}_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. "
            f"Please run 'python src/train.py' to generate model artifacts."
        )

    model = joblib.load(model_path)
    _LOADED_MODELS[target_key] = model
    return model


def get_risk_tier(prob: float) -> str:
    """Categorize prediction probability into educational risk tiers."""
    if prob < 0.30:
        return "Low Likelihood"
    elif prob < 0.60:
        return "Moderate Likelihood"
    elif prob < 0.80:
        return "High Likelihood"
    else:
        return "Elevated / Critical"


def predict_patient(patient_features: dict, targets: Optional[list] = None) -> dict:
    """
    Run prediction models for a single patient record.

    Parameters:
        patient_features: Dictionary of patient clinical features.
        targets: Optional list of target keys to evaluate (defaults to all four: cad, lad, lcx, rca).

    Returns:
        Dictionary mapping target keys to prediction results, including class, probability,
        label, and educational risk tier.
    """
    if not isinstance(patient_features, dict):
        raise TypeError(f"patient_features must be a dictionary, got {type(patient_features)}")

    # Strict check against target leakage
    forbidden = set(LEAKAGE_COLUMNS)
    overlap = forbidden.intersection(patient_features.keys())
    if overlap:
        raise ValueError(
            f"Target leakage detected! The following target columns must not be in model inputs: {sorted(overlap)}"
        )

    # Convert to DataFrame
    X = pd.DataFrame([patient_features])

    targets_to_run = targets or list(TARGETS.keys())
    output = {}

    for key in targets_to_run:
        key_lower = key.lower()
        if key_lower not in TARGETS:
            continue

        model = load_model(key_lower)
        labels_meta = TARGET_LABELS[key_lower]

        # Determine classifier classes to avoid assuming fixed index 1
        classifier = model.named_steps.get("classifier", model)
        classes = list(getattr(classifier, "classes_", [0, 1]))

        # Find positive class index (class 1)
        if 1 in classes:
            pos_idx = classes.index(1)
        else:
            pos_idx = -1

        prob_array = model.predict_proba(X)[0]
        pos_prob = float(prob_array[pos_idx]) if pos_idx >= 0 and len(prob_array) > pos_idx else float(prob_array[-1])
        pred_class = int(model.predict(X)[0])

        pos_label = labels_meta["positive"]
        neg_label = labels_meta["negative"]
        predicted_label = pos_label if pred_class == 1 else neg_label

        output[key_lower] = {
            "target_name": labels_meta["name"],
            "positive_label": pos_label,
            "negative_label": neg_label,
            "predicted_class": pred_class,
            "predicted_label": predicted_label,
            "positive_probability": float(np.round(pos_prob, 4)),
            "risk_tier": get_risk_tier(pos_prob),
            "disclaimer": "Educational research estimate only. Not a clinical diagnosis."
        }

    return output


def reset_model_cache():
    """Clear memory-cached models to force re-loading from disk."""
    global _LOADED_MODELS
    _LOADED_MODELS.clear()
