from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import shap

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT, TARGETS, TARGET_LABELS, LEAKAGE_COLUMNS, FEATURE_DESCRIPTIONS
from predict import load_model

_EXPLAINER_CACHE: Dict[str, Any] = {}


def clean_feature_name(name: str) -> str:
    """Transform raw pipeline feature names (e.g., 'numeric__BP', 'categorical__Sex_Male') into readable clinical labels."""
    if name.startswith("numeric__"):
        clean = name.replace("numeric__", "")
    elif name.startswith("categorical__"):
        parts = name.replace("categorical__", "").split("_", 1)
        if len(parts) == 2:
            base, val = parts
            val_clean = "Yes" if val.upper() == "Y" else ("No" if val.upper() == "N" else val)
            clean = f"{base} ({val_clean})"
        else:
            clean = parts[0]
    else:
        clean = name
    return clean


def get_feature_description(clean_name: str) -> str:
    """Look up feature description from clinical dictionary."""
    base_name = clean_name.split(" (")[0]
    return FEATURE_DESCRIPTIONS.get(base_name, "Clinical diagnostic biomarker or measurement")


def get_explainer(target_key: str):
    """Get or create cached SHAP TreeExplainer for the specified target model."""
    target_key = target_key.lower()
    if target_key in _EXPLAINER_CACHE:
        return _EXPLAINER_CACHE[target_key]

    model = load_model(target_key)
    classifier = model.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)
    _EXPLAINER_CACHE[target_key] = explainer
    return explainer


def get_global_feature_importance(target_key: str = "cad", top_n: int = 15) -> pd.DataFrame:
    """
    Compute global feature importances from the trained ensemble model.

    Returns:
        DataFrame with columns: ['feature', 'clean_feature', 'importance', 'description']
    """
    model = load_model(target_key)
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]

    raw_features = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    else:
        # Fallback for linear models or other estimators
        importances = np.abs(getattr(classifier, "coef_", np.ones((1, len(raw_features)))))[0]

    df_importance = pd.DataFrame({
        "raw_feature": raw_features,
        "clean_feature": [clean_feature_name(f) for f in raw_features],
        "importance": importances,
    })

    # Group by clean feature in case one-hot encoding split some categorical levels
    grouped = df_importance.groupby("clean_feature", as_index=False)["importance"].sum()
    grouped["description"] = [get_feature_description(f) for f in grouped["clean_feature"]]
    grouped = grouped.sort_values(by="importance", ascending=False).reset_index(drop=True)

    return grouped.head(top_n)


def explain_patient(
    patient_features: dict,
    target_key: str = "cad",
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Generate local patient-specific feature attribution using SHAP TreeExplainer.

    Returns:
        Dictionary containing:
        - target: target key
        - target_name: human readable target name
        - base_value: model base/prior probability
        - current_probability: model predicted positive class probability
        - top_positive: features increasing disease likelihood (positive SHAP push)
        - top_negative: features reducing disease likelihood (protective/negative SHAP push)
        - all_attributions: complete sorted DataFrame of feature attributions
        - method: explanation method used ('SHAP TreeExplainer')
        - disclaimer: explanation limitations notice
    """
    # Guard against leakage columns
    forbidden = set(LEAKAGE_COLUMNS)
    overlap = forbidden.intersection(patient_features.keys())
    if overlap:
        raise ValueError(f"Target leakage detected in explain_patient: {sorted(overlap)}")

    target_key = target_key.lower()
    model = load_model(target_key)
    labels_meta = TARGET_LABELS.get(target_key, {"name": target_key.upper(), "positive": "Positive"})

    X = pd.DataFrame([patient_features])
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    raw_feature_names = preprocessor.get_feature_names_out()

    X_trans = preprocessor.transform(X)

    # Calculate SHAP values
    explainer = get_explainer(target_key)
    shap_vals = explainer.shap_values(X_trans)

    # For binary classification TreeExplainer, shap_vals is (n_samples, n_features, 2)
    # or list [class0_shap, class1_shap]
    if isinstance(shap_vals, list):
        class1_shap = shap_vals[1][0]
    elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 3:
        class1_shap = shap_vals[0, :, 1]
    elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 2:
        class1_shap = shap_vals[0, :]
    else:
        class1_shap = np.array(shap_vals).flatten()

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1] if len(base_val) > 1 else base_val[0])
    else:
        base_val = float(base_val)

    # Build attribution records
    records = []
    for raw_name, shap_val in zip(raw_feature_names, class1_shap):
        clean_name = clean_feature_name(raw_name)
        # Try to extract the original raw feature value for user reference
        base_feat = clean_name.split(" (")[0]
        val_display = patient_features.get(base_feat, "—")

        direction = "Increases Likelihood" if shap_val > 0 else "Reduces Likelihood / Protective"

        records.append({
            "raw_feature": raw_name,
            "feature": clean_name,
            "patient_value": val_display,
            "shap_value": float(shap_val),
            "abs_shap": float(abs(shap_val)),
            "direction": direction,
            "description": get_feature_description(clean_name),
        })

    df_attr = pd.DataFrame(records).sort_values(by="abs_shap", ascending=False).reset_index(drop=True)

    pos_features = df_attr[df_attr["shap_value"] > 0].sort_values(by="shap_value", ascending=False).head(top_n)
    neg_features = df_attr[df_attr["shap_value"] < 0].sort_values(by="shap_value", ascending=True).head(top_n)

    prob_array = model.predict_proba(X)[0]
    pos_prob = float(prob_array[1] if len(prob_array) > 1 else prob_array[0])

    return {
        "target": target_key,
        "target_name": labels_meta["name"],
        "base_value": round(base_val, 4),
        "predicted_probability": round(pos_prob, 4),
        "top_positive": pos_features.to_dict(orient="records"),
        "top_negative": neg_features.to_dict(orient="records"),
        "all_attributions": df_attr,
        "method": "SHAP TreeExplainer (Additive Feature Attributions)",
        "limitations": (
            "Feature attribution scores reflect model mathematical dependencies on this training dataset. "
            "They indicate which variables shifted this model's prediction higher or lower, but do NOT prove "
            "biological causation, mechanism of action, or clinical diagnosis."
        )
    }


def explain_model(target_key: str = "cad"):
    """Global model explanation summary for backward compatibility."""
    return get_global_feature_importance(target_key=target_key)
