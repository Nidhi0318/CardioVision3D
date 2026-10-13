import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT, METRICS_PATH, TARGETS, TARGET_LABELS


def load_metrics() -> Dict[str, Any]:
    """Load serialized training and test evaluation metrics."""
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_PATH}. Please run 'python src/train.py' first."
        )
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def get_metrics_summary_df() -> pd.DataFrame:
    """Format held-out test evaluation metrics into a clean summary DataFrame."""
    metrics_data = load_metrics()
    rows = []
    for key, data in metrics_data.items():
        test_m = data.get("held_out_test_metrics", {})
        cv_m = data.get("cross_validation_candidates", {}).get(data.get("selected_model", ""), {})
        
        target_name = TARGET_LABELS.get(key, {}).get("name", key.upper())
        pos_def = data.get("positive_class_definition", "Positive")
        pos_cnt = data.get("class_counts_positive_negative", {}).get("positive", 0)
        neg_cnt = data.get("class_counts_positive_negative", {}).get("negative", 0)

        rows.append({
            "Target Key": key.upper(),
            "Clinical Target": target_name,
            "Selected Model": data.get("selected_model", "Unknown").replace("_", " ").title(),
            "CV ROC-AUC (Mean ± SD)": f"{cv_m.get('cv_roc_auc_mean', 0):.3f} ± {cv_m.get('cv_roc_auc_std', 0):.3f}",
            "Test Accuracy": f"{test_m.get('accuracy', 0) * 100:.1f}%",
            "Test Precision": f"{test_m.get('precision', 0) * 100:.1f}%",
            "Test Recall": f"{test_m.get('recall', 0) * 100:.1f}%",
            "Test F1-Score": f"{test_m.get('f1_score', 0):.3f}",
            "Test ROC-AUC": f"{test_m.get('roc_auc', 0):.3f}" if test_m.get("roc_auc") is not None else "N/A",
            "Positive Count": pos_cnt,
            "Negative Count": neg_cnt,
            "Prevalence": f"{(pos_cnt / (pos_cnt + neg_cnt) * 100):.1f}%" if (pos_cnt + neg_cnt) > 0 else "N/A"
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    try:
        metrics = load_metrics()
        print("=" * 70)
        print("CARDIOVISION3D — MODEL EVALUATION SUMMARY (HELD-OUT TEST SET)")
        print("=" * 70)
        df_summary = get_metrics_summary_df()
        print(df_summary[["Clinical Target", "Selected Model", "Test Accuracy", "Test Recall", "Test F1-Score", "Test ROC-AUC"]].to_string(index=False))
        print("\nFull metrics JSON loaded successfully.")
    except Exception as e:
        print(f"Error loading evaluation metrics: {e}")
