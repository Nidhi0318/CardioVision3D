import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PATH, SHEET_NAME, TARGETS, LEAKAGE_COLUMNS, MODEL_DIR, RESULTS_DIR
from preprocessing import encode_target, build_preprocessor


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Copy the supplied Excel file into data/raw/."
        )
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    if df.empty:
        raise ValueError(f"No patient records found in sheet: {SHEET_NAME}")

    feature_columns = [c for c in df.columns if c not in LEAKAGE_COLUMNS]
    X_all = df[feature_columns].copy()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for model_key, target_column in TARGETS.items():
        y_all = encode_target(df[target_column])
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
        )
        candidates = {
            "logistic_regression": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=42
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                random_state=42
            ),
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        candidate_scores = {}
        pipelines = {}
        for name, estimator in candidates.items():
            pipeline = Pipeline([
                ("preprocess", build_preprocessor(X_train)),
                ("classifier", estimator),
            ])
            scores = cross_val_score(
                pipeline, X_train, y_train, cv=cv, scoring="roc_auc"
            )
            candidate_scores[name] = {
                "cv_roc_auc_mean": float(scores.mean()),
                "cv_roc_auc_std": float(scores.std()),
            }
            pipelines[name] = pipeline

        best_name = max(candidate_scores, key=lambda n: candidate_scores[n]["cv_roc_auc_mean"])
        best_model = pipelines[best_name]
        best_model.fit(X_train, y_train)
        predictions = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)[:, 1]
        try:
            auc = float(roc_auc_score(y_test, probabilities))
        except ValueError:
            auc = None

        metrics = {
            "target": target_column,
            "selected_model": best_name,
            "n_records": int(len(df)),
            "n_features_before_encoding": int(len(feature_columns)),
            "positive_class_definition": "CAD" if target_column == "Cath" else "Stenotic",
            "class_counts_positive_negative": {
                "positive": int(y_all.sum()), "negative": int((1 - y_all).sum())
            },
            "cross_validation_candidates": candidate_scores,
            "held_out_test_metrics": {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision": float(precision_score(y_test, predictions, zero_division=0)),
                "recall": float(recall_score(y_test, predictions, zero_division=0)),
                "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
                "roc_auc": auc,
                "confusion_matrix_labels_0_then_1": confusion_matrix(
                    y_test, predictions, labels=[0, 1]
                ).tolist(),
                "classification_report": classification_report(
                    y_test, predictions, zero_division=0, output_dict=True
                ),
            },
        }
        joblib.dump(best_model, MODEL_DIR / f"{model_key}_model.joblib")
        all_results[model_key] = metrics
        print(f"\n{target_column} — selected {best_name}")
        print(json.dumps(metrics["held_out_test_metrics"], indent=2))

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved models to: {MODEL_DIR}")
    print(f"Saved metrics to: {RESULTS_DIR / 'metrics.json'}")
    print("Reminder: experimental prototype only; not clinically validated.")


if __name__ == "__main__":
    main()
