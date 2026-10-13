import pytest
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import METRICS_PATH, TARGETS
from evaluate import load_metrics, get_metrics_summary_df


def test_metrics_json_integrity():
    assert METRICS_PATH.exists(), f"Metrics file {METRICS_PATH} missing"
    metrics = load_metrics()
    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == set(TARGETS.keys())

    for target_key, data in metrics.items():
        assert "target" in data
        assert "selected_model" in data
        assert "n_records" in data
        assert data["n_records"] == 303
        assert "held_out_test_metrics" in data

        test_m = data["held_out_test_metrics"]
        assert "accuracy" in test_m
        assert "precision" in test_m
        assert "recall" in test_m
        assert "f1_score" in test_m
        assert 0.0 <= test_m["accuracy"] <= 1.0
        assert 0.0 <= test_m["recall"] <= 1.0

        assert "confusion_matrix_labels_0_then_1" in test_m
        cm = test_m["confusion_matrix_labels_0_then_1"]
        assert len(cm) == 2 and len(cm[0]) == 2


def test_metrics_summary_df():
    df = get_metrics_summary_df()
    assert not df.empty
    assert len(df) == 4
    assert "Clinical Target" in df.columns
    assert "Test Accuracy" in df.columns
    assert "Test Recall" in df.columns
