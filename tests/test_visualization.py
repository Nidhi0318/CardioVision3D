import pytest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visualization import create_heart_figure, get_vessel_status_dict


def test_create_heart_figure_defaults():
    fig = create_heart_figure()
    assert fig is not None
    assert len(fig.data) >= 7  # Surface + aorta + 3 vessels (lines + markers)


def test_create_heart_figure_with_status():
    status = {
        "LAD": {"status": "Stenotic", "probability": 0.85},
        "LCX": {"status": "Normal", "probability": 0.12},
        "RCA": "Normal"
    }
    fig = create_heart_figure(vessel_status=status, show_orientation_labels=True, theme="dark")
    assert fig is not None
    trace_names = [t.name for t in fig.data if t.name]
    assert any("LAD: Stenotic" in n for n in trace_names)
    assert any("LCX: Normal" in n for n in trace_names)


def test_get_vessel_status_dict():
    preds = {
        "cad": {"predicted_class": 1, "positive_probability": 0.75},
        "lad": {"predicted_class": 1, "positive_probability": 0.82},
        "lcx": {"predicted_class": 0, "positive_probability": 0.25},
        "rca": {"predicted_class": 0, "positive_probability": 0.18},
    }
    v_dict = get_vessel_status_dict(preds)
    assert v_dict["LAD"]["status"] == "Stenotic"
    assert v_dict["LAD"]["probability"] == 0.82
    assert v_dict["LCX"]["status"] == "Normal"
    assert v_dict["LCX"]["probability"] == 0.25
    assert v_dict["RCA"]["status"] == "Normal"
