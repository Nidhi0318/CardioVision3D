import pytest
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import DATA_PATH, SHEET_NAME, LEAKAGE_COLUMNS, TARGETS
from predict import load_model, predict_patient, reset_model_cache


@pytest.fixture(scope="module")
def sample_patient_dict():
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    features = [c for c in df.columns if c not in LEAKAGE_COLUMNS]
    return df.loc[0, features].to_dict()


def test_all_models_load():
    reset_model_cache()
    for target_key in TARGETS.keys():
        model = load_model(target_key)
        assert model is not None
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")


def test_predict_patient_structure(sample_patient_dict):
    preds = predict_patient(sample_patient_dict)
    assert isinstance(preds, dict)
    assert set(preds.keys()) == {"cad", "lad", "lcx", "rca"}

    for key, res in preds.items():
        assert "predicted_class" in res
        assert res["predicted_class"] in [0, 1]
        assert "positive_probability" in res
        assert 0.0 <= res["positive_probability"] <= 1.0
        assert "predicted_label" in res
        assert "risk_tier" in res
        assert "disclaimer" in res


def test_predict_patient_leakage_detection(sample_patient_dict):
    # Add a leakage column
    leaked_patient = dict(sample_patient_dict)
    leaked_patient["Cath"] = "CAD"
    with pytest.raises(ValueError, match="Target leakage detected"):
        predict_patient(leaked_patient)

    leaked_patient2 = dict(sample_patient_dict)
    leaked_patient2["LAD"] = "Stenotic"
    with pytest.raises(ValueError, match="Target leakage detected"):
        predict_patient(leaked_patient2)


def test_predict_patient_invalid_type():
    with pytest.raises(TypeError):
        predict_patient(["invalid", "list"])
