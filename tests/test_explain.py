import pytest
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import DATA_PATH, SHEET_NAME, LEAKAGE_COLUMNS
from explain import clean_feature_name, get_global_feature_importance, explain_patient


@pytest.fixture(scope="module")
def sample_patient_dict():
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    features = [c for c in df.columns if c not in LEAKAGE_COLUMNS]
    return df.loc[0, features].to_dict()


def test_clean_feature_name():
    assert clean_feature_name("numeric__Age") == "Age"
    assert clean_feature_name("categorical__Sex_Male") == "Sex (Male)"
    assert clean_feature_name("categorical__Atypical_Y") == "Atypical (Yes)"
    assert clean_feature_name("categorical__Atypical_N") == "Atypical (No)"


def test_global_feature_importance():
    for target in ["cad", "lad", "lcx", "rca"]:
        df_imp = get_global_feature_importance(target, top_n=10)
        assert not df_imp.empty
        assert len(df_imp) == 10
        assert "clean_feature" in df_imp.columns
        assert "importance" in df_imp.columns
        assert "description" in df_imp.columns
        assert (df_imp["importance"] >= 0).all()


def test_explain_patient_shap(sample_patient_dict):
    exp = explain_patient(sample_patient_dict, target_key="cad", top_n=5)
    assert isinstance(exp, dict)
    assert "base_value" in exp
    assert "predicted_probability" in exp
    assert "top_positive" in exp
    assert "top_negative" in exp
    assert "all_attributions" in exp
    assert "limitations" in exp
    assert len(exp["top_positive"]) <= 5
    assert len(exp["top_negative"]) <= 5


def test_explain_patient_leakage_raises(sample_patient_dict):
    leaked = dict(sample_patient_dict)
    leaked["Cath"] = "CAD"
    with pytest.raises(ValueError, match="Target leakage detected"):
        explain_patient(leaked, target_key="cad")
