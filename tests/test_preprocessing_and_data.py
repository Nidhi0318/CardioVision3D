import pytest
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import DATA_PATH, SHEET_NAME, LEAKAGE_COLUMNS, TARGETS
from preprocessing import encode_target, build_preprocessor


def test_dataset_exists_and_loads():
    assert DATA_PATH.exists(), f"Dataset file not found at {DATA_PATH}"
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    assert not df.empty, "Dataset sheet is empty"
    assert len(df) >= 300, f"Expected ~303 records, found {len(df)}"
    # Target columns must be present in raw data
    for col in LEAKAGE_COLUMNS:
        assert col in df.columns, f"Target column '{col}' missing from raw data"


def test_target_label_encoding():
    s_cad = pd.Series(["CAD", "Normal", "cad", "normal"])
    encoded_cad = encode_target(s_cad)
    assert encoded_cad.tolist() == [1, 0, 1, 0]

    s_vessel = pd.Series(["Stenotic", "Normal", "stenotic"])
    encoded_vessel = encode_target(s_vessel)
    assert encoded_vessel.tolist() == [1, 0, 1]


def test_target_label_encoding_invalid_raises():
    s_invalid = pd.Series(["CAD", "Unknown_Value"])
    with pytest.raises(ValueError, match="Unexpected target labels"):
        encode_target(s_invalid)


def test_leakage_columns_excluded():
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    features = [c for c in df.columns if c not in LEAKAGE_COLUMNS]
    for leak in LEAKAGE_COLUMNS:
        assert leak not in features


def test_preprocessor_build_and_transform():
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    features = [c for c in df.columns if c not in LEAKAGE_COLUMNS]
    X_sample = df[features].head(10)
    pre = build_preprocessor(X_sample)
    X_trans = pre.fit_transform(X_sample)
    assert X_trans.shape[0] == 10
    assert X_trans.shape[1] > 0
    assert not pd.isna(X_trans).any()
