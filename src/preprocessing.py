import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def encode_target(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.lower()
    mapping = {"cad": 1, "stenotic": 1, "normal": 0}
    encoded = cleaned.map(mapping)
    if encoded.isna().any():
        unknown = sorted(cleaned[encoded.isna()].unique().tolist())
        raise ValueError(f"Unexpected target labels in {series.name}: {unknown}")
    return encoded.astype(int)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [c for c in X.columns if c not in numeric_columns]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])
