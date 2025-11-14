
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# DATA_DIR = Path("../data")
DATA_DIR = Path("data")

# Reproducible splits: 60/20/20 (train/val/test)
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.25  # of the 80% temp set -> overall 20%

# Columns for the Heart dataset (Cleveland-style)
HEART_CATEGORICAL: List[str] = ["cp", "restecg", "slope", "ca", "thal"]
HEART_BINARIES: List[str] = ["sex", "fbs", "exang"]
HEART_NUMERIC: List[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]

def _split_stratified(X: pd.DataFrame, y: pd.Series, seed: int = RANDOM_STATE):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, stratify=y_temp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def _build_preprocessor_heart(X: pd.DataFrame) -> ColumnTransformer:
    """One-hot encode categoricals; scale numeric + binary."""
    cat_cols = [c for c in HEART_CATEGORICAL if c in X.columns]
    num_cols = [c for c in HEART_NUMERIC + HEART_BINARIES if c in X.columns]

    preproc = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )
    return preproc

def load_heart(csv_path: Path | None = None) -> Tuple[pd.DataFrame, ...]:
    """
    Load and prep the Heart dataset where the target column is named 'condition'.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test, preprocessor
    """
    if csv_path is None:
        csv_path = DATA_DIR / "heart.csv"

    df = pd.read_csv(csv_path)

    # Standardize target name to 'target'
    if "condition" in df.columns:
        df = df.rename(columns={"condition": "target"})
    elif "target" not in df.columns and "num" in df.columns:
        # Safety: if someone provides a 'num' version (0–4), binarize >0
        df["target"] = (pd.to_numeric(df["num"], errors="coerce").fillna(0) > 0).astype(int)
        df = df.drop(columns=["num"])
    elif "target" not in df.columns:
        raise ValueError("Expected a 'condition', 'target', or 'num' column for the label.")

    # Clean any legacy missing markers like '?'
    df = df.replace("?", np.nan)

    # Split features/label
    y = df["target"].astype(int)
    X = df.drop(columns=["target"])

    # Cast types for preprocessing
    for col in HEART_CATEGORICAL:
        if col in X.columns:
            X[col] = X[col].astype("category")
    for col in HEART_BINARIES + HEART_NUMERIC:
        for c in [col]:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce")

    # Your copy shows no NaNs; if needed later we can add SimpleImputer.
    X_train, X_val, X_test, y_train, y_val, y_test = _split_stratified(X, y)
    preproc = _build_preprocessor_heart(X)

    return X_train, X_val, X_test, y_train, y_val, y_test, preproc

# --- Manual smoke test: `python -m src.load`
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, pre = load_heart()
    def pct_pos(y): 
        return f"{(y.mean() if len(y) else float('nan')):.3f}"
    print("Splits (rows):", len(X_train), len(X_val), len(X_test))
    print("Pos rate (train/val/test):", pct_pos(y_train), pct_pos(y_val), pct_pos(y_test))
    pipe = Pipeline([("pre", pre)])
    pipe.fit(X_train, y_train)
    Xt = pipe.transform(X_train)
    print("Transformed train shape:", Xt.shape)


# ---------- PIMA (Indians Diabetes) ----------
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

PIMA_ZERO_IS_NAN = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def _build_preprocessor_pima(X: pd.DataFrame) -> ColumnTransformer:
    # All columns are numeric; we’ll impute and scale.
    from sklearn.impute import SimpleImputer
    numeric_cols = X.columns.tolist()
    preproc = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler())
            ]), numeric_cols)
        ],
        remainder="drop",
    )
    return preproc

def load_pima(csv_path: Path | None = None) -> Tuple[pd.DataFrame, ...]:
    """
    Load and prep the Pima Indians Diabetes dataset (Outcome is the label).
    - Replaces biologically-impossible zeros with NaN in selected columns
    - Stratified 60/20/20 split
    - Returns: X_train, X_val, X_test, y_train, y_val, y_test, preprocessor
    """
    if csv_path is None:
        csv_path = DATA_DIR / "diabetes.csv"

    df = pd.read_csv(csv_path)

    # Label
    if "Outcome" not in df.columns:
        raise ValueError("Expected label column 'Outcome' in diabetes.csv")
    y = df["Outcome"].astype(int)
    X = df.drop(columns=["Outcome"]).copy()

    # Replace zeros with NaN in the known problematic fields
    for col in PIMA_ZERO_IS_NAN:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X.loc[X[col] == 0, col] = np.nan

    # Ensure numeric dtype for all features
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Stratified 60/20/20 split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )

    preproc = _build_preprocessor_pima(X)

    return X_train, X_val, X_test, y_train, y_val, y_test, preproc

# --- Optional quick check: `python -m src.load_pima_check`
# if __name__ == "__main__" and False:
#     X_train, X_val, X_test, y_train, y_val, y_test, pre = load_pima()
#     from sklearn.pipeline import Pipeline
#     pipe = Pipeline([("pre", pre)])
#     pipe.fit(X_train, y_train)
#     print("Pima splits:", len(X_train), len(X_val), len(X_test))
#     print("Pima pos rates:", y_train.mean(), y_val.mean(), y_test.mean())
#     print("Transformed shape (train):", pipe.transform(X_train).shape)
