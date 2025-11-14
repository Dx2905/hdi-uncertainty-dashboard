
from __future__ import annotations
from typing import List
import numpy as np
from sklearn.base import clone
from sklearn.utils import resample
from sklearn.pipeline import Pipeline

def bootstrap_predict(
    X_train,
    y_train,
    preproc,
    estimator,
    X_row,
    B: int = 200,
    random_state: int = 42,
):
    """
    Estimate prediction uncertainty for one patient via bootstrap resampling.

    Args:
        X_train, y_train: original training data
        preproc: fitted or fit-able preprocessing pipeline
        estimator: sklearn estimator (e.g., LogisticRegression)
        X_row: a single patient row (DataFrame shape (1, n_features))
        B: number of bootstrap resamples
    Returns:
        dict with median, q025, q975, and full list of predictions
    """
    rng = np.random.RandomState(random_state)
    preds: List[float] = []

    for b in range(B):
        # draw random resample (same size, with replacement)
        idx = resample(
            np.arange(len(y_train)),
            replace=True,
            n_samples=len(y_train),
            random_state=rng.randint(0, 1_000_000),
            stratify=y_train,
        )
        est = clone(estimator)
        pipe = Pipeline([("pre", preproc), ("clf", est)])
        pipe.fit(X_train.iloc[idx], y_train.iloc[idx])
        p = pipe.predict_proba(X_row)[:, 1][0]
        preds.append(float(p))

    preds = np.array(preds)
    return {
        "median": float(np.median(preds)),
        "q025": float(np.quantile(preds, 0.025)),
        "q975": float(np.quantile(preds, 0.975)),
        "all": preds.tolist(),
    }
