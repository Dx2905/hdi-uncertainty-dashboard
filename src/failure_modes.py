
from __future__ import annotations
import numpy as np
import pandas as pd

def top_failures(X_test: pd.DataFrame, y_test, proba, k=5):
    y_test = np.asarray(y_test)
    proba = np.asarray(proba)
    y_pred = (proba >= 0.5).astype(int)
    # confidence = distance from threshold
    conf = np.abs(proba - 0.5)

    df = X_test.copy()
    df["y_true"] = y_test
    df["proba"] = proba
    df["conf"] = conf
    df["y_pred"] = y_pred
    df["err"] = (y_pred != y_test).astype(int)

    fps = df[(df.y_true == 0) & (df.y_pred == 1)].sort_values("conf", ascending=False).head(k)
    fns = df[(df.y_true == 1) & (df.y_pred == 0)].sort_values("conf", ascending=False).head(k)
    return fps, fns
