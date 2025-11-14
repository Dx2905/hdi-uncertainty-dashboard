
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump

import json, sys, platform, time
import sklearn, numpy, pandas, shap
from pathlib import Path


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

from src.load import load_heart, load_pima
from src.utils import ARTIFACTS_DIR, METRICS_DIR, dump_json

RANDOM_STATE = 42

def _models():
    """Return the base models we’ll train."""
    lr = LogisticRegression(
        penalty="l2",
        solver="liblinear",            # stable for small tabular data
        class_weight="balanced",
        max_iter=200,
        random_state=RANDOM_STATE,
    )
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return {"logreg": lr, "rf": rf}

def _eval_and_save(dataset: str, model_name: str, y_true: np.ndarray, proba: np.ndarray, suffix: str):
    """Compute metrics + reliability, then save JSON."""
    roc = float(roc_auc_score(y_true, proba))
    ap = float(average_precision_score(y_true, proba))
    brier = float(brier_score_loss(y_true, proba))
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="uniform")

    metrics = {
        "dataset": dataset,
        "model": model_name,
        "variant": suffix,         # "base", "sigmoid", "isotonic"
        "roc_auc": roc,
        "avg_precision": ap,
        "brier": brier,
        "n_test": int(len(y_true)),
    }
    reliability = {"mean_pred": mean_pred.tolist(), "frac_pos": frac_pos.tolist()}

    dump_json(metrics, METRICS_DIR / f"{dataset}_{model_name}_{suffix}.json")
    dump_json(reliability, METRICS_DIR / f"{dataset}_{model_name}_reliability_{suffix}.json")

def _train_one_dataset(dataset: str):
    """Load data, fit models, calibrate, evaluate, and save artifacts+metrics."""
    if dataset == "heart":
        X_train, X_val, X_test, y_train, y_val, y_test, pre = load_heart()
    elif dataset == "pima":
        X_train, X_val, X_test, y_train, y_val, y_test, pre = load_pima()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    X_tr = pd.concat([X_train, X_val], axis=0)
    y_tr = pd.concat([y_train, y_val], axis=0)

    for name, base_est in _models().items():
        # ----- Base pipeline -----
        pipe = Pipeline([("pre", pre), ("clf", base_est)])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_test)[:, 1]

        # Save base
        dump(pipe, ARTIFACTS_DIR / f"{dataset}_{name}_base.joblib")
        _eval_and_save(dataset, name, y_test, proba, "base")

        # ----- Calibrations (Platt + Isotonic) -----
        for method in ["sigmoid", "isotonic"]:
            calib = CalibratedClassifierCV(
                base_est,
                method=method,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            )
            pipe_cal = Pipeline([("pre", pre), ("cal", calib)])
            pipe_cal.fit(X_tr, y_tr)
            proba_c = pipe_cal.predict_proba(X_test)[:, 1]

            dump(pipe_cal, ARTIFACTS_DIR / f"{dataset}_{name}_{method}.joblib")
            _eval_and_save(dataset, name, y_test, proba_c, method)
def _write_env_manifest():
    """Write versions/env info so artifacts are reproducible."""
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "shap": shap.__version__,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "_versions.json").write_text(json.dumps(meta, indent=2))
    print("Wrote versions to results/metrics/_versions.json")


def main():
    for ds in ["heart", "pima"]:
        print(f"=== Training on {ds} ===")
        _train_one_dataset(ds)
    _write_env_manifest()    
    print("Done. Artifacts in results/artifacts/, metrics in results/metrics/")

if __name__ == "__main__":
    main()
