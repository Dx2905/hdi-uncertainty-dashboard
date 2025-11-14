# src/shap_utils.py
from __future__ import annotations
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def _get_model_steps(pipe: Pipeline):
    pre = pipe.named_steps.get("pre")
    clf = pipe.named_steps.get("clf")
    if clf is None:
        clf = pipe.named_steps.get("cal")
        base = getattr(clf, "base_estimator", None) or getattr(clf, "estimator", None)
        return pre, clf, base
    return pre, clf, clf

def _safe_feature_names(pre, X_ref: pd.DataFrame):
    """
    Robustly derive feature names from a fitted ColumnTransformer across sklearn versions.
    Falls back to input column names when transformers don't expose names.
    """
    names = []
    for name, trans, cols in pre.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if trans == "drop":
            continue

        # Materialize column list
        if isinstance(cols, (list, tuple, np.ndarray)):
            col_list = list(cols)
        else:
            # column indices or boolean mask
            col_list = list(X_ref.columns[cols])

        # If the transformer is a Pipeline, use its last step
        if isinstance(trans, Pipeline):
            last = trans.steps[-1][1]
        else:
            last = trans

        # Try modern API
        try:
            fn = last.get_feature_names_out(col_list)
            names.extend(list(fn))
            continue
        except Exception:
            pass

        # Try legacy API
        try:
            fn = last.get_feature_names()
            # Some return ndarray
            fn = fn.tolist() if hasattr(fn, "tolist") else fn
            names.extend(list(fn))
            continue
        except Exception:
            pass

        # Fallback: just use the original column names
        names.extend(col_list)

    return np.array(names, dtype=object)

def shap_explain_global(model: Pipeline, X_train: pd.DataFrame, max_display: int = 10) -> pd.DataFrame:
    pre, clf_or_cal, base_for_type = _get_model_steps(model)
    feature_names = _safe_feature_names(pre, X_train)
    X_proc = pre.transform(X_train)

    est_for_type = base_for_type or clf_or_cal
    if isinstance(est_for_type, LogisticRegression):
        explainer = shap.LinearExplainer(est_for_type, X_proc, feature_names=feature_names)
        sv = explainer(X_proc)
    elif isinstance(est_for_type, RandomForestClassifier):
        explainer = shap.TreeExplainer(est_for_type)
        sv = explainer(X_proc)
    else:
        explainer = shap.Explainer(lambda z: clf_or_cal.predict_proba(z)[:, 1], X_proc, feature_names=feature_names)
        sv = explainer(X_proc)

    mean_abs = np.abs(sv.values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).head(max_display)

def shap_explain_local(model: Pipeline, X_row: pd.DataFrame, X_background: pd.DataFrame, class_index: int = 1):
    """
    Local explanation for a single patient.
    Ensures returned feature_names length matches SHAP value vector length to avoid indexing errors.
    """
    pre, clf_or_cal, base_for_type = _get_model_steps(model)

    # Try to get detailed feature names; may be shorter than transformed width on older sklearn
    base_feature_names = _safe_feature_names(pre, X_background)

    # Transform to model space
    X_bg_proc = pre.transform(X_background)
    X_row_proc = pre.transform(X_row)

    est_for_type = base_for_type or clf_or_cal

    # Compute SHAP values
    if isinstance(est_for_type, LogisticRegression):
        explainer = shap.LinearExplainer(est_for_type, X_bg_proc, feature_names=base_feature_names)
        sv = explainer(X_row_proc)
        values = np.atleast_2d(sv.values)[0]
        base_value = float(np.atleast_1d(sv.base_values)[0])
    elif isinstance(est_for_type, RandomForestClassifier):
        explainer = shap.TreeExplainer(est_for_type)
        sv = explainer(X_row_proc)
        values = np.atleast_2d(sv.values)[0]
        base_value = float(np.atleast_1d(sv.base_values)[0])
    else:
        f = lambda Z: clf_or_cal.predict_proba(Z)[:, class_index]
        explainer = shap.Explainer(f, X_bg_proc, feature_names=base_feature_names)
        sv = explainer(X_row_proc)
        values = np.atleast_2d(sv.values)[0]
        base_value = float(np.atleast_1d(sv.base_values)[0])

    n_vals = len(values)

    # Align feature names length to SHAP values length
    if base_feature_names is not None and len(base_feature_names) == n_vals:
        feature_names = list(base_feature_names)
    else:
        # Best effort: try ColumnTransformer.get_feature_names_out (newer sklearn)
        try:
            feature_names = list(pre.get_feature_names_out())
        except Exception:
            feature_names = []

        # If still mismatched, pad/truncate to exact length
        if len(feature_names) != n_vals:
            feature_names = (feature_names + [f"feature_{i}" for i in range(len(feature_names), n_vals)])[:n_vals]

    return {
        "feature_names": feature_names,
        "shap_values": values.tolist(),
        "base_value": base_value,
    }


# def shap_explain_local(model: Pipeline, X_row: pd.DataFrame, X_background: pd.DataFrame, class_index: int = 1):
#     pre, clf_or_cal, base_for_type = _get_model_steps(model)
#     feature_names = _safe_feature_names(pre, X_background)

#     X_bg_proc = pre.transform(X_background)
#     X_row_proc = pre.transform(X_row)

#     est_for_type = base_for_type or clf_or_cal
#     if isinstance(est_for_type, LogisticRegression):
#         explainer = shap.LinearExplainer(est_for_type, X_bg_proc, feature_names=feature_names)
#         sv = explainer(X_row_proc)
#         values = sv.values[0]
#         base_value = float(np.atleast_1d(sv.base_values)[0])
#     elif isinstance(est_for_type, RandomForestClassifier):
#         explainer = shap.TreeExplainer(est_for_type)
#         sv = explainer(X_row_proc)
#         values = sv.values[0]
#         base_value = float(np.atleast_1d(sv.base_values)[0])
#     else:
#         f = lambda Z: clf_or_cal.predict_proba(Z)[:, class_index]
#         explainer = shap.Explainer(f, X_bg_proc, feature_names=feature_names)
#         sv = explainer(X_row_proc)
#         values = sv.values[0]
#         base_value = float(np.atleast_1d(sv.base_values)[0])

#     return {
#         "feature_names": list(feature_names),
#         "shap_values": values.tolist(),
#         "base_value": base_value,
#     }
