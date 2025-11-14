
from __future__ import annotations

import io
# --- make project root importable (so `src` resolves when running Streamlit) ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

from src.load import load_heart, load_pima
from src.bootstrap import bootstrap_predict
from src.shap_utils import shap_explain_local
from src.viz_uncertainty import plot_interval_band, plot_quantile_dotplot
from src.viz_quality import plot_roc_pr, plot_reliability_from_json

# ---------- Streamlit page ----------
st.set_page_config(page_title="HDI Uncertainty & Explainability", layout="wide")

# ---------- Helpers ----------
ART = Path("results/artifacts")

DATASETS = {
    "Heart (Cleveland)": {
        "loader": load_heart,
        "default_model": ART / "heart_logreg_isotonic.joblib",
    },
    "Pima (Diabetes)": {
        "loader": load_pima,
        "default_model": ART / "pima_logreg_isotonic.joblib",
    },
}

def _ds_key(name: str) -> str:
    return "heart" if "Heart" in name else "pima"

def _default_row_from_train(X_train: pd.DataFrame) -> pd.Series:
    """Use median for numeric columns; mode for categoricals."""
    s = {}
    for c in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[c]):
            s[c] = float(np.nanmedian(pd.to_numeric(X_train[c], errors="coerce")))
        else:
            s[c] = X_train[c].mode().iloc[0]
    return pd.Series(s, index=X_train.columns)

def _numeric_input(label, value, step=1.0, fmt=None, minv=None, maxv=None):
    return st.number_input(label, value=float(value), step=step, format=fmt or "%0.2f", min_value=minv, max_value=maxv)

def _int_input(label, value, minv=None, maxv=None):
    return st.number_input(label, value=int(value), step=1, min_value=minv, max_value=maxv)

def ss_get(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

import time
ss_get("last_predict_ts", None)
ss_get("latency_ms_list", [])


# ---------- Title ----------
st.title("HDI Uncertainty & Explainability – Phase 1 prototype")

# ---------- Dataset / model load ----------
ds_name = st.sidebar.selectbox("Dataset", list(DATASETS.keys()))
loader = DATASETS[ds_name]["loader"]
model_path = DATASETS[ds_name]["default_model"]

# Detect dataset switches and clear old caches
current_ds = "heart" if "Heart" in ds_name else "pima"
prev_ds = st.session_state.get("ds_key_runtime", None)
if prev_ds is None or prev_ds != current_ds:
    for k in ["pred_ready", "proba_test", "y_test_cur", "unc_samples", "point_pred", "input_row"]:
        st.session_state[k] = None
    st.session_state["pred_ready"] = False
    st.session_state["ds_key_runtime"] = current_ds


X_train, X_val, X_test, y_train, y_val, y_test, pre = loader()

if not model_path.exists():
    st.error(f"Model not found: {model_path}. Please run `python -m src.train` first.")
    st.stop()
model = load(model_path)

# ---------- Session keys (for slider-friendly state) ----------
ss_get("pred_ready", False)          # whether we have a prediction computed
ss_get("proba_test", None)           # cached test-set probabilities
ss_get("y_test_cur", None)           # cached y_test
ss_get("unc_samples", None)          # bootstrap samples
ss_get("point_pred", None)           # last point prediction
ss_get("input_row", None)            # last input row
ss_get("ds_key", None)               # 'heart' or 'pima'
ss_get("thresh_slider", 0.50)        # persistent threshold slider value

# ---------- Tabs ----------
tab_input, tab_unc, tab_why, tab_quality, tab_reflect = st.tabs(
    ["Patient Input", "Risk & Uncertainty", "Why (SHAP)", "Model Quality", "Reflections"]
)

# ========================= Patient Input =========================
with tab_input:
    st.subheader("Enter patient features")
    defaults = _default_row_from_train(X_train)

    # Build the input form
    inputs = {}
    for col in X_train.columns:
        # Use int widget only when dtype is integer and there are no NaNs in the column
        col_is_int_like = pd.api.types.is_integer_dtype(X_train[col].dtype) and not X_train[col].isna().any()
        if col_is_int_like:
            start_val = int(round(defaults[col])) if pd.notna(defaults[col]) else 0
            inputs[col] = _int_input(col, start_val, minv=0)
        else:
            start_val = float(defaults[col]) if pd.notna(defaults[col]) else 0.0
            inputs[col] = _numeric_input(col, start_val, step=0.1)

    # single-row DataFrame in the correct column order
    input_row = pd.DataFrame([inputs]).reindex(columns=X_train.columns)

    # Predict button: compute and cache everything for other tabs
    # Predict button: compute and cache everything for other tabs
    if st.button("Predict"):
        import time
        from sklearn.linear_model import LogisticRegression

        start = time.time()
        with st.spinner("Running model & bootstrap..."):
            # Point prediction
            p_point = float(model.predict_proba(input_row)[:, 1][0])

            # Bootstrap uncertainty
            est = LogisticRegression(max_iter=200, class_weight="balanced", solver="liblinear")
            unc = bootstrap_predict(X_train, y_train, pre, est, input_row, B=100)

            # Also cache test-set probabilities for the threshold slider
            proba_test = model.predict_proba(X_test)[:, 1]

        elapsed_ms = int((time.time() - start) * 1000)

        # Cache everything for other tabs + latency
        st.session_state.point_pred = p_point
        st.session_state.unc_samples = unc["all"]
        st.session_state.input_row = input_row
        st.session_state.proba_test = proba_test
        st.session_state.y_test_cur = y_test
        st.session_state.ds_key = _ds_key(ds_name)
        st.session_state.pred_ready = True

        # >>> LATENCY <<<
        # ensure the list exists (you already created it via ss_get above)
        if "latency_ms_list" not in st.session_state or st.session_state.latency_ms_list is None:
            st.session_state.latency_ms_list = []
        st.session_state.last_predict_ts = elapsed_ms
        st.session_state.latency_ms_list.append(elapsed_ms)

        st.success(f"Prediction ready in {elapsed_ms} ms")


# ========================= Risk & Uncertainty =========================
with tab_unc:
    st.subheader("Risk prediction with bootstrap uncertainty")

    if not st.session_state.pred_ready or st.session_state.ds_key != current_ds:
        st.info("Enter inputs and click **Predict** on the Patient Input tab.")
    else:
        p_point = st.session_state.point_pred
        samples = np.array(st.session_state.unc_samples, dtype=float)

        st.metric("Predicted risk (point)", f"{p_point:.3f}")
        q025, q50, q975 = np.quantile(samples, [0.025, 0.5, 0.975])
        st.write(f"**Bootstrap 95% interval:** {q025:.3f} – {q975:.3f} (median {q50:.3f}, B={len(samples)})")

        figA = plot_interval_band(samples)
        st.pyplot(figA, use_container_width=False)
        bufA = io.BytesIO()
        figA.savefig(bufA, format="png", dpi=220, bbox_inches="tight")
        st.download_button("Download interval-band PNG", bufA.getvalue(), file_name="interval_band.png", mime="image/png")

        figB = plot_quantile_dotplot(samples, n_dots=50)
        st.pyplot(figB, use_container_width=False)
        bufB = io.BytesIO()
        figB.savefig(bufB, format="png", dpi=220, bbox_inches="tight")
        st.download_button("Download quantile-dotplot PNG", bufB.getvalue(), file_name="quantile_dotplot.png", mime="image/png")

        with st.expander("Show raw bootstrap samples (first 30)"):
            st.write([float(f"{x:.3f}") for x in samples[:30]])

        st.markdown("---")
        st.subheader("Calibration-aware threshold")

        # Stable slider value across reruns
        thresh = st.slider("Decision threshold", 0.0, 1.0, st.session_state.thresh_slider, 0.01, key="thresh_slider")

        proba_test = st.session_state.proba_test
        y_test_cur = st.session_state.y_test_cur
        if proba_test is None:
            st.info("Run **Predict** once to enable the threshold slider.")
        else:
            y_pred = (proba_test >= thresh).astype(int)
            tp = int(((y_test_cur == 1) & (y_pred == 1)).sum())
            fp = int(((y_test_cur == 0) & (y_pred == 1)).sum())
            tn = int(((y_test_cur == 0) & (y_pred == 0)).sum())
            fn = int(((y_test_cur == 1) & (y_pred == 0)).sum())
            st.write(f"**At threshold {thresh:.2f}:** TP={tp}, FP={fp}, TN={tn}, FN={fn}")

# ========================= Why (SHAP) =========================
with tab_why:
    st.subheader("Why this prediction? (Local SHAP)")
    if not st.session_state.pred_ready or st.session_state.ds_key != current_ds:
        st.info("Predict first, then come back to see SHAP for that patient.")
    else:
        input_row = st.session_state.input_row
        X_bg = X_train.sample(n=min(200, len(X_train)), random_state=0)
        loc = shap_explain_local(model, input_row, X_bg)

        vals = np.array(loc["shap_values"])
        feats = np.array(loc["feature_names"])
        k = min(10, len(vals), len(feats))
        order = np.argsort(-np.abs(vals))[:k]
        df_local = pd.DataFrame({"feature": feats[order], "shap_value": vals[order]})
        st.dataframe(df_local, use_container_width=True)
        st.caption("Positive SHAP → pushes risk up; Negative SHAP → pushes risk down.")

        # in app/streamlit_app.py, inside tab_why after the local table
        from src.shap_utils import shap_explain_global

        st.markdown("---")
        st.subheader("Which features matter overall? (Global SHAP)")
        df_global = shap_explain_global(model, X_train, max_display=12)
        st.dataframe(df_global, use_container_width=True)


# ========================= Model Quality =========================
with tab_quality:
    st.subheader("Model quality (ROC/PR + Reliability)")
    fig_roc, fig_pr = plot_roc_pr(y_test, model.predict_proba(X_test)[:, 1])
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_roc, use_container_width=True)
        buf_roc = io.BytesIO()
        fig_roc.savefig(buf_roc, format="png", dpi=220, bbox_inches="tight")
        st.download_button(
            "Download ROC PNG",
            buf_roc.getvalue(),
            file_name=f"{_ds_key(ds_name)}_roc.png",
            mime="image/png",
            key="dl_roc"
        )
    with col2:
        st.pyplot(fig_pr, use_container_width=True)
        buf_pr = io.BytesIO()
        fig_pr.savefig(buf_pr, format="png", dpi=220, bbox_inches="tight")
        st.download_button(
            "Download PR PNG",
            buf_pr.getvalue(),
            file_name=f"{_ds_key(ds_name)}_pr.png",
            mime="image/png",
            key="dl_pr"
        )

    ds_key = _ds_key(ds_name)
    rel_path = Path("results/metrics") / f"{ds_key}_logreg_reliability_isotonic.json"
    if rel_path.exists():
        fig_rel = plot_reliability_from_json(rel_path)
        st.pyplot(fig_rel, use_container_width=True)
        buf_rel = io.BytesIO()
        fig_rel.savefig(buf_rel, format="png", dpi=220, bbox_inches="tight")
        st.download_button(
            "Download Reliability PNG",
            buf_rel.getvalue(),
            file_name=f"{_ds_key(ds_name)}_reliability.png",
            mime="image/png",
            key="dl_rel"
        )

    else:
        st.info(f"Reliability data not found at {rel_path}. Run training first.")

    # in app/streamlit_app.py, near tab_quality bottom

    from src.failure_modes import top_failures

    st.markdown("---")
    st.caption("Sorted by distance from threshold (most confident mistakes first).")
    st.subheader("Failure modes (most confident mistakes)")
    proba_cur = model.predict_proba(X_test)[:, 1]
    fps, fns = top_failures(X_test, y_test, proba_cur, k=5)
    c1, c2 = st.columns(2)
    with c1: st.write("**Top False Positives**"); st.dataframe(fps.reset_index(drop=True))
    with c2: st.write("**Top False Negatives**"); st.dataframe(fns.reset_index(drop=True))

    # Optional: pick one row index to inspect with local SHAP
    idx = st.number_input("Inspect test index", min_value=0, max_value=len(X_test)-1, value=0, step=1)
    if st.button("Explain this test case"):
        X_bg = X_train.sample(n=min(200, len(X_train)), random_state=0)
        loc2 = shap_explain_local(model, X_test.iloc[[idx]], X_bg)
        vals2 = np.array(loc2["shap_values"]); feats2 = np.array(loc2["feature_names"])
        k2 = min(10, len(vals2), len(feats2))
        order2 = np.argsort(-np.abs(vals2))[:k2]
        st.dataframe(pd.DataFrame({"feature": feats2[order2], "shap_value": vals2[order2]}))
        

# ========================= Reflections =========================
with tab_reflect:
    # existing text...
    if st.session_state.latency_ms_list:
        st.markdown(f"**Last predict latency:** {st.session_state.last_predict_ts} ms")
        st.markdown(f"**Median latency (this session):** {int(np.median(st.session_state.latency_ms_list))} ms")

    st.markdown("---")
    st.subheader("Ethics Linter (quick checklist)")
    checks = [
        "Show uncertainty when appropriate (no naked point estimates)",
        "No truncated axes in charts",
        "Legible labels & units",
        "Color-blind friendly choices",
        "Explainability shown alongside risk (no dark patterns)",
        "Calibration assessed (reliability diagram present)"
    ]
    sel = {c: st.checkbox(c, value=True) for c in checks}

# with tab_reflect:
#     st.subheader("Design reflections")
#     st.write(
#         """
#         Use this space to jot observations as you test cases:
#         - When did the **interval band** feel clearer than the **dotplot** (and vice-versa)?
#         - Did calibration change your threshold decisions?
#         - Any confusing SHAP artifacts (e.g., one-hot features splitting importance)?
#         """
#     )


# from __future__ import annotations

# import sys
# from pathlib import Path
# ROOT = Path(__file__).resolve().parents[1]
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

# import streamlit as st
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from joblib import load


# from src.viz_uncertainty import plot_interval_band, plot_quantile_dotplot
# from src.viz_quality import plot_roc_pr, plot_reliability_from_json

# from src.load import load_heart, load_pima
# from src.bootstrap import bootstrap_predict
# from src.shap_utils import shap_explain_local

# def ss_get(key, default):
#     if key not in st.session_state:
#         st.session_state[key] = default
#     return st.session_state[key]

# st.set_page_config(page_title="HDI Uncertainty & Explainability", layout="wide")

# # ---------- Helpers ----------
# ART = Path("results/artifacts")

# DATASETS = {
#     "Heart (Cleveland)": {
#         "loader": load_heart,
#         "default_model": ART / "heart_logreg_isotonic.joblib",  # calibrated & interpretable
#     },
#     "Pima (Diabetes)": {
#         "loader": load_pima,
#         "default_model": ART / "pima_logreg_isotonic.joblib",
#     },
# }

# def _ds_key(name: str) -> str:
#     return "heart" if "Heart" in name else "pima"


# def _default_row_from_train(X_train: pd.DataFrame) -> pd.Series:
#     """Use median for numeric columns; mode for categoricals if present."""
#     s = {}
#     for c in X_train.columns:
#         if pd.api.types.is_numeric_dtype(X_train[c]):
#             s[c] = float(np.nanmedian(pd.to_numeric(X_train[c], errors="coerce")))
#         else:
#             s[c] = X_train[c].mode().iloc[0]
#     return pd.Series(s, index=X_train.columns)

# def _numeric_input(label, value, step=1.0, fmt=None, minv=None, maxv=None):
#     return st.number_input(label, value=float(value), step=step, format=fmt or "%0.2f", min_value=minv, max_value=maxv)

# def _int_input(label, value, minv=None, maxv=None):
#     return st.number_input(label, value=int(value), step=1, min_value=minv, max_value=maxv)

# # ---------- UI ----------
# st.title("HDI Uncertainty & Explainability – Phase 1 prototype")

# ds_name = st.sidebar.selectbox("Dataset", list(DATASETS.keys()))
# loader = DATASETS[ds_name]["loader"]
# model_path = DATASETS[ds_name]["default_model"]

# # Load data splits + preprocessor (used for background + bootstrap)
# X_train, X_val, X_test, y_train, y_val, y_test, pre = loader()

# # Load calibrated model (if missing, prompt user to run training)
# if not model_path.exists():
#     st.error(f"Model not found: {model_path}. Please run `python -m src.train` first.")
#     st.stop()
# model = load(model_path)

# # Stable per-run keys
# ss_get("pred_ready", False)          # whether we have a prediction computed
# ss_get("proba_test", None)           # cached test-set probabilities for current model
# ss_get("y_test_cur", None)           # cached y_test for current dataset
# ss_get("unc_samples", None)          # bootstrap samples for the last input_row
# ss_get("point_pred", None)           # last point prediction
# ss_get("input_row", None)            # last input row
# ss_get("ds_key", None)               # 'heart' or 'pima'


# # Tabs scaffold
# tab_input, tab_unc, tab_why, tab_quality, tab_reflect = st.tabs(
#     ["Patient Input", "Risk & Uncertainty", "Why (SHAP)", "Model Quality", "Reflections"]
# )

# # --- Patient Input ---
# # --- Patient Input ---
# input_row = pd.DataFrame([inputs], columns=X_train.columns)

# # Compute on click and store in session so sliders work after reruns
# if st.button("Predict"):
#     # Point prediction
#     p_point = float(model.predict_proba(input_row)[:, 1][0])

#     # Bootstrap uncertainty
#     from sklearn.linear_model import LogisticRegression
#     est = LogisticRegression(max_iter=200, class_weight="balanced", solver="liblinear")
#     unc = bootstrap_predict(X_train, y_train, pre, est, input_row, B=100)

#     # Cache test set probabilities for the threshold slider
#     st.session_state.proba_test = model.predict_proba(X_test)[:, 1]
#     st.session_state.y_test_cur = y_test
#     st.session_state.unc_samples = unc["all"]
#     st.session_state.point_pred = p_point
#     st.session_state.input_row = input_row
#     st.session_state.ds_key = "heart" if "Heart" in ds_name else "pima"
#     st.session_state.pred_ready = True

# st.caption("Tip: values are prefilled from training-set medians/modes for a quick demo.")


# with tab_input:
#     st.subheader("Enter patient features")
#     defaults = _default_row_from_train(X_train)

#     # Ensure inputs dict exists even before interaction
#     inputs = {}

#     # Build the input widgets safely
#     for col in X_train.columns:
#         # Check if it's integer-like and has no NaNs
#         col_is_int_like = pd.api.types.is_integer_dtype(X_train[col].dtype) and not X_train[col].isna().any()

#         if col_is_int_like:
#             start_val = int(round(defaults[col])) if pd.notna(defaults[col]) else 0
#             inputs[col] = _int_input(col, start_val, minv=0)
#         else:
#             start_val = float(defaults[col]) if pd.notna(defaults[col]) else 0.0
#             inputs[col] = _numeric_input(col, start_val, step=0.1)

#     # Build a 1-row DataFrame from the input dictionary
#     input_row = pd.DataFrame([inputs]).reindex(columns=X_train.columns)

#     st.caption("Tip: values are prefilled from training-set medians/modes for a quick demo.")

#     # --- Predict button logic ---
#     if st.button("Predict"):
#         from sklearn.linear_model import LogisticRegression

#         # Compute prediction
#         p_point = float(model.predict_proba(input_row)[:, 1][0])

#         # Bootstrap uncertainty
#         est = LogisticRegression(max_iter=200, class_weight="balanced", solver="liblinear")
#         unc = bootstrap_predict(X_train, y_train, pre, est, input_row, B=100)

#         # Cache everything for other tabs
#         st.session_state.point_pred = p_point
#         st.session_state.unc_samples = unc["all"]
#         st.session_state.input_row = input_row
#         st.session_state.proba_test = model.predict_proba(X_test)[:, 1]
#         st.session_state.y_test_cur = y_test
#         st.session_state.ds_key = "heart" if "Heart" in ds_name else "pima"
#         st.session_state.pred_ready = True

#     st.caption("Tip: values are prefilled from training-set medians/modes for a quick demo.")

# # with tab_input:
# #     st.subheader("Enter patient features")
# #     defaults = _default_row_from_train(X_train)

# #     inputs = {}
# #     for col in X_train.columns:
# #         # Safe “int-like” check: only use int widget if the column dtype is integer AND has no NaNs
# #         col_is_int_like = pd.api.types.is_integer_dtype(X_train[col].dtype) and not X_train[col].isna().any()

# #         # Use int widget for clean integer columns (e.g., Pregnancies, Age in many copies)
# #         if col_is_int_like:
# #             start_val = int(round(defaults[col])) if pd.notna(defaults[col]) else 0
# #             inputs[col] = _int_input(col, start_val, minv=0)  # optional minv for counts
# #         else:
# #             # Fallback to numeric widget; tolerate NaNs in training distribution
# #             start_val = float(defaults[col]) if pd.notna(defaults[col]) else 0.0
# #             inputs[col] = _numeric_input(col, start_val, step=0.1)

# #     input_row = pd.DataFrame([inputs], columns=X_train.columns)
# #     st.caption("Tip: values are prefilled from training-set medians/modes for a quick demo.")
# #     go = st.button("Predict")


# # --- Risk & Uncertainty ---

# with tab_unc:
#     st.subheader("Risk prediction with bootstrap uncertainty")

#     if not st.session_state.pred_ready:
#         st.info("Enter inputs and click **Predict** on the Patient Input tab.")
#     else:
#         p_point = st.session_state.point_pred
#         samples = np.array(st.session_state.unc_samples, dtype=float)

#         st.metric("Predicted risk (point)", f"{p_point:.3f}")
#         q025, q50, q975 = np.quantile(samples, [0.025, 0.5, 0.975])
#         st.write(f"**Bootstrap 95% interval:** {q025:.3f} – {q975:.3f} (median {q50:.3f}, B={len(samples)})")

#         # Charts
#         figA = plot_interval_band(samples)
#         st.pyplot(figA, use_container_width=False)
#         figB = plot_quantile_dotplot(samples, n_dots=50)
#         st.pyplot(figB, use_container_width=False)

#         with st.expander("Show raw bootstrap samples (first 30)"):
#             st.write([float(f"{x:.3f}") for x in samples[:30]])

#         st.markdown("---")
#         st.subheader("Calibration-aware threshold")

#         # Stable slider key so value persists across reruns
#         thresh = st.slider("Decision threshold", 0.0, 1.0, 0.50, 0.01, key="thresh_slider")

#         proba_test = st.session_state.proba_test
#         y_test_cur = st.session_state.y_test_cur
#         if proba_test is None:
#             st.info("Run **Predict** once to enable the threshold slider.")
#         else:
#             y_pred = (proba_test >= thresh).astype(int)
#             tp = int(((y_test_cur==1) & (y_pred==1)).sum())
#             fp = int(((y_test_cur==0) & (y_pred==1)).sum())
#             tn = int(((y_test_cur==0) & (y_pred==0)).sum())
#             fn = int(((y_test_cur==1) & (y_pred==0)).sum())
#             st.write(f"**At threshold {thresh:.2f}:** TP={tp}, FP={fp}, TN={tn}, FN={fn}")

# # with tab_unc:
# #     st.subheader("Risk prediction with bootstrap uncertainty")
# #     if not go:
# #         st.info("Enter inputs and click **Predict** on the Patient Input tab.")
# #         st.stop()

# #     # Point prediction (using the loaded calibrated model)
# #     p_point = float(model.predict_proba(input_row)[:, 1][0])

# #     # Bootstrap-based uncertainty (LogReg for speed; you can swap to RF if you like)
# #     from sklearn.linear_model import LogisticRegression
# #     est = LogisticRegression(max_iter=200, class_weight="balanced", solver="liblinear")
# #     unc = bootstrap_predict(X_train, y_train, pre, est, input_row, B=100)

# #     st.metric("Predicted risk (point)", f"{p_point:.3f}")
# #     st.write(
# #         f"**Bootstrap 95% interval:** {unc['q025']:.3f} – {unc['q975']:.3f}  "
# #         f"(median {unc['median']:.3f}, B={len(unc['all'])})"
# #     )

# #     # (We’ll add proper visuals in Step 5B: band + quantile dotplot)
# #     with st.expander("Show raw bootstrap samples (first 30)"):
# #         st.write([float(f"{x:.3f}") for x in unc["all"][:30]])

# #     import matplotlib.pyplot as plt
# #     import streamlit as st

# #     figA = plot_interval_band(np.array(unc["all"]))
# #     st.pyplot(figA, use_container_width=False)

# #     figB = plot_quantile_dotplot(np.array(unc["all"]), n_dots=50)
# #     st.pyplot(figB, use_container_width=False)
    

# #     st.markdown("---")
# #     st.subheader("Calibration-aware threshold")
# #     thresh = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
# #     proba_test = model.predict_proba(X_test)[:, 1]
# #     y_pred = (proba_test >= thresh).astype(int)

# #     tp = int(((y_test==1) & (y_pred==1)).sum())
# #     fp = int(((y_test==0) & (y_pred==1)).sum())
# #     tn = int(((y_test==0) & (y_pred==0)).sum())
# #     fn = int(((y_test==1) & (y_pred==0)).sum())

# #     st.write(f"**At threshold {thresh:.2f}:** TP={tp}, FP={fp}, TN={tn}, FN={fn}")


# # --- Why (SHAP) ---

# with tab_why:
#     st.subheader("Why this prediction? (Local SHAP)")
#     if not st.session_state.pred_ready:
#         st.info("Predict first, then come back to see SHAP for that patient.")
#     else:
#         input_row = st.session_state.input_row
#         X_bg = X_train.sample(n=min(200, len(X_train)), random_state=0)
#         loc = shap_explain_local(model, input_row, X_bg)

#         vals = np.array(loc["shap_values"])
#         feats = np.array(loc["feature_names"])
#         k = min(10, len(vals), len(feats))
#         order = np.argsort(-np.abs(vals))[:k]
#         df_local = pd.DataFrame({"feature": feats[order], "shap_value": vals[order]})
#         st.dataframe(df_local, use_container_width=True)
#         st.caption("Positive SHAP → pushes risk up; Negative SHAP → pushes risk down.")

# # with tab_why:
# #     st.subheader("Why this prediction? (Local SHAP)")
# #     # Use a small background sample for SHAP’s masker
# #     X_bg = X_train.sample(n=min(200, len(X_train)), random_state=0)
# #     loc = shap_explain_local(model, input_row, X_bg)

# #     # Show top 10 contributions by absolute value
# #     # vals = np.array(loc["shap_values"])
# #     # feats = np.array(loc["feature_names"])
# #     # order = np.argsort(-np.abs(vals))[:10]
# #     # df_local = pd.DataFrame({
# #     #     "feature": feats[order],
# #     #     "shap_value": vals[order],
# #     # })
# #     # st.dataframe(df_local, use_container_width=True)
# #     vals = np.array(loc["shap_values"])
# #     feats = np.array(loc["feature_names"])
# #     k = min(10, len(vals), len(feats))
# #     order = np.argsort(-np.abs(vals))[:k]
# #     df_local = pd.DataFrame({"feature": feats[order], "shap_value": vals[order]})
# #     st.dataframe(df_local, use_container_width=True)

# #     st.caption("Positive SHAP → pushes risk up; Negative SHAP → pushes risk down.")

# # --- Model Quality (placeholder for Step 5B visuals) ---
# with tab_quality:
#     st.subheader("Model quality (AUROC, Brier, reliability)")
#     # st.info("In Step 5B we’ll plot ROC/PR and a reliability diagram using saved metrics.")


#     # Current model’s ROC/PR from test split
#     fig_roc, fig_pr = plot_roc_pr(y_test, model.predict_proba(X_test)[:, 1])
#     col1, col2 = st.columns(2)
#     with col1: st.pyplot(fig_roc, use_container_width=True)
#     with col2: st.pyplot(fig_pr, use_container_width=True)

#     # Reliability diagram from saved metrics JSON
#     ds_key = _ds_key(ds_name)               # 'heart' or 'pima'
#     rel_path = Path("results/metrics") / f"{ds_key}_logreg_reliability_isotonic.json"
#     if rel_path.exists():
#         fig_rel = plot_reliability_from_json(rel_path)
#         st.pyplot(fig_rel, use_container_width=True)
#     else:
#         st.info(f"Reliability data not found at {rel_path}. Run training first.")


# # --- Reflections (notes) ---
# with tab_reflect:
#     st.subheader("Design reflections")
#     st.write(
#         """
#         Use this space to jot observations as you test cases:
#         - When did the **interval band** feel clearer than the **dotplot** (and vice-versa)?
#         - Did calibration change your threshold decisions?
#         - Any confusing SHAP artifacts (e.g., one-hot features splitting importance)?
#         """
#     )
