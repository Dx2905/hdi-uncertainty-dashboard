# src/select_cases.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from sklearn.linear_model import LogisticRegression

from src.load import load_heart, load_pima
from src.bootstrap import bootstrap_predict

ART = Path("results/artifacts")

# Selection targets (tweak if a bucket comes back empty)
HI_RISK = 0.70
LO_RISK = 0.20
MID_LO, MID_HI = 0.40, 0.60
UNC_TIGHT = 0.20     # width <= 0.20 → “low uncertainty”
UNC_WIDE  = 0.35     # width >= 0.35 → “high uncertainty”
B = 100
MAX_ROWS_FOR_BOOT = 150  # speed guard

def eval_dataset(ds_key: str, loader, model_path: Path):
    print(f"\n=== {ds_key.upper()} ===")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    # Load honest splits + preprocessor
    X_train, X_val, X_test, y_train, y_val, y_test, pre = loader()

    # 1) Score candidates using the calibrated LR pipeline (exactly what the app uses)
    model = load(model_path)  # e.g., heart_logreg_isotonic.joblib
    X_pool = X_test.copy()
    proba = model.predict_proba(X_pool)[:, 1]
    pool = X_pool.reset_index(drop=True).copy()
    pool["_proba"] = proba

    # 2) For bootstrap, use a plain LR (same as app's uncertainty tab)
    est_boot = LogisticRegression(
        max_iter=200, class_weight="balanced", solver="liblinear", random_state=42
    )

    def add_uncertainty(rows: pd.DataFrame) -> pd.DataFrame:
        rows = rows.copy()
        qlo, qhi, qmed, width = [], [], [], []
        for i in range(len(rows)):
            r = rows.iloc[[i]]  # keep as DataFrame with column names
            unc = bootstrap_predict(X_train, y_train, pre, est_boot, r, B=B, random_state=123)
            qlo.append(unc["q025"]); qhi.append(unc["q975"]); qmed.append(unc["median"])
            width.append(unc["q975"] - unc["q025"])
        rows["_q025"] = qlo; rows["_q975"] = qhi; rows["_qmed"] = qmed; rows["_qwidth"] = width
        return rows

    # Candidate pools
    hi_cand  = pool.sort_values("_proba", ascending=False).head(MAX_ROWS_FOR_BOOT//3)
    lo_cand  = pool.sort_values("_proba", ascending=True).head(MAX_ROWS_FOR_BOOT//3)
    mid_band = pool[(pool["_proba"] >= MID_LO) & (pool["_proba"] <= MID_HI)]
    mid_cand = mid_band.sample(n=min(len(mid_band), MAX_ROWS_FOR_BOOT//3), random_state=42) if len(mid_band) else mid_band

    # Compute uncertainty on candidates only (fast)
    hi_eval  = add_uncertainty(hi_cand)  if len(hi_cand)  else pd.DataFrame()
    lo_eval  = add_uncertainty(lo_cand)  if len(lo_cand)  else pd.DataFrame()
    mid_eval = add_uncertainty(mid_cand) if len(mid_cand) else pd.DataFrame()

    # Pick final rows by rules
    hi_pick  = hi_eval[(hi_eval["_proba"] >= HI_RISK) & (hi_eval["_qwidth"] <= UNC_TIGHT)].head(1)
    lo_pick  = lo_eval[(lo_eval["_proba"] <= LO_RISK) & (lo_eval["_qwidth"] <= UNC_TIGHT)].head(1)
    mid_pick = mid_eval[(mid_eval["_proba"].between(MID_LO, MID_HI)) & (mid_eval["_qwidth"] >= UNC_WIDE)].head(1)

    # Graceful fallbacks if strict bucket empty
    if hi_pick.empty and not hi_eval.empty:
        hi_pick = hi_eval.sort_values(["_proba","_qwidth"], ascending=[False, True]).head(1)
    if lo_pick.empty and not lo_eval.empty:
        lo_pick = lo_eval.sort_values(["_proba","_qwidth"], ascending=[True, True]).head(1)
    if mid_pick.empty and not mid_eval.empty:
        mid_pick = mid_eval.sort_values("_qwidth", ascending=False).head(1)

    def show(tag, df):
        if df.empty:
            print(f"{tag}: NOT FOUND (consider relaxing thresholds)")
            return
        row = df.iloc[0]
        print(f"\n{tag}: proba={row['_proba']:.3f} | 95%=[{row['_q025']:.3f}, {row['_q975']:.3f}] | width={row['_qwidth']:.3f}")
        # print feature values
        feat_cols = [c for c in df.columns if not c.startswith("_")]
        for k, v in df[feat_cols].iloc[0].to_dict().items():
            print(f"  {k}: {v}")

    show("HIGH (tight)", hi_pick)
    show("LOW  (tight)", lo_pick)
    show("MID  (wide) ", mid_pick)

def main():
    eval_dataset("heart", load_heart, ART / "heart_logreg_isotonic.joblib")
    eval_dataset("pima",  load_pima,  ART / "pima_logreg_isotonic.joblib")

if __name__ == "__main__":
    main()
