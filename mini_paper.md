# Design Exploration of Uncertainty & Explainability in Clinical Risk Dashboards

**Author:** Fnu Gaurav  
**Date:** {{today}}  
**Note:** Design study; not a clinical tool.

## 1. Motivation
Clinical risk scores are often shown as single numbers. We explore two uncertainty encodings (interval band, quantile dotplot) and pair them with explainability (local/global SHAP) and calibration views to support more honest, interpretable decisions.

## 2. Approach
- **Datasets:** UCI Heart (Cleveland), Pima Indians Diabetes.
- **Models:** Logistic Regression baseline + Isotonic calibration (saved artifacts); Random Forest for sensitivity.
- **Uncertainty:** Bootstrap on patient input (B=100 by default).
- **Explainability:** Local (per-patient) + Global (dataset-level) SHAP.
- **Quality:** ROC/PR; reliability (10-bin diagram).
- **App:** Streamlit tabs; dataset toggle; calibration-aware threshold slider.

## 3. Visual Designs
### 3.1 Encoding A — Interval band (median + 95% CI)
_Insert FIG A (Heart case)_: one line band with a median tick; quick to parse.

### 3.2 Encoding B — Quantile dotplot
_Insert FIG B (Pima case)_: 50 quantile dots; shows skew/multimodality.

### 3.3 Explainability
- _Local SHAP table (Heart case)_: top 10 contributions (+ raises risk, − lowers).
- _Global SHAP table_: dominant drivers across the dataset.

### 3.4 Model Quality
- _ROC, PR_ for the test set.
- _Reliability diagram_ to assess calibration.

## 4. Case Observations
- **Heart-L:** Low point risk with **wide** interval → band communicates uncertainty succinctly; dotplot shows tail mass.
- **Heart-M:** Mid risk where **oldpeak** and **cp** drive increases (local SHAP aligns with expectations).
- **Pima-L:** Glucose low-variance case → **tight** interval; dotplot ~uniform around median.
- **Pima-Err:** Confident FN surfaced by failure-modes table; local SHAP reveals atypical feature mix.

## 5. Design Implications
- Bands are **fast**; dotplots reveal **distribution shape** (skew/bimodality).
- Pair uncertainty with **explanations** to support trust.
- **Calibrated** thresholds matter; the slider exposes the precision-recall tradeoff.
- Surfacing **failure modes** helps reason about model limits.

## 6. Ethics Notes
Checklist: show uncertainty; avoid truncated axes; readable labels; color-blind friendly defaults; display calibration; note dataset limits; not for clinical use.

## 7. Limitations & Next Steps
Small datasets; no user study; bootstrap ignores aleatoric label noise. Next: cost-aware thresholds, decision curves, clinician walkthroughs, prospective validation.

## 8. Appendix
Environment manifest (`results/metrics/_versions.json`), training details, hyperparameters, and preprocessing schema.
