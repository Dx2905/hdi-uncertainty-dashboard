# HDI Uncertainty & Explainability Dashboard

Design exploration with two uncertainty encodings (interval band, quantile dotplot),
local/global SHAP explanations, calibrated thresholds, and model quality views.

## Quick start
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
streamlit run app/streamlit_app.py

## Notes
- Not a clinical tool. Findings are qualitative design observations.
- Artifacts + environment manifest written to results/.
