# CareMP Fall Risk Prediction (V2)

This repository contains the **Version 2** machine learning pipeline for the CareMP Fall Risk prediction system. It ingests physiological smartband data from a PostgreSQL database, engineers rolling time-series features, and uses an XGBoost model layered with a hardcoded rule engine and trend engine to output clinical fall-risk assessments.

## Core Features (V2 Enhancements)
- **Multi-Domain Logic:** Evaluates risk across Vitals, Recovery, Mobility, and Sleep domains to prevent isolated sensor failures (e.g., band removal) from causing false alarms.
- **6-Hour Rapid Trend:** Detects rapid physiological deterioration alongside the standard 24-hour baseline trend.
- **Explainability:** CareGPT dynamically generates a natural-language clinical narrative based on SHAP risk drivers and trend severity.
- **Data Quality Gating:** The system enforces coverage thresholds and explicitly flags missing data (like missing sleep confidence) to prevent hallucinated predictions.

## Key Files & Modules

| File | Purpose |
|------|---------|
| `data_preprocessing1.py` | Extracts data from PostgreSQL (`band_log`) and engineers 105+ rolling-window physiological features and deviation ratios against personal baselines. |
| `train_v2_grouped_real_holdout.py` | Training script for the XGBoost ML model (`v23_final`) using a grouped holdout validation approach. |
| `predict_fall_risk_v1.py` | Core inference pipeline. Receives a resident ID, calls the preprocessor, and runs the ML model to output raw class probabilities. |
| `rolling_rule_engine_v2.py` | Immediate clinical safety net. Applies hardcoded rules (e.g., Tachycardia, SpO2 desaturation) that can override or explain the ML model's output. |
| `trend_engine_v2.py` | Evaluates historical risk scores (24h and 6h windows) to determine if a resident is `Improving`, `Worsening`, or `Stable`. |
| `fall_risk_layered_pipeline_v2.py` | The master orchestrator. Calls the ML model, Rule Engine, and Trend Engine, passes the results to CareGPT, and outputs the final `fall_risk_public_output_v2.json`. |



## How to Run

1. Ensure the PostgreSQL database connection string is properly configured in `env1.env`.
2. Ensure you have the required Python packages (`pandas`, `xgboost`, `scikit-learn`, `joblib`, `psycopg2`).
3. To generate a risk assessment for a specific resident (e.g., ID 15):
   ```bash
   python fall_risk_layered_pipeline_v2.py --resident-id 15
   ```
4. The final clinical output will be saved as `fall_risk_public_output_v2.json`.

##  Validation & Testing
You can run synthetic validation scenarios (like testing band-removal behavior) using:
```bash
python scratch/validate_multidomain.py
```
This script validates that the model correctly suppresses risk escalation when mobility drops but other vitals remain normal.
