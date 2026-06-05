# Fall Risk V1 Layered Pipeline

This document explains the current V1 fall-risk pipeline, what each script does, what files are produced, and how to execute the workflow.

The V1 pipeline is designed as a pilot architecture for real resident wearable data. It combines raw safety checks, 30-minute rolling features, XGBoost prediction, SHAP explainability, prediction history, trend analysis, resident clinical context, and caregiver-facing CareGPT output.

## 1. High-Level Flow

```text
PostgreSQL band_log
        +
Engineering API / SQLite resident cache
        ↓
30-minute feature generation
        ↓
V1 XGBoost prediction
        ↓
SHAP explanation
        ↓
Rule layer + final risk resolver
        ↓
SQLite prediction history
        ↓
24h / 7d trend engine
        ↓
CareGPT final insight
        ↓
Public JSON for UI / API
```

The pipeline intentionally separates engineering diagnostics from caregiver-facing output.

Use this for caregivers/UI:

```text
fall_risk_public_output_v1.json
```

Use this for debugging:

```text
fall_risk_layered_pipeline_v1.json
```

## 2. Main Scripts

### `predict_fall_risk_v1.py`

Builds V1 prediction features from database data and runs the saved XGBoost model.

Responsibilities:

- Fetches post-training BandLog rows from PostgreSQL.
- Fetches resident baselines from Engineering API / SQLite cache / database fallback.
- Builds 30-minute rolling features.
- Runs V1 XGBoost predictions.
- Optionally writes prediction history into SQLite.

Important output:

```text
fall_risk_prediction_history.db
fall_risk_predictions_v1.csv
```

### `trend_engine_v1.py`

Reads prediction history from SQLite and computes resident-level trends.

Responsibilities:

- Computes 24-hour trend.
- Falls back to 7-day trend if 24h is insufficient.
- Produces trend direction and severity:
  - `Strongly Improving`
  - `Improving`
  - `Stable`
  - `Deteriorating`
  - `Strongly Deteriorating`

Outputs:

```text
fall_risk_trend_summary_v1.csv
fall_risk_trend_summary_7d_v1.csv
```

### `fall_risk_layered_pipeline_v1.py`

This is the main combined pipeline.

Responsibilities:

- Fetches latest resident BandLog data.
- Builds 30-minute V1 feature rows.
- Runs V1 model prediction.
- Adds SHAP explainability.
- Applies raw safety and rolling rule checks.
- Loads trend context from SQLite prediction history.
- Resolves final risk.
- Adds resident context from API/cache:
  - age from DOB
  - medical history
  - condition
- Generates CareGPT final insight.
- Writes both full debug JSON and clean public JSON.

Outputs:

```text
fall_risk_layered_pipeline_v1.json
fall_risk_public_output_v1.json
fall_risk_layered_pipeline_v1_summary.csv
```

### `simulate_layered_pipeline_synthetic.py`

Runs the same layered pipeline using synthetic feature values for residents already present in prediction history.

Use this for testing scenarios without touching real history or database records.

Supported scenarios:

```text
normal
sitting
elevated
high
recovery
mixed
```

Outputs:

```text
synthetic_layered_pipeline_test.json
synthetic_public_layered_pipeline_test.json
synthetic_layered_pipeline_test_summary.csv
```

### `resident_context_cache.py`

Fetches and prepares resident baseline/context data.

Sources:

- Engineering API
- SQLite resident cache
- database fallback

It computes:

- resident age from DOB
- baseline vitals
- tolerances
- medical context from `medicalHistory` and `condition`

Notes and special notes are intentionally not used in the current pipeline.

### `rolling_rule_engine_v1.py`

Contains the raw safety and rolling feature rule checks used by the layered pipeline.

It checks:

- raw safety vitals
- 30-minute HR, HRV, SpO2, BP
- 30-minute activity
- fatigue/stress
- daily sleep deficit

## 3. Current Feature Groups

### V1 Model Features

The trained model uses these feature groups:

```text
Heart:
  avg_hr
  max_hr
  min_hr
  hr_std

HRV:
  avg_hrv
  hrv_availability

SpO2:
  avg_spo2
  min_spo2
  spo2_availability

Blood pressure:
  avg_sbp
  avg_dbp

Temperature:
  avg_temp

Stress / fatigue / breathing:
  avg_stress
  avg_fatigue
  avg_breathing

Activity:
  steps_30m
  step_std
  activity_ratio

Sleep:
  daily_total_sleep_minutes
  sleep_baseline_minutes
  sleep_ratio
  sleep_deficit
```

Age, medical history, condition, care notes, and allergies are not V1 model features.

## 4. How Step Data Is Used

Raw `stepCount` is converted into step increments and then 30-minute features.

```text
raw stepCount
→ step increment
→ rolling 30-minute step sum
→ steps_30m
→ activity_ratio
```

The daily reference step count is converted into an approximate active 30-minute baseline:

```text
reference_steps_30m = reference_step_count / 16
activity_ratio = steps_30m / reference_steps_30m
```

This means low activity in one 30-minute window is only a mild signal unless it appears with abnormal vitals, poor sleep, high fatigue, or worsening trend.

## 5. How Sleep Is Used

Sleep is treated as daily recovery context.

Current V1 uses:

```text
daily_total_sleep_minutes
sleep_baseline_minutes
sleep_ratio
sleep_deficit
sleep_data_available
```

If sleep is missing, it is not scored as a sleep risk driver. Missing sleep is kept as internal data quality information and is not shown in the public CareGPT output.

## 6. How HRV and SpO2 Are Used

HRV and SpO2 are considered, but cautiously.

They can appear in three places:

```text
1. Rule risk drivers
2. V1 model features
3. SHAP explainability
```

Rules:

- HRV `0` is treated as invalid/missing.
- SpO2 is used only when `oxygenSaturationValid = true`.
- Low HRV/SpO2 availability is treated as reliability metadata, not as an abnormal clinical finding.
- CareGPT mentions HRV or SpO2 only when the values are clearly abnormal.

Examples:

```text
HRV mentioned if avg_hrv < reference_hrv - hrv_tolerance
SpO2 mentioned if min_spo2 < 92
SpO2 mentioned if avg_spo2 < reference_spo2 - oxygen_saturation_tolerance
```

## 7. SHAP Explainability

SHAP is added as a V1 explainability layer.

Output location:

```text
modelExplainabilityLayer
```

Example:

```json
{
  "available": true,
  "method": "SHAP TreeExplainer",
  "explainedClass": "Moderate",
  "topPositiveDrivers": [
    {
      "feature": "steps_30m",
      "displayName": "30-minute step count",
      "impact": 0.462441,
      "value": 14.0
    }
  ]
}
```

CareGPT does not say “SHAP” to caregivers. It only uses translated supporting patterns when safe.

## 8. Resident Context

The pipeline uses resident context from the Engineering API/cache.

Currently used:

```text
dateOfBirth → resident_age
medicalHistory
condition
```

Not used:

```text
notes
specialNotes
allergies
```

Medical context is used only for explanation, not model prediction.

Example extracted context:

```text
walking assistance or mobility difficulty
diabetes
blood-pressure history
memory or cognitive concerns
recent surgery recovery
respiratory history
```

## 9. Public Output vs Debug Output

### Public Output

Use this file for UI/API/CareGPT:

```text
fall_risk_public_output_v1.json
```

It contains:

```json
{
  "residentId": 15,
  "generatedAt": "...",
  "finalRisk": "Moderate",
  "confidence": {
    "model": 0.93
  },
  "finalInsight": "...",
  "recommendations": [],
  "trend": {
    "window": "24-hour",
    "direction": "improving",
    "severity": "Strongly Improving"
  },
  "topObservedDrivers": [],
  "supportingPatterns": [],
  "clinicalContext": {
    "age": 31,
    "ageGroup": "adult",
    "phrases": ["diabetes"]
  }
}
```

### Debug Output

Use this file for engineering:

```text
fall_risk_layered_pipeline_v1.json
```

It contains:

```text
rawSafetyLayer
aggregationLayer
dataQualityLayer
ruleEngineLayer
v1ModelLayer
modelExplainabilityLayer
resolvedRiskLayer
predictionStoreLayer
trendLayer
careGptLayer
```

Do not expose the full debug file directly to caregivers.

## 10. Execution Steps

Open PowerShell in the project folder:

```powershell
cd "C:\Users\Darsana N\Desktop\CareMP\caremp-fall-risk"
```

Use the local virtual environment:

```powershell
.\venv\Scripts\python.exe
```

### Step 1: Populate or update prediction history

Run this when new post-training BandLog data is available:

```powershell
.\venv\Scripts\python.exe predict_fall_risk_v1.py --mode history
```

This writes predictions into:

```text
fall_risk_prediction_history.db
```

If you do not want to write history, do not run this step.

### Step 2: Generate trend summaries

```powershell
.\venv\Scripts\python.exe trend_engine_v1.py
```

Outputs:

```text
fall_risk_trend_summary_v1.csv
fall_risk_trend_summary_7d_v1.csv
```

### Step 3: Run full layered pipeline

For all residents:

```powershell
.\venv\Scripts\python.exe fall_risk_layered_pipeline_v1.py --no-write-history
```

For one resident:

```powershell
.\venv\Scripts\python.exe fall_risk_layered_pipeline_v1.py --resident-id 15 --no-write-history
```

With SHAP enabled:

```powershell
.\venv\Scripts\python.exe fall_risk_layered_pipeline_v1.py --resident-id 15 --no-write-history --shap-top-n 5
```

Without SHAP:

```powershell
.\venv\Scripts\python.exe fall_risk_layered_pipeline_v1.py --resident-id 15 --no-write-history --disable-shap
```

Main outputs:

```text
fall_risk_layered_pipeline_v1.json
fall_risk_public_output_v1.json
fall_risk_layered_pipeline_v1_summary.csv
```

### Step 4: Run synthetic tests

All residents with mixed scenarios:

```powershell
.\venv\Scripts\python.exe simulate_layered_pipeline_synthetic.py
```

One resident, high-risk synthetic case:

```powershell
.\venv\Scripts\python.exe simulate_layered_pipeline_synthetic.py --resident-id 15 --scenario high
```

One resident, sitting case:

```powershell
.\venv\Scripts\python.exe simulate_layered_pipeline_synthetic.py --resident-id 15 --scenario sitting
```

Without SHAP:

```powershell
.\venv\Scripts\python.exe simulate_layered_pipeline_synthetic.py --disable-shap
```

Synthetic outputs:

```text
synthetic_layered_pipeline_test.json
synthetic_public_layered_pipeline_test.json
synthetic_layered_pipeline_test_summary.csv
```

## 11. Recommended Daily/Pilot Workflow

If new data arrives:

```powershell
.\venv\Scripts\python.exe predict_fall_risk_v1.py --mode history
.\venv\Scripts\python.exe trend_engine_v1.py
.\venv\Scripts\python.exe fall_risk_layered_pipeline_v1.py --no-write-history
```

If no new data arrived and you only want current public output:

```powershell
.\venv\Scripts\python.exe fall_risk_layered_pipeline_v1.py --no-write-history
```

If testing behavior:

```powershell
.\venv\Scripts\python.exe simulate_layered_pipeline_synthetic.py --scenario mixed
```

## 12. Important Notes

### Training data is not reused for live prediction

The prediction script excludes the V1 training period:

```text
2026-04-10T00:00:00+00:00
to
2026-05-01T23:59:59+00:00
```

This avoids mixing training rows into live prediction/history.

### Trend can differ from current risk

It is normal to see:

```text
finalRisk = High
trend = stable
```

This means the resident is currently high risk, but the historical trend is not rapidly worsening.

It is also possible in synthetic testing to see:

```text
finalRisk = Critical
trend = improving
```

because the synthetic row changes only the current feature row. It does not rewrite SQLite history.

### Care notes are not used

Care notes, notes, and special notes are not used in the current pipeline because they may be stale, language-dependent, and not consistently updated.

Medical history and condition are used only as stable clinical context.

### Allergies are not used

Allergy is not a fall-risk model feature. It may be useful for medication safety, but it is not used in V1 fall-risk prediction.

### Sklearn warning

You may see:

```text
InconsistentVersionWarning
```

This happens because the saved label encoder was created with a different sklearn version. It has not blocked pilot execution, but production should pin compatible package versions.

## 13. Current Limitations

Known limitations:

- Resident count is still small.
- Labels are rule-based, not outcome/fall-event labels.
- Sleep can be missing for many rows.
- Posture/activity state is not available, so sitting/resting cannot be perfectly separated from inactivity.
- Care notes are not used.
- Medical history is context-only.
- SHAP explains the model prediction, not the entire final resolved risk.

## 14. V2 Direction

V2 should add only carefully controlled improvements:

```text
REM sleep
sleep confidence
sleep computation state
sleep data reliability
structured medical-condition flags
possibly limited care-note flags if notes become timestamp-reliable
better trend features
stable prediction frequency
```

V2 should not directly train on raw care-note text until note quality, language handling, and timestamp reliability are validated.
