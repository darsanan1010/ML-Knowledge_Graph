# Fall Risk Baseline Model V1 Documentation

## Purpose

This document explains how the current fall-risk training dataset is generated, how each available field is used, and how the baseline XGBoost model was trained and evaluated.

The current model should be treated as a baseline risk classifier. It predicts the rule-generated risk category using real band log history and resident baseline context.

## Data Sources

The training dataset is generated from two sources:

- PostgreSQL `band_log`
- Engineering API resident context

The PostgreSQL `band_log` table provides time-series wearable data for each resident.

The Engineering API provides resident-specific reference and tolerance values, such as heart rate baseline, HRV baseline, BP baseline, SpO2 baseline, temperature baseline, step reference, and sleep reference values.

Only the last 3 months of `band_log` data are extracted for the current dataset.

## Raw Band Log Fields Used

The following fields are extracted from `band_log`:

| Raw Field | Normalized Field | Usage |
|---|---|---|
| `residentId` | `resident_id` | Resident join key and grouping key |
| `generatedAt` | `generated_at` | Timestamp for rolling windows and ordering |
| `heartRate` | `heart_rate` | Heart rate rolling features |
| `hrv` | `hrv` | HRV rolling feature, cleaned before use |
| `oxygenSaturation` | `oxygen_saturation` | SpO2 feature when valid |
| `oxygenSaturationValid` | `oxygen_saturation_valid` | Filters invalid SpO2 values |
| `systolicBP` | `systolic_bp` | BP rolling feature |
| `diastolicBP` | `diastolic_bp` | BP rolling feature |
| `bodyTemperature` | `body_temperature` | Temperature rolling feature |
| `stepCount` | `step_count` | Converted into step increments |
| `deepSleepTime` | `deep_sleep_time` | Daily sleep context |
| `lightSleepTime` | `light_sleep_time` | Daily sleep context |
| `fatigueLevel` | `fatigue_level` | Rolling fatigue feature |
| `stress` | `stress` | Rolling stress feature |
| `breathing` | `breathing` | Rolling breathing feature |

## Resident Context Fields Used

The following resident context fields are fetched from the Engineering API:

| Field | Usage |
|---|---|
| `referenceHeartRate` | Heart rate baseline |
| `heartRateTolerance` | Heart rate tolerance for risk scoring |
| `referenceHrv` / `referenceHRV` | HRV baseline |
| `hrvTolerance` | HRV tolerance for risk scoring |
| `referenceSystolicBP` | Systolic BP baseline |
| `systolicBPTolerance` | Systolic BP tolerance |
| `referenceDiastolicBP` | Diastolic BP baseline |
| `diastolicBPTolerance` | Diastolic BP tolerance |
| `referenceOxygenSaturation` | SpO2 baseline |
| `oxygenSaturationTolerance` | SpO2 tolerance |
| `referenceBodyTemperature` | Temperature baseline |
| `temperatureToleranceCelsius` | Temperature tolerance |
| `referenceStepCount` | Daily step reference |
| `stepCountTolerance` | Daily step tolerance |
| `referenceDeepSleep` | Deep sleep reference |
| `referenceLightSleep` | Light sleep reference |

If the Engineering API is unavailable, the script falls back to available reference values from `resident_vitals`.

## Data Cleaning

Before feature generation, the script applies these cleaning rules:

- `resident_id` is converted to numeric type.
- Duplicate rows with the same `resident_id` and `generated_at` are dropped.
- HRV value `0` is treated as unavailable and replaced with missing.
- SpO2 is used only when `oxygenSaturationValid` is true.
- Fatigue values outside the valid range are removed.
- Stress values outside `0-100` are removed.
- Stress and fatigue are clipped to `0-100`.
- Missing HRV and SpO2 values are later imputed using resident reference values.
- Missing stress and fatigue are imputed using resident median, then global median.

## Step Handling

The raw `stepCount` field is cumulative during the day, so the script converts it into increments.

For each resident:

- Normal same-day increase: `step_increment = current_step_count - previous_step_count`
- Midnight/day reset: use current `step_count` as the new day's increment
- Same-day negative difference: treated as device reset/reboot and set to `0`
- Step increments are clipped to avoid unrealistic spikes

The model uses:

| Feature | Meaning |
|---|---|
| `steps_30m` | Sum of step increments over the last 30 minutes |
| `step_std` | Variability of step increments over the last 30 minutes |
| `activity_ratio` | Current 30-minute activity compared with expected resident activity |

The API step reference is daily, so it is converted into a 30-minute active-window reference:

```text
reference_steps_30m = reference_step_count / 16
```

Here, `16` means 16 active 30-minute windows, equivalent to 8 active hours per day.

Then:

```text
activity_ratio = steps_30m / reference_steps_30m
```

This lets the model interpret vitals in activity context. For example, high heart rate during high activity is different from high heart rate during low activity.

## Sleep Handling

The current sleep fields are:

- `deepSleepTime`
- `lightSleepTime`

These are treated as minutes directly. No multiplication is applied.

The script computes:

```text
total_sleep_minutes = deepSleepTime + lightSleepTime
```

Sleep values are cumulative during the day and reset at 12. Therefore, daily sleep is calculated using the maximum value for the resident and date:

```text
daily_total_sleep_minutes = max(total_sleep_minutes) per resident per day
```

The script also computes a resident-specific sleep baseline:

```text
sleep_baseline_minutes = median available daily sleep for that resident
```

If a resident has no usable sleep data, the script falls back to the API/reference sleep value.

For missing sleep days, the resident is not penalized. Instead:

```text
daily_total_sleep_for_ratio = sleep_baseline_minutes
sleep_ratio = 1
sleep_deficit = 0
```

This prevents missing sleep data from creating false high-risk labels.

Sleep features used by the model:

| Feature | Meaning |
|---|---|
| `daily_total_sleep_minutes` | Daily sleep total from band data |
| `sleep_baseline_minutes` | Resident-specific median sleep baseline |
| `sleep_ratio` | Sleep compared with resident baseline |
| `sleep_deficit` | Sleep shortage compared with baseline |

## Rolling Feature Generation

Most physiological and activity features are calculated using a resident-wise 30-minute rolling window.

The current model uses the following training features:

| Feature | Description |
|---|---|
| `avg_hr` | Average heart rate over 30 minutes |
| `max_hr` | Maximum heart rate over 30 minutes |
| `min_hr` | Minimum heart rate over 30 minutes |
| `hr_std` | Heart rate variability within the 30-minute window |
| `avg_hrv` | Average HRV over 30 minutes |
| `avg_spo2` | Average valid SpO2 over 30 minutes |
| `min_spo2` | Minimum valid SpO2 over 30 minutes |
| `avg_sbp` | Average systolic BP over 30 minutes |
| `avg_dbp` | Average diastolic BP over 30 minutes |
| `avg_temp` | Average body temperature over 30 minutes |
| `avg_stress` | Average stress over 30 minutes |
| `avg_fatigue` | Average fatigue over 30 minutes |
| `avg_breathing` | Average breathing value over 30 minutes |
| `steps_30m` | Total step increment over 30 minutes |
| `step_std` | Step increment variability over 30 minutes |
| `activity_ratio` | 30-minute activity relative to expected activity |
| `daily_total_sleep_minutes` | Daily sleep context |
| `sleep_baseline_minutes` | Resident sleep baseline |
| `sleep_ratio` | Sleep compared with baseline |
| `sleep_deficit` | Sleep deficit compared with baseline |
| `hrv_availability` | HRV availability in the rolling window |
| `spo2_availability` | SpO2 availability in the rolling window |

## Quality Features

The script computes signal availability metrics for HRV and SpO2:

```text
hrv_availability = valid HRV count / total HRV count in 30 minutes
spo2_availability = valid SpO2 count / total SpO2 count in 30 minutes
```

These features help the model distinguish true physiological changes from poor device signal availability.

## Audit and Label Generation Columns

The dataset also contains audit columns that explain how the rule-based risk label was generated.

Examples:

- `hr_dev`
- `hrv_dev`
- `spo2_dev`
- `sbp_dev`
- `dbp_dev`
- `temp_dev`
- `step_dev`
- `hr_breach`
- `hrv_breach`
- `spo2_breach`
- `sbp_breach`
- `dbp_breach`
- `temp_breach`
- `step_breach`
- `weighted_hr_breach`
- `weighted_hrv_breach`
- `weighted_sbp_breach`
- `weighted_dbp_breach`
- `weighted_spo2_breach`
- `weighted_step_breach`
- `risk_score`

These columns are useful for explainability and alert reasoning, but they are not used as model input for the baseline model.

This separation avoids training the model directly on the same rule-engine outputs used to create the target label.

## Risk Label Generation

The training label is generated from a rule-based `risk_score`.

The score combines:

- Heart rate breach
- HRV breach
- Systolic BP breach
- Diastolic BP breach
- SpO2 breach
- Step breach
- Fatigue component
- Stress component
- Sleep deficit component
- Short-term slope components

Some breach values are weighted by activity context. For example:

```text
weighted_hr_breach = hr_breach * activity_factor
```

This means a high heart rate while inactive can be considered more concerning than high heart rate during activity.



Contextual Weighting

The model distinguishes between physiological changes occurring during activity and physiological changes occurring during inactivity.

Example:

Heart Rate = 120

Case A:
Activity Ratio = 1.5
(Resident actively walking)

Case B:
Activity Ratio = 0.1
(Resident inactive)

The same heart rate may have different clinical significance depending on activity context.

For this reason activity context is incorporated into risk generation through activity-aware weighting.

Risk labels are generated as:

```text
Low
Moderate
High
Critical
```

Important: these are rule-generated labels, not confirmed fall-event labels. The baseline model learns the current clinical/rule-based risk logic.

## Class Weights

Class weights are calculated from label frequency:

```text
class_weight = total rows / (number of classes * class count)
```

Weights are capped to avoid unstable training:

```text
MAX_CLASS_WEIGHT = 50
```

The class weight is passed to XGBoost as `sample_weight`.

## Model Training Setup

The model is trained using XGBoost multi-class classification.

The resident split is done by resident, not randomly by rows. This prevents the same resident's patterns from appearing in both train and test.

Current split:

```text
Train residents:      1, 2, 3, 4, 7, 8
Validation residents: 5, 10
Test residents:       6, 14
```

Row counts:

```text
Train rows:      423,646
Validation rows: 172,147
Test rows:        35,060
```

Label distribution:

Train:

```text
Moderate: 261,703
High:      78,395
Low:       64,487
Critical:  19,061
```

Validation:

```text
Moderate: 132,183
Low:       27,752
High:       8,977
Critical:   3,235
```

Test:

```text
Moderate: 25,646
Low:       5,004
High:      4,127
Critical:    283
```

## XGBoost Configuration

The baseline model uses:

```text
objective = multi:softprob
eval_metric = mlogloss
max_depth = 5
learning_rate = 0.05
subsample = 0.85
colsample_bytree = 0.85
reg_lambda = 2.0
reg_alpha = 0.2
tree_method = hist
early_stopping_rounds = 40
```

Early stopping was used with the validation set.

Best result:

```text
Best iteration: 176
Best validation mlogloss: 0.3404
```

The earlier 400-tree model showed overfitting after around 200 trees. Early stopping controlled this.

## High-Class Threshold Tuning

The `High` class had low precision in earlier runs. To reduce false High alerts, a post-processing threshold was tuned on the validation set.

Selected threshold:

```text
HIGH_THRESHOLD = 0.70
```

Logic:

```text
If predicted class is High
and probability of High is below 0.70
then downgrade prediction to Moderate
```

This improved High precision while keeping Critical recall acceptable.

## Final Test Performance

Final evaluation was performed once on the untouched test residents:

```text
Test residents: 6, 14
```

Performance:

```text
Class      Precision   Recall   F1-score   Support

Critical      0.82      1.00      0.90        283
High          0.56      0.63      0.59       4127
Low           0.66      0.97      0.78       5004
Moderate      0.93      0.82      0.87      25646

Accuracy                          0.82      35060
Macro Avg     0.74      0.85      0.79      35060
Weighted Avg  0.85      0.82      0.83      35060
```

Interpretation:

- Critical detection is strong.
- Critical recall reached `1.00` on the final test set.
- Moderate detection is good.
- Low detection has high recall.
- High is the weakest class, but threshold tuning improved it.
- Most remaining errors are between Low, Moderate, and High.

## Current Model Status

The current model is suitable as:

```text
Baseline model: yes
Prototype prediction model: yes
Final clinically validated model: no
```

The model can be saved as:

```text
fall_risk_xgb_baseline_v1.pkl
```

Recommended artifact contents:

- XGBoost model
- Label encoder
- Feature column list
- `HIGH_THRESHOLD`
- Train resident IDs
- Validation resident IDs
- Test resident IDs
- Best iteration
- Best validation score



Current Model Limitation

The current model is trained using rule-generated risk labels rather than confirmed fall events.

Therefore:

The model learns patterns associated with the current risk assessment framework.

The model does not directly predict confirmed falls.

Future versions should incorporate actual fall-event outcomes and incident reports for supervised fall prediction.
## Recommended Next Steps

Use the V1 model for prediction and history capture.

For each prediction, store:

- `resident_id`
- prediction timestamp
- predicted risk label
- class probabilities
- model version
- feature snapshot
- created timestamp

This prediction history can later support trend views such as:

- last 24-hour risk trend
- last 7-day risk trend
- High/Critical alert count
- increasing or decreasing risk pattern

Future V2 improvements should include:

- 7-day heart rate average
- 7-day HRV average
- 7-day SpO2 trend
- 7-day step trend
- 7-day sleep trend
- risk history features
- REM sleep
- sleep computation state
- sleep confidence

These should be added in a new dataset/model version, not mixed into the current V1 artifact.
