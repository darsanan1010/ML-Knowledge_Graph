import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import shap
except ImportError:
    shap = None

from predict_fall_risk_v1 import (
    write_prediction_history_postgres,
    DEFAULT_TRAINING_END_AT,
    DEFAULT_TRAINING_START_AT,
    build_engine,
    build_prediction_features,
    fetch_band_logs,
    predict_rows,
    prediction_model_version,
    write_prediction_history_sqlite,
)
try:
    from resident_context_cache import get_resident_baselines
except ImportError:
    def get_resident_baselines(*args, **kwargs):
        print("Warning: resident_context_cache module missing. Using empty mock baselines.")
        return pd.DataFrame(columns=[
            "resident_id", "reference_heart_rate", "reference_hrv",
            "reference_oxygen_saturation", "reference_systolic_bp",
            "reference_diastolic_bp", "reference_body_temperature",
            "reference_step_count", "resident_age", "height_cm", "weight_kg",
            "reference_total_sleep_minutes", "reference_deep_sleep", "reference_light_sleep"
        ])
from rolling_rule_engine_v2 import evaluate_rule_layer
from trend_engine_v2 import get_resident_trend_context


RISK_ORDER = {
    "Low": 0,
    "Moderate": 1,
    "High": 2,
    "Critical": 3,
}
RISK_LABELS = ["Low", "Moderate", "High", "Critical"]
SHAP_FEATURE_LABELS = {
    "avg_hr": "average heart rate",
    "max_hr": "maximum heart rate",
    "min_hr": "minimum heart rate",
    "hr_std": "heart-rate variability within window",
    "avg_hrv": "average HRV",
    "avg_spo2": "average SpO2",
    "min_spo2": "minimum SpO2",
    "avg_sbp": "average systolic BP",
    "avg_dbp": "average diastolic BP",
    "avg_temp": "average body temperature",
    "avg_stress": "average stress",
    "avg_fatigue": "average fatigue",
    "severe_domain_count": "multiple physiological domains showing severe abnormalities",
    "steps_30m": "short-term mobility drop",
    "steps_30m_vs_12h": "drop in step count compared to 12-hour baseline",
    "steps_30m_vs_2h": "drop in step count compared to 2-hour baseline",
    "vitals_domain_flag": "vital-sign deviations detected",
    "ewma_fatigue": "sustained high fatigue levels",
    "severe_recovery_flag": "severe disruption in physiological recovery signals",
    "abnormal_domain_count": "abnormalities detected across multiple clinical domains",
    "avg_breathing": "average breathing",
    "step_std": "step-count variation",
    "activity_ratio": "activity ratio",
    "daily_total_sleep_minutes": "daily sleep duration",
    "sleep_baseline_minutes": "sleep baseline",
    "sleep_ratio": "sleep ratio",
    "sleep_deficit": "sleep deficit",
    "hrv_availability": "HRV availability",
    "spo2_availability": "SpO2 availability",
}


def clean(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def age_group(age):
    age = clean(age)
    if age is None:
        return None
    if age <= 0:
        return None
    if age >= 85:
        return "advanced_age"
    if age >= 75:
        return "older_adult"
    if age >= 65:
        return "senior"
    return "adult"


def text_blob_from_fields(*values):
    parts = []
    for value in values:
        value = clean(value)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            parts.append(value)
    return " ".join(parts)


def has_any_pattern(text, patterns):
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def extract_medical_context_flags(text):
    if not text:
        return {}

    flag_patterns = {
        "walking_assistance_or_difficulty": [
            r"\bwalking assistance\b",
            r"\bassistance needed for walking\b",
            r"\bdifficulty .*walking\b",
            r"\bwalker\b",
            r"\bwheelchair\b",
            r"\bunsteady\b",
            r"\bgait\b",
        ],
        "memory_or_cognitive_concern": [
            r"\bmemory impairment\b",
            r"\bcognitive\b",
            r"\bconfusion\b",
            r"\bconfused\b",
            r"\bdementia\b",
        ],
        "recent_or_active_surgery_recovery": [
            r"\brecovering from surgery\b",
            r"\bpost[- ]?surgery\b",
            r"\brecent surgery\b",
        ],
        "diabetes": [
            r"\bdiabet",
            r"\bmetformin\b",
        ],
        "blood_pressure_condition": [
            r"\bhigh bp\b",
            r"\bblood pressure\b",
            r"\bhypertension\b",
            r"\belevated blood press",
        ],
        "respiratory_condition": [
            r"\basthma\b",
            r"\bcopd\b",
            r"\brespiratory\b",
        ],
        "medication_timing_note": [
            r"\bmedication\b",
            r"\bafter food\b",
            r"\bmedicine\b",
        ],
    }
    return {
        flag: has_any_pattern(text, patterns)
        for flag, patterns in flag_patterns.items()
    }


def medical_context_phrases(flags):
    phrase_map = {
        "walking_assistance_or_difficulty": "walking assistance or mobility difficulty",
        "memory_or_cognitive_concern": "memory or cognitive concerns",
        "recent_or_active_surgery_recovery": "recent surgery recovery",
        "diabetes": "diabetes",
        "blood_pressure_condition": "blood-pressure history",
        "respiratory_condition": "respiratory history",
        "medication_timing_note": "medication timing considerations",
    }
    return [phrase for flag, phrase in phrase_map.items() if flags.get(flag)]


def resident_context_payload(feature_row):
    age = clean(feature_row.get("resident_age"))
    if age is not None and age <= 0:
        age = None
    medical_text = text_blob_from_fields(
        feature_row.get("medical_history"),
        feature_row.get("resident_condition"),
    )
    medical_flags = extract_medical_context_flags(medical_text)
    clinical_phrases = medical_context_phrases(medical_flags)
    return {
        "age": age,
        "ageGroup": age_group(age),
        "dobAvailable": clean(feature_row.get("resident_dob")) is not None,
        "medicalContextAvailable": bool(medical_text),
        "medicalContextFlags": medical_flags,
        "clinicalContextPhrases": clinical_phrases,
        "rawMedicalContext": {
            "medicalHistory": clean(feature_row.get("medical_history")),
            "condition": clean(feature_row.get("resident_condition")),
        },
        "usedInV1ModelFeatures": False,
        "usage": (
            "Resident age and medical context are used as contextual information "
            "only; they are not part of the trained V1 model feature vector."
        ),
    }


def latest_rows_by_resident(df):
    return (
        df.sort_values(["resident_id", "generated_at"])
        .groupby("resident_id", as_index=False)
        .tail(1)
        .copy()
    )


def make_probability_payload(prediction_row):
    return {
        "label": prediction_row.get("predicted_risk_label"),
        "confidence": clean(prediction_row.get("prediction_confidence")),
        "probabilities": {
            "Critical": clean(prediction_row.get("prob_Critical")),
            "High": clean(prediction_row.get("prob_High")),
            "Low": clean(prediction_row.get("prob_Low")),
            "Moderate": clean(prediction_row.get("prob_Moderate")),
        },
    }


def derived_feature_group(feature_name):
    if feature_name in ["avg_hrv", "hrv_availability"]:
        return "HRV"
    if feature_name in ["avg_spo2", "min_spo2", "spo2_availability"]:
        return "SpO2"
    return None


def abnormal_derived_feature_context(feature_name, feature_row):
    group = derived_feature_group(feature_name)
    if group == "HRV":
        avg_hrv = clean(feature_row.get("avg_hrv"))
        reference_hrv = clean(feature_row.get("reference_hrv"))
        tolerance = clean(feature_row.get("hrv_tolerance"))
        availability = clean(feature_row.get("hrv_availability"))

        if (
            avg_hrv is not None
            and reference_hrv is not None
            and tolerance is not None
            and avg_hrv < reference_hrv - tolerance
        ):
            return (
                True,
                f"average HRV {avg_hrv:.2f} is below baseline {reference_hrv:.2f}",
            )
        return False, None

    if group == "SpO2":
        avg_spo2 = clean(feature_row.get("avg_spo2"))
        min_spo2 = clean(feature_row.get("min_spo2"))
        reference_spo2 = clean(feature_row.get("reference_oxygen_saturation"))
        tolerance = clean(feature_row.get("oxygen_saturation_tolerance"))
        availability = clean(feature_row.get("spo2_availability"))

        if min_spo2 is not None and min_spo2 < 92:
            return True, f"minimum SpO2 {min_spo2:.1f}% is below 92%"
        if (
            avg_spo2 is not None
            and reference_spo2 is not None
            and tolerance is not None
            and avg_spo2 < reference_spo2 - tolerance
        ):
            return (
                True,
                f"average SpO2 {avg_spo2:.1f}% is below baseline {reference_spo2:.1f}%",
            )
        return False, None

    return None, None


def derived_feature_reliability_context(feature_name, feature_row):
    group = derived_feature_group(feature_name)
    if group == "HRV":
        availability = clean(feature_row.get("hrv_availability"))
        if availability is not None and availability < 0.5:
            return True, f"HRV availability is low ({availability:.2f})"
        return False, None

    if group == "SpO2":
        availability = clean(feature_row.get("spo2_availability"))
        if availability is not None and availability < 0.5:
            return True, f"SpO2 availability is low ({availability:.2f})"
        return False, None

    return None, None


def shap_observation_term(feature_name, impact, feature_row=None):
    if impact <= 0:
        return None

    if feature_name in ["steps_30m", "step_std", "activity_ratio"]:
        return "reduced activity level"
    if feature_name in [
        "daily_total_sleep_minutes",
        "sleep_baseline_minutes",
        "sleep_ratio",
        "sleep_deficit",
    ]:
        return "sleep recovery"
    if feature_name in ["avg_hrv", "hrv_availability"]:
        is_abnormal, _ = abnormal_derived_feature_context(feature_name, feature_row)
        if is_abnormal is not True:
            return None
        return "recovery signals"
    if feature_name in ["avg_hr", "max_hr", "min_hr", "hr_std"]:
        return "heart-rate pattern"
    if feature_name in ["avg_sbp", "avg_dbp"]:
        return "blood-pressure pattern"
    if feature_name in ["avg_spo2", "min_spo2", "spo2_availability"]:
        is_abnormal, _ = abnormal_derived_feature_context(feature_name, feature_row)
        if is_abnormal is not True:
            return None
        return "oxygenation pattern"
    if feature_name == "avg_temp":
        return "temperature pattern"
    if feature_name == "avg_fatigue":
        return "fatigue pattern"
    if feature_name == "avg_stress":
        return "stress pattern"
    if feature_name == "avg_breathing":
        return "breathing pattern"
    return SHAP_FEATURE_LABELS.get(feature_name, feature_name)


def extract_class_shap_values(shap_values, row_idx, class_idx, class_count):
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_idx][row_idx])

    values = np.asarray(shap_values)
    if values.ndim == 3:
        if values.shape[2] == class_count:
            return values[row_idx, :, class_idx]
        if values.shape[1] == class_count:
            return values[row_idx, class_idx, :]
    if values.ndim == 2:
        return values[row_idx]

    raise ValueError(f"Unsupported SHAP values shape: {values.shape}")


def make_shap_entry(feature_name, impact, feature_value, feature_row):
    group = derived_feature_group(feature_name)
    is_abnormal, abnormal_reason = abnormal_derived_feature_context(
        feature_name,
        feature_row,
    )
    has_reliability_issue, reliability_reason = derived_feature_reliability_context(
        feature_name,
        feature_row,
    )
    return {
        "feature": feature_name,
        "displayName": SHAP_FEATURE_LABELS.get(feature_name, feature_name),
        "derivedFeatureGroup": group,
        "isClearlyAbnormal": is_abnormal,
        "abnormalReason": abnormal_reason,
        "hasReliabilityConcern": has_reliability_issue,
        "reliabilityReason": reliability_reason,
        "impact": round(float(impact), 6),
        "value": clean(feature_value),
    }


def unavailable_shap_payload(reason):
    return {
        "available": False,
        "method": "SHAP TreeExplainer",
        "reason": reason,
        "explainedClass": None,
        "topPositiveDrivers": [],
        "topNegativeDrivers": [],
        "supportingObservationTerms": [],
    }


def build_shap_explanations(feature_rows, prediction_rows, artifact, top_n=5):
    if shap is None:
        return {}, unavailable_shap_payload("shap package is not installed")

    feature_columns = artifact["feature_columns"]
    missing_features = [
        column for column in feature_columns if column not in feature_rows.columns
    ]
    if missing_features:
        return {}, unavailable_shap_payload(
            f"Missing features for SHAP: {missing_features}"
        )

    model = artifact["model"]
    label_encoder = artifact["label_encoder"]
    prediction_index = prediction_rows.set_index(
        ["resident_id", "generated_at"]
    ).sort_index()
    explanation_rows = feature_rows.sort_values(
        ["resident_id", "generated_at"]
    ).copy()
    x_values = explanation_rows[feature_columns]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_values)
    except Exception as exc:
        return {}, unavailable_shap_payload(f"SHAP failed: {exc}")

    explanations = {}
    class_count = len(label_encoder.classes_)
    for row_idx, (_, feature_row) in enumerate(explanation_rows.iterrows()):
        key = (feature_row["resident_id"], feature_row["generated_at"])
        if key not in prediction_index.index:
            continue

        prediction_row = prediction_index.loc[key]
        if isinstance(prediction_row, pd.DataFrame):
            prediction_row = prediction_row.iloc[-1]

        explained_class = prediction_row.get("predicted_risk_label")
        try:
            class_idx = label_encoder.transform([explained_class])[0]
            row_shap_values = extract_class_shap_values(
                shap_values,
                row_idx,
                class_idx,
                class_count,
            )
        except Exception as exc:
            explanations[key] = unavailable_shap_payload(
                f"Could not extract class SHAP values: {exc}"
            )
            continue

        impacts = pd.Series(row_shap_values, index=feature_columns)
        positive = impacts[impacts > 0].sort_values(ascending=False).head(top_n)
        negative = impacts[impacts < 0].sort_values(ascending=True).head(top_n)
        supporting_terms = []
        for feature_name, impact in positive.items():
            term = shap_observation_term(feature_name, impact, feature_row)
            if term:
                if translate_shap_feature(feature_name, term) is None:
                    continue
                if term not in supporting_terms:
                    supporting_terms.append(term)

        explanations[key] = {
            "available": True,
            "method": "SHAP TreeExplainer",
            "explainedClass": clean(explained_class),
            "topPositiveDrivers": [
                make_shap_entry(
                    feature_name,
                    impact,
                    feature_row[feature_name],
                    feature_row,
                )
                for feature_name, impact in positive.items()
            ],
            "topNegativeDrivers": [
                make_shap_entry(
                    feature_name,
                    impact,
                    feature_row[feature_name],
                    feature_row,
                )
                for feature_name, impact in negative.items()
            ],
            "supportingObservationTerms": supporting_terms[:top_n],
        }

    return explanations, None


def risk_rank(label):
    return RISK_ORDER.get(label, 0)


def risk_from_rank(rank):
    return RISK_LABELS[max(0, min(int(rank), len(RISK_LABELS) - 1))]


def evidence_based_rule_confidence(rule_result):
    """Confidence in the rule assessment, not a risk percentage."""
    drivers = rule_result.get("riskDrivers") or []
    scored_drivers = [driver for driver in drivers if driver.get("score", 0) < 0]
    raw_count = rule_result.get("rawSafetyDriverCount", 0)
    rolling_count = len(
        [
            driver
            for driver in drivers
            if driver.get("source") in ["30m_aggregation", "daily_context"]
            and driver.get("score", 0) < 0
        ]
    )
    risk = rule_result.get("layeredRuleRisk")

    if not scored_drivers:
        return 72

    base_by_risk = {
        "Low": 68,
        "Moderate": 74,
        "High": 82,
        "Critical": 90,
    }
    confidence = base_by_risk.get(risk, 70)
    confidence += min(8, len(scored_drivers) * 2)
    confidence += min(6, raw_count * 3)
    confidence += min(4, rolling_count)
    return int(max(55, min(96, confidence)))


def resolve_risk(rule_result, prediction_row, trend_context):
    rule_label = rule_result.get("layeredRuleRisk")
    model_label = prediction_row.get("predicted_risk_label")
    model_confidence = clean(prediction_row.get("prediction_confidence")) or 0
    trend_analysis = trend_context.get("analysis") or {}
    fallback_analysis = trend_context.get("fallbackAnalysis") or {}
    active_trend, trend_window = choose_active_trend(trend_analysis, fallback_analysis)

    rule_level = risk_rank(rule_label)
    model_level = risk_rank(model_label)
    resolved_level = max(rule_level, model_level)
    reasons = []

    if rule_level > model_level:
        reasons.append(
            "Rule layer is higher than model prediction because current safety or rolling-context drivers are active."
        )
    elif model_level > rule_level:
        if model_confidence >= 0.70:
            reasons.append(
                "Model layer is higher than rule assessment with sufficient confidence."
            )
        else:
            resolved_level = max(rule_level, model_level - 1)
            reasons.append(
                "Model layer is higher, but confidence is limited, so risk is not fully escalated by model output alone."
            )
    else:
        reasons.append("Rule layer and model layer agree.")

    trend_dir = active_trend.get("trend_direction")
    if trend_dir == "worsening":
        if active_trend.get("trend_severity") == "Strongly Deteriorating":
            resolved_level = min(3, max(resolved_level, rule_level + 1, model_level))
        reasons.append(
            f"Trend is worsening over the {trend_window} window."
        )
    elif trend_dir == "improving":
        reasons.append(
            f"Trend is improving over the {trend_window} window; monitoring remains appropriate."
        )

    if rule_result.get("rawSafetyDriverCount", 0) > 0:
        resolved_level = max(resolved_level, 2)
        reasons.append("Raw safety driver is present, so final risk is kept at High or above.")

    # Calculate 0-100 Risk Score
    p_low = clean(prediction_row.get("prob_Low", 0.0)) or 0.0
    p_mod = clean(prediction_row.get("prob_Moderate", 0.0)) or 0.0
    p_high = clean(prediction_row.get("prob_High", 0.0)) or 0.0
    p_critical = clean(prediction_row.get("prob_Critical", 0.0)) or 0.0
    
    risk_score = (p_low * 12.5) + (p_mod * 37.5) + (p_high * 62.5) + (p_critical * 87.5)
    
    if resolved_level > model_level:
        escalation_diff = resolved_level - model_level
        risk_score += (escalation_diff * 25)
        
    if trend_dir in ["deteriorating", "worsening"]:
        risk_score += 2
        
    severe_count = clean(prediction_row.get("severe_domain_count", 0)) or 0
    if severe_count >= 3:
        risk_score += 2
        
    risk_score = round(risk_score)
    if resolved_level == 0:
        risk_score = max(0, min(24, risk_score))
    elif resolved_level == 1:
        risk_score = max(25, min(49, risk_score))
    elif resolved_level == 2:
        risk_score = max(50, min(74, risk_score))
    else:
        risk_score = max(75, min(100, risk_score))

    conflict = rule_label != model_label
    return {
        "finalRisk": risk_from_rank(resolved_level),
        "riskScore": risk_score,
        "ruleRisk": rule_label,
        "modelRisk": model_label,
        "modelConfidence": model_confidence,
        "ruleConfidence": evidence_based_rule_confidence(rule_result),
        "conflict": conflict,
        "conflictType": (
            "rule_higher"
            if rule_level > model_level
            else "model_higher"
            if model_level > rule_level
            else "aligned"
        ),
        "trendUsed": active_trend,
        "trendWindowUsed": trend_window,
        "resolutionReasons": reasons,
    }


def scored_risk_drivers(rule_result):
    drivers = rule_result.get("riskDrivers") or []
    return [driver for driver in drivers if driver.get("score", 0) < 0]


SHAP_TRANSLATION = {
    "ewma_dbp": "Consistently elevated diastolic blood pressure",
    "transition_worsening_flag": "Multiple indicators have worsened compared with previous observations",
    "context_worsening_count": "Several clinical domains show simultaneous deterioration",
    "activity_ratio": "reduced daily activity compared to baseline",
    "activity_ratio_12h": "reduced recent activity",
    "avg_sbp": "recent deviations in systolic blood pressure",
    "avg_dbp": "recent deviations in diastolic blood pressure",
    "avg_hr": "recent deviations in heart rate",
}

def translate_shap_feature(raw_feat, disp_name=None):
    raw_lower = raw_feat.lower().strip() if raw_feat else ""
    disp_lower = disp_name.lower().strip() if disp_name else ""
    
    ignored_raw = {
        "steps_30m", "steps_30m_vs_2h", "steps_30m_vs_6h", "steps_30m_vs_12h",
        "abnormal_domain_count", "severe_domain_count", "daily_sleep_feature_supported"
    }
    ignored_disp = {
        "short-term mobility drop",
        "drop in step count compared to 12-hour baseline",
        "drop in step count compared to 6-hour baseline",
        "drop in step count compared to 2-hour baseline",
        "steps 30m vs 6h",
        "steps 30m vs 12h",
        "steps 30m vs 2h",
        "abnormalities detected across multiple clinical domains",
        "multiple physiological domains showing severe abnormalities",
        "abnormal domain count",
        "severe domain count"
    }
    
    if raw_lower in ignored_raw or disp_lower in ignored_disp:
        return None
    for term in ignored_disp:
        if term in raw_lower or term in disp_lower:
            return None
            
    if raw_feat in SHAP_TRANSLATION:
        return SHAP_TRANSLATION[raw_feat]
    return (disp_name or raw_feat).replace("_", " ").title()

def data_quality_alerts(rule_result):
    drivers = rule_result.get("riskDrivers") or []
    return [driver for driver in drivers if driver.get("source") == "data_quality"]


def describe_top_drivers(rule_result, max_items=3):
    drivers = scored_risk_drivers(rule_result)
    if not drivers:
        return [], "no active rule drivers"

    sorted_drivers = sorted(drivers, key=lambda item: item.get("score", 0))
    top_drivers = sorted_drivers[:max_items]
    driver_names = [driver.get("parameter") for driver in top_drivers]
    domain_names = []
    for driver in top_drivers:
        domain = driver.get("domain")
        if domain and domain not in domain_names:
            domain_names.append(domain)

    if len(driver_names) == 1:
        driver_phrase = driver_names[0]
    elif len(driver_names) == 2:
        driver_phrase = f"{driver_names[0]} and {driver_names[1]}"
    else:
        driver_phrase = ", ".join(driver_names[:-1]) + f", and {driver_names[-1]}"

    domain_phrase = ", ".join(domain_names)
    return top_drivers, f"{driver_phrase} across {domain_phrase}"


def observation_terms(top_drivers):
    terms = []
    for driver in top_drivers:
        domain = driver.get("domain")
        parameter = driver.get("parameter", "")
        detail = driver.get("detail", "")
        combined = f"{parameter} {detail}".lower()

        if "spo2" in combined or "oxygen" in combined:
            term = "low oxygen saturation"
        elif "hrv" in combined:
            term = "reduced HRV compared with baseline"
        elif "low bp" in combined or "bp below" in combined or "hypotension" in combined:
            term = "low blood pressure"
        elif "high bp" in combined or "bp above" in combined or "hypertension" in combined:
            term = "elevated blood pressure"
        elif "blood pressure" in combined:
            term = "abnormal blood pressure"
        elif "tachycardia" in combined or "hr above" in combined or "heart rate" in combined:
            term = "elevated heart rate"
        elif "bradycardia" in combined or "hr below" in combined:
            term = "low heart rate"
        elif "temperature" in combined:
            term = "abnormal body temperature"
        elif "sleep deficit" in combined or domain == "Sleep":
            term = "sleep deficit"
        elif "fatigue" in combined:
            term = "elevated fatigue"
        elif "stress" in combined:
            term = "elevated stress"
        elif "activity" in combined or "steps" in combined or domain == "Mobility":
            term = "reduced mobility"
        elif domain == "Safety":
            term = "acute safety concern"
        elif domain == "Data Quality":
            term = "incomplete data"
        elif domain == "Vitals":
            term = "vital-sign deviation"
        else:
            term = parameter.lower() if parameter else "risk indicator"

        if term not in terms:
            terms.append(term)

    return terms


def join_terms(terms):
    if not terms:
        return "limited current risk indicators"
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2:
        return f"{terms[0]} and {terms[1]}"
    return ", ".join(terms[:-1]) + f", and {terms[-1]}"


def risk_phrase(final_label):
    if final_label in ["Critical", "High"]:
        return "elevated fall risk"
    if final_label == "Moderate":
        return "increased fall risk"
    return "currently lower fall risk"


def choose_active_trend(trend_analysis, fallback_analysis):
    if (
        trend_analysis.get("trend_direction") == "insufficient_data"
        and fallback_analysis
        and fallback_analysis.get("trend_direction") != "insufficient_data"
    ):
        return fallback_analysis, "7-day fallback"
    return trend_analysis, "24-hour"


def broader_pattern_sentence(resolved_risk, has_current_drivers, shap_payload=None):
    final_label = resolved_risk.get("finalRisk")
    conflict_type = resolved_risk.get("conflictType")
    supporting_terms = []
    if shap_payload and shap_payload.get("available"):
        supporting_terms = shap_payload.get("supportingObservationTerms") or []
    supporting_phrase = join_terms(supporting_terms[:3]) if supporting_terms else None

    if final_label not in ["Moderate", "High", "Critical"]:
        return None

    if not has_current_drivers:
        if supporting_phrase:
            return (
                "The elevated assessment is influenced by broader recent patterns, "
                f"especially {supporting_phrase}, even though immediate safety "
                "indicators are not strongly active in the latest observation."
            )
        return (
            "The elevated assessment is influenced by broader recent physiological "
            "and behavioral patterns, even though immediate safety indicators are "
            "not strongly active in the latest observation."
        )

    if conflict_type == "model_higher":
        if supporting_phrase:
            return (
                "The assessment also considers broader recent patterns, including "
                f"{supporting_phrase}, beyond the immediately visible drivers."
            )
        return (
            "The assessment also considers broader recent physiological and "
            "behavioral patterns beyond the immediately visible drivers."
        )

    return None


def age_context_sentence(resident_context, final_label):
    age = resident_context.get("age")
    if age is None or final_label not in ["Moderate", "High", "Critical"]:
        return None
    if age >= 85:
        return (
            "Age-related vulnerability should be considered when reviewing the "
            "resident's current risk pattern."
        )
    if age >= 75 and final_label in ["High", "Critical"]:
        return (
            "Resident age may increase vulnerability if the current pattern persists."
        )
    return None


def medical_context_sentence(resident_context, final_label):
    if final_label not in ["Moderate", "High", "Critical"]:
        return None

    phrases = resident_context.get("clinicalContextPhrases") or []
    if not phrases:
        return None

    priority_order = [
        "walking assistance or mobility difficulty",
        "recent surgery recovery",
        "memory or cognitive concerns",
        "blood-pressure history",
        "diabetes",
        "respiratory history",
        "medication timing considerations",
    ]
    ordered_phrases = [
        phrase for phrase in priority_order if phrase in phrases
    ][:3]
    if not ordered_phrases:
        return None

    return (
        "Resident clinical context also notes "
        f"{join_terms(ordered_phrases)}, which should be considered during review."
    )


def make_caregpt_payload(
    rule_result,
    prediction_row,
    trend_context,
    resolved_risk,
    vitals_snapshot,
    shap_payload=None,
    resident_context=None,
):
    if resident_context is None:
        resident_context = {}
    
    trend_analysis = trend_context.get("analysis") or {}
    fallback_analysis = trend_context.get("fallbackAnalysis") or {}
    rapid_analysis = trend_context.get("rapidAnalysis") or {}
    
    active_trend, trend_window = choose_active_trend(trend_analysis, fallback_analysis)
    
    # If the 6h trend is active and definitive, we prioritize mentioning it in CareGPT text
    rapid_dir = rapid_analysis.get("trend_direction")
    if rapid_dir in ["worsening", "improving"]:
        overall_trend_text = f"{rapid_dir} (rapid 6-hour trajectory)"
    elif active_trend.get("trend_direction") in ["worsening", "improving"]:
        overall_trend_text = f"{active_trend.get('trend_direction')} (24-hour trajectory)"
    elif active_trend.get("trend_direction") == "insufficient_data":
        overall_trend_text = "unclear due to insufficient historical data"
    else:
        overall_trend_text = "stable"

    final_label = resolved_risk.get("finalRisk")
    score = resolved_risk.get("riskScore", 0)
    trend_strength = active_trend.get("trend_strength")

    # 1. Current Status (Vitals)
    feature_context = rule_result.get("featureContext", {})
    def fmt_val(v):
        return v if v is not None else "N/A"
        
    hr = vitals_snapshot.get("heartRate")
    spo2 = vitals_snapshot.get("spo2")
    bp_sys = vitals_snapshot.get("systolicBP")
    bp_dia = vitals_snapshot.get("diastolicBP")
    
    hr_str = fmt_val(hr)
    spo2_str = fmt_val(spo2)
    bp_sys_str = fmt_val(bp_sys)
    bp_dia_str = fmt_val(bp_dia)
    
    vitals_abnormal = False
    if (hr and (hr > 100 or hr < 50)) or (spo2 and spo2 < 94) or (bp_sys and (bp_sys > 160 or bp_sys < 90)):
        vitals_abnormal = True

    # --- V3: Drift Severity Classification ---
    # Translates internal breach ratio math into Mild/Moderate/Severe labels
    # for individual vital domains so caregivers see actionable text, not raw numbers.
    def classify_drift_severity(breach_ratio):
        """Returns Mild/Moderate/Severe based on how far outside the tolerance zone a vital is."""
        if breach_ratio is None:
            return None
        if abs(breach_ratio) < 0.5:
            return None  # Within tolerance, not worth labelling
        elif abs(breach_ratio) < 1.0:
            return "Mild"
        elif abs(breach_ratio) < 1.5:
            return "Moderate"
        else:
            return "Severe"

    hr_breach = feature_context.get("hr_breach")
    sbp_breach = feature_context.get("sbp_breach")
    hr_severity = classify_drift_severity(hr_breach)
    sbp_severity = classify_drift_severity(sbp_breach)

    drift_phrases = []
    if hr_severity:
        direction = "elevated" if (hr_breach or 0) > 0 else "depressed"
        drift_phrases.append(f"{hr_severity} heart rate drift ({direction}, {hr_str} bpm)")
    if sbp_severity:
        direction = "elevated" if (sbp_breach or 0) > 0 else "low"
        drift_phrases.append(f"{sbp_severity} blood pressure drift ({direction}, {bp_sys_str}/{bp_dia_str} mmHg)")
    drift_severity_text = ("; ".join(drift_phrases) + ".") if drift_phrases else None
    # --- End Drift Severity Classification ---


    # 2. Recent Trend
    mob_ratio = feature_context.get("activity_ratio")
    mob_ratio_12h = feature_context.get("activity_ratio_12h")
    mob_trend = "stable"
    if mob_ratio is not None and mob_ratio_12h is not None:
        if (mob_ratio - mob_ratio_12h) < 0:
            mob_trend = "worsened"
        elif (mob_ratio - mob_ratio_12h) > 0:
            mob_trend = "improved"
            
    recent_trend = f"Overall risk trajectory is {overall_trend_text}. Mobility has {mob_trend} over the last 12 hours."
    
    # Add domain details instead of generic phrase
    top_drivers, _ = describe_top_drivers(rule_result)
    driver_terms = observation_terms(top_drivers)
    if not top_drivers and shap_payload and "topPositiveDrivers" in shap_payload:
        for feat in shap_payload.get("topPositiveDrivers", []):
            disp_name = feat.get("displayName") or feat.get("feature", "")
            raw_feat = feat.get("feature", "")
            translated = translate_shap_feature(raw_feat, disp_name)
            if translated is None:
                continue
            driver_terms.append(translated.lower())
            if len(driver_terms) >= 2:
                break

    if len(driver_terms) > 0:
        if len(driver_terms) == 1:
            recent_trend += f" Recent observations show signs of {driver_terms[0]}."
        elif len(driver_terms) == 2:
            recent_trend += f" Recent observations show signs of {driver_terms[0]} and {driver_terms[1]}."
        else:
            recent_trend += f" Recent observations show signs of {', '.join(driver_terms[:-1])}, and {driver_terms[-1]}."

    # 3. Risk Assessment
    if len(driver_terms) == 1:
        driver_str = driver_terms[0]
    elif len(driver_terms) == 2:
        driver_str = f"{driver_terms[0]} and {driver_terms[1]}"
    elif len(driver_terms) > 2:
        driver_str = ", ".join(driver_terms[:-1]) + f", and {driver_terms[-1]}"
    else:
        driver_str = "underlying baseline factors"

    risk_assessment = f"Overall fall risk is {final_label} ({score}/100). The assessment is primarily driven by {driver_str}."

    if vitals_abnormal:
        abnormal_list = []
        if hr and (hr > 100 or hr < 50):
            abnormal_list.append(f"heart rate ({hr_str} bpm)")
        if spo2 and spo2 < 94:
            abnormal_list.append(f"SPO2 ({spo2_str}%)")
        if bp_sys and (bp_sys > 160 or bp_sys < 90):
            abnormal_list.append(f"blood pressure ({bp_sys_str}/{bp_dia_str} mmHg)")
            
        abnormal_str = ", ".join(abnormal_list)
        current_status = f"Current vitals show abnormalities in {abnormal_str}. Recent readings: heart rate {hr_str} bpm, SPO2 {spo2_str}%, blood pressure {bp_sys_str}/{bp_dia_str} mmHg."
    else:
        if len(driver_terms) > 0:
            current_status = f"While current point-in-time vitals are within normal ranges (heart rate {hr_str} bpm, SPO2 {spo2_str}%, BP {bp_sys_str}/{bp_dia_str} mmHg), resident shows {driver_str}, which continue to contribute to the elevated fall-risk assessment."
        else:
            current_status = f"Current point-in-time vitals are within normal ranges, with heart rate {hr_str} bpm, SPO2 {spo2_str}%, and blood pressure {bp_sys_str}/{bp_dia_str} mmHg."
    # Append drift severity label if any vital is drifting outside its tolerance zone
    if drift_severity_text:
        current_status += f" Drift detected: {drift_severity_text}"

    # 4. Clinical Context
    context_notes = []
    age = resident_context.get("age")
    if age:
        context_notes.append(f"Resident is {age} years old.")
    bmi = resident_context.get("bmi")
    if bmi is not None:
        if bmi < 18.5:
            context_notes.append(f"Resident is clinically underweight (BMI {bmi:.1f}).")
        elif bmi >= 30:
            context_notes.append(f"Resident is clinically obese (BMI {bmi:.1f}).")
            
    carenote_phrases = resident_context.get("clinicalContextPhrases", [])
    if carenote_phrases:
        context_notes.append(f"History of {', '.join(carenote_phrases).lower()}.")
    
    clinical_context = " ".join(context_notes) if context_notes else "No significant clinical context notes available."

    # 5. Confidence
    model_conf = prediction_row.get("prediction_confidence", 0)
    conf_pct = round(float(model_conf) * 100) if model_conf else 0
    
    cov_12h = feature_context.get("coverage_12h")
    cov_12h_pct = int(cov_12h * 100) if cov_12h is not None else 0
    max_gap = feature_context.get("max_gap_minutes")

    if max_gap is not None and max_gap > 1440:
        confidence = f"Assessment confidence is low ({conf_pct}%). Coverage is sparse due to significant historical data gaps (>24h)."
    elif cov_12h is not None and cov_12h >= 0.85 and (max_gap is None or max_gap < 60):
        confidence = f"Assessment confidence is high ({conf_pct}%). 12h coverage: {cov_12h_pct}%. No significant wearable gaps detected."
    elif cov_12h is not None and cov_12h >= 0.5:
        if max_gap and max_gap >= 60 and cov_12h >= 0.85:
            confidence = f"Assessment confidence is moderate ({conf_pct}%). Coverage is good ({cov_12h_pct}%), but a {int(max_gap)}-minute data gap was detected."
        else:
            confidence = f"Assessment confidence is moderate ({conf_pct}%). Coverage reduced to {cov_12h_pct}%. Several data interruptions detected."
    else:
        confidence = f"Assessment confidence is low ({conf_pct}%). Large wearable gaps detected. Trend calculations may be incomplete."
        if cov_12h is not None and cov_12h < 0.25:
            confidence += f" Note: Assessment is based on limited recent data (<25% coverage). Verify with direct observation."

    # 6. Recommendation
    recommendations = []
    driver_text = " ".join(driver_terms).lower()
    
    if "sleep deficit" in driver_text or "fatigue" in driver_text:
        recommendations.append("Review sleep recovery and nighttime interruptions.")
    if "mobility" in driver_text or "activity" in driver_text or "steps" in driver_text:
        recommendations.append("Assess mobility status and recent activity decline.")
    if "heart rate" in driver_text or "blood pressure" in driver_text:
        recommendations.append("Evaluate cardiovascular stability and hydration status.")
    if "abnormalities detected" in driver_text or "multiple domains" in driver_text or "safety concern" in driver_text:
        recommendations.append("Perform caregiver review within current shift.")
        
    if not recommendations:
        if final_label in ["High", "Critical"]:
            recommendations.append("Caregiver review is recommended to assess whether the current drivers represent an emerging decline or a temporary condition.")
        elif final_label == "Moderate":
            recommendations.append("Closer monitoring is recommended, with caregiver review if symptoms, mobility, or vitals continue to worsen.")
        elif active_trend.get("trend_direction") == "worsening":
            recommendations.append("Because the risk trend is worsening, prioritize review of mobility, vitals, sleep recovery, and recent care changes.")
        elif active_trend.get("trend_direction") == "improving":
            recommendations.append("Trend is improving; continue monitoring and confirm that improvement is sustained across later observations.")
        else:
            recommendations.append("Routine monitoring can continue unless new concerns appear.")
            
    recommendations = list(dict.fromkeys(recommendations))
    recommendation_text = " ".join(recommendations)

    final_insight_dict = {
        "currentStatus": current_status,
        "recentTrend": recent_trend,
        "clinicalContext": clinical_context,
        "recommendation": recommendations,
        "supportingPatterns": list(set(driver_terms)),
    }

    return {
        "purpose": "CareGPT explanation layer input",
        "finalInsight": final_insight_dict,
        "summary": " ".join([current_status, recent_trend, risk_assessment, clinical_context, confidence, recommendation_text]),
        "driverNarrative": current_status,
        "assessmentNarrative": risk_assessment,
        "broaderPatternNarrative": recent_trend,
        "ageContextNarrative": clinical_context,
        "medicalContextNarrative": confidence,
        "trendNarrative": recommendation_text,
        "recommendations": recommendations,
        "structuredContext": {
            "finalRisk": final_label,
            "residentAge": resident_context.get("age"),
            "residentAgeGroup": resident_context.get("ageGroup"),
            "clinicalContextPhrases": resident_context.get(
                "clinicalContextPhrases",
                [],
            ),
            "topDrivers": top_drivers,
            "observationTerms": observation_terms(top_drivers),
            "trendWindowUsed": trend_window,
            "trendDirectionUsed": active_trend.get("trend_direction"),
            "trendSeverityUsed": active_trend.get("trend_severity"),
            "trendStrengthUsed": trend_strength,
            "supportingPatterns": (
                shap_payload.get("supportingObservationTerms")
                if shap_payload and shap_payload.get("available")
                else []
            ),
            "trendDirection24h": trend_analysis.get("trend_direction"),
            "trendSeverity24h": trend_analysis.get("trend_severity"),
            "fallbackTrendDirection7d": fallback_analysis.get("trend_direction"),
            "fallbackTrendSeverity7d": fallback_analysis.get("trend_severity"),
            "driftSeverityHR": hr_severity,
            "driftSeveritySBP": sbp_severity,
            "driftSeverityText": drift_severity_text,
        },
    }


def load_synthetic_carenotes(csv_path="synthetic_carenotes.csv"):
    if not Path(csv_path).exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def get_latest_carenote(resident_id, notes_df):
    if notes_df.empty or "resident_id" not in notes_df.columns:
        return None, None
    resident_notes = notes_df[notes_df["resident_id"] == resident_id]
    if resident_notes.empty:
        return None, None
    
    if "generated_at" in resident_notes.columns:
        resident_notes = resident_notes.sort_values("generated_at")
        
    latest_note = resident_notes.iloc[-1]["synthetic_care_note"]
    
    bmi = None
    if isinstance(latest_note, str):
        match = re.search(r"BMI:\s*([\d\.]+)", latest_note)
        if match:
            bmi = float(match.group(1))
            
    return latest_note, bmi


def build_pipeline_record(
    feature_row,
    raw_row,
    prediction_row,
    history_db,
    shap_payload=None,
    carenotes_df=None,
):
    if shap_payload is None:
        shap_payload = unavailable_shap_payload("SHAP was not computed for this row")

    rule_result = evaluate_rule_layer(feature_row, raw_row, history_db)
    trend_context = get_resident_trend_context(
        resident_id=feature_row["resident_id"],
        hours=24,
        fallback_hours=168,
        db_path=history_db,
    )
    resolved_risk = resolve_risk(rule_result, prediction_row, trend_context)
    resident_context = resident_context_payload(feature_row)

    caregpt_note, bmi = None, None
    if carenotes_df is not None:
        caregpt_note, bmi = get_latest_carenote(feature_row["resident_id"], carenotes_df)
    
    if bmi is not None:
        resident_context["bmi"] = bmi

    def safe_round(val, decimals=2):
        if val is None:
            return None
        return round(float(val), decimals)

    mob_ratio = rule_result["featureContext"].get("activity_ratio")
    mob_ratio_12h = rule_result["featureContext"].get("activity_ratio_12h")
    mob_trend_dir = "stable"
    mob_delta = 0
    if mob_ratio is not None and mob_ratio_12h is not None:
        delta_val = round((mob_ratio - mob_ratio_12h) * 100)
        mob_delta = abs(delta_val)
        if delta_val < 0:
            mob_trend_dir = "worsening"
        elif delta_val > 0:
            mob_trend_dir = "improving"

    vitals_snapshot = {
        "heartRate": safe_round(feature_row.get("heart_rate") or rule_result["featureContext"].get("avg_hr")),
        "spo2": safe_round(feature_row.get("oxygen_saturation") or rule_result["featureContext"].get("avg_spo2")),
        "systolicBP": safe_round(feature_row.get("systolic_bp") or rule_result["featureContext"].get("avg_sbp")),
        "diastolicBP": safe_round(feature_row.get("diastolic_bp") or rule_result["featureContext"].get("avg_dbp")),
        "temperature": safe_round(feature_row.get("body_temperature")),
        "sleepScore": min(100, round(rule_result["featureContext"].get("sleep_ratio") * 100)) if rule_result["featureContext"].get("sleep_ratio") is not None else None,
        "mobility": round(mob_ratio * 100) if mob_ratio is not None else None,
        "mobilityDelta12h": mob_delta,
        "trendDirection": mob_trend_dir,
        "currentSteps30m": safe_round(rule_result["featureContext"].get("steps_30m")),
    }

    return {
        "residentId": clean(feature_row["resident_id"]),
        "generatedAt": clean(feature_row["generated_at"]),
        "residentContextLayer": resident_context,
        "rawSafetyLayer": {
            "description": "Immediate safety checks from raw latest band_log.",
            "drivers": [
                driver
                for driver in rule_result["riskDrivers"]
                if driver.get("source") == "raw_bandlog"
            ],
        },
        "aggregationLayer": {
            "description": "30-minute rolling and daily context features.",
            "features": rule_result["featureContext"],
            "drivers": [
                driver
                for driver in rule_result["riskDrivers"]
                if driver.get("source") in ["30m_aggregation", "daily_context"]
            ],
        },
        "dataQualityLayer": {
            "description": "Non-scoring data-quality alerts that may affect interpretation.",
            "drivers": [
                driver
                for driver in rule_result["riskDrivers"]
                if driver.get("source") == "data_quality"
            ],
        },
        "ruleEngineLayer": {
            "risk": rule_result["layeredRuleRisk"],
            "score": rule_result["layeredRuleScore"],
            "confidence": resolved_risk["ruleConfidence"],
            "legacyScoreConfidence": rule_result["confidence"],
            "recommendedAction": rule_result["recommendedAction"],
            "drivers": rule_result["riskDrivers"],
            "rawCurrentVitals": {
                "heart_rate": feature_row.get("heart_rate"),
                "oxygen_saturation": feature_row.get("oxygen_saturation"),
                "systolic_bp": feature_row.get("systolic_bp"),
                "diastolic_bp": feature_row.get("diastolic_bp"),
                "body_temperature": feature_row.get("body_temperature"),
            },
            "featureContext": {
                "avg_hr": feature_row.get("avg_hr"),
                "avg_spo2": feature_row.get("avg_spo2"),
                "avg_sbp": feature_row.get("avg_sbp"),
                "avg_dbp": feature_row.get("avg_dbp"),
                "sleep_ratio": feature_row.get("sleep_ratio"),
                "activity_ratio": feature_row.get("activity_ratio"),
                "activity_ratio_12h": feature_row.get("activity_ratio_12h"),
                "steps_30m": feature_row.get("steps_30m"),
                "steps_6h": feature_row.get("steps_6h"),
                "steps_12h": feature_row.get("steps_12h"),
                "reference_step_count": feature_row.get("reference_step_count"),
                "reference_heart_rate": feature_row.get("reference_heart_rate"),
                "reference_hrv": feature_row.get("reference_hrv"),
                "reference_oxygen_saturation": feature_row.get("reference_oxygen_saturation"),
                "reference_systolic_bp": feature_row.get("reference_systolic_bp"),
                "reference_diastolic_bp": feature_row.get("reference_diastolic_bp"),
                "coverage_6h": feature_row.get("coverage_6h"),
                "coverage_12h": feature_row.get("coverage_12h"),
                "max_gap_minutes": feature_row.get("max_gap_minutes"),
            }
        },
        "v1ModelLayer": make_probability_payload(prediction_row),
        "modelExplainabilityLayer": shap_payload,
        "resolvedRiskLayer": resolved_risk,
        "predictionStoreLayer": {
            "historyDb": history_db,
            "storedByKey": "resident_id + generated_at + model_version",
        },
        "trendLayer": {
            "latest24h": trend_context.get("analysis"),
            "fallback7d": trend_context.get("fallbackAnalysis"),
            "rapid6h": trend_context.get("rapidAnalysis"),
            "window": trend_context.get("window"),
            "fallbackWindow": trend_context.get("fallbackWindow"),
        },
        "careGptLayer": make_caregpt_payload(
            rule_result,
            prediction_row,
            trend_context,
            resolved_risk,
            vitals_snapshot,
            shap_payload,
            resident_context,
        ),
        "vitalsSnapshotLayer": vitals_snapshot,
        "careGptNotes": caregpt_note,
    }


def top_public_drivers(record, max_items=3):
    drivers = []
    for driver in record.get("rawSafetyLayer", {}).get("drivers", []):
        if driver.get("score", 0) < 0:
            drivers.append(driver)
    for driver in record.get("aggregationLayer", {}).get("drivers", []):
        if driver.get("score", 0) < 0:
            drivers.append(driver)

    drivers = sorted(drivers, key=lambda item: item.get("score", 0))
    public_drivers = []
    seen_titles = set()
    
    for driver in drivers:
        title = driver.get("parameter")
        detail = driver.get("detail", "")
        driver_score = driver.get("score", 0)
        
        if title:
            if "tachycardia" in title.lower():
                title = title.replace("Tachycardia", "Elevated Heart Rate").replace("tachycardia", "elevated heart rate")
            if "tachycardia" in detail.lower():
                detail = detail.replace("Tachycardia", "Elevated Heart Rate").replace("tachycardia", "elevated heart rate")
                
            if title.lower() not in seen_titles:
                seen_titles.add(title.lower())
                
                status_val = "worsening"
                if driver.get("trend") == "Worsening":
                    if driver_score == -1:
                        status_val = "slightly worsening"
                    elif driver_score == -2:
                        status_val = "moderately worsening"
                    elif driver_score <= -3:
                        status_val = "significantly worsening"
                else:
                    status_val = "stable"
                    
                public_drivers.append(
                    {
                        "title": title,
                        "status": status_val,
                        "detail": detail,
                    }
                )
            if len(public_drivers) >= max_items:
                break
                
    if not public_drivers:
        # Fallback to SHAP features if rule drivers are empty
        shap_payload = record.get("modelExplainabilityLayer") or {}
        shap_features = shap_payload.get("topPositiveDrivers", [])
        
        for feat in shap_features:
            raw_feat = feat.get("feature", "unknown_feature")
            disp_name = feat.get("displayName") or raw_feat
            
            translated_title = translate_shap_feature(raw_feat, disp_name)
            
            if not translated_title:
                continue
                
            # Dynamic formatting for specific features in fallback
            feature_context = record.get("ruleEngineLayer", {}).get("featureContext", {})
            detail_text = None
            driver_pct = None
            
            if raw_feat == "activity_ratio":
                ratio_to_use = feature_context.get("activity_ratio_12h")
                if ratio_to_use is None:
                    ratio_to_use = feature_context.get("activity_ratio", 0.0)
                
                pct_decline = int((1.0 - ratio_to_use) * 100)
                
                steps_12h = feature_context.get("steps_12h")
                ref_steps = feature_context.get("reference_step_count") or 2000.0
                ref_steps_12h = ref_steps * 0.50
                
                if pct_decline > 0:
                    pct_decline = min(100, pct_decline)
                    driver_pct = pct_decline
                    translated_title = f"Mobility declined {pct_decline}% over 12h from baseline"
                    if steps_12h is not None:
                        detail_text = f"12-Hour steps {steps_12h:.0f} below baseline {ref_steps_12h:.0f}"
                    else:
                        detail_text = f"Mobility declined {pct_decline}% over 12h from baseline"
                else:
                    translated_title = "Mobility Pattern"
                    if steps_12h is not None:
                        detail_text = f"12-Hour steps {steps_12h:.0f} is above baseline {ref_steps_12h:.0f}"
                    else:
                        detail_text = "12-Hour mobility level is above baseline"
            elif raw_feat == "sleep_deficit":
                sleep_def = feature_context.get("sleep_deficit") or 0.0
                daily_sleep = feature_context.get("daily_total_sleep_minutes")
                sleep_baseline = max(300.0, feature_context.get("sleep_baseline_minutes") or 420.0)
                
                pct_decline = int(sleep_def * 100)
                pct_decline = max(0, min(100, pct_decline))
                driver_pct = pct_decline
                
                translated_title = f"Sleep reduced {pct_decline}% below baseline over 24h"
                if daily_sleep is not None:
                    detail_text = f"Daily sleep {daily_sleep:.0f}min below baseline {sleep_baseline:.0f}min"
                else:
                    detail_text = f"Sleep reduced {pct_decline}% below baseline over 24h"
            elif raw_feat in ["avg_hr", "max_hr", "min_hr"]:
                ref_hr = feature_context.get("reference_heart_rate") or 75.0
                val = feature_context.get(raw_feat)
                if val is not None and ref_hr > 0:
                    if val > ref_hr:
                        pct = min(100, ((val - ref_hr) / ref_hr) * 100)
                        driver_pct = pct
                        translated_title = f"Heart rate elevated {pct:.0f}% above baseline over 6h"
                        detail_text = f"6-Hour average HR {val:.0f} bpm above baseline {ref_hr:.0f} bpm"
                    else:
                        pct = min(100, ((ref_hr - val) / ref_hr) * 100)
                        driver_pct = pct
                        translated_title = f"Heart rate reduced {pct:.0f}% below baseline over 6h"
                        detail_text = f"6-Hour average HR {val:.0f} bpm below baseline {ref_hr:.0f} bpm"
                else:
                    detail_text = f"Model detected anomalous {translated_title.lower()}"
            elif raw_feat == "avg_hrv":
                ref_hrv = feature_context.get("reference_hrv") or 40.0
                val = feature_context.get("avg_hrv")
                if val is not None and ref_hrv > 0:
                    pct = min(100, ((ref_hrv - val) / ref_hrv) * 100)
                    driver_pct = pct
                    translated_title = f"HRV reduced {pct:.0f}% below baseline over 6h"
                    detail_text = f"6-Hour average HRV {val:.0f} below baseline {ref_hrv:.0f}"
                else:
                    detail_text = f"Model detected anomalous {translated_title.lower()}"
            elif raw_feat in ["avg_spo2", "min_spo2"]:
                ref_spo2 = feature_context.get("reference_oxygen_saturation") or 98.0
                val = feature_context.get(raw_feat)
                if val is not None and ref_spo2 > 0:
                    pct = min(100, ((ref_spo2 - val) / ref_spo2) * 100)
                    driver_pct = pct
                    translated_title = f"SpO2 reduced {pct:.0f}% below baseline over 6h"
                    detail_text = f"6-Hour average SpO2 {val:.0f}% below baseline {ref_spo2:.0f}%"
                else:
                    detail_text = f"Model detected anomalous {translated_title.lower()}"
            elif raw_feat == "avg_sbp":
                ref_sbp = feature_context.get("reference_systolic_bp") or 120.0
                val = feature_context.get("avg_sbp")
                if val is not None and ref_sbp > 0:
                    if val > ref_sbp:
                        pct = min(100, ((val - ref_sbp) / ref_sbp) * 100)
                        driver_pct = pct
                        translated_title = f"Systolic blood pressure elevated {pct:.0f}% above baseline over 6h"
                        detail_text = f"6-Hour average SBP {val:.0f} mmHg above baseline {ref_sbp:.0f} mmHg"
                    else:
                        pct = min(100, ((ref_sbp - val) / ref_sbp) * 100)
                        driver_pct = pct
                        translated_title = f"Systolic blood pressure reduced {pct:.0f}% below baseline over 6h"
                        detail_text = f"6-Hour average SBP {val:.0f} mmHg below baseline {ref_sbp:.0f} mmHg"
                else:
                    detail_text = f"Model detected anomalous {translated_title.lower()}"
            elif raw_feat == "avg_dbp":
                ref_dbp = feature_context.get("reference_diastolic_bp") or 80.0
                val = feature_context.get("avg_dbp")
                if val is not None and ref_dbp > 0:
                    if val > ref_dbp:
                        pct = min(100, ((val - ref_dbp) / ref_dbp) * 100)
                        driver_pct = pct
                        translated_title = f"Diastolic blood pressure elevated {pct:.0f}% above baseline over 6h"
                        detail_text = f"6-Hour average DBP {val:.0f} mmHg above baseline {ref_dbp:.0f} mmHg"
                    else:
                        pct = min(100, ((ref_dbp - val) / ref_dbp) * 100)
                        driver_pct = pct
                        translated_title = f"Diastolic blood pressure reduced {pct:.0f}% below baseline over 6h"
                        detail_text = f"6-Hour average DBP {val:.0f} mmHg below baseline {ref_dbp:.0f} mmHg"
                else:
                    detail_text = f"Model detected anomalous {translated_title.lower()}"
            elif raw_feat == "transition_worsening_flag":
                detail_text = "Resident is shifting from their stable baseline into a deteriorating physiological state"
            elif raw_feat == "context_worsening_count":
                worsening = []
                act_ratio = feature_context.get("activity_ratio_12h") or feature_context.get("activity_ratio", 1.0)
                if act_ratio < 0.95:
                    worsening.append("activity levels")
                if feature_context.get("sleep_deficit", 0.0) > 0.0:
                    worsening.append("sleep recovery")
                hr_breach = feature_context.get("hr_breach", 0.0)
                if hr_breach is not None and abs(hr_breach) > 0.0:
                    worsening.append("heart rate stability")
                sbp_breach = feature_context.get("sbp_breach", 0.0)
                dbp_breach = feature_context.get("dbp_breach", 0.0)
                if (sbp_breach is not None and abs(sbp_breach) > 0.0) or (dbp_breach is not None and abs(dbp_breach) > 0.0):
                    worsening.append("blood pressure stability")
                
                if worsening:
                    if len(worsening) == 1:
                        detail_text = f"Subtle deterioration observed in {worsening[0]}"
                    elif len(worsening) == 2:
                        detail_text = f"Subtle deterioration observed in {worsening[0]} and {worsening[1]}"
                    else:
                        detail_text = f"Subtle deterioration observed in {', '.join(worsening[:-1])}, and {worsening[-1]}"
                else:
                    detail_text = "Simultaneous subtle deterioration detected across multiple physiological domains"
            else:
                if raw_feat in SHAP_TRANSLATION:
                    detail_text = translated_title
                else:
                    detail_text = f"Model detected anomalous {translated_title.lower()}"
                    
            if translated_title.lower() not in seen_titles:
                seen_titles.add(translated_title.lower())
                
                status_val = "stable"
                if feat.get("impact", 0) > 0:
                    if driver_pct is not None:
                        if driver_pct < 5:
                            status_val = "slightly worsening"
                        elif driver_pct < 15:
                            status_val = "moderately worsening"
                        else:
                            status_val = "significantly worsening"
                    else:
                        status_val = "worsening"
                        
                public_drivers.append(
                    {
                        "title": translated_title,
                        "status": status_val,
                        "detail": detail_text,
                    }
                )
                if len(public_drivers) >= max_items:
                    break

    return public_drivers

def active_public_trend(record):
    latest = (record.get("trendLayer") or {}).get("latest24h") or {}
    fallback = (record.get("trendLayer") or {}).get("fallback7d") or {}
    rapid = (record.get("trendLayer") or {}).get("rapid6h") or {}
    
    if (
        latest.get("trend_direction") == "insufficient_data"
        and fallback
        and fallback.get("trend_direction") != "insufficient_data"
    ):
        trend = fallback
        window = "7-day"
    else:
        trend = latest
        window = "24-hour"

    return {
        "window": window,
        "direction": trend.get("trend_direction"),
        "severity": trend.get("trend_severity"),
        "isPartial": trend.get("is_partial_window"),
        "trend6h": {
            "direction": rapid.get("trend_direction"),
            "severity": rapid.get("trend_severity"),
        } if rapid else None
    }


def public_pipeline_record(record):
    care = record.get("careGptLayer") or {}
    resolved = record.get("resolvedRiskLayer") or {}
    resident_context = record.get("residentContextLayer") or {}
    trend_info = active_public_trend(record)
    
    raw_vitals = record.get("ruleEngineLayer", {}).get("rawCurrentVitals", {})
    feature_context = record.get("ruleEngineLayer", {}).get("featureContext", {})
    
    # Calculate risk score delta
    latest_score = resolved.get("riskScore")
    delta = None
    
    # trendLayer contains both latest24h and fallback7d which have previous_numeric_score
    active_trend_src = record.get("trendLayer", {}).get("fallback7d") if trend_info.get("window") == "7-day" else record.get("trendLayer", {}).get("latest24h")
    if active_trend_src and active_trend_src.get("previous_numeric_score") is not None and latest_score is not None:
        delta = latest_score - active_trend_src.get("previous_numeric_score")

    cov_12h = feature_context.get("coverage_12h")
    max_gap = feature_context.get("max_gap_minutes")
    data_quality = "good"
    if cov_12h is not None and cov_12h < 0.5:
        data_quality = "poor"
    if max_gap is not None and max_gap > 120:
        data_quality = "poor"

    mob_ratio = feature_context.get("activity_ratio")
    mob_ratio_12h = feature_context.get("activity_ratio_12h")
    mob_delta = None
    mob_trend_dir = "stable"
    
    if mob_ratio is not None and mob_ratio_12h is not None:
        delta_val = round((mob_ratio - mob_ratio_12h) * 100)
    care = record.get("careGptLayer") or {}
    trend_info = active_public_trend(record)

    final_drivers = top_public_drivers(record)
    driver_titles = {d.get("title", "").lower() for d in final_drivers}
    
    raw_patterns = (care.get("structuredContext") or {}).get("supportingPatterns", [])
    supporting_patterns = []
    seen_patterns = set(driver_titles)
    
    for pat in raw_patterns:
        translated = translate_shap_feature(pat, pat)
        if not translated:
            continue
        if translated.lower() not in seen_patterns:
            seen_patterns.add(translated.lower())
            supporting_patterns.append(translated)
            
    return {
        "residentId": record.get("residentId"),
        "generatedAt": record.get("generatedAt"),
        "finalRisk": resolved.get("finalRisk"),
        "riskScore": latest_score,
        "riskScoreDelta": delta,
        "vitalsSnapshot": record.get("vitalsSnapshotLayer", {}),
        "finalInsight": care.get("finalInsight"),
        "trend": trend_info,
        "riskDrivers": final_drivers,
        "supportingPatterns": supporting_patterns,
        "clinicalContext": {
            "age": resident_context.get("age"),
            "ageGroup": resident_context.get("ageGroup"),
            "phrases": resident_context.get("clinicalContextPhrases", []),
        },
    }

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Single-script layered V1 pilot: raw safety rules, 30m aggregation, "
            "V1 model prediction, SQLite history store, trend engine, and CareGPT context."
        )
    )
    parser.add_argument("--resident-id", type=int, default=None)
    parser.add_argument("--artifact", default="models_v2_grouped_real_holdout/fall_risk_xgboost_v23_final.pkl")
    parser.add_argument("--history-db", default="fall_risk_prediction_history.db")
    parser.add_argument("--mode", choices=["latest", "history"], default="latest")
    parser.add_argument("--output-json", default="fall_risk_layered_pipeline_v2.json")
    parser.add_argument("--public-output-json", default="fall_risk_public_output_v2.json")
    parser.add_argument("--output-csv", default="fall_risk_layered_pipeline_v2_summary.csv")
    parser.add_argument("--no-write-history", action="store_true")
    parser.add_argument("--disable-shap", action="store_true")
    parser.add_argument("--shap-top-n", type=int, default=5)
    parser.add_argument("--mock-raw-csv", default=None, help="Bypass DB and use this raw band log CSV directly.")
    parser.add_argument("--batch-json", default=None, help="Bypass DB and use a batch JSON array payload from Engineering.")
    args = parser.parse_args()

    if args.batch_json:
        print(f"Bypassing DB. Processing batch JSON payload from {args.batch_json}...")
        with open(args.batch_json, "r", encoding="utf-8") as f:
            batch_data = json.load(f)
        band_df = pd.DataFrame(batch_data)
        import re
        band_df = band_df.rename(columns=lambda x: re.sub(r'(?<!^)(?=[A-Z])', '_', x).lower())
        band_df = band_df.rename(columns={
            "systolic_b_p": "systolic_bp",
            "diastolic_b_p": "diastolic_bp",
        })
        if "is_synthetic" not in band_df.columns:
            band_df["is_synthetic"] = False
        engine = None
    elif args.mock_raw_csv:
        print(f"Bypassing DB. Loading raw mock data from {args.mock_raw_csv}...")
        band_df = pd.read_csv(args.mock_raw_csv)
        engine = None
    else:
        engine = build_engine()
        band_df = fetch_band_logs(
            engine=engine,
            hours=24,
            sleep_baseline_days=30,
            training_start_at=DEFAULT_TRAINING_START_AT,
            training_end_at=DEFAULT_TRAINING_END_AT,
            use_latest_available=True,
        )
    if band_df.empty:
        raise SystemExit("No post-training band_log rows found.")

    band_df["resident_id"] = pd.to_numeric(
        band_df["resident_id"], errors="coerce"
    ).astype("Int64")
    band_df["generated_at"] = pd.to_datetime(band_df["generated_at"], format="mixed", utc=True)
    if args.resident_id is not None:
        band_df = band_df[band_df["resident_id"].eq(args.resident_id)]
        if band_df.empty:
            raise SystemExit(
                f"No post-training band_log rows found for resident {args.resident_id}."
            )

    if args.mock_raw_csv:
        baseline_df = pd.DataFrame(columns=[
            "resident_id", "reference_heart_rate", "reference_hrv",
            "reference_oxygen_saturation", "reference_systolic_bp",
            "reference_diastolic_bp", "reference_body_temperature",
            "reference_step_count", "resident_age", "height_cm", "weight_kg",
            "reference_total_sleep_minutes", "reference_deep_sleep", "reference_light_sleep"
        ])
    else:
        baseline_df = get_resident_baselines(
            engine=engine,
            skip_engineering_api=False,
            use_sqlite_cache=True,
            use_database_fallback=True,
        )
    features_df = build_prediction_features(
        band_df,
        baseline_df,
        hours=24,
        use_latest_available=True,
    )
    if features_df.empty:
        raise SystemExit("No prediction feature rows were generated.")

    artifact_path = Path(args.artifact)
    if artifact_path.name.startswith("fall_risk_xgboost_"):
        model_dir = artifact_path.parent
        suffix = artifact_path.name.replace("fall_risk_xgboost_", "")
        artifact = {
            "model": joblib.load(artifact_path),
            "label_encoder": joblib.load(model_dir / f"risk_label_encoder_{suffix}"),
            "feature_columns": joblib.load(model_dir / f"model_features_{suffix}"),
            "model_version": suffix.replace(".pkl", "")
        }
    else:
        artifact = joblib.load(artifact_path)
    predictions = predict_rows(features_df, artifact, mode=args.mode)
    model_version = prediction_model_version(artifact)

    predictions["risk_score"] = ((predictions["prob_Low"] * 12.5) + (predictions["prob_Moderate"] * 37.5) + (predictions["prob_High"] * 62.5) + (predictions["prob_Critical"] * 87.5)).round()

    if not args.no_write_history:
        write_prediction_history_postgres(
            predictions=predictions,
            model_version=model_version,
            training_start_at=DEFAULT_TRAINING_START_AT,
            training_end_at=DEFAULT_TRAINING_END_AT,
            )

    if args.mode == "latest":
        selected_features = latest_rows_by_resident(features_df)
        selected_predictions = latest_rows_by_resident(predictions)
    else:
        selected_features = features_df.copy()
        selected_predictions = predictions.copy()

    if args.disable_shap:
        shap_explanations = {}
        shap_fallback = unavailable_shap_payload("SHAP disabled by CLI")
    else:
        shap_explanations, shap_fallback = build_shap_explanations(
            selected_features,
            selected_predictions,
            artifact,
            top_n=args.shap_top_n,
        )

    raw_index = band_df.set_index(["resident_id", "generated_at"]).sort_index()
    prediction_index = selected_predictions.set_index(
        ["resident_id", "generated_at"]
    ).sort_index()

    carenotes_df = load_synthetic_carenotes()

    records = []
    public_records = []
    summary_rows = []
    for _, feature_row in selected_features.iterrows():
        key = (feature_row["resident_id"], feature_row["generated_at"])
        if key not in raw_index.index or key not in prediction_index.index:
            continue

        raw_row = raw_index.loc[key]
        prediction_row = prediction_index.loc[key]
        if isinstance(raw_row, pd.DataFrame):
            raw_row = raw_row.iloc[-1]
        if isinstance(prediction_row, pd.DataFrame):
            prediction_row = prediction_row.iloc[-1]

        shap_payload = shap_explanations.get(key)
        if shap_payload is None:
            shap_payload = shap_fallback or unavailable_shap_payload(
                "No SHAP explanation found for this row"
            )

        record = build_pipeline_record(
            feature_row=feature_row,
            raw_row=raw_row,
            prediction_row=prediction_row,
            history_db=args.history_db,
            shap_payload=shap_payload,
            carenotes_df=carenotes_df,
        )
        records.append(record)
        public_records.append(public_pipeline_record(record))
        summary_rows.append(
            {
                "resident_id": record["residentId"],
                "resident_age": record["residentContextLayer"]["age"],
                "resident_age_group": record["residentContextLayer"]["ageGroup"],
                "clinical_context_available": record["residentContextLayer"][
                    "medicalContextAvailable"
                ],
                "clinical_context_phrases": "; ".join(
                    record["residentContextLayer"]["clinicalContextPhrases"]
                ),
                "generated_at": record["generatedAt"],
                "final_risk": record["resolvedRiskLayer"]["finalRisk"],
                "conflict": record["resolvedRiskLayer"]["conflict"],
                "conflict_type": record["resolvedRiskLayer"]["conflictType"],
                "rule_risk": record["ruleEngineLayer"]["risk"],
                "rule_score": record["ruleEngineLayer"]["score"],
                "rule_confidence": record["ruleEngineLayer"]["confidence"],
                "v1_model_risk": record["v1ModelLayer"]["label"],
                "v1_model_confidence": record["v1ModelLayer"]["confidence"],
                "trend_24h": (record["trendLayer"]["latest24h"] or {}).get(
                    "trend_direction"
                ),
                "trend_severity_24h": (record["trendLayer"]["latest24h"] or {}).get(
                    "trend_severity"
                ),
                "fallback_7d": (record["trendLayer"]["fallback7d"] or {}).get(
                    "trend_direction"
                ),
                "raw_safety_driver_count": len(
                    record["rawSafetyLayer"]["drivers"]
                ),
                "aggregation_driver_count": len(
                    record["aggregationLayer"]["drivers"]
                ),
                "data_quality_alert_count": len(
                    record["dataQualityLayer"]["drivers"]
                ),
                "shap_available": record["modelExplainabilityLayer"].get(
                    "available"
                ),
                "shap_top_positive": "; ".join(
                    item["feature"]
                    for item in record["modelExplainabilityLayer"].get(
                        "topPositiveDrivers",
                        [],
                    )[:3]
                ),
                "caregpt_recommendation_count": len(
                    record["careGptLayer"]["recommendations"]
                ),
            }
        )

    Path(args.output_json).write_text(
        json.dumps(records, indent=2, default=str),
        encoding="utf-8",
    )
    Path(args.public_output_json).write_text(
        json.dumps(public_records, indent=2, default=str),
        encoding="utf-8",
    )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.output_csv, index=False)

    print("\n=== Layered V1 Pipeline Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved full layered output to {args.output_json}")
    print(f"Saved public output to {args.public_output_json}")
    print(f"Saved summary to {args.output_csv}")


if __name__ == "__main__":
    main()
