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
    DEFAULT_TRAINING_END_AT,
    DEFAULT_TRAINING_START_AT,
    build_engine,
    build_prediction_features,
    fetch_band_logs,
    predict_rows,
    prediction_model_version,
    write_prediction_history_sqlite,
)
from resident_context_cache import get_resident_baselines
from rolling_rule_engine_v1 import evaluate_rule_layer
from trend_engine_v1 import get_resident_trend_context


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
    "avg_breathing": "average breathing",
    "steps_30m": "30-minute step count",
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
            if term and term not in supporting_terms:
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

    if active_trend.get("trend_direction") == "worsening":
        if active_trend.get("trend_severity") == "Strongly Deteriorating":
            resolved_level = min(3, max(resolved_level, rule_level + 1, model_level))
        reasons.append(
            f"Trend is worsening over the {trend_window} window."
        )
    elif active_trend.get("trend_direction") == "improving":
        reasons.append(
            f"Trend is improving over the {trend_window} window; monitoring remains appropriate."
        )

    if rule_result.get("rawSafetyDriverCount", 0) > 0:
        resolved_level = max(resolved_level, 2)
        reasons.append("Raw safety driver is present, so final risk is kept at High or above.")

    conflict = rule_label != model_label
    return {
        "finalRisk": risk_from_rank(resolved_level),
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
        elif "low bp" in combined or "bp below" in combined or "blood pressure" in combined:
            term = "low blood pressure"
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
    shap_payload=None,
    resident_context=None,
):
    if resident_context is None:
        resident_context = {}
    trend_analysis = trend_context.get("analysis") or {}
    fallback_analysis = trend_context.get("fallbackAnalysis") or {}
    top_drivers, driver_phrase = describe_top_drivers(rule_result)
    active_trend, trend_window = choose_active_trend(trend_analysis, fallback_analysis)
    final_label = resolved_risk.get("finalRisk")

    if top_drivers:
        observation_text = join_terms(observation_terms(top_drivers))
        opening = (
            "Current observations show several concerning indicators, including "
            f"{observation_text}."
        )
    else:
        opening = (
            "Current observations do not show strong immediate fall-risk indicators."
        )

    broader_sentence = broader_pattern_sentence(
        resolved_risk,
        bool(top_drivers),
        shap_payload,
    )
    age_sentence = age_context_sentence(resident_context, final_label)
    medical_sentence = medical_context_sentence(resident_context, final_label)

    trend_direction = active_trend.get("trend_direction")
    trend_severity = active_trend.get("trend_severity")
    trend_strength = active_trend.get("trend_strength")
    if trend_direction and trend_direction != "insufficient_data":
        if trend_direction == "worsening":
            trend_modifier = "with a worsening recent pattern"
        elif trend_direction == "improving":
            trend_modifier = "while recent observations show improvement"
        else:
            trend_modifier = "with a stable recent trend"
    else:
        trend_modifier = None
        trend_sentence = (
            "Additional trend information should be reviewed to determine whether "
            "the condition is temporary or part of a worsening pattern."
        )

    if trend_modifier:
        risk_sentence = (
            f"The overall risk assessment is {final_label}, suggesting "
            f"{risk_phrase(final_label)} {trend_modifier}."
        )
        trend_sentence = (
            "The recent pattern should be reviewed alongside current observations "
            "to determine whether this is a sustained change or a temporary condition."
        )
    else:
        risk_sentence = (
            f"The overall risk assessment is {final_label}, suggesting "
            f"{risk_phrase(final_label)}."
        )

    summary_parts = [opening, risk_sentence]
    if broader_sentence:
        summary_parts.append(broader_sentence)
    if age_sentence:
        summary_parts.append(age_sentence)
    if medical_sentence:
        summary_parts.append(medical_sentence)
    if trend_sentence:
        summary_parts.append(trend_sentence)
    if final_label in ["High", "Critical"]:
        summary_parts.append(
            "Caregiver review is recommended to determine whether preventive action is needed."
        )
    elif final_label == "Moderate":
        summary_parts.append("Continued monitoring is recommended.")
    else:
        summary_parts.append("Routine monitoring can continue unless new concerns appear.")
    summary = " ".join(summary_parts)

    recommendations = []
    if final_label in ["High", "Critical"]:
        recommendations.append(
            "Caregiver review is recommended to assess whether the current drivers represent an emerging decline or a temporary condition."
        )
    elif final_label == "Moderate":
        recommendations.append(
            "Closer monitoring is recommended, with caregiver review if symptoms, mobility, or vitals continue to worsen."
        )
    if active_trend.get("trend_direction") == "worsening":
        recommendations.append(
            "Because the risk trend is worsening, prioritize review of mobility, vitals, sleep recovery, and recent care changes."
        )
    elif active_trend.get("trend_direction") == "improving":
        recommendations.append(
            "Trend is improving; continue monitoring and confirm that improvement is sustained across later observations."
        )

    return {
        "purpose": "CareGPT explanation layer input",
        "finalInsight": summary,
        "summary": summary,
        "driverNarrative": opening,
        "assessmentNarrative": risk_sentence,
        "broaderPatternNarrative": broader_sentence,
        "ageContextNarrative": age_sentence,
        "medicalContextNarrative": medical_sentence,
        "trendNarrative": trend_sentence,
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
        },
    }


def build_pipeline_record(
    feature_row,
    raw_row,
    prediction_row,
    history_db,
    shap_payload=None,
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
            "window": trend_context.get("window"),
            "fallbackWindow": trend_context.get("fallbackWindow"),
        },
        "careGptLayer": make_caregpt_payload(
            rule_result,
            prediction_row,
            trend_context,
            resolved_risk,
            shap_payload,
            resident_context,
        ),
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
    for driver in drivers[:max_items]:
        public_drivers.append(
            {
                "domain": driver.get("domain"),
                "parameter": driver.get("parameter"),
                "detail": driver.get("detail"),
            }
        )
    return public_drivers


def active_public_trend(record):
    latest = (record.get("trendLayer") or {}).get("latest24h") or {}
    fallback = (record.get("trendLayer") or {}).get("fallback7d") or {}
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
    }


def public_pipeline_record(record):
    care = record.get("careGptLayer") or {}
    resolved = record.get("resolvedRiskLayer") or {}
    resident_context = record.get("residentContextLayer") or {}

    return {
        "residentId": record.get("residentId"),
        "generatedAt": record.get("generatedAt"),
        "finalRisk": resolved.get("finalRisk"),
        "confidence": {
            "model": (record.get("v1ModelLayer") or {}).get("confidence"),
        },
        "finalInsight": care.get("finalInsight"),
        "recommendations": care.get("recommendations", []),
        "trend": active_public_trend(record),
        "topObservedDrivers": top_public_drivers(record),
        "supportingPatterns": (
            (care.get("structuredContext") or {}).get("supportingPatterns", [])
        ),
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
    parser.add_argument("--artifact", default="fall_risk_xgb_baseline_v1.pkl")
    parser.add_argument("--history-db", default="fall_risk_prediction_history.db")
    parser.add_argument("--mode", choices=["latest", "history"], default="latest")
    parser.add_argument("--output-json", default="fall_risk_layered_pipeline_v1.json")
    parser.add_argument("--public-output-json", default="fall_risk_public_output_v1.json")
    parser.add_argument("--output-csv", default="fall_risk_layered_pipeline_v1_summary.csv")
    parser.add_argument("--no-write-history", action="store_true")
    parser.add_argument("--disable-shap", action="store_true")
    parser.add_argument("--shap-top-n", type=int, default=5)
    args = parser.parse_args()

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
    band_df["generated_at"] = pd.to_datetime(band_df["generated_at"], utc=True)
    if args.resident_id is not None:
        band_df = band_df[band_df["resident_id"].eq(args.resident_id)]
        if band_df.empty:
            raise SystemExit(
                f"No post-training band_log rows found for resident {args.resident_id}."
            )

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

    artifact = joblib.load(args.artifact)
    predictions = predict_rows(features_df, artifact, mode=args.mode)
    model_version = prediction_model_version(artifact)

    if not args.no_write_history:
        write_prediction_history_sqlite(
            predictions=predictions,
            model_version=model_version,
            training_start_at=DEFAULT_TRAINING_START_AT,
            training_end_at=DEFAULT_TRAINING_END_AT,
            sqlite_path=args.history_db,
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
