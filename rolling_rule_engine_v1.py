import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import get_risk_level, score_to_confidence
from predict_fall_risk_v1 import (
    DEFAULT_TRAINING_END_AT,
    DEFAULT_TRAINING_START_AT,
    build_engine,
    build_prediction_features,
    fetch_band_logs,
)
from resident_context_cache import get_resident_baselines
from trend_engine_v1 import get_resident_trend_context


REFERENCE_ACTIVE_WINDOWS_PER_DAY = 16.0


def clean_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def add_driver(drivers, domain, parameter, detail, score, source):
    drivers.append(
        {
            "domain": domain,
            "parameter": parameter,
            "detail": detail,
            "score": score,
            "source": source,
            "trend": "Worsening",
        }
    )


def evaluate_raw_safety_rules(row):
    drivers = []

    hr = row.get("heart_rate")
    if pd.notna(hr):
        if hr < 45:
            add_driver(
                drivers,
                "Safety",
                "Critical Bradycardia",
                f"Raw HR critically low at {hr:.0f} bpm",
                -3,
                "raw_bandlog",
            )
        elif hr > 125:
            add_driver(
                drivers,
                "Safety",
                "Critical Tachycardia",
                f"Raw HR critically high at {hr:.0f} bpm",
                -3,
                "raw_bandlog",
            )

    sbp = row.get("systolic_bp")
    if pd.notna(sbp) and sbp < 90:
        add_driver(
            drivers,
            "Safety",
            "Critical Low BP",
            f"Raw systolic BP critically low at {sbp:.0f} mmHg",
            -3,
            "raw_bandlog",
        )

    spo2 = row.get("oxygen_saturation")
    spo2_valid = bool(row.get("oxygen_saturation_valid"))
    if spo2_valid and pd.notna(spo2) and spo2 < 92:
        add_driver(
            drivers,
            "Safety",
            "Critical Low SpO2",
            f"Raw SpO2 critically low at {spo2:.0f}%",
            -3,
            "raw_bandlog",
        )

    temp = row.get("body_temperature")
    if pd.notna(temp) and (temp < 35.5 or temp > 38.0):
        add_driver(
            drivers,
            "Safety",
            "Temperature Safety Alert",
            f"Raw body temperature {temp:.1f}C outside safety range",
            -2,
            "raw_bandlog",
        )

    return drivers


def evaluate_rolling_risk_rules(row):
    drivers = []

    avg_hr = row.get("avg_hr")
    ref_hr = row.get("reference_heart_rate")
    hr_tol = row.get("heart_rate_tolerance")
    if pd.notna(avg_hr) and pd.notna(ref_hr) and pd.notna(hr_tol):
        if avg_hr > ref_hr + hr_tol:
            add_driver(
                drivers,
                "Vitals",
                "30m HR Above Baseline",
                f"30m avg HR {avg_hr:.0f} above personal max {ref_hr + hr_tol:.0f}",
                -1,
                "30m_aggregation",
            )
        elif avg_hr < ref_hr - hr_tol:
            add_driver(
                drivers,
                "Vitals",
                "30m HR Below Baseline",
                f"30m avg HR {avg_hr:.0f} below personal min {ref_hr - hr_tol:.0f}",
                -1,
                "30m_aggregation",
            )

    avg_hrv = row.get("avg_hrv")
    ref_hrv = row.get("reference_hrv")
    hrv_availability = row.get("hrv_availability")
    if (
        pd.notna(avg_hrv)
        and pd.notna(ref_hrv)
        and ref_hrv > 0
        and pd.notna(hrv_availability)
        and hrv_availability >= 0.25
    ):
        hrv_drop = ((ref_hrv - avg_hrv) / ref_hrv) * 100
        if hrv_drop > 25:
            add_driver(
                drivers,
                "Vitals",
                "30m HRV Reduction",
                f"30m avg HRV dropped {hrv_drop:.1f}% from baseline",
                -2,
                "30m_aggregation",
            )

    min_spo2 = row.get("min_spo2")
    avg_spo2 = row.get("avg_spo2")
    ref_spo2 = row.get("reference_oxygen_saturation")
    spo2_availability = row.get("spo2_availability")
    if pd.notna(spo2_availability) and spo2_availability >= 0.25:
        if pd.notna(min_spo2) and min_spo2 < 92:
            add_driver(
                drivers,
                "Vitals",
                "30m SpO2 Drop",
                f"30m min SpO2 dropped to {min_spo2:.0f}%",
                -2,
                "30m_aggregation",
            )
        elif pd.notna(avg_spo2) and pd.notna(ref_spo2) and avg_spo2 < ref_spo2 - 3:
            add_driver(
                drivers,
                "Vitals",
                "30m SpO2 Below Baseline",
                f"30m avg SpO2 {avg_spo2:.0f}% below baseline {ref_spo2:.0f}%",
                -1,
                "30m_aggregation",
            )

    avg_sbp = row.get("avg_sbp")
    ref_sbp = row.get("reference_systolic_bp")
    sbp_tol = row.get("systolic_bp_tolerance")
    if pd.notna(avg_sbp):
        if avg_sbp < 95:
            add_driver(
                drivers,
                "Vitals",
                "30m Low BP",
                f"30m avg systolic BP low at {avg_sbp:.0f} mmHg",
                -2,
                "30m_aggregation",
            )
        elif pd.notna(ref_sbp) and pd.notna(sbp_tol) and avg_sbp < ref_sbp - sbp_tol:
            add_driver(
                drivers,
                "Vitals",
                "30m BP Below Baseline",
                f"30m avg systolic BP {avg_sbp:.0f} below personal min {ref_sbp - sbp_tol:.0f}",
                -1,
                "30m_aggregation",
            )

    steps_30m = row.get("steps_30m")
    ref_steps = row.get("reference_step_count")
    if pd.notna(steps_30m) and pd.notna(ref_steps) and ref_steps > 0:
        ref_steps_30m = ref_steps / REFERENCE_ACTIVE_WINDOWS_PER_DAY
        activity_ratio = row.get("activity_ratio")
        if pd.notna(activity_ratio) and activity_ratio < 0.25:
            add_driver(
                drivers,
                "Mobility",
                "Low 30m Activity",
                f"30m activity ratio {activity_ratio:.2f} is low",
                -2,
                "30m_aggregation",
            )
        elif steps_30m < ref_steps_30m * 0.5:
            add_driver(
                drivers,
                "Mobility",
                "30m Steps Below Baseline",
                f"30m steps {steps_30m:.0f} below expected {ref_steps_30m:.0f}",
                -1,
                "30m_aggregation",
            )

    fatigue = row.get("avg_fatigue")
    if pd.notna(fatigue) and fatigue >= 70:
        add_driver(
            drivers,
            "Mobility",
            "30m High Fatigue",
            f"30m avg fatigue {fatigue:.0f}",
            -1,
            "30m_aggregation",
        )

    stress = row.get("avg_stress")
    if pd.notna(stress) and stress >= 80:
        add_driver(
            drivers,
            "Stress",
            "30m High Stress",
            f"30m avg stress {stress:.0f}",
            -1,
            "30m_aggregation",
        )

    sleep_deficit = row.get("sleep_deficit")
    daily_sleep = row.get("daily_total_sleep_minutes")
    sleep_baseline = row.get("sleep_baseline_minutes")
    sleep_available = row.get("sleep_data_available")
    if pd.notna(sleep_available) and int(sleep_available) == 0:
        add_driver(
            drivers,
            "Data Quality",
            "Sleep Data Missing",
            "Daily sleep value is zero or unavailable; sleep risk is not scored from this row",
            0,
            "data_quality",
        )
    elif pd.notna(sleep_deficit) and sleep_deficit >= 0.4:
        add_driver(
            drivers,
            "Sleep",
            "Sleep Deficit",
            f"Daily sleep {daily_sleep:.0f}min below baseline {sleep_baseline:.0f}min",
            -2,
            "daily_context",
        )

    return drivers


def risk_action(level):
    actions = {
        "Low": "Continue monitoring. No immediate action required.",
        "Moderate": "Notify caregiver. Schedule assessment within 2 hours.",
        "High": "Trigger red alert. Initiate preventive protocol immediately.",
        "Critical": "Immediate intervention. Consider hospital evaluation.",
    }
    return actions.get(level)


def trend_recommendations(trend_context):
    if not trend_context or not trend_context.get("available"):
        return []

    analysis = trend_context.get("analysis") or {}
    fallback = trend_context.get("fallbackAnalysis") or {}
    active = (
        fallback
        if analysis.get("trend_direction") == "insufficient_data"
        and fallback
        and fallback.get("trend_direction") != "insufficient_data"
        else analysis
    )

    direction = active.get("trend_direction")
    severity = active.get("trend_severity")
    strength = active.get("trend_strength")
    if direction == "worsening":
        return [
            {
                "type": "trend_recommendation",
                "priority": "urgent" if severity == "Strongly Deteriorating" else "high",
                "message": f"Risk trend is {severity or 'worsening'} with strength {strength}.",
            }
        ]
    if direction == "improving":
        return [
            {
                "type": "trend_recommendation",
                "priority": "low",
                "message": f"Risk trend is {severity or 'improving'} with strength {strength}.",
            }
        ]
    return []


def evaluate_rule_layer(feature_row, raw_row, history_db):
    raw_drivers = evaluate_raw_safety_rules(raw_row)
    rolling_drivers = evaluate_rolling_risk_rules(feature_row)
    drivers = raw_drivers + rolling_drivers
    scoring_rolling_drivers = [
        driver
        for driver in rolling_drivers
        if driver.get("source") in ["30m_aggregation", "daily_context"]
        and driver.get("score", 0) < 0
    ]

    score = sum(driver["score"] for driver in drivers)
    level, _ = get_risk_level(score)
    trend_context = get_resident_trend_context(
        resident_id=feature_row["resident_id"],
        hours=24,
        fallback_hours=168,
        db_path=history_db,
    )

    return {
        "residentId": clean_value(feature_row["resident_id"]),
        "generatedAt": feature_row["generated_at"].isoformat(),
        "layeredRuleRisk": level,
        "layeredRuleScore": score,
        "confidence": score_to_confidence(score),
        "recommendedAction": risk_action(level),
        "riskDrivers": drivers,
        "rawSafetyDriverCount": len(raw_drivers),
        "rollingRiskDriverCount": len(scoring_rolling_drivers),
        "featureContext": {
            key: clean_value(feature_row.get(key))
            for key in [
                "avg_hr",
                "avg_hrv",
                "avg_spo2",
                "min_spo2",
                "avg_sbp",
                "avg_dbp",
                "steps_30m",
                "step_std",
                "activity_ratio",
                "daily_total_sleep_minutes",
                "sleep_data_available",
                "sleep_baseline_minutes",
                "sleep_ratio",
                "sleep_deficit",
                "hrv_availability",
                "spo2_availability",
            ]
        },
        "rawCurrentVitals": {
            key: clean_value(raw_row.get(key))
            for key in [
                "heart_rate",
                "hrv",
                "systolic_bp",
                "diastolic_bp",
                "oxygen_saturation",
                "oxygen_saturation_valid",
                "body_temperature",
                "step_count",
                "deep_sleep_time",
                "light_sleep_time",
                "fatigue_level",
                "stress",
                "breathing",
            ]
        },
        "trendContext": trend_context,
        "careGptContext": {
            "summary": (
                f"Resident {clean_value(feature_row['resident_id'])} is {level} "
                f"by layered rule engine using raw safety checks and 30m context."
            ),
            "recommendations": trend_recommendations(trend_context),
            "explanationInputs": {
                "rawSafetyChecks": len(raw_drivers),
                "rollingRiskChecks": len(scoring_rolling_drivers),
                "trendDirection24h": (trend_context.get("analysis") or {}).get("trend_direction"),
                "trendSeverity24h": (trend_context.get("analysis") or {}).get("trend_severity"),
                "fallbackTrendDirection7d": (
                    trend_context.get("fallbackAnalysis") or {}
                ).get("trend_direction"),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Standalone layered rule engine: raw safety + 30m features + trend context."
    )
    parser.add_argument("--resident-id", type=int, default=None)
    parser.add_argument("--history-db", default="fall_risk_prediction_history.db")
    parser.add_argument("--output-json", default="layered_rule_engine_results.json")
    parser.add_argument("--output-csv", default="layered_rule_engine_summary.csv")
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

    band_df["resident_id"] = pd.to_numeric(band_df["resident_id"], errors="coerce").astype("Int64")
    band_df["generated_at"] = pd.to_datetime(band_df["generated_at"], utc=True)
    if args.resident_id is not None:
        band_df = band_df[band_df["resident_id"].eq(args.resident_id)]
        if band_df.empty:
            raise SystemExit(f"No post-training rows found for resident {args.resident_id}.")

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
        raise SystemExit("No feature rows generated.")

    latest_features = (
        features_df.sort_values(["resident_id", "generated_at"])
        .groupby("resident_id", as_index=False)
        .tail(1)
    )
    latest_raw = (
        band_df.sort_values(["resident_id", "generated_at"])
        .drop_duplicates("resident_id", keep="last")
        .set_index("resident_id")
    )

    results = []
    summary_rows = []
    for _, feature_row in latest_features.iterrows():
        resident_id = feature_row["resident_id"]
        raw_row = latest_raw.loc[resident_id]
        result = evaluate_rule_layer(feature_row, raw_row, args.history_db)
        results.append(result)
        summary_rows.append(
            {
                "resident_id": result["residentId"],
                "generated_at": result["generatedAt"],
                "layered_rule_risk": result["layeredRuleRisk"],
                "layered_rule_score": result["layeredRuleScore"],
                "raw_safety_drivers": result["rawSafetyDriverCount"],
                "rolling_risk_drivers": result["rollingRiskDriverCount"],
                "trend_direction_24h": (
                    result["trendContext"].get("analysis") or {}
                ).get("trend_direction"),
                "trend_severity_24h": (
                    result["trendContext"].get("analysis") or {}
                ).get("trend_severity"),
                "fallback_direction_7d": (
                    result["trendContext"].get("fallbackAnalysis") or {}
                ).get("trend_direction"),
                "caregpt_recommendations": len(result["careGptContext"]["recommendations"]),
            }
        )

    Path(args.output_json).write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.output_csv, index=False)

    print("\n=== Layered Rule Engine Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved full results to {args.output_json}")
    print(f"Saved summary to {args.output_csv}")


if __name__ == "__main__":
    main()
