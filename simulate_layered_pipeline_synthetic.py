import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import pandas as pd

from fall_risk_layered_pipeline_v1 import (
    build_pipeline_record,
    build_shap_explanations,
    public_pipeline_record,
    unavailable_shap_payload,
)
from predict_fall_risk_v1 import build_engine, predict_rows, prediction_model_version
from resident_context_cache import get_resident_baselines


SCENARIOS = ["normal", "sitting", "elevated", "high", "recovery"]


def get_history_resident_ids(history_db):
    db_path = Path(history_db)
    if not db_path.exists():
        raise FileNotFoundError(f"Prediction history DB not found: {history_db}")

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT resident_id
            FROM fall_risk_prediction_history
            ORDER BY resident_id
            """
        ).fetchall()
    return [int(row[0]) for row in rows]


def resident_baseline_lookup(baseline_df):
    prepared = baseline_df.copy()
    prepared["resident_id"] = pd.to_numeric(
        prepared["resident_id"], errors="coerce"
    ).astype("Int64")
    return {
        int(row["resident_id"]): row
        for _, row in prepared.dropna(subset=["resident_id"]).iterrows()
    }


def value(row, name, default):
    current = row.get(name, default)
    if pd.isna(current):
        return default
    return float(current)


def synthetic_feature_row(resident_id, baseline_row, scenario, generated_at):
    ref_hr = value(baseline_row, "reference_heart_rate", 75.0)
    hr_tol = value(baseline_row, "heart_rate_tolerance", 20.0)
    ref_hrv = value(baseline_row, "reference_hrv", 40.0)
    hrv_tol = value(baseline_row, "hrv_tolerance", 10.0)
    ref_sbp = value(baseline_row, "reference_systolic_bp", 120.0)
    sbp_tol = value(baseline_row, "systolic_bp_tolerance", 20.0)
    ref_dbp = value(baseline_row, "reference_diastolic_bp", 80.0)
    dbp_tol = value(baseline_row, "diastolic_bp_tolerance", 15.0)
    ref_spo2 = value(baseline_row, "reference_oxygen_saturation", 98.0)
    ref_temp = value(baseline_row, "reference_body_temperature", 36.8)
    ref_steps = value(baseline_row, "reference_step_count", 2000.0)
    ref_sleep = value(baseline_row, "reference_total_sleep_minutes", 350.0)
    ref_steps_30m = max(ref_steps / 16.0, 1.0)

    scenario_values = {
        "normal": {
            "hr": ref_hr,
            "hrv": ref_hrv,
            "spo2": ref_spo2,
            "min_spo2": ref_spo2 - 1,
            "sbp": ref_sbp,
            "dbp": ref_dbp,
            "steps": ref_steps_30m,
            "stress": 30,
            "fatigue": 25,
            "sleep_ratio": 1.0,
        },
        "sitting": {
            "hr": ref_hr,
            "hrv": ref_hrv,
            "spo2": ref_spo2,
            "min_spo2": ref_spo2 - 1,
            "sbp": ref_sbp,
            "dbp": ref_dbp,
            "steps": max(ref_steps_30m * 0.12, 5),
            "stress": 30,
            "fatigue": 30,
            "sleep_ratio": 1.0,
        },
        "elevated": {
            "hr": ref_hr + hr_tol * 1.1,
            "hrv": max(ref_hrv - hrv_tol * 1.3, 1),
            "spo2": ref_spo2 - 2,
            "min_spo2": ref_spo2 - 3,
            "sbp": ref_sbp - sbp_tol * 0.8,
            "dbp": ref_dbp - dbp_tol * 0.5,
            "steps": max(ref_steps_30m * 0.18, 8),
            "stress": 68,
            "fatigue": 65,
            "sleep_ratio": 0.7,
        },
        "high": {
            "hr": ref_hr + hr_tol * 1.8,
            "hrv": max(ref_hrv - hrv_tol * 2.0, 1),
            "spo2": min(ref_spo2 - 5, 91),
            "min_spo2": 90,
            "sbp": ref_sbp - sbp_tol * 1.5,
            "dbp": ref_dbp - dbp_tol,
            "steps": max(ref_steps_30m * 0.05, 2),
            "stress": 86,
            "fatigue": 82,
            "sleep_ratio": 0.45,
        },
        "recovery": {
            "hr": ref_hr - 2,
            "hrv": ref_hrv + 4,
            "spo2": ref_spo2,
            "min_spo2": ref_spo2 - 1,
            "sbp": ref_sbp,
            "dbp": ref_dbp,
            "steps": max(ref_steps_30m * 0.75, 20),
            "stress": 22,
            "fatigue": 20,
            "sleep_ratio": 1.05,
        },
    }[scenario]

    sleep_ratio = scenario_values["sleep_ratio"]
    daily_sleep = max(ref_sleep * sleep_ratio, 0)
    sleep_deficit = max(0.0, min(1.0, 1.0 - sleep_ratio))
    activity_ratio = max(0.0, min(2.0, scenario_values["steps"] / ref_steps_30m))

    row = {
        "resident_id": resident_id,
        "generated_at": generated_at,
        "synthetic_scenario": scenario,
        "avg_hr": scenario_values["hr"],
        "max_hr": scenario_values["hr"] + 4,
        "min_hr": scenario_values["hr"] - 4,
        "hr_std": 2.0,
        "avg_hrv": scenario_values["hrv"],
        "avg_spo2": scenario_values["spo2"],
        "min_spo2": scenario_values["min_spo2"],
        "avg_sbp": scenario_values["sbp"],
        "avg_dbp": scenario_values["dbp"],
        "avg_temp": ref_temp,
        "avg_stress": scenario_values["stress"],
        "avg_fatigue": scenario_values["fatigue"],
        "avg_breathing": 18.0 if scenario != "high" else 23.0,
        "steps_30m": scenario_values["steps"],
        "step_std": max(scenario_values["steps"] * 0.08, 0.5),
        "activity_ratio": activity_ratio,
        "daily_total_sleep_minutes": daily_sleep,
        "sleep_baseline_minutes": ref_sleep,
        "sleep_ratio": sleep_ratio,
        "sleep_deficit": sleep_deficit,
        "sleep_data_available": 1,
        "hrv_availability": 1.0,
        "spo2_availability": 1.0,
    }

    for column in baseline_row.index:
        row[column] = baseline_row[column]
    row["resident_id"] = resident_id
    row["generated_at"] = generated_at
    return row


def synthetic_raw_row(feature_row):
    return pd.Series(
        {
            "resident_id": feature_row["resident_id"],
            "generated_at": feature_row["generated_at"],
            "heart_rate": feature_row["avg_hr"],
            "hrv": feature_row["avg_hrv"],
            "systolic_bp": feature_row["avg_sbp"],
            "diastolic_bp": feature_row["avg_dbp"],
            "oxygen_saturation": feature_row["avg_spo2"],
            "oxygen_saturation_valid": True,
            "body_temperature": feature_row["avg_temp"],
            "step_count": feature_row["steps_30m"],
            "deep_sleep_time": feature_row["daily_total_sleep_minutes"] * 0.25,
            "light_sleep_time": feature_row["daily_total_sleep_minutes"] * 0.75,
            "fatigue_level": feature_row["avg_fatigue"],
            "stress": feature_row["avg_stress"],
            "breathing": feature_row["avg_breathing"],
        }
    )


def choose_scenario(index, requested):
    if requested != "mixed":
        return requested
    return SCENARIOS[index % len(SCENARIOS)]


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic layered V1 pipeline test using resident IDs from prediction history."
    )
    parser.add_argument("--history-db", default="fall_risk_prediction_history.db")
    parser.add_argument("--artifact", default="fall_risk_xgb_baseline_v1.pkl")
    parser.add_argument(
        "--scenario",
        choices=["mixed"] + SCENARIOS,
        default="mixed",
    )
    parser.add_argument("--resident-id", type=int, default=None)
    parser.add_argument("--output-json", default="synthetic_layered_pipeline_test.json")
    parser.add_argument(
        "--public-output-json",
        default="synthetic_public_layered_pipeline_test.json",
    )
    parser.add_argument("--output-csv", default="synthetic_layered_pipeline_test_summary.csv")
    parser.add_argument("--disable-shap", action="store_true")
    parser.add_argument("--shap-top-n", type=int, default=5)
    args = parser.parse_args()

    resident_ids = get_history_resident_ids(args.history_db)
    if args.resident_id is not None:
        resident_ids = [rid for rid in resident_ids if rid == args.resident_id]
    if not resident_ids:
        raise SystemExit("No resident IDs found for synthetic test.")

    engine = build_engine()
    baseline_df = get_resident_baselines(
        engine=engine,
        skip_engineering_api=True,
        use_sqlite_cache=True,
        use_database_fallback=True,
    )
    baseline_lookup = resident_baseline_lookup(baseline_df)
    generated_at = pd.Timestamp.now(tz="UTC")

    feature_rows = []
    for index, resident_id in enumerate(resident_ids):
        if resident_id not in baseline_lookup:
            continue
        scenario = choose_scenario(index, args.scenario)
        feature_rows.append(
            synthetic_feature_row(
                resident_id,
                baseline_lookup[resident_id],
                scenario,
                generated_at + pd.Timedelta(seconds=index),
            )
        )

    features_df = pd.DataFrame(feature_rows)
    if features_df.empty:
        raise SystemExit("No synthetic feature rows could be created.")

    artifact = joblib.load(args.artifact)
    predictions = predict_rows(features_df, artifact, mode="history")

    if args.disable_shap:
        shap_explanations = {}
        shap_fallback = unavailable_shap_payload("SHAP disabled by CLI")
    else:
        shap_explanations, shap_fallback = build_shap_explanations(
            features_df,
            predictions,
            artifact,
            top_n=args.shap_top_n,
        )

    prediction_index = predictions.set_index(["resident_id", "generated_at"])
    records = []
    public_records = []
    summary_rows = []
    for _, feature_row in features_df.iterrows():
        key = (feature_row["resident_id"], feature_row["generated_at"])
        prediction_row = prediction_index.loc[key]
        if isinstance(prediction_row, pd.DataFrame):
            prediction_row = prediction_row.iloc[-1]

        shap_payload = shap_explanations.get(key)
        if shap_payload is None:
            shap_payload = shap_fallback or unavailable_shap_payload(
                "No SHAP explanation found for this synthetic row"
            )

        record = build_pipeline_record(
            feature_row=feature_row,
            raw_row=synthetic_raw_row(feature_row),
            prediction_row=prediction_row,
            history_db=args.history_db,
            shap_payload=shap_payload,
        )
        record["syntheticTestLayer"] = {
            "isSynthetic": True,
            "scenario": feature_row["synthetic_scenario"],
            "source": "simulate_layered_pipeline_synthetic.py",
        }
        records.append(record)
        public_record = public_pipeline_record(record)
        public_record["syntheticTest"] = record["syntheticTestLayer"]
        public_records.append(public_record)
        summary_rows.append(
            {
                "resident_id": record["residentId"],
                "scenario": feature_row["synthetic_scenario"],
                "final_risk": record["resolvedRiskLayer"]["finalRisk"],
                "rule_risk": record["ruleEngineLayer"]["risk"],
                "v1_model_risk": record["v1ModelLayer"]["label"],
                "model_confidence": record["v1ModelLayer"]["confidence"],
                "trend_24h": (record["trendLayer"]["latest24h"] or {}).get(
                    "trend_direction"
                ),
                "top_shap": "; ".join(
                    item["feature"]
                    for item in record["modelExplainabilityLayer"].get(
                        "topPositiveDrivers",
                        [],
                    )[:3]
                ),
                "caregpt": record["careGptLayer"]["finalInsight"],
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

    print("\n=== Synthetic Layered Pipeline Test ===")
    print(summary_df.drop(columns=["caregpt"]).to_string(index=False))
    print(f"\nSaved synthetic JSON to {args.output_json}")
    print(f"Saved synthetic public JSON to {args.public_output_json}")
    print(f"Saved synthetic summary to {args.output_csv}")


if __name__ == "__main__":
    main()
