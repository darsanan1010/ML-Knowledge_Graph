import argparse
import json
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import joblib
import jwt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.engine import URL

from resident_context_cache import get_resident_baselines


env_path = Path(__file__).parent / "env1.env"
load_dotenv(env_path)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "mydb4")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Metrok@1357")
RESIDENT_CACHE_DB_PATH = os.getenv("DB_PATH", "./cache.db")

DEFAULT_REFERENCE_STEP_COUNT = 2000.0
REFERENCE_ACTIVE_WINDOWS_PER_DAY = 16.0
MAX_STEP_INCREMENT = 5000.0
MAX_STEPS_30M = 6000.0
MAX_STRESS_SCORE = 100.0
MAX_FATIGUE_SCORE = 100.0
RESIDENT_PAGE_SIZE = 200
DEFAULT_TRAINING_START_AT = "2026-04-10T00:00:00+00:00"
DEFAULT_TRAINING_END_AT = "2026-05-01T23:59:59+00:00"

ENGINEERING_API_URL = os.getenv("URL")
CARE_MP_SERVICE_SECRET = os.getenv("CARE_MP_SERVICE_SECRET")
ROLE = os.getenv("CARE_MP_SERVICE_ROLE", "caremp-ml-engine")
CARE_MP_SERVICE_ISS = os.getenv("CARE_MP_SERVICE_ISS")
CARE_MP_SERVICE_AUD = os.getenv("CARE_MP_SERVICE_AUD", "caremp-api")


def build_engine():
    db_url = URL.create(
        "postgresql+psycopg2",
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
    )
    return create_engine(db_url)


def safe_ratio(numerator, denominator):
    denominator = denominator.replace(0, np.nan)
    return numerator.div(denominator).replace([np.inf, -np.inf], 0).fillna(0)


def generate_engineering_token():
    if not CARE_MP_SERVICE_SECRET:
        raise ValueError("CARE_MP_SERVICE_SECRET is required when URL is configured")

    now = int(time.time())
    payload = {
        "iss": CARE_MP_SERVICE_ISS,
        "aud": CARE_MP_SERVICE_AUD,
        "role": ROLE,
        "iat": now,
        "exp": now + 60,
    }
    return jwt.encode(payload, CARE_MP_SERVICE_SECRET, algorithm="HS256")


def first_present(data: dict, keys: list[str]):
    for key in keys:
        value = data.get(key)
        if value not in [None, ""]:
            return value
    return None


def compute_age_from_dob(dob_value):
    if dob_value in [None, ""]:
        return None

    dob = pd.to_datetime(dob_value, errors="coerce", utc=True)
    if pd.isna(dob):
        return None

    today = pd.Timestamp.utcnow().date()
    birth_date = dob.date()
    age = today.year - birth_date.year
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        age -= 1
    if age < 0 or age > 130:
        return None
    return age


def normalize_resident_age(resident):
    existing_age = first_present(resident, ["age", "residentAge"])
    if existing_age not in [None, ""]:
        age = pd.to_numeric(existing_age, errors="coerce")
        if pd.notna(age) and 0 < age <= 130:
            return int(age)

    dob = first_present(
        resident,
        [
            "dob",
            "dateOfBirth",
            "date_of_birth",
            "birthDate",
            "birth_date",
            "residentDob",
            "residentDOB",
        ],
    )
    return compute_age_from_dob(dob)


def normalize_resident_context(resident):
    dob = first_present(
        resident,
        [
            "dob",
            "dateOfBirth",
            "date_of_birth",
            "birthDate",
            "birth_date",
            "residentDob",
            "residentDOB",
        ],
    )
    return {
        "resident_id": resident.get("id") or resident.get("resident_id"),
        "resident_age": normalize_resident_age(resident),
        "resident_dob": dob,
        "medical_history": resident.get("medicalHistory"),
        "resident_condition": resident.get("condition"),
        "reference_heart_rate": resident.get("referenceHeartRate"),
        "heart_rate_tolerance": resident.get("heartRateTolerance"),
        "reference_hrv": (
            resident.get("referenceHrv")
            if resident.get("referenceHrv") is not None
            else resident.get("referenceHRV")
        ),
        "hrv_tolerance": resident.get("hrvTolerance"),
        "reference_systolic_bp": resident.get("referenceSystolicBP"),
        "systolic_bp_tolerance": resident.get("systolicBPTolerance"),
        "reference_diastolic_bp": resident.get("referenceDiastolicBP"),
        "diastolic_bp_tolerance": resident.get("diastolicBPTolerance"),
        "reference_oxygen_saturation": resident.get("referenceOxygenSaturation"),
        "oxygen_saturation_tolerance": resident.get("oxygenSaturationTolerance"),
        "reference_body_temperature": (
            resident.get("referenceBodyTemperature")
            if resident.get("referenceBodyTemperature") is not None
            else resident.get("referenceTemperature")
        ),
        "temperature_tolerance_celsius": (
            resident.get("temperatureToleranceCelsius")
            if resident.get("temperatureToleranceCelsius") is not None
            else resident.get("temperatureTolerance")
        ),
        "reference_step_count": resident.get("referenceStepCount"),
        "step_count_tolerance": resident.get("stepCountTolerance"),
        "reference_deep_sleep": resident.get("referenceDeepSleep"),
        "reference_light_sleep": resident.get("referenceLightSleep"),
    }


def init_resident_cache():
    with sqlite3.connect(RESIDENT_CACHE_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS residents (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
            """
        )


def store_residents_in_sqlite_cache(residents):
    if not residents:
        return

    init_resident_cache()
    stored = 0
    with sqlite3.connect(RESIDENT_CACHE_DB_PATH) as conn:
        for resident in residents:
            resident_id = resident.get("id") or resident.get("resident_id")
            if not resident_id:
                continue
            conn.execute(
                """
                INSERT OR REPLACE INTO residents (id, data, updated_at)
                VALUES (?, ?, ?)
                """,
                (
                    str(resident_id),
                    json.dumps(resident),
                    datetime.now().isoformat(),
                ),
            )
            stored += 1

    print(f"Cached {stored} residents in SQLite: {RESIDENT_CACHE_DB_PATH}")


def fetch_residents_from_sqlite_cache():
    cache_path = Path(RESIDENT_CACHE_DB_PATH)
    if not cache_path.exists():
        return None

    init_resident_cache()
    residents = []
    with sqlite3.connect(RESIDENT_CACHE_DB_PATH) as conn:
        rows = conn.execute("SELECT data FROM residents").fetchall()

    for (data_json,) in rows:
        try:
            residents.append(json.loads(data_json))
        except json.JSONDecodeError:
            continue

    if not residents:
        return None

    print(f"Loaded {len(residents)} residents from SQLite cache: {RESIDENT_CACHE_DB_PATH}")
    baseline_df = pd.DataFrame(
        normalize_resident_context(resident) for resident in residents
    )
    return baseline_df.dropna(subset=["resident_id"])


def extract_resident_page(payload):
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        page = payload.get("data") or payload.get("residents") or payload.get("items") or []
        pagination = payload.get("pagination") or {}
        return page, pagination.get("total")
    return [], None


def fetch_residents_from_engineering_api():
    if not ENGINEERING_API_URL:
        return None

    token = generate_engineering_token()
    headers = {"Authorization": f"Bearer {token}"}
    residents = []
    skip = 0

    while True:
        try:
            response = requests.get(
                ENGINEERING_API_URL,
                headers=headers,
                params={"skip": skip, "limit": RESIDENT_PAGE_SIZE, "showDeleted": False},
                timeout=20,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            print(
                "Engineering API unavailable; falling back to resident_vitals. "
                f"Reason: {exc}"
            )
            return None

        page, total = extract_resident_page(response.json())
        if not page:
            break

        residents.extend(page)
        if total is not None and len(residents) >= int(total):
            break
        if len(page) < RESIDENT_PAGE_SIZE:
            break

        skip += RESIDENT_PAGE_SIZE

    store_residents_in_sqlite_cache(residents)
    baseline_df = pd.DataFrame(normalize_resident_context(resident) for resident in residents)
    return baseline_df.dropna(subset=["resident_id"])


def fetch_residents_from_database(engine):
    query = """
    SELECT
        id AS resident_id,
        "referenceHeartRate" AS reference_heart_rate,
        "referenceSystolicBP" AS reference_systolic_bp,
        "referenceDiastolicBP" AS reference_diastolic_bp,
        "referenceOxygenSaturation" AS reference_oxygen_saturation,
        "referenceBodyTemperature" AS reference_body_temperature,
        "referenceStepCount" AS reference_step_count
    FROM resident_vitals
    WHERE "isDeleted" = false
    """
    return pd.read_sql(query, engine)


def fetch_band_logs(
    engine,
    hours,
    sleep_baseline_days,
    training_start_at,
    training_end_at,
    use_latest_available=True,
):
    lookback_days = max(sleep_baseline_days, int(np.ceil(hours / 24)))
    time_filter = (
        'AND "generatedAt" >= NOW() - (%(lookback_days)s * INTERVAL \'1 day\')'
        if not use_latest_available
        else ""
    )
    query = f"""
    SELECT
        "residentId" AS resident_id,
        "generatedAt" AS generated_at,
        "heartRate" AS heart_rate,
        "systolicBP" AS systolic_bp,
        "diastolicBP" AS diastolic_bp,
        "oxygenSaturation" AS oxygen_saturation,
        "oxygenSaturationValid" AS oxygen_saturation_valid,
        hrv,
        rri,
        "stepCount" AS step_count,
        "fatigueLevel" AS fatigue_level,
        stress,
        breathing,
        "bodyTemperature" AS body_temperature,
        "deepSleepTime" AS deep_sleep_time,
        "lightSleepTime" AS light_sleep_time
    FROM band_log
    WHERE (
        "generatedAt" < %(training_start_at)s
        OR "generatedAt" > %(training_end_at)s
    )
      {time_filter}
    ORDER BY "residentId", "generatedAt"
    """
    return pd.read_sql(
        query,
        engine,
        params={
            "lookback_days": lookback_days,
            "training_start_at": training_start_at,
            "training_end_at": training_end_at,
        },
    )


def prepare_baselines(baseline_df):
    baseline_df = baseline_df.copy()
    baseline_df["resident_id"] = pd.to_numeric(
        baseline_df["resident_id"], errors="coerce"
    ).astype("Int64")
    baseline_df = baseline_df.dropna(subset=["resident_id"]).drop_duplicates(
        "resident_id", keep="last"
    )

    default_references = {
        "reference_heart_rate": 75.0,
        "reference_hrv": 40.0,
        "reference_systolic_bp": 120.0,
        "reference_diastolic_bp": 80.0,
        "reference_oxygen_saturation": 98.0,
        "reference_body_temperature": 36.8,
        "reference_step_count": DEFAULT_REFERENCE_STEP_COUNT,
        "reference_deep_sleep": 100.0,
        "reference_light_sleep": 250.0,
    }
    default_tolerances = {
        "heart_rate_tolerance": 20.0,
        "hrv_tolerance": 10.0,
        "systolic_bp_tolerance": 20.0,
        "diastolic_bp_tolerance": 15.0,
        "oxygen_saturation_tolerance": 4.0,
        "temperature_tolerance_celsius": 1.5,
        "step_count_tolerance": 500.0,
    }

    for column, default_value in {**default_references, **default_tolerances}.items():
        if column not in baseline_df.columns:
            baseline_df[column] = default_value
        baseline_df[column] = pd.to_numeric(baseline_df[column], errors="coerce")
        baseline_df[column] = baseline_df[column].fillna(default_value)
        baseline_df.loc[baseline_df[column] <= 0, column] = default_value

    if "resident_age" not in baseline_df.columns:
        baseline_df["resident_age"] = pd.NA
    baseline_df["resident_age"] = pd.to_numeric(
        baseline_df["resident_age"], errors="coerce"
    )
    baseline_df.loc[
        (baseline_df["resident_age"] <= 0) | (baseline_df["resident_age"] > 130),
        "resident_age",
    ] = pd.NA

    if "resident_dob" not in baseline_df.columns:
        baseline_df["resident_dob"] = pd.NA

    for column in [
        "medical_history",
        "resident_condition",
    ]:
        if column not in baseline_df.columns:
            baseline_df[column] = ""
        baseline_df[column] = baseline_df[column].fillna("").astype(str)

    baseline_df["reference_total_sleep_minutes"] = (
        baseline_df["reference_deep_sleep"] + baseline_df["reference_light_sleep"]
    )
    return baseline_df


def create_rolling_features(df):
    df = df.copy().sort_values(["resident_id", "generated_at"])
    df = df.set_index("generated_at")

    value_columns = [
        "heart_rate",
        "hrv",
        "spo2",
        "systolic_bp",
        "diastolic_bp",
        "body_temperature",
        "stress",
        "fatigue_level",
        "breathing",
        "step_increment",
    ]

    def per_resident(group):
        features = pd.DataFrame(index=group.index)
        features["resident_id"] = group.name
        features["avg_hr"] = group["heart_rate"].rolling("30min").mean()
        features["max_hr"] = group["heart_rate"].rolling("30min").max()
        features["min_hr"] = group["heart_rate"].rolling("30min").min()
        features["hr_std"] = group["heart_rate"].rolling("30min").std().fillna(0)
        features["avg_hrv"] = group["hrv"].rolling("30min").mean()
        features["avg_spo2"] = group["spo2"].rolling("30min").mean()
        features["min_spo2"] = group["spo2"].rolling("30min").min()
        features["avg_sbp"] = group["systolic_bp"].rolling("30min").mean()
        features["avg_dbp"] = group["diastolic_bp"].rolling("30min").mean()
        features["avg_temp"] = group["body_temperature"].rolling("30min").mean()
        features["avg_stress"] = group["stress"].rolling("30min").mean()
        features["avg_fatigue"] = group["fatigue_level"].rolling("30min").mean()
        features["avg_breathing"] = group["breathing"].rolling("30min").mean()
        features["steps_30m"] = (
            group["step_increment"].rolling("30min").sum().clip(upper=MAX_STEPS_30M)
        )
        features["step_std"] = group["step_increment"].rolling("30min").std().fillna(0)
        return features

    return (
        df.groupby("resident_id", group_keys=False)[value_columns]
        .apply(per_resident)
        .reset_index()
    )


def create_quality_features(df, value_column, prefix):
    df = df.copy().sort_values(["resident_id", "generated_at"])
    df[f"{prefix}_valid"] = df[value_column].notna().astype(int)
    df = df.set_index("generated_at")

    def per_resident(group):
        quality = pd.DataFrame(index=group.index)
        quality["resident_id"] = group.name
        quality[f"valid_{prefix}_count"] = group[f"{prefix}_valid"].rolling("30min").sum()
        quality[f"total_{prefix}_count"] = group[f"{prefix}_valid"].rolling("30min").count()
        quality[f"{prefix}_availability"] = safe_ratio(
            quality[f"valid_{prefix}_count"],
            quality[f"total_{prefix}_count"],
        )
        return quality

    return (
        df.groupby("resident_id", group_keys=False)[[value_column, f"{prefix}_valid"]]
        .apply(per_resident)
        .reset_index()
    )


def build_prediction_features(band_df, baseline_df, hours, use_latest_available=False):
    band_df = band_df.copy()
    band_df["resident_id"] = pd.to_numeric(
        band_df["resident_id"], errors="coerce"
    ).astype("Int64")
    band_df = band_df.dropna(subset=["resident_id"])
    band_df["generated_at"] = pd.to_datetime(band_df["generated_at"])
    band_df = (
        band_df.sort_values(["resident_id", "generated_at"])
        .drop_duplicates(["resident_id", "generated_at"], keep="last")
    )

    if band_df.empty:
        return pd.DataFrame()

    band_df["hrv"] = band_df["hrv"].replace(0, np.nan)
    band_df["fatigue_level"] = pd.to_numeric(band_df["fatigue_level"], errors="coerce")
    band_df.loc[
        (band_df["fatigue_level"] < 0) | (band_df["fatigue_level"] >= 255),
        "fatigue_level",
    ] = np.nan
    band_df["fatigue_level"] = band_df["fatigue_level"].clip(
        lower=0, upper=MAX_FATIGUE_SCORE
    )

    band_df["stress"] = pd.to_numeric(band_df["stress"], errors="coerce")
    band_df.loc[
        (band_df["stress"] < 0) | (band_df["stress"] > MAX_STRESS_SCORE),
        "stress",
    ] = np.nan
    band_df["stress"] = band_df["stress"].clip(lower=0, upper=MAX_STRESS_SCORE)

    band_df["spo2"] = np.where(
        band_df["oxygen_saturation_valid"].fillna(False).astype(bool),
        band_df["oxygen_saturation"],
        np.nan,
    )

    band_df["log_date"] = band_df["generated_at"].dt.date
    band_df["previous_log_date"] = band_df.groupby("resident_id")["log_date"].shift()
    band_df["raw_step_increment"] = band_df.groupby("resident_id")["step_count"].diff()
    band_df["is_new_step_day"] = (
        band_df["previous_log_date"].notna()
        & band_df["log_date"].ne(band_df["previous_log_date"])
    )
    band_df["step_increment"] = np.where(
        band_df["is_new_step_day"], band_df["step_count"], band_df["raw_step_increment"]
    )
    band_df["step_increment"] = np.where(
        (~band_df["is_new_step_day"]) & (band_df["raw_step_increment"] < 0),
        0,
        band_df["step_increment"],
    )
    band_df["step_increment"] = (
        band_df["step_increment"].fillna(0).clip(lower=0, upper=MAX_STEP_INCREMENT)
    )

    band_df["deep_sleep_minutes"] = band_df["deep_sleep_time"].fillna(0)
    band_df["light_sleep_minutes"] = band_df["light_sleep_time"].fillna(0)
    band_df["total_sleep_minutes"] = (
        band_df["deep_sleep_minutes"] + band_df["light_sleep_minutes"]
    )

    daily_sleep_df = (
        band_df.groupby(["resident_id", "log_date"], as_index=False)
        .agg(
            daily_deep_sleep_minutes=("deep_sleep_minutes", "max"),
            daily_light_sleep_minutes=("light_sleep_minutes", "max"),
            daily_total_sleep_minutes=("total_sleep_minutes", "max"),
        )
    )
    daily_sleep_df["sleep_data_available"] = (
        daily_sleep_df["daily_total_sleep_minutes"] > 0
    ).astype(int)

    sleep_context_df = (
        daily_sleep_df[daily_sleep_df["sleep_data_available"].eq(1)]
        .groupby("resident_id", as_index=False)
        .agg(resident_median_sleep_minutes=("daily_total_sleep_minutes", "median"))
    )

    feature_df = create_rolling_features(band_df)
    hrv_quality = create_quality_features(band_df, "hrv", "hrv")
    spo2_quality = create_quality_features(band_df, "spo2", "spo2")

    dataset = feature_df.merge(baseline_df, on="resident_id", how="left")
    dataset = dataset.merge(hrv_quality, on=["resident_id", "generated_at"], how="left")
    dataset = dataset.merge(spo2_quality, on=["resident_id", "generated_at"], how="left")
    dataset["log_date"] = dataset["generated_at"].dt.date
    dataset = dataset.merge(daily_sleep_df, on=["resident_id", "log_date"], how="left")
    dataset = dataset.merge(sleep_context_df, on="resident_id", how="left")
    dataset["sleep_data_available"] = dataset["sleep_data_available"].fillna(0).astype(int)

    dataset["avg_hrv"] = dataset["avg_hrv"].fillna(dataset["reference_hrv"])
    dataset["avg_spo2"] = dataset["avg_spo2"].fillna(
        dataset["reference_oxygen_saturation"]
    )
    dataset["min_spo2"] = dataset["min_spo2"].fillna(
        dataset["reference_oxygen_saturation"]
    )

    for source_col in ["avg_stress", "avg_fatigue"]:
        resident_median = dataset.groupby("resident_id")[source_col].transform("median")
        global_median = dataset[source_col].median()
        if pd.isna(global_median):
            global_median = 0
        dataset[source_col] = (
            dataset[source_col].fillna(resident_median).fillna(global_median).fillna(0)
        )

    dataset["reference_steps_30m"] = (
        dataset["reference_step_count"] / REFERENCE_ACTIVE_WINDOWS_PER_DAY
    )
    dataset["activity_ratio"] = safe_ratio(
        dataset["steps_30m"], dataset["reference_steps_30m"]
    ).clip(lower=0, upper=2)

    dataset["sleep_baseline_minutes"] = dataset["resident_median_sleep_minutes"].where(
        dataset["resident_median_sleep_minutes"] > 0
    ).fillna(dataset["reference_total_sleep_minutes"])
    dataset["daily_total_sleep_for_ratio"] = dataset["daily_total_sleep_minutes"].where(
        dataset["sleep_data_available"].eq(1), dataset["sleep_baseline_minutes"]
    )
    dataset["sleep_ratio"] = safe_ratio(
        dataset["daily_total_sleep_for_ratio"], dataset["sleep_baseline_minutes"]
    )
    dataset["sleep_deficit"] = (1 - dataset["sleep_ratio"]).clip(lower=0, upper=1)

    if use_latest_available:
        return dataset.copy()

    cutoff = pd.Timestamp.now(tz=dataset["generated_at"].dt.tz)
    prediction_start = cutoff - pd.Timedelta(hours=hours)
    return dataset[dataset["generated_at"] >= prediction_start].copy()


def predict_rows(features_df, artifact, mode="latest"):
    model = artifact["model"]
    label_encoder = artifact["label_encoder"]
    feature_columns = artifact["feature_columns"]
    high_threshold = artifact.get("high_threshold", 0.70)

    missing_features = [col for col in feature_columns if col not in features_df.columns]
    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")

    if mode == "latest":
        prediction_rows = (
            features_df.sort_values("generated_at")
            .groupby("resident_id", as_index=False)
            .tail(1)
            .copy()
        )
    elif mode == "history":
        prediction_rows = features_df.sort_values(
            ["resident_id", "generated_at"]
        ).copy()
    else:
        raise ValueError("mode must be either 'latest' or 'history'")

    x_values = prediction_rows[feature_columns]
    base_pred = model.predict(x_values)
    proba = model.predict_proba(x_values)

    high_idx = label_encoder.transform(["High"])[0]
    moderate_idx = label_encoder.transform(["Moderate"])[0]

    final_pred = base_pred.copy()
    final_pred[
        (base_pred == high_idx) & (proba[:, high_idx] < high_threshold)
    ] = moderate_idx

    prediction_rows["predicted_risk_label"] = label_encoder.inverse_transform(final_pred)
    prediction_rows["prediction_confidence"] = proba.max(axis=1)

    for label in label_encoder.classes_:
        idx = label_encoder.transform([label])[0]
        prediction_rows[f"prob_{label}"] = proba[:, idx]

    output_columns = [
        "resident_id",
        "generated_at",
        "predicted_risk_label",
        "prediction_confidence",
    ]
    output_columns.extend(f"prob_{label}" for label in label_encoder.classes_)
    return prediction_rows[output_columns]


def predict_latest(features_df, artifact):
    return predict_rows(features_df, artifact, mode="latest")


def summarize_probability_distribution(predictions):
    probability_columns = [
        column for column in predictions.columns if column.startswith("prob_")
    ]
    probability_summary = predictions[probability_columns].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    label_counts = predictions["predicted_risk_label"].value_counts().rename("count")
    label_percent = (
        predictions["predicted_risk_label"]
        .value_counts(normalize=True)
        .mul(100)
        .rename("percent")
    )
    label_distribution = pd.concat([label_counts, label_percent], axis=1)
    return probability_summary, label_distribution


def add_confusion_metrics_if_labels_available(
    predictions,
    label_encoder,
    actual_labels_csv,
    actual_label_column,
    output_prefix,
):
    if not actual_labels_csv:
        return

    actual_df = pd.read_csv(actual_labels_csv)
    required_columns = {"resident_id", "generated_at", actual_label_column}
    missing_columns = required_columns.difference(actual_df.columns)
    if missing_columns:
        raise ValueError(
            f"Actual labels CSV is missing required columns: {sorted(missing_columns)}"
        )

    actual_df["generated_at"] = pd.to_datetime(actual_df["generated_at"])
    predictions_for_eval = predictions.copy()
    predictions_for_eval["generated_at"] = pd.to_datetime(
        predictions_for_eval["generated_at"]
    )

    eval_df = predictions_for_eval.merge(
        actual_df[["resident_id", "generated_at", actual_label_column]],
        on=["resident_id", "generated_at"],
        how="inner",
    )
    if eval_df.empty:
        print(
            "No prediction rows matched actual labels exactly. "
            "Confusion metrics were not generated."
        )
        return

    y_true = label_encoder.transform(eval_df[actual_label_column])
    y_pred = label_encoder.transform(eval_df["predicted_risk_label"])

    report_df = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report_path = f"{output_prefix}_classification_report.csv"
    report_df.to_csv(report_path)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=range(len(label_encoder.classes_)),
    )
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in label_encoder.classes_],
        columns=[f"predicted_{label}" for label in label_encoder.classes_],
    )
    matrix_path = f"{output_prefix}_confusion_matrix.csv"
    matrix_df.to_csv(matrix_path)

    print("\nClassification report:")
    print(report_df)
    print("\nConfusion matrix:")
    print(matrix_df)
    print(f"Saved classification report to {report_path}")
    print(f"Saved confusion matrix to {matrix_path}")


def prediction_model_version(artifact):
    return (
        artifact.get("model_version")
        or artifact.get("version")
        or artifact.get("metadata", {}).get("version")
        or "fall_risk_xgb_baseline_v1"
    )


def write_prediction_history(
    engine,
    predictions,
    model_version,
    training_start_at,
    training_end_at,
    table_name,
):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        raise ValueError(
            "history table name must contain only letters, numbers, and underscores"
        )

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id BIGSERIAL PRIMARY KEY,
        resident_id BIGINT NOT NULL,
        generated_at TIMESTAMPTZ NOT NULL,
        predicted_risk_label TEXT NOT NULL,
        prediction_confidence DOUBLE PRECISION NOT NULL,
        prob_critical DOUBLE PRECISION,
        prob_high DOUBLE PRECISION,
        prob_low DOUBLE PRECISION,
        prob_moderate DOUBLE PRECISION,
        model_version TEXT NOT NULL,
        training_start_at TIMESTAMPTZ,
        training_end_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE (resident_id, generated_at, model_version)
    )
    """
    upsert_sql = f"""
    INSERT INTO {table_name} (
        resident_id,
        generated_at,
        predicted_risk_label,
        prediction_confidence,
        prob_critical,
        prob_high,
        prob_low,
        prob_moderate,
        model_version,
        training_start_at,
        training_end_at
    )
    VALUES (
        :resident_id,
        :generated_at,
        :predicted_risk_label,
        :prediction_confidence,
        :prob_critical,
        :prob_high,
        :prob_low,
        :prob_moderate,
        :model_version,
        :training_start_at,
        :training_end_at
    )
    ON CONFLICT (resident_id, generated_at, model_version)
    DO UPDATE SET
        predicted_risk_label = EXCLUDED.predicted_risk_label,
        prediction_confidence = EXCLUDED.prediction_confidence,
        prob_critical = EXCLUDED.prob_critical,
        prob_high = EXCLUDED.prob_high,
        prob_low = EXCLUDED.prob_low,
        prob_moderate = EXCLUDED.prob_moderate,
        training_start_at = EXCLUDED.training_start_at,
        training_end_at = EXCLUDED.training_end_at
    """

    history_df = predictions.copy()
    history_df["model_version"] = model_version
    history_df["training_start_at"] = training_start_at
    history_df["training_end_at"] = training_end_at
    history_df["generated_at"] = pd.to_datetime(
        history_df["generated_at"]
    ).dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    history_df = history_df.rename(
        columns={
            "prob_Critical": "prob_critical",
            "prob_High": "prob_high",
            "prob_Low": "prob_low",
            "prob_Moderate": "prob_moderate",
        }
    )
    expected_columns = [
        "resident_id",
        "generated_at",
        "predicted_risk_label",
        "prediction_confidence",
        "prob_critical",
        "prob_high",
        "prob_low",
        "prob_moderate",
        "model_version",
        "training_start_at",
        "training_end_at",
    ]
    for column in expected_columns:
        if column not in history_df.columns:
            history_df[column] = None

    records = history_df[expected_columns].to_dict("records")
    with engine.begin() as conn:
        conn.execute(text(create_table_sql))
        for start in range(0, len(records), 1000):
            conn.execute(text(upsert_sql), records[start : start + 1000])

    print(f"Upserted {len(records)} prediction rows into {table_name}")


def write_prediction_history_sqlite(
    predictions,
    model_version,
    training_start_at,
    training_end_at,
    sqlite_path,
):
    history_df = predictions.copy()
    history_df["model_version"] = model_version
    history_df["training_start_at"] = training_start_at
    history_df["training_end_at"] = training_end_at
    history_df["generated_at"] = pd.to_datetime(
        history_df["generated_at"]
    ).dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    history_df = history_df.rename(
        columns={
            "prob_Critical": "prob_critical",
            "prob_High": "prob_high",
            "prob_Low": "prob_low",
            "prob_Moderate": "prob_moderate",
        }
    )

    expected_columns = [
        "resident_id",
        "generated_at",
        "predicted_risk_label",
        "prediction_confidence",
        "prob_critical",
        "prob_high",
        "prob_low",
        "prob_moderate",
        "model_version",
        "training_start_at",
        "training_end_at",
    ]
    for column in expected_columns:
        if column not in history_df.columns:
            history_df[column] = None

    records = history_df[expected_columns].to_dict("records")
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fall_risk_prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resident_id INTEGER NOT NULL,
                generated_at TEXT NOT NULL,
                predicted_risk_label TEXT NOT NULL,
                prediction_confidence REAL NOT NULL,
                prob_critical REAL,
                prob_high REAL,
                prob_low REAL,
                prob_moderate REAL,
                model_version TEXT NOT NULL,
                training_start_at TEXT,
                training_end_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (resident_id, generated_at, model_version)
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO fall_risk_prediction_history (
                resident_id,
                generated_at,
                predicted_risk_label,
                prediction_confidence,
                prob_critical,
                prob_high,
                prob_low,
                prob_moderate,
                model_version,
                training_start_at,
                training_end_at
            )
            VALUES (
                :resident_id,
                :generated_at,
                :predicted_risk_label,
                :prediction_confidence,
                :prob_critical,
                :prob_high,
                :prob_low,
                :prob_moderate,
                :model_version,
                :training_start_at,
                :training_end_at
            )
            ON CONFLICT (resident_id, generated_at, model_version)
            DO UPDATE SET
                predicted_risk_label = excluded.predicted_risk_label,
                prediction_confidence = excluded.prediction_confidence,
                prob_critical = excluded.prob_critical,
                prob_high = excluded.prob_high,
                prob_low = excluded.prob_low,
                prob_moderate = excluded.prob_moderate,
                training_start_at = excluded.training_start_at,
                training_end_at = excluded.training_end_at
            """,
            records,
        )

    print(
        "Upserted "
        f"{len(records)} prediction rows into SQLite fall_risk_prediction_history "
        f"at {sqlite_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Predict fall risk with the saved V1 baseline model."
    )
    parser.add_argument(
        "--artifact",
        default="fall_risk_xgb_baseline_v1.pkl",
        help="Path to the saved V1 model artifact.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Recent hours to generate live prediction rows for.",
    )
    parser.add_argument(
        "--sleep-baseline-days",
        type=int,
        default=30,
        help=(
            "Days of recent DB history used only to estimate resident sleep baseline. "
            "Set to 0 to rely on API/reference sleep fallback."
        ),
    )
    parser.add_argument(
        "--training-start-at",
        default=os.getenv("FALL_RISK_V1_TRAINING_START_AT", DEFAULT_TRAINING_START_AT),
        help=(
            "Start timestamp of the V1 training period. Rows from this timestamp "
            "through --training-end-at are excluded from prediction."
        ),
    )
    parser.add_argument(
        "--training-end-at",
        default=os.getenv("FALL_RISK_V1_TRAINING_END_AT", DEFAULT_TRAINING_END_AT),
        help=(
            "End timestamp of the V1 training period. Rows from --training-start-at "
            "through this timestamp are excluded from prediction."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default="fall_risk_predictions_v2.csv",
        help="Optional CSV output path for predictions.",
    )
    parser.add_argument(
        "--mode",
        choices=["latest", "history"],
        default="latest",
        help=(
            "latest predicts one latest row per resident. history predicts every "
            "eligible non-training feature row for trend/history storage."
        ),
    )
    parser.add_argument(
        "--write-history-db",
        action="store_true",
        help="Write predictions to PostgreSQL fall risk prediction history table.",
    )
    parser.add_argument(
        "--write-history-sqlite",
        action="store_true",
        help="Write predictions to a local SQLite fall risk prediction history table.",
    )
    parser.add_argument(
        "--history-sqlite-path",
        default="fall_risk_prediction_history.db",
        help="SQLite DB path for --write-history-sqlite.",
    )
    parser.add_argument(
        "--history-table",
        default="fall_risk_prediction_history",
        help="PostgreSQL table for --write-history-db.",
    )
    parser.add_argument(
        "--probability-summary-csv",
        default="fall_risk_probability_summary_v1.csv",
        help="CSV output path for prediction probability distribution summary.",
    )
    parser.add_argument(
        "--label-distribution-csv",
        default="fall_risk_label_distribution_v1.csv",
        help="CSV output path for predicted label distribution.",
    )
    parser.add_argument(
        "--actual-labels-csv",
        default=None,
        help=(
            "Optional CSV with resident_id, generated_at, and true risk label. "
            "If provided, classification report and confusion matrix are generated."
        ),
    )
    parser.add_argument(
        "--actual-label-column",
        default="risk_label",
        help="Column name in --actual-labels-csv containing the true label.",
    )
    parser.add_argument(
        "--metrics-output-prefix",
        default="fall_risk_v1",
        help="Prefix for optional classification report and confusion matrix CSV files.",
    )
    parser.add_argument(
        "--live-window",
        action="store_true",
        help=(
            "Use only rows inside the recent --hours window. By default, the script "
            "uses the latest available DB rows for pilot prediction."
        ),
    )
    parser.add_argument(
        "--skip-engineering-api",
        action="store_true",
        help="Use resident_vitals DB fallback directly instead of calling the Engineering API.",
    )
    args = parser.parse_args()

    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {artifact_path}. "
            "Place fall_risk_xgb_baseline_v1.pkl in this folder or pass --artifact."
        )

    artifact = joblib.load(artifact_path)
    engine = build_engine()

    band_df = fetch_band_logs(
        engine,
        args.hours,
        args.sleep_baseline_days,
        args.training_start_at,
        args.training_end_at,
        use_latest_available=not args.live_window,
    )
    if band_df.empty:
        print("No non-training band_log rows found for prediction.")
        print(f"Excluded training range: {args.training_start_at} to {args.training_end_at}")
        return

    baseline_df = get_resident_baselines(
        engine=engine,
        skip_engineering_api=args.skip_engineering_api,
        use_sqlite_cache=True,
        use_database_fallback=True,
    )

    features_df = build_prediction_features(
        band_df,
        baseline_df,
        args.hours,
        use_latest_available=not args.live_window,
    )
    print("\n=== FEATURE DATASET DEBUG ===")
    print("Feature rows:", len(features_df))
    print("Residents:", features_df["resident_id"].nunique())
    print("\nRows per resident:")
    print(features_df.groupby("resident_id").size())
    if features_df.empty:
        min_time = band_df["generated_at"].min()
        max_time = band_df["generated_at"].max()
        print("No prediction feature rows were generated.")
        print(f"Fetched band_log range: {min_time} to {max_time}")
        print(
            "The script defaults to latest-available mode. Increase "
            "--sleep-baseline-days if the DB range is too narrow."
        )
        return

    predictions = predict_rows(features_df, artifact, mode=args.mode)
    predictions.to_csv(args.output_csv, index=False)

    probability_summary, label_distribution = summarize_probability_distribution(
        predictions
    )
    probability_summary.to_csv(args.probability_summary_csv)
    label_distribution.to_csv(args.label_distribution_csv)

    print(predictions.to_string(index=False))
    print(f"Saved {len(predictions)} predictions to {args.output_csv}")
    print(f"Prediction mode: {args.mode}")
    print("\nProbability distribution summary:")
    print(probability_summary)
    print("\nPredicted label distribution:")
    print(label_distribution)
    print(f"Saved probability summary to {args.probability_summary_csv}")
    print(f"Saved label distribution to {args.label_distribution_csv}")

    add_confusion_metrics_if_labels_available(
        predictions,
        artifact["label_encoder"],
        args.actual_labels_csv,
        args.actual_label_column,
        args.metrics_output_prefix,
    )

    if args.write_history_db:
        write_prediction_history(
            engine,
            predictions,
            prediction_model_version(artifact),
            args.training_start_at,
            args.training_end_at,
            args.history_table,
        )

    if args.write_history_sqlite:
        write_prediction_history_sqlite(
            predictions,
            prediction_model_version(artifact),
            args.training_start_at,
            args.training_end_at,
            args.history_sqlite_path,
        )


if __name__ == "__main__":
    main()
