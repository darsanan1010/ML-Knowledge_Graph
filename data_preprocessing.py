import os
import time

import jwt
import numpy as np
import pandas as pd
import requests
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from dotenv import load_dotenv
from pathlib import Path

env_path  = Path(__file__).parent / "env1.env"
load_dotenv(env_path)
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "mydb3"
DB_USER = "postgres"
DB_PASSWORD = ""

DB_URL = URL.create(
    "postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
)
OUTPUT_CSV = "fall_risk_training_dataset5.csv"
DEFAULT_REFERENCE_STEP_COUNT = 2000.0
REFERENCE_ACTIVE_WINDOWS_PER_DAY = 16.0
MAX_STEP_INCREMENT = 5000.0
MAX_STEPS_30M = 6000.0
MAX_STRESS_SCORE = 100.0
MAX_FATIGUE_SCORE = 100.0
MAX_CLASS_WEIGHT = 50.0

ENGINEERING_API_URL = os.getenv("URL")
CARE_MP_SERVICE_SECRET = os.getenv("CARE_MP_SERVICE_SECRET")
ROLE = os.getenv("CARE_MP_SERVICE_ROLE", "caremp-ml-engine")
CARE_MP_SERVICE_ISS = os.getenv("CARE_MP_SERVICE_ISS")
CARE_MP_SERVICE_AUD = os.getenv("CARE_MP_SERVICE_AUD", "caremp-api")
RESIDENT_PAGE_SIZE = 200

engine = create_engine(DB_URL)

bandlog_query = """
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
WHERE "generatedAt" >= NOW() - INTERVAL '3 months'
ORDER BY "residentId", "generatedAt"
"""

baseline_query = """
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


def safe_ratio(numerator, denominator):
    denominator = denominator.replace(0, np.nan)
    return (
        numerator.div(denominator)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )


def safe_breach(deviation, tolerance):
    return safe_ratio(deviation.abs(), tolerance)


def generate_engineering_token():
    if not CARE_MP_SERVICE_SECRET:
        raise ValueError("CARE_MP_SERVICE_SECRET is required for ENGINEERING_API_URL")

    now = int(time.time())
    payload = {
        "iss": CARE_MP_SERVICE_ISS,
        "aud": CARE_MP_SERVICE_AUD,
        "role": ROLE,
        "iat": now,
        "exp": now + 60,
    }
    return jwt.encode(payload, CARE_MP_SERVICE_SECRET, algorithm="HS256")


def normalize_resident_context(resident):
    return {
        "resident_id": resident.get("id") or resident.get("resident_id"),
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
        "reference_deep_sleep_tolerance": resident.get("referenceDeepSleepTolerance"),
        "reference_light_sleep": resident.get("referenceLightSleep"),
        "reference_light_sleep_tolerance": resident.get("referenceLightSleepTolerance"),
    }


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
        response = requests.get(
            ENGINEERING_API_URL,
            headers=headers,
            params={"skip": skip, "limit": RESIDENT_PAGE_SIZE, "showDeleted": False},
            timeout=20,
        )
        response.raise_for_status()

        page, total = extract_resident_page(response.json())
        if not page:
            break

        residents.extend(page)
        if total is not None and len(residents) >= int(total):
            break
        if len(page) < RESIDENT_PAGE_SIZE:
            break

        skip += RESIDENT_PAGE_SIZE

    baseline_df = pd.DataFrame(
        normalize_resident_context(resident)
        for resident in residents
    )
    baseline_df = baseline_df.dropna(subset=["resident_id"])
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
        features["spo2_std"] = group["spo2"].rolling("30min").std().fillna(0)
        features["avg_sbp"] = group["systolic_bp"].rolling("30min").mean()
        features["avg_dbp"] = group["diastolic_bp"].rolling("30min").mean()
        features["avg_temp"] = group["body_temperature"].rolling("30min").mean()
        features["avg_stress"] = group["stress"].rolling("30min").mean()
        features["avg_fatigue"] = group["fatigue_level"].rolling("30min").mean()
        features["avg_breathing"] = group["breathing"].rolling("30min").mean()
        features["steps_30m"] = (
            group["step_increment"]
            .rolling("30min")
            .sum()
            .clip(upper=MAX_STEPS_30M)
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
        quality[f"valid_{prefix}_count"] = (
            group[f"{prefix}_valid"].rolling("30min").sum()
        )
        quality[f"total_{prefix}_count"] = (
            group[f"{prefix}_valid"].rolling("30min").count()
        )
        quality[f"missing_{prefix}_count"] = (
            quality[f"total_{prefix}_count"] - quality[f"valid_{prefix}_count"]
        )
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


band_df = pd.read_sql(bandlog_query, engine)
baseline_df = fetch_residents_from_engineering_api()
if baseline_df is None or baseline_df.empty:
    baseline_df = pd.read_sql(baseline_query, engine)
print("\n=== Sleep Baseline Statistics ===")

print(
    baseline_df[
        [
              "resident_id",
            "reference_deep_sleep",
            "reference_light_sleep"
        ]
    ].head(20)
)   

band_df["resident_id"] = pd.to_numeric(
    band_df["resident_id"],
    errors="coerce",
).astype("Int64")
baseline_df["resident_id"] = pd.to_numeric(
    baseline_df["resident_id"],
    errors="coerce",
).astype("Int64")
band_df = band_df.dropna(subset=["resident_id"])
baseline_df = (
    baseline_df.dropna(subset=["resident_id"])
    .drop_duplicates("resident_id", keep="last")
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
for column, default_value in default_references.items():
    if column not in baseline_df.columns:
        baseline_df[column] = default_value
    else:
        baseline_df[column] = baseline_df[column].fillna(default_value)

print(
    "Band log resident IDs:",
    sorted(band_df["resident_id"].dropna().unique())[:20],
)
print(
    "Reference resident IDs:",
    sorted(baseline_df["resident_id"].dropna().unique())[:20],
)

default_tolerances = {
    "heart_rate_tolerance": 20.0,
    "hrv_tolerance": 10.0,
    "systolic_bp_tolerance": 20.0,
    "diastolic_bp_tolerance": 15.0,
    "oxygen_saturation_tolerance": 4.0,
    "temperature_tolerance_celsius": 1.5,
    "step_count_tolerance": 500.0,
    "reference_deep_sleep_tolerance": 10.0,
    "reference_light_sleep_tolerance": 20.0,
}
for column, default_value in default_tolerances.items():
    if column not in baseline_df.columns:
        baseline_df[column] = default_value
    else:
        baseline_df[column] = baseline_df[column].fillna(default_value)

numeric_baseline_columns = list(default_references) + list(default_tolerances)
for column in numeric_baseline_columns:
    baseline_df[column] = pd.to_numeric(baseline_df[column], errors="coerce")
    fallback = default_references.get(column, default_tolerances.get(column))
    baseline_df[column] = baseline_df[column].fillna(fallback)
    baseline_df.loc[baseline_df[column] <= 0, column] = fallback

baseline_df["reference_deep_sleep_minutes"] = (
    baseline_df["reference_deep_sleep"].fillna(0) 
)
baseline_df["reference_light_sleep_minutes"] = (
    baseline_df["reference_light_sleep"].fillna(0) 
)
baseline_df["reference_total_sleep_minutes"] = (
    baseline_df["reference_deep_sleep_minutes"]
    + baseline_df["reference_light_sleep_minutes"]
)
print("\n=== Sleep Baselines After Conversion ===")
print(
    baseline_df[
        [
            "resident_id",
            "reference_deep_sleep",
            "reference_light_sleep",
            "reference_deep_sleep_minutes",
            "reference_light_sleep_minutes",
            "reference_total_sleep_minutes",
        ]
    ]
    .head(20)
)
band_df["generated_at"] = pd.to_datetime(band_df["generated_at"])
duplicate_count = band_df.duplicated(["resident_id", "generated_at"]).sum()
if duplicate_count:
    print(f"Dropping duplicate resident/timestamp rows: {duplicate_count}")
band_df = (
    band_df.sort_values(["resident_id", "generated_at"])
    .drop_duplicates(["resident_id", "generated_at"], keep="last")
)

# HRV 0 usually means invalid/unavailable, not a true physiological value.
band_df["hrv"] = band_df["hrv"].replace(0, np.nan)

band_df["fatigue_level"] = pd.to_numeric(
    band_df["fatigue_level"],
    errors="coerce",
)
band_df.loc[
    (band_df["fatigue_level"] < 0)
    | (band_df["fatigue_level"] >= 255),
    "fatigue_level",
] = np.nan
band_df["fatigue_level"] = band_df["fatigue_level"].clip(
    lower=0,
    upper=MAX_FATIGUE_SCORE,
)

band_df["stress"] = pd.to_numeric(band_df["stress"], errors="coerce")
band_df.loc[
    (band_df["stress"] < 0)
    | (band_df["stress"] > MAX_STRESS_SCORE),
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
    band_df["is_new_step_day"],
    band_df["step_count"],
    band_df["raw_step_increment"],
)
band_df["step_increment"] = np.where(
    (~band_df["is_new_step_day"]) & (band_df["raw_step_increment"] < 0),
    0,
    band_df["step_increment"],
)
band_df["step_increment"] = (
    band_df["step_increment"]
    .fillna(0)
    .clip(lower=0, upper=MAX_STEP_INCREMENT)
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
    .agg(
        resident_median_sleep_minutes=("daily_total_sleep_minutes", "median"),
    )
)

feature_df = create_rolling_features(band_df)
hrv_quality = create_quality_features(band_df, "hrv", "hrv")
spo2_quality = create_quality_features(band_df, "spo2", "spo2")

dataset = feature_df.merge(baseline_df, on="resident_id", how="left")

missing_reference_rate = dataset["reference_heart_rate"].isna().mean()
print(f"Missing resident reference coverage: {missing_reference_rate:.2%}")
if missing_reference_rate > 0:
    missing_resident_ids = sorted(
        dataset.loc[
            dataset["reference_heart_rate"].isna(),
            "resident_id",
        ]
        .dropna()
        .unique()
    )
    print(
        "WARNING: Missing reference/tolerance data for resident IDs:",
        missing_resident_ids[:20],
    )

dataset = dataset.merge(
    hrv_quality,
    on=["resident_id", "generated_at"],
    how="left",
)
dataset = dataset.merge(
    spo2_quality,
    on=["resident_id", "generated_at"],
    how="left",
)

dataset["log_date"] = dataset["generated_at"].dt.date
dataset = dataset.merge(
    daily_sleep_df,
    on=["resident_id", "log_date"],
    how="left",
)
dataset = dataset.merge(
    sleep_context_df,
    on="resident_id",
    how="left",
)
dataset["sleep_data_available"] = dataset["sleep_data_available"].fillna(0).astype(int)

dataset["hrv_imputed"] = (
    dataset["avg_hrv"].isna() | dataset["valid_hrv_count"].fillna(0).eq(0)
).astype(int)
dataset["avg_hrv"] = dataset["avg_hrv"].fillna(dataset["reference_hrv"])

dataset["spo2_imputed"] = (
    dataset["avg_spo2"].isna() | dataset["valid_spo2_count"].fillna(0).eq(0)
).astype(int)
dataset["avg_spo2"] = dataset["avg_spo2"].fillna(
    dataset["reference_oxygen_saturation"]
)
dataset["min_spo2"] = dataset["min_spo2"].fillna(
    dataset["reference_oxygen_saturation"]
)

for source_col, imputed_col in [
    ("avg_stress", "stress_imputed"),
    ("avg_fatigue", "fatigue_imputed"),
]:
    dataset[imputed_col] = dataset[source_col].isna().astype(int)
    resident_median = dataset.groupby("resident_id")[source_col].transform("median")
    global_median = dataset[source_col].median()
    if pd.isna(global_median):
        global_median = 0
    dataset[source_col] = (
        dataset[source_col]
        .fillna(resident_median)
        .fillna(global_median)
        .fillna(0)
    )

dataset["hr_dev"] = dataset["avg_hr"] - dataset["reference_heart_rate"]
dataset["hrv_dev"] = dataset["avg_hrv"] - dataset["reference_hrv"]
dataset["spo2_dev"] = dataset["avg_spo2"] - dataset["reference_oxygen_saturation"]
dataset["sbp_dev"] = dataset["avg_sbp"] - dataset["reference_systolic_bp"]
dataset["dbp_dev"] = dataset["avg_dbp"] - dataset["reference_diastolic_bp"]
dataset["temp_dev"] = dataset["avg_temp"] - dataset["reference_body_temperature"]
dataset["reference_steps_30m"] = (
    dataset["reference_step_count"]
    /
    REFERENCE_ACTIVE_WINDOWS_PER_DAY
)
dataset["step_count_tolerance_30m"] = (
    dataset["step_count_tolerance"]
    /
    REFERENCE_ACTIVE_WINDOWS_PER_DAY
)
dataset["step_dev"] = dataset["steps_30m"] - dataset["reference_steps_30m"]

dataset["activity_ratio"] = safe_ratio(
    dataset["steps_30m"],
    dataset["reference_steps_30m"],
).clip(lower=0, upper=2)
dataset["activity_factor"] = 1 + (1 - dataset["activity_ratio"]).clip(
    lower=0,
    upper=1,
)
dataset["sleep_baseline_minutes"] = dataset["resident_median_sleep_minutes"].where(
    dataset["resident_median_sleep_minutes"] > 0
).fillna(
    dataset["reference_total_sleep_minutes"]
)
dataset["daily_total_sleep_for_ratio"] = dataset["daily_total_sleep_minutes"].where(
    dataset["sleep_data_available"].eq(1),
    dataset["sleep_baseline_minutes"],
)
dataset["sleep_ratio"] = safe_ratio(
    dataset["daily_total_sleep_for_ratio"],
    dataset["sleep_baseline_minutes"],
)
print("\n=== Sleep Ratio Validation ===")
print(
    dataset[
        [
            "resident_id",
            "daily_total_sleep_minutes",
            "reference_total_sleep_minutes",
            "sleep_ratio",
        ]
    ]
    .head(20)
)
dataset["sleep_deficit"] = (1 - dataset["sleep_ratio"]).clip(lower=0, upper=1)
dataset["sleep_component"] = dataset["sleep_deficit"]

dataset["hr_breach"] = safe_breach(
    dataset["hr_dev"],
    dataset["heart_rate_tolerance"],
)
dataset["hrv_breach"] = safe_breach(dataset["hrv_dev"], dataset["hrv_tolerance"])
dataset["spo2_breach"] = safe_breach(
    dataset["spo2_dev"],
    dataset["oxygen_saturation_tolerance"],
)
dataset["sbp_breach"] = safe_breach(
    dataset["sbp_dev"],
    dataset["systolic_bp_tolerance"],
)
dataset["dbp_breach"] = safe_breach(
    dataset["dbp_dev"],
    dataset["diastolic_bp_tolerance"],
)
dataset["temp_breach"] = safe_breach(
    dataset["temp_dev"],
    dataset["temperature_tolerance_celsius"],
)
dataset["step_breach"] = safe_breach(
    dataset["step_dev"],
    dataset["step_count_tolerance_30m"],
)

dataset["weighted_hr_breach"] = dataset["hr_breach"] * dataset["activity_factor"]
dataset["weighted_hrv_breach"] = dataset["hrv_breach"] * dataset["activity_factor"]
dataset["weighted_sbp_breach"] = dataset["sbp_breach"] * dataset["activity_factor"]
dataset["weighted_dbp_breach"] = dataset["dbp_breach"] * dataset["activity_factor"]
dataset["weighted_spo2_breach"] = dataset["spo2_breach"]
dataset["weighted_step_breach"] = dataset["step_breach"]

dataset = dataset.sort_values(["resident_id", "generated_at"])
for source_col, slope_col in [
    ("avg_hr", "hr_slope"),
    ("avg_hrv", "hrv_slope"),
    ("avg_sbp", "sbp_slope"),
    ("avg_stress", "stress_slope"),
    ("avg_fatigue", "fatigue_slope"),
    ("avg_spo2", "spo2_slope"),
    ("steps_30m", "step_slope"),
]:
    dataset[slope_col] = dataset.groupby("resident_id")[source_col].diff().fillna(0)

dataset["fatigue_component"] = (
    dataset["avg_fatigue"]
    .fillna(0)
    .clip(lower=0, upper=MAX_FATIGUE_SCORE)
    /
    MAX_FATIGUE_SCORE
)
dataset["stress_component"] = (
    dataset["avg_stress"]
    .fillna(0)
    .clip(lower=0, upper=MAX_STRESS_SCORE)
    /
    MAX_STRESS_SCORE
)

dataset["hr_slope_component"] = dataset["hr_slope"].abs().clip(upper=20)
dataset["hrv_slope_component"] = dataset["hrv_slope"].abs().clip(upper=20)
dataset["spo2_slope_component"] = dataset["spo2_slope"].abs().clip(upper=10)
dataset["step_slope_component"] = dataset["step_slope"].abs().clip(upper=MAX_STEPS_30M)

dataset["risk_score"] = (
    0.20 * dataset["weighted_hr_breach"]
    + 0.20 * dataset["weighted_hrv_breach"]
    + 0.10 * dataset["weighted_sbp_breach"]
    + 0.10 * dataset["weighted_dbp_breach"]
    + 0.10 * dataset["weighted_spo2_breach"]
    + 0.15 * dataset["weighted_step_breach"]
    + 0.075 * dataset["fatigue_component"]
    + 0.075 * dataset["stress_component"]
    + 0.05 * dataset["sleep_component"]
    + 0.05 * dataset["hr_slope_component"]
    + 0.05 * dataset["hrv_slope_component"]
    + 0.05 * dataset["spo2_slope_component"]
    + 0.00005 * dataset["step_slope_component"]
)

dataset["risk_label"] = pd.cut(
    dataset["risk_score"],
    bins=[-np.inf, 1.0, 2.0, 3.5, np.inf],
    labels=["Low", "Moderate", "High", "Critical"],
)

label_counts = dataset["risk_label"].value_counts()
class_weights = {
    label: len(dataset) / (len(label_counts) * count)
    for label, count in label_counts.items()
}
dataset["class_weight"] = (
    dataset["risk_label"]
    .map(class_weights)
    .astype(float)
    .clip(upper=MAX_CLASS_WEIGHT)
)
print("Class weights:", class_weights)
print(f"Capped class weights at: {MAX_CLASS_WEIGHT}")

model_training_features = [
    "avg_hr",
    "max_hr",
    "min_hr",
    "hr_std",
    "avg_hrv",
    "avg_spo2",
    "min_spo2",
    "avg_sbp",
    "avg_dbp",
    "avg_temp",
    "avg_stress",
    "avg_fatigue",
    "avg_breathing",
    "steps_30m",
    "step_std",
    "activity_ratio",
    "daily_total_sleep_minutes",
    "sleep_baseline_minutes",
    "sleep_ratio",
    "sleep_deficit",
    "hrv_availability",
    "spo2_availability",
]

audit_columns = [
    "hr_dev",
    "hrv_dev",
    "spo2_dev",
    "sbp_dev",
    "dbp_dev",
    "temp_dev",
    "step_dev",
    "reference_steps_30m",
    "step_count_tolerance_30m",
    "activity_factor",
    "resident_median_sleep_minutes",
    "daily_total_sleep_for_ratio",
    "sleep_data_available",
    "daily_deep_sleep_minutes",
    "daily_light_sleep_minutes",
    "hrv_imputed",
    "spo2_imputed",
    "stress_imputed",
    "fatigue_imputed",
    "sleep_component",
    "hr_breach",
    "hrv_breach",
    "spo2_breach",
    "sbp_breach",
    "dbp_breach",
    "temp_breach",
    "step_breach",
    "weighted_hr_breach",
    "weighted_hrv_breach",
    "weighted_sbp_breach",
    "weighted_dbp_breach",
    "weighted_spo2_breach",
    "weighted_step_breach",
    "hr_slope",
    "hrv_slope",
    "sbp_slope",
    "stress_slope",
    "fatigue_slope",
    "spo2_slope",
    "step_slope",
]

training_features = [
    *model_training_features,
    *audit_columns,
]

model_feature_manifest = pd.DataFrame(
    {"feature": model_training_features}
)
model_feature_manifest.to_csv("model_training_features.csv", index=False)
target_columns = ["risk_score", "risk_label", "class_weight"]

final_training_dataset = dataset[
    ["resident_id", "generated_at"] + training_features + target_columns
]

final_training_dataset.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(final_training_dataset)} rows to {OUTPUT_CSV}")
print(final_training_dataset.describe(include="all"))
print(final_training_dataset["risk_label"].value_counts())
