import os
import time

import jwt
import numpy as np
import pandas as pd
import requests
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import OperationalError
from sqlalchemy.engine import URL
from dotenv import load_dotenv
from pathlib import Path

env_path  = Path(__file__).parent / "env1.env"
load_dotenv(env_path)
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "mydb4"
DB_USER = "postgres"
DB_PASSWORD = "Metrok@1357"

DB_URL = URL.create(
    "postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
)
OUTPUT_CSV = "fall_risk_training_dataset_v23.csv"
SYNTHETIC_BANDLOG_CSV = os.getenv("SYNTHETIC_BANDLOG_CSV")
SYNTHETIC_BASELINE_CSV = os.getenv("SYNTHETIC_BASELINE_CSV")
DEFAULT_REFERENCE_STEP_COUNT = 2000.0
REFERENCE_ACTIVE_WINDOWS_PER_DAY = 16.0
MAX_STEP_INCREMENT = 5000.0
MAX_STEPS_30M = 6000.0
MAX_STRESS_SCORE = 100.0
MAX_FATIGUE_SCORE = 100.0
MAX_CLASS_WEIGHT = 50.0
EWMA_SPAN = 12
MAX_SLEEP_CONFIDENCE = 100.0

ENGINEERING_API_URL = os.getenv("URL")
CARE_MP_SERVICE_SECRET = os.getenv("CARE_MP_SERVICE_SECRET")
ROLE = os.getenv("CARE_MP_SERVICE_ROLE", "caremp-ml-engine")
CARE_MP_SERVICE_ISS = os.getenv("CARE_MP_SERVICE_ISS")
CARE_MP_SERVICE_AUD = os.getenv("CARE_MP_SERVICE_AUD", "caremp-api")
RESIDENT_PAGE_SIZE = 200
C2_BAND_IDS = {
    value.strip()
    for value in os.getenv("C2_BAND_IDS", "").split(",")
    if value.strip()
}
S6_BAND_IDS = {
    value.strip()
    for value in os.getenv("S6_BAND_IDS", "").split(",")
    if value.strip()
}
ADVANCED_SLEEP_BAND_IDS = C2_BAND_IDS | S6_BAND_IDS

engine = create_engine(DB_URL)
try:
    band_log_columns = {
        column["name"]
        for column in inspect(engine).get_columns("band_log")
    }
except OperationalError as exc:
    raise SystemExit(
        "Could not connect to PostgreSQL for data_preprocessing1.py.\n"
        "Add the database credentials to env1.env, for example:\n\n"
        "DB_HOST=localhost\n"
        "DB_PORT=5432\n"
        "DB_NAME=mydb3\n"
        "DB_USER=postgres\n"
        "DB_PASSWORD=your_postgres_password\n\n"
        "The current connection reached PostgreSQL but authentication failed."
    ) from exc


def select_optional_column(candidates, alias, default_sql="NULL"):
    for column_name in candidates:
        if column_name in band_log_columns:
            return f'"{column_name}" AS {alias}'
    return f"{default_sql} AS {alias}"


def load_optional_synthetic_csv(path, column_map, required_columns):
    if not path:
        return pd.DataFrame(columns=required_columns)

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Synthetic CSV not found: {csv_path}")

    synthetic_df = pd.read_csv(csv_path)
    synthetic_df = synthetic_df.rename(columns=column_map)

    for column in required_columns:
        if column not in synthetic_df.columns:
            synthetic_df[column] = np.nan

    return synthetic_df[required_columns]

band_id_select = select_optional_column(["bandId", "band_id"], "band_id")
rem_sleep_select = select_optional_column(
    ["remSleepTime", "rem_sleep_time", "remSleepMinutes", "remSleep"],
    "rem_sleep_time",
)
sleep_confidence_select = select_optional_column(
    ["sleepConfidence", "sleep_confidence"],
    "sleep_confidence",
)
sleep_status_select = select_optional_column(
    [
        "sleepComputationStatus",
        "sleepComputationState",
        "sleep_computation_status",
        "sleep_computation_state",
    ],
    "sleep_computation_status",
)
sleep_score_select = select_optional_column(
    ["sleepComputationScore", "sleep_computation_score"],
    "sleep_computation_score",
)

bandlog_query = f"""
SELECT
    "residentId" AS resident_id,
    "generatedAt" AS generated_at,
    {band_id_select},

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
    "lightSleepTime" AS light_sleep_time,
    {rem_sleep_select},
    {sleep_confidence_select},
    {sleep_status_select},
    {sleep_score_select}

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

    def rolling_mean(group, column, window):
        return group[column].rolling(window).mean()

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

        features["ewma_hr"] = group["heart_rate"].ewm(
            span=EWMA_SPAN,
            adjust=False,
        ).mean()
        features["ewma_hrv"] = group["hrv"].ewm(
            span=EWMA_SPAN,
            adjust=False,
        ).mean()
        features["ewma_spo2"] = group["spo2"].ewm(
            span=EWMA_SPAN,
            adjust=False,
        ).mean()
        features["ewma_sbp"] = group["systolic_bp"].ewm(
            span=EWMA_SPAN,
            adjust=False,
        ).mean()
        features["ewma_dbp"] = group["diastolic_bp"].ewm(
            span=EWMA_SPAN,
            adjust=False,
        ).mean()
        features["ewma_stress"] = group["stress"].ewm(
            span=EWMA_SPAN,
            adjust=False,
        ).mean()
        features["ewma_fatigue"] = group["fatigue_level"].ewm(
            span=EWMA_SPAN,
            adjust=False,
        ).mean()
        features["ewma_steps"] = group["step_increment"].ewm(
            span=EWMA_SPAN,
            adjust=False,
        ).mean()

        features["avg_hr_2h"] = rolling_mean(group, "heart_rate", "2h")
        features["avg_hrv_2h"] = rolling_mean(group, "hrv", "2h")
        features["avg_spo2_2h"] = rolling_mean(group, "spo2", "2h")
        features["min_spo2_2h"] = group["spo2"].rolling("2h").min()
        features["steps_2h"] = group["step_increment"].rolling("2h").sum()
        features["avg_stress_2h"] = rolling_mean(group, "stress", "2h")
        features["avg_fatigue_2h"] = rolling_mean(group, "fatigue_level", "2h")

        features["avg_hr_6h"] = rolling_mean(group, "heart_rate", "6h")
        features["avg_hrv_6h"] = rolling_mean(group, "hrv", "6h")
        features["avg_spo2_6h"] = rolling_mean(group, "spo2", "6h")
        features["min_spo2_6h"] = group["spo2"].rolling("6h").min()
        features["steps_6h"] = group["step_increment"].rolling("6h").sum()
        features["avg_stress_6h"] = rolling_mean(group, "stress", "6h")
        features["avg_fatigue_6h"] = rolling_mean(group, "fatigue_level", "6h")

        features["avg_hr_12h"] = rolling_mean(group, "heart_rate", "12h")
        features["avg_hrv_12h"] = rolling_mean(group, "hrv", "12h")
        features["avg_spo2_12h"] = rolling_mean(group, "spo2", "12h")
        features["min_spo2_12h"] = group["spo2"].rolling("12h").min()
        features["steps_12h"] = group["step_increment"].rolling("12h").sum()
        features["avg_stress_12h"] = rolling_mean(group, "stress", "12h")
        features["avg_fatigue_12h"] = rolling_mean(group, "fatigue_level", "12h")

        features["coverage_2h"] = (group["heart_rate"].rolling("2h").count() / 120.0).clip(upper=1.0).round(2)
        features["coverage_6h"] = (group["heart_rate"].rolling("6h").count() / 360.0).clip(upper=1.0).round(2)
        features["coverage_12h"] = (group["heart_rate"].rolling("12h").count() / 720.0).clip(upper=1.0).round(2)

        time_gaps = pd.Series(group.index).diff().dt.total_seconds() / 60.0
        time_gaps.index = group.index
        features["max_gap_minutes"] = time_gaps.rolling("12h").max().round(1)

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
band_df["is_synthetic"] = 0
baseline_df = fetch_residents_from_engineering_api()
if baseline_df is None or baseline_df.empty:
    baseline_df = pd.read_sql(baseline_query, engine)

bandlog_column_map = {
    "residentId": "resident_id",
    "generatedAt": "generated_at",
    "bandId": "band_id",
    "heartRate": "heart_rate",
    "systolicBP": "systolic_bp",
    "diastolicBP": "diastolic_bp",
    "oxygenSaturation": "oxygen_saturation",
    "oxygenSaturationValid": "oxygen_saturation_valid",
    "stepCount": "step_count",
    "fatigueLevel": "fatigue_level",
    "bodyTemperature": "body_temperature",
    "deepSleepTime": "deep_sleep_time",
    "lightSleepTime": "light_sleep_time",
    "remSleepTime": "rem_sleep_time",
    "sleepConfidence": "sleep_confidence",
    "sleepComputationStatus": "sleep_computation_status",
    "sleepComputationState": "sleep_computation_status",
    "sleepComputationScore": "sleep_computation_score",
}
required_bandlog_columns = [
    "resident_id",
    "generated_at",
    "band_id",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "oxygen_saturation",
    "oxygen_saturation_valid",
    "hrv",
    "rri",
    "step_count",
    "fatigue_level",
    "stress",
    "breathing",
    "body_temperature",
    "deep_sleep_time",
    "light_sleep_time",
    "rem_sleep_time",
    "sleep_confidence",
    "sleep_computation_status",
    "sleep_computation_score",
    "is_synthetic",
]
synthetic_band_df = load_optional_synthetic_csv(
    SYNTHETIC_BANDLOG_CSV,
    bandlog_column_map,
    required_bandlog_columns,
)
if not synthetic_band_df.empty:
    synthetic_band_df["is_synthetic"] = 1
    band_df = pd.concat(
        [band_df.reindex(columns=required_bandlog_columns), synthetic_band_df],
        ignore_index=True,
    )
    print(
        f"Appended {len(synthetic_band_df)} synthetic C2 BandLog rows "
        f"from {SYNTHETIC_BANDLOG_CSV}"
    )

baseline_column_map = {
    "residentId": "resident_id",
    "referenceHeartRate": "reference_heart_rate",
    "referenceHrv": "reference_hrv",
    "referenceHRV": "reference_hrv",
    "referenceSystolicBP": "reference_systolic_bp",
    "referenceDiastolicBP": "reference_diastolic_bp",
    "referenceOxygenSaturation": "reference_oxygen_saturation",
    "referenceBodyTemperature": "reference_body_temperature",
    "referenceTemperature": "reference_body_temperature",
    "referenceStepCount": "reference_step_count",
    "heartRateTolerance": "heart_rate_tolerance",
    "hrvTolerance": "hrv_tolerance",
    "systolicBPTolerance": "systolic_bp_tolerance",
    "diastolicBPTolerance": "diastolic_bp_tolerance",
    "oxygenSaturationTolerance": "oxygen_saturation_tolerance",
    "temperatureToleranceCelsius": "temperature_tolerance_celsius",
    "temperatureTolerance": "temperature_tolerance_celsius",
    "stepCountTolerance": "step_count_tolerance",
    "referenceDeepSleep": "reference_deep_sleep",
    "referenceDeepSleepTolerance": "reference_deep_sleep_tolerance",
    "referenceLightSleep": "reference_light_sleep",
    "referenceLightSleepTolerance": "reference_light_sleep_tolerance",
}
required_baseline_columns = [
    "resident_id",
    "reference_heart_rate",
    "reference_hrv",
    "reference_systolic_bp",
    "reference_diastolic_bp",
    "reference_oxygen_saturation",
    "reference_body_temperature",
    "reference_step_count",
    "heart_rate_tolerance",
    "hrv_tolerance",
    "systolic_bp_tolerance",
    "diastolic_bp_tolerance",
    "oxygen_saturation_tolerance",
    "temperature_tolerance_celsius",
    "step_count_tolerance",
    "reference_deep_sleep",
    "reference_deep_sleep_tolerance",
    "reference_light_sleep",
    "reference_light_sleep_tolerance",
]
synthetic_baseline_df = load_optional_synthetic_csv(
    SYNTHETIC_BASELINE_CSV,
    baseline_column_map,
    required_baseline_columns,
)
if not synthetic_baseline_df.empty:
    baseline_df = pd.concat(
        [baseline_df, synthetic_baseline_df],
        ignore_index=True,
    )
    print(
        f"Appended {len(synthetic_baseline_df)} synthetic resident baselines "
        f"from {SYNTHETIC_BASELINE_CSV}"
    )

print("\n=== Sleep Baseline Statistics ===")

def normalize_temperature(temp):
    if pd.isna(temp):
        return temp
    if temp > 50.0:
        return (temp - 32.0) * 5.0 / 9.0
    return temp

band_df["body_temperature"] = band_df["body_temperature"].apply(normalize_temperature)

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

band_df["band_id"] = band_df["band_id"].astype("string")
if ADVANCED_SLEEP_BAND_IDS:
    has_c2_sleep_payload = (
        band_df["rem_sleep_time"].notna()
        | band_df["sleep_confidence"].notna()
        | band_df["sleep_computation_status"].notna()
        | band_df["sleep_computation_score"].notna()
    )
    band_df["is_c2_band"] = (
        band_df["band_id"].isin(ADVANCED_SLEEP_BAND_IDS)
        | (band_df["is_synthetic"].eq(1) & has_c2_sleep_payload)
    ).astype(int)
else:
    band_df["is_c2_band"] = (
        band_df["rem_sleep_time"].notna()
        | band_df["sleep_confidence"].notna()
        | band_df["sleep_computation_status"].notna()
        | band_df["sleep_computation_score"].notna()
    ).astype(int)

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

# Add physiological bounds clipping
band_df["heart_rate"] = pd.to_numeric(band_df["heart_rate"], errors="coerce")
band_df.loc[(band_df["heart_rate"] < 30) | (band_df["heart_rate"] > 220), "heart_rate"] = np.nan

band_df["systolic_bp"] = pd.to_numeric(band_df["systolic_bp"], errors="coerce")
band_df.loc[(band_df["systolic_bp"] < 60) | (band_df["systolic_bp"] > 250), "systolic_bp"] = np.nan

band_df["diastolic_bp"] = pd.to_numeric(band_df["diastolic_bp"], errors="coerce")
band_df.loc[(band_df["diastolic_bp"] < 30) | (band_df["diastolic_bp"] > 150), "diastolic_bp"] = np.nan

band_df["body_temperature"] = pd.to_numeric(band_df["body_temperature"], errors="coerce")
band_df.loc[(band_df["body_temperature"] < 30.0) | (band_df["body_temperature"] > 43.0), "body_temperature"] = np.nan

band_df["spo2"] = pd.to_numeric(band_df["spo2"], errors="coerce")
band_df.loc[(band_df["spo2"] < 50) | (band_df["spo2"] > 100), "spo2"] = np.nan

band_df["rem_sleep_minutes"] = pd.to_numeric(
    band_df["rem_sleep_time"],
    errors="coerce",
).where(band_df["is_c2_band"].eq(1))
band_df["sleep_confidence"] = pd.to_numeric(
    band_df["sleep_confidence"],
    errors="coerce",
).where(band_df["is_c2_band"].eq(1))
band_df["sleep_confidence"] = band_df["sleep_confidence"].clip(
    lower=0,
    upper=MAX_SLEEP_CONFIDENCE,
)
band_df["sleep_confidence"] = band_df["sleep_confidence"].where(
    band_df["sleep_confidence"].gt(0)
)
band_df["sleep_confidence_normalized"] = (
    band_df["sleep_confidence"] / MAX_SLEEP_CONFIDENCE
)
band_df["sleep_computation_score"] = pd.to_numeric(
    band_df["sleep_computation_score"],
    errors="coerce",
).where(band_df["is_c2_band"].eq(1))
band_df["sleep_computation_score"] = band_df["sleep_computation_score"].clip(
    lower=0,
    upper=MAX_SLEEP_CONFIDENCE,
)
band_df["sleep_computation_score"] = band_df["sleep_computation_score"].where(
    band_df["sleep_computation_score"].gt(0)
)
band_df["sleep_computation_score_normalized"] = (
    band_df["sleep_computation_score"] / MAX_SLEEP_CONFIDENCE
)

sleep_status = (
    band_df["sleep_computation_status"]
    .astype("string")
    .str.strip()
    .str.lower()
)
band_df["sleep_status_completed"] = sleep_status.isin(
    ["completed", "complete", "computed", "success", "successful"]
).astype(int)
band_df["sleep_status_partial"] = sleep_status.isin(
    ["partial", "pending", "processing", "in_progress", "in progress"]
).astype(int)
band_df["sleep_status_failed"] = sleep_status.isin(
    ["failed", "failure", "error", "invalid"]
).astype(int)
band_df["sleep_status_unknown"] = (
    band_df["is_c2_band"].eq(1)
    & sleep_status.isna()
).astype(int)
band_df["sleep_feature_supported"] = band_df["is_c2_band"]
band_df["sleep_data_reliable"] = (
    band_df["is_c2_band"].eq(1)
    & band_df["sleep_status_completed"].eq(1)
    & band_df["sleep_confidence_normalized"].fillna(0).ge(0.70)
).astype(int)

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
band_df["rem_sleep_minutes_for_total"] = band_df["rem_sleep_minutes"].fillna(0)
band_df["total_sleep_minutes"] = (
    band_df["deep_sleep_minutes"]
    + band_df["light_sleep_minutes"]
    + band_df["rem_sleep_minutes_for_total"]
)

daily_sleep_df = (
    band_df.groupby(["resident_id", "log_date"], as_index=False)
    .agg(
        daily_deep_sleep_minutes=("deep_sleep_minutes", "max"),
        daily_light_sleep_minutes=("light_sleep_minutes", "max"),
        daily_rem_sleep_minutes=("rem_sleep_minutes", "max"),
        daily_total_sleep_minutes=("total_sleep_minutes", "max"),
        daily_sleep_confidence=("sleep_confidence_normalized", "max"),
        daily_sleep_computation_score=(
            "sleep_computation_score_normalized",
            "max",
        ),
        daily_sleep_feature_supported=("sleep_feature_supported", "max"),
        daily_sleep_status_completed=("sleep_status_completed", "max"),
        daily_sleep_status_partial=("sleep_status_partial", "max"),
        daily_sleep_status_failed=("sleep_status_failed", "max"),
        daily_sleep_status_unknown=("sleep_status_unknown", "max"),
        daily_sleep_data_reliable=("sleep_data_reliable", "max"),
    )
)
daily_sleep_df["sleep_data_available"] = (
    daily_sleep_df["daily_total_sleep_minutes"] > 0
).astype(int)
daily_sleep_df["rem_sleep_available"] = (
    daily_sleep_df["daily_rem_sleep_minutes"].fillna(0) > 0
).astype(int)
daily_sleep_df["sleep_confidence_available"] = (
    daily_sleep_df["daily_sleep_confidence"].notna()
).astype(int)
daily_sleep_df = daily_sleep_df.sort_values(["resident_id", "log_date"])
daily_sleep_df["previous_daily_total_sleep_minutes"] = (
    daily_sleep_df.groupby("resident_id")["daily_total_sleep_minutes"].shift()
)
daily_sleep_df["sleep_value_changed_from_previous_day"] = (
    daily_sleep_df["previous_daily_total_sleep_minutes"].isna()
    | (
        daily_sleep_df["daily_total_sleep_minutes"]
        .sub(daily_sleep_df["previous_daily_total_sleep_minutes"])
        .abs()
        .gt(1)
    )
).astype(int)
sleep_duration_plausible = daily_sleep_df["daily_total_sleep_minutes"].between(60, 720)
daily_sleep_df["estimated_sleep_reliability"] = (
    0.35 * daily_sleep_df["sleep_data_available"]
    + 0.25 * sleep_duration_plausible.astype(int)
    + 0.20 * daily_sleep_df["sleep_value_changed_from_previous_day"]
    + 0.20 * daily_sleep_df["daily_sleep_status_completed"]
).clip(lower=0, upper=1)
daily_sleep_df["sleep_reliability_source_estimated"] = (
    daily_sleep_df["sleep_confidence_available"].eq(0)
    & daily_sleep_df["estimated_sleep_reliability"].gt(0)
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
source_flags = (
    band_df[["resident_id", "generated_at", "is_synthetic"]]
    .drop_duplicates(["resident_id", "generated_at"], keep="last")
)

dataset = feature_df.merge(baseline_df, on="resident_id", how="left")
dataset = dataset.merge(
    source_flags,
    on=["resident_id", "generated_at"],
    how="left",
)
dataset["is_synthetic"] = dataset["is_synthetic"].fillna(0).astype(int)

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
for column in [
    "daily_rem_sleep_minutes",
    "daily_sleep_confidence",
    "daily_sleep_computation_score",
    "daily_sleep_feature_supported",
    "daily_sleep_status_completed",
    "daily_sleep_status_partial",
    "daily_sleep_status_failed",
    "daily_sleep_status_unknown",
    "daily_sleep_data_reliable",
    "rem_sleep_available",
    "sleep_confidence_available",
    "estimated_sleep_reliability",
    "sleep_reliability_source_estimated",
    "sleep_value_changed_from_previous_day",
]:
    if column in dataset.columns:
        dataset[column] = dataset[column].fillna(0)

dataset["hrv_imputed"] = (
    dataset["avg_hrv"].isna() | dataset["valid_hrv_count"].fillna(0).eq(0)
).astype(int)
dataset["avg_hrv"] = dataset["avg_hrv"].fillna(dataset["reference_hrv"])
for column in ["ewma_hrv", "avg_hrv_2h", "avg_hrv_6h", "avg_hrv_12h"]:
    dataset[column] = dataset[column].fillna(dataset["reference_hrv"])

dataset["spo2_imputed"] = (
    dataset["avg_spo2"].isna() | dataset["valid_spo2_count"].fillna(0).eq(0)
).astype(int)
dataset["avg_spo2"] = dataset["avg_spo2"].fillna(
    dataset["reference_oxygen_saturation"]
)
dataset["min_spo2"] = dataset["min_spo2"].fillna(
    dataset["reference_oxygen_saturation"]
)
for column in [
    "ewma_spo2",
    "avg_spo2_2h",
    "min_spo2_2h",
    "avg_spo2_6h",
    "min_spo2_6h",
    "avg_spo2_12h",
    "min_spo2_12h",
]:
    dataset[column] = dataset[column].fillna(dataset["reference_oxygen_saturation"])

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

for source_col in [
    "ewma_stress",
    "avg_stress_2h",
    "avg_stress_6h",
    "avg_stress_12h",
    "ewma_fatigue",
    "avg_fatigue_2h",
    "avg_fatigue_6h",
    "avg_fatigue_12h",
]:
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

for source_col, reference_col in [
    ("ewma_hr", "reference_heart_rate"),
    ("avg_hr_2h", "reference_heart_rate"),
    ("avg_hr_6h", "reference_heart_rate"),
    ("avg_hr_12h", "reference_heart_rate"),
    ("ewma_sbp", "reference_systolic_bp"),
    ("avg_sbp", "reference_systolic_bp"),
    ("ewma_dbp", "reference_diastolic_bp"),
    ("avg_dbp", "reference_diastolic_bp"),
]:
    dataset[source_col] = dataset[source_col].fillna(dataset[reference_col])

for source_col in ["steps_2h", "steps_6h", "steps_12h", "ewma_steps"]:
    dataset[source_col] = dataset[source_col].fillna(0)

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
dataset["estimated_sleep_data_reliable"] = (
    dataset["sleep_confidence_available"].eq(0)
    & dataset["estimated_sleep_reliability"].ge(0.70)
).astype(int)

dataset["implausible_historical_sleep"] = (
    (dataset["sleep_baseline_minutes"] > 0)
    & (dataset["sleep_baseline_minutes"] < 240)
)

base_sleep_used = np.where(
    dataset["daily_sleep_feature_supported"].eq(1),
    dataset["daily_sleep_data_reliable"].eq(1)
    | dataset["estimated_sleep_data_reliable"].eq(1),
    dataset["sleep_data_available"].eq(1),
)

dataset["sleep_used_for_risk"] = np.where(
    dataset["implausible_historical_sleep"],
    0,
    base_sleep_used
).astype(int)
dataset["sleep_stale_or_unreliable"] = (
    dataset["daily_sleep_feature_supported"].eq(1)
    & dataset["sleep_used_for_risk"].eq(0)
).astype(int)
dataset["daily_total_sleep_for_ratio"] = dataset["daily_total_sleep_minutes"].where(
    dataset["sleep_used_for_risk"].eq(1),
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
dataset["rem_sleep_ratio"] = safe_ratio(
    dataset["daily_rem_sleep_minutes"],
    dataset["daily_total_sleep_minutes"],
)
dataset["deep_sleep_ratio"] = safe_ratio(
    dataset["daily_deep_sleep_minutes"],
    dataset["daily_total_sleep_minutes"],
)
dataset["light_sleep_ratio"] = safe_ratio(
    dataset["daily_light_sleep_minutes"],
    dataset["daily_total_sleep_minutes"],
)

dataset["hr_30m_vs_2h"] = dataset["avg_hr"] - dataset["avg_hr_2h"]
dataset["hrv_30m_vs_2h"] = dataset["avg_hrv"] - dataset["avg_hrv_2h"]
dataset["spo2_30m_vs_2h"] = dataset["avg_spo2"] - dataset["avg_spo2_2h"]
dataset["steps_30m_vs_2h"] = dataset["steps_30m"] - (
    dataset["steps_2h"] / 4
)
dataset["stress_30m_vs_2h"] = dataset["avg_stress"] - dataset["avg_stress_2h"]
dataset["fatigue_30m_vs_2h"] = (
    dataset["avg_fatigue"] - dataset["avg_fatigue_2h"]
)

dataset["hr_30m_vs_6h"] = dataset["avg_hr"] - dataset["avg_hr_6h"]
dataset["hrv_30m_vs_6h"] = dataset["avg_hrv"] - dataset["avg_hrv_6h"]
dataset["spo2_30m_vs_6h"] = dataset["avg_spo2"] - dataset["avg_spo2_6h"]
dataset["steps_30m_vs_6h"] = dataset["steps_30m"] - (
    dataset["steps_6h"] / 12
)
dataset["stress_30m_vs_6h"] = dataset["avg_stress"] - dataset["avg_stress_6h"]
dataset["fatigue_30m_vs_6h"] = (
    dataset["avg_fatigue"] - dataset["avg_fatigue_6h"]
)

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
dataset["rem_sleep_ratio"] = safe_ratio(
    dataset["daily_rem_sleep_minutes"],
    dataset["daily_total_sleep_minutes"],
)
dataset["deep_sleep_ratio"] = safe_ratio(
    dataset["daily_deep_sleep_minutes"],
    dataset["daily_total_sleep_minutes"],
)
dataset["light_sleep_ratio"] = safe_ratio(
    dataset["daily_light_sleep_minutes"],
    dataset["daily_total_sleep_minutes"],
)

dataset["hr_30m_vs_2h"] = dataset["avg_hr"] - dataset["avg_hr_2h"]
dataset["hrv_30m_vs_2h"] = dataset["avg_hrv"] - dataset["avg_hrv_2h"]
dataset["spo2_30m_vs_2h"] = dataset["avg_spo2"] - dataset["avg_spo2_2h"]
dataset["steps_30m_vs_2h"] = dataset["steps_30m"] - (
    dataset["steps_2h"] / 4
)
dataset["stress_30m_vs_2h"] = dataset["avg_stress"] - dataset["avg_stress_2h"]
dataset["fatigue_30m_vs_2h"] = (
    dataset["avg_fatigue"] - dataset["avg_fatigue_2h"]
)

dataset["hr_30m_vs_6h"] = dataset["avg_hr"] - dataset["avg_hr_6h"]
dataset["hrv_30m_vs_6h"] = dataset["avg_hrv"] - dataset["avg_hrv_6h"]
dataset["spo2_30m_vs_6h"] = dataset["avg_spo2"] - dataset["avg_spo2_6h"]
dataset["steps_30m_vs_6h"] = dataset["steps_30m"] - (
    dataset["steps_6h"] / 12
)
dataset["stress_30m_vs_6h"] = dataset["avg_stress"] - dataset["avg_stress_6h"]
dataset["fatigue_30m_vs_6h"] = (
    dataset["avg_fatigue"] - dataset["avg_fatigue_6h"]
)

dataset["hr_30m_vs_12h"] = dataset["avg_hr"] - dataset["avg_hr_12h"]
dataset["hrv_30m_vs_12h"] = dataset["avg_hrv"] - dataset["avg_hrv_12h"]
dataset["spo2_30m_vs_12h"] = dataset["avg_spo2"] - dataset["avg_spo2_12h"]
dataset["steps_30m_vs_12h"] = dataset["steps_30m"] - (
    dataset["steps_12h"] / 24
)
dataset["stress_30m_vs_12h"] = dataset["avg_stress"] - dataset["avg_stress_12h"]
dataset["fatigue_30m_vs_12h"] = (
    dataset["avg_fatigue"] - dataset["avg_fatigue_12h"]
)

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
    0.15 * dataset["weighted_hr_breach"]
    + 0.15 * dataset["weighted_hrv_breach"]
    + 0.10 * dataset["weighted_sbp_breach"]
    + 0.10 * dataset["weighted_dbp_breach"]
    + 0.10 * dataset["weighted_spo2_breach"]
    + 0.20 * dataset["weighted_step_breach"]
    + 0.075 * dataset["fatigue_component"]
    + 0.075 * dataset["stress_component"]
    + 0.10 * dataset["sleep_component"]
    + 0.05 * dataset["hr_slope_component"]
    + 0.05 * dataset["hrv_slope_component"]
    + 0.05 * dataset["spo2_slope_component"]
    + 0.00005 * dataset["step_slope_component"]
)

dataset["vitals_domain_flag"] = (
    (dataset["hr_breach"] >= 1)
    | (dataset["spo2_breach"] >= 1)
    | (dataset["sbp_breach"] >= 1)
    | (dataset["dbp_breach"] >= 1)
    | (dataset["temp_breach"] >= 1)
).astype(int)
dataset["recovery_domain_flag"] = (
    (dataset["hrv_breach"] >= 1)
    | (dataset["fatigue_component"] >= 0.70)
    | (dataset["stress_component"] >= 0.70)
).astype(int)
dataset["mobility_domain_flag"] = (
    (dataset["step_breach"] >= 1)
    | (dataset["activity_ratio"] <= 0.40)
).astype(int)
dataset["sleep_domain_flag"] = (
    dataset["sleep_used_for_risk"].eq(1)
    & (dataset["sleep_deficit"] >= 0.25)
).astype(int)
dataset["abnormal_domain_count"] = (
    dataset["vitals_domain_flag"]
    + dataset["recovery_domain_flag"]
    + dataset["mobility_domain_flag"]
    + dataset["sleep_domain_flag"]
)

dataset["severe_vitals_flag"] = (
    (dataset["hr_breach"] >= 2)
    | (dataset["spo2_breach"] >= 2)
    | (dataset["sbp_breach"] >= 2)
    | (dataset["dbp_breach"] >= 2)
    | (dataset["temp_breach"] >= 2)
).astype(int)
dataset["severe_recovery_flag"] = (
    (dataset["hrv_breach"] >= 2)
    | (dataset["fatigue_component"] >= 0.85)
    | (dataset["stress_component"] >= 0.85)
).astype(int)
dataset["severe_mobility_flag"] = (
    (dataset["step_breach"] >= 2)
    | (dataset["activity_ratio"] <= 0.20)
).astype(int)
dataset["severe_sleep_flag"] = (
    dataset["sleep_used_for_risk"].eq(1)
    & (dataset["sleep_deficit"] >= 0.45)
).astype(int)
dataset["severe_domain_count"] = (
    dataset["severe_vitals_flag"]
    + dataset["severe_recovery_flag"]
    + dataset["severe_mobility_flag"]
    + dataset["severe_sleep_flag"]
)

dataset["short_term_worsening_count"] = (
    (dataset["hr_30m_vs_2h"] >= 10).astype(int)
    + (dataset["hrv_30m_vs_2h"] <= -5).astype(int)
    + (dataset["spo2_30m_vs_2h"] <= -2).astype(int)
    + (dataset["steps_30m_vs_2h"] <= -100).astype(int)
    + (dataset["stress_30m_vs_2h"] >= 1.5).astype(int)
    + (dataset["fatigue_30m_vs_2h"] >= 1.5).astype(int)
)
dataset["context_worsening_count"] = (
    (dataset["hr_30m_vs_6h"] >= 12).astype(int)
    + (dataset["hrv_30m_vs_6h"] <= -7).astype(int)
    + (dataset["spo2_30m_vs_6h"] <= -3).astype(int)
    + (dataset["steps_30m_vs_6h"] <= -150).astype(int)
    + (dataset["stress_30m_vs_6h"] >= 2).astype(int)
    + (dataset["fatigue_30m_vs_6h"] >= 2).astype(int)
)
dataset["transition_worsening_flag"] = (
    (dataset["short_term_worsening_count"] >= 2)
    | (dataset["context_worsening_count"] >= 2)
).astype(int)

critical_mask = (
    (dataset["risk_score"] >= 4.2)
    | (
        (dataset["severe_domain_count"] >= 3)
        & (dataset["risk_score"] >= 2.8)
    )
    | (
        (dataset["risk_score"] >= 3.4)
        & (dataset["severe_domain_count"] >= 2)
        & (dataset["abnormal_domain_count"] >= 3)
    )
)
high_mask = (
    ~critical_mask
    & (
        (
            (dataset["risk_score"] >= 2.0)
            & (dataset["abnormal_domain_count"] >= 2)
        )
        | (
            (dataset["risk_score"] >= 1.8)
            & (dataset["abnormal_domain_count"] >= 3)
            & dataset["transition_worsening_flag"].eq(1)
        )
        | (
            (dataset["risk_score"] >= 1.7)
            & (dataset["severe_domain_count"] >= 2)
            & dataset["transition_worsening_flag"].eq(1)
        )
    )
)
moderate_mask = (
    ~critical_mask
    & ~high_mask
    & (
        (dataset["risk_score"] >= 1.2)
        | (dataset["abnormal_domain_count"] >= 2)
    )
)
dataset["risk_label"] = np.select(
    [critical_mask, high_mask, moderate_mask],
    ["Critical", "High", "Moderate"],
    default="Low",
)
dataset["risk_label"] = pd.Categorical(
    dataset["risk_label"],
    categories=["Low", "Moderate", "High", "Critical"],
    ordered=True,
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
    "spo2_std",
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
    "daily_rem_sleep_minutes",
    "rem_sleep_ratio",
    "deep_sleep_ratio",
    "light_sleep_ratio",
    "rem_sleep_available",
    "daily_sleep_confidence",
    "daily_sleep_computation_score",
    "daily_sleep_feature_supported",
    "daily_sleep_status_completed",
    "daily_sleep_status_partial",
    "daily_sleep_status_failed",
    "daily_sleep_status_unknown",
    "daily_sleep_data_reliable",
    "sleep_confidence_available",
    "estimated_sleep_reliability",
    "estimated_sleep_data_reliable",
    "sleep_reliability_source_estimated",
    "sleep_value_changed_from_previous_day",
    "sleep_used_for_risk",
    "sleep_stale_or_unreliable",
    "vitals_domain_flag",
    "recovery_domain_flag",
    "mobility_domain_flag",
    "sleep_domain_flag",
    "abnormal_domain_count",
    "severe_vitals_flag",
    "severe_recovery_flag",
    "severe_mobility_flag",
    "severe_sleep_flag",
    "severe_domain_count",
    "short_term_worsening_count",
    "context_worsening_count",
    "transition_worsening_flag",
    "hrv_availability",
    "spo2_availability",
    "ewma_hr",
    "ewma_hrv",
    "ewma_spo2",
    "ewma_sbp",
    "ewma_dbp",
    "ewma_stress",
    "ewma_fatigue",
    "ewma_steps",
    "avg_hr_2h",
    "avg_hrv_2h",
    "avg_spo2_2h",
    "min_spo2_2h",
    "steps_2h",
    "avg_stress_2h",
    "avg_fatigue_2h",
    "avg_hr_6h",
    "avg_hrv_6h",
    "avg_spo2_6h",
    "min_spo2_6h",
    "steps_6h",
    "avg_stress_6h",
    "avg_fatigue_6h",
    "avg_hr_12h",
    "avg_hrv_12h",
    "avg_spo2_12h",
    "min_spo2_12h",
    "steps_12h",
    "avg_stress_12h",
    "avg_fatigue_12h",
    "hr_30m_vs_2h",
    "hrv_30m_vs_2h",
    "spo2_30m_vs_2h",
    "steps_30m_vs_2h",
    "stress_30m_vs_2h",
    "fatigue_30m_vs_2h",
    "hr_30m_vs_6h",
    "hrv_30m_vs_6h",
    "spo2_30m_vs_6h",
    "steps_30m_vs_6h",
    "stress_30m_vs_6h",
    "fatigue_30m_vs_6h",
    "hr_30m_vs_12h",
    "hrv_30m_vs_12h",
    "spo2_30m_vs_12h",
    "steps_30m_vs_12h",
    "stress_30m_vs_12h",
    "fatigue_30m_vs_12h",
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
    ["resident_id", "generated_at", "is_synthetic"] + training_features + target_columns
]

final_training_dataset.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(final_training_dataset)} rows to {OUTPUT_CSV}")
print(final_training_dataset.describe(include="all"))
print(final_training_dataset["risk_label"].value_counts())
