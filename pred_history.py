import sqlite3
import pandas as pd

DB_PATH = "fall_risk_prediction_history.db"

conn = sqlite3.connect(DB_PATH)

summary = pd.read_sql("""
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT resident_id) AS resident_count,
    MIN(generated_at) AS first_prediction_time,
    MAX(generated_at) AS last_prediction_time
FROM fall_risk_prediction_history;
""", conn)

print("\n=== Overall History Summary ===")
print(summary)

per_resident = pd.read_sql("""
SELECT
    resident_id,
    COUNT(*) AS rows,
    MIN(generated_at) AS first_prediction_time,
    MAX(generated_at) AS last_prediction_time
FROM fall_risk_prediction_history
GROUP BY resident_id
ORDER BY resident_id;
""", conn)

print("\n=== Per Resident History ===")
print(per_resident)

label_counts = pd.read_sql("""
SELECT
    predicted_risk_label,
    COUNT(*) AS count
FROM fall_risk_prediction_history
GROUP BY predicted_risk_label
ORDER BY count DESC;
""", conn)

print("\n=== Label Distribution ===")
print(label_counts)
per_resident["first_prediction_time"] = pd.to_datetime(
    per_resident["first_prediction_time"]
)
per_resident["last_prediction_time"] = pd.to_datetime(
    per_resident["last_prediction_time"]
)

per_resident["history_days"] = (
    per_resident["last_prediction_time"]
    - per_resident["first_prediction_time"]
).dt.total_seconds() / (24 * 3600)

print(per_resident[
    [
        "resident_id",
        "rows",
        "first_prediction_time",
        "last_prediction_time",
        "history_days",
    ]
])
rows_per_resident = pd.read_sql("""
SELECT
    resident_id,
    COUNT(*) AS prediction_rows,
    MIN(generated_at) AS first_prediction_time,
    MAX(generated_at) AS last_prediction_time
FROM fall_risk_prediction_history
GROUP BY resident_id
ORDER BY resident_id;
""", conn)
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print("\n=== Rows Per Resident ===")
print(rows_per_resident)

conn.close()