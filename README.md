# CareMP Fall Risk V2 - Fast-Track API Architecture

This directory contains the production-ready REST API for the V2 Fall Risk model. The architecture has been specifically engineered to operate within highly constrained memory environments (1GB RAM, 2 vCPUs) while processing high-frequency telemetry from wearable bands.

## Core Architectural Pillars

### 1. Hybrid Smart-Batch Ingestion 
Wearable bands transmit high-frequency telemetry packets (up to 1 packet per minute per resident). Running the full ML pipeline synchronously for every single packet would overwhelm the server's CPU and RAM. To solve this, the `/predict` endpoint utilizes a **Hybrid Fast-Track & Batching Architecture** backed by a Redis Sorted Set:
* **First-Packet Fast-Track:** When a resident's Redis queue is empty (e.g., they just put their band on or synced after a gap), the API processes the telemetry synchronously, triggering the XGBoost model instantly to return immediate clinical feedback to the dashboard.
* **15-Minute Vectorized Batching:** If a resident is actively transmitting and their queue already contains recent packets, the API ingests the packet in milliseconds and returns a success response *without* running the ML model. An `AsyncIOScheduler` background job wakes up every 15 minutes, aggregates all queued data into a single, highly-efficient Pandas DataFrame (Vectorization), and processes predictions for all residents simultaneously.

### 2. Concurrency Management & OOM Protection
To protect the 1GB RAM limitation, the API implements strict concurrency controls using Python's `asyncio.Semaphore(4)`. 
In the event of a network outage recovery—where 200 bands might attempt to sync historical data at the exact same millisecond—the system would normally attempt to load 200 Pandas DataFrames into memory, resulting in an immediate Out-Of-Memory (OOM) crash. The Semaphore acts as an application-level traffic light, physically restricting the system to a maximum of 4 concurrent ML executions. Excess requests are queued politely, guaranteeing that the server remains stable and responsive under infinite load.

### 3. Synchronous Clinical Translation Layer
While the core V2 XGBoost model outputs mathematical probabilities (e.g., `0.607`), the front-end Engineering Dashboard requires rich, human-readable clinical context. 
To maintain backward compatibility with the dashboard, the API features an integrated **Translation Layer**. When the ML pipeline executes, it intercepts the raw XGBoost probability and feeds it into the legacy Clinical Rule Engine. The API then returns a deeply-nested JSON payload containing:
* A normalized **0-100 Risk Score**
* Calculated **Clinical Risk Drivers** (e.g., *"Mobility declined 99% over 12h from baseline"*)
* Automated **AI Recommendations** mapped to the specific physiological deviations

### 4. Automated Baseline Synchronization
To ensure the ML feature engineering process is always utilizing up-to-date patient context, an asynchronous 12-hour background scheduler runs continuously. This process reaches out to the central Engineering API to fetch the latest resident demographics and daily baseline metrics, caching them securely in a local SQLite database for instant retrieval during prediction scoring.

---

## Setup & Execution

Because this API uses Redis to manage high-frequency data queues, **the Redis server must be running alongside the pipeline.** The FastAPI server will automatically connect to it upon startup.

### 1. Prerequisites
* **Python 3.10+**
* **Redis Server**
  * *Mac/Linux:* `brew install redis` or `sudo apt install redis-server`
  * *Docker:* `docker run -p 6379:6379 -d redis`

### 3. Running the Infrastructure
You must start Redis *before* you start the FastAPI pipeline.

**Terminal 1 (Redis):**
Start your Redis server so it is ready to receive packets.
```bash
redis-server
```

**Terminal 2 (Fall Risk Pipeline):**
Navigate to the folder containing `main.py` and start the Uvicorn web server.
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

*Note: Once Uvicorn starts, you will immediately see the APScheduler initialize in the logs, which will start the 15-minute background batching job and the 12-hour baseline sync job.*

### 4. Testing the API
You can securely test the Fast-Track architecture using PowerShell:
```powershell
$body = @{
    bandlogId = 1
    residentId = 101
    heartRate = 75
    systolicBP = 120
    diastolicBP = 80
    bodyTemperature = 98.6
    stepCount = 0
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"
```
* **Packet 1:** Returns instant JSON Fall Risk feedback (Queue was empty -> Fast-Track ML Execution).
  ```json
  {
    "residentId": 101,
    "mlRiskScore": 54.0,
    "riskLevel": "Medium Risk",
    "predictionTimestamp": "2026-06-25T12:00:00Z",
    "predictiveHorizon": "Next 24 hrs",
    "vitalsSnapshot": {
      "heartRate": 75.0,
      "systolicBP": 120.0,
      "diastolicBP": 80.0,
      "temperature": 98.6
    },
    "keyDrivers": [
      "No movement detected in last 12h (99.0% decline)"
    ],
    "recommendations": [
      "Fall Risk Alert: Perform unassisted transfer check."
    ],
    "dataConfidence": {
      "confidenceScore": 100,
      "dataGapDetected": false
    }
  }
  ```
* **Packet 2:** Returns instantly with `"queued": true` (Queue exists -> Deferred to 15-minute batch).
  ```json
  {
    "status": "success",
    "queued": true,
    "message": "Packet buffered for resident 101. ML pipeline deferred to scheduled batch."
  }
  ```
