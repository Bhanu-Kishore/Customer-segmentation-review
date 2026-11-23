from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent.parent
DRIFT_REPORT_PATH = BASE_DIR / "monitoring" / "data_drift_report.html"

app = FastAPI(
    title="Customer Segmentation API",
    description="Customer segmentation using K-Means",
    version="1.0"
)

model = joblib.load(BASE_DIR / "models" / "kmeans_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")


@app.get("/")
def home():
    return {
        "message": "Customer Segmentation API is running"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/drift-report", response_class=FileResponse)
def drift_report():
    if not DRIFT_REPORT_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail="Drift report not generated yet"
        )

    return FileResponse(
        DRIFT_REPORT_PATH,
        media_type="text/html"
    )


@app.post("/predict")
def predict(
    age: int,
    income: float,
    spending_score: float,
    frequency: int
):

    customer_data = np.array([
        [age, income, spending_score, frequency]
    ])

    scaled_data = scaler.transform(customer_data)

    segment = model.predict(scaled_data)

    return {
        "customer_segment": int(segment[0])
    }
