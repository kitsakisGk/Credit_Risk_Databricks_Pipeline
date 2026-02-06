"""
Credit Risk Scoring API

FastAPI service for real-time credit risk predictions.
Designed for Swiss banking compliance (FINMA).

Usage:
    uvicorn api.main:app --reload
    Open: http://localhost:8000/docs
"""

import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from api.schemas import (
    CreditApplication, PredictionResponse,
    BatchRequest, BatchResponse,
    ModelInfo, HealthResponse
)
from api.feature_engine import engineer_features

# Model storage
model = None
model_metadata = {}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "xgboost_model.pkl")

FEATURE_ORDER = [
    "credit_limit", "age",
    "bill_amt_1", "bill_amt_2", "bill_amt_3", "bill_amt_4", "bill_amt_5", "bill_amt_6",
    "pay_amt_1", "pay_amt_2", "pay_amt_3", "pay_amt_4", "pay_amt_5", "pay_amt_6",
    "pay_status_1", "pay_status_2", "pay_status_3", "pay_status_4", "pay_status_5", "pay_status_6",
    "avg_bill_amount", "avg_payment_amount", "credit_utilization", "log_credit_limit",
    "payment_ratio", "pays_full_balance", "is_young_borrower",
    "delay_risk_1", "delay_risk_2", "delay_risk_3",
    "education_risk", "marital_risk", "utilization_risk", "total_risk_score",
    "months_delayed", "max_delay_months", "total_bill_amt", "total_pay_amt"
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, model_metadata
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            saved = pickle.load(f)
            model = saved["model"]
            model_metadata = saved.get("metadata", {})
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"WARNING: No model found at {MODEL_PATH}")
        print("Run 'python api/train_and_export.py' to train and export the model.")
    yield


app = FastAPI(
    title="Credit Risk Scoring API",
    description="Real-time credit risk assessment for Swiss banking compliance (FINMA)",
    version="1.0.0",
    lifespan=lifespan
)


def score_application(application: CreditApplication) -> PredictionResponse:
    """Score a single credit application."""
    # Engineer features (mirrors Gold layer)
    features = engineer_features(application.model_dump())

    # Create feature vector in correct order
    feature_vector = np.array([[features[col] for col in FEATURE_ORDER]])

    # Predict
    probability = float(model.predict_proba(feature_vector)[0][1])

    # Risk label
    if probability < 0.3:
        risk_label = "LOW"
    elif probability < 0.6:
        risk_label = "MEDIUM"
    else:
        risk_label = "HIGH"

    # Top risk factors
    top_factors = []
    risk_indicators = {
        "pay_status_1": ("Recent payment delay", features["pay_status_1"]),
        "credit_utilization": ("Credit utilization", round(features["credit_utilization"], 2)),
        "months_delayed": ("Months with delays", features["months_delayed"]),
        "total_risk_score": ("Total risk score", features["total_risk_score"]),
        "payment_ratio": ("Payment ratio", round(features["payment_ratio"], 2)),
    }
    for key, (name, value) in risk_indicators.items():
        top_factors.append({"factor": name, "value": value})

    return PredictionResponse(
        default_probability=round(probability, 4),
        risk_label=risk_label,
        risk_score=int(features["total_risk_score"]),
        top_risk_factors=top_factors
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "no_model",
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.get("/model_info", response_model=ModelInfo)
async def get_model_info():
    """Get model metadata."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfo(
        model_type=model_metadata.get("model_type", "XGBoost"),
        n_features=len(FEATURE_ORDER),
        training_samples=model_metadata.get("training_samples", 0),
        auc_score=model_metadata.get("auc_score", 0.0),
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(application: CreditApplication):
    """Score a single credit application.

    Returns default probability, risk label, and top risk factors.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return score_application(application)


@app.post("/predict_batch", response_model=BatchResponse)
async def predict_batch(batch: BatchRequest):
    """Score multiple credit applications.

    Accepts up to 1000 applications per request.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(batch.applications) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 applications per batch")

    predictions = [score_application(app) for app in batch.applications]
    avg_prob = np.mean([p.default_probability for p in predictions])

    return BatchResponse(
        predictions=predictions,
        total_processed=len(predictions),
        avg_default_probability=round(float(avg_prob), 4)
    )
