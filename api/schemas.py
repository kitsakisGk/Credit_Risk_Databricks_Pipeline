"""Pydantic models for request/response validation."""

from pydantic import BaseModel, Field
from typing import List, Optional


class CreditApplication(BaseModel):
    """Single credit application for scoring."""
    credit_limit: float = Field(..., gt=0, description="Credit card limit")
    age: int = Field(..., ge=18, le=100, description="Customer age")
    sex: int = Field(..., ge=1, le=2, description="1=male, 2=female")
    education: int = Field(..., ge=1, le=4, description="1=grad, 2=uni, 3=high school, 4=other")
    marriage: int = Field(..., ge=1, le=3, description="1=married, 2=single, 3=other")
    pay_status_1: int = Field(0, description="Payment status month 1 (-2 to 8)")
    pay_status_2: int = Field(0, description="Payment status month 2")
    pay_status_3: int = Field(0, description="Payment status month 3")
    pay_status_4: int = Field(0, description="Payment status month 4")
    pay_status_5: int = Field(0, description="Payment status month 5")
    pay_status_6: int = Field(0, description="Payment status month 6")
    bill_amt_1: float = Field(0, description="Bill amount month 1")
    bill_amt_2: float = Field(0, description="Bill amount month 2")
    bill_amt_3: float = Field(0, description="Bill amount month 3")
    bill_amt_4: float = Field(0, description="Bill amount month 4")
    bill_amt_5: float = Field(0, description="Bill amount month 5")
    bill_amt_6: float = Field(0, description="Bill amount month 6")
    pay_amt_1: float = Field(0, description="Payment amount month 1")
    pay_amt_2: float = Field(0, description="Payment amount month 2")
    pay_amt_3: float = Field(0, description="Payment amount month 3")
    pay_amt_4: float = Field(0, description="Payment amount month 4")
    pay_amt_5: float = Field(0, description="Payment amount month 5")
    pay_amt_6: float = Field(0, description="Payment amount month 6")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "credit_limit": 200000,
                "age": 35,
                "sex": 1,
                "education": 2,
                "marriage": 1,
                "pay_status_1": 0,
                "pay_status_2": 0,
                "pay_status_3": 0,
                "pay_status_4": 0,
                "pay_status_5": 0,
                "pay_status_6": 0,
                "bill_amt_1": 50000,
                "bill_amt_2": 45000,
                "bill_amt_3": 40000,
                "bill_amt_4": 35000,
                "bill_amt_5": 30000,
                "bill_amt_6": 25000,
                "pay_amt_1": 5000,
                "pay_amt_2": 4500,
                "pay_amt_3": 4000,
                "pay_amt_4": 3500,
                "pay_amt_5": 3000,
                "pay_amt_6": 2500
            }]
        }
    }


class PredictionResponse(BaseModel):
    """Single prediction result."""
    default_probability: float = Field(..., description="Probability of default (0-1)")
    risk_label: str = Field(..., description="LOW, MEDIUM, or HIGH")
    risk_score: int = Field(..., description="Composite risk score (0-18)")
    top_risk_factors: List[dict] = Field(..., description="Top contributing factors")


class BatchRequest(BaseModel):
    """Batch prediction request."""
    applications: List[CreditApplication]


class BatchResponse(BaseModel):
    """Batch prediction result."""
    predictions: List[PredictionResponse]
    total_processed: int
    avg_default_probability: float


class ModelInfo(BaseModel):
    """Model metadata."""
    model_type: str
    n_features: int
    training_samples: int
    auc_score: float
    version: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str
