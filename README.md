# Credit Risk ML Pipeline

[![CI](https://github.com/kitsakisGk/Credit_Risk_Databricks_Pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/kitsakisGk/Credit_Risk_Databricks_Pipeline/actions/workflows/ci.yml)
![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=flat&logo=databricks&logoColor=white)
![Delta Lake](https://img.shields.io/badge/Delta%20Lake-00ADD8?style=flat&logo=delta&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat&logo=xgboost&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)

End-to-end **credit risk assessment platform** built on **Databricks Lakehouse**, featuring batch + streaming processing, real-time scoring API, data quality monitoring, and model drift detection — aligned with **Swiss banking standards (FINMA)**.

---

## Pipeline Overview

![Credit Risk Pipeline](docs/Images/Credit_Risk_Pipeline.png)

**Medallion Architecture:** Bronze (raw) → Silver (cleaned) → Gold (features) → ML Models → Monitoring

---

## Quick Start

### 1. Clone & Upload Data

```bash
git clone https://github.com/kitsakisGk/Credit_Risk_Databricks_Pipeline.git
```

Upload `data/default of credit card clients.xls` to Databricks:
- **Catalog** → **Create Table** → **Upload File**
- Schema: `kitsakis_credit_risk`
- Table: `bronze_credit_raw`
- Check "First row contains header"

### 2. Run Notebooks in Order

| Step | Notebook | Purpose |
|------|----------|---------|
| 1 | `00_setup_environment.py` | Create Bronze table |
| 2 | `01_silver_transformation.py` | Data cleaning |
| 3 | `02_gold_features.py` | Feature engineering |
| 4 | `03_ml_training.py` | Baseline ML models |
| 5 | `04_advanced_ml.py` | XGBoost + SHAP |
| 6 | `05_data_quality_monitoring.py` | Data validation |
| 7 | `06_model_monitoring.py` | Drift detection |
| 8 | `07_streaming_simulation.py` | Real-time scoring |

---

## Model Results

### Model Comparison

![Model Comparison](docs/Images/Model_Comparison_Table.png)

| Model | AUC | Accuracy |
|-------|-----|----------|
| **XGBoost** | 0.78 | 82% |
| Gradient Boosting | 0.78 | 82% |
| Random Forest | 0.78 | 82% |
| Logistic Regression | 0.76 | 82% |

### Confusion Matrix

![Confusion Matrix](docs/Images/Confusion_Matrix.png)

---

## SHAP Explainability (Swiss Banking Compliance)

Swiss financial regulations (FINMA) require model interpretability. We use **SHAP values** to explain predictions.

### Feature Importance

![SHAP Summary Plot](docs/Images/SHAP_Summary_Plot.png)

### Top Features by SHAP

![SHAP Feature Importance](docs/Images/SHAP_Feature_mportance_Table.png)

### Individual Customer Explanation

![Individual Explanation](docs/Images/Individual_Customer_Explanation.png)

This is exactly what regulators need - **transparent, explainable AI** that can justify credit decisions.

---

## Data Quality Monitoring

Automated validation across all Medallion layers (Basel III / BCBS 239 compliance):

- **Bronze**: Schema completeness, null checks, duplicate detection
- **Silver**: Value ranges, categorical validity, cross-layer consistency
- **Gold**: Feature distributions, ML-readiness, NaN detection
- **Quality Gate**: Pipeline blocks on critical failures

Results are logged to Delta tables for audit history.

---

## Model Monitoring & Drift Detection

Production model monitoring with:

- **PSI (Population Stability Index)**: Detects feature distribution shifts
- **Prediction Stability**: Monitors model output consistency over time
- **Segment Analysis**: Performance breakdown by age and credit segments
- **Fairness Check**: Disparate impact ratio across demographic groups
- **Retraining Alerts**: Automated recommendations when drift exceeds thresholds

---

## Streaming Pipeline

Simulated real-time credit risk scoring:

- **Micro-batch processing**: Applications scored in streaming batches
- **Consistent features**: Same feature engineering as batch pipeline
- **Risk classification**: LOW / MEDIUM / HIGH labels in real-time
- **Production-ready**: Architecture maps directly to Kafka + Spark Structured Streaming

---

## Scoring API (FastAPI + Docker)

Real-time credit risk scoring service for production deployment:

```bash
# Train and export model
python api/train_and_export.py

# Start API
uvicorn api.main:app --reload

# Or with Docker
docker-compose up
```

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Score a single application |
| POST | `/predict_batch` | Score up to 1000 applications |
| GET | `/model_info` | Model metadata |
| GET | `/health` | Health check |

API docs available at `http://localhost:8000/docs`

---

## Testing

27 unit tests covering feature engineering logic:

```bash
pytest tests/ -v
```

Tests cover: credit utilization, payment ratios, delay risk scores, education/marital risk, aggregate features, and feature completeness.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Platform | Databricks Lakehouse |
| Storage | Delta Lake |
| ML | XGBoost, scikit-learn |
| Explainability | SHAP Values |
| API | FastAPI, Docker |
| Monitoring | PSI drift detection |
| CI/CD | GitHub Actions + pytest |

---

## Swiss Banking Standards

| Technique | Purpose | Relevance |
|-----------|---------|-----------|
| **XGBoost** | Classification | Industry standard at UBS, Credit Suisse |
| **SHAP Values** | Explainability | Required by FINMA for audit compliance |
| **Cross-Validation** | Validation | Basel III/IV model requirements |
| **Data Quality** | Governance | BCBS 239 data quality standards |
| **Model Monitoring** | Drift detection | Ongoing model validation (SR 11-7) |
| **Fairness Analysis** | Bias detection | Anti-discrimination compliance |

---

## Dataset

**Taiwan Credit Card Default Dataset** - 30,000 records from [UCI ML Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

---

## License

MIT
