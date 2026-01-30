# Credit Risk ML Pipeline

[![CI](https://github.com/kitsakisGk/credit-risk-databricks-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/kitsakisGk/credit-risk-databricks-pipeline/actions/workflows/ci.yml)
![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=flat&logo=databricks&logoColor=white)
![Delta Lake](https://img.shields.io/badge/Delta%20Lake-00ADD8?style=flat&logo=delta&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C?style=flat&logo=apachespark&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)

End-to-end **credit risk assessment pipeline** built on **Databricks Lakehouse**, demonstrating production-grade data engineering and machine learning practices.

## Overview

This project implements a complete ML pipeline for credit risk prediction using the **Medallion Architecture** (Bronze → Silver → Gold), with MLflow experiment tracking and rule-based risk scoring.

### Key Highlights

- **Data Engineering**: Medallion architecture with Delta Lake
- **Machine Learning**: Model training, comparison, and tracking with MLflow
- **Risk Scoring**: Rule-based credit scoring model
- **Feature Engineering**: Payment behavior analysis and risk indicators
- **Two Implementations**: SQL (for SQL Warehouse) and Python (for full compute)

## Tech Stack

| Layer | Technology |
|-------|------------|
| Platform | Databricks Lakehouse |
| Storage | Delta Lake |
| Compute | Apache Spark / SQL Warehouse |
| ML | MLflow, Spark MLlib |
| CI/CD | GitHub Actions |

## Dataset

**Taiwan Credit Card Default Dataset** from UCI ML Repository
- **30,000 records** of credit card customers
- 23 features including payment history, bill amounts, demographics
- Binary target: default payment (yes/no)
- ~22% default rate

Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

## Getting Started

### Two Options Available

| Option | Compute Type | Features |
|--------|--------------|----------|
| **SQL Notebooks** | SQL Warehouse (Free tier) | Data pipeline + Rule-based scoring |
| **Python Notebooks** | All-Purpose Compute | Full ML with MLflow + 3 ML models |

---

### Option 1: SQL Warehouse (Free Tier)

For Databricks with only SQL Warehouse access.

**Setup:**
1. Download dataset: [UCI Credit Card Data](https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls)
2. In Databricks: **Catalog** → **Create Table** → Upload the Excel file
   - Schema: `kitsakis_credit_risk`
   - Table: `bronze_credit_applications`

**Run Notebooks:**
```
notebooks/sql/01_bronze_cleanup.sql      → Fix column names
notebooks/sql/02_silver_transformation.sql → Data cleaning + features
notebooks/sql/03_gold_aggregation.sql    → Feature engineering
notebooks/sql/04_risk_analysis.sql       → Risk scoring + metrics
```

---

### Option 2: Python Compute (Full ML)

For Databricks with Python compute cluster or serverless.

**Run Notebooks:**
```
notebooks/python/00_setup_environment.py  → Downloads data, creates Bronze
notebooks/python/01_silver_transformation.py → Data cleaning
notebooks/python/02_gold_features.py      → Feature engineering
notebooks/python/03_ml_training.py        → MLflow + 3 ML models
```

---

## Pipeline Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   SOURCE    │───▶│   BRONZE    │───▶│   SILVER    │───▶│    GOLD     │
│  (UCI Data) │    │    (Raw)    │    │  (Cleaned)  │    │ (Features)  │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                               │
                  ┌─────────────┐    ┌─────────────┐           │
                  │   MLFLOW    │◀───│  TRAINING   │◀──────────┘
                  │ (Tracking)  │    │  (Models)   │
                  └─────────────┘    └─────────────┘
```

## Project Structure

```
├── notebooks/
│   ├── sql/                              # SQL Warehouse compatible
│   │   ├── 01_bronze_cleanup.sql
│   │   ├── 02_silver_transformation.sql
│   │   ├── 03_gold_aggregation.sql
│   │   └── 04_risk_analysis.sql
│   └── python/                           # Full Python/MLflow
│       ├── 00_setup_environment.py
│       ├── 01_silver_transformation.py
│       ├── 02_gold_features.py
│       └── 03_ml_training.py
├── src/utils/                            # Reusable modules
├── data/                                 # Sample data
└── .github/workflows/                    # CI/CD
```

## Features Engineered

| Feature | Description |
|---------|-------------|
| `months_delayed` | Count of months with payment delay |
| `max_delay_months` | Worst payment delay |
| `credit_utilization` | Current balance / Credit limit |
| `total_risk_score` | Combined risk indicator |
| `payment_ratio` | Payment amount / Bill amount |

## Model Performance (Python)

| Model | AUC | Accuracy |
|-------|-----|----------|
| Gradient Boosted Trees | ~0.78 | ~82% |
| Random Forest | ~0.77 | ~81% |
| Logistic Regression | ~0.72 | ~78% |

## License

MIT
