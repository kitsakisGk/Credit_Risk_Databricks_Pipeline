# Credit Risk ML Pipeline

[![CI](https://github.com/kitsakisGk/credit-risk-databricks-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/kitsakisGk/credit-risk-databricks-pipeline/actions/workflows/ci.yml)
![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=flat&logo=databricks&logoColor=white)
![Delta Lake](https://img.shields.io/badge/Delta%20Lake-00ADD8?style=flat&logo=delta&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat&logo=xgboost&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

End-to-end **credit risk assessment pipeline** built on **Databricks Lakehouse**, demonstrating production-grade data engineering and machine learning practices aligned with **Swiss banking standards**.

## Overview

This project implements a complete ML pipeline for credit risk prediction using the **Medallion Architecture** (Bronze → Silver → Gold), featuring techniques used by Swiss financial institutions (UBS, Credit Suisse, Julius Baer).

### Key Highlights

- **Medallion Architecture**: Bronze → Silver → Gold data layers with Delta Lake
- **Swiss Banking ML**: XGBoost + SHAP (used at UBS, Credit Suisse)
- **Regulatory Compliance**: Model explainability for FINMA requirements
- **Production Ready**: CI/CD, cross-validation, hyperparameter tuning

## Tech Stack

| Layer | Technology |
|-------|------------|
| Platform | Databricks Lakehouse |
| Storage | Delta Lake |
| ML Models | XGBoost, Random Forest, Gradient Boosting, Logistic Regression |
| Explainability | SHAP Values |
| Data Processing | pandas, NumPy |
| CI/CD | GitHub Actions |

## Swiss Banking Standards

This project demonstrates ML techniques required by Swiss financial regulators:

| Technique | Purpose | Swiss Relevance |
|-----------|---------|-----------------|
| **XGBoost** | State-of-art classification | Industry standard at UBS, Credit Suisse |
| **SHAP Values** | Model explainability | Required by FINMA for audit compliance |
| **Cross-Validation** | Robust validation | Basel III/IV model validation |
| **Hyperparameter Tuning** | Optimization | Industry best practice |

## Dataset

**Taiwan Credit Card Default Dataset** from UCI ML Repository
- **30,000 records** of credit card customers
- 23 features including payment history, bill amounts, demographics
- Binary target: default payment (yes/no)
- ~22% default rate

Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

## Getting Started

### Run Notebooks in Order:

```
notebooks/python/00_setup_environment.py   → Download data, create Bronze table
notebooks/python/01_silver_transformation.py → Data cleaning, feature creation
notebooks/python/02_gold_features.py        → Feature engineering, risk scores
notebooks/python/03_ml_training.py          → Train baseline ML models
notebooks/python/04_advanced_ml.py          → XGBoost + SHAP (Swiss banking)
```

### SQL Version (for SQL Warehouse):
```
notebooks/sql/01_bronze_cleanup.sql
notebooks/sql/02_silver_transformation.sql
notebooks/sql/03_gold_aggregation.sql
notebooks/sql/04_risk_analysis.sql
```

## Pipeline Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   SOURCE    │───▶│   BRONZE    │───▶│   SILVER    │───▶│    GOLD     │
│  (UCI Data) │    │    (Raw)    │    │  (Cleaned)  │    │ (Features)  │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                               │
                   ┌─────────────┐    ┌─────────────┐          │
                   │    SHAP     │◀───│   XGBOOST   │◀─────────┘
                   │(Explainability)  │  (Training) │
                   └─────────────┘    └─────────────┘
```

## Project Structure

```
├── notebooks/
│   ├── python/                           # Main pipeline (pandas + sklearn)
│   │   ├── 00_setup_environment.py       # Data ingestion
│   │   ├── 01_silver_transformation.py   # Data cleaning
│   │   ├── 02_gold_features.py           # Feature engineering
│   │   ├── 03_ml_training.py             # Baseline models
│   │   └── 04_advanced_ml.py             # XGBoost + SHAP
│   └── sql/                              # SQL Warehouse version
├── src/utils/                            # Reusable modules
└── .github/workflows/                    # CI/CD
```

## Features Engineered

| Feature | Description |
|---------|-------------|
| `months_delayed` | Count of months with payment delay |
| `max_delay_months` | Worst payment delay severity |
| `credit_utilization` | Current balance / Credit limit |
| `total_risk_score` | Combined risk indicator (0-18) |
| `payment_ratio` | Payment amount / Bill amount |
| `delay_risk_*` | Risk scores per month |

## Model Performance

| Model | AUC | Accuracy | F1 Score |
|-------|-----|----------|----------|
| **XGBoost** | ~0.78 | ~82% | ~0.47 |
| Gradient Boosting | ~0.77 | ~81% | ~0.46 |
| Random Forest | ~0.77 | ~81% | ~0.45 |
| Logistic Regression | ~0.72 | ~78% | ~0.40 |

## SHAP Explainability

The `04_advanced_ml.py` notebook demonstrates:

- **Global feature importance** - Which features matter most overall
- **Individual explanations** - Why a specific customer was flagged as high risk
- **Regulatory compliance** - Audit trail for credit decisions

Example output:
```
INDIVIDUAL CUSTOMER RISK EXPLANATION
=====================================
Prediction: DEFAULT
Probability of Default: 73.5%

Top factors increasing risk:
  - pay_status_1: 2.00 (impact: +0.234)
  - credit_utilization: 0.89 (impact: +0.156)
  - months_delayed: 4.00 (impact: +0.098)
```

## License

MIT
