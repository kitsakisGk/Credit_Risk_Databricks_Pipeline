# Credit Risk ML Pipeline

[![CI](https://github.com/kitsakisGk/Credit_Risk_Databricks_Pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/kitsakisGk/Credit_Risk_Databricks_Pipeline/actions/workflows/ci.yml)
![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=flat&logo=databricks&logoColor=white)
![Delta Lake](https://img.shields.io/badge/Delta%20Lake-00ADD8?style=flat&logo=delta&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat&logo=xgboost&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

End-to-end **credit risk assessment pipeline** built on **Databricks Lakehouse**, demonstrating production-grade data engineering and machine learning practices aligned with **Swiss banking standards**.

## Quick Start Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/kitsakisGk/Credit_Risk_Databricks_Pipeline.git
```

### Step 2: Upload Data to Databricks

The dataset is included in this repo: `data/default of credit card clients.xls`

1. Open **Databricks** workspace
2. Go to **Catalog** (left sidebar)
3. Click **Create Table** → **Upload File**
4. Upload the file: `data/default of credit card clients.xls`
5. Configure:
   - **Catalog**: `workspace` (or your catalog)
   - **Schema**: `kitsakis_credit_risk` (create new if needed)
   - **Table name**: `bronze_credit_raw`
   - **First row contains header**: ✅ Check this
6. Click **Create Table**

### Step 3: Import Notebooks to Databricks

1. In Databricks, go to **Workspace**
2. Click **Import** → **URL**
3. Enter: `https://github.com/kitsakisGk/Credit_Risk_Databricks_Pipeline`
4. Or manually copy notebook contents

### Step 4: Run Notebooks in Order

```
1. notebooks/python/00_setup_environment.py   → Creates Bronze table
2. notebooks/python/01_silver_transformation.py → Data cleaning
3. notebooks/python/02_gold_features.py        → Feature engineering
4. notebooks/python/03_ml_training.py          → Baseline ML models
5. notebooks/python/04_advanced_ml.py          → XGBoost + SHAP
```

**Note:** The notebooks handle both:
- ✅ Manual upload (recommended)
- ✅ Auto-download from mirror (fallback)
- ✅ Different column formats (Excel upload vs CSV)

---

## Overview

This project implements a complete ML pipeline for credit risk prediction using the **Medallion Architecture** (Bronze → Silver → Gold), featuring techniques used by Swiss financial institutions (UBS, Credit Suisse, Julius Baer).

### Key Highlights

- **Medallion Architecture**: Bronze → Silver → Gold data layers with Delta Lake
- **Swiss Banking ML**: XGBoost + SHAP (used at UBS, Credit Suisse)
- **Regulatory Compliance**: Model explainability for FINMA requirements
- **Production Ready**: CI/CD, cross-validation, hyperparameter tuning

## Dataset

**Taiwan Credit Card Default Dataset** from UCI ML Repository

| Property | Value |
|----------|-------|
| Records | 30,000 |
| Features | 23 |
| Target | Default payment (binary) |
| Default Rate | ~22% |
| File | `data/default of credit card clients.xls` |

Source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

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

## Pipeline Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   SOURCE    │───▶│   BRONZE    │───▶│   SILVER    │───▶│    GOLD     │
│  (Excel)    │    │    (Raw)    │    │  (Cleaned)  │    │ (Features)  │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                               │
                   ┌─────────────┐    ┌─────────────┐          │
                   │    SHAP     │◀───│   XGBOOST   │◀─────────┘
                   │(Explainability)  │  (Training) │
                   └─────────────┘    └─────────────┘
```

## Project Structure

```
Credit_Risk_Databricks_Pipeline/
├── data/
│   └── default of credit card clients.xls   # Dataset (30K records)
├── notebooks/
│   ├── python/                              # Main pipeline
│   │   ├── 00_setup_environment.py          # Bronze ingestion
│   │   ├── 01_silver_transformation.py      # Data cleaning
│   │   ├── 02_gold_features.py              # Feature engineering
│   │   ├── 03_ml_training.py                # Baseline models
│   │   └── 04_advanced_ml.py                # XGBoost + SHAP
│   └── sql/                                 # SQL Warehouse version
├── workflows/
│   └── credit_risk_pipeline.json            # Databricks Workflow config
├── docs/
│   ├── PROJECT_REPORT.md                    # Full technical report
│   └── LINKEDIN_POST.md                     # LinkedIn post templates
└── .github/workflows/ci.yml                 # CI/CD
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
| Gradient Boosting | ~0.78 | ~82% | ~0.48 |
| Random Forest | ~0.78 | ~82% | ~0.47 |
| Logistic Regression | ~0.76 | ~82% | ~0.47 |

## SHAP Explainability

The `04_advanced_ml.py` notebook demonstrates model interpretability:

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

## Troubleshooting

### Column Names Issue

The notebook automatically handles different column formats:

| Source | Columns Look Like |
|--------|-------------------|
| Excel Upload | `_c0`, `X1`, `X2`, `X3`... |
| CSV Download | `ID`, `LIMIT_BAL`, `SEX`... |

Both are automatically converted to: `id`, `credit_limit`, `sex`, `education`...

### Common Errors

| Error | Solution |
|-------|----------|
| `Table not found` | Run `00_setup_environment.py` first |
| `Schema not found` | The notebook creates it automatically |
| `Column not found` | Notebook handles column renaming |

## License

MIT

## Author

Built for Swiss banking/finance job applications. Demonstrates:
- Data Engineering (Medallion Architecture)
- Machine Learning (XGBoost, sklearn)
- Regulatory Compliance (SHAP explainability)
- MLOps (CI/CD, Workflows)
