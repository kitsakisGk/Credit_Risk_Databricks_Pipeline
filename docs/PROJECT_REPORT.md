# Credit Risk ML Pipeline - Project Report

## Executive Summary

This project implements a **production-grade credit risk assessment pipeline** on Databricks, demonstrating skills in data engineering, machine learning, and regulatory compliance aligned with **Swiss banking standards**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Data Pipeline (Medallion)](#data-pipeline-medallion)
4. [Machine Learning Models](#machine-learning-models)
5. [Swiss Banking Compliance](#swiss-banking-compliance)
6. [Code Walkthrough](#code-walkthrough)
7. [Results & Metrics](#results--metrics)
8. [How to Present](#how-to-present)

---

## Project Overview

### Business Problem
Banks need to predict which customers are likely to default on credit card payments. This helps with:
- **Risk Management**: Identify high-risk customers
- **Regulatory Compliance**: Meet Basel III/IV requirements
- **Business Decisions**: Set credit limits, interest rates

### Solution
An end-to-end ML pipeline that:
1. Ingests 30,000 customer records
2. Transforms and cleans data (Medallion Architecture)
3. Engineers 37 features
4. Trains 4 ML models
5. Provides explainable predictions (SHAP)

### Tech Stack
| Component | Technology |
|-----------|------------|
| Platform | Databricks Lakehouse |
| Storage | Delta Lake |
| Processing | pandas, NumPy |
| ML | scikit-learn, XGBoost |
| Explainability | SHAP |
| CI/CD | GitHub Actions |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                               │
│                   UCI ML Repository (30K records)                 │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      BRONZE LAYER (Raw)                          │
│  • Raw data ingestion                                            │
│  • Schema: 25 columns                                            │
│  • Table: bronze_credit_applications                             │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      SILVER LAYER (Cleaned)                      │
│  • Decoded categoricals (gender, education, marital)             │
│  • Payment behavior features                                     │
│  • Table: silver_credit_applications                             │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      GOLD LAYER (Features)                       │
│  • 37 engineered features                                        │
│  • Risk scores                                                   │
│  • Table: gold_credit_features                                   │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
┌─────────────────────────┐ ┌─────────────────────────┐
│   BASELINE MODELS       │ │   ADVANCED ML           │
│  • Logistic Regression  │ │  • XGBoost              │
│  • Random Forest        │ │  • SHAP Explainability  │
│  • Gradient Boosting    │ │  • Cross-Validation     │
└─────────────────────────┘ └─────────────────────────┘
```

---

## Data Pipeline (Medallion)

### Bronze Layer (`00_setup_environment.py`)

**Purpose**: Raw data ingestion

**What it does**:
```python
# Download from UCI repository
df = pd.read_excel(DATASET_URL, header=1, engine='openpyxl')

# Rename columns to clean names
df.columns = ['id', 'credit_limit', 'sex', 'education', ...]

# Save as Delta table
spark_df.write.format("delta").saveAsTable("bronze_credit_applications")
```

**Output**: 30,000 records with 25 columns

---

### Silver Layer (`01_silver_transformation.py`)

**Purpose**: Data cleaning and transformation

**What it does**:
```python
# Decode gender (1=male, 2=female)
df['gender'] = df['sex'].map({1: 'male', 2: 'female'})

# Decode education
df['education_level'] = df['education'].map({
    1: 'graduate_school',
    2: 'university',
    3: 'high_school'
})

# Create payment behavior features
df['months_delayed'] = (df[pay_status_cols] > 0).sum(axis=1)
df['total_bill_amt'] = df[bill_cols].sum(axis=1)
```

**Features Created**:
| Feature | Description |
|---------|-------------|
| `gender` | Decoded from numeric |
| `education_level` | graduate_school, university, high_school, other |
| `marital_status` | married, single, other |
| `months_delayed` | Count of months with late payments |
| `max_delay_months` | Worst delay severity |
| `total_bill_amt` | Sum of 6-month bills |
| `total_pay_amt` | Sum of 6-month payments |

---

### Gold Layer (`02_gold_features.py`)

**Purpose**: Feature engineering for ML

**What it does**:
```python
# Credit utilization
df['credit_utilization'] = df['bill_amt_1'] / df['credit_limit']

# Payment ratio
df['payment_ratio'] = df['pay_amt_1'] / df['bill_amt_1']

# Risk scores
df['delay_risk_1'] = df['pay_status_1'].apply(delay_risk)
df['education_risk'] = df['education_level'].map(education_risk_map)

# Combined risk score
df['total_risk_score'] = (
    df['delay_risk_1'] + df['delay_risk_2'] + df['delay_risk_3'] +
    df['education_risk'] + df['marital_risk'] + df['utilization_risk']
)
```

**Features Created** (37 total):
| Category | Features |
|----------|----------|
| Credit | `credit_utilization`, `log_credit_limit`, `payment_ratio` |
| Age | `age_group`, `is_young_borrower`, `credit_bucket` |
| Risk Scores | `delay_risk_1/2/3`, `education_risk`, `marital_risk`, `utilization_risk` |
| Aggregates | `avg_bill_amount`, `avg_payment_amount`, `total_risk_score` |

---

## Machine Learning Models

### Baseline Models (`03_ml_training.py`)

**Models Trained**:
1. **Logistic Regression** - Linear baseline
2. **Random Forest** - 100 trees, max depth 10
3. **Gradient Boosting** - 100 estimators, max depth 5

**Code**:
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)

# Evaluate
y_prob = rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
```

---

### Advanced ML (`04_advanced_ml.py`)

**XGBoost** - Industry standard in Swiss banking:
```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])  # Handle imbalance
)
```

**SHAP Explainability**:
```python
import shap

# Calculate SHAP values
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Feature importance
shap_importance = np.abs(shap_values).mean(axis=0)
```

**Cross-Validation**:
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='roc_auc')
# Mean AUC: 0.78 (+/- 0.02)
```

---

## Swiss Banking Compliance

### Why This Matters

Swiss financial regulations (FINMA) require:
1. **Model Interpretability** - Must explain why a loan was denied
2. **Robust Validation** - Cross-validation, not single train/test
3. **Audit Trail** - Document all model decisions

### How We Address It

| Requirement | Solution |
|-------------|----------|
| Interpretability | SHAP values for individual predictions |
| Validation | 5-fold cross-validation |
| Audit Trail | Save predictions with probabilities |

### Individual Explanation Example:
```
CUSTOMER RISK EXPLANATION
==========================
Prediction: DEFAULT
Probability: 73.5%

Top factors increasing risk:
  - pay_status_1: 2.00 (impact: +0.234)
  - credit_utilization: 0.89 (impact: +0.156)
  - months_delayed: 4.00 (impact: +0.098)
```

---

## Results & Metrics

### Model Performance

| Model | AUC | Accuracy | F1 Score |
|-------|-----|----------|----------|
| **XGBoost** | 0.78 | 82% | 0.47 |
| Gradient Boosting | 0.78 | 82% | 0.48 |
| Random Forest | 0.78 | 82% | 0.47 |
| Logistic Regression | 0.76 | 82% | 0.47 |

### Confusion Matrix (Best Model)

```
                    Predicted
                 |  No   |  Yes  |
        ---------|-------|-------|
Actual    No     | 4400  |  273  |
          Yes    |  823  |  504  |
        ---------|-------|-------|

Precision: 0.65
Recall: 0.38
```

### Key Insights

1. **Payment history** is the strongest predictor of default
2. **Credit utilization** above 80% significantly increases risk
3. **Young borrowers** (< 30) have higher default rates
4. Tree-based models outperform logistic regression

---

## Tables Created

| Table | Layer | Records | Purpose |
|-------|-------|---------|---------|
| `bronze_credit_applications` | Bronze | 30,000 | Raw data |
| `silver_credit_applications` | Silver | 30,000 | Cleaned data |
| `gold_credit_features` | Gold | 30,000 | ML-ready features |
| `model_predictions` | Output | 6,000 | Baseline predictions |
| `xgboost_predictions` | Output | 6,000 | XGBoost predictions |

---

## Workflow Configuration

The pipeline can be scheduled as a Databricks Workflow:

```
01_bronze_ingestion
        │
        ▼
02_silver_transformation
        │
        ▼
03_gold_features
        │
   ┌────┴────┐
   ▼         ▼
04_ml    05_advanced_ml
```

- **Schedule**: Daily at 6 AM (Europe/Zurich)
- **Retries**: 2 attempts for data tasks, 1 for ML
- **Notifications**: Email on success/failure

---

## Skills Demonstrated

### Data Engineering
- Medallion Architecture (Bronze/Silver/Gold)
- Delta Lake tables
- Data quality checks
- Feature engineering

### Machine Learning
- Multiple model training
- Hyperparameter tuning
- Cross-validation
- Handling imbalanced data

### MLOps
- CI/CD with GitHub Actions
- Workflow orchestration
- Model versioning (via table saves)

### Regulatory Compliance
- SHAP explainability
- Audit trails
- Swiss banking standards (FINMA)

---

## Repository Structure

```
credit-risk-databricks-pipeline/
├── notebooks/
│   ├── python/
│   │   ├── 00_setup_environment.py      # Bronze ingestion
│   │   ├── 01_silver_transformation.py  # Silver cleaning
│   │   ├── 02_gold_features.py          # Gold features
│   │   ├── 03_ml_training.py            # Baseline models
│   │   └── 04_advanced_ml.py            # XGBoost + SHAP
│   └── sql/                             # SQL Warehouse version
├── workflows/
│   └── credit_risk_pipeline.json        # Databricks Workflow
├── docs/
│   ├── PROJECT_REPORT.md                # This document
│   └── LINKEDIN_POST.md                 # LinkedIn template
├── src/utils/                           # Reusable modules
├── .github/workflows/ci.yml             # CI/CD
└── README.md                            # Project overview
```
