# Credit Risk ML Pipeline

[![CI](https://github.com/kitsakisGk/credit-risk-databricks-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/kitsakisGk/credit-risk-databricks-pipeline/actions/workflows/ci.yml)
![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=flat&logo=databricks&logoColor=white)
![Delta Lake](https://img.shields.io/badge/Delta%20Lake-00ADD8?style=flat&logo=delta&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C?style=flat&logo=apachespark&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)

End-to-end **credit risk assessment pipeline** built on **Databricks Lakehouse**, demonstrating production-grade data engineering and machine learning practices.

## Overview

This project implements a complete ML pipeline for credit risk prediction using the **Medallion Architecture** (Bronze → Silver → Gold), with MLflow experiment tracking.

### Key Highlights

- **Data Engineering**: Medallion architecture with Delta Lake
- **Machine Learning**: Model training, comparison, and tracking with MLflow
- **Data Quality**: Automated validation with quality checks
- **Feature Engineering**: Risk scoring and feature creation

## Tech Stack

| Layer | Technology |
|-------|------------|
| Platform | Databricks Lakehouse |
| Storage | Delta Lake |
| Compute | Apache Spark |
| ML | MLflow, Spark MLlib |
| CI/CD | GitHub Actions |

## Getting Started

### Prerequisites

- [Databricks Workspace](https://databricks.com/) (Free trial or Community Edition)

### Installation

1. **Import Repository to Databricks**
   ```
   Workspace → Create → Git folder → https://github.com/kitsakisGk/credit-risk-databricks-pipeline.git
   ```

2. **Run Notebooks in Order**
   ```
   00_setup_environment    →  Downloads data, creates Bronze table
   02_silver_transformation →  Cleans data, creates Silver table
   03_gold_aggregation     →  Feature engineering, creates Gold table
   04_ml_training          →  Trains ML models with MLflow
   ```

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
│   ├── 00_setup_environment.py    # Setup + data ingestion
│   ├── 02_silver_transformation.py # Data cleaning
│   ├── 03_gold_aggregation.py     # Feature engineering
│   └── 04_ml_training.py          # Model training
├── src/utils/                      # Reusable modules
├── config/                         # Configuration files
└── .github/workflows/              # CI/CD
```

## Dataset

[German Credit Risk Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) from UCI ML Repository (1,000 records, 20 features).

## License

MIT
