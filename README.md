# Credit Risk ML Pipeline

[![CI](https://github.com/kitsakisGk/credit-risk-databricks-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/kitsakisGk/credit-risk-databricks-pipeline/actions/workflows/ci.yml)
![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=flat&logo=databricks&logoColor=white)
![Delta Lake](https://img.shields.io/badge/Delta%20Lake-00ADD8?style=flat&logo=delta&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C?style=flat&logo=apachespark&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

End-to-end **credit risk assessment pipeline** built on **Databricks Lakehouse**, demonstrating production-grade data engineering and machine learning practices.

## Overview

This project implements a complete ML pipeline for credit risk prediction using the **Medallion Architecture** (Bronze → Silver → Gold), with real-time streaming capabilities and MLflow experiment tracking.

### Key Highlights

- **Data Engineering**: Medallion architecture with Delta Lake for ACID transactions and time travel
- **Streaming**: Real-time data ingestion using Structured Streaming and Auto Loader
- **Machine Learning**: Model training, hyperparameter tuning, and registry with MLflow
- **Data Quality**: Automated validation with quarantine handling for bad records
- **Orchestration**: End-to-end workflow with Databricks Jobs

## Tech Stack

| Layer | Technology |
|-------|------------|
| Platform | Databricks Lakehouse |
| Storage | Delta Lake |
| Compute | Apache Spark |
| Streaming | Structured Streaming, Auto Loader |
| ML | MLflow, Spark MLlib |
| CI/CD | GitHub Actions |

## Getting Started

### Prerequisites

- [Databricks Community Edition](https://community.cloud.databricks.com/) (free)
- Cluster with **Runtime 13.3+ ML**

### Installation

1. **Import Repository to Databricks**
   ```
   Workspace → Create → Git folder → https://github.com/kitsakisGk/credit-risk-databricks-pipeline.git
   ```

2. **Create Compute Cluster**
   ```
   Compute → Create compute → Select ML Runtime → Create
   ```

3. **Run Notebooks in Order**
   ```
   00_setup_environment
   01_bronze_ingestion
   02_silver_transformation
   03_gold_aggregation
   04_ml_training
   05_streaming_simulation
   06_model_inference
   ```

## Pipeline Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   SOURCE    │───▶│   BRONZE    │───▶│   SILVER    │───▶│    GOLD     │
│  (Raw Data) │    │  (Landing)  │    │ (Validated) │    │ (Features)  │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                │
                   ┌─────────────┐    ┌─────────────┐           │
                   │   MODEL     │◀───│  TRAINING   │◀──────────┘
                   │  REGISTRY   │    │  (MLflow)   │
                   └─────────────┘    └─────────────┘
```

## Project Structure

```
├── notebooks/                    # Databricks notebooks
│   ├── 00_setup_environment.py   # Configuration and setup
│   ├── 01_bronze_ingestion.py    # Raw data ingestion
│   ├── 02_silver_transformation.py # Data cleaning
│   ├── 03_gold_aggregation.py    # Feature engineering
│   ├── 04_ml_training.py         # Model training
│   ├── 05_streaming_simulation.py # Streaming demo
│   └── 06_model_inference.py     # Predictions
├── src/utils/                    # Reusable modules
├── config/                       # Configuration files
└── .github/workflows/            # CI/CD
```

## Dataset

[German Credit Risk Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) from UCI ML Repository (1,000 records, 20 features).

## License

MIT
