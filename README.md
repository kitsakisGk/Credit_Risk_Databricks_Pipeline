# Credit Risk ML Pipeline on Databricks

[![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)](https://databricks.com)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)](https://spark.apache.org)
[![Delta Lake](https://img.shields.io/badge/Delta%20Lake-00ADD8?style=for-the-badge&logo=delta&logoColor=white)](https://delta.io)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)

A production-grade **data engineering** and **ML pipeline** for credit risk assessment, demonstrating modern Databricks Lakehouse capabilities. Built to showcase skills for Data/ML Engineering roles in financial services.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES                                       │
│    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│    │  Batch Files │    │   Streaming  │    │   API Data   │                 │
│    │    (CSV)     │    │   (Kafka)    │    │   (REST)     │                 │
│    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                 │
└───────────┼───────────────────┼───────────────────┼─────────────────────────┘
            │                   │                   │
            ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATABRICKS LAKEHOUSE                                  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         MEDALLION ARCHITECTURE                         │ │
│  │                                                                        │ │
│  │   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐           │ │
│  │   │   BRONZE    │      │   SILVER    │      │    GOLD     │           │ │
│  │   │─────────────│      │─────────────│      │─────────────│           │ │
│  │   │ • Raw Data  │ ──▶  │ • Cleaned   │ ──▶  │ • Features  │           │ │
│  │   │ • Append    │      │ • Validated │      │ • Aggregates│           │ │
│  │   │ • Streaming │      │ • Deduped   │      │ • ML-Ready  │           │ │
│  │   └─────────────┘      └─────────────┘      └──────┬──────┘           │ │
│  │                                                    │                   │ │
│  └────────────────────────────────────────────────────┼───────────────────┘ │
│                                                       │                      │
│  ┌────────────────────────────────────────────────────▼───────────────────┐ │
│  │                          ML PIPELINE                                   │ │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │ │
│  │   │   Feature    │  │    Model     │  │    Model     │                │ │
│  │   │ Engineering  │──│   Training   │──│   Registry   │                │ │
│  │   │              │  │   (MLflow)   │  │  (Staging)   │                │ │
│  │   └──────────────┘  └──────────────┘  └──────────────┘                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Platform** | Databricks (Community Edition compatible) |
| **Storage** | Delta Lake - ACID transactions, time travel, schema evolution |
| **Processing** | Apache Spark (PySpark + Spark SQL) |
| **Streaming** | Structured Streaming, Auto Loader |
| **ML Tracking** | MLflow - experiments, model registry |
| **Orchestration** | Databricks Workflows |
| **Data Quality** | Custom validation framework |

## Project Structure

```
credit-risk-pipeline/
│
├── notebooks/                          # Databricks notebooks
│   ├── 00_setup_environment.py         # Environment configuration
│   ├── 01_bronze_ingestion.py          # Raw data ingestion (batch + streaming)
│   ├── 02_silver_transformation.py     # Data cleaning & validation
│   ├── 03_gold_aggregation.py          # Feature engineering
│   ├── 04_ml_training.py               # Model training with MLflow
│   ├── 05_streaming_simulation.py      # Real-time processing demo
│   └── 06_model_inference.py           # Batch & streaming predictions
│
├── src/utils/                          # Reusable Python modules
│   ├── __init__.py
│   ├── data_quality.py                 # Data quality validation framework
│   └── feature_engineering.py          # Feature transformation utilities
│
├── config/
│   ├── pipeline_config.yaml            # Pipeline configuration
│   └── workflow_config.json            # Databricks Workflow definition
│
├── data/sample/                        # Sample data for testing
│   └── sample_credit_data.csv
│
├── tests/
│   └── test_transformations.py         # Unit tests
│
├── requirements.txt
└── README.md
```

## Dataset

This project uses the **German Credit Risk** dataset from the UCI Machine Learning Repository:
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- **Records**: 1,000 loan applications
- **Features**: 20 attributes (demographics, financial history, loan details)
- **Target**: Credit risk classification (Good/Bad)

The dataset is automatically downloaded when running the setup notebook.

## Key Features Demonstrated

### Data Engineering
- **Medallion Architecture** - Bronze/Silver/Gold layers with clear data contracts
- **Delta Lake** - ACID transactions, time travel, schema enforcement
- **Streaming Ingestion** - Auto Loader with exactly-once semantics
- **Data Quality Framework** - Configurable validation rules with quarantine
- **Incremental Processing** - Efficient MERGE/upsert operations

### Machine Learning
- **MLflow Tracking** - Full experiment tracking with parameters and metrics
- **Model Comparison** - Automated evaluation of multiple algorithms
- **Hyperparameter Tuning** - Cross-validation with grid search
- **Model Registry** - Versioning and staging for deployment
- **Feature Engineering** - 40+ engineered features with domain knowledge

### Production Patterns
- **Workflow Orchestration** - End-to-end pipeline with dependencies
- **Configuration Management** - Externalized config for environments
- **Testing** - Unit tests for transformations
- **Monitoring** - Data quality metrics and model performance

## Getting Started

### Prerequisites
- Databricks account (free [Community Edition](https://community.cloud.databricks.com/) works!)
- Git installed locally

### Quick Start

#### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-databricks-pipeline.git
cd credit-risk-databricks-pipeline
```

#### 2. Import to Databricks

**Option A: Import via Git (Recommended)**
1. In Databricks, go to **Workspace** > **Users** > Your username
2. Click **Create** > **Git folder**
3. Enter the repository URL
4. Click **Create Git folder**

**Option B: Manual Import**
1. In Databricks, go to **Workspace**
2. Right-click your user folder > **Import**
3. Upload the notebooks folder as a `.dbc` archive or individual files

#### 3. Create a Cluster
1. Go to **Compute** > **Create Cluster**
2. Settings:
   - **Runtime**: 13.3 LTS or later (ML Runtime for notebook 04+)
   - **Node type**: Smallest available (Community Edition auto-selects)
   - **Workers**: 0 (single node for Community Edition)
3. Click **Create Cluster**

#### 4. Run the Pipeline
Execute notebooks in order:
```
00_setup_environment  →  Creates database, downloads data
01_bronze_ingestion   →  Ingests raw data to Bronze layer
02_silver_transformation  →  Cleans and validates data
03_gold_aggregation   →  Engineers ML features
04_ml_training        →  Trains and registers models
05_streaming_simulation  →  Demonstrates real-time processing
06_model_inference    →  Runs batch/streaming predictions
```

## Sample Results

### Model Performance
| Model | Test AUC | Test Accuracy | Test F1 |
|-------|----------|---------------|---------|
| Gradient Boosted Trees | 0.78 | 0.75 | 0.72 |
| Random Forest | 0.76 | 0.73 | 0.70 |
| Logistic Regression | 0.74 | 0.71 | 0.68 |

### Top Features
1. `checking_risk_score` - Checking account status
2. `combined_risk_score` - Composite risk indicator
3. `duration_months` - Loan duration
4. `credit_amount` - Loan amount
5. `age` - Applicant age

## Skills Showcased

This project demonstrates proficiency in:

- **Databricks Platform** - Notebooks, clusters, workflows, DBFS
- **Apache Spark** - DataFrames, SQL, transformations, optimizations
- **Delta Lake** - ACID, time travel, schema evolution, MERGE
- **Streaming** - Structured Streaming, watermarks, checkpoints
- **MLflow** - Experiment tracking, model registry, serving
- **Python** - PySpark, pandas, data engineering patterns
- **Software Engineering** - Testing, configuration, documentation

## Future Enhancements

- [ ] Unity Catalog integration for data governance
- [ ] Delta Live Tables declarative pipelines
- [ ] Feature Store integration
- [ ] Model serving endpoint with REST API
- [ ] CI/CD with GitHub Actions
- [ ] Dashboard with Databricks SQL

## License

MIT License - feel free to use this for your own portfolio!

## Author

Built as a portfolio project for Data/ML Engineering roles.

---

**Questions or feedback?** Open an issue or reach out on LinkedIn!
