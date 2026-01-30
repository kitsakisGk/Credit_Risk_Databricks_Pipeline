# Credit Risk ML Pipeline

[![CI](https://github.com/kitsakisGk/credit-risk-databricks-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/kitsakisGk/credit-risk-databricks-pipeline/actions/workflows/ci.yml)
[![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=flat&logo=databricks&logoColor=white)](https://databricks.com)
[![Delta Lake](https://img.shields.io/badge/Delta%20Lake-00ADD8?style=flat&logo=delta&logoColor=white)](https://delta.io)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org)

End-to-end credit risk assessment pipeline built on Databricks Lakehouse, featuring Delta Lake, Structured Streaming, and MLflow.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Platform | Databricks |
| Storage | Delta Lake |
| Processing | Apache Spark |
| Streaming | Structured Streaming, Auto Loader |
| ML | MLflow, Spark ML |
| CI/CD | GitHub Actions |

## Features

- **Medallion Architecture** - Bronze, Silver, Gold data layers
- **Real-time Processing** - Streaming ingestion with exactly-once semantics
- **ML Pipeline** - Automated training, hyperparameter tuning, model registry
- **Data Quality** - Validation framework with quarantine handling
- **Production Ready** - Orchestration, CI/CD

## Quick Start

### Prerequisites
- Databricks workspace ([Community Edition](https://community.cloud.databricks.com/) works)
- Cluster with Runtime 13.3+ ML

### Setup

1. **Clone to Databricks**
   - Workspace → Git folder → `https://github.com/kitsakisGk/credit-risk-databricks-pipeline.git`

2. **Create Cluster**
   - Compute → Create → Select ML Runtime → Start

3. **Run Pipeline**
   ```
   00_setup_environment → 01_bronze_ingestion → 02_silver_transformation
   → 03_gold_aggregation → 04_ml_training → 05_streaming_simulation → 06_model_inference
   ```

## Project Structure

```
├── notebooks/
│   ├── 00_setup_environment.py
│   ├── 01_bronze_ingestion.py
│   ├── 02_silver_transformation.py
│   ├── 03_gold_aggregation.py
│   ├── 04_ml_training.py
│   ├── 05_streaming_simulation.py
│   └── 06_model_inference.py
├── src/utils/
│   ├── data_quality.py
│   └── feature_engineering.py
├── config/
│   ├── pipeline_config.yaml
│   └── workflow_config.json
└── .github/workflows/
    └── ci.yml
```

## Dataset

German Credit Risk dataset from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) - automatically downloaded during setup.

## License

MIT
