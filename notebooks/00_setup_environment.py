# Databricks notebook source
# MAGIC %md
# MAGIC # Environment Setup
# MAGIC
# MAGIC This notebook configures the environment for the Credit Risk ML Pipeline.
# MAGIC Run this first to set up the schema and download data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Get username for MLflow path
username = spark.sql("SELECT current_user()").first()[0]

# Project configuration - clean schema name
SCHEMA_NAME = "kitsakis_credit_risk"

# MLflow experiment
EXPERIMENT_PATH = f"/Users/{username}/credit_risk_experiment"

print(f"""
╔══════════════════════════════════════════════════════════════╗
║           Credit Risk ML Pipeline - Configuration            ║
╠══════════════════════════════════════════════════════════════╣
║ Schema:       {SCHEMA_NAME:<45} ║
║ MLflow Exp:   {EXPERIMENT_PATH:<45} ║
╚══════════════════════════════════════════════════════════════╝
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Schema (Database)

# COMMAND ----------

# Create schema in default catalog
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}")
spark.sql(f"USE {SCHEMA_NAME}")

print(f"✓ Schema '{SCHEMA_NAME}' created and selected")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download and Load Dataset
# MAGIC
# MAGIC We'll use the German Credit Risk dataset from UCI Machine Learning Repository.

# COMMAND ----------

import urllib.request
import pandas as pd

# Column names for German Credit dataset
COLUMN_NAMES = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_duration",
    "installment_rate",
    "personal_status",
    "other_parties",
    "residence_duration",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
    "credit_risk"
]

# Download dataset
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
local_path = "/tmp/german_credit.data"
urllib.request.urlretrieve(DATASET_URL, local_path)

# Load into pandas then convert to Spark
pdf = pd.read_csv(local_path, sep=' ', header=None, names=COLUMN_NAMES)
raw_df = spark.createDataFrame(pdf)

print(f"✓ Dataset downloaded: {raw_df.count()} records, {len(raw_df.columns)} columns")
raw_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Raw Data Table (Bronze)

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, lit, sha2, concat_ws, col

# Add metadata columns
bronze_df = (
    raw_df
    .withColumn("_ingestion_timestamp", current_timestamp())
    .withColumn("_source", lit("uci_german_credit"))
    .withColumn("_row_hash", sha2(concat_ws("||", *[col(c) for c in COLUMN_NAMES]), 256))
)

# Save as managed Delta table
bronze_df.write.format("delta").mode("overwrite").saveAsTable("bronze_credit_applications")

print(f"✓ Bronze table created: {SCHEMA_NAME}.bronze_credit_applications")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Setup

# COMMAND ----------

# Verify table
tables = spark.sql(f"SHOW TABLES IN {SCHEMA_NAME}").collect()
print(f"Schema '{SCHEMA_NAME}' has {len(tables)} table(s):")
for t in tables:
    count = spark.table(f"{SCHEMA_NAME}.{t.tableName}").count()
    print(f"  - {t.tableName}: {count} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Complete!
# MAGIC
# MAGIC ✅ Schema created
# MAGIC ✅ Data downloaded
# MAGIC ✅ Bronze table created
# MAGIC
# MAGIC **Next:** Run `01_bronze_ingestion` (already done above) or go to `02_silver_transformation`
