# Databricks notebook source
# MAGIC %md
# MAGIC # Environment Setup
# MAGIC
# MAGIC This notebook configures the environment for the Credit Risk ML Pipeline.
# MAGIC Using the **Taiwan Credit Card Default** dataset (30,000 records) from UCI ML Repository.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Get username for MLflow path
username = spark.sql("SELECT current_user()").first()[0]

# Project configuration
SCHEMA_NAME = "kitsakis_credit_risk"

# MLflow experiment
EXPERIMENT_PATH = f"/Users/{username}/credit_risk_experiment"

print(f"""
╔══════════════════════════════════════════════════════════════╗
║           Credit Risk ML Pipeline - Configuration            ║
╠══════════════════════════════════════════════════════════════╣
║ Schema:       {SCHEMA_NAME:<45} ║
║ MLflow Exp:   {EXPERIMENT_PATH:<45} ║
║ Dataset:      Taiwan Credit Card Default (30,000 records)    ║
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
# MAGIC ## Download Dataset
# MAGIC
# MAGIC **Taiwan Credit Card Default Dataset** from UCI ML Repository:
# MAGIC - 30,000 credit card clients from Taiwan
# MAGIC - Features: credit limit, gender, education, marital status, age, payment history
# MAGIC - Target: default payment (Yes=1, No=0)

# COMMAND ----------

import pandas as pd

# Download from UCI repository (Excel format)
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

print("Downloading dataset from UCI Repository...")
pdf = pd.read_excel(DATASET_URL, header=1)  # Skip first row (header is in row 2)

# Rename columns to be more readable
pdf.columns = [
    'id', 'credit_limit', 'gender', 'education', 'marital_status', 'age',
    'pay_status_sep', 'pay_status_aug', 'pay_status_jul',
    'pay_status_jun', 'pay_status_may', 'pay_status_apr',
    'bill_amt_sep', 'bill_amt_aug', 'bill_amt_jul',
    'bill_amt_jun', 'bill_amt_may', 'bill_amt_apr',
    'pay_amt_sep', 'pay_amt_aug', 'pay_amt_jul',
    'pay_amt_jun', 'pay_amt_may', 'pay_amt_apr',
    'default_payment'
]

# Convert to Spark DataFrame
raw_df = spark.createDataFrame(pdf)

print(f"✓ Dataset downloaded: {raw_df.count():,} records, {len(raw_df.columns)} columns")
raw_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset Overview

# COMMAND ----------

# Show column info
print("=== Dataset Columns ===")
for col in raw_df.columns:
    print(f"  - {col}")

print(f"\n=== Default Rate ===")
raw_df.groupBy("default_payment").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Bronze Table

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, lit

# Add metadata columns
bronze_df = (
    raw_df
    .withColumn("_ingestion_timestamp", current_timestamp())
    .withColumn("_source", lit("uci_taiwan_credit"))
)

# Save as managed Delta table
bronze_df.write.format("delta").mode("overwrite").saveAsTable("bronze_credit_applications")

print(f"✓ Bronze table created: {SCHEMA_NAME}.bronze_credit_applications")
print(f"  Records: {bronze_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Setup

# COMMAND ----------

# Verify table
tables = spark.sql(f"SHOW TABLES IN {SCHEMA_NAME}").collect()
print(f"Schema '{SCHEMA_NAME}' has {len(tables)} table(s):")
for t in tables:
    count = spark.table(f"{SCHEMA_NAME}.{t.tableName}").count()
    print(f"  - {t.tableName}: {count:,} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Complete!
# MAGIC
# MAGIC ✅ Schema created
# MAGIC ✅ Dataset downloaded (30,000 records)
# MAGIC ✅ Bronze table created
# MAGIC
# MAGIC **Next:** Run `02_silver_transformation`
