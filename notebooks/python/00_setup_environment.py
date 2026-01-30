# Databricks notebook source
# MAGIC %md
# MAGIC # Environment Setup & Data Ingestion
# MAGIC
# MAGIC Download Taiwan Credit Card Default dataset and create Bronze table.
# MAGIC
# MAGIC **Dataset:** UCI Taiwan Credit Card Default (30,000 records)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

SCHEMA_NAME = "kitsakis_credit_risk"
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Schema

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}")
spark.sql(f"USE {SCHEMA_NAME}")
print(f"Using schema: {SCHEMA_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies

# COMMAND ----------

%pip install openpyxl --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Dataset

# COMMAND ----------

import pandas as pd
import numpy as np

# Download directly from UCI repository
print(f"Downloading from: {DATASET_URL}")
df = pd.read_excel(DATASET_URL, header=1, engine='openpyxl')

print(f"Downloaded {len(df)} records with {len(df.columns)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean Column Names

# COMMAND ----------

# Rename columns to clean names
df.columns = [
    'id', 'credit_limit', 'sex', 'education', 'marriage', 'age',
    'pay_status_1', 'pay_status_2', 'pay_status_3', 'pay_status_4', 'pay_status_5', 'pay_status_6',
    'bill_amt_1', 'bill_amt_2', 'bill_amt_3', 'bill_amt_4', 'bill_amt_5', 'bill_amt_6',
    'pay_amt_1', 'pay_amt_2', 'pay_amt_3', 'pay_amt_4', 'pay_amt_5', 'pay_amt_6',
    'default_payment'
]

print("Columns renamed:")
print(df.columns.tolist())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Overview

# COMMAND ----------

print(f"Shape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")

# COMMAND ----------

df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Statistics

# COMMAND ----------

df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Bronze Table

# COMMAND ----------

# Drop old table if exists and save new one
spark.sql("DROP TABLE IF EXISTS bronze_credit_applications")

# Convert to Spark DataFrame and save
spark_df = spark.createDataFrame(df)
spark_df.write.format("delta").mode("overwrite").saveAsTable("bronze_credit_applications")

print(f"Bronze table created: {len(df)} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

spark.sql("SELECT COUNT(*) as total FROM bronze_credit_applications").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC - Downloaded 30,000 credit card records from UCI repository
# MAGIC - Cleaned column names
# MAGIC - Saved as Bronze Delta table
# MAGIC
# MAGIC **Next:** Run `01_silver_transformation`
