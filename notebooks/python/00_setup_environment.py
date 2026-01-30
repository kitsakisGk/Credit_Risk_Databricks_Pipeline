# Databricks notebook source
# MAGIC %md
# MAGIC # Environment Setup & Data Ingestion
# MAGIC
# MAGIC Download Taiwan Credit Card Default dataset and create Bronze table.
# MAGIC
# MAGIC **Dataset:** UCI Taiwan Credit Card Default (30,000 records)
# MAGIC
# MAGIC **Requirements:** Databricks with Python compute (not SQL Warehouse)

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
# MAGIC ## Download and Load Dataset

# COMMAND ----------

import pandas as pd

# Download directly from UCI repository
print(f"Downloading from: {DATASET_URL}")
pdf = pd.read_excel(DATASET_URL, header=1)

print(f"Downloaded {len(pdf)} records with {len(pdf.columns)} columns")
print(f"Columns: {list(pdf.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean Column Names

# COMMAND ----------

# Rename columns to clean names
pdf.columns = [
    'id', 'credit_limit', 'sex', 'education', 'marriage', 'age',
    'pay_status_1', 'pay_status_2', 'pay_status_3', 'pay_status_4', 'pay_status_5', 'pay_status_6',
    'bill_amt_1', 'bill_amt_2', 'bill_amt_3', 'bill_amt_4', 'bill_amt_5', 'bill_amt_6',
    'pay_amt_1', 'pay_amt_2', 'pay_amt_3', 'pay_amt_4', 'pay_amt_5', 'pay_amt_6',
    'default_payment'
]

print("Columns renamed successfully")
pdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Bronze Table

# COMMAND ----------

from pyspark.sql.functions import current_timestamp

# Convert to Spark DataFrame
bronze_df = spark.createDataFrame(pdf)

# Add ingestion timestamp
bronze_df = bronze_df.withColumn("_ingested_at", current_timestamp())

# Save as Delta table
bronze_df.write.format("delta").mode("overwrite").saveAsTable("bronze_credit_applications")

print(f"Bronze table created: {bronze_df.count()} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Data

# COMMAND ----------

# Quick verification
spark.sql("SELECT COUNT(*) as total FROM bronze_credit_applications").show()

# COMMAND ----------

# Sample data
spark.sql("SELECT * FROM bronze_credit_applications LIMIT 5").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Summary

# COMMAND ----------

spark.sql("""
    SELECT
        MIN(credit_limit) as min_credit,
        MAX(credit_limit) as max_credit,
        ROUND(AVG(credit_limit), 2) as avg_credit,
        MIN(age) as min_age,
        MAX(age) as max_age,
        SUM(default_payment) as total_defaults,
        ROUND(100.0 * SUM(default_payment) / COUNT(*), 2) as default_rate_pct
    FROM bronze_credit_applications
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC - Downloaded 30,000 credit card records from UCI repository
# MAGIC - Created Bronze Delta table with raw data
# MAGIC - Default rate: ~22%
# MAGIC
# MAGIC **Next:** Run `01_silver_transformation`
