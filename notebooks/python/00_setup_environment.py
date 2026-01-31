# Databricks notebook source
# MAGIC %md
# MAGIC # Environment Setup & Data Ingestion
# MAGIC
# MAGIC Create Bronze table from the Taiwan Credit Card Default dataset.
# MAGIC
# MAGIC **Dataset:** UCI Taiwan Credit Card Default (30,000 records)
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC 1. Download dataset from: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
# MAGIC 2. Upload to Databricks: **Catalog** → **Create Table** → Upload Excel file
# MAGIC 3. Schema: `kitsakis_credit_risk`, Table: `bronze_credit_raw`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

SCHEMA_NAME = "kitsakis_credit_risk"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Schema

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}")
spark.sql(f"USE {SCHEMA_NAME}")
print(f"Using schema: {SCHEMA_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Uploaded Data
# MAGIC
# MAGIC If you uploaded the Excel file via Catalog, load it here.
# MAGIC If not, we'll try to download from a mirror.

# COMMAND ----------

import pandas as pd
import numpy as np

# Try to load from uploaded table first
try:
    df = spark.table("bronze_credit_raw").toPandas()
    print(f"Loaded {len(df)} records from uploaded table")
    FROM_UPLOAD = True
except:
    print("No uploaded table found. Trying to download from mirror...")
    FROM_UPLOAD = False

    # Try GitHub mirror
    DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/default_of_credit_card_clients.csv"
    try:
        df = pd.read_csv(DATA_URL)
        print(f"Downloaded {len(df)} records from GitHub mirror")
    except Exception as e:
        print(f"Download failed: {e}")
        print("\n" + "="*60)
        print("MANUAL UPLOAD REQUIRED")
        print("="*60)
        print("1. Download: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients")
        print("2. In Databricks: Catalog → Create Table → Upload")
        print("3. Schema: kitsakis_credit_risk")
        print("4. Table name: bronze_credit_raw")
        print("5. Re-run this notebook")
        raise Exception("Please upload data manually")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Examine Raw Data

# COMMAND ----------

print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean Column Names

# COMMAND ----------

# Standardize column names (handles both upload and download formats)
column_mapping = {
    # From Excel upload (X1, X2, etc.)
    '_c0': 'id', 'X1': 'credit_limit', 'X2': 'sex', 'X3': 'education',
    'X4': 'marriage', 'X5': 'age',
    'X6': 'pay_status_1', 'X7': 'pay_status_2', 'X8': 'pay_status_3',
    'X9': 'pay_status_4', 'X10': 'pay_status_5', 'X11': 'pay_status_6',
    'X12': 'bill_amt_1', 'X13': 'bill_amt_2', 'X14': 'bill_amt_3',
    'X15': 'bill_amt_4', 'X16': 'bill_amt_5', 'X17': 'bill_amt_6',
    'X18': 'pay_amt_1', 'X19': 'pay_amt_2', 'X20': 'pay_amt_3',
    'X21': 'pay_amt_4', 'X22': 'pay_amt_5', 'X23': 'pay_amt_6',
    'X24': 'default_payment', 'Y': 'default_payment',

    # From CSV download (LIMIT_BAL, etc.)
    'ID': 'id', 'LIMIT_BAL': 'credit_limit', 'SEX': 'sex',
    'EDUCATION': 'education', 'MARRIAGE': 'marriage', 'AGE': 'age',
    'PAY_0': 'pay_status_1', 'PAY_2': 'pay_status_2', 'PAY_3': 'pay_status_3',
    'PAY_4': 'pay_status_4', 'PAY_5': 'pay_status_5', 'PAY_6': 'pay_status_6',
    'BILL_AMT1': 'bill_amt_1', 'BILL_AMT2': 'bill_amt_2', 'BILL_AMT3': 'bill_amt_3',
    'BILL_AMT4': 'bill_amt_4', 'BILL_AMT5': 'bill_amt_5', 'BILL_AMT6': 'bill_amt_6',
    'PAY_AMT1': 'pay_amt_1', 'PAY_AMT2': 'pay_amt_2', 'PAY_AMT3': 'pay_amt_3',
    'PAY_AMT4': 'pay_amt_4', 'PAY_AMT5': 'pay_amt_5', 'PAY_AMT6': 'pay_amt_6',
    'default.payment.next.month': 'default_payment'
}

# Rename columns that exist
df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

# Remove header row if it got imported as data (from Excel)
if df['id'].dtype == 'object' and 'ID' in df['id'].values:
    df = df[df['id'] != 'ID']

# Convert to numeric
numeric_cols = ['id', 'credit_limit', 'sex', 'education', 'marriage', 'age',
                'pay_status_1', 'pay_status_2', 'pay_status_3', 'pay_status_4', 'pay_status_5', 'pay_status_6',
                'bill_amt_1', 'bill_amt_2', 'bill_amt_3', 'bill_amt_4', 'bill_amt_5', 'bill_amt_6',
                'pay_amt_1', 'pay_amt_2', 'pay_amt_3', 'pay_amt_4', 'pay_amt_5', 'pay_amt_6',
                'default_payment']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("Cleaned columns:")
print(df.columns.tolist())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Overview

# COMMAND ----------

print(f"Shape: {df.shape}")
df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Statistics

# COMMAND ----------

df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Target Distribution

# COMMAND ----------

print("Default Payment Distribution:")
print(df['default_payment'].value_counts())
print(f"\nDefault Rate: {df['default_payment'].mean()*100:.2f}%")

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

spark.sql("SELECT * FROM bronze_credit_applications LIMIT 5").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC - Loaded 30,000 credit card records
# MAGIC - Cleaned and standardized column names
# MAGIC - Saved as Bronze Delta table
# MAGIC - Default rate: ~22%
# MAGIC
# MAGIC **Next:** Run `01_silver_transformation`
