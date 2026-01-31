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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Schema

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}")
spark.sql(f"USE {SCHEMA_NAME}")
print(f"Using schema: {SCHEMA_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Dataset
# MAGIC
# MAGIC Using Kaggle mirror which is more reliable than UCI direct link.

# COMMAND ----------

import pandas as pd
import numpy as np

# Direct CSV URL from a reliable source
# This is the same UCI dataset hosted on a stable mirror
DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/default_of_credit_card_clients.csv"

print(f"Downloading from: {DATA_URL}")
df = pd.read_csv(DATA_URL)
print(f"Downloaded {len(df)} records with {len(df.columns)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Examine Raw Data

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean Column Names

# COMMAND ----------

# The CSV has different column names, let's standardize them
# First check what we have
print("Original columns:")
print(df.columns.tolist())

# COMMAND ----------

# Rename columns to clean, consistent names
df = df.rename(columns={
    'ID': 'id',
    'LIMIT_BAL': 'credit_limit',
    'SEX': 'sex',
    'EDUCATION': 'education',
    'MARRIAGE': 'marriage',
    'AGE': 'age',
    'PAY_0': 'pay_status_1',
    'PAY_2': 'pay_status_2',
    'PAY_3': 'pay_status_3',
    'PAY_4': 'pay_status_4',
    'PAY_5': 'pay_status_5',
    'PAY_6': 'pay_status_6',
    'BILL_AMT1': 'bill_amt_1',
    'BILL_AMT2': 'bill_amt_2',
    'BILL_AMT3': 'bill_amt_3',
    'BILL_AMT4': 'bill_amt_4',
    'BILL_AMT5': 'bill_amt_5',
    'BILL_AMT6': 'bill_amt_6',
    'PAY_AMT1': 'pay_amt_1',
    'PAY_AMT2': 'pay_amt_2',
    'PAY_AMT3': 'pay_amt_3',
    'PAY_AMT4': 'pay_amt_4',
    'PAY_AMT5': 'pay_amt_5',
    'PAY_AMT6': 'pay_amt_6',
    'default.payment.next.month': 'default_payment'
})

print("Cleaned columns:")
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
# MAGIC - Downloaded 30,000 credit card records
# MAGIC - Cleaned and standardized column names
# MAGIC - Saved as Bronze Delta table
# MAGIC - Default rate: ~22%
# MAGIC
# MAGIC **Next:** Run `01_silver_transformation`
