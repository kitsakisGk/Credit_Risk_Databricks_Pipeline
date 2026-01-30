# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer - Data Transformation
# MAGIC
# MAGIC Clean and transform Bronze data with business logic using pandas.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import pandas as pd
import numpy as np

SCHEMA_NAME = "kitsakis_credit_risk"
spark.sql(f"USE {SCHEMA_NAME}")
print(f"Using schema: {SCHEMA_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Bronze Data

# COMMAND ----------

# Load from Delta table to pandas
df = spark.table("bronze_credit_applications").toPandas()
print(f"Loaded {len(df)} records from Bronze table")

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Cleaning

# COMMAND ----------

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# COMMAND ----------

# Check for invalid values
print(f"\nRecords with credit_limit <= 0: {(df['credit_limit'] <= 0).sum()}")
print(f"Records with age <= 0: {(df['age'] <= 0).sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decode Categorical Variables

# COMMAND ----------

# Decode gender (1=male, 2=female)
df['gender'] = df['sex'].map({1: 'male', 2: 'female'}).fillna('unknown')

# Decode education (1=graduate school, 2=university, 3=high school, 4+=other)
education_map = {1: 'graduate_school', 2: 'university', 3: 'high_school'}
df['education_level'] = df['education'].map(education_map).fillna('other')

# Decode marital status (1=married, 2=single, 3=other)
marriage_map = {1: 'married', 2: 'single'}
df['marital_status'] = df['marriage'].map(marriage_map).fillna('other')

print("Categorical variables decoded:")
print(f"  Gender: {df['gender'].value_counts().to_dict()}")
print(f"  Education: {df['education_level'].value_counts().to_dict()}")
print(f"  Marital Status: {df['marital_status'].value_counts().to_dict()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Payment Behavior Features

# COMMAND ----------

# Payment status columns
pay_status_cols = ['pay_status_1', 'pay_status_2', 'pay_status_3',
                   'pay_status_4', 'pay_status_5', 'pay_status_6']

# Count months with payment delay (pay_status > 0 means delayed)
df['months_delayed'] = (df[pay_status_cols] > 0).sum(axis=1)

# Maximum delay across all months
df['max_delay_months'] = df[pay_status_cols].max(axis=1)

# Bill amount columns
bill_cols = ['bill_amt_1', 'bill_amt_2', 'bill_amt_3',
             'bill_amt_4', 'bill_amt_5', 'bill_amt_6']
df['total_bill_amt'] = df[bill_cols].sum(axis=1)

# Payment amount columns
pay_cols = ['pay_amt_1', 'pay_amt_2', 'pay_amt_3',
            'pay_amt_4', 'pay_amt_5', 'pay_amt_6']
df['total_pay_amt'] = df[pay_cols].sum(axis=1)

print("Payment behavior features created:")
print(f"  months_delayed range: {df['months_delayed'].min()} - {df['months_delayed'].max()}")
print(f"  max_delay_months range: {df['max_delay_months'].min()} - {df['max_delay_months'].max()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Summary

# COMMAND ----------

print("="*50)
print("DATA QUALITY SUMMARY")
print("="*50)
print(f"Total records: {len(df)}")
print(f"Unique customers: {df['id'].nunique()}")
print(f"\nGender distribution:")
print(df['gender'].value_counts())
print(f"\nEducation distribution:")
print(df['education_level'].value_counts())
print(f"\nDefault rate: {df['default_payment'].mean()*100:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Payment Delay Analysis

# COMMAND ----------

# Default rate by months delayed
delay_analysis = df.groupby('months_delayed').agg({
    'id': 'count',
    'default_payment': 'mean'
}).rename(columns={'id': 'customer_count', 'default_payment': 'default_rate'})
delay_analysis['default_rate'] = (delay_analysis['default_rate'] * 100).round(1)

print("Default Rate by Months Delayed:")
display(delay_analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Silver Table

# COMMAND ----------

# Select columns for Silver table
silver_cols = [
    'id', 'credit_limit', 'age', 'gender', 'education_level', 'marital_status',
    'pay_status_1', 'pay_status_2', 'pay_status_3', 'pay_status_4', 'pay_status_5', 'pay_status_6',
    'bill_amt_1', 'bill_amt_2', 'bill_amt_3', 'bill_amt_4', 'bill_amt_5', 'bill_amt_6',
    'pay_amt_1', 'pay_amt_2', 'pay_amt_3', 'pay_amt_4', 'pay_amt_5', 'pay_amt_6',
    'months_delayed', 'max_delay_months', 'total_bill_amt', 'total_pay_amt',
    'default_payment'
]

silver_df = df[silver_cols].copy()

# Save to Delta
spark.sql("DROP TABLE IF EXISTS silver_credit_applications")
spark_df = spark.createDataFrame(silver_df)
spark_df.write.format("delta").mode("overwrite").saveAsTable("silver_credit_applications")

print(f"Silver table created: {len(silver_df)} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

spark.sql("SELECT COUNT(*) as total FROM silver_credit_applications").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC - Decoded categorical variables (gender, education, marital status)
# MAGIC - Created payment behavior features (months_delayed, total amounts)
# MAGIC - Data quality validated
# MAGIC - Silver table saved
# MAGIC
# MAGIC **Next:** Run `02_gold_features`
