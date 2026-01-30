# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer - Feature Engineering
# MAGIC
# MAGIC Create ML-ready features from Silver data using pandas.

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
# MAGIC ## Load Silver Data

# COMMAND ----------

df = spark.table("silver_credit_applications").toPandas()
print(f"Loaded {len(df)} records from Silver table")

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering

# COMMAND ----------

# Bill amount columns
bill_cols = ['bill_amt_1', 'bill_amt_2', 'bill_amt_3',
             'bill_amt_4', 'bill_amt_5', 'bill_amt_6']

# Payment amount columns
pay_cols = ['pay_amt_1', 'pay_amt_2', 'pay_amt_3',
            'pay_amt_4', 'pay_amt_5', 'pay_amt_6']

# Average bill and payment amounts
df['avg_bill_amount'] = df[bill_cols].mean(axis=1)
df['avg_payment_amount'] = df[pay_cols].mean(axis=1)

# Credit utilization (most recent bill / credit limit)
df['credit_utilization'] = np.where(
    df['credit_limit'] > 0,
    df['bill_amt_1'] / df['credit_limit'],
    0
)

# Log credit limit (for better distribution)
df['log_credit_limit'] = np.log(df['credit_limit'] + 1)

# Payment ratio (payment / bill for most recent month)
df['payment_ratio'] = np.where(
    df['bill_amt_1'] > 0,
    df['pay_amt_1'] / df['bill_amt_1'],
    1
)

# Pays full balance flag
df['pays_full_balance'] = (df['pay_amt_1'] >= df['bill_amt_1']).astype(int)

print("Credit features created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Age Features

# COMMAND ----------

# Age group
def age_group(age):
    if age < 25:
        return 'young'
    elif age < 35:
        return 'young_adult'
    elif age < 50:
        return 'middle_aged'
    elif age < 65:
        return 'senior'
    else:
        return 'elderly'

df['age_group'] = df['age'].apply(age_group)

# Young borrower flag
df['is_young_borrower'] = (df['age'] < 30).astype(int)

# Credit bucket
def credit_bucket(limit):
    if limit < 50000:
        return 'low'
    elif limit < 150000:
        return 'medium'
    elif limit < 300000:
        return 'high'
    else:
        return 'very_high'

df['credit_bucket'] = df['credit_limit'].apply(credit_bucket)

print("Age and credit features created")
print(f"\nAge group distribution:\n{df['age_group'].value_counts()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Risk Score Features

# COMMAND ----------

# Payment delay risk (0-3 scale based on severity)
def delay_risk(status):
    if status <= 0:
        return 0
    elif status == 1:
        return 1
    elif status == 2:
        return 2
    else:
        return 3

df['delay_risk_1'] = df['pay_status_1'].apply(delay_risk)
df['delay_risk_2'] = df['pay_status_2'].apply(delay_risk)
df['delay_risk_3'] = df['pay_status_3'].apply(delay_risk)

# Education risk (lower education = higher risk)
education_risk_map = {
    'graduate_school': 0,
    'university': 1,
    'high_school': 2,
    'other': 3
}
df['education_risk'] = df['education_level'].map(education_risk_map)

# Marital risk
marital_risk_map = {'married': 0, 'single': 1, 'other': 2}
df['marital_risk'] = df['marital_status'].map(marital_risk_map)

# Utilization risk
def utilization_risk(util):
    if util < 0.3:
        return 0
    elif util < 0.5:
        return 1
    elif util < 0.8:
        return 2
    else:
        return 3

df['utilization_risk'] = df['credit_utilization'].apply(utilization_risk)

# Total risk score
df['total_risk_score'] = (
    df['delay_risk_1'] + df['delay_risk_2'] + df['delay_risk_3'] +
    df['education_risk'] + df['marital_risk'] + df['utilization_risk']
)

print("Risk scores created")
print(f"\nTotal risk score range: {df['total_risk_score'].min()} - {df['total_risk_score'].max()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Summary

# COMMAND ----------

print("="*50)
print("FEATURE SUMMARY")
print("="*50)
print(f"Total records: {len(df)}")
print(f"Total features: {len(df.columns)}")
print(f"\nKey statistics:")
print(f"  Avg credit limit: ${df['credit_limit'].mean():,.0f}")
print(f"  Avg age: {df['age'].mean():.1f}")
print(f"  Avg credit utilization: {df['credit_utilization'].mean():.2%}")
print(f"  Avg risk score: {df['total_risk_score'].mean():.2f}")
print(f"  Default rate: {df['default_payment'].mean():.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Risk Score vs Default Rate

# COMMAND ----------

# Analyze risk score effectiveness
def risk_category(score):
    if score <= 4:
        return '1. Low Risk (0-4)'
    elif score <= 8:
        return '2. Medium Risk (5-8)'
    else:
        return '3. High Risk (9+)'

df['risk_category'] = df['total_risk_score'].apply(risk_category)

risk_analysis = df.groupby('risk_category').agg({
    'id': 'count',
    'default_payment': 'mean'
}).rename(columns={'id': 'customer_count', 'default_payment': 'default_rate'})
risk_analysis['default_rate'] = (risk_analysis['default_rate'] * 100).round(1)

print("Default Rate by Risk Category:")
display(risk_analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Gold Table

# COMMAND ----------

# Select all columns for Gold table (drop temporary columns)
gold_df = df.drop(columns=['risk_category'])

# Save to Delta
spark.sql("DROP TABLE IF EXISTS gold_credit_features")
spark_df = spark.createDataFrame(gold_df)
spark_df.write.format("delta").mode("overwrite").saveAsTable("gold_credit_features")

print(f"Gold table created: {len(gold_df)} records with {len(gold_df.columns)} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

spark.sql("SELECT COUNT(*) as total FROM gold_credit_features").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature List

# COMMAND ----------

print("="*50)
print("FEATURES IN GOLD TABLE")
print("="*50)
for i, col in enumerate(gold_df.columns, 1):
    print(f"{i:2}. {col}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### Features Created:
# MAGIC - **Credit Features:** utilization, log_credit_limit, payment_ratio
# MAGIC - **Age Features:** age_group, is_young_borrower, credit_bucket
# MAGIC - **Risk Scores:** delay_risk, education_risk, marital_risk, total_risk_score
# MAGIC
# MAGIC ### Key Insight:
# MAGIC Higher risk scores correlate strongly with higher default rates!
# MAGIC
# MAGIC **Next:** Run `03_ml_training`
