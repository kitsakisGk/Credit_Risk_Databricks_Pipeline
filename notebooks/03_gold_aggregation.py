# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer - Feature Engineering & Aggregations
# MAGIC
# MAGIC The Gold layer contains business-level aggregates and ML-ready features.
# MAGIC This is the consumption layer for analytics and machine learning.
# MAGIC
# MAGIC ## Key Concepts:
# MAGIC - **Feature Engineering**: Create predictive features for ML
# MAGIC - **Business Aggregations**: Calculate KPIs and metrics
# MAGIC - **Data Marts**: Purpose-built tables for specific use cases
# MAGIC - **Feature Store patterns**: Reusable feature pipelines

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %run ./00_setup_environment

# COMMAND ----------

from pyspark.sql.functions import (
    col, when, lit, current_timestamp, count, sum as spark_sum,
    avg, min as spark_min, max as spark_max, stddev,
    percentile_approx, log, exp, sqrt, pow as spark_pow,
    concat, concat_ws, array, create_map, explode,
    dense_rank, percent_rank, ntile
)
from pyspark.sql.window import Window
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler,
    StandardScaler, Bucketizer
)
from pyspark.ml import Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Silver Data

# COMMAND ----------

silver_df = spark.table(f"{DATABASE_NAME}.silver_credit_applications")
print(f"Silver records: {silver_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering
# MAGIC
# MAGIC Create features that will be predictive for credit risk.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical Features

# COMMAND ----------

# Create derived numerical features
features_df = (
    silver_df
    # Risk indicators
    .withColumn("credit_per_month",
        col("credit_amount") / col("duration_months"))

    .withColumn("credit_income_proxy",
        col("credit_amount") / (col("installment_rate") + 1))

    .withColumn("age_credit_ratio",
        col("age") / (col("credit_amount") / 1000))

    # Age-based features
    .withColumn("age_group",
        when(col("age") < 25, "young")
        .when(col("age") < 35, "young_adult")
        .when(col("age") < 50, "middle_aged")
        .when(col("age") < 65, "senior")
        .otherwise("elderly"))

    .withColumn("is_young_borrower",
        when(col("age") < 30, 1).otherwise(0))

    .withColumn("is_senior_borrower",
        when(col("age") >= 55, 1).otherwise(0))

    # Credit amount features
    .withColumn("log_credit_amount",
        log(col("credit_amount") + 1))

    .withColumn("credit_amount_bucket",
        when(col("credit_amount") < 1000, "very_low")
        .when(col("credit_amount") < 2500, "low")
        .when(col("credit_amount") < 5000, "medium")
        .when(col("credit_amount") < 10000, "high")
        .otherwise("very_high"))

    # Duration features
    .withColumn("is_short_term",
        when(col("duration_months") <= 12, 1).otherwise(0))

    .withColumn("is_long_term",
        when(col("duration_months") > 36, 1).otherwise(0))

    # Stability indicators
    .withColumn("stability_score",
        (col("residence_duration") + col("existing_credits")) / 2)

    .withColumn("dependency_ratio",
        col("num_dependents") / (col("existing_credits") + 1))
)

print("✓ Numerical features created")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Risk Score Features

# COMMAND ----------

# Create composite risk indicators
risk_features_df = (
    features_df
    # Checking account risk (important predictor)
    .withColumn("checking_risk_score",
        when(col("checking_status") == "A14", 0)  # no account = higher risk
        .when(col("checking_status") == "A11", 3)  # negative balance
        .when(col("checking_status") == "A12", 2)  # low balance
        .when(col("checking_status") == "A13", 1)  # good balance
        .otherwise(2))

    # Savings risk score
    .withColumn("savings_risk_score",
        when(col("savings_status") == "A65", 4)  # unknown/none
        .when(col("savings_status") == "A61", 3)  # < 100
        .when(col("savings_status") == "A62", 2)  # 100-500
        .when(col("savings_status") == "A63", 1)  # 500-1000
        .when(col("savings_status") == "A64", 0)  # >= 1000
        .otherwise(2))

    # Employment risk score
    .withColumn("employment_risk_score",
        when(col("employment_duration") == "A71", 4)  # unemployed
        .when(col("employment_duration") == "A72", 3)  # < 1 year
        .when(col("employment_duration") == "A73", 2)  # 1-4 years
        .when(col("employment_duration") == "A74", 1)  # 4-7 years
        .when(col("employment_duration") == "A75", 0)  # >= 7 years
        .otherwise(2))

    # Credit history risk score
    .withColumn("history_risk_score",
        when(col("credit_history") == "A34", 4)  # critical
        .when(col("credit_history") == "A33", 3)  # delay
        .when(col("credit_history") == "A32", 1)  # existing paid
        .when(col("credit_history") == "A31", 0)  # all paid this bank
        .when(col("credit_history") == "A30", 0)  # no credits
        .otherwise(2))

    # Combined risk score
    .withColumn("combined_risk_score",
        col("checking_risk_score") +
        col("savings_risk_score") +
        col("employment_risk_score") +
        col("history_risk_score"))
)

print("✓ Risk score features created")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical Encoding Preparation

# COMMAND ----------

# Identify categorical columns to encode
categorical_cols = [
    "checking_status", "credit_history", "purpose",
    "savings_status", "employment_duration", "personal_status",
    "other_parties", "property_magnitude", "other_payment_plans",
    "housing", "job"
]

# Add encoded versions using labels (for tree-based models)
encoded_df = risk_features_df

for cat_col in categorical_cols:
    encoded_df = encoded_df.withColumn(
        f"{cat_col}_encoded",
        dense_rank().over(Window.orderBy(col(cat_col))) - 1
    )

print(f"✓ Created label encodings for {len(categorical_cols)} categorical columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interaction Features

# COMMAND ----------

# Create interaction features (important for linear models)
interaction_df = (
    encoded_df
    # Age * Credit interactions
    .withColumn("age_x_credit",
        col("age") * col("log_credit_amount"))

    # Risk score interactions
    .withColumn("checking_x_savings",
        col("checking_risk_score") * col("savings_risk_score"))

    .withColumn("employment_x_history",
        col("employment_risk_score") * col("history_risk_score"))

    # Financial stress indicator
    .withColumn("financial_stress_indicator",
        col("credit_per_month") * col("combined_risk_score"))
)

print("✓ Interaction features created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create ML-Ready Feature Table

# COMMAND ----------

# Select final features for ML
ml_feature_cols = [
    # ID and target
    "_record_id",
    "credit_risk",

    # Original numerical features
    "duration_months", "credit_amount", "installment_rate",
    "residence_duration", "age", "existing_credits", "num_dependents",

    # Engineered numerical features
    "credit_per_month", "credit_income_proxy", "age_credit_ratio",
    "log_credit_amount", "stability_score", "dependency_ratio",

    # Binary features
    "is_young_borrower", "is_senior_borrower",
    "is_short_term", "is_long_term",
    "is_foreign_worker", "has_telephone",

    # Risk scores
    "checking_risk_score", "savings_risk_score",
    "employment_risk_score", "history_risk_score",
    "combined_risk_score",

    # Interaction features
    "age_x_credit", "checking_x_savings",
    "employment_x_history", "financial_stress_indicator",

    # Encoded categorical features
    "checking_status_encoded", "credit_history_encoded", "purpose_encoded",
    "savings_status_encoded", "employment_duration_encoded",
    "personal_status_encoded", "other_parties_encoded",
    "property_magnitude_encoded", "other_payment_plans_encoded",
    "housing_encoded", "job_encoded",

    # Categorical columns (for analysis)
    "age_group", "credit_amount_bucket",

    # Metadata
    "_silver_timestamp"
]

# Create ML features table
ml_features_df = (
    interaction_df
    .select(ml_feature_cols)
    .withColumn("_gold_timestamp", current_timestamp())
)

print(f"ML Features table: {ml_features_df.count()} rows, {len(ml_features_df.columns)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Gold Tables

# COMMAND ----------

# MAGIC %md
# MAGIC ### ML Features Table

# COMMAND ----------

# Write ML features table
(
    ml_features_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.autoOptimize.optimizeWrite", "true")
    .save(f"{GOLD_PATH}/ml_features")
)

# Register table
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {DATABASE_NAME}.gold_ml_features
    USING DELTA
    LOCATION '{GOLD_PATH}/ml_features'
""")

# Optimize with Z-ordering on commonly filtered columns
spark.sql(f"""
    OPTIMIZE {DATABASE_NAME}.gold_ml_features
    ZORDER BY (credit_risk, age_group)
""")

print(f"✓ ML Features table created at {GOLD_PATH}/ml_features")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Business Aggregations Table

# COMMAND ----------

# Create business-level aggregations
business_agg_df = (
    interaction_df
    .groupBy("age_group", "credit_amount_bucket", "purpose_decoded")
    .agg(
        count("*").alias("application_count"),
        spark_sum("credit_amount").alias("total_credit_amount"),
        avg("credit_amount").alias("avg_credit_amount"),
        avg("duration_months").alias("avg_duration_months"),
        avg("age").alias("avg_age"),
        avg("combined_risk_score").alias("avg_risk_score"),
        spark_sum(when(col("credit_risk") == 1, 1).otherwise(0)).alias("default_count"),
        (spark_sum(when(col("credit_risk") == 1, 1).otherwise(0)) / count("*") * 100)
            .alias("default_rate_pct")
    )
    .withColumn("_aggregation_timestamp", current_timestamp())
)

# Write business aggregations
(
    business_agg_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(f"{GOLD_PATH}/business_aggregations")
)

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {DATABASE_NAME}.gold_business_aggregations
    USING DELTA
    LOCATION '{GOLD_PATH}/business_aggregations'
""")

print(f"✓ Business aggregations table created")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Risk Segment Summary

# COMMAND ----------

# Create risk segment summary for dashboards
risk_segments_df = (
    interaction_df
    .withColumn("risk_segment",
        when(col("combined_risk_score") <= 4, "Low Risk")
        .when(col("combined_risk_score") <= 8, "Medium Risk")
        .when(col("combined_risk_score") <= 12, "High Risk")
        .otherwise("Very High Risk"))
    .groupBy("risk_segment")
    .agg(
        count("*").alias("customer_count"),
        spark_sum("credit_amount").alias("total_exposure"),
        avg("credit_amount").alias("avg_loan_size"),
        avg("age").alias("avg_age"),
        (spark_sum(when(col("credit_risk") == 1, 1).otherwise(0)) / count("*") * 100)
            .alias("actual_default_rate_pct"),
        percentile_approx("credit_amount", 0.5).alias("median_credit_amount")
    )
    .orderBy("risk_segment")
)

display(risk_segments_df)

# Write risk segments
(
    risk_segments_df
    .write
    .format("delta")
    .mode("overwrite")
    .save(f"{GOLD_PATH}/risk_segments")
)

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {DATABASE_NAME}.gold_risk_segments
    USING DELTA
    LOCATION '{GOLD_PATH}/risk_segments'
""")

print("✓ Risk segments table created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Statistics & Profiling

# COMMAND ----------

# Generate feature statistics for documentation
numeric_features = [
    "duration_months", "credit_amount", "age",
    "credit_per_month", "combined_risk_score", "log_credit_amount"
]

feature_stats = ml_features_df.select(numeric_features).describe()
display(feature_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Correlation Analysis

# COMMAND ----------

# Check correlations with target
from pyspark.sql.functions import corr

correlation_cols = [
    "duration_months", "credit_amount", "age", "installment_rate",
    "checking_risk_score", "savings_risk_score", "combined_risk_score",
    "credit_per_month", "is_young_borrower"
]

print("=== Feature Correlations with Credit Risk (Default) ===")
for col_name in correlation_cols:
    correlation = ml_features_df.stat.corr(col_name, "credit_risk")
    print(f"{col_name:30} : {correlation:+.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Gold Layer

# COMMAND ----------

# Show all Gold tables
print("=== Gold Layer Tables ===")
gold_tables = spark.sql(f"SHOW TABLES IN {DATABASE_NAME} LIKE 'gold_*'").collect()
for table in gold_tables:
    count = spark.table(f"{DATABASE_NAME}.{table.tableName}").count()
    print(f"{table.tableName}: {count} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook we created:
# MAGIC
# MAGIC | Table | Purpose | Key Features |
# MAGIC |-------|---------|--------------|
# MAGIC | gold_ml_features | ML Training | 40+ engineered features |
# MAGIC | gold_business_aggregations | Reporting | KPIs by segment |
# MAGIC | gold_risk_segments | Dashboards | Risk distribution summary |
# MAGIC
# MAGIC **Feature Engineering Techniques Used:**
# MAGIC - Ratio features (credit_per_month, age_credit_ratio)
# MAGIC - Log transformations
# MAGIC - Risk scoring
# MAGIC - Label encoding
# MAGIC - Interaction features
# MAGIC - Binning/Bucketing
# MAGIC
# MAGIC **Next:** Run `04_ml_training.py` to train a credit risk model.
