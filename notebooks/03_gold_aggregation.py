# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer - Feature Engineering
# MAGIC
# MAGIC Create ML-ready features from Silver data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# Schema name
SCHEMA_NAME = "kitsakis_credit_risk"

spark.sql(f"USE {SCHEMA_NAME}")
print(f"Using schema: {SCHEMA_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Silver Data

# COMMAND ----------

silver_df = spark.table("silver_credit_applications")
print(f"Silver records: {silver_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering

# COMMAND ----------

from pyspark.sql.functions import col, when, log, current_timestamp
from pyspark.sql.window import Window

# Create features
features_df = (
    silver_df
    # Payment features
    .withColumn("credit_per_month", col("credit_amount") / col("duration_months"))
    .withColumn("log_credit_amount", log(col("credit_amount") + 1))

    # Age features
    .withColumn("age_group",
        when(col("age") < 25, "young")
        .when(col("age") < 35, "young_adult")
        .when(col("age") < 50, "middle_aged")
        .when(col("age") < 65, "senior")
        .otherwise("elderly"))
    .withColumn("is_young_borrower", when(col("age") < 30, 1).otherwise(0))

    # Credit amount bucket
    .withColumn("credit_bucket",
        when(col("credit_amount") < 2500, "low")
        .when(col("credit_amount") < 5000, "medium")
        .when(col("credit_amount") < 10000, "high")
        .otherwise("very_high"))

    # Duration features
    .withColumn("is_short_term", when(col("duration_months") <= 12, 1).otherwise(0))
    .withColumn("is_long_term", when(col("duration_months") > 36, 1).otherwise(0))
)

print("✓ Basic features created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Risk Score Features

# COMMAND ----------

# Create risk indicators based on domain knowledge
risk_df = (
    features_df
    # Checking account risk
    .withColumn("checking_risk",
        when(col("checking_status") == "A11", 3)  # negative balance
        .when(col("checking_status") == "A12", 2)  # 0-200
        .when(col("checking_status") == "A13", 1)  # 200+
        .when(col("checking_status") == "A14", 0)  # no account
        .otherwise(2))

    # Savings risk
    .withColumn("savings_risk",
        when(col("savings_status") == "A65", 4)  # unknown
        .when(col("savings_status") == "A61", 3)  # < 100
        .when(col("savings_status") == "A62", 2)  # 100-500
        .when(col("savings_status") == "A63", 1)  # 500-1000
        .otherwise(0))

    # Employment risk
    .withColumn("employment_risk",
        when(col("employment_duration") == "A71", 4)  # unemployed
        .when(col("employment_duration") == "A72", 3)  # < 1 year
        .when(col("employment_duration") == "A73", 2)  # 1-4 years
        .when(col("employment_duration") == "A74", 1)  # 4-7 years
        .otherwise(0))

    # Credit history risk
    .withColumn("history_risk",
        when(col("credit_history") == "A34", 4)  # critical
        .when(col("credit_history") == "A33", 3)  # delay
        .when(col("credit_history") == "A32", 1)  # existing paid
        .otherwise(0))

    # Combined risk score
    .withColumn("total_risk_score",
        col("checking_risk") + col("savings_risk") +
        col("employment_risk") + col("history_risk"))
)

print("✓ Risk scores created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Gold Table

# COMMAND ----------

# Select final features
gold_df = (
    risk_df
    .withColumn("_gold_timestamp", current_timestamp())
)

# Save as Gold table
gold_df.write.format("delta").mode("overwrite").saveAsTable("gold_credit_features")

print(f"✓ Gold table created: {gold_df.count()} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Summary

# COMMAND ----------

# Show feature statistics
spark.sql("""
    SELECT
        ROUND(AVG(credit_amount), 2) as avg_credit,
        ROUND(AVG(age), 1) as avg_age,
        ROUND(AVG(duration_months), 1) as avg_duration,
        ROUND(AVG(total_risk_score), 2) as avg_risk_score,
        SUM(CASE WHEN credit_risk = 1 THEN 1 ELSE 0 END) as defaults,
        COUNT(*) as total
    FROM gold_credit_features
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Risk Score vs Default Rate

# COMMAND ----------

spark.sql("""
    SELECT
        CASE
            WHEN total_risk_score <= 4 THEN 'Low Risk'
            WHEN total_risk_score <= 8 THEN 'Medium Risk'
            ELSE 'High Risk'
        END as risk_category,
        COUNT(*) as count,
        ROUND(100.0 * SUM(credit_risk) / COUNT(*), 1) as default_rate_pct
    FROM gold_credit_features
    GROUP BY 1
    ORDER BY 1
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ✅ Feature engineering complete
# MAGIC ✅ Risk scores calculated
# MAGIC ✅ Gold table created
# MAGIC
# MAGIC **Next:** Run `04_ml_training`
