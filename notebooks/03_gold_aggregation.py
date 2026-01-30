# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer - Feature Engineering
# MAGIC
# MAGIC Create ML-ready features from Silver data for credit default prediction.

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
silver_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering

# COMMAND ----------

from pyspark.sql.functions import col, when, log, current_timestamp, greatest, least, abs as spark_abs

# Create features
features_df = (
    silver_df
    # Credit utilization features
    .withColumn("avg_bill_amount",
        (col("bill_amt_1") + col("bill_amt_2") + col("bill_amt_3") +
         col("bill_amt_4") + col("bill_amt_5") + col("bill_amt_6")) / 6)
    .withColumn("avg_payment_amount",
        (col("pay_amt_1") + col("pay_amt_2") + col("pay_amt_3") +
         col("pay_amt_4") + col("pay_amt_5") + col("pay_amt_6")) / 6)
    .withColumn("credit_utilization",
        when(col("credit_limit") > 0, col("bill_amt_1") / col("credit_limit")).otherwise(0))
    .withColumn("log_credit_limit", log(col("credit_limit") + 1))

    # Payment behavior features
    .withColumn("payment_ratio",
        when(col("bill_amt_1") > 0, col("pay_amt_1") / col("bill_amt_1")).otherwise(1))
    .withColumn("pays_full_balance",
        when(col("pay_amt_1") >= col("bill_amt_1"), 1).otherwise(0))

    # Age features
    .withColumn("age_group",
        when(col("age") < 25, "young")
        .when(col("age") < 35, "young_adult")
        .when(col("age") < 50, "middle_aged")
        .when(col("age") < 65, "senior")
        .otherwise("elderly"))
    .withColumn("is_young_borrower", when(col("age") < 30, 1).otherwise(0))

    # Credit limit buckets
    .withColumn("credit_bucket",
        when(col("credit_limit") < 50000, "low")
        .when(col("credit_limit") < 150000, "medium")
        .when(col("credit_limit") < 300000, "high")
        .otherwise("very_high"))
)

print("Basic features created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Risk Score Features

# COMMAND ----------

# Create risk indicators based on payment history
risk_df = (
    features_df
    # Payment delay risk (higher delay status = higher risk)
    .withColumn("delay_risk_1",
        when(col("pay_status_1") <= 0, 0)
        .when(col("pay_status_1") == 1, 1)
        .when(col("pay_status_1") == 2, 2)
        .otherwise(3))
    .withColumn("delay_risk_2",
        when(col("pay_status_2") <= 0, 0)
        .when(col("pay_status_2") == 1, 1)
        .when(col("pay_status_2") == 2, 2)
        .otherwise(3))
    .withColumn("delay_risk_3",
        when(col("pay_status_3") <= 0, 0)
        .when(col("pay_status_3") == 1, 1)
        .when(col("pay_status_3") == 2, 2)
        .otherwise(3))

    # Education risk (higher education typically = lower risk)
    .withColumn("education_risk",
        when(col("education") == "graduate_school", 0)
        .when(col("education") == "university", 1)
        .when(col("education") == "high_school", 2)
        .otherwise(3))

    # Marital status risk factor
    .withColumn("marital_risk",
        when(col("marital_status") == "married", 0)
        .when(col("marital_status") == "single", 1)
        .otherwise(2))

    # Credit utilization risk
    .withColumn("utilization_risk",
        when(col("credit_utilization") < 0.3, 0)
        .when(col("credit_utilization") < 0.5, 1)
        .when(col("credit_utilization") < 0.8, 2)
        .otherwise(3))

    # Combined risk score
    .withColumn("total_risk_score",
        col("delay_risk_1") + col("delay_risk_2") + col("delay_risk_3") +
        col("education_risk") + col("marital_risk") + col("utilization_risk"))
)

print("Risk scores created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Gold Table

# COMMAND ----------

# Add timestamp and save as Gold table
gold_df = risk_df.withColumn("_gold_timestamp", current_timestamp())

gold_df.write.format("delta").mode("overwrite").saveAsTable("gold_credit_features")

print(f"Gold table created: {gold_df.count()} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Summary

# COMMAND ----------

# Show feature statistics
spark.sql("""
    SELECT
        ROUND(AVG(credit_limit), 2) as avg_credit_limit,
        ROUND(AVG(age), 1) as avg_age,
        ROUND(AVG(credit_utilization), 3) as avg_utilization,
        ROUND(AVG(total_risk_score), 2) as avg_risk_score,
        SUM(CASE WHEN default_payment = 1 THEN 1 ELSE 0 END) as defaults,
        COUNT(*) as total,
        ROUND(100.0 * SUM(default_payment) / COUNT(*), 2) as default_rate_pct
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
        ROUND(100.0 * SUM(default_payment) / COUNT(*), 1) as default_rate_pct
    FROM gold_credit_features
    GROUP BY 1
    ORDER BY 1
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Default Rate by Demographics

# COMMAND ----------

# By education
spark.sql("""
    SELECT
        education,
        COUNT(*) as count,
        ROUND(100.0 * SUM(default_payment) / COUNT(*), 1) as default_rate_pct
    FROM gold_credit_features
    GROUP BY education
    ORDER BY default_rate_pct DESC
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC - Feature engineering complete
# MAGIC - Risk scores calculated based on payment history and demographics
# MAGIC - Gold table created with 30,000 records
# MAGIC
# MAGIC **Next:** Run `04_ml_training`
