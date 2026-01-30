# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer - Data Transformation
# MAGIC
# MAGIC Clean and transform Bronze data with business logic.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

SCHEMA_NAME = "kitsakis_credit_risk"

spark.sql(f"USE {SCHEMA_NAME}")
print(f"Using schema: {SCHEMA_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Bronze Data

# COMMAND ----------

bronze_df = spark.table("bronze_credit_applications")
print(f"Bronze records: {bronze_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Transformation

# COMMAND ----------

from pyspark.sql.functions import col, when, current_timestamp, greatest

# Transform data
silver_df = (
    bronze_df
    # Decode gender
    .withColumn("gender",
        when(col("sex") == 1, "male")
        .when(col("sex") == 2, "female")
        .otherwise("unknown"))

    # Decode education
    .withColumn("education_level",
        when(col("education") == 1, "graduate_school")
        .when(col("education") == 2, "university")
        .when(col("education") == 3, "high_school")
        .otherwise("other"))

    # Decode marital status
    .withColumn("marital_status",
        when(col("marriage") == 1, "married")
        .when(col("marriage") == 2, "single")
        .otherwise("other"))

    # Payment behavior features
    .withColumn("months_delayed",
        (when(col("pay_status_1") > 0, 1).otherwise(0) +
         when(col("pay_status_2") > 0, 1).otherwise(0) +
         when(col("pay_status_3") > 0, 1).otherwise(0) +
         when(col("pay_status_4") > 0, 1).otherwise(0) +
         when(col("pay_status_5") > 0, 1).otherwise(0) +
         when(col("pay_status_6") > 0, 1).otherwise(0)))

    .withColumn("max_delay_months",
        greatest(col("pay_status_1"), col("pay_status_2"), col("pay_status_3"),
                 col("pay_status_4"), col("pay_status_5"), col("pay_status_6")))

    # Total amounts
    .withColumn("total_bill_amt",
        col("bill_amt_1") + col("bill_amt_2") + col("bill_amt_3") +
        col("bill_amt_4") + col("bill_amt_5") + col("bill_amt_6"))

    .withColumn("total_pay_amt",
        col("pay_amt_1") + col("pay_amt_2") + col("pay_amt_3") +
        col("pay_amt_4") + col("pay_amt_5") + col("pay_amt_6"))

    # Metadata
    .withColumn("_silver_timestamp", current_timestamp())

    # Drop original encoded columns
    .drop("sex", "education", "marriage", "_ingested_at")
)

print("Transformation complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Check

# COMMAND ----------

# Check for invalid records
invalid_records = silver_df.filter((col("credit_limit") <= 0) | (col("age") <= 0)).count()
print(f"Invalid records (credit_limit <= 0 or age <= 0): {invalid_records}")

# Filter out invalid records
silver_df = silver_df.filter((col("credit_limit") > 0) & (col("age") > 0))
print(f"Valid records: {silver_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Silver Table

# COMMAND ----------

silver_df.write.format("delta").mode("overwrite").saveAsTable("silver_credit_applications")
print(f"Silver table created: {silver_df.count()} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Report

# COMMAND ----------

spark.sql("""
    SELECT
        COUNT(*) as total_records,
        COUNT(DISTINCT id) as unique_customers,
        SUM(CASE WHEN gender = 'male' THEN 1 ELSE 0 END) as male_count,
        SUM(CASE WHEN gender = 'female' THEN 1 ELSE 0 END) as female_count,
        SUM(default_payment) as total_defaults,
        ROUND(100.0 * SUM(default_payment) / COUNT(*), 2) as default_rate_pct
    FROM silver_credit_applications
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Payment Behavior Analysis

# COMMAND ----------

spark.sql("""
    SELECT
        months_delayed,
        COUNT(*) as customer_count,
        ROUND(100.0 * SUM(default_payment) / COUNT(*), 1) as default_rate_pct
    FROM silver_credit_applications
    GROUP BY months_delayed
    ORDER BY months_delayed
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC - Decoded categorical variables (gender, education, marital status)
# MAGIC - Created payment behavior features
# MAGIC - Data quality validated
# MAGIC
# MAGIC **Next:** Run `02_gold_features`
