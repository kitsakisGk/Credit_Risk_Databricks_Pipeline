# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer - Data Cleaning & Transformation
# MAGIC
# MAGIC Clean, validate, and transform raw credit card data into analysis-ready format.

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
# MAGIC ## Read Bronze Data

# COMMAND ----------

bronze_df = spark.table("bronze_credit_applications")
print(f"Bronze records: {bronze_df.count():,}")
bronze_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Exploration

# COMMAND ----------

# Check data types
bronze_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decode Categorical Values

# COMMAND ----------

from pyspark.sql.functions import col, when, current_timestamp

# Decode categorical variables to meaningful labels
decoded_df = (
    bronze_df
    # Gender: 1=male, 2=female
    .withColumn("gender_decoded",
        when(col("gender") == 1, "male")
        .when(col("gender") == 2, "female")
        .otherwise("unknown"))

    # Education: 1=graduate school, 2=university, 3=high school, 4=others
    .withColumn("education_decoded",
        when(col("education") == 1, "graduate_school")
        .when(col("education") == 2, "university")
        .when(col("education") == 3, "high_school")
        .when(col("education") == 4, "others")
        .otherwise("unknown"))

    # Marital status: 1=married, 2=single, 3=others
    .withColumn("marital_status_decoded",
        when(col("marital_status") == 1, "married")
        .when(col("marital_status") == 2, "single")
        .when(col("marital_status") == 3, "others")
        .otherwise("unknown"))
)

print("✓ Categorical values decoded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Validation

# COMMAND ----------

from pyspark.sql.functions import sum as spark_sum

# Add quality flags
quality_df = (
    decoded_df
    .withColumn("_dq_valid_age", (col("age") >= 18) & (col("age") <= 100))
    .withColumn("_dq_valid_credit_limit", col("credit_limit") > 0)
    .withColumn("_dq_valid_target", col("default_payment").isin([0, 1]))
    .withColumn("_dq_passed",
        col("_dq_valid_age") &
        col("_dq_valid_credit_limit") &
        col("_dq_valid_target"))
)

# Quality report
total = quality_df.count()
passed = quality_df.filter(col("_dq_passed")).count()
print(f"""
╔══════════════════════════════════════╗
║       DATA QUALITY REPORT            ║
╠══════════════════════════════════════╣
║ Total Records:    {total:>6,}             ║
║ Passed Checks:    {passed:>6,}             ║
║ Failed Checks:    {total-passed:>6,}             ║
║ Pass Rate:        {100*passed/total:>5.1f}%            ║
╚══════════════════════════════════════╝
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Payment Behavior Features

# COMMAND ----------

# Create features for payment behavior
# pay_status: -1=pay duly, 1=payment delay 1 month, 2=delay 2 months, etc.

features_df = (
    quality_df
    .filter(col("_dq_passed") == True)

    # Count months with delayed payment (pay_status > 0)
    .withColumn("months_delayed",
        (when(col("pay_status_sep") > 0, 1).otherwise(0) +
         when(col("pay_status_aug") > 0, 1).otherwise(0) +
         when(col("pay_status_jul") > 0, 1).otherwise(0) +
         when(col("pay_status_jun") > 0, 1).otherwise(0) +
         when(col("pay_status_may") > 0, 1).otherwise(0) +
         when(col("pay_status_apr") > 0, 1).otherwise(0)))

    # Max delay severity (worst payment status)
    .withColumn("max_delay_months",
        greatest(
            col("pay_status_sep"), col("pay_status_aug"), col("pay_status_jul"),
            col("pay_status_jun"), col("pay_status_may"), col("pay_status_apr")
        ))

    # Total bill amount
    .withColumn("total_bill_amt",
        col("bill_amt_sep") + col("bill_amt_aug") + col("bill_amt_jul") +
        col("bill_amt_jun") + col("bill_amt_may") + col("bill_amt_apr"))

    # Total payment amount
    .withColumn("total_pay_amt",
        col("pay_amt_sep") + col("pay_amt_aug") + col("pay_amt_jul") +
        col("pay_amt_jun") + col("pay_amt_may") + col("pay_amt_apr"))

    # Add silver timestamp
    .withColumn("_silver_timestamp", current_timestamp())
)

# Need to import greatest
from pyspark.sql.functions import greatest

# Re-run with greatest imported
features_df = (
    quality_df
    .filter(col("_dq_passed") == True)
    .withColumn("months_delayed",
        (when(col("pay_status_sep") > 0, 1).otherwise(0) +
         when(col("pay_status_aug") > 0, 1).otherwise(0) +
         when(col("pay_status_jul") > 0, 1).otherwise(0) +
         when(col("pay_status_jun") > 0, 1).otherwise(0) +
         when(col("pay_status_may") > 0, 1).otherwise(0) +
         when(col("pay_status_apr") > 0, 1).otherwise(0)))
    .withColumn("max_delay_months",
        greatest(
            col("pay_status_sep"), col("pay_status_aug"), col("pay_status_jul"),
            col("pay_status_jun"), col("pay_status_may"), col("pay_status_apr")
        ))
    .withColumn("total_bill_amt",
        col("bill_amt_sep") + col("bill_amt_aug") + col("bill_amt_jul") +
        col("bill_amt_jun") + col("bill_amt_may") + col("bill_amt_apr"))
    .withColumn("total_pay_amt",
        col("pay_amt_sep") + col("pay_amt_aug") + col("pay_amt_jul") +
        col("pay_amt_jun") + col("pay_amt_may") + col("pay_amt_apr"))
    .withColumn("_silver_timestamp", current_timestamp())
)

print("✓ Payment behavior features created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Silver Table

# COMMAND ----------

# Save as Silver table
features_df.write.format("delta").mode("overwrite").saveAsTable("silver_credit_applications")

print(f"✓ Silver table created: {features_df.count():,} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Silver Table

# COMMAND ----------

# Show sample with new features
spark.table("silver_credit_applications").select(
    "id", "credit_limit", "age", "gender_decoded", "education_decoded",
    "months_delayed", "max_delay_months", "default_payment"
).show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Target Distribution

# COMMAND ----------

spark.sql("""
    SELECT
        default_payment,
        CASE WHEN default_payment = 0 THEN 'No Default' ELSE 'Default' END as label,
        COUNT(*) as count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) as percentage
    FROM silver_credit_applications
    GROUP BY default_payment
    ORDER BY default_payment
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ✅ Categorical values decoded
# MAGIC ✅ Data quality validated
# MAGIC ✅ Payment behavior features created
# MAGIC ✅ Silver table created
# MAGIC
# MAGIC **Next:** Run `03_gold_aggregation`
