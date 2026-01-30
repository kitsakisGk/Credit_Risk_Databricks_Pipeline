# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer - Data Cleaning & Transformation
# MAGIC
# MAGIC Clean, validate, and transform raw data into analysis-ready format.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# Get schema name
username = spark.sql("SELECT current_user()").first()[0]
username_prefix = username.split("@")[0].replace(".", "_").replace("-", "_")
SCHEMA_NAME = f"{username_prefix}_credit_risk"

spark.sql(f"USE {SCHEMA_NAME}")
print(f"Using schema: {SCHEMA_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Bronze Data

# COMMAND ----------

bronze_df = spark.table("bronze_credit_applications")
print(f"Bronze records: {bronze_df.count()}")
bronze_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Assessment

# COMMAND ----------

from pyspark.sql.functions import col, when, sum as spark_sum, count

# Check for nulls
null_counts = bronze_df.select([
    spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in bronze_df.columns if not c.startswith("_")
])

print("=== Null Counts per Column ===")
null_counts.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Type Casting

# COMMAND ----------

from pyspark.sql.functions import col, when, current_timestamp
from pyspark.sql.types import IntegerType, DoubleType

# Cast to proper types
typed_df = (
    bronze_df
    .withColumn("duration_months", col("duration_months").cast(IntegerType()))
    .withColumn("credit_amount", col("credit_amount").cast(DoubleType()))
    .withColumn("installment_rate", col("installment_rate").cast(IntegerType()))
    .withColumn("residence_duration", col("residence_duration").cast(IntegerType()))
    .withColumn("age", col("age").cast(IntegerType()))
    .withColumn("existing_credits", col("existing_credits").cast(IntegerType()))
    .withColumn("num_dependents", col("num_dependents").cast(IntegerType()))
    # Convert target: 1 = Good (0), 2 = Bad (1)
    .withColumn("credit_risk",
                when(col("credit_risk") == 1, 0)
                .when(col("credit_risk") == 2, 1)
                .otherwise(None).cast(IntegerType()))
)

print("✓ Type casting complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decode Categorical Values

# COMMAND ----------

# Decode codes to meaningful labels
decoded_df = (
    typed_df
    # Checking account status
    .withColumn("checking_status_decoded",
        when(col("checking_status") == "A11", "< 0 DM")
        .when(col("checking_status") == "A12", "0-200 DM")
        .when(col("checking_status") == "A13", ">= 200 DM")
        .when(col("checking_status") == "A14", "no checking account")
        .otherwise("unknown"))

    # Credit history
    .withColumn("credit_history_decoded",
        when(col("credit_history") == "A30", "no credits/all paid")
        .when(col("credit_history") == "A31", "all credits paid at this bank")
        .when(col("credit_history") == "A32", "existing credits paid till now")
        .when(col("credit_history") == "A33", "delay in past payments")
        .when(col("credit_history") == "A34", "critical account")
        .otherwise("unknown"))

    # Purpose
    .withColumn("purpose_decoded",
        when(col("purpose") == "A40", "car (new)")
        .when(col("purpose") == "A41", "car (used)")
        .when(col("purpose") == "A42", "furniture/equipment")
        .when(col("purpose") == "A43", "radio/television")
        .when(col("purpose") == "A44", "domestic appliances")
        .when(col("purpose") == "A45", "repairs")
        .when(col("purpose") == "A46", "education")
        .when(col("purpose") == "A48", "retraining")
        .when(col("purpose") == "A49", "business")
        .otherwise("other"))

    # Employment duration
    .withColumn("employment_decoded",
        when(col("employment_duration") == "A71", "unemployed")
        .when(col("employment_duration") == "A72", "< 1 year")
        .when(col("employment_duration") == "A73", "1-4 years")
        .when(col("employment_duration") == "A74", "4-7 years")
        .when(col("employment_duration") == "A75", ">= 7 years")
        .otherwise("unknown"))

    # Housing
    .withColumn("housing_decoded",
        when(col("housing") == "A151", "rent")
        .when(col("housing") == "A152", "own")
        .when(col("housing") == "A153", "for free")
        .otherwise("unknown"))

    # Gender
    .withColumn("gender",
        when(col("personal_status").isin(["A91", "A93", "A94"]), "male")
        .when(col("personal_status") == "A92", "female")
        .otherwise("unknown"))
)

print("✓ Categorical values decoded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Validation

# COMMAND ----------

# Add quality flags
quality_df = (
    decoded_df
    .withColumn("_dq_valid_age", (col("age") >= 18) & (col("age") <= 100))
    .withColumn("_dq_valid_amount", col("credit_amount") > 0)
    .withColumn("_dq_valid_duration", col("duration_months") > 0)
    .withColumn("_dq_valid_target", col("credit_risk").isNotNull())
    .withColumn("_dq_passed",
        col("_dq_valid_age") &
        col("_dq_valid_amount") &
        col("_dq_valid_duration") &
        col("_dq_valid_target"))
)

# Quality report
total = quality_df.count()
passed = quality_df.filter(col("_dq_passed")).count()
print(f"""
╔══════════════════════════════════════╗
║       DATA QUALITY REPORT            ║
╠══════════════════════════════════════╣
║ Total Records:    {total:>6}              ║
║ Passed Checks:    {passed:>6}              ║
║ Failed Checks:    {total-passed:>6}              ║
║ Pass Rate:        {100*passed/total:>5.1f}%             ║
╚══════════════════════════════════════╝
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Silver Table

# COMMAND ----------

# Keep only good records
silver_df = (
    quality_df
    .filter(col("_dq_passed") == True)
    .withColumn("_silver_timestamp", current_timestamp())
)

# Save as Silver table
silver_df.write.format("delta").mode("overwrite").saveAsTable("silver_credit_applications")

print(f"✓ Silver table created: {silver_df.count()} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Silver Table

# COMMAND ----------

# Show sample
spark.table("silver_credit_applications").select(
    "age", "credit_amount", "duration_months",
    "checking_status_decoded", "purpose_decoded",
    "credit_risk"
).show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Target Distribution

# COMMAND ----------

spark.sql("""
    SELECT
        credit_risk,
        CASE WHEN credit_risk = 0 THEN 'Good Credit' ELSE 'Bad Credit' END as label,
        COUNT(*) as count
    FROM silver_credit_applications
    GROUP BY credit_risk
    ORDER BY credit_risk
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ✅ Type casting applied
# MAGIC ✅ Categorical values decoded
# MAGIC ✅ Data quality validated
# MAGIC ✅ Silver table created
# MAGIC
# MAGIC **Next:** Run `03_gold_aggregation`
