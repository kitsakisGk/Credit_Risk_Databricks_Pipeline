# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer - Data Cleaning & Transformation
# MAGIC
# MAGIC The Silver layer contains cleaned, validated, and conformed data.
# MAGIC This is where we apply business rules and ensure data quality.
# MAGIC
# MAGIC ## Key Concepts:
# MAGIC - **Data Quality Checks**: Validate data meets expectations
# MAGIC - **Type Casting**: Convert strings to proper types
# MAGIC - **Deduplication**: Remove duplicate records
# MAGIC - **Standardization**: Apply business rules and lookups
# MAGIC - **SCD Type 1/2**: Handle slowly changing dimensions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %run ./00_setup_environment

# COMMAND ----------

from pyspark.sql.functions import (
    col, when, lit, current_timestamp, trim, upper, lower,
    regexp_replace, to_date, datediff, current_date,
    count, sum as spark_sum, avg, min as spark_min, max as spark_max,
    row_number, dense_rank, monotonically_increasing_id
)
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, DoubleType, StringType
from delta.tables import DeltaTable

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Bronze Data

# COMMAND ----------

# Read from Bronze layer
bronze_df = spark.table(f"{DATABASE_NAME}.bronze_credit_applications")

print(f"Bronze records: {bronze_df.count()}")
display(bronze_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Assessment
# MAGIC
# MAGIC Before transforming, let's understand the data quality issues.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check for Nulls and Missing Values

# COMMAND ----------

from pyspark.sql.functions import sum as spark_sum, when, col

# Calculate null counts for each column
null_counts = bronze_df.select([
    spark_sum(when(col(c).isNull() | (col(c) == ""), 1).otherwise(0)).alias(c)
    for c in COLUMN_NAMES
])

print("=== Null/Empty Value Counts ===")
display(null_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check for Duplicates

# COMMAND ----------

total_count = bronze_df.count()
distinct_count = bronze_df.select("_row_hash").distinct().count()
duplicate_count = total_count - distinct_count

print(f"""
=== Duplicate Analysis ===
Total Records:    {total_count}
Distinct Records: {distinct_count}
Duplicates:       {duplicate_count}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Transformations

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Type Casting & Basic Cleaning

# COMMAND ----------

# Cast columns to appropriate types
typed_df = (
    bronze_df
    # Numeric columns
    .withColumn("duration_months", col("duration_months").cast(IntegerType()))
    .withColumn("credit_amount", col("credit_amount").cast(DoubleType()))
    .withColumn("installment_rate", col("installment_rate").cast(IntegerType()))
    .withColumn("residence_duration", col("residence_duration").cast(IntegerType()))
    .withColumn("age", col("age").cast(IntegerType()))
    .withColumn("existing_credits", col("existing_credits").cast(IntegerType()))
    .withColumn("num_dependents", col("num_dependents").cast(IntegerType()))
    # Target variable: 1 = Good, 2 = Bad -> convert to 0/1 (0 = Good, 1 = Bad/Default)
    .withColumn("credit_risk",
                when(col("credit_risk") == "1", 0)  # Good credit
                .when(col("credit_risk") == "2", 1)  # Bad credit
                .otherwise(None).cast(IntegerType()))
)

print("✓ Type casting complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Apply Business Rules & Decode Categorical Values
# MAGIC
# MAGIC The German Credit dataset uses codes (A11, A12, etc.). Let's decode them to meaningful values.

# COMMAND ----------

# Decode checking account status
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

    # Purpose of loan
    .withColumn("purpose_decoded",
        when(col("purpose") == "A40", "car (new)")
        .when(col("purpose") == "A41", "car (used)")
        .when(col("purpose") == "A42", "furniture/equipment")
        .when(col("purpose") == "A43", "radio/television")
        .when(col("purpose") == "A44", "domestic appliances")
        .when(col("purpose") == "A45", "repairs")
        .when(col("purpose") == "A46", "education")
        .when(col("purpose") == "A47", "vacation")
        .when(col("purpose") == "A48", "retraining")
        .when(col("purpose") == "A49", "business")
        .when(col("purpose") == "A410", "others")
        .otherwise("unknown"))

    # Savings status
    .withColumn("savings_status_decoded",
        when(col("savings_status") == "A61", "< 100 DM")
        .when(col("savings_status") == "A62", "100-500 DM")
        .when(col("savings_status") == "A63", "500-1000 DM")
        .when(col("savings_status") == "A64", ">= 1000 DM")
        .when(col("savings_status") == "A65", "unknown/none")
        .otherwise("unknown"))

    # Employment duration
    .withColumn("employment_duration_decoded",
        when(col("employment_duration") == "A71", "unemployed")
        .when(col("employment_duration") == "A72", "< 1 year")
        .when(col("employment_duration") == "A73", "1-4 years")
        .when(col("employment_duration") == "A74", "4-7 years")
        .when(col("employment_duration") == "A75", ">= 7 years")
        .otherwise("unknown"))

    # Personal status (contains gender info)
    .withColumn("gender",
        when(col("personal_status").isin(["A91", "A93", "A94"]), "male")
        .when(col("personal_status") == "A92", "female")
        .otherwise("unknown"))

    .withColumn("marital_status",
        when(col("personal_status") == "A91", "divorced/separated")
        .when(col("personal_status") == "A92", "divorced/separated/married")
        .when(col("personal_status") == "A93", "single")
        .when(col("personal_status") == "A94", "married/widowed")
        .otherwise("unknown"))

    # Housing
    .withColumn("housing_decoded",
        when(col("housing") == "A151", "rent")
        .when(col("housing") == "A152", "own")
        .when(col("housing") == "A153", "for free")
        .otherwise("unknown"))

    # Job type
    .withColumn("job_decoded",
        when(col("job") == "A171", "unemployed/unskilled (non-resident)")
        .when(col("job") == "A172", "unskilled (resident)")
        .when(col("job") == "A173", "skilled employee")
        .when(col("job") == "A174", "management/self-employed")
        .otherwise("unknown"))

    # Foreign worker
    .withColumn("is_foreign_worker",
        when(col("foreign_worker") == "A201", True)
        .when(col("foreign_worker") == "A202", False)
        .otherwise(None))

    # Has telephone
    .withColumn("has_telephone",
        when(col("own_telephone") == "A192", True)
        .when(col("own_telephone") == "A191", False)
        .otherwise(None))
)

print("✓ Business rules and decoding applied")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Data Quality Checks (Expectations)
# MAGIC
# MAGIC Define and apply data quality rules. Records failing critical checks go to quarantine.

# COMMAND ----------

# Define data quality expectations
quality_checks = (
    decoded_df
    # Check 1: Age must be valid (18-100)
    .withColumn("_dq_valid_age",
        (col("age") >= 18) & (col("age") <= 100))

    # Check 2: Credit amount must be positive
    .withColumn("_dq_valid_amount",
        col("credit_amount") > 0)

    # Check 3: Duration must be positive
    .withColumn("_dq_valid_duration",
        col("duration_months") > 0)

    # Check 4: Target variable must exist
    .withColumn("_dq_valid_target",
        col("credit_risk").isNotNull())

    # Check 5: Required fields present
    .withColumn("_dq_required_fields",
        col("checking_status").isNotNull() &
        col("credit_history").isNotNull() &
        col("purpose").isNotNull())
)

# Calculate overall quality flag
quality_df = (
    quality_checks
    .withColumn("_dq_passed",
        col("_dq_valid_age") &
        col("_dq_valid_amount") &
        col("_dq_valid_duration") &
        col("_dq_valid_target") &
        col("_dq_required_fields"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Quality Report

# COMMAND ----------

# Generate quality report
quality_report = quality_df.agg(
    count("*").alias("total_records"),
    spark_sum(when(col("_dq_valid_age"), 1).otherwise(0)).alias("valid_age"),
    spark_sum(when(col("_dq_valid_amount"), 1).otherwise(0)).alias("valid_amount"),
    spark_sum(when(col("_dq_valid_duration"), 1).otherwise(0)).alias("valid_duration"),
    spark_sum(when(col("_dq_valid_target"), 1).otherwise(0)).alias("valid_target"),
    spark_sum(when(col("_dq_required_fields"), 1).otherwise(0)).alias("valid_required"),
    spark_sum(when(col("_dq_passed"), 1).otherwise(0)).alias("passed_all_checks")
).collect()[0]

print(f"""
╔══════════════════════════════════════════════════════════════╗
║                  DATA QUALITY REPORT                         ║
╠══════════════════════════════════════════════════════════════╣
║ Total Records:        {quality_report.total_records:>10}                         ║
║ ─────────────────────────────────────────────────────────────║
║ Valid Age:            {quality_report.valid_age:>10} ({100*quality_report.valid_age/quality_report.total_records:.1f}%)                     ║
║ Valid Amount:         {quality_report.valid_amount:>10} ({100*quality_report.valid_amount/quality_report.total_records:.1f}%)                     ║
║ Valid Duration:       {quality_report.valid_duration:>10} ({100*quality_report.valid_duration/quality_report.total_records:.1f}%)                     ║
║ Valid Target:         {quality_report.valid_target:>10} ({100*quality_report.valid_target/quality_report.total_records:.1f}%)                     ║
║ Valid Required:       {quality_report.valid_required:>10} ({100*quality_report.valid_required/quality_report.total_records:.1f}%)                     ║
║ ─────────────────────────────────────────────────────────────║
║ PASSED ALL CHECKS:    {quality_report.passed_all_checks:>10} ({100*quality_report.passed_all_checks/quality_report.total_records:.1f}%)                     ║
╚══════════════════════════════════════════════════════════════╝
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Deduplication

# COMMAND ----------

# Deduplicate keeping the latest record based on ingestion timestamp
window_spec = Window.partitionBy("_row_hash").orderBy(col("_ingestion_timestamp").desc())

deduped_df = (
    quality_df
    .withColumn("_row_num", row_number().over(window_spec))
    .filter(col("_row_num") == 1)
    .drop("_row_num")
)

print(f"Records after deduplication: {deduped_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5: Split Good/Quarantine Records

# COMMAND ----------

# Good records -> Silver table
silver_df = (
    deduped_df
    .filter(col("_dq_passed") == True)
    .withColumn("_silver_timestamp", current_timestamp())
    .withColumn("_record_id", monotonically_increasing_id())
)

# Bad records -> Quarantine table
quarantine_df = (
    deduped_df
    .filter(col("_dq_passed") == False)
    .withColumn("_quarantine_timestamp", current_timestamp())
    .withColumn("_quarantine_reason",
        when(~col("_dq_valid_age"), "invalid_age")
        .when(~col("_dq_valid_amount"), "invalid_amount")
        .when(~col("_dq_valid_duration"), "invalid_duration")
        .when(~col("_dq_valid_target"), "missing_target")
        .when(~col("_dq_required_fields"), "missing_required_fields")
        .otherwise("multiple_failures"))
)

print(f"Silver records: {silver_df.count()}")
print(f"Quarantine records: {quarantine_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Silver Layer

# COMMAND ----------

# Select final columns for Silver table
silver_columns = [
    # Identifiers
    "_record_id", "_row_hash",

    # Original numeric features
    "duration_months", "credit_amount", "installment_rate",
    "residence_duration", "age", "existing_credits", "num_dependents",

    # Original categorical codes (for ML)
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment_duration", "personal_status", "other_parties",
    "property_magnitude", "other_payment_plans", "housing", "job",

    # Decoded categorical values (for analysis)
    "checking_status_decoded", "credit_history_decoded", "purpose_decoded",
    "savings_status_decoded", "employment_duration_decoded",
    "gender", "marital_status", "housing_decoded", "job_decoded",
    "is_foreign_worker", "has_telephone",

    # Target
    "credit_risk",

    # Metadata
    "_ingestion_timestamp", "_silver_timestamp", "_source_file"
]

final_silver_df = silver_df.select(silver_columns)

# Write Silver table
(
    final_silver_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.autoOptimize.optimizeWrite", "true")
    .save(f"{SILVER_PATH}/credit_applications")
)

# Register table
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {DATABASE_NAME}.silver_credit_applications
    USING DELTA
    LOCATION '{SILVER_PATH}/credit_applications'
""")

print(f"✓ Silver table created at {SILVER_PATH}/credit_applications")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Quarantine Table

# COMMAND ----------

# Write quarantine records for later review
(
    quarantine_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(f"{SILVER_PATH}/quarantine_credit_applications")
)

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {DATABASE_NAME}.quarantine_credit_applications
    USING DELTA
    LOCATION '{SILVER_PATH}/quarantine_credit_applications'
""")

print(f"✓ Quarantine table created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming Silver Transformation
# MAGIC
# MAGIC Apply the same transformations to streaming data.

# COMMAND ----------

def transform_to_silver(batch_df, batch_id):
    """
    Micro-batch transformation function for streaming.
    Applies all Silver transformations and writes to Delta.
    """
    from pyspark.sql.functions import lit

    # Apply same transformations as batch
    transformed = (
        batch_df
        # Type casting
        .withColumn("duration_months", col("duration_months").cast(IntegerType()))
        .withColumn("credit_amount", col("credit_amount").cast(DoubleType()))
        .withColumn("age", col("age").cast(IntegerType()))
        .withColumn("credit_risk",
                    when(col("credit_risk") == "1", 0)
                    .when(col("credit_risk") == "2", 1)
                    .otherwise(None).cast(IntegerType()))
        # Add metadata
        .withColumn("_silver_timestamp", current_timestamp())
        .withColumn("_batch_id", lit(batch_id))
    )

    # Write to Silver streaming table
    (
        transformed
        .write
        .format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(f"{DATABASE_NAME}.silver_credit_applications_streaming")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Silver Layer

# COMMAND ----------

# Check Silver table
silver_table = spark.table(f"{DATABASE_NAME}.silver_credit_applications")

print("=== Silver Table Schema ===")
silver_table.printSchema()

print(f"\n=== Record Counts ===")
print(f"Total Silver records: {silver_table.count()}")
print(f"Positive cases (default): {silver_table.filter(col('credit_risk') == 1).count()}")
print(f"Negative cases (good): {silver_table.filter(col('credit_risk') == 0).count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Statistics

# COMMAND ----------

display(
    silver_table.describe()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook we demonstrated:
# MAGIC
# MAGIC | Transformation | Purpose |
# MAGIC |---------------|---------|
# MAGIC | Type Casting | Convert strings to proper numeric/boolean types |
# MAGIC | Business Rules | Decode categorical codes to meaningful values |
# MAGIC | Data Quality | Validate records against defined expectations |
# MAGIC | Deduplication | Remove duplicate records keeping latest |
# MAGIC | Quarantine | Isolate bad records for review |
# MAGIC
# MAGIC **Next:** Run `03_gold_aggregation.py` to create ML-ready features.
