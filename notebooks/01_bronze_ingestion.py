# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer - Raw Data Ingestion
# MAGIC
# MAGIC The Bronze layer stores raw data exactly as received from source systems.
# MAGIC We demonstrate both **batch** and **streaming** ingestion patterns.
# MAGIC
# MAGIC ## Key Concepts:
# MAGIC - **Append-only**: Raw data is never modified
# MAGIC - **Schema-on-read**: Minimal transformation, preserve original data
# MAGIC - **Metadata enrichment**: Add ingestion timestamp, source info
# MAGIC - **Auto Loader**: Scalable file ingestion with exactly-once semantics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %run ./00_setup_environment

# COMMAND ----------

from pyspark.sql.functions import (
    col, lit, current_timestamp, input_file_name,
    sha2, concat_ws, to_date, when
)
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from delta.tables import DeltaTable

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option 1: Batch Ingestion
# MAGIC
# MAGIC Simple one-time load for initial data or small datasets.

# COMMAND ----------

# Read raw data with proper column names
raw_df = (
    spark.read
    .option("delimiter", " ")
    .option("header", "false")
    .schema(RAW_SCHEMA)
    .csv(f"{LANDING_PATH}/german_credit.data")
)

# Add metadata columns (Bronze layer best practice)
bronze_df = (
    raw_df
    .withColumn("_ingestion_timestamp", current_timestamp())
    .withColumn("_source_file", lit("german_credit.data"))
    .withColumn("_batch_id", lit(1))
    # Create a unique row hash for deduplication
    .withColumn("_row_hash", sha2(concat_ws("||", *[col(c) for c in COLUMN_NAMES]), 256))
)

print(f"Records to ingest: {bronze_df.count()}")
bronze_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write to Delta Lake (Bronze Table)

# COMMAND ----------

# Write as Delta table with optimizations
(
    bronze_df
    .write
    .format("delta")
    .mode("overwrite")  # Use 'append' in production for incremental loads
    .option("overwriteSchema", "true")
    .option("delta.autoOptimize.optimizeWrite", "true")
    .option("delta.autoOptimize.autoCompact", "true")
    .save(f"{BRONZE_PATH}/credit_applications")
)

# Register as table
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {DATABASE_NAME}.bronze_credit_applications
    USING DELTA
    LOCATION '{BRONZE_PATH}/credit_applications'
""")

print(f"✓ Bronze table created at {BRONZE_PATH}/credit_applications")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option 2: Streaming Ingestion with Auto Loader
# MAGIC
# MAGIC Production-grade pattern for continuous data ingestion.
# MAGIC Auto Loader provides:
# MAGIC - **Exactly-once processing** via checkpointing
# MAGIC - **Schema inference and evolution**
# MAGIC - **Scalable file discovery** (no need to list directories)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Simulate Streaming Source
# MAGIC
# MAGIC In production, this would be Kafka, Event Hubs, or files landing from external systems.
# MAGIC Here we'll create a function to simulate new files arriving.

# COMMAND ----------

import json
import random
from datetime import datetime

def generate_streaming_record():
    """Generate a synthetic credit application record."""
    return {
        "checking_status": random.choice(["A11", "A12", "A13", "A14"]),
        "duration_months": str(random.randint(6, 72)),
        "credit_history": random.choice(["A30", "A31", "A32", "A33", "A34"]),
        "purpose": random.choice(["A40", "A41", "A42", "A43", "A44", "A45", "A46"]),
        "credit_amount": str(random.randint(250, 20000)),
        "savings_status": random.choice(["A61", "A62", "A63", "A64", "A65"]),
        "employment_duration": random.choice(["A71", "A72", "A73", "A74", "A75"]),
        "installment_rate": str(random.randint(1, 4)),
        "personal_status": random.choice(["A91", "A92", "A93", "A94"]),
        "other_parties": random.choice(["A101", "A102", "A103"]),
        "residence_duration": str(random.randint(1, 4)),
        "property_magnitude": random.choice(["A121", "A122", "A123", "A124"]),
        "age": str(random.randint(19, 75)),
        "other_payment_plans": random.choice(["A141", "A142", "A143"]),
        "housing": random.choice(["A151", "A152", "A153"]),
        "existing_credits": str(random.randint(1, 4)),
        "job": random.choice(["A171", "A172", "A173", "A174"]),
        "num_dependents": str(random.randint(1, 2)),
        "own_telephone": random.choice(["A191", "A192"]),
        "foreign_worker": random.choice(["A201", "A202"]),
        "credit_risk": str(random.choice([1, 2]))
    }

def write_streaming_batch(num_records=10, batch_num=1):
    """Write a batch of records as JSON to landing zone for streaming."""
    records = [generate_streaming_record() for _ in range(num_records)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"credit_applications_{timestamp}_batch{batch_num}.json"

    # Write to temp file then copy to DBFS
    local_path = f"/tmp/{filename}"
    with open(local_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    dbutils.fs.cp(f"file:{local_path}", f"{LANDING_PATH}/streaming/{filename}")
    print(f"✓ Written {num_records} records to {LANDING_PATH}/streaming/{filename}")

# Create streaming landing directory
dbutils.fs.mkdirs(f"{LANDING_PATH}/streaming")

# Generate some initial streaming data
write_streaming_batch(num_records=50, batch_num=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Auto Loader Streaming Ingestion

# COMMAND ----------

# Schema for JSON streaming data
streaming_schema = StructType([
    StructField(col, StringType(), True) for col in COLUMN_NAMES
])

# Auto Loader configuration
auto_loader_options = {
    "cloudFiles.format": "json",
    "cloudFiles.schemaLocation": f"{CHECKPOINT_PATH}/bronze_schema",
    "cloudFiles.inferColumnTypes": "false",  # Use explicit schema
    "cloudFiles.schemaHints": "duration_months STRING, credit_amount STRING, age STRING"
}

# Read stream with Auto Loader
streaming_df = (
    spark.readStream
    .format("cloudFiles")
    .options(**auto_loader_options)
    .schema(streaming_schema)
    .load(f"{LANDING_PATH}/streaming")
)

# Add metadata columns
streaming_bronze_df = (
    streaming_df
    .withColumn("_ingestion_timestamp", current_timestamp())
    .withColumn("_source_file", input_file_name())
    .withColumn("_batch_id", lit(-1))  # Will be replaced by streaming batch id
    .withColumn("_row_hash", sha2(concat_ws("||", *[col(c) for c in COLUMN_NAMES]), 256))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write Stream to Delta (Bronze Streaming Table)

# COMMAND ----------

# Define streaming write with checkpoint
streaming_query = (
    streaming_bronze_df
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", f"{CHECKPOINT_PATH}/bronze_streaming")
    .option("mergeSchema", "true")
    .trigger(availableNow=True)  # Process all available data then stop
    # Use .trigger(processingTime="10 seconds") for continuous processing
    .table(f"{DATABASE_NAME}.bronze_credit_applications_streaming")
)

# Wait for completion (for availableNow trigger)
streaming_query.awaitTermination()

print("✓ Streaming batch processed successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Bronze Layer

# COMMAND ----------

# Check batch table
print("=== Batch Bronze Table ===")
batch_count = spark.table(f"{DATABASE_NAME}.bronze_credit_applications").count()
print(f"Total records: {batch_count}")

# Check streaming table
print("\n=== Streaming Bronze Table ===")
streaming_count = spark.table(f"{DATABASE_NAME}.bronze_credit_applications_streaming").count()
print(f"Total records: {streaming_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delta Lake Features Demo

# COMMAND ----------

# MAGIC %md
# MAGIC ### Table History (Audit Trail)

# COMMAND ----------

# View table history - every operation is logged
display(
    spark.sql(f"DESCRIBE HISTORY {DATABASE_NAME}.bronze_credit_applications")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Time Travel - Query Previous Versions

# COMMAND ----------

# Query data as of version 0 (initial write)
historical_df = (
    spark.read
    .format("delta")
    .option("versionAsOf", 0)
    .load(f"{BRONZE_PATH}/credit_applications")
)

print(f"Records at version 0: {historical_df.count()}")

# Can also query by timestamp:
# .option("timestampAsOf", "2024-01-01 00:00:00")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Table Details

# COMMAND ----------

# View table properties and statistics
display(
    spark.sql(f"DESCRIBE DETAIL {DATABASE_NAME}.bronze_credit_applications")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook we demonstrated:
# MAGIC
# MAGIC | Pattern | Use Case | Key Feature |
# MAGIC |---------|----------|-------------|
# MAGIC | Batch Ingestion | Initial load, small datasets | Simple, predictable |
# MAGIC | Auto Loader Streaming | Continuous ingestion, large scale | Exactly-once, schema evolution |
# MAGIC | Delta Lake | All Bronze storage | ACID, time travel, audit |
# MAGIC
# MAGIC **Next:** Run `02_silver_transformation.py` to clean and validate this data.
