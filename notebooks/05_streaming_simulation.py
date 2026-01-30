# Databricks notebook source
# MAGIC %md
# MAGIC # Real-Time Streaming Pipeline
# MAGIC
# MAGIC Simulate real-time credit application processing with streaming predictions.
# MAGIC
# MAGIC ## Key Concepts:
# MAGIC - **Structured Streaming**: Process data as it arrives
# MAGIC - **Stream-to-Stream Joins**: Enrich streaming data
# MAGIC - **Streaming Aggregations**: Real-time KPIs
# MAGIC - **Streaming ML Inference**: Score records in real-time
# MAGIC - **Watermarks**: Handle late-arriving data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %run ./00_setup_environment

# COMMAND ----------

from pyspark.sql.functions import (
    col, lit, current_timestamp, expr, from_json, to_json, struct,
    window, count, sum as spark_sum, avg, max as spark_max,
    rand, when, round as spark_round
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, TimestampType
)
import time
import json

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming Data Generator
# MAGIC
# MAGIC Simulates credit applications arriving in real-time.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rate Source Streaming
# MAGIC
# MAGIC Use Spark's built-in rate source to generate synthetic streaming data.

# COMMAND ----------

# Define schema for synthetic records
application_schema = StructType([
    StructField("application_id", StringType(), True),
    StructField("timestamp", TimestampType(), True),
    StructField("checking_status", StringType(), True),
    StructField("duration_months", IntegerType(), True),
    StructField("credit_history", StringType(), True),
    StructField("purpose", StringType(), True),
    StructField("credit_amount", DoubleType(), True),
    StructField("savings_status", StringType(), True),
    StructField("employment_duration", StringType(), True),
    StructField("installment_rate", IntegerType(), True),
    StructField("personal_status", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("housing", StringType(), True),
    StructField("existing_credits", IntegerType(), True),
    StructField("job", StringType(), True),
    StructField("num_dependents", IntegerType(), True)
])

# COMMAND ----------

# Generate synthetic streaming data using rate source
rate_stream = (
    spark.readStream
    .format("rate")
    .option("rowsPerSecond", 10)  # 10 applications per second
    .load()
)

# Transform rate stream into credit applications
synthetic_stream = (
    rate_stream
    .withColumn("application_id", expr("concat('APP-', cast(value as string))"))
    .withColumn("event_timestamp", col("timestamp"))

    # Generate random values for features
    .withColumn("checking_status",
        when(rand() < 0.25, "A11")
        .when(rand() < 0.5, "A12")
        .when(rand() < 0.75, "A13")
        .otherwise("A14"))

    .withColumn("duration_months",
        (rand() * 60 + 6).cast(IntegerType()))

    .withColumn("credit_history",
        when(rand() < 0.2, "A30")
        .when(rand() < 0.4, "A31")
        .when(rand() < 0.6, "A32")
        .when(rand() < 0.8, "A33")
        .otherwise("A34"))

    .withColumn("purpose",
        when(rand() < 0.3, "A43")  # radio/tv
        .when(rand() < 0.5, "A40")  # new car
        .when(rand() < 0.7, "A42")  # furniture
        .otherwise("A41"))  # used car

    .withColumn("credit_amount",
        spark_round(rand() * 15000 + 500, 2))

    .withColumn("savings_status",
        when(rand() < 0.2, "A61")
        .when(rand() < 0.4, "A62")
        .when(rand() < 0.6, "A63")
        .when(rand() < 0.8, "A64")
        .otherwise("A65"))

    .withColumn("employment_duration",
        when(rand() < 0.1, "A71")
        .when(rand() < 0.3, "A72")
        .when(rand() < 0.6, "A73")
        .when(rand() < 0.8, "A74")
        .otherwise("A75"))

    .withColumn("installment_rate",
        (rand() * 4 + 1).cast(IntegerType()))

    .withColumn("personal_status",
        when(rand() < 0.3, "A91")
        .when(rand() < 0.5, "A92")
        .when(rand() < 0.8, "A93")
        .otherwise("A94"))

    .withColumn("age",
        (rand() * 50 + 20).cast(IntegerType()))

    .withColumn("housing",
        when(rand() < 0.3, "A151")
        .when(rand() < 0.8, "A152")
        .otherwise("A153"))

    .withColumn("existing_credits",
        (rand() * 3 + 1).cast(IntegerType()))

    .withColumn("job",
        when(rand() < 0.1, "A171")
        .when(rand() < 0.3, "A172")
        .when(rand() < 0.7, "A173")
        .otherwise("A174"))

    .withColumn("num_dependents",
        (rand() * 2 + 1).cast(IntegerType()))

    # Add risk score for demo purposes
    .withColumn("checking_risk_score",
        when(col("checking_status") == "A14", 0)
        .when(col("checking_status") == "A11", 3)
        .when(col("checking_status") == "A12", 2)
        .otherwise(1))

    .withColumn("savings_risk_score",
        when(col("savings_status") == "A65", 4)
        .when(col("savings_status") == "A61", 3)
        .when(col("savings_status") == "A62", 2)
        .when(col("savings_status") == "A63", 1)
        .otherwise(0))

    .withColumn("employment_risk_score",
        when(col("employment_duration") == "A71", 4)
        .when(col("employment_duration") == "A72", 3)
        .when(col("employment_duration") == "A73", 2)
        .when(col("employment_duration") == "A74", 1)
        .otherwise(0))

    .withColumn("combined_risk_score",
        col("checking_risk_score") +
        col("savings_risk_score") +
        col("employment_risk_score"))

    # Simple rule-based prediction for demo
    .withColumn("risk_prediction",
        when(col("combined_risk_score") > 8, "HIGH")
        .when(col("combined_risk_score") > 5, "MEDIUM")
        .otherwise("LOW"))

    .withColumn("approval_recommendation",
        when(col("combined_risk_score") <= 5, "APPROVE")
        .when(col("combined_risk_score") <= 8, "REVIEW")
        .otherwise("DECLINE"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming Aggregations
# MAGIC
# MAGIC Calculate real-time metrics using sliding windows.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Windowed Aggregations

# COMMAND ----------

# Real-time KPIs with 1-minute sliding windows
streaming_kpis = (
    synthetic_stream
    # Add watermark for late data handling
    .withWatermark("event_timestamp", "10 seconds")

    # Group by 1-minute tumbling windows
    .groupBy(
        window(col("event_timestamp"), "1 minute"),
        "risk_prediction"
    )
    .agg(
        count("*").alias("application_count"),
        spark_sum("credit_amount").alias("total_amount_requested"),
        avg("credit_amount").alias("avg_amount_requested"),
        avg("age").alias("avg_applicant_age"),
        avg("combined_risk_score").alias("avg_risk_score")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write Streaming KPIs to Delta

# COMMAND ----------

# Write streaming aggregations to Delta table
kpi_query = (
    streaming_kpis
    .writeStream
    .format("delta")
    .outputMode("update")  # Update mode for aggregations
    .option("checkpointLocation", f"{CHECKPOINT_PATH}/streaming_kpis")
    .trigger(processingTime="10 seconds")
    .table(f"{DATABASE_NAME}.streaming_kpis")
)

print("✓ Streaming KPIs query started")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming Application Log

# COMMAND ----------

# Select columns for application log
application_log_stream = (
    synthetic_stream
    .select(
        "application_id",
        "event_timestamp",
        "checking_status",
        "duration_months",
        "credit_amount",
        "age",
        "purpose",
        "combined_risk_score",
        "risk_prediction",
        "approval_recommendation"
    )
)

# Write to Delta as append-only log
application_log_query = (
    application_log_stream
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", f"{CHECKPOINT_PATH}/application_log")
    .trigger(processingTime="5 seconds")
    .table(f"{DATABASE_NAME}.streaming_application_log")
)

print("✓ Application log query started")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitor Streaming Queries

# COMMAND ----------

# Let streams run for a bit to accumulate data
import time
time.sleep(30)

# Check streaming query status
for query in spark.streams.active:
    print(f"""
    Query: {query.name}
    Status: {query.status}
    Progress: {query.recentProgress[-1] if query.recentProgress else 'No progress yet'}
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC ## View Streaming Results

# COMMAND ----------

# MAGIC %md
# MAGIC ### Real-Time KPIs

# COMMAND ----------

# Query streaming KPIs table
display(
    spark.sql(f"""
        SELECT
            window.start as window_start,
            window.end as window_end,
            risk_prediction,
            application_count,
            round(total_amount_requested, 2) as total_requested,
            round(avg_amount_requested, 2) as avg_amount,
            round(avg_risk_score, 2) as avg_risk
        FROM {DATABASE_NAME}.streaming_kpis
        ORDER BY window.start DESC
        LIMIT 20
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Recent Applications

# COMMAND ----------

# Query application log
display(
    spark.sql(f"""
        SELECT *
        FROM {DATABASE_NAME}.streaming_application_log
        ORDER BY event_timestamp DESC
        LIMIT 20
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Real-Time Dashboard Query
# MAGIC
# MAGIC This query can be used for a live dashboard.

# COMMAND ----------

# Summary for dashboard
display(
    spark.sql(f"""
        SELECT
            approval_recommendation,
            COUNT(*) as count,
            ROUND(AVG(credit_amount), 2) as avg_amount,
            ROUND(AVG(combined_risk_score), 2) as avg_risk_score,
            ROUND(AVG(age), 1) as avg_age
        FROM {DATABASE_NAME}.streaming_application_log
        GROUP BY approval_recommendation
        ORDER BY approval_recommendation
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming State Management

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checkpointing
# MAGIC
# MAGIC Checkpoints enable exactly-once processing and fault tolerance.

# COMMAND ----------

# List checkpoint directories
display(dbutils.fs.ls(CHECKPOINT_PATH))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stop Streaming Queries

# COMMAND ----------

# Stop all streaming queries (run when done)
# for query in spark.streams.active:
#     query.stop()
#     print(f"Stopped: {query.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production Streaming Patterns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pattern 1: Kafka Source (Production)
# MAGIC
# MAGIC ```python
# MAGIC # In production, read from Kafka
# MAGIC kafka_stream = (
# MAGIC     spark.readStream
# MAGIC     .format("kafka")
# MAGIC     .option("kafka.bootstrap.servers", "broker:9092")
# MAGIC     .option("subscribe", "credit-applications")
# MAGIC     .option("startingOffsets", "latest")
# MAGIC     .load()
# MAGIC     .select(
# MAGIC         from_json(col("value").cast("string"), schema).alias("data")
# MAGIC     )
# MAGIC     .select("data.*")
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pattern 2: Event Hubs Source (Azure)
# MAGIC
# MAGIC ```python
# MAGIC # Azure Event Hubs integration
# MAGIC eh_stream = (
# MAGIC     spark.readStream
# MAGIC     .format("eventhubs")
# MAGIC     .options(**ehConf)
# MAGIC     .load()
# MAGIC     .select(
# MAGIC         from_json(col("body").cast("string"), schema).alias("data")
# MAGIC     )
# MAGIC     .select("data.*")
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pattern 3: Auto Loader for Files
# MAGIC
# MAGIC ```python
# MAGIC # Incrementally process new files
# MAGIC file_stream = (
# MAGIC     spark.readStream
# MAGIC     .format("cloudFiles")
# MAGIC     .option("cloudFiles.format", "json")
# MAGIC     .option("cloudFiles.schemaLocation", schema_path)
# MAGIC     .load(landing_path)
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook we demonstrated:
# MAGIC
# MAGIC | Pattern | Use Case |
# MAGIC |---------|----------|
# MAGIC | Rate Source | Synthetic data generation for testing |
# MAGIC | Windowed Aggregations | Real-time KPIs with tumbling windows |
# MAGIC | Watermarks | Late data handling |
# MAGIC | Append Streams | Event logging to Delta |
# MAGIC | Update Streams | Aggregation updates |
# MAGIC | Checkpointing | Exactly-once processing |
# MAGIC
# MAGIC **Production Extensions:**
# MAGIC - Kafka/Event Hubs integration
# MAGIC - Stream-to-ML model inference
# MAGIC - Alerting on thresholds
# MAGIC - Real-time dashboards
# MAGIC
# MAGIC **Next:** Run `06_model_inference.py` for batch and streaming predictions.
