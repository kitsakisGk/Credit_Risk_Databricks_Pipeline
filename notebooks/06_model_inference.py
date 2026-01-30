# Databricks notebook source
# MAGIC %md
# MAGIC # Model Inference - Batch & Real-Time Predictions
# MAGIC
# MAGIC Deploy the trained model for both batch and streaming inference.
# MAGIC
# MAGIC ## Key Concepts:
# MAGIC - **Model Loading**: Load from MLflow Registry
# MAGIC - **Batch Inference**: Score historical data
# MAGIC - **Streaming Inference**: Real-time predictions
# MAGIC - **Model Serving**: REST API endpoints (Databricks Model Serving)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %run ./00_setup_environment

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql.functions import (
    col, struct, to_json, from_json, current_timestamp,
    when, round as spark_round, array, lit
)
from pyspark.sql.types import DoubleType, ArrayType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Model from Registry

# COMMAND ----------

# Get model from registry
client = MlflowClient()
model_name = f"{DATABASE_NAME}_credit_risk_model_tuned"

# Load staging model
model_uri = f"models:/{model_name}/Staging"

print(f"Loading model: {model_uri}")

# Load as Spark ML pipeline
loaded_model = mlflow.spark.load_model(model_uri)

print("✓ Model loaded successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Inference

# COMMAND ----------

# MAGIC %md
# MAGIC ### Score Gold Features Table

# COMMAND ----------

# Load features
features_df = spark.table(f"{DATABASE_NAME}.gold_ml_features")

# Prepare features (same as training)
numerical_features = [
    "duration_months", "credit_amount", "installment_rate",
    "residence_duration", "age", "existing_credits", "num_dependents",
    "credit_per_month", "credit_income_proxy", "age_credit_ratio",
    "log_credit_amount", "stability_score", "dependency_ratio",
    "checking_risk_score", "savings_risk_score",
    "employment_risk_score", "history_risk_score",
    "combined_risk_score", "age_x_credit", "checking_x_savings",
    "employment_x_history", "financial_stress_indicator"
]

categorical_encoded_features = [
    "checking_status_encoded", "credit_history_encoded", "purpose_encoded",
    "savings_status_encoded", "employment_duration_encoded",
    "personal_status_encoded", "other_parties_encoded",
    "property_magnitude_encoded", "other_payment_plans_encoded",
    "housing_encoded", "job_encoded"
]

binary_features = [
    "is_young_borrower", "is_senior_borrower",
    "is_short_term", "is_long_term"
]

# Score with model
batch_predictions = loaded_model.transform(features_df.na.drop())

print(f"Scored {batch_predictions.count()} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extract Probability Scores

# COMMAND ----------

# Extract default probability from probability vector
from pyspark.ml.functions import vector_to_array

scored_df = (
    batch_predictions
    .withColumn("probability_array", vector_to_array(col("probability")))
    .withColumn("default_probability",
        spark_round(col("probability_array")[1], 4))
    .withColumn("prediction_label",
        when(col("prediction") == 1, "Default")
        .otherwise("No Default"))
    .withColumn("risk_tier",
        when(col("default_probability") < 0.2, "Low Risk")
        .when(col("default_probability") < 0.4, "Medium Risk")
        .when(col("default_probability") < 0.6, "High Risk")
        .otherwise("Very High Risk"))
    .withColumn("scored_timestamp", current_timestamp())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### View Predictions

# COMMAND ----------

display(
    scored_df
    .select(
        "_record_id",
        "credit_amount",
        "age",
        "combined_risk_score",
        "credit_risk",
        "prediction",
        "prediction_label",
        "default_probability",
        "risk_tier"
    )
    .limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prediction Distribution

# COMMAND ----------

# Analyze predictions
display(
    scored_df
    .groupBy("risk_tier")
    .agg(
        count("*").alias("count"),
        avg("default_probability").alias("avg_probability"),
        avg("credit_amount").alias("avg_credit_amount"),
        avg("age").alias("avg_age")
    )
    .orderBy("risk_tier")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Batch Predictions

# COMMAND ----------

# Save predictions to Gold layer
(
    scored_df
    .select(
        "_record_id",
        "credit_risk",
        "prediction",
        "prediction_label",
        "default_probability",
        "risk_tier",
        "scored_timestamp"
    )
    .write
    .format("delta")
    .mode("overwrite")
    .save(f"{GOLD_PATH}/batch_predictions")
)

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {DATABASE_NAME}.gold_batch_predictions
    USING DELTA
    LOCATION '{GOLD_PATH}/batch_predictions'
""")

print(f"✓ Batch predictions saved to {GOLD_PATH}/batch_predictions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Performance Monitoring

# COMMAND ----------

# Calculate confusion matrix
from pyspark.sql.functions import sum as spark_sum

confusion = (
    scored_df
    .groupBy("credit_risk", "prediction")
    .count()
    .orderBy("credit_risk", "prediction")
)

display(confusion)

# COMMAND ----------

# Calculate metrics
tp = scored_df.filter((col("credit_risk") == 1) & (col("prediction") == 1)).count()
tn = scored_df.filter((col("credit_risk") == 0) & (col("prediction") == 0)).count()
fp = scored_df.filter((col("credit_risk") == 0) & (col("prediction") == 1)).count()
fn = scored_df.filter((col("credit_risk") == 1) & (col("prediction") == 0)).count()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"""
╔══════════════════════════════════════════════════════════════╗
║               MODEL PERFORMANCE METRICS                      ║
╠══════════════════════════════════════════════════════════════╣
║  Accuracy:    {accuracy:.4f}                                         ║
║  Precision:   {precision:.4f}                                         ║
║  Recall:      {recall:.4f}                                         ║
║  F1 Score:    {f1:.4f}                                         ║
╠══════════════════════════════════════════════════════════════╣
║  True Pos:    {tp:>6}     False Pos:   {fp:>6}                    ║
║  True Neg:    {tn:>6}     False Neg:   {fn:>6}                    ║
╚══════════════════════════════════════════════════════════════╝
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming Inference

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Model as UDF for Streaming

# COMMAND ----------

# Register model as Spark UDF for streaming
predict_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=model_uri,
    result_type=DoubleType()
)

# Note: For complex models, you might need to use the full pipeline
# and transform in foreachBatch

# COMMAND ----------

# MAGIC %md
# MAGIC ### Streaming Inference Pattern

# COMMAND ----------

def score_micro_batch(batch_df, batch_id):
    """
    Score a micro-batch of streaming data with the ML model.
    This function is called for each micro-batch.
    """
    if batch_df.count() == 0:
        return

    # Apply feature engineering (simplified for demo)
    features_df = (
        batch_df
        .withColumn("credit_per_month",
            col("credit_amount") / col("duration_months"))
        .withColumn("log_credit_amount",
            log(col("credit_amount") + 1))
        # Add other features as needed...
    )

    # Score with model
    predictions = loaded_model.transform(features_df)

    # Extract probability and save
    scored = (
        predictions
        .withColumn("probability_array", vector_to_array(col("probability")))
        .withColumn("default_probability", col("probability_array")[1])
        .withColumn("risk_tier",
            when(col("default_probability") < 0.2, "Low Risk")
            .when(col("default_probability") < 0.4, "Medium Risk")
            .when(col("default_probability") < 0.6, "High Risk")
            .otherwise("Very High Risk"))
        .withColumn("scored_timestamp", current_timestamp())
        .withColumn("batch_id", lit(batch_id))
    )

    # Append to streaming predictions table
    (
        scored
        .select(
            "application_id",
            "event_timestamp",
            "credit_amount",
            "prediction",
            "default_probability",
            "risk_tier",
            "scored_timestamp",
            "batch_id"
        )
        .write
        .format("delta")
        .mode("append")
        .saveAsTable(f"{DATABASE_NAME}.streaming_predictions")
    )

    print(f"Batch {batch_id}: Scored {scored.count()} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example: Score Streaming Applications
# MAGIC
# MAGIC This would connect to the streaming application log.

# COMMAND ----------

# Example streaming inference (commented out - run if streaming source exists)
"""
# Read from streaming application log
streaming_source = (
    spark.readStream
    .format("delta")
    .table(f"{DATABASE_NAME}.streaming_application_log")
)

# Apply scoring using foreachBatch
streaming_query = (
    streaming_source
    .writeStream
    .foreachBatch(score_micro_batch)
    .option("checkpointLocation", f"{CHECKPOINT_PATH}/streaming_inference")
    .trigger(processingTime="30 seconds")
    .start()
)
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Serving (REST API)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Enable Model Serving
# MAGIC
# MAGIC In Databricks, you can enable Model Serving for REST API access:
# MAGIC
# MAGIC 1. Go to Machine Learning > Models
# MAGIC 2. Select your registered model
# MAGIC 3. Click "Use model for inference" > "Real-time"
# MAGIC 4. Configure endpoint settings
# MAGIC
# MAGIC This creates a REST endpoint for real-time predictions.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample API Request
# MAGIC
# MAGIC ```python
# MAGIC import requests
# MAGIC import json
# MAGIC
# MAGIC # Model serving endpoint URL
# MAGIC endpoint_url = "https://<workspace>.cloud.databricks.com/serving-endpoints/<model-name>/invocations"
# MAGIC
# MAGIC # Prepare request
# MAGIC headers = {
# MAGIC     "Authorization": f"Bearer {token}",
# MAGIC     "Content-Type": "application/json"
# MAGIC }
# MAGIC
# MAGIC # Sample input
# MAGIC data = {
# MAGIC     "dataframe_records": [
# MAGIC         {
# MAGIC             "duration_months": 24,
# MAGIC             "credit_amount": 5000,
# MAGIC             "age": 35,
# MAGIC             "checking_status": "A12",
# MAGIC             # ... other features
# MAGIC         }
# MAGIC     ]
# MAGIC }
# MAGIC
# MAGIC # Make prediction
# MAGIC response = requests.post(endpoint_url, headers=headers, json=data)
# MAGIC prediction = response.json()
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook we demonstrated:
# MAGIC
# MAGIC | Inference Pattern | Use Case | Latency |
# MAGIC |-------------------|----------|---------|
# MAGIC | Batch Scoring | Historical analysis, reporting | Minutes |
# MAGIC | Streaming foreachBatch | Near-real-time, micro-batches | Seconds |
# MAGIC | Model Serving API | Real-time single predictions | Milliseconds |
# MAGIC
# MAGIC **Production Considerations:**
# MAGIC - Model monitoring and drift detection
# MAGIC - A/B testing for model versions
# MAGIC - Fallback strategies for model failures
# MAGIC - Feature store integration for consistent features
# MAGIC
# MAGIC **This completes the Credit Risk ML Pipeline!**
