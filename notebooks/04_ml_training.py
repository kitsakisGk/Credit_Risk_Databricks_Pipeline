# Databricks notebook source
# MAGIC %md
# MAGIC # ML Model Training with MLflow
# MAGIC
# MAGIC Train credit risk classification models with experiment tracking.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# Schema name
SCHEMA_NAME = "kitsakis_credit_risk"

spark.sql(f"USE {SCHEMA_NAME}")
print(f"Using schema: {SCHEMA_NAME}")

# Get username for MLflow
username = spark.sql("SELECT current_user()").first()[0]

# COMMAND ----------

import mlflow
import mlflow.spark
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

# Set experiment
EXPERIMENT_PATH = f"/Users/{username}/credit_risk_experiment"
mlflow.set_experiment(EXPERIMENT_PATH)
print(f"MLflow experiment: {EXPERIMENT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Gold Features

# COMMAND ----------

gold_df = spark.table("gold_credit_features")
print(f"Gold records: {gold_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Features

# COMMAND ----------

# Select numeric features for ML
feature_cols = [
    "duration_months", "credit_amount", "installment_rate",
    "residence_duration", "age", "existing_credits", "num_dependents",
    "credit_per_month", "log_credit_amount",
    "checking_risk", "savings_risk", "employment_risk", "history_risk",
    "total_risk_score", "is_young_borrower", "is_short_term", "is_long_term"
]

# Create feature vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

# Prepare data
ml_df = gold_df.select(*feature_cols, "credit_risk").na.drop()
print(f"ML records (after dropping nulls): {ml_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/Test Split

# COMMAND ----------

train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)
print(f"Training: {train_df.count()}, Test: {test_df.count()}")

# Class distribution
print("\n=== Training Set Class Distribution ===")
train_df.groupBy("credit_risk").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Models

# COMMAND ----------

def train_and_log_model(model, model_name, train_data, test_data):
    """Train model and log to MLflow"""
    with mlflow.start_run(run_name=model_name):
        # Create pipeline
        pipeline = Pipeline(stages=[assembler, model])

        # Train
        pipeline_model = pipeline.fit(train_data)

        # Predict
        predictions = pipeline_model.transform(test_data)

        # Evaluate
        auc_eval = BinaryClassificationEvaluator(labelCol="credit_risk", metricName="areaUnderROC")
        acc_eval = MulticlassClassificationEvaluator(labelCol="credit_risk", metricName="accuracy")

        auc = auc_eval.evaluate(predictions)
        accuracy = acc_eval.evaluate(predictions)

        # Log metrics
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_metric("test_auc", auc)
        mlflow.log_metric("test_accuracy", accuracy)

        # Log model
        mlflow.spark.log_model(pipeline_model, "model")

        print(f"{model_name}: AUC={auc:.4f}, Accuracy={accuracy:.4f}")

        return {"model": model_name, "auc": auc, "accuracy": accuracy}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Comparison

# COMMAND ----------

results = []

# Logistic Regression
lr = LogisticRegression(labelCol="credit_risk", featuresCol="features", maxIter=100)
results.append(train_and_log_model(lr, "LogisticRegression", train_df, test_df))

# Random Forest
rf = RandomForestClassifier(labelCol="credit_risk", featuresCol="features", numTrees=100, seed=42)
results.append(train_and_log_model(rf, "RandomForest", train_df, test_df))

# Gradient Boosted Trees
gbt = GBTClassifier(labelCol="credit_risk", featuresCol="features", maxIter=50, seed=42)
results.append(train_and_log_model(gbt, "GradientBoostedTrees", train_df, test_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Summary

# COMMAND ----------

import pandas as pd

results_df = pd.DataFrame(results).sort_values("auc", ascending=False)
print("\n" + "="*50)
print("MODEL COMPARISON (sorted by AUC)")
print("="*50)
display(spark.createDataFrame(results_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Best Model Feature Importance

# COMMAND ----------

# Train best model (GBT) and get feature importance
best_pipeline = Pipeline(stages=[assembler, gbt]).fit(train_df)
gbt_model = best_pipeline.stages[-1]

# Feature importance
importance = list(zip(feature_cols, gbt_model.featureImportances.toArray()))
importance_df = pd.DataFrame(importance, columns=["feature", "importance"])
importance_df = importance_df.sort_values("importance", ascending=False)

print("\n=== Top 10 Features ===")
display(spark.createDataFrame(importance_df.head(10)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Best Model Predictions

# COMMAND ----------

# Generate predictions with best model
final_predictions = best_pipeline.transform(test_df)

# Save predictions
final_predictions.select(
    "credit_risk", "prediction", "probability"
).write.format("delta").mode("overwrite").saveAsTable("model_predictions")

print("✓ Predictions saved to model_predictions table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ✅ MLflow experiment tracking
# MAGIC ✅ Multiple models trained and compared
# MAGIC ✅ Feature importance analyzed
# MAGIC ✅ Predictions saved
# MAGIC
# MAGIC **Check MLflow UI:** Click "Experiments" in the left sidebar to see all runs!
