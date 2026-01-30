# Databricks notebook source
# MAGIC %md
# MAGIC # ML Model Training with MLflow
# MAGIC
# MAGIC Train credit default prediction models with experiment tracking.

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
EXPERIMENT_PATH = f"/Users/{username}/credit_default_experiment"
mlflow.set_experiment(EXPERIMENT_PATH)
print(f"MLflow experiment: {EXPERIMENT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Gold Features

# COMMAND ----------

gold_df = spark.table("gold_credit_features")
print(f"Gold records: {gold_df.count()}")

# Check class distribution
print("\n=== Target Variable Distribution ===")
gold_df.groupBy("default_payment").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Features

# COMMAND ----------

# Select numeric features for ML
feature_cols = [
    # Original features
    "credit_limit", "age",
    # Bill amounts (6 months)
    "bill_amt_1", "bill_amt_2", "bill_amt_3", "bill_amt_4", "bill_amt_5", "bill_amt_6",
    # Payment amounts (6 months)
    "pay_amt_1", "pay_amt_2", "pay_amt_3", "pay_amt_4", "pay_amt_5", "pay_amt_6",
    # Payment status (6 months)
    "pay_status_1", "pay_status_2", "pay_status_3", "pay_status_4", "pay_status_5", "pay_status_6",
    # Engineered features from Gold layer
    "avg_bill_amount", "avg_payment_amount", "credit_utilization", "log_credit_limit",
    "payment_ratio", "pays_full_balance", "is_young_borrower",
    # Risk scores
    "delay_risk_1", "delay_risk_2", "delay_risk_3",
    "education_risk", "marital_risk", "utilization_risk", "total_risk_score",
    # Silver layer features
    "months_delayed", "max_delay_months", "total_bill_amt", "total_pay_amt"
]

# Create feature vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

# Prepare data
ml_df = gold_df.select(*feature_cols, "default_payment").na.drop()
print(f"ML records (after dropping nulls): {ml_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/Test Split

# COMMAND ----------

train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)
print(f"Training: {train_df.count()}, Test: {test_df.count()}")

# Class distribution
print("\n=== Training Set Class Distribution ===")
train_df.groupBy("default_payment").count().show()

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
        auc_eval = BinaryClassificationEvaluator(labelCol="default_payment", metricName="areaUnderROC")
        acc_eval = MulticlassClassificationEvaluator(labelCol="default_payment", metricName="accuracy")
        f1_eval = MulticlassClassificationEvaluator(labelCol="default_payment", metricName="f1")

        auc = auc_eval.evaluate(predictions)
        accuracy = acc_eval.evaluate(predictions)
        f1 = f1_eval.evaluate(predictions)

        # Log metrics
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("train_size", train_data.count())
        mlflow.log_param("test_size", test_data.count())
        mlflow.log_metric("test_auc", auc)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1", f1)

        # Log model
        mlflow.spark.log_model(pipeline_model, "model")

        print(f"{model_name}: AUC={auc:.4f}, Accuracy={accuracy:.4f}, F1={f1:.4f}")

        return {"model": model_name, "auc": auc, "accuracy": accuracy, "f1": f1, "pipeline": pipeline_model}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Comparison

# COMMAND ----------

results = []

# Logistic Regression
lr = LogisticRegression(labelCol="default_payment", featuresCol="features", maxIter=100)
results.append(train_and_log_model(lr, "LogisticRegression", train_df, test_df))

# Random Forest
rf = RandomForestClassifier(labelCol="default_payment", featuresCol="features", numTrees=100, seed=42)
results.append(train_and_log_model(rf, "RandomForest", train_df, test_df))

# Gradient Boosted Trees
gbt = GBTClassifier(labelCol="default_payment", featuresCol="features", maxIter=50, seed=42)
results.append(train_and_log_model(gbt, "GradientBoostedTrees", train_df, test_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Summary

# COMMAND ----------

import pandas as pd

# Create results dataframe (without pipeline column)
results_summary = [{k: v for k, v in r.items() if k != "pipeline"} for r in results]
results_df = pd.DataFrame(results_summary).sort_values("auc", ascending=False)
print("\n" + "="*60)
print("MODEL COMPARISON (sorted by AUC)")
print("="*60)
display(spark.createDataFrame(results_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Best Model Feature Importance

# COMMAND ----------

# Get best model (highest AUC)
best_result = max(results, key=lambda x: x["auc"])
best_model_name = best_result["model"]
print(f"Best model: {best_model_name} (AUC: {best_result['auc']:.4f})")

# For tree-based models, show feature importance
if best_model_name in ["RandomForest", "GradientBoostedTrees"]:
    best_pipeline = best_result["pipeline"]
    tree_model = best_pipeline.stages[-1]

    # Feature importance
    importance = list(zip(feature_cols, tree_model.featureImportances.toArray()))
    importance_df = pd.DataFrame(importance, columns=["feature", "importance"])
    importance_df = importance_df.sort_values("importance", ascending=False)

    print("\n=== Top 15 Most Important Features ===")
    display(spark.createDataFrame(importance_df.head(15)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Best Model Predictions

# COMMAND ----------

# Generate predictions with best model
best_pipeline = best_result["pipeline"]
final_predictions = best_pipeline.transform(test_df)

# Save predictions
final_predictions.select(
    "default_payment", "prediction", "probability"
).write.format("delta").mode("overwrite").saveAsTable("model_predictions")

print("Predictions saved to model_predictions table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion Matrix

# COMMAND ----------

# Calculate confusion matrix
confusion = final_predictions.groupBy("default_payment", "prediction").count()
print("\n=== Confusion Matrix ===")
confusion.show()

# Calculate metrics
tp = final_predictions.filter((col("default_payment") == 1) & (col("prediction") == 1)).count()
tn = final_predictions.filter((col("default_payment") == 0) & (col("prediction") == 0)).count()
fp = final_predictions.filter((col("default_payment") == 0) & (col("prediction") == 1)).count()
fn = final_predictions.filter((col("default_payment") == 1) & (col("prediction") == 0)).count()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"True Positives: {tp}, True Negatives: {tn}")
print(f"False Positives: {fp}, False Negatives: {fn}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC - MLflow experiment tracking configured
# MAGIC - 3 models trained and compared (Logistic Regression, Random Forest, GBT)
# MAGIC - Best model selected based on AUC
# MAGIC - Feature importance analyzed
# MAGIC - Predictions saved to Delta table
# MAGIC
# MAGIC **Check MLflow UI:** Click "Experiments" in the left sidebar to see all runs!
