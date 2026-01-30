# Databricks notebook source
# MAGIC %md
# MAGIC # ML Model Training with MLflow
# MAGIC
# MAGIC Train a credit risk classification model with full experiment tracking.
# MAGIC
# MAGIC ## Key Concepts:
# MAGIC - **MLflow Tracking**: Log parameters, metrics, and artifacts
# MAGIC - **Model Comparison**: Evaluate multiple algorithms
# MAGIC - **Hyperparameter Tuning**: Cross-validation with grid search
# MAGIC - **Model Registry**: Version and stage models for deployment
# MAGIC - **Feature Importance**: Explain model decisions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %run ./00_setup_environment

# COMMAND ----------

import mlflow
import mlflow.spark
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier,
    GBTClassifier, DecisionTreeClassifier
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline

from pyspark.sql.functions import col, when
import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure MLflow

# COMMAND ----------

# Set MLflow experiment
mlflow.set_experiment(EXPERIMENT_PATH)

# Enable autologging for Spark ML
mlflow.spark.autolog()

print(f"MLflow Experiment: {EXPERIMENT_PATH}")
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and Prepare Data

# COMMAND ----------

# Load Gold features table
features_df = spark.table(f"{DATABASE_NAME}.gold_ml_features")
print(f"Total records: {features_df.count()}")

# Check class distribution
print("\n=== Target Distribution ===")
features_df.groupBy("credit_risk").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Feature Columns

# COMMAND ----------

# Numerical features for model
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

# Encoded categorical features
categorical_encoded_features = [
    "checking_status_encoded", "credit_history_encoded", "purpose_encoded",
    "savings_status_encoded", "employment_duration_encoded",
    "personal_status_encoded", "other_parties_encoded",
    "property_magnitude_encoded", "other_payment_plans_encoded",
    "housing_encoded", "job_encoded"
]

# Binary features
binary_features = [
    "is_young_borrower", "is_senior_borrower",
    "is_short_term", "is_long_term"
]

# Combine all features
all_features = numerical_features + categorical_encoded_features + binary_features

print(f"Total features: {len(all_features)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Handle Missing/Boolean Values

# COMMAND ----------

# Convert boolean columns to integers and handle nulls
ml_df = features_df.select(
    "_record_id",
    *all_features,
    "credit_risk"
)

# Handle nulls in boolean columns
for col_name in ["is_foreign_worker", "has_telephone"]:
    if col_name in ml_df.columns:
        ml_df = ml_df.withColumn(
            col_name,
            when(col(col_name) == True, 1)
            .when(col(col_name) == False, 0)
            .otherwise(0)
        )

# Drop rows with nulls in features
ml_df = ml_df.na.drop()
print(f"Records after cleaning: {ml_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train/Test Split

# COMMAND ----------

# Stratified split
train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)

print(f"Training set: {train_df.count()} records")
print(f"Test set: {test_df.count()} records")

# Check balance
print("\n=== Training Set Balance ===")
train_df.groupBy("credit_risk").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build ML Pipeline

# COMMAND ----------

# Assemble features into vector
assembler = VectorAssembler(
    inputCols=all_features,
    outputCol="features_raw",
    handleInvalid="skip"
)

# Scale features
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True
)

print("✓ Feature pipeline created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training & Comparison

# COMMAND ----------

# MAGIC %md
# MAGIC ### Helper Functions

# COMMAND ----------

def train_and_evaluate(model, model_name, train_data, test_data, feature_pipeline):
    """
    Train a model and log results to MLflow.
    """
    with mlflow.start_run(run_name=model_name) as run:
        # Log model parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("num_features", len(all_features))
        mlflow.log_param("train_size", train_data.count())
        mlflow.log_param("test_size", test_data.count())

        # Create full pipeline
        pipeline = Pipeline(stages=[
            *feature_pipeline,
            model
        ])

        # Train model
        pipeline_model = pipeline.fit(train_data)

        # Make predictions
        train_predictions = pipeline_model.transform(train_data)
        test_predictions = pipeline_model.transform(test_data)

        # Evaluate
        evaluator_auc = BinaryClassificationEvaluator(
            labelCol="credit_risk",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )

        evaluator_pr = BinaryClassificationEvaluator(
            labelCol="credit_risk",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderPR"
        )

        evaluator_acc = MulticlassClassificationEvaluator(
            labelCol="credit_risk",
            predictionCol="prediction",
            metricName="accuracy"
        )

        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol="credit_risk",
            predictionCol="prediction",
            metricName="f1"
        )

        # Calculate metrics
        train_auc = evaluator_auc.evaluate(train_predictions)
        test_auc = evaluator_auc.evaluate(test_predictions)
        test_pr = evaluator_pr.evaluate(test_predictions)
        test_acc = evaluator_acc.evaluate(test_predictions)
        test_f1 = evaluator_f1.evaluate(test_predictions)

        # Log metrics
        mlflow.log_metric("train_auc", train_auc)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("test_pr_auc", test_pr)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1", test_f1)

        # Log model
        mlflow.spark.log_model(
            pipeline_model,
            artifact_path="model",
            registered_model_name=f"{DATABASE_NAME}_credit_risk_model"
        )

        # Log feature names
        mlflow.log_text(",".join(all_features), "feature_names.txt")

        print(f"""
        ╔══════════════════════════════════════════════════════════════╗
        ║  {model_name:^56}  ║
        ╠══════════════════════════════════════════════════════════════╣
        ║  Train AUC:     {train_auc:.4f}                                      ║
        ║  Test AUC:      {test_auc:.4f}                                      ║
        ║  Test PR-AUC:   {test_pr:.4f}                                      ║
        ║  Test Accuracy: {test_acc:.4f}                                      ║
        ║  Test F1:       {test_f1:.4f}                                      ║
        ║  Run ID:        {run.info.run_id}            ║
        ╚══════════════════════════════════════════════════════════════╝
        """)

        return {
            "model_name": model_name,
            "run_id": run.info.run_id,
            "pipeline_model": pipeline_model,
            "train_auc": train_auc,
            "test_auc": test_auc,
            "test_accuracy": test_acc,
            "test_f1": test_f1
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Multiple Models

# COMMAND ----------

# Define feature pipeline stages
feature_stages = [assembler, scaler]

# Define models to compare
models = {
    "Logistic Regression": LogisticRegression(
        labelCol="credit_risk",
        featuresCol="features",
        maxIter=100,
        regParam=0.01
    ),
    "Random Forest": RandomForestClassifier(
        labelCol="credit_risk",
        featuresCol="features",
        numTrees=100,
        maxDepth=5,
        seed=42
    ),
    "Gradient Boosted Trees": GBTClassifier(
        labelCol="credit_risk",
        featuresCol="features",
        maxIter=50,
        maxDepth=5,
        seed=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        labelCol="credit_risk",
        featuresCol="features",
        maxDepth=10,
        seed=42
    )
}

# Train all models
results = []
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    result = train_and_evaluate(
        model, model_name, train_df, test_df, feature_stages
    )
    results.append(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Comparison

# COMMAND ----------

# Create comparison dataframe
comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values("test_auc", ascending=False)

print("=== Model Comparison (sorted by Test AUC) ===")
display(spark.createDataFrame(comparison_df[["model_name", "train_auc", "test_auc", "test_accuracy", "test_f1"]]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter Tuning (Best Model)

# COMMAND ----------

# Select best model type for tuning
best_model_name = comparison_df.iloc[0]["model_name"]
print(f"Tuning: {best_model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cross-Validation with Grid Search

# COMMAND ----------

with mlflow.start_run(run_name=f"{best_model_name}_Tuned") as run:
    # Create GBT model for tuning (often performs best)
    gbt = GBTClassifier(
        labelCol="credit_risk",
        featuresCol="features",
        seed=42
    )

    # Create pipeline
    pipeline = Pipeline(stages=[assembler, scaler, gbt])

    # Parameter grid
    param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, [3, 5, 7])
        .addGrid(gbt.maxIter, [30, 50, 100])
        .addGrid(gbt.stepSize, [0.05, 0.1])
        .build()
    )

    # Cross-validator
    evaluator = BinaryClassificationEvaluator(
        labelCol="credit_risk",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=5,
        parallelism=4,
        seed=42
    )

    print("Starting cross-validation (this may take a while)...")

    # Fit
    cv_model = cv.fit(train_df)

    # Get best model
    best_model = cv_model.bestModel

    # Evaluate
    test_predictions = best_model.transform(test_df)
    test_auc = evaluator.evaluate(test_predictions)

    # Get best params
    best_gbt = best_model.stages[-1]

    mlflow.log_param("best_maxDepth", best_gbt.getMaxDepth())
    mlflow.log_param("best_maxIter", best_gbt.getMaxIter())
    mlflow.log_param("best_stepSize", best_gbt.getStepSize())
    mlflow.log_metric("cv_best_auc", max(cv_model.avgMetrics))
    mlflow.log_metric("test_auc_tuned", test_auc)

    # Log tuned model
    mlflow.spark.log_model(
        best_model,
        artifact_path="tuned_model",
        registered_model_name=f"{DATABASE_NAME}_credit_risk_model_tuned"
    )

    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║           HYPERPARAMETER TUNING RESULTS                      ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Best Max Depth:   {best_gbt.getMaxDepth():<10}                               ║
    ║  Best Max Iter:    {best_gbt.getMaxIter():<10}                               ║
    ║  Best Step Size:   {best_gbt.getStepSize():<10}                               ║
    ║  CV Best AUC:      {max(cv_model.avgMetrics):.4f}                                      ║
    ║  Test AUC (Tuned): {test_auc:.4f}                                      ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance

# COMMAND ----------

# Get feature importance from GBT model
gbt_model = best_model.stages[-1]
feature_importances = gbt_model.featureImportances.toArray()

# Create importance dataframe
importance_df = pd.DataFrame({
    "feature": all_features,
    "importance": feature_importances
}).sort_values("importance", ascending=False)

print("=== Top 15 Features ===")
display(spark.createDataFrame(importance_df.head(15)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Registry

# COMMAND ----------

# Get the latest model version
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = f"{DATABASE_NAME}_credit_risk_model_tuned"

# Get latest version
latest_version = client.get_latest_versions(model_name, stages=["None"])[0]

print(f"""
Model Registry Info:
- Model Name: {model_name}
- Version: {latest_version.version}
- Status: {latest_version.status}
- Run ID: {latest_version.run_id}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transition to Staging

# COMMAND ----------

# Transition model to staging
client.transition_model_version_stage(
    name=model_name,
    version=latest_version.version,
    stage="Staging",
    archive_existing_versions=True
)

print(f"✓ Model {model_name} v{latest_version.version} transitioned to Staging")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Test Predictions

# COMMAND ----------

# Save predictions for analysis
predictions_df = (
    test_predictions
    .select(
        "_record_id",
        "credit_risk",
        "prediction",
        "probability"
    )
    .withColumn("predicted_default", col("prediction").cast("int"))
)

(
    predictions_df
    .write
    .format("delta")
    .mode("overwrite")
    .save(f"{GOLD_PATH}/model_predictions")
)

print(f"✓ Predictions saved to {GOLD_PATH}/model_predictions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook we demonstrated:
# MAGIC
# MAGIC | Capability | Implementation |
# MAGIC |------------|----------------|
# MAGIC | MLflow Experiment Tracking | Logged params, metrics, models |
# MAGIC | Model Comparison | 4 algorithms evaluated |
# MAGIC | Hyperparameter Tuning | 5-fold CV with grid search |
# MAGIC | Model Registry | Versioning and staging |
# MAGIC | Feature Importance | GBT feature rankings |
# MAGIC
# MAGIC **Next:** Run `05_streaming_simulation.py` to set up real-time inference.
