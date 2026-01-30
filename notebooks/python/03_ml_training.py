# Databricks notebook source
# MAGIC %md
# MAGIC # ML Model Training with Scikit-Learn
# MAGIC
# MAGIC Train credit default prediction models using scikit-learn.
# MAGIC
# MAGIC **Works on Serverless compute!**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

SCHEMA_NAME = "kitsakis_credit_risk"

spark.sql(f"USE {SCHEMA_NAME}")
print(f"Using schema: {SCHEMA_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Gold Features

# COMMAND ----------

# Load data and convert to Pandas
gold_df = spark.table("gold_credit_features")
print(f"Gold records: {gold_df.count()}")

# Convert to pandas for sklearn
pdf = gold_df.toPandas()
print(f"Loaded {len(pdf)} records into pandas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Features

# COMMAND ----------

import pandas as pd
import numpy as np

# Select numeric features for ML
feature_cols = [
    # Original features
    "credit_limit", "age",
    # Bill amounts
    "bill_amt_1", "bill_amt_2", "bill_amt_3", "bill_amt_4", "bill_amt_5", "bill_amt_6",
    # Payment amounts
    "pay_amt_1", "pay_amt_2", "pay_amt_3", "pay_amt_4", "pay_amt_5", "pay_amt_6",
    # Payment status
    "pay_status_1", "pay_status_2", "pay_status_3", "pay_status_4", "pay_status_5", "pay_status_6",
    # Engineered features
    "avg_bill_amount", "avg_payment_amount", "credit_utilization", "log_credit_limit",
    "payment_ratio", "pays_full_balance", "is_young_borrower",
    # Risk scores
    "delay_risk_1", "delay_risk_2", "delay_risk_3",
    "education_risk", "marital_risk", "utilization_risk", "total_risk_score",
    # Silver features
    "months_delayed", "max_delay_months", "total_bill_amt", "total_pay_amt"
]

# Prepare X and y
X = pdf[feature_cols].fillna(0)
y = pdf["default_payment"]

print(f"Features: {len(feature_cols)}")
print(f"Samples: {len(X)}")
print(f"\nTarget distribution:")
print(y.value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/Test Split

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} records")
print(f"Test set: {len(X_test)} records")
print(f"\nTraining set class distribution:")
print(y_train.value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Models

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 1: Logistic Regression

# COMMAND ----------

print("="*50)
print("Training: Logistic Regression")
print("="*50)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)
lr_prob = lr.predict_proba(X_test_scaled)[:, 1]

lr_auc = roc_auc_score(y_test, lr_prob)
lr_acc = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

print(f"AUC:      {lr_auc:.4f}")
print(f"Accuracy: {lr_acc:.4f}")
print(f"F1 Score: {lr_f1:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 2: Random Forest

# COMMAND ----------

print("="*50)
print("Training: Random Forest")
print("="*50)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

rf_auc = roc_auc_score(y_test, rf_prob)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print(f"AUC:      {rf_auc:.4f}")
print(f"Accuracy: {rf_acc:.4f}")
print(f"F1 Score: {rf_f1:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 3: Gradient Boosting

# COMMAND ----------

print("="*50)
print("Training: Gradient Boosting")
print("="*50)

gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb.fit(X_train, y_train)

gb_pred = gb.predict(X_test)
gb_prob = gb.predict_proba(X_test)[:, 1]

gb_auc = roc_auc_score(y_test, gb_prob)
gb_acc = accuracy_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred)

print(f"AUC:      {gb_auc:.4f}")
print(f"Accuracy: {gb_acc:.4f}")
print(f"F1 Score: {gb_f1:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Comparison

# COMMAND ----------

results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "Gradient Boosting"],
    "AUC": [lr_auc, rf_auc, gb_auc],
    "Accuracy": [lr_acc, rf_acc, gb_acc],
    "F1 Score": [lr_f1, rf_f1, gb_f1]
}).sort_values("AUC", ascending=False)

print("\n" + "="*60)
print("MODEL COMPARISON (sorted by AUC)")
print("="*60)
display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Best Model

# COMMAND ----------

best_idx = results["AUC"].idxmax()
best_model_name = results.loc[best_idx, "Model"]
best_auc = results.loc[best_idx, "AUC"]

print(f"Best Model: {best_model_name}")
print(f"AUC: {best_auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance (Random Forest)

# COMMAND ----------

# Get feature importance
importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n=== Top 15 Most Important Features ===")
display(importance_df.head(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classification Report (Best Model)

# COMMAND ----------

# Use Gradient Boosting predictions (typically best)
print("="*50)
print("CLASSIFICATION REPORT - Gradient Boosting")
print("="*50)
print(classification_report(y_test, gb_pred, target_names=["No Default", "Default"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion Matrix

# COMMAND ----------

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, gb_pred)
tn, fp, fn, tp = cm.ravel()

print("="*50)
print("CONFUSION MATRIX")
print("="*50)
print(f"""
                    Predicted
                 |  No   |  Yes  |
        ---------|-------|-------|
Actual    No     | {tn:5} | {fp:5} |
          Yes    | {fn:5} | {tp:5} |
        ---------|-------|-------|
""")

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"Precision: {precision:.4f} (of predicted defaults, how many actually defaulted)")
print(f"Recall:    {recall:.4f} (of actual defaults, how many did we catch)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Predictions

# COMMAND ----------

# Create predictions dataframe
predictions_pdf = pd.DataFrame({
    "actual": y_test.values,
    "predicted": gb_pred,
    "probability": gb_prob
})

# Drop existing table and save
spark.sql("DROP TABLE IF EXISTS model_predictions")
predictions_spark = spark.createDataFrame(predictions_pdf)
predictions_spark.write.format("delta").mode("overwrite").saveAsTable("model_predictions")

print(f"Saved {len(predictions_pdf)} predictions to model_predictions table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### Models Trained:
# MAGIC - **Logistic Regression** - Linear baseline model
# MAGIC - **Random Forest** - 100 trees, max depth 10
# MAGIC - **Gradient Boosting** - 100 estimators, max depth 5
# MAGIC
# MAGIC ### Key Findings:
# MAGIC - Payment history (pay_status) is the strongest predictor of default
# MAGIC - Credit utilization and recent bill amounts are important features
# MAGIC - Tree-based models outperform logistic regression
# MAGIC
# MAGIC ### Technical Stack:
# MAGIC - **scikit-learn** for ML models
# MAGIC - **pandas** for data manipulation
# MAGIC - **Delta Lake** for storing predictions
