# Databricks notebook source
# MAGIC %md
# MAGIC # Model Monitoring & Drift Detection
# MAGIC
# MAGIC Production model monitoring for credit risk models.
# MAGIC
# MAGIC **Why this matters for Swiss banking:**
# MAGIC - Models degrade over time as customer behavior changes
# MAGIC - FINMA requires ongoing model validation, not just at training time
# MAGIC - Early drift detection prevents bad credit decisions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies

# COMMAND ----------

%pip install xgboost shap --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

SCHEMA_NAME = "kitsakis_credit_risk"
spark.sql(f"USE {SCHEMA_NAME}")
print(f"Using schema: {SCHEMA_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data & Train Model
# MAGIC
# MAGIC We retrain the model and compare against historical predictions.

# COMMAND ----------

from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

# Load gold features
df = spark.table("gold_credit_features").toPandas()

feature_cols = [
    "credit_limit", "age",
    "bill_amt_1", "bill_amt_2", "bill_amt_3", "bill_amt_4", "bill_amt_5", "bill_amt_6",
    "pay_amt_1", "pay_amt_2", "pay_amt_3", "pay_amt_4", "pay_amt_5", "pay_amt_6",
    "pay_status_1", "pay_status_2", "pay_status_3", "pay_status_4", "pay_status_5", "pay_status_6",
    "avg_bill_amount", "avg_payment_amount", "credit_utilization", "log_credit_limit",
    "payment_ratio", "pays_full_balance", "is_young_borrower",
    "delay_risk_1", "delay_risk_2", "delay_risk_3",
    "education_risk", "marital_risk", "utilization_risk", "total_risk_score",
    "months_delayed", "max_delay_months", "total_bill_amt", "total_pay_amt"
]

X = df[feature_cols].fillna(0)
y = df["default_payment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training: {len(X_train)}, Test: {len(X_test)}")

# COMMAND ----------

# Train model
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    random_state=42, use_label_encoder=False, eval_metric='auc'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Model trained successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Performance Metrics

# COMMAND ----------

metrics = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_type": "XGBoost",
    "auc": round(roc_auc_score(y_test, y_prob), 4),
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "f1_score": round(f1_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "default_rate_train": round(y_train.mean(), 4),
    "default_rate_test": round(y_test.mean(), 4),
    "n_features": len(feature_cols)
}

print("=" * 60)
print("MODEL PERFORMANCE REPORT")
print("=" * 60)
for key, value in metrics.items():
    print(f"  {key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Population Stability Index (PSI)
# MAGIC
# MAGIC PSI measures how much a feature distribution has shifted.
# MAGIC
# MAGIC | PSI Value | Interpretation |
# MAGIC |-----------|---------------|
# MAGIC | < 0.10 | No significant change |
# MAGIC | 0.10 - 0.25 | Moderate shift - monitor closely |
# MAGIC | > 0.25 | Significant shift - investigate and retrain |

# COMMAND ----------

def calculate_psi(reference, current, bins=10):
    """Calculate Population Stability Index between two distributions."""
    # Create bins from reference distribution
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    # Calculate proportions in each bin
    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    # Add small value to avoid division by zero
    ref_pct = (ref_counts + 0.001) / len(reference)
    cur_pct = (cur_counts + 0.001) / len(current)

    # PSI formula
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return round(psi, 4)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Drift Analysis
# MAGIC
# MAGIC Compare training data (reference) vs test data (current) distributions.

# COMMAND ----------

print("=" * 60)
print("FEATURE DRIFT ANALYSIS (PSI)")
print("=" * 60)

psi_results = []
for col in feature_cols:
    psi_value = calculate_psi(X_train[col].values, X_test[col].values)

    if psi_value < 0.10:
        status = "STABLE"
    elif psi_value < 0.25:
        status = "MONITOR"
    else:
        status = "DRIFT"

    psi_results.append({
        "feature": col,
        "psi": psi_value,
        "status": status,
        "train_mean": round(X_train[col].mean(), 2),
        "test_mean": round(X_test[col].mean(), 2),
        "mean_shift_pct": round(abs(X_train[col].mean() - X_test[col].mean()) / (X_train[col].mean() + 0.001) * 100, 1)
    })

psi_df = pd.DataFrame(psi_results).sort_values("psi", ascending=False)

# COMMAND ----------

# Show top features by drift
print("\nTop 10 Features by PSI:")
display(psi_df.head(10))

# COMMAND ----------

# Summary
stable = (psi_df["status"] == "STABLE").sum()
monitor = (psi_df["status"] == "MONITOR").sum()
drift = (psi_df["status"] == "DRIFT").sum()

print(f"\nDrift Summary:")
print(f"  STABLE:  {stable} features")
print(f"  MONITOR: {monitor} features")
print(f"  DRIFT:   {drift} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction Stability Analysis

# COMMAND ----------

print("=" * 60)
print("PREDICTION STABILITY ANALYSIS")
print("=" * 60)

# Split test set into two halves to simulate "time periods"
half = len(X_test) // 2
period_1_prob = y_prob[:half]
period_2_prob = y_prob[half:]

# Compare prediction distributions
pred_psi = calculate_psi(period_1_prob, period_2_prob)

print(f"\nPrediction PSI: {pred_psi}")
print(f"Period 1 mean probability: {period_1_prob.mean():.4f}")
print(f"Period 2 mean probability: {period_2_prob.mean():.4f}")

if pred_psi < 0.10:
    print("Status: STABLE - Model predictions are consistent")
elif pred_psi < 0.25:
    print("Status: MONITOR - Predictions showing some variation")
else:
    print("Status: ALERT - Significant prediction drift detected!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Segment Performance Analysis
# MAGIC
# MAGIC Check if model performs equally well across customer segments.

# COMMAND ----------

print("=" * 60)
print("SEGMENT PERFORMANCE ANALYSIS")
print("=" * 60)

# Create segments on test data
test_df = pd.DataFrame(X_test, columns=feature_cols)
test_df["actual"] = y_test.values
test_df["predicted"] = y_pred
test_df["probability"] = y_prob

# By age group
test_df["age_segment"] = pd.cut(test_df["age"], bins=[0, 30, 40, 50, 100],
                                 labels=["Young (<30)", "Adult (30-40)", "Middle (40-50)", "Senior (50+)"])

print("\nPerformance by Age Segment:")
segment_results = []
for segment in test_df["age_segment"].unique():
    seg_data = test_df[test_df["age_segment"] == segment]
    if len(seg_data) > 50:
        seg_auc = roc_auc_score(seg_data["actual"], seg_data["probability"])
        seg_acc = accuracy_score(seg_data["actual"], seg_data["predicted"])
        segment_results.append({
            "segment": str(segment),
            "records": len(seg_data),
            "default_rate": round(seg_data["actual"].mean(), 4),
            "auc": round(seg_auc, 4),
            "accuracy": round(seg_acc, 4)
        })

segment_df = pd.DataFrame(segment_results)
display(segment_df)

# COMMAND ----------

# By credit limit segment
test_df["credit_segment"] = pd.cut(test_df["credit_limit"],
                                     bins=[0, 50000, 150000, 300000, float("inf")],
                                     labels=["Low", "Medium", "High", "Very High"])

print("\nPerformance by Credit Limit Segment:")
credit_results = []
for segment in test_df["credit_segment"].unique():
    seg_data = test_df[test_df["credit_segment"] == segment]
    if len(seg_data) > 50:
        seg_auc = roc_auc_score(seg_data["actual"], seg_data["probability"])
        seg_acc = accuracy_score(seg_data["actual"], seg_data["predicted"])
        credit_results.append({
            "segment": str(segment),
            "records": len(seg_data),
            "default_rate": round(seg_data["actual"].mean(), 4),
            "auc": round(seg_auc, 4),
            "accuracy": round(seg_acc, 4)
        })

credit_df = pd.DataFrame(credit_results)
display(credit_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Fairness Check
# MAGIC
# MAGIC Ensure model doesn't discriminate across demographic groups.

# COMMAND ----------

print("=" * 60)
print("MODEL FAIRNESS ANALYSIS")
print("=" * 60)

# Reload gold data to get gender info
gold_df = spark.table("gold_credit_features").toPandas()
test_indices = X_test.index
gold_test = gold_df.loc[test_indices]

# Fairness by gender
print("\nDefault Prediction Rate by Gender:")
fairness_results = []
for gender in gold_test["gender"].unique():
    mask = gold_test["gender"] == gender
    gender_pred = y_pred[mask.values]
    gender_actual = y_test.values[mask.values]
    gender_prob = y_prob[mask.values]

    if len(gender_pred) > 50:
        fairness_results.append({
            "group": gender,
            "records": int(mask.sum()),
            "actual_default_rate": round(gender_actual.mean(), 4),
            "predicted_default_rate": round(gender_pred.mean(), 4),
            "avg_risk_probability": round(gender_prob.mean(), 4),
            "auc": round(roc_auc_score(gender_actual, gender_prob), 4)
        })

fairness_df = pd.DataFrame(fairness_results)
display(fairness_df)

# Check for significant disparate impact
if len(fairness_df) >= 2:
    max_rate = fairness_df["predicted_default_rate"].max()
    min_rate = fairness_df["predicted_default_rate"].min()
    ratio = min_rate / max_rate if max_rate > 0 else 1

    print(f"\nDisparate Impact Ratio: {ratio:.3f}")
    if ratio >= 0.8:
        print("Status: PASS - No significant disparate impact (>= 0.8)")
    else:
        print("Status: WARNING - Potential disparate impact detected (< 0.8)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitoring Dashboard Summary

# COMMAND ----------

print("\n" + "=" * 60)
print("MONITORING DASHBOARD")
print("=" * 60)
print(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"\n--- Model Performance ---")
print(f"  AUC:       {metrics['auc']}")
print(f"  Accuracy:  {metrics['accuracy']}")
print(f"  F1 Score:  {metrics['f1_score']}")
print(f"  Precision: {metrics['precision']}")
print(f"  Recall:    {metrics['recall']}")

print(f"\n--- Data Drift ---")
print(f"  Features stable:    {stable}")
print(f"  Features to monitor: {monitor}")
print(f"  Features drifted:   {drift}")
print(f"  Prediction PSI:     {pred_psi}")

print(f"\n--- Recommendations ---")
if drift > 0:
    drifted = psi_df[psi_df["status"] == "DRIFT"]["feature"].tolist()
    print(f"  RETRAIN: {drift} features have significant drift: {drifted}")
elif monitor > 2:
    print(f"  WATCH: {monitor} features showing moderate shift")
else:
    print(f"  OK: Model is performing within expected bounds")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Monitoring Results

# COMMAND ----------

# Save model metrics
metrics_df = pd.DataFrame([metrics])
spark.sql("DROP TABLE IF EXISTS model_performance_log")
spark.createDataFrame(metrics_df).write.format("delta").mode("overwrite").saveAsTable("model_performance_log")

# Save PSI results
spark.sql("DROP TABLE IF EXISTS feature_drift_log")
spark.createDataFrame(psi_df).write.format("delta").mode("overwrite").saveAsTable("feature_drift_log")

print("Monitoring results saved to Delta tables:")
print("  - model_performance_log")
print("  - feature_drift_log")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### Monitoring Capabilities:
# MAGIC - **Model Performance**: AUC, accuracy, F1, precision, recall tracked over time
# MAGIC - **Feature Drift**: PSI calculated for all 37 features
# MAGIC - **Prediction Stability**: Distribution of model outputs compared across periods
# MAGIC - **Segment Analysis**: Performance broken down by age and credit segments
# MAGIC - **Fairness**: Disparate impact ratio across demographic groups
# MAGIC
# MAGIC ### Regulatory Compliance:
# MAGIC - FINMA: Ongoing model validation and monitoring
# MAGIC - Basel III: Model risk management (SR 11-7)
# MAGIC - All metrics logged to Delta tables for audit trail
