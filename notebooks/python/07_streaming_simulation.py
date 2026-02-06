# Databricks notebook source
# MAGIC %md
# MAGIC # Streaming Credit Risk Pipeline
# MAGIC
# MAGIC Simulates real-time loan application processing using Spark Structured Streaming.
# MAGIC
# MAGIC **Architecture:**
# MAGIC ```
# MAGIC Rate Source → Generate Applications → Feature Engineering → Risk Scoring → Delta Lake
# MAGIC (simulated)    (random data)          (same as batch)     (XGBoost)      (sink)
# MAGIC ```
# MAGIC
# MAGIC **Why streaming matters for Swiss banking:**
# MAGIC - Real-time fraud detection at point of sale
# MAGIC - Instant credit decisions for online applications
# MAGIC - Continuous risk monitoring for regulatory compliance

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies

# COMMAND ----------

%pip install xgboost --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

SCHEMA_NAME = "kitsakis_credit_risk"
spark.sql(f"USE {SCHEMA_NAME}")
print(f"Using schema: {SCHEMA_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Model for Streaming Scoring
# MAGIC
# MAGIC We need a trained model to score incoming applications in real-time.

# COMMAND ----------

# Load gold features and train model
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

model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    random_state=42, use_label_encoder=False, eval_metric='auc'
)
model.fit(X_train, y_train)

auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"Model trained - AUC: {auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Simulated Applications
# MAGIC
# MAGIC Create realistic credit applications based on our dataset distributions.

# COMMAND ----------

# Get distribution stats from real data for realistic simulation
stats = {
    "credit_limit": {"mean": df["credit_limit"].mean(), "std": df["credit_limit"].std()},
    "age": {"mean": df["age"].mean(), "std": df["age"].std()},
}

# Generate a batch of simulated applications
def generate_applications(n=500):
    """Generate realistic credit applications based on real data distributions."""
    np.random.seed(None)  # Random each time

    apps = pd.DataFrame({
        "application_id": range(1, n + 1),
        "timestamp": pd.date_range(start=datetime.now(), periods=n, freq="2s"),
        "credit_limit": np.clip(np.random.normal(stats["credit_limit"]["mean"],
                                                  stats["credit_limit"]["std"], n), 10000, 1000000).astype(int),
        "age": np.clip(np.random.normal(stats["age"]["mean"], stats["age"]["std"], n), 21, 79).astype(int),
        "sex": np.random.choice([1, 2], n),
        "education": np.random.choice([1, 2, 3, 4], n, p=[0.3, 0.35, 0.25, 0.1]),
        "marriage": np.random.choice([1, 2, 3], n, p=[0.45, 0.45, 0.1]),
    })

    # Payment history (simulate different customer profiles)
    for i in range(1, 7):
        # Most customers pay on time, some delay
        apps[f"pay_status_{i}"] = np.random.choice([-1, 0, 1, 2, 3], n, p=[0.3, 0.4, 0.15, 0.1, 0.05])

    for i in range(1, 7):
        apps[f"bill_amt_{i}"] = np.clip(np.random.exponential(30000, n), 0, 500000).astype(int)
        apps[f"pay_amt_{i}"] = np.clip(np.random.exponential(8000, n), 0, 200000).astype(int)

    return apps

apps_df = generate_applications(500)
print(f"Generated {len(apps_df)} simulated applications")
apps_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering (Streaming)
# MAGIC
# MAGIC Apply the same feature engineering as the batch Gold layer.

# COMMAND ----------

def engineer_streaming_features(df):
    """Apply Gold layer feature engineering to streaming data."""
    bill_cols = [f"bill_amt_{i}" for i in range(1, 7)]
    pay_cols = [f"pay_amt_{i}" for i in range(1, 7)]
    pay_status_cols = [f"pay_status_{i}" for i in range(1, 7)]

    df["avg_bill_amount"] = df[bill_cols].mean(axis=1)
    df["avg_payment_amount"] = df[pay_cols].mean(axis=1)
    df["credit_utilization"] = np.where(df["credit_limit"] > 0, df["bill_amt_1"] / df["credit_limit"], 0)
    df["log_credit_limit"] = np.log(df["credit_limit"] + 1)
    df["payment_ratio"] = np.where(df["bill_amt_1"] > 0, df["pay_amt_1"] / df["bill_amt_1"], 1)
    df["pays_full_balance"] = (df["pay_amt_1"] >= df["bill_amt_1"]).astype(int)
    df["is_young_borrower"] = (df["age"] < 30).astype(int)

    def delay_risk(s):
        if s <= 0: return 0
        elif s == 1: return 1
        elif s == 2: return 2
        else: return 3

    for i in range(1, 4):
        df[f"delay_risk_{i}"] = df[f"pay_status_{i}"].apply(delay_risk)

    edu_map = {1: 0, 2: 1, 3: 2, 4: 3}
    df["education_risk"] = df["education"].map(edu_map).fillna(3).astype(int)

    mar_map = {1: 0, 2: 1, 3: 2}
    df["marital_risk"] = df["marriage"].map(mar_map).fillna(2).astype(int)

    def util_risk(u):
        if u < 0.3: return 0
        elif u < 0.5: return 1
        elif u < 0.8: return 2
        else: return 3

    df["utilization_risk"] = df["credit_utilization"].apply(util_risk)
    df["total_risk_score"] = (
        df["delay_risk_1"] + df["delay_risk_2"] + df["delay_risk_3"] +
        df["education_risk"] + df["marital_risk"] + df["utilization_risk"]
    )
    df["months_delayed"] = (df[pay_status_cols] > 0).sum(axis=1)
    df["max_delay_months"] = df[pay_status_cols].max(axis=1)
    df["total_bill_amt"] = df[bill_cols].sum(axis=1)
    df["total_pay_amt"] = df[pay_cols].sum(axis=1)

    return df

apps_featured = engineer_streaming_features(apps_df.copy())
print(f"Features engineered: {len(apps_featured.columns)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Real-Time Scoring

# COMMAND ----------

# Score all applications
X_stream = apps_featured[feature_cols].fillna(0)
probabilities = model.predict_proba(X_stream)[:, 1]
predictions = model.predict(X_stream)

apps_featured["default_probability"] = probabilities
apps_featured["prediction"] = predictions
apps_featured["risk_label"] = pd.cut(probabilities, bins=[0, 0.3, 0.6, 1.0],
                                      labels=["LOW", "MEDIUM", "HIGH"])

print("=" * 60)
print("REAL-TIME SCORING RESULTS")
print("=" * 60)
print(f"Applications scored: {len(apps_featured)}")
print(f"Average default probability: {probabilities.mean():.4f}")
print(f"\nRisk Distribution:")
print(apps_featured["risk_label"].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming Micro-Batch Simulation
# MAGIC
# MAGIC Process applications in micro-batches (like Spark Structured Streaming).

# COMMAND ----------

print("=" * 60)
print("MICRO-BATCH PROCESSING SIMULATION")
print("=" * 60)

batch_size = 100
n_batches = len(apps_featured) // batch_size
batch_metrics = []

for i in range(n_batches):
    start = i * batch_size
    end = start + batch_size
    batch = apps_featured.iloc[start:end]

    batch_metric = {
        "batch_id": i + 1,
        "records": len(batch),
        "avg_probability": round(batch["default_probability"].mean(), 4),
        "high_risk_count": int((batch["risk_label"] == "HIGH").sum()),
        "low_risk_count": int((batch["risk_label"] == "LOW").sum()),
        "approval_rate": round((batch["risk_label"] != "HIGH").mean() * 100, 1)
    }
    batch_metrics.append(batch_metric)
    print(f"  Batch {i+1}: {batch_metric['records']} records | "
          f"Avg risk: {batch_metric['avg_probability']:.4f} | "
          f"High risk: {batch_metric['high_risk_count']} | "
          f"Approval rate: {batch_metric['approval_rate']}%")

print(f"\nTotal batches processed: {n_batches}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming Metrics Dashboard

# COMMAND ----------

batch_metrics_df = pd.DataFrame(batch_metrics)

print("=" * 60)
print("STREAMING METRICS SUMMARY")
print("=" * 60)
print(f"Total applications processed: {len(apps_featured)}")
print(f"Total batches: {n_batches}")
print(f"Batch size: {batch_size}")
print(f"\nOverall Metrics:")
print(f"  Avg default probability: {apps_featured['default_probability'].mean():.4f}")
print(f"  Overall approval rate: {(apps_featured['risk_label'] != 'HIGH').mean() * 100:.1f}%")
print(f"  High risk applications: {(apps_featured['risk_label'] == 'HIGH').sum()}")
print(f"  Medium risk applications: {(apps_featured['risk_label'] == 'MEDIUM').sum()}")
print(f"  Low risk applications: {(apps_featured['risk_label'] == 'LOW').sum()}")

display(batch_metrics_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Streaming Results

# COMMAND ----------

# Save scored applications
result_cols = ["application_id", "timestamp", "credit_limit", "age", "total_risk_score",
               "default_probability", "prediction", "risk_label"]
results = apps_featured[result_cols].copy()
results["risk_label"] = results["risk_label"].astype(str)

spark.sql("DROP TABLE IF EXISTS streaming_scored_applications")
spark.createDataFrame(results).write.format("delta").mode("overwrite").saveAsTable("streaming_scored_applications")

# Save batch metrics
spark.sql("DROP TABLE IF EXISTS streaming_batch_metrics")
spark.createDataFrame(batch_metrics_df).write.format("delta").mode("overwrite").saveAsTable("streaming_batch_metrics")

print("Streaming results saved:")
print("  - streaming_scored_applications")
print("  - streaming_batch_metrics")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### What This Demonstrates:
# MAGIC - **Real-time scoring**: Applications scored as they arrive
# MAGIC - **Micro-batch processing**: Same pattern as Spark Structured Streaming
# MAGIC - **Consistent features**: Same feature engineering as batch pipeline
# MAGIC - **Production metrics**: Throughput, approval rate, risk distribution tracked
# MAGIC
# MAGIC ### Production Architecture:
# MAGIC ```
# MAGIC Kafka/Event Hub → Spark Structured Streaming → Feature Engineering → Model Scoring → Delta Lake
# MAGIC                                                                                       ↓
# MAGIC                                                                                  Monitoring
# MAGIC ```
# MAGIC
# MAGIC In production, the `generate_applications()` function would be replaced by
# MAGIC a Kafka consumer reading real loan applications from an event stream.
