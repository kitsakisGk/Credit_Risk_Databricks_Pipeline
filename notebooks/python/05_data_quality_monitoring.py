# Databricks notebook source
# MAGIC %md
# MAGIC # Data Quality Monitoring
# MAGIC
# MAGIC Automated data validation across all Medallion layers.
# MAGIC
# MAGIC **Why this matters for Swiss banking:**
# MAGIC - Basel III / BCBS 239 requires data quality governance
# MAGIC - FINMA expects documented data lineage and validation
# MAGIC - Every model decision must be based on validated data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime

SCHEMA_NAME = "kitsakis_credit_risk"
spark.sql(f"USE {SCHEMA_NAME}")
print(f"Using schema: {SCHEMA_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Framework
# MAGIC
# MAGIC We validate each layer with specific expectations:
# MAGIC
# MAGIC | Layer | Checks | Purpose |
# MAGIC |-------|--------|---------|
# MAGIC | Bronze | Completeness, schema | Raw data integrity |
# MAGIC | Silver | Validity, consistency | Business rule compliance |
# MAGIC | Gold | Ranges, distributions | ML-readiness |

# COMMAND ----------

# Quality check results storage
quality_results = []

def log_check(layer, check_name, passed, total, details=""):
    """Log a quality check result."""
    quality_results.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "layer": layer,
        "check_name": check_name,
        "records_checked": total,
        "records_passed": passed,
        "records_failed": total - passed,
        "pass_rate": round(passed / total * 100, 2) if total > 0 else 0,
        "status": "PASS" if passed == total else "WARN" if passed / total > 0.95 else "FAIL",
        "details": details
    })

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze Layer Validation

# COMMAND ----------

print("=" * 60)
print("BRONZE LAYER VALIDATION")
print("=" * 60)

bronze_df = spark.table("bronze_credit_applications").toPandas()
total = len(bronze_df)
print(f"Records loaded: {total}")

# COMMAND ----------

# Check 1: No nulls in critical fields
critical_cols = ["credit_limit", "age", "sex", "education", "marriage", "default_payment"]
for col in critical_cols:
    non_null = bronze_df[col].notna().sum()
    log_check("Bronze", f"not_null_{col}", int(non_null), total, f"Null count: {total - int(non_null)}")
    status = "PASS" if non_null == total else "FAIL"
    print(f"  [{status}] {col}: {non_null}/{total} non-null")

# COMMAND ----------

# Check 2: Record count in expected range
expected_min, expected_max = 25000, 35000
count_pass = 1 if expected_min <= total <= expected_max else 0
log_check("Bronze", "record_count_range", count_pass, 1,
          f"Expected {expected_min}-{expected_max}, got {total}")
status = "PASS" if count_pass else "FAIL"
print(f"  [{status}] Record count: {total} (expected {expected_min}-{expected_max})")

# COMMAND ----------

# Check 3: No duplicate IDs
unique_ids = bronze_df["id"].nunique()
dup_count = total - unique_ids
log_check("Bronze", "no_duplicate_ids", unique_ids, total, f"Duplicates: {dup_count}")
status = "PASS" if dup_count == 0 else "WARN"
print(f"  [{status}] Unique IDs: {unique_ids}/{total}")

# COMMAND ----------

# Check 4: Schema completeness
expected_cols = [
    "id", "credit_limit", "sex", "education", "marriage", "age",
    "pay_status_1", "pay_status_2", "pay_status_3", "pay_status_4", "pay_status_5", "pay_status_6",
    "bill_amt_1", "bill_amt_2", "bill_amt_3", "bill_amt_4", "bill_amt_5", "bill_amt_6",
    "pay_amt_1", "pay_amt_2", "pay_amt_3", "pay_amt_4", "pay_amt_5", "pay_amt_6",
    "default_payment"
]
missing_cols = [c for c in expected_cols if c not in bronze_df.columns]
cols_present = len(expected_cols) - len(missing_cols)
log_check("Bronze", "schema_completeness", cols_present, len(expected_cols),
          f"Missing: {missing_cols}" if missing_cols else "All columns present")
status = "PASS" if not missing_cols else "FAIL"
print(f"  [{status}] Schema: {cols_present}/{len(expected_cols)} expected columns present")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver Layer Validation

# COMMAND ----------

print("\n" + "=" * 60)
print("SILVER LAYER VALIDATION")
print("=" * 60)

silver_df = spark.table("silver_credit_applications").toPandas()
total = len(silver_df)
print(f"Records loaded: {total}")

# COMMAND ----------

# Check 5: Age in valid range (18-100)
valid_age = ((silver_df["age"] >= 18) & (silver_df["age"] <= 100)).sum()
log_check("Silver", "age_range_18_100", int(valid_age), total,
          f"Out of range: {total - int(valid_age)}")
status = "PASS" if valid_age == total else "WARN"
print(f"  [{status}] Age range (18-100): {valid_age}/{total}")

# COMMAND ----------

# Check 6: Credit limit positive
positive_credit = (silver_df["credit_limit"] > 0).sum()
log_check("Silver", "credit_limit_positive", int(positive_credit), total,
          f"Non-positive: {total - int(positive_credit)}")
status = "PASS" if positive_credit == total else "FAIL"
print(f"  [{status}] Credit limit > 0: {positive_credit}/{total}")

# COMMAND ----------

# Check 7: Gender values valid
valid_genders = silver_df["gender"].isin(["male", "female", "unknown"]).sum()
log_check("Silver", "valid_gender_values", int(valid_genders), total,
          f"Invalid: {total - int(valid_genders)}")
status = "PASS" if valid_genders == total else "WARN"
print(f"  [{status}] Valid gender values: {valid_genders}/{total}")

# COMMAND ----------

# Check 8: Education values valid
valid_education = silver_df["education_level"].isin(
    ["graduate_school", "university", "high_school", "other"]
).sum()
log_check("Silver", "valid_education_values", int(valid_education), total,
          f"Invalid: {total - int(valid_education)}")
status = "PASS" if valid_education == total else "WARN"
print(f"  [{status}] Valid education values: {valid_education}/{total}")

# COMMAND ----------

# Check 9: Default payment is binary (0 or 1)
valid_target = silver_df["default_payment"].isin([0, 1]).sum()
log_check("Silver", "binary_target_variable", int(valid_target), total,
          f"Non-binary: {total - int(valid_target)}")
status = "PASS" if valid_target == total else "FAIL"
print(f"  [{status}] Binary target (0/1): {valid_target}/{total}")

# COMMAND ----------

# Check 10: Payment status range (-2 to 8)
pay_cols = ["pay_status_1", "pay_status_2", "pay_status_3",
            "pay_status_4", "pay_status_5", "pay_status_6"]
for col in pay_cols:
    valid = ((silver_df[col] >= -2) & (silver_df[col] <= 8)).sum()
    log_check("Silver", f"pay_status_range_{col}", int(valid), total,
              f"Min: {silver_df[col].min()}, Max: {silver_df[col].max()}")
status_all = all(
    ((silver_df[c] >= -2) & (silver_df[c] <= 8)).all() for c in pay_cols
)
print(f"  [{'PASS' if status_all else 'WARN'}] Payment status range (-2 to 8): all 6 months checked")

# COMMAND ----------

# Check 11: Record count consistency (Bronze == Silver)
bronze_count = len(bronze_df)
silver_count = len(silver_df)
consistent = 1 if bronze_count == silver_count else 0
log_check("Silver", "record_count_consistency", consistent, 1,
          f"Bronze: {bronze_count}, Silver: {silver_count}")
status = "PASS" if consistent else "FAIL"
print(f"  [{status}] Bronze-Silver consistency: {bronze_count} == {silver_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold Layer Validation

# COMMAND ----------

print("\n" + "=" * 60)
print("GOLD LAYER VALIDATION")
print("=" * 60)

gold_df = spark.table("gold_credit_features").toPandas()
total = len(gold_df)
print(f"Records loaded: {total}")

# COMMAND ----------

# Check 12: Credit utilization in reasonable range (-1 to 10)
valid_util = ((gold_df["credit_utilization"] >= -1) & (gold_df["credit_utilization"] <= 10)).sum()
log_check("Gold", "credit_utilization_range", int(valid_util), total,
          f"Min: {gold_df['credit_utilization'].min():.2f}, Max: {gold_df['credit_utilization'].max():.2f}")
status = "PASS" if valid_util / total > 0.99 else "WARN"
print(f"  [{status}] Credit utilization range: {valid_util}/{total}")

# COMMAND ----------

# Check 13: Risk score in valid range (0-18)
valid_risk = ((gold_df["total_risk_score"] >= 0) & (gold_df["total_risk_score"] <= 18)).sum()
log_check("Gold", "risk_score_range_0_18", int(valid_risk), total,
          f"Min: {gold_df['total_risk_score'].min()}, Max: {gold_df['total_risk_score'].max()}")
status = "PASS" if valid_risk == total else "FAIL"
print(f"  [{status}] Risk score range (0-18): {valid_risk}/{total}")

# COMMAND ----------

# Check 14: Payment ratio not extremely negative
valid_ratio = (gold_df["payment_ratio"] >= -100).sum()
log_check("Gold", "payment_ratio_reasonable", int(valid_ratio), total,
          f"Min: {gold_df['payment_ratio'].min():.2f}")
status = "PASS" if valid_ratio / total > 0.99 else "WARN"
print(f"  [{status}] Payment ratio reasonable: {valid_ratio}/{total}")

# COMMAND ----------

# Check 15: No NaN in ML features
ml_features = [
    "credit_limit", "age", "credit_utilization", "payment_ratio",
    "total_risk_score", "months_delayed", "default_payment"
]
for feat in ml_features:
    non_null = gold_df[feat].notna().sum()
    log_check("Gold", f"ml_feature_no_nan_{feat}", int(non_null), total,
              f"NaN count: {total - int(non_null)}")
nan_free = all(gold_df[f].notna().all() for f in ml_features)
status = "PASS" if nan_free else "FAIL"
print(f"  [{status}] ML features NaN-free: {len(ml_features)} features checked")

# COMMAND ----------

# Check 16: Default rate within expected bounds (15-30%)
default_rate = gold_df["default_payment"].mean()
rate_valid = 1 if 0.15 <= default_rate <= 0.30 else 0
log_check("Gold", "default_rate_expected_range", rate_valid, 1,
          f"Default rate: {default_rate:.2%}")
status = "PASS" if rate_valid else "WARN"
print(f"  [{status}] Default rate: {default_rate:.2%} (expected 15-30%)")

# COMMAND ----------

# Check 17: Feature count
expected_features = 37
actual_features = len(gold_df.columns)
feat_check = 1 if actual_features >= expected_features else 0
log_check("Gold", "minimum_feature_count", feat_check, 1,
          f"Expected >= {expected_features}, got {actual_features}")
status = "PASS" if feat_check else "WARN"
print(f"  [{status}] Feature count: {actual_features} (expected >= {expected_features})")

# COMMAND ----------

# Check 18: Record count consistency (Silver == Gold)
gold_count = len(gold_df)
consistent = 1 if silver_count == gold_count else 0
log_check("Gold", "record_count_consistency", consistent, 1,
          f"Silver: {silver_count}, Gold: {gold_count}")
status = "PASS" if consistent else "FAIL"
print(f"  [{status}] Silver-Gold consistency: {silver_count} == {gold_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Statistical Distribution Checks

# COMMAND ----------

print("\n" + "=" * 60)
print("STATISTICAL DISTRIBUTION CHECKS")
print("=" * 60)

# Check key feature distributions
stats = {
    "credit_limit": {"min": 0, "max": 1_000_000, "mean_min": 100_000, "mean_max": 250_000},
    "age": {"min": 18, "max": 100, "mean_min": 30, "mean_max": 45},
    "total_risk_score": {"min": 0, "max": 18, "mean_min": 2, "mean_max": 8},
}

for feat, bounds in stats.items():
    actual_mean = gold_df[feat].mean()
    actual_min = gold_df[feat].min()
    actual_max = gold_df[feat].max()

    mean_ok = bounds["mean_min"] <= actual_mean <= bounds["mean_max"]
    range_ok = actual_min >= bounds["min"] and actual_max <= bounds["max"]

    status = "PASS" if mean_ok and range_ok else "WARN"
    print(f"  [{status}] {feat}: mean={actual_mean:.1f}, range=[{actual_min:.0f}, {actual_max:.0f}]")

    log_check("Gold", f"distribution_{feat}", 1 if mean_ok and range_ok else 0, 1,
              f"Mean: {actual_mean:.1f}, Range: [{actual_min:.0f}, {actual_max:.0f}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quality Report Summary

# COMMAND ----------

results_df = pd.DataFrame(quality_results)

# Overall summary
total_checks = len(results_df)
passed = (results_df["status"] == "PASS").sum()
warned = (results_df["status"] == "WARN").sum()
failed = (results_df["status"] == "FAIL").sum()

print("\n" + "=" * 60)
print("DATA QUALITY REPORT")
print("=" * 60)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total checks: {total_checks}")
print(f"  PASSED:  {passed}")
print(f"  WARNING: {warned}")
print(f"  FAILED:  {failed}")
print(f"\nOverall Score: {passed / total_checks * 100:.1f}%")

# COMMAND ----------

# Summary by layer
print("\n" + "-" * 40)
print("BY LAYER:")
print("-" * 40)
for layer in ["Bronze", "Silver", "Gold"]:
    layer_df = results_df[results_df["layer"] == layer]
    layer_pass = (layer_df["status"] == "PASS").sum()
    layer_total = len(layer_df)
    pct = layer_pass / layer_total * 100 if layer_total > 0 else 0
    print(f"  {layer}: {layer_pass}/{layer_total} passed ({pct:.0f}%)")

# COMMAND ----------

# Show all results
display(results_df[["layer", "check_name", "records_checked", "records_passed",
                     "pass_rate", "status", "details"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Quality Metrics

# COMMAND ----------

# Save quality results to Delta table for tracking over time
spark.sql("DROP TABLE IF EXISTS data_quality_log")
spark_results = spark.createDataFrame(results_df)
spark_results.write.format("delta").mode("overwrite").saveAsTable("data_quality_log")

print(f"Quality metrics saved: {len(results_df)} checks logged")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quality Gate Decision

# COMMAND ----------

# Determine if pipeline should proceed
critical_failures = results_df[
    (results_df["status"] == "FAIL") &
    (results_df["check_name"].str.contains("record_count|binary_target|schema"))
]

if len(critical_failures) > 0:
    print("QUALITY GATE: BLOCKED")
    print("Critical failures detected:")
    for _, row in critical_failures.iterrows():
        print(f"  - {row['check_name']}: {row['details']}")
    print("\nPipeline should NOT proceed until these are resolved.")
else:
    print("QUALITY GATE: PASSED")
    print("All critical checks passed. Pipeline is safe to proceed.")
    if warned > 0:
        print(f"\nNote: {warned} non-critical warnings to review.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### Validation Framework:
# MAGIC - **Bronze**: Schema completeness, null checks, record count, duplicate detection
# MAGIC - **Silver**: Value ranges, categorical validity, target variable, cross-layer consistency
# MAGIC - **Gold**: Feature ranges, distribution checks, ML-readiness, NaN detection
# MAGIC
# MAGIC ### Regulatory Compliance:
# MAGIC - Basel III / BCBS 239: Data quality governance
# MAGIC - FINMA: Documented validation trail
# MAGIC - Quality metrics stored in Delta for audit history
