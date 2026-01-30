-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Risk Analysis & Scoring Model
-- MAGIC
-- MAGIC SQL-based credit risk scoring and analysis.
-- MAGIC
-- MAGIC Since SQL Warehouse doesn't support ML libraries, we use a **rule-based scoring model**
-- MAGIC derived from statistical analysis of the data.

-- COMMAND ----------

USE kitsakis_credit_risk;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Analyze Risk Factors
-- MAGIC
-- MAGIC First, let's understand which factors correlate with default.

-- COMMAND ----------

-- Payment delay is the strongest predictor
SELECT
    'Recent Payment Delay' AS factor,
    pay_status_1 AS value,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(default_payment) / COUNT(*), 1) AS default_rate_pct
FROM gold_credit_features
GROUP BY pay_status_1
ORDER BY pay_status_1;

-- COMMAND ----------

-- Months with any delay
SELECT
    'Months Delayed' AS factor,
    months_delayed AS value,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(default_payment) / COUNT(*), 1) AS default_rate_pct
FROM gold_credit_features
GROUP BY months_delayed
ORDER BY months_delayed;

-- COMMAND ----------

-- Credit utilization impact
SELECT
    'Credit Utilization' AS factor,
    CASE
        WHEN credit_utilization < 0.2 THEN '1. Very Low (<20%)'
        WHEN credit_utilization < 0.4 THEN '2. Low (20-40%)'
        WHEN credit_utilization < 0.6 THEN '3. Medium (40-60%)'
        WHEN credit_utilization < 0.8 THEN '4. High (60-80%)'
        ELSE '5. Very High (>80%)'
    END AS value,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(default_payment) / COUNT(*), 1) AS default_rate_pct
FROM gold_credit_features
GROUP BY 2
ORDER BY 2;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Create Risk Scoring Model
-- MAGIC
-- MAGIC Based on analysis, we create a weighted scoring model.

-- COMMAND ----------

CREATE OR REPLACE TABLE risk_model_scores AS
SELECT
    id,
    credit_limit,
    age,
    gender,
    education,
    marital_status,
    credit_utilization,
    months_delayed,
    max_delay_months,
    total_risk_score,
    default_payment,

    -- Weighted risk score (based on observed default correlations)
    (
        -- Payment history weight: 50% (strongest predictor)
        (CASE
            WHEN pay_status_1 >= 2 THEN 50
            WHEN pay_status_1 = 1 THEN 30
            WHEN pay_status_1 = 0 THEN 15
            ELSE 0
        END) +

        -- Months delayed weight: 20%
        (CASE
            WHEN months_delayed >= 4 THEN 20
            WHEN months_delayed >= 2 THEN 12
            WHEN months_delayed >= 1 THEN 6
            ELSE 0
        END) +

        -- Credit utilization weight: 15%
        (CASE
            WHEN credit_utilization > 0.8 THEN 15
            WHEN credit_utilization > 0.6 THEN 10
            WHEN credit_utilization > 0.4 THEN 5
            ELSE 0
        END) +

        -- Max delay severity weight: 10%
        (CASE
            WHEN max_delay_months >= 3 THEN 10
            WHEN max_delay_months >= 2 THEN 6
            WHEN max_delay_months >= 1 THEN 3
            ELSE 0
        END) +

        -- Age factor weight: 5%
        (CASE
            WHEN age < 25 THEN 5
            WHEN age < 30 THEN 3
            ELSE 0
        END)
    ) AS weighted_risk_score,

    -- Risk category
    CASE
        WHEN (
            (CASE WHEN pay_status_1 >= 2 THEN 50 WHEN pay_status_1 = 1 THEN 30 WHEN pay_status_1 = 0 THEN 15 ELSE 0 END) +
            (CASE WHEN months_delayed >= 4 THEN 20 WHEN months_delayed >= 2 THEN 12 WHEN months_delayed >= 1 THEN 6 ELSE 0 END) +
            (CASE WHEN credit_utilization > 0.8 THEN 15 WHEN credit_utilization > 0.6 THEN 10 WHEN credit_utilization > 0.4 THEN 5 ELSE 0 END) +
            (CASE WHEN max_delay_months >= 3 THEN 10 WHEN max_delay_months >= 2 THEN 6 WHEN max_delay_months >= 1 THEN 3 ELSE 0 END) +
            (CASE WHEN age < 25 THEN 5 WHEN age < 30 THEN 3 ELSE 0 END)
        ) >= 50 THEN 'HIGH RISK'
        WHEN (
            (CASE WHEN pay_status_1 >= 2 THEN 50 WHEN pay_status_1 = 1 THEN 30 WHEN pay_status_1 = 0 THEN 15 ELSE 0 END) +
            (CASE WHEN months_delayed >= 4 THEN 20 WHEN months_delayed >= 2 THEN 12 WHEN months_delayed >= 1 THEN 6 ELSE 0 END) +
            (CASE WHEN credit_utilization > 0.8 THEN 15 WHEN credit_utilization > 0.6 THEN 10 WHEN credit_utilization > 0.4 THEN 5 ELSE 0 END) +
            (CASE WHEN max_delay_months >= 3 THEN 10 WHEN max_delay_months >= 2 THEN 6 WHEN max_delay_months >= 1 THEN 3 ELSE 0 END) +
            (CASE WHEN age < 25 THEN 5 WHEN age < 30 THEN 3 ELSE 0 END)
        ) >= 25 THEN 'MEDIUM RISK'
        ELSE 'LOW RISK'
    END AS risk_category,

    -- Model prediction (1 = likely to default)
    CASE
        WHEN (
            (CASE WHEN pay_status_1 >= 2 THEN 50 WHEN pay_status_1 = 1 THEN 30 WHEN pay_status_1 = 0 THEN 15 ELSE 0 END) +
            (CASE WHEN months_delayed >= 4 THEN 20 WHEN months_delayed >= 2 THEN 12 WHEN months_delayed >= 1 THEN 6 ELSE 0 END) +
            (CASE WHEN credit_utilization > 0.8 THEN 15 WHEN credit_utilization > 0.6 THEN 10 WHEN credit_utilization > 0.4 THEN 5 ELSE 0 END) +
            (CASE WHEN max_delay_months >= 3 THEN 10 WHEN max_delay_months >= 2 THEN 6 WHEN max_delay_months >= 1 THEN 3 ELSE 0 END) +
            (CASE WHEN age < 25 THEN 5 WHEN age < 30 THEN 3 ELSE 0 END)
        ) >= 40 THEN 1
        ELSE 0
    END AS predicted_default

FROM gold_credit_features;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Model Performance Evaluation

-- COMMAND ----------

SELECT COUNT(*) AS total_scored FROM risk_model_scores;

-- COMMAND ----------

-- Risk category distribution
SELECT
    risk_category,
    COUNT(*) AS customers,
    SUM(default_payment) AS actual_defaults,
    ROUND(100.0 * SUM(default_payment) / COUNT(*), 1) AS actual_default_rate_pct,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM risk_model_scores), 1) AS pct_of_portfolio
FROM risk_model_scores
GROUP BY risk_category
ORDER BY
    CASE risk_category
        WHEN 'LOW RISK' THEN 1
        WHEN 'MEDIUM RISK' THEN 2
        WHEN 'HIGH RISK' THEN 3
    END;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Confusion Matrix

-- COMMAND ----------

SELECT
    CASE
        WHEN default_payment = 1 AND predicted_default = 1 THEN 'True Positive (Correctly predicted default)'
        WHEN default_payment = 0 AND predicted_default = 0 THEN 'True Negative (Correctly predicted no default)'
        WHEN default_payment = 0 AND predicted_default = 1 THEN 'False Positive (Wrongly predicted default)'
        WHEN default_payment = 1 AND predicted_default = 0 THEN 'False Negative (Missed default)'
    END AS prediction_result,
    COUNT(*) AS count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM risk_model_scores), 1) AS percentage
FROM risk_model_scores
GROUP BY 1
ORDER BY 1;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Model Metrics

-- COMMAND ----------

SELECT
    -- Accuracy
    ROUND(100.0 * SUM(CASE WHEN predicted_default = default_payment THEN 1 ELSE 0 END) / COUNT(*), 2) AS accuracy_pct,

    -- Precision (of those we predicted as default, how many actually defaulted)
    ROUND(100.0 * SUM(CASE WHEN predicted_default = 1 AND default_payment = 1 THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN predicted_default = 1 THEN 1 ELSE 0 END), 0), 2) AS precision_pct,

    -- Recall (of actual defaults, how many did we catch)
    ROUND(100.0 * SUM(CASE WHEN predicted_default = 1 AND default_payment = 1 THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN default_payment = 1 THEN 1 ELSE 0 END), 0), 2) AS recall_pct,

    -- Counts
    SUM(CASE WHEN predicted_default = 1 AND default_payment = 1 THEN 1 ELSE 0 END) AS true_positives,
    SUM(CASE WHEN predicted_default = 0 AND default_payment = 0 THEN 1 ELSE 0 END) AS true_negatives,
    SUM(CASE WHEN predicted_default = 1 AND default_payment = 0 THEN 1 ELSE 0 END) AS false_positives,
    SUM(CASE WHEN predicted_default = 0 AND default_payment = 1 THEN 1 ELSE 0 END) AS false_negatives

FROM risk_model_scores;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Risk Score Distribution

-- COMMAND ----------

SELECT
    CASE
        WHEN weighted_risk_score < 10 THEN '00-09'
        WHEN weighted_risk_score < 20 THEN '10-19'
        WHEN weighted_risk_score < 30 THEN '20-29'
        WHEN weighted_risk_score < 40 THEN '30-39'
        WHEN weighted_risk_score < 50 THEN '40-49'
        WHEN weighted_risk_score < 60 THEN '50-59'
        WHEN weighted_risk_score < 70 THEN '60-69'
        ELSE '70+'
    END AS score_range,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(default_payment) / COUNT(*), 1) AS default_rate_pct
FROM risk_model_scores
GROUP BY 1
ORDER BY 1;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## High Risk Customer Profile

-- COMMAND ----------

SELECT
    gender,
    education,
    marital_status,
    ROUND(AVG(age), 1) AS avg_age,
    ROUND(AVG(credit_limit), 0) AS avg_credit_limit,
    ROUND(AVG(credit_utilization), 2) AS avg_utilization,
    ROUND(AVG(months_delayed), 1) AS avg_months_delayed,
    COUNT(*) AS customer_count
FROM risk_model_scores
WHERE risk_category = 'HIGH RISK'
GROUP BY gender, education, marital_status
ORDER BY customer_count DESC
LIMIT 10;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Summary
-- MAGIC
-- MAGIC ### Model Results:
-- MAGIC - Rule-based credit scoring model created
-- MAGIC - Risk categories: LOW, MEDIUM, HIGH
-- MAGIC - Performance metrics calculated (accuracy, precision, recall)
-- MAGIC
-- MAGIC ### Key Findings:
-- MAGIC - Payment history is the strongest default predictor
-- MAGIC - High credit utilization correlates with higher default risk
-- MAGIC - Model successfully segments customers by risk level
-- MAGIC
-- MAGIC ### Tables Created:
-- MAGIC - `bronze_credit_clean` - Raw data with proper column names
-- MAGIC - `silver_credit_applications` - Cleaned and transformed data
-- MAGIC - `gold_credit_features` - Feature-engineered data
-- MAGIC - `risk_model_scores` - Final risk scores and predictions
