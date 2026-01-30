-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Gold Layer - Feature Engineering
-- MAGIC
-- MAGIC Create ML-ready features and risk scores from Silver data.

-- COMMAND ----------

USE kitsakis_credit_risk;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Create Gold Feature Table

-- COMMAND ----------

CREATE OR REPLACE TABLE gold_credit_features AS
SELECT
    id,
    credit_limit,
    age,
    gender,
    education,
    marital_status,

    -- Original payment data
    pay_status_1, pay_status_2, pay_status_3,
    pay_status_4, pay_status_5, pay_status_6,
    bill_amt_1, bill_amt_2, bill_amt_3,
    bill_amt_4, bill_amt_5, bill_amt_6,
    pay_amt_1, pay_amt_2, pay_amt_3,
    pay_amt_4, pay_amt_5, pay_amt_6,

    -- Silver features
    months_delayed,
    max_delay_months,
    total_bill_amt,
    total_pay_amt,

    -- Credit utilization features
    (bill_amt_1 + bill_amt_2 + bill_amt_3 +
     bill_amt_4 + bill_amt_5 + bill_amt_6) / 6 AS avg_bill_amount,

    (pay_amt_1 + pay_amt_2 + pay_amt_3 +
     pay_amt_4 + pay_amt_5 + pay_amt_6) / 6 AS avg_payment_amount,

    CASE
        WHEN credit_limit > 0 THEN ROUND(bill_amt_1 / credit_limit, 4)
        ELSE 0
    END AS credit_utilization,

    LN(credit_limit + 1) AS log_credit_limit,

    -- Payment behavior
    CASE
        WHEN bill_amt_1 > 0 THEN ROUND(pay_amt_1 / bill_amt_1, 4)
        ELSE 1
    END AS payment_ratio,

    CASE WHEN pay_amt_1 >= bill_amt_1 THEN 1 ELSE 0 END AS pays_full_balance,

    -- Age features
    CASE
        WHEN age < 25 THEN 'young'
        WHEN age < 35 THEN 'young_adult'
        WHEN age < 50 THEN 'middle_aged'
        WHEN age < 65 THEN 'senior'
        ELSE 'elderly'
    END AS age_group,

    CASE WHEN age < 30 THEN 1 ELSE 0 END AS is_young_borrower,

    -- Credit limit bucket
    CASE
        WHEN credit_limit < 50000 THEN 'low'
        WHEN credit_limit < 150000 THEN 'medium'
        WHEN credit_limit < 300000 THEN 'high'
        ELSE 'very_high'
    END AS credit_bucket,

    -- Risk scores (higher = riskier)

    -- Payment delay risk
    CASE
        WHEN pay_status_1 <= 0 THEN 0
        WHEN pay_status_1 = 1 THEN 1
        WHEN pay_status_1 = 2 THEN 2
        ELSE 3
    END AS delay_risk_1,

    CASE
        WHEN pay_status_2 <= 0 THEN 0
        WHEN pay_status_2 = 1 THEN 1
        WHEN pay_status_2 = 2 THEN 2
        ELSE 3
    END AS delay_risk_2,

    CASE
        WHEN pay_status_3 <= 0 THEN 0
        WHEN pay_status_3 = 1 THEN 1
        WHEN pay_status_3 = 2 THEN 2
        ELSE 3
    END AS delay_risk_3,

    -- Education risk
    CASE education
        WHEN 'graduate_school' THEN 0
        WHEN 'university' THEN 1
        WHEN 'high_school' THEN 2
        ELSE 3
    END AS education_risk,

    -- Marital risk
    CASE marital_status
        WHEN 'married' THEN 0
        WHEN 'single' THEN 1
        ELSE 2
    END AS marital_risk,

    -- Utilization risk
    CASE
        WHEN credit_limit > 0 AND bill_amt_1 / credit_limit < 0.3 THEN 0
        WHEN credit_limit > 0 AND bill_amt_1 / credit_limit < 0.5 THEN 1
        WHEN credit_limit > 0 AND bill_amt_1 / credit_limit < 0.8 THEN 2
        ELSE 3
    END AS utilization_risk,

    -- Target
    default_payment,

    -- Metadata
    current_timestamp() AS _gold_timestamp

FROM silver_credit_applications;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Add Total Risk Score

-- COMMAND ----------

-- Add combined risk score
CREATE OR REPLACE TABLE gold_credit_features AS
SELECT
    *,
    (delay_risk_1 + delay_risk_2 + delay_risk_3 +
     education_risk + marital_risk + utilization_risk) AS total_risk_score
FROM gold_credit_features;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Verify Gold Table

-- COMMAND ----------

SELECT COUNT(*) AS total_records FROM gold_credit_features;

-- COMMAND ----------

SELECT * FROM gold_credit_features LIMIT 10;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Feature Summary Statistics

-- COMMAND ----------

SELECT
    ROUND(AVG(credit_limit), 2) AS avg_credit_limit,
    ROUND(AVG(age), 1) AS avg_age,
    ROUND(AVG(credit_utilization), 3) AS avg_utilization,
    ROUND(AVG(total_risk_score), 2) AS avg_risk_score,
    SUM(default_payment) AS total_defaults,
    COUNT(*) AS total_records,
    ROUND(100.0 * SUM(default_payment) / COUNT(*), 2) AS default_rate_pct
FROM gold_credit_features;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Risk Score vs Default Rate

-- COMMAND ----------

SELECT
    CASE
        WHEN total_risk_score <= 4 THEN '1. Low Risk (0-4)'
        WHEN total_risk_score <= 8 THEN '2. Medium Risk (5-8)'
        ELSE '3. High Risk (9+)'
    END AS risk_category,
    COUNT(*) AS customer_count,
    SUM(default_payment) AS defaults,
    ROUND(100.0 * SUM(default_payment) / COUNT(*), 1) AS default_rate_pct
FROM gold_credit_features
GROUP BY 1
ORDER BY 1;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Default Rate by Demographics

-- COMMAND ----------

-- By education
SELECT
    education,
    COUNT(*) AS customer_count,
    ROUND(100.0 * SUM(default_payment) / COUNT(*), 1) AS default_rate_pct
FROM gold_credit_features
GROUP BY education
ORDER BY default_rate_pct DESC;

-- COMMAND ----------

-- By age group
SELECT
    age_group,
    COUNT(*) AS customer_count,
    ROUND(100.0 * SUM(default_payment) / COUNT(*), 1) AS default_rate_pct
FROM gold_credit_features
GROUP BY age_group
ORDER BY default_rate_pct DESC;

-- COMMAND ----------

-- By credit bucket
SELECT
    credit_bucket,
    COUNT(*) AS customer_count,
    ROUND(AVG(credit_limit), 0) AS avg_credit,
    ROUND(100.0 * SUM(default_payment) / COUNT(*), 1) AS default_rate_pct
FROM gold_credit_features
GROUP BY credit_bucket
ORDER BY avg_credit;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Summary
-- MAGIC
-- MAGIC - Gold feature table created with 30,000 records
-- MAGIC - Risk scores calculated from payment history and demographics
-- MAGIC - Feature engineering complete
-- MAGIC
-- MAGIC **Next:** Run `04_risk_analysis`
