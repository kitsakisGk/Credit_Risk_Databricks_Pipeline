-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Silver Layer - Data Transformation
-- MAGIC
-- MAGIC Clean and transform Bronze data with business logic.

-- COMMAND ----------

USE kitsakis_credit_risk;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Create Silver Table
-- MAGIC
-- MAGIC Transform raw data:
-- MAGIC - Decode categorical variables (sex, education, marriage)
-- MAGIC - Calculate payment behavior features
-- MAGIC - Add data quality flags

-- COMMAND ----------

CREATE OR REPLACE TABLE silver_credit_applications AS
SELECT
    id,
    credit_limit,
    age,

    -- Decode gender
    CASE sex
        WHEN 1 THEN 'male'
        WHEN 2 THEN 'female'
        ELSE 'unknown'
    END AS gender,

    -- Decode education (1=graduate school, 2=university, 3=high school, 4=others)
    CASE
        WHEN education = 1 THEN 'graduate_school'
        WHEN education = 2 THEN 'university'
        WHEN education = 3 THEN 'high_school'
        ELSE 'other'
    END AS education,

    -- Decode marital status (1=married, 2=single, 3=others)
    CASE
        WHEN marriage = 1 THEN 'married'
        WHEN marriage = 2 THEN 'single'
        ELSE 'other'
    END AS marital_status,

    -- Payment status columns (keep as-is for feature engineering)
    pay_status_1,
    pay_status_2,
    pay_status_3,
    pay_status_4,
    pay_status_5,
    pay_status_6,

    -- Bill amounts
    bill_amt_1,
    bill_amt_2,
    bill_amt_3,
    bill_amt_4,
    bill_amt_5,
    bill_amt_6,

    -- Payment amounts
    pay_amt_1,
    pay_amt_2,
    pay_amt_3,
    pay_amt_4,
    pay_amt_5,
    pay_amt_6,

    -- Payment behavior features
    -- Count months with payment delay (pay_status > 0 means delayed)
    (CASE WHEN pay_status_1 > 0 THEN 1 ELSE 0 END +
     CASE WHEN pay_status_2 > 0 THEN 1 ELSE 0 END +
     CASE WHEN pay_status_3 > 0 THEN 1 ELSE 0 END +
     CASE WHEN pay_status_4 > 0 THEN 1 ELSE 0 END +
     CASE WHEN pay_status_5 > 0 THEN 1 ELSE 0 END +
     CASE WHEN pay_status_6 > 0 THEN 1 ELSE 0 END) AS months_delayed,

    -- Maximum delay across all months
    GREATEST(pay_status_1, pay_status_2, pay_status_3,
             pay_status_4, pay_status_5, pay_status_6) AS max_delay_months,

    -- Total bill and payment amounts
    (bill_amt_1 + bill_amt_2 + bill_amt_3 +
     bill_amt_4 + bill_amt_5 + bill_amt_6) AS total_bill_amt,

    (pay_amt_1 + pay_amt_2 + pay_amt_3 +
     pay_amt_4 + pay_amt_5 + pay_amt_6) AS total_pay_amt,

    -- Target variable
    default_payment,

    -- Metadata
    current_timestamp() AS _silver_timestamp

FROM bronze_credit_clean
WHERE credit_limit > 0 AND age > 0;  -- Basic data quality filter

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Verify Silver Table

-- COMMAND ----------

SELECT COUNT(*) AS total_records FROM silver_credit_applications;

-- COMMAND ----------

SELECT * FROM silver_credit_applications LIMIT 10;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Data Quality Report

-- COMMAND ----------

SELECT
    COUNT(*) AS total_records,
    COUNT(DISTINCT id) AS unique_customers,

    -- Gender distribution
    SUM(CASE WHEN gender = 'male' THEN 1 ELSE 0 END) AS male_count,
    SUM(CASE WHEN gender = 'female' THEN 1 ELSE 0 END) AS female_count,

    -- Education distribution
    SUM(CASE WHEN education = 'graduate_school' THEN 1 ELSE 0 END) AS graduate_count,
    SUM(CASE WHEN education = 'university' THEN 1 ELSE 0 END) AS university_count,
    SUM(CASE WHEN education = 'high_school' THEN 1 ELSE 0 END) AS high_school_count,

    -- Default rate
    SUM(default_payment) AS total_defaults,
    ROUND(100.0 * SUM(default_payment) / COUNT(*), 2) AS default_rate_pct

FROM silver_credit_applications;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Payment Behavior Analysis

-- COMMAND ----------

SELECT
    months_delayed,
    COUNT(*) AS customer_count,
    ROUND(100.0 * SUM(default_payment) / COUNT(*), 1) AS default_rate_pct
FROM silver_credit_applications
GROUP BY months_delayed
ORDER BY months_delayed;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Summary
-- MAGIC
-- MAGIC - Silver table created with decoded categorical variables
-- MAGIC - Payment behavior features calculated
-- MAGIC - Data quality validated
-- MAGIC
-- MAGIC **Next:** Run `03_gold_aggregation`
