-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Bronze Layer - Data Cleanup
-- MAGIC
-- MAGIC Fix column names from the uploaded Excel file.

-- COMMAND ----------

USE kitsakis_credit_risk;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Check Raw Data

-- COMMAND ----------

SELECT * FROM bronze_credit_applications LIMIT 5;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Create Clean Bronze Table
-- MAGIC
-- MAGIC The uploaded Excel had column names in row 1, so we need to:
-- MAGIC 1. Skip the first row (contains header text)
-- MAGIC 2. Rename columns properly

-- COMMAND ----------

-- Drop if exists and recreate with proper column names
CREATE OR REPLACE TABLE bronze_credit_clean AS
SELECT
    CAST(_c0 AS INT) AS id,
    CAST(X1 AS DOUBLE) AS credit_limit,
    CAST(X2 AS INT) AS sex,
    CAST(X3 AS INT) AS education,
    CAST(X4 AS INT) AS marriage,
    CAST(X5 AS INT) AS age,
    CAST(X6 AS INT) AS pay_status_1,
    CAST(X7 AS INT) AS pay_status_2,
    CAST(X8 AS INT) AS pay_status_3,
    CAST(X9 AS INT) AS pay_status_4,
    CAST(X10 AS INT) AS pay_status_5,
    CAST(X11 AS INT) AS pay_status_6,
    CAST(X12 AS DOUBLE) AS bill_amt_1,
    CAST(X13 AS DOUBLE) AS bill_amt_2,
    CAST(X14 AS DOUBLE) AS bill_amt_3,
    CAST(X15 AS DOUBLE) AS bill_amt_4,
    CAST(X16 AS DOUBLE) AS bill_amt_5,
    CAST(X17 AS DOUBLE) AS bill_amt_6,
    CAST(X18 AS DOUBLE) AS pay_amt_1,
    CAST(X19 AS DOUBLE) AS pay_amt_2,
    CAST(X20 AS DOUBLE) AS pay_amt_3,
    CAST(X21 AS DOUBLE) AS pay_amt_4,
    CAST(X22 AS DOUBLE) AS pay_amt_5,
    CAST(X23 AS DOUBLE) AS pay_amt_6,
    CAST(X24 AS INT) AS default_payment
FROM bronze_credit_applications
WHERE _c0 != 'ID';  -- Skip the header row that got imported as data

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Verify Clean Data

-- COMMAND ----------

SELECT COUNT(*) AS total_records FROM bronze_credit_clean;

-- COMMAND ----------

SELECT * FROM bronze_credit_clean LIMIT 10;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Data Summary

-- COMMAND ----------

SELECT
    MIN(credit_limit) AS min_credit,
    MAX(credit_limit) AS max_credit,
    ROUND(AVG(credit_limit), 2) AS avg_credit,
    MIN(age) AS min_age,
    MAX(age) AS max_age,
    ROUND(AVG(age), 1) AS avg_age,
    SUM(default_payment) AS total_defaults,
    COUNT(*) AS total_records,
    ROUND(100.0 * SUM(default_payment) / COUNT(*), 2) AS default_rate_pct
FROM bronze_credit_clean;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Summary
-- MAGIC
-- MAGIC - Bronze table cleaned with proper column names
-- MAGIC - 30,000 credit card customer records
-- MAGIC - Ready for Silver transformation
-- MAGIC
-- MAGIC **Next:** Run `02_silver_transformation`
