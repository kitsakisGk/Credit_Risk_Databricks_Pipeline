# Databricks notebook source
# MAGIC %md
# MAGIC # Environment Setup
# MAGIC
# MAGIC This notebook configures the environment for the Credit Risk ML Pipeline.
# MAGIC Run this first to set up databases, paths, and configuration.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Define your username for unique paths (avoids conflicts in shared workspaces)
username = spark.sql("SELECT current_user()").first()[0]
username_prefix = username.split("@")[0].replace(".", "_")

# Project configuration
PROJECT_NAME = "credit_risk_pipeline"
DATABASE_NAME = f"{username_prefix}_{PROJECT_NAME}"

# Storage paths (using DBFS for Community Edition compatibility)
BASE_PATH = f"/FileStore/{username_prefix}/{PROJECT_NAME}"
BRONZE_PATH = f"{BASE_PATH}/bronze"
SILVER_PATH = f"{BASE_PATH}/silver"
GOLD_PATH = f"{BASE_PATH}/gold"
CHECKPOINT_PATH = f"{BASE_PATH}/checkpoints"
LANDING_PATH = f"{BASE_PATH}/landing"  # For streaming source simulation

# MLflow experiment
EXPERIMENT_PATH = f"/Users/{username}/{PROJECT_NAME}_experiment"

print(f"""
╔══════════════════════════════════════════════════════════════╗
║           Credit Risk ML Pipeline - Configuration            ║
╠══════════════════════════════════════════════════════════════╣
║ Database:     {DATABASE_NAME:<45} ║
║ Base Path:    {BASE_PATH:<45} ║
║ MLflow Exp:   {EXPERIMENT_PATH:<45} ║
╚══════════════════════════════════════════════════════════════╝
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Database and Paths

# COMMAND ----------

# Create database
spark.sql(f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}")
spark.sql(f"USE {DATABASE_NAME}")

# Create directories
dbutils.fs.mkdirs(BRONZE_PATH)
dbutils.fs.mkdirs(SILVER_PATH)
dbutils.fs.mkdirs(GOLD_PATH)
dbutils.fs.mkdirs(CHECKPOINT_PATH)
dbutils.fs.mkdirs(LANDING_PATH)

print("✓ Database and directories created successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Sample Dataset
# MAGIC
# MAGIC We'll use the German Credit Risk dataset from UCI Machine Learning Repository.

# COMMAND ----------

import urllib.request
import os

# German Credit dataset URL
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

# Download to DBFS
local_path = "/tmp/german_credit.data"
urllib.request.urlretrieve(DATASET_URL, local_path)

# Move to DBFS landing zone
dbutils.fs.cp(f"file:{local_path}", f"{LANDING_PATH}/german_credit.data")

print(f"✓ Dataset downloaded to {LANDING_PATH}/german_credit.data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Schema Definition
# MAGIC
# MAGIC The German Credit dataset has specific column definitions. Let's define them properly.

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# Column names and descriptions for German Credit dataset
COLUMN_NAMES = [
    "checking_status",           # Status of existing checking account
    "duration_months",           # Duration in months
    "credit_history",            # Credit history
    "purpose",                   # Purpose of loan
    "credit_amount",             # Credit amount
    "savings_status",            # Savings account/bonds status
    "employment_duration",       # Present employment since
    "installment_rate",          # Installment rate in percentage of disposable income
    "personal_status",           # Personal status and sex
    "other_parties",             # Other debtors / guarantors
    "residence_duration",        # Present residence since
    "property_magnitude",        # Property
    "age",                       # Age in years
    "other_payment_plans",       # Other installment plans
    "housing",                   # Housing
    "existing_credits",          # Number of existing credits at this bank
    "job",                       # Job
    "num_dependents",            # Number of people being liable for maintenance
    "own_telephone",             # Telephone
    "foreign_worker",            # Foreign worker
    "credit_risk"                # Target: 1 = Good, 2 = Bad
]

# Schema for raw data (all string initially for flexibility)
RAW_SCHEMA = StructType([
    StructField(col, StringType(), True) for col in COLUMN_NAMES
])

print(f"✓ Schema defined with {len(COLUMN_NAMES)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Store Configuration for Other Notebooks

# COMMAND ----------

# Store config in spark conf for access in other notebooks
spark.conf.set("pipeline.database", DATABASE_NAME)
spark.conf.set("pipeline.bronze_path", BRONZE_PATH)
spark.conf.set("pipeline.silver_path", SILVER_PATH)
spark.conf.set("pipeline.gold_path", GOLD_PATH)
spark.conf.set("pipeline.checkpoint_path", CHECKPOINT_PATH)
spark.conf.set("pipeline.landing_path", LANDING_PATH)
spark.conf.set("pipeline.experiment_path", EXPERIMENT_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Setup

# COMMAND ----------

# Verify database
tables = spark.sql(f"SHOW TABLES IN {DATABASE_NAME}").collect()
print(f"Database '{DATABASE_NAME}' exists with {len(tables)} tables")

# Verify paths
paths_to_check = [BRONZE_PATH, SILVER_PATH, GOLD_PATH, CHECKPOINT_PATH, LANDING_PATH]
for path in paths_to_check:
    try:
        dbutils.fs.ls(path)
        print(f"✓ {path}")
    except Exception as e:
        print(f"✗ {path} - {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Complete!
# MAGIC
# MAGIC You can now run the pipeline notebooks in order:
# MAGIC 1. `01_bronze_ingestion.py` - Ingest raw data
# MAGIC 2. `02_silver_transformation.py` - Clean and validate
# MAGIC 3. `03_gold_aggregation.py` - Create features
# MAGIC 4. `04_ml_training.py` - Train ML model
# MAGIC 5. `05_streaming_simulation.py` - Simulate real-time data
# MAGIC 6. `06_model_inference.py` - Make predictions
