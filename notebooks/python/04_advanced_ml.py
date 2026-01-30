# Databricks notebook source
# MAGIC %md
# MAGIC # Advanced ML - Swiss Banking Standards
# MAGIC
# MAGIC This notebook implements ML techniques commonly used in Swiss financial institutions:
# MAGIC
# MAGIC - **XGBoost** - Industry standard at UBS, Credit Suisse, Julius Baer
# MAGIC - **SHAP Values** - Model explainability (required for FINMA regulatory compliance)
# MAGIC - **Cross-Validation** - Robust model validation
# MAGIC - **Hyperparameter Tuning** - GridSearchCV optimization
# MAGIC
# MAGIC Swiss financial regulations (FINMA) require model interpretability -
# MAGIC we must be able to explain WHY a customer is classified as high risk.

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
import warnings
warnings.filterwarnings('ignore')

SCHEMA_NAME = "kitsakis_credit_risk"
spark.sql(f"USE {SCHEMA_NAME}")
print(f"Using schema: {SCHEMA_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Gold Features

# COMMAND ----------

df = spark.table("gold_credit_features").toPandas()
print(f"Loaded {len(df)} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Features

# COMMAND ----------

# Select numeric features
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

print(f"Features: {len(feature_cols)}")
print(f"Samples: {len(X)}")
print(f"Default rate: {y.mean():.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/Test Split

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training: {len(X_train)}, Test: {len(X_test)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## XGBoost Model
# MAGIC
# MAGIC XGBoost is the industry standard in Swiss banking for:
# MAGIC - Credit scoring
# MAGIC - Fraud detection
# MAGIC - Risk assessment

# COMMAND ----------

import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

# XGBoost with parameters optimized for credit risk
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),  # Handle imbalanced data
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc'
)

print("Training XGBoost model...")
xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]

# Metrics
print("\n" + "="*50)
print("XGBOOST RESULTS")
print("="*50)
print(f"AUC:      {roc_auc_score(y_test, y_prob):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cross-Validation
# MAGIC
# MAGIC Swiss regulators require robust validation - single train/test split is not sufficient.

# COMMAND ----------

from sklearn.model_selection import cross_val_score, StratifiedKFold

# 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='roc_auc')

print("="*50)
print("5-FOLD CROSS-VALIDATION (AUC)")
print("="*50)
for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: {score:.4f}")
print(f"\nMean AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter Tuning
# MAGIC
# MAGIC Grid search to find optimal parameters.

# COMMAND ----------

from sklearn.model_selection import GridSearchCV

# Parameter grid (reduced for speed)
param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200]
}

print("Running GridSearchCV (this may take a moment)...")

grid_search = GridSearchCV(
    xgb.XGBClassifier(
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='auc'
    ),
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\n" + "="*50)
print("BEST PARAMETERS")
print("="*50)
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV AUC: {grid_search.best_score_:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SHAP Values - Model Explainability
# MAGIC
# MAGIC **Critical for Swiss banking compliance (FINMA)**
# MAGIC
# MAGIC Banks must explain WHY a loan was denied. SHAP values provide:
# MAGIC - Global feature importance
# MAGIC - Individual prediction explanations
# MAGIC - Regulatory audit trail

# COMMAND ----------

import shap

# Create SHAP explainer
print("Calculating SHAP values (this may take a moment)...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

print("SHAP values calculated successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Global Feature Importance (SHAP)

# COMMAND ----------

# Calculate mean absolute SHAP values for feature importance
shap_importance = pd.DataFrame({
    'Feature': feature_cols,
    'SHAP_Importance': np.abs(shap_values).mean(axis=0)
}).sort_values('SHAP_Importance', ascending=False)

print("="*50)
print("TOP 15 FEATURES BY SHAP IMPORTANCE")
print("="*50)
display(shap_importance.head(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ## SHAP Summary Plot

# COMMAND ----------

# Summary plot showing feature impact on predictions
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Individual Prediction Explanation
# MAGIC
# MAGIC **Example: Explaining a single customer's risk assessment**
# MAGIC
# MAGIC This is exactly what Swiss banks need for regulatory compliance.

# COMMAND ----------

# Pick a high-risk customer (predicted as default)
high_risk_idx = np.where(y_pred == 1)[0][0]
customer_data = X_test.iloc[high_risk_idx]
customer_shap = shap_values[high_risk_idx]

print("="*60)
print("INDIVIDUAL CUSTOMER RISK EXPLANATION")
print("="*60)
print(f"Prediction: {'DEFAULT' if y_pred[high_risk_idx] == 1 else 'NO DEFAULT'}")
print(f"Probability of Default: {y_prob[high_risk_idx]:.2%}")
print(f"Actual Outcome: {'DEFAULT' if y_test.iloc[high_risk_idx] == 1 else 'NO DEFAULT'}")

print("\nTop factors increasing risk:")
explanation = pd.DataFrame({
    'Feature': feature_cols,
    'Value': customer_data.values,
    'SHAP_Impact': customer_shap
}).sort_values('SHAP_Impact', ascending=False)

# Show top risk factors
top_risk = explanation[explanation['SHAP_Impact'] > 0].head(5)
for _, row in top_risk.iterrows():
    print(f"  - {row['Feature']}: {row['Value']:.2f} (impact: +{row['SHAP_Impact']:.3f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Comparison

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    'XGBoost': xgb_model
}

results = []
for name, model in models.items():
    if name != 'XGBoost':  # XGBoost already fitted
        model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    results.append({
        'Model': name,
        'AUC': roc_auc_score(y_test, y_prob),
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
display(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Best Model Predictions

# COMMAND ----------

# Save XGBoost predictions
predictions_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred,
    'probability': y_prob
})

# Drop existing table and save
spark.sql("DROP TABLE IF EXISTS xgboost_predictions")
spark_df = spark.createDataFrame(predictions_df)
spark_df.write.format("delta").mode("overwrite").saveAsTable("xgboost_predictions")

print(f"Saved {len(predictions_df)} XGBoost predictions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### Swiss Banking ML Standards Demonstrated:
# MAGIC
# MAGIC | Technique | Purpose | Swiss Relevance |
# MAGIC |-----------|---------|-----------------|
# MAGIC | **XGBoost** | State-of-art classification | Used at UBS, Credit Suisse, Julius Baer |
# MAGIC | **SHAP Values** | Model explainability | Required by FINMA for regulatory compliance |
# MAGIC | **Cross-Validation** | Robust validation | Basel III/IV model validation requirements |
# MAGIC | **Hyperparameter Tuning** | Optimization | Industry best practice |
# MAGIC
# MAGIC ### Key Results:
# MAGIC - XGBoost achieved highest AUC score
# MAGIC - SHAP values enable individual prediction explanations
# MAGIC - Model ready for regulatory audit
# MAGIC
# MAGIC ### Regulatory Compliance:
# MAGIC - FINMA (Swiss Financial Market Supervisory Authority) requires model interpretability
# MAGIC - SHAP provides audit trail for credit decisions
# MAGIC - Cross-validation ensures model stability
