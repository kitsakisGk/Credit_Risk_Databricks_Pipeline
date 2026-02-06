"""
Train XGBoost model and export for API serving.

Run this script to create the model file:
    python api/train_and_export.py

The model will be saved to model/xgboost_model.pkl
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# Load dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "default of credit card clients.xls")

print("Loading dataset...")
df = pd.read_excel(DATA_PATH, header=1)

# Rename columns
column_mapping = {
    'ID': 'id', 'LIMIT_BAL': 'credit_limit', 'SEX': 'sex',
    'EDUCATION': 'education', 'MARRIAGE': 'marriage', 'AGE': 'age',
    'PAY_0': 'pay_status_1', 'PAY_2': 'pay_status_2', 'PAY_3': 'pay_status_3',
    'PAY_4': 'pay_status_4', 'PAY_5': 'pay_status_5', 'PAY_6': 'pay_status_6',
    'BILL_AMT1': 'bill_amt_1', 'BILL_AMT2': 'bill_amt_2', 'BILL_AMT3': 'bill_amt_3',
    'BILL_AMT4': 'bill_amt_4', 'BILL_AMT5': 'bill_amt_5', 'BILL_AMT6': 'bill_amt_6',
    'PAY_AMT1': 'pay_amt_1', 'PAY_AMT2': 'pay_amt_2', 'PAY_AMT3': 'pay_amt_3',
    'PAY_AMT4': 'pay_amt_4', 'PAY_AMT5': 'pay_amt_5', 'PAY_AMT6': 'pay_amt_6',
    'default payment next month': 'default_payment'
}
df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

print(f"Loaded {len(df)} records")

# Feature engineering (mirrors Gold layer)
bill_cols = ['bill_amt_1', 'bill_amt_2', 'bill_amt_3', 'bill_amt_4', 'bill_amt_5', 'bill_amt_6']
pay_cols = ['pay_amt_1', 'pay_amt_2', 'pay_amt_3', 'pay_amt_4', 'pay_amt_5', 'pay_amt_6']
pay_status_cols = ['pay_status_1', 'pay_status_2', 'pay_status_3', 'pay_status_4', 'pay_status_5', 'pay_status_6']

df['avg_bill_amount'] = df[bill_cols].mean(axis=1)
df['avg_payment_amount'] = df[pay_cols].mean(axis=1)
df['credit_utilization'] = np.where(df['credit_limit'] > 0, df['bill_amt_1'] / df['credit_limit'], 0)
df['log_credit_limit'] = np.log(df['credit_limit'] + 1)
df['payment_ratio'] = np.where(df['bill_amt_1'] > 0, df['pay_amt_1'] / df['bill_amt_1'], 1)
df['pays_full_balance'] = (df['pay_amt_1'] >= df['bill_amt_1']).astype(int)
df['is_young_borrower'] = (df['age'] < 30).astype(int)

def delay_risk(status):
    if status <= 0: return 0
    elif status == 1: return 1
    elif status == 2: return 2
    else: return 3

df['delay_risk_1'] = df['pay_status_1'].apply(delay_risk)
df['delay_risk_2'] = df['pay_status_2'].apply(delay_risk)
df['delay_risk_3'] = df['pay_status_3'].apply(delay_risk)

edu_map = {1: 0, 2: 1, 3: 2}
df['education_risk'] = df['education'].map(edu_map).fillna(3).astype(int)

marital_map = {1: 0, 2: 1}
df['marital_risk'] = df['marriage'].map(marital_map).fillna(2).astype(int)

def util_risk(u):
    if u < 0.3: return 0
    elif u < 0.5: return 1
    elif u < 0.8: return 2
    else: return 3

df['utilization_risk'] = df['credit_utilization'].apply(util_risk)
df['total_risk_score'] = (
    df['delay_risk_1'] + df['delay_risk_2'] + df['delay_risk_3'] +
    df['education_risk'] + df['marital_risk'] + df['utilization_risk']
)
df['months_delayed'] = (df[pay_status_cols] > 0).sum(axis=1)
df['max_delay_months'] = df[pay_status_cols].max(axis=1)
df['total_bill_amt'] = df[bill_cols].sum(axis=1)
df['total_pay_amt'] = df[pay_cols].sum(axis=1)

# Prepare features
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
y = df['default_payment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train
print("Training XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    random_state=42, use_label_encoder=False, eval_metric='auc'
)
model.fit(X_train, y_train)

# Evaluate
y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(f"AUC: {auc:.4f}")

# Save model
model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "xgboost_model.pkl")

with open(model_path, "wb") as f:
    pickle.dump({
        "model": model,
        "metadata": {
            "model_type": "XGBoost",
            "training_samples": len(X_train),
            "auc_score": round(auc, 4),
            "n_features": len(feature_cols),
            "feature_names": feature_cols
        }
    }, f)

print(f"Model saved to {model_path}")
print("Run the API with: uvicorn api.main:app --reload")
