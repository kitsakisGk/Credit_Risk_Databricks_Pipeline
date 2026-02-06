"""Feature engineering for scoring API - mirrors Gold layer logic."""

import numpy as np
import pandas as pd


def engineer_features(data: dict) -> dict:
    """Transform raw application data into ML features.

    Mirrors the logic from notebooks/python/02_gold_features.py
    """
    bill_cols = [data[f"bill_amt_{i}"] for i in range(1, 7)]
    pay_cols = [data[f"pay_amt_{i}"] for i in range(1, 7)]
    pay_status = [data[f"pay_status_{i}"] for i in range(1, 7)]

    # Average amounts
    avg_bill = np.mean(bill_cols)
    avg_payment = np.mean(pay_cols)

    # Credit utilization
    credit_limit = data["credit_limit"]
    credit_utilization = data["bill_amt_1"] / credit_limit if credit_limit > 0 else 0

    # Log credit limit
    log_credit_limit = np.log(credit_limit + 1)

    # Payment ratio
    payment_ratio = data["pay_amt_1"] / data["bill_amt_1"] if data["bill_amt_1"] > 0 else 1

    # Pays full balance
    pays_full = 1 if data["pay_amt_1"] >= data["bill_amt_1"] else 0

    # Young borrower
    is_young = 1 if data["age"] < 30 else 0

    # Delay risk scores
    def delay_risk(status):
        if status <= 0: return 0
        elif status == 1: return 1
        elif status == 2: return 2
        else: return 3

    dr1 = delay_risk(data["pay_status_1"])
    dr2 = delay_risk(data["pay_status_2"])
    dr3 = delay_risk(data["pay_status_3"])

    # Education risk
    edu_risk_map = {1: 0, 2: 1, 3: 2, 4: 3}
    education_risk = edu_risk_map.get(data["education"], 3)

    # Marital risk
    marital_risk_map = {1: 0, 2: 1, 3: 2}
    marital_risk = marital_risk_map.get(data["marriage"], 2)

    # Utilization risk
    if credit_utilization < 0.3: util_risk = 0
    elif credit_utilization < 0.5: util_risk = 1
    elif credit_utilization < 0.8: util_risk = 2
    else: util_risk = 3

    # Total risk score
    total_risk = dr1 + dr2 + dr3 + education_risk + marital_risk + util_risk

    # Months delayed and max delay
    months_delayed = sum(1 for s in pay_status if s > 0)
    max_delay = max(pay_status)

    # Totals
    total_bill = sum(bill_cols)
    total_pay = sum(pay_cols)

    return {
        "credit_limit": credit_limit,
        "age": data["age"],
        "bill_amt_1": data["bill_amt_1"], "bill_amt_2": data["bill_amt_2"],
        "bill_amt_3": data["bill_amt_3"], "bill_amt_4": data["bill_amt_4"],
        "bill_amt_5": data["bill_amt_5"], "bill_amt_6": data["bill_amt_6"],
        "pay_amt_1": data["pay_amt_1"], "pay_amt_2": data["pay_amt_2"],
        "pay_amt_3": data["pay_amt_3"], "pay_amt_4": data["pay_amt_4"],
        "pay_amt_5": data["pay_amt_5"], "pay_amt_6": data["pay_amt_6"],
        "pay_status_1": data["pay_status_1"], "pay_status_2": data["pay_status_2"],
        "pay_status_3": data["pay_status_3"], "pay_status_4": data["pay_status_4"],
        "pay_status_5": data["pay_status_5"], "pay_status_6": data["pay_status_6"],
        "avg_bill_amount": avg_bill,
        "avg_payment_amount": avg_payment,
        "credit_utilization": credit_utilization,
        "log_credit_limit": log_credit_limit,
        "payment_ratio": payment_ratio,
        "pays_full_balance": pays_full,
        "is_young_borrower": is_young,
        "delay_risk_1": dr1, "delay_risk_2": dr2, "delay_risk_3": dr3,
        "education_risk": education_risk,
        "marital_risk": marital_risk,
        "utilization_risk": util_risk,
        "total_risk_score": total_risk,
        "months_delayed": months_delayed,
        "max_delay_months": max_delay,
        "total_bill_amt": total_bill,
        "total_pay_amt": total_pay
    }
