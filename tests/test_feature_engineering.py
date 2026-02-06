"""Unit tests for feature engineering logic."""

import pytest
import numpy as np
from api.feature_engine import engineer_features


def make_application(**overrides):
    """Create a default test application with optional overrides."""
    defaults = {
        "credit_limit": 200000, "age": 35, "sex": 1, "education": 2, "marriage": 1,
        "pay_status_1": 0, "pay_status_2": 0, "pay_status_3": 0,
        "pay_status_4": 0, "pay_status_5": 0, "pay_status_6": 0,
        "bill_amt_1": 50000, "bill_amt_2": 45000, "bill_amt_3": 40000,
        "bill_amt_4": 35000, "bill_amt_5": 30000, "bill_amt_6": 25000,
        "pay_amt_1": 5000, "pay_amt_2": 4500, "pay_amt_3": 4000,
        "pay_amt_4": 3500, "pay_amt_5": 3000, "pay_amt_6": 2500,
    }
    defaults.update(overrides)
    return defaults


class TestCreditUtilization:
    def test_normal_utilization(self):
        features = engineer_features(make_application(credit_limit=100000, bill_amt_1=50000))
        assert features["credit_utilization"] == 0.5

    def test_zero_credit_limit(self):
        features = engineer_features(make_application(credit_limit=0))
        assert features["credit_utilization"] == 0

    def test_high_utilization(self):
        features = engineer_features(make_application(credit_limit=100000, bill_amt_1=120000))
        assert features["credit_utilization"] == 1.2


class TestPaymentRatio:
    def test_partial_payment(self):
        features = engineer_features(make_application(bill_amt_1=10000, pay_amt_1=5000))
        assert features["payment_ratio"] == 0.5

    def test_zero_bill(self):
        features = engineer_features(make_application(bill_amt_1=0, pay_amt_1=5000))
        assert features["payment_ratio"] == 1

    def test_full_payment(self):
        features = engineer_features(make_application(bill_amt_1=10000, pay_amt_1=10000))
        assert features["payment_ratio"] == 1.0
        assert features["pays_full_balance"] == 1

    def test_no_payment(self):
        features = engineer_features(make_application(bill_amt_1=10000, pay_amt_1=0))
        assert features["payment_ratio"] == 0
        assert features["pays_full_balance"] == 0


class TestDelayRisk:
    def test_no_delay(self):
        features = engineer_features(make_application(pay_status_1=-1))
        assert features["delay_risk_1"] == 0

    def test_one_month_delay(self):
        features = engineer_features(make_application(pay_status_1=1))
        assert features["delay_risk_1"] == 1

    def test_two_month_delay(self):
        features = engineer_features(make_application(pay_status_1=2))
        assert features["delay_risk_1"] == 2

    def test_severe_delay(self):
        features = engineer_features(make_application(pay_status_1=5))
        assert features["delay_risk_1"] == 3


class TestRiskScores:
    def test_education_risk_graduate(self):
        features = engineer_features(make_application(education=1))
        assert features["education_risk"] == 0

    def test_education_risk_high_school(self):
        features = engineer_features(make_application(education=3))
        assert features["education_risk"] == 2

    def test_marital_risk_married(self):
        features = engineer_features(make_application(marriage=1))
        assert features["marital_risk"] == 0

    def test_marital_risk_single(self):
        features = engineer_features(make_application(marriage=2))
        assert features["marital_risk"] == 1

    def test_utilization_risk_low(self):
        features = engineer_features(make_application(credit_limit=200000, bill_amt_1=40000))
        assert features["utilization_risk"] == 0

    def test_utilization_risk_high(self):
        features = engineer_features(make_application(credit_limit=100000, bill_amt_1=90000))
        assert features["utilization_risk"] == 3

    def test_total_risk_score_low(self):
        features = engineer_features(make_application(
            pay_status_1=-1, pay_status_2=-1, pay_status_3=-1,
            education=1, marriage=1, credit_limit=200000, bill_amt_1=20000
        ))
        assert features["total_risk_score"] == 0

    def test_total_risk_score_range(self):
        features = engineer_features(make_application())
        assert 0 <= features["total_risk_score"] <= 18


class TestAggregateFeatures:
    def test_months_delayed(self):
        features = engineer_features(make_application(
            pay_status_1=2, pay_status_2=1, pay_status_3=0,
            pay_status_4=-1, pay_status_5=3, pay_status_6=0
        ))
        assert features["months_delayed"] == 3

    def test_max_delay(self):
        features = engineer_features(make_application(
            pay_status_1=1, pay_status_2=3, pay_status_3=2,
            pay_status_4=0, pay_status_5=0, pay_status_6=0
        ))
        assert features["max_delay_months"] == 3

    def test_total_bill(self):
        features = engineer_features(make_application(
            bill_amt_1=10000, bill_amt_2=20000, bill_amt_3=30000,
            bill_amt_4=40000, bill_amt_5=50000, bill_amt_6=60000
        ))
        assert features["total_bill_amt"] == 210000

    def test_young_borrower(self):
        features = engineer_features(make_application(age=25))
        assert features["is_young_borrower"] == 1

    def test_not_young_borrower(self):
        features = engineer_features(make_application(age=35))
        assert features["is_young_borrower"] == 0

    def test_log_credit_limit(self):
        features = engineer_features(make_application(credit_limit=100000))
        assert features["log_credit_limit"] == pytest.approx(np.log(100001), rel=1e-4)


class TestFeatureCompleteness:
    def test_all_features_present(self):
        features = engineer_features(make_application())
        expected = [
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
        for feat in expected:
            assert feat in features, f"Missing feature: {feat}"

    def test_feature_count(self):
        features = engineer_features(make_application())
        assert len(features) == 38
