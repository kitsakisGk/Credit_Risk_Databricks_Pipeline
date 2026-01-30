"""
Feature Engineering Module for Credit Risk Pipeline.

This module provides reusable feature transformation functions
for the credit risk ML pipeline.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, lit, log, sqrt, pow as spark_pow,
    datediff, current_date, months_between,
    concat_ws, sha2, dense_rank, percent_rank
)
from pyspark.sql.window import Window
from typing import List, Dict, Optional


class FeatureEngineer:
    """
    Feature engineering utilities for credit risk modeling.

    Usage:
        fe = FeatureEngineer(df)
        df = fe.add_numerical_features()
        df = fe.add_risk_scores()
        df = fe.encode_categoricals()
    """

    def __init__(self, df: DataFrame):
        self.df = df

    def add_numerical_features(self) -> DataFrame:
        """
        Add derived numerical features.

        Features created:
            - credit_per_month: Monthly payment proxy
            - credit_income_proxy: Credit relative to installment
            - age_credit_ratio: Age normalized by credit
            - log_credit_amount: Log-transformed credit
            - stability_score: Residence and credit stability
        """
        self.df = (
            self.df
            # Payment-related features
            .withColumn(
                "credit_per_month",
                col("credit_amount") / col("duration_months")
            )
            .withColumn(
                "credit_income_proxy",
                col("credit_amount") / (col("installment_rate") + 1)
            )

            # Age-related features
            .withColumn(
                "age_credit_ratio",
                col("age") / (col("credit_amount") / 1000)
            )

            # Transformations
            .withColumn(
                "log_credit_amount",
                log(col("credit_amount") + 1)
            )

            # Stability metrics
            .withColumn(
                "stability_score",
                (col("residence_duration") + col("existing_credits")) / 2
            )
            .withColumn(
                "dependency_ratio",
                col("num_dependents") / (col("existing_credits") + 1)
            )
        )

        return self.df

    def add_age_features(self) -> DataFrame:
        """
        Add age-based categorical and binary features.
        """
        self.df = (
            self.df
            .withColumn(
                "age_group",
                when(col("age") < 25, "young")
                .when(col("age") < 35, "young_adult")
                .when(col("age") < 50, "middle_aged")
                .when(col("age") < 65, "senior")
                .otherwise("elderly")
            )
            .withColumn(
                "is_young_borrower",
                when(col("age") < 30, 1).otherwise(0)
            )
            .withColumn(
                "is_senior_borrower",
                when(col("age") >= 55, 1).otherwise(0)
            )
        )

        return self.df

    def add_credit_features(self) -> DataFrame:
        """
        Add credit amount categorical features.
        """
        self.df = (
            self.df
            .withColumn(
                "credit_amount_bucket",
                when(col("credit_amount") < 1000, "very_low")
                .when(col("credit_amount") < 2500, "low")
                .when(col("credit_amount") < 5000, "medium")
                .when(col("credit_amount") < 10000, "high")
                .otherwise("very_high")
            )
            .withColumn(
                "is_short_term",
                when(col("duration_months") <= 12, 1).otherwise(0)
            )
            .withColumn(
                "is_long_term",
                when(col("duration_months") > 36, 1).otherwise(0)
            )
        )

        return self.df

    def add_risk_scores(self) -> DataFrame:
        """
        Add risk score features based on categorical variables.

        These are domain-knowledge based risk indicators derived
        from the German Credit dataset documentation.
        """
        self.df = (
            self.df
            # Checking account risk (most important predictor)
            .withColumn(
                "checking_risk_score",
                when(col("checking_status") == "A14", 0)   # no account
                .when(col("checking_status") == "A11", 3)  # negative
                .when(col("checking_status") == "A12", 2)  # 0-200
                .when(col("checking_status") == "A13", 1)  # 200+
                .otherwise(2)
            )

            # Savings risk
            .withColumn(
                "savings_risk_score",
                when(col("savings_status") == "A65", 4)  # unknown/none
                .when(col("savings_status") == "A61", 3)  # < 100
                .when(col("savings_status") == "A62", 2)  # 100-500
                .when(col("savings_status") == "A63", 1)  # 500-1000
                .when(col("savings_status") == "A64", 0)  # >= 1000
                .otherwise(2)
            )

            # Employment stability risk
            .withColumn(
                "employment_risk_score",
                when(col("employment_duration") == "A71", 4)  # unemployed
                .when(col("employment_duration") == "A72", 3)  # < 1 year
                .when(col("employment_duration") == "A73", 2)  # 1-4 years
                .when(col("employment_duration") == "A74", 1)  # 4-7 years
                .when(col("employment_duration") == "A75", 0)  # >= 7 years
                .otherwise(2)
            )

            # Credit history risk
            .withColumn(
                "history_risk_score",
                when(col("credit_history") == "A34", 4)  # critical
                .when(col("credit_history") == "A33", 3)  # delay
                .when(col("credit_history") == "A32", 1)  # existing paid
                .when(col("credit_history") == "A31", 0)  # all paid
                .when(col("credit_history") == "A30", 0)  # no credits
                .otherwise(2)
            )

            # Combined risk score
            .withColumn(
                "combined_risk_score",
                col("checking_risk_score") +
                col("savings_risk_score") +
                col("employment_risk_score") +
                col("history_risk_score")
            )
        )

        return self.df

    def add_interaction_features(self) -> DataFrame:
        """
        Add interaction features between important variables.
        """
        self.df = (
            self.df
            # Numeric interactions
            .withColumn(
                "age_x_credit",
                col("age") * col("log_credit_amount")
            )

            # Risk score interactions
            .withColumn(
                "checking_x_savings",
                col("checking_risk_score") * col("savings_risk_score")
            )
            .withColumn(
                "employment_x_history",
                col("employment_risk_score") * col("history_risk_score")
            )

            # Financial stress indicator
            .withColumn(
                "financial_stress_indicator",
                col("credit_per_month") * col("combined_risk_score")
            )
        )

        return self.df

    def encode_categoricals(
        self,
        categorical_columns: List[str],
        method: str = "label"
    ) -> DataFrame:
        """
        Encode categorical variables.

        Args:
            categorical_columns: List of columns to encode
            method: Encoding method ('label' or 'frequency')

        Returns:
            DataFrame with encoded columns
        """
        for col_name in categorical_columns:
            if method == "label":
                # Label encoding using dense_rank
                window = Window.orderBy(col(col_name))
                self.df = self.df.withColumn(
                    f"{col_name}_encoded",
                    dense_rank().over(window) - 1
                )
            elif method == "frequency":
                # Frequency encoding
                freq_df = (
                    self.df
                    .groupBy(col_name)
                    .count()
                    .withColumnRenamed("count", f"{col_name}_freq")
                )
                total = self.df.count()
                freq_df = freq_df.withColumn(
                    f"{col_name}_encoded",
                    col(f"{col_name}_freq") / lit(total)
                )
                self.df = self.df.join(
                    freq_df.select(col_name, f"{col_name}_encoded"),
                    on=col_name,
                    how="left"
                )

        return self.df

    def decode_german_credit(self) -> DataFrame:
        """
        Decode German Credit dataset categorical codes to readable values.
        """
        self.df = (
            self.df
            # Checking account
            .withColumn(
                "checking_status_decoded",
                when(col("checking_status") == "A11", "< 0 DM")
                .when(col("checking_status") == "A12", "0-200 DM")
                .when(col("checking_status") == "A13", ">= 200 DM")
                .when(col("checking_status") == "A14", "no checking")
                .otherwise("unknown")
            )

            # Purpose
            .withColumn(
                "purpose_decoded",
                when(col("purpose") == "A40", "new car")
                .when(col("purpose") == "A41", "used car")
                .when(col("purpose") == "A42", "furniture")
                .when(col("purpose") == "A43", "radio/TV")
                .when(col("purpose") == "A44", "appliances")
                .when(col("purpose") == "A45", "repairs")
                .when(col("purpose") == "A46", "education")
                .when(col("purpose") == "A47", "vacation")
                .when(col("purpose") == "A48", "retraining")
                .when(col("purpose") == "A49", "business")
                .when(col("purpose") == "A410", "others")
                .otherwise("unknown")
            )

            # Gender extraction
            .withColumn(
                "gender",
                when(col("personal_status").isin(["A91", "A93", "A94"]), "male")
                .when(col("personal_status") == "A92", "female")
                .otherwise("unknown")
            )
        )

        return self.df

    def create_row_hash(self, columns: List[str]) -> DataFrame:
        """
        Create a hash of specified columns for deduplication.
        """
        self.df = self.df.withColumn(
            "_row_hash",
            sha2(concat_ws("||", *[col(c) for c in columns]), 256)
        )
        return self.df

    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        Get lists of feature names by category.
        """
        return {
            "numerical": [
                "duration_months", "credit_amount", "installment_rate",
                "residence_duration", "age", "existing_credits", "num_dependents",
                "credit_per_month", "credit_income_proxy", "age_credit_ratio",
                "log_credit_amount", "stability_score", "dependency_ratio"
            ],
            "risk_scores": [
                "checking_risk_score", "savings_risk_score",
                "employment_risk_score", "history_risk_score",
                "combined_risk_score"
            ],
            "interactions": [
                "age_x_credit", "checking_x_savings",
                "employment_x_history", "financial_stress_indicator"
            ],
            "binary": [
                "is_young_borrower", "is_senior_borrower",
                "is_short_term", "is_long_term"
            ],
            "categorical_encoded": [
                "checking_status_encoded", "credit_history_encoded",
                "purpose_encoded", "savings_status_encoded",
                "employment_duration_encoded", "personal_status_encoded",
                "other_parties_encoded", "property_magnitude_encoded",
                "other_payment_plans_encoded", "housing_encoded", "job_encoded"
            ]
        }


def create_ml_features(df: DataFrame) -> DataFrame:
    """
    Convenience function to create all ML features in one call.

    Args:
        df: Input DataFrame with raw features

    Returns:
        DataFrame with all engineered features
    """
    fe = FeatureEngineer(df)

    return (
        fe.df
        .transform(lambda d: FeatureEngineer(d).add_numerical_features())
        .transform(lambda d: FeatureEngineer(d).add_age_features())
        .transform(lambda d: FeatureEngineer(d).add_credit_features())
        .transform(lambda d: FeatureEngineer(d).add_risk_scores())
        .transform(lambda d: FeatureEngineer(d).add_interaction_features())
    )
