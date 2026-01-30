"""
Unit Tests for Credit Risk Pipeline Transformations.

These tests can be run locally with pytest or in Databricks notebooks.
For Databricks, use the Nutter framework or databricks-connect.
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing."""
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("CreditRiskPipelineTests")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )


@pytest.fixture
def sample_data(spark):
    """Create sample test data."""
    schema = StructType([
        StructField("checking_status", StringType(), True),
        StructField("duration_months", IntegerType(), True),
        StructField("credit_history", StringType(), True),
        StructField("purpose", StringType(), True),
        StructField("credit_amount", DoubleType(), True),
        StructField("savings_status", StringType(), True),
        StructField("employment_duration", StringType(), True),
        StructField("installment_rate", IntegerType(), True),
        StructField("personal_status", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("existing_credits", IntegerType(), True),
        StructField("num_dependents", IntegerType(), True),
        StructField("residence_duration", IntegerType(), True),
        StructField("credit_risk", IntegerType(), True),
    ])

    data = [
        ("A11", 12, "A32", "A43", 5000.0, "A61", "A73", 2, "A93", 35, 1, 1, 2, 0),
        ("A12", 24, "A34", "A40", 10000.0, "A62", "A74", 3, "A92", 45, 2, 2, 3, 1),
        ("A13", 36, "A31", "A41", 15000.0, "A63", "A75", 4, "A91", 55, 3, 1, 4, 0),
        ("A14", 6, "A33", "A42", 2000.0, "A65", "A71", 1, "A94", 25, 1, 0, 1, 1),
    ]

    return spark.createDataFrame(data, schema)


class TestDataQuality:
    """Tests for data quality checks."""

    def test_null_check_passes_complete_data(self, sample_data):
        """Test that null check passes for complete data."""
        from src.utils.data_quality import DataQualityValidator

        validator = DataQualityValidator(sample_data)
        result = validator.check_nulls(["age", "credit_amount"])
        assert result is True

    def test_range_check_valid_age(self, sample_data):
        """Test age range validation."""
        from src.utils.data_quality import DataQualityValidator

        validator = DataQualityValidator(sample_data)
        result = validator.check_range("age", min_val=18, max_val=100)
        assert result is True

    def test_range_check_invalid_values(self, spark):
        """Test range check fails for invalid values."""
        from src.utils.data_quality import DataQualityValidator

        # Create data with invalid age
        data = [(150,), (10,), (35,)]
        df = spark.createDataFrame(data, ["age"])

        validator = DataQualityValidator(df)
        result = validator.check_range("age", min_val=18, max_val=100)
        assert result is False

    def test_quality_flags_added(self, sample_data):
        """Test that quality flags are correctly added."""
        from src.utils.data_quality import DataQualityValidator, CREDIT_RISK_CHECKS

        validator = DataQualityValidator(sample_data)
        result_df = validator.add_quality_flags(CREDIT_RISK_CHECKS)

        assert "_dq_passed" in result_df.columns
        assert "_dq_valid_age" in result_df.columns

    def test_split_by_quality(self, sample_data):
        """Test splitting data by quality."""
        from src.utils.data_quality import DataQualityValidator, CREDIT_RISK_CHECKS

        validator = DataQualityValidator(sample_data)
        validator.add_quality_flags(CREDIT_RISK_CHECKS)
        good_df, bad_df = validator.split_by_quality()

        total = sample_data.count()
        assert good_df.count() + bad_df.count() == total


class TestFeatureEngineering:
    """Tests for feature engineering functions."""

    def test_numerical_features_created(self, sample_data):
        """Test that numerical features are created correctly."""
        from src.utils.feature_engineering import FeatureEngineer

        fe = FeatureEngineer(sample_data)
        result_df = fe.add_numerical_features()

        assert "credit_per_month" in result_df.columns
        assert "log_credit_amount" in result_df.columns
        assert "stability_score" in result_df.columns

    def test_credit_per_month_calculation(self, sample_data):
        """Test credit_per_month is calculated correctly."""
        from src.utils.feature_engineering import FeatureEngineer
        from pyspark.sql.functions import col

        fe = FeatureEngineer(sample_data)
        result_df = fe.add_numerical_features()

        # First row: 5000 / 12 = 416.67
        first_row = result_df.filter(col("credit_amount") == 5000.0).first()
        expected = 5000.0 / 12
        assert abs(first_row["credit_per_month"] - expected) < 0.01

    def test_risk_scores_created(self, sample_data):
        """Test that risk scores are created."""
        from src.utils.feature_engineering import FeatureEngineer

        fe = FeatureEngineer(sample_data)
        result_df = fe.add_risk_scores()

        assert "checking_risk_score" in result_df.columns
        assert "savings_risk_score" in result_df.columns
        assert "combined_risk_score" in result_df.columns

    def test_checking_risk_score_values(self, sample_data):
        """Test checking risk scores are assigned correctly."""
        from src.utils.feature_engineering import FeatureEngineer
        from pyspark.sql.functions import col

        fe = FeatureEngineer(sample_data)
        result_df = fe.add_risk_scores()

        # A11 should have risk score 3 (negative balance)
        a11_row = result_df.filter(col("checking_status") == "A11").first()
        assert a11_row["checking_risk_score"] == 3

        # A14 should have risk score 0 (no account)
        a14_row = result_df.filter(col("checking_status") == "A14").first()
        assert a14_row["checking_risk_score"] == 0

    def test_age_features_created(self, sample_data):
        """Test age-based features are created."""
        from src.utils.feature_engineering import FeatureEngineer

        fe = FeatureEngineer(sample_data)
        result_df = fe.add_age_features()

        assert "age_group" in result_df.columns
        assert "is_young_borrower" in result_df.columns
        assert "is_senior_borrower" in result_df.columns

    def test_age_group_assignment(self, sample_data):
        """Test age groups are assigned correctly."""
        from src.utils.feature_engineering import FeatureEngineer
        from pyspark.sql.functions import col

        fe = FeatureEngineer(sample_data)
        result_df = fe.add_age_features()

        # Age 25 should be "young"
        young_row = result_df.filter(col("age") == 25).first()
        assert young_row["age_group"] == "young"

        # Age 55 should be "senior"
        senior_row = result_df.filter(col("age") == 55).first()
        assert senior_row["age_group"] == "senior"

    def test_categorical_encoding(self, sample_data):
        """Test categorical encoding works."""
        from src.utils.feature_engineering import FeatureEngineer

        fe = FeatureEngineer(sample_data)
        result_df = fe.encode_categoricals(
            ["checking_status"],
            method="label"
        )

        assert "checking_status_encoded" in result_df.columns
        # Encoded values should be non-negative integers
        encoded_values = [
            row["checking_status_encoded"]
            for row in result_df.select("checking_status_encoded").collect()
        ]
        assert all(v >= 0 for v in encoded_values)

    def test_interaction_features(self, sample_data):
        """Test interaction features are created."""
        from src.utils.feature_engineering import FeatureEngineer

        fe = FeatureEngineer(sample_data)
        fe.add_numerical_features()
        fe.add_risk_scores()
        result_df = fe.add_interaction_features()

        assert "age_x_credit" in result_df.columns
        assert "checking_x_savings" in result_df.columns

    def test_row_hash_uniqueness(self, sample_data):
        """Test row hash creates unique values."""
        from src.utils.feature_engineering import FeatureEngineer

        fe = FeatureEngineer(sample_data)
        result_df = fe.create_row_hash(["checking_status", "credit_amount", "age"])

        assert "_row_hash" in result_df.columns
        # All hashes should be unique for our test data
        unique_hashes = result_df.select("_row_hash").distinct().count()
        assert unique_hashes == sample_data.count()


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_feature_pipeline(self, sample_data):
        """Test full feature engineering pipeline."""
        from src.utils.feature_engineering import create_ml_features

        result_df = create_ml_features(sample_data)

        # Check key features exist
        expected_features = [
            "credit_per_month",
            "combined_risk_score",
            "age_group",
            "is_young_borrower"
        ]

        for feature in expected_features:
            assert feature in result_df.columns, f"Missing feature: {feature}"

    def test_no_nulls_in_derived_features(self, sample_data):
        """Test that derived features don't introduce nulls."""
        from src.utils.feature_engineering import create_ml_features
        from pyspark.sql.functions import col, isnan, when, count

        result_df = create_ml_features(sample_data)

        # Check numerical derived features for nulls
        derived_cols = [
            "credit_per_month",
            "log_credit_amount",
            "combined_risk_score"
        ]

        for col_name in derived_cols:
            null_count = result_df.filter(col(col_name).isNull()).count()
            assert null_count == 0, f"Found nulls in {col_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
