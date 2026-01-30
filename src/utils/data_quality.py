"""
Data Quality Module for Credit Risk Pipeline.

This module provides data quality validation functions that can be
used across the pipeline for consistent quality checks.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, sum as spark_sum, count, lit,
    isnan, isnull
)
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Container for data quality metrics."""
    total_records: int
    valid_records: int
    invalid_records: int
    completeness: float
    validity: float
    checks_passed: Dict[str, bool]
    details: Dict[str, any]


class DataQualityValidator:
    """
    Data quality validator for the credit risk pipeline.

    Usage:
        validator = DataQualityValidator(df)
        metrics = validator.run_all_checks()
        good_df, bad_df = validator.split_by_quality()
    """

    def __init__(self, df: DataFrame):
        self.df = df
        self.checks_results = {}
        self.quality_column = "_dq_passed"

    def check_nulls(self, columns: List[str], threshold: float = 0.95) -> bool:
        """
        Check that specified columns have completeness above threshold.

        Args:
            columns: List of column names to check
            threshold: Minimum completeness ratio (default 0.95 = 95%)

        Returns:
            True if all columns pass, False otherwise
        """
        total = self.df.count()
        results = {}

        for col_name in columns:
            non_null_count = self.df.filter(
                col(col_name).isNotNull() &
                (~isnan(col(col_name)) if str(self.df.schema[col_name].dataType) in ['DoubleType', 'FloatType'] else lit(True))
            ).count()

            completeness = non_null_count / total if total > 0 else 0
            results[col_name] = {
                "completeness": completeness,
                "passed": completeness >= threshold
            }

        self.checks_results["null_check"] = results
        return all(r["passed"] for r in results.values())

    def check_range(
        self,
        column: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> bool:
        """
        Check that values in a column fall within specified range.

        Args:
            column: Column name to check
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)

        Returns:
            True if all non-null values are within range
        """
        conditions = []

        if min_val is not None:
            conditions.append(col(column) >= min_val)
        if max_val is not None:
            conditions.append(col(column) <= max_val)

        if not conditions:
            return True

        combined_condition = conditions[0]
        for cond in conditions[1:]:
            combined_condition = combined_condition & cond

        total = self.df.filter(col(column).isNotNull()).count()
        valid = self.df.filter(col(column).isNotNull() & combined_condition).count()

        validity = valid / total if total > 0 else 1.0

        self.checks_results[f"range_check_{column}"] = {
            "total": total,
            "valid": valid,
            "validity": validity,
            "min": min_val,
            "max": max_val,
            "passed": validity == 1.0
        }

        return validity == 1.0

    def check_allowed_values(
        self,
        column: str,
        allowed_values: List[str]
    ) -> bool:
        """
        Check that categorical column only contains allowed values.

        Args:
            column: Column name to check
            allowed_values: List of allowed values

        Returns:
            True if all values are in allowed list
        """
        total = self.df.filter(col(column).isNotNull()).count()
        valid = self.df.filter(col(column).isin(allowed_values)).count()

        validity = valid / total if total > 0 else 1.0

        self.checks_results[f"allowed_values_{column}"] = {
            "total": total,
            "valid": valid,
            "validity": validity,
            "allowed": allowed_values,
            "passed": validity == 1.0
        }

        return validity == 1.0

    def check_uniqueness(
        self,
        columns: List[str],
        threshold: float = 1.0
    ) -> bool:
        """
        Check uniqueness of values across specified columns.

        Args:
            columns: Columns to check for uniqueness
            threshold: Minimum uniqueness ratio (1.0 = no duplicates)

        Returns:
            True if uniqueness is above threshold
        """
        total = self.df.count()
        distinct = self.df.select(columns).distinct().count()

        uniqueness = distinct / total if total > 0 else 1.0

        self.checks_results["uniqueness_check"] = {
            "total": total,
            "distinct": distinct,
            "uniqueness": uniqueness,
            "columns": columns,
            "passed": uniqueness >= threshold
        }

        return uniqueness >= threshold

    def add_quality_flags(self, checks: Dict[str, any]) -> DataFrame:
        """
        Add data quality flag columns to the dataframe.

        Args:
            checks: Dictionary of check definitions
                Example:
                {
                    "valid_age": {"column": "age", "min": 18, "max": 100},
                    "valid_amount": {"column": "credit_amount", "min": 0}
                }

        Returns:
            DataFrame with quality flag columns added
        """
        result_df = self.df

        for check_name, check_def in checks.items():
            col_name = check_def["column"]
            conditions = []

            if "min" in check_def:
                conditions.append(col(col_name) >= check_def["min"])
            if "max" in check_def:
                conditions.append(col(col_name) <= check_def["max"])
            if "not_null" in check_def and check_def["not_null"]:
                conditions.append(col(col_name).isNotNull())
            if "allowed_values" in check_def:
                conditions.append(col(col_name).isin(check_def["allowed_values"]))

            if conditions:
                combined = conditions[0]
                for cond in conditions[1:]:
                    combined = combined & cond
                result_df = result_df.withColumn(f"_dq_{check_name}", combined)
            else:
                result_df = result_df.withColumn(f"_dq_{check_name}", lit(True))

        # Add overall quality flag
        dq_columns = [c for c in result_df.columns if c.startswith("_dq_")]
        if dq_columns:
            overall_condition = col(dq_columns[0])
            for dq_col in dq_columns[1:]:
                overall_condition = overall_condition & col(dq_col)
            result_df = result_df.withColumn(self.quality_column, overall_condition)

        self.df = result_df
        return result_df

    def split_by_quality(self) -> Tuple[DataFrame, DataFrame]:
        """
        Split dataframe into good and bad records based on quality flags.

        Returns:
            Tuple of (good_records_df, bad_records_df)
        """
        if self.quality_column not in self.df.columns:
            raise ValueError("Run add_quality_flags first")

        good_df = self.df.filter(col(self.quality_column) == True)
        bad_df = self.df.filter(col(self.quality_column) == False)

        return good_df, bad_df

    def get_quality_report(self) -> QualityMetrics:
        """
        Generate comprehensive quality report.

        Returns:
            QualityMetrics object with all check results
        """
        if self.quality_column not in self.df.columns:
            raise ValueError("Run add_quality_flags first")

        total = self.df.count()
        valid = self.df.filter(col(self.quality_column) == True).count()
        invalid = total - valid

        return QualityMetrics(
            total_records=total,
            valid_records=valid,
            invalid_records=invalid,
            completeness=valid / total if total > 0 else 0,
            validity=valid / total if total > 0 else 0,
            checks_passed={k: v.get("passed", False) for k, v in self.checks_results.items()},
            details=self.checks_results
        )


# Credit Risk specific quality checks
CREDIT_RISK_CHECKS = {
    "valid_age": {
        "column": "age",
        "min": 18,
        "max": 100,
        "not_null": True
    },
    "valid_credit_amount": {
        "column": "credit_amount",
        "min": 0,
        "not_null": True
    },
    "valid_duration": {
        "column": "duration_months",
        "min": 1,
        "max": 120,
        "not_null": True
    },
    "valid_target": {
        "column": "credit_risk",
        "allowed_values": [0, 1],
        "not_null": True
    }
}
