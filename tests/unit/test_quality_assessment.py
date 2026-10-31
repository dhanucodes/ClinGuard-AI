
import pandas as pd
import pytest

from clinguard.preprocessing.quality_assessment import (
    assess_data_quality,
)


def create_sample_dataset():

    return pd.DataFrame({
        "has_missing_value": [True, False, False],
        "has_noise": [False, True, False],
        "has_sensor_error": [False, False, True],
        "has_irregular_timestamp": [False, False, False],
        "data_quality_issue": [True, True, True],
        "heart_rate_was_missing": [True, False, False],
    })


def test_total_record_count():

    report = assess_data_quality(create_sample_dataset())

    assert report["total_records"] == 3


def test_affected_record_count():

    report = assess_data_quality(create_sample_dataset())

    assert report["affected_records"] == 3


def test_unaffected_record_count():

    report = assess_data_quality(create_sample_dataset())

    assert report["unaffected_records"] == 0


def test_quality_issue_counts():

    report = assess_data_quality(create_sample_dataset())

    assert report["issue_counts"]["has_noise"]["count"] == 1
    assert report["issue_counts"]["has_sensor_error"]["count"] == 1


def test_original_missing_readings():

    report = assess_data_quality(create_sample_dataset())

    assert report["original_missing_counts"]["heart_rate"] == 1


def test_missing_required_quality_columns():

    data = create_sample_dataset().drop(
        columns=["has_noise"]
    )

    with pytest.raises(ValueError):
        assess_data_quality(data)


def test_empty_dataset():

    data = create_sample_dataset().iloc[0:0]

    with pytest.raises(ValueError):
        assess_data_quality(data)
