
import pandas as pd
import pytest

from clinguard.preprocessing.validation import validate_dataset


def create_sample_dataset():

    return pd.DataFrame({
        "patient_id": ["P001", "P001", "P001"],
        "timestamp": [
            "2026-01-01 10:00:00",
            "2026-01-01 10:05:00",
            "2026-01-01 10:10:00",
        ],
        "heart_rate": [72, 75, 80],
        "spo2": [98, 97, 96],
        "systolic_bp": [118, 120, 122],
        "diastolic_bp": [76, 78, 80],
        "respiratory_rate": [16, 17, 18],
        "temperature": [36.8, 36.9, 37.0],
    })


def test_valid_dataset_has_no_validation_issues():

    processed, report = validate_dataset(
        create_sample_dataset()
    )

    assert report["validation_issue_records"] == 0


def test_invalid_timestamp_is_detected():

    data = create_sample_dataset()
    data.loc[1, "timestamp"] = "invalid-date"

    _, report = validate_dataset(data)

    assert report["invalid_timestamps"] == 1


def test_irregular_timestamp_interval_is_detected():

    data = create_sample_dataset()
    data.loc[1, "timestamp"] = "2026-01-01 10:07:00"

    _, report = validate_dataset(data)

    assert report["timestamp_interval_mismatches"] >= 1


def test_implausible_heart_rate_is_detected():

    data = create_sample_dataset()
    data.loc[1, "heart_rate"] = 250

    processed, report = validate_dataset(data)

    assert processed.loc[1, "heart_rate_out_of_range"]
    assert report["implausible_value_records"] == 1


def test_original_dataset_is_not_modified():

    data = create_sample_dataset()
    original = data.copy(deep=True)

    validate_dataset(data)

    pd.testing.assert_frame_equal(data, original)


def test_empty_dataset_is_rejected():

    data = create_sample_dataset().iloc[0:0]

    with pytest.raises(ValueError):
        validate_dataset(data)


def test_invalid_interval_is_rejected():

    data = create_sample_dataset()

    with pytest.raises(ValueError):
        validate_dataset(data, expected_interval_minutes=0)
