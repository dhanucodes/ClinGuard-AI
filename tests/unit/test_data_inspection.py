
import pandas as pd
import pytest

from clinguard.preprocessing.inspection import (
    REQUIRED_COLUMNS,
    inspect_dataset,
)


def create_sample_dataset():
    """Create a small dataset for testing."""

    data = {
        "patient_id": ["P001", "P001", "P002"],
        "timestamp": [
            "2026-01-01 00:00:00",
            "2026-01-01 00:05:00",
            "2026-01-01 00:00:00",
        ],
        "heart_rate": [72.0, None, 80.0],
        "spo2": [98.0, 97.0, 96.0],
        "systolic_bp": [118.0, 120.0, 115.0],
        "diastolic_bp": [76.0, 78.0, 75.0],
        "respiratory_rate": [16.0, 17.0, 18.0],
        "temperature": [36.8, 36.9, 37.0],
        "anomaly_label": [0, 0, 1],
        "scenario_type": ["normal", "normal", "sudden_anomaly"],
        "event_id": ["", "", "E002"],
    }

    return pd.DataFrame(data)


def test_inspection_record_and_patient_counts():

    data = create_sample_dataset()

    report = inspect_dataset(data)

    assert report["total_records"] == 3
    assert report["total_patients"] == 2


def test_inspection_detects_missing_values():

    data = create_sample_dataset()

    report = inspect_dataset(data)

    assert report["missing_values"]["heart_rate"] == 1


def test_inspection_detects_invalid_timestamp():

    data = create_sample_dataset()

    data.loc[0, "timestamp"] = "not-a-date"

    report = inspect_dataset(data)

    assert report["invalid_timestamps"] == 1


def test_inspection_detects_duplicate_patient_timestamp():

    data = create_sample_dataset()

    data.loc[2, "patient_id"] = "P001"
    data.loc[2, "timestamp"] = "2026-01-01 00:05:00"

    report = inspect_dataset(data)

    assert report["duplicate_patient_timestamps"] == 1


def test_inspection_rejects_missing_required_columns():

    data = create_sample_dataset()

    data = data.drop(columns=["heart_rate"])

    with pytest.raises(ValueError):
        inspect_dataset(data)


def test_inspection_rejects_empty_dataset():

    data = create_sample_dataset().iloc[0:0]

    with pytest.raises(ValueError):
        inspect_dataset(data)
