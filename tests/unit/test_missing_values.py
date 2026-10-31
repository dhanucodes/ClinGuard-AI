
import pandas as pd
import pytest

from clinguard.preprocessing.missing_values import (
    handle_missing_values,
)


def create_sample_dataset():

    return pd.DataFrame({
        "patient_id": [
            "P001", "P001", "P001", "P001",
            "P002", "P002",
        ],
        "timestamp": pd.to_datetime([
            "2026-01-01 10:00:00",
            "2026-01-01 10:05:00",
            "2026-01-01 10:10:00",
            "2026-01-01 10:15:00",
            "2026-01-01 10:00:00",
            "2026-01-01 10:05:00",
        ]),
        "heart_rate": [
            72.0, None, None, 76.0, 80.0, None
        ],
        "spo2": [98, 98, 97, 97, 96, 96],
        "systolic_bp": [118, 118, 120, 120, 115, 115],
        "diastolic_bp": [76, 76, 78, 78, 75, 75],
        "respiratory_rate": [16, 16, 17, 17, 18, 18],
        "temperature": [36.8, 36.8, 36.9, 36.9, 37.0, 37.0],
    })


def test_short_gap_is_forward_filled():

    data = create_sample_dataset()

    processed, _ = handle_missing_values(data)

    assert processed.loc[1, "heart_rate"] == 72.0


def test_long_gap_remains_missing():

    data = create_sample_dataset()

    processed, _ = handle_missing_values(data)

    assert pd.isna(processed.loc[2, "heart_rate"])


def test_original_missing_indicator_is_preserved():

    data = create_sample_dataset()

    processed, _ = handle_missing_values(data)

    assert processed.loc[1, "heart_rate_was_missing"] == True


def test_patients_are_processed_separately():

    data = create_sample_dataset()

    processed, _ = handle_missing_values(data)

    # P002's first missing reading must not use P001's last reading.
    assert processed.loc[5, "heart_rate"] == 80.0


def test_original_dataset_is_not_modified():

    data = create_sample_dataset()

    original = data.copy(deep=True)

    handle_missing_values(data)

    pd.testing.assert_frame_equal(data, original)


def test_remaining_missing_values_are_reported():

    data = create_sample_dataset()

    _, remaining = handle_missing_values(data)

    assert remaining["heart_rate"] == 1


def test_invalid_fill_limit():

    data = create_sample_dataset()

    with pytest.raises(ValueError):
        handle_missing_values(data, max_forward_fill=-1)


def test_empty_dataset():

    data = create_sample_dataset().iloc[0:0]

    with pytest.raises(ValueError):
        handle_missing_values(data)
