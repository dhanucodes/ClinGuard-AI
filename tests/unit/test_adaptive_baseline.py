
import pandas as pd
import pytest

from clinguard.adaptive_baseline.baseline import (
    calculate_adaptive_baselines,
)

from clinguard.data_generation.generator import VITAL_COLUMNS


def create_sample_data():

    records = []

    for patient_id, values in [
        ("P001", [10, 20, 30, 40, 50, 60, 100]),
        ("P002", [50, 60, 70, 80, 90, 100, 120]),
    ]:

        for index, value in enumerate(values):

            record = {
                "patient_id": patient_id,
                "timestamp": pd.Timestamp("2026-01-01")
                + pd.Timedelta(minutes=5 * index),
            }

            for vital in VITAL_COLUMNS:
                record[vital] = value

            records.append(record)

    return pd.DataFrame(records)


def test_baseline_uses_previous_readings_only():

    data = create_sample_data()

    result = calculate_adaptive_baselines(
        data,
        window_size=6,
        min_periods=6,
    )

    patient_data = result[
        result["patient_id"] == "P001"
    ].reset_index(drop=True)

    # Median of the first six readings:
    # (10, 20, 30, 40, 50, 60) -> 35
    assert patient_data.loc[6, "heart_rate_baseline"] == 35


def test_baseline_is_not_available_too_early():

    data = create_sample_data()

    result = calculate_adaptive_baselines(
        data,
        window_size=6,
        min_periods=6,
    )

    patient_data = result[
        result["patient_id"] == "P001"
    ].reset_index(drop=True)

    assert pd.isna(
        patient_data.loc[5, "heart_rate_baseline"]
    )

    assert pd.notna(
        patient_data.loc[6, "heart_rate_baseline"]
    )


def test_each_patient_gets_separate_baseline():

    data = create_sample_data()

    result = calculate_adaptive_baselines(
        data,
        window_size=6,
        min_periods=6,
    )

    patient_one = result[
        (result["patient_id"] == "P001")
        & (result.index == 6)
    ]["heart_rate_baseline"].iloc[0]

    patient_two = result[
        (result["patient_id"] == "P002")
        & (result.index == 13)
    ]["heart_rate_baseline"].iloc[0]

    assert patient_one == 35
    assert patient_two == 75


def test_baseline_ready_flag():

    data = create_sample_data()

    result = calculate_adaptive_baselines(
        data,
        window_size=6,
        min_periods=6,
    )

    assert result["baseline_ready"].sum() == 2


def test_original_dataset_is_not_modified():

    data = create_sample_data()
    original_columns = list(data.columns)

    calculate_adaptive_baselines(data)

    assert list(data.columns) == original_columns


def test_empty_dataset_is_rejected():

    data = create_sample_data().iloc[0:0]

    with pytest.raises(ValueError):
        calculate_adaptive_baselines(data)


def test_invalid_window_size_is_rejected():

    data = create_sample_data()

    with pytest.raises(ValueError):
        calculate_adaptive_baselines(
            data,
            window_size=0,
        )


def test_invalid_min_periods_is_rejected():

    data = create_sample_data()

    with pytest.raises(ValueError):
        calculate_adaptive_baselines(
            data,
            window_size=6,
            min_periods=7,
        )
