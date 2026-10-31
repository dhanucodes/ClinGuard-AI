
import pandas as pd
import pytest

from clinguard.adaptive_baseline.baseline import (
    calculate_adaptive_baselines,
)
from clinguard.adaptive_baseline.deviation import (
    calculate_deviations,
)
from clinguard.data_generation.generator import VITAL_COLUMNS


def create_sample_data(values):

    records = []

    for index, value in enumerate(values):

        record = {
            "patient_id": "P001",
            "timestamp": (
                pd.Timestamp("2026-01-01")
                + pd.Timedelta(minutes=5 * index)
            ),
        }

        for vital in VITAL_COLUMNS:
            record[vital] = value

        records.append(record)

    return pd.DataFrame(records)


def prepare_data(values):

    data = create_sample_data(values)

    return calculate_adaptive_baselines(
        data,
        window_size=6,
        min_periods=6,
    )


def test_signed_deviation():

    data = prepare_data([10, 20, 30, 40, 50, 60, 100])

    result = calculate_deviations(
        data,
        window_size=6,
        min_periods=6,
    )

    # Baseline = median of first six readings = 35.
    # Current reading = 100.
    assert result.loc[6, "heart_rate_deviation"] == 65


def test_absolute_deviation():

    data = prepare_data([10, 20, 30, 40, 50, 60, 100])

    result = calculate_deviations(
        data,
        window_size=6,
        min_periods=6,
    )

    assert result.loc[6, "heart_rate_absolute_deviation"] == 65


def test_z_score_is_calculated():

    data = prepare_data([10, 20, 30, 40, 50, 60, 100])

    result = calculate_deviations(
        data,
        window_size=6,
        min_periods=6,
    )

    assert pd.notna(result.loc[6, "heart_rate_z_score"])
    assert result.loc[6, "heart_rate_z_score"] > 0


def test_deviation_unavailable_before_baseline():

    data = prepare_data([10, 20, 30, 40, 50, 60, 100])

    result = calculate_deviations(
        data,
        window_size=6,
        min_periods=6,
    )

    assert pd.isna(result.loc[5, "heart_rate_deviation"])
    assert pd.isna(result.loc[5, "heart_rate_z_score"])


def test_constant_previous_readings_produce_no_z_score():

    data = prepare_data([70, 70, 70, 70, 70, 70, 90])

    result = calculate_deviations(
        data,
        window_size=6,
        min_periods=6,
    )

    assert pd.isna(result.loc[6, "heart_rate_z_score"])


def test_current_reading_does_not_change_its_baseline():

    first_data = prepare_data([10, 20, 30, 40, 50, 60, 100])
    second_data = prepare_data([10, 20, 30, 40, 50, 60, 200])

    first_result = calculate_deviations(first_data)
    second_result = calculate_deviations(second_data)

    assert (
        first_result.loc[6, "heart_rate_baseline"]
        == second_result.loc[6, "heart_rate_baseline"]
    )


def test_original_dataset_is_not_modified():

    data = prepare_data([10, 20, 30, 40, 50, 60, 100])

    original_columns = list(data.columns)

    calculate_deviations(data)

    assert list(data.columns) == original_columns


def test_empty_dataset_is_rejected():

    data = prepare_sample = prepare_data(
        [10, 20, 30, 40, 50, 60, 100]
    ).iloc[0:0]

    with pytest.raises(ValueError):
        calculate_deviations(data)


def test_invalid_window_size_is_rejected():

    data = prepare_data([10, 20, 30, 40, 50, 60, 100])

    with pytest.raises(ValueError):
        calculate_deviations(data, window_size=0)

