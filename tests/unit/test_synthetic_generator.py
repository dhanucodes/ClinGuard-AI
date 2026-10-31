
import pandas as pd
import pytest

from clinguard.data_generation.generator import (
    VITAL_COLUMNS,
    generate_dataset,
    save_dataset,
)


# TEST 1: Check dataset structure and patient count
def test_schema_and_patient_count():

    data = generate_dataset(
        n_patients=3,
        duration_hours=1,
        interval_minutes=5,
        seed=7
    )

    required_columns = [
        "patient_id",
        "timestamp",
        *VITAL_COLUMNS,
        "anomaly_label",
        "scenario_type",
        "scenario_description",
        "event_id",
        "has_missing_value",
        "has_noise",
        "has_sensor_error",
        "has_irregular_timestamp",
        "data_quality_issue",
        "data_origin",
    ]

    for column in required_columns:
        assert column in data.columns

    assert data["patient_id"].nunique() == 3

    assert len(data) == 39


# TEST 2: Check reproducibility
def test_reproducibility():

    data1 = generate_dataset(
        n_patients=2,
        duration_hours=1,
        seed=123
    )

    data2 = generate_dataset(
        n_patients=2,
        duration_hours=1,
        seed=123
    )

    pd.testing.assert_frame_equal(data1, data2)


# TEST 3: Check anomaly labels
def test_anomaly_labels():

    data = generate_dataset(
        n_patients=3,
        duration_hours=2,
        seed=9
    )

    assert set(data["anomaly_label"].unique()).issubset({0, 1})

    expected = data["scenario_type"] != "normal"

    actual = data["anomaly_label"] == 1

    assert (expected == actual).all()


# TEST 4: Check data-quality flags
def test_quality_flags():

    data = generate_dataset(
        n_patients=2,
        duration_hours=2,
        seed=5
    )

    expected = (
        data["has_missing_value"]
        | data["has_noise"]
        | data["has_sensor_error"]
        | data["has_irregular_timestamp"]
    )

    assert (
        data["data_quality_issue"] == expected
    ).all()


# TEST 5: Check invalid configurations
def test_invalid_configuration():

    with pytest.raises(ValueError):
        generate_dataset(n_patients=0)

    with pytest.raises(ValueError):
        generate_dataset(interval_minutes=0)


# TEST 6: Check CSV export
def test_csv_export(tmp_path):

    data = generate_dataset(
        n_patients=1,
        duration_hours=1,
        seed=4
    )

    output_file = tmp_path / "sample.csv"

    save_dataset(data, output_file)

    assert output_file.exists()

    loaded_data = pd.read_csv(output_file)

    assert len(loaded_data) == len(data)
