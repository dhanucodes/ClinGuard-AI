
"""Synthetic patient vital-sign data generator for ClinGuard AI."""

from pathlib import Path

import numpy as np
import pandas as pd


VITAL_COLUMNS = [
    "heart_rate",
    "spo2",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "temperature",
]

# Starting population averages and normal variation.
# Each virtual patient receives their own slightly different baseline.
BASELINES = {
    "heart_rate": (72.0, 7.0),
    "spo2": (97.5, 0.8),
    "systolic_bp": (118.0, 9.0),
    "diastolic_bp": (76.0, 7.0),
    "respiratory_rate": (16.0, 2.0),
    "temperature": (36.8, 0.25),
}


def generate_dataset(
    n_patients=8,
    duration_hours=24,
    interval_minutes=5,
    seed=42,
):
    """Generate reproducible synthetic patient monitoring records."""

    if not isinstance(n_patients, int) or n_patients < 1:
        raise ValueError("n_patients must be a positive integer.")

    if duration_hours <= 0:
        raise ValueError("duration_hours must be greater than zero.")

    if not isinstance(interval_minutes, int) or interval_minutes < 1:
        raise ValueError("interval_minutes must be a positive integer.")

    rng = np.random.default_rng(seed)
    all_patients = []

    start_time = pd.Timestamp("2026-01-01 00:00:00")

    for patient_number in range(1, n_patients + 1):

        patient_id = f"P{patient_number:03d}"

        count = int(duration_hours * 60 / interval_minutes) + 1

        elapsed_minutes = np.arange(count) * interval_minutes

        timestamps = (
            start_time
            + pd.to_timedelta(elapsed_minutes, unit="m")
        )

        # 1. Create an individual baseline for this patient.
        patient_profile = {}

        for vital, (average, variation) in BASELINES.items():

            personal_average = rng.normal(
                average,
                variation * 0.35
            )

            patient_profile[vital] = (
                personal_average,
                variation
            )

        
        # 2. Generate normal vital-sign readings.
        patient_data = {}

        daily_cycle = np.sin(
            2 * np.pi * elapsed_minutes / (24 * 60)
        )

        for vital, (personal_average, variation) in patient_profile.items():

            readings = rng.normal(
                personal_average,
                variation,
                count
            )

            if vital == "heart_rate":
                readings += daily_cycle * 2.0

            elif vital == "temperature":
                readings += daily_cycle * 0.08

            # Keep normally generated SpO2 readings within
            # the synthetic engineering bounds.
            if vital == "spo2":
                readings = np.clip(readings, 50, 100)

            patient_data[vital] = readings

        

        # 3. Create the initial normal scenario labels.
        anomaly_label = np.zeros(count, dtype=int)

        scenario_type = np.full(
            count, "normal", dtype=object
        )

        scenario_description = np.full(
            count,
            "No injected physiological event",
            dtype=object
        )

        event_id = np.full(count, "", dtype=object)

        # 4. Inject a simulated physiological event.
        if count >= 8:

            event_length = max(3, min(count // 8, 18))

            event_start = int(
                rng.integers(
                    2,
                    max(3, count - event_length + 1)
                )
            )

            event_end = min(
                count,
                event_start + event_length
            )

            event_slice = slice(event_start, event_end)

            scenarios = [
                "sudden_anomaly",
                "gradual_deterioration",
                "multivariate_anomaly",
            ]

            selected_scenario = scenarios[
                (patient_number - 1) % len(scenarios)
            ]

            anomaly_label[event_slice] = 1

            scenario_type[event_slice] = selected_scenario

            event_id[event_slice] = f"E{patient_number:03d}"

            if selected_scenario == "sudden_anomaly":

                patient_data["heart_rate"][event_slice] += 30

                scenario_description[event_slice] = (
                    "Injected abrupt heart-rate deviation"
                )

            elif selected_scenario == "gradual_deterioration":

                ramp = np.linspace(
                    0.2,
                    1.0,
                    event_end - event_start
                )

                patient_data["spo2"][event_slice] -= 4 * ramp

                patient_data["respiratory_rate"][event_slice] += 5 * ramp

                scenario_description[event_slice] = (
                    "Injected gradual oxygen and respiration change"
                )

            else:

                patient_data["heart_rate"][event_slice] += 22

                patient_data["spo2"][event_slice] -= 3.5

                patient_data["systolic_bp"][event_slice] -= 18

                scenario_description[event_slice] = (
                    "Injected simultaneous multi-vital deviation"
                )

        # 5. Create a DataFrame for the patient.
        frame = pd.DataFrame(patient_data)

        frame.insert(0, "timestamp", timestamps)

        frame.insert(0, "patient_id", patient_id)

        frame["anomaly_label"] = anomaly_label

        frame["scenario_type"] = scenario_type

        frame["scenario_description"] = scenario_description

        frame["event_id"] = event_id

        # 6. Simulate data-quality problems separately.
        missing_flag = np.zeros(count, dtype=bool)

        noise_flag = np.zeros(count, dtype=bool)

        sensor_error_flag = np.zeros(count, dtype=bool)

        irregular_flag = np.zeros(count, dtype=bool)

        if count > 5:

            # Missing readings.
            missing_indices = rng.choice(
                count,
                size=max(1, count // 100),
                replace=False
            )

            for idx in missing_indices:

                vital = rng.choice(VITAL_COLUMNS)

                frame.loc[idx, vital] = np.nan

                missing_flag[idx] = True

            # Noisy readings.
            noise_indices = rng.choice(
                count,
                size=max(1, count // 100),
                replace=False
            )

            for idx in noise_indices:

                vital = rng.choice(VITAL_COLUMNS)

                if pd.notna(frame.loc[idx, vital]):

                    frame.loc[idx, vital] += rng.normal(0, 4)

                    noise_flag[idx] = True

            # Simulated sensor errors.
            sensor_indices = rng.choice(
                count,
                size=max(1, count // 200),
                replace=False
            )

            for idx in sensor_indices:

                vital = rng.choice(VITAL_COLUMNS)

                frame.loc[idx, vital] = np.nan

                sensor_error_flag[idx] = True

            # Irregular timestamps.
            irregular_indices = rng.choice(
                count,
                size=max(1, count // 100),
                replace=False
            )

            offsets = np.zeros(count, dtype=int)

            for idx in irregular_indices:

                offsets[idx:] += 2

                irregular_flag[idx] = True

            frame["timestamp"] = (
                frame["timestamp"]
                + pd.to_timedelta(offsets, unit="m")
            )

        # 7. Store individual quality flags.
        frame["has_missing_value"] = missing_flag

        frame["has_noise"] = noise_flag

        frame["has_sensor_error"] = sensor_error_flag

        frame["has_irregular_timestamp"] = irregular_flag

        frame["data_quality_issue"] = (
            missing_flag
            | noise_flag
            | sensor_error_flag
            | irregular_flag
        )

        frame["data_origin"] = "synthetic_generated"

        all_patients.append(frame)

    # 8. Combine all virtual patients into one dataset.
    result = pd.concat(
        all_patients,
        ignore_index=True
    )

    result = result.sort_values(
        ["patient_id", "timestamp"]
    ).reset_index(drop=True)

    return result


def save_dataset(dataset, output_path):
    """Save the generated dataset as a CSV file."""

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset.to_csv(path, index=False)

    return path
