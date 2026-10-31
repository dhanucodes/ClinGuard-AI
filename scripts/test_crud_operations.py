"""Test ClinGuard AI database CRUD operations."""

from pathlib import Path
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinguard.database.connection import engine
from clinguard.database.session import SessionLocal
from clinguard.database.models import Base
from clinguard.database.crud import (
    create_patient,
    get_patient,
    create_vital_reading,
    get_patient_readings,
    update_vital_reading,
    delete_vital_reading,
)


def main():

    Base.metadata.create_all(bind=engine)

    db = SessionLocal()

    test_patient_id = "TEST001"

    try:
        # CREATE PATIENT
        patient = get_patient(db, test_patient_id)

        if patient is None:
            patient = create_patient(db, test_patient_id)

        print("Patient created:", patient.patient_id)

        # CREATE READING
        reading_data = {
            "patient_id": test_patient_id,
            "timestamp": datetime(2026, 1, 1, 10, 0),
            "heart_rate": 75,
            "spo2": 98,
            "systolic_bp": 120,
            "diastolic_bp": 80,
            "respiratory_rate": 16,
            "temperature": 36.8,
        }

        reading = create_vital_reading(db, reading_data)

        print("Reading created:", reading.reading_id)

        # READ
        readings = get_patient_readings(db, test_patient_id)

        print("Patient readings:", len(readings))

        # UPDATE
        updated = update_vital_reading(
            db,
            reading.reading_id,
            {"heart_rate": 80},
        )

        print("Updated heart rate:", updated.heart_rate)

        # DELETE
        deleted = delete_vital_reading(
            db,
            reading.reading_id,
        )

        print("Reading deleted:", deleted)

        print("\nCRUD test completed successfully!")

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    main()