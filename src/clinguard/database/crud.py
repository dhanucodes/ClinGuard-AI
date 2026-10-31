
"""Database CRUD operations for ClinGuard AI."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from clinguard.database.models import (
    Patient,
    VitalReading,
    DetectionResult,
    VerificationResult,
)


def create_patient(db: Session, patient_id: str):
    patient = Patient(patient_id=patient_id)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


def get_patient(db: Session, patient_id: str):
    statement = select(Patient).where(
        Patient.patient_id == patient_id
    )
    return db.scalar(statement)


def create_vital_reading(db: Session, reading_data: dict):
    reading = VitalReading(**reading_data)
    db.add(reading)
    db.commit()
    db.refresh(reading)
    return reading


def get_patient_readings(db: Session, patient_id: str):
    statement = (
        select(VitalReading)
        .where(VitalReading.patient_id == patient_id)
        .order_by(VitalReading.timestamp)
    )
    return list(db.scalars(statement).all())


def get_vital_reading(db: Session, reading_id: int):
    return db.get(VitalReading, reading_id)


def update_vital_reading(
    db: Session,
    reading_id: int,
    updated_data: dict,
):
    reading = db.get(VitalReading, reading_id)

    if reading is None:
        return None

    for field, value in updated_data.items():
        if hasattr(reading, field):
            setattr(reading, field, value)

    db.commit()
    db.refresh(reading)
    return reading


def delete_vital_reading(db: Session, reading_id: int):
    reading = db.get(VitalReading, reading_id)

    if reading is None:
        return False

    db.delete(reading)
    db.commit()
    return True


def save_detection_and_verification(
    db: Session,
    reading_id: int,
    detection_data: dict,
    verification_data: dict,
):
    """
    Save or update detection and verification results
    for one existing vital reading.

    Both records are committed together.
    """

    try:
        # 1. Find an existing detection result.
        statement = select(DetectionResult).where(
            DetectionResult.reading_id == reading_id
        )
        detection = db.scalar(statement)

        # 2. Create a detection result if one doesn't exist.
        if detection is None:
            detection = DetectionResult(
                reading_id=reading_id
            )
            db.add(detection)
            db.flush()

        # 3. Update detection fields.
        for field, value in detection_data.items():
            if hasattr(detection, field):
                setattr(detection, field, value)

        # Make sure the detection ID is available.
        db.flush()

        # 4. Find an existing verification result.
        statement = select(VerificationResult).where(
            VerificationResult.detection_id
            == detection.detection_id
        )
        verification = db.scalar(statement)

        # 5. Create a verification result if necessary.
        if verification is None:
            verification = VerificationResult(
                detection_id=detection.detection_id,
                verification_status="pending",
                verification_reason="Pending verification.",
                context_issues="",
            )
            db.add(verification)

        # 6. Update verification fields.
        for field, value in verification_data.items():
            if hasattr(verification, field):
                setattr(verification, field, value)

        # 7. Commit both records together.
        db.commit()

        db.refresh(detection)
        db.refresh(verification)

        return {
            "detection": detection,
            "verification": verification,
        }

    except Exception:
        db.rollback()
        raise
