"""SQLAlchemy database models for ClinGuard AI."""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class Patient(Base):
    __tablename__ = "patients"

    patient_id: Mapped[str] = mapped_column(
        String(50), primary_key=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class VitalReading(Base):
    __tablename__ = "vital_readings"

    __table_args__ = (
        UniqueConstraint(
            "patient_id",
            "timestamp",
            name="uq_patient_reading_timestamp",
        ),
    )

    reading_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )

    patient_id: Mapped[str] = mapped_column(
        ForeignKey("patients.patient_id"),
        nullable=False,
        index=True,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False
    )

    heart_rate: Mapped[float | None] = mapped_column(Float)
    spo2: Mapped[float | None] = mapped_column(Float)
    systolic_bp: Mapped[float | None] = mapped_column(Float)
    diastolic_bp: Mapped[float | None] = mapped_column(Float)
    respiratory_rate: Mapped[float | None] = mapped_column(Float)
    temperature: Mapped[float | None] = mapped_column(Float)

    has_missing_value: Mapped[bool] = mapped_column(
        Boolean, default=False
    )
    has_noise: Mapped[bool] = mapped_column(
        Boolean, default=False
    )
    has_sensor_error: Mapped[bool] = mapped_column(
        Boolean, default=False
    )
    data_quality_issue: Mapped[bool] = mapped_column(
        Boolean, default=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class DetectionResult(Base):
    __tablename__ = "detection_results"

    detection_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )

    reading_id: Mapped[int] = mapped_column(
        ForeignKey("vital_readings.reading_id"),
        nullable=False,
        unique=True,
    )

    statistical_anomaly_flag: Mapped[bool] = mapped_column(
        Boolean, default=False
    )
    isolation_forest_anomaly_flag: Mapped[bool] = mapped_column(
        Boolean, default=False
    )
    hybrid_or_flag: Mapped[bool] = mapped_column(
        Boolean, default=False
    )
    hybrid_and_flag: Mapped[bool] = mapped_column(
        Boolean, default=False
    )

    isolation_forest_score: Mapped[float | None] = mapped_column(
        Float
    )


class VerificationResult(Base):
    __tablename__ = "verification_results"

    verification_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )

    detection_id: Mapped[int] = mapped_column(
        ForeignKey("detection_results.detection_id"),
        nullable=False,
        unique=True,
    )

    verification_status: Mapped[str] = mapped_column(
        String(100), nullable=False
    )

    verification_reason: Mapped[str] = mapped_column(
        Text, nullable=False
    )

    context_issues: Mapped[str] = mapped_column(
        Text, nullable=False
    )

    explainable_reason: Mapped[str | None] = mapped_column(Text)