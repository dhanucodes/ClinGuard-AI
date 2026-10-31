"""Database session management for ClinGuard AI."""

from clinguard.database.connection import engine
from sqlalchemy.orm import sessionmaker

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def get_db():

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()