"""PostgreSQL connection configuration for ClinGuard AI."""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, URL, text

# Locate the main project folder.
ROOT = Path(__file__).resolve().parents[3]

# Load the .env file from the project root.
ENV_PATH = ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Read database settings.
required_settings = [
    "DB_USER",
    "DB_PASSWORD",
    "DB_NAME",
]

missing_settings = [
    setting for setting in required_settings
    if not os.getenv(setting)
]

if missing_settings:
    raise ValueError(
        f"Missing database settings: {missing_settings}. "
        "Check your .env file."
    )

database_url = URL.create(
    drivername="postgresql+psycopg",
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    database=os.getenv("DB_NAME"),
)

engine = create_engine(
    database_url,
    pool_pre_ping=True,
)


def test_database_connection():

    with engine.connect() as connection:
        result = connection.execute(text("SELECT version()"))
        version = result.scalar()

    return version