"""Create all ClinGuard AI PostgreSQL tables."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinguard.database.connection import engine
from clinguard.database.models import Base


def main():

    print("\nCLINGUARD AI - DATABASE INITIALISATION")
    print("=" * 45)

    Base.metadata.create_all(bind=engine)

    print("Database tables created successfully!")

    for table_name in Base.metadata.tables:
        print("-", table_name)


if __name__ == "__main__":
    main()