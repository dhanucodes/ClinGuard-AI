"""Test the ClinGuard AI PostgreSQL connection."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinguard.database.connection import test_database_connection


def main():

    print("\nCLINGUARD AI - DATABASE CONNECTION")
    print("=" * 40)

    try:
        version = test_database_connection()

        print("Database connection successful!")
        print("PostgreSQL version:")
        print(version)

    except Exception as error:
        print("Database connection failed.")
        print("Error:", error)


if __name__ == "__main__":
    main()