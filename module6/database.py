"""
Module 6 — Database Engine & Session Factory

Provides connection pooling, safe schema migrations, foreign key enforcement,
and transaction context managers. Works seamlessly with PostgreSQL and SQLite.
"""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from module6.models import Base

# Root project directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Default to SQLite in data/attendance.db if DATABASE_URL is not set
DEFAULT_DB_PATH = ROOT_DIR / "data" / "attendance.db"
DEFAULT_SQLITE_URL = f"sqlite:///{DEFAULT_DB_PATH.as_posix()}"


def get_database_url() -> str:
    """Retrieve database URL from environment variable or fallback to default SQLite."""
    return os.getenv("DATABASE_URL", DEFAULT_SQLITE_URL)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable SQLite foreign key support on connection."""
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def get_engine(database_url: str = None) -> Engine:
    """Create a SQLAlchemy engine configured for PostgreSQL or SQLite."""
    url = database_url or get_database_url()
    is_sqlite = url.startswith("sqlite")

    if is_sqlite:
        # Ensure parent directory exists for SQLite
        db_path_str = url.replace("sqlite:///", "")
        if db_path_str and not db_path_str.startswith(":memory:"):
            Path(db_path_str).parent.mkdir(parents=True, exist_ok=True)

        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            echo=False,
        )
    else:
        # PostgreSQL with connection pooling
        return create_engine(
            url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False,
        )


# Global default engine and session factory
_engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def init_db(engine: Engine = None) -> None:
    """
    Safely initialize tables and apply non-destructive schema migrations.
    Preserves all existing registered student rows.
    """
    eng = engine or _engine

    # Check if this is SQLite with an existing students table missing face_embedding
    with eng.connect() as conn:
        try:
            # Check if students table already exists
            result = conn.execute(text("SELECT 1 FROM students LIMIT 1"))
            # Check existing columns
            cols_info = conn.execute(text("PRAGMA table_info(students)")).fetchall()
            col_names = [col[1] for col in cols_info]

            if "face_embedding" not in col_names:
                conn.execute(text("ALTER TABLE students ADD COLUMN face_embedding BLOB"))
                conn.commit()
        except Exception:
            # Table does not exist yet; create_all will create it
            pass

    # Create any missing tables (e.g. attendance)
    Base.metadata.create_all(bind=eng)


@contextmanager
def get_db_session(engine: Engine = None) -> Generator[Session, None, None]:
    """
    Context manager providing a transactional database session.
    Commits on successful completion; rolls back automatically on error.
    """
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine or _engine)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
