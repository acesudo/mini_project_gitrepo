
import sqlite3
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Database location
DB_PATH = BASE_DIR / "data" / "attendance.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    return sqlite3.connect(DB_PATH)


def initialize_database():
    with get_connection() as conn:
        cursor = conn.cursor()

        # Student information
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                semester INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Face image records
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_samples (
                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                image_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (student_id)
                REFERENCES students(student_id)
                ON DELETE CASCADE,

                UNIQUE(student_id, image_path)
            )
        """)

        # Enable foreign key enforcement
        cursor.execute("PRAGMA foreign_keys = ON")

        conn.commit()


def add_student(student):
    with get_connection() as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        conn.execute("""
            INSERT INTO students
            (student_id, name, department, semester)
            VALUES (?, ?, ?, ?)
        """, (
            student.student_id,
            student.name,
            student.department,
            student.semester
        ))


def add_face_sample(student_id, image_path):
    with get_connection() as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        conn.execute("""
            INSERT INTO face_samples (student_id, image_path)
            VALUES (?, ?)
        """, (student_id, str(image_path)))


def get_student(student_id):
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT student_id, name, department, semester
            FROM students
            WHERE student_id = ?
        """, (student_id,))

        return cursor.fetchone()


def get_face_samples(student_id):
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT image_path
            FROM face_samples
            WHERE student_id = ?
            ORDER BY sample_id
        """, (student_id,))

        return [row[0] for row in cursor.fetchall()]


def get_all_students():
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT student_id, name, department, semester
            FROM students
            ORDER BY student_id
        """)

        return cursor.fetchall()
