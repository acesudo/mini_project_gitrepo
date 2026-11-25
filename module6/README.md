# Module 6 — Attendance Database Layer

## 1. Overview

Module 6 provides the **relational persistence layer** for the Face Attendance system. Built with **SQLAlchemy ORM**, it stores student profiles, biometric face representations (512-dimensional FaceNet embeddings), and daily attendance logs.

It enforces **duplicate attendance prevention** both at the database level and application level, handles safe transactional commits/rollbacks, and provides analytical queries for daily and monthly attendance reporting.

---

## 2. Relational Database Design (ER Diagram)

The schema adheres strictly to [`database.md`](../database.md):

``` mermaid
erDiagram
    STUDENTS {
        string student_id PK "Unique Student ID (e.g. CSE1025)"
        string name "Full student name"
        string department "Academic department"
        int semester "Current semester"
        blob face_embedding "512-d float32 FaceNet representation (2048 bytes)"
        datetime created_at "Registration timestamp"
    }

    ATTENDANCE {
        int attendance_id PK "Auto-increment primary key"
        string student_id FK "References STUDENTS.student_id (CASCADE)"
        date date "Attendance date (Indexed)"
        time time "Attendance timestamp"
        string status "Attendance status ('Present', 'Late', etc.)"
        float confidence "Facial recognition similarity score"
        datetime created_at "Record creation timestamp"
    }

    STUDENTS ||--o{ ATTENDANCE : "has many"
```

---

## 3. Duplicate Attendance Protection

Preventing students from being logged multiple times on the same date is a critical system requirement:

1. **Database-Level Constraint**:
   ```python
   __table_args__ = (
       UniqueConstraint("student_id", "date", name="uq_student_daily_attendance"),
   )
   ```
   Ensures that even if two concurrent threads try to record attendance simultaneously, the relational database guarantees that only one row can exist for any `(student_id, date)` combination.

2. **Application-Level Interception (`mark_attendance`)**:
   Before inserting, `AttendanceManager.mark_attendance()` checks if a record for `(student_id, today)` already exists. If found:
   - It gracefully logs `already_marked = True`.
   - Returns the existing record timestamp without throwing errors or breaking the webcam stream.

---

## 4. Dual Database Engine Support (SQLite + PostgreSQL)

The database engine is dynamically configured via the `DATABASE_URL` environment variable:

```text
DATABASE_URL=sqlite:///data/attendance.db          # Default local development
DATABASE_URL=postgresql://user:pass@db:5432/attend # Docker Compose / Production
```

- **Local Development**: Runs out-of-the-box on `data/attendance.db` without requiring a PostgreSQL installation on Windows. SQLite foreign keys are explicitly enforced via `PRAGMA foreign_keys = ON`.
- **Production Deployment (Module 12)**: Connects to PostgreSQL with connection pooling (`pool_size=10, max_overflow=20`) with **zero code modifications**.

---

## 5. Key Classes & Methods

### `AttendanceManager`
Located in `module6/attendance_manager.py`:

| Method | Parameters | Return Type | Description |
|---|---|---|---|
| `mark_attendance` | `student_id, target_date, target_time, status, confidence` | `(bool, str, dict)` | Logs student attendance while preventing duplicates. |
| `get_daily_attendance` | `target_date, department` | `list[dict]` | Returns all attendance logs for a specific day with student details joined. |
| `get_student_attendance_history` | `student_id, start_date, end_date` | `list[dict]` | Chronological attendance logs for an individual student. |
| `get_monthly_attendance_summary` | `year, month, department` | `list[dict]` | Aggregates active days, present sessions, and attendance percentages. |
| `register_student` | `student_id, name, department, semester, embedding` | `dict` | Creates or updates a student profile with their 512-d FaceNet representation. |
| `load_all_registered_embeddings` | *None* | `dict[str, np.ndarray]` | Deserializes all enrolled embeddings to directly populate Module 5 `FaceMatcher`. |

---

## 6. How to Run & Verify

### Running the Module 6 Demo
```powershell
.\.venv\Scripts\python module6/attendance_manager.py
```
**Expected Output:**
```text
============================================================
MODULE 6 — Attendance Database Demo
============================================================
Database initialized successfully!

--- 1. Registered Student: Aarav Sharma (DEMO_CSE01) ---

--- 2. First Attendance Entry ---
Success: True
Message: Attendance marked successfully for student 'DEMO_CSE01' on 2026-11-25.

--- 3. Duplicate Attendance Entry Attempt ---
Duplicate Blocked (Expected False): False
Message: Attendance already marked for student 'DEMO_CSE01' on 2026-11-25.

--- 4. Today's Attendance List (1 students) ---
  [15:40:43] DEMO_CSE01 - Aarav Sharma (Present, confidence: 0.942)

--- 5. Loaded 1 Biometric Embeddings for FaceMatcher ---
  DEMO_CSE01 Embedding Shape: (512,), Norm: 1.0000

Module 6 demonstration completed successfully.
```

### Running Automated Unit Tests
```powershell
.\.venv\Scripts\pytest tests/test_module6_database.py -v
```
All 11 tests run against an isolated in-memory SQLite database, verifying:
- Table schema, primary keys, and foreign keys
- Embedding serialization and restoration
- First entry success and duplicate entry prevention
- Unique constraints at the database level
- Multi-date support
- Daily queries and monthly percentage aggregations
- Foreign key cascade deletions

---

## 7. How Files Communicate

``` mermaid
flowchart LR
    M3[Module 3: Registration<br/>Student Info] --> M6[Module 6: AttendanceManager<br/>register_student]
    M4[Module 4: FaceNet<br/>512-d Embedding] --> M6
    M6 -->|Registered Embeddings| M5[Module 5: FaceMatcher<br/>Gallery Matrix]
    M5 -->|Identified Student ID| M6A[Module 6: AttendanceManager<br/>mark_attendance]
    M6A -->|Unique Constraint Check| DB[(PostgreSQL / SQLite)]
    DB --> M7[Module 7: FastAPI Backend<br/>Endpoints]
    M7 --> M8[Module 8: Dashboard<br/>Daily & Monthly Reports]
```
