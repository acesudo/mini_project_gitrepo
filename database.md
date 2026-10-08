# Database Design

## Step 11 — Database and ER Diagram

### Technologies

- PostgreSQL
- SQL
- Python database integration
- SQLAlchemy as the planned database access layer

## ER Diagram

``` mermaid
erDiagram
    STUDENTS {
        string student_id PK
        string name
        string department
        int semester
        text face_embedding
    }

    ATTENDANCE {
        int attendance_id PK
        string student_id FK
        date date
        time time
        string status
    }

    STUDENTS ||--o{ ATTENDANCE : has
```

## Students

| Field          | Role                           |
|----------------|--------------------------------|
| Student_ID     | Primary key                    |
| Name           | Student name                   |
| Department     | Academic department            |
| Semester       | Current semester               |
| Face_Embedding | Registered face representation |

## Attendance

| Field         | Role              |
|---------------|-------------------|
| Attendance_ID | Primary key       |
| Student_ID    | Foreign key       |
| Date          | Attendance date   |
| Time          | Attendance time   |
| Status        | Attendance status |

## Relationship

``` text
STUDENTS
   │ 1
   │
   │ N
   ↓
ATTENDANCE
```

## Example Queries

``` sql
SELECT *
FROM attendance
WHERE date = CURRENT_DATE;
```

``` sql
SELECT *
FROM attendance
WHERE student_id = 'CSE1025';
```

``` sql
SELECT student_id, COUNT(*) AS attendance_count
FROM attendance
WHERE date >= '2026-08-01'
  AND date <= '2026-08-31'
GROUP BY student_id;
```

## Acceptance Criteria

- PostgreSQL database exists.
- Students table exists.
- Attendance table exists.
- Primary and foreign keys are configured.
- Student records can be inserted/retrieved.
- Attendance records can be inserted/retrieved.
- Daily/monthly queries work.
- Duplicate attendance rules are enforced.
