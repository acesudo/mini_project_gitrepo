# Module 7 — Backend API (FastAPI)

## 1. Overview

Module 7 exposes the whole Face Attendance system over **HTTP** so that the Module 8 dashboard (or Postman) can use it. It does not contain new recognition logic — it **connects** the earlier modules:

``` mermaid
flowchart LR
    Client[Dashboard / Postman / Camera app] -->|HTTP + JSON| API[Module 7: FastAPI]
    API -->|detect_faces| M2[Module 2: Haar Cascade]
    API -->|preprocess_face, validate| M3[Module 3: Preprocessing]
    API -->|FaceNet embeddings| M4[Module 4]
    API -->|FaceMatcher gallery| M5[Module 5]
    API -->|AttendanceManager| M6[(Module 6: Database)]
```

Interactive documentation is generated automatically at **`/docs`** (Swagger UI) and **`/redoc`**.

---

## 2. Endpoints

| Method | Path | Purpose | Success | Errors |
|---|---|---|---|---|
| GET | `/health` | Server, database and gallery status | 200 | 503 |
| POST | `/students` | Create a student (no face yet) | 201 | 409 exists, 422 invalid |
| POST | `/students/register-with-faces` | Create/update student **and** enroll face from 3+ photos | 201 | 422 too few valid samples |
| GET | `/students` | List students (`department`, `limit`, `offset`) | 200 | |
| GET | `/students/{id}` | One student | 200 | 404 |
| PUT | `/students/{id}` | Update name / department / semester | 200 | 404, 422 |
| DELETE | `/students/{id}` | Delete student + attendance + face | 204 | 404 |
| POST | `/recognition/frame` | Upload a camera image → recognize all faces, mark attendance | 200 | 400 bad image, 413 too large |
| POST | `/recognition/embedding` | Match a raw 512-d vector | 200 | 422 |
| GET | `/attendance/daily` | Records for `date` (default today), optional `department` | 200 | 422 |
| GET | `/attendance/student/{id}` | One student's history (`start_date`, `end_date`) | 200 | 404, 422 |
| GET | `/attendance/monthly` | Monthly counts and percentages (`year`, `month`) | 200 | 422 |
| POST | `/attendance/mark` | Manual attendance entry | 201 | 404 unregistered, 409 duplicate |

### Rules enforced by the API
- **Unknown faces are never marked present** (see `_to_face_result` in `routers/recognition.py`).
- **Duplicate attendance** is blocked by Module 6; the API reports `already_marked: true` for recognition and `409` for manual entry.
- **Student fields** follow the same rules as Module 3 (`module3/student.py`), so the CLI and API accept exactly the same data.
- **Registration** needs at least 3 usable photos, each with exactly one face that passes Module 3's brightness/blur check. Rejected photos are listed with a reason.
- Student IDs are normalized (`" cse1025 "` → `CSE1025`).

### Example: recognition response
``` json
{
  "faces_detected": 2,
  "known_faces": 1,
  "unknown_faces": 1,
  "results": [
    {
      "student_id": "CSE1025", "confidence": 0.9123, "distance": 0.4189,
      "is_known": true, "bbox": [120, 80, 140, 140],
      "attendance": { "marked": true, "already_marked": false, "message": "Attendance marked successfully ...", "record": { "...": "..." } }
    },
    { "student_id": "UNKNOWN", "confidence": 0.31, "distance": 1.17, "is_known": false, "bbox": [400, 90, 130, 130], "attendance": null }
  ]
}
```

---

## 3. Files

| File | Responsibility |
|---|---|
| `app.py` | `create_app()` — builds the app, loads the face gallery from the database, adds CORS, includes routers |
| `main.py` | Entry point: `app = create_app()` for `uvicorn` |
| `schemas.py` | Pydantic request/response models (validation + `/docs`) |
| `dependencies.py` | `AppState` (manager, matcher, lock) and `get_state` dependency |
| `crud.py` | List / get / delete student queries missing from Module 6 |
| `utils.py` | Safe image-upload decoding (size limit, invalid-file errors) |
| `routers/health.py`, `students.py`, `recognition.py`, `attendance.py` | One router per resource |
| `postman_collection.json` | Import into Postman to try every endpoint |

### Design notes
- **Database is the source of truth.** The in-memory `FaceMatcher` gallery is rebuilt from the database at startup and after every registration or deletion, so a server restart loses nothing.
- **One lock** (`AppState.lock`) keeps gallery updates and FaceNet inference from overlapping when several requests arrive together.
- **FaceNet loads lazily** on the first registration/recognition request, so the server starts quickly.
- **CORS** is open (`*`) for development so the Module 8 dashboard can call the API; restrict it with `CORS_ORIGINS`.
- ⚠️ The API has **no authentication yet** — that is Module 9. Do not expose it publicly before then.

---

## 4. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `sqlite:///data/attendance.db` | Database (SQLite locally, PostgreSQL in Docker) |
| `FACE_MATCH_THRESHOLD` | `0.70` | Minimum cosine similarity to accept a match |
| `CORS_ORIGINS` | `*` | Comma-separated allowed browser origins |

---

## 5. How to Run & Verify

### Install the new dependencies
```powershell
.\.venv\Scripts\pip install -r requirements.txt
```

### Start the server (from the project root)
```powershell
.\.venv\Scripts\uvicorn module7.main:app --reload
```
Open **http://127.0.0.1:8000/docs** and use *Try it out* on any endpoint.

### Try it with Postman
Import `module7/postman_collection.json`. The `base_url` variable is `http://127.0.0.1:8000`. For the two image requests, click the `images` / `image` rows and choose files from your computer.

### Run the automated tests
```powershell
.\.venv\Scripts\pytest tests/test_module7_api.py -v
```
The tests use an in-memory database and a fake FaceNet generator, so they need no webcam, no model download, and never touch `data/attendance.db`. They cover: health, student CRUD and validation, face registration (too few / bad / multi-face samples), recognition (known, unknown, no face, invalid image, no-duplicate marking), attendance reports, cascade deletion, and reloading the gallery after a restart.
