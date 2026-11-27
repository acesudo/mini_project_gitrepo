# Module 8 — Teacher & Admin Dashboard

Modern, responsive web dashboard for teachers and administrators to monitor real-time face recognition attendance, inspect daily records, manage students, view monthly analytics, detect defaulters, and export attendance reports.

---

## 1. Overview & Architecture

Module 8 is built as a single-page application (SPA) using HTML5, modern CSS3 (flexbox, grid, glassmorphism design), and vanilla ES6+ JavaScript. It connects directly with the **Module 7 REST API** (`/api/v1/*`) to deliver a real-time experience without external frontend frameworks or heavy dependencies.

```
+----------------------------------------------------------------------+
|                      Teacher / Admin Dashboard                       |
|                          (Module 8 Frontend)                         |
+----------------------------------------------------------------------+
                                   |
         REST API calls (JSON & multipart frame uploads)
                                   v
+----------------------------------------------------------------------+
|                         FastAPI Backend                              |
|                       (Module 7 /api/v1/*)                           |
+----------------------------------------------------------------------+
         |                                           |
         v                                           v
+-----------------------+                 +---------------------+
|   Face Recognition    |                 | Database Management |
|   (Modules 4 & 5)     |                 |     (Module 6)      |
+-----------------------+                 +---------------------+
```

---

## 2. Key Features

1. **Live Attendance Kiosk (WebRTC)**:
   - Live camera preview with canvas face-tracking overlays.
   - Capture loop with configurable interval (1s-5s).
   - Instant visual feedback on recognized faces (student ID, name, match confidence score).
   - Audio feedback chime and banner notification upon successful attendance marking.
   - Automatic prevention of redundant duplicate recognition alerts.

2. **Today's Attendance View**:
   - Live KPI cards: Total Students, Present Today, Absent Today, Attendance Percentage.
   - Real-time tabular listing of students marked present today with timestamps, recognition status, and confidence scores.
   - Instant one-click CSV export for daily records.
   - Manual attendance override button for edge cases.

3. **Student Directory & History**:
   - Searchable student catalog by Student ID, Name, Department, or Year.
   - Individual student profile cards with attendance metrics.
   - Date-range attendance history logs.
   - Quick registration modal with face image upload.

4. **Monthly Analytics & Defaulters Detection**:
   - Monthly attendance percentage computation per student.
   - Configurable minimum attendance threshold (default: 75%).
   - Auto-highlighted defaulters list with one-click export for academic administration.

5. **Toast Notifications & Status Indicators**:
   - Live backend API connectivity status badge (`Online` / `Offline`).
   - Non-intrusive toast messages for actions, errors, and recognition events.

---

## 3. Directory Structure

```
module8/
|-- __init__.py
|-- README.md                    # Documentation
`-- static/                      # Static assets mounted at /dashboard/
    |-- index.html               # Main SPA markup & templates
    |-- css/
    |   `-- style.css            # Responsive dark/modern theme stylesheet
    `-- js/
        `-- app.js               # Frontend controller, API client & WebRTC loop
```

---

## 4. Running the Dashboard

### Step 1: Start the Backend Server
From the project root:
```powershell
python -m uvicorn module7.app:create_app --factory --reload --port 8000
```

### Step 2: Access in Browser
Open your browser and navigate to:
- **Dashboard UI**: [http://localhost:8000/dashboard/](http://localhost:8000/dashboard/) (or simply [http://localhost:8000/](http://localhost:8000/))
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 5. Automated Testing

Module 8 includes automated tests verifying static file availability, route redirection, MIME types, and UI structural components:

```powershell
pytest tests/test_module8_dashboard.py -v
```
