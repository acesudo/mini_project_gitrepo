"""
Module 7 — FastAPI application factory

create_app() wires together Module 6 (database), Module 5 (matching) and the
routers. Keeping it a function (instead of a global) lets tests build an app
with an isolated in-memory database.
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from module5.face_matcher import FaceMatcher
from module6.attendance_manager import AttendanceManager
from module7.dependencies import AppState
from module7.routers import attendance, health, recognition, students
from module9.middleware import SecurityHeadersMiddleware
from module9.router import router as auth_router
from module9.user_manager import UserManager

DESCRIPTION = """
Backend for the **Face Attendance** system.

* **Students** — register students and enroll their faces
* **Recognition** — upload a camera frame, identify students, mark attendance
* **Attendance** — daily records, student history, monthly percentages

Environment variables: `DATABASE_URL`, `FACE_MATCH_THRESHOLD` (default 0.70),
`CORS_ORIGINS` (comma-separated, default `*`).
"""


def create_app(
    engine=None,
    matcher: Optional[FaceMatcher] = None,
    manager: Optional[AttendanceManager] = None,
) -> FastAPI:
    """
    Build the FastAPI app.

    Args:
        engine:  SQLAlchemy engine (default: DATABASE_URL, or local SQLite).
        matcher: Pre-built FaceMatcher (tests inject one with a fake generator).
        manager: Pre-built AttendanceManager.
    """
    app = FastAPI(title="Face Attendance API", version="1.0.0", description=DESCRIPTION)

    manager = manager or AttendanceManager(engine=engine)
    if matcher is None:
        threshold = float(os.getenv("FACE_MATCH_THRESHOLD", "0.70"))
        matcher = FaceMatcher(cosine_threshold=threshold)  # FaceNet loads lazily on first use

    state = AppState(manager=manager, matcher=matcher)
    state.reload_gallery()  # load registered faces from the database
    app.state.core = state

    # Seed initial administrator account for Module 9
    try:
        with manager.get_session() as session:
            UserManager(session).seed_initial_admin()
    except Exception:
        pass

    # Security headers middleware (Module 9)
    app.add_middleware(SecurityHeadersMiddleware)

    # Allow the Module 8 dashboard (served from another origin) to call this API.
    origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(students.router)
    app.include_router(recognition.router)
    app.include_router(attendance.router)
    app.include_router(auth_router)

    # Mount Module 8 Dashboard static files if directory exists
    dashboard_dir = Path(__file__).resolve().parent.parent / "module8" / "static"
    if dashboard_dir.is_dir():
        app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")

        @app.get("/", include_in_schema=False)
        def root_redirect():
            return RedirectResponse(url="/dashboard/")

    return app
