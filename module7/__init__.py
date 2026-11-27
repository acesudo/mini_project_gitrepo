"""
Module 7 — Backend API (FastAPI)

REST API that exposes the Face Attendance system (Modules 2-6) over HTTP:
student management, face recognition, and attendance reporting.

Run the server from the project root:
    uvicorn module7.main:app --reload
Then open http://127.0.0.1:8000/docs for interactive API documentation.
"""

from .app import create_app

__all__ = ["create_app"]
