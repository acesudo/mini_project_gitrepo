"""
Module 7 — Shared Application State & Dependencies

One AppState object lives for the whole life of the server. It holds:
  * the AttendanceManager (Module 6) for database access
  * the FaceMatcher (Module 5) with the in-memory gallery of face embeddings
  * a lock, because FastAPI handles several requests at the same time and the
    gallery / FaceNet model must not be changed and used simultaneously.
"""

import threading
from dataclasses import dataclass, field

from fastapi import Request

from module5.face_matcher import FaceMatcher
from module6.attendance_manager import AttendanceManager


@dataclass
class AppState:
    manager: AttendanceManager
    matcher: FaceMatcher
    lock: threading.RLock = field(default_factory=threading.RLock)

    def reload_gallery(self) -> int:
        """
        Rebuild the matcher's gallery from the database.

        Call this after any change to registered faces (register / delete) so the
        database stays the single source of truth. Returns the gallery size.
        """
        embeddings = self.manager.load_all_registered_embeddings()
        with self.lock:
            self.matcher.registered_embeddings.clear()
            self.matcher.register_students(embeddings)
            return len(self.matcher.registered_embeddings)


def get_state(request: Request) -> AppState:
    """FastAPI dependency: returns the shared AppState."""
    return request.app.state.core
