"""
Module 8 Dashboard tests.

Verifies:
1. Static files exist (index.html, css/style.css, js/app.js).
2. FastAPI mounts /dashboard/ and serves index.html (HTTP 200).
3. FastAPI mounts /dashboard/css/style.css and /dashboard/js/app.js with proper content types.
4. GET / redirects to /dashboard/ (HTTP 307).
5. HTML file contains essential UI components:
   - Live Attendance Kiosk / WebRTC camera stream
   - Today's Live Attendance table & CSV export
   - Student Directory & Search section
   - Monthly Attendance & Defaulters Analytics
   - Student Enrollment form
   - History modal & Toast container
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from module7.app import create_app
from module6.attendance_manager import AttendanceManager
from module5.face_matcher import FaceMatcher


@pytest.fixture
def client(tmp_path):
    """FastAPI TestClient with in-memory database."""
    from sqlalchemy import create_engine
    test_db = tmp_path / "test_dash.db"
    engine = create_engine(f"sqlite:///{test_db}")
    manager = AttendanceManager(engine=engine)
    matcher = FaceMatcher()
    app = create_app(engine=engine, manager=manager, matcher=matcher)
    return TestClient(app)


def test_static_files_exist():
    """Verify all dashboard assets are present on disk."""
    base = Path(__file__).resolve().parent.parent / "module8" / "static"
    assert (base / "index.html").is_file()
    assert (base / "css" / "style.css").is_file()
    assert (base / "js" / "app.js").is_file()


def test_root_redirects_to_dashboard(client):
    """Visiting / should redirect to /dashboard/."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/dashboard/"


def test_dashboard_serves_html(client):
    """Visiting /dashboard/ should return index.html with 200 OK."""
    response = client.get("/dashboard/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Face Attendance" in response.text
    assert "Live Attendance Kiosk" in response.text


def test_dashboard_serves_css_and_js(client):
    """Static assets should be served with appropriate content types."""
    css_resp = client.get("/dashboard/css/style.css")
    assert css_resp.status_code == 200
    assert "text/css" in css_resp.headers["content-type"]

    js_resp = client.get("/dashboard/js/app.js")
    assert js_resp.status_code == 200
    assert "javascript" in js_resp.headers["content-type"]


def test_dashboard_ui_elements_present():
    """Verify index.html contains all required functional components."""
    html_path = Path(__file__).resolve().parent.parent / "module8" / "static" / "index.html"
    content = html_path.read_text(encoding="utf-8")

    # Navigation / Views
    assert 'data-tab="kiosk"' in content
    assert 'data-tab="today"' in content
    assert 'data-tab="directory"' in content
    assert 'data-tab="enroll"' in content
    assert 'data-tab="monthly"' in content

    # KPI elements
    assert 'id="kpiTotalStudents"' in content
    assert 'id="kpiPresentToday"' in content
    assert 'id="kpiAbsentToday"' in content
    assert 'id="kpiRate"' in content

    # Kiosk elements
    assert 'id="cameraVideo"' in content
    assert 'id="overlayCanvas"' in content
    assert 'id="btnStartCamera"' in content
    assert 'id="btnToggleCamera"' in content
    assert 'id="liveDetectionLog"' in content

    # Tables & Actions
    assert 'id="btnExportTodayCSV"' in content
    assert 'id="searchToday"' in content
    assert 'id="historyModal"' in content
    assert 'id="toastContainer"' in content
