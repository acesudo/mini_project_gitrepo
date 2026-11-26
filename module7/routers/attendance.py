"""
Attendance endpoints

GET  /attendance/daily             all records for a date (default: today)
GET  /attendance/student/{id}      one student's history
GET  /attendance/monthly           monthly summary with attendance percentages
POST /attendance/mark              manual attendance entry
"""

from datetime import date as date_type
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from module7 import crud
from module7.dependencies import AppState, get_state
from module7.schemas import AttendanceMarkRequest, AttendanceOut, MonthlySummaryOut

router = APIRouter(prefix="/attendance", tags=["Attendance"])


@router.get("/daily", response_model=List[AttendanceOut])
def daily_attendance(
    on_date: Optional[date_type] = Query(None, alias="date", description="YYYY-MM-DD (default: today)"),
    department: Optional[str] = None,
    state: AppState = Depends(get_state),
):
    return state.manager.get_daily_attendance(target_date=on_date, department=department)


@router.get("/student/{student_id}", response_model=List[AttendanceOut])
def student_history(
    student_id: str,
    start_date: Optional[date_type] = None,
    end_date: Optional[date_type] = None,
    state: AppState = Depends(get_state),
):
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=422,
            detail="start_date must not be after end_date.",
        )
    if not crud.student_exists(state.manager, student_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student '{student_id.strip().upper()}' not found.",
        )
    return state.manager.get_student_attendance_history(student_id, start_date, end_date)


@router.get("/monthly", response_model=List[MonthlySummaryOut])
def monthly_summary(
    year: int = Query(..., ge=2000, le=2100),
    month: int = Query(..., ge=1, le=12),
    department: Optional[str] = None,
    state: AppState = Depends(get_state),
):
    return state.manager.get_monthly_attendance_summary(year, month, department)


@router.post("/mark", response_model=AttendanceOut, status_code=status.HTTP_201_CREATED)
def mark_attendance(body: AttendanceMarkRequest, state: AppState = Depends(get_state)):
    success, message, record = state.manager.mark_attendance(body.student_id, status=body.status)
    if success:
        return record
    if record is not None:  # existing row -> duplicate
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=message)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
