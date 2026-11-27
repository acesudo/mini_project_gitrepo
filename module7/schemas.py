"""
Module 7 — Request & Response Schemas (Pydantic)

These classes define the exact JSON shape of every API request and response.
FastAPI uses them to validate input automatically and to generate /docs.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------- Students
class StudentCreate(BaseModel):
    """Body for POST /students."""
    student_id: str = Field(..., min_length=2, max_length=30, examples=["CSE1025"])
    name: str = Field(..., min_length=2, max_length=80, examples=["Kunal Tiwari"])
    department: str = Field(..., min_length=2, max_length=60, examples=["Computer Science"])
    semester: int = Field(..., ge=1, le=12, examples=[3])


class StudentUpdate(BaseModel):
    """Body for PUT /students/{student_id} (the ID itself cannot change)."""
    name: str = Field(..., min_length=2, max_length=80)
    department: str = Field(..., min_length=2, max_length=60)
    semester: int = Field(..., ge=1, le=12)


class StudentOut(BaseModel):
    student_id: str
    name: str
    department: str
    semester: int
    has_embedding: bool
    created_at: Optional[str] = None


class RejectedSample(BaseModel):
    filename: str
    reason: str


class FaceRegistrationOut(BaseModel):
    """Response of POST /students/register-with-faces."""
    student: StudentOut
    re_enrolled: bool = Field(..., description="True if the student already existed and was updated.")
    samples_received: int
    samples_used: int
    rejected_samples: List[RejectedSample]


# -------------------------------------------------------------- Attendance
class AttendanceOut(BaseModel):
    attendance_id: int
    student_id: str
    student_name: Optional[str] = None
    department: Optional[str] = None
    semester: Optional[int] = None
    date: str
    time: Optional[str] = None
    status: str
    confidence: Optional[float] = None
    created_at: Optional[str] = None


class AttendanceMarkRequest(BaseModel):
    """Body for POST /attendance/mark (manual entry by teacher/admin)."""
    student_id: str = Field(..., min_length=2, max_length=30)
    status: Literal["Present", "Late"] = "Present"


class MonthlySummaryOut(BaseModel):
    student_id: str
    name: str
    department: str
    semester: int
    month: int
    year: int
    total_active_days: int
    present_days: int
    attendance_percentage: float


# ------------------------------------------------------------- Recognition
class AttendanceResult(BaseModel):
    """What happened to attendance for one recognized face."""
    marked: bool = Field(..., description="True if a NEW attendance record was created.")
    already_marked: bool = Field(..., description="True if the student was already present today.")
    message: str
    record: Optional[AttendanceOut] = None


class FaceResult(BaseModel):
    """Recognition decision for one detected face."""
    student_id: Optional[str]
    confidence: float
    distance: float
    is_known: bool
    bbox: Optional[List[int]] = Field(None, description="[x, y, width, height] in the uploaded image.")
    attendance: Optional[AttendanceResult] = None


class FrameRecognitionOut(BaseModel):
    faces_detected: int
    known_faces: int
    unknown_faces: int
    results: List[FaceResult]


class EmbeddingMatchRequest(BaseModel):
    """Body for POST /recognition/embedding (match a raw 512-d vector)."""
    embedding: List[float] = Field(..., min_length=512, max_length=512)
    mark_attendance: bool = False

    @field_validator("embedding")
    @classmethod
    def _not_all_zero(cls, v: List[float]) -> List[float]:
        if not any(v):
            raise ValueError("embedding cannot be all zeros")
        return v


# ------------------------------------------------------------------ Health
class HealthOut(BaseModel):
    status: str
    database: str
    registered_faces: int
    match_threshold: float
