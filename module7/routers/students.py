"""
Student endpoints

POST   /students                      create a student (no face yet)
POST   /students/register-with-faces  create/update a student AND enroll their face
GET    /students                      list students
GET    /students/{student_id}         get one student
PUT    /students/{student_id}         update name / department / semester
DELETE /students/{student_id}         delete student + their attendance
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status

from module2.harr_classifier import detect_faces
from module3.face_preprocessing import preprocess_face
from module3.face_quality import validate_face_image
from module3.student import Student as StudentRules  # same validation rules as Module 3
from module7 import crud
from module7.dependencies import AppState, get_state
from module7.schemas import (
    FaceRegistrationOut,
    RejectedSample,
    StudentCreate,
    StudentOut,
    StudentUpdate,
)
from module7.utils import read_image

router = APIRouter(prefix="/students", tags=["Students"])

MIN_FACE_SAMPLES = 3  # Module 3 captures 5; we require at least 3 usable ones


def _validate_fields(student_id: str, name: str, department: str, semester: int) -> StudentRules:
    """Apply the Module 3 student rules; turn failures into HTTP 422."""
    try:
        return StudentRules(
            student_id=student_id, name=name, department=department, semester=semester
        )
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))


def _not_found(student_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Student '{student_id.strip().upper()}' not found.",
    )


@router.post("", response_model=StudentOut, status_code=status.HTTP_201_CREATED)
def create_student(body: StudentCreate, state: AppState = Depends(get_state)):
    rules = _validate_fields(body.student_id, body.name, body.department, body.semester)
    if crud.student_exists(state.manager, rules.student_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Student '{rules.student_id}' already exists.",
        )
    return state.manager.register_student(**rules.to_dict())


@router.post(
    "/register-with-faces",
    response_model=FaceRegistrationOut,
    status_code=status.HTTP_201_CREATED,
)
def register_with_faces(
    student_id: str = Form(...),
    name: str = Form(...),
    department: str = Form(...),
    semester: int = Form(...),
    images: List[UploadFile] = File(..., description="3 or more photos containing ONE face each."),
    state: AppState = Depends(get_state),
):
    """
    Full registration pipeline (Modules 2 -> 3 -> 4 -> 6 -> 5):
    detect face -> quality check -> crop -> FaceNet embedding -> average -> save.
    """
    rules = _validate_fields(student_id, name, department, semester)

    crops, rejected = [], []
    for upload in images:
        filename = upload.filename or "image"
        try:
            frame = read_image(upload)
        except HTTPException as err:
            rejected.append(RejectedSample(filename=filename, reason=err.detail))
            continue

        faces = detect_faces(frame)
        if len(faces) == 0:
            rejected.append(RejectedSample(filename=filename, reason="No face detected."))
            continue
        if len(faces) > 1:
            rejected.append(
                RejectedSample(filename=filename, reason=f"{len(faces)} faces detected; exactly one is required.")
            )
            continue

        try:
            face = preprocess_face(frame, faces[0], size=(160, 160), padding=0.15)
        except ValueError as err:
            rejected.append(RejectedSample(filename=filename, reason=str(err)))
            continue

        ok, message = validate_face_image(face)
        if not ok:
            rejected.append(RejectedSample(filename=filename, reason=message))
            continue
        crops.append(face)

    if len(crops) < MIN_FACE_SAMPLES:
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"Need at least {MIN_FACE_SAMPLES} valid face samples, got {len(crops)}.",
                "rejected_samples": [r.model_dump() for r in rejected],
            },
        )

    already_registered = crud.student_exists(state.manager, rules.student_id)

    with state.lock:  # FaceNet inference + gallery update must not overlap with recognition
        representation = state.matcher.generator.generate_student_representation(crops)
        student = state.manager.register_student(embedding=representation, **rules.to_dict())
    state.reload_gallery()

    return FaceRegistrationOut(
        student=student,
        re_enrolled=already_registered,
        samples_received=len(images),
        samples_used=len(crops),
        rejected_samples=rejected,
    )


@router.get("", response_model=List[StudentOut])
def list_students(
    department: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    state: AppState = Depends(get_state),
):
    return crud.list_students(state.manager, department, limit, offset)


@router.get("/{student_id}", response_model=StudentOut)
def get_student(student_id: str, state: AppState = Depends(get_state)):
    student = crud.get_student(state.manager, student_id)
    if student is None:
        raise _not_found(student_id)
    return student


@router.put("/{student_id}", response_model=StudentOut)
def update_student(student_id: str, body: StudentUpdate, state: AppState = Depends(get_state)):
    if not crud.student_exists(state.manager, student_id):
        raise _not_found(student_id)
    rules = _validate_fields(student_id, body.name, body.department, body.semester)
    return state.manager.register_student(**rules.to_dict())  # keeps the stored face embedding


@router.delete("/{student_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_student(student_id: str, state: AppState = Depends(get_state)):
    if not crud.delete_student(state.manager, student_id):
        raise _not_found(student_id)
    state.reload_gallery()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
