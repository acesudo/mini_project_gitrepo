"""
Recognition endpoints

POST /recognition/frame      upload a camera image -> detect, recognize, (optionally) mark attendance
POST /recognition/embedding  send a raw 512-d embedding -> match against the gallery
"""

from typing import List

import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile

from module5.face_matcher import MatchResult
from module7.dependencies import AppState, get_state
from module7.schemas import (
    AttendanceResult,
    EmbeddingMatchRequest,
    FaceResult,
    FrameRecognitionOut,
)
from module7.utils import read_image

router = APIRouter(prefix="/recognition", tags=["Recognition"])


def _to_face_result(match: MatchResult, state: AppState, mark: bool) -> FaceResult:
    """Convert a Module 5 MatchResult into an API result, recording attendance if asked."""
    result = FaceResult(**match.to_dict())

    # Unknown faces are NEVER marked present.
    if mark and match.is_known:
        success, message, record = state.manager.mark_attendance(
            match.student_id, confidence=match.confidence
        )
        result.attendance = AttendanceResult(
            marked=success,
            already_marked=(not success and record is not None),
            message=message,
            record=record,
        )
    return result


@router.post("/frame", response_model=FrameRecognitionOut)
def recognize_frame(
    image: UploadFile = File(..., description="A camera frame (JPEG/PNG) that may contain several faces."),
    mark_attendance: bool = Form(True),
    state: AppState = Depends(get_state),
):
    frame = read_image(image)

    with state.lock:
        matches = state.matcher.match_frame(frame)
        results: List[FaceResult] = [_to_face_result(m, state, mark_attendance) for m in matches]

    known = sum(1 for r in results if r.is_known)
    return FrameRecognitionOut(
        faces_detected=len(results),
        known_faces=known,
        unknown_faces=len(results) - known,
        results=results,
    )


@router.post("/embedding", response_model=FaceResult)
def match_embedding(body: EmbeddingMatchRequest, state: AppState = Depends(get_state)):
    query = np.array(body.embedding, dtype=np.float32)
    with state.lock:
        match = state.matcher.match_embedding(query)
        return _to_face_result(match, state, body.mark_attendance)
