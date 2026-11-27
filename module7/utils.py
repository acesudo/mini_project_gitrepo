"""Module 7 — Image upload helpers."""

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB per uploaded image


def read_image(upload: UploadFile) -> np.ndarray:
    """
    Decode an uploaded image file into an OpenCV BGR array.

    Raises HTTP 400 if the file is empty, too large, or not a valid image.
    """
    data = upload.file.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail=f"'{upload.filename}' is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"'{upload.filename}' is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB.",
        )

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail=f"'{upload.filename}' is not a valid image.")
    return image
