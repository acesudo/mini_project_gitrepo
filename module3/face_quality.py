
import cv2
import numpy as np


def validate_face_image(face):
    """
    Validate a preprocessed RGB face image.

    Returns:
        is_valid: Boolean
        message: Explanation of validation result
    """

    # 1. Check whether image exists
    if face is None or not isinstance(face, np.ndarray) or face.size == 0:
        return False, "Invalid or empty image."

    # 2. Check image dimensions
    if face.shape != (160, 160, 3):
        return False, "Image must have dimensions 160x160x3."

    # 3. Check brightness
    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))

    if brightness < 40:
        return False, "Image is too dark."

    if brightness > 220:
        return False, "Image is too bright."

    # 4. Check blur using Laplacian variance
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    if blur_score < 50:
        return False, "Image is too blurry."

    return True, "Face image quality is acceptable."
