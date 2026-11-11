
import cv2


def crop_face(frame, bbox, padding=0.15):
    """
    Crop a detected face from the original frame.

    bbox format: (x, y, width, height)
    padding: Extra area around the face.
    """

    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty.")

    x, y, w, h = map(int, bbox)

    if w <= 0 or h <= 0:
        raise ValueError("Invalid face dimensions.")

    height, width = frame.shape[:2]

    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)

    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        raise ValueError("Face crop is empty.")

    return face


def resize_face(face, size=(160, 160)):
    """
    Resize face and convert BGR to RGB.
    """

    if face is None or face.size == 0:
        raise ValueError("Cannot resize an empty image.")

    resized = cv2.resize(face, size, interpolation=cv2.INTER_AREA)

    rgb_face = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    return rgb_face


def preprocess_face(frame, bbox, size=(160, 160), padding=0.15):
    """
    Complete face preprocessing pipeline.
    """

    cropped_face = crop_face(frame, bbox, padding)

    processed_face = resize_face(cropped_face, size)

    return processed_face
