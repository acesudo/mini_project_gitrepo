
import cv2
from pathlib import Path


# Get the current module directory
BASE_DIR = Path(__file__).resolve().parent


# Load Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    str(BASE_DIR / "haarcascade_eye.xml")
)

smile_cascade = cv2.CascadeClassifier(
    str(BASE_DIR / "haarcascade_smile.xml")
)


# Validate classifiers
if face_cascade.empty():
    raise FileNotFoundError("Face Cascade XML not found.")

if eye_cascade.empty():
    raise FileNotFoundError("Eye Cascade XML not found.")

if smile_cascade.empty():
    raise FileNotFoundError("Smile Cascade XML not found.")


# Reusable face detection function
def detect_faces(image):

    # Convert BGR image into grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # Return actual coordinates
    return faces


# Reusable eye detection function
def detect_eyes(image):

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    return eyes


# Reusable smile detection function
def detect_smiles(image):

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    smiles = smile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.7,
        minNeighbors=20
    )

    return smiles


# Standalone testing function
def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Face detection started. Press Q to exit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Could not read camera frame.")
            break

        frame = cv2.flip(frame, 1)

        # Call reusable face detection
        faces = detect_faces(frame)

        # Draw rectangles
        for (x, y, w, h) in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"Faces Detected: {len(faces)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Reusable Face Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# Webcam only starts when this file is executed directly
if __name__ == "__main__":
    main()
