
import cv2
from pathlib import Path

from module2.harr_classifier import detect_faces
from module3.face_preprocessing import preprocess_face
from module3.face_quality import validate_face_image


def capture_face_samples(student_id, target_samples=5):
    output_dir = Path("data/face_samples") / student_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Saving images to:", output_dir.resolve())

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    saved_count = 0
    processed_face = None
    is_valid = False
    quality_message = "No face detected"

    try:
        while saved_count < target_samples:
            ret, frame = cap.read()

            if not ret:
                print("Failed to read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            faces = detect_faces(frame)

            processed_face = None
            is_valid = False
            quality_message = "No face detected"

            display = frame.copy()

            if len(faces) == 1:
                bbox = faces[0]

                try:
                    processed_face = preprocess_face(
                        frame, bbox, size=(160, 160), padding=0.15
                    )

                    is_valid, quality_message = validate_face_image(
                        processed_face
                    )

                    x, y, w, h = map(int, bbox)
                    color = (0, 255, 0) if is_valid else (0, 0, 255)

                    cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)

                    preview = cv2.cvtColor(
                        processed_face, cv2.COLOR_RGB2BGR
                    )
                    cv2.imshow("Processed Face Preview", preview)

                except Exception as error:
                    quality_message = f"Processing error: {error}"
                    print(quality_message)

            elif len(faces) > 1:
                quality_message = "Multiple faces detected"

            cv2.putText(
                display, quality_message, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
            )

            cv2.putText(
                display, f"Saved: {saved_count}/{target_samples}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )

            cv2.imshow("Student Face Registration", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if processed_face is None or not is_valid:
                    print("NOT SAVED:", quality_message)
                    continue

                filename = output_dir / f"sample_{saved_count + 1}.jpg"

                image_to_save = cv2.cvtColor(
                    processed_face, cv2.COLOR_RGB2BGR
                )

                success = cv2.imwrite(str(filename), image_to_save)

                if success:
                    saved_count += 1
                    print("SUCCESS: Image saved at", filename.resolve())
                else:
                    print("ERROR: cv2.imwrite() failed.")

            elif key == ord("q"):
                print("Capture stopped.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Total images saved: {saved_count}")
    return saved_count


if __name__ == "__main__":
    student_id = input("Enter Student ID: ").strip().upper()

    if student_id:
        capture_face_samples(student_id)
    else:
        print("Student ID cannot be empty.")
