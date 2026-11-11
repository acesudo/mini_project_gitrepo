
import cv2

from module2.harr_classifier import detect_faces
from module3.face_preprocessing import preprocess_face


def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Face preprocessing integration started.")
    print("Press Q to exit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Frame capture failed.")
            break

        frame = cv2.flip(frame, 1)

        # Module 2: Detect faces
        faces = detect_faces(frame)

        output = frame.copy()

        for index, bbox in enumerate(faces):

            x, y, w, h = map(int, bbox)

            try:

                # Module 3: Crop, resize and convert to RGB
                processed_face = preprocess_face(
                    frame,
                    bbox,
                    size=(160, 160),
                    padding=0.15
                )

                # Convert RGB back to BGR for display
                preview = cv2.cvtColor(
                    processed_face,
                    cv2.COLOR_RGB2BGR
                )

                # Draw detection rectangle
                cv2.rectangle(
                    output,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    output,
                    f"Face {index + 1}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # Display processed face
                cv2.imshow(
                    f"Processed Face {index + 1}",
                    preview
                )

            except ValueError as error:
                print(f"Processing error: {error}")

        cv2.putText(
            output,
            f"Faces Detected: {len(faces)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Face Detection", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
