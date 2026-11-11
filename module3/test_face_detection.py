
import cv2

# Import function from Module 2
from module2.harr_classifier import detect_faces


def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera could not be opened.")
        return

    print("Module 3 integration test started.")
    print("Press Q to exit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Frame not captured.")
            break

        frame = cv2.flip(frame, 1)

        # Get face coordinates from Module 2
        faces = detect_faces(frame)

        # Display detected faces
        for (x, y, w, h) in faces:

            # Crop detected face
            face_crop = frame[y:y+h, x:x+w]

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"Detected: {len(faces)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Module 3 - Detection Integration", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
