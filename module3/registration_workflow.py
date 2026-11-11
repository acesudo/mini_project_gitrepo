
from pathlib import Path

from module3.student import Student
from module3.database import (
    initialize_database,
    add_student,
    add_face_sample,
    get_student
)
from module3.face_capture import capture_face_samples


def register_student():
    initialize_database()

    print("\n===== STUDENT REGISTRATION =====")

    student_id = input("Enter Student ID: ").strip().upper()
    name = input("Enter Student Name: ").strip()
    department = input("Enter Department: ").strip()

    try:
        semester = int(input("Enter Semester: "))

        # Validate student information
        student = Student(
            student_id=student_id,
            name=name,
            department=department,
            semester=semester
        )

        # Check for duplicate registration
        if get_student(student.student_id):
            print("Student already registered!")
            return

        # Save student details
        add_student(student)
        print("\nStudent details saved successfully!")

        # Capture face images
        saved_count = capture_face_samples(
            student.student_id,
            target_samples=5
        )

        # Save actual image paths in database
        image_directory = (
            Path("data") / "face_samples" / student.student_id
        )

        for index in range(1, saved_count + 1):
            image_path = image_directory / f"sample_{index}.jpg"

            if image_path.exists():
                add_face_sample(
                    student.student_id,
                    image_path.resolve()
                )

        print("\n===== REGISTRATION RESULT =====")
        print("Student ID:", student.student_id)
        print("Student Name:", student.name)
        print("Samples captured:", saved_count)
        print("Registration process completed.")

    except ValueError as error:
        print("Registration error:", error)


if __name__ == "__main__":
    register_student()
