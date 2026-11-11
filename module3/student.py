
from dataclasses import dataclass
import re


@dataclass
class Student:

    student_id: str
    name: str
    department: str
    semester: int

    def __post_init__(self):

        # Normalize input
        self.student_id = self.student_id.strip().upper()
        self.name = self.name.strip()
        self.department = self.department.strip()

        # Validate student ID
        if not re.fullmatch(r"[A-Z0-9_-]{2,30}", self.student_id):
            raise ValueError("Invalid Student ID")

        # Validate student name
        if not 2 <= len(self.name) <= 80:
            raise ValueError("Name must contain 2-80 characters")

        if not any(char.isalpha() for char in self.name):
            raise ValueError("Name must contain letters")

        if not all(char.isalpha() or char in " .'-" for char in self.name):
            raise ValueError("Name contains invalid characters")

        # Validate department
        if not 2 <= len(self.department) <= 60:
            raise ValueError("Department must contain 2-60 characters")

        # Validate semester
        if type(self.semester) is not int or not 1 <= self.semester <= 12:
            raise ValueError("Semester must be an integer between 1 and 12")

    def to_dict(self):

        return {
            "student_id": self.student_id,
            "name": self.name,
            "department": self.department,
            "semester": self.semester
        }


if __name__ == "__main__":

    student = Student(
        student_id="CSE1025",
        name="Kunal Tiwari",
        department="Computer Science",
        semester=3
    )

    print("Student Registered Successfully!")
    print(student.to_dict())
