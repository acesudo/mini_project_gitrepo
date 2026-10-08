# Face Attendance

Automated attendance using computer vision, face recognition, and a
database.

## Project Overview

The system captures faces from a camera, preprocesses frames, detects
faces, generates face embeddings, compares them with registered student
representations, records attendance, and provides teacher/admin
interfaces for review and reports.

## Core Pipeline

``` mermaid
flowchart TD
    A[Camera / Video Stream] --> B[Frame Capture]
    B --> C[Preprocessing]
    C --> D[Face Detection]
    D --> E[Face Cropping]
    E --> F[FaceNet Model]
    F --> G[Face Embedding]
    G --> H[Compare with Registered Embeddings]
    H --> I{Match Found?}
    I -->|Yes| J[Identify Student]
    I -->|No| K[Unknown Person]
    J --> L[Attendance Validation]
    L --> M[Record Attendance]
    M --> N[(PostgreSQL Database)]
    N --> O[Teacher / Admin Dashboard]
    K --> O
```

## Technology Pipeline

``` mermaid
flowchart LR
    A[Webcam] --> B[OpenCV]
    B --> C[NumPy + Image Processing]
    C --> D[OpenCV Haar Cascade]
    D --> E[Face Crop]
    E --> F[FaceNet]
    F --> G[PyTorch]
    G --> H[Embedding]
    H --> I[Cosine Similarity / Distance]
    I --> J[FastAPI]
    J --> K[(PostgreSQL)]
    K --> L[HTML + CSS + JavaScript Dashboard]
```

## Main Modules

1.  Project and GitHub setup
2.  Camera and data capture
3.  Image preprocessing
4.  Face detection
5.  Student registration
6.  Face recognition and embeddings
7.  Face matching
8.  Attendance management
9.  PostgreSQL database
10. FastAPI backend
11. Teacher/admin dashboard
12. Security and privacy
13. Testing
14. Technical documentation
15. Docker deployment
16. GitHub Actions / CI

## Repository Structure

``` text
mini_project_gitrepo/
├── README.md
├── PROJECT_PLAN.md
├── TECHNOLOGY_STACK.md
├── PROJECT_STATUS.md
├── docs/
│   ├── workflow.md
│   ├── student_registration.md
│   ├── attendance_workflow.md
│   ├── database.md
│   └── github_workflow.md
├── src/
│   ├── camera/
│   ├── preprocessing/
│   ├── detection/
│   ├── recognition/
│   ├── attendance/
│   ├── database/
│   ├── api/
│   └── dashboard/
├── tests/
├── data/
├── requirements.txt
└── .gitignore
```

## Development Rule

``` text
Learn Technology
      ↓
Practice Small Example
      ↓
Understand the Concept
      ↓
Implement Project Task
      ↓
Test
      ↓
Commit
      ↓
Pull Request
      ↓
Review
      ↓
Merge
```

The team learns each technology before implementing the related project
feature.

## Documentation

- [Project Plan](PROJECT_PLAN.md)
- [Technology Stack](TECHNOLOGY_STACK.md)
- [Complete Workflow](docs/workflow.md)
- [Student Registration](docs/student_registration.md)
- [Attendance Workflow](docs/attendance_workflow.md)
- [Database Design](docs/database.md)
- [GitHub Development Workflow](docs/github_workflow.md)

## GitHub Board

Recommended statuses:

``` text
TODO → IN PROGRESS → REVIEW → DONE
```

The Project board tracks work. The repository Markdown files explain the
system.

## Model Decision

The project synopsis proposes FaceNet/ArcFace as face-embedding model
options. The initial implementation uses **FaceNet** so the team can
learn one model thoroughly before considering ArcFace as a comparison.

## Source

This documentation is based on the team's Face Attendance project
synopsis and implementation roadmap.
