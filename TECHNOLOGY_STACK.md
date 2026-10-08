# Face Attendance — Technology Stack

## 1. Stack

| Layer                | Technology          | Purpose                                |
|----------------------|---------------------|----------------------------------------|
| Language             | Python              | Main implementation                    |
| Version Control      | Git                 | Source control                         |
| Collaboration        | GitHub              | Repository, Issues, PRs, Project board |
| Computer Vision      | OpenCV              | Camera and image processing            |
| Numerical Processing | NumPy               | Image arrays and numerical operations  |
| Face Detection       | OpenCV Haar Cascade | Initial face detector                  |
| Deep Learning        | PyTorch             | FaceNet inference                      |
| Face Recognition     | FaceNet             | Face embeddings                        |
| Matching             | NumPy               | Similarity/distance calculations       |
| Backend              | FastAPI             | REST API                               |
| API Server           | Uvicorn             | Run FastAPI                            |
| Database             | PostgreSQL          | Student and attendance data            |
| Database Layer       | SQL / SQLAlchemy    | Database interaction                   |
| Frontend             | HTML                | Dashboard structure                    |
| Styling              | CSS                 | Dashboard styling                      |
| Browser Logic        | JavaScript          | API calls and interaction              |
| API Testing          | Postman             | API testing                            |
| Testing              | Pytest              | Automated tests                        |
| Containerization     | Docker              | Packaging                              |
| Orchestration        | Docker Compose      | Local multi-service setup              |
| Documentation        | Markdown            | Documentation                          |
| Diagrams             | Mermaid             | GitHub-rendered diagrams               |
| CI                   | GitHub Actions      | Automated checks                       |

## 2. Technology Architecture

``` mermaid
flowchart TD
    A[Webcam] --> B[OpenCV]
    B --> C[NumPy]
    C --> D[Haar Cascade]
    D --> E[Face Crop]
    E --> F[FaceNet]
    F --> G[PyTorch]
    G --> H[Face Embedding]
    H --> I[Similarity Matching]
    I --> J[FastAPI]
    J --> K[(PostgreSQL)]
    K --> L[HTML + CSS + JavaScript Dashboard]
    N[Postman] --> J
    O[Pytest] --> B
    O --> F
    O --> J
    O --> K
    P[Docker] --> J
    P --> K
    Q[GitHub Actions] --> O
```

## 3. Face Recognition Stack

``` mermaid
flowchart LR
    A[Face Image] --> B[OpenCV]
    B --> C[Face Crop]
    C --> D[PyTorch]
    D --> E[FaceNet]
    E --> F[Face Embedding]
    F --> G[Cosine Similarity / Distance]
    G --> H[Best Match]
    H --> I{Threshold}
    I -->|Accept| J[Student ID]
    I -->|Reject| K[Unknown]
```

## 4. Technology Responsibilities

### Python

Primary language for computer vision, model inference, backend
integration, and automation.

### OpenCV

Camera capture, frame processing, image manipulation, detection,
cropping, and display.

### NumPy

Image arrays and numerical calculations.

### Haar Cascade

Initial face detector.

### PyTorch

Load and execute FaceNet.

### FaceNet

Convert face images into embeddings that can be compared with registered
representations.

### PostgreSQL

Relational storage for students and attendance.

### FastAPI

REST API for backend services.

### HTML/CSS/JavaScript

Teacher/admin dashboard.

### Postman

API testing.

### Pytest

Automated tests.

### Docker

Reproducible packaging and deployment.

### Mermaid

Diagrams directly in GitHub Markdown.

## 5. Model Decision

The project synopsis proposes **FaceNet/ArcFace** as face-embedding
model options.

### Initial implementation

**FaceNet**

### Later experiment

**ArcFace**

The team will learn FaceNet thoroughly before adding an ArcFace
comparison.

## 6. Learning Rule

``` text
Technology
    ↓
Concepts
    ↓
Small Practice
    ↓
Understand Output
    ↓
Project Implementation
    ↓
Testing
```
