# Face Attendance — Project Plan

## 1. Objective

Build an attendance system that captures student faces, detects and
recognizes registered students, records attendance in a relational
database, and provides teacher/admin interfaces for attendance review
and reporting.

## 2. Problem

Manual roll calls consume classroom time and can enable proxy
attendance. Manual sign-in also creates data-entry work and
opportunities for errors.

## 3. Proposed Solution

``` mermaid
flowchart TD
    A[Camera] --> B[Capture Frame]
    B --> C[Preprocess]
    C --> D[Detect Face]
    D --> E[Crop Face]
    E --> F[Generate Face Embedding]
    F --> G[Compare with Registered Embeddings]
    G --> H{Known Student?}
    H -->|Yes| I[Identify Student]
    H -->|No| J[Unknown]
    I --> K[Validate Attendance]
    K --> L[Write Attendance]
    L --> M[(Database)]
    M --> N[Teacher/Admin Dashboard]
```

## 4. Development Modules

### Module 0 — Project Setup

- Repository structure
- Python environment
- Git/GitHub workflow
- README and documentation

### Module 1 — Camera and Data Capture

- Open webcam
- Capture frames
- Convert frames to grayscale
- Resize images
- Basic image preprocessing
- Save/test captured frames

### Module 2 — Face Detection

- Learn Haar Cascade
- Load detector
- Detect faces
- Draw bounding boxes
- Crop detected faces
- Handle multiple faces

### Module 3 — Student Registration

- Student registration flow
- Capture multiple face samples
- Label samples using Student_ID
- Validate samples
- Organize student face data

### Module 4 — Face Recognition

- Learn neural-network/CNN fundamentals
- Learn PyTorch inference
- Learn FaceNet
- Generate face embeddings
- Create stable registered representations
- Store/retrieve embeddings

### Module 5 — Face Matching

- Learn cosine similarity and/or Euclidean distance
- Compare query embeddings with registered embeddings
- Select best candidate
- Evaluate matching threshold
- Handle unknown faces
- Support multiple recognized faces

### Module 6 — Attendance Database

- Learn SQL
- Create PostgreSQL database
- Create Students table
- Create Attendance table
- Store student information
- Store face representations
- Record date/time/status
- Prevent duplicate attendance
- Generate daily/monthly queries

### Module 7 — Backend API

- Learn REST and HTTP
- Build FastAPI application
- Student APIs
- Recognition endpoint
- Attendance APIs
- Database integration
- Validation and error handling
- Test APIs with Postman

### Module 8 — Teacher/Admin Dashboard

- HTML
- CSS
- JavaScript
- Today's attendance
- Student search
- Attendance filters
- Attendance percentage
- Mismatch review
- Report export

### Module 9 — Security and Privacy

- Authentication
- Authorization/roles
- Protect student information
- Protect face embeddings
- Consent/privacy documentation
- Secure configuration and secrets

### Module 10 — Testing

- Unit tests
- Face detection tests
- Recognition tests
- Unknown-face tests
- Lighting tests
- Pose/angle tests
- Multiple-face tests
- Duplicate-attendance tests
- End-to-end tests

### Module 11 — Technical Documentation

- Use Case Diagram
- Sequence Diagram
- Data Flow Diagram
- ER Diagram
- State Transition Diagram
- Technical Workflow Diagram
- System Architecture Diagram

### Module 12 — Deployment

- Docker
- Backend container
- PostgreSQL container
- Docker Compose
- Environment variables
- Deployment documentation

### Module 13 — CI/CD

- GitHub Actions
- Automated tests
- Pull-request checks
- Protected main branch

## 5. End-to-End Architecture

``` mermaid
flowchart TD
    A[Student / Classroom] --> B[Webcam]
    B --> C[OpenCV Frame Capture]
    C --> D[Image Preprocessing]
    D --> E[Face Detection]
    E --> F[Face Crop]
    F --> G[FaceNet]
    G --> H[Face Embedding]
    H --> I[Similarity Matching]
    I --> J{Known?}
    J -->|Yes| K[Student ID]
    J -->|No| L[Unknown]
    K --> M[FastAPI]
    M --> N[Attendance Validation]
    N --> O[(PostgreSQL)]
    O --> P[Teacher/Admin Dashboard]
    L --> P
```

## 6. Development Order

``` text
Project Setup
      ↓
Camera
      ↓
Image Processing
      ↓
Face Detection
      ↓
Student Registration
      ↓
PyTorch Fundamentals
      ↓
FaceNet
      ↓
Embeddings
      ↓
Matching
      ↓
PostgreSQL
      ↓
Attendance
      ↓
FastAPI
      ↓
Dashboard
      ↓
Testing
      ↓
Docker
      ↓
CI/CD
```

## 7. Definition of Done

A task is complete only when:

- Required technology has been learned.
- Implementation works.
- Implementation is tested.
- Documentation is updated.
- Code is committed to a feature branch.
- Pull Request is created when required.
- Review comments are resolved.
- Change is merged into `main`.

## 8. Project Board

``` text
TODO → IN PROGRESS → REVIEW → DONE
```

## 9. Source Alignment

The plan follows the Face Attendance synopsis: capture and label faces,
choose a face-embedding model such as FaceNet/ArcFace, create stable
representations, perform real-time inference, record attendance through
a backend/database, and provide teacher/admin interfaces.
