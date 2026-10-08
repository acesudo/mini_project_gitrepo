# Student Registration Workflow

## Step 9 — Student Registration

### Technologies

- Python
- OpenCV
- NumPy
- FaceNet
- PyTorch
- PostgreSQL

## Diagram

``` mermaid
flowchart TD
    A[Teacher/Admin] --> B[Enter Student Details]
    B --> C[Student ID]
    B --> D[Name]
    B --> E[Department]
    B --> F[Semester]
    C --> G[Start Face Capture]
    D --> G
    E --> G
    F --> G
    G --> H[Capture Multiple Images]
    H --> I[Detect Face]
    I --> J{Valid Sample?}
    J -->|No| K[Reject / Recapture]
    K --> H
    J -->|Yes| L[Crop Face]
    L --> M[Preprocess]
    M --> N[FaceNet]
    N --> O[Generate Embedding]
    O --> P[Create Stable Student Representation]
    P --> Q[(PostgreSQL)]
```

## Data

``` text
Student_ID
Name
Department
Semester
Face representation / embedding
```

## Acceptance Criteria

- Student details can be entered.
- Multiple face samples can be captured.
- Invalid samples can be rejected.
- Face crops are generated.
- FaceNet can generate embeddings.
- Registered representation can be stored.
- Student information can be retrieved from PostgreSQL.

## Learn Before Coding

1.  OpenCV capture
2.  Face detection
3.  NumPy arrays
4.  PyTorch tensors/inference
5.  FaceNet embeddings
6.  PostgreSQL INSERT/SELECT
