# Face Attendance — Technical Workflows

## 1. Main System Workflow

``` mermaid
flowchart TD
    A[Camera / Video Stream] --> B[Frame Capture]
    B --> C[Preprocessing]
    C --> D[Face Detection]
    D --> E[Face Cropping]
    E --> F[FaceNet]
    F --> G[Face Embedding]
    G --> H[Compare with Registered Embeddings]
    H --> I{Match Found?}
    I -->|Yes| J[Identify Student]
    I -->|No| K[Unknown Person]
    J --> L[Attendance Validation]
    L --> M[Record Attendance]
    M --> N[(PostgreSQL)]
    N --> O[Teacher/Admin Dashboard]
    K --> O
```

## 2. Technology-Aware Workflow

``` mermaid
flowchart TD
    A[Webcam<br/>OpenCV] --> B[Frame Processing<br/>OpenCV + NumPy]
    B --> C[Face Detection<br/>Haar Cascade]
    C --> D[Face Crop<br/>OpenCV]
    D --> E[Face Recognition<br/>FaceNet + PyTorch]
    E --> F[Embedding]
    F --> G[Similarity Matching<br/>NumPy]
    G --> H{Known Student?}
    H -->|Yes| I[Student ID]
    H -->|No| J[Unknown]
    I --> K[FastAPI]
    K --> L[(PostgreSQL)]
    L --> M[Dashboard<br/>HTML + CSS + JavaScript]
    J --> M
```

## 3. Student Registration

``` mermaid
flowchart TD
    A[Teacher/Admin] --> B[Enter Student Information]
    B --> C[Student ID]
    B --> D[Name]
    B --> E[Department]
    B --> F[Semester]
    C --> G[Capture Face Samples]
    D --> G
    E --> G
    F --> G
    G --> H[Face Detection]
    H --> I[Face Preprocessing]
    I --> J[FaceNet]
    J --> K[Generate Embeddings]
    K --> L[Create Registered Face Representation]
    L --> M[(PostgreSQL)]
```

## 4. Attendance Sequence

``` mermaid
sequenceDiagram
    participant C as Camera
    participant O as OpenCV
    participant F as FaceNet
    participant M as Matcher
    participant API as FastAPI
    participant DB as PostgreSQL
    participant D as Dashboard

    C->>O: Capture frame
    O->>O: Detect faces
    O->>O: Crop face
    O->>F: Send face crop
    F->>F: Generate embedding
    F->>M: Return embedding
    M->>DB: Retrieve registered embeddings
    DB-->>M: Registered embeddings
    M->>M: Compare embeddings

    alt Known student
        M->>API: Student ID + match
        API->>DB: Validate attendance
        DB-->>API: Save attendance
        API-->>D: Attendance result
    else Unknown student
        M->>API: Unknown
        API-->>D: Unknown result
    end
```

## 5. Multiple Face Workflow

``` mermaid
flowchart TD
    A[Single Camera Frame] --> B[Detect All Faces]
    B --> C[Face 1]
    B --> D[Face 2]
    B --> E[Face 3]
    C --> F[FaceNet]
    D --> G[FaceNet]
    E --> H[FaceNet]
    F --> I[Match Face 1]
    G --> J[Match Face 2]
    H --> K[Match Face 3]
    I --> L[Attendance Validation]
    J --> L
    K --> L
    L --> M[(Attendance Database)]
```

## 6. Unknown Face Workflow

``` mermaid
flowchart TD
    A[Detected Face] --> B[FaceNet Embedding]
    B --> C[Compare with Registered Embeddings]
    C --> D{Similarity Meets Threshold?}
    D -->|Yes| E[Known Student]
    D -->|No| F[Unknown Person]
    E --> G[Attendance Validation]
    F --> H[Do Not Mark Attendance]
```

## 7. Duplicate Attendance

``` mermaid
flowchart TD
    A[Recognized Student] --> B[Student ID]
    B --> C[Check Today's Attendance]
    C --> D{Already Present?}
    D -->|Yes| E[Do Not Create Duplicate]
    D -->|No| F[Create Attendance Record]
    F --> G[(PostgreSQL)]
```

## 8. System States

``` mermaid
stateDiagram-v2
    [*] --> CameraReady
    CameraReady --> FrameCaptured
    FrameCaptured --> FaceDetected
    FaceDetected --> FaceRecognized
    FaceDetected --> Unknown
    FaceRecognized --> AttendanceChecked
    AttendanceChecked --> AttendanceMarked
    AttendanceChecked --> AlreadyMarked
    AttendanceMarked --> CameraReady
    AlreadyMarked --> CameraReady
    Unknown --> CameraReady
```
