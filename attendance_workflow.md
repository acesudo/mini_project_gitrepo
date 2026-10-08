# Attendance Workflow

## Step 10 — Attendance Recognition and Recording

### Technologies

- OpenCV
- FaceNet
- PyTorch
- NumPy
- FastAPI
- PostgreSQL
- HTML/CSS/JavaScript

## Diagram

``` mermaid
sequenceDiagram
    participant Camera
    participant OpenCV
    participant FaceNet
    participant Matcher
    participant API as FastAPI
    participant DB as PostgreSQL
    participant Dashboard

    Camera->>OpenCV: Capture frame
    OpenCV->>OpenCV: Detect faces
    OpenCV->>OpenCV: Crop face
    OpenCV->>FaceNet: Face image
    FaceNet->>FaceNet: Generate embedding
    FaceNet-->>Matcher: Query embedding
    Matcher->>DB: Get registered embeddings
    DB-->>Matcher: Registered embeddings
    Matcher->>Matcher: Calculate similarity

    alt Match accepted
        Matcher->>API: Student ID
        API->>DB: Check today's attendance
        alt Not already present
            DB->>DB: Insert attendance
            DB-->>API: Saved
        else Already present
            DB-->>API: Duplicate prevented
        end
        API-->>Dashboard: Attendance status
    else Match rejected
        Matcher-->>Dashboard: Unknown person
    end
```

## Attendance Record

``` text
Attendance_ID
Student_ID
Date
Time
Status
```

## Acceptance Criteria

- Camera captures frames.
- Faces are detected.
- Each detected face can be processed.
- FaceNet generates embeddings.
- Embeddings are compared with registered embeddings.
- A matching student can be identified.
- Unknown faces are not marked present.
- Duplicate attendance is prevented.
- Date/time are stored.
- Dashboard receives the result.
