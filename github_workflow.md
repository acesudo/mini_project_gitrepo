# GitHub Development Workflow

## Step 12 — Connect Documentation, Issues, Branches and Project Board

## Relationship

``` mermaid
flowchart TD
    A[PROJECT_PLAN.md] --> B[Development Module]
    B --> C[GitHub Issue]
    C --> D[Assigned Developer]
    D --> E[Feature Branch]
    E --> F[Code + Tests]
    F --> G[Pull Request]
    G --> H[Code Review]
    H --> I{Approved?}
    I -->|No| J[Requested Changes]
    J --> E
    I -->|Yes| K[Merge to main]
    K --> L[Issue Closed]
    L --> M[Project Board → DONE]
    F --> N[Documentation Update]
    N --> A
```

## Board

``` text
TODO → IN PROGRESS → REVIEW → DONE
```

## Issue Structure

``` markdown
## Objective

## Technology

## What to Learn

## Implementation Requirements

## Acceptance Criteria

## Expected Output

## Related Documentation
```

## Example Issue

``` markdown
# Capture webcam frames

## Technology
- Python
- OpenCV

## What to Learn
- cv2.VideoCapture
- frames
- cv2.imshow
- cv2.waitKey
- cv2.release

## Acceptance Criteria
- [ ] Camera opens
- [ ] Frames are captured
- [ ] Frames are displayed
- [ ] Camera closes correctly
```

## Branch Naming

``` text
feature/camera-capture
feature/face-detection
feature/student-registration
feature/facenet-recognition
feature/attendance-database
feature/fastapi-backend
feature/dashboard
```

## Commit Examples

``` text
feat: add webcam capture
feat: add face detection
feat: add FaceNet embedding generation
feat: add attendance database
fix: prevent duplicate attendance
test: add face matching tests
docs: update architecture diagram
```

## Pull Request Flow

``` text
Issue
  ↓
Branch
  ↓
Code
  ↓
Test
  ↓
Commit
  ↓
Push
  ↓
Pull Request
  ↓
Review
  ↓
GitHub Actions
  ↓
Merge
  ↓
DONE
```

## Documentation Rule

When implementation changes the architecture, database, workflow, API,
or technology stack, update the corresponding Markdown document in the
same Pull Request.
