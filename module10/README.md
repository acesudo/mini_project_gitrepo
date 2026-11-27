# Module 10 — Comprehensive Testing & Quality Verification

Robustness evaluation framework for the **Face Attendance** system. Validates edge cases including illumination variance, head rotations, multi-student recognition, duplicate handling, and full end-to-end workflows.

---

## 1. Test Architecture & Coverage

The testing framework is structured across three dedicated test suites in `tests/`:

```
tests/
|-- test_module10_detection.py   # Haar Cascade & image quality gates (blur, illumination, geometry)
|-- test_module10_robustness.py  # Lighting, pose rotation, multi-face frames, impostors, noise
`-- test_module10_e2e.py         # Full student lifecycle & high-concurrency race condition testing
```

---

## 2. Test Suites Summary

### 2.1 Detection & Image Quality Gates (`test_module10_detection.py`)
- **Haar Frontal Detection**: Evaluates detection on frontal portraits.
- **Laplacian Blur Rejection**: Discards blurry captures with variance $< 50.0$.
- **Illumination Boundaries**: Rejects overexposed (average $> 245$) and pitch-black ($< 15$) frames.
- **Bounding Box Geometry**: Verifies face crops maintain proper aspect ratios and minimum width/height ($80 \times 80$ px).

### 2.2 Robustness & Environmental Variations (`test_module10_robustness.py`)
- **Lighting Sensitivity**:
  - Tests underexposure (50% brightness reduction).
  - Tests glare / overexposure (150% brightness boost).
  - Validates pre-processing normalization preserves cosine similarity above threshold $\tau = 0.70$.
- **Pose Angle Invariance**:
  - In-plane rotations ($\pm 15^\circ$, $\pm 30^\circ$).
  - Validates stability for natural student posture changes.
- **Multi-Subject Detection**:
  - Synthesizes frames containing 3 distinct students simultaneously.
  - Verifies all 3 faces are bounded, cropped, and recognized independently.
- **Adversarial & Impostor Defense**:
  - Zero-shot rejection of unregistered subjects (marked as `Unknown`).
  - Gaussian noise and texture frames rejected without false positive triggers.

### 2.3 End-to-End Workflow & Stress Testing (`test_module10_e2e.py`)
- **Complete Student Lifecycle**:
  1. Student registration via REST API.
  2. Multi-sample enrollment with 5 face crops.
  3. Feature embedding aggregation and database storage.
  4. Camera frame ingestion with Haar face detection.
  5. Cosine similarity matching against gallery.
  6. Attendance record generation (`Present`).
  7. Duplicate entry suppression on subsequent frame capture.
  8. Today's attendance listing & monthly defaulters report.
  9. Cascade deletion ("Right to Be Forgotten").
- **Concurrency Stress Test**:
  - Rapid parallel attendance requests testing SQLite/PostgreSQL transaction isolation.

---

## 3. Running Module 10 Tests

Run the dedicated Module 10 suites:
```powershell
pytest tests/test_module10_detection.py tests/test_module10_robustness.py tests/test_module10_e2e.py -v
```

Run the complete project test suite (all modules):
```powershell
pytest tests/ -v
```
