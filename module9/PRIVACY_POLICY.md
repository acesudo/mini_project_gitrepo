# Face Attendance — Biometric Data Privacy & Protection Policy

**Effective Date:** 2026-11-27  
**Compliance Framework:** Digital Personal Data Protection (DPDP) Act & General Data Protection Regulation (GDPR) Biometric Safeguards.

---

## 1. Purpose & Scope
This policy outlines the biometric data governance, privacy protections, and security safeguards implemented across the **Face Attendance** system for student and faculty attendance management.

---

## 2. Guiding Privacy Principles

### 2.1 Storage Limitation & Irreversibility
- **Zero Raw Image Retention**: Raw webcam frames and cropped facial images used during student enrollment are processed strictly in volatile memory. Immediately following 512-dimensional feature embedding generation via FaceNet, all raw images are permanently destroyed.
- **Mathematical Abstraction**: The stored embedding vector represents a non-invertible directional coordinate in Euclidean space. It is mathematically impossible to reconstruct the student's original facial photograph from the 512-d floating point values.

### 2.2 Biometric Integrity Signing
- Every face representation stored in the system is cryptographically signed using **HMAC-SHA256**.
- The signature binds the student ID to the feature vector, immediately identifying any out-of-band tampering or database injection attempts.

### 2.3 Purpose Limitation
- Biometric embeddings are strictly utilized for attendance verification within the academic institution.
- Facial biometric records will never be sold, leased, shared with third-party vendors, or cross-referenced for non-academic tracking.

### 2.4 Right to Erasure ("Right to Be Forgotten")
- Upon student graduation, transfer, or formal withdrawal of consent, a single deletion request triggers an immediate **cascade purge**.
- Deleting a student record via `DELETE /api/v1/students/{id}` permanently deletes:
  1. The student profile.
  2. All associated daily attendance records.
  3. The 512-d facial embedding vector.
  4. The in-memory FaceNet matching gallery cache.

---

## 3. Data Flow & Security Controls

```
+----------------+      Volatile RAM      +--------------------+
|  Camera Frame  | ---------------------> | Face Crop Preproc  |
+----------------+                        +--------------------+
                                                     |
                                                     v
                                          +--------------------+
                                          | FaceNet Embedding  |
                                          |  (512-d Vector)    |
                                          +--------------------+
                                                     |
                                                     v
  Raw Image Wiped From Memory <----------------------+
                                                     |
                                                     v
+-----------------------+     HMAC-SHA256     +--------------------+
| Database (Embeddings) | <------------------ | Cryptographic Sign |
+-----------------------+                     +--------------------+
```

---

## 4. Student Consent Statement Template

> *"I hereby authorize the institution to process my facial feature embeddings solely for the purpose of automated classroom attendance recording. I understand that my photographic images are not retained, that my biometric representation is protected with cryptographic integrity checks, and that I retain the right to request deletion upon departure."*
