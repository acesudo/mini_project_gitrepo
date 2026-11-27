# Module 9 — Security, Authentication & Privacy

Comprehensive security layer providing JSON Web Token (JWT) authentication, Role-Based Access Control (RBAC), biometric embedding integrity signing, security HTTP middleware, and privacy policy documentation.

---

## 1. Key Capabilities

1. **Authentication**:
   - `bcrypt` password hashing with auto-generated salts (12 rounds).
   - Stateless JWT tokens (`HS256`) containing user identity, role, and expiration timestamps.
   - HTTP Bearer authorization header (`Authorization: Bearer <token>`).

2. **Role-Based Access Control (RBAC)**:
   - **`admin`**: Full system administration (manage students, delete records, register system users).
   - **`teacher`**: Read daily attendance logs, view student histories, export monthly defaulters, mark manual overrides.
   - **`kiosk`**: Machine account dedicated to classroom attendance kiosks calling frame recognition.

3. **Biometric Integrity (`BiometricSigner`)**:
   - Cryptographic HMAC-SHA256 signature calculated over the 512-dimensional face embedding vector.
   - Detects out-of-band database tampering or unauthorized biometric vector alterations.

4. **Security Middleware**:
   - Injects defense-in-depth headers into all HTTP responses (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection: 1; mode=block`, `Referrer-Policy: strict-origin-when-cross-origin`).

5. **Privacy & DPDP Compliance**:
   - Explicit zero-raw-image policy: raw camera captures are destroyed immediately after mathematical feature extraction.
   - Full legal consent documentation in `PRIVACY_POLICY.md`.

---

## 2. API Endpoints

| Method | Endpoint | Description | Role Required |
|---|---|---|---|
| `POST` | `/api/v1/auth/login` | Authenticate with username & password; returns JWT token | Public |
| `GET` | `/api/v1/auth/me` | Fetch currently logged-in user profile & role | Authenticated |
| `POST` | `/api/v1/auth/users` | Create a new user (`admin`, `teacher`, `kiosk`) | `admin` |
| `GET` | `/api/v1/auth/users` | List all system accounts | `admin` |

---

## 3. Configuration

Configured via environment variables (see `.env.example`):
- `SECRET_KEY`: Minimum 32-byte secret for JWT signing.
- `ALGORITHM`: Token signature algorithm (`HS256`).
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Token lifetime in minutes (default: 60).
- `AUTH_ENABLED`: Set to `true` to strictly require Bearer tokens across protected endpoints.
- `BIOMETRIC_HMAC_KEY`: Cryptographic key for HMAC embedding verification.

---

## 4. Running Tests

```powershell
pytest tests/test_module9_security.py -v
```
