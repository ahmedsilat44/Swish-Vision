# SwishVision — Implementation Guide

> **Project:** SwishVision Basketball Analytics Platform  
> **Version:** 1.0  
> **Team:** Mallick Mikaal Imam · Minhaj ul Hassan · Muhammad Ahmed Silat · Syed Muhammad Sameer Hassan  
> **Jira Project:** [SCRUM @ talisdemha.atlassian.net](https://talisdemha.atlassian.net/jira/software/projects/SCRUM/boards)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Tech Stack & Dependencies](#2-tech-stack--dependencies)
3. [Project Structure](#3-project-structure)
4. [Environment Setup](#4-environment-setup)
5. [Epic 1 — User Authentication & Account Management](#5-epic-1--user-authentication--account-management)
   - [SCRUM-10: User Registration](#scrum-10-user-registration)
   - [SCRUM-11: Login & Logout](#scrum-11-login--logout)
   - [SCRUM-12: Auth Middleware & Ownership Checks](#scrum-12-auth-middleware--ownership-checks)
6. [Epic 2 — Video Upload & Job Queue](#6-epic-2--video-upload--job-queue)
   - [SCRUM-13: Video Upload UI](#scrum-13-video-upload-ui)
   - [SCRUM-14: File Storage & Celery Queue](#scrum-14-file-storage--celery-queue)
   - [SCRUM-15: Session Deletion](#scrum-15-session-deletion)
7. [Epic 3 — CV Pipeline Integration & Processing Worker](#7-epic-3--cv-pipeline-integration--processing-worker)
   - [SCRUM-16: Celery Worker & Pipeline Execution](#scrum-16-celery-worker--pipeline-execution)
   - [SCRUM-17: Pipeline Failure Handling](#scrum-17-pipeline-failure-handling)
   - [SCRUM-18: Report Parsing & DB Persistence](#scrum-18-report-parsing--db-persistence)
8. [Epic 4 — Analytics Dashboard & Results Views](#8-epic-4--analytics-dashboard--results-views)
   - [SCRUM-19: Shot Analytics Screen](#scrum-19-shot-analytics-screen)
   - [SCRUM-20: Form Analysis Screen](#scrum-20-form-analysis-screen)
   - [SCRUM-21: Session History & Trend Chart](#scrum-21-session-history--trend-chart)
   - [SCRUM-22: Home Dashboard](#scrum-22-home-dashboard)
9. [Epic 5 — Database, Security & Infrastructure](#9-epic-5--database-security--infrastructure)
   - [SCRUM-23: Database Schema & ORM Models](#scrum-23-database-schema--orm-models)
   - [SCRUM-24: Security Hardening](#scrum-24-security-hardening)
   - [SCRUM-25: Documentation & In-App Help](#scrum-25-documentation--in-app-help)
10. [API Reference](#10-api-reference)
11. [Database Schema Reference](#11-database-schema-reference)
12. [CV Pipeline Reference](#12-cv-pipeline-reference)
13. [Session State Machine](#13-session-state-machine)
14. [Testing Strategy](#14-testing-strategy)

---

## 1. Architecture Overview

SwishVision follows a three-tier architecture with an asynchronous background processing layer:

```
┌──────────────────────────────────────────┐
│            Client Browser                │
│   (Chrome / Firefox / Safari / Edge)     │
└────────────────────┬─────────────────────┘
                     │  HTTPS / REST
┌────────────────────▼─────────────────────┐
│         Web Application Server           │
│  ┌─────────────────────────────────────┐ │
│  │  FastAPI / Flask Application        │ │
│  │  - Authentication Service (JWT)     │ │
│  │  - Upload Endpoint                  │ │
│  │  - Session / Report API             │ │
│  │  - Dashboard API                    │ │
│  │  - Static Frontend (HTML/CSS/JS)    │ │
│  └─────────────────────────────────────┘ │
└────────────────────┬─────────────────────┘
                     │  Celery Task Queue (Redis)
┌────────────────────▼─────────────────────┐
│        Background Worker Process         │
│  ┌─────────────────────────────────────┐ │
│  │  Celery Worker                      │ │
│  │  - Invokes main_pipeline()          │ │
│  │  - BallTracker / RimTracker         │ │
│  │  - HumanTracker (pose estimation)   │ │
│  │  - ShotTracker                      │ │
│  │  - Report Generator                 │ │
│  └─────────────────────────────────────┘ │
└────────────────────┬─────────────────────┘
                     │  File I/O + ORM
┌────────────────────▼─────────────────────┐
│              Storage Layer               │
│  ┌──────────────────┐  ┌───────────────┐ │
│  │  PostgreSQL DB   │  │  File System  │ │
│  │  (SQLAlchemy)    │  │  - uploads/   │ │
│  │                  │  │  - output_    │ │
│  │                  │  │    videos/    │ │
│  │                  │  │  - reports/   │ │
│  └──────────────────┘  └───────────────┘ │
└──────────────────────────────────────────┘
```

---

## 2. Tech Stack & Dependencies

### Backend
| Package | Version | Purpose |
|---|---|---|
| `fastapi` | `>=0.100` | Web framework & API routing |
| `uvicorn` | latest | ASGI server |
| `sqlalchemy` | `>=2.0` | ORM |
| `alembic` | latest | Database migrations |
| `celery` | `>=5.3` | Async task queue |
| `redis` | latest | Celery broker |
| `python-jose` | latest | JWT signing/verification |
| `passlib[bcrypt]` | latest | Password hashing |
| `python-multipart` | latest | File upload support |
| `psycopg2-binary` | latest | PostgreSQL adapter |

### CV Pipeline (existing)
| Package | Version | Purpose |
|---|---|---|
| `torch` | `>=2.0.0` | PyTorch (CUDA GPU support) |
| `ultralytics` | `>=8.0.0` | YOLOv8 models |
| `opencv-python` | `>=4.8.0` | Frame I/O & annotation |
| `numpy` | `>=1.24.0` | Numerical operations |
| `matplotlib` | `>=3.7.0` | Chart generation |

### Frontend
| Package | Purpose |
|---|---|
| `Chart.js` | Doughnut chart, line charts |
| Vanilla JS / React | UI components |
| Tailwind CSS | Styling |

### Dev/Test
| Package | Purpose |
|---|---|
| `pytest` | Unit & integration testing |
| `httpx` | Async test client for FastAPI |
| `pytest-asyncio` | Async test support |

---

## 3. Project Structure

```
Swish-Vision/
├── main.py                          # CV pipeline entry point
├── trackers/
│   ├── ball_tracker.py              # BallTracker class
│   ├── rim_tracker.py               # RimTracker class
│   └── human_tracker.py             # HumanTracker class (pose estimation)
├── drawers/
│   ├── human_tracks_drawer.py       # HumanTracksDrawer + analysis()
│   └── shot_tracker.py              # ShotTracker class
├── utils/
│   ├── vid_utils.py                 # read_video(), write_video()
│   ├── ball_hand.py                 # ball_hand(), shot_started()
│   └── stubs_utils.py               # Caching for dev/test
├── models/
│   ├── best.pt                      # Fine-tuned YOLOv8 (ball/rim, 3 classes)
│   └── yolov8m-pose.pt              # Pre-trained pose model (17 keypoints)
│
├── input_videos/                    # Worker copies uploaded video here pre-processing
├── output_videos/                   # Pipeline writes annotated .avi output here
├── reports/                         # Pipeline writes {vidname}_report.txt here
│
└── app/                             # ← New web application layer
    ├── main.py                      # FastAPI app factory
    ├── config.py                    # Settings from environment variables
    ├── database.py                  # SQLAlchemy engine & session
    ├── api/
    │   ├── auth.py                  # /api/register, /api/login, /api/logout
    │   ├── sessions.py              # /api/sessions/* endpoints
    │   └── dashboard.py             # /api/dashboard/* endpoints
    ├── models/
    │   ├── user.py                  # User ORM model
    │   ├── session.py               # Session ORM model
    │   ├── report.py                # Report ORM model
    │   ├── shot_event.py            # ShotEvent ORM model
    │   └── angle_frame.py           # AngleFrame ORM model
    ├── schemas/
    │   ├── auth.py                  # Pydantic schemas for auth requests/responses
    │   └── session.py               # Pydantic schemas for session data
    ├── tasks/
    │   └── pipeline_task.py         # Celery task wrapping main_pipeline()
    ├── core/
    │   ├── security.py              # JWT helpers, bcrypt, auth dependency
    │   └── middleware.py            # HTTPS, CSRF, security headers
    ├── migrations/                  # Alembic migration scripts
    │   ├── env.py
    │   └── versions/
    │       └── 001_initial.py
    ├── frontend/
    │   ├── index.html               # Dashboard / Home
    │   ├── upload.html              # Upload page
    │   ├── session.html             # Session results (Shot Analytics + Form tabs)
    │   ├── history.html             # Session history
    │   ├── help.html                # In-app help / onboarding
    │   └── static/
    │       ├── css/
    │       └── js/
    └── uploads/                     # Raw uploaded videos (pre-processing, outside web root)
```

---

## 4. Environment Setup

### Prerequisites
- Python 3.8+
- Node.js (optional, for frontend tooling)
- PostgreSQL (production) or SQLite (local dev)
- Redis
- CUDA-compatible NVIDIA GPU (strongly recommended)

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd Swish-Vision

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Copy and configure environment variables
cp .env.example .env
# Edit .env with your values (see table below)

# 5. Run database migrations
alembic upgrade head

# 6. Start Redis (required for Celery)
redis-server

# 7. Start the Celery worker (in a separate terminal)
celery -A app.tasks.pipeline_task worker --loglevel=info

# 8. Start the web server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Environment Variables (`.env`)

| Variable | Example Value | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://user:pass@localhost/swishvision` | PostgreSQL connection string (use `sqlite:///./dev.db` for local dev) |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis broker URL for Celery |
| `SECRET_KEY` | `change-me-to-a-random-256-bit-secret` | JWT signing secret — **never commit to git** |
| `UPLOAD_DIR` | `/var/swishvision/uploads` | Absolute path for raw uploaded videos |
| `OUTPUT_DIR` | `/var/swishvision/output_videos` | Absolute path for processed output videos |
| `MODEL_DIR` | `/path/to/Swish-Vision/models` | Absolute path to model weight files |
| `MAX_UPLOAD_SIZE_MB` | `500` | Maximum allowed video upload size in MB |

> **Security note:** `UPLOAD_DIR` and `OUTPUT_DIR` must be outside the web root so files are never directly URL-accessible.

---

## 5. Epic 1 — User Authentication & Account Management

**Epic key:** SCRUM-5  
**References:** SRS UC-01, SDS Activity: User Registration, SRS NFR-7.3

---

### SCRUM-10: User Registration

**Story:** As a new user, I can register with name, email, and password so that I can access the platform.

#### Subtasks
| Key | Subtask |
|---|---|
| SCRUM-26 | Build `POST /api/register` endpoint with input validation and bcrypt password hashing |
| SCRUM-27 | Build registration form UI (name, email, password fields with inline validation) |
| SCRUM-28 | Write unit tests for registration endpoint (happy path, duplicate email, weak password) |

#### Backend Implementation

```python
# app/api/auth.py
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.core.security import hash_password, create_access_token
from app.database import get_db
from app.models.user import User
from app.schemas.auth import RegisterRequest, TokenResponse
import re

router = APIRouter(prefix="/api", tags=["auth"])

def validate_password_strength(password: str) -> bool:
    """Minimum 8 chars, at least one letter and one number."""
    return len(password) >= 8 and re.search(r'[A-Za-z]', password) and re.search(r'\d', password)

@router.post("/register", response_model=TokenResponse, status_code=201)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    # Validate email format
    if not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', payload.email):
        raise HTTPException(status_code=422, detail="Invalid email format")
    
    # Check for duplicate email
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    
    # Validate password strength
    if not validate_password_strength(payload.password):
        raise HTTPException(status_code=422, detail="Password too weak")
    
    # Create user with hashed password
    user = User(
        name=payload.name,
        email=payload.email,
        password_hash=hash_password(payload.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    token = create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer"}
```

#### Password Hashing

```python
# app/core/security.py
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict, expires_minutes: int = 60) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=expires_minutes)
    return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")
```

#### Acceptance Criteria
- [ ] Registration form accepts name, email, and password
- [ ] Email format validated; duplicate emails return 409
- [ ] Password strength enforced (min 8 chars, letter + number)
- [ ] Password stored as bcrypt salted hash — never plaintext
- [ ] JWT token returned on success (HTTP 201)

---

### SCRUM-11: Login & Logout

**Story:** As a registered user, I can log in and log out so that my account is secure.

#### Subtasks
| Key | Subtask |
|---|---|
| SCRUM-29 | Build `POST /api/login` endpoint with credential check and JWT token issuance |
| SCRUM-30 | Build `POST /api/logout` endpoint and token invalidation logic |
| SCRUM-31 | Build login/logout UI pages with error state handling |

#### Backend Implementation

```python
# app/api/auth.py (continued)
from app.core.security import verify_password

@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    
    # Use constant-time comparison to prevent timing attacks
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer"}

@router.post("/logout")
def logout(current_user: User = Depends(get_current_user)):
    # For JWT: client-side token removal is sufficient for stateless auth.
    # For server-side invalidation, add token to a Redis blocklist.
    return {"message": "Logged out successfully"}
```

#### Acceptance Criteria
- [ ] Login accepts email and password; incorrect credentials return 401 with generic error
- [ ] JWT/session token issued and stored securely on success
- [ ] Logout endpoint invalidates the token/session
- [ ] UI shows appropriate error states for failed login

---

### SCRUM-12: Auth Middleware & Ownership Checks

**Story:** As the system, all protected API endpoints enforce authentication and ownership checks.

#### Subtasks
| Key | Subtask |
|---|---|
| SCRUM-32 | Implement JWT middleware/dependency that validates token on every protected route |
| SCRUM-33 | Add resource ownership check to session endpoints |

#### Implementation

```python
# app/core/security.py (continued)
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User

bearer_scheme = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db)
) -> User:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def verify_session_ownership(session_id: int, user: User, db: Session):
    """Raises 403 if the session does not belong to the authenticated user."""
    from app.models.session import Session as SessionModel
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    return session
```

#### Acceptance Criteria
- [ ] All protected endpoints return 401 for unauthenticated requests
- [ ] Requests for another user's session return 403
- [ ] JWT middleware applied globally to all routes except `/api/register` and `/api/login`

---

## 6. Epic 2 — Video Upload & Job Queue

**Epic key:** SCRUM-6  
**References:** SRS UC-02, UC-07, SRS NFR-7.1, NFR-7.2, NFR-7.3, SDS Screen 2

---

### SCRUM-13: Video Upload UI

**Story:** As a player, I can upload a basketball practice video (up to 500MB) so that it can be queued for analysis.

#### Subtasks
| Key | Subtask |
|---|---|
| SCRUM-34 | Build upload page UI with drag-and-drop zone, file picker, size/format labels, and progress bar |
| SCRUM-35 | Build `POST /api/sessions/upload` endpoint with MIME/extension validation and 500MB limit |

#### Backend Implementation

```python
# app/api/sessions.py
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.core.security import get_current_user
from app.models.session import Session as SessionModel, SessionStatus
from app.tasks.pipeline_task import run_pipeline
import os, shutil, uuid

ALLOWED_MIME_TYPES = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"}
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_SIZE_BYTES = int(os.environ.get("MAX_UPLOAD_SIZE_MB", 500)) * 1024 * 1024

@router.post("/sessions/upload", status_code=201)
async def upload_video(
    file: UploadFile = File(...),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Validate MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type. Upload a video file.")
    
    # Validate extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Extension {ext} not allowed.")
    
    # Read and check file size
    contents = await file.read()
    if len(contents) > MAX_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 500MB limit.")
    
    # Save to isolated upload directory
    upload_dir = os.environ["UPLOAD_DIR"]
    os.makedirs(upload_dir, exist_ok=True)
    unique_name = f"{uuid.uuid4()}{ext}"
    dest_path = os.path.join(upload_dir, unique_name)
    
    with open(dest_path, "wb") as f:
        f.write(contents)
    
    # Create session record
    session = SessionModel(
        user_id=current_user.id,
        original_filename=file.filename,
        stored_filename=unique_name,
        status=SessionStatus.QUEUED
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    
    # Dispatch to Celery queue
    run_pipeline.delay(session.id)
    
    return {"session_id": session.id, "status": "queued"}
```

#### Frontend Upload UI (key snippet)

```javascript
// Drag-and-drop with progress bar
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    uploadFile(file);
});

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/sessions/upload');
    xhr.setRequestHeader('Authorization', `Bearer ${getToken()}`);
    
    xhr.upload.addEventListener('progress', (e) => {
        const pct = Math.round((e.loaded / e.total) * 100);
        document.getElementById('progress-bar').style.width = `${pct}%`;
        document.getElementById('progress-label').textContent = `${pct}%`;
    });
    
    xhr.onload = () => {
        const res = JSON.parse(xhr.responseText);
        if (xhr.status === 201) {
            window.location.href = `/session.html?id=${res.session_id}`;
        }
    };
    xhr.send(formData);
}
```

#### Acceptance Criteria
- [ ] Upload page has drag-and-drop area and file picker
- [ ] Accepted formats and 500MB limit displayed to the user
- [ ] Progress bar reflects real-time upload progress
- [ ] MIME type and file extension validated server-side
- [ ] On success: session created with status `queued`, user redirected with session link

---

### SCRUM-14: File Storage & Celery Queue

**Story:** As the system, uploaded videos are stored securely and dispatched to the Celery job queue.

#### Subtasks
| Key | Subtask |
|---|---|
| SCRUM-36 | Configure isolated upload directory (outside web root) and file storage logic |
| SCRUM-37 | Set up Celery + Redis worker and integrate task dispatch on successful upload |

#### Celery Configuration

```python
# app/tasks/pipeline_task.py
from celery import Celery
from app.config import settings

celery_app = Celery(
    "swishvision",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    worker_max_tasks_per_child=10,   # Restart worker every 10 tasks (prevent memory leaks)
)

@celery_app.task(bind=True, max_retries=0)
def run_pipeline(self, session_id: int):
    # Implementation detailed in SCRUM-16
    pass
```

#### Starting the Worker

```bash
# Start Celery worker (supports up to 5 concurrent jobs)
celery -A app.tasks.pipeline_task worker \
    --concurrency=5 \
    --loglevel=info \
    --queues=pipeline
```

#### Acceptance Criteria
- [ ] Uploaded files stored in an isolated directory not accessible via URL
- [ ] Celery + Redis task queue dispatches a processing job on upload
- [ ] System handles at least 5 concurrent queued jobs without degradation
- [ ] Session status updates to `queued` immediately after enqueue

---

### SCRUM-15: Session Deletion

**Story:** As a player, I can delete a session so that its data is permanently removed.

#### Subtasks
| Key | Subtask |
|---|---|
| SCRUM-38 | Implement `DELETE /api/sessions/{id}` endpoint with processing-state guard |
| SCRUM-39 | Add delete confirmation button and remove session row from history UI on success |

#### Backend Implementation

```python
# app/api/sessions.py (continued)
@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(
    session_id: int,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = verify_session_ownership(session_id, current_user, db)
    
    # Block deletion if currently processing
    if session.status == SessionStatus.PROCESSING:
        raise HTTPException(status_code=409, detail="Cannot delete a session that is currently processing.")
    
    # Delete associated files
    for path_attr in ["upload_path", "output_video_path", "report_path"]:
        path = getattr(session, path_attr, None)
        if path and os.path.exists(path):
            os.remove(path)
    
    # Cascade delete all DB records (reports, shot_events, angle_frames)
    db.delete(session)
    db.commit()
    return  # 204 No Content
```

#### Acceptance Criteria
- [ ] User can delete their own session and all associated data
- [ ] Deletion blocked with 409 if session status is `processing`
- [ ] Deleted session no longer appears in user history

---

## 7. Epic 3 — CV Pipeline Integration & Processing Worker

**Epic key:** SCRUM-7  
**References:** SRS UC-03, SRS NFR-7.1, NFR-7.2, SDS Activity: Full Pipeline Execution

---

### SCRUM-16: Celery Worker & Pipeline Execution

**Story:** As the system, a Celery worker executes the full 12-step CV pipeline on a queued video.

#### Subtasks
| Key | Subtask |
|---|---|
| SCRUM-40 | Write Celery task wrapper that invokes `main_pipeline()` and updates session status |
| SCRUM-41 | Integrate BallTracker, RimTracker, HumanTracker, and ShotTracker (pipeline steps 1–22) |
| SCRUM-42 | Verify annotated output `.avi` and `_report.txt` are written to correct directories |

#### The 12-Step Pipeline

The pipeline executes the following steps in order. Steps are already implemented in `main.py` — the web layer only needs to wrap them:

| Step | Function | Module |
|---|---|---|
| 1 | Frame extraction | `read_video(video_path)` → `utils/vid_utils.py` |
| 2 | Ball detection | `BallTracker.get_object_tracks(frames)` |
| 3 | Rim detection | `RimTracker` (derived from ball tracks) |
| 4 | Pose estimation | `HumanTracker.detect_frame(frames)` |
| 5 | Track validation | `ball_tracker.remove_wrong_tracks()` |
| 6 | Trajectory interpolation | `ball_tracker.interpolate_missing_tracks()` |
| 7 | Joint angle calculation | `human_tracker.calc_angles(frames, human_tracks)` |
| 8 | Ball-hand detection | `ball_hand(ball_loco, points, frames)` → `utils/ball_hand.py` |
| 9 | Shot start identification | `shot_started(points, ball_left_frames)` |
| 10 | Frame annotation | `HumanTracksDrawer.draw()` |
| 11 | Shot outcome detection | `ShotTracker.detect_shot()` + `draw_shots()` |
| 12 | Output write | `write_video(frames, output_path, fps)` + report `.txt` |

#### Celery Task Implementation

```python
# app/tasks/pipeline_task.py
import sys, os, traceback, logging
sys.path.insert(0, os.environ.get("PIPELINE_ROOT", "."))

from app.database import SessionLocal
from app.models.session import Session as SessionModel, SessionStatus

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, max_retries=0)
def run_pipeline(self, session_id: int):
    db = SessionLocal()
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    
    try:
        # Step 1: Mark as processing
        session.status = SessionStatus.PROCESSING
        db.commit()
        
        # Step 2: Resolve paths
        input_path  = session.upload_path
        output_path = os.path.join(os.environ["OUTPUT_DIR"], f"{session_id}_output.avi")
        report_path = os.path.join("reports", f"{session_id}_report.txt")
        
        # Step 3: Run the pipeline (existing main.py logic)
        from main import main_pipeline
        main_pipeline(
            video_path=input_path,
            output_path=output_path,
            report_path=report_path,
            model_dir=os.environ["MODEL_DIR"]
        )
        
        # Step 4: Verify outputs exist before marking complete
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output video not found: {output_path}")
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"Report not found: {report_path}")
        
        # Step 5: Parse report and persist to DB (see SCRUM-18)
        from app.tasks.report_parser import parse_and_persist
        parse_and_persist(session_id, report_path, db)
        
        # Step 6: Update session as completed
        session.status = SessionStatus.COMPLETED
        session.output_video_path = output_path
        session.report_path = report_path
        db.commit()
        
    except Exception as e:
        logger.error(f"Pipeline failed for session {session_id}: {traceback.format_exc()}")
        session.status = SessionStatus.FAILED
        session.error_message = str(e)
        db.commit()
    finally:
        db.close()
        # Clean up transient intermediate files
        for fname in ["angs.txt", "xy_coords.txt", "detections.txt", "ball_locl.txt"]:
            if os.path.exists(fname):
                os.remove(fname)
```

> **Important:** Always use `best.pt` as the canonical production model. Do **not** load `bestOld.pt`, `bestYT.pt`, or `ballRim.pt` in the production pipeline.

#### Acceptance Criteria
- [ ] Celery task receives `session_id`, sets status to `processing`, invokes `main_pipeline()`
- [ ] All 12 pipeline steps execute in order
- [ ] Uses `best.pt` for detection and `yolov8m-pose.pt` for pose
- [ ] Output `.avi` and `_report.txt` written to correct directories
- [ ] Session status updated to `completed` on success

---

### SCRUM-17: Pipeline Failure Handling

**Story:** As the system, pipeline failures are caught, logged, and marked on the session without crashing the worker.

#### Subtasks
| Key | Subtask |
|---|---|
| SCRUM-43 | Wrap pipeline execution in try/except; set session status to `failed` and store `error_message` |
| SCRUM-44 | Ensure original uploaded video is preserved on failure |

#### Implementation Notes

The `try/except/finally` block in the Celery task (shown above in SCRUM-16) handles this. Key rules:

- **Never delete** the original uploaded file on the error path
- Set `session.status = SessionStatus.FAILED` and store `session.error_message = str(e)`
- Log the full traceback with `logger.error(...)` for debugging
- The `finally` block only cleans up **transient intermediate files**, not the original upload
- If no shots are detected, the pipeline writes a zero-shot-count report — this is **not** a failure

#### Session Status Enum

```python
# app/models/session.py
import enum

class SessionStatus(str, enum.Enum):
    QUEUED      = "queued"
    PROCESSING  = "processing"
    COMPLETED   = "completed"
    FAILED      = "failed"
    DELETED     = "deleted"
```

#### Acceptance Criteria
- [ ] Any unhandled exception sets status to `failed` and stores `error_message`
- [ ] Failed jobs do not crash the worker or affect other queued jobs
- [ ] Original uploaded video is preserved on failure
- [ ] Zero-shot-detection result generates a report (not a failure)

---

### SCRUM-18: Report Parsing & DB Persistence

**Story:** As the system, pipeline output is parsed and persisted to the database.

#### Subtasks
| Key | Subtask |
|---|---|
| SCRUM-45 | Write parser for `_report.txt` and INSERT into `reports` and `shot_events` tables |
| SCRUM-46 | INSERT per-frame joint angle data into `angle_frames` table |
| SCRUM-47 | Delete transient files after successful DB insert |

#### Report Parser

```python
# app/tasks/report_parser.py
import re
from app.models.report import Report
from app.models.shot_event import ShotEvent
from app.models.angle_frame import AngleFrame

def parse_and_persist(session_id: int, report_path: str, db):
    with open(report_path, "r") as f:
        content = f.read()
    
    # Parse summary metrics (adapt regex to actual report format)
    shot_pct   = float(re.search(r'Shot Percentage:\s*([\d.]+)', content).group(1))
    made       = int(re.search(r'Made:\s*(\d+)', content).group(1))
    missed     = int(re.search(r'Missed:\s*(\d+)', content).group(1))
    consistency = float(re.search(r'Consistency Score:\s*([\d.]+)', content).group(1))
    avg_release = float(re.search(r'Avg Release Angle:\s*([\d.]+)', content).group(1))
    feedback   = re.search(r'Feedback:\s*(.+)', content, re.DOTALL).group(1).strip()
    
    # Persist Report
    report = Report(
        session_id=session_id,
        shot_percentage=shot_pct,
        shots_made=made,
        shots_missed=missed,
        consistency_score=consistency,
        avg_release_angle=avg_release,
        feedback_text=feedback
    )
    db.add(report)
    db.flush()  # Get report.id before inserting child records
    
    # Parse and persist individual shot events
    for match in re.finditer(r'Shot (\d+): (made|missed), release=(\d+\.\d+), elbow=(\d+\.\d+)', content):
        shot = ShotEvent(
            session_id=session_id,
            shot_number=int(match.group(1)),
            outcome=match.group(2),
            release_angle=float(match.group(3)),
            elbow_angle_at_release=float(match.group(4))
        )
        db.add(shot)
    
    db.commit()
    
    # Clean up transient files
    for fname in ["angs.txt", "xy_coords.txt", "detections.txt", "ball_locl.txt"]:
        if os.path.exists(fname):
            os.remove(fname)
```

> **Note:** Adjust the regex patterns to match the exact format produced by `HumanTracksDrawer.analysis()`. Log the actual report output during development to confirm the format.

#### Acceptance Criteria
- [ ] `_report.txt` parsed; data inserted into `reports` and `shot_events` tables
- [ ] Per-frame angle data inserted into `angle_frames` table
- [ ] Transient intermediate files deleted after successful persist
- [ ] Output video existence verified before session marked `completed`

---

## 8. Epic 4 — Analytics Dashboard & Results Views

**Epic key:** SCRUM-8  
**References:** SRS UC-04, UC-05, UC-06, SDS Screens 1–5

---

### SCRUM-19: Shot Analytics Screen

**Story:** As a player, I can view shot analytics for a completed session.

#### Subtasks
| Key | Subtask |
|---|---|
| SCRUM-48 | Build `GET /api/sessions/{id}/shots` and `GET /api/sessions/{id}/report` endpoints |
| SCRUM-49 | Build shot% hero stat and made/missed doughnut chart component |
| SCRUM-50 | Build per-shot results table (shot #, outcome, release angle, elbow angle) |
| SCRUM-51 | Implement streaming video player for annotated output via `GET /api/sessions/{id}/output_video` |
| SCRUM-52 | Add processing spinner and failed-session error message states |

#### API Endpoints

```python
@router.get("/sessions/{session_id}/report")
def get_report(session_id: int, current_user=Depends(get_current_user), db=Depends(get_db)):
    session = verify_session_ownership(session_id, current_user, db)
    report = db.query(Report).filter(Report.session_id == session_id).first()
    if not report:
        raise HTTPException(404, "Report not found")
    return report

@router.get("/sessions/{session_id}/shots")
def get_shots(session_id: int, current_user=Depends(get_current_user), db=Depends(get_db)):
    verify_session_ownership(session_id, current_user, db)
    return db.query(ShotEvent).filter(ShotEvent.session_id == session_id).all()

@router.get("/sessions/{session_id}/output_video")
def stream_output_video(session_id: int, current_user=Depends(get_current_user), db=Depends(get_db)):
    session = verify_session_ownership(session_id, current_user, db)
    if not session.output_video_path or not os.path.exists(session.output_video_path):
        raise HTTPException(404, "Output video not available")
    return FileResponse(session.output_video_path, media_type="video/x-msvideo")
```

#### Doughnut Chart (Chart.js)

```javascript
// session.html
const ctx = document.getElementById('doughnut-chart').getContext('2d');
new Chart(ctx, {
    type: 'doughnut',
    data: {
        labels: ['Made', 'Missed'],
        datasets: [{
            data: [report.shots_made, report.shots_missed],
            backgroundColor: ['#22c55e', '#ef4444']
        }]
    },
    options: { cutout: '70%', plugins: { legend: { position: 'bottom' } } }
});
```

#### Acceptance Criteria
- [ ] Shot percentage displayed prominently
- [ ] Made/Missed doughnut chart rendered
- [ ] Per-shot table: shot #, outcome, release angle, elbow angle
- [ ] Embedded video player streams annotated output `.avi`
- [ ] Processing spinner shown when session status is `processing`
- [ ] Failure message shown when session status is `failed`

---

### SCRUM-20: Form Analysis Screen

**Story:** As a player, I can view my shooting form analysis.

#### Subtasks
| Key | Subtask |
|---|---|
| SCRUM-53 | Build `GET /api/sessions/{id}/angles` endpoint |
| SCRUM-54 | Build overlaid elbow angle line chart (one series per shot) |
| SCRUM-55 | Render consistency score, average release angle with 45–55° reference band, and feedback |
| SCRUM-56 | Add inline tooltips for key metrics; handle no-pose-data fallback |

#### Elbow Angle Line Chart (Chart.js)

```javascript
// One dataset (line series) per detected shot
const datasets = shots.map((shot, i) => ({
    label: `Shot ${shot.shot_number}`,
    data: angleFrames.filter(f => f.shot_number === shot.shot_number).map(f => f.elbow_angle),
    borderColor: colors[i % colors.length],
    fill: false,
    tension: 0.3
}));

new Chart(ctx, {
    type: 'line',
    data: { labels: frameLabels, datasets },
    options: {
        plugins: { 
            annotation: {
                annotations: {
                    idealZone: {
                        type: 'box',
                        yMin: 45, yMax: 55,
                        backgroundColor: 'rgba(34,197,94,0.1)',
                        label: { content: 'Ideal 45–55°', display: true }
                    }
                }
            }
        }
    }
});
```

#### Tooltip Component

```html
<!-- Reusable tooltip for metric labels -->
<span class="metric-label">
    Consistency Score
    <span class="tooltip-icon" data-tip="A score from 0–100 measuring how uniform your shooting form is across all detected shots. Higher = more consistent.">ⓘ</span>
</span>
```

#### Acceptance Criteria
- [ ] Consistency Score (0–100) displayed with descriptive label
- [ ] Overlaid line chart shows elbow angle per shot
- [ ] Average release angle shown with 45–55° ideal reference band
- [ ] Plain-language feedback text from report displayed
- [ ] No-pose-data fallback message shown when pose estimation unavailable
- [ ] Inline tooltips explain Consistency Score, Release Angle, Elbow Angle

---

### SCRUM-21: Session History & Trend Chart

**Story:** As a player, I can view my session history and a shot percentage trend chart.

#### Subtasks
| Key | Subtask |
|---|---|
| SCRUM-57 | Build `GET /api/sessions` endpoint |
| SCRUM-58 | Build session history table UI |
| SCRUM-59 | Build shot% trend line chart; add empty-state prompt |

#### API Endpoint

```python
@router.get("/sessions")
def list_sessions(current_user=Depends(get_current_user), db=Depends(get_db)):
    sessions = (
        db.query(SessionModel)
        .filter(SessionModel.user_id == current_user.id)
        .order_by(SessionModel.created_at.desc())
        .all()
    )
    return sessions

@router.get("/dashboard/trends")
def get_trends(current_user=Depends(get_current_user), db=Depends(get_db)):
    sessions = (
        db.query(SessionModel, Report)
        .join(Report, Report.session_id == SessionModel.id)
        .filter(SessionModel.user_id == current_user.id, SessionModel.status == SessionStatus.COMPLETED)
        .order_by(SessionModel.created_at.asc())
        .all()
    )
    return [{"date": s.created_at, "shot_percentage": r.shot_percentage} for s, r in sessions]
```

#### Acceptance Criteria
- [ ] History page lists all sessions sorted by date
- [ ] Table columns: Date, Video Name, Shots, Made%, Consistency Score, Status badge, Delete action
- [ ] Trend line chart shows shot% over time
- [ ] Empty state shows prompt to upload first video

---

### SCRUM-22: Home Dashboard

**Story:** As a player, I can see a home dashboard with aggregate stats, recent sessions, and a quick upload CTA.

#### Subtasks
| Key | Subtask |
|---|---|
| SCRUM-60 | Build `GET /api/dashboard/summary` and `GET /api/dashboard/trends` endpoints |
| SCRUM-61 | Build summary stats card (total sessions, lifetime shot%, avg consistency score) |
| SCRUM-62 | Build recent sessions list (last 5) and Upload CTA button |
| SCRUM-63 | Build global navigation header |

#### API Endpoint

```python
@router.get("/dashboard/summary")
def get_summary(current_user=Depends(get_current_user), db=Depends(get_db)):
    reports = (
        db.query(Report)
        .join(SessionModel, SessionModel.id == Report.session_id)
        .filter(SessionModel.user_id == current_user.id)
        .all()
    )
    if not reports:
        return {"total_sessions": 0, "lifetime_shot_pct": None, "avg_consistency": None}
    
    return {
        "total_sessions": len(reports),
        "lifetime_shot_pct": round(sum(r.shot_percentage for r in reports) / len(reports), 1),
        "avg_consistency": round(sum(r.consistency_score for r in reports) / len(reports), 1)
    }
```

#### Acceptance Criteria
- [ ] Dashboard shows total sessions, lifetime shot%, average consistency score
- [ ] Recent sessions list shows last 5 with date, shot%, status badge
- [ ] Prominent "Upload New Video" CTA button displayed
- [ ] Navigation header: Dashboard, History, Upload, Logout
- [ ] Page loads within 2 seconds under normal load

---

## 9. Epic 5 — Database, Security & Infrastructure

**Epic key:** SCRUM-9  
**References:** SRS NFR-7.2, NFR-7.3, NFR-7.4, SDS Section 5.1, SDS Appendices B & C

---

### SCRUM-23: Database Schema & ORM Models

**Story:** As the system, the database schema and ORM models are defined for all entities.

#### Subtasks
| Key | Subtask |
|---|---|
| SCRUM-64 | Define SQLAlchemy ORM models for User, Session, Report, ShotEvent, AngleFrame |
| SCRUM-65 | Set up Alembic and write initial migration |
| SCRUM-66 | Create `.env.example` documenting all environment variables |

#### ORM Models

```python
# app/models/user.py
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base

class User(Base):
    __tablename__ = "users"
    id            = Column(Integer, primary_key=True, index=True)
    name          = Column(String(100), nullable=False)
    email         = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())
    sessions      = relationship("Session", back_populates="user", cascade="all, delete-orphan")
```

```python
# app/models/session.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import relationship
from app.database import Base
from app.models.session_status import SessionStatus

class Session(Base):
    __tablename__ = "sessions"
    id                 = Column(Integer, primary_key=True, index=True)
    user_id            = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    original_filename  = Column(String(255))
    stored_filename    = Column(String(255))
    upload_path        = Column(String(512))
    output_video_path  = Column(String(512))
    report_path        = Column(String(512))
    status             = Column(SAEnum(SessionStatus), default=SessionStatus.QUEUED)
    error_message      = Column(String(1000))
    created_at         = Column(DateTime(timezone=True), server_default=func.now())
    completed_at       = Column(DateTime(timezone=True))
    user               = relationship("User", back_populates="sessions")
    report             = relationship("Report", back_populates="session", uselist=False, cascade="all, delete-orphan")
    shot_events        = relationship("ShotEvent", back_populates="session", cascade="all, delete-orphan")
    angle_frames       = relationship("AngleFrame", back_populates="session", cascade="all, delete-orphan")
```

```python
# app/models/report.py
from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

class Report(Base):
    __tablename__ = "reports"
    id                = Column(Integer, primary_key=True)
    session_id        = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), unique=True)
    shot_percentage   = Column(Float)
    shots_made        = Column(Integer)
    shots_missed      = Column(Integer)
    consistency_score = Column(Float)
    avg_release_angle = Column(Float)
    feedback_text     = Column(String(2000))
    session           = relationship("Session", back_populates="report")
```

```python
# app/models/shot_event.py
from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

class ShotEvent(Base):
    __tablename__ = "shot_events"
    id                      = Column(Integer, primary_key=True)
    session_id              = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"))
    shot_number             = Column(Integer)
    outcome                 = Column(String(10))   # "made" or "missed"
    release_angle           = Column(Float)
    elbow_angle_at_release  = Column(Float)
    session                 = relationship("Session", back_populates="shot_events")
```

```python
# app/models/angle_frame.py
from sqlalchemy import Column, Integer, Float, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

class AngleFrame(Base):
    __tablename__ = "angle_frames"
    id           = Column(Integer, primary_key=True)
    session_id   = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"))
    shot_number  = Column(Integer)
    frame_index  = Column(Integer)
    elbow_angle  = Column(Float)
    session      = relationship("Session", back_populates="angle_frames")
```

#### Running Migrations

```bash
# Generate initial migration
alembic revision --autogenerate -m "initial schema"

# Apply migrations
alembic upgrade head

# Rollback one step
alembic downgrade -1
```

#### Acceptance Criteria
- [ ] All 5 ORM models defined with correct relationships and foreign keys
- [ ] Alembic migrations runnable from scratch on a clean database
- [ ] PostgreSQL used in production; SQLite works for local dev
- [ ] `.env.example` documents all 7 required environment variables

---

### SCRUM-24: Security Hardening

**Story:** As the system, all security NFRs are implemented.

#### Subtasks
| Key | Subtask |
|---|---|
| SCRUM-67 | Enforce HTTPS; configure HTTP → HTTPS redirect middleware |
| SCRUM-68 | Enable CSRF protection and configure CORS policy |
| SCRUM-69 | Add secure response headers (CSP, X-Frame-Options, X-Content-Type-Options) |
| SCRUM-70 | Validate MIME type and file extension on upload; reject non-video files with 400 |
| SCRUM-71 | Write security integration tests |

#### Security Middleware

```python
# app/core/middleware.py
from fastapi import FastAPI
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

def add_security_middleware(app: FastAPI, production: bool = False):
    if production:
        # Redirect all HTTP to HTTPS
        app.add_middleware(HTTPSRedirectMiddleware)
    
    # CORS — restrict to your actual domain in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://yourdomain.com"] if production else ["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )
    
    # Secure response headers
    class SecurityHeadersMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            response: Response = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self' https://cdnjs.cloudflare.com; "
                "style-src 'self' 'unsafe-inline';"
            )
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            return response
    
    app.add_middleware(SecurityHeadersMiddleware)
```

#### Security Test Examples

```python
# tests/test_security.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_unauthenticated_request_returns_401(client: AsyncClient):
    response = await client.get("/api/sessions")
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_cross_user_session_access_returns_403(client: AsyncClient, user_a_token, user_b_session_id):
    response = await client.get(
        f"/api/sessions/{user_b_session_id}",
        headers={"Authorization": f"Bearer {user_a_token}"}
    )
    assert response.status_code == 403

@pytest.mark.asyncio
async def test_malicious_file_upload_rejected(client: AsyncClient, auth_token):
    response = await client.post(
        "/api/sessions/upload",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("evil.exe", b"MZ\x90\x00", "application/octet-stream")}
    )
    assert response.status_code == 400
```

#### Acceptance Criteria
- [ ] HTTPS enforced in production; HTTP redirects to HTTPS
- [ ] CSRF protection enabled on all state-mutating endpoints
- [ ] Secure response headers set on all responses
- [ ] Upload endpoint rejects non-video MIME types with 400
- [ ] Security integration tests pass for auth bypass, cross-user access, bad file upload

---

### SCRUM-25: Documentation & In-App Help

**Story:** As a developer and user, I have access to a README and in-app documentation.

#### Subtasks
| Key | Subtask |
|---|---|
| SCRUM-72 | Write developer README covering setup, dependencies, running app, Celery, and tests |
| SCRUM-73 | Build in-app help / onboarding page |
| SCRUM-74 | Add UI tooltips for Consistency Score, Release Angle, and Elbow Angle across all screens |

#### In-App Help Content

The help page (`/help`) should cover at minimum:

**Recording a Suitable Video**
- Record from a **side-on angle** at a fixed camera position
- Ensure the **full player body and basket are visible** in every frame
- Minimum resolution: **720p** at standard frame rate (30fps recommended)
- Good lighting — avoid strong backlighting or deep shadows
- Avoid panning; keep the camera stationary

**Understanding Your Results**
- **Shot Percentage:** Shots made ÷ total shot attempts × 100
- **Consistency Score (0–100):** Measures how similar your form is across all shots. A score above 70 indicates reliable mechanics.
- **Release Angle:** The angle of the ball's trajectory at the moment it leaves your hands. Ideal range: 45–55°.
- **Elbow Angle:** The angle at your shooting elbow at the point of release. Consistent values indicate repeatable form.

#### Acceptance Criteria
- [ ] Developer README covers: setup, install, run app, run Celery, run tests, file system layout, env vars
- [ ] In-app help page explains video recording requirements and metric definitions
- [ ] Inline tooltips present on all Consistency Score, Release Angle, Elbow Angle labels

---

## 10. API Reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/api/register` | No | Create new user account |
| `POST` | `/api/login` | No | Authenticate and receive JWT token |
| `POST` | `/api/logout` | Yes | Invalidate current session |
| `POST` | `/api/sessions/upload` | Yes | Upload a video for processing |
| `GET` | `/api/sessions` | Yes | List all sessions for authenticated user |
| `GET` | `/api/sessions/{id}` | Yes | Get session status and metadata |
| `DELETE` | `/api/sessions/{id}` | Yes | Delete a session and all its data |
| `GET` | `/api/sessions/{id}/report` | Yes | Get full report data for a session |
| `GET` | `/api/sessions/{id}/shots` | Yes | Get all shot events for a session |
| `GET` | `/api/sessions/{id}/angles` | Yes | Get per-frame angle data for all shots |
| `GET` | `/api/sessions/{id}/output_video` | Yes | Stream annotated output video |
| `GET` | `/api/dashboard/summary` | Yes | Get aggregate lifetime stats |
| `GET` | `/api/dashboard/trends` | Yes | Get shot% trend data over time |

---

## 11. Database Schema Reference

```
users
├── id            INTEGER PK
├── name          VARCHAR(100)
├── email         VARCHAR(255) UNIQUE
├── password_hash VARCHAR(255)
└── created_at    TIMESTAMPTZ

sessions
├── id                  INTEGER PK
├── user_id             INTEGER FK → users.id
├── original_filename   VARCHAR(255)
├── stored_filename     VARCHAR(255)
├── upload_path         VARCHAR(512)
├── output_video_path   VARCHAR(512)
├── report_path         VARCHAR(512)
├── status              ENUM(queued, processing, completed, failed, deleted)
├── error_message       VARCHAR(1000)
├── created_at          TIMESTAMPTZ
└── completed_at        TIMESTAMPTZ

reports
├── id                INTEGER PK
├── session_id        INTEGER FK → sessions.id (UNIQUE — one per session)
├── shot_percentage   FLOAT
├── shots_made        INTEGER
├── shots_missed      INTEGER
├── consistency_score FLOAT
├── avg_release_angle FLOAT
└── feedback_text     VARCHAR(2000)

shot_events
├── id                     INTEGER PK
├── session_id             INTEGER FK → sessions.id
├── shot_number            INTEGER
├── outcome                VARCHAR(10)   -- "made" | "missed"
├── release_angle          FLOAT
└── elbow_angle_at_release FLOAT

angle_frames
├── id           INTEGER PK
├── session_id   INTEGER FK → sessions.id
├── shot_number  INTEGER
├── frame_index  INTEGER
└── elbow_angle  FLOAT
```

---

## 12. CV Pipeline Reference

The existing `main.py` pipeline executes 12 steps. The web layer calls `main_pipeline()` — do not modify these modules.

| Step | Call | Output |
|---|---|---|
| 1 | `read_video(video_path)` | `frames`, `fps` |
| 2 | `BallTracker.get_object_tracks(frames)` | `ball_tracks` |
| 3 | `RimTracker` (derived from ball tracks) | `rim_tracks` |
| 4 | `HumanTracker.detect_frame(frames)` | `human_tracks` |
| 5 | `ball_tracker.remove_wrong_tracks(ball_tracks)` | cleaned tracks |
| 6 | `ball_tracker.interpolate_missing_tracks(ball_tracks)` | interpolated tracks |
| 7 | `human_tracker.calc_angles(frames, human_tracks)` | `angles` per frame |
| 8 | `ball_hand(ball_loco, points, frames)` | `ball_left_frames` |
| 9 | `shot_started(points, ball_left_frames)` | `shot_starts` |
| 10 | `HumanTracksDrawer.draw()` + `analysis(...)` | annotated `frames`, `_report.txt` |
| 11 | `ShotTracker.detect_shot()` + `draw_shots()` | final annotated `frames` |
| 12 | `write_video(frames, output_path, fps)` | `.avi` output file |

**Model files:**
| File | Size | Purpose |
|---|---|---|
| `best.pt` | ~43MB | Ball/rim detection (fine-tuned YOLOv8, 3 classes) |
| `yolov8m-pose.pt` | ~78MB | Human pose estimation (17 keypoints) |

**Known limitations to account for in the UI:**
- Accuracy degrades below 720p or with poor lighting
- Ball track interpolation may introduce false positives on short occlusions
- `shot_started()` may miss shots where the player's hand is occluded
- CPU-only inference is ~10× slower than GPU inference

---

## 13. Session State Machine

```
[queued]
   │
   │  Worker picks up job
   ▼
[processing]
   │                    │
   │  Pipeline success  │  Unhandled exception
   ▼                    ▼
[completed]          [failed]
   │                    │
   │  User deletes      │  User deletes
   ▼                    ▼
[deleted]            [deleted]
```

**State transitions:**

| From | To | Trigger |
|---|---|---|
| `queued` | `processing` | Celery worker dequeues job |
| `processing` | `completed` | Pipeline succeeds, outputs verified |
| `processing` | `failed` | Unhandled exception in pipeline |
| `completed` | `deleted` | User confirms deletion |
| `failed` | `deleted` | User confirms deletion |
| `failed` | `queued` | User retries (future feature) |

> **Deletion is blocked** when status is `processing`. The UI must disable the delete action and show a message.

---

## 14. Testing Strategy

### Unit Tests

Recommended coverage for each epic:

| Area | Tests |
|---|---|
| Auth | Registration happy path, duplicate email, weak password, login success, login failure, token expiry |
| Upload | Valid video accepted, oversized file rejected, invalid MIME rejected, invalid extension rejected |
| Pipeline | Mock `main_pipeline()` — test status transitions, file existence checks, failure handling |
| Report Parser | Parse known report fixture, verify DB inserts, verify transient file cleanup |
| Dashboard API | Summary endpoint with 0 sessions, with N sessions; trends endpoint |

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=app --cov-report=html

# Run only security tests
pytest tests/test_security.py -v
```

### Integration Test Setup

```python
# tests/conftest.py
import pytest
from httpx import AsyncClient
from app.main import app
from app.database import Base, engine

@pytest.fixture(autouse=True)
def reset_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as c:
        yield c

@pytest.fixture
async def auth_token(client):
    await client.post("/api/register", json={
        "name": "Test User", "email": "test@test.com", "password": "Secure123"
    })
    res = await client.post("/api/login", json={
        "email": "test@test.com", "password": "Secure123"
    })
    return res.json()["access_token"]
```

---
