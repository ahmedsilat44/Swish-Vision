# SwishVision Web App — Developer Guide

For full local setup and run instructions, see `app/DEVELOPER_README.md`.

SwishVision's web application is split into two parts:

- **Backend** — FastAPI (Python), located in `app/`
- **Frontend** — React, located in `app/frontend/`

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.13+ |
| Node.js | 18+ |
| SQL Server | 2019+ (local or remote) |
| ODBC Driver | 17 for SQL Server |
| Redis | 7+ (for background video processing) |

---

## Backend Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r app/requirements.txt
```

### 3. Create the database

Open SQL Server Management Studio (or `sqlcmd`) and run:

```sql
CREATE DATABASE SwishVision;
```

### 4. Configure environment variables

Create a `.env` file in the **project root** (next to this README):

```env
DB_SERVER=localhost
DB_NAME=SwishVision
DB_TRUSTED_CONNECTION=true

# Or use SQL auth:
# DB_TRUSTED_CONNECTION=false
# DB_USERNAME=your_user
# DB_PASSWORD=your_password

REDIS_URL=redis://localhost:6379/0
SECRET_KEY=replace-with-a-random-256-bit-secret
```

### 5. Run the backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## Frontend Setup

### 1. Install dependencies

```bash
cd app/frontend
npm install
```

### 2. Configure environment variables

The file `app/frontend/.env` already contains:

```env
REACT_APP_API_URL=http://localhost:8000/api
```

Update this value if your backend runs on a different host/port.

### 3. Run the dev server

```bash
cd app/frontend
npm start
```

The React app will open at `http://localhost:3000`.  
API calls are proxied to `http://localhost:8000` via the `proxy` field in `package.json`.

---

## Running Both Together

Open two terminals:

```bash
# Terminal 1 — backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend
cd app/frontend && npm start
```

---

## Project Structure

```
app/
├── main.py               # FastAPI app entry point
├── config.py             # Settings (reads .env)
├── database.py           # SQLAlchemy engine & session
├── requirements.txt      # Python dependencies
│
├── api/
│   ├── auth.py           # /api/register, /api/login, /api/logout
│   ├── sessions.py       # /api/sessions  (upload, list, get, delete)
│   └── dashboard.py      # /api/dashboard/summary
│
├── core/
│   ├── middleware.py     # CORS + security headers
│   └── security.py       # JWT auth, password hashing
│
├── models/               # SQLAlchemy ORM models
├── schemas/              # Pydantic request/response schemas
├── tasks/
│   └── pipeline_task.py  # Celery task — runs CV pipeline on uploaded video
├── migrations/           # Alembic migration scripts
└── uploads/              # Uploaded video files (git-ignored)

app/frontend/
├── public/
├── src/
│   ├── pages/            # Page-level React components (LoginPage, etc.)
│   ├── App.js
│   └── index.js
├── .env                  # REACT_APP_API_URL
└── package.json
```

---

## API Overview

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/register` | No | Create account |
| POST | `/api/login` | No | Get JWT token |
| POST | `/api/logout` | Yes | Logout |
| POST | `/api/sessions/upload` | Yes | Upload video |
| GET | `/api/sessions/` | Yes | List user sessions |
| GET | `/api/sessions/{id}` | Yes | Get session detail |
| GET | `/api/sessions/{id}/shots` | Yes | Shot analytics |
| GET | `/api/sessions/{id}/angles` | Yes | Angle frame data |
| DELETE | `/api/sessions/{id}` | Yes | Delete session |
| GET | `/api/dashboard/summary` | Yes | User stats summary |
| GET | `/api/health` | No | Health check |

All authenticated endpoints require `Authorization: Bearer <token>` in the request header.

---

## Frontend Component Conventions

- Place page-level components in `app/frontend/src/pages/`
- Use `process.env.REACT_APP_API_URL` as the base URL for all API calls
- Auth token should be stored in `localStorage` and sent as a `Bearer` token

---

## Background Tasks (Celery)

Video processing runs as an async Celery task (`app/tasks/pipeline_task.py`). To run a worker locally:

```bash
celery -A app.tasks.pipeline_task.celery_app worker --loglevel=info
```

Redis must be running before starting the worker.
