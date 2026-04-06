# SwishVision Web App - Developer README

This guide covers local developer setup for:

- Environment variables
- Dependencies
- Running backend and frontend
- Running Celery worker
- Running tests

## Project Layout

- Backend: `app/` (FastAPI + SQLAlchemy + Celery)
- Frontend: `app/frontend/` (React)
- Tests: `tests/`

## Prerequisites

| Tool | Recommended Version |
|------|---------------------|
| Python | 3.8+ |
| Node.js | 18+ |
| SQL Server | 2019+ |
| ODBC Driver | SQL Server ODBC driver installed locally |
| Redis | 7+ |

Notes:

- The backend uses SQL Server through `pyodbc`.
- ODBC driver names vary by machine (examples: `ODBC Driver 17 for SQL Server`, `ODBC Driver 18 for SQL Server`, `SQL Server`).

## 1. Environment Setup

Run commands from the repository root.

### 1.1 Create and activate virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 1.2 Backend env file

Copy the template and edit values:

```bash
copy .env.example .env
```

If `copy` is unavailable (macOS/Linux):

```bash
cp .env.example .env
```

Required keys in `.env`:

```env
DB_SERVER=localhost
DB_NAME=SwishVision
DB_DRIVER=ODBC Driver 17 for SQL Server
DB_TRUSTED_CONNECTION=true
DB_USERNAME=
DB_PASSWORD=

REDIS_URL=redis://localhost:6379/0
SECRET_KEY=change-me-to-a-random-256-bit-secret

UPLOAD_DIR=app/uploads
OUTPUT_DIR=output_videos
MODEL_DIR=models
MAX_UPLOAD_SIZE_MB=500
```

Important:

- Set `DB_DRIVER` to a driver that exists on your machine.
- Use `DB_TRUSTED_CONNECTION=true` for Windows auth.
- Set `DB_TRUSTED_CONNECTION=false` and fill username/password for SQL auth.

### 1.3 Frontend env file

No frontend `.env` file is currently required for local development.

The React app uses the Create React App development proxy for backend API requests, so you do not need to set `REACT_APP_API_URL` unless the frontend is updated to explicitly read that variable in the future.

## 2. Dependency Installation

### 2.1 Backend dependencies

```bash
pip install -r app/requirements.txt
```

### 2.2 Test dependencies

If `pytest` is not installed in your environment:

```bash
pip install pytest
```

### 2.3 Frontend dependencies

```bash
cd app/frontend
npm install
cd ../..
```

## 3. Database and Redis

### 3.1 Create SQL Server database

Run in SSMS or `sqlcmd`:

```sql
CREATE DATABASE SwishVision;
```

### 3.2 Start Redis

Start Redis locally before running Celery worker.

### 3.3 Run migrations

Apply the initial schema migration after the database is available:

```bash
alembic upgrade head
```

If the `alembic` command is not available, use:

```bash
python -m alembic upgrade head
```

## 4. Run the App

Use two terminals.

### Terminal A: Backend API

From repository root:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend URLs:

- API base: `http://localhost:8000`
- OpenAPI docs: `http://localhost:8000/docs`

### Terminal B: Frontend

```bash
cd app/frontend
npm start
```

Frontend URL:

- App: `http://localhost:3000`

## 5. Celery Worker

Video processing task is defined in `app/tasks/pipeline_task.py`.

Run worker from repository root:

```bash
celery -A app.tasks.pipeline_task.celery_app worker --loglevel=info
```

If `celery` command is not found, run with Python module:

```bash
python -m celery -A app.tasks.pipeline_task.celery_app worker --loglevel=info
```

## 6. Run Tests

Run from repository root:

```bash
pytest -q
```

Run specific test files:

```bash
pytest tests/test_auth.py -q
pytest tests/test_sessions.py -q
```

Test notes:

- Tests override the app's `get_db` dependency in pytest to use a temporary SQLite database.
- You do not need to create or configure a separate local test database for the standard test suite.
- The configured application database connection is not used by these tests.

## 7. Common Issues

### ODBC driver error

Symptom: `Data source name not found`.

Fix:

- Install SQL Server ODBC driver.
- Update `DB_DRIVER` in `.env` to match installed driver name exactly.

### SQL Server connection error

Symptom: `SQL Server does not exist or access denied`.

Fix:

- Verify SQL Server service is running.
- Verify server name/instance in `DB_SERVER`.
- Verify auth mode and credentials.

### Celery stuck or not processing

Fix:

- Verify Redis is running and reachable via `REDIS_URL`.
- Start worker from repository root.

## 8. Security Notes

- Do not commit `.env`.
- Replace `SECRET_KEY` before shared/staging/prod environments.
- Use HTTPS termination in non-local environments (reverse proxy or ingress).
