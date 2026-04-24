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
CELERY_RESULT_BACKEND=redis://localhost:6379/0
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

Redis must be running before starting the Celery worker. The upload endpoint will still accept requests if Redis is down (status becomes `pending`), but video processing will not start until a worker connects.

**Windows (native binary — recommended)**

Download and extract the Redis Windows binary once (run in PowerShell):

```powershell
Invoke-WebRequest -Uri "https://github.com/tporadowski/redis/releases/download/v5.0.14.1/Redis-x64-5.0.14.1.zip" -OutFile "$env:TEMP\redis.zip" -UseBasicParsing
Expand-Archive -Path "$env:TEMP\redis.zip" -DestinationPath "$env:TEMP\redis" -Force
```

Start Redis (runs minimised in the background):

```powershell
Start-Process -FilePath "$env:TEMP\redis\redis-server.exe" -ArgumentList "$env:TEMP\redis\redis.windows.conf" -WindowStyle Minimized
```

Verify it is running:

```powershell
& "$env:TEMP\redis\redis-cli.exe" ping
# Expected output: PONG
```

> **Note:** The binary is extracted to `$env:TEMP\redis`. You can move it to a permanent location (e.g. `C:\redis`) and update the paths accordingly.

**Windows (via WSL) — alternative**

Only use this if WSL has internet access:

```bash
wsl -- sudo apt-get install -y redis-server
wsl -- sudo service redis-server start
wsl -- redis-cli ping
```

**macOS**

```bash
brew install redis
brew services start redis
```

**Linux**

```bash
sudo apt-get install -y redis-server
sudo service redis-server start
```

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

Four services must be running for full end-to-end video processing. Start them in this order.

### Terminal A: Redis

```powershell
Start-Process -FilePath "$env:TEMP\redis\redis-server.exe" -ArgumentList "--port 6379 --bind 127.0.0.1" -WindowStyle Minimized
```

Verify: `& "$env:TEMP\redis\redis-cli.exe" ping` → `PONG`

### Terminal B: Backend API

From repository root:

```powershell
.venv\Scripts\Activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend URLs:

- API base: `http://localhost:8000`
- OpenAPI docs: `http://localhost:8000/docs`

### Terminal C: Frontend

```bash
cd app/frontend
npm start
```

Frontend URL:

- App: `http://localhost:3000`

### Terminal D: Celery Worker

See section 5.

## 5. Celery Worker

Video processing task is defined in `app/tasks/pipeline_task.py`.

Run from the repository root with the venv Python. `--pool=solo` is **required on Windows** (no `os.fork`):

```powershell
.venv\Scripts\python.exe -m celery -A app.tasks.pipeline_task.celery_app worker --loglevel=info --pool=solo
```

macOS/Linux:

```bash
.venv/bin/python -m celery -A app.tasks.pipeline_task.celery_app worker --loglevel=info
```

Expected output once ready:

```
[INFO/MainProcess] Connected to redis://localhost:6379/0
[INFO/MainProcess] celery@<hostname> ready.
```

When a video is uploaded, the worker terminal will print:

```
[INFO/MainProcess] Task app.tasks.pipeline_task.process_video[<uuid>] received
```

### Session status lifecycle

| Status | Meaning |
|---|---|
| `uploading` | File being saved to disk |
| `queued` | Task dispatched to Redis, waiting for worker |
| `pending` | Redis was unreachable at dispatch time |
| `processing` | Worker has picked up the task |
| `completed` | Pipeline finished successfully |
| `failed` | Pipeline errored (check worker logs) |

### Monitoring progress

Poll the session status:

```
GET http://localhost:8000/api/sessions/{id}
Authorization: Bearer <token>
```

Or list all sessions:

```
GET http://localhost:8000/api/sessions/
Authorization: Bearer <token>
```

Or use Swagger UI at `http://localhost:8000/docs`.

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

### Redis connection refused on Windows

Symptom: `ConnectionRefusedError: [WinError 10061]` in uvicorn logs when uploading a video.

Fix:

- Start Redis using the native Windows binary (see section 3.2).
- Confirm Redis is listening: `& "$env:TEMP\redis\redis-cli.exe" ping` should return `PONG`.
- If the `$env:TEMP\redis` folder is missing (e.g. after a reboot), re-run the `Start-Process` command — the ZIP stays in `$env:TEMP` so you only need to re-extract if it was also deleted.

### Celery stuck or not processing

Fix:

- Verify Redis is running and reachable via `REDIS_URL` (see Redis issue above).
- Verify `CELERY_RESULT_BACKEND` is set in `.env`.
- Make sure `--pool=solo` is used on Windows.
- Start worker from repository root with the venv Python (section 5).

### Session stuck at `queued` after Redis restart

Symptom: A session shows `status=queued` in the database but the worker never picks it up.

Cause: Redis lost its queue when it was restarted (Redis is in-memory — the queue is wiped on restart).

Fix — manually re-dispatch the task:

```powershell
.venv\Scripts\python.exe -c "from app.tasks.pipeline_task import process_video; process_video.delay(<session_id>)"
```

Replace `<session_id>` with the numeric ID from the database.

### `Stop-Process python` kills uvicorn

If you use `Stop-Process -Name python` to kill a Celery worker, it will also kill uvicorn because both run as `python` processes. Instead, kill by PID:

```powershell
# Find PIDs
Get-Process python | Select-Object Id, CommandLine
# Kill only the worker
Stop-Process -Id <pid>
```

## 8. Security Notes

- Do not commit `.env`.
- Replace `SECRET_KEY` before shared/staging/prod environments.
- Use HTTPS termination in non-local environments (reverse proxy or ingress).

## 9. Running Behind a Reverse Proxy (HTTPS / Proxy Headers)

In production the app is typically placed behind a reverse proxy (nginx, a cloud load balancer, Kubernetes ingress, etc.) that terminates TLS. The proxy then forwards plain HTTP to the app.

When `ENV=production` is set, `HTTPSRedirectMiddleware` is enabled. Without proxy-header trust, the app always sees plain HTTP—even for requests that arrived over HTTPS—and will redirect every request to HTTPS indefinitely (**infinite redirect loop**).

### Trusted proxy headers (recommended in-app fix)

Start Uvicorn with `--proxy-headers` and restrict which upstream IPs may set those headers:

```bash
# Trust a specific proxy IP (recommended for production)
uvicorn app.main:app --proxy-headers --forwarded-allow-ips=<proxy-ip> --host 0.0.0.0 --port 8000

# Trust all upstream IPs (only use if the app is fully behind a controlled network)
uvicorn app.main:app --proxy-headers --forwarded-allow-ips=* --host 0.0.0.0 --port 8000
```

With `--proxy-headers` enabled, Uvicorn reads `X-Forwarded-Proto` from the proxy and reports the correct scheme to the middleware, preventing the redirect loop.

### Preferred alternative: redirect at the proxy layer

Handle HTTP→HTTPS redirection in nginx/ingress and remove the need for in-app middleware entirely. Example nginx snippet:

```nginx
server {
    listen 80;
    server_name example.com;
    return 301 https://$host$request_uri;
}
```

This is the simpler and more robust approach for most deployments.

### Local development

No action needed. `ENV` defaults to `development`, so `HTTPSRedirectMiddleware` is never added and the app runs over plain HTTP on `localhost:8000`.

