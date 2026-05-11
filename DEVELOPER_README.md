# SwishVision — Developer README

Quick-start guide for running the SwishVision web application locally on Windows (PowerShell). For deeper detail on individual subsystems, see [`app/DEVELOPER_README.md`](app/DEVELOPER_README.md), [`docs/SwishVision_SRS.md`](docs/SwishVision_SRS.md), and [`docs/SwishVision_SDS.md`](docs/SwishVision_SDS.md).

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | React (Create React App) — `app/frontend/` |
| Backend | FastAPI + Uvicorn — `app/main.py` |
| Database | Microsoft SQL Server (via `pyodbc`) |
| Task queue | Celery worker — `app/tasks/pipeline_task.py` |
| Broker | Redis (`redis://localhost:6379/0`) |
| CV pipeline | YOLOv8 (Ultralytics) + OpenCV + ffmpeg (imageio-ffmpeg) — `main.py`, `trackers/`, `drawers/`, `utils/` |

The flow: React UI → FastAPI REST API → Celery enqueues a job on Redis → Celery worker invokes the CV pipeline → annotated H.264 mp4 + parsed report rows land in SQL Server → React reads them back.

---

## Prerequisites

- Python 3.8+ (tested on 3.13) with `venv`
- Node.js 18+
- Microsoft SQL Server 2019+ with the **ODBC Driver 17 for SQL Server** installed (`Get-OdbcDriver` to verify)
- Redis (a portable Windows binary works; see step 1 of "Running the app" below)
- (Recommended) NVIDIA GPU with CUDA-compatible PyTorch for inference. CPU-only inference works but is roughly 10× slower.

---

## One-time setup

```powershell
# 1. Create venv + activate
python -m venv .venv
.venv\Scripts\Activate

# 2. Install backend (FastAPI / SQLAlchemy / Celery / Redis client)
pip install -r app/requirements.txt

# 3. Install the CV pipeline runtime deps (NOT in app/requirements.txt — see "Known dependency gaps" below)
pip install opencv-python ultralytics torch torchvision numpy matplotlib supervision pandas imageio-ffmpeg

# 4. Pin a passlib-compatible bcrypt (see "Known dependency gaps")
pip install "bcrypt==3.2.2"

# 5. Install frontend deps
cd app/frontend
npm install
cd ../..

# 6. Configure env — copy template and edit
copy env.example .env
#   Critical edits:
#     DB_DRIVER=ODBC Driver 17 for SQL Server     (NOT the legacy "SQL Server" driver)
#     DB_TRUSTED_CONNECTION=true                   (Windows auth)
#     SECRET_KEY=<random 256-bit secret>

# 7. Create the database (SSMS or sqlcmd):
#       CREATE DATABASE SwishVision;
#    The schema itself is auto-created by SQLAlchemy's metadata.create_all() in app/main.py
#    on first startup — running `alembic upgrade head` is optional and only needed if you
#    are working with versioned migrations.
```

Make sure `models/best.pt` and `models/yolov8m-pose.pt` are present in `models/`. See [`README.md`](README.md) for sources.

### Known dependency gaps in `app/requirements.txt`

`app/requirements.txt` only lists web-stack packages — running it alone will not give you a working pipeline. We hit four real installation problems while bringing the system up; these are documented here so the next person doesn't repeat them:

| Missing / wrong | Symptom | Fix |
|---|---|---|
| `redis` Python client | Celery worker dies on startup with `AttributeError: 'NoneType' object has no attribute 'Redis'` (kombu fails to import the redis transport) | `pip install redis==5.2.1` (already in `app/requirements.txt` but verify it actually installed) |
| CV pipeline deps (`opencv-python`, `ultralytics`, `torch`, `torchvision`, `numpy`, `matplotlib`, `supervision`, `pandas`) | Sessions fail almost instantly with `ModuleNotFoundError: No module named 'cv2'` (or `supervision`, etc.) — surfaced in the worker log | Step 3 above |
| `bcrypt==4.0.1` (pinned by `app/requirements.txt`) is incompatible with `passlib==1.7.4` | Every register/login crashes with `ValueError: password cannot be longer than 72 bytes` *inside passlib's own backend probe* | `pip install "bcrypt==3.2.2"` (step 4 above), or update the pin in `app/requirements.txt` |
| `imageio-ffmpeg` not installed | Pipeline completes but the annotated video shows a blank player in the browser. Worker log says `ffmpeg not available (No module named 'imageio_ffmpeg'), keeping mp4v output.` mp4v in MP4 plays in VLC but not in Chrome / Firefox / Edge `<video>` elements. | `pip install imageio-ffmpeg`. The pipeline transcodes to H.264 (`libx264`, `yuv420p`, `+faststart`) on the second pass once this is present. |

---

## Running the app — four terminals

You need **four** processes running simultaneously. Open a separate PowerShell terminal for each.

### 1. Redis

```powershell
Start-Process -FilePath "$env:TEMP\redis\redis-server.exe" -ArgumentList "--port 6379 --bind 127.0.0.1" -WindowStyle Minimized
```

> First time only — download and extract the binary:
> ```powershell
> Invoke-WebRequest -Uri "https://github.com/tporadowski/redis/releases/download/v5.0.14.1/Redis-x64-5.0.14.1.zip" -OutFile "$env:TEMP\redis.zip" -UseBasicParsing
> Expand-Archive -Path "$env:TEMP\redis.zip" -DestinationPath "$env:TEMP\redis" -Force
> ```
>
> Verify it's up: `& "$env:TEMP\redis\redis-cli.exe" ping` → `PONG`

### 2. Backend (FastAPI / Uvicorn)

```powershell
.venv\Scripts\Activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API base: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`

### 3. Frontend (React)

```powershell
cd app/frontend
npm start
```

- App: `http://localhost:3000`
- Dev-server proxy forwards `/api/*` requests to the backend on port 8000.

### 4. Celery worker

```powershell
.venv\Scripts\python.exe -m celery -A app.tasks.pipeline_task.celery_app worker --loglevel=info --pool=solo
```

`--pool=solo` is **required on Windows** (Celery's default pool relies on `os.fork`, which isn't available).

Worker is ready when you see:

```
[INFO/MainProcess] celery@<hostname> ready.
```

---

## Session lifecycle

When a user uploads a video, the session row moves through these states:

| Status | Meaning |
|---|---|
| `uploading` | File being saved to disk |
| `queued` | Task dispatched to Redis, waiting for a worker |
| `pending` | Redis was unreachable at dispatch — re-dispatch needed |
| `processing` | Worker has picked up the task |
| `completed` | Pipeline finished; H.264 `.mp4` written, report parsed into DB |
| `failed` | Pipeline errored — check the worker terminal |

A successful run produces:

- `output_videos/output_session_{id}_{n}_processed.mp4` — annotated H.264 video served to the browser via `GET /api/sessions/{id}/output_video`
- `reports/session_{id}_{n}_report.txt` — text report parsed into `reports`, `shot_events`, and `angle_frames` rows

---

## Tests

```powershell
pytest -q
```

Tests override the app's `get_db` dependency to use a temporary SQLite database, so you don't need a separate test SQL Server.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ConnectionRefusedError: [WinError 10061]` from uvicorn on upload | Redis isn't running — see step 1. |
| `Data source name not found` (ODBC) | Install ODBC Driver 17 for SQL Server; set `DB_DRIVER` in `.env` to the exact installed driver name. |
| `[08001] SQL Server does not exist or access denied` | Most likely `.env` is using the legacy `DB_DRIVER=SQL Server` (DBNETLIB) which can't reach the local instance. Switch to `ODBC Driver 17 for SQL Server`. |
| `ModuleNotFoundError: No module named 'cv2'` (or `supervision`, `pandas`) in worker log | CV pipeline deps not installed. See "Known dependency gaps" → step 3 of one-time setup. |
| Register/login returns 500, log shows `ValueError: password cannot be longer than 72 bytes` | bcrypt 4.x + passlib 1.7.4 incompatibility. `pip install "bcrypt==3.2.2"` and restart the backend. |
| Annotated video player is blank but session shows `completed` | `imageio-ffmpeg` is missing, so output is left in `mp4v` codec which browsers won't play. `pip install imageio-ffmpeg`, restart the worker, and re-process. |
| Session stuck at `queued` after Redis restart | Redis is in-memory — the queue is wiped on restart. Re-dispatch: `.venv\Scripts\python.exe -c "from app.tasks.pipeline_task import process_video; process_video.delay(<session_id>)"` |
| Celery never picks up tasks on Windows | Make sure `--pool=solo` is on the command. |
| `Stop-Process -Name python` killed uvicorn too | Both run as `python` — kill by PID instead: `Get-Process python \| Select Id, CommandLine` then `Stop-Process -Id <pid>`. |

For more detail, see [`app/DEVELOPER_README.md`](app/DEVELOPER_README.md) §7.
