# Software Design Specification

## SwishVision — Basketball Analytics Platform

**Version:** 1.10

| Project Team | |
|---|---|
| Mallick Mikaal Imam | |
| Minhaj ul Hassan | |
| Muhammad Ahmed Silat | |
| Syed Muhammad Sameer Hassan | |

| Submission Date | 11th May 2026 |
|---|---|

---

## Document History

| Version | Name of Person | Date | Description of Change |
|---|---|---|---|
| 1.00 | Ahmed Silat | 16th March 2026 | Initial draft — architecture and design documented |
| 1.10 | Minhaj ul Hassan | 11th May 2026 | Post-implementation update — finalised technology stack table, replaced PostgreSQL with SQL Server (pyodbc), updated file system layout to match shipped `app/` structure, updated environment variables to match `env.example` |
| 1.11 | Minhaj ul Hassan | 11th May 2026 | Corrected output codec across the document — pipeline writes H.264 `.mp4` (via `mp4v` then `imageio-ffmpeg` transcode), not XVID `.avi`. Updated output filename pattern to the actual `output_session_{id}_{n}_processed.mp4` / `session_{id}_{n}_report.txt`. |

---

## Document Information

| Category | Information |
|---|---|
| Project | SwishVision |
| Document | Software Design Specification |
| Document Version | 1.00 |
| Status | Draft |
| Author(s) | Mallick Mikaal Imam, Minhaj ul Hassan, Muhammad Ahmed Silat, Syed Muhammad Sameer Hassan |
| Approver(s) | Course Supervisor |
| Document Location | Karachi |

---

## Table of Contents

1. Introduction
   - 1.1 Purpose of Document
   - 1.2 Intended Audience
   - 1.3 Document Convention
   - 1.4 Project Overview
   - 1.5 Scope
2. Design Considerations
   - 2.1 Assumptions and Dependencies
   - 2.2 Risks and Volatile Areas
3. System Architecture
   - 3.1 System Level Architecture
   - 3.2 Software Architecture
4. Design Strategy
5. Detailed System Design
   - 5.1 Database Design
     - 5.1.1 ER Diagram
     - 5.1.2 Data Dictionary
   - 5.2 Application Design
     - 5.2.1 Class Diagram
     - 5.2.2 Sequence Diagrams
     - 5.2.3 State Diagrams
     - 5.2.4 Activity Diagrams
     - 5.2.5 System User Interface
6. References
7. Appendices

---

## 1. Introduction

### 1.1 Purpose of Document

This document provides the design specifications for SwishVision. It outlines both the high-level architectural view and the detailed low-level design of the system, covering the web platform, the CV processing pipeline, data models, and their interactions. The document serves as the blueprint that guides implementation and ensures alignment between the design decisions made and the requirements stated in the SRS.

### 1.2 Intended Audience

This document is intended for the development team building the system, the course supervisor evaluating the design, and any future maintainers who need to understand the internal structure of the platform in order to make modifications or extensions.

### 1.3 Document Convention

- Font: Arial
- Size: 11
- Line Spacing: 1.5

### 1.4 Project Overview

SwishVision is a web-based analytics platform that transforms basketball practice footage into structured performance data using computer vision and pose estimation. A fully working CLI prototype already exists, built on YOLOv8 for object detection and pose estimation. The object detection model (`best.pt`) was fine-tuned from YOLOv8 over 13 training iterations on the Basketball-1 Roboflow dataset (3 classes: basketball, rim, sports ball; CC BY 4.0). Human pose estimation uses `yolov8m-pose.pt`, a pre-trained YOLOv8 medium pose model that tracks 17 body keypoints (head, shoulders, elbows, wrists, hips, knees, ankles, neck, chest).

The design objective of this phase is to wrap that pipeline in a web application, adding user management, video upload, asynchronous processing, and a results dashboard.

The pipeline executes the following 12 steps: (1) frame extraction via OpenCV, (2) ball detection with `BallTracker`, (3) rim detection with `RimTracker`, (4) human pose detection with `HumanTracker`, (5) track validation and filtering, (6) trajectory interpolation, (7) joint angle calculation, (8) ball-hand interaction detection (`ball_hand()`), (9) shot start frame identification (`shot_started()`), (10) frame annotation via drawer classes, (11) shot outcome detection via `ShotTracker`, and (12) output video write (XVID/AVI) and text report generation.

### 1.5 Scope

- Design of the web application layer (frontend + backend API)
- Design of the user authentication subsystem
- Design of the video upload and storage subsystem
- Design of the asynchronous job queue for video processing
- Design of the database schema
- Integration design between the web layer and the existing CLI pipeline
- Design of the results dashboard and reporting views

---

## 2. Design Considerations

### 2.1 Assumptions and Dependencies

- The existing pipeline modules (`BallTracker`, `RimTracker`, `HumanTracker`, `ShotTracker`, and drawer classes) are treated as stable internal libraries. They will be invoked by a web-layer worker, not modified.
- Pre-trained model files are available on the server under `models/`: `best.pt` (~43MB, fine-tuned YOLOv8 for ball/rim detection) and `yolov8m-pose.pt` (~78MB, Ultralytics pretrained medium pose model).
- Additional model variants (`bestOld.pt`, `bestYT.pt`, `ballRim.pt`) exist but are not used in the primary pipeline.
- Video storage is local filesystem-based for the current version. Cloud storage is acknowledged as a future migration path.
- A Microsoft SQL Server 2019+ instance is available (Windows trusted connection or SQL auth) with the appropriate ODBC driver installed locally.
- The server environment has Python 3.8+, `torch>=2.0.0`, `ultralytics>=8.0.0`, `opencv-python>=4.8.0`, `numpy>=1.24.0`, and `matplotlib>=3.7.0` installed.
- The task queue uses Celery with Redis (or an equivalent) for managing background processing jobs.
- Output videos are written in two stages: `utils/vid_utils.py::write_video()` first writes frames as `.mp4` using OpenCV's `mp4v` fourcc, then transcodes to H.264 (`libx264`, `yuv420p`, `+faststart`) using the `ffmpeg` binary bundled by `imageio-ffmpeg`. The H.264 step is required for the React `<video>` player to play the file directly; if `imageio-ffmpeg` is missing the pipeline falls back to keeping the `mp4v` output, which most browsers will not play.
- Intermediate data files (`angs.txt`, `xy_coords.txt`, `detections.txt`, `ball_locl.txt`) produced during processing are treated as transient and are not persisted to the database.
- The `stubs_utils.py` module can be used to cache and reload intermediate tracking results, avoiding re-inference during development and testing.

### 2.2 Risks and Volatile Areas

- **Model output variability:** The accuracy of ball and pose detection varies with video quality, lighting, and camera angle. Input should be 720p minimum at a side-on angle. If a video produces no detectable shots, the report will reflect zero detections — the pipeline handles this gracefully.
- **Processing time unpredictability:** Inference time varies significantly based on video length and GPU availability. The README notes CUDA-compatible GPU is recommended; CPU-only will be substantially slower. The async job queue decouples upload from results, but the UI must communicate expected wait times clearly.
- **File storage growth:** Uploaded videos and H.264 `.mp4` output files can be large. Without a cleanup or archival policy, disk usage will grow unbounded. A retention policy should be defined before production deployment.
- **Interpolation artefacts:** The ball track interpolation step (`interpolate_missing_tracks()`) can introduce false positives for short missing segments. This is a known limitation acknowledged in the existing codebase.
- **Shot start detection sensitivity:** The `shot_started()` function in `utils/ball_hand.py` depends on pose keypoints and ball-hand distance estimates. Edge cases (e.g., player in unusual posture, occluded hand) may cause missed or duplicate shot start detections.
- **Multiple model variants:** Several model files exist (`best.pt`, `bestOld.pt`, `bestYT.pt`, `ballRim.pt`). The web layer must use `best.pt` as the canonical production model; loading the wrong model variant would silently produce degraded results.

---

## 3. System Architecture

### 3.1 System Level Architecture

SwishVision follows a three-tier architecture:

```
[ Client Browser ]
       |
       | HTTPS / REST
       |
[ Web Application Server ]
   - Authentication Service
   - Upload Endpoint
   - API / Results Endpoints
   - Frontend (HTML/CSS/JS or React)
       |
       | Task Queue (Celery + Redis)
       |
[ Background Worker Process ]
   - CV Pipeline Executor
   - Report Generator
       |
       | File I/O
       |
[ Storage Layer ]
   - Database (Microsoft SQL Server via pyodbc)
   - Video File Store (local filesystem: app/uploads/, output_videos/)
```

The client browser communicates exclusively with the Web Application Server over HTTPS. Video processing is offloaded to a Background Worker via a task queue. The worker writes results to the database and file system, which the web server then reads and serves to the client.

### 3.2 Software Architecture

#### 3.2.1 Module Decomposition

The system is decomposed into the following top-level modules:

**1. Auth Module**
Handles user registration, login, session management, and password operations. Exposes: `POST /api/register`, `POST /api/login`, `POST /api/logout`.

**2. Video Module**
Handles video upload, storage, and lifecycle management. Exposes: `POST /api/sessions/upload`, `GET /api/sessions`, `GET /api/sessions/{id}`, `DELETE /api/sessions/{id}`.

**3. Processing Module (Worker)**
Background Celery task that invokes the CV pipeline. Consumes jobs from the Redis queue. Writes results to the database on completion or failure.

**4. CV Pipeline (existing)**
The existing Python modules: `BallTracker`, `RimTracker`, `HumanTracker`, `ShotTracker`, and associated drawers. Called exclusively by the Processing Module. No direct web exposure.

**5. Report Module**
Reads pipeline output data from the database/filesystem and structures it for API responses. Exposes: `GET /api/sessions/{id}/report`, `GET /api/sessions/{id}/shots`.

**6. Dashboard Module**
Aggregates session data for longitudinal analytics views. Exposes: `GET /api/dashboard/summary`, `GET /api/dashboard/trends`.

**7. Frontend**
A browser-based UI consuming the REST API. Key views: Landing/Login, Registration, Dashboard, Upload, Session Results (Shot Analytics tab, Form Analysis tab), History.

#### 3.2.2 Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 (Create React App), JavaScript, served by Node.js dev server in development; built static bundle in production |
| Backend Framework | FastAPI 0.135 on Uvicorn 0.42 (ASGI) |
| ORM / Migrations | SQLAlchemy 2.0 + Alembic 1.14 |
| Task Queue | Celery 5.6 (worker run with `--pool=solo` on Windows) |
| Message Broker | Redis 7+ (`redis://localhost:6379/0`) |
| Database | Microsoft SQL Server 2019+ via `pyodbc` (`mssql+pyodbc` driver, ODBC Driver 17 for SQL Server) |
| CV Pipeline | Python 3.8+, PyTorch ≥2.0, Ultralytics YOLOv8 ≥8.0, OpenCV ≥4.8 |
| Authentication | JWT (HS256) via `python-jose`; passwords hashed with `bcrypt` (`passlib`) |
| Video Storage | Local filesystem — `app/uploads/` (raw uploads) and `output_videos/` (annotated H.264 `.mp4`, named `output_session_{id}_{n}_processed.mp4`); reports written to `reports/session_{id}_{n}_report.txt` |

---

## 4. Design Strategy

The core design strategy is **wrapper-first**: the CV pipeline is treated as a stable black box, and the web application is built around it rather than integrating deeply with its internals. This minimizes regression risk to the existing working system.

**Asynchronous by default:** Video processing is never done synchronously in a request-response cycle. Every upload enqueues a job; the client polls or receives a notification when complete.

**Separation of concerns:** The web API, the background worker, and the CV pipeline are isolated processes. This allows each to be scaled, debugged, and deployed independently.

**Database as source of truth for results:** Pipeline output (shot events, angles, report summary) is parsed from the text report and structured video data, then stored in the relational database. This enables the dashboard and history views without re-running the pipeline.

**Progressive enhancement:** The frontend is designed so that core functionality (upload, view results) works without JavaScript-heavy frameworks if needed, though a richer experience is provided where JS is available.

---

## 5. Detailed System Design

### 5.1 Database Design

#### 5.1.1 ER Diagram

> **Note:** ER diagram image to be inserted here. Described textually below.

**Entities and Relationships:**

- `User` (1) ——< `Session` (Many): one user owns many sessions
- `Session` (1) ——< `ShotEvent` (Many): one session contains many shot events
- `Session` (1) ——— `Report` (1): one session has one report
- `ShotEvent` (1) ——< `AngleFrame` (Many): one shot event has angle data across multiple frames

#### 5.1.2 Data Dictionary

**Table: `users`**

| Column | Type | Constraints | Description |
|---|---|---|---|
| `user_id` | UUID / SERIAL | PRIMARY KEY | Unique user identifier |
| `email` | VARCHAR(255) | NOT NULL, UNIQUE | User's email address (used as login) |
| `password_hash` | VARCHAR(255) | NOT NULL | bcrypt hash of the user's password |
| `display_name` | VARCHAR(100) | | User's chosen display name |
| `created_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | Account creation timestamp |

---

**Table: `sessions`**

| Column | Type | Constraints | Description |
|---|---|---|---|
| `session_id` | UUID / SERIAL | PRIMARY KEY | Unique session identifier |
| `user_id` | UUID / INT | FK → users.user_id, NOT NULL | Owner of the session |
| `video_filename` | VARCHAR(500) | NOT NULL | Original uploaded filename |
| `video_path` | VARCHAR(1000) | NOT NULL | Server-side path to raw video |
| `output_video_path` | VARCHAR(1000) | | Path to annotated output video |
| `status` | VARCHAR(50) | NOT NULL, DEFAULT 'queued' | One of: queued, processing, completed, failed |
| `upload_time` | TIMESTAMP | NOT NULL, DEFAULT NOW() | When video was uploaded |
| `completed_time` | TIMESTAMP | | When processing finished |
| `error_message` | TEXT | | Error detail if status = failed |
| `fps` | FLOAT | | Frames per second of source video |

---

**Table: `reports`**

| Column | Type | Constraints | Description |
|---|---|---|---|
| `report_id` | UUID / SERIAL | PRIMARY KEY | Unique report identifier |
| `session_id` | UUID / INT | FK → sessions.session_id, NOT NULL, UNIQUE | Parent session |
| `total_shots` | INT | NOT NULL | Total number of detected shot attempts |
| `made_shots` | INT | NOT NULL | Number of made shots |
| `shot_percentage` | FLOAT | | Made shots / total shots × 100 |
| `consistency_score` | FLOAT | | Derived score for form consistency (0–100) |
| `avg_release_angle` | FLOAT | | Average release angle across all shot attempts (degrees) |
| `feedback_text` | TEXT | | Plain-language form feedback generated by the pipeline |
| `report_file_path` | VARCHAR(1000) | | Path to the raw `.txt` report file on disk |

---

**Table: `shot_events`**

| Column | Type | Constraints | Description |
|---|---|---|---|
| `shot_id` | UUID / SERIAL | PRIMARY KEY | Unique shot event identifier |
| `session_id` | UUID / INT | FK → sessions.session_id, NOT NULL | Parent session |
| `shot_index` | INT | NOT NULL | Ordinal index of shot in the session (1-based) |
| `start_frame` | INT | | Frame index where shot attempt began |
| `release_frame` | INT | | Frame index where ball left the hand |
| `outcome` | VARCHAR(10) | | "made" or "missed" |
| `release_angle` | FLOAT | | Ball release angle at release frame (degrees) |
| `elbow_angle_at_release` | FLOAT | | Shooting elbow angle at release frame (degrees) |
| `wrist_angle_at_release` | FLOAT | | Shooting wrist angle at release frame (degrees) |

---

**Table: `angle_frames`**

| Column | Type | Constraints | Description |
|---|---|---|---|
| `angle_frame_id` | UUID / SERIAL | PRIMARY KEY | Unique record identifier |
| `shot_id` | UUID / INT | FK → shot_events.shot_id, NOT NULL | Parent shot event |
| `frame_index` | INT | NOT NULL | Video frame number |
| `elbow_angle` | FLOAT | | Elbow joint angle at this frame (degrees) |
| `wrist_angle` | FLOAT | | Wrist joint angle at this frame (degrees) |
| `knee_angle` | FLOAT | | Knee joint angle at this frame (degrees) |
| `hip_angle` | FLOAT | | Hip joint angle at this frame (degrees) |

---

### 5.2 Application Design

#### 5.2.1 Class Diagram

> **Note:** Class diagram image to be inserted here. Key classes described below.

**Existing Pipeline Classes (from codebase):**

```
BallTracker                                         # trackers/ball_tracker.py
  - model_path: str                                 # "models/best.pt"
  + get_object_tracks(frames) → list                # YOLOv8 detection per frame
  + remove_wrong_tracks(tracks) → list              # Filters invalid/noisy detections
  + interpolate_missing_tracks(tracks) → list       # Fills gaps in track sequence
  + get_ball_loco(frames, tracks) → list            # Extracts ball location data

RimTracker                                          # trackers/rim_tracker.py
  - model_path: str                                 # "models/best.pt" (same backbone)
  + remove_wrong_tracks(ball_tracks) → list         # Derives rim tracks from ball tracks
  + interpolate_missing_tracks(tracks) → list

HumanTracker                                        # trackers/human_tracker.py
  - model_path: str                                 # "models/yolov8m-pose.pt"
  + detect_frame(frames) → list                     # YOLOv8-Pose detection (17 keypoints)
  + calc_angles(frames, tracks) → list              # Computes joint angles per frame
  + get_points(frames, tracks) → list               # Extracts keypoint (x,y) coordinates
  # 17 keypoints: nose, eyes, ears, shoulders, elbows,
  # wrists, hips, knees, ankles (COCO format)

ShotTracker                                         # drawers/shot_tracker.py
  + detect_shot(frames, ball_tracks, rim_tracks) → void
  + draw_shots(frames) → list

BallTracksDrawer                                    # drawers/ball_tracks_drawer.py
  + draw(frames, tracks) → list
  + draw_ball_left(frames, ball_left_frames) → list

RimTracksDrawer                                     # drawers/rim_tracks_drawer.py
  + draw(frames, tracks) → list

HumanTracksDrawer                                   # drawers/human_tracks_drawer.py
  + draw(frames, tracks, angles, draw_boxes, draw_keypoints) → list
  + analysis(frames, angles, ball_left_frames, shot_starts, report_path) → list

# utils/vid_utils.py
read_video(path: str) → (frames: list, fps: float)
write_video(frames: list, path: str, fps: float) → void   # mp4v → transcoded to H.264 .mp4 via imageio-ffmpeg

# utils/ball_hand.py
ball_hand(ball_loco, points, frames) → list         # Returns ball_left_frames
shot_started(points, ball_left_frames) → list       # Returns shot_starts
```

**New Web Layer Classes:**

```
User
  - user_id: UUID
  - email: str
  - password_hash: str                              # bcrypt
  - display_name: str
  - created_at: datetime
  + check_password(plain: str) → bool

Session
  - session_id: UUID
  - user_id: UUID
  - video_filename: str                             # Original upload name
  - video_path: str                                 # Server storage path
  - output_video_path: str                          # H.264 .mp4 output path
  - report_file_path: str                           # Path to .txt report
  - status: str                                     # queued|processing|completed|failed
  - upload_time: datetime
  - completed_time: datetime
  - fps: float
  + is_owned_by(user_id: UUID) → bool

Report
  - report_id: UUID
  - session_id: UUID
  - total_shots: int
  - made_shots: int
  - shot_percentage: float
  - consistency_score: float
  - avg_release_angle: float
  - feedback_text: str                              # Parsed from {vidname}_report.txt

ShotEvent
  - shot_id: UUID
  - session_id: UUID
  - shot_index: int
  - start_frame: int                                # From shot_starts list
  - release_frame: int
  - outcome: str                                    # "made" | "missed"
  - release_angle: float
  - elbow_angle_at_release: float
  - wrist_angle_at_release: float

ProcessingTask                                      # Celery task
  + run(session_id: UUID) → void
  - _invoke_pipeline(video_path: str, vid_name: str) → void
  - _parse_report(report_path: str) → dict          # Reads {vidname}_report.txt
  - _save_results(session_id, results: dict) → void
  - _update_status(session_id, status: str) → void
  - _cleanup_intermediates(vid_name: str) → void    # Removes angs.txt, xy_coords.txt etc.
```

#### 5.2.2 Sequence Diagrams

> **Note:** Sequence diagram images to be inserted here. Key flows described below.

**Sequence 1: Video Upload and Processing**

```
Player Browser → API Server: POST /api/sessions/upload (video file)
API Server → Storage: Save video to disk
API Server → Database: INSERT sessions (status=queued)
API Server → Task Queue: Enqueue process_video(session_id)
API Server → Player Browser: 202 Accepted { session_id }

[Background]
Task Queue → Worker: Dequeue process_video(session_id)
Worker → Database: UPDATE sessions SET status=processing
Worker → CV Pipeline: main_pipeline(vidname)
CV Pipeline → Storage: Write output video + report
Worker → Database: INSERT reports, shot_events, angle_frames
Worker → Database: UPDATE sessions SET status=completed

[Polling]
Player Browser → API Server: GET /api/sessions/{id}
API Server → Database: SELECT sessions WHERE id=...
API Server → Player Browser: { status: "completed" }
```

**Sequence 2: View Shot Analytics**

```
Player Browser → API Server: GET /api/sessions/{id}/report
API Server → Auth: Verify token, check ownership
API Server → Database: SELECT reports, shot_events WHERE session_id=...
API Server → Player Browser: { shot_percentage, consistency_score, shots: [...] }
Player Browser → API Server: GET /api/sessions/{id}/output_video
API Server → Storage: Stream output video file
API Server → Player Browser: Video stream (chunked)
```

#### 5.2.3 State Diagrams

> **Note:** State diagram images to be inserted here. Key states described below.

**Session State Machine:**

```
[queued] → (worker picks up job) → [processing]
[processing] → (pipeline completes successfully) → [completed]
[processing] → (pipeline throws exception) → [failed]
[completed] → (user deletes) → [deleted]
[failed] → (user retries) → [queued]
```

**User Authentication State:**

```
[anonymous] → (register / login success) → [authenticated]
[authenticated] → (logout / token expiry) → [anonymous]
[authenticated] → (password change) → [authenticated]
```

#### 5.2.4 Activity Diagrams

> **Note:** Activity diagram images to be inserted here. Key flows described below.

**Activity: Full Pipeline Execution (Worker)**

1. Receive `session_id` from Celery queue
2. Load session record from DB; set status = `processing`
3. Read video frames and FPS using `read_video(video_path)` → `utils/vid_utils.py`
4. Instantiate `BallTracker(model_path="models/best.pt")`
5. Instantiate `RimTracker(model_path="models/best.pt")`
6. Instantiate `HumanTracker(model_path="models/yolov8m-pose.pt")`
7. Run `ball_tracker.get_object_tracks(frames)` → raw ball tracks
8. Run `human_tracker.detect_frame(frames)` → human keypoint tracks
9. Run `human_tracker.calc_angles(frames, human_tracks)` → joint angles per frame
10. Run `human_tracker.get_points(frames, human_tracks)` → keypoint coordinates
11. Run `ball_tracker.remove_wrong_tracks(ball_tracks)` → cleaned ball tracks
12. Run `rim_tracker.remove_wrong_tracks(ball_tracks)` → rim tracks (derived from ball)
13. Run `ball_tracker.interpolate_missing_tracks(ball_tracks)` → interpolated ball tracks
14. Run `ball_tracker.get_ball_loco(frames, interpolated_ball_tracks)` → ball locations
15. Run `rim_tracker.interpolate_missing_tracks(rim_tracks)` → interpolated rim tracks
16. Run `ball_hand(ball_loco, points, frames)` → `ball_left_frames`
17. Run `shot_started(points, ball_left_frames)` → `shot_starts`
18. Instantiate drawer classes; run `HumanTracksDrawer.draw()` with keypoints overlay
19. Run `HumanTracksDrawer.analysis(frames, angles, ball_left_frames, shot_starts, report_path)` → annotated frames + report `.txt` written to `reports/`
20. Run `ShotTracker.detect_shot(frames, interpolated_ball_tracks, rim_tracks)`
21. Run `ShotTracker.draw_shots(frames)` → final annotated frames
22. Run `write_video(frames, output_path, fps)` → H.264 `.mp4` written to `output_videos/output_session_{id}_{n}_processed.mp4` (via `mp4v` then `imageio-ffmpeg` transcode to `libx264`)
23. Parse `{vidname}_report.txt`; INSERT into `reports`, `shot_events`, `angle_frames` tables
24. UPDATE session status = `completed`, set `completed_time`
25. Clean up transient intermediate files (`angs.txt`, `xy_coords.txt`, `detections.txt`, `ball_locl.txt`)
26. **[On any unhandled exception]** → UPDATE session status = `failed`, store `error_message`, log stack trace

**Activity: User Registration**

1. User submits registration form (name, email, password)
2. Validate email format → if invalid, return error
3. Check email uniqueness in DB → if exists, return error
4. Validate password strength → if weak, return error
5. Hash password with bcrypt
6. Insert user record into DB
7. Create session token
8. Return 201 Created with token

#### 5.2.5 System User Interface

> **Note:** UI mockup images to be inserted here. Key screens described below.

**Screen 1: Dashboard / Home**
- Header with logo, nav links (Dashboard, History, Upload, Logout)
- Summary stats card: total sessions, lifetime shot percentage, average consistency score
- Recent sessions list (last 5): thumbnail, date, shot%, status badge
- Prominent "Upload New Video" button

**Screen 2: Upload Page**
- Drag-and-drop area for video file with supported formats noted
- File size limit displayed
- Progress bar for upload
- After successful upload: "Your video is queued for processing" with session link

**Screen 3: Session Results — Shot Analytics Tab**
- Shot percentage displayed prominently (large number)
- Made / Missed doughnut chart
- Per-shot table: shot #, outcome, release angle, elbow angle at release
- Embedded video player for annotated output video

**Screen 4: Session Results — Form Analysis Tab**
- Consistency Score displayed (0–100, with descriptive label)
- Line chart: elbow angle across frames for each detected shot (overlaid)
- Average release angle with reference range (ideal: 45–55°)
- Feedback text block from report

**Screen 5: Session History**
- Table/list of all sessions sorted by date
- Columns: Date, Video Name, Shots, Made%, Consistency Score, Status, Actions
- Trend chart: shot percentage over time (line chart)

---

## 6. References

- Project SRS: SwishVision Software Requirements Specification v1.00
- Ultralytics YOLOv8 Documentation: https://docs.ultralytics.com
- Existing codebase: `main.py`, `trackers/`, `drawers/`, `utils/`

---

## 7. Appendices

### Appendix A — API Endpoint Summary

| Method | Endpoint | Auth Required | Description |
|---|---|---|---|
| POST | `/api/register` | No | Create new user account |
| POST | `/api/login` | No | Authenticate and receive token |
| POST | `/api/logout` | Yes | Invalidate current session |
| POST | `/api/sessions/upload` | Yes | Upload a video for processing |
| GET | `/api/sessions` | Yes | List all sessions for authenticated user |
| GET | `/api/sessions/{id}` | Yes | Get session status and metadata |
| DELETE | `/api/sessions/{id}` | Yes | Delete a session and its data |
| GET | `/api/sessions/{id}/report` | Yes | Get full report data for a session |
| GET | `/api/sessions/{id}/shots` | Yes | Get all shot events for a session |
| GET | `/api/sessions/{id}/output_video` | Yes | Stream annotated output video |
| GET | `/api/dashboard/summary` | Yes | Get aggregate stats for authenticated user |
| GET | `/api/dashboard/trends` | Yes | Get historical trend data for charts |

### Appendix B — File System Layout

The web application maps to the existing project structure as follows:

```
Swish-Vision/                        ← Project root
├── main.py                          ← Pipeline entry point (called by worker)
├── trackers/                        ← Tracker modules (BallTracker, RimTracker, HumanTracker)
├── drawers/                         ← Drawer/visualization modules + ShotTracker
├── utils/                           ← vid_utils.py, ball_hand.py, stubs_utils.py
├── models/                          ← Model weights (best.pt, yolov8m-pose.pt, variants)
│
├── input_videos/                    ← Worker copies uploaded video here before processing
├── output_videos/                   ← Pipeline writes H.264 .mp4 output here
├── reports/                         ← Pipeline writes {vidname}_report.txt here
│
├── Basketball-1/                    ← Training dataset (Roboflow, not used at runtime)
├── runs/                            ← YOLO training run history (not used at runtime)
│
└── app/                            ← Web application layer (delivered)
    ├── main.py                     ← FastAPI app factory + lifespan
    ├── config.py                   ← Pydantic Settings (reads .env)
    ├── database.py                 ← SQLAlchemy engine + SessionLocal (mssql+pyodbc)
    ├── api/
    │   ├── auth.py                 ← /api/auth/* (register, login, logout, JWT)
    │   ├── sessions.py             ← /api/sessions/* (upload, list, get, delete, report, shots, video)
    │   └── dashboard.py            ← /api/dashboard/* (summary, trends)
    ├── core/
    │   ├── security.py             ← bcrypt + JWT helpers, get_current_user dependency
    │   └── middleware.py           ← CORS, security headers, optional HTTPS redirect
    ├── models/                     ← SQLAlchemy ORM models
    │   ├── user.py
    │   ├── session.py
    │   ├── report.py
    │   ├── shot_event.py
    │   ├── angle_frame.py
    │   └── revoked_token.py
    ├── schemas/                    ← Pydantic request/response schemas
    ├── tasks/
    │   ├── pipeline_task.py        ← Celery app + process_video task
    │   └── report_parser.py        ← Parses {vidname}_report.txt → DB rows
    ├── migrations/                 ← Alembic environment + versions/
    ├── frontend/                   ← React (Create React App)
    │   ├── package.json
    │   ├── public/
    │   └── src/                    ← App.js, AuthContext.js, NavBar.jsx, pages/, components/
    ├── uploads/                    ← Raw user-uploaded videos (pre-processing)
    ├── requirements.txt            ← Backend Python dependencies
    └── DEVELOPER_README.md         ← Web-app-specific developer guide
```

### Appendix C — Environment Variables

| Variable | Description |
|---|---|
| `ENV` | Environment name: `development` \| `staging` \| `production` (controls HTTPS redirect middleware) |
| `DB_SERVER` | SQL Server hostname/instance (e.g. `localhost`) |
| `DB_NAME` | SQL Server database name (e.g. `SwishVision`) |
| `DB_DRIVER` | ODBC driver name (e.g. `ODBC Driver 17 for SQL Server`) |
| `DB_TRUSTED_CONNECTION` | `true` for Windows auth; `false` to use `DB_USERNAME`/`DB_PASSWORD` |
| `DB_USERNAME` / `DB_PASSWORD` | SQL auth credentials (only when `DB_TRUSTED_CONNECTION=false`) |
| `REDIS_URL` | Redis connection string for the Celery broker (e.g. `redis://localhost:6379/0`) |
| `CELERY_RESULT_BACKEND` | Celery result backend URL (typically same as `REDIS_URL`) |
| `SECRET_KEY` | JWT signing secret — must be replaced before any non-local deploy |
| `ADMIN_RESET_KEY` | Shared secret required for `POST /api/auth/admin/force-reset-password` |
| `UPLOAD_DIR` | Path for raw uploaded videos (default `app/uploads`) |
| `OUTPUT_DIR` | Path for annotated H.264 `.mp4` outputs (default `output_videos`) |
| `MODEL_DIR` | Path to YOLOv8 model weights (default `models`) |
| `MAX_UPLOAD_SIZE_MB` | Maximum allowed video upload size in MB (default `500`) |
