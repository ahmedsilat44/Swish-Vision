# Software Requirements Specification

## SwishVision — Basketball Analytics Platform

**Version:** 1.00

| Project Team | |
|---|---|
| Mallick Mikaal Imam | |
| Minhaj ul Hassan | |
| Muhammad Ahmed Silat | |
| Syed Muhammad Sameer Hassan | |

| Submission Date | TBD |
|---|---|

---

## Document History

| Version | Name of Person | Date | Description of Change |
|---|---|---|---|
| 1.00 | Ahmed Silat | 16th March | Initial draft — project proposal translated to SRS |

---

## Distribution List

| Name | Role |
|---|---|
| | Supervisor / Course Instructor |
| | Project Manager |

---

## Document Sign-Off

| Version | Sign-off Authority | Project Role | Sign-off Date |
|---|---|---|---|
| 1.00 | | Supervisor | |

---

## Table of Contents

1. Introduction
   - 1.1 Purpose of Document
   - 1.2 Intended Audience
   - 1.3 Document Convention
2. Overall System Description
   - 2.1 Project Background
   - 2.2 Project Scope
   - 2.3 Not In Scope
   - 2.4 Project Objectives
   - 2.5 Stakeholders
   - 2.6 Operating Environment
   - 2.7 System Constraints
   - 2.8 Assumptions & Dependencies
3. Software Development Methodology
4. System Analysis & Requirement Gathering
5. External Interface Requirements
   - 5.1 Hardware Interfaces
   - 5.2 Software Interfaces
   - 5.3 Communications Interfaces
6. Functional Requirements
   - 6.1 Data Flow Diagrams
   - 6.2 CRUD and Association Matrices
   - 6.3 Use Cases
7. Non-Functional Requirements
   - 7.1 Performance Requirements
   - 7.2 Safety Requirements
   - 7.3 Security Requirements
   - 7.4 User Documentation
8. References
9. Appendices

---

## 1. Introduction

### 1.1 Purpose of Document

The purpose of this document is to outline the requirements for the construction of SwishVision, a web-based basketball analytics platform. It documents functional and non-functional requirements, system constraints, interface requirements, design constraints, and other factors vital to ensuring the software meets its intended goals. It serves as the primary reference for development, validation, and stakeholder alignment.

### 1.2 Intended Audience

The intended audience includes the course supervisor, the project development team, and any future maintainers of the system. The development team requires this document to ensure that implementation efforts are aligned with the stated purpose of the software. Supervisors and evaluators use it to assess the completeness and clarity of requirements. Future developers or contributors may reference it to understand system boundaries and expected behaviors.

### 1.3 Document Convention

- Font: Arial
- Size: 11
- Line Spacing: 1.5

---

## 2. Overall System Description

### 2.1 Project Background

Basketball players at the amateur and semi-professional level lack access to the kind of analytical feedback that professional athletes receive from coaching staff and sports scientists. Critical aspects of shooting mechanics — such as release angle, elbow alignment, and follow-through consistency — occur too rapidly to be reliably analyzed by the naked eye. Furthermore, there exist no affordable, accessible tools that can automatically process practice footage and return structured, data-driven feedback.

SwishVision addresses this gap by building a web-based platform that accepts uploaded basketball footage and returns detailed performance analytics powered by computer vision and pose estimation. A fully working command-line prototype already exists (`main.py`), capable of detecting ball trajectory, rim interaction, and player skeletal keypoints using YOLOv8-based models. The object detection model (`best.pt`) was fine-tuned over 13 training iterations on the Basketball-1 dataset sourced from Roboflow (Eagle Eye workspace, CC BY 4.0), which provides labeled bounding boxes for three classes: basketball, rim, and sports ball. Human pose estimation uses the pre-trained `yolov8m-pose.pt` model, tracking 17 body joint keypoints per player.

### 2.2 Project Scope

- Allow users to create accounts and upload basketball practice footage via a web interface.
- Process uploaded video using the existing CV pipeline: ball tracking, rim tracking, and human pose estimation.
- Detect shot attempts and classify them as made or missed based on ball-rim interaction.
- Analyze shooting form using skeletal keypoints and compute joint angles across frames.
- Generate a per-session report including shot percentage, consistency scores, and mechanical feedback.
- Display session history and allow longitudinal comparison of performance across sessions.
- Provide a personal analytics dashboard summarizing trends over time.

### 2.3 Not In Scope

- Real-time (live) video analysis — the system processes pre-recorded footage only.
- Multi-player team analytics — the current scope focuses on individual player analysis.
- Mobile native applications (iOS/Android) — the platform is web-only.
- Analysis of non-shooting basketball actions such as dribbling, defense, or passing.

### 2.4 Project Objectives

The primary objective of SwishVision is to democratize access to professional-grade shooting analytics for amateur basketball players. Specific objectives include:

- Automate the detection and classification of shot attempts with high accuracy.
- Provide quantifiable metrics for shooting form that are consistent and reproducible.
- Enable players to track their progress over time through a persistent session history.
- Present results in a format that is intuitive and actionable for non-technical users.
- Expose the underlying CLI pipeline through a web interface without requiring users to have any technical knowledge.

### 2.5 Stakeholders

**Amateur Basketball Players (End Users)**
The primary users of the system. They upload footage, view their analytics, and act on the feedback provided. Their goal is to improve shooting form and track progress.

**Development Team**
Responsible for building the system. Must ensure the web platform correctly integrates with the existing CV pipeline and delivers results reliably.

**Course Supervisor**
Evaluates the project from an academic standpoint. Requires documentation, working demonstrations, and compliance with software engineering standards.


### 2.6 Operating Environment

- **Server:** Linux or Windows environment running Python 3.8+, supporting PyTorch and YOLOv8 inference
- **Client:** Any modern web browser (Chrome, Firefox, Safari, Edge) — no installation required
- **GPU:** CUDA-compatible NVIDIA GPU strongly recommended on the server side (CUDA-compatible PyTorch 2.0+); CPU fallback supported but will result in significantly longer processing times
- **RAM:** Minimum 4GB; 8GB+ recommended for smooth inference
- **Storage:** Minimum 10GB for model files, uploaded videos, and generated outputs
- **Network:** Standard HTTPS internet connection for the client; server requires sufficient bandwidth for video uploads
- **Output Codec:** XVID (AVI container) for processed output videos

### 2.7 System Constraints

- Video processing is computationally intensive; processing time scales with video length and resolution. Users should expect non-instantaneous results.
- The accuracy of shot detection and pose estimation is constrained by video quality, camera angle, and lighting conditions.
- The system relies on pre-trained YOLOv8 models (`best.pt` for ball/rim, `yolov8m-pose.pt` for human pose). Model performance is bounded by the training data.
- The web application must handle asynchronous video processing — the upload and the results must be decoupled with a job queue or background worker.

### 2.8 Assumptions & Dependencies

- Uploaded videos are recorded from a side-on or slightly elevated angle that captures both the player and the basket simultaneously.
- The player being analyzed occupies the majority of the frame; multi-player crowded scenes may reduce accuracy.
- The server environment has Python 3.8+, PyTorch 2.0+, Ultralytics YOLOv8 (`ultralytics>=8.0.0`), OpenCV (`opencv-python>=4.8.0`), NumPy, and Matplotlib installed.
- Model files (`models/best.pt` and `models/yolov8m-pose.pt`) are present on the server. `best.pt` is the fine-tuned ball/rim detection model; `yolov8m-pose.pt` is the Ultralytics pretrained medium pose model.
- Users have access to a device capable of recording and uploading video (smartphone or camera).
- The existing CLI pipeline (`main.py` and its associated `trackers/`, `drawers/`, and `utils/` modules) is treated as a stable, working subsystem and will be wrapped — not rewritten — by the web layer.
- Intermediate data files produced by the pipeline (e.g., `angs.txt`, `xy_coords.txt`, `detections.txt`, `ball_locl.txt`) are transient and do not need to be persisted beyond a single processing run.
- Text reports are generated per video and stored in the `reports/` directory (e.g., `reports/vid18_report.txt`).

---

## 3. Software Development Methodology

An **Incremental Development** methodology will be used. This choice is appropriate because:

- A functioning core (the CV pipeline) already exists and can serve as the first increment.
- Requirements for the web layer can be developed and validated progressively.
- Early increments (video upload + basic results) can be demonstrated to the supervisor for feedback before more advanced features (dashboard, history, detailed analytics) are built.
- The team can parallelize work on the backend processing pipeline and the frontend interface.

**Planned Increments:**

| Increment | Deliverable |
|---|---|
| 1 | User registration/login + video upload endpoint |
| 2 | Background processing integration — pipeline runs on uploaded video |
| 3 | Shot detection results displayed on frontend |
| 4 | Pose/form analysis results and report generation |
| 5 | Session history and longitudinal dashboard |

---

## 4. System Analysis & Requirement Gathering

Requirements were gathered through the following approaches:

- **Prototype Analysis:** The existing CLI pipeline (`main.py`) was examined to understand what the system already produces: ball tracks, rim tracks, human keypoints, joint angles, ball-hand contact frames, shot start events, and output video with overlays.
- **README Review:** The project README was reviewed to extract the confirmed module inventory (`trackers/`, `drawers/`, `utils/`), model files, dataset information, dependency versions, and the pipeline's 12-step processing flow.
- **Output Inspection:** Sample output reports (e.g., `reports/vid13_1_report.txt`, `reports/vid14_1_report.txt`) and annotated output videos in `output_videos/` were reviewed to understand what data the pipeline already surfaces.
- **Stub File Analysis:** The presence of `stubs_utils.py` and intermediate data files (`angs.txt`, `xy_coords.txt`, `detections.txt`, `ball_locl.txt`) indicates the pipeline already supports caching intermediate results — this will be leveraged in the web processing architecture to avoid re-running stages on failure.
- **Project Proposal Review:** The project proposal document was analyzed to extract stated goals, key features, and value propositions.
- **Gap Analysis:** The difference between what the CLI pipeline produces and what a web user needs was analyzed to derive web-specific requirements (authentication, upload, async processing, results display, dashboard).

---

## 5. External Interface Requirements

### 5.1 Hardware Interfaces

- A server machine with sufficient CPU/RAM to run Python-based inference workloads. A CUDA-compatible NVIDIA GPU is strongly recommended.
- End users require only a standard computer or smartphone with internet access and a web browser.
- A camera or smartphone capable of recording video at a minimum of 720p resolution is recommended for users to capture footage.

### 5.2 Software Interfaces

- **Python 3.8+** — core runtime for the CV pipeline
- **Ultralytics YOLOv8** (`ultralytics>=8.0.0`) — object detection and pose estimation
- **PyTorch 2.0+** (`torch>=2.0.0`, `torchvision>=0.15.0`) — deep learning inference backend
- **OpenCV** (`opencv-python>=4.8.0`) — video reading, frame processing, and output writing (XVID codec)
- **NumPy** (`numpy>=1.24.0`) — array operations and numerical calculations
- **Matplotlib** (`matplotlib>=3.7.0`) — plotting and visualization within reports
- **Web Framework (e.g., Flask or ASP.NET)** — to expose the pipeline as a web service
- **Database (SQL Server)** — to store user accounts, session metadata, and report data
- **Task Queue (e.g., Celery + Redis, or equivalent)** — to manage asynchronous video processing jobs
- **Web Browser** — Chrome, Firefox, Safari, or Edge (client side)

### 5.3 Communications Interfaces

- **HTTPS** — all client-server communication encrypted via TLS
- **REST API** — the frontend communicates with the backend via HTTP/REST endpoints for upload, status polling, and results retrieval
- **WebSocket or Polling** — for job status updates (i.e., notifying the user when processing is complete)

---

## 6. Functional Requirements

### 6.1 Data Flow Diagrams

> **Note:** DFD diagrams to be added as images upon creation. Described textually below.

**Context Level DFD:**
The user interacts with the SwishVision Web Platform. The user submits a video and receives back analytics results and reports. The platform internally invokes the CV Processing Engine and reads/writes from the Database.

**Level 0 DFD:**
Major processes: (1) User Authentication, (2) Video Upload & Management, (3) CV Processing Pipeline, (4) Report Generation, (5) Dashboard & History View.

**Level 1 DFD — Process 3: CV Processing Pipeline:**
Inputs: raw video file. Sub-processes: Ball Tracking, Rim Tracking, Human Pose Estimation, Ball-Hand Contact Detection, Shot Start Detection, Shot Outcome Detection. Outputs: annotated output video, joint angle data, shot events, report text.

### 6.2 CRUD and Association Matrices

**Data to Location CRUD Matrix:**

| Entity.Attribute | Player (Self) | System (Backend) | Admin |
|---|---|---|---|
| **User** | | | ALL |
| .UserID | R | CRUD | CRUD |
| .Email | RU | CRUD | CRUD |
| .PasswordHash | | CRUD | CRUD |
| **Session** | | | ALL |
| .SessionID | R | CRUD | CRUD |
| .UserID | R | R | CRUD |
| .UploadDate | R | CRUD | CRUD |
| .VideoPath | | CRUD | CRUD |
| .Status | R | CRUD | CRUD |
| **Report** | | | ALL |
| .ReportID | R | CRUD | CRUD |
| .ShotPercentage | R | CRU | CRUD |
| .ConsistencyScore | R | CRU | CRUD |
| .FormFeedback | R | CRU | CRUD |
| **ShotEvent** | | | ALL |
| .FrameIndex | R | CRU | CRUD |
| .Outcome (made/missed) | R | CRU | CRUD |
| .ReleaseAngle | R | CRU | CRUD |

**Process to Location Association Matrix:**

| Process | Player | Backend | Admin |
|---|---|---|---|
| Register / Login | X | X | X |
| Upload Video | X | X | |
| Trigger Processing | | X | |
| View Results | X | X | |
| View Session History | X | X | |
| Delete Session | X | | X |
| Manage Users | | | X |

### 6.3 Use Cases

**Use Case Diagram (described textually):**
Actors: Player, System Backend, Admin. Primary use cases: Register, Login, Upload Video, View Processing Status, View Shot Analytics, View Form Analysis, View Session History, Delete Session. Include relationships: Upload Video «includes» Trigger Processing; View Shot Analytics «extends» View Session History.

---

**Use Case 1: User Registration**

| Field | Details |
|---|---|
| Use-Case Name | User Registration |
| Use-Case ID | SV-01 |
| Use-Case Type | Business Requirement |
| Priority | High |
| Primary Actor | Player |
| Other Actors | System Backend |
| Description | A new user creates an account on SwishVision by providing their name, email address, and password. The system validates the inputs and creates the account. |
| Precondition | The user does not already have an account with the provided email. |
| Trigger | The user navigates to the registration page and submits the form. |
| Typical Course | **Actor Action:** User provides name, email, password. **System Response:** Validates inputs, hashes password, stores user record, redirects to dashboard. |
| Alternate Courses | Alt 1 — Email already registered: system displays error and prompts login. Alt 2 — Weak password: system rejects and shows requirements. |
| Conclusion | User account is created successfully. |
| Post Condition | User is logged in and redirected to their (empty) dashboard. |

---

**Use Case 2: Upload Practice Video**

| Field | Details |
|---|---|
| Use-Case Name | Upload Practice Video |
| Use-Case ID | SV-02 |
| Use-Case Type | Business Requirement |
| Priority | High |
| Primary Actor | Player |
| Other Actors | System Backend |
| Description | A logged-in user uploads a basketball practice video file. The system accepts the file, stores it, and enqueues it for processing. |
| Precondition | User is authenticated. |
| Trigger | User selects a video file and clicks Upload on the dashboard. |
| Typical Course | **Actor Action:** User selects video and submits. **System Response:** Validates file format and size, stores video, creates a Session record with status "queued", returns a job ID. |
| Alternate Courses | Alt 1 — Unsupported format: system rejects file with format error. Alt 2 — File exceeds size limit: system rejects and shows size limit message. |
| Conclusion | Video is stored and queued for processing. |
| Post Condition | Session record created with status "queued"; user can monitor processing status. |

---

**Use Case 3: Process Video (CV Pipeline)**

| Field | Details |
|---|---|
| Use-Case Name | Process Video |
| Use-Case ID | SV-03 |
| Use-Case Type | System |
| Priority | High |
| Primary Actor | System Backend |
| Other Actors | None |
| Description | The backend worker picks up a queued video job and runs the CV pipeline: ball tracking, rim tracking, pose estimation, shot event detection, and report generation. |
| Precondition | A video has been uploaded and a job is in the queue. |
| Trigger | Background task worker dequeues the job. |
| Typical Course | **System Action:** Runs `main_pipeline()` on the video. Generates annotated output video, shot events, joint angle data, and report text file. Updates Session status to "completed". |
| Alternate Courses | Alt 1 — Inference error (e.g., model not loaded): session status set to "failed", error logged. Alt 2 — No shots detected: report generated with zero shot count and appropriate message. |
| Conclusion | Session results are stored and status updated. |
| Post Condition | Session status is "completed"; Report and ShotEvent records are created in the database. |

---

**Use Case 4: View Shot Analytics**

| Field | Details |
|---|---|
| Use-Case Name | View Shot Analytics |
| Use-Case ID | SV-04 |
| Use-Case Type | Business Requirement |
| Priority | High |
| Primary Actor | Player |
| Other Actors | System Backend |
| Description | The user views the results of a processed session, including shot percentage, annotated output video, per-shot outcomes, and a consistency score. |
| Precondition | The session has been processed (status = "completed"). |
| Trigger | User clicks on a completed session from their dashboard. |
| Typical Course | **Actor Action:** User selects session. **System Response:** Displays shot percentage, made/missed breakdown, annotated output video player, and consistency score. |
| Alternate Courses | Alt 1 — Session still processing: user sees a progress indicator. Alt 2 — Session failed: user sees a failure message with guidance. |
| Conclusion | User reviews their shot data. |
| Post Condition | No state change; data is read-only from user perspective. |

---

**Use Case 5: View Form Analysis**

| Field | Details |
|---|---|
| Use-Case Name | View Form Analysis |
| Use-Case ID | SV-05 |
| Use-Case Type | Business Requirement |
| Priority | High |
| Primary Actor | Player |
| Other Actors | System Backend |
| Description | The user views the biomechanical analysis of their shooting form, including release angle data, elbow alignment feedback, and a consistency rating derived from joint angle calculations across all detected shot attempts. |
| Precondition | The session has been processed and pose estimation data is available. |
| Trigger | User navigates to the Form tab of a completed session. |
| Typical Course | **Actor Action:** User opens form tab. **System Response:** Displays per-shot joint angle charts, average release angle, consistency score, and plain-language feedback highlights. |
| Alternate Courses | Alt 1 — Pose data unavailable (player not visible in frame): system shows a message indicating insufficient pose data. |
| Conclusion | User gains insight into the technical aspects of their shooting form. |
| Post Condition | None. |

---

**Use Case 6: View Session History**

| Field | Details |
|---|---|
| Use-Case Name | View Session History |
| Use-Case ID | SV-06 |
| Use-Case Type | Business Requirement |
| Priority | Medium |
| Primary Actor | Player |
| Other Actors | System Backend |
| Description | The user views a chronological list of all their past sessions, with summary statistics for each, and can compare metrics across sessions to track progress over time. |
| Precondition | The user has at least one completed session. |
| Trigger | User navigates to the History or Dashboard page. |
| Typical Course | **Actor Action:** User views session list. **System Response:** Displays sessions sorted by date with shot percentage, consistency score, and a trend chart showing change over time. |
| Alternate Courses | Alt 1 — No sessions: displays a prompt encouraging the user to upload their first video. |
| Conclusion | User views longitudinal progress. |
| Post Condition | None. |

---

**Use Case 7: Delete Session**

| Field | Details |
|---|---|
| Use-Case Name | Delete Session |
| Use-Case ID | SV-07 |
| Use-Case Type | Business Requirement |
| Priority | Low |
| Primary Actor | Player |
| Other Actors | System Backend |
| Description | The user permanently deletes a session and all associated data (video files, reports, shot events). |
| Precondition | Session belongs to the authenticated user. |
| Trigger | User clicks Delete on a session entry. |
| Typical Course | **Actor Action:** User confirms deletion. **System Response:** Removes video files, report, and all associated database records. |
| Alternate Courses | Alt 1 — Session currently processing: system rejects deletion until processing completes or fails. |
| Conclusion | Session and all related data removed. |
| Post Condition | Session no longer appears in user history. |

---

## 7. Non-Functional Requirements

### 7.1 Performance Requirements

- Video processing time should complete within a reasonable duration relative to video length. As a guideline, a 1-minute video should complete processing within 5 minutes on a server with a CUDA GPU.
- The web frontend should respond to user interactions (page loads, navigation) within 2 seconds under normal load.
- File upload should support videos up to 500MB without timeout.

### 7.2 Safety Requirements

- Uploaded video files must be stored in an isolated server directory not directly accessible via URL, to prevent unauthorized access.
- Processing jobs that fail must be caught and logged without crashing the worker or affecting other queued jobs.
- Output files (annotated video, report) should be verified for existence before being served to the user.
- Data loss on processing failure should be minimal — the original uploaded video must not be deleted on a failed processing run.

### 7.3 Security Requirements

- User passwords must be stored as hashes.
- All endpoints that access user-specific data must verify session authentication and that the requesting user owns the resource (authorization check).
- File upload endpoints must validate MIME type and file extension to prevent malicious file uploads.
- The system must enforce HTTPS in production.
- API endpoints should be protected against vulnerabilities

### 7.4 User Documentation

- An in-app help section or onboarding guide explaining how to record a suitable video and what the analytics results mean.
- A README for developers describing how to set up the environment, install dependencies, and run the application.
- Inline tooltips or labels in the UI explaining key metrics (e.g., "Consistency Score", "Release Angle").

---

## 8. References

- YOLOv8 Documentation — Ultralytics: https://docs.ultralytics.com
- Project Proposal: SwishVision — submitted by team members Mallick Mikaal Imam, Minhaj ul Hassan, Muhammad Ahmed Silat, Syed Muhammad Sameer Hassan
- Existing CLI codebase: `main.py` and associated tracker/drawer modules
- IEEE Std 830-1998 — Recommended Practice for Software Requirements Specifications (used as structural reference)

---

## 9. Appendices

### Appendix A — Glossary

| Term | Definition |
|---|---|
| YOLOv8 | A state-of-the-art real-time object detection model from Ultralytics, used here for ball, rim, and human pose detection. |
| Pose Estimation | A computer vision technique that detects and tracks the positions of human body joints (keypoints) in video frames. |
| Release Angle | The angle of the ball's trajectory at the point it leaves the player's hands during a shot attempt. |
| Consistency Score | A derived metric indicating how uniform a player's shooting form is across multiple shot attempts in a session. |
| Shot Event | A detected instance where the player attempts a shot, including the frame it begins, the ball trajectory, and the outcome (made/missed). |
| CV Pipeline | The chain of computer vision processing steps: ball tracking → rim tracking → pose estimation → shot detection → report generation. |
| Session | A single uploaded video and all data derived from its processing. |
