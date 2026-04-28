# Swish-Vision
**AI Basketball Shooting Practice Analysis System**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-brightgreen)
![OpenCV](https://img.shields.io/badge/OpenCV-Video%20Processing-red)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Technical Stack](#technical-stack)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Modules and Components](#modules-and-components)
- [Models and Datasets](#models-and-datasets)
- [Configuration](#configuration)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

**Swish-Vision** is an advanced computer vision system designed to analyze basketball shooting practice videos. It uses state-of-the-art deep learning models (YOLOv8 and pose estimation models) to detect, track, and analyze:

- **Basketball movement** and trajectory during shots
- **Rim and hoop detection** for target localization
- **Human pose and body mechanics** during shooting
- **Shot outcomes** and performance metrics

The system processes video input, extracts frame data, applies multi-object tracking algorithms, and generates annotated output videos with detailed analysis reports.

### Key Use Cases
- Basketball training analytics
- Performance improvement tracking
- Shooting form analysis
- Shot success prediction and feedback
- Video-based coaching assistance

---

## Features

### Core Capabilities
- ✅ **Multi-Object Detection**: Simultaneous detection of basketball, rim, and players
- ✅ **Advanced Tracking**: State-of-the-art tracker implementation for ball and rim tracking
- ✅ **Pose Estimation**: Human body pose detection and joint tracking
- ✅ **Shot Detection**: Automatic detection of shot initiation and completion
- ✅ **Trajectory Analysis**: Ball trajectory interpolation and smoothing
- ✅ **Hand-Ball Interaction**: Detection of hand position relative to the ball
- ✅ **Video Annotation**: Frame-by-frame visualization with bounding boxes and tracking data
- ✅ **Performance Reporting**: Comprehensive shot analysis reports with metrics

### Supported Detections
- Basketball (ball)
- Rim/Hoop
- Players (human pose with 17 joint keypoints)

---

## System Architecture

```
Input Video
    ↓
[Frame Extraction] → OpenCV Video Reader
    ↓
┌─────────────────────────────────────────┐
│  Multi-Model Detection Pipeline         │
├─────────────────────────────────────────┤
│ • BallTracker (YOLOv8)                  │
│ • RimTracker (YOLOv8)                   │
│ • HumanTracker (YOLOv8-Pose)           │
└─────────────────────────────────────────┘
    ↓
[Data Processing & Filtering]
    ├── Track Validation
    ├── Trajectory Interpolation
    ├── Angle Calculation (Pose)
    ├── Ball-Hand Detection
    └── Shot Analysis
    ↓
[Visualization & Annotation]
    ├── BallTracksDrawer
    ├── RimTracksDrawer
    ├── HumanTracksDrawer
    ├── ShotTracker
    └── Frame Composition
    ↓
Output Video + Analysis Reports
```

---

## Technical Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Framework** | Python | 3.8+ | Core programming language |
| **Object Detection** | YOLOv8 | Latest | Ball, rim, player detection |
| **Pose Estimation** | YOLOv8-Pose | Latest | Human body joint tracking |
| **Video Processing** | OpenCV | 4.x+ | Video I/O and image processing |
| **Deep Learning** | PyTorch/GPU | CUDA compatible | Model inference and acceleration |
| **Scientific Computing** | NumPy | 1.x+ | Array operations and calculations |
| **Visualization** | Matplotlib | 3.x+ | Plotting and visualization |

---

## Project Structure

```
Swish-Vision/
├── main.py                          # Main execution pipeline
├── README.md                        # This file
├── train.ipynb                      # Jupyter notebook for model training
├── 
├── Basketball-1/                    # YOLO dataset (Roboflow format)
│   ├── data.yaml                   # Dataset configuration
│   ├── train/                      # Training images and labels
│   ├── valid/                      # Validation images and labels
│   └── test/                       # Test images and labels
│
├── models/                          # Pre-trained model weights
│   ├── best.pt                     # Fine-tuned YOLOv8 (best model)
│   ├── bestOld.pt                  # Previous iteration
│   ├── bestYT.pt                   # YouTube-trained variant
│   ├── ballRim.pt                  # Ball-Rim specific model
│   ├── yolov8m-pose.pt            # Pose estimation model (medium)
│   └── [Other model variants]
│
├── input_videos/                    # Input video files
│   └── [Video files for processing]
│
├── output_videos/                   # Processed output videos
│   └── [Annotated output videos]
│
├── runs/                            # YOLO training runs and results
│   └── detect/
│       ├── train/
│       ├── train2-train13/
│       └── [Training iterations]
│
├── reports/                         # Analysis output reports
│   ├── vid13_1_report.txt
│   ├── vid14_1_report.txt
│   └── [Other analysis reports]
│
├── trackers/                        # Object tracking modules
│   ├── __init__.py
│   ├── ball_tracker.py             # Ball detection and tracking
│   ├── rim_tracker.py              # Rim/hoop detection and tracking
│   ├── human_tracker.py            # Human pose detection and tracking
│   └── __pycache__/
│
├── drawers/                         # Visualization and annotation modules
│   ├── __init__.py
│   ├── ball_tracks_drawer.py       # Ball trajectory visualization
│   ├── rim_tracks_drawer.py        # Rim detection visualization
│   ├── human_tracks_drawer.py      # Pose visualization
│   ├── shot_tracker.py             # Shot analysis visualization
│   ├── utils.py                    # Drawing utilities
│   └── __pycache__/
│
├── utils/                           # Utility functions
│   ├── __init__.py
│   ├── vid_utils.py                # Video I/O functions
│   ├── ball_hand.py                # Ball-hand interaction detection
│   ├── stubs_utils.py              # Stub utilities
│   └── __pycache__/
│
├── Basketball-1/                    # Dataset directory
│   ├── data.yaml                   # YAML configuration for dataset
│   ├── train/, valid/, test/       # Roboflow dataset splits
│   └── README.roboflow.txt
│
└── [Data files]
    ├── angs.txt                    # Angle calculations
    ├── xy_coords.txt               # Coordinate data
    ├── detections.txt              # Raw detections
    ├── ball_locl.txt               # Ball location data
    └── [Other intermediate data files]

```

---

## Installation and Setup

### Prerequisites
- **Python 3.8+** installed
- **GPU support recommended** (CUDA-compatible NVIDIA GPU)
- **4GB+ RAM** minimum
- **10GB+ disk space** for models and datasets

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd Swish-Vision
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Enable Pre-Commit Secret Scanning (Recommended)
Use pre-commit with detect-secrets to block accidental commits of credentials and tokens.

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

The repository includes [.pre-commit-config.yaml](.pre-commit-config.yaml), which runs detect-secrets on staged code files before each commit.

**Key Dependencies:**
```
ultralytics>=8.0.0     # YOLOv8
opencv-python>=4.8.0   # Video processing
torch>=2.0.0          # Deep learning framework
torchvision>=0.15.0   # Vision models
numpy>=1.24.0         # Numerical computing
matplotlib>=3.7.0     # Visualization
```

### Step 4: Download Pre-trained Models
Place the following model files in the `models/` directory:
- `best.pt` - Fine-tuned YOLOv8 object detection
- `yolov8m-pose.pt` - YOLOv8 medium pose estimation
- `ballRim.pt` - Optional specialized model

Models can be downloaded from:
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow Universe](https://universe.roboflow.com/)

### Step 5: Prepare Input Videos
Place video files in the `input_videos/` directory:
```
input_videos/
├── video1.mp4
├── video2.mp4
└── ...
```

---

## Usage

### Basic Pipeline Execution

```python
from main import main_pipeline

# Run analysis on a video
# Video should be named "video_name.mp4" in input_videos/ directory
main_pipeline("video_name")
```

### Processing Steps

The main pipeline performs the following steps:

1. **Frame Extraction**: Reads video and extracts all frames
2. **Ball Detection**: Detects basketball in each frame
3. **Rim Detection**: Detects hoop/rim position
4. **Pose Detection**: Detects human body joints and keypoints
5. **Track Validation**: Filters and validates detection tracks
6. **Track Interpolation**: Fills gaps in tracking data
7. **Angle Calculation**: Computes body angles and joint positions
8. **Shot Detection**: Identifies shot initiation and completion
9. **Ball-Hand Detection**: Detects hand-ball interaction
10. **Visualization**: Annotates frames with detection data
11. **Video Output**: Writes annotated video to output_videos/
12. **Report Generation**: Creates analysis reports

### Command Line Usage
```bash
python main.py [video_name]
```

---

## Modules and Components

### Trackers Module (`trackers/`)

#### **BallTracker** (`ball_tracker.py`)
Detects and tracks the basketball throughout video frames.

**Key Methods:**
- `get_object_tracks(frames)` - Detects ball in all frames
- `remove_wrong_tracks(tracks)` - Filters invalid detections
- `interpolate_missing_tracks(tracks)` - Fills tracking gaps
- `get_ball_loco(frames, tracks)` - Extracts ball location data

**Config:**
- Model: `models/best.pt` (YOLOv8)
- Confidence threshold: Configurable

#### **RimTracker** (`rim_tracker.py`)
Detects and tracks the basketball rim/hoop.

**Key Methods:**
- Same interface as BallTracker
- Specialized for rim detection

**Config:**
- Model: `models/best.pt` (same backbone)

#### **HumanTracker** (`human_tracker.py`)
Detects human players and estimates their pose.

**Key Methods:**
- `detect_frame(frames)` - Returns human detections
- `calc_angles(frames, detections)` - Computes joint angles
- `get_points(frames, detections)` - Extracts keypoint coordinates

**Key Points (17 joints):**
- Head, shoulders, elbows, wrists
- Hips, knees, ankles
- Plus neck, chest

**Config:**
- Model: `models/yolov8m-pose.pt` (YOLOv8 Medium Pose)
- 17 keypoint output

### Drawers Module (`drawers/`)

#### **BallTracksDrawer** (`ball_tracks_drawer.py`)
Visualizes basketball trajectory and detection.

#### **RimTracksDrawer** (`rim_tracks_drawer.py`)
Visualizes rim/hoop detection and position.

#### **HumanTracksDrawer** (`human_tracks_drawer.py`)
Visualizes human pose with skeletal connections.

#### **ShotTracker** (`shot_tracker.py`)
Annotates and visualizes shot-specific information.

**Key Utilities:** (`utils.py`)
- `get_center()` - Computes bounding box center
- Frame annotation helpers
- Color management for visualization

### Utils Module (`utils/`)

#### **Video Utilities** (`vid_utils.py`)
- `read_video(path)` - Reads MP4/video file, returns frames and FPS
- `write_video(frames, path, fps)` - Writes frames to video file

**Features:**
- Progress bar for long operations
- Automatic directory creation

#### **Ball-Hand Detection** (`ball_hand.py`)
- `ball_hand(location_data, points, frames)` - Detects hand-ball proximity
- `shot_started(frame_data)` - Detects shot initiation

#### **Stub Utilities** (`stubs_utils.py`)
Helper utilities for stubbing and testing.

---

## Models and Datasets

### Pre-trained Models

| Model | Purpose | Source | Size |
|-------|---------|--------|------|
| **YOLOv8 (best.pt)** | Ball/Rim detection | Fine-tuned on Basketball-1 | ~43MB |
| **YOLOv8m-Pose** | Human pose estimation | Ultralytics pretrained | ~78MB |
| **YOLOv11n.pt** | Nano variant | Ultralytics | ~12MB |
| **YOLOv8m.pt** | Medium variant | Ultralytics | ~47MB |
| **YOLOv8n.pt** | Nano variant | Ultralytics | ~13MB |

### Dataset Information

**Basketball-1 Dataset** (Roboflow)
- **Source**: [Roboflow Universe](https://universe.roboflow.com/eagle-eye/basketball-1zhpe)
- **Classes**: 
  - Basketball (ball)
  - Rim
  - Sports ball
- **Total**: 3 detection classes
- **Splits**:
  - **Train**: ~70% of images
  - **Valid**: ~15% of images
  - **Test**: ~15% of images
- **Format**: YOLO format (txt labeled bounding boxes)
- **License**: CC BY 4.0

### Training Information

Model training/fine-tuning results are stored in `runs/detect/` with multiple training iterations:
- `train/` through `train13/` - Different training epochs and configurations
- Each contains:
  - `weights/best.pt` - Best model weights
  - `results.csv` - Training metrics
  - `confusion_matrix.png` - Confusion matrix visualization

---

## Configuration

### YAML Configuration (`Basketball-1/data.yaml`)

```yaml
names:
  - basketball
  - rim
  - sports ball
nc: 3  # Number of classes
train: ../train/images
val: ../valid/images
test: ../test/images
roboflow:
  workspace: eagle-eye
  project: basketball-1zhpe
  version: 1
  license: CC BY 4.0
```

### Model Configuration Parameters

**Ball/Rim Tracker:**
- Model path: `models/best.pt`
- Confidence threshold: [Adjustable in code]
- Input size: 640x640 (YOLOv8 default)

**Human Tracker:**
- Model path: `models/yolov8m-pose.pt`
- Keypoints: 17 joints
- Confidence threshold: [Adjustable in code]

**Video Processing:**
- Supported formats: MP4, AVI, MOV, etc.
- Output codec: XVID
- Frame rate: Original FPS preserved

---

## Future Enhancements

### Short-term Improvements
- [ ] Implement real-time processing for live camera feeds
- [ ] Add shot success prediction model
- [ ] Create interactive web dashboard for analysis
- [ ] Implement multi-player tracking
- [ ] Add performance metrics calculation (accuracy, arc, release point)

### Medium-term Enhancements
- [ ] Develop mobile app for on-court feedback
- [ ] Create database for historical tracking and progression
- [ ] Implement advanced 3D reconstruction from video
- [ ] Add AI-powered coaching recommendations
- [ ] Support for multiple camera angles

### Long-term Goals
- [ ] Real-time AR overlay coaching
- [ ] Integration with wearable sensors
- [ ] Automated training program generation
- [ ] Professional analytics platform
- [ ] AI comparison with professional athletes

### Technical Debt & Optimization
- [ ] Code refactoring and modularization
- [ ] Unit test coverage
- [ ] Documentation generation (API docs)
- [ ] Performance optimization for batch processing
- [ ] Configuration file structure (YAML/JSON)
- [ ] Logging and error handling improvements

---

## Contributing

Guidelines for future contributors:

1. Create a feature branch from `main`
2. Follow PEP 8 style guidelines
3. Add docstrings to functions and classes
4. Test changes with sample videos
5. Update documentation for new features
6. Submit pull request with detailed description

---

## License

This project is built on top of Roboflow's Basketball-1 dataset (CC BY 4.0 License).

---

## Support and Documentation

For detailed technical information about specific modules, see:
- **Software Design Specification** (TBD)
- **API Documentation** (TBD)
- **System Requirements** (TBD)
- **Deployment Guide** (TBD)

---

**Last Updated**: March 2026  
**Version**: 1.0.0
