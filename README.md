# Swish-Vision
AI Basketball Shooting Practice Analysis

## Overview
Swish-Vision is an AI-powered system for analyzing basketball shooting practice. It uses YOLOv8 models to detect and track:
- Basketball
- Basket rim
- Player poses and movements

## Features
- Real-time basketball and rim detection
- Shot tracking and analysis
- Player pose estimation
- Shooting angle calculation
- Video output with annotations

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training and inference)

### Install Dependencies
```bash
pip install ultralytics supervision pandas numpy opencv-python torch torchvision
```

## Usage

### Running Analysis
```bash
python main.py
```

By default, the script processes videos from the `input_videos/` directory and saves results to `output_videos/`.

### Model Training
To retrain the model with a larger dataset, see [TRAINING.md](TRAINING.md) for detailed instructions.

Quick start:
```bash
# Download and train automatically
python train_model.py --download --epochs 100 --batch 16

# Or use manually downloaded dataset
python train_model.py --data ./dataset/data.yaml --epochs 100
```

## Project Structure
```
Swish-Vision/
├── main.py                 # Main application entry point
├── train_model.py          # Model training script
├── TRAINING.md             # Detailed training guide
├── models/                 # Pre-trained model weights
│   ├── best.pt            # Ball detection model
│   ├── bestYT.pt          # Rim detection model
│   └── yolov8m-pose.pt    # Pose estimation model
├── trackers/               # Object tracking modules
├── drawers/                # Visualization modules
├── utils/                  # Utility functions
├── input_videos/           # Input video directory
└── output_videos/          # Output video directory
```

## Models

The system uses three YOLOv8 models:
1. **Ball Detector** (`best.pt`) - Detects basketball in frames
2. **Rim Detector** (`bestYT.pt`) - Detects basket rim
3. **Pose Estimator** (`yolov8m-pose.pt`) - Estimates player pose

## Training Your Own Model

See [TRAINING.md](TRAINING.md) for complete training instructions using the Roboflow dataset:
- Dataset: https://universe.roboflow.com/ds/Dln6t1SSH6?key=z3fhm33DMn

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is open source and available under the MIT License.
