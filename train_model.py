"""
Script to train YOLOv8 model with Roboflow dataset
Dataset: https://universe.roboflow.com/ds/Dln6t1SSH6?key=z3fhm33DMn

Usage:
    1. Download dataset manually from Roboflow or use the script with internet access
    2. Run: python train_model.py --data <path_to_data.yaml>
    
    Or let the script download it automatically (requires internet access):
    3. Run: python train_model.py --download
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from roboflow import Roboflow
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False

from ultralytics import YOLO
import torch


def download_dataset_roboflow(api_key=None, project_id=None):
    """Download the dataset from Roboflow using the API
    
    Args:
        api_key: Roboflow API key (if None, tries to load from env var ROBOFLOW_API_KEY)
        project_id: Roboflow project ID (if None, uses default from dataset URL)
    """
    if not ROBOFLOW_AVAILABLE:
        print("ERROR: roboflow package not installed. Install with: pip install roboflow")
        return None
    
    # Get API key from environment variable if not provided
    if api_key is None:
        api_key = os.environ.get('ROBOFLOW_API_KEY', 'z3fhm33DMn')
    
    # Use project ID from the dataset URL if not provided
    # The dataset URL format suggests using the dataset ID directly
    if project_id is None:
        project_id = "Dln6t1SSH6"
    
    print("Downloading dataset from Roboflow...")
    print(f"Note: Using dataset ID: {project_id}")
    
    try:
        # Initialize Roboflow with the API key
        rf = Roboflow(api_key=api_key)
        
        # Get the workspace
        workspace = rf.workspace()
        
        # Access the dataset using the project ID
        project = workspace.project(project_id)
        
        # Download the dataset in YOLOv8 format
        dataset = project.version(1).download("yolov8")
        
        print(f"Dataset downloaded to: {dataset.location}")
        return dataset.location
        
    except Exception as e:
        print(f"Error downloading dataset from Roboflow: {e}")
        print("\nAlternative: Download the dataset manually:")
        print("1. Visit: https://universe.roboflow.com/ds/Dln6t1SSH6?key=z3fhm33DMn")
        print("2. Export in YOLOv8 format")
        print("3. Extract to a local directory")
        print("4. Run: python train_model.py --data <path_to_extracted_dataset>/data.yaml")
        return None


def train_model(data_yaml_path, epochs=100, batch_size=16, img_size=640, model_size='n'):
    """Train YOLOv8 model on the dataset
    
    Args:
        data_yaml_path: Path to the data.yaml file
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        model_size: YOLOv8 model size (n, s, m, l, x)
    """
    print("Starting model training...")
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Check if data.yaml exists
    if not os.path.exists(data_yaml_path):
        print(f"ERROR: data.yaml not found at {data_yaml_path}")
        return None
    
    # Load YOLOv8 model
    model_name = f'yolov8{model_size}.pt'
    print(f"Loading pretrained model: {model_name}")
    model = YOLO(model_name)
    
    # Train the model
    print(f"\nTraining configuration:")
    print(f"  Data: {data_yaml_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")
    print(f"  Device: {device}")
    
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name='basketball_rim_detection',
        patience=20,
        save=True,
        device=device,
        project='runs/detect'
    )
    
    print("\nTraining completed!")
    print(f"Best model saved to: runs/detect/basketball_rim_detection/weights/best.pt")
    
    return results


def main():
    """Main function to orchestrate dataset download and model training"""
    parser = argparse.ArgumentParser(description='Train YOLOv8 model on basketball dataset')
    parser.add_argument('--download', action='store_true', 
                        help='Download dataset from Roboflow')
    parser.add_argument('--data', type=str, 
                        help='Path to data.yaml file')
    parser.add_argument('--api-key', type=str,
                        help='Roboflow API key (can also use ROBOFLOW_API_KEY env var)')
    parser.add_argument('--project-id', type=str,
                        help='Roboflow project ID (default: Dln6t1SSH6)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size (default: 640)')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size: n(nano), s(small), m(medium), l(large), x(extra large)')
    
    args = parser.parse_args()
    
    data_path = None
    
    # Download dataset if requested
    if args.download:
        data_path = download_dataset_roboflow(
            api_key=args.api_key,
            project_id=args.project_id
        )
        if data_path:
            data_yaml = os.path.join(data_path, 'data.yaml')
        else:
            print("Dataset download failed. Please download manually and use --data option.")
            sys.exit(1)
    elif args.data:
        data_yaml = args.data
    else:
        print("ERROR: Please provide either --download or --data <path_to_data.yaml>")
        parser.print_help()
        sys.exit(1)
    
    # Train model
    results = train_model(
        data_yaml_path=data_yaml,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        model_size=args.model_size
    )
    
    if results:
        print("\n" + "="*50)
        print("Training Summary")
        print("="*50)
        print("The trained model is saved in the 'runs/detect/basketball_rim_detection' directory")
        print("Best model weights: runs/detect/basketball_rim_detection/weights/best.pt")
        print("Last model weights: runs/detect/basketball_rim_detection/weights/last.pt")
        print("\nTo use the new model, update the model paths in your code:")
        print("  - models/best.pt (for ball detection)")
        print("  - models/bestYT.pt (for rim detection)")


if __name__ == "__main__":
    main()
