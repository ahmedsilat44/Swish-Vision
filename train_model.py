"""
Script to train YOLOv8 model with Roboflow dataset
Dataset: https://universe.roboflow.com/ds/Dln6t1SSH6?key=z3fhm33DMn
"""

from roboflow import Roboflow
from ultralytics import YOLO
import os

def download_dataset():
    """Download the dataset from Roboflow"""
    print("Downloading dataset from Roboflow...")
    
    # Initialize Roboflow with the API key from the URL
    rf = Roboflow(api_key="z3fhm33DMn")
    
    # Get the project and version from the Roboflow link
    # The dataset ID format is: workspace/project/version
    project = rf.workspace().project("Dln6t1SSH6")
    dataset = project.version(1).download("yolov8")
    
    return dataset

def train_model(data_path):
    """Train YOLOv8 model on the dataset"""
    print("Starting model training...")
    
    # Load YOLOv8 model (using yolov8n for faster training, can change to yolov8m/yolov8l for better accuracy)
    model = YOLO('yolov8n.pt')  # Start with pretrained model
    
    # Train the model
    results = model.train(
        data=f'{data_path}/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='basketball_rim_detection',
        patience=20,
        save=True,
        device='cuda' if os.system('command -v nvidia-smi > /dev/null 2>&1') == 0 else 'cpu'
    )
    
    print("Training completed!")
    return results

def main():
    """Main function to orchestrate dataset download and model training"""
    # Download dataset
    dataset = download_dataset()
    
    # Train model
    train_model(dataset.location)
    
    print("\nTraining complete! The trained model is saved in the 'runs/detect/basketball_rim_detection' directory")
    print("The best model weights are saved as 'runs/detect/basketball_rim_detection/weights/best.pt'")

if __name__ == "__main__":
    main()
