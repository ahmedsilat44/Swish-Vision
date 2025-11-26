# Model Training Guide

This guide explains how to retrain the YOLOv8 model with a larger dataset from Roboflow.

## Dataset Information

**Dataset URL**: [https://universe.roboflow.com/ds/Dln6t1SSH6?key=z3fhm33DMn](https://universe.roboflow.com/ds/Dln6t1SSH6?key=z3fhm33DMn)

The dataset contains annotated basketball images for detecting:
- Basketball
- Rim

## Prerequisites

Make sure you have the required packages installed:

```bash
pip install ultralytics roboflow torch torchvision supervision pandas numpy opencv-python
```

## Training Methods

### Method 1: Automatic Download (Recommended)

If you have internet access and can reach Roboflow servers:

```bash
# Using default API key from dataset URL
python train_model.py --download --epochs 100 --batch 16 --model-size n

# Or with custom API key
python train_model.py --download --api-key YOUR_API_KEY --epochs 100 --batch 16

# Or using environment variable (more secure)
export ROBOFLOW_API_KEY=your_api_key_here
python train_model.py --download --epochs 100 --batch 16
```

### Method 2: Manual Download

1. **Download the Dataset**:
   - Visit: [https://universe.roboflow.com/ds/Dln6t1SSH6?key=z3fhm33DMn](https://universe.roboflow.com/ds/Dln6t1SSH6?key=z3fhm33DMn)
   - Click "Download" and select "YOLOv8" format
   - Extract the downloaded ZIP file to a directory (e.g., `./dataset`)

2. **Train the Model**:
   ```bash
   python train_model.py --data ./dataset/data.yaml --epochs 100 --batch 16 --model-size n
   ```

## Training Parameters

- `--download`: Download dataset from Roboflow automatically
- `--data PATH`: Path to data.yaml file (for manual dataset)
- `--epochs N`: Number of training epochs (default: 100)
- `--batch N`: Batch size (default: 16, adjust based on GPU memory)
- `--img-size N`: Input image size (default: 640)
- `--model-size {n,s,m,l,x}`: YOLOv8 model size
  - `n` (nano): Fastest, least accurate
  - `s` (small): Good balance
  - `m` (medium): Better accuracy
  - `l` (large): High accuracy
  - `x` (extra large): Best accuracy, slowest

## Model Size Recommendations

- **For CPU or limited GPU**: Use `n` or `s`
- **For moderate GPU (4-8GB)**: Use `s` or `m`
- **For powerful GPU (>8GB)**: Use `m`, `l`, or `x`

## Training Examples

### Quick Training (Fast, Less Accurate)
```bash
python train_model.py --data ./dataset/data.yaml --epochs 50 --batch 16 --model-size n
```

### Balanced Training (Recommended)
```bash
python train_model.py --data ./dataset/data.yaml --epochs 100 --batch 16 --model-size m
```

### High-Quality Training (Slow, Most Accurate)
```bash
python train_model.py --data ./dataset/data.yaml --epochs 150 --batch 8 --model-size l
```

## After Training

Once training is complete, the model weights will be saved to:
- `runs/detect/basketball_rim_detection/weights/best.pt` (best performing model)
- `runs/detect/basketball_rim_detection/weights/last.pt` (last epoch)

### Using the New Model

Replace the existing model files with your newly trained model:

```bash
# Backup existing models
cp models/best.pt models/best.pt.backup
cp models/bestYT.pt models/bestYT.pt.backup

# Copy new model
cp runs/detect/basketball_rim_detection/weights/best.pt models/best.pt
# OR
cp runs/detect/basketball_rim_detection/weights/best.pt models/bestYT.pt
```

The main application will now use your retrained model when you run:

```bash
python main.py
```

## Training on GPU

To utilize GPU acceleration (highly recommended):

1. **Check CUDA availability**:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

2. **Install CUDA-enabled PyTorch** (if not already installed):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Train with GPU**: The script automatically detects and uses GPU if available

## Monitoring Training

During training, you can monitor:
- Training loss and metrics in the console output
- TensorBoard logs (if enabled): `tensorboard --logdir runs/detect`
- Validation results saved in the run directory

## Troubleshooting

### Out of Memory Error
- Reduce `--batch` size (try 8 or 4)
- Use smaller `--model-size` (n or s)
- Reduce `--img-size` to 416 or 512

### Slow Training
- Ensure GPU is being used (check console output)
- Increase `--batch` size if GPU has memory available
- Use a smaller model size for faster iterations

### Poor Performance
- Increase `--epochs` (try 150-200)
- Use larger `--model-size` (m or l)
- Ensure dataset quality and annotations are correct

## Advanced Configuration

For advanced users, you can modify training hyperparameters by editing the `train_model.py` script or by creating a custom hyperparameters YAML file.

Example hyperparameters to tune:
- Learning rate
- Augmentation settings
- Optimizer settings
- Loss function weights

Refer to [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/modes/train/) for more details.
