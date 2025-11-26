# Quick Start Guide for Model Training

## TL;DR

Train a new basketball detection model with the Roboflow dataset in 3 steps:

### Step 1: Install Dependencies
```bash
pip install ultralytics roboflow torch torchvision supervision pandas numpy opencv-python
```

### Step 2: Train the Model
```bash
# Option A: Let the script download the dataset
python train_model.py --download --epochs 100 --batch 16 --model-size n

# Option B: Use a manually downloaded dataset
python train_model.py --data ./dataset/data.yaml --epochs 100 --batch 16 --model-size n
```

### Step 3: Update the Model
```bash
# Backup current model
cp models/best.pt models/best.pt.backup

# Copy new model
cp runs/detect/basketball_rim_detection/weights/best.pt models/best.pt
```

## Testing the New Model
```bash
python main.py
```

Check `output_videos/` for results!

---

## Common Issues

### "Connection Error" when downloading dataset
**Solution**: Download dataset manually from [Roboflow Universe](https://universe.roboflow.com/ds/Dln6t1SSH6?key=z3fhm33DMn), export as YOLOv8, then use `--data` option.

### "Out of Memory" during training
**Solution**: Reduce batch size: `--batch 8` or `--batch 4`

### Training is too slow
**Solution**: Use smaller model: `--model-size n` or `--model-size s`

### Model not detecting objects well
**Solution**: Train longer with larger model: `--epochs 150 --model-size m`

---

## Full Documentation

- **Training Guide**: [TRAINING.md](TRAINING.md) - Comprehensive training documentation
- **Model Update Guide**: [UPDATE_MODELS.md](UPDATE_MODELS.md) - How to integrate trained models
- **Main README**: [README.md](README.md) - Project overview

## Dataset Information

**Dataset URL**: https://universe.roboflow.com/ds/Dln6t1SSH6?key=z3fhm33DMn

This dataset contains annotated images for:
- Basketball detection
- Rim detection

## Training Parameters Quick Reference

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--model-size` | n, s, m, l, x | Model complexity (n=fastest, x=most accurate) |
| `--epochs` | Number | Training iterations (default: 100) |
| `--batch` | Number | Batch size (default: 16, reduce if OOM) |
| `--img-size` | Number | Input image size (default: 640) |
| `--download` | Flag | Auto-download from Roboflow |
| `--data` | Path | Path to local data.yaml |
| `--api-key` | String | Roboflow API key (or use ROBOFLOW_API_KEY env var) |

## Example Commands

### Fast Training (Testing)
```bash
python train_model.py --data ./dataset/data.yaml --epochs 30 --batch 16 --model-size n
```

### Balanced Training (Recommended)
```bash
python train_model.py --data ./dataset/data.yaml --epochs 100 --batch 16 --model-size m
```

### High-Quality Training
```bash
python train_model.py --data ./dataset/data.yaml --epochs 150 --batch 8 --model-size l
```

## Hardware Recommendations

- **CPU Only**: Use `--model-size n --batch 4 --epochs 50`
- **GPU 4GB**: Use `--model-size s --batch 8 --epochs 100`
- **GPU 8GB+**: Use `--model-size m --batch 16 --epochs 100`
- **GPU 16GB+**: Use `--model-size l --batch 32 --epochs 150`

## Need Help?

1. Check [TRAINING.md](TRAINING.md) for detailed explanations
2. Check [UPDATE_MODELS.md](UPDATE_MODELS.md) for model integration
3. Open an issue on GitHub if problems persist
