# Updating Models After Training

This guide explains how to integrate newly trained models into the Swish-Vision application.

## After Training Completion

Once you've completed training using `train_model.py`, you'll have new model weights saved in:
```
runs/detect/basketball_rim_detection/weights/
├── best.pt    # Best performing model during training
└── last.pt    # Model from the last epoch
```

## Model Selection

The `best.pt` file is typically the one you want to use, as it represents the model with the best validation performance during training.

## Integration Steps

### Step 1: Backup Current Models

Before replacing the existing models, create backups:

```bash
# Backup current models
cp models/best.pt models/best.pt.backup
cp models/bestYT.pt models/bestYT.pt.backup
```

### Step 2: Identify Model Purpose

The Swish-Vision system uses different models for different purposes:

1. **`models/best.pt`** - Used for ball detection in `trackers/ball_tracker.py`
2. **`models/bestYT.pt`** - Used for rim detection in `trackers/rim_tracker.py`
3. **`models/yolov8m-pose.pt`** - Used for pose estimation (not affected by this training)

### Step 3: Replace the Appropriate Model

Depending on what your training focused on:

#### If you trained a model for BOTH basketball and rim detection:
```bash
# Copy the new model to both locations
cp runs/detect/basketball_rim_detection/weights/best.pt models/best.pt
cp runs/detect/basketball_rim_detection/weights/best.pt models/bestYT.pt
```

#### If you trained specifically for basketball detection:
```bash
cp runs/detect/basketball_rim_detection/weights/best.pt models/best.pt
```

#### If you trained specifically for rim detection:
```bash
cp runs/detect/basketball_rim_detection/weights/best.pt models/bestYT.pt
```

### Step 4: Verify Class Names

Check that your trained model uses the correct class names. Open a Python shell:

```python
from ultralytics import YOLO

# Load your new model
model = YOLO('runs/detect/basketball_rim_detection/weights/best.pt')

# Check class names
print("Model classes:", model.names)
# Should output: {0: 'Basketball', 1: 'Rim'}
```

### Step 5: Update Code if Necessary

If your model uses different class names, update the tracker code:

#### In `trackers/ball_tracker.py`:
```python
# Line 50-51 (approximately)
if cls_id == cls_names_inv['Basketball']:  # Update class name if different
    tracks[frame_num][1] = {
        "bbox": bbox,
        "class": "Basketball",
    }
```

#### In `trackers/rim_tracker.py`:
Similar updates for rim detection if class names differ.

### Step 6: Test the New Model

Run the application on a test video:

```bash
python main.py
```

Check the output in `output_videos/` to verify the new model is working correctly.

## Validation Checklist

- [ ] Backup original models created
- [ ] New model copied to correct location(s)
- [ ] Class names verified
- [ ] Test run completed successfully
- [ ] Output video shows correct detections
- [ ] Performance is acceptable (detection accuracy and speed)

## Troubleshooting

### Model Not Detecting Objects

1. **Check confidence threshold**: The trackers use `conf=0.25`. If your model needs a different threshold, update it in the tracker files:
   ```python
   results = self.model.predict(batch, conf=0.25, device=device)  # Adjust conf value
   ```

2. **Verify training data**: Ensure the training dataset had good quality annotations

3. **Check model size**: Larger models (m, l, x) generally perform better but are slower

### Performance Issues

1. **Model too slow**: Consider training a smaller model (n or s)
2. **Using CPU instead of GPU**: Ensure CUDA is properly installed
3. **Batch size too large**: Reduce batch size in the tracker code

### Incorrect Detections

1. **Need more training**: Increase epochs (try 150-200)
2. **Dataset issues**: Review and improve training dataset
3. **Hyperparameter tuning**: Adjust learning rate, augmentation settings

## Rolling Back

If the new model doesn't work well, restore the backup:

```bash
# Restore original models
cp models/best.pt.backup models/best.pt
cp models/bestYT.pt.backup models/bestYT.pt
```

## Comparing Models

To compare old vs new model performance:

1. **Keep both models**:
   ```bash
   # Keep new model with a different name
   cp runs/detect/basketball_rim_detection/weights/best.pt models/best_new.pt
   ```

2. **Test each model separately** by temporarily updating the model path in the code

3. **Compare metrics**:
   - Detection accuracy
   - Processing speed
   - False positive/negative rates

## Best Practices

1. Always backup models before replacing
2. Test on multiple videos before deploying
3. Document model version and training parameters
4. Keep training logs for reference
5. Consider maintaining a model repository for version control

## Model Versioning

Consider creating a versioning scheme:

```
models/
├── best.pt              # Current production model
├── best.pt.backup       # Previous version backup
├── best_v1.pt          # Version 1
├── best_v2_epoch100.pt # Version 2, 100 epochs
└── README_models.txt    # Model version log
```

Example `README_models.txt`:
```
Model Version History
=====================

best_v2_epoch100.pt (2024-01-15)
- Dataset: Roboflow Basketball Detection v1
- Epochs: 100
- Model size: YOLOv8n
- mAP50: 0.89
- Notes: Improved rim detection

best_v1.pt (2024-01-01)
- Original model
- Dataset: Unknown
- Notes: Baseline model
```

## Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Model Evaluation Guide](https://docs.ultralytics.com/modes/val/)
- [Hyperparameter Tuning](https://docs.ultralytics.com/usage/cfg/)
