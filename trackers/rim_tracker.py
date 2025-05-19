from ultralytics import YOLO
import supervision as sv
import torch
import pandas as pd
import numpy as np

class RimTracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
    def detect_frame(self, frame):
        batch_size = 20
        detections = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for i in range(0, len(frame), batch_size):
            batch = frame[i:i + batch_size]
            results = self.model.predict(batch, conf=0.25, device=device)
            detections += results
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        detections = self.detect_frame(frames)
        tracks = []
        for frame_num,detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            


            detection_supervision = sv.Detections.from_ultralytics(detection)

            # detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks.append({})

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                
               
                if cls_id == cls_names_inv['rim']:
                    tracks[frame_num][2] = {
                        "bbox": bbox,
                        "class": "Rim"
                    }



        return tracks

    def interpolate_missing_tracks(self,rim_positions):
            print("rim_positions",rim_positions)
            rim_positions = [x.get(2, {}).get("bbox", []) for x in rim_positions]
            print("rim_positions",rim_positions)
            df_rim_positions = pd.DataFrame(rim_positions,columns=["x1", "y1", "x2", "y2"])

            df_rim_positions = df_rim_positions.interpolate()
            df_rim_positions = df_rim_positions.bfill()

            rim_positions = [{2:{"bbox":x , "class": "Rim"}} for x in df_rim_positions.to_numpy().tolist()]
            print("rim_positions",rim_positions)

            return rim_positions