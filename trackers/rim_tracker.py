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
        print("is cuda available?", torch.cuda.is_available())
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for i in range(0, len(frame), batch_size):
            batch = frame[i:i + batch_size]
            results = self.model.predict(batch, conf=0.25, device=device, verbose=False)
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

        if len(rim_positions) == 0:
            return rim_positions
            
        rim_positions = [x.get(2, {}).get("bbox", []) for x in rim_positions]
        df_rim_positions = pd.DataFrame(rim_positions,columns=["x1", "y1", "x2", "y2"])

        # df_rim_positions = df_rim_positions.interpolate()
        df_rim_positions = df_rim_positions.bfill()
        df_rim_positions = df_rim_positions.ffill()

        rim_positions = [{2:{"bbox":x , "class": "Rim"}} for x in df_rim_positions.to_numpy().tolist()]

        return rim_positions
    

    def remove_wrong_tracks(self, tracks):
        max_distance = 150  # Maximum distance to consider a track valid (increased for rim)
        last_good_track = -1

        for i in range(len(tracks)):
            # skip empty track entries
            if not tracks[i]:
                continue

            current_box = tracks[i].get(2, {}).get("bbox", [])  # Use track ID 2 for rim

            # ensure current_box has at least x1,y1 and x2,y2
            if len(current_box) < 4:
                continue

            if last_good_track == -1:
                last_good_track = i
                continue

            last_good_box = tracks[last_good_track].get(2, {}).get("bbox", [])  # Use track ID 2 for rim
            # if last good box is malformed, treat current as new good
            if len(last_good_box) < 4:
                last_good_track = i
                continue

            gap = i - last_good_track
            # Reduce threshold as gap increases (rim should be stationary)
            adjusted_distance = max(max_distance - (gap * 2), 10)

            # compute centers for a more robust distance check
            current_center = np.array([(current_box[0] + current_box[2]) / 2.0,
                                       (current_box[1] + current_box[3]) / 2.0])
            last_center = np.array([(last_good_box[0] + last_good_box[2]) / 2.0,
                                    (last_good_box[1] + last_good_box[3]) / 2.0])

            distance = np.linalg.norm(last_center - current_center)

            if distance > adjusted_distance:
                # Remove the track 
                tracks[i] = {}
            else:
                last_good_track = i

        return tracks