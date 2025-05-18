from ultralytics import YOLO
import supervision as sv
import torch
import pandas as pd
import numpy as np

class BallTracker:
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
            
            results = self.model.predict(batch, conf=0.25, device=device) 
            detections += results
        return detections
    
    def get_object_tracks(self, frames):
        detections = self.detect_frame(frames)
        tracks = []
        for frame_num,detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            


            detection_supervision = sv.Detections.from_ultralytics(detection)


            tracks.append({})
            chosen_bbox =None
            max_confidence = 0

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                conf = frame_detection[2]
                

                
                
                if cls_id == cls_names_inv['ball']:
                    tracks[frame_num][1] = {
                        "bbox": bbox,
                        "class": "Basketball",
                        
                    }
                elif cls_id == cls_names_inv['rim']:
                    tracks[frame_num][2] = {
                        "bbox": bbox,
                        "class": "Rim",
                        
                    }
            # if chosen_bbox is not None:
            #     tracks[frame_num][1] = {"bbox":chosen_bbox}
               



        return tracks
    
    def remove_wrong_tracks(self, tracks):
        max_distance = 25  # Maximum distance to consider a track valid
        last_good_track = -1

        for i in range(len(tracks)):
            if len(tracks[i]) == 0:
                continue
            current_box = tracks[i].get( 1, {}).get("bbox", [])

            if len(current_box) == 0:
                continue

            if last_good_track == -1:
                last_good_track = i
                continue

            last_good_box = tracks[last_good_track].get(1, {}).get("bbox", [])
            gap = i - last_good_track
            adjusted_distance = max_distance - gap

            if np.linalg.norm(np.array(last_good_box[:2]) - np.array(current_box[:2])) > adjusted_distance:
                # Remove the track 
                tracks[i] = {}
            else:
                last_good_track = i

        return tracks


    def interpolate_missing_tracks(self,ball_positions):
        print("ball_positions",ball_positions)
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        print("ball_positions",ball_positions)
        df_ball_positions = pd.DataFrame(ball_positions,columns=["x1", "y1", "x2", "y2"])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:{"bbox":x , "class": "Basketball"}} for x in df_ball_positions.to_numpy().tolist()]
        print("ball_positions",ball_positions)

        return ball_positions
