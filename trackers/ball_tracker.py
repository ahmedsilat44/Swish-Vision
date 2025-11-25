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
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
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
                

                
                
                # if cls_id == cls_names_inv['ball']:
                if cls_id == cls_names_inv['Basketball']:
                    tracks[frame_num][1] = {
                        "bbox": bbox,
                        "class": "Basketball",
                        
                    }
                elif cls_id == cls_names_inv['Rim']:
                    height = bbox[3] - bbox[1]
                    # add a margin to the height
                    margin = 0.5 * height
                    # increase top of the bbox
                    bbox[1] = bbox[1] - margin
                    
                    
                    tracks[frame_num][2] = {
                        "bbox": bbox,
                        "class": "Rim",
                    }
               


        # print(f"ball_track: {tracks}")
        return tracks
    
    def remove_wrong_tracks(self, tracks):
        max_distance = 500  # Maximum distance to consider a track valid
        last_good_track = -1

        for i in range(len(tracks)):
            # skip empty track entries
            if not tracks[i]:
                continue

            current_box = tracks[i].get(1, {}).get("bbox", [])

            # ensure current_box has at least x1,y1 and x2,y2
            if len(current_box) < 4:
                continue

            if last_good_track == -1:
                last_good_track = i
                continue

            last_good_box = tracks[last_good_track].get(1, {}).get("bbox", [])
            # if last good box is malformed, treat current as new good
            if len(last_good_box) < 4:
                last_good_track = i
                continue

            gap = i - last_good_track
            # avoid negative thresholds when gap grows
            adjusted_distance = max(max_distance - gap, 0)

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


    def interpolate_missing_tracks(self,ball_positions):

        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=["x1", "y1", "x2", "y2"])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:{"bbox":x , "class": "Basketball"}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
