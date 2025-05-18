from ultralytics import YOLO
import supervision as sv

class RimTracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
    def detect_frame(self, frame):
        batch_size = 50
        detections = []
        for i in range(0, len(frame), batch_size):
            batch = frame[i:i + batch_size]
            results = self.model.predict(batch, conf=0.05)
            detections += results
        return detections
    
    def get_object_tracks(self, frames):
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

                track_id = frame_detection[4]
                
               
                if cls_id == cls_names_inv['Rim']:
                    tracks[frame_num][track_id] = {
                        "box": bbox,
                        "class": "Rim",
                        "track_id": track_id
                    }



        return tracks
