from ultralytics import YOLO
import supervision as sv
import torch
import pandas as pd
import numpy as np

class HumanTracker:

    def __init__(self, model_path: str):
        self.model = YOLO(model_path, )
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
            
            # with open("detections.txt", "w") as f:
            #     f.write(str(results))
        return detections
    
    def angle_bw_points(self, a, b, c):
        """
        Computes the angle ABC (at point B)
        a, b, c are (x, y) tuples.
        """
        # Create vectors BA and BC
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])

        # Dot product and magnitudes
        dot = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = np.sqrt(ba[0]**2 + ba[1]**2)
        mag_bc = np.sqrt(bc[0]**2 + bc[1]**2)

        # Avoid division by zero
        if mag_ba == 0 or mag_bc == 0:
            return None

        # Compute angle in radians
        cos_angle = dot / (mag_ba * mag_bc)

        # Clamp to avoid numerical errors
        cos_angle = max(min(cos_angle, 1), -1)

        angle_rad = np.acos(cos_angle)

        # Convert to degrees
        return np.degrees(angle_rad)

    def calc_angles(
        self,
        video_frames,
        detections):
        
        anngles = [] #shoulder-elbow-wrist
        for i, frame in enumerate(video_frames):
            res = detections[i]

            # --- Keypoints ---
            kps = getattr(res, "keypoints", None)
            kps_xy = kps.xy.cpu().numpy()  # shape: [N, K, 2]
            kps_cf = kps.conf.cpu().numpy()  # [N, K]

            if kps_xy is not None:
                N = kps_xy.shape[0]
                for n in range(N): #num peoplpe
                    joints = [(float(x), float(y)) for x, y in kps_xy[n]]
                    if kps_cf is not None:
                        confs = [float(c) if c is not None else None for c in kps_cf[n]]
                    else:
                        confs = [1.0] * len(joints)

                    with open("xy_coords.txt", "a") as f:
                        f.write(str(joints))
                        f.write("\n")
                    
                    # parts_oi = [6,8,10] #right arm stuff
                    right_shoulder = joints[6]
                    right_elbow = joints[8]
                    right_wrist = joints[10]

                    # Skip angle calculation if any keypoint is invalid (None or at origin)
                    if (None in right_shoulder or None in right_elbow or None in right_wrist or
                        (abs(right_shoulder[0]) < 1 and abs(right_shoulder[1]) < 1) or
                        (abs(right_elbow[0]) < 1 and abs(right_elbow[1]) < 1) or
                        (abs(right_wrist[0]) < 1 and abs(right_wrist[1]) < 1)):
                        anngles.append(None)
                    else:
                        angle = self.angle_bw_points(right_shoulder, right_elbow, right_wrist)
                        anngles.append(angle)
        return anngles


    
