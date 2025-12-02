import cv2
import numpy as np


class HumanTracksDrawer:
    """
    Draw YOLOv8 Results on frames (bboxes, labels, scores, keypoints + skeleton).
    Works with Ultralytics Results list aligned to frames: detections[i] corresponds to video_frames[i].
    """

    # COCO-17 keypoint skeleton (Ultralytics default order)
    # (index pairs to connect with lines)
    COCO_SKELETON = [
        (5, 7), (7, 9),        # left arm: L-shoulder->L-elbow->L-wrist
        (6, 8), (8,10),        # right arm: R-shoulder->R-elbow->R-wrist
        (11,13), (13,15),      # left leg: L-hip->L-knee->L-ankle
        (12,14), (14,16),      # right leg: R-hip->R-knee->R-ankle
        (5,6), (11,12),        # shoulders, hips
        (5,11), (6,12),        # torso diagonals
        (0,1), (1,2), (2,3), (3,4), (0,5), (0,6)  # head/neck connections
    ]
    COCO_SKELETON_Names = [
        "Nose", 
        "Left Eye", 
        "Right Eye", 
        "Left Ear", 
        "Right Ear", 
        "Left Shoulder", 
        "Right Shoulder", 
        "Left Elbow", 
        "Right Elbow", 
        "Left Wrist", 
        "Right Wrist", 
        "Left Hip", 
        "Right Hip", 
        "Left Knee", 
        "Right Knee", 
        "Left Ankle", 
        "Right Ankle"
    ]

    def __init__(
        self,
        box_color=(0, 255, 0),
        kp_color=(255, 0, 0),
        skeleton_color=(0, 255, 255),
        skeleton_color_rhs=(255,20,147),
        text_color=(255, 255, 255),
        box_thickness=2,
        kp_radius=3,
        sk_thickness=2,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.5,
    ):
        self.box_color = box_color
        self.kp_color = kp_color
        self.skeleton_color = skeleton_color
        self.skeleton_color_rhs = skeleton_color_rhs        
        self.text_color = text_color
        self.box_thickness = box_thickness
        self.kp_radius = kp_radius
        self.sk_thickness = sk_thickness
        self.font = font
        self.font_scale = font_scale

    def _put_label(self, img, label, x1, y1):
        (tw, th), _ = cv2.getTextSize(label, self.font, self.font_scale, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), self.box_color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), self.font, self.font_scale, self.text_color, 1, cv2.LINE_AA)

    def _draw_box(self, img, box, label=None):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), self.box_color, self.box_thickness)
        if label:
            self._put_label(img, label, x1, y1)

    def _draw_keypoints(self, img, kps_xy, kps_conf=None, conf_thr=0.2):
        """
        kps_xy: (K,2) ndarray or list of (x,y)
        kps_conf: (K,) confidences or None
        """
        K = len(kps_xy)
        # draw joints
        for i in range(K):
            x, y = kps_xy[i]
            if x is None or y is None:
                continue
            # Skip invalid coordinates (0,0) or very close to origin
            if abs(x) < 1 and abs(y) < 1:
                continue
            if kps_conf is not None and kps_conf[i] is not None and kps_conf[i] < conf_thr:
                continue
            cv2.circle(img, (int(x), int(y)), self.kp_radius, self.kp_color, -1)

        # draw skeleton
        for a, b in self.COCO_SKELETON:
            if a >= K or b >= K:
                continue
            xa, ya = kps_xy[a]
            xb, yb = kps_xy[b]
            if None in (xa, ya, xb, yb):
                continue
            # Skip invalid coordinates (0,0) or very close to origin
            if (abs(xa) < 1 and abs(ya) < 1) or (abs(xb) < 1 and abs(yb) < 1):
                continue
            if kps_conf is not None:
                ca = kps_conf[a] if kps_conf[a] is not None else 1.0
                cb = kps_conf[b] if kps_conf[b] is not None else 1.0
                if ca < conf_thr or cb < conf_thr:
                    continue
            if ((a,b) == (6, 8) or (a,b) == (8,10)):
                cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), self.skeleton_color_rhs, self.sk_thickness)
            else:
                cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), self.skeleton_color, self.sk_thickness)
            
    def write_coords(self, img, kps_xy, kps_conf=None, conf_thr=0.2):
        parts_oi = [6,8,10,12] #right arm stuff

        # for parts in range(len(kps_xy)):
        #     coord = kps_xy[parts]
        #     part = self.COCO_SKELETON_Names[parts]
        #     cv2.putText(img, f"{part} coords: {coord}", (10, 90+parts*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        offset = 0
        for parts in parts_oi:
            offset += 30
            coord = kps_xy[parts]
            part = self.COCO_SKELETON_Names[parts]
            
            # Check if coordinate is valid
            if coord[0] is None or coord[1] is None:
                cv2.putText(img, f"{part} coords: N/A", (10, 60+offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                continue
                
            # Check confidence threshold
            if kps_conf is not None and parts < len(kps_conf):
                if kps_conf[parts] is not None and kps_conf[parts] < conf_thr:
                    cv2.putText(img, f"{part} coords: Low confidence", (10, 60+offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    continue
            
            x = round(coord[0],4)
            y = round(coord[1],4)
            coord = (x,y)
            cv2.putText(img, f"{part} coords: {coord}", (10, 60+offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        

    def write_angles(self, img, angle_sew, angle_esh):
        if angle_sew is None:
            cv2.putText(img, "Right S-E-W angle: N/A", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            angle_sew = round(angle_sew, 4)
            cv2.putText(img, f"Right S-E-W angle: {angle_sew} deg", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if angle_esh is None:
            cv2.putText(img, "Right E-S-H angle: N/A", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            angle_esh = round(angle_esh, 4)
            cv2.putText(img, f"Right E-S-H angle: {angle_esh} deg", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            

    def draw(
        self,
        video_frames,
        detections,
        angles,
        draw_boxes=True,
        draw_keypoints=True,
        draw_labels=True,
        score_thr=0.25,
        kpt_thr=0.2,
        class_names=None  # optional override; otherwise uses Ultralytics model names if present
    ):
        """
        Args:
            video_frames: list[np.ndarray] BGR frames
            detections: list[ultralytics.engine.results.Results] aligned 1:1 with frames
            draw_boxes, draw_keypoints, draw_labels: toggles
            score_thr: min confidence for bbox label display
            kpt_thr: min confidence for a keypoint to be drawn
            class_names: optional list of class names
        Returns:
            list[np.ndarray]: frames with overlays (same length/order as input)
        """
        with open("xy_coords.txt", "w") as f:
            f.write("")

        out_frames = []
        for i, frame in enumerate(video_frames):
            res = detections[i]
            # img = frame.copy()
            img = frame
            current_angle_sew = angles[0][i]
            current_angle_esh = angles[1][i]

            # --- Boxes, labels, ids ---
            boxes = getattr(res, "boxes", None)
            names = getattr(res, "names", None) or class_names

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
                clss  = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else None
                ids   = boxes.id.cpu().numpy().astype(int) if hasattr(boxes, "id") and boxes.id is not None else None

                for j, box in enumerate(xyxy):
                    label_text = None
                    if draw_labels:
                        parts = []
                        if clss is not None and names is not None and clss[j] < len(names):
                            parts.append(str(names[clss[j]]))
                        if confs is not None and confs[j] is not None and confs[j] >= score_thr:
                            parts.append(f"{confs[j]:.2f}")
                        if ids is not None:
                            parts.append(f"ID:{ids[j]}")
                        if parts:
                            label_text = " ".join(parts)

                    if draw_boxes:
                        self._draw_box(img, box, label_text)

            # --- Keypoints ---
            kps = getattr(res, "keypoints", None)
            if draw_keypoints and (kps is not None) and (len(kps) > 0):
                # Ultralytics exposes .xy (n,k,2) and .conf (n,k) when available
                try:
                    kps_xy = kps.xy.cpu().numpy()  # shape: [N, K, 2]
                except:
                    # Fallback to data[..., :2]
                    data = kps.data
                    if data is None or len(data) == 0:
                        kps_xy = None
                    else:
                        arr = data[..., :2].cpu().numpy()
                        kps_xy = arr

                kps_cf = None
                if hasattr(kps, "conf") and kps.conf is not None:
                    kps_cf = kps.conf.cpu().numpy()  # [N, K]
                else:
                    # If confidence isn’t provided, treat all as confident
                    pass

                if kps_xy is not None:
                    N = kps_xy.shape[0] #num people
                    for n in range(N):
                        joints = [(float(x), float(y)) for x, y in kps_xy[n]]
                        if kps_cf is not None:
                            confs = [float(c) if c is not None else None for c in kps_cf[n]]
                        else:
                            confs = [1.0] * len(joints)

                        # with open("xy_coords.txt", "a") as f:
                        #     f.write(str(joints))
                        #     f.write("\n")

                        self._draw_keypoints(img, joints, confs, conf_thr=kpt_thr)
                        self.write_coords(img, joints, confs, conf_thr=kpt_thr)
                        self.write_angles(img, current_angle_sew, current_angle_esh)

            out_frames.append(img)

        return out_frames

    def analysis():
        pass