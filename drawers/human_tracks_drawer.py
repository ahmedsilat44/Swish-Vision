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

    def __init__(
        self,
        box_color=(0, 255, 0),
        kp_color=(255, 0, 0),
        skeleton_color=(0, 255, 255),
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
            if kps_conf is not None:
                ca = kps_conf[a] if kps_conf[a] is not None else 1.0
                cb = kps_conf[b] if kps_conf[b] is not None else 1.0
                if ca < conf_thr or cb < conf_thr:
                    continue
            cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), self.skeleton_color, self.sk_thickness)

    def draw(
        self,
        video_frames,
        detections,
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
        out_frames = []
        for i, frame in enumerate(video_frames):
            res = detections[i]
            img = frame.copy()

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
                    N = kps_xy.shape[0]
                    for n in range(N):
                        joints = [(float(x), float(y)) for x, y in kps_xy[n]]
                        if kps_cf is not None:
                            confs = [float(c) if c is not None else None for c in kps_cf[n]]
                        else:
                            confs = [1.0] * len(joints)
                        self._draw_keypoints(img, joints, confs, conf_thr=kpt_thr)

            out_frames.append(img)

        return out_frames

