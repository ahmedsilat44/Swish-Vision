from .utils import draw_elipse, get_center, get_box_width, draw_trajectory
import cv2


class BallTracksDrawer:
    def __init__(self):
        self.ball_tracks = {}  # Store ball centers by track_id
        self.max_trail_length = 5  # Limit trail length to avoid clutter
        

    def draw(self,video_frames, tracks):
        
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):

            # frame = frame.copy()
            # frame = cv2.flip(frame,0)

            player_dict = tracks[frame_num]

            for track_id, track in player_dict.items():
                if track["bbox"] is None:
                    continue
                # print(track_id)
                # frame = draw_elipse(frame, track["box"], (0, 255, 0))
                box = track["bbox"]
                # Assume track has a "class" key (e.g., "ball" or "basket")
                label = track["class"]

                if label == "Basketball":
                    # Store ball center
                    center = get_center(box)
                    

                    if track_id not in self.ball_tracks:
                        self.ball_tracks[track_id] = []
                    self.ball_tracks[track_id].append(center)
                    # Limit trail length
                    if len(self.ball_tracks[track_id]) > self.max_trail_length:
                        self.ball_tracks[track_id].pop(0)

                    # Draw trajectory
                    frame = draw_trajectory(frame, self.ball_tracks[track_id], (0, 255, 0))

                    # Draw current bounding box
                    frame = draw_elipse(frame, box, (0, 255, 0), track_id, "Basketball")
                
                
            output_video_frames.append(frame)
        
        return output_video_frames

    def draw_ball_left(self, video_frames, leave_frames):
        linger=30
        h, w = video_frames[0].shape[:2]
        position = (w // 2, h // 2)

        font_scale = 1.0
        color = (0, 0, 255)
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Ball left hand"

        annotated_frames = []

        # Convert leave_frames to pure Python ints
        leave_frames = [int(f) for f in leave_frames]

        # Build a set of frames to draw on, including LINGER window
        draw_frames = set()
        for lf in leave_frames:
            # Add lf, lf+1, lf+2, ..., lf+linger
            for k in range(linger + 1):
                draw_frames.add(lf + k)

        for i, frame in enumerate(video_frames):
            frame_out = frame

            if i in draw_frames:
                cv2.putText(frame_out, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
                cv2.imwrite(f"temp/frame_{i}.png", frame_out)

            annotated_frames.append(frame_out)

        return annotated_frames

            
