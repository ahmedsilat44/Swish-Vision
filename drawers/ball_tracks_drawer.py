from .utils import draw_elipse, get_center, get_box_width, draw_trajectory


class BallTracksDrawer:
    def __init__(self):
        self.ball_tracks = {}  # Store ball centers by track_id
        self.max_trail_length = 30  # Limit trail length to avoid clutter

    def draw(self,video_frames, tracks):
        
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):

            frame = frame.copy()
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