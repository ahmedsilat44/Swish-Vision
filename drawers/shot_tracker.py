import cv2
import numpy as np
import matplotlib.pyplot as plt
from .utils import get_center

class ShotTracker:
    def __init__(self):
        self.ball_tracks = {}  # Store ball centers by track_id
        self.shots = []  # Store shot events
        self.shot_zone_factor = 2  # Expand rim bbox for shot zone
        self.rim_overlap_threshold = 0.2  # Fraction of rim size for overlap
        self.look_ahead_frames = 10  # Frames to check for shot outcome
        self.max_trail_length = 30  # Match BallTracksDrawer

    def detect_shot(self, video_frames, ball_tracks, rim_tracks):
        # iterate through each frame and then keep track of the ball and rim. keep track of the ball staying above or below the rim once the ball goes from below to above the room , save the latest point of the ball. Then check if the ball is in the shot zone of the rim. If it is, then put a pending flag. then check when the ball goes below the rim again and save that point.  Then using y = mx + c check if the ball is going in the rim or not. If it is, then save the shot event as a make. If it is not, then save the shot event as a miss.
        # also track the ball bounding box size so that ther eis a posssibility of false positive when ball is infront of the rim.
        pending_shot = False
        latest_ball_point = None
        ball_box_width = 0
        ball_box_height = 0
        
        for frame_num, frame in enumerate(video_frames):
            player_dict = ball_tracks[frame_num]
            rim_dict = rim_tracks[frame_num]

            for track_id, track in player_dict.items():
                if track["bbox"] is None:
                    continue

                box = track["bbox"]
                label = track["class"]

                if label == "Basketball":
                    # Store ball center
                    center = get_center(box)
                    if track_id not in self.ball_tracks:
                        self.ball_tracks[track_id] = []
                    self.ball_tracks[track_id].append(center)

                   #check if there is a pending shot
                    if pending_shot:
                        # Check if ball is below the rim
                        for rim_track in rim_dict.values():
                            if rim_track["bbox"] is None:
                                continue

                            rim_box = rim_track["bbox"]
                            rim_center = get_center(rim_box)

                            if center[1] > rim_box[3]:
                                # check if the current ball box is smaller or larger than the previous ball box
                                current_ball_box_width = box[2] - box[0]
                                current_ball_box_height = box[3] - box[1]
                                if ((current_ball_box_width > ball_box_width * 1.2) and (current_ball_box_height > ball_box_height * 1.2)) or ((current_ball_box_width < ball_box_width * 0.8) and (current_ball_box_height < ball_box_height * 0.8)):
                                    # Ball is too big, ignore this shot
                                    self.shots.append({
                                        "frame": frame_num,
                                        "outcome": "miss",
                                        "center": center
                                    })
                                    pending_shot = False
                                    print(frame_num, "False positive shot ignored")

                                    break


                                # Ball is below the rim, check for shot outcome
                                outcome = self.check_shot_outcome(frame_num, latest_ball_point, center, rim_tracks)
                                if outcome is not None:
                                    self.shots.append({
                                        "frame": frame_num,
                                        "outcome": outcome,
                                        "center": center
                                    })
                                    pending_shot = False
                                break
                            #  if ball is still above the rim then updated the latest ball point
                            else:
                                if center[1] < rim_box[1]:
                                    latest_ball_point = center
                                    frame = cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 0,255), -1)
                                break
                            

                    else:
                        # Check if ball is above the rim
                        for rim_track in rim_dict.values():
                            if rim_track["bbox"] is None:
                                continue

                            rim_box = rim_track["bbox"]
                            rim_center = get_center(rim_box)
                            # check if ball box bottom is above rim box top
                            if center[1] < rim_box[1]:
                                shot_zone_box = [
                                                int(rim_box[0] - (rim_box[2] - rim_box[0]) * self.shot_zone_factor),
                                                int(rim_box[1] - (rim_box[3] - rim_box[1]) * self.shot_zone_factor),
                                                int(rim_box[2] + (rim_box[2] - rim_box[0]) * self.shot_zone_factor),
                                                int(rim_box[3] + (rim_box[3] - rim_box[1]) * self.shot_zone_factor)
                                                ]
                                if self.is_in_shot_zone(center, shot_zone_box):
                                    pending_shot = True
                                    latest_ball_point = center
                                    # draw point at the latest ball point
                                    frame = cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0,0, 255), -1)
                                    ball_box_width = box[2] - box[0]
                                    ball_box_height = box[3] - box[1]
                                    break

        
        
        
    
    def is_in_shot_zone(self, center, shot_zone_box):
        x = center[0]
        y = center[1]
        x1, y1, x2, y2 = shot_zone_box
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def check_shot_outcome(self, frame_num, latest_ball_point, ball_point_below_rim, rim_track):
        # Calculate the line equation y = mx + c using the two ball points
        x1, y1 = latest_ball_point
        x2, y2 = ball_point_below_rim
      
        
       

        m = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
        c = y1 - m * x1

        # Get rim center from rim_track (should be dict of rim tracks for this frame)
        rim_box = None
        for rim in rim_track[frame_num].values():
            if rim["bbox"] is not None:
                rim_box = rim["bbox"]
                break
        if rim_box is None:
            return None

        rim_center = get_center(rim_box)
        adjustment = (rim_center[0] - rim_box[0] ) / 4
        y_rim = rim_center[1]
        # Solve for x when y = y_rim
        if m == 0:
            x_at_rim = x1
        else:
            x_at_rim = (y_rim - c) / m

        # Check if x_at_rim is inside the rim bounding box
        x1_rim, y1_rim, x2_rim, y2_rim = rim_box
        if x1_rim <= x_at_rim <= x2_rim:
            return "make"
        else:
            return "miss"
        
       
    
    def draw_shots(self, video_frames):

        if len(self.shots) == 0:
            for frame_num, frame in enumerate(video_frames):
                frame = frame.copy()
                cv2.putText(frame, "No shots detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                video_frames[frame_num] = frame
            return video_frames

        #  keep track of all shots made and missed. draw on top left of video of percentage of shots made and missed. draw on top right of video the number of shots made and missed
        output_video_frames = []
        print("Total shots: ", len(self.shots))
        print("Shots made: ", len([shot for shot in self.shots if shot["outcome"] == "make"]))
        print("Shots missed: ", len([shot for shot in self.shots if shot["outcome"] == "miss"]))



        total_shots = 0
        made_shots = 0
        missed_shots = 0
        made_percentage = 0
        first_shot_frame = self.shots[0]["frame"]
        last_shot_frame = self.shots[-1]["frame"]

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            if frame_num < first_shot_frame:
                frame = cv2.putText(frame, "No shots detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif frame_num > last_shot_frame:
                frame = cv2.putText(frame, f"{made_shots} / {total_shots}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                frame = cv2.putText(frame, f"Made Percentage: {made_percentage:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                if frame_num == self.shots[total_shots]["frame"]:
                    if self.shots[total_shots]["outcome"] == "make":
                        made_shots += 1
                    else:
                        missed_shots += 1
                    total_shots+= 1
                    made_percentage = (made_shots / total_shots) * 100
                frame = cv2.putText(frame, f"{made_shots} / {total_shots}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                frame = cv2.putText(frame, f"Made Percentage: {made_percentage:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                

            output_video_frames.append(frame)



        return output_video_frames

    

        
