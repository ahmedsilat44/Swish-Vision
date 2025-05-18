# from ultralytics import YOLO

# model = YOLO("yolov8x.pt")  # Load a pretrained model (or specify a custom path)

# results = model.predict("input_videos/vid1.mp4", save=True)  # Use webcam as source

# print(results)  # Print results to console
# print("=====================")
# for box in results[0].boxes:
#     print(box)  # Print box coordinates, confidence, and class

# import torch
# print(torch.__version__)
# print(hasattr(torch, 'load'))

from utils import read_video, write_video
from trackers.ball_tracker import BallTracker
from trackers.rim_tracker import RimTracker
from drawers.ball_tracks_drawer import BallTracksDrawer
from drawers.rim_tracks_drawer import RimTracksDrawer


def main():
    video_frames = read_video("input_videos/vid8.mp4")

    ball_tracker = BallTracker(model_path="models/bestYT.pt")
    # rim_tracker = RimTracker(model_path="models/best.pt")

    ball_tracks = ball_tracker.get_object_tracks(video_frames)
    # rim_tracks = ball_tracker.get_object_tracks(video_frames)

    # ball_tracks = ball_tracker.remove_wrong_tracks(ball_tracks)

    ball_tracks = ball_tracker.interpolate_missing_tracks(ball_tracks)

    ball_tracks_drawer = BallTracksDrawer()
    out_video_frames = ball_tracks_drawer.draw(video_frames, ball_tracks)

    rim_tracks_drawer = RimTracksDrawer()
    two_out_video_frames = rim_tracks_drawer.draw(out_video_frames, ball_tracks)

    # print(video_frames)

    write_video(two_out_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()