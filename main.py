# from ultralytics import YOLO

# model = YOLO("yolov8x.pt")  # Load a pretrained model (or specify a custom path)

# results = model.predict("input_videos/vid1.mp4", save=True)  # Use webcam as source

# print(results)  # Print results to console
# print("=====================")
# for box in results[0].boxes:
#     print(box)  # Print box coordinates, confidence, and class

# imports are always needed
# import torch


# # get index of currently selected device
# torch.cuda.current_device() # returns 0 in my case


# # get number of GPUs available
# torch.cuda.device_count() # returns 1 in my case


# # get the name of the device
# torch.cuda.get_device_name(0) # good old Tesla K80

# # setting device on GPU if available, else CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)
# print()


# #Additional Info when using cuda
# if device.type == 'cuda':
#     print(torch.cuda.get_device_name(0))
#     print('Memory Usage:')
#     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
#     print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')




from utils import read_video, write_video
from trackers.ball_tracker import BallTracker
from trackers.rim_tracker import RimTracker
from drawers.shot_tracker import ShotTracker
from drawers.ball_tracks_drawer import BallTracksDrawer
from drawers.rim_tracks_drawer import RimTracksDrawer


def main():
    vidname = "vid1"
    video_frames = read_video(f"input_videos/{vidname}.mp4")

    ball_tracker = BallTracker(model_path="models/bestYT.pt")
    rim_tracker = RimTracker(model_path="models/bestYT.pt")

    ball_tracks = ball_tracker.get_object_tracks(video_frames)
    # rim_tracks = rim_tracker.get_object_tracks(video_frames)

    # ball_tracks = ball_tracker.remove_wrong_tracks(ball_tracks)

    interpolated_ball_tracks = ball_tracker.interpolate_missing_tracks(ball_tracks)
    rim_tracks = rim_tracker.interpolate_missing_tracks(ball_tracks)

    ball_tracks_drawer = BallTracksDrawer()
    out_video_frames = ball_tracks_drawer.draw(video_frames, interpolated_ball_tracks)

    rim_tracks_drawer = RimTracksDrawer()
    two_out_video_frames = rim_tracks_drawer.draw(out_video_frames, rim_tracks)

    shot_tracker = ShotTracker()
    shot_tracker.detect_shot(two_out_video_frames, interpolated_ball_tracks, rim_tracks)
    four_out_video_frames = shot_tracker.draw_shots(two_out_video_frames)



    # print(video_frames)

    write_video(four_out_video_frames, f"output_videos/output_{vidname}.avi")

if __name__ == "__main__":
    main()