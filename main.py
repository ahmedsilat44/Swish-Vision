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
from trackers.human_tracker import HumanTracker
from drawers.shot_tracker import ShotTracker
from drawers.ball_tracks_drawer import BallTracksDrawer
from drawers.rim_tracks_drawer import RimTracksDrawer
from drawers.human_tracks_drawer import HumanTracksDrawer
from utils.ball_hand import ball_hand, shot_started
import os

def main_pipeline(vidname):



    print(f"Reading video: input_videos/{vidname}.mp4")
    video_frames, fps = read_video(f"input_videos/{vidname}.mp4")


    print("ball_tracker = BallTracker(model_path=")
    ball_tracker = BallTracker(model_path="models/best.pt")

    print("rim_tracker = RimTracker(model_path=")
    rim_tracker = RimTracker(model_path="models/best.pt")

    print("human_tracker = HumanTracker(model_path=")
    human_tracker = HumanTracker(model_path="models/yolov8m-pose.pt")

    print("ball_tracks = ball_tracker.get_object_tracks(video_frames)")
    ball_tracks = ball_tracker.get_object_tracks(video_frames)

    # rim_tracks = rim_tracker.get_object_tracks(video_frames)


    print("human_tracks = human_tracker.detect_frame(video_frames)")
    human_tracks = human_tracker.detect_frame(video_frames)
    angles = human_tracker.calc_angles(video_frames, human_tracks)
    points = human_tracker.get_points(video_frames, human_tracks)



    # with open("angs.txt", "w") as f:
    #     f.write(str(angles))
    #     f.write("\n")


    # rim_tracks = rim_tracker.get_object_tracks(video_frames)


    ball_tracks = ball_tracker.remove_wrong_tracks(ball_tracks)
    rim_tracks = rim_tracker.remove_wrong_tracks(ball_tracks)
    interpolated_ball_tracks = ball_tracker.interpolate_missing_tracks(ball_tracks)

    ball_loco = ball_tracker.get_ball_loco(video_frames, interpolated_ball_tracks)
    rim_tracks = rim_tracker.interpolate_missing_tracks(rim_tracks)

    ball_left_frames = ball_hand(ball_loco, points, video_frames)
    print(f"Ball left frames: {ball_left_frames}")

    shot_starts = shot_started(points, ball_left_frames)
    print(f"Shot starts: {shot_starts}")


    # Drawers
    ball_tracks_drawer = BallTracksDrawer()
    # out_video_frames = ball_tracks_drawer.draw(video_frames, ball_tracks)
    print("1")
    # out_video_frames = ball_tracks_drawer.draw(video_frames, interpolated_ball_tracks)

    rim_tracks_drawer = RimTracksDrawer()
    print("2")
    # two_out_video_frames = rim_tracks_drawer.draw(out_video_frames, rim_tracks)

    # Add human keypoints drawing
    human_tracks_drawer = HumanTracksDrawer()
    print("3")
    three_out_video_frames = human_tracks_drawer.draw(video_frames, human_tracks, angles, draw_boxes=False, draw_keypoints=True)
    four_out_video_frames = human_tracks_drawer.analysis(three_out_video_frames, angles, ball_left_frames, shot_starts, f"{vidname}_report.txt")




    shot_tracker = ShotTracker()
    print("4")
    # shot_tracker.detect_shot(three_out_video_frames, interpolated_ball_tracks, rim_tracks)
    shot_tracker.detect_shot(video_frames, interpolated_ball_tracks, rim_tracks)
    # shot_tracker.detect_shot(three_out_video_frames, ball_tracks, rim_tracks)

    print("5")
    five_out_video_frames = shot_tracker.draw_shots(four_out_video_frames)

    # six_out_video_frames = ball_tracks_drawer.draw_ball_left(five_out_video_frames, ball_left_frames)



    print("Making video...")

   # write_video(four_out_video_frames, f"output_videos/output_{vidname}_second_angle.avi", fps=fps)
    write_video(five_out_video_frames, f"output_videos/output_{vidname}_processed.avi", fps=fps)

    ##if using this dont forget to change in vid_utils.py the fourcc to mp4v
    # write_video(four_out_video_frames, f"output_videos/output_{vidname}.mp4", fps=fps)

    # print(len(ball_loco))
    # print(len(angles[0]))
    # print(len(angles[1]))
    # print(len(points))
    # print(len(video_frames))


def run_pipeline(input_path: str, session_id: int = None) -> tuple:
    """
    Entry point for the Celery task.

    Parameters
    ----------
    input_path : str
        Absolute or relative path to the input video that has already been
        copied into input_videos/.  Only the stem (name without extension) is
        used to locate the file, so the file must reside in input_videos/.
    session_id : int, optional
        Database session ID (unused by the CV pipeline itself, reserved for
        future report-persistence work).

    Returns
    -------
    tuple[str, str]
        (output_path, report_path) – paths produced by main_pipeline.
    """
    vidname = os.path.splitext(os.path.basename(input_path))[0]
    main_pipeline(vidname)
    output_path = os.path.join("output_videos", f"output_{vidname}_processed.avi.avi")
    report_path = f"{vidname}_report.txt"
    return output_path, report_path


def main():
    # main_pipeline("vid14_1")
    # main_pipeline("vid13_1")
    main_pipeline("vid18")
    main_pipeline("vid20")

if __name__ == "__main__":
    main()