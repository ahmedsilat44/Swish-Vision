import cv2
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        # frame = cv2.flip(frame,0)
        if not ret:
            break
        frames.append(frame)
    return frames, fps

def write_video(frames, output_dir, fps=30):
    if not os.path.exists(os.path.dirname(output_dir)):
        os.mkdir(os.path.dirname(frames))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_dir, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()
