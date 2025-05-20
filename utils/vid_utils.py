import cv2
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        # frame = cv2.flip(frame,0)
        if not ret:
            break
        frames.append(frame)
    return frames

def write_video(frames, output_dir):
    if not os.path.exists(os.path.dirname(output_dir)):
        os.mkdir(os.path.dirname(frames))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_dir, fourcc, 30.0, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()
