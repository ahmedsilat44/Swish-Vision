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

import cv2
import os
import sys

def write_video(frames, output_path, fps=30):
    # Ensure directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Setup video writer
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total = len(frames)

    def progress_bar(i, total):
        pct = (i / total) * 100
        bar_len = 40
        filled = int((i / total) * bar_len)
        bar = "█" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\rWriting video: [{bar}] {pct:6.2f}% ({i}/{total})")
        sys.stdout.flush()

    # Write frames with progress bar
    for i, frame in enumerate(frames, start=1):
        out.write(frame)
        progress_bar(i, total)

    out.release()
    print("\nDone!")  # move to new line
