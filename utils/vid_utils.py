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

    height, width = frames[0].shape[:2]

    # Write frames to a temporary mp4v file, then transcode to H.264 with
    # ffmpeg so the result is browser-compatible (mp4v/MPEG-4 Part 2 is not).
    base, ext = os.path.splitext(output_path)
    temp_path = base + "_tmp.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    total = len(frames)

    def progress_bar(i, total):
        pct = (i / total) * 100
        bar_len = 40
        filled = int((i / total) * bar_len)
        bar = "█" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\rWriting video: [{bar}] {pct:6.2f}% ({i}/{total})")
        sys.stdout.flush()

    for i, frame in enumerate(frames, start=1):
        out.write(frame)
        progress_bar(i, total)

    out.release()
    print("\nTranscoding to H.264...")

    # Transcode to H.264 for browser compatibility
    transcoded = False
    try:
        import subprocess
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        result = subprocess.run(
            [
                ffmpeg_exe, "-y", "-i", temp_path,
                "-vcodec", "libx264", "-preset", "fast", "-crf", "23",
                "-movflags", "+faststart",
                "-an",
                output_path,
            ],
            capture_output=True,
        )
        if result.returncode == 0:
            os.remove(temp_path)
            transcoded = True
            print("Done!")
        else:
            print("ffmpeg transcode failed, keeping mp4v output.")
            print(result.stderr.decode(errors="replace")[-500:])
    except Exception as e:
        print(f"ffmpeg not available ({e}), keeping mp4v output.")

    if not transcoded:
        # Fallback: rename temp file to final path
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_path, output_path)
