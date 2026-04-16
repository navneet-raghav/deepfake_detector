import os
import argparse
import cv2
import numpy as np
from pathlib import Path

# Arguments
parser = argparse.ArgumentParser(description="Extract faces from videos into dataset folders")
parser.add_argument("--video_dir",        default=None, help="Folder containing video files")
parser.add_argument("--video",            default=None, help="Single video file path")
parser.add_argument("--output_dir",       required=True, help="Where to save face images")
parser.add_argument("--label",            required=True, choices=["real", "fake"])
parser.add_argument("--frames_per_video", type=int, default=20,
                    help="Face frames to extract per video (default: 20)")
parser.add_argument("--img_size",         type=int, default=299,
                    help="Output face image size in pixels (default: 299)")
parser.add_argument("--margin",           type=float, default=0.3,
                    help="Extra margin around detected face (default: 0.3)")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

#Face detector 
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("[ERROR] Could not load Haar Cascade. Reinstall opencv-python.")
    exit(1)

print("[INFO] Face detector loaded")


def extract_face(frame, margin=0.3, output_size=299):
    """Detect largest face in frame and return cropped square. None if no face found."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        # Try with lower minNeighbors if nothing found
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60)
        )
    if len(faces) == 0:
        return None

    # Pick largest face by area
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

    fh, fw = frame.shape[:2]
    m      = int(max(w, h) * margin)
    x1     = max(0, x - m)
    y1     = max(0, y - m)
    x2     = min(fw, x + w + m)
    y2     = min(fh, y + h + m)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (output_size, output_size))


def process_video(video_path, output_dir, label, frames_per_video, img_size, margin):
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return 0

   
    indices = set(np.linspace(0, total - 1, frames_per_video, dtype=int).tolist())

    stem    = Path(video_path).stem
    saved   = 0
    frame_i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_i in indices:
            face = extract_face(frame, margin=margin, output_size=img_size)
            if face is not None:
                fname = f"{label}_{stem}_frame{frame_i:05d}.jpg"
                cv2.imwrite(os.path.join(output_dir, fname), face)
                saved += 1
        frame_i += 1

    cap.release()
    return saved


VIDEO_EXTS  = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
video_paths = []

if args.video:
    video_paths = [args.video]
elif args.video_dir:
    video_paths = [
        str(p) for p in Path(args.video_dir).iterdir()
        if p.suffix.lower() in VIDEO_EXTS
    ]
else:
    print("[ERROR] Provide --video or --video_dir")
    exit(1)

print(f"[INFO] Found {len(video_paths)} video(s)")
print(f"[INFO] Output dir  : {args.output_dir}")
print(f"[INFO] Label       : {args.label}")
print(f"[INFO] Frames/video: {args.frames_per_video}")
print()

total_saved = 0
for i, vpath in enumerate(sorted(video_paths), 1):
    print(f"  [{i:3d}/{len(video_paths)}] {os.path.basename(vpath)} ... ", end="", flush=True)
    n = process_video(vpath, args.output_dir, args.label,
                      args.frames_per_video, args.img_size, args.margin)
    print(f"{n} faces saved")
    total_saved += n

print(f"\n{'='*45}")
print(f"  Total saved : {total_saved} images")
print(f"  Location    : {args.output_dir}")
print(f"{'='*45}")
print("Face extraction complete")