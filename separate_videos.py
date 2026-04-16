import json
import shutil
import os

SAMPLE_DIR   = "train_sample_videos"          
FAKE_OUT_DIR = "raw_videos/fake_videos"
REAL_OUT_DIR = "raw_videos/real_videos"

os.makedirs(FAKE_OUT_DIR, exist_ok=True)
os.makedirs(REAL_OUT_DIR, exist_ok=True)

meta_path = os.path.join(SAMPLE_DIR, "metadata.json")
if not os.path.exists(meta_path):
    print(f"[ERROR] metadata.json not found at: {meta_path}")
    print(f"        Make sure SAMPLE_DIR points to your extracted train_sample_videos folder.")
    exit(1)

with open(meta_path) as f:
    metadata = json.load(f)

fake_count = 0
real_count = 0
missing    = 0

for filename, info in metadata.items():
    src = os.path.join(SAMPLE_DIR, filename)
    if not os.path.exists(src):
        missing += 1
        continue

    label   = info.get("label", "").upper()
    dst_dir = FAKE_OUT_DIR if label == "FAKE" else REAL_OUT_DIR
    shutil.copy(src, os.path.join(dst_dir, filename))

    if label == "FAKE":
        fake_count += 1
    else:
        real_count += 1

print(f"\n{'='*40}")
print(f"  FAKE videos copied : {fake_count}  -> {FAKE_OUT_DIR}")
print(f"  REAL videos copied : {real_count}  -> {REAL_OUT_DIR}")
if missing:
    print(f"  Missing files      : {missing}  (skipped)")
print(f"{'='*40}")
print("Done! Now run extract_faces.py on each folder.")