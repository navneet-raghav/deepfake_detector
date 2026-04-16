
import os
import sys
import argparse

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

try:
    import timm
except ImportError:
    print("[ERROR] timm not installed. Run: pip install timm")
    sys.exit(1)

#Arguments 
parser = argparse.ArgumentParser()
parser.add_argument("--model",     required=True,       help="Path to XceptionNet .pth")
parser.add_argument("--model2",    default=None,         help="Path to EfficientNet-B0 .pth (enables ensemble)")
parser.add_argument("--input",     required=True,       help="Path to image or video file")
parser.add_argument("--video",     action="store_true", help="Treat input as video")
parser.add_argument("--threshold", type=float, default=0.75,
                    help="Confidence threshold — below this outputs UNCERTAIN (default 0.75)")
parser.add_argument("--output",    default=None,        help="Save annotated image to this path")
args = parser.parse_args()

CLASSES  = ["FAKE", "REAL"]  
IMG_SIZE = 299
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_timm_model(path, arch):
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    m = timm.create_model(arch, pretrained=False, num_classes=2)
    m.load_state_dict(torch.load(path, map_location=device))
    m = m.to(device)
    m.eval()
    return m


print(f"[INFO] Loading XceptionNet: {args.model}")



if "efficientnet" in args.model:
    model1 = load_timm_model(args.model, "efficientnet_b0")
else:
    model1 = load_timm_model(args.model, "xception")



model2 = None
if args.model2:
    print(f"[INFO] Loading EfficientNet-B0: {args.model2}")
    model2 = load_timm_model(args.model2, "efficientnet_b0")
    print("[INFO] Mode: ensemble (averaging both models)")
else:
    print("[INFO] Mode: single model")

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def predict_array(img_bgr):
    """BGR numpy array -> (label, confidence, probs list)."""
    rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)

    with torch.no_grad():
        p1 = F.softmax(model1(tensor), dim=1)[0].cpu().numpy()

    if model2 is not None:
        with torch.no_grad():
            p2 = F.softmax(model2(tensor), dim=1)[0].cpu().numpy()
        probs = ((p1 + p2) / 2).tolist()
    else:
        probs = p1.tolist()

    idx   = int(np.argmax(probs))
    conf  = probs[idx]
    label = CLASSES[idx] if conf >= args.threshold else "UNCERTAIN"
    return label, conf, probs


def annotate(img_bgr, label, confidence, probs):
    out   = img_bgr.copy()
    h, w  = out.shape[:2]
    colors = {"REAL": (0, 200, 0), "FAKE": (0, 0, 220), "UNCERTAIN": (0, 165, 255)}
    color  = colors.get(label, (200, 200, 200))

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    suffix = " [ensemble]" if model2 else ""
    cv2.putText(out, f"{label}  {confidence*100:.1f}%{suffix}",
                (10, 42), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)

    bar_y  = h - 28
    fake_w = int(w * probs[0])
    cv2.rectangle(out, (0, bar_y), (w, h), (30, 30, 30), -1)
    cv2.rectangle(out, (0, bar_y), (fake_w, h), (0, 0, 200), -1)
    cv2.rectangle(out, (fake_w, bar_y), (w, h), (0, 200, 0), -1)
    cv2.putText(out, f"FAKE {probs[0]*100:.0f}%", (5, h-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(out, f"REAL {probs[1]*100:.0f}%", (w-110, h-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def run_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] Cannot read image: {path}")
        sys.exit(1)

    label, conf, probs = predict_array(img)
    mode = "ensemble" if model2 else "single"

    print(f"\n{'='*48}")
    print(f"  File       : {os.path.basename(path)}")
    print(f"  Mode       : {mode}")
    print(f"  Prediction : {label}")
    print(f"  Confidence : {conf*100:.2f}%")
    print(f"  FAKE prob  : {probs[0]*100:.2f}%")
    print(f"  REAL prob  : {probs[1]*100:.2f}%")
    print(f"{'='*48}\n")

    out_path  = args.output or path.rsplit(".", 1)[0] + "_result.jpg"
    cv2.imwrite(out_path, annotate(img, label, conf, probs))
    print(f"[INFO] Saved -> {out_path}")


def run_video(path, max_frames=30):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {path}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    step  = max(1, total // max_frames)
    print(f"[INFO] {total} frames @ {fps:.1f} FPS — sampling every {step}")

    all_probs = []
    fi        = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % step == 0 and len(all_probs) < max_frames:
            _, _, p = predict_array(frame)
            all_probs.append(p)
        fi += 1
    cap.release()

    if not all_probs:
        print("[ERROR] No frames processed.")
        sys.exit(1)

    avg    = np.mean(all_probs, axis=0).tolist()
    idx    = int(np.argmax(avg))
    conf   = avg[idx]
    label  = CLASSES[idx] if conf >= args.threshold else "UNCERTAIN"
    fake_n = sum(1 for p in all_probs if p[0] > p[1])

    print(f"\n{'='*52}")
    print(f"  File           : {os.path.basename(path)}")
    print(f"  Frames sampled : {len(all_probs)}")
    print(f"  FAKE frames    : {fake_n}  ({fake_n/len(all_probs)*100:.1f}%)")
    print(f"  REAL frames    : {len(all_probs)-fake_n}")
    print(f"  Final verdict  : {label}")
    print(f"  Avg confidence : {conf*100:.2f}%")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    if args.video:
        run_video(args.input)
    else:
        run_image(args.input)