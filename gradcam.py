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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#Arguments
parser = argparse.ArgumentParser(description="Grad-CAM visualizer — PyTorch")
parser.add_argument("--model",  required=True, help="Path to .pth model file")
parser.add_argument("--input",  required=True, help="Path to input image")
parser.add_argument("--output", default=None,  help="Save path for output image")
args = parser.parse_args()

CLASSES  = ["FAKE", "REAL"]
IMG_SIZE = 299
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load model
print(f"[INFO] Loading model: {args.model}")
if not os.path.exists(args.model):
    print(f"[ERROR] Model file not found: {args.model}")
    sys.exit(1)

model = timm.create_model("xception", pretrained=False, num_classes=2)
model.load_state_dict(torch.load(args.model, map_location=device))
model = model.to(device)
model.eval()
print("[INFO] Model loaded")

# Transform 
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def find_last_conv(model):
    """Walk all named modules and return the name + module of the last Conv2d."""
    last_name   = None
    last_module = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_name   = name
            last_module = module
    if last_name is None:
        raise RuntimeError("No Conv2d layer found in model.")
    return last_name, last_module


# GradCAM
def compute_gradcam(model, img_tensor, target_class=None):
   
    _, target_layer = find_last_conv(model)

    saved_activation = {}
    saved_gradient   = {}

    def forward_hook(module, input, output):
        saved_activation["value"] = output

    def backward_hook(module, grad_input, grad_output):
        saved_gradient["value"] = grad_output[0]

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        # Forward pass
        output = model(img_tensor)          
        probs  = F.softmax(output, dim=1)[0]
        probs_list = probs.detach().cpu().numpy().tolist()

        if target_class is None:
            target_class = int(probs.argmax().item())

        # Backward pass
        model.zero_grad()
        score = output[0, target_class]
        score.backward()

    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    activations = saved_activation["value"].detach().cpu()   
    gradients   = saved_gradient["value"].detach().cpu()     

    weights = gradients[0].mean(dim=(1, 2))                  

    heatmap = torch.zeros(activations.shape[2:])            
    for i, w in enumerate(weights):
        heatmap += w * activations[0, i]                     

    heatmap = torch.clamp(heatmap, min=0)
    heatmap = heatmap / (heatmap.max() + 1e-8)

    return heatmap.numpy(), target_class, probs_list


#Overlay heatmap
def overlay_heatmap(heatmap, original_bgr, alpha=0.45):
    """
    Resize heatmap to match original_bgr and overlay with JET colormap.
    Returns (side_by_side, overlay_only).
    """
    h, w = original_bgr.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay   = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_color, alpha, 0)
    divider   = np.ones((h, 10, 3), dtype=np.uint8) * 200
    side_by_side = np.hstack([original_bgr, divider, overlay])

    return side_by_side, overlay


if __name__ == "__main__":
    # Load image
    original_bgr = cv2.imread(args.input)
    if original_bgr is None:
        print(f"[ERROR] Cannot read: {args.input}")
        sys.exit(1)

    # Prepare tensor
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    pil_img      = Image.fromarray(original_rgb)
    tensor       = preprocess(pil_img).unsqueeze(0).to(device)

    # Compute Grad-CAM
    print("[INFO] Computing Grad-CAM ...")
    heatmap, predicted_idx, probs = compute_gradcam(model, tensor)

    label      = CLASSES[predicted_idx]
    confidence = probs[predicted_idx]

    print(f"\n  Prediction : {label}")
    print(f"  Confidence : {confidence * 100:.2f}%")
    print(f"  FAKE prob  : {probs[0] * 100:.2f}%")
    print(f"  REAL prob  : {probs[1] * 100:.2f}%")

    display_bgr = cv2.resize(original_bgr, (IMG_SIZE, IMG_SIZE))
    side_by_side, overlay = overlay_heatmap(heatmap, display_bgr)

    color_map = {"REAL": (0, 200, 0), "FAKE": (0, 0, 220)}
    color     = color_map.get(label, (200, 200, 200))
    h_out, w_out = side_by_side.shape[:2]
    cv2.rectangle(side_by_side, (0, 0), (w_out, 50), (0, 0, 0), -1)
    cv2.putText(side_by_side, "ORIGINAL", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    cv2.putText(side_by_side, f"GRAD-CAM  {label} ({confidence*100:.1f}%)",
                (w_out // 2 + 20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    base         = args.input.rsplit(".", 1)[0]
    output_path  = args.output if args.output else f"{base}_gradcam.jpg"
    cv2.imwrite(output_path, side_by_side)
    print(f"\n[INFO] Grad-CAM saved -> {output_path}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Grad-CAM — {label}  ({confidence*100:.1f}% confidence)",
                 fontsize=14, fontweight="bold")

    axes[0].imshow(cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Heatmap  (red = focus region)")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    report_path = f"{base}_gradcam_report.png"
    plt.savefig(report_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Report figure  -> {report_path}")
    print("\nGrad-CAM complete!")