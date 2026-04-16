import os
import tempfile

import cv2
import numpy as np
from PIL import Image

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms

try:
    import timm
except ImportError:
    st.error("timm not installed. Run: pip install timm")
    st.stop()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title {
    font-size:2.4rem; font-weight:800; text-align:center;
    background:linear-gradient(90deg,#e63946,#457b9d);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:0.2rem;
}
.subtitle  { text-align:center; color:#888; font-size:1rem; margin-bottom:2rem; }
.r-fake    { background:#ff4b4b22; border:2px solid #ff4b4b; border-radius:12px; padding:1.2rem; text-align:center; }
.r-real    { background:#00c85322; border:2px solid #00c853; border-radius:12px; padding:1.2rem; text-align:center; }
.r-unc     { background:#ff980022; border:2px solid #ff9800; border-radius:12px; padding:1.2rem; text-align:center; }
.tip-box   { background:#1e3a5f22; border:1px solid #457b9d; border-radius:8px; padding:0.8rem; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

CLASSES  = ["FAKE", "REAL"]
IMG_SIZE = 299
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


@st.cache_resource
def load_model_cached(path, arch):
    m = timm.create_model(arch, pretrained=False, num_classes=2)
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device).eval()
    return m


def predict(m1, m2, img_rgb, threshold):
    t = preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        p1 = F.softmax(m1(t), dim=1)[0].cpu().numpy()
    if m2 is not None:
        with torch.no_grad():
            p2 = F.softmax(m2(t), dim=1)[0].cpu().numpy()
        probs = ((p1 + p2) / 2).tolist()
    else:
        probs = p1.tolist()
    idx   = int(np.argmax(probs))
    conf  = probs[idx]
    label = CLASSES[idx] if conf >= threshold else "UNCERTAIN"
    return label, conf, probs


def find_last_conv(model):
    last = None
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last


def compute_gradcam(model, img_rgb):
    tgt = find_last_conv(model)
    if tgt is None:
        return None
    act, grad = {}, {}
    fh = tgt.register_forward_hook(lambda m, i, o: act.update({"v": o}))
    bh = tgt.register_full_backward_hook(lambda m, gi, go: grad.update({"v": go[0]}))
    try:
        t   = preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
        out = model(t)
        model.zero_grad()
        out[0, out[0].argmax().item()].backward()
    except Exception:
        fh.remove(); bh.remove(); return None
    fh.remove(); bh.remove()
    a = act["v"].detach().cpu()
    g = grad["v"].detach().cpu()
    w = g[0].mean(dim=(1, 2))
    hm = sum(w[i] * a[0, i] for i in range(len(w)))
    hm = torch.clamp(hm, min=0)
    hm = hm / (hm.max() + 1e-8)
    return hm.numpy()


def gradcam_fig(img_rgb, hm):
    h, w = img_rgb.shape[:2]
    hm_r = cv2.resize(hm, (w, h))
    col  = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*hm_r), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    ov   = cv2.addWeighted(img_rgb, 0.55, col, 0.45, 0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117"); ax.axis("off")
    axes[0].imshow(img_rgb);       axes[0].set_title("Original",        color="white")
    axes[1].imshow(hm_r, cmap="jet"); axes[1].set_title("Attention map", color="white")
    axes[2].imshow(ov);            axes[2].set_title("Overlay",          color="white")
    plt.tight_layout(pad=1.0)
    return fig


def process_video(m1, m2, path, threshold, max_frames=20):
    cap   = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total // max_frames)
    all_p = []
    prog  = st.progress(0, text="Analysing video ...")
    fi    = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if fi % step == 0 and len(all_p) < max_frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, _, p = predict(m1, m2, rgb, threshold=0.0)
            all_p.append(p)
            prog.progress(min(len(all_p)/max_frames, 1.0),
                          text=f"Frame {len(all_p)}/{max_frames}")
        fi += 1
    cap.release(); prog.empty()
    if not all_p:
        return "UNCERTAIN", 0.0, [0.5, 0.5]
    avg  = np.mean(all_p, axis=0).tolist()
    idx  = int(np.argmax(avg))
    conf = avg[idx]
    return (CLASSES[idx] if conf >= threshold else "UNCERTAIN"), conf, avg


#Sidebar
with st.sidebar:
    st.markdown("## Settings")
    xc_path  = st.text_input("XceptionNet model", value="results/best_model_xception.pth")
    b0_path  = st.text_input("EfficientNet-B0 model (optional)",
                              value="results/best_model_efficientnet_b0.pth")
    threshold    = st.slider("Confidence threshold", 0.50, 0.99, 0.75, 0.01)
    show_gradcam = st.checkbox("Show Grad-CAM", value=True)
    st.divider()
    st.markdown(f"**Device:** `{device}`")
    if device.type == "cuda":
        st.markdown(f"**GPU:** `{torch.cuda.get_device_name(0)}`")
    b0_exists = os.path.exists(b0_path)
    if b0_exists:
        st.success("Ensemble mode: ON")
    else:
        st.info("Single model mode (XceptionNet)")
    st.divider()
    st.markdown("""
**FAKE** — manipulation artifacts detected  
**REAL** — no artifacts found  
**UNCERTAIN** — model confidence is low  
    """)

#Header 
st.markdown('<div class="main-title">🔍 DeepFake Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">XceptionNet + EfficientNet-B0 ensemble | Grad-CAM explainability</div>',
            unsafe_allow_html=True)

#Load models 
if not os.path.exists(xc_path):
    st.warning(f"Model not found at `{xc_path}`. "
               "Train first: `python train_dataset.py dataset/ classes.txt results/ --model xception`")
    st.stop()

with st.spinner("Loading model(s) ..."):
    try:
        m1   = load_model_cached(xc_path, "xception")
        m2   = load_model_cached(b0_path, "efficientnet_b0") if b0_exists else None
        mode = "Ensemble (XceptionNet + EfficientNet-B0)" if m2 else "XceptionNet only"
        st.success(f"Ready  |  **{mode}**  |  Device: `{device}`")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

st.divider()

# Tabs 
tab_img, tab_vid, tab_about = st.tabs(["Image", "Video", "About"])

#IMAGE TAB 
with tab_img:
    st.markdown('<div class="tip-box">Tip: for best results upload a cropped face image. The model was trained on face crops, not full photos.</div>',
                unsafe_allow_html=True)
    st.write("")
    uploaded = st.file_uploader("Upload image (JPG / PNG / WEBP)",
                                type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        pil_img   = Image.open(uploaded).convert("RGB")
        img_array = np.array(pil_img)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.subheader("Uploaded image")
            st.image(pil_img, use_container_width=True)

        with col2:
            st.subheader("Result")
            with st.spinner("Running prediction ..."):
                label, conf, probs = predict(m1, m2, img_array, threshold)

            css = {"REAL": "r-real", "FAKE": "r-fake"}.get(label, "r-unc")
            emo = {"REAL": "✅", "FAKE": "⛔"}.get(label, "⚠️")
            st.markdown(f"""
            <div class="{css}">
                <h1 style="margin:0;font-size:3rem">{emo}</h1>
                <h2 style="margin:.3rem 0 0;font-size:2rem">{label}</h2>
                <p style="margin:.3rem 0 0;opacity:.85">Confidence: <strong>{conf*100:.1f}%</strong></p>
            </div>""", unsafe_allow_html=True)
            st.write("")
            st.metric("FAKE probability", f"{probs[0]*100:.1f}%"); st.progress(probs[0])
            st.metric("REAL probability", f"{probs[1]*100:.1f}%"); st.progress(probs[1])
            if label == "UNCERTAIN":
                st.caption(f"Confidence {conf*100:.1f}% is below threshold {threshold*100:.0f}%. "
                           "Lower the threshold slider to get a definite result.")

        if show_gradcam:
            st.divider()
            st.subheader("Grad-CAM")
            st.caption("Red/warm areas = regions that most influenced the decision.")
            with st.spinner("Computing Grad-CAM ..."):
                hm = compute_gradcam(m1, img_array)
                if hm is not None:
                    fig = gradcam_fig(cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)), hm)
                    st.pyplot(fig); plt.close(fig)
                else:
                    st.warning("Grad-CAM unavailable for this model.")
 
with tab_vid:
    vid_file   = st.file_uploader("Upload video (MP4 / AVI / MOV)",
                                  type=["mp4", "avi", "mov", "webm"])
    max_frames = st.slider("Frames to sample", 5, 50, 20, 5)

    if vid_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(vid_file.read()); tmp_path = tmp.name

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.subheader("Uploaded video"); st.video(vid_file)
        with col2:
            st.subheader("Result")
            label, conf, avg = process_video(m1, m2, tmp_path, threshold, max_frames)
            css = {"REAL": "r-real", "FAKE": "r-fake"}.get(label, "r-unc")
            emo = {"REAL": "✅", "FAKE": "⛔"}.get(label, "⚠️")
            st.markdown(f"""
            <div class="{css}">
                <h1 style="margin:0;font-size:3rem">{emo}</h1>
                <h2 style="margin:.3rem 0 0;font-size:2rem">{label}</h2>
                <p style="margin:.3rem 0 0;opacity:.85">Avg confidence: <strong>{conf*100:.1f}%</strong></p>
            </div>""", unsafe_allow_html=True)
            st.write("")
            st.metric("FAKE", f"{avg[0]*100:.1f}%"); st.progress(float(avg[0]))
            st.metric("REAL", f"{avg[1]*100:.1f}%"); st.progress(float(avg[1]))
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

with tab_about:
    st.markdown("""
    ### How this works

    The uploaded image is resized to 299×299 and passed through a trained convolutional neural network.
    If both model files are present in `results/`, the app automatically runs in ensemble mode —
    it averages the outputs of XceptionNet and EfficientNet-B0 before making a decision.
    Predictions below the confidence threshold show as UNCERTAIN rather than making a wrong confident call.

    **Grad-CAM** (Gradient-weighted Class Activation Mapping) highlights which parts of the face
    the model focused on. For deepfakes, the model typically highlights eye corners, face edges,
    and the mouth area — places where blending artifacts are usually visible.

    **Limitations:**
    - Trained on Kaggle DFDC video face crops. Real-world photos with different compression may be misclassified.
    - A REAL result does not guarantee authenticity — it means no detectable artifacts were found.
    - The model performs better on face crops than full photos.
    """)

#Footer 
st.divider()
st.markdown("""
<div style="text-align:center;color:#555;font-size:.85rem">
XceptionNet + EfficientNet-B0 + PyTorch + Streamlit |
<a href="https://www.kaggle.com/c/deepfake-detection-challenge">Kaggle DFDC dataset</a>
</div>""", unsafe_allow_html=True)