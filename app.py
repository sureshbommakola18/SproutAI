import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ---------------- CONFIG ----------------
YOLO_MODEL = "models/yolo_best.pt"
CLS_MODEL  = "models/dinov2_vits14_germ.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLS_IMG_SIZE = 196

GERM_COLOR = (72, 161, 17)     # Green
NON_COLOR  = (255, 0, 0)    # Blue

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SproutAI",
    page_icon="🌱",
    layout="wide"
)

# ---------------- CUSTOM BACKGROUND ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

h1, h2, h3 {
    color: #ffffff;
}

.block-container {
    padding-top: 2rem;
}

.metric-container {
    background-color: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("🌱 SproutAI")
st.markdown("### Intelligent Seed Germination Analysis System")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    yolo = YOLO(YOLO_MODEL)

    ckpt = torch.load(CLS_MODEL, map_location=DEVICE)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    backbone = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
        pretrained=True
    )

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    EMB_DIM = 384

    class GermHead(torch.nn.Module):
        def __init__(self, bb):
            super().__init__()
            self.bb = bb
            self.head = torch.nn.Sequential(
                torch.nn.LayerNorm(EMB_DIM),
                torch.nn.Linear(EMB_DIM, 2)
            )

        def forward(self, x):
            with torch.no_grad():
                feats = self.bb.forward_features(x)
                x = feats["x_norm_clstoken"]
            return self.head(x)

    model = GermHead(backbone).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return yolo, model, idx_to_class


def preprocess_crop(bgr_crop):
    crop = cv2.resize(bgr_crop, (CLS_IMG_SIZE, CLS_IMG_SIZE))
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    return x.to(DEVICE)


def classify_crop(model, idx_to_class, crop_bgr):
    x = preprocess_crop(crop_bgr)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
        cls_idx = int(torch.argmax(prob).item())
    return idx_to_class[cls_idx]


def process_image(img, yolo, clf, idx_to_class):

    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    res = yolo.predict(frame, imgsz=420, conf=0.25, iou=0.6, verbose=False)[0]

    germ_count = 0
    nong_count = 0

    if res.boxes is not None:
        boxes = res.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            label = classify_crop(clf, idx_to_class, crop)

            if label == "germinated":
                germ_count += 1
                color = GERM_COLOR
            else:
                nong_count += 1
                color = NON_COLOR

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    total = germ_count + nong_count
    pct = (germ_count / total * 100) if total > 0 else 0

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb, germ_count, nong_count, pct


# ---------------- UI ----------------
uploaded_file = st.file_uploader("Upload Seed Image", type=["jpg","png","jpeg"])

if uploaded_file:

    yolo, clf, idx_to_class = load_models()
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, width="stretch")

    result_img, g, n, pct = process_image(image, yolo, clf, idx_to_class)

    with col2:
        st.subheader("Predicted Image")
        st.image(result_img, width="stretch")

    st.markdown("---")

    colA, colB, colC = st.columns(3)
    colA.metric("Total Seeds", g + n)
    colB.metric("Germinated", g)
    colC.metric("Non-Germinated", n)

    st.progress(pct / 100)
    st.write(f"### Germination Percentage: {pct:.1f}%")

    st.markdown("---")
    st.markdown("### Legend")

    st.markdown(
        """
        <div style="display:flex; gap:40px; align-items:center;">
            <div>
                <div style="width:25px;height:25px;background-color:rgb(72, 161, 17);display:inline-block;"></div>
                Germinated
            </div>
            <div>
                <div style="width:25px;height:25px;background-color:rgb(30,144,255);display:inline-block;"></div>
                Non-Germinated
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )