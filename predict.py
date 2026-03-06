import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

# ---------------- PATHS ----------------
YOLO_SEG_MODEL = r"C:\Users\BOMMAKOLA SURESH\Downloads\runs\segment\train\weights\best.pt"
CLS_MODEL      = r"C:\Users\BOMMAKOLA SURESH\SeedGermination_RT\models\dinov2_vits14_germ.pt"
OUT_DIR        = r"C:\Users\BOMMAKOLA SURESH\SeedGermination_RT\outputs"
# --------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLS_IMG_SIZE = 196  # must be multiple of 14

# Professional Colors
GERM_COLOR = (50, 205, 50)      # Bright Green
NON_COLOR  = (30, 144, 255)     # Professional Blue
TEXT_COLOR = (255, 255, 255)

# -------------------------------------------------
# CLASSIFIER LOADING
# -------------------------------------------------

def load_classifier():
    ckpt = torch.load(CLS_MODEL, map_location=DEVICE)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
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

    return model, idx_to_class

# -------------------------------------------------
# PREPROCESS
# -------------------------------------------------

def preprocess_crop(bgr_crop):
    crop = cv2.resize(bgr_crop, (CLS_IMG_SIZE, CLS_IMG_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    return x.to(DEVICE)

# -------------------------------------------------
# CLASSIFICATION
# -------------------------------------------------

def classify_crop(model, idx_to_class, crop_bgr):
    x = preprocess_crop(crop_bgr)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
        cls_idx = int(torch.argmax(prob).item())
        conf = int(prob[cls_idx].item() * 100)
    return idx_to_class[cls_idx], conf

# -------------------------------------------------
# CLEAN PROFESSIONAL LABEL DRAWING
# -------------------------------------------------

def draw_label(img, x1, y1, x2, label, conf, color):

    text = f"{label.upper()}  {conf}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1

    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)

    # Prevent going outside image
    y1 = max(y1, h + 15)

    # Draw filled rectangle above box
    cv2.rectangle(img, (x1, y1 - h - 12), (x1 + w + 12, y1), color, -1)

    # Draw text
    cv2.putText(
        img,
        text,
        (x1 + 6, y1 - 6),
        font,
        scale,
        TEXT_COLOR,
        thickness,
        cv2.LINE_AA
    )

# -------------------------------------------------
# MAIN FRAME PROCESSING
# -------------------------------------------------

def process_frame(frame, yolo, clf, idx_to_class):

    res = yolo.predict(frame, imgsz=640, conf=0.25, iou=0.6, verbose=False)[0]

    germ_count = 0
    nong_count = 0

    if res.masks is None:
        return frame, 0, 0, 0.0

    boxes = res.boxes.xyxy.cpu().numpy()

    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])

        pad = 4
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(frame.shape[1]-1, x2 + pad)
        y2p = min(frame.shape[0]-1, y2 + pad)

        crop = frame[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            continue

        label, conf = classify_crop(clf, idx_to_class, crop)

        if label == "germinated":
            germ_count += 1
            box_color = GERM_COLOR
        else:
            nong_count += 1
            box_color = NON_COLOR

        # Slightly thinner, cleaner box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        draw_label(frame, x1, y1, x2, label, conf, box_color)

    total = germ_count + nong_count
    pct = (germ_count / total * 100.0) if total > 0 else 0.0

    # ---- Modern Transparent Summary Panel ----
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (520, 80), (0, 0, 0), -1)
    alpha = 0.65
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    summary_text = f"🌱 Germinated: {germ_count}    ❌ Non-Germinated: {nong_count}    Germination Rate: {pct:.1f}%"

    cv2.putText(frame, summary_text, (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA)

    return frame, germ_count, nong_count, pct

# -------------------------------------------------
# IMAGE MODE
# -------------------------------------------------

def run_on_image(path):
    os.makedirs(OUT_DIR, exist_ok=True)

    yolo = YOLO(YOLO_SEG_MODEL)
    clf, idx_to_class = load_classifier()

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)

    out_img, g, n, pct = process_frame(img, yolo, clf, idx_to_class)

    out_path = os.path.join(OUT_DIR, "pred_" + os.path.basename(path))
    cv2.imwrite(out_path, out_img)

    print("✅ Saved:", out_path)
    print(f"Result -> Germinated={g}, Non-germ={n}, Germ%={pct:.1f}")

# -------------------------------------------------
# VIDEO MODE
# -------------------------------------------------

def run_on_video(path):
    os.makedirs(OUT_DIR, exist_ok=True)

    yolo = YOLO(YOLO_SEG_MODEL)
    clf, idx_to_class = load_classifier()

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(OUT_DIR,
                            "pred_" + os.path.splitext(os.path.basename(path))[0] + ".mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        out_frame, _, _, _ = process_frame(frame, yolo, clf, idx_to_class)
        out.write(out_frame)

    cap.release()
    out.release()
    print("✅ Saved:", out_path)

# -------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    args = p.parse_args()

    ext = os.path.splitext(args.source)[1].lower()

    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        run_on_image(args.source)
    else:
        run_on_video(args.source)