import streamlit as st
import tempfile, os, cv2
import numpy as np
from PIL import Image
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="🎥📷 Detectron2 Demo", layout="wide")
st.title("🎥📷 Object Detection (Faster R-CNN)")

# Load Detectron2 model
@st.cache_resource
def load_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cpu"  # ⚡ change to "cuda" if GPU available
    return DefaultPredictor(cfg)

predictor = load_model()

# =========================
# HELPERS
# =========================
def run_inference_image(image: Image.Image):
    img = np.array(image)[:, :, ::-1]  # RGB -> BGR
    outputs = predictor(img)
    return draw_boxes(img.copy(), outputs)

def draw_boxes(frame, outputs):
    v = outputs["instances"].to("cpu")
    boxes = v.pred_boxes.tensor.numpy()
    for box in boxes:
        x1,y1,x2,y2 = map(int, box)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    return frame[:, :, ::-1]  # back to RGB for display

def run_inference_video(video_path):
    cap = cv2.VideoCapture(video_path)
    out_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        outputs = predictor(frame)
        processed = draw_boxes(frame, outputs)
        out_frames.append(processed)
    cap.release()
    return out_frames

# =========================
# UI
# =========================
mode = st.radio("Choose mode:", ["Image", "Video"])

if mode == "Image":
    img_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, caption="Uploaded Image", width=400)
        with st.spinner("Running detection..."):
            result = run_inference_image(img)
        st.image(result, caption="Detections", width=600)

else:  # Video mode
    vid_file = st.file_uploader("Upload a video", type=["mp4","mov","avi"])
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(vid_file.read())
        video_path = tfile.name
        st.video(video_path)

        with st.spinner("Processing video... (might take time)"):
            frames = run_inference_video(video_path)

        st.success(f"✅ Processed {len(frames)} frames!")
        # Show first few frames
        for i, f in enumerate(frames[:5]):
            st.image(f, caption=f"Frame {i}", width=400)

        # Optionally save processed video
        save_path = os.path.join(tempfile.gettempdir(), "processed.mp4")
        h, w, _ = frames[0].shape
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w,h))
        for f in frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
        st.success("🎞️ Processed video saved!")
        with open(save_path, "rb") as file:
            st.download_button("Download Processed Video", file, file_name="processed.mp4")
