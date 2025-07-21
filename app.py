import streamlit as st
import os
import tempfile
import json
import cv2
import torch
import numpy as np
from PIL import Image
from collections import Counter
import faiss
import open_clip
from ultralytics import YOLO
import shutil

# =========================
# SETTINGS
# =========================
FRAMES_FOLDER = "frames"
os.makedirs(FRAMES_FOLDER, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="🎥 Video Search with YOLOv11 + CLIP", layout="wide")
st.title("🎥🔎 Video Search with YOLOv11 + CLIP")

# =========================
# LOAD MODELS
# =========================
# Load YOLOv11
if "yolo_model" not in st.session_state:
    st.session_state.yolo_model = YOLO("yolo11x.pt")  # latest high-accuracy model
yolo_model = st.session_state.yolo_model

@st.cache_resource
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model.to(device).eval(), preprocess, tokenizer

clip_model, clip_preprocess, tokenizer = load_clip()

# =========================
# FUNCTIONS
# =========================
def extract_frames(video_path, every_n=30):
    """Extract every_n-th frame from the video and save to FRAMES_FOLDER."""
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n == 0:
            fname = f"{os.path.splitext(os.path.basename(video_path))[0]}_{count}.jpg"
            fpath = os.path.join(FRAMES_FOLDER, fname)
            cv2.imwrite(fpath, frame)
            saved_frames.append(fname)
        count += 1
    cap.release()
    return saved_frames

def detect_objects(img_path):
    """Run YOLOv11 object detection on an image."""
    results = yolo_model(img_path, conf=0.3, verbose=False)
    objs = []
    for box in results[0].boxes:
        objs.append({
            "label": results[0].names[int(box.cls[0])],
            "conf": float(box.conf[0])
        })
    return objs

def encode_image(img_path):
    """Get CLIP embedding for an image."""
    img = Image.open(img_path).convert("RGB")
    tens = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(tens)
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()

# =========================
# VIDEO UPLOAD & PROCESSING
# =========================
uploaded_videos = st.file_uploader(
    "🎞️ Upload one or more videos",
    type=["mp4", "avi", "mov"],
    accept_multiple_files=True
)

if uploaded_videos:
    all_filenames = []
    all_objects = {}
    all_embeddings = []

    for video in uploaded_videos:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.name)[1]) as tmp:
            tmp.write(video.read())
            tmp_path = tmp.name

        st.write(f"⏳ Processing **{video.name}**...")
        frames = extract_frames(tmp_path, every_n=30)  # adjust for more/less frames
        for frame in frames:
            src = os.path.join(FRAMES_FOLDER, frame)
            all_filenames.append(frame)

            # detect and encode
            objs = detect_objects(src)
            all_objects[frame] = objs
            emb = encode_image(src)
            all_embeddings.append(emb)

        # clean up temp video
        os.remove(tmp_path)

    # Save metadata
    with open("metadata.txt", "w") as f:
        for m in all_filenames:
            f.write(f"{m}\n")
    with open("objects.json", "w") as f:
        json.dump(all_objects, f, indent=2)

    # Build FAISS index
    if all_embeddings:
        dim = all_embeddings[0].shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(np.vstack(all_embeddings))
        faiss.write_index(index, "video_index.faiss")
        st.success("✅ Processing complete! You can now search.")
    else:
        st.error("⚠️ No embeddings created.")

st.markdown("---")
st.header("🔎 Search processed frames")

if os.path.exists("video_index.faiss") and os.path.exists("metadata.txt"):
    query = st.text_input("Enter a text description to search (e.g. 'car', 'dog'): ")
    if query:
        index = faiss.read_index("video_index.faiss")
        metadata = [line.strip() for line in open("metadata.txt")]
        objects_map = json.load(open("objects.json"))

        tok = tokenizer([query])
        with torch.no_grad():
            tfeat = clip_model.encode_text(tok.to(device))
            tfeat /= tfeat.norm(dim=-1, keepdim=True)

        D, I = index.search(tfeat.cpu().numpy(), k=5)

        results_found = False
        st.subheader(f"Top matches for **'{query}'**:")
        for idx, score in zip(I[0], D[0]):
            if score < 0.2:  # threshold to filter irrelevant results
                continue
            results_found = True
            fname = metadata[idx]
            img_path = os.path.join(FRAMES_FOLDER, fname)
            if os.path.exists(img_path):
                st.image(img_path, caption=f"{fname} (score {score:.3f})", width=400)
                st.write("Objects detected:")
                for obj in objects_map.get(fname, []):
                    st.write(f"- **{obj['label']}** ({obj['conf']:.2f})")
            else:
                st.write(f"❌ Image not found: {img_path}")
        if not results_found:
            st.warning("❌ No relevant results found for your query.")

    st.markdown("---")
    st.header("📋 Summarize detected objects")
    if st.button("Summarize"):
        objects_map = json.load(open("objects.json"))
        all_labels = [obj["label"] for objs in objects_map.values() for obj in objs]
        if not all_labels:
            st.warning("No objects detected yet.")
        else:
            counts = Counter(all_labels)
            st.write("✅ **Summary of detected objects:**")
            for label, count in counts.most_common():
                st.write(f"- **{label}** appears in {count} frame(s)")
else:
    st.info("Upload and process videos first.")
