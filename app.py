import streamlit as st
import os, tempfile, json, shutil
import torch
import numpy as np
from PIL import Image
from collections import Counter
import faiss
import open_clip
from ultralytics import YOLO
import cv2

# =======================
# SETTINGS
# =======================
FRAMES_FOLDER = "frames"
os.makedirs(FRAMES_FOLDER, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="📷🎥 Search with YOLOv11 + CLIP", layout="wide")
st.title("📷🎥🔎 Image & Video Search with YOLOv11 + CLIP")

# =======================
# LOAD MODELS
# =======================
if "yolo_model" not in st.session_state:
    st.session_state.yolo_model = YOLO("yolov11n.pt")  # ✅ use YOLOv11
yolo_model = st.session_state.yolo_model

@st.cache_resource
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model.to(device).eval(), preprocess, tokenizer

clip_model, clip_preprocess, tokenizer = load_clip()

# =======================
# HELPER FUNCTIONS
# =======================
def detect_objects(img_path):
    results = yolo_model(img_path, conf=0.3, verbose=False)
    objs = []
    for box in results[0].boxes:
        objs.append({
            "label": results[0].names[int(box.cls[0])],
            "conf": float(box.conf[0])
        })
    return objs

def encode_image(img_path):
    img = Image.open(img_path).convert("RGB")
    tens = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(tens)
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()

def extract_frames(video_path, every_n=30):
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n == 0:
            # ✅ convert BGR to RGB to avoid blank image issue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            fname = f"{os.path.splitext(os.path.basename(video_path))[0]}_{count}.jpg"
            fpath = os.path.join(FRAMES_FOLDER, fname)
            img.save(fpath)
            saved.append(fname)
        count += 1
    cap.release()
    return saved

# =======================
# UPLOAD SECTION
# =======================
uploaded_images = st.file_uploader(
    "Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)
uploaded_videos = st.file_uploader(
    "Upload videos", type=["mp4", "avi", "mov"], accept_multiple_files=True
)

all_filenames = []
all_objects = {}
all_embeddings = []

# --- Process Images ---
if uploaded_images:
    for image_file in uploaded_images:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_file.name)[1]) as tmp:
            tmp.write(image_file.read())
            tmp_path = tmp.name

        dst_path = os.path.join(FRAMES_FOLDER, os.path.basename(tmp_path))
        shutil.move(tmp_path, dst_path)
        all_filenames.append(os.path.basename(tmp_path))

        objs = detect_objects(dst_path)
        all_objects[os.path.basename(tmp_path)] = objs
        emb = encode_image(dst_path)
        all_embeddings.append(emb)

# --- Process Videos ---
if uploaded_videos:
    for video_file in uploaded_videos:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        st.write(f"⏳ Extracting frames from {video_file.name}...")
        frames = extract_frames(tmp_path, every_n=30)
        for f in frames:
            img_path = os.path.join(FRAMES_FOLDER, f)
            objs = detect_objects(img_path)
            all_objects[f] = objs
            emb = encode_image(img_path)
            all_embeddings.append(emb)
            all_filenames.append(f)

# --- Save metadata if anything was uploaded ---
if uploaded_images or uploaded_videos:
    with open("metadata.txt", "w") as f:
        for m in all_filenames:
            f.write(f"{m}\n")
    with open("objects.json", "w") as f:
        json.dump(all_objects, f, indent=2)

    if all_embeddings:
        dim = all_embeddings[0].shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(np.vstack(all_embeddings))
        faiss.write_index(index, "media_index.faiss")
        st.success("✅ Media processed successfully! You can now search.")
    else:
        st.error("⚠️ No embeddings created.")

# =======================
# SEARCH SECTION
# =======================
st.markdown("---")
st.header("🔎 Search Media")

if os.path.exists("media_index.faiss") and os.path.exists("metadata.txt"):
    query = st.text_input("Enter a description (e.g. 'car', 'dog'):")
    if query:
        index = faiss.read_index("media_index.faiss")
        metadata = [line.strip() for line in open("metadata.txt")]
        objects_map = json.load(open("objects.json"))

        tok = tokenizer([query])
        with torch.no_grad():
            tfeat = clip_model.encode_text(tok.to(device))
            tfeat /= tfeat.norm(dim=-1, keepdim=True)

        D, I = index.search(tfeat.cpu().numpy(), k=5)

        # Show results only if similarity score > threshold
        threshold = 0.25
        filtered = [(idx, score) for idx, score in zip(I[0], D[0]) if score > threshold]

        if filtered:
            st.subheader(f"Top matches for **'{query}'**:")
            for idx, score in filtered:
                fname = metadata[idx]
                img_path = os.path.join(FRAMES_FOLDER, fname)
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"{fname} (score {score:.3f})", width=400)
                else:
                    st.write(f"❌ Image not found: {img_path}")
                st.write("Objects detected:")
                for obj in objects_map.get(fname, []):
                    st.write(f"- **{obj['label']}** ({obj['conf']:.2f})")
        else:
            st.error("❌ No results found for your query.")
else:
    st.info("Upload and process images or videos first.")

# =======================
# SUMMARY
# =======================
st.markdown("---")
st.header("📋 Summarize All Uploaded Media")
if st.button("Summarize Objects"):
    if os.path.exists("objects.json"):
        objects_map = json.load(open("objects.json"))
        all_labels = [obj["label"] for objs in objects_map.values() for obj in objs]
        if not all_labels:
            st.warning("No objects detected yet.")
        else:
            counts = Counter(all_labels)
            st.write("✅ **Summary of objects detected:**")
            for label, count in counts.most_common():
                st.write(f"- **{label}** appears in {count} image(s)/frame(s)")
