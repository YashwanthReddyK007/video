import streamlit as st
import os, tempfile, json, shutil
import torch
import numpy as np
from PIL import Image
from collections import Counter
import faiss
import open_clip
import cv2
from ultralytics import YOLO

# =======================
# SETTINGS
# =======================
FRAMES_FOLDER = "frames"
os.makedirs(FRAMES_FOLDER, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="📷 Video & Image Search", layout="wide")
st.title("📷🔎 Search with YOLOv11 + CLIP + FAISS")

# =======================
# LOAD MODELS
# =======================
if "yolo_model" not in st.session_state:
    st.session_state.yolo_model = YOLO("yolo11n.pt")  # place yolo11n.pt in project folder
yolo_model = st.session_state.yolo_model

@st.cache_resource
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model.to(device).eval(), preprocess, tokenizer

clip_model, clip_preprocess, tokenizer = load_clip()

# =======================
# FUNCTIONS
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
            fname = f"{os.path.basename(video_path)}_{count}.jpg"
            fpath = os.path.join(FRAMES_FOLDER, fname)
            cv2.imwrite(fpath, frame)
            saved.append(fname)
        count += 1
    cap.release()
    return saved

# =======================
# UPLOAD SECTION
# =======================
st.header("📤 Upload Images or Videos")
uploaded_files = st.file_uploader("Upload files", type=["jpg", "jpeg", "png", "mp4"], accept_multiple_files=True)

if uploaded_files:
    all_filenames, all_objects, all_embeddings = [], {}, []

    for up in uploaded_files:
        ext = os.path.splitext(up.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(up.read())
            tmp_path = tmp.name

        if ext in [".jpg", ".jpeg", ".png"]:
            # move image
            fname = os.path.basename(tmp_path)
            dst = os.path.join(FRAMES_FOLDER, fname)
            shutil.move(tmp_path, dst)
            frames = [fname]
        else:
            # video -> extract frames
            frames = extract_frames(tmp_path)

        for frame in frames:
            fpath = os.path.join(FRAMES_FOLDER, frame)
            objs = detect_objects(fpath)
            all_objects[frame] = objs
            emb = encode_image(fpath)
            all_embeddings.append(emb)
            all_filenames.append(frame)

    if all_embeddings:
        with open("metadata.txt", "w") as f:
            for m in all_filenames:
                f.write(f"{m}\n")
        with open("objects.json", "w") as f:
            json.dump(all_objects, f, indent=2)
        dim = all_embeddings[0].shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(np.vstack(all_embeddings))
        faiss.write_index(index, "media_index.faiss")
        st.success("✅ Processing done! You can search now.")
    else:
        st.error("⚠️ No embeddings created.")

# =======================
# SEARCH SECTION
# =======================
st.markdown("---")
st.header("🔎 Search")
if os.path.exists("media_index.faiss") and os.path.exists("metadata.txt"):
    query = st.text_input("Enter a description to search:")
    if query:
        index = faiss.read_index("media_index.faiss")
        metadata = [line.strip() for line in open("metadata.txt")]
        objects_map = json.load(open("objects.json"))

        tok = tokenizer([query])
        with torch.no_grad():
            tfeat = clip_model.encode_text(tok.to(device))
            tfeat /= tfeat.norm(dim=-1, keepdim=True)

        D, I = index.search(tfeat.cpu().numpy(), k=5)
        # threshold to ignore irrelevant matches
        st.subheader(f"Results for **'{query}'**:")
        shown = False
        for idx, score in zip(I[0], D[0]):
            if score < 0.3:  # ignore low similarity
                continue
            shown = True
            fname = metadata[idx]
            img_path = os.path.join(FRAMES_FOLDER, fname)
            if os.path.exists(img_path):
                st.image(img_path, caption=f"{fname} (score {score:.3f})", width=400)
                for obj in objects_map.get(fname, []):
                    st.write(f"- **{obj['label']}** ({obj['conf']:.2f})")
        if not shown:
            st.warning("❌ No relevant results found.")

# =======================
# SUMMARY SECTION
# =======================
st.markdown("---")
st.header("📋 Summary")
if st.button("Summarize Objects"):
    if os.path.exists("objects.json"):
        objects_map = json.load(open("objects.json"))
        all_labels = [obj["label"] for objs in objects_map.values() for obj in objs]
        if not all_labels:
            st.warning("No objects detected.")
        else:
            counts = Counter(all_labels)
            for label, count in counts.most_common():
                st.write(f"- **{label}** appears in {count} frames")
    else:
        st.info("No objects data yet.")
