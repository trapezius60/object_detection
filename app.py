import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import time

# ------------------- Page Config -------------------
st.set_page_config(page_title="Object Detection App", page_icon="🤕", layout="wide")

# ------------------- Header -------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🔎 Object Detection by yolov8x-oiv7 </h1>
    """,
    unsafe_allow_html=True
)
st.write("Upload an image or use your webcam for live detection")

# ------------------- Load Model -------------------

MODEL_OPTIONS = {
    "YOLOv8x (OIV7 – 600 Objects Detection)": "https://huggingface.co/trapezius60/yolov8x-oiv7/resolve/main/yolov8x-oiv7.pt",
    "YOLO8-wound-detection (Forensic Wound Detection)": "https://huggingface.co/trapezius60/forensic_wound_detection/resolve/main/best.pt"

selected_model = st.selectbox("Choose Detection Model", list(MODEL_OPTIONS.keys()))

@st.cache_resource
def load_model(model_url):
    return YOLO(model_url)

model = load_model(MODEL_OPTIONS[selected_model])

# ------------------- Confidence Slider -------------------
conf_thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)

# ------------------- Image Upload -------------------
uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # BGR for YOLO

    # Run detection
    results = model(img_cv, conf=conf_thresh)
    annotated_bgr = results[0].plot()  # YOLO returns BGR
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)  # convert to RGB for display

    # Display annotated image
    st.image(annotated_rgb, caption="Detection Result", use_container_width=True)

    # Save for download
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(temp_file.name, annotated_bgr)  # save BGR

    st.download_button(
        "Download Annotated Image",
        data=open(temp_file.name, "rb").read(),
        file_name="detection.png"
    )

# ------------------- Webcam Live Detection -------------------
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.captured_frame = None  # store BGR frame for capture/download

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  # input BGR from webcam
        results = model(img, conf=conf_thresh)
        annotated = results[0].plot()  # BGR for WebRTC preview
        self.captured_frame = annotated  # store for capture/download
        return annotated  # BGR preview (WebRTC expects BGR)

# Initialize webcam
webrtc_ctx = webrtc_streamer(
    key="wound-detection",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    async_transform=True,
)

# ------------------- Capture Button -------------------
st.markdown("---")

if webrtc_ctx.video_transformer:
    if st.button("📸 Capture & Download Current Frame"):
        frame_bgr = webrtc_ctx.video_transformer.captured_frame
        if frame_bgr is not None:
            # Convert BGR -> RGB -> BGR for saving (ensures correct color)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            cv2.imwrite(temp_file.name, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))  # save correctly

            st.download_button(
                "Download Captured Image",
                data=open(temp_file.name, "rb").read(),
                file_name="capture.png"
            )
        else:
            st.warning("No frame captured yet! Please wait for the webcam to initialize.")

# ------------------- Footer -------------------
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align:center; font-size:14px; color:gray;'>
        Object Detection Version: 1.0.0 | © 2025 BH <br>
        <div>
            <a href="https://docs.ultralytics.com/datasets/detect/open-images-v7/" target="_blank">📄 User Manual</a> |
            <a href="https://www.ultralytics.com/glossary/object-detection" target="_blank">👍more about object detection</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
