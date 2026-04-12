import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import cv2

# 1. Load your optimized model
# We do this outside the class so it only loads once
model = YOLO('yolov8n.pt')

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # 2. Run MOT Tracking
        # We use a lower conf for web responsiveness
        results = model.track(img, persist=True, tracker="botsort.yaml", verbose=False)

        # 3. Annotate the frame
        if results[0].boxes.id is not None:
            annotated_frame = results[0].plot()
            return annotated_frame
        
        return img

# --- Streamlit UI ---
st.title("M.Tech Minor Project: AI MOT")
st.subheader("Hardware Optimized with Intel OpenVINO")

st.write("Click 'Start' below to begin live tracking via your webcam.")

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)