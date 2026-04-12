import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import av
from ultralytics import YOLO

# 1. Load the basic model (Safe for Cloud)
model = YOLO('yolov8n.pt') 

# 2. WebRTC configuration for STABLE connection
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # 3. Optimization: Run tracking only on high-confidence detections
    # Reduce 'imgsz' to 320 to make it 4x faster on the cloud
    results = model.track(img, persist=True, imgsz=320, conf=0.5, verbose=False)

    annotated_frame = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

st.title("Live MOT Deployment")

# 4. Use 'SENDONLY' mode to reduce server load
webrtc_streamer(
    key="mot",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False}, # No audio = Faster
    async_processing=True, # Critical: This prevents the UI from freezing
)