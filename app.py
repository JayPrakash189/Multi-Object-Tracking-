import streamlit as st
import cv2
from ultralytics import YOLO

# 1. Load standard model
model = YOLO('yolov8n.pt') 

st.title("M.Tech Minor Project: Live MOT")
st.write("Click 'Start' to activate the camera. If it freezes, try reducing your browser zoom.")

# 2. Setup placeholders to prevent UI lag
run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])

# 3. Use standard OpenCV capture (More stable for beginners)
camera = cv2.VideoCapture(0)

while run:
    _, frame = camera.read()
    if frame is None:
        break
    
    # 4. Convert BGR to RGB for Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 5. Fast Tracking: Small image size for Cloud speed
    results = model.track(frame, persist=True, imgsz=320, verbose=False)
    
    # 6. Plot and Display
    annotated_frame = results[0].plot()
    FRAME_WINDOW.image(annotated_frame)

else:
    st.write('Camera Stopped')
    camera.release()