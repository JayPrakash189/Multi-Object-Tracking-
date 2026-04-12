Real-Time Multi-Object Tracking (MOT) with Hardware Optimization
This project implements a high-performance Multi-Object Tracking system using YOLOv8 and BoT-SORT. It is designed with a dual-deployment strategy: a hardware-optimized local version for Intel CPUs and a cloud-compatible web application.

### Deployment Modes
1. Local Deployment (High Performance)
Optimized for Intel 13th Gen Core i7 using the OpenVINO toolkit.

Inference Engine: OpenVINO (FP16 Quantization).

Performance: Real-time tracking at 30+ FPS on local hardware.

How to run:

pip install -r requirements.txt

python optimize_model.py (Run once to generate the optimized folder)

python live_tracking_mot.py

2. Web Deployment (Streamlit Cloud)
A cross-platform web app for remote accessibility.

Inference Engine: Standard PyTorch (CPU-based).

Web Framework: Streamlit using st.camera_input for secure browser access.

Live Demo: [https://motjay.streamlit.app]

🛠 Tech Stack-
AI Model: Ultralytics YOLOv8 (Nano).

Tracking: BoT-SORT (Identity preservation and motion prediction).

Optimization: Intel OpenVINO Toolkit.

UI/UX: Streamlit & OpenCV.

 Project Structure-
live_tracking_mot.py: Main local execution script.

app.py: Streamlit web application.

optimize_model.py: Hardware optimization script.

requirements.txt: Python dependency list.

packages.txt: Linux system dependencies for cloud deployment.

 Performance Metrics-
The system tracks unique object IDs and monitors live FPS to demonstrate the efficiency of hardware-specific optimization.

Final Pro-Tip for your Demo:
On GitHub, your file list should now look like this:

app.py (Must be in the main folder to avoid "Main module does not exist")

live_tracking_mot.py

optimize_model.py

requirements.txt

packages.txt

yolov8n.pt

.gitignore 
