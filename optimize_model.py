# The setup :This is the script you need for the "Intel Optimization" part of your project.
from ultralytics import YOLO

# Load the base model
model = YOLO('yolov8n.pt')

# Export to OpenVINO format (Intel Optimization)
# This creates the 'yolov8n_openvino_model' folder
model.export(format='openvino', half=True)

print("SUCCESS: Optimized model folder created!")