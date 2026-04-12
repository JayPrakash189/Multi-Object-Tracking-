#The Live Demo:This is the code provid in my message. Use this for my actual presentation.
import cv2
from ultralytics import YOLO
import time

# 1. Load the OPTIMIZED folder
model = YOLO('yolov8n_openvino_model/')

cap = cv2.VideoCapture(0)
unique_ids = set()
start_time = time.time()
frame_count = 0

print("Live MOT Running. Press 'Q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (640, 480))

    # 4. RUN TRACKING
    results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        unique_ids.update(ids)

    annotated_frame = results[0].plot()
    
    frame_count += 1
    fps = frame_count / (time.time() - start_time)

    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Unique Tracks: {len(unique_ids)}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Minor Project: Live Multi-Object Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()