# ============================================================
#   MINOR PROJECT — Multi Object Tracking (Live Webcam)
#   Lenovo IdeaPad Slim 3 | Windows 11 | Python
# ============================================================

# STEP 1 — Import libraries
from ultralytics import YOLO
import cv2
import time

# ============================================================
# STEP 2 — Load AI Model
# ============================================================
print("=" * 50)
print("  MINOR PROJECT - Multi Object Tracking")
print("=" * 50)
print("\nLoading AI model... please wait")

model = YOLO('yolov8n.pt')  # Downloads automatically on first run (~6MB)
print("Model loaded successfully!")

# ============================================================
# STEP 3 — Open Webcam
# ============================================================
print("\nStarting webcam...")
cap = cv2.VideoCapture(0)  # 0 = built-in laptop webcam

# Check if webcam opened
if not cap.isOpened():
    print("ERROR: Cannot open webcam!")
    print("Try changing VideoCapture(0) to VideoCapture(1)")
    exit()

print("Webcam started!")
print("\nControls:")
print("  Press Q = Quit")
print("  Press S = Save screenshot")
print("  Press P = Pause/Resume")
print("=" * 50)

# ============================================================
# STEP 4 — Setup variables
# ============================================================
frame_count = 0
fps_start_time = time.time()
fps = 0
paused = False
unique_ids = set()

# ============================================================
# STEP 5 — Main Tracking Loop
# ============================================================
while True:

    # If paused, wait for key press
    if paused:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('p'):
            paused = False
            print("Resumed!")
        elif key == ord('q'):
            break
        continue

    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Cannot read from webcam. Exiting...")
        break

    # Resize frame — smaller = faster on your laptop
    frame = cv2.resize(frame, (640, 480))

    # --------------------------------------------------------
    # STEP 6 — Run YOLO Detection + Tracking
    # --------------------------------------------------------
    results = model.track(
        frame,
        persist=True,           # Remember IDs across frames
        tracker="botsort.yaml", # Tracking algorithm
        conf=0.4,               # Confidence threshold (0-1)
        verbose=False           # Suppress output logs
    )

    # --------------------------------------------------------
    # STEP 7 — Collect unique tracked IDs
    # --------------------------------------------------------
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        unique_ids.update(ids)

    # --------------------------------------------------------
    # STEP 8 — Draw boxes and labels on frame
    # --------------------------------------------------------
    annotated_frame = results[0].plot()

    # --------------------------------------------------------
    # STEP 9 — Calculate and display FPS
    # --------------------------------------------------------
    frame_count += 1
    if frame_count % 10 == 0:
        fps_end_time = time.time()
        fps = 10 / (fps_end_time - fps_start_time)
        fps_start_time = time.time()

    # --------------------------------------------------------
    # STEP 10 — Add info text on screen
    # --------------------------------------------------------
    # FPS counter (top left)
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 255, 0), 2
    )

    # Total unique IDs (top center)
    cv2.putText(
        annotated_frame,
        f"Total Tracked: {len(unique_ids)}",
        (200, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 0), 2
    )

    # Project title (bottom)
    cv2.putText(
        annotated_frame,
        "Minor Project - Multi Object Tracking | Press Q to quit",
        (10, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (200, 200, 200), 1
    )

    # --------------------------------------------------------
    # STEP 11 — Show the frame on screen
    # --------------------------------------------------------
    cv2.imshow("Minor Project - Multi Object Tracking", annotated_frame)

    # --------------------------------------------------------
    # STEP 12 — Handle keyboard input
    # --------------------------------------------------------
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):          # Q = Quit
        print("\nQuitting...")
        break

    elif key == ord('s'):        # S = Save screenshot
        filename = f"screenshot_frame_{frame_count}.jpg"
        cv2.imwrite(filename, annotated_frame)
        print(f"Screenshot saved: {filename}")

    elif key == ord('p'):        # P = Pause
        paused = True
        print("Paused. Press P to resume.")

# ============================================================
# STEP 13 — Cleanup and print results
# ============================================================
cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print("  PROJECT COMPLETE!")
print("=" * 50)
print(f"  Total frames processed : {frame_count}")
print(f"  Total unique IDs tracked: {len(unique_ids)}")
print(f"  Average FPS            : {fps:.1f}")
print("=" * 50)