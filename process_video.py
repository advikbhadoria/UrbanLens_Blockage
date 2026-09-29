import cv2
import time
import requests
import os
import sys
from ultralytics import YOLO

MODEL_PATH = "best.pt"
VIDEO_PATH = "road_blockage_sample.mp4" if os.path.exists("road_blockage_sample.mp4") else 0
SERVER_URL = "http://127.0.0.1:8000/report_hazard"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"[ERROR] Could not open video source: {VIDEO_PATH}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("annotated_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

bus_lat, bus_lon = 18.5204, 73.8567
last_logged_time = 0
ALERT_COOLDOWN = 3.0

print(f"[INFO] Running Obstruction Detection on: {VIDEO_PATH}")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, conf=0.35, device="cpu", verbose=False)
    current_time = time.time()

    for r in results:
        if r.boxes is not None:
            for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                class_name = model.names[int(cls_id)]
                confidence = float(conf)

                if current_time - last_logged_time > ALERT_COOLDOWN:
                    payload = {
                        "hazard_type": class_name,
                        "latitude": bus_lat,
                        "longitude": bus_lon,
                        "confidence": confidence
                    }
                    try:
                        requests.post(SERVER_URL, json=payload, timeout=0.3)
                        print(f"[OBSTRUCTION DETECTED] Type: {class_name} | Confidence: {confidence:.2f}")
                        last_logged_time = current_time
                    except Exception:
                        pass

    annotated_frame = results[0].plot()
    out.write(annotated_frame)

    cv2.imshow("SIH 2026 - Edge Obstruction Detection", annotated_frame)
    delay = int(1000 / fps) if isinstance(VIDEO_PATH, str) else 1
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("[SUCCESS] Processing complete. Saved to 'annotated_output.mp4'.")