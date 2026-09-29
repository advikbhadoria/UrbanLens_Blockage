import os
import cv2
from ultralytics import YOLO

MODEL_PATH = "best.pt"
TEST_IMG = "test_media/1.png"  # Or select any of your test images

if not os.path.exists(MODEL_PATH):
  print(f"[ERROR] '{MODEL_PATH}' not found.")
  exit()

model = YOLO(MODEL_PATH)
print("Model classes:", model.names)

# Run with low threshold (0.10) to catch all raw proposals
results = model(TEST_IMG, conf=0.10, device="cpu")

print("\n--- DETECTIONS FOUND ---")
boxes = results[0].boxes
if boxes is not None and len(boxes) > 0:
  for box, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
    class_name = model.names[int(cls_id)]
    print(f"-> Detected: {class_name} | Confidence: {float(conf)*100:.2f}%")
else:
  print("Zero detections even at 10% confidence.")

annotated = results[0].plot()
cv2.imshow("Raw Model Output (conf=0.10)", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()