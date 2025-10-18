import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

frames_dir = "processed_frames"
roi_mask_path = "roi_mask.png"
output_csv = "detections.csv"
model_path = "yolov8m.pt"

print("[INFO] Memuat model YOLO...")
model = YOLO(model_path)

print("[INFO] Memuat ROI mask...")
roi_mask = cv2.imread(roi_mask_path, cv2.IMREAD_GRAYSCALE)
roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)[1]

vehicle_classes = {"car", "bus", "truck", "motorcycle"}

results_data = []

print("[INFO] Mulai deteksi kendaraan...")
for frame_name in sorted(os.listdir(frames_dir)):
    if not frame_name.endswith(".png"):
        continue

    frame_path = os.path.join(frames_dir, frame_name)
    frame_id = int(frame_name.split("_")[-1].split(".")[0])

    frame = cv2.imread(frame_path)

    detections = model(frame)[0]

    for box in detections.boxes:
        cls_name = model.names[int(box.cls)]
        if cls_name not in vehicle_classes:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        w, h = x2 - x1, y2 - y1

        cx, cy = x1 + w // 2, y1 + h // 2
        if roi_mask[cy, cx] == 0:
            continue

        results_data.append({
            "frame_id": frame_id,
            "bbox_x": x1,
            "bbox_y": y1,
            "bbox_w": w,
            "bbox_h": h,
            "class": cls_name
        })

df = pd.DataFrame(results_data)
df.to_csv(output_csv, index=False)
print(f"[INFO] Selesai. Hasil tersimpan di {output_csv}")
