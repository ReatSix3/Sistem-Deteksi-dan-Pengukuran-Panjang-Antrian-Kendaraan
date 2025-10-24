import os
import cv2
import joblib
import numpy as np
import pandas as pd
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict

VIDEO_PATH = "short_1.mp4"
CLASSIFIER_PATH = "queue_classifier.pkl"
YOLO_MODEL_PATH = "runs/detect/yolov8m_cctv_hasil_v15/weights/best.pt"
ROI_MASK_PATH = "roi_mask.png"
PPM = 20.0

OUTPUT_CSV = "tracked_data.csv"

TARGET_FPS = 5
VEHICLE_CLASSES = [0, 1, 2]

print("[INFO] Memuat model dan konfigurasi...")

try:
    classifier_data = joblib.load(CLASSIFIER_PATH)
    queue_classifier = classifier_data['model']
    scaler = classifier_data['scaler']
    print("[INFO] Model klasifikasi berhasil dimuat.")
except FileNotFoundError:
    print(f"[ERROR] File '{CLASSIFIER_PATH}' tidak ditemukan. Jalankan skrip 2d terlebih dahulu.")
    exit()

model = YOLO(YOLO_MODEL_PATH)
print(f"[INFO] Model YOLO '{YOLO_MODEL_PATH}' berhasil dimuat.")

roi_mask = cv2.imread(ROI_MASK_PATH, cv2.IMREAD_GRAYSCALE)
if roi_mask is None:
    print(f"[ERROR] Gagal memuat ROI mask dari '{ROI_MASK_PATH}'.")
    exit()
roi_area_px = np.sum(roi_mask > 0)
roi_area_m2 = roi_area_px / (PPM ** 2)

tracker = sv.ByteTrack()
annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(VIDEO_PATH)
original_fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(original_fps / TARGET_FPS)

track_history = defaultdict(lambda: [])
all_tracked_data = []
frame_counter = 0

print("[INFO] Memulai pemrosesan video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_counter % frame_interval != 0:
        frame_counter += 1
        continue

    masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
    results = model(masked_frame, classes=VEHICLE_CLASSES, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    num_vehicles = len(detections)
    if num_vehicles > 0:
        density = num_vehicles / roi_area_m2
        occupancy = np.sum(detections.box_area) / roi_area_px
    else:
        density = 0.0
        occupancy = 0.0
        
    features = np.array([[density, occupancy]])
    features_scaled = scaler.transform(features)
    queue_state = queue_classifier.predict(features_scaled)[0]

    if queue_state == 'FREE_FLOW':
        print(f"Frame {frame_counter}: FREE_FLOW")
        cv2.putText(frame, "STATUS: FREE FLOW", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Tracking", frame)
    
    else:
        print(f"Frame {frame_counter}: QUEUE DETECTED - Running Tracker")

        tracked_detections = tracker.update_with_detections(detections)
        
        current_frame_data = []
        for i in range(len(tracked_detections)):
            bbox = tracked_detections.xyxy[i]
            track_id = tracked_detections.tracker_id[i]
            x1, y1, x2, y2 = map(int, bbox)
            centroid = sv.Point(x=(x1 + x2) / 2, y=(y1 + y2) / 2)
            
            speed_kmh = 0
            history = track_history[track_id]
            history.append(centroid)
            
            if len(history) > 1:
                distance_px = np.linalg.norm(np.array([history[-1].x, history[-1].y]) - np.array([history[-2].x, history[-2].y]))
                distance_m = distance_px / PPM
                time_s = frame_interval / original_fps
                speed_ms = distance_m / time_s
                speed_kmh = speed_ms * 3.6

            is_queuing = 1 if speed_kmh < 5.0 else 0

            current_frame_data.append({
                "frame_id": frame_counter,
                "track_id": track_id,
                "bbox_x": x1, "bbox_y": y1, "bbox_w": x2 - x1, "bbox_h": y2 - y1,
                "centroid_x": centroid.x, "centroid_y": centroid.y,
                "speed_kmh": round(speed_kmh, 2),
                "is_queuing": is_queuing
            })
        
        all_tracked_data.extend(current_frame_data)
        
        annotated_frame = annotator.annotate(scene=frame.copy(), detections=tracked_detections)

        if current_frame_data:
            labels = [
                f"ID:{d['track_id']} S:{d['speed_kmh']:.1f}"
                for d in current_frame_data
            ]
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)

        cv2.putText(annotated_frame, "STATUS: QUEUE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Tracking", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
        
    frame_counter += 1

cap.release()
cv2.destroyAllWindows()

if all_tracked_data:
    df = pd.DataFrame(all_tracked_data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[INFO] Proses selesai. Data tracking tersimpan di '{OUTPUT_CSV}'.")
else:
    print("\n[INFO] Proses selesai. Tidak ada data tracking yang dihasilkan.")
