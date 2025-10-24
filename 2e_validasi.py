import cv2
import pandas as pd
import os

FRAME_ID_YANG_DIPERIKSA = 1542

frames_dir = "processed_frames"
detections_csv = "detections.csv"
features_csv = "features.csv"
features_labeled_csv = "features_labeled.csv"

frame_name = f"frame_{FRAME_ID_YANG_DIPERIKSA:05d}.png"
frame_path = os.path.join(frames_dir, frame_name)

if not os.path.exists(frame_path):
    print(f"Error: Frame {frame_name} tidak ditemukan di folder '{frames_dir}'")
    exit()

frame = cv2.imread(frame_path)
print(f"Memvalidasi Frame ID: {FRAME_ID_YANG_DIPERIKSA}...")

try:
    df_detections = pd.read_csv(detections_csv)
    frame_detections = df_detections[df_detections["frame_id"] == FRAME_ID_YANG_DIPERIKSA]
    print(f"\n[Detections.csv] Menemukan {len(frame_detections)} kendaraan di frame ini.")
    for _, row in frame_detections.iterrows():
        x, y, w, h = int(row['bbox_x']), int(row['bbox_y']), int(row['bbox_w']), int(row['bbox_h'])
        label = row['class']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
except FileNotFoundError:
    print(f"Warning: File {detections_csv} tidak ditemukan. Validasi deteksi dilewati.")

try:
    df_features = pd.read_csv(features_csv)
    frame_features = df_features[df_features["frame_id"] == FRAME_ID_YANG_DIPERIKSA].iloc[0]
    density = frame_features['density']
    occupancy = frame_features['occupancy']
    print(f"[Features.csv] Density: {density:.4f}, Occupancy: {occupancy:.4f}")
    cv2.putText(frame, f"Density: {density:.4f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Occupancy: {occupancy:.4f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
except (FileNotFoundError, IndexError):
    print(f"Warning: Data untuk frame {FRAME_ID_YANG_DIPERIKSA} tidak ditemukan di {features_csv}.")

try:
    df_labeled = pd.read_csv(features_labeled_csv)
    frame_labeled = df_labeled[df_labeled["frame_id"] == FRAME_ID_YANG_DIPERIKSA].iloc[0]
    label = frame_labeled['label']
    print(f"[Features_labeled.csv] Label: {label}")
    cv2.putText(frame, f"Label: {label}", (frame.shape[1] - 300, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
except (FileNotFoundError, IndexError):
    print(f"Warning: Data untuk frame {FRAME_ID_YANG_DIPERIKSA} tidak ditemukan di {features_labeled_csv}.")

cv2.imshow(f"Validator untuk Frame {FRAME_ID_YANG_DIPERIKSA}", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
