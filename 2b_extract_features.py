import pandas as pd
import cv2
import numpy as np

detections_csv = "detections.csv"
roi_mask_path = "roi_mask.png"
output_csv = "features.csv"
PPM = 20.0

detections = pd.read_csv(detections_csv)
roi_mask = cv2.imread(roi_mask_path, cv2.IMREAD_GRAYSCALE)
roi_area_px = np.sum(roi_mask > 0)
roi_area_m2 = roi_area_px / (PPM**2)

features = []
for frame_id, group in detections.groupby("frame_id"):
    num_vehicles = len(group)
    vehicle_density_metric = num_vehicles / roi_area_m2
    total_vehicle_area = np.sum(group["bbox_w"] * group["bbox_h"])
    occupancy_rate = total_vehicle_area / roi_area_px
    features.append({
        "frame_id": frame_id,
        "density": round(vehicle_density_metric, 6),
        "occupancy": round(occupancy_rate, 6),
        "speed": None,
        "clustering": None,
        "label": "UNLABELED"
    })

features_df = pd.DataFrame(features)
features_df.to_csv(output_csv, index=False)
print(f"[INFO] Fitur tersimpan di {output_csv}")
