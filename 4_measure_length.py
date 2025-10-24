import cv2
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from numpy.linalg import LinAlgError

TRACKED_DATA_CSV = "tracked_data.csv"
VIDEO_PATH = "short_1.mp4"
OUTPUT_CSV = "queue_lengths_raw.csv"

PPM = 20.0
real_world_length_m = 50.0
real_world_width_m = 10.0

src_pts = np.float32([
    [510, 209], [680, 202],
    [1209, 551], [412, 583]
])

bev_width_px = int(real_world_width_m * PPM)
bev_height_px = int(real_world_length_m * PPM)
dst_pts = np.float32([
    [0, 0], [bev_width_px, 0],
    [bev_width_px, bev_height_px], [0, bev_height_px]
])

M = cv2.getPerspectiveTransform(src_pts, dst_pts)

AVG_VEHICLE_LENGTH_M = 4.5

def measure_queue_direct(queuing_vehicles, M, PPM):
    if len(queuing_vehicles) < 2:
        return 0
    centroids_image = np.array([queuing_vehicles[['centroid_x', 'centroid_y']].values], dtype=np.float32)
    centroids_bev = cv2.perspectiveTransform(centroids_image, M)[0]
    y_coords = centroids_bev[:, 1]
    queue_start_px = np.min(y_coords)
    queue_end_px = np.max(y_coords)
    length_pixels = queue_end_px - queue_start_px
    return length_pixels / PPM

def measure_queue_density(queuing_vehicles, M, PPM, frame_shape):
    if len(queuing_vehicles) < 2:
        return 0
    density_map = np.zeros(frame_shape[:2], dtype=np.float32)
    centroids = queuing_vehicles[['centroid_x', 'centroid_y']].values
    try:
        jitter = np.random.randn(centroids.shape[0], centroids.shape[1]) * 1e-3
        centroids = centroids + jitter
        kde = gaussian_kde(centroids.T)
        h, w = frame_shape[:2]
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()])
        density_map_values = kde(grid_points).reshape(h, w)
        if density_map_values.max() > density_map_values.min():
            density_map = (density_map_values - density_map_values.min()) / (density_map_values.max() - density_map_values.min())
    except LinAlgError:
        print(f"  [WARNING] KDE failed for frame due to collinear data. Density measurement will be 0.")
        return 0
    density_bev = cv2.warpPerspective(density_map, M, (bev_width_px, bev_height_px))
    _, binary = cv2.threshold((density_bev * 255).astype(np.uint8), 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    queue_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(queue_contour) < 100:
        return 0
    rect = cv2.minAreaRect(queue_contour)
    length_pixels = max(rect[1])
    return length_pixels / PPM

try:
    df = pd.read_csv(TRACKED_DATA_CSV)
except FileNotFoundError:
    print(f"[ERROR] File '{TRACKED_DATA_CSV}' tidak ditemukan. Jalankan skrip 3 terlebih dahulu.")
    exit()

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    print("[ERROR] Tidak bisa membaca video. Menggunakan default frame size 1280x720.")
    frame_shape = (720, 1280, 3)
else:
    frame_shape = frame.shape
cap.release()
print(f"[INFO] Menggunakan ukuran frame: {frame_shape}")

print("[INFO] Memulai perhitungan panjang antrian mentah...")

results = []
grouped = df.groupby("frame_id")
all_frame_ids = sorted(df['frame_id'].unique())

for frame_id in all_frame_ids:
    group = grouped.get_group(frame_id)
    queuing_vehicles = group[group['is_queuing'] == 1]
    print(f"Frame {frame_id}: ", end="")
    if len(queuing_vehicles) < 2:
        l_direct, l_density, l_compensated = 0, 0, 0
    else:
        l_direct = measure_queue_direct(queuing_vehicles, M, PPM)
        l_density = measure_queue_density(queuing_vehicles, M, PPM, frame_shape)
        alpha = 0.7 if len(queuing_vehicles) > 5 else 0.4
        l_final = alpha * l_direct + (1 - alpha) * l_density
        l_compensated = l_final + 0.5 * AVG_VEHICLE_LENGTH_M
    print(f"Direct={l_direct:.2f}m, Density={l_density:.2f}m, Hybrid={l_compensated:.2f}m")
    results.append({
        "frame_id": frame_id,
        "length_direct_m": round(l_direct, 2),
        "length_density_m": round(l_density, 2),
        "length_hybrid_m": round(l_compensated, 2)
    })

if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[INFO] Perhitungan selesai. Hasil tersimpan di '{OUTPUT_CSV}'.")
else:
    print("\n[INFO] Perhitungan selesai. Tidak ada data yang dihasilkan.")
