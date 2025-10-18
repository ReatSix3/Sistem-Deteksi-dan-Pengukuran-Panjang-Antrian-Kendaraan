import cv2
import numpy as np

NAMA_FILE_GAMBAR = 'annotate0.png'

src_pts = np.float32([
    [510, 209],
    [680, 202],
    [1209, 551],
    [412, 583]
])

real_world_length_m = 50.0
real_world_width_m = 10.0

PPM = 20.0 

bev_width_px = int(real_world_width_m * PPM)
bev_height_px = int(real_world_length_m * PPM)

dst_pts = np.float32([
    [0, 0],
    [bev_width_px, 0],
    [bev_width_px, bev_height_px],
    [0, bev_height_px]
])

image = cv2.imread(NAMA_FILE_GAMBAR)

if image is None:
    print(f"Error: Gambar tidak ditemukan di '{NAMA_FILE_GAMBAR}'")
    print("Pastikan nama file sudah benar dan file berada di folder yang sama.")
else:
    cv2.polylines(image, [np.int32(src_pts)], isClosed=True, color=(0, 255, 0), thickness=3)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    bev_image = cv2.warpPerspective(image, M, (bev_width_px, bev_height_px))

    cv2.imshow("Gambar Asli CCTV dengan ROI", image)
    cv2.imshow(f"Hasil Bird's-Eye View ({bev_width_px}x{bev_height_px}px)", bev_image)

    cv2.imwrite("bev_image.png", bev_image)
    cv2.imwrite("image_ROI.png", image)

    print("="*50)
    print("Proses Kalibrasi Selesai!")
    print(f"Ukuran Gambar BEV: {bev_width_px} x {bev_height_px} piksel")
    print(f"Skala (PPM): {PPM} piksel per meter")
    print("\nTekan tombol apa saja untuk keluar.")
    print("="*50)

    cv2.waitKey(0)
    cv2.destroyAllWindows()