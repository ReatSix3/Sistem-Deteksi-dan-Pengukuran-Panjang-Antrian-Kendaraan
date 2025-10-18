import cv2
import os
import numpy as np

NAMA_FILE_VIDEO = "short_1.mp4"
NAMA_FILE_MASK = "roi_mask.png"

FOLDER_OUTPUT = "processed_frames"

FPS_VIDEO_ASLI = 30
FPS_YANG_DIINGINKAN = 5

if not os.path.exists(FOLDER_OUTPUT):
    os.makedirs(FOLDER_OUTPUT)
    print(f"Folder '{FOLDER_OUTPUT}' berhasil dibuat.")

cap = cv2.VideoCapture(NAMA_FILE_VIDEO)
if not cap.isOpened():
    print(f"Error: Tidak bisa membuka video '{NAMA_FILE_VIDEO}'")
    exit()

mask = cv2.imread(NAMA_FILE_MASK, 0)
if mask is None:
    print(f"Error: Tidak bisa membaca mask '{NAMA_FILE_MASK}'")
    exit()

frame_interval = int(FPS_VIDEO_ASLI / FPS_YANG_DIINGINKAN)
frame_count = 0
saved_frame_count = 0

print("\nMemulai proses ekstraksi frame...")
while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_count % frame_interval == 0:
        processed_frame = cv2.bitwise_and(frame, frame, mask=mask)

        nama_file_simpan = os.path.join(FOLDER_OUTPUT, f"frame_{saved_frame_count:05d}.png")
        cv2.imwrite(nama_file_simpan, processed_frame)

        saved_frame_count += 1
        print(f"Menyimpan frame ke-{saved_frame_count}: {nama_file_simpan}")

    frame_count += 1

cap.release()

print("Proses Selesai!")
print(f"Total frame yang disimpan: {saved_frame_count}")
print(f"Hasil disimpan di folder: '{FOLDER_OUTPUT}'")