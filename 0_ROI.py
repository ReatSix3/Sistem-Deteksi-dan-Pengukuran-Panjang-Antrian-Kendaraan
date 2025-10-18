import cv2
import numpy as np

NAMA_FILE_GAMBAR = 'annotate0.png'

src_pts = np.array([
    [510, 209],
    [680, 202],
    [1209, 551],
    [412, 583]
], dtype=np.int32)

image = cv2.imread(NAMA_FILE_GAMBAR)

if image is None:
    print(f"Error: Gambar tidak ditemukan di '{NAMA_FILE_GAMBAR}'")
else:
    height, width, _ = image.shape

    mask = np.zeros((height, width), dtype=np.uint8)

    cv2.fillPoly(mask, [src_pts], (255))

    image_with_mask_applied = cv2.bitwise_and(image, image, mask=mask)
    
    cv2.imwrite("roi_mask.png", mask)

    cv2.imshow("Gambar Asli", image)
    cv2.imshow("Gambar Mask (roi_mask.png)", mask)
    cv2.imshow("Hasil Gambar Setelah Diberi Mask", image_with_mask_applied)

    print("="*50)
    print("Mask image berhasil dibuat dan disimpan sebagai 'roi_mask.png'")
    print("Tekan tombol apa saja untuk keluar.")
    print("="*50)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()