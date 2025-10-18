import cv2
import pandas as pd
import os

frames_dir = "processed_frames"
features_csv = "features.csv"
output_csv = "features_labeled.csv"

df = pd.read_csv(features_csv)
if "label" not in df.columns:
    df["label"] = "UNLABELED"

print("[INFO] Mulai proses labeling frame.")
print("Tekan 'q' untuk QUEUE, 'f' untuk FREE_FLOW, 's' untuk skip, atau 'ESC' untuk keluar.")

for i, row in df.iterrows():
    frame_id = int(row["frame_id"])
    frame_name = f"frame_{frame_id:05d}.png"
    frame_path = os.path.join(frames_dir, frame_name)

    if not os.path.exists(frame_path):
        print(f"[WARNING] Frame {frame_name} tidak ditemukan. Lewati.")
        continue

    frame = cv2.imread(frame_path)
    cv2.putText(frame, f"Frame {frame_id}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Current Label: {row['label']}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Labeling Tool", frame)
    key = cv2.waitKey(0) & 0xFF

    if key == 27:  # ESC keluar
        break
    elif key == ord("q"):
        df.at[i, "label"] = "QUEUE"
    elif key == ord("f"):
        df.at[i, "label"] = "FREE_FLOW"
    elif key == ord("s"):
        continue

    print(f"[INFO] Frame {frame_id} → Label: {df.at[i, 'label']}")

cv2.destroyAllWindows()

df.to_csv(output_csv, index=False)
print(f"[INFO] Labeling selesai. Data tersimpan di {output_csv}")
