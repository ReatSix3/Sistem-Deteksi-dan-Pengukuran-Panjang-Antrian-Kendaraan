import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

RAW_LENGTHS_CSV = "queue_lengths_raw.csv"
FINAL_OUTPUT_CSV = "queue_lengths_final.csv"

class QueueLengthKalmanFilter:
    def __init__(self, dt=0.2, r_noise=1.5, q_noise=0.1):
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        self.kf.x = np.array([0., 0.])
        self.kf.F = np.array([[1., dt], [0., 1.]])
        self.kf.H = np.array([[1., 0.]])
        self.kf.P *= 1000.
        self.kf.R = np.array([[r_noise]])
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=q_noise)

    def update(self, measurement):
        self.kf.predict()
        self.kf.update(measurement)
        return self.kf.x[0]

def reject_outliers(measurements, window=10, sigma=3):
    if len(measurements) < window:
        return measurements[-1]
    recent = np.array(measurements[-window:])
    mean = np.mean(recent)
    std = np.std(recent)
    current = measurements[-1]
    if std > 0 and abs(current - mean) > sigma * std:
        return mean
    else:
        return current

def smooth_ema(new_value, prev_smoothed, alpha=0.3):
    if prev_smoothed is None:
        return new_value
    return alpha * new_value + (1 - alpha) * prev_smoothed

try:
    df = pd.read_csv(RAW_LENGTHS_CSV)
except FileNotFoundError:
    exit()

kalman_filter = QueueLengthKalmanFilter()
measurements_history = []
smoothed_results = []
prev_ema_value = None

if not df.empty:
    raw_lengths = df['length_hybrid_m'].values
    for i, raw_length in enumerate(raw_lengths):
        measurements_history.append(raw_length)
        filtered_length = reject_outliers(measurements_history)
        kalman_output = kalman_filter.update(filtered_length)
        final_length = smooth_ema(kalman_output, prev_ema_value)
        prev_ema_value = final_length
        final_length = max(0, final_length)
        smoothed_results.append({
            "frame_id": df['frame_id'].iloc[i],
            "raw_hybrid_m": raw_length,
            "smoothed_length_m": round(final_length, 2)
        })

if smoothed_results:
    results_df = pd.DataFrame(smoothed_results)
    results_df.to_csv(FINAL_OUTPUT_CSV, index=False)
