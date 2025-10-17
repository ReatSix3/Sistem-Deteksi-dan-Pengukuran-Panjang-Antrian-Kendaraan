# 🚦 Traffic Queue Length Measurement System

This repository contains an **end-to-end traffic queue detection and measurement pipeline** built for real-time CCTV applications.  
It combines deep learning detection (YOLOv8), tracking (ByteTrack), Bird’s Eye View (BEV) transformation, and hybrid measurement methods to **estimate queue length in meters** with high accuracy.

---

## 📌 Features

- 🧠 **Vehicle Detection & Tracking** — YOLOv8 + ByteTrack for robust multi-object tracking  
- 📐 **Camera Calibration** — BEV transformation using homography for real-world mapping  
- 📏 **Queue Length Measurement** — Hybrid Direct + Density-based measurement  
- 🧮 **Temporal Smoothing** — Kalman Filter + EMA for stable output  
- 📊 **Performance Evaluation** — MAE, RMSE, MAPE, R², Accuracy, F1 score  
- 🧪 **Ablation Study & Error Analysis** — To evaluate each component’s contribution

---

## 📂 Project Structure

