# 🚗 Driver Drowsiness Prediction using Haar Cascade and MSDA

A real-time driver monitoring system that detects **drowsiness** and nearby **objects (people 🚶, vehicles 🚗)** using a **dual-camera setup** with voice alerts 🔊.

---

## ✨ Features

- 👀 **Eye & Mouth Detection** using Haar Cascades
- 🧠 **Micro-saccadic Drift Analysis** with Optical Flow (PyrLK)
- 🎯 **YOLOv4-Tiny** for fast object detection
- 🔊 **Voice Alerts** using `gTTS` and `pygame`
- 🎥 **Dual Camera Setup** — internal for driver, external for surroundings

---

## 🧰 Requirements

- Python 3.7+
- `opencv-python`
- `numpy`
- `pygame`
- `gTTS`

📦 Install all dependencies:

```bash
pip install opencv-python numpy pygame gTTS

