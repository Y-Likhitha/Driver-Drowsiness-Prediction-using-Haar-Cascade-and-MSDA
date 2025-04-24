# 🚗 Driver Drowsiness Prediction using Haar Cascade and MSDA

A real-time driver monitoring system that detects **drowsiness** and nearby **objects (people 🚶, vehicles 🚗)** using a **dual-camera setup** with voice alerts 🔊.

---

## ✨ Features

- 👀 **Eye & Mouth Detection** using Haar Cascades
- 🧠 **Micro-saccadic Drift Analysis** with Optical Flow (PyrLK)
- 🎯 **YOLOv4-Tiny** for fast object detection
- 🔊 **Voice Alerts** using `gTTS` and `pygame`
- 🎥 **Dual Camera Setup** — internal for driver, external for surroundings
  - **Internal Camera** for driver monitoring  
  - **External Camera** for obstacle & pedestrian detection  
  - 📱 Supports **DroidCam** as an external camera source

---

## 🧰 Requirements

- Python 3.7+
- `opencv-python`
- `numpy`
- `pygame`
- `gTTS`

📦 Install all dependencies:
pip install opencv-python numpy pygame gTTS


## 🔗 Download Required Files

This project requires large files that are not stored in the repository due to GitHub size limits.

Please download the following manually from Google Drive:

➡️ [Download YOLOv4-Tiny Weights & Config](https://drive.google.com/drive/folders/1ACHpGLPvmv-3LK71tVsvh71_pcRG0NCo?usp=sharing)
➡️ [Download Dataset](https://drive.google.com/drive/folders/1yU7N33xs394ED3IjRl3WU8EGubprpWXB?usp=sharing)

**After downloading**, place them in the appropriate folders:

- `yolov4.weights` → `yolo/`
- Dataset folders → `Dataset/`
- Audio files → `alarm/`

Then, you're ready to run the project! 🚀
