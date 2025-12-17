# 🎵 Gesture-Controlled Spotify AI

By Siddharth Inamdar

---

## 🧩 Overview
A real-time AI system that controls Spotify using hand gestures. Built with OpenCV, MediaPipe, and a KNN classifier for fast, reliable predictions.

- Dataset: 2115 manually captured samples
- Accuracy: 94.9% (KNN)
- Features: 63 coordinates per frame (21 landmarks × x,y,z)

---

## 🎥 Demo
🎬 Watch Demo Video:I have added the video in "Uploaded" folder. Please download the video to watch.
This demo showcases full control of Spotify through gestures such as play/pause, like, volume control, and track switching.

---

## 🖐️ Supported Gestures
| Gesture | Action |
|---|---|
| 👍 Thumbs Up | Play |
| 🖐️ Open Palm | Pause |
| 🤘 Rock On | Like Song |
| ☝️ Index Swipe Up | Volume Up |
| 👇 Index Swipe Down | Volume Down |
| 👉 Swipe Right | Next Song |
| 👈 Swipe Left | Previous Song |

---

## ⚙️ Tech Stack
- Python 3.11+
- OpenCV — Captures real-time frames
- MediaPipe — Detects 21 3D hand landmarks
- Scikit-learn (KNN) — Gesture classification
- PyAutoGUI — Automates Spotify keyboard shortcuts
- PyGetWindow (optional) — Window focus management
- Pickle — Saves trained ML model

---

## 🧠 Model Summary
- Dataset: 2115 manually labeled samples
- Model: KNN Classifier
- Accuracy: 94.9%
- Features: 63 numerical coordinates (21 landmarks × 3 axes)

---

## 🚀 How It Works
1) Run `collect_data.py` → Collect gesture data
2) Run `train_model.py` → Train KNN model
3) Run `main.py` → Control Spotify in real-time

Each frame is converted into 63 numerical coordinates by MediaPipe, classified by the ML model, and translated into Spotify actions.

---

## 🔧 Project Architecture
```
Camera → MediaPipe (21 Landmarks) → 63 Features → KNN Model → Action → Spotify
```

---

## 📦 Project Structure
```
Gesture Controlled Spotify/
├── collect_data.py           # Dataset collection
├── train_model.py            # Model training
├── main.py                   # Real-time gesture control
├── gesture_model.pkl         # Trained ML model
├── gestures.csv              # Dataset (generated)
├── README.md                 # Portfolio documentation
└── Problem Solving and Learnings/
    ├── problems_and_solutions.txt
    └── project_learnings.txt
```

---

## ❤️ Key Learnings
- I learned how to extract and use real-time hand landmarks with MediaPipe.
- I understood how classical ML (KNN) can outperform heavier models for lightweight, real-time tasks.
- I discovered the importance of buffer and cooldown logic for gesture stability.
- I learned to optimize FPS for real-time AI systems without GPU dependency.

---

## 📈 Future Enhancements
- Add minimize/maximize gestures 🪟
- Integrate TensorFlow for CNN-based deep learning recognition 🧠
- Add dashboard visualization for gesture detection accuracy 📊

---

## 🏁 Final Details
- Dataset: 2115 samples
- Accuracy: 94.9%
- Language: Python
- Author: Siddharth Inamdar
- Completion: November 2025

---

## 📜 License
Open-source project — Free to use for educational purposes. Attribution appreciated.
