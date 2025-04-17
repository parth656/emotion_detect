
# 🎭 Real-Time Emotion Detection System using OpenCV & PyQt5

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-brightgreen?style=for-the-badge&logo=opencv)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-red?style=for-the-badge&logo=qt)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)
![Emotion AI](https://img.shields.io/badge/Emotion%20AI-Simulation-purple?style=for-the-badge&logo=ai)

> 👁️‍🗨️ Detect emotions from live webcam feed and receive real-time mental wellness recommendations — all within a modern, interactive desktop GUI!

---

## 🌟 Features

- 📸 Real-time webcam capture using OpenCV
- 🧠 Simulated emotion recognition: Happy, Sad, Angry, Fear, Disgust, Surprise, Neutral
- 💬 Personalized wellness suggestions for each emotion
- 🚨 Auto fallback to demo mode if no camera is detected
- 🎨 Beautiful PyQt5 interface with color-coded feedback
- 🔄 Live frame rendering & dynamic emotion updates every few seconds

---

## 🚀 Quick Preview

> Want to see it in action?



---

## 🛠️ Getting Started

### Prerequisites

Ensure Python 3.8+ is installed. Then install dependencies:

```bash
pip install opencv-python PyQt5 numpy
```

### Run the Application

```bash
git clone https://github.com/your-username/emotion-detection-gui.git
cd emotion-detection-gui
python emotion_detct_final.py
```

---
<!-- 
## 🖼 UI Overview

| Main Window | Real-Time Emotion | Smart Suggestion |
|-------------|-------------------|------------------|
| ![UI](https://via.placeholder.com/250x150?text=Main+UI) | ![Detection](https://via.placeholder.com/250x150?text=Emotion+Detection) | ![Advice](https://via.placeholder.com/250x150?text=Wellness+Tip) |

--- -->

## 📂 Project Structure

```
emotion-detection-gui/
├── app.py                          # Main GUI & logic
├── haarcascade_frontalface_default.xml  # Face detection model
└── README.md
```

---

## 💡 Behind the Scenes

- Haar Cascades detect faces from webcam frames
- Every few seconds, a random emotion is selected (simulated for now)
- Recommendations rotate based on detected emotion
- Includes multiple camera access methods: DirectShow, V4L2, GStreamer

---

## 🎯 Future Upgrades

- 🤖 Replace random emotions with real ML model (CNN / FER+ / DeepFace)
- 🎙️ Add text-to-speech for recommendations
- 📊 Include emotion history timeline or logs
- 🌈 Add animated Lottie or SVG icons beside emotion text
- ☁️ Package into a cross-platform executable

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Got ideas to improve detection or the GUI? Found a bug?  
Pull requests and stars are always welcome! ⭐

---

## 👤 Author

**Parth Bijpuriya**  
💼 [Portfolio](https://superb-gingersnap-2a78b0.netlify.app/)  
📬 [Email](mailto:parthbijpuriya416.com)  
<!-- 🐦 [Twitter](https://twitter.com/yourhandle) -->

---

### ✨ If you found this helpful or cool, give it a ⭐ and share it!
