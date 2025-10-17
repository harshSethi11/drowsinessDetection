#  Real-time Drowsiness Detection

This project detects **driver drowsiness** in real-time using a webcam feed.  
It calculates the **Eye Aspect Ratio (EAR)** using either **dlib** or **MediaPipe FaceMesh** to detect when the eyes remain closed for a certain duration.

---

##  Features
- Real-time detection using webcam
- EAR-based eye closure monitoring
- Automatic calibration for open-eye threshold
- Visual overlay (eye contour, EAR value)
- Audio alerts (TTS or beep)
- Streamlit web version for online demo
