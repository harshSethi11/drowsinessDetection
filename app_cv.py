import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import math
from collections import deque
import time

# ── Streamlit UI ─────────────────────────────────────────────
st.set_page_config(page_title="Drowsiness Detection", layout="centered")

st.title("😴 Real-Time Drowsiness Detection")
st.markdown("""
This app detects **drowsiness** in real time using your webcam.  
It computes the **Eye Aspect Ratio (EAR)** — when it stays below a threshold for several frames, it flags drowsiness.
""")

FRAME_WIDTH = 640
CONSEC_FRAMES = st.sidebar.slider("Frames below threshold before alert", 5, 30, 15)
EAR_THRESHOLD = st.sidebar.slider("EAR Threshold", 0.15, 0.35, 0.25, 0.01)
SMOOTHING_WINDOW = 5
DRAW_STYLE = "dots"
DOT_RADIUS = 2

st.sidebar.markdown("---")
st.sidebar.info("Press 'Stop' above the video to end the stream.")

# ── Helpers ──────────────────────────────────────────────────
def euclid(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def eye_aspect_ratio(eye_pts):
    A = euclid(eye_pts[1], eye_pts[5])
    B = euclid(eye_pts[2], eye_pts[4])
    C = euclid(eye_pts[0], eye_pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def draw_eye_overlay(img, points, style="dots", color=(0, 255, 0)):
    pts = np.asarray(points, dtype=np.int32)
    if style == "lines":
        for i in range(len(pts)):
            a = tuple(pts[i]); b = tuple(pts[(i + 1) % len(pts)])
            cv2.line(img, a, b, color, 1, cv2.LINE_AA)
    else:
        for p in pts:
            cv2.circle(img, tuple(p), DOT_RADIUS, color, -1, lineType=cv2.LINE_AA)

# EAR landmark indices (MediaPipe FaceMesh canonical)
LEFT = [33, 160, 158, 133, 153, 144]
RIGHT = [362, 385, 387, 263, 373, 380]

# ── Video transformer ────────────────────────────────────────
class DrowsinessTransformer(VideoTransformerBase):
    def __init__(self):
        # Create FaceMesh inside transformer (thread/hotreload safe)
        import mediapipe as mp
        self.mp_face = mp.solutions.face_mesh
        self.mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.ear_buffer = deque(maxlen=SMOOTHING_WINDOW)
        self.below_counter = 0
        self.last_alert_t = 0.0
        self.alert_text = ""

    def __del__(self):
        # Clean up mediapipe resources
        if hasattr(self, "mesh") and self.mesh:
            self.mesh.close()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Optional mirror for user-facing UX
        img = cv2.flip(img, 1)

        # Resize for consistent perf
        h, w = img.shape[:2]
        if FRAME_WIDTH and w != FRAME_WIDTH:
            scale = FRAME_WIDTH / w
            img = cv2.resize(img, (FRAME_WIDTH, int(h * scale)))
            h, w = img.shape[:2]

        # Face landmarks → EAR
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            res = self.mesh.process(rgb)
        except Exception:
            res = None  # safety: never crash the stream

        if not res or not res.multi_face_landmarks:
            cv2.putText(img, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        lm = res.multi_face_landmarks[0].landmark

        def get_xy(idx):
            p = lm[idx]
            return (p.x * w, p.y * h)

        left_pts = np.array([get_xy(i) for i in LEFT], dtype=np.float32)
        right_pts = np.array([get_xy(i) for i in RIGHT], dtype=np.float32)

        draw_eye_overlay(img, left_pts, style=DRAW_STYLE)
        draw_eye_overlay(img, right_pts, style=DRAW_STYLE)

        left_ear = eye_aspect_ratio(left_pts)
        right_ear = eye_aspect_ratio(right_pts)
        ear = (left_ear + right_ear) / 2.0

        self.ear_buffer.append(float(ear))
        smooth_ear = float(np.mean(self.ear_buffer)) if self.ear_buffer else 0.0

        color = (0, 255, 0)
        if smooth_ear < EAR_THRESHOLD:
            color = (0, 165, 255)
            self.below_counter += 1
        else:
            self.below_counter = 0

        if self.below_counter >= CONSEC_FRAMES:
            now = time.time()
            if now - self.last_alert_t > 2.0:
                self.last_alert_t = now
                self.alert_text = "DROWSINESS DETECTED!"
        else:
            self.alert_text = ""

        # HUD
        cv2.rectangle(img, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(img, f"EAR: {smooth_ear:.3f}  (thr {EAR_THRESHOLD:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if self.alert_text:
            cv2.putText(img, self.alert_text, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ── Start WebRTC stream ──────────────────────────────────────
webrtc_streamer(
    key="drowsiness-demo",
    video_transformer_factory=DrowsinessTransformer,
    media_stream_constraints={"video": True, "audio": False},
)

