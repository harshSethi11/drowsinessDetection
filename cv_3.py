"""
Real-time Drowsiness Detector (EAR-based)
- Uses either dlib (68-point landmarks) or MediaPipe FaceMesh.
- Non-blocking TTS on Windows via SAPI5 (falls back to a beep).
- 'C' to quick-calibrate threshold using your "eyes open" baseline.
- 'Q' or ESC to quit.
"""

import os
import time
import math
from collections import deque
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

# Optional imports (loaded if available) 
try:
    import dlib
    _HAS_DLIB = True
except Exception:
    _HAS_DLIB = False

try:
    import mediapipe as mp
    _HAS_MP = True
except Exception:
    _HAS_MP = False


# Configuration 

CAM_INDEX = 0                
FRAME_WIDTH = 960           

EAR_THRESHOLD = 0.25         
CONSEC_FRAMES = 15           
SMOOTHING_WINDOW = 5        
ALERT_COOLDOWN_SEC = 2.0    

# dlib 68-landmarks model path (only used if _HAS_DLIB True)
DLIB_LANDMARKS_PATH = r"C:\Users\Harsh Sethi\Downloads\shape_predictor_68_face_landmarks.dat"

# HUD colors (BGR)
OK_COLOR = (60, 220, 60)
WARN_COLOR = (40, 140, 255)
ALERT_COLOR = (36, 36, 255)

WINDOW_NAME = "Real-time Drowsiness Detector (Q=quit, C=calibrate)"

#  Visualization style 
DRAW_STYLE = "dots"   # options: "dots" or "lines"
DOT_RADIUS = 2        # pixel radius for dots



# TTS: non-blocking SAPI5

_TTS = None
_TTS_OK = False
_TTS_INIT_LOCK = threading.Lock()

def tts_init() -> None:
    """Initialize a non-blocking SAPI5 TTS engine ."""
    global _TTS, _TTS_OK
    with _TTS_INIT_LOCK:
        if _TTS_OK:
            return
        try:
            import pyttsx3  # Imported here so the script runs even if not installed
            _TTS = pyttsx3.init(driverName='sapi5')
            _TTS.setProperty('rate', 175)
            _TTS.setProperty('volume', 1.0)
            _TTS.startLoop(False)  # Non-blocking event loop
            _TTS_OK = True
        except Exception:
            _TTS = None
            _TTS_OK = False

def tts_say(msg: str) -> None:
    """Queue a message to speak; safe to call frequently."""
    if _TTS_OK and _TTS:
        try:
            _TTS.say(msg)
        except Exception:
            pass  # Ignore TTS hiccups

def tts_iterate() -> None:
    """Pump the TTS event loop once; call this every frame."""
    if _TTS_OK and _TTS:
        try:
            _TTS.iterate()
        except Exception:
            pass

def tts_shutdown() -> None:
    """Stop the TTS loop cleanly on exit."""
    if _TTS_OK and _TTS:
        try:
            _TTS.endLoop()
        except Exception:
            pass

def speak_or_beep(message: str = "Alert! Drowsiness detected.") -> None:
    """High-level alert: try TTS, else beep."""
    if _TTS_OK:
        tts_say(message)
    else:
        try:
            import winsound
            winsound.Beep(1000, 700)
        except Exception:
            pass  # Silent fallback if can't beep


# Utility functions

def safe_path(p: str) -> str:
    """Normalize a Windows path to avoid unicode-escape surprises."""
    return p.replace("\\", "/")

def euclid(p1, p2) -> float:
    """2D Euclidean distance."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    eye_pts must be ordered as: [p1, p2, p3, p4, p5, p6]
    """
    A = euclid(eye_pts[1], eye_pts[5])
    B = euclid(eye_pts[2], eye_pts[4])
    C = euclid(eye_pts[0], eye_pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def order_for_ear(six_points: np.ndarray) -> np.ndarray:
    """Already ordered at source; kept for clarity / future swaps."""
    return six_points


# Dlib backend

class DlibBackend:
    """Dlib 68-landmarks backend for computing EAR."""

    # 68-landmarks indices (right eye: 42-47, left eye: 36-41)
    LEFT = list(range(36, 42))
    RIGHT = list(range(42, 48))

    def __init__(self, model_path: str):
        model_path = safe_path(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"dlib model not found: {model_path}")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)

    def get_ear(self, gray: np.ndarray) -> Optional[Tuple[Tuple[float, float], np.ndarray, Tuple[list, list]]]:
        """
        Returns:
            ( (left_ear, right_ear), all_pts, (LEFT_idx_list, RIGHT_idx_list) )
        or:
            None if no face detected.
        """
        faces = self.detector(gray, 0)
        if len(faces) == 0:
            return None  
        face = faces[0]
        shape = self.predictor(gray, face)
        pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float32)

        left_eye = pts[self.LEFT]
        right_eye = pts[self.RIGHT]

        left_ear = eye_aspect_ratio(order_for_ear(left_eye))
        right_ear = eye_aspect_ratio(order_for_ear(right_eye))

        return (left_ear, right_ear), pts, (self.LEFT, self.RIGHT)

# MediaPipe backend

class MediaPipeBackend:
    """MediaPipe FaceMesh backend approximating EAR using canonical eye points."""

    # EAR order [p1, p2, p3, p4, p5, p6]
    LEFT = [33, 160, 158, 133, 153, 144]
    RIGHT = [362, 385, 387, 263, 373, 380]

    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_ear(self, bgr: np.ndarray) -> Optional[Tuple[Tuple[float, float], Tuple[np.ndarray, np.ndarray], Tuple[list, list]]]:
        """
        Returns:
            ( (left_ear, right_ear), (left_pts, right_pts), (LEFT_idx_list, RIGHT_idx_list) )
        or:
            None if no face detected.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None

        h, w = bgr.shape[:2]
        lm = res.multi_face_landmarks[0].landmark

        def get_xy(idx):
            p = lm[idx]
            return (p.x * w, p.y * h)

        left_pts = np.array([get_xy(i) for i in self.LEFT], dtype=np.float32)
        right_pts = np.array([get_xy(i) for i in self.RIGHT], dtype=np.float32)

        left_ear = eye_aspect_ratio(order_for_ear(left_pts))
        right_ear = eye_aspect_ratio(order_for_ear(right_pts))

        return (left_ear, right_ear), (left_pts, right_pts), (self.LEFT, self.RIGHT)

def draw_eye_overlay(img, points, style="dots", color=(0, 255, 0)):
    """
    Draw either small dots or connecting lines for a sequence of eye points.
    `points` should be an (N, 2) array-like of (x, y) coordinates.
    """
    pts = np.asarray(points, dtype=np.int32)

    if style == "lines":
        # draw a closed loop
        for i in range(len(pts)):
            a = tuple(pts[i])
            b = tuple(pts[(i + 1) % len(pts)])
            cv2.line(img, a, b, color, 1, cv2.LINE_AA)
    else:
        # dots (filled circles)
        for p in pts:
            cv2.circle(img, tuple(p), DOT_RADIUS, color, -1, lineType=cv2.LINE_AA)



# Main app

def main() -> None:
    # Prefer dlib if available and model exists; else fall back to MediaPipe.
    backend = None
    using_dlib = False

    if _HAS_DLIB and os.path.exists(safe_path(DLIB_LANDMARKS_PATH)):
        try:
            backend = DlibBackend(DLIB_LANDMARKS_PATH)
            using_dlib = True
            print("[INFO] Using dlib 68-landmarks backend")
        except Exception as e:
            print(f"[WARN] dlib init failed ({e}); falling back to MediaPipe...")

    if backend is None:
        if not _HAS_MP:
            print("[ERROR] Neither dlib nor mediapipe available. Install one of them:")
            print("  pip install mediapipe opencv-python numpy")
            return
        backend = MediaPipeBackend()
        print("[INFO] Using MediaPipe FaceMesh backend")

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f("[ERROR] Cannot open camera index {CAM_INDEX}"))
        return

    # State
    last_alert_t = 0.0
    below_counter = 0
    ear_buffer: deque = deque(maxlen=SMOOTHING_WINDOW)
    ear_thresh = EAR_THRESHOLD
    calib_open_baseline: Optional[float] = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    tts_init()

    try:
        while True:
            tts_iterate()  # keep TTS responsive

            ok, frame = cap.read()
            if not ok:
                print("[WARN] Empty frame; continuing...")
                continue

            # Resize for speed / consistent scaling
            if FRAME_WIDTH is not None:
                scale = FRAME_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, (FRAME_WIDTH, int(frame.shape[0] * scale)))

            draw = frame.copy()

            # Compute EAR via backend
            left_right_ear = None
            if using_dlib:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                res = backend.get_ear(gray)
                if res is not None:
                    left_right_ear, pts, loops = res
                    # visualize eye loops
                    for loop in loops:
                        eye_pts = pts[loop]
                        draw_eye_overlay(draw, eye_pts, style=DRAW_STYLE, color=(0, 255, 0))

            else:
                res = backend.get_ear(frame)
                if res is not None:
                    left_right_ear, (lpts, rpts), _ = res
                    # visualize approximate eye contours
                    draw_eye_overlay(draw, lpts, style=DRAW_STYLE, color=(0, 255, 0))
                    draw_eye_overlay(draw, rpts, style=DRAW_STYLE, color=(0, 255, 0))

            # Smooth EAR with moving average 
            smooth_ear: Optional[float] = None
            if left_right_ear is not None:
                ear = float(np.mean(left_right_ear))
                ear_buffer.append(ear)
                smooth_ear = float(np.mean(ear_buffer)) if ear_buffer else None

            # HUD 
            h, w = draw.shape[:2]
            cv2.rectangle(draw, (0, 0), (w, 80), (0, 0, 0), -1)

            if smooth_ear is None:
                status = "NO FACE"
                color = WARN_COLOR
            else:
                status = f"EAR: {smooth_ear:.3f}  (thr {ear_thresh:.3f})"
                color = WARN_COLOR if smooth_ear < ear_thresh else OK_COLOR

            cv2.putText(draw, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            cv2.putText(
                draw,
                "[Q] Quit   [C] Calibrate   Frames<Thr: %d/%d" % (below_counter, CONSEC_FRAMES),
                (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # Logic: consecutive frames below threshold => alert
            if smooth_ear is not None:
                if smooth_ear < ear_thresh:
                    below_counter += 1
                else:
                    below_counter = 0

                if below_counter >= CONSEC_FRAMES:
                    now = time.time()
                    if now - last_alert_t >= ALERT_COOLDOWN_SEC:
                        last_alert_t = now
                        cv2.putText(
                            draw,
                            "DROWSINESS DETECTED!",
                            (12, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            ALERT_COLOR,
                            3,
                            cv2.LINE_AA,
                        )
                        speak_or_beep("Alert! Drowsiness detected.")

            cv2.imshow(WINDOW_NAME, draw)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):  # 'q' or ESC
                break

            # Quick calibration: press 'C' while eyes are naturally open
            if key in (ord('c'), ord('C')):
                tmp_vals = []
                t0 = time.time()
                while time.time() - t0 < 0.8:  # ~0.8s sample window
                    ok2, fr2 = cap.read()
                    if not ok2:
                        continue
                    if FRAME_WIDTH is not None:
                        scale2 = FRAME_WIDTH / fr2.shape[1]
                        fr2 = cv2.resize(fr2, (FRAME_WIDTH, int(fr2.shape[0] * scale2)))
                    if using_dlib:
                        r2 = backend.get_ear(cv2.cvtColor(fr2, cv2.COLOR_BGR2GRAY))
                        if r2 is not None and r2[0] is not None:
                            tmp_vals.append(float(np.mean(r2[0])))
                    else:
                        r2 = backend.get_ear(fr2)
                        if r2 is not None and r2[0] is not None:
                            tmp_vals.append(float(np.mean(r2[0])))

                if tmp_vals:
                    calib_open_baseline = float(np.median(tmp_vals))
                    # Typical closed-eye EAR is ~35–45% of open.
                    # Using ~72% of open as threshold is a valid safety margin.
                    ear_thresh = max(0.15, calib_open_baseline * 0.72)
                    print(f"[CALIB] open_EAR≈{calib_open_baseline:.3f}  -> threshold={ear_thresh:.3f}")
                    below_counter = 0
                    ear_buffer.clear()

    except KeyboardInterrupt:
        pass
    finally:
        # release resources cleanly
        cap.release()
        cv2.destroyAllWindows()
        tts_shutdown()


if __name__ == "__main__":
    main()
