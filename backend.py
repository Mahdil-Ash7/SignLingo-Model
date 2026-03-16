"""
04_fastapi_server.py
====================
FastAPI backend for BIM Sign Language detection.
Flutter sends cropped 16:9 JPEG frames → server runs MediaPipe + model → returns prediction.

COMPATIBLE WITH: 02_train_model.py (enhanced augmentation version)
  Model architecture : Conv1D → Conv1D → BiLSTM(128) → BiLSTM(64) → Dense → Softmax
  Feature pipeline   : identical to training (FEATURE_SIZE=204, SEQUENCE_LENGTH=8)
  Normalization      : identical to training (per-frame, per-region)
  Auto dual-hand     : detected from MediaPipe results per session (no client flag)

CONSTANTS (must always match 02_train_model.py):
  SEQUENCE_LENGTH = 8
  FEATURE_SIZE    = 204  (126 hands + 60 face + 18 pose)
  MIN_NONZERO     = 0.45
  STRIDES         = [1, 2]
  RAW_BUFFER_LEN  = 16

Install:
  pip install fastapi uvicorn mediapipe tensorflow opencv-python python-multipart numpy

Run:
  python 04_fastapi_server.py

API Docs: http://localhost:8000/docs
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import base64
import time
import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Suppress TF logs and optimize CPU threading for local machine
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# ==============================
# CONFIG — must be identical to 02_train_model.py
# ==============================
MODEL_PATH   = "bim_model/handface_pose_cnn_lstm.h5"   # saved by 02_train_model.py
LABEL_PATH   = "bim_model/labels.json"

SEQUENCE_LENGTH = 8
HAND_FEATURES   = 126
FACE_FEATURES   = 60
POSE_FEATURES   = 18
FEATURE_SIZE    = HAND_FEATURES + FACE_FEATURES + POSE_FEATURES  # 204

STRIDES          = [1, 2]
RAW_BUFFER_LEN   = max(STRIDES) * SEQUENCE_LENGTH  # 16
MIN_NONZERO      = 0.45
CONF_THRESHOLD   = 0.65   # slightly higher — deeper model is more calibrated
SMOOTHING        = 5      # wider vote window — deeper model is more stable

# Auto dual-hand detection rolling window
# Majority of last HAND_HISTORY_LEN frames having 2 hands → dual mode
HAND_HISTORY_LEN = 10

# ==============================
# MOTION-AWARE PREDICTION CONFIG
# ==============================
# Velocity = mean absolute displacement of hand wrist keypoints between frames.
# Below STATIC_THRESHOLD  → person is holding a pose  → predict fast (static mode)
# Above DYNAMIC_THRESHOLD → person is signing motion  → wait for arc to complete

STATIC_THRESHOLD  = 0.015  # wrist displacement/frame → holding still
DYNAMIC_THRESHOLD = 0.030  # wrist displacement/frame → actively moving

# Static mode: how many consecutive still frames before we emit a prediction
STATIC_CONFIRM_FRAMES = 4   # ~400ms at 100ms throttle

# Dynamic mode: smoothing window — wider to capture full motion arc
DYNAMIC_SMOOTHING = 5

# Static mode: smoothing window — tighter for fast response
STATIC_SMOOTHING  = 2

# Velocity history length for motion state decisions
VELOCITY_HISTORY_LEN = 6

# FIX #5: Same face indices as training + desktop testing
SELECTED_FACE_IDX = [
    13, 14, 78, 308, 82, 312, 33, 133, 362, 263,
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300
]
POSE_IDX = [11, 13, 15, 12, 14, 16]  # shoulders, elbows, wrists

# ==============================
# LOAD MODEL + LABELS
# ==============================
print("🔄 Loading model and labels...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Run 02_train_model.py first."
    )

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Warm up model — first predict() call is always slow due to JIT compilation.
# Running a dummy prediction here means the first real frame is fast.
_dummy = np.zeros((1, SEQUENCE_LENGTH, FEATURE_SIZE), dtype=np.float32)
model.predict(_dummy, verbose=0)
print("✅ Model warmed up")

with open(LABEL_PATH, "r") as f:
    LABELS = json.load(f)

print(f"✅ Model loaded | {len(LABELS)} classes: {LABELS}")

# Thread pool for running model.predict() without blocking MediaPipe.
# 1 worker — TF model is not thread-safe for concurrent calls.
_infer_pool = ThreadPoolExecutor(max_workers=1)

# ==============================
# MEDIAPIPE  (single shared instance)
# ==============================
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# ==============================
# PER-SESSION STATE
# ==============================
class SessionState:
    def __init__(self):
        self.raw_buffer      = deque(maxlen=RAW_BUFFER_LEN)
        self.smooth_buffer   = deque(maxlen=DYNAMIC_SMOOTHING)
        self.accepted        = 0
        self.hand_history    = deque(maxlen=HAND_HISTORY_LEN)

        # Motion tracking
        self.velocity_history  = deque(maxlen=VELOCITY_HISTORY_LEN)
        self.prev_wrist_pos    = None
        self.prev_spread       = None    # previous fingertip spread value
        self.still_frame_count = 0
        self.motion_state      = 'static'
        self.arc_started       = False
        self.last_emitted_sign = None
        self.buffer_cleared    = False

    @property
    def is_dual_hand(self) -> bool:
        if not self.hand_history:
            return False
        return sum(self.hand_history) / len(self.hand_history) >= 0.5

    def update_motion(self, kp: np.ndarray) -> float:
        """
        Combined velocity = wrist displacement + fingertip spread change.

        Wrist displacement alone misses signs that share the same wrist
        position but differ in finger orientation (e.g. A / S / E).

        Fingertip spread = mean pairwise distance between the 5 fingertips.
        When fingers open, close, or reconfigure, spread changes → detected
        as motion even if the wrist never moves.

        Final velocity = 0.6 × wrist_vel + 0.4 × spread_change
        Both components normalised to similar scale (~0.0–0.05 at rest,
        >0.03 when actively changing).
        """
        # ── Fingertip landmark offsets within one hand block (63 values) ──
        # landmark 4=thumb, 8=index, 12=middle, 16=ring, 20=pinky
        # each landmark occupies 3 values (x, y, z) → offset = landmark × 3
        FINGERTIP_OFFSETS = [12, 24, 36, 48, 60]  # × 3 → x coordinate

        def _wrist_xy(base):
            """Return (x, y) of wrist for hand at kp[base], or None if absent."""
            x, y = kp[base], kp[base + 1]
            return np.array([x, y]) if (x != 0 or y != 0) else None

        def _fingertip_spread(base):
            """
            Mean distance between all pairs of fingertips for one hand.
            Returns 0.0 if hand absent.
            """
            tips = []
            for off in FINGERTIP_OFFSETS:
                x, y = kp[base + off], kp[base + off + 1]
                if x != 0 or y != 0:
                    tips.append(np.array([x, y]))
            if len(tips) < 2:
                return 0.0
            dists = []
            for i in range(len(tips)):
                for j in range(i + 1, len(tips)):
                    dists.append(np.linalg.norm(tips[i] - tips[j]))
            return float(np.mean(dists))

        # ── Pick active hand (prefer right, fallback left) ──
        right_wrist = _wrist_xy(63)
        left_wrist  = _wrist_xy(0)

        if right_wrist is not None:
            wrist        = right_wrist
            spread       = _fingertip_spread(63)
        elif left_wrist is not None:
            wrist        = left_wrist
            spread       = _fingertip_spread(0)
        else:
            # No hand — treat as still
            self.velocity_history.append(0.0)
            self._update_state(0.0)
            return 0.0

        # ── Wrist displacement ──
        wrist_vel = 0.0
        if self.prev_wrist_pos is not None:
            wrist_vel = float(np.mean(np.abs(wrist - self.prev_wrist_pos)))
        self.prev_wrist_pos = wrist

        # ── Fingertip spread change ──
        spread_vel = 0.0
        if self.prev_spread is not None:
            spread_vel = abs(spread - self.prev_spread)
        self.prev_spread = spread

        # ── Combined velocity ──
        # Wrist movement weighted higher (more reliable signal),
        # finger change weighted lower but enough to catch same-wrist signs.
        velocity = 0.6 * wrist_vel + 0.4 * spread_vel

        self.velocity_history.append(velocity)
        self._update_state(velocity)
        return velocity

    def _update_state(self, velocity: float):
        """State machine: static → dynamic → settling → static"""
        avg_velocity = float(np.mean(self.velocity_history)) if self.velocity_history else 0.0
        prev_state   = self.motion_state

        if avg_velocity > DYNAMIC_THRESHOLD:
            # Actively moving
            self.motion_state      = 'dynamic'
            self.arc_started       = True
            self.still_frame_count = 0
            # ── KEY FIX ──────────────────────────────────────────────────────
            # First frame of motion after being static = sign transition.
            # Clear the raw buffer immediately so old-sign frames don't
            # contaminate the next prediction.
            if prev_state in ('static', 'settling'):
                self.buffer_cleared = True   # signal to predict() to flush
            # Widen smoothing buffer for dynamic mode
            if self.smooth_buffer.maxlen != DYNAMIC_SMOOTHING:
                old = list(self.smooth_buffer)
                self.smooth_buffer = deque(old, maxlen=DYNAMIC_SMOOTHING)

        elif avg_velocity < STATIC_THRESHOLD:
            self.still_frame_count += 1
            if self.arc_started and self.still_frame_count >= STATIC_CONFIRM_FRAMES:
                # Motion arc just completed → settling → predict
                self.motion_state  = 'settling'
                self.arc_started   = False
            elif not self.arc_started:
                self.motion_state = 'static'
            # Tighten smoothing buffer for static mode
            if self.smooth_buffer.maxlen != STATIC_SMOOTHING:
                old = list(self.smooth_buffer)[-STATIC_SMOOTHING:]
                self.smooth_buffer = deque(old, maxlen=STATIC_SMOOTHING)
        else:
            # In between — keep current state
            self.still_frame_count = 0

    @property
    def should_predict(self) -> bool:
        """
        Static  → predict after STATIC_CONFIRM_FRAMES still frames
        Dynamic → don't predict mid-motion
        Settling → predict once (arc just completed)
        """
        if self.motion_state == 'static':
            return self.still_frame_count >= STATIC_CONFIRM_FRAMES
        if self.motion_state == 'settling':
            return True
        return False  # dynamic — wait for arc to complete

    def mark_settling_done(self):
        """Call after emitting a prediction in settling state."""
        self.motion_state      = 'static'
        self.still_frame_count = 0

sessions: dict[str, SessionState] = {}

def get_session(session_id: str) -> SessionState:
    if session_id not in sessions:
        sessions[session_id] = SessionState()
    return sessions[session_id]

# ==============================
# NORMALIZATION  — identical to training
# ==============================
def normalize_hand(frame):
    frame = frame.copy()
    xs, ys = frame[0::3], frame[1::3]
    valid  = xs != 0
    if not valid.any(): return frame
    min_x, max_x = xs[valid].min(), xs[valid].max()
    min_y, max_y = ys[valid].min(), ys[valid].max()
    scale = max(max_x - min_x, max_y - min_y)
    if scale == 0: scale = 1
    for i in range(0, len(frame), 3):
        frame[i]     = (frame[i]     - min_x) / scale
        frame[i + 1] = (frame[i + 1] - min_y) / scale
    return frame

def normalize_hand_pair(h): return np.concatenate([normalize_hand(h[:63]), normalize_hand(h[63:])])

def normalize_face(frame):
    frame = frame.copy()
    xs, ys = frame[0::3], frame[1::3]
    valid  = xs != 0
    if not valid.any(): return frame
    cx, cy = xs[0], ys[0]
    scale  = max(xs[valid].max() - xs[valid].min(), ys[valid].max() - ys[valid].min())
    if scale == 0: scale = 1
    for i in range(0, len(frame), 3):
        frame[i]     = (frame[i]     - cx) / scale
        frame[i + 1] = (frame[i + 1] - cy) / scale
    return frame

def normalize_pose(frame):
    frame = frame.copy()
    xs, ys = frame[0::3], frame[1::3]
    valid  = xs != 0
    if not valid.any(): return frame
    cx, cy = xs[0], ys[0]
    scale  = max(xs[valid].max() - xs[valid].min(), ys[valid].max() - ys[valid].min())
    if scale == 0: scale = 1
    for i in range(0, len(frame), 3):
        frame[i]     = (frame[i]     - cx) / scale
        frame[i + 1] = (frame[i + 1] - cy) / scale
    return frame

# ==============================
# CENTER-OF-FRAME HAND FILTER
# ==============================
# Used automatically when only one hand is detected as the signing hand.
# Picks the hand whose wrist x is closest to frame centre (x=0.5).
# The signing hand is raised and centred; the idle hand hangs to the side.
# Tiebreaker: MediaPipe visibility score.

CENTER_TIE = 0.05

# Vertical resting-hand filter.
# In the cropped 16:9 frame a signing hand's wrist is in the upper portion.
# A resting hand hangs down — its wrist y is near the bottom of the frame.
# Any hand whose wrist y > this threshold is considered resting and suppressed.
# 0.72 means: if wrist is in the bottom 28% of the crop → treat as resting.
RESTING_Y_THRESHOLD = 0.72

def _is_resting(hand_lms) -> bool:
    """Return True if the hand's wrist is too low to be a signing hand."""
    if hand_lms is None:
        return False
    return hand_lms.landmark[0].y > RESTING_Y_THRESHOLD

def _pick_center_hand(left_lms, right_lms):
    """
    Given left and right hand landmarks (either may be None):
    1. Suppress any hand whose wrist is below RESTING_Y_THRESHOLD.
    2. If both remain, keep the one closest to frame centre (x=0.5).
    3. Tiebreaker: MediaPipe visibility score.
    Returns (left_result, right_result) with suppressed hand set to None.
    """
    # Step 1: vertical resting-hand filter
    if _is_resting(left_lms):
        left_lms = None
    if _is_resting(right_lms):
        right_lms = None

    if left_lms is None or right_lms is None:
        return left_lms, right_lms  # zero or one hand left — nothing to filter

    # Step 2: centre-of-frame bias
    left_dist  = abs(left_lms.landmark[0].x  - 0.5)
    right_dist = abs(right_lms.landmark[0].x - 0.5)

    if abs(left_dist - right_dist) <= CENTER_TIE:
        # Step 3: tiebreak by visibility
        left_vis  = getattr(left_lms.landmark[0],  'visibility', 1.0)
        right_vis = getattr(right_lms.landmark[0], 'visibility', 1.0)
        return (left_lms, None) if left_vis >= right_vis else (None, right_lms)
    elif left_dist < right_dist:
        return left_lms, None
    else:
        return None, right_lms


# ==============================
# KEYPOINT EXTRACTION
# ==============================
def extract_keypoints(results, is_dual_hand: bool) -> np.ndarray:
    """
    Extract one frame's keypoints from MediaPipe Holistic results.

    is_dual_hand is determined automatically by SessionState.is_dual_hand —
    True  → keep both hands (both slots filled)
    False → apply center-of-frame filter to suppress the idle hand
    """
    kp = np.zeros(FEATURE_SIZE, dtype=np.float32)

    # Hand selection
    left_lms  = results.left_hand_landmarks
    right_lms = results.right_hand_landmarks

    if not is_dual_hand:
        left_lms, right_lms = _pick_center_hand(left_lms, right_lms)

    for h_idx, hand in enumerate([left_lms, right_lms]):
        if hand:
            base = h_idx * 63
            for i, lm in enumerate(hand.landmark):
                kp[base + i * 3]     = lm.x
                kp[base + i * 3 + 1] = lm.y
                kp[base + i * 3 + 2] = lm.z

    # Face
    if results.face_landmarks:
        for i, idx in enumerate(SELECTED_FACE_IDX):
            lm  = results.face_landmarks.landmark[idx]
            off = HAND_FEATURES + i * 3
            kp[off]     = lm.x
            kp[off + 1] = lm.y
            kp[off + 2] = lm.z

    # Pose
    if results.pose_landmarks:
        pose_start = HAND_FEATURES + FACE_FEATURES
        for i, idx in enumerate(POSE_IDX):
            lm  = results.pose_landmarks.landmark[idx]
            off = pose_start + i * 3
            kp[off]     = lm.x
            kp[off + 1] = lm.y
            kp[off + 2] = lm.z

    return kp

def normalize_keypoints(kp: np.ndarray) -> np.ndarray:
    return np.concatenate([
        normalize_hand_pair(kp[:HAND_FEATURES]),
        normalize_face(kp[HAND_FEATURES:HAND_FEATURES + FACE_FEATURES]),
        normalize_pose(kp[HAND_FEATURES + FACE_FEATURES:])
    ])

# ==============================
# USER VISIBILITY CHECK
# ==============================
# Minimum landmarks required before prediction is allowed.
# This ensures the user is properly framed — avoids garbage predictions
# from partial frames where the person is too far, too close, or off-center.
#
# Required:
#   - Face detected
#   - Both shoulders visible (pose landmarks 11, 12)
#   - Both elbows visible   (pose landmarks 13, 14)
#
# Not required:
#   - Hands (user may not be signing yet)
#
# Visibility threshold: MediaPipe assigns 0.0–1.0 per landmark.
# We require > 0.5 to count a landmark as "seen".
VISIBILITY_THRESHOLD = 0.3   # lenient — landmark just needs to be partially visible

# POSE_IDX = [11, 13, 15, 12, 14, 16]
#  slot 0 → landmark 11 (left shoulder)
#  slot 1 → landmark 13 (left elbow)     ← excluded: unreliable when arms at rest
#  slot 3 → landmark 12 (right shoulder)
#  slot 4 → landmark 14 (right elbow)    ← excluded: unreliable when arms at rest
#
# Shoulders + face is sufficient to confirm proper framing.
# Elbows are intentionally excluded — MediaPipe gives them low visibility
# scores even when the person is correctly positioned.
_REQUIRED_POSE_SLOTS = [0, 3]  # left shoulder + right shoulder only

def check_user_visibility(results) -> tuple[bool, str]:
    """
    Returns (is_visible, reason).
    is_visible = True  → all required landmarks present → proceed with prediction
    is_visible = False → something missing → block prediction, reason explains what
    """
    # 1. Face
    if not results.face_landmarks:
        return False, "Face not detected — move closer or face the camera"

    # 2. Pose landmarks exist at all
    if not results.pose_landmarks:
        return False, "Upper body not detected — step back so shoulders are visible"

    # 3. Both shoulders + both elbows must be visible
    missing = []
    names   = {0: "left shoulder", 1: "left elbow", 3: "right shoulder", 4: "right elbow"}

    for slot in _REQUIRED_POSE_SLOTS:
        pose_lm_idx = POSE_IDX[slot]
        lm          = results.pose_landmarks.landmark[pose_lm_idx]
        vis         = getattr(lm, 'visibility', 1.0)
        if vis < VISIBILITY_THRESHOLD:
            missing.append(names[slot])

    if missing:
        return False, f"Not visible: {', '.join(missing)}"

    return True, "ok"


# Minimum frames in buffer before prediction starts.
# Model always receives SEQUENCE_LENGTH=8 frames — early frames are
# left-padded by repeating the first available frame.
# Lower = faster first prediction but lower early confidence.
# CONF_THRESHOLD filters out low-confidence early predictions naturally.
MIN_FRAMES_TO_START = 2   # start predicting after just 2 frames


def _pad_sequence(frames: list, target_len: int) -> np.ndarray:
    """
    Pad a short frame list to target_len by repeating the first frame on the left.

    Example (target=8):
      2 frames [A, B]       → [A, A, A, A, A, A, A, B]
      4 frames [A, B, C, D] → [A, A, A, A, A, B, C, D]
      8 frames (full)       → unchanged
    """
    if len(frames) >= target_len:
        return np.array(frames[-target_len:], dtype=np.float32)
    pad_count = target_len - len(frames)
    padded    = [frames[0]] * pad_count + frames
    return np.array(padded, dtype=np.float32)


def _run_inference(seq: np.ndarray) -> tuple[int, float]:
    """Run model.predict on a single sequence. Runs in thread pool."""
    inp   = np.expand_dims(seq, axis=0)   # (1, 8, 204)
    probs = model.predict(inp, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return idx, float(probs[idx])


def predict_multiscale(raw_buffer: deque):
    """
    Returns (best_idx, best_conf) or (None, None).
    Builds all sequences first, then submits all strides to the thread pool
    concurrently so MediaPipe overlap is maximised.
    """
    buf  = list(raw_buffer)
    seqs = []

    for stride in STRIDES:
        needed = stride * SEQUENCE_LENGTH
        if stride == 1:
            if len(buf) < MIN_FRAMES_TO_START:
                continue
            tail = buf[-needed:] if len(buf) >= needed else buf
            seqs.append(_pad_sequence(tail, SEQUENCE_LENGTH))
        else:
            if len(buf) < needed:
                continue
            tail = buf[-needed:]
            seq  = np.array(tail[::stride], dtype=np.float32)
            if len(seq) == SEQUENCE_LENGTH:
                seqs.append(seq)

    if not seqs:
        return None, None

    # Submit all strides to thread pool and wait for results
    futures = [_infer_pool.submit(_run_inference, seq) for seq in seqs]
    best_idx, best_conf = None, -1.0
    for f in futures:
        idx, conf = f.result()
        if conf > best_conf:
            best_conf, best_idx = conf, idx

    return (best_idx, best_conf) if best_idx is not None else (None, None)


# ==============================
# IMAGE DECODING
# ==============================
def decode_frame(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",")[1]
    buf   = base64.b64decode(b64)
    arr   = np.frombuffer(buf, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image")
    return frame

# ==============================
# FASTAPI APP
# ==============================
app = FastAPI(
    title="BIM Sign Language API",
    description="Real-time Malay Sign Language (BIM) detection backend",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ──
class PredictRequest(BaseModel):
    session_id : str
    frame      : str        # base64 JPEG
    reset      : bool = False
    # is_dual_hand removed — server detects this automatically

class PredictResponse(BaseModel):
    sign              : str | None   # None = not confident yet
    confidence        : float
    buffer_size       : int
    ready             : bool
    landmarks         : dict
    processing_ms     : float
    dual_hand_detected: bool
    motion_state      : str          # 'static' | 'dynamic' | 'settling'
    velocity          : float
    user_visible      : bool         # True = properly framed, False = blocked
    visibility_reason : str          # explains what's missing when user_visible=False

class HealthResponse(BaseModel):
    status          : str
    model_loaded    : bool
    classes         : list
    sequence_length : int
    feature_size    : int
    strides         : list
    conf_threshold  : float
    smoothing       : int

# ── Endpoints ──
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status         ="ok",
        model_loaded   =model is not None,
        classes        =LABELS,
        sequence_length=SEQUENCE_LENGTH,
        feature_size   =FEATURE_SIZE,
        strides        =STRIDES,
        conf_threshold =CONF_THRESHOLD,
        smoothing      =SMOOTHING,
    )

# ==============================
# DEBUG FRAME BUFFER
# Stores the last received frame + MediaPipe overlay for inspection.
# Access via GET /debug/frame to download the annotated JPEG.
# Toggle saving with GET /debug/enable and GET /debug/disable.
# ==============================

_debug_lock          = threading.Lock()
_debug_enabled       = True          # flip to False in production
_debug_frame_path    = "debug_last_frame.jpg"
_debug_overlay_path  = "debug_last_overlay.jpg"
_debug_frame_counter = 0
_debug_save_every    = 10            # save every N frames (not every frame)

def _save_debug_frame(frame: np.ndarray, results, frame_counter: int):
    """
    Save two debug images to disk:
      debug_last_frame.jpg   — raw cropped JPEG as received from Flutter
      debug_last_overlay.jpg — same frame with MediaPipe landmarks drawn on it
    """
    try:
        # Raw frame
        cv2.imwrite(_debug_frame_path, frame)

        # Annotated overlay
        annotated = frame.copy()
        mp_drawing = mp.solutions.drawing_utils
        mp_styles  = mp.solutions.drawing_styles

        rgb_ann = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.left_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
            )
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.right_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
            )
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp.solutions.holistic.POSE_CONNECTIONS,
            )

        # Print wrist Y positions to console for threshold tuning
        h, w = frame.shape[:2]
        if results.left_hand_landmarks:
            wy = results.left_hand_landmarks.landmark[0].y
            print(f"  [DEBUG frame#{frame_counter}] LEFT  wrist y={wy:.3f}  "
                  f"({'RESTING' if wy > RESTING_Y_THRESHOLD else 'active'})")
        if results.right_hand_landmarks:
            wy = results.right_hand_landmarks.landmark[0].y
            print(f"  [DEBUG frame#{frame_counter}] RIGHT wrist y={wy:.3f}  "
                  f"({'RESTING' if wy > RESTING_Y_THRESHOLD else 'active'})")

        # Label frame size on the image
        label = f"#{frame_counter}  {w}x{h}px"
        cv2.putText(annotated, label, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite(_debug_overlay_path, annotated)

    except Exception as e:
        print(f"  [DEBUG] save failed: {e}")


@app.get("/classes")
def get_classes():
    return {"classes": LABELS, "total": len(LABELS)}

@app.get("/debug/enable")
def debug_enable():
    global _debug_enabled
    _debug_enabled = True
    return {"debug": True, "message": "Debug frame saving enabled"}

@app.get("/debug/disable")
def debug_disable():
    global _debug_enabled
    _debug_enabled = False
    return {"debug": False, "message": "Debug frame saving disabled"}

@app.get("/debug/frame")
def debug_get_frame():
    """Download the last raw frame received from Flutter (as sent to MediaPipe)."""
    from fastapi.responses import FileResponse
    if not os.path.exists(_debug_frame_path):
        raise HTTPException(status_code=404,
            detail="No debug frame yet. Send a /predict request first.")
    return FileResponse(_debug_frame_path, media_type="image/jpeg",
                        filename="debug_last_frame.jpg")

@app.get("/debug/overlay")
def debug_get_overlay():
    """Download the last frame with MediaPipe landmarks drawn on it.
    Also shows wrist Y positions — useful for tuning RESTING_Y_THRESHOLD."""
    from fastapi.responses import FileResponse
    if not os.path.exists(_debug_overlay_path):
        raise HTTPException(status_code=404,
            detail="No debug overlay yet. Send a /predict request first.")
    return FileResponse(_debug_overlay_path, media_type="image/jpeg",
                        filename="debug_last_overlay.jpg")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.time()

    # Handle reset
    if req.reset:
        if req.session_id in sessions:
            del sessions[req.session_id]
        return PredictResponse(
            sign=None, confidence=0.0, buffer_size=0,
            ready=False, landmarks={}, processing_ms=0.0,
            dual_hand_detected=False, motion_state='static', velocity=0.0,
            user_visible=False, visibility_reason='reset'
        )

    sess = get_session(req.session_id)

    try:
        # 1. Decode frame
        frame = decode_frame(req.frame)

        # 2. Run MediaPipe Holistic
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)

        # ── Debug: save frame + overlay every N frames ──
        global _debug_frame_counter
        _debug_frame_counter += 1
        if _debug_enabled and _debug_frame_counter % _debug_save_every == 0:
            threading.Thread(
                target=_save_debug_frame,
                args=(frame.copy(), results, _debug_frame_counter),
                daemon=True
            ).start()

        # 3. Check user is properly framed
        user_visible, visibility_reason = check_user_visibility(results)

        # 4. Track landmarks
        landmarks = {
            "left_hand"  : results.left_hand_landmarks  is not None,
            "right_hand" : results.right_hand_landmarks is not None,
            "pose"       : results.pose_landmarks       is not None,
            "face"       : results.face_landmarks       is not None,
        }

        # 5. Auto dual-hand detection
        hands_this_frame = sum([
            results.left_hand_landmarks  is not None,
            results.right_hand_landmarks is not None,
        ])
        sess.hand_history.append(hands_this_frame == 2)
        is_dual = sess.is_dual_hand

        # 6. Extract + normalize keypoints
        kp = extract_keypoints(results, is_dual_hand=is_dual)

        # 7. Update motion state
        velocity = sess.update_motion(kp)

        # ── Buffer flush on sign transition ──────────────────────────────────
        # When motion starts after being still, the old sign's frames are stale.
        # Clear immediately so the new sign fills a clean buffer.
        if sess.buffer_cleared:
            sess.raw_buffer.clear()
            sess.smooth_buffer.clear()
            sess.buffer_cleared = False
            print(f"  [MOTION] Sign transition — buffer cleared")

        # 8. Quality gate — only add to buffer when user is properly visible
        #    This prevents contaminating the buffer with partial-frame keypoints
        if user_visible:
            nonzero = np.count_nonzero(kp) / kp.size
            print(f"  [QC] nonzero={nonzero:.3f}  "
                  f"left={landmarks['left_hand']}  right={landmarks['right_hand']}  "
                  f"face={landmarks['face']}  pose={landmarks['pose']}  "
                  f"user_visible={user_visible}")
            if nonzero >= MIN_NONZERO:
                kp_norm = normalize_keypoints(kp)
                sess.raw_buffer.append(kp_norm)
                sess.accepted += 1
            else:
                print(f"  [QC] REJECTED — nonzero={nonzero:.3f} < MIN_NONZERO={MIN_NONZERO}")
        else:
            print(f"  [QC] BLOCKED — user_visible=False reason='{visibility_reason}'")

        buf_size  = len(sess.raw_buffer)
        ready     = buf_size >= MIN_FRAMES_TO_START and user_visible

        sign = None
        conf = 0.0

        # 9. Motion-aware inference — only when user is visible
        if ready and user_visible and sess.should_predict:
            best_idx, best_conf = predict_multiscale(sess.raw_buffer)

            if best_idx is not None and best_conf >= CONF_THRESHOLD:
                sess.smooth_buffer.append(best_idx)
                majority = int(np.bincount(
                    np.array(sess.smooth_buffer, dtype=np.int32)).argmax())
                sign = LABELS[majority]
                conf = best_conf

                if sess.motion_state == 'settling':
                    sess.mark_settling_done()
                    sess.smooth_buffer.clear()
            else:
                sess.smooth_buffer.clear()

        ms = round((time.time() - start) * 1000, 2)
        return PredictResponse(
            sign=sign, confidence=conf,
            buffer_size=buf_size, ready=ready,
            landmarks=landmarks, processing_ms=ms,
            dual_hand_detected=sess.is_dual_hand,
            motion_state=sess.motion_state,
            velocity=round(velocity, 4),
            user_visible=user_visible,
            visibility_reason=visibility_reason,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"message": f"Session {session_id} cleared"}

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    print("\n🚀 BIM Sign Language API")
    print("   URL      : http://0.0.0.0:8000")
    print("   API Docs : http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)