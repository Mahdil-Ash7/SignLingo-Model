"""
05_cloud_server.py
==================
Lightweight BIM Sign Language backend for Render (512MB RAM / 0.1 CPU).

KEY DIFFERENCE from 04_fastapi_server.py:
  - NO MediaPipe — keypoints are extracted on the phone instead
  - Accepts pre-extracted float arrays (204 values) directly
  - Only runs model inference (~15-30ms vs ~130ms full pipeline)
  - Fits comfortably within 512MB RAM

Flutter sends keypoints (not JPEG frames) to this server.
Flutter runs MediaPipe on-device using google_mlkit or mediapipe_flutter.

RAM breakdown:
  TensorFlow + model  : ~180MB
  FastAPI + uvicorn   : ~50MB
  Python runtime      : ~80MB
  Total               : ~310MB  ← well within 512MB

CONSTANTS — must match 02_train_model.py exactly:
  SEQUENCE_LENGTH = 8
  FEATURE_SIZE    = 204
  STRIDES         = [1, 2]
  RAW_BUFFER_LEN  = 16

Deploy on Render:
  Build command : pip install -r requirements_cloud.txt
  Start command : uvicorn 05_cloud_server:app --host 0.0.0.0 --port $PORT --workers 1
  Instance type : Free (512MB / 0.1 CPU)
"""

import numpy as np
import tensorflow as tf
import json
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

# Optimized for low-CPU environment
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# ==============================
# CONFIG — must match 02_train_model.py
# ==============================
MODEL_PATH  = "bim_model/handface_pose_cnn_lstm.h5"
LABEL_PATH  = "bim_model/labels.json"

SEQUENCE_LENGTH = 8
FEATURE_SIZE    = 204

STRIDES             = [1, 2]
RAW_BUFFER_LEN      = max(STRIDES) * SEQUENCE_LENGTH  # 16
MIN_NONZERO         = 0.45
CONF_THRESHOLD      = 0.65
MIN_FRAMES_TO_START = 2

# Motion-aware config — identical to 04_fastapi_server.py
STATIC_THRESHOLD      = 0.015
DYNAMIC_THRESHOLD     = 0.030
STATIC_CONFIRM_FRAMES = 4
DYNAMIC_SMOOTHING     = 5
STATIC_SMOOTHING      = 2
VELOCITY_HISTORY_LEN  = 6

# ==============================
# LOAD MODEL
# ==============================
print("🔄 Loading model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Warm up
_dummy = np.zeros((1, SEQUENCE_LENGTH, FEATURE_SIZE), dtype=np.float32)
model.predict(_dummy, verbose=0)
print("✅ Model warmed up")

with open(LABEL_PATH) as f:
    LABELS = json.load(f)

print(f"✅ {len(LABELS)} classes: {LABELS}")

_infer_pool = ThreadPoolExecutor(max_workers=1)

# ==============================
# NORMALIZATION — identical to training + local server
# ==============================
def normalize_hand(frame):
    frame = frame.copy()
    xs, ys = frame[0::3], frame[1::3]
    valid  = xs != 0
    if not valid.any(): return frame
    min_x, max_x = xs[valid].min(), xs[valid].max()
    min_y, max_y = ys[valid].min(), ys[valid].max()
    scale = max(max_x - min_x, max_y - min_y) or 1
    for i in range(0, len(frame), 3):
        frame[i]     = (frame[i]     - min_x) / scale
        frame[i + 1] = (frame[i + 1] - min_y) / scale
    return frame

def normalize_hand_pair(h):
    return np.concatenate([normalize_hand(h[:63]), normalize_hand(h[63:])])

def normalize_face(frame):
    frame = frame.copy()
    xs, ys = frame[0::3], frame[1::3]
    valid  = xs != 0
    if not valid.any(): return frame
    cx, cy = xs[0], ys[0]
    scale  = max(xs[valid].max() - xs[valid].min(),
                 ys[valid].max() - ys[valid].min()) or 1
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
    scale  = max(xs[valid].max() - xs[valid].min(),
                 ys[valid].max() - ys[valid].min()) or 1
    for i in range(0, len(frame), 3):
        frame[i]     = (frame[i]     - cx) / scale
        frame[i + 1] = (frame[i + 1] - cy) / scale
    return frame

def normalize_keypoints(kp: np.ndarray) -> np.ndarray:
    return np.concatenate([
        normalize_hand_pair(kp[:126]),
        normalize_face(kp[126:186]),
        normalize_pose(kp[186:]),
    ])

# ==============================
# PER-SESSION STATE
# Same motion state machine as local server — no changes needed in Flutter
# ==============================
class SessionState:
    def __init__(self):
        self.raw_buffer      = deque(maxlen=RAW_BUFFER_LEN)
        self.smooth_buffer   = deque(maxlen=DYNAMIC_SMOOTHING)
        self.accepted        = 0
        self.velocity_history  = deque(maxlen=VELOCITY_HISTORY_LEN)
        self.prev_wrist_pos    = None
        self.prev_spread       = None
        self.still_frame_count = 0
        self.motion_state      = 'static'
        self.arc_started       = False
        self.buffer_cleared    = False

    def update_motion(self, kp: np.ndarray) -> float:
        FINGERTIP_OFFSETS = [12, 24, 36, 48, 60]

        def _wrist_xy(base):
            x, y = kp[base], kp[base + 1]
            return np.array([x, y]) if (x != 0 or y != 0) else None

        def _spread(base):
            tips = []
            for off in FINGERTIP_OFFSETS:
                x, y = kp[base + off], kp[base + off + 1]
                if x != 0 or y != 0:
                    tips.append(np.array([x, y]))
            if len(tips) < 2: return 0.0
            return float(np.mean([
                np.linalg.norm(tips[i] - tips[j])
                for i in range(len(tips))
                for j in range(i + 1, len(tips))
            ]))

        right_wrist = _wrist_xy(63)
        left_wrist  = _wrist_xy(0)
        if right_wrist is not None:
            wrist, spread = right_wrist, _spread(63)
        elif left_wrist is not None:
            wrist, spread = left_wrist, _spread(0)
        else:
            self.velocity_history.append(0.0)
            self._update_state(0.0)
            return 0.0

        wrist_vel  = float(np.mean(np.abs(wrist - self.prev_wrist_pos))) \
                     if self.prev_wrist_pos is not None else 0.0
        spread_vel = abs(spread - self.prev_spread) \
                     if self.prev_spread is not None else 0.0
        self.prev_wrist_pos = wrist
        self.prev_spread    = spread

        velocity = 0.6 * wrist_vel + 0.4 * spread_vel
        self.velocity_history.append(velocity)
        self._update_state(velocity)
        return velocity

    def _update_state(self, velocity: float):
        avg        = float(np.mean(self.velocity_history)) if self.velocity_history else 0.0
        prev_state = self.motion_state

        if avg > DYNAMIC_THRESHOLD:
            self.motion_state      = 'dynamic'
            self.arc_started       = True
            self.still_frame_count = 0
            if prev_state in ('static', 'settling'):
                self.buffer_cleared = True
            if self.smooth_buffer.maxlen != DYNAMIC_SMOOTHING:
                self.smooth_buffer = deque(list(self.smooth_buffer), maxlen=DYNAMIC_SMOOTHING)
        elif avg < STATIC_THRESHOLD:
            self.still_frame_count += 1
            if self.arc_started and self.still_frame_count >= STATIC_CONFIRM_FRAMES:
                self.motion_state  = 'settling'
                self.arc_started   = False
            elif not self.arc_started:
                self.motion_state = 'static'
            if self.smooth_buffer.maxlen != STATIC_SMOOTHING:
                old = list(self.smooth_buffer)[-STATIC_SMOOTHING:]
                self.smooth_buffer = deque(old, maxlen=STATIC_SMOOTHING)
        else:
            self.still_frame_count = 0

    @property
    def should_predict(self) -> bool:
        if self.motion_state == 'static':   return self.still_frame_count >= STATIC_CONFIRM_FRAMES
        if self.motion_state == 'settling': return True
        return False

    def mark_settling_done(self):
        self.motion_state      = 'static'
        self.still_frame_count = 0

sessions: dict[str, SessionState] = {}

def get_session(sid: str) -> SessionState:
    if sid not in sessions:
        sessions[sid] = SessionState()
    return sessions[sid]

# ==============================
# INFERENCE
# ==============================
def _pad_sequence(frames: list, target_len: int) -> np.ndarray:
    if len(frames) >= target_len:
        return np.array(frames[-target_len:], dtype=np.float32)
    pad_count = target_len - len(frames)
    return np.array([frames[0]] * pad_count + frames, dtype=np.float32)

def _run_inference(seq: np.ndarray) -> tuple[int, float]:
    probs = model.predict(np.expand_dims(seq, 0), verbose=0)[0]
    idx   = int(np.argmax(probs))
    return idx, float(probs[idx])

def predict_multiscale(raw_buffer: deque):
    buf, seqs = list(raw_buffer), []
    for stride in STRIDES:
        needed = stride * SEQUENCE_LENGTH
        if stride == 1:
            if len(buf) < MIN_FRAMES_TO_START: continue
            tail = buf[-needed:] if len(buf) >= needed else buf
            seqs.append(_pad_sequence(tail, SEQUENCE_LENGTH))
        else:
            if len(buf) < needed: continue
            tail = buf[-needed:]
            seq  = np.array(tail[::stride], dtype=np.float32)
            if len(seq) == SEQUENCE_LENGTH:
                seqs.append(seq)

    if not seqs: return None, None
    futures = [_infer_pool.submit(_run_inference, seq) for seq in seqs]
    best_idx, best_conf = None, -1.0
    for f in futures:
        idx, conf = f.result()
        if conf > best_conf:
            best_conf, best_idx = conf, idx
    return (best_idx, best_conf) if best_idx is not None else (None, None)

# ==============================
# FASTAPI APP
# ==============================
app = FastAPI(
    title      = "BIM Sign Language Cloud API",
    description= "Lightweight inference-only backend (no MediaPipe)",
    version    = "1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins    = ["*"],
    allow_methods    = ["*"],
    allow_headers    = ["*"],
    allow_credentials= True,
)

# ── Schemas ──
class CloudPredictRequest(BaseModel):
    session_id : str
    keypoints  : List[float]   # 204 floats — extracted by Flutter MediaPipe on-device
    reset      : bool = False

class CloudPredictResponse(BaseModel):
    sign              : str | None
    confidence        : float
    buffer_size       : int
    ready             : bool
    processing_ms     : float
    motion_state      : str
    velocity          : float

class HealthResponse(BaseModel):
    status         : str
    model_loaded   : bool
    classes        : list
    sequence_length: int
    feature_size   : int
    mode           : str   # 'cloud' — Flutter knows it must extract keypoints itself

# ── Endpoints ──
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status         = "ok",
        model_loaded   = model is not None,
        classes        = LABELS,
        sequence_length= SEQUENCE_LENGTH,
        feature_size   = FEATURE_SIZE,
        mode           = "cloud",
    )

@app.get("/classes")
def get_classes():
    return {"classes": LABELS, "total": len(LABELS)}

@app.post("/predict", response_model=CloudPredictResponse)
async def predict(req: CloudPredictRequest):
    start = time.time()

    if req.reset:
        sessions.pop(req.session_id, None)
        return CloudPredictResponse(
            sign=None, confidence=0.0, buffer_size=0,
            ready=False, processing_ms=0.0,
            motion_state='static', velocity=0.0,
        )

    if len(req.keypoints) != FEATURE_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {FEATURE_SIZE} keypoints, got {len(req.keypoints)}"
        )

    sess = get_session(req.session_id)
    kp   = np.array(req.keypoints, dtype=np.float32)

    # Motion update
    velocity = sess.update_motion(kp)

    # Buffer flush on sign transition
    if sess.buffer_cleared:
        sess.raw_buffer.clear()
        sess.smooth_buffer.clear()
        sess.buffer_cleared = False

    # Quality gate
    nonzero = np.count_nonzero(kp) / kp.size
    if nonzero >= MIN_NONZERO:
        kp_norm = normalize_keypoints(kp)
        sess.raw_buffer.append(kp_norm)
        sess.accepted += 1

    buf_size = len(sess.raw_buffer)
    ready    = buf_size >= MIN_FRAMES_TO_START

    sign = None
    conf = 0.0

    if ready and sess.should_predict:
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
    return CloudPredictResponse(
        sign=sign, confidence=conf,
        buffer_size=buf_size, ready=ready,
        processing_ms=ms,
        motion_state=sess.motion_state,
        velocity=round(velocity, 4),
    )

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"message": f"Session {session_id} cleared"}

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    print("\n🚀 BIM Sign Language Cloud API (inference-only)")
    print("   URL : http://0.0.0.0:8000")
    print("   Mode: keypoints-only (no MediaPipe)\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)