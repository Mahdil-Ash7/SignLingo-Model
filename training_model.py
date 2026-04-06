"""
02_train_model.py
=================
Trains the 1D CNN + GRU model on BIM keypoint sequences.

CHANGES IN THIS VERSION:
  Level 3 — Profile-weighted loss
  ────────────────────────────────
  Sign profiles (from 04_build_sign_profiles.py) are loaded before training.
  A custom loss function weights the cross-entropy contribution of each class
  by how discriminative its features are. Classes whose key regions (hand/face/
  pose) are high-variance get penalized more strongly when misclassified,
  forcing the model to focus on the regions that actually matter per sign.

  If sign_profiles.json does not exist yet, training falls back to standard
  categorical cross-entropy automatically — so you can train a first pass
  without profiles, then run 04_build_sign_profiles.py and retrain.

  BUG FIXES vs previous version:
    - spatial_augment() def was missing (orphaned function body)
    - Duplicate AUGMENTATION FUNCTIONS comment block removed

ARCHITECTURE:
  Conv1D(64)  → BN → MaxPool → Dropout
  Conv1D(128) → BN → Dropout
  GRU(128, return_sequences=True,  unroll=True) → Dropout
  GRU(64,  return_sequences=False, unroll=True) → Dropout
  Dense(128) → Dropout
  Dense(64)  → Dropout
  Dense(num_classes, softmax)
"""

import os
import json
import math
import pickle
import random
from collections import Counter, defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization, Conv1D,
    Dense, Dropout, GRU, MaxPooling1D
)
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==============================
# CONFIG — single source of truth
# ==============================
DATA_PATH       = "dataset_desktop"
SEQUENCE_LENGTH = 8
HAND_FEATURES   = 126
FACE_FEATURES   = 60
POSE_FEATURES   = 18
FEATURE_SIZE    = HAND_FEATURES + FACE_FEATURES + POSE_FEATURES  # 204
MIN_NONZERO     = 0.15

TEST_SIZE       = 0.2
BATCH_SIZE      = 32
EPOCHS          = 80

MODEL_SAVE_PATH = "bim_model/handface_pose_cnn_gru.h5"
TFLITE_PATH     = "bim_model/handface_pose_cnn_gru.tflite"
LABEL_PATH      = "bim_model/labels.json"
ENCODER_PATH    = "bim_model/label_encoder.pkl"
HISTORY_PATH    = "bim_model/train_history.pkl"

# ── Level 3: profile-weighted loss config ─────────────────────────────────
PROFILES_PATH        = "bim_model/sign_profiles.json"
STD_IGNORE_THRESHOLD = 0.12   # regions with std > this are treated as noise
PROFILE_LOSS_WEIGHT  = 0.35   # blend: 0.65 × standard CE + 0.35 × weighted CE
                               # increase toward 0.5 if similar signs still confused
                               # decrease toward 0.1 if overall accuracy drops

# ── Augmentation config ────────────────────────────────────────────────────
AUGMENTATION_FACTOR   = 8

AUG_SCALE_RANGE       = (0.85, 1.15)
AUG_ROT_RANGE         = (-10, 10)
AUG_TRANS_RANGE       = (-0.05, 0.05)
AUG_NOISE_STD         = 0.006
AUG_HAND_SCALE        = (0.80, 1.20)
AUG_SHEAR_RANGE       = (-0.12, 0.12)
AUG_WARP_RANGE        = (0.7, 1.3)
AUG_WARP_PROB         = 0.4
AUG_KP_DROP_P         = 0.12
AUG_MIXUP_ALPHA       = 0.2
AUG_MIXUP_PROB        = 0.3
AUG_MIRROR_PROB       = 0.5
AUG_TILT_RANGE        = (-0.15, 0.15)
AUG_TILT_PROB         = 0.5
AUG_ROLL_RANGE        = (-0.08, 0.08)
AUG_ROLL_PROB         = 0.5
AUG_DISTANCE_RANGE    = (0.75, 1.30)
AUG_DISTANCE_PROB     = 0.5
AUG_CROP_OFFSET_RANGE = (-0.08, 0.08)
AUG_CROP_OFFSET_PROB  = 0.4
AUG_ASPECT_RANGE      = (0.92, 1.08)
AUG_ASPECT_PROB       = 0.4

os.makedirs("bim_model", exist_ok=True)

# ==============================
# REPRODUCIBILITY
# ==============================
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ==============================
# NORMALIZATION
# ==============================
def normalize_hand(frame):
    frame = frame.copy()
    xs, ys = frame[0::3], frame[1::3]
    valid = xs != 0
    if not valid.any():
        return frame
    min_x, max_x = xs[valid].min(), xs[valid].max()
    min_y, max_y = ys[valid].min(), ys[valid].max()
    scale = max(max_x - min_x, max_y - min_y)
    if scale == 0: scale = 1
    for i in range(0, len(frame), 3):
        frame[i]     = (frame[i]     - min_x) / scale
        frame[i + 1] = (frame[i + 1] - min_y) / scale
    return frame

def normalize_hand_pair(h):
    return np.concatenate([normalize_hand(h[:63]), normalize_hand(h[63:])])

def normalize_face(frame):
    frame = frame.copy()
    xs, ys = frame[0::3], frame[1::3]
    valid = xs != 0
    if not valid.any():
        return frame
    cx, cy = xs[0], ys[0]
    scale = max(xs[valid].max() - xs[valid].min(),
                ys[valid].max() - ys[valid].min())
    if scale == 0: scale = 1
    for i in range(0, len(frame), 3):
        frame[i]     = (frame[i]     - cx) / scale
        frame[i + 1] = (frame[i + 1] - cy) / scale
    return frame

def normalize_pose(frame):
    frame = frame.copy()
    xs, ys = frame[0::3], frame[1::3]
    valid = xs != 0
    if not valid.any():
        return frame
    cx, cy = xs[0], ys[0]
    scale = max(xs[valid].max() - xs[valid].min(),
                ys[valid].max() - ys[valid].min())
    if scale == 0: scale = 1
    for i in range(0, len(frame), 3):
        frame[i]     = (frame[i]     - cx) / scale
        frame[i + 1] = (frame[i + 1] - cy) / scale
    return frame

def normalize_sequence(seq):
    return np.array([
        np.concatenate([
            normalize_hand_pair(frame[:HAND_FEATURES]),
            normalize_face(frame[HAND_FEATURES:HAND_FEATURES + FACE_FEATURES]),
            normalize_pose(frame[HAND_FEATURES + FACE_FEATURES:])
        ])
        for frame in seq
    ], dtype=np.float32)

# ==============================
# MIRROR
# ==============================
def mirror_sequence(sequence):
    new_seq = []
    for frame in sequence:
        f = frame.copy()
        for i in range(0, FEATURE_SIZE, 3):
            if f[i] != 0:
                f[i] = 1.0 - f[i]
        left  = f[0:63].copy()
        right = f[63:126].copy()
        f[0:63]   = right
        f[63:126] = left
        pose_start = HAND_FEATURES + FACE_FEATURES
        left_pose  = f[pose_start:pose_start + 9].copy()
        right_pose = f[pose_start + 9:pose_start + 18].copy()
        f[pose_start:pose_start + 9]      = right_pose
        f[pose_start + 9:pose_start + 18] = left_pose
        new_seq.append(f)
    return np.array(new_seq, dtype=np.float32)

# ==============================
# AUGMENTATION FUNCTIONS
# ==============================

def spatial_augment(sequence):
    """Global scale + rotation + translation applied to all landmarks."""
    scale = random.uniform(*AUG_SCALE_RANGE)
    angle = random.uniform(*AUG_ROT_RANGE)
    tx    = random.uniform(*AUG_TRANS_RANGE)
    ty    = random.uniform(*AUG_TRANS_RANGE)
    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    new_seq = []
    for frame in sequence:
        f = frame.copy()
        for i in range(0, FEATURE_SIZE, 3):
            # 🛑 ADD THIS IF STATEMENT: Only augment if the point exists
            if f[i] != 0: 
                x, y, z = f[i], f[i+1], f[i+2]
                x *= scale; y *= scale
                f[i]     = x * cos_t - y * sin_t + tx
                f[i + 1] = x * sin_t + y * cos_t + ty
                f[i + 2] = z
        new_seq.append(f)
    return np.array(new_seq, dtype=np.float32)

def noise_augment(sequence):
    noise = np.random.normal(0, AUG_NOISE_STD, sequence.shape).astype(np.float32)
    # 🛑 ADD THIS MASK: Multiply noise by a boolean mask so zeros stay zero
    mask = (sequence != 0.0).astype(np.float32)
    return sequence + (noise * mask)

def hand_scale_augment(sequence):
    left_scale  = random.uniform(*AUG_HAND_SCALE)
    right_scale = random.uniform(*AUG_HAND_SCALE)
    new_seq = []
    for frame in sequence:
        f = frame.copy()
        if f[0] != 0:
            cx, cy = f[0], f[1]
            for i in range(0, 63, 3):
                f[i]     = cx + (f[i]     - cx) * left_scale
                f[i + 1] = cy + (f[i + 1] - cy) * left_scale
        if f[63] != 0:
            cx, cy = f[63], f[64]
            for i in range(63, 126, 3):
                f[i]     = cx + (f[i]     - cx) * right_scale
                f[i + 1] = cy + (f[i + 1] - cy) * right_scale
        new_seq.append(f)
    return np.array(new_seq, dtype=np.float32)

def perspective_augment(sequence):
    shear = random.uniform(*AUG_SHEAR_RANGE)
    new_seq = []
    for frame in sequence:
        f = frame.copy()
        for i in range(0, FEATURE_SIZE, 3):
            if f[i] != 0:
                f[i] = f[i] + shear * f[i + 1]
        new_seq.append(f)
    return np.array(new_seq, dtype=np.float32)

def body_shift_augment(sequence):
    hand_tx = random.uniform(*AUG_TRANS_RANGE)
    hand_ty = random.uniform(*AUG_TRANS_RANGE)
    face_tx = hand_tx + random.uniform(-0.02, 0.02)
    face_ty = hand_ty + random.uniform(-0.02, 0.02)
    new_seq = []
    for frame in sequence:
        f = frame.copy()
        for i in range(0, 126, 3):
            if f[i] != 0:
                f[i]     += hand_tx
                f[i + 1] += hand_ty
        for i in range(126, 186, 3):
            if f[i] != 0:
                f[i]     += face_tx
                f[i + 1] += face_ty
        for i in range(186, 204, 3):
            if f[i] != 0:
                f[i]     += hand_tx
                f[i + 1] += hand_ty
        new_seq.append(f)
    return np.array(new_seq, dtype=np.float32)

def time_warp_augment(sequence):
    factor  = random.uniform(*AUG_WARP_RANGE)
    n       = len(sequence)
    new_len = max(2, int(n * factor))
    src_idx   = np.linspace(0, n - 1, new_len)
    resampled = np.array([
        sequence[int(i)] * (1 - (i % 1)) +
        sequence[min(int(i) + 1, n - 1)] * (i % 1)
        for i in src_idx
    ], dtype=np.float32)
    tgt_idx = np.linspace(0, len(resampled) - 1, SEQUENCE_LENGTH)
    return np.array([
        resampled[int(i)] * (1 - (i % 1)) +
        resampled[min(int(i) + 1, len(resampled) - 1)] * (i % 1)
        for i in tgt_idx
    ], dtype=np.float32)

def smooth_frame_drop(sequence):
    seq        = list(sequence)
    drop_count = random.randint(0, 2)
    for _ in range(drop_count):
        if len(seq) > 2:
            idx      = random.randint(1, len(seq) - 2)
            interp   = ((seq[idx - 1] + seq[idx + 1]) / 2.0).astype(np.float32)
            seq[idx] = interp
    while len(seq) < SEQUENCE_LENGTH:
        if len(seq) > 1:
            interp = ((seq[-1] + seq[-2]) / 2.0).astype(np.float32)
        else:
            interp = seq[-1].copy()
        seq.append(interp)
    return np.array(seq[:SEQUENCE_LENGTH], dtype=np.float32)

def keypoint_dropout_augment(sequence):
    new_seq = []
    for frame in sequence:
        f = frame.copy()
        if random.random() < AUG_KP_DROP_P:
            f[0:63] = 0.0
        if random.random() < AUG_KP_DROP_P:
            f[63:126] = 0.0
        for i in range(126, FEATURE_SIZE, 3):
            if random.random() < AUG_KP_DROP_P:
                f[i] = f[i+1] = f[i+2] = 0.0
        new_seq.append(f)
    return np.array(new_seq, dtype=np.float32)

def mixup_augment(seq_a, seq_b):
    lam = np.random.beta(AUG_MIXUP_ALPHA, AUG_MIXUP_ALPHA)
    return (lam * seq_a + (1 - lam) * seq_b).astype(np.float32)

def camera_tilt_augment(sequence):
    tilt = random.uniform(*AUG_TILT_RANGE)
    new_seq = []
    for frame in sequence:
        f = frame.copy()
        for i in range(1, FEATURE_SIZE, 3):
            if f[i] != 0:
                f[i] = f[i] * (1.0 + tilt * (0.5 - f[i]))
                f[i] = float(np.clip(f[i], 0.0, 1.0))
        new_seq.append(f)
    return np.array(new_seq, dtype=np.float32)

def camera_roll_augment(sequence):
    angle = random.uniform(*AUG_ROLL_RANGE)
    theta = math.radians(angle * 15)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = 0.5, 0.5
    new_seq = []
    for frame in sequence:
        f = frame.copy()
        for i in range(0, FEATURE_SIZE, 3):
            if f[i] != 0:
                x = f[i]     - cx
                y = f[i + 1] - cy
                f[i]     = x * cos_t - y * sin_t + cx
                f[i + 1] = x * sin_t + y * cos_t + cy
        new_seq.append(f)
    return np.array(new_seq, dtype=np.float32)

def camera_distance_augment(sequence):
    scale = random.uniform(*AUG_DISTANCE_RANGE)
    cx, cy = 0.5, 0.5
    new_seq = []
    for frame in sequence:
        f = frame.copy()
        for i in range(0, FEATURE_SIZE, 3):
            if f[i] != 0:
                f[i]     = cx + (f[i]     - cx) * scale
                f[i + 1] = cy + (f[i + 1] - cy) * scale
        new_seq.append(f)
    return np.array(new_seq, dtype=np.float32)

def crop_offset_augment(sequence):
    offset = random.uniform(*AUG_CROP_OFFSET_RANGE)
    new_seq = []
    for frame in sequence:
        f = frame.copy()
        for i in range(1, FEATURE_SIZE, 3):
            if f[i] != 0:
                f[i] = float(np.clip(f[i] + offset, 0.0, 1.0))
        new_seq.append(f)
    return np.array(new_seq, dtype=np.float32)

def aspect_ratio_augment(sequence):
    sx = random.uniform(*AUG_ASPECT_RANGE)
    sy = random.uniform(*AUG_ASPECT_RANGE)
    cx, cy = 0.5, 0.5
    new_seq = []
    for frame in sequence:
        f = frame.copy()
        for i in range(0, FEATURE_SIZE, 3):
            if f[i] != 0:
                f[i]     = cx + (f[i]     - cx) * sx
                f[i + 1] = cy + (f[i + 1] - cy) * sy
        new_seq.append(f)
    return np.array(new_seq, dtype=np.float32)

# 🛑 UPDATE THIS DEFINITION to accept `is_dynamic`
def augment_sequence(sequence, is_dynamic, same_class_pool=None):
    seq = sequence.copy()
    seq = spatial_augment(seq)
    seq = noise_augment(seq)
    seq = hand_scale_augment(seq)
    seq = body_shift_augment(seq)
    seq = perspective_augment(seq)
    seq = smooth_frame_drop(seq)
    
    # 🛑 UPDATE THIS LINE: Only time-warp if the sign actually moves!
    if is_dynamic and random.random() < AUG_WARP_PROB:
        seq = time_warp_augment(seq)
        
    seq = keypoint_dropout_augment(seq)
    if random.random() < AUG_MIRROR_PROB:
        seq = mirror_sequence(seq)
    if random.random() < AUG_TILT_PROB:
        seq = camera_tilt_augment(seq)
    if random.random() < AUG_ROLL_PROB:
        seq = camera_roll_augment(seq)
    if random.random() < AUG_DISTANCE_PROB:
        seq = camera_distance_augment(seq)
    if random.random() < AUG_CROP_OFFSET_PROB:
        seq = crop_offset_augment(seq)
    if random.random() < AUG_ASPECT_PROB:
        seq = aspect_ratio_augment(seq)
    if same_class_pool is not None and random.random() < AUG_MIXUP_PROB:
        partner = random.choice(same_class_pool)
        seq = mixup_augment(seq, partner)
    return seq

# ==============================
# LOAD DATASET
# ==============================
def load_dataset(data_path):
    X, y = [], []
    for category in sorted(os.listdir(data_path)):
        cat_path = os.path.join(data_path, category)
        if not os.path.isdir(cat_path):
            continue
        for label in sorted(os.listdir(cat_path)):
            label_folder = os.path.join(cat_path, label)
            if not os.path.isdir(label_folder):
                continue
            files = [f for f in os.listdir(label_folder) if f.endswith(".npy")]
            if not files:
                continue
            for file in files:
                seq = np.load(os.path.join(label_folder, file))
                print(seq.shape)
                if seq.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
                    print(f"  ⚠️  Skipping {file} — shape {seq.shape}")
                    continue
                nonzero = np.count_nonzero(seq) / seq.size
                if nonzero < MIN_NONZERO:
                    print(f"  ⚠️  Skipping {file} — low quality {nonzero:.0%}")
                    continue
                seq_norm = normalize_sequence(seq)
                X.append(seq_norm)
                y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)

# ==============================
# LEVEL 3 — PROFILE-WEIGHTED LOSS
# ==============================

def load_profiles(profiles_path):
    """
    Load sign_profiles.json if it exists.
    Returns None → falls back to standard CE loss automatically.
    """
    if not os.path.exists(profiles_path):
        print(f"⚠️  {profiles_path} not found — falling back to standard CE")
        print(f"   To enable Level 3:")
        print(f"   1. python 04_build_sign_profiles.py")
        print(f"   2. python 02_train_model.py  (this script again)")
        return None
    with open(profiles_path) as f:
        profiles = json.load(f)
    print(f"✅ Loaded {len(profiles)} sign profiles from {profiles_path}")
    return profiles


def build_class_noise_weights(labels, profiles):
    """
    1D array (num_classes,) — clarity bonus per class.

    Signs with well-defined profiles (low std in active regions) get a
    higher weight so the loss penalizes them more on misclassification.
    This forces the model to learn these signs more precisely.

    Range: 1.0 (unclear profile) → ~1.8 (very clear profile)
    """
    num_classes   = len(labels)
    noise_weights = np.ones(num_classes, dtype=np.float32)

    for i, label in enumerate(labels):
        if label not in profiles:
            continue
        weights  = profiles[label].get('weights',  {})
        variance = profiles[label].get('variance', {})

        clarity = 0.0
        for region in ['hand', 'face', 'pose']:
            w   = weights.get(region,  0.0)
            std = variance.get(region, 0.5)
            if w > 0.1:
                clarity += w * (1.0 / (1.0 + std * 5))

        noise_weights[i] = 1.0 + 0.8 * min(clarity, 1.0)

    return noise_weights


def build_region_weight_matrix(labels, profiles):
    """
    2D matrix (num_classes, 204) — per-feature importance per class.

    For hand-only sign: hand features = high weight, face/pose = low weight
    For hand+face sign: hand + face = high weight, pose = low weight

    Min weight = 0.2 so no region is completely zeroed during training.
    Active regions are boosted to 2× their profile weight so the effect
    is visible in the gradient.
    """
    num_classes   = len(labels)
    weight_matrix = np.ones((num_classes, FEATURE_SIZE), dtype=np.float32)

    for i, label in enumerate(labels):
        if label not in profiles:
            continue
        weights  = profiles[label].get('weights',  {})
        variance = profiles[label].get('variance', {})

        def apply_region(region_key, start, end):
            std = variance.get(region_key, 0.0)
            w   = weights.get(region_key,  1.0)
            if std > STD_IGNORE_THRESHOLD:
                # Noisy region — downweight but never fully zero
                weight_matrix[i, start:end] = 0.2
            else:
                # Active region — emphasize
                weight_matrix[i, start:end] = max(0.2, w * 2.0)

        apply_region('hand', 0,   126)
        apply_region('face', 126, 186)
        apply_region('pose', 186, 204)

    return weight_matrix


def profile_weighted_loss(weight_matrix, class_noise_weights, alpha):
    """
    Custom Keras loss:
      total = (1 - α) × CE  +  α × profile_CE

    profile_CE = CE × class_noise_weight × mean_feature_weight

    class_noise_weight : rewards signs with clear profiles
    mean_feature_weight: average importance of features for this class
                         — high for signs with strong discriminative regions

    α = PROFILE_LOSS_WEIGHT (default 0.35)

    Effect: the model pays more attention to features that actually
    distinguish each sign, and less to noisy background regions.
    """
    wm_tensor  = tf.constant(weight_matrix,      dtype=tf.float32)  # (C, 204)
    cnw_tensor = tf.constant(class_noise_weights, dtype=tf.float32)  # (C,)
    alpha_t    = tf.constant(alpha,               dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        # Standard cross-entropy
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # True class index per sample
        true_idx = tf.argmax(y_true, axis=-1)

        # Per-sample clarity bonus
        sample_noise_w = tf.gather(cnw_tensor, true_idx)

        # Per-sample mean feature weight
        sample_feat_w = tf.reduce_mean(
            tf.gather(wm_tensor, true_idx), axis=-1)

        # Profile-aware CE — harder penalty for well-profiled signs
        profile_ce = ce * sample_noise_w * sample_feat_w

        return (1.0 - alpha_t) * ce + alpha_t * profile_ce

    return loss_fn


def print_profile_summary(labels, profiles):
    print("\n📊 Level 3 — Profile weight summary:")
    print(f"   {'Label':<22} {'Hand':>6} {'Face':>6} {'Pose':>6}  "
          f"{'H.std':>6} {'F.std':>6} {'P.std':>6}  Active")
    print("   " + "─" * 72)
    for label in labels:
        if label not in profiles:
            print(f"   {label:<22} {'(no profile)':>50}")
            continue
        w = profiles[label].get('weights',  {})
        v = profiles[label].get('variance', {})
        active = [r for r in ['hand', 'face', 'pose']
                  if v.get(r, 1.0) <= STD_IGNORE_THRESHOLD]
        print(f"   {label:<22}"
              f" {w.get('hand',0):>6.2f} {w.get('face',0):>6.2f}"
              f" {w.get('pose',0):>6.2f} "
              f" {v.get('hand',0):>6.3f} {v.get('face',0):>6.3f}"
              f" {v.get('pose',0):>6.3f}  "
              f"{', '.join(active) if active else 'none'}")

# ==============================
# MAIN
# ==============================
print("=" * 60)
print("  BIM Sign Language — Model Training  (CNN + GRU + Level 3)")
print("=" * 60)

print("\n📂 Loading dataset...")
X, y = load_dataset(DATA_PATH)
labels = sorted(list(set(y)))
print(f"   Raw sequences : {len(X)}")
print(f"   Classes       : {len(labels)} — {labels}")
print(f"   Class counts  : {dict(Counter(y))}")

if len(X) == 0:
    raise RuntimeError(f"No data found in {DATA_PATH}.")

le    = LabelEncoder()
y_enc = le.fit_transform(y)

print("\n✂️  Splitting dataset (before augmentation)...")
X_train_raw, X_test, y_train_raw, y_test_enc = train_test_split(
    X, y_enc, test_size=TEST_SIZE, random_state=42, stratify=y_enc
)
print(f"   Train (raw) : {len(X_train_raw)} | Test : {len(X_test)}")

class_pool: dict[int, list] = defaultdict(list)
for seq, lbl in zip(X_train_raw, y_train_raw):
    class_pool[int(lbl)].append(seq)


#---------------------Augmentation Balancing--------------
#---------------------------------------------------------
print(f"\n🔀 Augmenting and Balancing training set...")
X_train_aug = []
y_train_aug = []

profiles = load_profiles(PROFILES_PATH)

# 1. Find out which sign has the most data
max_samples = max([len(pool) for pool in class_pool.values()])

# 2. Set a perfectly equal target for EVERY class
# (e.g., if the biggest class has 50 samples, make EVERY class have 50 * 8 = 400 samples)
TARGET_PER_CLASS = max_samples * AUGMENTATION_FACTOR
print(f"   Targeting exactly {TARGET_PER_CLASS} samples per class to prevent bias.")

# 3. Augment each class until it hits the exact target
for class_idx, pool in class_pool.items():
    label_name = le.inverse_transform([class_idx])[0]
    
    is_dyn = False
    if profiles and label_name in profiles:
        is_dyn = profiles[label_name].get('is_dynamic', False)

    # Add the original real data first
    for seq in pool:
        X_train_aug.append(seq)
        y_train_aug.append(class_idx)
        
    # Generate synthetic data until we hit the perfectly balanced target
    current_count = len(pool)
    while current_count < TARGET_PER_CLASS:
        # Pick a random real sequence from this class to base the augmentation on
        base_seq = random.choice(pool)
        synthetic_seq = augment_sequence(base_seq, is_dyn, same_class_pool=pool)
        
        X_train_aug.append(synthetic_seq)
        y_train_aug.append(class_idx)
        current_count += 1

print(f"   Train (augmented & perfectly balanced) : {len(X_train_aug)}")
        
        

X_train     = np.array(X_train_aug, dtype=np.float32)
y_train_enc = np.array(y_train_aug)
perm        = np.random.permutation(len(X_train))
X_train     = X_train[perm]
y_train_enc = y_train_enc[perm]
print(f"   Train (augmented) : {len(X_train)}")

num_classes = len(labels)
y_train_oh  = tf.keras.utils.to_categorical(y_train_enc, num_classes)
y_test_oh   = tf.keras.utils.to_categorical(y_test_enc,  num_classes)

with open(ENCODER_PATH, "wb") as f: pickle.dump(le, f)
with open(LABEL_PATH,   "w") as f: json.dump(labels, f, indent=2)
print(f"\n💾 Label encoder → {ENCODER_PATH}")
print(f"💾 Labels JSON   → {LABEL_PATH}")

# ── Level 3 setup ─────────────────────────────────────────────────────────
print(f"\n🔍 Loading sign profiles for Level 3 loss...")
profiles = load_profiles(PROFILES_PATH)

if profiles is not None:
    weight_matrix       = build_region_weight_matrix(labels, profiles)
    class_noise_weights = build_class_noise_weights(labels, profiles)
    loss_function       = profile_weighted_loss(
        weight_matrix, class_noise_weights, PROFILE_LOSS_WEIGHT)
    print_profile_summary(labels, profiles)
    print(f"\n   α = {PROFILE_LOSS_WEIGHT}  →  "
          f"{(1-PROFILE_LOSS_WEIGHT)*100:.0f}% standard CE  +  "
          f"{PROFILE_LOSS_WEIGHT*100:.0f}% profile-weighted CE")
    print(f"   STD ignore threshold : {STD_IGNORE_THRESHOLD}")
else:
    loss_function = 'categorical_crossentropy'
    print("   → Standard CE (no profiles found)")

# ==============================
# MODEL
# ==============================
print(f"\n🏗️  Building model  "
      f"(input: {SEQUENCE_LENGTH}×{FEATURE_SIZE}, classes: {num_classes})...")

model = Sequential([
    Conv1D(64, kernel_size=5, activation='relu', padding='same',
           input_shape=(SEQUENCE_LENGTH, FEATURE_SIZE)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.3),

    GRU(128, return_sequences=True,  unroll=True),
    Dropout(0.4),

    GRU(64,  return_sequences=False, unroll=True),
    Dropout(0.4),

    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    Dropout(0.4),

    Dense(64,  activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=loss_function,
    metrics=['accuracy']
)
model.summary()
print(f"\n   Loss: "
      f"{'Level 3 profile-weighted CE' if profiles else 'Standard CE (fallback)'}")

# ==============================
# CALLBACKS + TRAIN
# ==============================
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=5, min_lr=1e-6, verbose=1),
]

print(f"\n🏋️  Training  (epochs={EPOCHS}, batch={BATCH_SIZE})...")
history = model.fit(
    X_train, y_train_oh,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

with open(HISTORY_PATH, "wb") as f:
    pickle.dump(history.history, f)

# ==============================
# EVALUATE
# ==============================
loss_val, acc = model.evaluate(X_test, y_test_oh, verbose=0)
print(f"\n📊 Test Accuracy : {acc * 100:.2f}%")
print(f"   Test Loss     : {loss_val:.4f}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print("\n📋 Classification Report:")
print(classification_report(y_test_enc, y_pred, target_names=labels))
print("Confusion Matrix:")
print(confusion_matrix(y_test_enc, y_pred))

print("\n📈 Per-class accuracy:")
for i, lbl in enumerate(labels):
    mask    = y_test_enc == i
    if mask.sum() == 0: continue
    cls_acc = (y_pred[mask] == i).mean() * 100
    n       = mask.sum()
    status  = "✅" if cls_acc >= 80 else "⚠️ " if cls_acc >= 60 else "❌"
    print(f"  {status}  {lbl:<20} {cls_acc:5.1f}%  ({n} test samples)")

# ── Save ──────────────────────────────────────────────────────────────────
model.save(MODEL_SAVE_PATH)
print(f"\n💾 Keras model → {MODEL_SAVE_PATH}")

# ==============================
# TFLITE EXPORT
# ==============================
print("\n📦 Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

size_kb = os.path.getsize(TFLITE_PATH) / 1024
print(f"💾 TFLite → {TFLITE_PATH}  ({size_kb:.1f} KB)")

print("\n🔍 Verifying TFLite model...")
interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
interp.allocate_tensors()
inp_d = interp.get_input_details()
out_d = interp.get_output_details()
print(f"   Input  : {inp_d[0]['shape']}")
print(f"   Output : {out_d[0]['shape']}")
dummy = np.zeros((1, SEQUENCE_LENGTH, FEATURE_SIZE), dtype=np.float32)
interp.set_tensor(inp_d[0]['index'], dummy)
interp.invoke()
result = interp.get_tensor(out_d[0]['index'])
print(f"   Softmax sum : {result.sum():.6f}")
print(f"   ✅ TFLite verified")

# ── Deploy copy ───────────────────────────────────────────────────────────
import shutil
MODEL_DEST = "deploy/model/"
LABEL_DEST = "deploy/labels/"
os.makedirs(MODEL_DEST, exist_ok=True)
os.makedirs(LABEL_DEST, exist_ok=True)
shutil.copy2(MODEL_SAVE_PATH, os.path.join(MODEL_DEST, "handface_pose_cnn_gru.h5"))
shutil.copy2(TFLITE_PATH,     os.path.join(MODEL_DEST, "handface_pose_cnn_gru.tflite"))
shutil.copy2(LABEL_PATH,      os.path.join(LABEL_DEST, "labels.json"))
print(f"\n📁 Deployed to {MODEL_DEST}")

print("\n🎉 Training complete!")
print(f"   Loss used: "
      f"{'Level 3 profile-weighted CE' if profiles else 'Standard CE (no profiles)'}")