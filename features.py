"""Turn MediaPipe hand landmarks into a pose-invariant feature vector.

Supports:
- 1-hand / 2-hand mode
- Static pose features (126-d)
- Motion aggregate features (mean + std + delta) for moving signs
"""

from __future__ import annotations

import numpy as np

LANDMARKS_PER_HAND = 21
COORDS = 3
HANDS = 2
POSE_SIZE = LANDMARKS_PER_HAND * COORDS * HANDS  # 126
# Motion: mean(126) + std(126) + delta(126) = 378
MOTION_SIZE = POSE_SIZE * 3
FEATURE_SIZE = POSE_SIZE  # default static size; motion uses MOTION_SIZE


def _points(hand) -> np.ndarray:
    pts = hand.landmark if hasattr(hand, "landmark") else hand
    return np.array([[lm.x, lm.y, lm.z] for lm in pts], dtype=np.float32)


def _normalize_hand(xyz: np.ndarray) -> np.ndarray:
    """xyz shape (21, 3). Center on wrist and scale by palm size."""
    wrist = xyz[0].copy()
    centered = xyz - wrist
    scale = np.linalg.norm(centered[9])  # middle-finger MCP
    if scale < 1e-6:
        scale = np.max(np.linalg.norm(centered, axis=1)) or 1.0
    return (centered / scale).reshape(-1)


def _hand_label(hand_info, index: int) -> str:
    if not hand_info:
        return "Right"
    item = hand_info[index]
    if hasattr(item, "classification"):
        return item.classification[0].label
    if isinstance(item, (list, tuple)):
        cat = item[0]
        return getattr(cat, "category_name", None) or getattr(cat, "display_name", "Right")
    return getattr(item, "category_name", None) or "Right"


def landmarks_to_features(
    multi_hand_landmarks,
    handedness,
    one_hand_mode: bool = False,
) -> np.ndarray | None:
    """Build a 126-d vector: left hand then right hand. Missing hands are zeros.

    When one_hand_mode=True we keep only the first / right-most detected hand
    and zero the other side so the model stays consistent.
    """
    if not multi_hand_landmarks:
        return None

    left = np.zeros(LANDMARKS_PER_HAND * COORDS, dtype=np.float32)
    right = np.zeros(LANDMARKS_PER_HAND * COORDS, dtype=np.float32)

    for i, landmarks in enumerate(multi_hand_landmarks):
        vec = _normalize_hand(_points(landmarks))
        label = _hand_label(handedness, i)
        if str(label).lower().startswith("left"):
            left = vec
        else:
            right = vec

    if one_hand_mode:
        # Prefer right hand; if only left is present keep left and zero right
        if np.any(right):
            left = np.zeros_like(left)
        # else keep left, right already zero

    return np.concatenate([left, right]).astype(np.float32)


def sequence_to_motion_features(seq: list[np.ndarray]) -> np.ndarray | None:
    """Turn a short sequence of pose vectors into a fixed-size motion feature.

    mean + std + (last - first)  → 378-d
    """
    if not seq:
        return None
    arr = np.stack([s for s in seq if s is not None and s.shape == (POSE_SIZE,)])
    if len(arr) < 3:
        return None
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    delta = arr[-1] - arr[0]
    return np.concatenate([mean, std, delta]).astype(np.float32)


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)
