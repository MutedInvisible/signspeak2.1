"""Webcam capture + MediaPipe hands, running off the UI thread."""

from __future__ import annotations

import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from features import HAND_CONNECTIONS, landmarks_to_features

FrameCallback = Callable[[np.ndarray, np.ndarray | None], None]
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


def ensure_model() -> Path:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return MODEL_PATH
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def _draw_hands(frame: np.ndarray, hands) -> None:
    h, w = frame.shape[:2]
    for hand in hands:
        pts = hand.landmark if hasattr(hand, "landmark") else hand
        pixels = [(int(lm.x * w), int(lm.y * h)) for lm in pts]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pixels[a], pixels[b], (45, 212, 191), 2, cv2.LINE_AA)
        for x, y in pixels:
            cv2.circle(frame, (x, y), 4, (110, 231, 208), -1, cv2.LINE_AA)


class HandCamera:
    def __init__(self, on_frame: FrameCallback, camera_index: int = 0) -> None:
        self.on_frame = on_frame
        self.camera_index = camera_index
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self.one_hand_mode: bool = False

    def set_one_hand_mode(self, enabled: bool) -> None:
        self.one_hand_mode = enabled

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    def _make_landmarker(self):
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=RunningMode.VIDEO,
            num_hands=2,  # always detect up to 2; UI mode decides how to use them
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return HandLandmarker.create_from_options(options)

    def _run(self) -> None:
        self._cap = self._open()
        if not self._cap.isOpened():
            blank = np.zeros((480, 720, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                "Camera not found",
                (80, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (180, 180, 255),
                2,
            )
            self.on_frame(blank, None)
            return

        import mediapipe as mp

        landmarker = self._make_landmarker()
        stamp = 0

        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            stamp += 33
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, stamp)
            hands = result.hand_landmarks
            handed = result.handedness

            features = None
            if hands:
                features = landmarks_to_features(hands, handed, one_hand_mode=self.one_hand_mode)
                _draw_hands(frame, hands)

            self.on_frame(frame, features)

        try:
            landmarker.close()
        except Exception:
            pass
        self._cap.release()
        self._cap = None
