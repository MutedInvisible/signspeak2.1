"""Store labeled samples, train a classifier, and predict live signs.

Supports static poses (126-d) and motion signs (378-d aggregates).
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_SIZE, MOTION_SIZE, POSE_SIZE, sequence_to_motion_features

DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLES_PATH = DATA_DIR / "samples.npz"
MODEL_PATH = DATA_DIR / "model.joblib"
META_PATH = DATA_DIR / "meta.json"


class SignTrainer:
    def __init__(self) -> None:
        self.X: list[np.ndarray] = []
        self.y: list[str] = []
        self.is_motion: list[bool] = []  # parallel flag per sample
        self.model: Pipeline | None = None
        self.motion_model: Pipeline | None = None
        self._history: deque[str] = deque(maxlen=8)
        self.load()

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for label in self.y:
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))

    def labels(self) -> list[str]:
        return list(self.counts().keys())

    def add_samples(self, label: str, samples: list[np.ndarray], motion: bool = False) -> int:
        name = label.strip().lower()
        if not name or not samples:
            return 0
        added = 0
        if motion:
            # samples is a list of pose vectors → turn into one motion feature
            feat = sequence_to_motion_features(samples)
            if feat is not None and feat.shape == (MOTION_SIZE,):
                self.X.append(feat.astype(np.float32))
                self.y.append(name)
                self.is_motion.append(True)
                added = 1
        else:
            for sample in samples:
                if sample is None or sample.shape != (POSE_SIZE,):
                    continue
                self.X.append(sample.astype(np.float32))
                self.y.append(name)
                self.is_motion.append(False)
                added += 1
        self.save_samples()
        return added

    def delete_label(self, label: str) -> None:
        name = label.strip().lower()
        kept_x, kept_y, kept_m = [], [], []
        for x, y, m in zip(self.X, self.y, self.is_motion):
            if y != name:
                kept_x.append(x)
                kept_y.append(y)
                kept_m.append(m)
        self.X, self.y, self.is_motion = kept_x, kept_y, kept_m
        self.model = None
        self.motion_model = None
        self.save_samples()
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()

    def can_train(self, motion: bool = False) -> bool:
        flags = [m for m in self.is_motion if m == motion]
        ys = [y for y, m in zip(self.y, self.is_motion) if m == motion]
        return len(set(ys)) >= 2 and len(ys) >= 10

    def train(self, motion: bool = False) -> dict:
        if not self.can_train(motion=motion):
            raise ValueError(
                "Need at least 2 signs and 10 samples "
                + ("(motion mode)" if motion else "(static mode)")
                + " to train."
            )

        X_list = [x for x, m in zip(self.X, self.is_motion) if m == motion]
        y_list = [y for y, m in zip(self.y, self.is_motion) if m == motion]
        X = np.stack(X_list)
        y = np.array(y_list)
        n_labels = len(set(y_list))

        forest = RandomForestClassifier(
            n_estimators=220,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", forest),
            ]
        )
        pipe.fit(X, y)
        acc = float(pipe.score(X, y))

        if motion:
            self.motion_model = pipe
        else:
            self.model = pipe

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Save both models if present
        payload = {
            "static": self.model,
            "motion": self.motion_model,
        }
        joblib.dump(payload, MODEL_PATH)
        META_PATH.write_text(
            json.dumps(
                {
                    "train_accuracy": acc,
                    "n_samples": int(len(y)),
                    "labels": sorted(set(y_list)),
                    "mode": "motion" if motion else "static",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"accuracy": acc, "samples": len(y), "signs": n_labels, "mode": "motion" if motion else "static"}

    def predict(self, features: np.ndarray | None, motion: bool = False) -> tuple[str | None, float]:
        model = self.motion_model if motion else self.model
        expected = MOTION_SIZE if motion else POSE_SIZE
        if model is None or features is None or features.shape != (expected,):
            self._history.clear()
            return None, 0.0

        proba = model.predict_proba([features])[0]
        idx = int(np.argmax(proba))
        label = str(model.classes_[idx])
        confidence = float(proba[idx])

        self._history.append(label)
        if self._history.count(label) < max(3, len(self._history) // 2):
            return None, confidence
        return label, confidence

    def save_samples(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not self.X:
            if SAMPLES_PATH.exists():
                SAMPLES_PATH.unlink()
            return
        # pad different sizes? store as object array
        np.savez_compressed(
            SAMPLES_PATH,
            X=np.array(self.X, dtype=object),
            y=np.array(self.y),
            is_motion=np.array(self.is_motion),
        )

    def load(self) -> None:
        if SAMPLES_PATH.exists():
            blob = np.load(SAMPLES_PATH, allow_pickle=True)
            self.X = [np.asarray(row, dtype=np.float32) for row in blob["X"]]
            self.y = [str(label) for label in blob["y"]]
            if "is_motion" in blob:
                self.is_motion = [bool(m) for m in blob["is_motion"]]
            else:
                # legacy static-only data
                self.is_motion = [False] * len(self.y)
        if MODEL_PATH.exists():
            try:
                payload = joblib.load(MODEL_PATH)
                if isinstance(payload, dict):
                    self.model = payload.get("static")
                    self.motion_model = payload.get("motion")
                else:
                    # old single-model format
                    self.model = payload
                    self.motion_model = None
            except Exception:
                self.model = None
                self.motion_model = None
