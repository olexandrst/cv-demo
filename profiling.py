"""Face → emotion + gender + age via OpenCV DNN / ONNX (no TensorFlow).

Used by the profiling mode.

Pipeline:
  1. Face detection — OpenCV YuNet (~230 KB ONNX). Gives bbox + 5 landmarks.
  2. Emotion       — FER+ (8 classes), ONNX, ~35 MB. We apply an
                     "anti-neutral" threshold to fight the model's
                     well-known bias toward labelling everything neutral —
                     if neutral isn't dominant (default <55%), the best
                     non-neutral class wins. This dramatically improves
                     responsiveness on a conference-stand setup where
                     visitors make exaggerated faces and want immediate
                     feedback.
  3. Gender        — Levi-Hassner Caffe model (~45 MB).
  4. Age           — Levi-Hassner Caffe model, 8 age buckets (~45 MB).

Output record format (matches what detector.py expects):
  {"region":           {"x", "y", "w", "h"},
   "dominant_emotion": <one of 7 keys>,
   "dominant_gender":  "Man" | "Woman" | "",
   "dominant_age":     "(25-32)" | …  | ""}
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from downloads import fetch as _fetch_model

# Set EMOTION_DEBUG=1 to print top-3 emotion probabilities + dump face crops
# to ./debug/face_*.jpg for visual inspection.
_EMOTION_DEBUG = os.environ.get("EMOTION_DEBUG", "0") not in ("", "0", "false", "False")
_DEBUG_DIR = Path(__file__).resolve().parent / "debug"

# Threshold: neutral wins only if its probability is at least this.
# Otherwise the best non-neutral class wins. 0.55 is a reasonable demo
# default — visitors making faces get instant feedback even when the
# model is uncertain. Tunable via env.
_NEUTRAL_THRESHOLD = float(os.environ.get("EMOTION_NEUTRAL_THRESHOLD", "0.55"))

# FER+ class order (from the onnx/models card).
_FERPLUS_LABELS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt",
]
_NEUTRAL_IDX = 0
# Map FER+ classes -> the short keys detector.py paints colours for.
_EMOTION_KEY = {
    "neutral":   "neutral",
    "happiness": "happy",
    "surprise":  "surprise",
    "sadness":   "sad",
    "anger":     "angry",
    "disgust":   "disgust",
    "fear":      "fear",
    "contempt":  "angry",   # fold contempt into angry for the UI
}

# Levi-Hassner mean values (BGR, 227×227 input).
_LH_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
_GENDER_LABELS = ["Man", "Woman"]
_AGE_LABELS = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)",
]


class FaceAnalyzer:
    """Lazy-loading ONNX/Caffe-based face / emotion / gender / age analyzer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._face_detector = None
        self._emotion_net = None      # cv2.dnn.Net (FER+ ONNX)
        self._gender_net = None       # cv2.dnn.Net (Caffe)
        self._age_net = None          # cv2.dnn.Net (Caffe)
        self._cached_size = (0, 0)
        # Per-face running average of emotion logits (key = rounded face centre).
        self._emotion_history: dict[tuple[int, int], np.ndarray] = {}

    # ------------------------------------------------------------- loaders
    def _ensure_face(self):
        if self._face_detector is not None:
            return self._face_detector
        with self._lock:
            if self._face_detector is None:
                p = _fetch_model("yunet.onnx")
                self._face_detector = cv2.FaceDetectorYN.create(
                    str(p), "", (320, 320), 0.6, 0.3, 5000
                )
        return self._face_detector

    def _ensure_emotion(self):
        if self._emotion_net is not None:
            return self._emotion_net
        with self._lock:
            if self._emotion_net is None:
                p = _fetch_model("emotion-ferplus-8.onnx")
                self._emotion_net = cv2.dnn.readNet(str(p))
        return self._emotion_net

    def _ensure_gender(self):
        if self._gender_net is not None:
            return self._gender_net
        with self._lock:
            if self._gender_net is None:
                proto = _fetch_model("gender_deploy.prototxt")
                weights = _fetch_model("gender_net.caffemodel")
                self._gender_net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        return self._gender_net

    def _ensure_age(self):
        if self._age_net is not None:
            return self._age_net
        with self._lock:
            if self._age_net is None:
                proto = _fetch_model("age_deploy.prototxt")
                weights = _fetch_model("age_net.caffemodel")
                self._age_net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        return self._age_net

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / np.sum(ex)

    # ------------------------------------------------------------- inference
    def _predict_emotion(self, face_bgr: np.ndarray, history_key: tuple[int, int]) -> str:
        try:
            net = self._ensure_emotion()
            # FER+ takes a single-channel 64×64 image, pixel range [0, 255]
            # as float32. No further normalisation.
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)  # boost contrast — helps on dim webcam frames
            gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
            x = gray.astype(np.float32).reshape(1, 1, 64, 64)
            net.setInput(x)
            logits = net.forward().flatten()
            probs = self._softmax(logits)

            # Temporal smoothing — exponential moving average. Bias toward
            # the latest reading so the demo feels responsive.
            prev = self._emotion_history.get(history_key)
            if prev is not None and prev.shape == probs.shape:
                probs = 0.3 * prev + 0.7 * probs
            self._emotion_history[history_key] = probs

            # Anti-neutral threshold: only call it neutral if neutral
            # *clearly* dominates. Otherwise pick the best non-neutral class.
            if probs[_NEUTRAL_IDX] >= _NEUTRAL_THRESHOLD:
                idx = _NEUTRAL_IDX
            else:
                non_neutral = probs.copy()
                non_neutral[_NEUTRAL_IDX] = -1.0
                idx = int(np.argmax(non_neutral))

            if _EMOTION_DEBUG:
                top3 = np.argsort(probs)[::-1][:3]
                msg = "  ".join(
                    f"{_FERPLUS_LABELS[i]}={probs[i]:.2f}" for i in top3
                )
                chosen = _FERPLUS_LABELS[idx]
                print(f"[emotion@{history_key}] {msg}  →  {chosen}", flush=True)

            label = _FERPLUS_LABELS[idx]
            return _EMOTION_KEY.get(label, "neutral")
        except Exception as exc:
            if _EMOTION_DEBUG:
                print(f"[emotion] error: {exc}", flush=True)
            return "neutral"

    def _predict_gender(self, face_bgr: np.ndarray) -> str:
        try:
            net = self._ensure_gender()
            blob = cv2.dnn.blobFromImage(
                face_bgr, 1.0, (227, 227), _LH_MEAN, swapRB=False, crop=False,
            )
            net.setInput(blob)
            preds = net.forward()[0]
            return _GENDER_LABELS[int(np.argmax(preds))]
        except Exception:
            return ""

    def _predict_age(self, face_bgr: np.ndarray) -> str:
        try:
            net = self._ensure_age()
            blob = cv2.dnn.blobFromImage(
                face_bgr, 1.0, (227, 227), _LH_MEAN, swapRB=False, crop=False,
            )
            net.setInput(blob)
            preds = net.forward()[0]
            return _AGE_LABELS[int(np.argmax(preds))]
        except Exception:
            return ""

    # ------------------------------------------------------------- public
    def analyze(self, frame_bgr: np.ndarray) -> list[dict]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        h, w = frame_bgr.shape[:2]
        det = self._ensure_face()
        if (w, h) != self._cached_size:
            det.setInputSize((w, h))
            self._cached_size = (w, h)
        try:
            _, faces = det.detect(frame_bgr)
        except Exception:
            return []
        results: list[dict] = []
        if faces is None:
            return results

        seen_keys: set[tuple[int, int]] = set()

        for f in faces:
            x, y, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            if fw < 24 or fh < 24:
                continue

            # Padded crop — gender/age (Levi-Hassner) was trained on padded faces,
            # FER+ benefits from a tighter crop. Compute both.
            pad_lh = int(0.20 * max(fw, fh))
            lx1 = max(0, x - pad_lh); ly1 = max(0, y - pad_lh)
            lx2 = min(w, x + fw + pad_lh); ly2 = min(h, y + fh + pad_lh)
            padded = frame_bgr[ly1:ly2, lx1:lx2]
            if padded.size == 0:
                continue

            pad_em = int(0.05 * max(fw, fh))
            ex1 = max(0, x - pad_em); ey1 = max(0, y - pad_em)
            ex2 = min(w, x + fw + pad_em); ey2 = min(h, y + fh + pad_em)
            em_crop = frame_bgr[ey1:ey2, ex1:ex2]
            if em_crop.size == 0:
                em_crop = padded

            # Use coarse face center as identity key for temporal smoothing.
            cx = (x + fw // 2) // 32 * 32
            cy = (y + fh // 2) // 32 * 32
            key = (int(cx), int(cy))
            seen_keys.add(key)

            if _EMOTION_DEBUG:
                _DEBUG_DIR.mkdir(exist_ok=True)
                tag = f"{key[0]:04d}_{key[1]:04d}"
                cv2.imwrite(str(_DEBUG_DIR / f"face_{tag}.jpg"), em_crop)

            results.append({
                "region": {"x": int(x), "y": int(y), "w": int(fw), "h": int(fh)},
                "dominant_emotion": self._predict_emotion(em_crop, key),
                "dominant_gender":  self._predict_gender(padded),
                "dominant_age":     self._predict_age(padded),
            })

        # Forget smoothing state for faces that disappeared.
        if seen_keys != set(self._emotion_history.keys()):
            self._emotion_history = {k: v for k, v in self._emotion_history.items() if k in seen_keys}

        return results
