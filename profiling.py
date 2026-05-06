"""Face → emotion + gender + age via OpenCV DNN / ONNX (no TensorFlow).

Used by the profiling mode.

Pipeline:
  1. Face detection — OpenCV YuNet (~230 KB ONNX). Gives bbox + 5 landmarks.
  2. Emotion       — opencv_zoo `facial_expression_recognition_mobilefacenet`
                     (~13 MB ONNX). 7 classes: angry, disgust, fear, happy,
                     neutral, sad, surprise. Run on a 5-landmark-aligned
                     112×112 face crop; predictions are smoothed across
                     consecutive runs to suppress jitter.
  3. Gender        — Levi-Hassner Caffe model (~45 MB).
  4. Age           — Levi-Hassner Caffe model, 8 age buckets (~45 MB).

Output record format (matches what detector.py expects):
  {"region":           {"x", "y", "w", "h"},
   "dominant_emotion": <one of 7 keys>,
   "dominant_gender":  "Man" | "Woman" | "",
   "dominant_age":     "(25-32)" | …  | ""}
"""
from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np

from downloads import fetch as _fetch_model

# Class order for facial_expression_recognition_mobilefacenet_2022july.onnx
_EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# 5-landmark template the alignment maps to (positions on a 112×112 canvas).
# Same landmark order as YuNet returns: right-eye, left-eye, nose, mouth-right, mouth-left.
_FACE_TEMPLATE_112 = np.array([
    [38.2946, 51.6963],   # right eye
    [73.5318, 51.5014],   # left eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # right mouth corner
    [70.7299, 92.2041],   # left mouth corner
], dtype=np.float32)

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
        self._emotion_net = None      # cv2.dnn.Net (ONNX)
        self._gender_net = None       # cv2.dnn.Net (Caffe)
        self._age_net = None          # cv2.dnn.Net (Caffe)
        self._cached_size = (0, 0)
        # Per-face running average of emotion logits.
        # Keys: rounded face center; values: (logits_avg, last_seen_frame).
        self._emotion_history: dict[tuple[int, int], np.ndarray] = {}
        self._frame_idx = 0

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
                p = _fetch_model("emotion_mobilefacenet.onnx")
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
    def _align_face(frame_bgr: np.ndarray, landmarks: np.ndarray) -> Optional[np.ndarray]:
        """Affine-warp face to a 112×112 canvas using 5 landmarks."""
        try:
            # estimateAffinePartial2D needs at least 3 points.
            M, _ = cv2.estimateAffinePartial2D(
                landmarks.astype(np.float32),
                _FACE_TEMPLATE_112,
            )
            if M is None:
                return None
            return cv2.warpAffine(
                frame_bgr, M, (112, 112), flags=cv2.INTER_LINEAR,
                borderValue=(0, 0, 0),
            )
        except cv2.error:
            return None

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / np.sum(ex)

    # ------------------------------------------------------------- inference
    def _predict_emotion(self, aligned_112: np.ndarray, history_key: tuple[int, int]) -> str:
        try:
            net = self._ensure_emotion()
            # opencv_zoo's mobilefacenet demo normalises to [-1, 1]:
            #   blob = (pixel/255 - 0.5) / 0.5  ==  (pixel - 127.5) / 127.5
            # Feeding it the [0, 1] range we used before made it collapse
            # to "neutral" for every face.
            blob = cv2.dnn.blobFromImage(
                aligned_112,
                scalefactor=1.0 / 127.5,
                size=(112, 112),
                mean=(127.5, 127.5, 127.5),
                swapRB=False,
                crop=False,
            )
            net.setInput(blob)
            logits = net.forward().flatten()
            probs = self._softmax(logits)

            # Temporal smoothing — exponential moving average.
            prev = self._emotion_history.get(history_key)
            if prev is not None and prev.shape == probs.shape:
                probs = 0.5 * prev + 0.5 * probs
            self._emotion_history[history_key] = probs

            idx = int(np.argmax(probs))
            return _EMOTION_LABELS[idx]
        except Exception:
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

        self._frame_idx += 1
        seen_keys: set[tuple[int, int]] = set()

        for f in faces:
            x, y, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            if fw < 24 or fh < 24:
                continue

            # Padded crop for gender/age (Levi-Hassner was trained on padded faces).
            pad = int(0.20 * max(fw, fh))
            cx1 = max(0, x - pad); cy1 = max(0, y - pad)
            cx2 = min(w, x + fw + pad); cy2 = min(h, y + fh + pad)
            padded = frame_bgr[cy1:cy2, cx1:cx2]
            if padded.size == 0:
                continue

            # Aligned 112x112 crop for emotion (uses YuNet's 5 landmarks).
            try:
                lms = np.array([
                    [f[4],  f[5]],   # right eye
                    [f[6],  f[7]],   # left eye
                    [f[8],  f[9]],   # nose tip
                    [f[10], f[11]],  # right mouth corner
                    [f[12], f[13]],  # left mouth corner
                ], dtype=np.float32)
                aligned = self._align_face(frame_bgr, lms)
            except Exception:
                aligned = None
            if aligned is None or aligned.size == 0:
                aligned = cv2.resize(padded, (112, 112), interpolation=cv2.INTER_AREA)

            # Use coarse face center as identity key for temporal smoothing.
            cx = (x + fw // 2) // 32 * 32
            cy = (y + fh // 2) // 32 * 32
            key = (int(cx), int(cy))
            seen_keys.add(key)

            results.append({
                "region": {"x": int(x), "y": int(y), "w": int(fw), "h": int(fh)},
                "dominant_emotion": self._predict_emotion(aligned, key),
                "dominant_gender":  self._predict_gender(padded),
                "dominant_age":     self._predict_age(padded),
            })

        # Forget smoothing state for faces that disappeared.
        if seen_keys != set(self._emotion_history.keys()):
            self._emotion_history = {k: v for k, v in self._emotion_history.items() if k in seen_keys}

        return results
