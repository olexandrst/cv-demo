"""Face → emotion + gender via ONNX (no TensorFlow / DeepFace).

Used by the profiling mode. Models are downloaded on first use into ./models.

Pipeline:
  1. Face detection — OpenCV YuNet (lightweight, ~230 KB ONNX).
  2. Emotion       — FER+ (8 classes), ONNX, ~35 MB.
  3. Gender        — Levi-Hassner Caffe model, ~45 MB.

Returned record format matches what detector.py expects:
  {"region": {"x", "y", "w", "h"},
   "dominant_emotion": <our short key>,
   "dominant_gender":  "Man" | "Woman" | ""}
"""
from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np

from downloads import fetch as _fetch_model

_FERPLUS_LABELS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt",
]
# Map FER+ classes -> short keys used by detector.py palette.
_EMOTION_KEY = {
    "neutral":   "neutral",
    "happiness": "happy",
    "surprise":  "surprise",
    "sadness":   "sad",
    "anger":     "angry",
    "disgust":   "disgust",
    "fear":      "fear",
    "contempt":  "angry",
}
# Levi-Hassner mean values (BGR).
_GENDER_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
_GENDER_LABELS = ["Man", "Woman"]


class FaceAnalyzer:
    """Lazy-loading ONNX-based face / emotion / gender analyzer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._face_detector: Optional["cv2.FaceDetectorYN"] = None
        self._emotion_session = None        # onnxruntime InferenceSession
        self._emotion_input: str = ""
        self._gender_net = None             # cv2.dnn.Net
        self._cached_size = (0, 0)

    # -------------------------------------------------------------- loaders
    def _ensure_face(self) -> "cv2.FaceDetectorYN":
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
        if self._emotion_session is not None:
            return self._emotion_session
        with self._lock:
            if self._emotion_session is None:
                import onnxruntime as ort  # heavy import, lazy
                p = _fetch_model("emotion-ferplus-8.onnx")
                self._emotion_session = ort.InferenceSession(
                    str(p), providers=["CPUExecutionProvider"]
                )
                self._emotion_input = self._emotion_session.get_inputs()[0].name
        return self._emotion_session

    def _ensure_gender(self):
        if self._gender_net is not None:
            return self._gender_net
        with self._lock:
            if self._gender_net is None:
                proto = _fetch_model("gender_deploy.prototxt")
                weights = _fetch_model("gender_net.caffemodel")
                self._gender_net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        return self._gender_net

    # -------------------------------------------------------------- inference
    def _predict_emotion(self, face_bgr: np.ndarray) -> str:
        try:
            sess = self._ensure_emotion()
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
            x = gray.astype(np.float32).reshape(1, 1, 64, 64)
            out = sess.run(None, {self._emotion_input: x})[0][0]
            idx = int(np.argmax(out))
            return _EMOTION_KEY.get(_FERPLUS_LABELS[idx], "neutral")
        except Exception:
            return "neutral"

    def _predict_gender(self, face_bgr: np.ndarray) -> str:
        try:
            net = self._ensure_gender()
            blob = cv2.dnn.blobFromImage(
                face_bgr, 1.0, (227, 227), _GENDER_MEAN, swapRB=False, crop=False
            )
            net.setInput(blob)
            preds = net.forward()
            idx = int(np.argmax(preds[0]))
            return _GENDER_LABELS[idx]
        except Exception:
            return ""

    # -------------------------------------------------------------- public
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
        for f in faces:
            x, y, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            # clamp + a small margin helps gender net (it was trained on padded faces)
            pad = int(0.15 * max(fw, fh))
            x1 = max(0, x - pad); y1 = max(0, y - pad)
            x2 = min(w, x + fw + pad); y2 = min(h, y + fh + pad)
            if x2 - x1 < 24 or y2 - y1 < 24:
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            results.append({
                "region": {"x": int(x), "y": int(y), "w": int(fw), "h": int(fh)},
                "dominant_emotion": self._predict_emotion(crop),
                "dominant_gender":  self._predict_gender(crop),
            })
        return results
