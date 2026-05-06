"""Flask server: webcam capture + detection pipeline + MJPEG stream."""
from __future__ import annotations

# Relax HTTPS verification BEFORE any third-party HTTP client gets imported.
# Corporate networks often inject a TLS-intercepting CA that lacks fields
# Python 3.12+ now requires (Authority Key Identifier, revocation list, …).
# Without this, ultralytics / huggingface_hub fail to download model weights.
from downloads import relax_global_ssl, validate_cache
relax_global_ssl()
for _msg in validate_cache():
    print(f"[startup] {_msg}", flush=True)

import threading
import time
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from detector import Detector


# ---------------------------------------------------------------------------
# Webcam capture thread
# ---------------------------------------------------------------------------
class CameraStream:
    def __init__(self, src: int = 0, width: int = 1280, height: int = 720) -> None:
        self.src = src
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stopped = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "CameraStream":
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            # fallback without DSHOW
            self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        assert self.cap is not None
        while not self._stopped:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._stopped = True
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
app = Flask(__name__)
camera = CameraStream(src=0).start()
detector = Detector()


def _placeholder(text: str = "очікуємо камеру…") -> np.ndarray:
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(img, text, (40, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2, cv2.LINE_AA)
    return img


def _mjpeg_generator():
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    target_dt = 1.0 / 25.0
    last_t = 0.0
    while True:
        now = time.time()
        if now - last_t < target_dt:
            time.sleep(target_dt - (now - last_t))
        last_t = time.time()

        frame = camera.read()
        if frame is None:
            frame = _placeholder()
        else:
            frame = detector.process(frame)

        ok, buf = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            continue
        chunk = buf.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(chunk)).encode() + b"\r\n\r\n"
            + chunk + b"\r\n"
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/mode", methods=["POST"])
def set_mode():
    data = request.get_json(force=True) or {}
    mode = data.get("mode", "detection")
    detector.set_mode(mode)
    return jsonify(ok=True, mode=detector.mode)


@app.route("/api/zones", methods=["POST"])
def set_zones():
    data = request.get_json(force=True) or {}
    zones_in = data.get("zones", [])
    zones: list[list[tuple[float, float]]] = []
    for poly in zones_in:
        pts: list[tuple[float, float]] = []
        for p in poly:
            try:
                x = float(p[0]); y = float(p[1])
            except (TypeError, ValueError, IndexError):
                continue
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            pts.append((x, y))
        if len(pts) >= 3:
            zones.append(pts)
    detector.set_zones(zones)
    return jsonify(ok=True, zones=len(zones))


@app.route("/api/status")
def status():
    return jsonify(
        mode=detector.mode,
        zones=len(detector.zones),
        errors=detector.errors[-5:],
    )


if __name__ == "__main__":
    # threaded=True so /video_feed and /api/* don't block each other
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
