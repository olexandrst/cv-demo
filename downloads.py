"""All model downloads, SSL-tolerant.

Corporate networks frequently terminate TLS with their own CA that doesn't
satisfy Python's default certificate chain (you'll see errors like
"Missing Authority Key Identifier" or "CRYPT_E_NO_REVOCATION_CHECK").

We work around that by using an unverified SSL context for these specific
downloads of public, well-known model weights. This is a deliberate, scoped
trade-off: weights are large, public, and our app simply can't function
without them — and the data is read-only.

If a file already exists in ./models/ we skip the download. So you can also
manually drop a weight file in there to bypass the network entirely.
"""
from __future__ import annotations

import os
import shutil
import ssl
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

URLS: dict[str, str] = {
    "yolov8n.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
    ),
    "yolov8n-hardhat.pt": (
        "https://huggingface.co/keremberke/yolov8n-hard-hat-detection/"
        "resolve/main/best.pt"
    ),
    "yunet.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    "emotion-ferplus-8.onnx": (
        "https://github.com/onnx/models/raw/main/validated/vision/"
        "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
    ),
    "gender_deploy.prototxt": (
        "https://github.com/spmallick/learnopencv/raw/master/"
        "AgeGender/gender_deploy.prototxt"
    ),
    "gender_net.caffemodel": (
        "https://github.com/spmallick/learnopencv/raw/master/"
        "AgeGender/gender_net.caffemodel"
    ),
}

# Minimum size to consider a download "complete" (sanity check vs HTML error pages).
_MIN_SIZE = {
    "yolov8n.pt":              5_000_000,
    "yolov8n-hardhat.pt":      5_000_000,
    "yunet.onnx":                100_000,
    "emotion-ferplus-8.onnx": 30_000_000,
    "gender_deploy.prototxt":      1_000,
    "gender_net.caffemodel":  40_000_000,
}


def _insecure_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _download(url: str, dest: Path, attempts: int = 3) -> None:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "cv-demo/1.0 (+https://example.local)"},
            )
            with urllib.request.urlopen(req, context=_insecure_ctx(), timeout=120) as r, \
                 open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1024 * 256)
            tmp.replace(dest)
            return
        except Exception as exc:
            last_exc = exc
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            print(f"[downloads] attempt {attempt}/{attempts} failed: {exc}", flush=True)
    raise RuntimeError(f"failed to download {url}: {last_exc}")


def fetch(name: str) -> Path:
    """Return a path to the local weight file, downloading if needed."""
    if name not in URLS:
        raise KeyError(f"unknown model name: {name}")
    p = MODELS_DIR / name
    min_size = _MIN_SIZE.get(name, 1_000)
    if p.exists() and p.stat().st_size >= min_size:
        return p
    print(f"[downloads] fetching {name} -> {p}", flush=True)
    _download(URLS[name], p)
    if p.stat().st_size < min_size:
        raise RuntimeError(
            f"downloaded {name} is too small ({p.stat().st_size} bytes); "
            f"the network probably injected a captive-portal page"
        )
    print(f"[downloads]   ok ({p.stat().st_size:,} bytes)", flush=True)
    return p


def relax_global_ssl() -> None:
    """Best-effort: tell every HTTPS client in this process to skip cert checks.

    Used as a safety net for libraries that initiate their own downloads
    (ultralytics auto-update probes, huggingface_hub, requests, …).
    """
    os.environ.setdefault("PYTHONHTTPSVERIFY", "0")
    os.environ.setdefault("CURL_CA_BUNDLE", "")
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
    os.environ.setdefault("SSL_CERT_FILE", "")
    os.environ.setdefault("YOLO_OFFLINE", "True")
    try:
        ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]
    except Exception:
        pass
