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
    # FER+ emotion classifier (8 classes, ~35 MB). Less accurate than
    # mobilefacenet on paper but doesn't collapse to "neutral" on every
    # face — combined with our anti-neutral threshold it actually works.
    "emotion-ferplus-8.onnx": (
        "https://github.com/onnx/models/raw/main/validated/vision/"
        "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
    ),
    # Levi-Hassner gender model. spmallick/learnopencv only ships a
    # downloader script that pulls from Dropbox; smahesh29/Gender-and-Age-Detection
    # mirrors the prototxt + caffemodel directly in the repo.
    "gender_deploy.prototxt": (
        "https://github.com/smahesh29/Gender-and-Age-Detection/"
        "raw/master/gender_deploy.prototxt"
    ),
    "gender_net.caffemodel": (
        "https://github.com/smahesh29/Gender-and-Age-Detection/"
        "raw/master/gender_net.caffemodel"
    ),
    # Levi-Hassner age model (8 age buckets), same mirror.
    "age_deploy.prototxt": (
        "https://github.com/smahesh29/Gender-and-Age-Detection/"
        "raw/master/age_deploy.prototxt"
    ),
    "age_net.caffemodel": (
        "https://github.com/smahesh29/Gender-and-Age-Detection/"
        "raw/master/age_net.caffemodel"
    ),
}

# Minimum size to consider a download "complete" (sanity check vs HTML error pages).
_MIN_SIZE = {
    "yolov8n.pt":                     5_000_000,
    "yolov8n-hardhat.pt":             5_000_000,
    "yunet.onnx":                       100_000,
    "emotion-ferplus-8.onnx":        30_000_000,
    "gender_deploy.prototxt":             1_000,
    "gender_net.caffemodel":         40_000_000,
    "age_deploy.prototxt":                1_000,
    "age_net.caffemodel":            40_000_000,
}

# Magic-number prefixes (first few bytes) we expect to see for each format.
# Used to reject HTML block-pages saved as ".pt", etc.
_MAGIC = {
    ".pt":          (b"PK",),                    # PyTorch checkpoint = ZIP
    ".onnx":        (b"\x08", b"\x0a"),          # protobuf varint tags
    ".caffemodel":  (b"\x08", b"\x0a", b"\x12"),
    ".prototxt":    (b"name", b"layer", b"inp", b"#"),  # text protobuf
}


def _insecure_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _looks_valid(path: Path) -> tuple[bool, str]:
    """Returns (ok, reason). Only rejects files that look like HTML proxy
    block-pages — we can't reliably know every binary format's exact magic
    bytes, and false-positives here would erase a working model file."""
    try:
        with open(path, "rb") as f:
            head = f.read(512)
    except OSError as exc:
        return False, f"can't read: {exc}"
    head_lower = head.lower().lstrip()
    if (
        head_lower.startswith(b"<!doctype")
        or head_lower.startswith(b"<html")
        or head_lower.startswith(b"<?xml")
        or b"<head" in head_lower[:300]
        or b"<title" in head_lower[:300]
        or b"<body" in head_lower[:300]
    ):
        return False, "file is HTML (proxy/captive-portal page)"
    return True, "ok"


def _download(url: str, dest: Path, attempts: int = 3) -> None:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "cv-demo/1.0 (+https://example.local)"},
            )
            with urllib.request.urlopen(req, context=_insecure_ctx(), timeout=120) as r:
                ctype = (r.headers.get("Content-Type") or "").lower()
                if "html" in ctype:
                    raise RuntimeError(
                        f"server returned HTML (Content-Type: {ctype}); "
                        f"corporate proxy is likely intercepting the request"
                    )
                with open(tmp, "wb") as f:
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
    """Return a path to the local weight file, downloading if needed.

    If a previously-cached file is corrupted (e.g. an HTML block-page saved
    as ``.pt``), it is deleted and re-downloaded. If automatic download
    fails, the raised error tells the user exactly where to drop the file
    by hand.
    """
    if name not in URLS:
        raise KeyError(f"unknown model name: {name}")
    p = MODELS_DIR / name
    min_size = _MIN_SIZE.get(name, 1_000)

    # Validate any pre-existing file; remove if it's bogus.
    if p.exists():
        ok, reason = _looks_valid(p)
        if not ok or p.stat().st_size < min_size:
            print(
                f"[downloads] discarding cached {name} — invalid: {reason} "
                f"({p.stat().st_size:,} bytes)",
                flush=True,
            )
            try:
                p.unlink()
            except OSError:
                pass
        else:
            return p

    print(f"[downloads] fetching {name} -> {p}", flush=True)
    try:
        _download(URLS[name], p)
    except Exception as exc:
        raise RuntimeError(
            f"could not download {name} from {URLS[name]} "
            f"({exc}). Download it manually and place it as: {p}"
        ) from exc

    if p.stat().st_size < min_size:
        try:
            p.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"downloaded {name} is too small ({p.stat().st_size} bytes) — "
            f"the network probably returned a captive-portal page. "
            f"Download it manually from {URLS[name]} and place it as: {p}"
        )
    ok, reason = _looks_valid(p)
    if not ok:
        try:
            p.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"downloaded {name} is not a valid model file ({reason}). "
            f"Download it manually from {URLS[name]} and place it as: {p}"
        )
    print(f"[downloads]   ok ({p.stat().st_size:,} bytes)", flush=True)
    return p


def validate_cache() -> list[str]:
    """Sweep ./models/ on startup. Only deletes files that are obvious
    HTML proxy block-pages — never touches anything that might be a real
    weight file, even if it's smaller than expected. False-positives here
    silently break detection."""
    messages: list[str] = []
    for name in URLS:
        p = MODELS_DIR / name
        if not p.exists():
            continue
        ok, reason = _looks_valid(p)
        if not ok:
            size = p.stat().st_size if p.exists() else 0
            try:
                p.unlink()
                messages.append(f"{name}: removed (invalid — {reason}, {size} bytes)")
            except OSError as exc:
                messages.append(f"{name}: invalid but couldn't delete: {exc}")
    return messages


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
