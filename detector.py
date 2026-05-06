"""Computer vision pipeline for the conference demo.

Modes:
  - detection: person detection with multi-coloured boxes
  - zones:     person detection + zone-of-interest intrusion alarm
  - ppe:       hard-hat / helmet detection on each person
  - profiling: face emotion + gender, colour-coded by mood
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Palette (BGR — OpenCV order). Red is reserved for alarm states.
# ---------------------------------------------------------------------------
PERSON_PALETTE_BGR = [
    (255, 200,   0),   # cyan
    (  0, 200, 255),   # orange
    (200, 100, 255),   # pink
    (  0, 255, 200),   # lime
    (255, 100, 100),   # light blue
    (100, 255,   0),   # bright green
    (255,   0, 200),   # magenta
    (  0, 200, 100),   # teal
]
RED_BGR     = (0, 0, 255)
GREEN_BGR   = (0, 200, 0)
GRAY_BGR    = (160, 160, 160)
WHITE_BGR   = (255, 255, 255)
BLACK_BGR   = (0, 0, 0)

EMOTION_COLORS_BGR = {
    "happy":    (0, 200, 0),       # green
    "sad":      (255, 80, 0),      # blue
    "angry":    (0, 0, 255),       # red
    "fear":     (0, 255, 255),     # yellow
    "neutral":  GRAY_BGR,
    "surprise": (0, 165, 255),     # orange
    "disgust":  (160, 32, 240),    # purple
}
EMOTION_UA = {
    "happy":    "веселий",
    "sad":      "сумний",
    "angry":    "сердитий",
    "fear":     "наляканий",
    "neutral":  "нормальний",
    "surprise": "здивований",
    "disgust":  "роздратований",
}
GENDER_UA = {"Man": "чоловік", "Woman": "жінка"}


# ---------------------------------------------------------------------------
# Cyrillic-aware text drawing (cv2 cannot render Ukrainian glyphs).
# ---------------------------------------------------------------------------
def _find_unicode_font() -> Optional[str]:
    candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


_FONT_PATH = _find_unicode_font()
_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


def _font(size: int) -> ImageFont.ImageFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    if _FONT_PATH:
        f = ImageFont.truetype(_FONT_PATH, size)
    else:
        f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


def draw_texts(frame_bgr: np.ndarray, items: list[tuple[str, tuple[int, int], tuple[int, int, int], int]]) -> np.ndarray:
    """Render multiple texts in one PIL pass.

    items: list of (text, (x, y), color_bgr, font_size).
    Returns new BGR ndarray.
    """
    if not items:
        return frame_bgr
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    for text, pos, color_bgr, size in items:
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        font = _font(size)
        # background pill for legibility
        try:
            bbox = draw.textbbox(pos, text, font=font)
            pad = 4
            draw.rounded_rectangle(
                (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
                radius=6,
                fill=(0, 0, 0, 180),
            )
        except Exception:
            pass
        draw.text(pos, text, fill=color_rgb, font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int], thickness: int = 3) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    # corner accents
    L = max(12, (x2 - x1) // 8)
    for (cx, cy, dx, dy) in [
        (x1, y1,  1,  1), (x2, y1, -1,  1),
        (x1, y2,  1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * L, cy), color, thickness + 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * L), color, thickness + 2, cv2.LINE_AA)


def draw_helmet_icon(frame: np.ndarray, cx: int, cy: int, color: tuple[int, int, int], crossed: bool = False, size: int = 28) -> None:
    """Schematic hard-hat. If `crossed` — draw a red diagonal line over it."""
    s = size
    # dome (semi-ellipse)
    cv2.ellipse(frame, (cx, cy), (s, int(s * 0.7)), 0, 180, 360, color, -1, cv2.LINE_AA)
    # brim
    cv2.rectangle(frame, (cx - s - 4, cy), (cx + s + 4, cy + 5), color, -1, cv2.LINE_AA)
    # ridge stripe
    cv2.line(frame, (cx - s + 4, cy - int(s * 0.5)), (cx + s - 4, cy - int(s * 0.5)), BLACK_BGR, 2, cv2.LINE_AA)
    if crossed:
        cv2.line(frame, (cx - s - 6, cy + 8), (cx + s + 6, cy - s - 4), RED_BGR, 4, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
@dataclass
class Profile:
    label_ua: str = ""
    color_bgr: tuple[int, int, int] = GRAY_BGR


@dataclass
class _ProfilingCache:
    faces: list = field(default_factory=list)   # list of dicts: region + dominant_emotion + dominant_gender
    timestamp: float = 0.0


class Detector:
    def __init__(self) -> None:
        self.mode: str = "detection"           # detection | zones | ppe | profiling
        self.zones: list[list[tuple[float, float]]] = []  # polygons in normalised coords
        self._person_model = None
        self._helmet_model = None
        self._face_analyzer = None             # profiling.FaceAnalyzer (lazy)
        self._models_lock = threading.Lock()
        self._profiling_cache = _ProfilingCache()
        self._profiling_thread: Optional[threading.Thread] = None
        self._profiling_busy = False
        self.errors: list[str] = []

    # -------- public API ----------------------------------------------------
    def set_mode(self, mode: str) -> None:
        if mode in ("detection", "zones", "ppe", "profiling"):
            self.mode = mode

    def set_zones(self, zones: list[list[tuple[float, float]]]) -> None:
        self.zones = zones or []

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        try:
            if self.mode == "detection":
                return self._mode_detection(frame_bgr)
            if self.mode == "zones":
                return self._mode_zones(frame_bgr)
            if self.mode == "ppe":
                return self._mode_ppe(frame_bgr)
            if self.mode == "profiling":
                return self._mode_profiling(frame_bgr)
        except Exception as exc:  # never break the stream
            self._note_error(f"process({self.mode}): {exc}")
        return frame_bgr

    # -------- model loaders -------------------------------------------------
    def _get_person_model(self):
        if self._person_model is not None:
            return self._person_model
        with self._models_lock:
            if self._person_model is None:
                from ultralytics import YOLO  # heavy import — lazy
                self._person_model = YOLO("yolov8n.pt")
        return self._person_model

    def _get_helmet_model(self):
        if self._helmet_model is not None:
            return self._helmet_model
        with self._models_lock:
            if self._helmet_model is None:
                from ultralytics import YOLO
                from huggingface_hub import hf_hub_download
                weights = hf_hub_download(
                    repo_id="keremberke/yolov8n-hard-hat-detection",
                    filename="best.pt",
                )
                self._helmet_model = YOLO(weights)
        return self._helmet_model

    # -------- person detection ---------------------------------------------
    def _detect_persons(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        model = self._get_person_model()
        # imgsz lower = faster; conf 0.4 keeps demo clean
        res = model.predict(frame_bgr, imgsz=480, conf=0.4, classes=[0], verbose=False)
        out: list[tuple[int, int, int, int, float]] = []
        if not res:
            return out
        r = res[0]
        if r.boxes is None:
            return out
        for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            out.append((x1, y1, x2, y2, float(conf)))
        return out

    # -------- mode: plain detection -----------------------------------------
    def _mode_detection(self, frame: np.ndarray) -> np.ndarray:
        persons = self._detect_persons(frame)
        texts: list = []
        for i, (x1, y1, x2, y2, conf) in enumerate(persons):
            color = PERSON_PALETTE_BGR[i % len(PERSON_PALETTE_BGR)]
            draw_box(frame, x1, y1, x2, y2, color)
            texts.append((f"людина #{i+1}  {conf*100:.0f}%", (x1 + 4, max(0, y1 - 28)), color, 18))
        frame = draw_texts(frame, texts)
        return frame

    # -------- mode: zones of interest ---------------------------------------
    def _mode_zones(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        persons = self._detect_persons(frame)

        # Convert zones -> pixel polygons
        zone_polys: list[np.ndarray] = []
        for poly in self.zones:
            if len(poly) < 3:
                continue
            pts = np.array([[int(x * w), int(y * h)] for x, y in poly], dtype=np.int32)
            zone_polys.append(pts)

        intruded = [False] * len(zone_polys)
        person_alarms = [False] * len(persons)

        for pi, (x1, y1, x2, y2, _conf) in enumerate(persons):
            # Use bottom-centre of bbox as person's position on the floor.
            foot = (int((x1 + x2) / 2), int(y2))
            for zi, poly in enumerate(zone_polys):
                if cv2.pointPolygonTest(poly, foot, False) >= 0:
                    intruded[zi] = True
                    person_alarms[pi] = True

        # draw zones
        overlay = frame.copy()
        for zi, poly in enumerate(zone_polys):
            color = RED_BGR if intruded[zi] else (0, 220, 255)  # yellow when calm
            cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        for zi, poly in enumerate(zone_polys):
            color = RED_BGR if intruded[zi] else (0, 220, 255)
            cv2.polylines(frame, [poly], True, color, 4, cv2.LINE_AA)

        # draw persons
        texts: list = []
        for i, (x1, y1, x2, y2, _c) in enumerate(persons):
            if person_alarms[i]:
                color = RED_BGR
                label = f"ТРИВОГА — людина #{i+1}"
            else:
                color = PERSON_PALETTE_BGR[i % len(PERSON_PALETTE_BGR)]
                label = f"людина #{i+1}"
            draw_box(frame, x1, y1, x2, y2, color)
            texts.append((label, (x1 + 4, max(0, y1 - 28)), color, 18))
        frame = draw_texts(frame, texts)
        return frame

    # -------- mode: PPE / hard-hat ------------------------------------------
    def _mode_ppe(self, frame: np.ndarray) -> np.ndarray:
        persons = self._detect_persons(frame)
        try:
            model = self._get_helmet_model()
            hres = model.predict(frame, imgsz=480, conf=0.35, verbose=False)
        except Exception as exc:
            self._note_error(f"helmet model: {exc}")
            return self._mode_detection(frame)

        # Build list of (head_box, has_helmet)
        heads: list[tuple[int, int, int, int, bool]] = []
        if hres and hres[0].boxes is not None:
            names = hres[0].names  # dict idx->name
            for box, cls_idx in zip(
                hres[0].boxes.xyxy.cpu().numpy(),
                hres[0].boxes.cls.cpu().numpy().astype(int),
            ):
                cls_name = str(names.get(int(cls_idx), "")).lower()
                x1, y1, x2, y2 = map(int, box)
                if "hardhat" in cls_name or "helmet" in cls_name:
                    has_helmet = not (
                        cls_name.startswith("no")
                        or "no-" in cls_name
                        or "no_" in cls_name
                        or "without" in cls_name
                    )
                    heads.append((x1, y1, x2, y2, has_helmet))

        texts: list = []
        for i, (px1, py1, px2, py2, _c) in enumerate(persons):
            # find the head whose centre lies in the upper region of this person
            top_pad = int((py2 - py1) * 0.10)
            upper_y = py1 + int((py2 - py1) * 0.45)
            best = None
            for hx1, hy1, hx2, hy2, has in heads:
                cx = (hx1 + hx2) // 2
                cy = (hy1 + hy2) // 2
                if px1 <= cx <= px2 and (py1 - top_pad) <= cy <= upper_y:
                    if best is None or (cy < best[2]):
                        best = (has, (hx1, hy1, hx2, hy2), cy)

            if best is None:
                color = GRAY_BGR
                draw_box(frame, px1, py1, px2, py2, color)
                texts.append(("шолом не визначено", (px1 + 4, max(0, py1 - 30)), color, 18))
                continue

            has_helmet = best[0]
            color = GREEN_BGR if has_helmet else RED_BGR
            draw_box(frame, px1, py1, px2, py2, color)

            # icon above the box
            ix = (px1 + px2) // 2
            iy = max(40, py1 - 14)
            draw_helmet_icon(frame, ix, iy, color, crossed=not has_helmet, size=22)

            label = "у шоломі" if has_helmet else "БЕЗ ШОЛОМА"
            texts.append((label, (px1 + 4, max(0, py1 - 30)), color, 18))

        frame = draw_texts(frame, texts)
        return frame

    # -------- mode: profiling (emotion + gender) ---------------------------
    def _mode_profiling(self, frame: np.ndarray) -> np.ndarray:
        persons = self._detect_persons(frame)
        self._maybe_run_profiling(frame)
        cache = self._profiling_cache.faces

        # match each face from cache to a person
        person_info: list[Optional[dict]] = [None] * len(persons)
        for face in cache:
            r = face.get("region", {})
            fx, fy, fw, fh = r.get("x", 0), r.get("y", 0), r.get("w", 0), r.get("h", 0)
            cx, cy = fx + fw // 2, fy + fh // 2
            for i, (px1, py1, px2, py2, _c) in enumerate(persons):
                if px1 <= cx <= px2 and py1 <= cy <= py2:
                    if person_info[i] is None:
                        person_info[i] = face
                    break

        texts: list = []
        for i, (px1, py1, px2, py2, _c) in enumerate(persons):
            info = person_info[i]
            if info:
                emotion = (info.get("dominant_emotion") or "neutral").lower()
                gender = info.get("dominant_gender") or ""
                color = EMOTION_COLORS_BGR.get(emotion, GRAY_BGR)
                gender_ua = GENDER_UA.get(gender, "")
                mood_ua = EMOTION_UA.get(emotion, emotion)
                lines = []
                if gender_ua:
                    lines.append(gender_ua)
                lines.append(f"настрій: {mood_ua}")
                label = " · ".join(lines)
            else:
                color = PERSON_PALETTE_BGR[i % len(PERSON_PALETTE_BGR)]
                label = f"людина #{i+1}"

            draw_box(frame, px1, py1, px2, py2, color)
            texts.append((label, (px1 + 4, max(0, py1 - 28)), color, 18))

        frame = draw_texts(frame, texts)
        return frame

    def _get_face_analyzer(self):
        if self._face_analyzer is not None:
            return self._face_analyzer
        with self._models_lock:
            if self._face_analyzer is None:
                from profiling import FaceAnalyzer  # lazy import
                self._face_analyzer = FaceAnalyzer()
        return self._face_analyzer

    def _maybe_run_profiling(self, frame: np.ndarray) -> None:
        # Throttle: rerun every 0.5s while previous worker is idle.
        now = time.time()
        if self._profiling_busy:
            return
        if now - self._profiling_cache.timestamp < 0.5:
            return
        snapshot = frame.copy()
        self._profiling_busy = True

        def _worker() -> None:
            try:
                analyzer = self._get_face_analyzer()
                results = analyzer.analyze(snapshot)
                self._profiling_cache = _ProfilingCache(faces=results, timestamp=time.time())
            except Exception as exc:
                self._note_error(f"profiling: {exc}")
            finally:
                self._profiling_busy = False

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        self._profiling_thread = t

    def _note_error(self, msg: str) -> None:
        self.errors.append(msg)
        if len(self.errors) > 20:
            self.errors = self.errors[-20:]
        print(f"[detector] {msg}", flush=True)
