"""Lightweight MediaPipe hand tracker (Tasks API, mediapipe 0.10.30+).

Downloads hand_landmarker.task on first run (~8 MB).
Returns full per-hand landmark data (wrist, thumb tip, index tip, palm base)
for both hands, sorted by screen-x.
"""

from __future__ import annotations

import math
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np


_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _ensure_model() -> None:
    if not os.path.exists(_MODEL_PATH):
        print("[VisionDrift] Downloading hand landmark model (~8 MB) ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[VisionDrift] Model ready.")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Anchor:
    x: float   # normalized 0-1
    y: float   # normalized 0-1


@dataclass(frozen=True)
class HandData:
    wrist:      Anchor   # landmark  0
    thumb_tip:  Anchor   # landmark  4
    index_tip:  Anchor   # landmark  8
    palm_base:  Anchor   # landmark  9  (middle-finger MCP — used for scale)

    @property
    def pinch_norm(self) -> float:
        """Pinch distance (thumb–index) normalized by palm size (wrist–palm_base)."""
        raw   = math.hypot(self.thumb_tip.x - self.index_tip.x,
                           self.thumb_tip.y - self.index_tip.y)
        scale = math.hypot(self.wrist.x     - self.palm_base.x,
                           self.wrist.y     - self.palm_base.y)
        return raw / scale if scale > 1e-6 else 0.0


@dataclass(frozen=True)
class HandPair:
    left:  HandData   # smaller screen-x
    right: HandData   # larger screen-x


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class HandTracker:
    def __init__(self) -> None:
        _ensure_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._t0         = time.perf_counter()

    def process(self, frame_bgr: np.ndarray) -> Optional[HandPair]:
        """Return HandPair if both hands visible, else None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms  = int((time.perf_counter() - self._t0) * 1000)

        result = self._landmarker.detect_for_video(mp_img, ts_ms)
        if not result.hand_landmarks or len(result.hand_landmarks) < 2:
            return None

        hands = [_extract(lm) for lm in result.hand_landmarks]
        hands.sort(key=lambda h: h.wrist.x)
        return HandPair(left=hands[0], right=hands[1])

    def close(self) -> None:
        self._landmarker.close()


def _extract(landmarks) -> HandData:
    def a(i): return Anchor(x=landmarks[i].x, y=landmarks[i].y)
    return HandData(wrist=a(0), thumb_tip=a(4), index_tip=a(8), palm_base=a(9))
