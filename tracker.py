"""Lightweight MediaPipe hand tracker using the Tasks API (mediapipe 0.10.30+).

Downloads hand_landmarker.task on first run (~8 MB).
Returns the two wrist anchors sorted by screen-x when both hands are visible.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class Anchor:
    x: float   # normalized 0-1
    y: float   # normalized 0-1


@dataclass(frozen=True)
class HandPair:
    left:  Anchor   # anchor with smaller screen-x
    right: Anchor   # anchor with larger screen-x


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
        self._landmarker  = mp_vision.HandLandmarker.create_from_options(options)
        self._start_time  = time.perf_counter()

    def process(self, frame_bgr: np.ndarray) -> Optional[HandPair]:
        """Return HandPair if both hands visible, else None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.perf_counter() - self._start_time) * 1000)

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks or len(result.hand_landmarks) < 2:
            return None

        anchors = [
            Anchor(x=lm[0].x, y=lm[0].y)   # landmark 0 = wrist
            for lm in result.hand_landmarks
        ]
        anchors.sort(key=lambda a: a.x)     # positional left → right
        return HandPair(left=anchors[0], right=anchors[1])

    def close(self) -> None:
        self._landmarker.close()
