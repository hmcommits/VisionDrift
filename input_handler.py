"""PyDirectInput wrapper — sends hardware-level scan codes the OS can't block."""

from __future__ import annotations
from typing import List

import pydirectinput

pydirectinput.PAUSE = 0.0   # remove the default artificial delay between events


class InputHandler:
    def apply(self, to_press: List[str], to_release: List[str]) -> None:
        for k in to_release: pydirectinput.keyUp(k)
        for k in to_press:   pydirectinput.keyDown(k)

    def release_all(self, keys: List[str]) -> None:
        for k in keys: pydirectinput.keyUp(k)
