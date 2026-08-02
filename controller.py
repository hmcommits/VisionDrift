"""Steering + pedal state machine.

Converts a HandPair into key-press / key-release events, emitting only on
state transitions to avoid redundant input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import config
from tracker import HandPair


@dataclass
class ControlState:
    steer: str   = 'STRAIGHT'   # 'LEFT' | 'STRAIGHT' | 'RIGHT'
    pedal: str   = 'COAST'      # 'ACCEL' | 'COAST'   | 'BRAKE'
    angle: float = 0.0
    pinch: float = 0.5


class SteeringController:
    def __init__(self) -> None:
        self.state = ControlState()

    def update(self, pair: HandPair) -> Tuple[List[str], List[str]]:
        """Return (keys_to_press, keys_to_release) for this frame."""
        angle     = _angle(pair)
        pinch     = pair.right.pinch_norm
        new_steer = _steer(angle)
        new_pedal = _pedal(pinch)

        press:   List[str] = []
        release: List[str] = []

        if new_steer != self.state.steer:
            _steer_keys(self.state.steer, release)
            _steer_keys(new_steer, press)
            self.state.steer = new_steer

        if new_pedal != self.state.pedal:
            _pedal_keys(self.state.pedal, release)
            _pedal_keys(new_pedal, press)
            self.state.pedal = new_pedal

        self.state.angle = angle
        self.state.pinch = pinch
        return press, release

    def release_all(self) -> List[str]:
        """Release every currently held key and reset state."""
        held: List[str] = []
        _steer_keys(self.state.steer, held)
        _pedal_keys(self.state.pedal, held)
        self.state.steer = 'STRAIGHT'
        self.state.pedal = 'COAST'
        return held


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _angle(pair: HandPair) -> float:
    dx = pair.right.wrist.x - pair.left.wrist.x
    dy = pair.right.wrist.y - pair.left.wrist.y
    return math.degrees(math.atan2(dy, dx))


def _steer(a: float) -> str:
    if a < -config.DEADZONE_DEGREES: return 'LEFT'
    if a >  config.DEADZONE_DEGREES: return 'RIGHT'
    return 'STRAIGHT'


def _pedal(p: float) -> str:
    if p > config.THROTTLE_THRESHOLD: return 'ACCEL'
    if p < config.BRAKE_THRESHOLD:    return 'BRAKE'
    return 'COAST'


def _steer_keys(state: str, lst: List[str]) -> None:
    if state == 'LEFT':  lst.append(config.KEY_LEFT)
    if state == 'RIGHT': lst.append(config.KEY_RIGHT)


def _pedal_keys(state: str, lst: List[str]) -> None:
    if state == 'ACCEL': lst.append(config.KEY_ACCEL)
    if state == 'BRAKE': lst.append(config.KEY_BRAKE)
