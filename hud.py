"""All OpenCV drawing code: steering wheel, key indicators, and status text."""

from __future__ import annotations

import math

import cv2
import numpy as np

from tracker import HandPair
from controller import ControlState

# ── Palette (BGR) ─────────────────────────────────────────────────────────
_GLOW_DIM   = (155, 135,  90)
_GLOW_MID   = (215, 205, 165)
_RIM        = (248, 245, 238)
_INNER_RNG  = (190, 186, 175)
_SPOKE_FLL  = (230, 226, 218)
_SPOKE_EDG  = (248, 245, 238)
_GOLD       = (  0, 165, 255)   # RGB 255,165,  0
_GOLD_GLOW  = (  0,  80, 170)
_GRIP_ACT   = (  0, 240, 120)   # active grip arc
_GRIP_DIM   = ( 40,  80,  60)   # inactive grip arc
_HUB_FILL   = (210, 207, 200)
_GREEN      = (  0, 210,  80)   # accel indicator
_RED        = ( 50,  50, 230)   # brake indicator
_CYAN       = (255, 210,   0)   # title accent


# ---------------------------------------------------------------------------
# Steering wheel — geometry helpers
# ---------------------------------------------------------------------------

_CUT_HALF = 0.58   # radians each side of the flat bottom (~33°)

def _dcut_pts(cx: int, cy: int, r: int, a: float, n: int = 90) -> np.ndarray:
    """Polygon points for a D-cut circle. Flat section at 6-o'clock of wheel."""
    bottom    = a + math.pi / 2          # screen angle of wheel's 6-o'clock
    arc_start = bottom + _CUT_HALF       # right edge of flat gap
    arc_span  = 2 * math.pi - 2 * _CUT_HALF   # long arc (the D shape)
    pts = [
        (cx + r * math.cos(arc_start + arc_span * i / n),
         cy + r * math.sin(arc_start + arc_span * i / n))
        for i in range(n + 1)
    ]
    return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))


def _spoke(ov: np.ndarray, cx: int, cy: int, theta: float,
           r0: int, r1: int) -> None:
    """Wide tapered spoke with a structural cross-bar."""
    dx, dy = math.cos(theta), math.sin(theta)
    px, py = -math.sin(theta), math.cos(theta)
    w0, w1 = 11, 5   # half-width at hub / rim end

    pts = np.array([
        [cx + r0*dx + w0*px,  cy + r0*dy + w0*py],
        [cx + r0*dx - w0*px,  cy + r0*dy - w0*py],
        [cx + r1*dx - w1*px,  cy + r1*dy - w1*py],
        [cx + r1*dx + w1*px,  cy + r1*dy + w1*py],
    ], dtype=np.int32)
    cv2.fillPoly(ov, [pts], _SPOKE_FLL)
    cv2.polylines(ov, [pts], True, _SPOKE_EDG, 1, cv2.LINE_AA)

    # Cross-bar at 55 % of spoke length
    t   = 0.55
    crx = cx + (r0 + t * (r1 - r0)) * dx
    cry = cy + (r0 + t * (r1 - r0)) * dy
    chw = int(w0 + t * (w1 - w0)) + 7   # slightly wider than the spoke there
    cv2.line(ov,
             (int(crx + chw * px), int(cry + chw * py)),
             (int(crx - chw * px), int(cry - chw * py)),
             _SPOKE_EDG, 2, cv2.LINE_AA)


def _hub_pts(cx: int, cy: int, r_hub: int, a: float, n: int = 8) -> np.ndarray:
    """Octagonal hub polygon."""
    pts = [
        (int(cx + r_hub * math.cos(a + math.radians(i * 360 / n + 22.5))),
         int(cy + r_hub * math.sin(a + math.radians(i * 360 / n + 22.5))))
        for i in range(n)
    ]
    return np.array(pts, dtype=np.int32)


# ---------------------------------------------------------------------------
# Steering wheel — main draw call
# ---------------------------------------------------------------------------

def draw_steering_wheel(frame: np.ndarray, pair: HandPair,
                        state: ControlState) -> None:
    h, w = frame.shape[:2]

    lx = int(pair.left.wrist.x  * w);  ly = int(pair.left.wrist.y  * h)
    rx = int(pair.right.wrist.x * w);  ry = int(pair.right.wrist.y * h)

    cx = (lx + rx) // 2
    cy = (ly + ry) // 2
    r  = int(math.hypot(rx - lx, ry - ly) * 0.38)

    if r < 36:
        return

    a     = math.radians(state.angle)
    r_hub = 24

    ov = frame.copy()

    # ── D-cut rim — three-layer glow ───────────────────────────────────────
    rim = _dcut_pts(cx, cy, r, a)
    cv2.polylines(ov, [rim], True, _GLOW_DIM, 20, cv2.LINE_AA)
    cv2.polylines(ov, [rim], True, _GLOW_MID,  8, cv2.LINE_AA)
    cv2.polylines(ov, [rim], True, _RIM,        2, cv2.LINE_AA)

    # Inner grip channel (D-cut at r-9, thin accent line)
    inner = _dcut_pts(cx, cy, r - 9, a)
    cv2.polylines(ov, [inner], True, _INNER_RNG, 1, cv2.LINE_AA)

    # ── Three spokes at 12 / 4 / 8 o'clock ────────────────────────────────
    # (-π/2 = 12 o'clock; ±2π/3 puts the other two at 4 and 8 o'clock)
    for offset in (-math.pi / 2,
                   -math.pi / 2 + 2 * math.pi / 3,
                   -math.pi / 2 - 2 * math.pi / 3):
        _spoke(ov, cx, cy, a + offset, r_hub + 2, r - 8)

    # ── Octagonal hub ──────────────────────────────────────────────────────
    hub = _hub_pts(cx, cy, r_hub, a)
    cv2.fillPoly(ov, [hub], _HUB_FILL)
    cv2.polylines(ov, [hub], True, _INNER_RNG, 1, cv2.LINE_AA)

    # Small recessed ring inside hub
    cv2.circle(ov, (cx, cy), r_hub - 7, _INNER_RNG, 1, cv2.LINE_AA)

    # Hub pip — colour signals pedal state
    pip = _GREEN if state.pedal == 'ACCEL' else \
          _RED   if state.pedal == 'BRAKE' else _GOLD
    cv2.circle(ov, (cx, cy), 6, pip, -1)

    # ── 12-o'clock stripe on inner channel ────────────────────────────────
    top_ang = a - math.pi / 2
    for rr, rc, rt in ((r - 1, _GOLD_GLOW, 6), (r - 1, _GOLD, 3)):
        tx = int(cx + rr * math.cos(top_ang))
        ty = int(cy + rr * math.sin(top_ang))
        cv2.circle(ov, (tx, ty), rt, rc, -1)

    # ── Grip arcs at 3 and 9 o'clock ──────────────────────────────────────
    for side, offset in (('right', 0.0), ('left', math.pi)):
        active = (side == 'right' and state.steer == 'RIGHT') or \
                 (side == 'left'  and state.steer == 'LEFT')
        gd = math.degrees(a + offset)
        cv2.ellipse(ov, (cx, cy), (r, r), 0, gd - 28, gd + 28,
                    _GOLD_GLOW, 10, cv2.LINE_AA)
        cv2.ellipse(ov, (cx, cy), (r, r), 0, gd - 28, gd + 28,
                    _GRIP_ACT if active else _GOLD, 4, cv2.LINE_AA)

    # ── Wrist tracking dots ────────────────────────────────────────────────
    for px, py in ((lx, ly), (rx, ry)):
        cv2.circle(ov, (px, py),  9, _GOLD_GLOW, -1)
        cv2.circle(ov, (px, py),  5, _GOLD,       -1)

    frame[:] = cv2.addWeighted(ov, 0.80, frame, 0.20, 0)


# ---------------------------------------------------------------------------
# Key indicator bar
# ---------------------------------------------------------------------------

def _key_box(frame: np.ndarray, x: int, y: int,
             label: str, active: bool, color) -> None:
    kw, kh = 52, 40
    x2, y2 = x + kw, y + kh

    if active:
        cv2.rectangle(frame, (x, y), (x2, y2), color, -1)
        cv2.rectangle(frame, (x, y), (x2, y2), (255, 255, 255), 1)
        tc = (255, 255, 255)
    else:
        cv2.rectangle(frame, (x, y), (x2, y2), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, y), (x2, y2), (80, 80, 80), 1)
        tc = (80, 80, 80)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(frame, label,
                (x + (kw - tw) // 2, y + (kh + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, tc, 1, cv2.LINE_AA)


def draw_key_indicators(frame: np.ndarray, state: ControlState) -> None:
    h, w = frame.shape[:2]
    by   = h - 52         # top of key-box row
    gap  = 8

    # Semi-transparent bottom strip
    ov = frame.copy()
    cv2.rectangle(ov, (0, by - 10), (w, h), (18, 18, 18), -1)
    frame[:] = cv2.addWeighted(ov, 0.60, frame, 0.40, 0)

    # Steering keys  A / D  (bottom-left)
    _key_box(frame, 20,          by, 'A', state.steer == 'LEFT',  (200,  80,  80))
    _key_box(frame, 20 + 52 + gap, by, 'D', state.steer == 'RIGHT', (200,  80,  80))

    # Pedal keys  W / S  (bottom-right)
    _key_box(frame, w - 20 - 52 - gap - 52, by, 'W',
             state.pedal == 'ACCEL', _GREEN)
    _key_box(frame, w - 20 - 52,            by, 'S',
             state.pedal == 'BRAKE', _RED)

    # Pinch gauge — thin vertical bar just left of W/S group
    _draw_pinch_gauge(frame, w - 20 - 52 - gap - 52 - 18, by, state.pinch)


def _draw_pinch_gauge(frame: np.ndarray, x: int, y: int,
                      pinch: float) -> None:
    """Thin vertical bar showing normalized pinch (0=brake, 1=accel)."""
    bar_h = 40
    filled = int(bar_h * min(max(pinch, 0.0), 1.5) / 1.5)

    cv2.rectangle(frame, (x, y), (x + 8, y + bar_h), (50, 50, 50), -1)
    if filled > 0:
        color = _GREEN if pinch > 0.6 else (_RED if pinch < 0.25 else _GOLD)
        cv2.rectangle(frame, (x, y + bar_h - filled),
                      (x + 8, y + bar_h), color, -1)
    cv2.rectangle(frame, (x, y), (x + 8, y + bar_h), (90, 90, 90), 1)


# ---------------------------------------------------------------------------
# HUD text
# ---------------------------------------------------------------------------

def draw_hud(frame: np.ndarray, state: ControlState) -> None:
    h, w = frame.shape[:2]

    cv2.putText(frame, "VisionDrift",
                (20, 44), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, _CYAN, 2, cv2.LINE_AA)

    label = f"{state.angle:+.1f}°"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
    cv2.putText(frame, label,
                (w - tw - 20, 44), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (140, 140, 140), 1, cv2.LINE_AA)

    steer_label = {'LEFT': '← LEFT', 'RIGHT': 'RIGHT →',
                   'STRAIGHT': 'STRAIGHT'}.get(state.steer, '')
    color = (_GRIP_ACT if state.steer != 'STRAIGHT' else (130, 130, 130))
    cv2.putText(frame, steer_label,
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 1, cv2.LINE_AA)


def draw_hud_idle(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]

    cv2.putText(frame, "VisionDrift",
                (20, 44), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, _CYAN, 2, cv2.LINE_AA)

    msg = "Place both hands in frame"
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
    cv2.putText(frame, msg,
                (w // 2 - tw // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (155, 155, 155), 1, cv2.LINE_AA)
