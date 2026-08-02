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

def _spoke(ov: np.ndarray, cx: int, cy: int, theta: float,
           r0: int, r1: int, w0: int = 11, w1: int = 5) -> None:
    """Filled tapered spoke; returns the (pos-perp, neg-perp) edge points for
    later use by the bridge panel."""
    dx, dy = math.cos(theta), math.sin(theta)
    px, py = -math.sin(theta), math.cos(theta)

    pts = np.array([
        [cx + r0*dx + w0*px,  cy + r0*dy + w0*py],
        [cx + r0*dx - w0*px,  cy + r0*dy - w0*py],
        [cx + r1*dx - w1*px,  cy + r1*dy - w1*py],
        [cx + r1*dx + w1*px,  cy + r1*dy + w1*py],
    ], dtype=np.int32)
    cv2.fillPoly(ov, [pts], _SPOKE_FLL)
    cv2.polylines(ov, [pts], True, _SPOKE_EDG, 1, cv2.LINE_AA)


def _bridge_panel(ov: np.ndarray, cx: int, cy: int,
                  theta_l: float, theta_r: float,
                  r_hub: int, r_rim: int) -> None:
    """Wide panel that bridges the two upper spokes (10 and 2 o'clock).

    The panel is a quadrilateral whose four corners sit on the
    inward-facing edges of the left and right upper spokes.
    """
    def edge_pt(theta: float, t: float, sign: int):
        """Point on one edge of a spoke at fractional length t (0=hub, 1=rim)."""
        r_  = r_hub + t * (r_rim - r_hub)
        w_  = max(11 + t * (5 - 11), 5)
        dx, dy = math.cos(theta), math.sin(theta)
        px, py = -math.sin(theta), math.cos(theta)
        return (cx + r_*dx + sign*w_*px,
                cy + r_*dy + sign*w_*py)

    # Left spoke  (upper-left, theta_l): the +perp side faces inward
    # Right spoke (upper-right, theta_r): the -perp side faces inward
    pts = np.array([
        edge_pt(theta_l, 0.28, +1),   # left spoke inner edge, near hub
        edge_pt(theta_l, 0.76, +1),   # left spoke inner edge, near rim
        edge_pt(theta_r, 0.76, -1),   # right spoke inner edge, near rim
        edge_pt(theta_r, 0.28, -1),   # right spoke inner edge, near hub
    ], dtype=np.int32)

    cv2.fillConvexPoly(ov, pts, _SPOKE_FLL)
    cv2.polylines(ov, [pts.reshape(-1, 1, 2)], True, _SPOKE_EDG, 1, cv2.LINE_AA)


def _hub_ring(ov: np.ndarray, cx: int, cy: int, r_hub: int) -> None:
    """Simple round hub with two concentric rings."""
    cv2.circle(ov, (cx, cy), r_hub,      _HUB_FILL,  -1)
    cv2.circle(ov, (cx, cy), r_hub,      _INNER_RNG,  1, cv2.LINE_AA)
    cv2.circle(ov, (cx, cy), r_hub - 7,  _INNER_RNG,  1, cv2.LINE_AA)


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
    r_hub = 26

    # Spoke directions: 10 o'clock, 2 o'clock, 6 o'clock (120° apart)
    theta_l = a - 5 * math.pi / 6   # upper-left  (10 o'clock)
    theta_r = a -     math.pi / 6   # upper-right (2 o'clock)
    theta_b = a +     math.pi / 2   # bottom      (6 o'clock)

    ov = frame.copy()

    # ── Full round rim — three-layer glow ──────────────────────────────────
    cv2.circle(ov, (cx, cy), r, _GLOW_DIM, 20, cv2.LINE_AA)
    cv2.circle(ov, (cx, cy), r, _GLOW_MID,  8, cv2.LINE_AA)
    cv2.circle(ov, (cx, cy), r, _RIM,        2, cv2.LINE_AA)

    # Inner grip channel
    cv2.circle(ov, (cx, cy), r - 9, _INNER_RNG, 1, cv2.LINE_AA)

    # ── Bridge panel between the two upper spokes ──────────────────────────
    _bridge_panel(ov, cx, cy, theta_l, theta_r, r_hub, r - 8)

    # ── Three spokes: upper-left, upper-right, bottom ─────────────────────
    _spoke(ov, cx, cy, theta_l, r_hub + 2, r - 8)
    _spoke(ov, cx, cy, theta_r, r_hub + 2, r - 8)
    _spoke(ov, cx, cy, theta_b, r_hub + 2, r - 8)

    # ── Round hub ──────────────────────────────────────────────────────────
    _hub_ring(ov, cx, cy, r_hub)

    # Hub pip — colour signals pedal state
    pip = _GREEN if state.pedal == 'ACCEL' else \
          _RED   if state.pedal == 'BRAKE' else _GOLD
    cv2.circle(ov, (cx, cy), 6, pip, -1)

    # ── 12-o'clock radial stripe on the rim ───────────────────────────────
    top_ang = a - math.pi / 2
    tx0 = int(cx + (r - 12) * math.cos(top_ang))
    ty0 = int(cy + (r - 12) * math.sin(top_ang))
    tx1 = int(cx +  r       * math.cos(top_ang))
    ty1 = int(cy +  r       * math.sin(top_ang))
    cv2.line(ov, (tx0, ty0), (tx1, ty1), _GOLD_GLOW, 7, cv2.LINE_AA)
    cv2.line(ov, (tx0, ty0), (tx1, ty1), _GOLD,      3, cv2.LINE_AA)

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
