"""VisionDrift — camera + hand-tracked steering-wheel HUD."""

import math
import sys

import cv2
import numpy as np

import config
from tracker import HandPair, HandTracker

# ── Color palette (BGR) ────────────────────────────────────────────────────
_GLOW_DIM  = (155, 135,  90)   # outer halo
_GLOW_MID  = (215, 205, 165)   # mid ring
_RIM       = (248, 245, 238)   # bright rim core
_INNER_RNG = (190, 186, 175)   # inner accent ring
_SPOKE_FLL = (230, 226, 218)   # spoke fill
_SPOKE_EDG = (248, 245, 238)   # spoke edge highlight
_GOLD      = (  0, 165, 255)   # grip arcs + center dot  (RGB 255,165,0)
_GOLD_GLOW = (  0,  80, 170)   # grip glow (darker gold)
_HUB_FILL  = (210, 207, 200)   # hub background


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def compute_angle(pair: HandPair) -> float:
    dx = pair.right.x - pair.left.x
    dy = pair.right.y - pair.left.y
    return math.degrees(math.atan2(dy, dx))


# ---------------------------------------------------------------------------
# Steering wheel renderer
# ---------------------------------------------------------------------------

def _spoke(ov: np.ndarray, cx: int, cy: int, theta: float,
           r0: int, r1: int) -> None:
    """Filled tapered trapezoid spoke from radius r0 to r1 at angle theta."""
    dx, dy   = math.cos(theta), math.sin(theta)
    px, py   = -math.sin(theta), math.cos(theta)   # perpendicular
    w0, w1   = 7, 3                                 # half-width hub / rim

    pts = np.array([
        [cx + r0*dx + w0*px,  cy + r0*dy + w0*py],
        [cx + r0*dx - w0*px,  cy + r0*dy - w0*py],
        [cx + r1*dx - w1*px,  cy + r1*dy - w1*py],
        [cx + r1*dx + w1*px,  cy + r1*dy + w1*py],
    ], dtype=np.int32)

    cv2.fillPoly(ov, [pts], _SPOKE_FLL)
    cv2.polylines(ov, [pts], True, _SPOKE_EDG, 1, cv2.LINE_AA)


def draw_steering_wheel(frame: np.ndarray, pair: HandPair,
                        angle_deg: float) -> None:
    h, w = frame.shape[:2]

    lx = int(pair.left.x  * w);  ly = int(pair.left.y  * h)
    rx = int(pair.right.x * w);  ry = int(pair.right.y * h)

    cx = (lx + rx) // 2
    cy = (ly + ry) // 2
    r  = int(math.hypot(rx - lx, ry - ly) * 0.38)   # snug fit

    if r < 36:
        return

    a       = math.radians(angle_deg)
    r_inner = int(r * 0.66)   # inner accent ring radius
    r_hub   = 20              # hub circle radius

    ov = frame.copy()

    # ── Rim glow (three concentric draws on same radius → layered halo) ────
    cv2.circle(ov, (cx, cy), r, _GLOW_DIM, 18, cv2.LINE_AA)
    cv2.circle(ov, (cx, cy), r, _GLOW_MID,  7, cv2.LINE_AA)
    cv2.circle(ov, (cx, cy), r, _RIM,        2, cv2.LINE_AA)

    # ── Inner accent ring ──────────────────────────────────────────────────
    cv2.circle(ov, (cx, cy), r_inner, _INNER_RNG, 1, cv2.LINE_AA)

    # ── Three tapered spokes ───────────────────────────────────────────────
    for i in range(3):
        _spoke(ov, cx, cy, math.radians(i * 120) + a, r_hub + 2, r - 5)

    # ── Hub ────────────────────────────────────────────────────────────────
    cv2.circle(ov, (cx, cy), r_hub, _HUB_FILL, -1)
    cv2.circle(ov, (cx, cy), r_hub, _INNER_RNG, 1, cv2.LINE_AA)
    cv2.circle(ov, (cx, cy), 5, _GOLD, -1)              # gold center pip

    # ── 12-o'clock indicator (rotates with wheel) ─────────────────────────
    top_x = int(cx + (r - 10) * math.cos(a - math.pi / 2))
    top_y = int(cy + (r - 10) * math.sin(a - math.pi / 2))
    cv2.circle(ov, (top_x, top_y), 5, _GOLD, -1)

    # ── Grip arcs — gold glow then bright core ────────────────────────────
    for offset in (0.0, math.pi):
        gd = math.degrees(a + offset)
        cv2.ellipse(ov, (cx, cy), (r, r), 0,
                    gd - 26, gd + 26, _GOLD_GLOW, 9, cv2.LINE_AA)
        cv2.ellipse(ov, (cx, cy), (r, r), 0,
                    gd - 26, gd + 26, _GOLD,      4, cv2.LINE_AA)

    # ── Wrist tracking dots ────────────────────────────────────────────────
    for px, py in ((lx, ly), (rx, ry)):
        cv2.circle(ov, (px, py),  9, _GOLD_GLOW, -1)
        cv2.circle(ov, (px, py),  5, _GOLD,       -1)

    frame[:] = cv2.addWeighted(ov, 0.80, frame, 0.20, 0)


# ---------------------------------------------------------------------------
# HUD overlays
# ---------------------------------------------------------------------------

def draw_hud(frame: np.ndarray, angle_deg: float) -> None:
    h, w = frame.shape[:2]

    cv2.putText(frame, "VisionDrift",
                (20, 44), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (0, 210, 255), 2, cv2.LINE_AA)

    label = f"{angle_deg:+.1f}°"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
    cv2.putText(frame, label,
                (w - tw - 20, 44), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (140, 140, 140), 1, cv2.LINE_AA)

    cv2.putText(frame, "Q  quit",
                (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (75, 75, 75), 1, cv2.LINE_AA)


def draw_hud_idle(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]

    cv2.putText(frame, "VisionDrift",
                (20, 44), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (0, 210, 255), 2, cv2.LINE_AA)

    msg = "Place both hands in frame"
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
    cv2.putText(frame, msg,
                (w // 2 - tw // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (155, 155, 155), 1, cv2.LINE_AA)

    cv2.putText(frame, "Q  quit",
                (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (75, 75, 75), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    tracker = HandTracker()
    print("[VisionDrift] Camera ready. Press Q to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)
            pair  = tracker.process(frame)

            if pair is not None:
                angle = compute_angle(pair)
                draw_steering_wheel(frame, pair, angle)
                draw_hud(frame, angle)
            else:
                draw_hud_idle(frame)

            cv2.imshow("VisionDrift", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[VisionDrift] Exited cleanly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
