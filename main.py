"""VisionDrift — camera + hand-tracked steering-wheel HUD."""

import math
import sys

import cv2
import numpy as np

import config
from tracker import HandPair, HandTracker


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

def draw_steering_wheel(frame: np.ndarray, pair: HandPair, angle_deg: float) -> None:
    h, w = frame.shape[:2]

    lx = int(pair.left.x  * w)
    ly = int(pair.left.y  * h)
    rx = int(pair.right.x * w)
    ry = int(pair.right.y * h)

    cx = (lx + rx) // 2
    cy = (ly + ry) // 2
    r  = int(math.hypot(rx - lx, ry - ly) / 2)

    if r < 40:          # hands too close — skip
        return

    a = math.radians(angle_deg)
    ov = frame.copy()

    # ── Rim  (thick dim glow → medium → thin bright core) ──────────────────
    cv2.circle(ov, (cx, cy), r, (0,  90, 130), 18, cv2.LINE_AA)
    cv2.circle(ov, (cx, cy), r, (0, 185, 220),  7, cv2.LINE_AA)
    cv2.circle(ov, (cx, cy), r, (0, 245, 255),  2, cv2.LINE_AA)

    # ── Spokes (120° apart, rotated by steering angle) ─────────────────────
    for i in range(3):
        theta = math.radians(i * 120) + a
        ex = int(cx + r * math.cos(theta))
        ey = int(cy + r * math.sin(theta))
        cv2.line(ov, (cx, cy), (ex, ey), (0,  90, 130), 7, cv2.LINE_AA)
        cv2.line(ov, (cx, cy), (ex, ey), (0, 220, 255), 2, cv2.LINE_AA)

    # ── Hub ─────────────────────────────────────────────────────────────────
    cv2.circle(ov, (cx, cy), 18, (0,  90, 130), 16, cv2.LINE_AA)
    cv2.circle(ov, (cx, cy), 18, (0, 220, 255),  3, cv2.LINE_AA)
    cv2.circle(ov, (cx, cy),  5, (0, 245, 255), -1)

    # ── Grip highlights — bright arcs where hands hold the rim ─────────────
    for offset in (0.0, math.pi):
        gd = math.degrees(a + offset)
        cv2.ellipse(ov, (cx, cy), (r, r), 0,
                    gd - 24, gd + 24, (0, 255, 190), 6, cv2.LINE_AA)

    # ── Wrist tracking dots ─────────────────────────────────────────────────
    for px, py in ((lx, ly), (rx, ry)):
        cv2.circle(ov, (px, py), 10, (0,  90, 130), -1)
        cv2.circle(ov, (px, py),  7, (0, 255, 190), -1)

    frame[:] = cv2.addWeighted(ov, 0.78, frame, 0.22, 0)


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
