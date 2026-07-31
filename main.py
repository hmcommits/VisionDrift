"""VisionDrift — Phase 1+2: camera, hand tracking, and steering-wheel HUD."""

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
    """Angle (degrees) of the line from left anchor to right anchor.
    Positive = right hand lower (steer right); negative = left hand lower (steer left).
    """
    dx = pair.right.x - pair.left.x
    dy = pair.right.y - pair.left.y   # y increases downward in normalized space
    return math.degrees(math.atan2(dy, dx))


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_hand_axis(frame: np.ndarray, pair: HandPair) -> None:
    """Draw the axis line and anchor dots on the two wrist positions."""
    h, w = frame.shape[:2]
    lx, ly = int(pair.left.x  * w), int(pair.left.y  * h)
    rx, ry = int(pair.right.x * w), int(pair.right.y * h)

    cv2.line(frame, (lx, ly), (rx, ry), (0, 210, 255), 2, cv2.LINE_AA)
    for px, py in ((lx, ly), (rx, ry)):
        cv2.circle(frame, (px, py), 9, (0, 210, 255), -1)
        cv2.circle(frame, (px, py), 9, (0, 100, 140), 2)


def draw_steering_wheel(frame: np.ndarray, angle_deg: float) -> None:
    """Render a rotating steering-wheel graphic centred at the bottom of frame."""
    h, w = frame.shape[:2]
    cx = w // 2
    cy = h - config.WHEEL_OFFSET_Y
    r  = config.WHEEL_RADIUS
    a  = math.radians(angle_deg)

    overlay = frame.copy()

    # Dashboard strip
    cv2.rectangle(overlay,
                  (0, cy - r - 28), (w, h),
                  (18, 18, 18), -1)
    frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    overlay  = frame.copy()

    # Rim shadow
    cv2.circle(overlay, (cx + 3, cy + 3), r, (25, 25, 25), 7)
    # Outer rim
    cv2.circle(overlay, (cx, cy), r, (220, 220, 220), 6)

    # Three spokes (120° apart, rotated by angle)
    for i in range(3):
        theta = math.radians(i * 120) + a
        ex = int(cx + r * math.cos(theta))
        ey = int(cy + r * math.sin(theta))
        cv2.line(overlay, (cx, cy), (ex, ey), (175, 175, 175), 4, cv2.LINE_AA)

    # Center hub
    cv2.circle(overlay, (cx, cy), 18, (170, 170, 170), -1)
    cv2.circle(overlay, (cx, cy), 18, (55, 55, 55), 2)

    # 12-o'clock indicator dot
    ind_theta = a - math.pi / 2
    ix = int(cx + (r - 12) * math.cos(ind_theta))
    iy = int(cy + (r - 12) * math.sin(ind_theta))
    cv2.circle(overlay, (ix, iy), 9, (0, 210, 255), -1)

    frame[:] = cv2.addWeighted(overlay, 0.90, frame, 0.10, 0)


def draw_hud_active(frame: np.ndarray, angle_deg: float) -> None:
    h, w = frame.shape[:2]
    cx = w // 2

    cv2.putText(frame, "VisionDrift",
                (20, 44), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (0, 210, 255), 2, cv2.LINE_AA)

    label = f"{angle_deg:+.1f} deg"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    label_y = h - config.WHEEL_OFFSET_Y + config.WHEEL_RADIUS + 36
    cv2.putText(frame, label,
                (cx - tw // 2, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (150, 150, 150), 1, cv2.LINE_AA)

    cv2.putText(frame, "Q  quit",
                (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (80, 80, 80), 1, cv2.LINE_AA)


def draw_hud_idle(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]

    cv2.putText(frame, "VisionDrift",
                (20, 44), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (0, 210, 255), 2, cv2.LINE_AA)

    msg = "Place both hands in frame"
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
    cv2.putText(frame, msg,
                (w // 2 - tw // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (160, 160, 160), 1, cv2.LINE_AA)

    cv2.putText(frame, "Q  quit",
                (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (80, 80, 80), 1, cv2.LINE_AA)


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

            pair = tracker.process(frame)

            if pair is not None:
                angle = compute_angle(pair)
                draw_hand_axis(frame, pair)
                draw_steering_wheel(frame, angle)
                draw_hud_active(frame, angle)
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
