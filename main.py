"""VisionDrift — entry point."""

import sys

import cv2

import config
from controller import SteeringController
from hud import draw_hud, draw_hud_idle, draw_key_indicators, draw_steering_wheel
from input_handler import InputHandler
from tracker import HandTracker


def main() -> int:
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    tracker    = HandTracker()
    controller = SteeringController()
    inp        = InputHandler()

    hands_lost = 0   # consecutive frames without both hands visible
    print("[VisionDrift] Ready — press Q in the camera window to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Frame read failed.")
                break

            frame = cv2.flip(frame, 1)
            pair  = tracker.process(frame)

            if pair is not None:
                hands_lost = 0
                press, release = controller.update(pair)
                inp.apply(press, release)

                draw_steering_wheel(frame, pair, controller.state)
                draw_key_indicators(frame, controller.state)
                draw_hud(frame, controller.state)

            else:
                hands_lost += 1
                # Grace period: release keys only after N consecutive lost frames
                if hands_lost == config.HANDS_LOST_TIMEOUT:
                    held = controller.release_all()
                    inp.release_all(held)

                draw_key_indicators(frame, controller.state)
                draw_hud_idle(frame)

            cv2.imshow("VisionDrift", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Guarantee no key stays stuck on exit
        held = controller.release_all()
        inp.release_all(held)
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[VisionDrift] Exited cleanly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
