# VisionDrift — Implementation Plan

## Overview

Build a real-time, webcam-based virtual steering controller that translates hand geometry into game keystrokes. No game modifications required; the output is indistinguishable from a physical keyboard.

---

## Phase 1 — Project Setup

**Goal:** Establish a clean, reproducible Python environment.

- [ ] Initialize a `requirements.txt` with pinned dependencies:
  - `opencv-python` — webcam capture and frame rendering
  - `mediapipe` — hand landmark detection
  - `pydirectinput` — low-level hardware-scan-code keyboard simulation
  - `numpy` — vector/angle math
- [ ] Create a `venv` and verify all packages install cleanly on the target machine (Windows, Intel i5 class CPU)
- [ ] Create the entry-point file `main.py`
- [ ] Create `config.py` for all tunable constants (deadzone, thresholds, key bindings)

**Deliverable:** `python main.py` starts without errors and prints a ready message.

---

## Phase 2 — Vision Layer (OpenCV)

**Goal:** Open the webcam, capture frames, and display a mirrored preview window.

- [ ] Initialize `cv2.VideoCapture(0)` and confirm a frame is returned
- [ ] Horizontally flip every frame with `cv2.flip(frame, 1)` — this makes on-screen hands mirror physical movement
- [ ] Display the frame in a named window (`cv2.imshow`)
- [ ] Implement the main loop with a clean exit on `Q` keypress (`cv2.waitKey`)
- [ ] Release the capture and destroy windows on exit

**Deliverable:** A live, mirrored webcam window that closes cleanly on `Q`.

---

## Phase 3 — Tracking Engine (MediaPipe)

**Goal:** Detect both hands and extract the palm/index-base landmarks in real time.

- [ ] Initialize `mp.solutions.hands.Hands` with:
  - `max_num_hands=2`
  - `min_detection_confidence=0.7`
  - `min_tracking_confidence=0.5`
- [ ] Convert each BGR frame to RGB before passing to MediaPipe
- [ ] Parse `results.multi_hand_landmarks` and `results.multi_handedness` to label LEFT vs RIGHT hand reliably (note: handedness labels are flipped after mirroring — account for this)
- [ ] Extract the `(x, y)` pixel coordinates of landmark 0 (wrist) or landmark 5 (index finger base) for each hand
- [ ] Draw landmark overlays on the preview frame using `mp.solutions.drawing_utils`

**Deliverable:** Both hands tracked in real time with left/right labels visible in the preview window.

---

## Phase 4 — Logic and Math Engine

**Goal:** Convert hand positions into a steering angle and a throttle/brake signal.

### 4a — Steering

- [ ] Compute the angle between the two hand anchor points using:
  ```
  dx = right_hand.x - left_hand.x
  dy = right_hand.y - left_hand.y
  angle_deg = math.degrees(math.atan2(dy, dx))
  ```
- [ ] Define `DEADZONE_DEGREES` (default `±15°`) in `config.py`
- [ ] Map angle to steering state:
  - `angle < -DEADZONE` → `STEER_LEFT`
  - `angle > +DEADZONE` → `STEER_RIGHT`
  - otherwise → `STRAIGHT`

### 4b — Throttle / Brake

- [ ] Compute Euclidean distance between thumb tip (landmark 4) and index fingertip (landmark 8) on the designated hand
- [ ] Normalize distance against a calibrated hand size (e.g., wrist-to-middle-finger distance) so it is scale-invariant
- [ ] Define `THROTTLE_THRESHOLD` and `BRAKE_THRESHOLD` in `config.py`
- [ ] Map distance to pedal state:
  - `distance > THROTTLE_THRESHOLD` → `ACCELERATE`
  - `distance < BRAKE_THRESHOLD` → `BRAKE`
  - otherwise → `COAST`

### 4c — State Machine

- [ ] Track the *previous* steering and pedal state to avoid redundant key events (press only on state change, release only on state change)

**Deliverable:** Console output printing the current angle, pinch distance, and derived state on every frame.

---

## Phase 5 — Execution Layer (PyDirectInput)

**Goal:** Convert logical states into actual low-level keystrokes.

- [ ] Define key mappings in `config.py`:
  ```python
  KEY_LEFT  = 'a'
  KEY_RIGHT = 'd'
  KEY_ACCEL = 'w'
  KEY_BRAKE = 's'
  ```
- [ ] On state transition to active → `pydirectinput.keyDown(key)`
- [ ] On state transition away from active → `pydirectinput.keyUp(key)`
- [ ] Ensure all held keys are released in the cleanup path (exit handler / `finally` block) so no key gets "stuck" after the script stops

**Deliverable:** A racing game responds correctly to hand gestures with no stuck keys on exit.

---

## Phase 6 — HUD Overlay

**Goal:** Render useful debug/status information directly on the preview window.

- [ ] Draw the imaginary line between the two hand anchor points
- [ ] Display current angle value (degrees) as text
- [ ] Display current steering state (`LEFT` / `STRAIGHT` / `RIGHT`) and pedal state
- [ ] Highlight active key labels (turn green when pressed)
- [ ] Show a "NO HANDS DETECTED" warning when fewer than 2 hands are visible

**Deliverable:** All control state is readable at a glance in the preview window.

---

## Phase 7 — Robustness and Edge Cases

**Goal:** Handle real-world failure modes gracefully.

- [ ] One hand lost mid-session — hold last known state for at most N frames, then release all keys and coast
- [ ] Both hands lost — immediately release all keys
- [ ] Webcam not available — print a clear error and exit with a non-zero code
- [ ] Add a configurable `FRAME_SKIP` to drop processing on every Nth frame if CPU load is high
- [ ] Optional: add a `--calibrate` CLI flag that walks the user through setting their personal deadzone and pinch thresholds

---

## Phase 8 — Configuration and Docs

- [ ] Consolidate all tunable values into `config.py` with inline comments explaining each parameter
- [ ] Write a concise `README.md` covering: prerequisites, installation, how to run, controls reference
- [ ] Add a `--debug` flag that enables verbose console logging without modifying game behavior

---

## File Structure

```
VisionDrift/
├── main.py           # Entry point; main loop
├── config.py         # All constants and key mappings
├── tracker.py        # MediaPipe wrapper; returns hand landmarks per frame
├── controller.py     # Angle/pinch math + state machine
├── input_handler.py  # PyDirectInput wrapper; key press/release management
├── hud.py            # OpenCV overlay drawing functions
├── requirements.txt
└── README.md
```

---

## Build Order Summary

| Phase | Dependency | Risk |
|-------|-----------|------|
| 1 — Setup | none | Low |
| 2 — OpenCV loop | Phase 1 | Low |
| 3 — MediaPipe tracking | Phase 2 | Medium (handedness flip after mirror) |
| 4 — Math engine | Phase 3 | Medium (scale-invariant pinch calibration) |
| 5 — PyDirectInput | Phase 4 | Low (Windows-only; run as admin if games block input) |
| 6 — HUD | Phase 4 | Low |
| 7 — Robustness | Phase 5 | Low |
| 8 — Config + Docs | Phase 7 | Low |

---

## Known Risks

- **Handedness flip:** After `cv2.flip`, MediaPipe's left/right labels are inverted. Must swap them after detection.
- **Admin privileges:** Some games require the script to run with elevated permissions for `pydirectinput` scan codes to register.
- **Lighting sensitivity:** MediaPipe confidence drops in low or backlit environments. Document minimum lighting requirements.
- **Scale variance:** Raw pixel distance for pinch is not consistent across different distances from the webcam. Normalize against hand size.
