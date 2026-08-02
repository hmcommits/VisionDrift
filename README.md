# VisionDrift

**Drive any racing game with your bare hands — no wheel, no controller, no keyboard.**

VisionDrift is a real-time computer vision controller that turns your webcam into a steering wheel. Hold both hands in the air, tilt them like a wheel to steer, and open or close your fingers to accelerate or brake. The game receives standard keyboard scan codes — it cannot tell the difference between VisionDrift and a physical keyboard.

---

## Table of Contents

- [Demo](#demo)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running](#running)
- [Controls](#controls)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Demo

> Point your webcam at your hands, run `python main.py`, and start playing.

When both hands are detected, a transparent steering wheel graphic appears between them and rotates in real time. The bottom bar shows which keys (`A` `D` `W` `S`) are currently being pressed.

---

## How It Works

1. **Capture** — OpenCV grabs each webcam frame and mirrors it horizontally so your hands move naturally (like a mirror).
2. **Track** — Google MediaPipe's hand landmarker model detects 21 3-D landmarks on each hand at high speed. VisionDrift uses the wrist, thumb tip, index fingertip, and palm base from both hands.
3. **Compute** — The angle of the line between your two wrists (via `atan2`) becomes the steering angle. The normalized distance between your right hand's thumb tip and index fingertip becomes the throttle/brake signal.
4. **Act** — A state machine emits key-down / key-up events only when state changes. PyDirectInput delivers these as low-level hardware scan codes that bypass game input filters.
5. **Render** — A transparent, modern steering-wheel HUD is drawn over the camera feed. The wheel rotates with your hands and scales to fit the gap between them.

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.10 – 3.13 |
| Windows | 10 / 11 (PyDirectInput is Windows-only) |
| Webcam | Any USB or built-in webcam |
| CPU | Intel Core i5 or equivalent (no GPU needed) |

> **Note:** PyDirectInput requires the game window to be in focus when keys are sent. The VisionDrift camera window can be visible alongside the game.

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/hmcommits/VisionDrift.git
cd VisionDrift
```

**2. Create and activate a virtual environment** *(recommended)*

```bash
python -m venv .venv
.venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

> On first launch, VisionDrift will automatically download the MediaPipe hand landmark model (~8 MB) and save it locally as `hand_landmarker.task`. Subsequent launches are instant.

---

## Running

```bash
python main.py
```

Press **Q** in the camera window to exit. All held keys are guaranteed to release before the process terminates.

---

## Controls

### Steering

Hold both hands up in front of the camera as if gripping an invisible wheel, roughly shoulder-width apart.

| Gesture | Action | Key |
|---|---|---|
| Tilt right — right hand dips lower | Steer right | `D` |
| Tilt left — left hand dips lower | Steer left | `A` |
| Hands level (within ±15°) | Drive straight | *(none)* |

### Throttle & Brake

Controlled by the **right hand** finger pinch.

| Gesture | Action | Key |
|---|---|---|
| Open hand — fingers spread wide | Accelerate | `W` |
| Relaxed / neutral hand | Coast | *(none)* |
| Pinch — finger tips together / closed fist | Brake | `S` |

### HUD

| Element | Meaning |
|---|---|
| Rotating steering wheel | Live angle and rotation between your hands |
| Gold grip arcs | Active side brightens when a turn is registered |
| Hub pip colour | Gold = coasting · Green = accelerating · Red = braking |
| `A` `D` `W` `S` boxes (bottom bar) | Light up when that key is being pressed |
| Vertical bar (bottom right) | Real-time throttle/pinch level |
| Angle readout (top right) | Current tilt in degrees |

### Kill Switch

Press **`Q`** in the camera window at any time. Held keys are released before exit — nothing gets stuck.

---

## Configuration

All tunable parameters live in [`config.py`](config.py). No code changes are needed for common adjustments.

| Parameter | Default | Description |
|---|---|---|
| `DEADZONE_DEGREES` | `15.0` | Tilt angle (±) that counts as straight — increase if steering feels jittery |
| `THROTTLE_THRESHOLD` | `0.60` | Normalised pinch above which the gas key is held |
| `BRAKE_THRESHOLD` | `0.25` | Normalised pinch below which the brake key is held |
| `KEY_LEFT` | `'a'` | Key sent when steering left |
| `KEY_RIGHT` | `'d'` | Key sent when steering right |
| `KEY_ACCEL` | `'w'` | Key sent when accelerating |
| `KEY_BRAKE` | `'s'` | Key sent when braking |
| `CAMERA_INDEX` | `0` | OpenCV camera index (try `1` or `2` if the wrong camera opens) |
| `FRAME_WIDTH` | `1280` | Capture resolution width |
| `FRAME_HEIGHT` | `720` | Capture resolution height |
| `HANDS_LOST_TIMEOUT` | `8` | Frames of missing hands before all keys are released |

---

## Project Structure

```
VisionDrift/
│
├── main.py            Entry point — camera loop, event wiring, robustness
├── tracker.py         MediaPipe Tasks wrapper; returns per-hand landmarks
├── controller.py      Steering & pedal state machine; emits key events
├── input_handler.py   PyDirectInput wrapper; sends hardware scan codes
├── hud.py             All OpenCV drawing — wheel, key indicators, HUD text
├── config.py          Every tunable constant in one place
│
├── requirements.txt
├── hand_landmarker.task   Downloaded automatically on first run (~8 MB)
└── README.md
```

---

## Architecture

```
Webcam
  │
  ▼
OpenCV (flip)
  │
  ▼
MediaPipe HandLandmarker          tracker.py
  │  wrist · thumb tip · index tip · palm base (per hand)
  ▼
SteeringController                controller.py
  │  angle = atan2(dy, dx)
  │  pinch = thumb–index dist / palm size
  │  state machine → key press / release events (on transition only)
  ▼
InputHandler                      input_handler.py
  │  pydirectinput.keyDown / keyUp (hardware scan codes)
  ▼
Game receives WASD keystrokes

     ┌──────────────────────────┐
     │  HUD overlay             │  hud.py
     │  · Steering wheel        │
     │  · Grip arcs (state)     │
     │  · Key indicator bar     │
     │  · Pinch gauge           │
     └──────────────────────────┘
```

### Key design decisions

- **Model complexity 0** — MediaPipe's lightest hand model; runs comfortably on a mid-range CPU with no GPU.
- **State machine, not polling** — `keyDown` / `keyUp` are only called when state changes, not every frame. This prevents key-repeat spam and reduces OS overhead.
- **Positional hand sorting** — Hands are sorted by screen-x rather than by MediaPipe's handedness label, which can be unreliable after horizontal mirroring.
- **Normalised pinch** — Raw thumb–index pixel distance is divided by the wrist-to-palm-base distance, making the throttle signal consistent regardless of how far the user sits from the camera.
- **Grace period** — If both hands leave the frame, keys are held for `HANDS_LOST_TIMEOUT` frames before releasing, preventing false key-up events from brief occlusion.
- **Guaranteed key release** — A `finally` block in the main loop releases all held keys on any exit path, including exceptions and Ctrl-C.

---

## Known Limitations

- **Windows only** — PyDirectInput uses Windows-specific scan codes. Mac/Linux support would require a different input library.
- **Two hands required** — The controller is disabled when fewer than two hands are visible. Single-hand mode is not currently supported.
- **Lighting sensitivity** — MediaPipe hand detection degrades in very dark or strongly backlit environments. A front-facing light source improves reliability.
- **Right hand for throttle** — The pinch signal always uses the rightmost detected hand. Left-hand dominance is not currently configurable via `config.py`.
- **No analogue output** — Steering and pedals are binary (key held or not). True analogue input (e.g., variable steering angle → variable joystick axis) is not implemented.

---

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
