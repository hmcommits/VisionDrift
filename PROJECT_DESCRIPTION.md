## VisionDrift

This project is a computer vision-based virtual controller that allows users to play racing games by turning their empty hands in the air, mimicking a physical steering wheel. It eliminates the need for a keyboard, mouse, or expensive sim-racing hardware by relying entirely on a standard webcam and real-time AI processing.

---

## The Core Concept

The system operates on a simple principle: translation of physical geometry into digital commands. By capturing the user's hands on a webcam, the system identifies the spatial relationship between the left and right hand. When the user tilts this imaginary horizontal line—dipping the right hand lower than the left, for instance—the system calculates that angle and translates it into a standard keyboard command (like pressing the `D` key to turn right).

The game being played requires no modifications or special API access; it simply receives these keystrokes as if they were being typed on a physical keyboard.

---

## System Architecture

The project is built entirely in Python and operates through a continuous, high-speed loop consisting of four main stages:

### 1. The Vision Layer (OpenCV)

The foundation of the project is OpenCV, which interfaces with the computer's webcam. It captures video frame-by-frame and immediately mirrors the image horizontally. This mirroring is crucial for user experience, as it ensures the on-screen hands move in the same direction as the user's physical hands, much like looking into a mirror.

### 2. The Tracking Engine (Google MediaPipe)

The raw video frames are passed into Google’s MediaPipe, a highly optimized machine-learning pipeline. Instead of scanning the entire image for fingers, MediaPipe uses a two-step process: it first runs a lightweight palm detector to find the general location of the hands, and then applies a denser neural network to that cropped area to map 21 distinct 3D landmarks (joints and fingertips) on each hand. This occurs in milliseconds, ensuring the system runs smoothly even on standard processors like an Intel Core i5.

### 3. The Logic and Math Engine

Once MediaPipe outputs the $(x, y)$ coordinates of the hand landmarks, the Python script extracts the position of the palm center or the base of the index fingers.

The script calculates the slope between the left and right hand using basic trigonometry (the arctangent function). This slope is converted into an angle in degrees.

* **The Deadzone:** A buffer (e.g., -15° to +15°) is established around the horizontal zero-point. If the angle falls within this zone, the vehicle drives straight. This prevents micro-jitters from causing the car to swerve.
* **Steering:** If the angle exceeds the threshold, it triggers a turn.
* **Pedals:** To handle acceleration and braking, the script calculates the Euclidean distance between the thumb and index finger on one hand. An open hand (large distance) signals acceleration, while a closed fist or pinch (small distance) signals braking.

### 4. The Execution Layer (PyDirectInput)

The final stage converts the mathematical logic into action. Instead of using standard automation libraries that are often blocked by video games, the system uses a library that sends low-level, hardware-scan codes to the operating system. This simulates physical keystrokes so perfectly that games cannot differentiate between the Python script and a real mechanical keyboard.

## Controls
Steer Left: Dip your left hand down (< -15°)
Steer Right: Dip your right hand down (> 15°)
Kill Switch: Press Q while the camera window is active to safely terminate the script.