# NoodleAI Studio — Mode 1

A static, browser-based NoodleAI application for the **Waveshare ESP32-S3-Matrix** using its onboard **QMI8658** IMU.

Mode 1 uses the board's physical **BOOT button** to delimit each gesture:

```text
hold BOOT → perform one gesture → release BOOT
```

The complete gesture is then normalized to a fixed number of points, transformed into the selected NAI4 motion representation, and used for browser-side MLP training or on-device Noodle inference.

## Mode 1 workflow

- **Live IMU** — raw six-axis accelerometer + gyroscope plots over Web Bluetooth.
- **Capture & Train** — define labels and normalized gesture length; use BOOT to capture one gesture at a time.
- **Training Curves** — view train/validation loss and accuracy.
- **Deploy & Infer** — save/deploy the NAI4 package, select device mode, then classify BOOT-delimited gestures on the ESP32-S3-Matrix.

There is no application server. TensorFlow.js training, dataset processing, NAI4 packaging, and BLE communication run in the browser.

## Hardware target

This Mode 1 site is intentionally tied to:

- Waveshare **ESP32-S3-Matrix**
- onboard **QMI8658** IMU
- physical **BOOT** button for gesture start/end

It is not the continuous/sliding-window Mode 2 workflow.

## Run locally

Web Bluetooth requires a secure context. `localhost` is allowed for development:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

## Publish with GitHub Pages

1. Put the files in the repository root.
2. Push to `main`.
3. Open **Settings → Pages**.
4. Select **Deploy from a branch**.
5. Select `main` and `/ (root)`.
6. Open the HTTPS Pages URL.

## Browser support

Use a Chromium-based browser with Web Bluetooth support.

## Dependencies

Pinned CDN dependencies:

- TensorFlow.js 4.22.0
- JSZip 3.10.1

The first page load needs internet access for the CDN files. Gesture data, training data, and model processing stay local in the browser.

## NAI4 motion representations

The raw six-axis gesture can be transformed into:

- Accelerometer
- Gyroscope
- Accel + Gyro
- Relative Quaternion
- Estimated Velocity
- Velocity + Quaternion

The defining Mode 1 behavior remains the same for every representation: **BOOT supplies the gesture boundaries; the browser/device normalizes the complete gesture before inference.**
