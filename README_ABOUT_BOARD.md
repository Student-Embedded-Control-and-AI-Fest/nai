# About the ESP32-S3 Matrix — NoodleAI Mode 1

The **Waveshare ESP32-S3-Matrix** is a compact ESP32-S3 development board with an **8×8 RGB LED matrix**, **QMI8658 six-axis IMU**, **USB-C**, **Bluetooth LE**, and **Wi-Fi**. The ESP32-S3 uses a dual-core processor operating up to 240 MHz, with 512 KB on-chip SRAM and 4 MB flash on the documented board variant. For the hardware specification, see [Waveshare's official guide](https://docs.waveshare.com/ESP32-S3-Matrix).

## See it in action

**[Watch the NoodleAI gesture-recognition demo (demo.mp4)](https://github.com/Student-Embedded-Control-and-AI-Fest/nai/blob/main/demo.mp4)** on the project's GitHub repository.

## Finding the board in Indonesia

The board may be harder to source locally. The user-provided example is [CNC Store Bandung on Tokopedia](https://www.tokopedia.com/cncstorebandung/esp32-s3-development-board-dengan-rgb-led-matrix-wifi-bluetooth-sensor-gyroscope-qmi8658c-papan-pengembangan-iot-motion-control-1734224430143342142). Availability and pricing may change; NoodleAI is not affiliated with the store.

## Mode 1 in four steps

1. **Flash Firmware:** first-time USB installation through a compatible desktop Chromium browser, using the website's installer.
2. **Capture:** connect via Web Bluetooth. Hold **BOOT** to record a gesture and release it to stop.
3. **Train:** select a representation, collect labeled gestures, and train a model locally in TensorFlow.js.
4. **Deploy:** transfer the NAI4 model over Bluetooth LE. Gesture classification then runs on the device using Noodle.

The onboard IMU provides raw `ax,ay,az` (g) and `gx,gy,gz` (deg/s) at a nominal NoodleAI acquisition rate of **50 Hz**. Each BOOT-delimited gesture has variable duration, so Mode 1 linearly resamples the gesture to **N points (100 by default)**.

## Shared preprocessing

- Apply a first-order **5 Hz LPF** to the required raw IMU channels **before** resampling. The LPF is restarted per gesture and initialized with its first input sample.
- **No HPF** is used in this Mode 1 firmware.
- Arrange the resulting features in **time-major order**.
- Direct sensor representations then subtract each channel's **own mean for the gesture**. This is not division by gesture-wise standard deviation.
- Training-derived **StandardScaler** statistics (training set only) are stored in the model and used again during device inference. The three unit-direction channels of the direction + magnitude representation are intentionally left unscaled.
- Startup gyro bias is subtracted from gyroscope measurements when stationary calibration succeeds.

## All five input kinds

| Input kind | NAI4 ID | Channels | Processing | N=100 inputs |
|---|---:|---:|---|---:|
| **Accelerometer** | 1 | 3 | LPF5 → resample → per-axis gesture mean subtraction → StandardScaler | 300 |
| **Accel direction + magnitude** | 9 | 4 | LPF5 → resample XYZ → unit direction + magnitude; **no centering**, standardize **magnitude only** | 400 |
| **Gyroscope** | 2 | 3 | Gyro bias correction → LPF5 → resample → per-axis gesture mean subtraction → StandardScaler | 300 |
| **Accel + Gyro** | 3 | 6 | Gyro bias correction → LPF5 on six channels → resample → center each axis → StandardScaler | 600 |
| **Estimated Velocity** | 5 | 3 | Gyro bias correction → LPF5 → Madgwick orientation → world-frame gravity subtraction → integration + endpoint correction → resample → StandardScaler | 300 |

For direction + magnitude, let filtered/resampled acceleration be `a=(ax,ay,az)` and `m=||a||`. The four channels are `(ax/m,ay/m,az/m,m)` where defined. For `m ≤ 0.001 g`, direction is set to `(0,0,0)`; magnitude is retained. A stationary sensor typically sees approximately `1 g`; in free fall the measured magnitude approaches zero after filter settling. This represents the measured acceleration vector, not independently measured gravitational acceleration.

These descriptions apply to **Mode 1**. A continuous/sliding-window Mode 2 implementation has different acquisition and preprocessing behavior and should not be assumed identical.

For full firmware/model compatibility details, also consult [README_INPUT_PREPROCESSING.md](README_INPUT_PREPROCESSING.md). Previously trained models must match their firmware's preprocessing version.
