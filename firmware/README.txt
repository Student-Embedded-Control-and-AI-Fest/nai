NoodleAI ESP32-S3 Matrix — LPF5 Firmware
========================================

Use with matching NoodleAI Mode 1 Web. See README_DIRECTION_MAGNITUDE.md
and web/README_INPUT_PREPROCESSING.md for exact feature definitions.

Capture: BOOT press/release at nominal 50 Hz. Per-gesture 5 Hz first-order
low-pass filter (state initialized to the gesture's first raw sample), then
resampling to the model's normalized length (usually 100).

Input representations (NAI4):
1 accel             (3 channels), LPF5 then per-gesture mean centering
2 gyro              (3 channels), LPF5 then per-gesture mean centering
3 accel+gyro        (6 channels), LPF5 then per-gesture mean centering
5 velocity          (3 channels), gesture-local LPF5 accel + continuous Madgwick gravity compensation
9 accel-dir+mag     (4 channels), LPF5 unit direction + magnitude; no centering
10 accel+roll+pitch (5 channels), LPF5 acceleration + continuous Madgwick roll/pitch; no centering

All use training-derived scaler; for ID9 only magnitude uses non-identity
scaling. Quaternion representations 4 and 6 remain recognized by the
firmware but are not available for new training through the web dropdown.

Continuous Madgwick is initialized once after boot and updated between BOOT holds.
New model exports use the 0x0010 continuous-orientation flag, and raw dataset
exports include aligned device quaternion samples. See README_CONTINUOUS_MADGWICK.md.

Models require NAI4 version 4 with LPF5 flag 0x0008; older packages cannot
be loaded, and must be retrained/exported. Existing raw NPZ gesture data
with raw_data/raw_offsets can be reused for retraining.

Firmware builds: use platformio.ini environments, including FP32 file-backed,
FP32 SRAM, and INT8. BLE and model deployment behavior is otherwise unchanged.
