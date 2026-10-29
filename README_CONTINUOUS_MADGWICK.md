# New NoodleAI input: Accel XYZ + continuous Madgwick Roll and Pitch

The new `accel+roll+pitch` representation has five time-major channels `ax, ay, az, roll, pitch` and requires ESP32-S3 firmware from the matching `firmware` source archive. Angular channels are in radians, acceleration in g; orientation is gravity-referenced and does not use yaw. No per-gesture mean subtraction is applied to this representation. StandardScaler parameters remain learned from the training split and deployed unchanged.

The firmware continuously updates Madgwick across BOOT presses and releases, using startup gyro bias correction and a continuous 5 Hz LPF. The gesture-local accelerometer LPF and velocity endpoint correction, where applicable, remain separate. The firmware sends a 208-byte BLE notification with the quaternion appended after each raw six-axis sample. The web trainer records both in the dataset; `quat_data.npy` stores one quaternion per raw IMU sample. Existing 128-byte streams remain supported for legacy input kinds, but do not support the new continuous orientation mode.

**Do not train continuous-orientation models with an old raw-only NPZ**: the between-gesture orientation history is unavailable. Record a new dataset with the updated firmware, retrain, and redeploy. Updated model configurations set `0x0010` (continuous orientation), while the usual LPF5 flag remains `0x0008`.

**Installer warning:** the provided merged factory `.bin` in this web archive remains the *older* build. The browser installer still installs that older firmware, not the updated Madgwick implementation. Compile the accompanying firmware source and flash it before using the new input kind; replace the installer factory image only after a successful target build.

Suggested hardware acceptance test: perform two BOOT gestures separated by a slow, unrecorded 30° tilt; verify the first roll/pitch sample in the second gesture reflects the new tilt rather than resetting to 0°. Compare the same recorded example's browser-prepared features against the firmware's pre-scaler window values within Float32 numerical tolerance.
