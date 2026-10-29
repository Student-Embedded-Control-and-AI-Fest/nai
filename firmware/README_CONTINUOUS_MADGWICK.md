# NoodleAI ESP32-S3 Mode 1 — one-time Madgwick initialization

This revision changes the Madgwick-based input kinds while retaining BOOT-triggered segmentation and all original direct IMU input kinds.

## Orientation state

1. Startup: calibrate the gyro zero-rate offset using the existing stationary procedure.
2. First valid sample after startup: initialize the quaternion from the accelerometer's gravity direction (initial yaw is arbitrary).
3. Every subsequent 50 Hz IMU sample, **including when BOOT is released**: update a separate continuous 5 Hz LPF and the Madgwick IMU quaternion with gyro bias corrected. Beta remains 0.10.
4. BOOT press: reset only the gesture recording buffer. **Do not reset the continuous LPF or quaternion.**
5. BOOT release: compute the selected input representation; only the velocity integrator is gesture-local and endpoint-corrected. The StandardScaler uses stored training-set statistics.

The separate 5 Hz LPF used for classifier acceleration inputs is gesture-local as before. No additional gyro high-pass filter is applied before orientation estimation.

## New input representation

`accel+roll+pitch` / NAI4 representation **10**: `[LPF5 ax, LPF5 ay, LPF5 az, roll, pitch]` in time-major layout. Roll/pitch in **radians**, gravity-referenced via the continuous quaternion; yaw is excluded. No gesture mean subtraction; the model's training-derived StandardScaler is applied. Default length 100 gives 500 MLP inputs.

For continuous model representations `quaternion`, `velocity`, `velocity+quaternion`, and `accel+roll+pitch`, the NAI4 `cfg.bin` flag **0x0010** signals continuous-orientation semantics. Older saved models without this flag retain the previous gesture-local orientation behavior, for compatibility; new models should be retrained.

## Browser/device parity

The BLE IMU notification expands from **128 to 208 bytes**. Every sample now includes the original `ax,ay,az,gx,gy,gz` followed by the device's **absolute quaternion** `qw,qx,qy,qz`, all Float32. The browser accepts both packet sizes, but continuous-orientation input kinds require 208-byte packets. Saved `.npz` datasets include `quat_data.npy` (`[total_raw_samples,4]`) aligned to `raw_data.npy` and `raw_offsets.npy`. The browser does **not** reinitialize Madgwick from the first sample of each saved gesture.

Existing raw-only `.npz` datasets can still train direct IMU representations, but **cannot** reconstruct a continuous quaternion from isolated gestures (inter-gesture measurements are missing). For continuous orientation inputs, record new data using the updated firmware. A mixed raw-only + quaternion dataset cannot be exported with complete orientation metadata.

## Build / test status

Sources updated and JavaScript syntax + synthetic feature tests passed. **PlatformIO compilation and hardware tests were not run here** because the required target platform and local Noodle library checkout are unavailable in this environment. Before deployment, compile and upload the new firmware and verify BLE 208-byte notifications and agreement of training/inference features. The web installer currently references a **legacy factory `.bin`**, which does not implement the new protocol; rebuild and republish that binary separately.

For gyroscope bias calibration, keep the board stationary during initial boot. Absolute tilt is gravity-referenced, but yaw is not observable from a six-axis IMU and remains relative/drift-prone. Accelerometer-based tilt may be disturbed by large linear accelerations.
