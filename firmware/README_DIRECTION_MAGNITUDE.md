# NoodleAI Mode 1 firmware — 5 Hz LPF preprocessing

This firmware matches `web/README_INPUT_PREPROCESSING.md` from the same release. Use it together with the matching web project.

Every BOOT-triggered gesture is processed at the **raw 50 Hz** sample rate using `src/nai_motion_lpf5.h`. The 5 Hz first-order low-pass filter is initialized from the first sample **on every gesture**, then the filtered gesture is resampled to the trained model's fixed length (default 100).

- **Accel (ID 1):** LPF → temporal resampling → per-gesture axis mean subtraction → training-derived StandardScaler.
- **Gyro (ID 2):** gyro-only HPF0.5 → LPF → temporal resampling → per-gesture axis mean subtraction → StandardScaler. Startup zero-rate calibration is independent.
- **Accel + Gyro (ID 3):** HPF0.5 on gyro XYZ only, then LPF on all six axes → resampling → per-gesture mean subtraction on six axes → StandardScaler.
- **Estimated Velocity (ID 5):** gesture-local LPF acceleration + continuous Madgwick quaternion → world-frame gravity compensation → trapezoidal integration + endpoint correction → resampling → StandardScaler.
- **Accel direction + magnitude (ID 9):** LPF accel → resampling → unit XYZ direction + magnitude of the **same filtered vector**. No gesture centering. The direction scaler is identity; only magnitude is standardized. At `m <= 0.001 g`, direction is `[0,0,0]` rather than a noisy unit vector.

**The only HPF is the fixed 0.5 Hz stage on direct gyro classification features; Madgwick and accelerometer channels do not use it.** The firmware requires the `NAI4` `cfg.bin` bit `0x0008` for LPF5 and rejects incompatible old packages (including IDs 7/8). Retrain/re-export the model with this version of the web trainer before deployment.



Note: Madgwick initializes once after startup, not on every BOOT press. See README_CONTINUOUS_MADGWICK.md.
