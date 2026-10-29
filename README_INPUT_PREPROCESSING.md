# NoodleAI Mode 1 — Input kinds and preprocessing (gyro HPF0.5 + LPF5)

This documents the **Mode 1 ESP32-S3-Matrix** browser trainer and firmware shipped together in this package. Input processing is part of the model definition: web training and firmware inference must use the **same version**.

## Acquisition and shared operations

- **QMI8658**: `ax,ay,az` in **g**, `gx,gy,gz` in **deg/s**; nominal sampling rate **50 Hz**.
- Each **BOOT press/release** defines a variable-duration gesture. The raw six-axis readings for that gesture are saved in `raw_data` with `raw_offsets` in the dataset `.npz`.
- Apply **first-order 0.5 Hz high-pass filtering to gyro XYZ only** for the direct `gyro` and `accel+gyro` classification representations. Reset HPF state on every BOOT gesture. The HPF is applied **before** the existing 5 Hz LPF. **No HPF is applied to Madgwick orientation or any derived input kind.**
- Apply **first-order 5 Hz low-pass filtering** to the required raw IMU channels **before** temporal resampling. The LPF state resets independently each gesture and starts at the first sample of each filtered axis; constant gravity therefore remains observable.
- Resample each selected/derived time series linearly to the configured fixed length **N** (default 100). Resampling changes the number of points, **not** the source 50-Hz sampling rate.
- Arrange channels in **time-major order**: `[ch0_t0,ch1_t0,...,ch0_t1,...]`.
- Training uses a `StandardScaler` fitted **only on training samples**; the browser exports `mean.bin` and `scale.bin` for firmware inference. Any feature with near-zero standard deviation uses a divisor of 1 rather than 0.

The filter recurrence at 50 Hz is

```
dt   = 1 / 50
τ    = 1 / (2π × 5)
beta = dt / (τ + dt)
y[0] = x[0]
y[t] = y[t-1] + beta × (x[t] - y[t-1]), t >= 1
```

For direct gyroscope classification inputs, the 0.5 Hz HPF is:

```
dt = 1 / 50
tauHP = 1 / (2*pi*0.5)
alphaHP = tauHP / (tauHP + dt)
hp[0] = 0
hp[t] = alphaHP * (hp[t-1] + raw[t] - raw[t-1])
```

Feed this HPF output to the 5 Hz LPF, initialized to the **first HPF output (zero)**. The HPF state is gesture-local, not continuous across BOOT captures. Other input kinds retain their previous preprocessing, and the separate continuous Madgwick filter remains unmodified.

## Selectable input kinds

| Input kind | NAI4 ID | Samples per time point | Processing after LPF (before scaler) | N=100 input |
|---|---:|---|---|---:|
| **Accelerometer** | 1 | `ax, ay, az` | Resample, then subtract each axis's **mean within this gesture** | 300 |
| **Accel direction + magnitude** | **9** | `ux, uy, uz, m` | Resample filtered acceleration XYZ, then compute unit direction and magnitude **without centering** | **400** |
| **Gyroscope** | 2 | `gx, gy, gz` | Gyro HPF0.5 -> LPF5, resample, then subtract the gesture mean of each axis | 300 |
| **Accel + Gyro** | 3 | `ax, ay, az, gx, gy, gz` | Gyro-only HPF0.5 -> LPF5 plus accel LPF5, resample, then subtract the gesture mean of each of the six axes | 600 |
| **Estimated Velocity** | 5 | `vx, vy, vz` (m/s) | Per-gesture LPF acceleration, continuous device Madgwick orientation, gravity subtraction, gesture-local integration and endpoint correction, then resampling | 300 |
| **Accel XYZ + Roll + Pitch** | **10** | `ax,ay,az,roll,pitch` | LPF5 accel with continuous device quaternion-derived gravity roll/pitch in radians; no per-gesture mean subtraction | **500** |

**Mean centering** is within *each gesture*, and only for the three direct motion-feature kinds. It subtracts an axis mean but **does not divide by the gesture's standard deviation**. `StandardScaler` remains a separate training-derived operation afterward.

**Accel direction + magnitude** is a different representation: from the **same LPF-filtered acceleration vector** `a_f(t)` after resampling,

```
m = sqrt(ax_f² + ay_f² + az_f²)
if m > 0.001 g:
    [ux,uy,uz] = [ax_f,ay_f,az_f] / m
else:
    [ux,uy,uz] = [0,0,0]   # undefined direction near free fall
output = [ux,uy,uz,m]
```

At rest, its LPF magnitude is approximately **1 g**; under sustained free fall it is approximately **0 g**, with a filter transition delay. This is total accelerometer specific-force magnitude, *not a separate measurement of gravity*. To **preserve unit direction at the FP32 network input**, only each magnitude feature receives training-derived standardization; the direction channels have stored scaler `mean=0, scale=1` (with small errors possible after INT8 quantization).

For **Estimated Velocity**, a separate continuous 5 Hz IMU filter feeds Madgwick while a gesture-local LPF prepares acceleration for world-frame gravity subtraction. Velocity integration resets and is corrected at each BOOT-delimited gesture. The quaternion is never reset by BOOT. This feature is a short-gesture classifier input, **not** a drift-free inertial-navigation estimate. Startup gyroscope bias calibration is separate from mean centering.

## Model metadata and compatibility

- All new exports are `NAI4` version 4 with the **required `LPF5` flag `0x0008`**. Direct `gyro` and `accel+gyro` input models also **require `GYRO_HPF05` flag `0x0020`**. Other representations must not set this flag. Packed INT8 also carries `0x0002`.
- **ID 9** identifies the new unit-direction plus physical LPF magnitude feature. The old magnitude IDs 7 and 8 are not supported.
- The firmware explicitly **rejects older models** lacking LPF5 and old gyro/accel+gyro models lacking the new HPF0.5 flag. Other previously exported LPF5 models remain compatible when their representation is unchanged. This is intentional: the input distributions differ.
- To deploy after upgrading firmware, **retrain and re-export** from saved *raw* gesture data (or record fresh gestures), and deploy the new matching NAI4 model. Old `.npz` datasets *without* `raw_data`/`raw_offsets` cannot reproduce the required LPF.
- For direct input kinds, BOOT-captured gestures need not be recollected if their raw six-axis samples are present; for continuous orientation they must be recollected with matching quaternion samples. The class manager still permits deleting classes with recorded examples (with confirmation), and requires retraining after changes.

## Validation status

Host tests verify HPF0.5 + LPF5 C++/JavaScript parity, unchanged non-gyro representations, zero-offset behavior, and NAI4 compatibility flags. Previous LPF5 tests and browser/BLE integration tests remain available separately. Full ESP32-S3 PlatformIO compilation, physical-device inference, browser-training integration, and BLE deployment still require verification on the target system.

## Continuous Madgwick update

See **README_CONTINUOUS_MADGWICK.md** for the new device-streamed quaternion protocol, input representation 10, and the factory-installer limitation. The bundled factory binary is not yet rebuilt for continuous Madgwick.
