# NoodleAI Mode 1 — Input Representations and Preprocessing

**Applies to:** the ESP32-S3 Matrix, BOOT-triggered **Mode 1** Web Studio with the **HPF/LPF + Acceleration Magnitude** update. This document describes the behavior implemented in `js/core.js` and selected through `js/app.js`; it is not a description of the continuous BL702 Mode 2 pipeline.

## 1. Common acquisition and processing

1. Hold **BOOT** during a gesture; release it to finish recording. The IMU supplies six raw channels in this order: `ax, ay, az, gx, gy, gz`. Nominal acquisition is **50 Hz** (`ax/ay/az` in **g**, `gx/gy/gz` in **degrees/s**).
2. Every gesture may contain a different number of original samples. After representation-specific processing, each gesture is **linearly resampled to `N` positions** (default **100**). `N=100` is **not** a 100-Hz sensor sampling rate and does **not** make each gesture last two seconds.
3. Arrange channels in **time-major, interleaved** order: all channels for sample 0, then all channels for sample 1, etc.
4. Apply a **feature-wise StandardScaler fitted only to training examples**, then feed the resulting vector into the MLP. The same stored scaler is used for validation and on-device inference.

Some operations occur *before* temporal resampling (filtering, orientation and velocity estimation); others occur *after* resampling (legacy centering, acceleration magnitude computation and StandardScaler). Their order matters.

## 2. What each Input representation does

The **HPF → LPF** checkbox is initially **ON** for direct sensor inputs. It can be turned **OFF** for Accelerometer, Gyroscope, and Accel + Gyro, in which case those inputs use the **legacy per-gesture centering** path instead. It is **forced ON** for Accel + magnitude and is **not used** for the derived Quaternion/Velocity selections.

| Input selection | NAI4 ID | Channels per time step (exact order) | Processing before resampling | Processing after resampling | FCN width at `N=100` |
|---|---:|---|---|---|---:|
| **Accelerometer** | 1 | `ax, ay, az` | **ON:** HPF → LPF on raw acceleration. **OFF:** no filter. | **ON:** no gesture centering. **OFF:** subtract the per-gesture mean of each axis. Then StandardScaler. | **300** |
| **Accel + magnitude** | 7 | `ax, ay, az, m` | **Always:** HPF → LPF on raw acceleration. | Resample `ax,ay,az`, calculate `m = sqrt(ax² + ay² + az²)` from those three resampled filtered values; **no** gesture centering. Then StandardScaler. | **400** |
| **Gyroscope** | 2 | `gx, gy, gz` | **ON:** HPF → LPF on raw angular rates. **OFF:** no filter. | **ON:** no gesture centering. **OFF:** subtract the per-gesture mean of each axis. Then StandardScaler. | **300** |
| **Accel + Gyro** | 3 | `ax, ay, az, gx, gy, gz` | **ON:** HPF → LPF on all six raw channels. **OFF:** no filter. | **ON:** no gesture centering. **OFF:** subtract the per-gesture mean of each of the six channels. Then StandardScaler. | **600** |
| **Relative Quaternion** | 4 | `qw, qx, qy, qz` | Estimate orientation from raw acceleration and gyro using **Madgwick IMU** (`beta=0.10`); express each quaternion relative to the gesture's starting orientation. **No UI HPF/LPF.** | Linearly resample quaternion components; **renormalize to unit length** at each position. Then StandardScaler; **no per-gesture channel centering**. | **400** |
| **Estimated Velocity** | 5 | `vx, vy, vz` | Estimate orientation with Madgwick; rotate measured acceleration into the world frame, **subtract world-frame gravity**, convert `g` to m/s², integrate using the trapezoidal rule. Apply a **linear endpoint correction** to force the estimated velocity at both ends to zero. **No UI HPF/LPF.** | Resample velocity; then StandardScaler. **No per-gesture mean subtraction**. | **300** |
| **Velocity + Quaternion** | 6 | `vx, vy, vz, qw, qx, qy, qz` | Compute velocity and relative quaternion exactly as above. **No UI HPF/LPF.** | Resample the seven channels; **renormalize quaternion channels only**; then StandardScaler. **No per-gesture mean subtraction**. | **700** |

### Acceleration magnitude is **not** normalized direction and is **not** gravity magnitude

For the new four-channel input, after filtering and resampling:

```text
m[t] = sqrt(ax[t]^2 + ay[t]^2 + az[t]^2)
features[t] = [ax[t], ay[t], az[t], m[t]]
```

This is the magnitude of the **filtered, measured acceleration**. It is **not** an estimate of gravity's magnitude, and the XYZ components are **not** divided by `m[t]`. The HPF suppresses approximately constant gravity/offset components, but **direct accelerometer inputs do not explicitly compensate for gravity changes during device rotation**. Only the velocity-derived path explicitly rotates acceleration and subtracts gravity.

## 3. HPF/LPF behavior (direct input kinds only)

**Order:** `raw IMU at 50 Hz → first-order HPF (0.10 Hz) → first-order LPF (8 Hz) → temporal resampling`.

For each chosen axis, the implementation uses `dt = 1 / sample_rate`, `tau = 1 / (2*pi*cutoff_hz)` and:

```text
alpha = tauHP / (tauHP + dt)
hp[t] = alpha * (hp[t-1] + x[t] - x[t-1])

beta  = dt / (tauLP + dt)
lp[t] = lp[t-1] + beta * (hp[t] - lp[t-1])
```

The filter state **resets for each BOOT gesture**, with the first filtered output set to **zero**. Thus, the cutoff frequencies are defined in physical time, **before** resampling, but the short-gesture startup response is not identical to continuously running the same filter on a Mode 2 stream.

With the checkbox **OFF**, no HPF/LPF is applied. Instead, direct sensor inputs are resampled and then **centered separately for that gesture and each channel**: `x'[t,c] = x[t,c] - mean_t(x[t,c])`. That legacy operation does **not** divide by the gesture's standard deviation.

**Important:** The derived quaternion/velocity processing in this **Mode 1 web version** does **not** use the new HPF/LPF stage, even though the continuous **Mode 2** implementation has a separate LPF-before-Madgwick configuration. Do not assume their derived-feature processing is identical.

## 4. Orientation and velocity details

- **Relative Quaternion:** Madgwick estimates an orientation using accelerometer and gyroscope samples. The initial accelerometer reading establishes a reference alignment; the resulting quaternion represents rotation **relative to the beginning of the BOOT gesture**. Output order is **scalar first**: `[qw,qx,qy,qz]`. The quaternion is renormalized after interpolation because component-wise interpolation does not necessarily preserve unit magnitude.
- **Estimated Velocity:** Orientation rotates acceleration into the world frame. A nominal gravity vector `[0,0,1] g` is removed; the result is multiplied by **9.80665 m/s² per g** and integrated over the raw sample sequence. A linear residual/endpoint correction forces the estimated velocity to end near zero. This is a **short-gesture classification feature**, not drift-free navigation or a reliable absolute displacement estimate.
- **Velocity + Quaternion:** Concatenates those representations at every time step: **velocity first, quaternion second**.

If supported and successfully completed by the matching firmware, **startup gyroscope zero-rate calibration** subtracts a stationary-estimated gyro bias before the readings reach feature construction; it is a **separate** operation from the per-gesture centering or HPF/LPF checkbox. The browser does not independently perform this device calibration.

## 5. StandardScaler: applied to **all seven** kinds

For every output position/channel, the browser fits training-set statistics and computes:

```text
z[j] = (feature[j] - train_mean[j]) / train_scale[j]
```

`j` is the **flattened feature index**. This is **per-feature** scaling: the 100 time positions of the same channel can have *different* fitted means and scales. It is **not** per-gesture normalization; a powerful gesture can still differ from a weak gesture after scaling.

- The mean and population standard deviation are computed **only from the training split** (`ddof=0`).
- If the standard deviation is non-finite or **≤ 1e-12**, the stored divisor is **1.0**.
- For `N=100`, the new four-channel input has **400 stored means and 400 scales**.
- INT8 inference additionally **quantizes the standardized model input** using model-specific quantization parameters; it does not replace StandardScaler.

## 6. Saved data, exported models, and compatibility

- New filtered or Accel + magnitude training **requires the original raw six-axis samples** (`raw_data` and `raw_offsets` in a NAI4-style dataset). A legacy dataset containing only normalized/centered `X` rows **cannot reconstruct** the pre-filter raw signal or the new magnitude channel.
- Direct sensor features with HPF/LPF use NAI4 **flag `0x0004`**. The new Accel + magnitude kind has **representation ID 7**. Packed INT8 models additionally set flag `0x0002`. These flags are stored in `cfg.bin`.
- An **older model without flag `0x0004`** follows its original direct-input **centering** path. Do not change the preprocessing of deployed legacy models implicitly.
- New filtered models and ID-7 models require the **matching updated Mode 1 firmware**, for whichever FP32 or INT8 backend is used. Older firmware cannot be assumed to understand them.
- Changing a representation or enabling/disabling filtering **changes the model input contract**. Retrain and redeploy; do not only toggle a checkbox for an existing model.

## 7. Example input vectors (`N=100`)

```text
Accelerometer           [ax0, ay0, az0, ax1, ay1, az1, ...]          300 values
Accel + magnitude       [ax0, ay0, az0, m0, ax1, ay1, az1, m1, ...] 400 values
Gyroscope               [gx0, gy0, gz0, gx1, gy1, gz1, ...]          300 values
Accel + Gyro            [ax0, ay0, az0, gx0, gy0, gz0, ...]          600 values
Relative Quaternion     [qw0, qx0, qy0, qz0, qw1, ...]              400 values
Estimated Velocity      [vx0, vy0, vz0, vx1, vy1, vz1, ...]          300 values
Velocity + Quaternion   [vx0, vy0, vz0, qw0, qx0, qy0, qz0, ...]     700 values
```

**Where to inspect the implementation:** `js/app.js` (`selectedDirectHplp`, `refreshFilterControl`, training/model export); `js/core.js` (`REP`, `filterRaw6Hplp`, `buildRepresentationOne`, `deriveQuatVelocity`, `fitScaler`, `exportNai4`, `exportNai4Int8`). The matching firmware implements the same model-defined direct-feature filter and representation dispatch.

**Validation status:** The supplied web patch was syntax-checked and tested for the direct 400-feature path, flags and legacy behavior. End-to-end real-hardware equivalence for **all** seven representations has **not** been independently verified in this deliverable.
