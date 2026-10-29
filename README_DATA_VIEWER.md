# NoodleAI Mode 1 — Data Viewer tab

## Where it appears

Tab **06 Data Viewer**, directly before **07 About the Board**. This edition adds optional orientation diagnostics. The viewer is independent of the live capture and model-training dataset, and does not require a BLE connection.

## How to use

1. Open the **Data Viewer** tab, then choose **Load .npz dataset**.
2. Pick a NoodleAI six-axis `.npz` file. The browser parses it locally (no upload).
3. Choose a gesture **class** in the dropdown. Six separate plots show `ax`, `ay`, `az` (g), `gx`, `gy`, `gz` (deg/s).
4. The default display overlays up to 32 example gesture traces and plots the **mean of every gesture** in the selected class. Select a specific gesture in the second dropdown to inspect one sample.
5. Change the class. **Y-axis bounds stay fixed**: all acceleration panels share one domain, and all gyroscope panels share another. Bounds were computed from all loaded classes, include zero, and are padded by 5.5%.

## NPZ compatibility

Supported NPY arrays in the archive:

- `X.npy`: float data in one of these six-channel shapes:
  - `[samples, time_steps*6]` (classic NoodleAI export, time-major flattened)
  - `[samples, time_steps, 6]`
  - `[samples, 6, time_steps]`
- `y.npy`: one class ID or class name per sample.
- `labels.npy`: optional class labels (Unicode or byte strings). Otherwise numeric classes are named `Class N`.
- `normalized_length.npy` / `window_length.npy`: optional. If included, it must agree with `X`.

Newer NoodleAI archives may also contain `raw_data.npy`, `raw_offsets.npy`, and `quat_data.npy`. The Data Viewer always displays the **stored fixed-length six-channel X data** without running the model preprocessing again. When all three arrays are present and valid, it additionally shows an optional **Orientation diagnostics** section, collapsed by default:

- `qw, qx, qy, qz`: four quaternion components resampled to the normalized gesture timeline using spherical interpolation (SLERP), unit normalization, and local quaternion sign alignment.
- `roll` and `pitch` (degrees): derived from the resampled continuous Madgwick quaternion using the same conventions as the NoodleAI training code.
- Fixed axes across every class and gesture: each quaternion component `[-1,+1]`, roll `[-180,+180]` degrees, and pitch `[-90,+90]` degrees.
- Individual quaternion and angle trajectories are shown, with up to 32 class examples, but **no quaternion mean**: `q` and `-q` denote the same orientation and averaging components is generally misleading.

The new schema stores `quat_data` as `[total_raw_samples,4]` in `[qw,qx,qy,qz]` order and `raw_offsets` as `[gestures+1]`. If quaternion history is absent, the orientation section stays hidden. If it is malformed, the six-channel plots remain usable and the viewer explains why orientation was skipped. The viewer does not interpolate or infer the missing intervals between separate BOOT presses.


Some very old archives that store Python object arrays, a different number of channels, or use a custom schema may not load. Inspect an incompatible file to extend the reader; do not convert object arrays with `allow_pickle` without understanding their provenance.

## Design notes

- The viewer does **not** fit or apply the StandardScaler, zero the gyro, subtract per-window means, add an HPF or LPF, or run Madgwick. It plots what is stored in `X`.
- The x-axis is the normalized sample index. It is **not** elapsed seconds because BOOT-held gestures may have different original durations before resampling.
- This is a web-only change. Quaternion visualization does not recompute Madgwick or affect model inputs. The firmware is **unchanged** and the already-bundled factory firmware may be older than the latest source.
- The site uses the existing JSZip CDN dependency to read `.npz`. TensorFlow.js is not used by the viewer.

## Testing

The Data Viewer parser and selector logic have been tested using synthetic legacy arrays with 2-D flattened, 3-D time-major, and 3-D channel-major six-axis `X`. The orientation parser has been tested with synthetic continuous quaternions, irregular original gesture lengths, ±quaternion sign flips, roll/pitch calculation, class selection, and invalid quaternion data. A synthetic NPZ with `quat_data` is included for local testing. Browser screenshot testing was blocked by this execution environment; a local browser smoke test remains recommended. Hardware testing is not required for this read-only view.
