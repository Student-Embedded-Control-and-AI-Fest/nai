# NoodleAI Mode 1 — Data Viewer tab

## Where it appears

Tab **06 Data Viewer**, directly before **07 About the Board**. The viewer is independent of the live capture and model-training dataset, and does not require a BLE connection.

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

Newer NoodleAI archives may also contain `raw_data.npy`, `raw_offsets.npy`, and `quat_data.npy`; the Data Viewer intentionally displays the **stored fixed-length six-channel X data**, for direct comparison between class labels, rather than preprocessing it again.

Some very old archives that store Python object arrays, a different number of channels, or use a custom schema may not load. Inspect an incompatible file to extend the reader; do not convert object arrays with `allow_pickle` without understanding their provenance.

## Design notes

- The viewer does **not** fit or apply the StandardScaler, zero the gyro, subtract per-window means, add an HPF or LPF, or run Madgwick. It plots what is stored in `X`.
- The x-axis is the normalized sample index. It is **not** elapsed seconds because BOOT-held gestures may have different original durations before resampling.
- This is a web-only change. The firmware is **unchanged** and the already-bundled factory firmware may be older than the latest source.
- The site uses the existing JSZip CDN dependency to read `.npz`. TensorFlow.js is not used by the viewer.

## Testing

The Data Viewer parser and selector logic have been tested using synthetic legacy arrays with 2-D flattened, 3-D time-major, and 3-D channel-major six-axis `X`; a synthetic NPZ is supplied separately for interactive smoke testing. Hardware testing is not required for this read-only view.
