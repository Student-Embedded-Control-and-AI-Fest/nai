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
- **Flash Firmware** — first-time USB flashing of the ESP32-S3 INT8 runtime using ESP Web Tools.
- **About the Board** — hardware introduction, video demonstration, Indonesian purchase link, and preprocessing guide for all five input kinds.

There is no application server. TensorFlow.js training, dataset processing, NAI4 packaging, and BLE communication run in the browser.

## Video demo and board resources

- **[Watch the NoodleAI demo (demo.mp4)](https://github.com/Student-Embedded-Control-and-AI-Fest/nai/blob/main/demo.mp4)** — demonstration of the browser-to-device workflow.
- **[Waveshare ESP32-S3-Matrix hardware guide](https://docs.waveshare.com/ESP32-S3-Matrix)** — board specifications, pinout, and examples.
- **[CNC Store Bandung on Tokopedia](https://www.tokopedia.com/cncstorebandung/esp32-s3-development-board-dengan-rgb-led-matrix-wifi-bluetooth-sensor-gyroscope-qmi8658c-papan-pengembangan-iot-motion-control-1734224430143342142)** — a possible Indonesian seller; check current availability directly.
- **[About / preprocessing guide](README_ABOUT_BOARD.md)** — the text companion to the final website tab.

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

## Detailed preprocessing reference

See **[README_INPUT_PREPROCESSING.md](README_INPUT_PREPROCESSING.md)** for the exact filtering, unit-direction encoding, temporal resampling, gravity handling, channel layouts, and StandardScaler behavior of the five selectable input kinds.

## NAI4 motion representations

The training dropdown supports:

- Accelerometer
- **Accel direction + magnitude** (`ux,uy,uz,|a|`; ID 9)
- Gyroscope
- Accel + Gyro
- Estimated Velocity

The web trains five input kinds. Quaternion and Velocity + Quaternion are not selectable for new models. Previous high-pass-filter model packages are intentionally incompatible with this LPF5-only firmware; export fresh models from raw gesture data.

The defining Mode 1 behavior is unchanged: **BOOT supplies each gesture boundary, and the gesture is resampled to a fixed number of time positions before FCN inference.**

## Editing classes after capture or training

You can add another class at any time from **Capture & Train → Add class**,
even after locking the gesture length, collecting samples, or training.
The selected recording class switches to the newly added class, and existing
recordings (including their numeric class IDs) are preserved. If a model has
been trained, adding/removing a class invalidates that *browser-side* model;
record examples for the new class and train again before deploying.

**Lock gesture length & start** fixes only the normalized gesture length.
**Remove class** works whether or not the class has saved samples. When a class
has recorded gestures, the browser asks for confirmation, then removes exactly
those gestures and their raw samples, normalized samples, lengths, and durations.
The remaining class IDs are compacted consistently for `.npz` export and training;
all other gestures remain intact. Any previously trained browser model is invalidated,
so train and deploy again after changing classes. Deleted recordings cannot be
recovered unless you previously saved the original `.npz` dataset. The ESP32-S3
firmware supports at most **10 classes** and at most **31 UTF-8 bytes per class label**.

If a previously deployed model is still running on the board, it is not
automatically modified by edits in the browser; deploy the retrained model
when ready.

## Paper-ready classification metrics (Results & Curves)

After **Train model** completes, open **Results & Curves**. The site displays:

- Training and validation **loss/accuracy curves** with numeric final-epoch results and epoch axis labels.
- Final-epoch **training and validation accuracy**, including number of correct gestures and split sizes.
- Holdout-validation **macro precision**, **macro recall**, **macro F1**, and **support-weighted F1**.
- A per-class **precision / recall / F1 / support** table and **confusion matrix** (actual class in rows, predicted class in columns).
- **Export paper report (.zip)**: full six-decimal scores, per-class scores, confusion matrices (training and validation), full epoch history, per-gesture held-out predictions and class probabilities, split membership, run metadata, and PNG plots.

### Definitions and reproducibility

Metrics are computed *after the final training epoch* from predicted class IDs, separately for the training and validation subsets. For each class, precision is `TP/(TP+FP)`, recall is `TP/(TP+FN)`, and F1 is `2*precision*recall/(precision+recall)`; zero denominators yield 0. Macro averages are unweighted across all classes, while weighted averages use ground-truth class support. For this single-label classification task, micro F1 equals accuracy.

The existing **seed-42 stratified approximately 80/20 holdout** is preserved. Validation size is at least one gesture per class (so for very small datasets it can exceed 20%). A deterministic swap ensures every class appears in both subsets if an uneven distribution would otherwise leave a class missing, without changing the split sizes. A per-feature scaler is fitted to *training samples only*. The report records class names, counts, selected representation, topology, epochs, optimizer, batch size, split membership, validation predictions, and whether a coverage adjustment was needed. The exported values are from the **last epoch**, not the best validation epoch.

**Publication caveat:** These are *holdout validation* metrics, not independent test-set metrics and not on-device evaluation. Running many experiments, selecting the best representation/topology, and reporting only its same-split score will overestimate generalization. For a paper, reserve separate test participants or gesture sessions, or use appropriately designed subject-wise folds and independent testing. At least 5 validation gestures per class is recommended for less volatile per-class scores; this is a practical suggestion, not a statistical guarantee. Tiny validation supports are flagged in the UI.

To run the local pure-JavaScript smoke tests (requires Node.js):

```bash
node tests/test_preprocessing.cjs
node tests/test_class_deletion.cjs
```


**Current preprocessing:** The input-kind dropdown has five choices; see [README_INPUT_PREPROCESSING.md](README_INPUT_PREPROCESSING.md) for the LPF5 preprocessing and the new unit-direction + magnitude representation ID 9.

## Browser-based USB firmware installer

The **Flash Firmware** tab lets a student install the precompiled ESP32-S3
Matrix INT8 firmware by USB instead of installing PlatformIO. After flashing,
NoodleAI models are still trained locally and sent over BLE. The installer
requires a merged factory binary in `firmware/` and uses ESP Web Tools.
The install control is deliberately hidden until the published binary is found.

See **[README_FIRMWARE_INSTALLER.md](README_FIRMWARE_INSTALLER.md)** for
merging, publishing, erase/FFat warnings, and a one-command publishing helper.
