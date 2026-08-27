# NoodleAI Web

A static, client-side NoodleAI application for the ESP32-S3-Matrix/QMI8658 platform.

**Collect. Train. Upload. Infer.**

The application mirrors the desktop NoodleAI workflow:

- **Live IMU** — raw 6-axis accelerometer + gyroscope plots over Web Bluetooth.
- **Dataset & Train** — define labels, record BOOT-delimited gestures, save/load Python-compatible `.npz` datasets, select a NAI4 motion representation, and train an N-layer MLP with TensorFlow.js.
- **Training Curves** — train/validation loss and accuracy history.
- **Deploy & Infer** — save the binary `.nai` package, deploy it transactionally over BLE, switch device modes, and show the latest Noodle prediction.

There is no application server. TensorFlow.js training, dataset processing, NAI4 packaging, and BLE communication all happen in the browser.

## Run locally

Web Bluetooth needs a secure browser context. `localhost` is allowed for development:

```bash
python3 -m http.server 8000
```

Open:

```text
http://localhost:8000
```

The Python command is only a static file server. It does **not** train models or handle Bluetooth.

## Publish with GitHub Pages

1. Create a GitHub repository and put these files in the repository root.
2. Push to `main`.
3. Open **Settings → Pages**.
4. Under **Build and deployment**, choose **Deploy from a branch**.
5. Select `main` and `/ (root)`.
6. Save and open the HTTPS Pages URL GitHub gives you.

GitHub Pages provides the secure HTTPS context Web Bluetooth expects.

## Browser support

Use a Chromium-based browser with Web Bluetooth support. On Chrome/Linux, `navigator.bluetooth` may require enabling **Experimental Web Platform features** in `chrome://flags`.

## Dependencies

Pinned CDN dependencies:

- TensorFlow.js 4.22.0
- JSZip 3.10.1

Because they are loaded from CDN, the first page load needs internet access. Model training and BLE data stay local in the browser.

## NAI4 representations

The browser stores the original raw 6-axis gesture and can train any supported representation from the same dataset:

- Accelerometer
- Gyroscope
- Accel + Gyro
- Relative Quaternion
- Estimated Velocity
- Velocity + Quaternion

Derived representations mirror the current NAI4 motion pipeline (6-axis Madgwick orientation, gravity compensation, velocity integration with endpoint correction, then temporal normalization).

## Dataset compatibility

Saved `.npz` files contain the same NAI4 fields as the desktop GUI, including `X`, `y`, `labels`, `normalized_length`, `raw_lengths`, `durations_ms`, and—when available—`raw_data` + `raw_offsets`.

Older normalized-only NAI3 datasets can still train the raw accelerometer/gyro representations. Quaternion/velocity modes require the retained raw 6-axis gesture data.

## Model deployment

The page uses the existing NoodleAI BLE service and transactional A/B model update protocol. Each binary file is sent with its size and CRC32; the device responds with `FILE_READY`, `FILE_OK`, and finally `MODEL_OK` after Noodle validates and activates the new slot.

The NAI4 archive contains binary float32 files such as:

```text
cfg.bin
mean.bin
scale.bin
w00.bin
b00.bin
...
```

TensorFlow.js Dense kernels `[I,O]` are transposed to the Noodle `[O,I]` layout before serialization.
