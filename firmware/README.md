# NoodleAI firmware — INT8-only, two hardware targets

This is the source-only port made from the supplied `firmware(6).zip`. The **existing NAI4 model format, INT8 FCNFile/FFat executor, BLE service UUIDs, 208-byte IMU/quaternion notifications, model A/B deployment, and Mode-1 gesture segmentation** are shared across both boards.

## Targets

| PlatformIO environment | Board | IMU | Feedback | Gesture trigger |
|---|---|---|---|---|
| `esp32-s3-matrix-int8` | ESP32-S3 Matrix | QMI8658 (I2C GPIO11/12) | 8x8 NeoPixel GPIO14 | BOOT GPIO0 |
| `esp32-mpu6050-int8` | Generic ESP32 DevKit/WROOM-32, 4 MB flash | MPU6050 via I2C | Serial, optional GPIO2 LED | BOOT GPIO0 (or external button) |

### Generic ESP32 wiring

| ESP32 DevKit | MPU6050 (GY-521) |
|---|---|
| 3V3 | VCC (use **3.3 V-compatible** module) |
| GND | GND |
| GPIO21 | SDA |
| GPIO22 | SCL |

MPU6050 AD0 grounded gives I2C address `0x68`; AD0 high gives `0x69`. The driver probes both. I2C pull-ups must reference **3.3 V**.

For gestures, use the on-board **BOOT (GPIO0)** button after the firmware has started, or connect an external normally-open pushbutton between **GPIO27** and GND and enable `-D NAI_GESTURE_BUTTON_PIN=27` in the generic environment. Do **not** hold BOOT during reset, because GPIO0 also selects the ROM downloader. GPIO2 LED is optional; set `-D NAI_STATUS_LED_PIN=-1` to disable it. Pin availability varies by board.

### MPU6050 configuration

Direct I2C register driver `src/nai_mpu6050.h` needs no additional MPU library. It configures ±4 g (8192 LSB/g), ±1000°/s (32.8 LSB/°/s), internal DLPF about 42–44 Hz, sensor output rate 100 Hz; NoodleAI samples it at nominal 50 Hz. Raw output is `[ax,ay,az]` **in g** and `[gx,gy,gz]` **in degrees/s**, same application interface as QMI8658. Keep the board still for startup gyro-bias calibration (~3.5 seconds). IMU body-axis mounting differs between boards: **verify signs/orientation**, and prefer collecting/retraining on the target board rather than assuming a QMI8658-trained model transfers unchanged.

## Build and upload

Requires local PlatformIO, platform `espressif32@6.12.0` and the Noodle library checkout. The original dependency path `symlink:///home/auralius/works/noodle` is retained; edit it in `platformio.ini` if your checkout is elsewhere.

```bash
cd firmware
pio run -e esp32-mpu6050-int8
pio run -e esp32-mpu6050-int8 -t upload
pio device monitor -b 115200

# Existing S3 Matrix is retained:
pio run -e esp32-s3-matrix-int8
pio run -e esp32-s3-matrix-int8 -t upload
```

The generic ESP32 uses UART serial (`/dev/ttyUSB*`) rather than ESP32-S3 native USB CDC. Select the correct device via `-d` or `upload_port` as appropriate.

## Existing inference/preprocessing contract retained

- NAI4 v4 **INT8-only** inference. `cfg.bin` must carry INT8 flag `0x0002` and LPF5 flag `0x0008`; gyro/accel+gyro models require HPF0.5 flag `0x0020`. New continuous-orientation models carry `0x0010`.
- `wNN.bin` packed INT8 weights, `pNN.bin` quantization metadata, and the same training-derived `mean.bin`/`scale.bin` (float statistics) from the web trainer. **FP32 neural-network inference and SRAM-FP32 backend have been removed, not the float32 sensor preprocessing/scaler or browser training.**
- Accelerometer classifier: 5 Hz LPF. Direct gyroscope classifier: **0.5 Hz HPF followed by 5 Hz LPF**. Madgwick inputs: bias-corrected gyro + independent continuous 5 Hz LPF, **no gyro HPF**.
- Continuous Madgwick estimates are initialized once from gravity, updated between gestures; each press/release resets only the gesture buffer, not the quaternion. BLE notifications still contain six float IMU channels and four float quaternion components per sample, 208 bytes per batch of five.
- A/B FFat model slots, BLE upload, NAI4 flags, model packaging, class predictions and mode switching are unchanged.

## Browser web installer

This upload contains **source only**. It does not update the existing web flasher binary or manifest. Build and verify a device-specific factory image before publishing. **Classic ESP32 flash offsets differ from the ESP32-S3**: the conventional ESP32 uses bootloader `0x1000`, partition table `0x8000`, boot_app0 `0xe000`, app `0x10000`; use esptool version-appropriate `merge_bin`/`merge-bin` with the actual build artifacts. The ESP32-S3 bootloader generally goes at `0x0000`. Do not publish a generic ESP32 image under the S3 board manifest.

## Validation status

- Tested offline: MPU6050 I2C probing/configuration/signed conversion using a C++ mock; code-level checks of build targets and preprocessor structure.
- **Not yet validated:** PlatformIO builds (toolchain and the original external Noodle checkout unavailable in this execution environment), flashing, FFat, BLE transfer, physical gesture inference, sensor axis parity, or timing/heap. The original S3 source already had working INT8 deployments, but this shared-source revision should be rebuilt and regression-tested on hardware.
