# NoodleAI INT8 web flasher — ESP32 and ESP32-S3

The **Flash Firmware** tab uses ESP Web Tools to install one of two *different*
4 MB factory images. **ESP Web Tools detects the chip family, not the IMU
board**: flash only the listed hardware combination.

| Board | IMU | Gesture button | Factory image |
| --- | --- | --- | --- |
| Waveshare ESP32-S3 Matrix | Onboard QMI8658 | BOOT | `noodleai-esp32s3-int8-factory.bin` |
| Generic ESP32-WROOM DevKit (4 MB flash) | External MPU6050 over I²C | External GPIO27 → GND (recommended) | `noodleai-esp32-int8-factory.bin` |

## Generic ESP32 + MPU6050 wiring

| ESP32 | MPU6050 breakout |
| --- | --- |
| 3V3 | VCC (3.3 V-compatible breakout) |
| GND | GND |
| GPIO21 | SDA |
| GPIO22 | SCL |

The external *normally-open* gesture button connects GPIO27 to GND; use
`INPUT_PULLUP`. No capacitor is needed for short, quiet wiring. GPIO0 remains
a boot strapping pin: do not hold it while powering up. Confirm that your
firmware build enables GPIO27 for the gesture button.

## 1. Produce chip-specific merged factory images

Run the command in each corresponding PlatformIO build directory. **The ESP32
bootloader begins at 0x1000, while ESP32-S3 begins at 0x0000.**

### Classic ESP32 + MPU6050

```bash
cd nai/firmware/.pio/build/esp32-mpu6050-int8
python -m esptool --chip esp32 merge-bin \
  -o noodleai-esp32-int8-factory.bin \
  --flash-mode dio --flash-size 4MB \
  0x1000 bootloader.bin \
  0x8000 partitions.bin \
  0xe000 ~/.platformio/packages/framework-arduinoespressif32/tools/partitions/boot_app0.bin \
  0x10000 firmware.bin
```

### ESP32-S3 Matrix

```bash
cd nai/firmware/.pio/build/esp32-s3-matrix-int8
python -m esptool --chip esp32s3 merge-bin \
  -o noodleai-esp32s3-int8-factory.bin \
  --flash-mode dio --flash-size 4MB \
  0x0000 bootloader.bin \
  0x8000 partitions.bin \
  0xe000 ~/.platformio/packages/framework-arduinoespressif32/tools/partitions/boot_app0.bin \
  0x10000 firmware.bin
```

Use the actual `platformio.ini` flash mode/frequency and partitions for the
build. The fixed offsets above match this NoodleAI 4 MB layout, but do not
assume they apply to other projects. `merge-bin` success alone does not prove
the board firmware is compatible.

## 2. Publish both binaries (from `nai/web/`)

```bash
python3 tools/publish_firmware.py --chip esp32 \
  ../firmware/.pio/build/esp32-mpu6050-int8/noodleai-esp32-int8-factory.bin
python3 tools/publish_firmware.py --chip esp32s3 \
  ../firmware/.pio/build/esp32-s3-matrix-int8/noodleai-esp32s3-int8-factory.bin
```

The helper validates the merged-image layout, copies the correct filename into
`web/firmware/`, and updates **only that chip family's manifest build**.
Both entries are retained. You may also copy the binaries manually and keep
the supplied two-build `manifest.json`.

**Important:** The downloadable web template includes the previously packaged
ESP32-S3 image but **does not contain your locally built generic ESP32
binary**. Publish your own current images before deploying this website.
For the newest continuous Madgwick and gyro-HPF features, rebuild both from
the latest firmware source rather than reusing stale factory binaries.

## 3. Deploy GitHub Pages

```bash
cd nai
# if 'web/' is the folder served by GitHub Pages:
git add web/index.html web/style.css web/js/firmware-installer.js \
  web/tools/publish_firmware.py web/README_FIRMWARE_INSTALLER.md \
  web/firmware/manifest.json web/firmware/*.bin
git commit -m "Enable ESP32 and ESP32-S3 NoodleAI web flashing"
git push
```

If the `web/` directory is a **separate** Git repository, run Git from inside
that directory and omit the `web/` prefixes. Confirm the published manifest
URL loads and both `.bin` paths return HTTP 200. The installer requires both
images before exposing the Flash button, to avoid offering a chip with a
missing image. Use HTTPS (or localhost) and a desktop Chromium-based browser
with Web Serial support.

For a first install, connect the board by USB, open **Flash Firmware**, and
click **Connect & Install Firmware**. Once flashed, use BLE to upload the NAI4
INT8 model. Firmware installation may erase saved FFat models; back up
important data and verify partition compatibility.

### Application-only update

With the *same existing partition layout*, an application-only write at
`0x10000` may preserve FFat, but must use the matching board image and port.
For a generic ESP32, for example:

```bash
python -m esptool --chip esp32 --port /dev/ttyUSB0 write-flash \
  0x10000 firmware/.pio/build/esp32-mpu6050-int8/firmware.bin
```

This example is for a local `nai/` project root; adapt the serial port and
paths for your system. Do not use it across incompatible partitions.
