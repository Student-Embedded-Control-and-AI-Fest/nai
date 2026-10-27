# Browser-based NoodleAI firmware installation (ESP32-S3 Matrix)

The NoodleAI Mode 1 website includes a **Flash Firmware** tab powered by
[ESP Web Tools](https://esphome.github.io/esp-web-tools/). Firmware flashing
uses USB/Web Serial, **not BLE**. BLE is used later for model deployment.

## 1. Publish the compiled INT8 factory image

After building the PlatformIO environment `esp32-s3-matrix-int8` and merging
bootloader, partition table, boot_app0, and application into one factory image,
copy it into the website:

```bash
# Run in the website's repository root (the folder with index.html):
python3 tools/publish_firmware.py \
  firmware/.pio/build/esp32-s3-matrix-int8/noodleai-esp32s3-int8-factory.bin
```

**Important:** If the web files are in a `web/` subfolder instead of the `nai/` repository
root, use the sibling firmware path from that subfolder:

```bash
python3 tools/publish_firmware.py \
  ../firmware/.pio/build/esp32-s3-matrix-int8/noodleai-esp32s3-int8-factory.bin
```

Or supply the **absolute path** to the merged `.bin`.

If the merged binary has not yet been created, run in the PlatformIO
`.pio/build/esp32-s3-matrix-int8/` directory:

```bash
python -m esptool --chip esp32s3 merge-bin \
  -o noodleai-esp32s3-int8-factory.bin \
  --flash-mode dio --flash-size 4MB \
  0x0000 bootloader.bin \
  0x8000 partitions.bin \
  0xe000 ~/.platformio/packages/framework-arduinoespressif32/tools/partitions/boot_app0.bin \
  0x10000 firmware.bin
```

Verify the offsets against your board's actual PlatformIO upload settings.
The current NoodleAI ESP32-S3 Matrix `platformio.ini` specifies DIO and 4MB
flash. Confirm these settings on your physical board before publication.
A merged image is not universal across all ESP32-S3 boards.

The publishing helper copies the image to:

`firmware/noodleai-esp32s3-int8-factory.bin`

and writes the single-build ESP Web Tools manifest to:

`firmware/manifest.json`

The helper assigns a SHA-based firmware version by default, so updating the
binary updates the published version. Run with `--version 1.2.3-int8` to
specify your own version string. The helper validates the basic image header,
not the underlying board, firmware features, or partition layout.

## 2. Publish GitHub Pages

Commit the static web files, `firmware/manifest.json`, and the merged binary.
Confirm Pages is configured to serve the directory containing `index.html`
(e.g. repository root). If the binary is missing, the website safely hides
the install button and displays the reason. Avoid Git LFS for the binary when
GitHub Pages is serving it; use a normal Git-tracked `.bin` file.

```bash
git add index.html style.css js/app.js js/firmware-installer.js \
  firmware/manifest.json firmware/noodleai-esp32s3-int8-factory.bin \
  README.md README_FIRMWARE_INSTALLER.md tools/publish_firmware.py
git commit -m "Add ESP32-S3 INT8 web installer"
git push
```

The website must be served over **HTTPS** (or localhost for development) and
the browser must support Web Serial. Desktop Chrome or Edge is recommended.

## 3. First-time installation

1. Connect the **Waveshare ESP32-S3 Matrix** using a data-capable USB cable.
2. Open the NoodleAI page → **Flash Firmware**.
3. Choose **Connect & Install Firmware** and select the USB serial device.
4. Follow ESP Web Tools prompts. On some boards, BOOT/RESET may be needed
   for entering download mode.
5. When flashing completes, reconnect the device over BLE on the **Live IMU**
   and **Deploy & Infer** tabs, then deploy a compatible NAI4 INT8 model.

**WARNING:** A factory install may erase flash, including saved FFat models.
Back up model packages/datasets before flashing. The installer manifest asks
ESP Web Tools to present an erase-choice prompt when treated as a new install.
Do not assume firmware upgrades preserve data if the partition table changes.

## 4. Application-only CLI update (advanced)

If the existing board already has the matching bootloader/partition layout,
an application-only update generally preserves FFat. Example:

```bash
python -m esptool --chip esp32s3 --port /dev/ttyACM0 \
  --baud 460800 write-flash 0x10000 \
  ../firmware/.pio/build/esp32-s3-matrix-int8/firmware.bin
```

Confirm the correct port, application offset, and partition table before
using this. Do not use application-only updates across incompatible flash
partition layouts or firmware variants.

## 5. Notes for maintainers

- The website hosts **INT8 firmware only**; this installer does not build or
  flash FP32 firmware. NAI4 models are deployed separately over BLE.
- ESP Web Tools chooses the detected *chip family*. It cannot distinguish
  between all physical ESP32-S3 board revisions. Label the target clearly.
- Keep the published firmware aligned with the web's NAI4 preprocessing and
  INT8 model format. Building a new binary without retesting compatibility
  does not guarantee that trained models will execute correctly.
- The installer intentionally does **not** attempt to flash automatically.
- ESP Web Tools source and documented manifest schema:
  https://esphome.github.io/esp-web-tools/
