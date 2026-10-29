NoodleAI INT8-only (ESP32-S3 Matrix + generic ESP32/MPU6050)
============================================================
See README.md for build instructions, wiring and verification.

Build environments:
  pio run -e esp32-s3-matrix-int8
  pio run -e esp32-mpu6050-int8

This source drops the obsolete FP32 model inference/SRAM environments.
Float IMU acquisition, preprocessing, scaler, and TensorFlow.js training remain
necessary and are unchanged. NAI4 INT8/FFat, BLE protocol and Madgwick remain.

Generic ESP32 defaults: I2C SDA21 SCL22, MPU6050 at 0x68/0x69, gesture BOOT
GPIO0, optional LED GPIO2. Requires 4MB flash and FFat partition table.
