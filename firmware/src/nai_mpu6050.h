#pragma once

#include <Arduino.h>
#include <Wire.h>

// Register-level MPU-6050 adapter. No additional IMU library is required.
// Output units deliberately match SensorQMI8658: acceleration [g],
// gyroscope [degrees/second]. Raw axes preserve the sensor's chip frame;
// board-mounting differences must be calibrated/validated before model reuse.
class NaiMpu6050 {
public:
    bool begin(TwoWire &bus) {
        _wire = &bus;
        uint8_t identity = 0;
        bool found = false;
        const uint8_t candidates[2] = {0x68, 0x69};
        for (uint8_t a : candidates) {
            _address = a;
            if (readRegister(0x75, identity) && (identity & 0x7e) == 0x68) {
                found = true;
                break;
            }
        }
        if (!found) return false;
        // Wake the sensor; use the PLL clock referenced to X gyro.
        if (!writeRegister(0x6b, 0x01) || !writeRegister(0x6c, 0x00)) return false;
        delay(100);
        // Internal DLPF on (~42/44 Hz), output rate 1 kHz / (1+9) = 100 Hz.
        // Application samples the latest readings at 50 Hz; its own 5 Hz LPF
        // and gyro classification 0.5 Hz HPF remain unchanged.
        if (!writeRegister(0x1a, 0x03) || !writeRegister(0x19, 0x09)) return false;
        // ±1000 dps -> 32.8 LSB/(deg/s), ±4 g -> 8192 LSB/g.
        if (!writeRegister(0x1b, 0x10) || !writeRegister(0x1c, 0x08)) return false;
        delay(100);
        return true;
    }

    bool read(float result[6]) {
        if (!_wire || !result) return false;
        _wire->beginTransmission(_address);
        _wire->write(uint8_t(0x3b)); // ACCEL_XOUT_H
        if (_wire->endTransmission(false) != 0) return false;
        if (_wire->requestFrom(_address, uint8_t(14), uint8_t(true)) != 14) return false;
        int16_t raw[7];
        for (int i = 0; i < 7; ++i) {
            const uint16_t hi = static_cast<uint8_t>(_wire->read());
            const uint16_t lo = static_cast<uint8_t>(_wire->read());
            raw[i] = static_cast<int16_t>((hi << 8) | lo);
        }
        // Includes the two temperature bytes at raw[3], which are skipped.
        result[0] = raw[0] / 8192.0f;
        result[1] = raw[1] / 8192.0f;
        result[2] = raw[2] / 8192.0f;
        result[3] = raw[4] / 32.8f;
        result[4] = raw[5] / 32.8f;
        result[5] = raw[6] / 32.8f;
        return true;
    }

    uint8_t address() const { return _address; }

private:
    bool writeRegister(uint8_t reg, uint8_t value) {
        _wire->beginTransmission(_address);
        _wire->write(reg);
        _wire->write(value);
        return _wire->endTransmission(true) == 0;
    }
    bool readRegister(uint8_t reg, uint8_t &value) {
        _wire->beginTransmission(_address);
        _wire->write(reg);
        if (_wire->endTransmission(false) != 0) return false;
        if (_wire->requestFrom(_address, uint8_t(1), uint8_t(true)) != 1) return false;
        value = static_cast<uint8_t>(_wire->read());
        return true;
    }
    TwoWire *_wire = nullptr;
    uint8_t _address = 0x68;
};
