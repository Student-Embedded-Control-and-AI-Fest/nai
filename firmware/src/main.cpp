#include <Arduino.h>
#include <new>
#include <math.h>
#include <Wire.h>
#include <FFat.h>
#include <SensorQMI8658.hpp>
#include <Adafruit_NeoPixel.h>

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>

#include "NoodleAIModel.h"

// -----------------------------------------------------------------------------
// Waveshare ESP32-S3-Matrix hardware
// -----------------------------------------------------------------------------
static constexpr int PIN_IMU_SDA = 11;
static constexpr int PIN_IMU_SCL = 12;
static constexpr int PIN_MATRIX = 14;
static constexpr int PIN_BOOT = 0;
static constexpr uint32_t BOOT_TRAIN_WINDOW_MS = 1200;
static constexpr uint32_t BOOT_DEBOUNCE_MS = 60;
static constexpr uint32_t RUNTIME_BOOT_DEBOUNCE_MS = 30;
// Six-axis acquisition is sampled by software at 50 Hz. BLE always streams all
// six QMI8658 channels in batches of five so the GUI can plot and record the
// same raw data regardless of which channels the deployed model selects.
static constexpr uint16_t TRAIN_SAMPLE_RATE_HZ = 50;
static constexpr uint8_t TRAIN_BLE_BATCH_SIZE = 5;
static constexpr uint32_t TRAIN_SAMPLE_PERIOD_MS = 1000 / TRAIN_SAMPLE_RATE_HZ;
// Gesture capture is variable-duration. Five seconds is deliberately generous
// for air-written digits while still using only 6 kB of raw gesture SRAM.
static constexpr uint16_t MIN_GESTURE_SAMPLES = 10;
static constexpr uint16_t MAX_GESTURE_SAMPLES = 250;
static constexpr uint32_t MAX_FILE_BYTES = 550 * 1024;
static constexpr float MADGWICK_BETA = 0.10f;
static constexpr float GRAVITY_MPS2 = 9.80665f;

// -----------------------------------------------------------------------------
// BLE protocol
// -----------------------------------------------------------------------------
static const char *DEVICE_NAME = "NoodleAI";
static const char *SERVICE_UUID = "7f8b0001-5f5b-4f4a-a5d5-2e889aa10001";
static const char *IMU_UUID     = "7f8b0002-5f5b-4f4a-a5d5-2e889aa10001"; // notify: <IB3x30f (1..5 six-axis samples)
static const char *CONTROL_UUID = "7f8b0003-5f5b-4f4a-a5d5-2e889aa10001"; // write
static const char *MODEL_UUID   = "7f8b0004-5f5b-4f4a-a5d5-2e889aa10001"; // file bytes
static const char *STATUS_UUID  = "7f8b0005-5f5b-4f4a-a5d5-2e889aa10001"; // notify/read ASCII

enum ControlOpcode : uint8_t {
    OP_SET_TRAINING = 0x01,
    OP_SET_INFERENCE = 0x02,
    OP_ERASE_MODEL = 0x12,

    OP_DEPLOY_BEGIN = 0x20,
    OP_FILE_BEGIN = 0x21,   // [op][nameLen:u8][size:u32][crc:u32][name bytes]
    OP_FILE_END = 0x22,
    OP_DEPLOY_COMMIT = 0x23,
    OP_DEPLOY_ABORT = 0x24,
};

enum class DeviceMode : uint8_t { TRAINING, INFERENCE };

#pragma pack(push, 1)
struct ImuSample {
    float ax;
    float ay;
    float az;
    float gx;
    float gy;
    float gz;
};

// 128 bytes: t0_ms (4) + count (1) + pad (3) + 5*6*float32 (120).
// MTU is requested as 247, so one notification can carry a complete batch.
struct ImuPacket {
    uint32_t t0_ms;
    uint8_t count;
    uint8_t reserved[3];
    ImuSample samples[TRAIN_BLE_BATCH_SIZE];
};
#pragma pack(pop)

static_assert(sizeof(ImuPacket) == 128, "ImuPacket must remain exactly 128 bytes");

SensorQMI8658 qmi;
Adafruit_NeoPixel pixels(64, PIN_MATRIX, NEO_GRB + NEO_KHZ800);
NoodleAIModel model;

BLEServer *bleServer = nullptr;
BLECharacteristic *imuCharacteristic = nullptr;
BLECharacteristic *statusCharacteristic = nullptr;
bool bleConnected = false;

// BLE callbacks run in the Bluetooth stack task.  Keep them short and never
// call notify(), FFat model management, or mode-switch logic from inside a
// GATT callback.  Control packets are copied into this FreeRTOS queue and
// executed later from Arduino loop().
static constexpr size_t CONTROL_MESSAGE_MAX = 32;
struct ControlMessage {
    uint8_t len = 0;
    uint8_t data[CONTROL_MESSAGE_MAX]{};
};
QueueHandle_t controlQueue = nullptr;

DeviceMode mode = DeviceMode::TRAINING;
// Event-based gesture inference: raw samples are captured only between BOOT
// press/release, then resampled into the fixed model input length. Parameters
// remain in FFat; only raw gesture + normalized input + Noodle activations use SRAM.
float *gestureRaw = nullptr;
float *derivedRaw = nullptr;      // only allocated for quaternion/velocity representations
float *inferenceWindow = nullptr;
uint32_t lastSampleMs = 0;

bool gestureActive = false;
bool gestureTruncated = false;
uint16_t gestureSampleCount = 0;
uint32_t gestureStartMs = 0;

bool runtimeBootRawPressed = false;
bool runtimeBootStablePressed = false;
uint32_t runtimeBootLastChangeMs = 0;

ImuPacket trainingPacket{};
uint8_t trainingBatchIndex = 0;

struct DeployState {
    bool active = false;
    bool fileActive = false;
    char targetSlot = 0;
    char oldSlot = 0;
    char canonical[16]{};
    char targetName[NOODLE_MAX_FILENAME + 1]{};
    uint32_t expectedBytes = 0;
    uint32_t expectedCrc = 0;
    uint32_t receivedBytes = 0;
    uint32_t crc = 0xFFFFFFFFu;
    NDL_File file;
} deploy;

// -----------------------------------------------------------------------------
// Small 5x7 status glyphs on the onboard 8x8 matrix
// -----------------------------------------------------------------------------
static uint16_t xyToIndex(uint8_t x, uint8_t y) {
    return (x % 2 == 0) ? (x * 8 + y) : (x * 8 + (7 - y));
}

static void drawRows(const uint8_t rows[7], uint32_t color) {
    pixels.clear();
    for (uint8_t y = 0; y < 7; ++y) {
        for (uint8_t x = 0; x < 5; ++x) {
            if (rows[y] & (1u << (4 - x))) pixels.setPixelColor(xyToIndex(x + 1, y), color);
        }
    }
    pixels.show();
}

static void showGlyph(char c) {
    static const uint8_t glyphs[13][7] = {
        {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E},
        {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E},
        {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F},
        {0x1E,0x01,0x01,0x0E,0x01,0x01,0x1E},
        {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02},
        {0x1F,0x10,0x10,0x1E,0x01,0x01,0x1E},
        {0x0E,0x10,0x10,0x1E,0x11,0x11,0x0E},
        {0x1F,0x01,0x02,0x04,0x08,0x08,0x08},
        {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E},
        {0x0E,0x11,0x11,0x0F,0x01,0x01,0x0E},
        {0x1F,0x04,0x04,0x04,0x04,0x04,0x04}, // T
        {0x0E,0x11,0x01,0x02,0x04,0x00,0x04}, // ?
        {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}, // R
    };
    const uint32_t color = pixels.Color(18, 18, 18);
    if (c >= '0' && c <= '9') drawRows(glyphs[c - '0'], color);
    else if (c == 'T') drawRows(glyphs[10], color);
    else if (c == 'R') drawRows(glyphs[12], color);
    else drawRows(glyphs[11], color);
}

// GPIO0 is the ESP32-S3 BOOT strapping pin. Holding it while resetting can
// enter the ROM downloader, so NoodleAI samples it only AFTER the application
// firmware has started. Press BOOT during this short window to force TRAINING.
static bool startupTrainingRequested() {
    pinMode(PIN_BOOT, INPUT_PULLUP);
    Serial.printf("Press BOOT within %.1f s to force TRAINING mode...\n",
                  BOOT_TRAIN_WINDOW_MS / 1000.0f);
    showGlyph('?');

    const uint32_t started = millis();
    while (millis() - started < BOOT_TRAIN_WINDOW_MS) {
        if (digitalRead(PIN_BOOT) == LOW) {
            const uint32_t pressed = millis();
            while (digitalRead(PIN_BOOT) == LOW &&
                   millis() - pressed < BOOT_DEBOUNCE_MS) {
                delay(1);
            }
            if (millis() - pressed >= BOOT_DEBOUNCE_MS) {
                Serial.println("BOOT pressed: forcing TRAINING mode");
                showGlyph('T');
                return true;
            }
        }
        delay(5);
    }

    Serial.println("No BOOT request; using normal model-based startup");
    return false;
}

static void sendStatus(const String &s) {
    Serial.println(s);
    if (statusCharacteristic) statusCharacteristic->setValue(s.c_str());
    if (bleConnected && statusCharacteristic) statusCharacteristic->notify();
}

static void resetGestureState() {
    gestureActive = false;
    gestureTruncated = false;
    gestureSampleCount = 0;
    gestureStartMs = 0;
    trainingBatchIndex = 0;
    trainingPacket.count = 0;
}

static bool representationNeedsDerivedBuffer() {
    if (!model.loaded()) return false;
    const auto rep = model.representation();
    return rep == NoodleAIModel::REP_QUATERNION ||
           rep == NoodleAIModel::REP_VELOCITY ||
           rep == NoodleAIModel::REP_VELOCITY_QUATERNION;
}

static bool inferenceBuffersReady() {
    if (!gestureRaw || !inferenceWindow) return false;
    if (representationNeedsDerivedBuffer() && !derivedRaw) return false;
    return true;
}

static void rebuildInferenceWindow() {
    delete[] gestureRaw;
    delete[] derivedRaw;
    delete[] inferenceWindow;
    gestureRaw = nullptr;
    derivedRaw = nullptr;
    inferenceWindow = nullptr;
    resetGestureState();

    if (!model.loaded()) return;

    const uint16_t channels = model.channelCount();
    if (model.inputDim() == 0 || channels == 0 || channels > 7 ||
        model.windowLength() * channels != model.inputDim()) {
        sendStatus("ERR:INPUT_DIM");
        return;
    }

    // Raw acquisition always stays six-axis. NAI4 metadata chooses the feature
    // representation built at BOOT release.
    gestureRaw = new (std::nothrow) float[MAX_GESTURE_SAMPLES * 6u];
    inferenceWindow = new (std::nothrow) float[model.inputDim()];
    if (representationNeedsDerivedBuffer()) {
        derivedRaw = new (std::nothrow) float[MAX_GESTURE_SAMPLES * channels];
    }

    if (!inferenceBuffersReady()) {
        delete[] gestureRaw;
        delete[] derivedRaw;
        delete[] inferenceWindow;
        gestureRaw = nullptr;
        derivedRaw = nullptr;
        inferenceWindow = nullptr;
        sendStatus("ERR:RAM");
    }
}

static void setMode(DeviceMode newMode) {
    if (newMode == DeviceMode::INFERENCE && (!model.loaded() || !inferenceBuffersReady())) {
        mode = DeviceMode::TRAINING;
        resetGestureState();
        showGlyph('?');
        sendStatus("NO_MODEL");
        return;
    }

    mode = newMode;
    resetGestureState();
    if (mode == DeviceMode::TRAINING) {
        showGlyph('T');
        sendStatus("MODE:T");
        Serial.println("Gesture training: select a label in the GUI, hold BOOT, perform the gesture, then release BOOT.");
    } else {
        showGlyph('?');
        sendStatus("MODE:I");
        Serial.printf("Gesture inference ready: BOOT press=start, release=end; representation=%s, normalized=%u, input=%u, rate=%u Hz\n",
                      model.representationName(),
                      (unsigned)model.windowLength(),
                      (unsigned)model.inputDim(),
                      (unsigned)model.sampleRateHz());
    }
}

// zlib-compatible CRC32 (polynomial 0xEDB88320)
static uint32_t crc32Update(uint32_t crc, const uint8_t *data, size_t len) {
    while (len--) {
        crc ^= *data++;
        for (uint8_t k = 0; k < 8; ++k)
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return crc;
}

static void resetDeployState() {
    if (deploy.file) deploy.file.close();
    deploy = DeployState{};
}

static void abortDeploy(const char *status) {
    if (deploy.file) deploy.file.close();
    if (deploy.fileActive && deploy.targetName[0]) noodle_fs_remove(deploy.targetName);
    const char slot = deploy.targetSlot;
    resetDeployState();
    if (slot == 'A' || slot == 'B') model.eraseSlot(slot);
    sendStatus(status);
}

static void beginDeploy() {
    if (deploy.active) abortDeploy("ERR:RESET");
    const char oldSlot = model.loaded() ? model.activeSlot() : model.readActiveSlot();
    const char newSlot = (oldSlot == 'A') ? 'B' : 'A';
    model.eraseSlot(newSlot);

    deploy = DeployState{};
    deploy.active = true;
    deploy.targetSlot = newSlot;
    deploy.oldSlot = oldSlot;
    setMode(DeviceMode::TRAINING);
    sendStatus(String("DEPLOY:") + newSlot);
}

static void beginFile(const uint8_t *p, size_t n) {
    if (!deploy.active || deploy.fileActive || n < 10) { sendStatus("ERR:FILE_CMD"); return; }
    const uint8_t nameLen = p[1];
    if (nameLen == 0 || nameLen >= sizeof(deploy.canonical) || n != static_cast<size_t>(10 + nameLen)) {
        // packet length = op(1)+nameLen(1)+size(4)+crc(4)+name
        sendStatus("ERR:NAME");
        return;
    }

    uint32_t size = 0, crc = 0;
    memcpy(&size, p + 2, 4);
    memcpy(&crc, p + 6, 4);
    if (size == 0 || size > MAX_FILE_BYTES) { sendStatus("ERR:SIZE"); return; }

    memcpy(deploy.canonical, p + 10, nameLen);
    deploy.canonical[nameLen] = '\0';
    if (!NoodleAIModel::canonicalFilenameAllowed(deploy.canonical) ||
        !NoodleAIModel::slotFilename(deploy.targetSlot, deploy.canonical, deploy.targetName, sizeof(deploy.targetName))) {
        sendStatus("ERR:NAME");
        return;
    }

    noodle_fs_remove(deploy.targetName);
    deploy.file = noodle_fs_open_write(deploy.targetName);
    if (!deploy.file) { sendStatus("ERR:FILE"); return; }

    deploy.fileActive = true;
    deploy.expectedBytes = size;
    deploy.expectedCrc = crc;
    deploy.receivedBytes = 0;
    deploy.crc = 0xFFFFFFFFu;
    sendStatus(String("FILE_READY:") + deploy.canonical);
}

static void finishFile() {
    if (!deploy.active || !deploy.fileActive) { sendStatus("ERR:NO_FILE"); return; }
    deploy.file.flush();
    deploy.file.close();
    const uint32_t finalCrc = deploy.crc ^ 0xFFFFFFFFu;

    if (deploy.receivedBytes != deploy.expectedBytes) {
        noodle_fs_remove(deploy.targetName);
        deploy.fileActive = false;
        sendStatus("ERR:COUNT");
        return;
    }
    if (finalCrc != deploy.expectedCrc) {
        noodle_fs_remove(deploy.targetName);
        deploy.fileActive = false;
        sendStatus("ERR:CRC");
        return;
    }

    const String ok = String("FILE_OK:") + deploy.canonical;
    deploy.fileActive = false;
    deploy.expectedBytes = deploy.receivedBytes = 0;
    deploy.expectedCrc = deploy.crc = 0;
    deploy.canonical[0] = '\0';
    deploy.targetName[0] = '\0';
    sendStatus(ok);
}

static void commitDeploy() {
    if (!deploy.active || deploy.fileActive) { sendStatus("ERR:COMMIT"); return; }
    const char newSlot = deploy.targetSlot;
    const char oldSlot = deploy.oldSlot;

    model.clearRuntime();
    if (!model.loadSlot(newSlot)) {
        if (oldSlot == 'A' || oldSlot == 'B') model.loadSlot(oldSlot);
        rebuildInferenceWindow();
        model.eraseSlot(newSlot);
        resetDeployState();
        sendStatus("ERR:MODEL");
        return;
    }

    if (!model.writeActiveSlot(newSlot)) {
        model.clearRuntime();
        if (oldSlot == 'A' || oldSlot == 'B') model.loadSlot(oldSlot);
        rebuildInferenceWindow();
        resetDeployState();
        sendStatus("ERR:ACTIVE");
        return;
    }

    rebuildInferenceWindow();
    resetDeployState();
    Serial.printf("Active model slot: %c\n", model.activeSlot());
    Serial.printf("Noodle tensor arena: %u bytes\n", (unsigned)model.tensorArenaBytes());
    sendStatus("MODEL_OK");
    setMode(DeviceMode::INFERENCE);
}

class ServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer *) override {
        // Do not notify here: the central has connected, but it has not yet
        // subscribed to the STATUS CCCD.  Just prime the readable value.
        bleConnected = true;
        const char *s = model.loaded() ? "CONNECTED:M" : "CONNECTED:N";
        Serial.println(s);
        if (statusCharacteristic) statusCharacteristic->setValue(s);
    }
    void onDisconnect(BLEServer *server) override {
        bleConnected = false;
        if (deploy.active) abortDeploy("RX:ABORT");
        if (model.loaded() && inferenceBuffersReady()) setMode(DeviceMode::INFERENCE);
        else setMode(DeviceMode::TRAINING);
        delay(50);
        server->getAdvertising()->start();
    }
};

class ControlCallbacks : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *c) override {
        const std::string v = c->getValue();
        if (v.empty() || !controlQueue) return;
        if (v.size() > CONTROL_MESSAGE_MAX) {
            Serial.printf("BLE CTRL too long (%u bytes)\n", (unsigned)v.size());
            return;
        }

        ControlMessage msg{};
        msg.len = static_cast<uint8_t>(v.size());
        memcpy(msg.data, v.data(), v.size());

        // Never block the Bluetooth task.  Eight queued control commands are
        // ample because the PC intentionally spaces deployment commands.
        if (xQueueSend(controlQueue, &msg, 0) != pdTRUE) {
            Serial.println("BLE CTRL queue full");
            return;
        }
        Serial.printf("BLE CTRL queued 0x%02X (%u bytes)\n", msg.data[0], (unsigned)msg.len);
    }
};

static void executeControl(const uint8_t *p, size_t n) {
    if (!p || n == 0) return;
    Serial.printf("BLE CTRL exec 0x%02X (%u bytes)\n", p[0], (unsigned)n);

    switch (p[0]) {
    case OP_SET_TRAINING: setMode(DeviceMode::TRAINING); break;
    case OP_SET_INFERENCE: setMode(DeviceMode::INFERENCE); break;
    case OP_ERASE_MODEL:
        if (deploy.active) abortDeploy("RX:ABORT");
        model.clearRuntime();
        model.eraseAll();
        delete[] gestureRaw;
        delete[] derivedRaw;
        delete[] inferenceWindow;
        gestureRaw = nullptr;
        derivedRaw = nullptr;
        inferenceWindow = nullptr;
        resetGestureState();
        setMode(DeviceMode::TRAINING);
        sendStatus("MODEL_ERASED");
        break;
    case OP_DEPLOY_BEGIN: beginDeploy(); break;
    case OP_FILE_BEGIN: beginFile(p, n); break;
    case OP_FILE_END: finishFile(); break;
    case OP_DEPLOY_COMMIT: commitDeploy(); break;
    case OP_DEPLOY_ABORT: if (deploy.active) abortDeploy("RX:ABORT"); break;
    default: sendStatus("ERR:CMD"); break;
    }
}

static void processControlQueue() {
    if (!controlQueue) return;
    ControlMessage msg{};
    while (xQueueReceive(controlQueue, &msg, 0) == pdTRUE) {
        executeControl(msg.data, msg.len);
    }
}

class ModelCallbacks : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *c) override {
        if (!deploy.active || !deploy.fileActive) return;
        std::string v = c->getValue();
        if (v.empty()) return;
        const uint8_t *data = reinterpret_cast<const uint8_t *>(v.data());
        const size_t len = v.size();
        if (deploy.receivedBytes + len > deploy.expectedBytes) {
            abortDeploy("ERR:OVER");
            return;
        }
        if (deploy.file.write(data, len) != len) {
            abortDeploy("ERR:WRITE");
            return;
        }
        deploy.crc = crc32Update(deploy.crc, data, len);
        deploy.receivedBytes += len;
    }
};

static bool initImu() {
    Wire.begin(PIN_IMU_SDA, PIN_IMU_SCL);
    Wire.setClock(400000);
    delay(20);

    bool ok = qmi.begin(Wire, QMI8658_L_SLAVE_ADDRESS);
    if (!ok) ok = qmi.begin(Wire, QMI8658_H_SLAVE_ADDRESS);
    if (!ok) return false;

    if (!qmi.configAccelerometer(
            SensorQMI8658::ACC_RANGE_4G,
            SensorQMI8658::ACC_ODR_125Hz,
            SensorQMI8658::LPF_OFF)) return false;
    if (!qmi.configGyroscope(
            SensorQMI8658::GYR_RANGE_1024DPS,
            SensorQMI8658::GYR_ODR_112_1Hz,
            SensorQMI8658::LPF_OFF)) return false;

    if (!qmi.enableAccelerometer()) return false;
    return qmi.enableGyroscope();
}

static void initBle() {
    BLEDevice::init(DEVICE_NAME);
    BLEDevice::setMTU(247);
    bleServer = BLEDevice::createServer();
    bleServer->setCallbacks(new ServerCallbacks());

    BLEService *service = bleServer->createService(SERVICE_UUID);
    imuCharacteristic = service->createCharacteristic(IMU_UUID, BLECharacteristic::PROPERTY_NOTIFY);
    imuCharacteristic->addDescriptor(new BLE2902());

    // Control commands are deliberately Write Without Response.  This avoids
    // re-entrant ATT response/notification traffic when a command immediately
    // emits a STATUS notification (e.g. MODE:T).
    BLECharacteristic *control = service->createCharacteristic(
        CONTROL_UUID, BLECharacteristic::PROPERTY_WRITE_NR);
    control->setCallbacks(new ControlCallbacks());

    BLECharacteristic *modelData = service->createCharacteristic(
        MODEL_UUID, BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR);
    modelData->setCallbacks(new ModelCallbacks());

    statusCharacteristic = service->createCharacteristic(
        STATUS_UUID, BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
    statusCharacteristic->addDescriptor(new BLE2902());
    statusCharacteristic->setValue("BOOT");

    service->start();
    BLEAdvertising *advertising = BLEDevice::getAdvertising();
    advertising->addServiceUUID(SERVICE_UUID);
    advertising->setScanResponse(true);
    advertising->start();
}

static void flushTrainingPacket() {
    if (trainingBatchIndex == 0) return;
    if (bleConnected && imuCharacteristic) {
        trainingPacket.count = trainingBatchIndex;
        imuCharacteristic->setValue(
            reinterpret_cast<uint8_t *>(&trainingPacket), sizeof(trainingPacket));
        imuCharacteristic->notify();
    }
    trainingBatchIndex = 0;
    trainingPacket.count = 0;
}

static void streamImuSample(float ax, float ay, float az,
                            float gx, float gy, float gz, uint32_t now) {
    // Keep the live BLE path identical to the previously working six-axis
    // firmware. Pause only while FFat deployment owns the BLE link.
    if (deploy.active || !bleConnected || !imuCharacteristic) return;

    if (trainingBatchIndex == 0) trainingPacket.t0_ms = now;
    ImuSample &dst = trainingPacket.samples[trainingBatchIndex];
    dst.ax = ax; dst.ay = ay; dst.az = az;
    dst.gx = gx; dst.gy = gy; dst.gz = gz;
    ++trainingBatchIndex;

    if (trainingBatchIndex >= TRAIN_BLE_BATCH_SIZE) flushTrainingPacket();
}

static uint8_t rawAxisForRepresentationChannel(uint8_t modelChannel) {
    switch (model.representation()) {
    case NoodleAIModel::REP_ACCEL:
        return modelChannel;                       // ax ay az
    case NoodleAIModel::REP_GYRO:
        return static_cast<uint8_t>(modelChannel + 3u); // gx gy gz
    case NoodleAIModel::REP_ACCEL_GYRO:
    default:
        return modelChannel;                       // all six
    }
}

static void resampleGeneric(const float *src, uint16_t rawSamples, uint8_t channels,
                            float *out, uint16_t normalizedSamples) {
    if (!src || !out || rawSamples == 0 || normalizedSamples == 0 || channels == 0) return;
    if (rawSamples == 1 || normalizedSamples == 1) {
        for (uint16_t i = 0; i < normalizedSamples; ++i)
            for (uint8_t c = 0; c < channels; ++c)
                out[i * channels + c] = src[c];
        return;
    }

    const float scale = static_cast<float>(rawSamples - 1) /
                        static_cast<float>(normalizedSamples - 1);
    for (uint16_t i = 0; i < normalizedSamples; ++i) {
        const float pos = i * scale;
        const uint16_t i0 = static_cast<uint16_t>(pos);
        const uint16_t i1 = (i0 + 1 < rawSamples) ? (i0 + 1) : i0;
        const float alpha = pos - static_cast<float>(i0);
        for (uint8_t c = 0; c < channels; ++c) {
            const float a = src[i0 * channels + c];
            const float b = src[i1 * channels + c];
            out[i * channels + c] = a + alpha * (b - a);
        }
    }
}

static void resampleRawRepresentation(const float *raw6, uint16_t rawSamples,
                                      float *out, uint16_t normalizedSamples) {
    const uint8_t channels = static_cast<uint8_t>(model.channelCount());
    if (!raw6 || !out || channels == 0) return;

    if (rawSamples == 1 || normalizedSamples == 1) {
        for (uint16_t i = 0; i < normalizedSamples; ++i)
            for (uint8_t c = 0; c < channels; ++c)
                out[i * channels + c] = raw6[rawAxisForRepresentationChannel(c)];
        return;
    }

    const float scale = static_cast<float>(rawSamples - 1) /
                        static_cast<float>(normalizedSamples - 1);
    for (uint16_t i = 0; i < normalizedSamples; ++i) {
        const float pos = i * scale;
        const uint16_t i0 = static_cast<uint16_t>(pos);
        const uint16_t i1 = (i0 + 1 < rawSamples) ? (i0 + 1) : i0;
        const float alpha = pos - static_cast<float>(i0);
        for (uint8_t c = 0; c < channels; ++c) {
            const uint8_t axis = rawAxisForRepresentationChannel(c);
            const float a = raw6[i0 * 6u + axis];
            const float b = raw6[i1 * 6u + axis];
            out[i * channels + c] = a + alpha * (b - a);
        }
    }
}

static void centerNormalizedGesture(float *data, uint16_t samples,
                                    uint8_t channels, float meanOut[7]) {
    for (uint8_t c = 0; c < 7; ++c) meanOut[c] = 0.0f;
    if (!data || samples == 0 || channels == 0) return;

    for (uint16_t i = 0; i < samples; ++i)
        for (uint8_t c = 0; c < channels; ++c)
            meanOut[c] += data[i * channels + c];

    const float invN = 1.0f / static_cast<float>(samples);
    for (uint8_t c = 0; c < channels; ++c) meanOut[c] *= invN;

    for (uint16_t i = 0; i < samples; ++i)
        for (uint8_t c = 0; c < channels; ++c)
            data[i * channels + c] -= meanOut[c];
}

struct Quaternion {
    float w, x, y, z;
};

static Quaternion qNormalize(Quaternion q) {
    const float n2 = q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z;
    if (!isfinite(n2) || n2 < 1e-20f) return {1.0f, 0.0f, 0.0f, 0.0f};
    const float inv = 1.0f / sqrtf(n2);
    q.w *= inv; q.x *= inv; q.y *= inv; q.z *= inv;
    return q;
}

static Quaternion qConj(const Quaternion &q) {
    return {q.w, -q.x, -q.y, -q.z};
}

static Quaternion qMul(const Quaternion &a, const Quaternion &b) {
    return {
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
    };
}

static float qDot(const Quaternion &a, const Quaternion &b) {
    return a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z;
}

static void qRotate(const Quaternion &q, float vx, float vy, float vz,
                    float &ox, float &oy, float &oz) {
    const Quaternion p{0.0f, vx, vy, vz};
    const Quaternion r = qMul(qMul(q, p), qConj(q));
    ox = r.x; oy = r.y; oz = r.z;
}

static Quaternion quatAlignAccelToWorldZ(float ax, float ay, float az) {
    const float n2 = ax*ax + ay*ay + az*az;
    if (!isfinite(n2) || n2 < 1e-12f) return {1.0f, 0.0f, 0.0f, 0.0f};
    const float inv = 1.0f / sqrtf(n2);
    const float ux = ax * inv, uy = ay * inv, uz = az * inv;
    const float d = uz; // dot([ux,uy,uz], [0,0,1])

    if (d < -0.999999f) {
        // Stable 180-degree fallback; axis is perpendicular to measured gravity.
        float bx = (fabsf(ux) < 0.9f) ? 1.0f : 0.0f;
        float by = (fabsf(ux) < 0.9f) ? 0.0f : 1.0f;
        float cx = -uz * by;
        float cy = uz * bx;
        float cz = ux * by - uy * bx;
        const float cn = sqrtf(cx*cx + cy*cy + cz*cz);
        if (cn < 1e-8f) return {1.0f, 0.0f, 0.0f, 0.0f};
        return {0.0f, cx/cn, cy/cn, cz/cn};
    }

    // cross(u, z) = [uy, -ux, 0]
    return qNormalize({1.0f + d, uy, -ux, 0.0f});
}

static Quaternion madgwickUpdateImu(Quaternion q,
                                    float ax, float ay, float az,
                                    float gxDps, float gyDps, float gzDps,
                                    float dt) {
    const float deg2rad = 0.01745329251994329577f;
    const float gx = gxDps * deg2rad;
    const float gy = gyDps * deg2rad;
    const float gz = gzDps * deg2rad;

    float q1=q.w, q2=q.x, q3=q.y, q4=q.z;
    float qDot1 = 0.5f * (-q2*gx - q3*gy - q4*gz);
    float qDot2 = 0.5f * ( q1*gx + q3*gz - q4*gy);
    float qDot3 = 0.5f * ( q1*gy - q2*gz + q4*gx);
    float qDot4 = 0.5f * ( q1*gz + q2*gy - q3*gx);

    const float an2 = ax*ax + ay*ay + az*az;
    if (isfinite(an2) && an2 > 1e-12f) {
        const float invA = 1.0f / sqrtf(an2);
        ax *= invA; ay *= invA; az *= invA;

        const float _2q1=2.0f*q1, _2q2=2.0f*q2, _2q3=2.0f*q3, _2q4=2.0f*q4;
        const float _4q1=4.0f*q1, _4q2=4.0f*q2, _4q3=4.0f*q3;
        const float _8q2=8.0f*q2, _8q3=8.0f*q3;
        const float q1q1=q1*q1, q2q2=q2*q2, q3q3=q3*q3, q4q4=q4*q4;

        float s1 = _4q1*q3q3 + _2q3*ax + _4q1*q2q2 - _2q2*ay;
        float s2 = _4q2*q4q4 - _2q4*ax + 4.0f*q1q1*q2 - _2q1*ay - _4q2
                 + _8q2*q2q2 + _8q2*q3q3 + _4q2*az;
        float s3 = 4.0f*q1q1*q3 + _2q1*ax + _4q3*q4q4 - _2q4*ay - _4q3
                 + _8q3*q2q2 + _8q3*q3q3 + _4q3*az;
        float s4 = 4.0f*q2q2*q4 - _2q2*ax + 4.0f*q3q3*q4 - _2q3*ay;
        const float sn2 = s1*s1 + s2*s2 + s3*s3 + s4*s4;
        if (isfinite(sn2) && sn2 > 1e-20f) {
            const float invS = 1.0f / sqrtf(sn2);
            s1*=invS; s2*=invS; s3*=invS; s4*=invS;
            qDot1 -= MADGWICK_BETA*s1;
            qDot2 -= MADGWICK_BETA*s2;
            qDot3 -= MADGWICK_BETA*s3;
            qDot4 -= MADGWICK_BETA*s4;
        }
    }

    q.w += qDot1*dt;
    q.x += qDot2*dt;
    q.y += qDot3*dt;
    q.z += qDot4*dt;
    return qNormalize(q);
}

static void storeDerivedSample(float *out, uint16_t k,
                               const Quaternion &qRel,
                               float vx, float vy, float vz) {
    switch (model.representation()) {
    case NoodleAIModel::REP_QUATERNION: {
        const uint16_t b = k * 4u;
        out[b+0]=qRel.w; out[b+1]=qRel.x; out[b+2]=qRel.y; out[b+3]=qRel.z;
        break;
    }
    case NoodleAIModel::REP_VELOCITY: {
        const uint16_t b = k * 3u;
        out[b+0]=vx; out[b+1]=vy; out[b+2]=vz;
        break;
    }
    case NoodleAIModel::REP_VELOCITY_QUATERNION: {
        const uint16_t b = k * 7u;
        out[b+0]=vx; out[b+1]=vy; out[b+2]=vz;
        out[b+3]=qRel.w; out[b+4]=qRel.x; out[b+5]=qRel.y; out[b+6]=qRel.z;
        break;
    }
    default:
        break;
    }
}

static bool buildDerivedRawRepresentation(const float *raw6, uint16_t rawSamples,
                                          float *out) {
    if (!raw6 || !out || rawSamples < 2 || model.sampleRateHz() == 0) return false;
    const auto rep = model.representation();
    if (rep != NoodleAIModel::REP_QUATERNION &&
        rep != NoodleAIModel::REP_VELOCITY &&
        rep != NoodleAIModel::REP_VELOCITY_QUATERNION) return false;

    const float dt = 1.0f / static_cast<float>(model.sampleRateHz());
    Quaternion qAbs = quatAlignAccelToWorldZ(raw6[0], raw6[1], raw6[2]);
    const Quaternion qRef = qAbs;
    Quaternion qRelPrev{1.0f,0.0f,0.0f,0.0f};

    float awx=0.0f, awy=0.0f, awz=0.0f;
    qRotate(qAbs, raw6[0], raw6[1], raw6[2], awx, awy, awz);
    float axPrev = awx * GRAVITY_MPS2;
    float ayPrev = awy * GRAVITY_MPS2;
    float azPrev = (awz - 1.0f) * GRAVITY_MPS2;

    float vx=0.0f, vy=0.0f, vz=0.0f;
    storeDerivedSample(out, 0, qRelPrev, vx, vy, vz);

    for (uint16_t k=1; k<rawSamples; ++k) {
        const uint16_t b = k * 6u;
        qAbs = madgwickUpdateImu(qAbs,
                                 raw6[b+0], raw6[b+1], raw6[b+2],
                                 raw6[b+3], raw6[b+4], raw6[b+5], dt);
        Quaternion qRel = qNormalize(qMul(qConj(qRef), qAbs));
        if (qDot(qRel, qRelPrev) < 0.0f) {
            qRel.w=-qRel.w; qRel.x=-qRel.x; qRel.y=-qRel.y; qRel.z=-qRel.z;
        }
        qRelPrev = qRel;

        qRotate(qAbs, raw6[b+0], raw6[b+1], raw6[b+2], awx, awy, awz);
        const float axNow = awx * GRAVITY_MPS2;
        const float ayNow = awy * GRAVITY_MPS2;
        const float azNow = (awz - 1.0f) * GRAVITY_MPS2;
        vx += 0.5f * (axPrev + axNow) * dt;
        vy += 0.5f * (ayPrev + ayNow) * dt;
        vz += 0.5f * (azPrev + azNow) * dt;
        axPrev=axNow; ayPrev=ayNow; azPrev=azNow;

        storeDerivedSample(out, k, qRel, vx, vy, vz);
    }

    // Button-delimited gesture assumption: approximately at rest at both ends.
    // Remove the linear residual so v(0)=v(T)=0. Quaternion channels are untouched.
    if (rep == NoodleAIModel::REP_VELOCITY || rep == NoodleAIModel::REP_VELOCITY_QUATERNION) {
        const uint8_t channels = static_cast<uint8_t>(model.channelCount());
        const uint16_t last = (rawSamples - 1u) * channels;
        const float endVx=out[last+0], endVy=out[last+1], endVz=out[last+2];
        for (uint16_t k=0; k<rawSamples; ++k) {
            const float a = static_cast<float>(k) / static_cast<float>(rawSamples - 1u);
            const uint16_t b = k * channels;
            out[b+0] -= a*endVx;
            out[b+1] -= a*endVy;
            out[b+2] -= a*endVz;
        }
        out[0]=out[1]=out[2]=0.0f;
        out[last+0]=out[last+1]=out[last+2]=0.0f;
    }
    return true;
}

static void normalizeQuaternionChannels(float *data, uint16_t samples,
                                        uint8_t channels, uint8_t offset) {
    if (!data || channels < offset + 4u) return;
    Quaternion prev{1.0f,0.0f,0.0f,0.0f};
    for (uint16_t i=0; i<samples; ++i) {
        const uint16_t b = i*channels + offset;
        Quaternion q{data[b+0],data[b+1],data[b+2],data[b+3]};
        q=qNormalize(q);
        if (i>0 && qDot(q,prev)<0.0f) {
            q.w=-q.w; q.x=-q.x; q.y=-q.y; q.z=-q.z;
        }
        data[b+0]=q.w; data[b+1]=q.x; data[b+2]=q.y; data[b+3]=q.z;
        prev=q;
    }
}

static bool buildInferenceRepresentation(uint16_t rawSamples) {
    const uint16_t normalized = model.windowLength();
    const uint8_t channels = static_cast<uint8_t>(model.channelCount());
    const auto rep = model.representation();

    if (rep == NoodleAIModel::REP_ACCEL || rep == NoodleAIModel::REP_GYRO ||
        rep == NoodleAIModel::REP_ACCEL_GYRO) {
        resampleRawRepresentation(gestureRaw, rawSamples, inferenceWindow, normalized);
        float gestureMean[7];
        centerNormalizedGesture(inferenceWindow, normalized, channels, gestureMean);
        Serial.printf("REP raw=%s centered; normalized=%u channels=%u input=%u\n",
                      model.representationName(), (unsigned)normalized,
                      (unsigned)channels, (unsigned)model.inputDim());
        return true;
    }

    if (!derivedRaw || !buildDerivedRawRepresentation(gestureRaw, rawSamples, derivedRaw)) return false;
    resampleGeneric(derivedRaw, rawSamples, channels, inferenceWindow, normalized);
    if (rep == NoodleAIModel::REP_QUATERNION) {
        normalizeQuaternionChannels(inferenceWindow, normalized, channels, 0);
    } else if (rep == NoodleAIModel::REP_VELOCITY_QUATERNION) {
        normalizeQuaternionChannels(inferenceWindow, normalized, channels, 3);
    }
    Serial.printf("REP derived=%s Madgwick(beta=%.3f) -> normalized=%u channels=%u input=%u\n",
                  model.representationName(), MADGWICK_BETA, (unsigned)normalized,
                  (unsigned)channels, (unsigned)model.inputDim());
    return true;
}

static void runGestureInference(uint16_t rawSamples, uint32_t durationMs) {
    if (!model.loaded() || !inferenceBuffersReady()) return;
    if (rawSamples < MIN_GESTURE_SAMPLES) {
        Serial.printf("GESTURE too short: %u samples (%.3f s); minimum=%u\n",
                      (unsigned)rawSamples,
                      durationMs / 1000.0f,
                      (unsigned)MIN_GESTURE_SAMPLES);
        sendStatus(String("GESTURE:SHORT:") + rawSamples);
        showGlyph('?');
        return;
    }

    if (!buildInferenceRepresentation(rawSamples)) {
        Serial.println("Representation preprocessing failed");
        sendStatus("ERR:REP");
        showGlyph('?');
        return;
    }

    uint16_t cls = 0;
    float confidence = 0.0f;
    const uint32_t inferStartUs = micros();
    const bool ok = model.predict(inferenceWindow, cls, confidence);
    const uint32_t inferUs = micros() - inferStartUs;

    if (!ok) {
        Serial.println("PRED failed");
        sendStatus("ERR:PRED");
        showGlyph('?');
        return;
    }

    const String &name = model.label(cls);
    Serial.printf("PRED class=%u label=%s confidence=%.4f inference=%.3f ms raw=%u duration=%lu ms rep=%s\n",
                  (unsigned)cls, name.c_str(), confidence,
                  inferUs / 1000.0f, (unsigned)rawSamples, (unsigned long)durationMs,
                  model.representationName());

    char glyph = '?';
    if (name.length() == 1 && name[0] >= '0' && name[0] <= '9') glyph = name[0];
    else if (cls < 10) glyph = static_cast<char>('0' + cls);
    showGlyph(glyph);

    if (bleConnected) {
        char msg[24];
        snprintf(msg, sizeof(msg), "P:%u:%.3f", (unsigned)cls, confidence);
        sendStatus(msg);
    }
}

static void beginGesture(uint32_t now) {
    if (deploy.active || gestureActive) return;
    if (mode == DeviceMode::TRAINING && (!bleConnected || !imuCharacteristic)) {
        Serial.println("GESTURE ignored: TRAINING mode requires BLE connection");
        return;
    }
    if (mode == DeviceMode::INFERENCE && (!model.loaded() || !inferenceBuffersReady())) {
        sendStatus("NO_MODEL");
        return;
    }

    gestureActive = true;
    gestureTruncated = false;
    gestureSampleCount = 0;
    gestureStartMs = now;

    // Preserve the previously working live-stream behavior: finish any partial
    // live packet before GESTURE:START instead of silently dropping it.
    flushTrainingPacket();
    showGlyph('R');

    Serial.printf("GESTURE START mode=%s t=%lu ms\n",
                  mode == DeviceMode::TRAINING ? "TRAINING" : "INFERENCE",
                  (unsigned long)now);
    sendStatus("GESTURE:START");
}

static void endGesture(uint32_t now) {
    if (!gestureActive) return;
    gestureActive = false;
    const uint32_t durationMs = now - gestureStartMs;

    if (gestureSampleCount < MIN_GESTURE_SAMPLES) {
        // Preserve the final partial packet for the live plot. GESTURE:SHORT
        // prevents the GUI from saving it as a training example.
        flushTrainingPacket();
        Serial.printf("GESTURE too short: raw=%u duration=%lu ms; minimum=%u samples\n",
                      (unsigned)gestureSampleCount, (unsigned long)durationMs,
                      (unsigned)MIN_GESTURE_SAMPLES);
        sendStatus(String("GESTURE:SHORT:") + gestureSampleCount);
        showGlyph(mode == DeviceMode::TRAINING ? 'T' : '?');
        return;
    }

    flushTrainingPacket();

    Serial.printf("GESTURE END mode=%s raw=%u duration=%lu ms%s\n",
                  mode == DeviceMode::TRAINING ? "TRAINING" : "INFERENCE",
                  (unsigned)gestureSampleCount,
                  (unsigned long)durationMs,
                  gestureTruncated ? " TRUNCATED" : "");

    // The count lets the GUI wait for the final partial IMU notification before
    // resampling, even if STATUS and IMU notifications are delivered out of order.
    sendStatus(String("GESTURE:END:") + gestureSampleCount + ":" + durationMs);

    if (mode == DeviceMode::TRAINING) {
        showGlyph('T');
    } else {
        runGestureInference(gestureSampleCount, durationMs);
    }
}

static void updateRuntimeBootButton(uint32_t now) {
    const bool rawPressed = (digitalRead(PIN_BOOT) == LOW);
    if (rawPressed != runtimeBootRawPressed) {
        runtimeBootRawPressed = rawPressed;
        runtimeBootLastChangeMs = now;
    }

    if (rawPressed != runtimeBootStablePressed &&
        now - runtimeBootLastChangeMs >= RUNTIME_BOOT_DEBOUNCE_MS) {
        runtimeBootStablePressed = rawPressed;
        if (runtimeBootStablePressed) beginGesture(now);
        else endGesture(now);
    }
}

static void processImuSample(float ax, float ay, float az,
                             float gx, float gy, float gz, uint32_t now) {
    // Stream continuously in both TRAINING and INFERENCE modes.
    // This is intentionally kept separate from gesture capture/model selection.
    streamImuSample(ax, ay, az, gx, gy, gz, now);

    if (!gestureActive || deploy.active) return;
    if (gestureSampleCount >= MAX_GESTURE_SAMPLES) {
        if (!gestureTruncated) {
            gestureTruncated = true;
            Serial.printf("GESTURE reached maximum %u samples (%.1f s); extra samples ignored until BOOT release\n",
                          (unsigned)MAX_GESTURE_SAMPLES,
                          MAX_GESTURE_SAMPLES / static_cast<float>(TRAIN_SAMPLE_RATE_HZ));
        }
        return;
    }

    if (mode == DeviceMode::TRAINING) {
        // The PC records from the same continuous six-axis BLE stream.
        ++gestureSampleCount;
        return;
    }

    if (!model.loaded() || !inferenceBuffersReady()) return;
    const uint16_t base = gestureSampleCount * 6u;
    gestureRaw[base + 0] = ax;
    gestureRaw[base + 1] = ay;
    gestureRaw[base + 2] = az;
    gestureRaw[base + 3] = gx;
    gestureRaw[base + 4] = gy;
    gestureRaw[base + 5] = gz;
    ++gestureSampleCount;
}

void setup() {
    Serial.begin(115200);
    delay(400);
    Serial.println("\nNoodleAI ESP32-S3-Matrix — NAI4 motion representations + restored live BLE");

    pixels.begin();
    pixels.setBrightness(24);
    pixels.clear();
    pixels.show();

    // Physical startup mode selection. Do not hold BOOT during reset/power-on;
    // press it after the NoodleAI application starts and while '?' is displayed.
    const bool forceTraining = startupTrainingRequested();

    if (!noodle_fs_init()) {
        Serial.println("Noodle FFat mount failed; formatting partition...");
        if (!FFat.format() || !noodle_fs_init()) {
            Serial.println("FFat mount failed");
            showGlyph('?');
            while (true) delay(1000);
        }
    }
    Serial.printf("FFat total=%u used=%u bytes\n",
                  (unsigned)FFat.totalBytes(), (unsigned)FFat.usedBytes());

    if (!initImu()) {
        Serial.println("QMI8658 init failed");
        showGlyph('?');
        while (true) delay(1000);
    }
    Serial.println("QMI8658 accelerometer + gyroscope OK");
    Serial.printf("Gesture acquisition: %u Hz, BOOT press=start, release=end, raw range=%u..%u samples\n",
                  (unsigned)TRAIN_SAMPLE_RATE_HZ,
                  (unsigned)MIN_GESTURE_SAMPLES,
                  (unsigned)MAX_GESTURE_SAMPLES);
    Serial.println("NAI4 reps: accel | gyro | accel+gyro | quaternion | velocity | velocity+quaternion");
    Serial.println("Derived reps: 6-axis Madgwick -> relative quaternion / gravity-compensated velocity -> temporal resampling -> StandardScaler");
    Serial.printf("Training BLE: up to %u samples/notification (count-aware final packet)\n",
                  (unsigned)TRAIN_BLE_BATCH_SIZE);

    const bool modelOk = model.loadActive();
    rebuildInferenceWindow();
    if (modelOk) {
        Serial.printf("Loaded model slot %c: representation=%s normalized=%u input=%u layers=%u classes=%u rate=%u Hz\n",
                      model.activeSlot(), model.representationName(), model.windowLength(), model.inputDim(),
                      model.layerCount(), model.classCount(), model.sampleRateHz());
        Serial.printf("Noodle tensor arena: %u bytes\n", (unsigned)model.tensorArenaBytes());
    }

    controlQueue = xQueueCreate(8, sizeof(ControlMessage));
    if (!controlQueue) {
        Serial.println("Failed to create BLE control queue");
        showGlyph('?');
        while (true) delay(1000);
    }

    initBle();
    if (forceTraining) setMode(DeviceMode::TRAINING);
    else setMode((modelOk && inferenceBuffersReady()) ? DeviceMode::INFERENCE : DeviceMode::TRAINING);

    // If BOOT is still held from the startup selection, require release + a new
    // press before starting the first gesture.
    runtimeBootRawPressed = (digitalRead(PIN_BOOT) == LOW);
    runtimeBootStablePressed = runtimeBootRawPressed;
    runtimeBootLastChangeMs = millis();
    lastSampleMs = millis();
}

void loop() {
    processControlQueue();

    const uint32_t now = millis();
    updateRuntimeBootButton(now);

    uint32_t periodMs = TRAIN_SAMPLE_PERIOD_MS;
    if (mode == DeviceMode::INFERENCE && model.loaded()) {
        const uint32_t candidate = 1000u / model.sampleRateHz();
        periodMs = candidate > 0 ? candidate : 1;
    }

    if (now - lastSampleMs >= periodMs) {
        lastSampleMs += periodMs;
        float ax = 0.0f, ay = 0.0f, az = 0.0f;
        float gx = 0.0f, gy = 0.0f, gz = 0.0f;
        const bool accOk = qmi.getAccelerometer(ax, ay, az);
        const bool gyrOk = qmi.getGyroscope(gx, gy, gz);
        if (accOk && gyrOk) processImuSample(ax, ay, az, gx, gy, gz, now);
    }
    delay(1);
}
