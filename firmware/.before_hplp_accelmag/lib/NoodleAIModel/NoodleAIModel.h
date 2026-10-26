#pragma once

#include <Arduino.h>
#include <noodle.h>
#if defined(NOODLE_USE_INT8)
#include <noodle_int8.h>
#endif

/**
 * NoodleAI runtime adapter.
 *
 * Model parameter files remain in FFat. At build time SRAM_BACKEND selects:
 *   0: Noodle FCNFile streams FP32 parameters from FFat each inference;
 *   1: Noodle FCNMem loads parameters into SRAM once per model activation.
 * Both backends reuse the same two NoodleTensor activation arenas.
 *
 * NAI4 extends the model metadata from raw sensor selection to an input
 * representation. The firmware constructs that representation before predict().
 */
class NoodleAIModel {
public:
    static constexpr uint16_t MAX_LAYERS = 8;
    static constexpr uint16_t MAX_CLASSES = 10;
    static constexpr uint16_t MAX_DIM = 1536;
    static constexpr uint8_t MAX_LABEL_BYTES = 31;
#if defined(NOODLE_USE_INT8)
#endif

    enum InputRepresentation : uint16_t {
        REP_ACCEL = 1,
        REP_GYRO = 2,
        REP_ACCEL_GYRO = 3,
        REP_QUATERNION = 4,
        REP_VELOCITY = 5,
        REP_VELOCITY_QUATERNION = 6,

        // NAI3 source-compatibility aliases.
        SENSOR_ACCEL = REP_ACCEL,
        SENSOR_GYRO = REP_GYRO,
        SENSOR_ACCEL_GYRO = REP_ACCEL_GYRO,
    };
    using SensorMode = InputRepresentation;

    NoodleAIModel();
    ~NoodleAIModel();

    bool loadActive();
    bool loadSlot(char slot);
    void clearRuntime();

    bool writeActiveSlot(char slot);
    char readActiveSlot() const;
    char activeSlot() const { return _slot; }
    char inactiveSlot() const { return (_slot == 'A') ? 'B' : 'A'; }

    bool eraseSlot(char slot) const;
    bool eraseAll() const;

    static bool slotFilename(char slot, const char *canonical, char *out, size_t outCap);
    static bool canonicalFilenameAllowed(const char *name);

    bool loaded() const { return _loaded; }
    uint16_t inputDim() const { return _inputDim; }
    uint16_t windowLength() const { return _windowLength; }
    uint16_t sampleRateHz() const { return _sampleRateHz; }
    uint16_t classCount() const { return _nClasses; }
    uint16_t layerCount() const { return _nLayers; }

    InputRepresentation representation() const { return _representation; }
    uint16_t channelCount() const;
    const char *representationName() const;

    // Compatibility with NAI3-era application code.
    SensorMode sensorMode() const { return _representation; }
    const char *sensorModeName() const { return representationName(); }

    const String &label(uint16_t index) const;

    // x is already converted to the selected NAI4 representation, temporally
    // normalized, and laid out time-major. StandardScaler is applied here.
    bool predict(const float *x, uint16_t &classIndex, float &confidence);

    size_t tensorArenaBytes() const;
#if defined(NAI_BENCH_FCN) && NAI_BENCH_FCN
    // FCN-only time, excludes scaler, output decoding and serial printing.
    uint32_t lastFcnUs() const { return _lastFcnUs; }
#endif

private:
#pragma pack(push, 1)
    struct ConfigHeader {
        char magic[4];
        uint16_t version;
        uint16_t flags;
        uint16_t windowLength;
        uint16_t sampleRateHz;
        uint16_t nLayers;
        uint16_t nClasses;
        uint16_t inputDim;
        uint16_t reserved; // NAI4: InputRepresentation; NAI3: raw SensorMode
    };
#pragma pack(pop)

    bool readExact(NDL_File &f, void *dst, size_t n) const;
    bool readConfig(char slot);
    bool validateParameterFiles(char slot);
    bool loadScaler(char slot);
    void configureLayers(char slot);
#if defined(NOODLE_USE_INT8)
    bool readQuantization(NDL_File &file);
#endif
#if defined(SRAM_BACKEND) && SRAM_BACKEND
    bool loadRamParameters();
    void clearRamParameters();
#endif
    void initTensorsIfNeeded();
    static bool fileHasSize(const char *name, size_t expected);
    static uint16_t representationChannelCount(InputRepresentation rep);

#if defined(NAI_BENCH_FCN) && NAI_BENCH_FCN
    uint32_t _lastFcnUs = 0;
#endif
    bool _loaded = false;
    bool _binaryClassifier = false;
    bool _tensorsInitialized = false;
    char _slot = 0;

    InputRepresentation _representation = REP_ACCEL;
    uint16_t _windowLength = 0;
    uint16_t _sampleRateHz = 0;
    uint16_t _nLayers = 0;
    uint16_t _nClasses = 0;
    uint16_t _inputDim = 0;

    uint16_t _dims[MAX_LAYERS + 1]{};
    float *_mean = nullptr;
    float *_scale = nullptr;
    FCNFile _layers[MAX_LAYERS]{};  // INT8: wNN + pNN; FP32: wNN + bNN.
#if defined(NOODLE_USE_INT8)
    float _inputScale = 1.0f;
    int32_t _inputZeroPoint = 0;
    float _layerOutputScales[MAX_LAYERS]{};
    int32_t _layerOutputZeroPoints[MAX_LAYERS]{};
#endif
#if defined(SRAM_BACKEND) && SRAM_BACKEND && !defined(NOODLE_USE_INT8)
    // FP32 parameter mirrors, loaded once from active FFat slot.
    // Output-major [O][I] layout is *already* Noodle FCNMem format.
    FCNMem _ramLayers[MAX_LAYERS]{};
    float *_ramWeights[MAX_LAYERS]{};
    float *_ramBiases[MAX_LAYERS]{};
    size_t _ramParameterBytes = 0;
#endif
    String _labels[MAX_CLASSES];

    char _weightNames[MAX_LAYERS][NOODLE_MAX_FILENAME + 1]{};
    char _biasNames[MAX_LAYERS][NOODLE_MAX_FILENAME + 1]{};  // INT8: pNN metadata

    NoodleTensor _actA{};
    NoodleTensor _actB{};

};
