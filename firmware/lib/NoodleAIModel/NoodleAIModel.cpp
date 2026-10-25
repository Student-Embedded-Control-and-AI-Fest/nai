#include "NoodleAIModel.h"

#include <math.h>
#include <new>
#include <string.h>


namespace {
constexpr uint16_t MODEL_VERSION = 4;
constexpr uint16_t FLAG_BINARY_CLASSIFIER = 0x0001;
constexpr uint16_t FLAG_INT8 = 0x0002;
constexpr const char *ACTIVE_FILE = "nai_active";
constexpr const char *CFG_FILE = "cfg.bin";
constexpr const char *MEAN_FILE = "mean.bin";
constexpr const char *SCALE_FILE = "scale.bin";
}


NoodleAIModel::NoodleAIModel() = default;
NoodleAIModel::~NoodleAIModel() { clearRuntime(); }

bool NoodleAIModel::readExact(NDL_File &f, void *dst, size_t n) const {
    return f.readBytes(reinterpret_cast<char *>(dst), n) == n;
}

void NoodleAIModel::initTensorsIfNeeded() {
    if (_tensorsInitialized) return;
    noodle_tensor_init(&_actA);
    noodle_tensor_init(&_actB);
    _tensorsInitialized = true;
}

void NoodleAIModel::clearRuntime() {
#if defined(NAI_BENCH_FCN) && NAI_BENCH_FCN
    _lastFcnUs = 0;
#endif
#if defined(SRAM_BACKEND) && SRAM_BACKEND && !defined(NOODLE_USE_INT8)
    clearRamParameters();
#endif
    delete[] _mean;
    delete[] _scale;
    _mean = nullptr;
    _scale = nullptr;

    if (_tensorsInitialized) {
        noodle_tensor_free(&_actA);
        noodle_tensor_free(&_actB);
        _tensorsInitialized = false;
    }

    for (uint16_t i = 0; i < MAX_LAYERS; ++i) {
        _layers[i] = FCNFile{};
        _weightNames[i][0] = '\0';
        _biasNames[i][0] = '\0';
    }
    for (uint16_t i = 0; i < MAX_CLASSES; ++i) _labels[i] = "";
    memset(_dims, 0, sizeof(_dims));

    _loaded = false;
    _binaryClassifier = false;
    _slot = 0;
    _representation = REP_ACCEL;
    _windowLength = 0;
    _sampleRateHz = 0;
    _nLayers = 0;
    _nClasses = 0;
    _inputDim = 0;
#if defined(NOODLE_USE_INT8)
    _inputScale = 1.0f;
    _inputZeroPoint = 0;
    memset(_layerOutputScales,0,sizeof(_layerOutputScales));
    memset(_layerOutputZeroPoints,0,sizeof(_layerOutputZeroPoints));
#endif
}

bool NoodleAIModel::slotFilename(char slot, const char *canonical, char *out, size_t outCap) {
    if (!out || outCap == 0 || !canonical || (slot != 'A' && slot != 'B')) return false;
    if (!canonicalFilenameAllowed(canonical)) return false;
    const char prefix = (slot == 'A') ? 'a' : 'b';
    const int n = snprintf(out, outCap, "%c_%s", prefix, canonical);
    return n > 0 && static_cast<size_t>(n) < outCap && static_cast<size_t>(n) <= NOODLE_MAX_FILENAME;
}

bool NoodleAIModel::canonicalFilenameAllowed(const char *name) {
    if (!name) return false;
    if (strcmp(name, CFG_FILE) == 0 || strcmp(name, MEAN_FILE) == 0 || strcmp(name, SCALE_FILE) == 0)
        return true;
    if (strlen(name) != 7) return false; // w00.bin / b00.bin
    if ((name[0] != 'w' && name[0] != 'b' && name[0] != 'p') || name[3] != '.' || strcmp(name + 4, "bin") != 0)
        return false;
    if (name[1] < '0' || name[1] > '9' || name[2] < '0' || name[2] > '9') return false;
    const int idx = (name[1] - '0') * 10 + (name[2] - '0');
    return idx >= 0 && idx < MAX_LAYERS;
}

bool NoodleAIModel::fileHasSize(const char *name, size_t expected) {
    NDL_File f = noodle_fs_open_read(name);
    if (!f) return false;
    const size_t n = f.size();
    f.close();
    return n == expected;
}

bool NoodleAIModel::readConfig(char slot) {
    char name[NOODLE_MAX_FILENAME + 1];
    if (!slotFilename(slot, CFG_FILE, name, sizeof(name))) return false;
    NDL_File f = noodle_fs_open_read(name);
    if (!f) return false;

    ConfigHeader h{};
    if (!readExact(f, &h, sizeof(h))) { f.close(); return false; }

    // Backward compatibility:
    //   NAI2 = accelerometer only
    //   NAI3 = accel / gyro / accel+gyro
    //   NAI4 = raw or derived motion representation
    InputRepresentation parsedRepresentation = REP_ACCEL;
    if (memcmp(h.magic, "NAI2", 4) == 0 && h.version == 2) {
        parsedRepresentation = REP_ACCEL;
    } else if (memcmp(h.magic, "NAI3", 4) == 0 && h.version == 3) {
        if (h.reserved < REP_ACCEL || h.reserved > REP_ACCEL_GYRO) {
            f.close(); return false;
        }
        parsedRepresentation = static_cast<InputRepresentation>(h.reserved);
    } else if (memcmp(h.magic, "NAI4", 4) == 0 && h.version == MODEL_VERSION) {
        if (h.reserved < REP_ACCEL || h.reserved > REP_VELOCITY_QUATERNION) {
            f.close(); return false;
        }
        parsedRepresentation = static_cast<InputRepresentation>(h.reserved);
    } else {
        f.close(); return false;
    }

    const uint16_t channels = representationChannelCount(parsedRepresentation);
    if (h.nLayers == 0 || h.nLayers > MAX_LAYERS ||
        h.nClasses < 2 || h.nClasses > MAX_CLASSES ||
        h.windowLength == 0 || h.sampleRateHz == 0 ||
        h.inputDim == 0 || h.inputDim > MAX_DIM ||
        h.inputDim != static_cast<uint32_t>(h.windowLength) * channels) {
        f.close(); return false;
    }

    const bool packageInt8 = (h.flags & FLAG_INT8) != 0;
#if defined(NOODLE_USE_INT8)
    if (!packageInt8 || memcmp(h.magic,"NAI4",4) != 0) { f.close(); return false; }
#else
    if (packageInt8) { f.close(); return false; }
#endif
    if (h.flags & ~(FLAG_BINARY_CLASSIFIER | FLAG_INT8)) { f.close(); return false; }
    _binaryClassifier = (h.flags & FLAG_BINARY_CLASSIFIER) != 0;
    _representation = parsedRepresentation;
    _windowLength = h.windowLength;
    _sampleRateHz = h.sampleRateHz;
    _nLayers = h.nLayers;
    _nClasses = h.nClasses;
    _inputDim = h.inputDim;

    if (!readExact(f, _dims, sizeof(uint16_t) * (_nLayers + 1)) || _dims[0] != _inputDim) {
        f.close(); return false;
    }
    for (uint16_t l = 0; l < _nLayers; ++l) {
        if (_dims[l] == 0 || _dims[l] > MAX_DIM || _dims[l + 1] == 0 || _dims[l + 1] > MAX_DIM) {
            f.close(); return false;
        }
    }

    const uint16_t finalDim = _dims[_nLayers];
    if (_binaryClassifier) {
        if (_nClasses != 2 || finalDim != 1) { f.close(); return false; }
    } else if (finalDim != _nClasses) {
        f.close(); return false;
    }

    for (uint16_t c = 0; c < _nClasses; ++c) {
        uint8_t len = 0;
        if (!readExact(f, &len, 1) || len == 0 || len > MAX_LABEL_BYTES) { f.close(); return false; }
        char buf[MAX_LABEL_BYTES + 1]{};
        if (!readExact(f, buf, len)) { f.close(); return false; }
        buf[len] = '\0';
        _labels[c] = String(buf);
    }

#if defined(NOODLE_USE_INT8)
    if (!readQuantization(f)) { f.close(); return false; }
#endif
    const bool noTrailingBytes = !f.available();
    f.close();
    return noTrailingBytes;
}

#if defined(NOODLE_USE_INT8)
bool NoodleAIModel::readQuantization(NDL_File &f) {
    // Q8M1 exactly as written by the browser: 28 fixed bytes plus 8 per layer.
    char magic[4];
    uint16_t stride=0, reserved=0;
    float thresholdMin=0, thresholdMax=0, confidenceThreshold=0;
    if (!readExact(f,magic,4) || memcmp(magic,"Q8M1",4)!=0 ||
        !readExact(f,&stride,2) || !readExact(f,&reserved,2) ||
        !readExact(f,&thresholdMin,4) || !readExact(f,&thresholdMax,4) ||
        !readExact(f,&confidenceThreshold,4) ||
        !readExact(f,&_inputScale,4) || !readExact(f,&_inputZeroPoint,4)) return false;
    if (stride != 0 || reserved != 0 || !isfinite(_inputScale) || _inputScale <= 0 ||
        _inputZeroPoint < -128 || _inputZeroPoint > 127 ||
        !isfinite(thresholdMin) || !isfinite(thresholdMax) ||
        !isfinite(confidenceThreshold)) return false;
    for (uint16_t l=0;l<_nLayers;++l) {
        if (!readExact(f,&_layerOutputScales[l],4) ||
            !readExact(f,&_layerOutputZeroPoints[l],4)) return false;
        if (!isfinite(_layerOutputScales[l]) || _layerOutputScales[l] <= 0 ||
            _layerOutputZeroPoints[l] < -128 || _layerOutputZeroPoints[l] > 127) return false;
    }
    return true;
}
#endif

bool NoodleAIModel::validateParameterFiles(char slot) {
    char name[NOODLE_MAX_FILENAME + 1];
    if (!slotFilename(slot, MEAN_FILE, name, sizeof(name)) || !fileHasSize(name, sizeof(float) * _inputDim)) return false;
    if (!slotFilename(slot, SCALE_FILE, name, sizeof(name)) || !fileHasSize(name, sizeof(float) * _inputDim)) return false;

    for (uint16_t l = 0; l < _nLayers; ++l) {
        char canonical[8];
        snprintf(canonical, sizeof(canonical), "w%02u.bin", l);
        if (!slotFilename(slot, canonical, name, sizeof(name)) ||
            !fileHasSize(name, static_cast<size_t>(_dims[l]) * _dims[l + 1]
#if !defined(NOODLE_USE_INT8)
                * sizeof(float)
#endif
            )) return false;
#if defined(NOODLE_USE_INT8)
        snprintf(canonical, sizeof(canonical), "p%02u.bin", l);
        const size_t metadataSize = 12u * _dims[l+1];
#else
        snprintf(canonical, sizeof(canonical), "b%02u.bin", l);
        const size_t metadataSize = sizeof(float) * _dims[l+1];
#endif
        if (!slotFilename(slot, canonical, name, sizeof(name)) ||
            !fileHasSize(name, metadataSize)) return false;
    }
    return true;
}

bool NoodleAIModel::loadScaler(char slot) {
    _mean = new (std::nothrow) float[_inputDim];
    _scale = new (std::nothrow) float[_inputDim];
    if (!_mean || !_scale) return false;

    char name[NOODLE_MAX_FILENAME + 1];
    if (!slotFilename(slot, MEAN_FILE, name, sizeof(name))) return false;
    NDL_File fm = noodle_fs_open_read(name);
    if (!fm || !readExact(fm, _mean, sizeof(float) * _inputDim)) { if (fm) fm.close(); return false; }
    fm.close();

    if (!slotFilename(slot, SCALE_FILE, name, sizeof(name))) return false;
    NDL_File fs = noodle_fs_open_read(name);
    if (!fs || !readExact(fs, _scale, sizeof(float) * _inputDim)) { if (fs) fs.close(); return false; }
    fs.close();

    for (uint16_t i = 0; i < _inputDim; ++i) {
        if (!isfinite(_mean[i]) || !isfinite(_scale[i]) || fabsf(_scale[i]) < 1e-12f) return false;
    }
    return true;
}

void NoodleAIModel::configureLayers(char slot) {
    for (uint16_t l = 0; l < _nLayers; ++l) {
        char canonical[8];
        snprintf(canonical, sizeof(canonical), "w%02u.bin", l);
        slotFilename(slot, canonical, _weightNames[l], sizeof(_weightNames[l]));
#if defined(NOODLE_USE_INT8)
        snprintf(canonical, sizeof(canonical), "p%02u.bin", l);
#else
        snprintf(canonical, sizeof(canonical), "b%02u.bin", l);
#endif
        slotFilename(slot, canonical, _biasNames[l], sizeof(_biasNames[l]));

        _layers[l] = FCNFile{};
        _layers[l].weight_fn = _weightNames[l];
#if defined(NOODLE_USE_INT8)
        _layers[l].metadata_fn = _biasNames[l];
        _layers[l].input_scale = (l == 0) ? _inputScale : _layerOutputScales[l-1];
        _layers[l].input_zero_point = (l == 0) ? _inputZeroPoint : _layerOutputZeroPoints[l-1];
        _layers[l].output_scale = _layerOutputScales[l];
        _layers[l].output_zero_point = _layerOutputZeroPoints[l];
        _layers[l].activation_min = -128;
        _layers[l].activation_max = 127;
#else
        _layers[l].bias_fn = _biasNames[l];
#endif
        _layers[l].O = _dims[l + 1];
        _layers[l].act = (l + 1 < _nLayers)
            ? ACT_RELU
            : ACT_NONE;
#if !defined(NOODLE_USE_INT8)
        if (l + 1 == _nLayers && !_binaryClassifier) _layers[l].act = ACT_SOFTMAX;
#endif
#if defined(SRAM_BACKEND) && SRAM_BACKEND && !defined(NOODLE_USE_INT8)
        _ramLayers[l] = FCNMem{};
        _ramLayers[l].O = _layers[l].O;
        _ramLayers[l].act = _layers[l].act;
#endif
    }
}

#if defined(SRAM_BACKEND) && SRAM_BACKEND && !defined(NOODLE_USE_INT8)
void NoodleAIModel::clearRamParameters() {
    for (uint16_t l = 0; l < MAX_LAYERS; ++l) {
        delete[] _ramWeights[l];
        delete[] _ramBiases[l];
        _ramWeights[l] = nullptr;
        _ramBiases[l] = nullptr;
        _ramLayers[l] = FCNMem{};
    }
    _ramParameterBytes = 0;
}

bool NoodleAIModel::loadRamParameters() {
    const uint32_t started = micros();
    // The FFat parameter files remain canonical, so BLE model replacement and
    // A/B slots are unchanged. In SRAM mode we copy them once at model load.
    // Do not silently fall back to FFat if allocation or file I/O fails:
    // a timed SRAM run must actually use SRAM for every layer.
    clearRamParameters();
    for (uint16_t l = 0; l < _nLayers; ++l) {
        const size_t ni = _dims[l];
        const size_t no = _dims[l + 1];
        if (ni == 0 || no == 0 || ni > SIZE_MAX / no / sizeof(float)) {
            Serial.printf("NAI_SRAM error=invalid_layer_size layer=%u\n", (unsigned)l);
            clearRamParameters();
            return false;
        }
        const size_t weightBytes = ni * no * sizeof(float);
        const size_t biasBytes = no * sizeof(float);
        _ramWeights[l] = new (std::nothrow) float[ni * no];
        _ramBiases[l] = new (std::nothrow) float[no];
        if (!_ramWeights[l] || !_ramBiases[l]) {
            Serial.printf("NAI_SRAM error=out_of_memory layer=%u need=%u_bytes\n",
                (unsigned)l, (unsigned)(weightBytes + biasBytes));
            clearRamParameters();
            return false;
        }
        NDL_File weights = noodle_fs_open_read(_weightNames[l]);
        if (!weights) {
            Serial.printf("NAI_SRAM error=weight_open layer=%u\n", (unsigned)l);
            clearRamParameters();
            return false;
        }
        // Noodle's original .nai weights are already row-major [O][I],
        // unlike the transposed emlearn mirror. Preserve every bit.
        const bool gotWeights = readExact(weights, _ramWeights[l], weightBytes);
        weights.close();
        if (!gotWeights) {
            Serial.printf("NAI_SRAM error=weight_read layer=%u\n", (unsigned)l);
            clearRamParameters();
            return false;
        }
        NDL_File biases = noodle_fs_open_read(_biasNames[l]);
        if (!biases) {
            Serial.printf("NAI_SRAM error=bias_open layer=%u\n", (unsigned)l);
            clearRamParameters();
            return false;
        }
        const bool gotBiases = readExact(biases, _ramBiases[l], biasBytes);
        biases.close();
        if (!gotBiases) {
            Serial.printf("NAI_SRAM error=bias_read layer=%u\n", (unsigned)l);
            clearRamParameters();
            return false;
        }
        _ramLayers[l].weight = _ramWeights[l];
        _ramLayers[l].bias = _ramBiases[l];
        _ramLayers[l].O = _layers[l].O;
        _ramLayers[l].act = _layers[l].act;
        _ramParameterBytes += weightBytes + biasBytes;
    }
    Serial.printf("NAI_BACKEND=SRAM layers=%u param_ram_bytes=%u load_us=%lu free_heap=%u\n",
        (unsigned)_nLayers, (unsigned)_ramParameterBytes,
        (unsigned long)(micros() - started), (unsigned)ESP.getFreeHeap());
    return true;
}
#endif

bool NoodleAIModel::loadSlot(char slot) {
    clearRuntime();
    if (slot != 'A' && slot != 'B') return false;
    if (!readConfig(slot) || !validateParameterFiles(slot) || !loadScaler(slot)) {
        clearRuntime();
        return false;
    }

    configureLayers(slot);
#if defined(NOODLE_USE_INT8)
    Serial.println("NAI_EXEC backend=FFAT kernel=FCNFile_PACKED_INT8 emlearn=OFF");
#elif defined(SRAM_BACKEND) && SRAM_BACKEND
    Serial.println("NAI_EXEC backend=SRAM kernel=FAST_FP32 emlearn=OFF");
#elif defined(NAI_BENCH_FCN) && NAI_BENCH_FCN
    Serial.println("NAI_EXEC backend=FFAT kernel=FCNFile emlearn=OFF");
#endif
#if defined(SRAM_BACKEND) && SRAM_BACKEND && !defined(NOODLE_USE_INT8)
    if (!loadRamParameters()) {
        clearRuntime();
        return false;
    }
#elif defined(NAI_BENCH_FCN) && NAI_BENCH_FCN
    Serial.println("NAI_BACKEND=FFAT (parameters streamed during every inference)");
#endif
    initTensorsIfNeeded();
#if defined(NOODLE_USE_INT8)
    if (!noodle_i8_tensor_require_vector(&_actA, _inputDim, _inputScale, _inputZeroPoint) ||
        !noodle_i8_tensor_require_vector(&_actB, _dims[1], _layerOutputScales[0], _layerOutputZeroPoints[0])) {
#else
    if (!noodle_tensor_require_vector(&_actA, _inputDim) || !noodle_tensor_require_vector(&_actB, _dims[1])) {
#endif
        clearRuntime();
        return false;
    }

    _slot = slot;
    _loaded = true;
    return true;
}

char NoodleAIModel::readActiveSlot() const {
    NDL_File f = noodle_fs_open_read(ACTIVE_FILE);
    if (!f) return 0;
    const int c = f.read();
    f.close();
    return (c == 'A' || c == 'B') ? static_cast<char>(c) : 0;
}

bool NoodleAIModel::writeActiveSlot(char slot) {
    if (slot != 'A' && slot != 'B') return false;
    NDL_File f = noodle_fs_open_write(ACTIVE_FILE);
    if (!f) return false;
    const uint8_t c = static_cast<uint8_t>(slot);
    const bool ok = f.write(&c, 1) == 1;
    f.flush();
    f.close();
    return ok;
}

bool NoodleAIModel::loadActive() {
    const char preferred = readActiveSlot();
    if (preferred == 'A' || preferred == 'B') {
        if (loadSlot(preferred)) return true;
        const char other = (preferred == 'A') ? 'B' : 'A';
        if (loadSlot(other)) {
            writeActiveSlot(other);
            return true;
        }
    } else {
        if (loadSlot('A')) {
            writeActiveSlot('A');
            return true;
        }
        if (loadSlot('B')) {
            writeActiveSlot('B');
            return true;
        }
    }
    clearRuntime();
    return false;
}

bool NoodleAIModel::eraseSlot(char slot) const {
    if (slot != 'A' && slot != 'B') return false;
    bool ok = true;
    char name[NOODLE_MAX_FILENAME + 1];
    const char *fixed[] = {CFG_FILE, MEAN_FILE, SCALE_FILE};
    for (const char *canonical : fixed) {
        if (slotFilename(slot, canonical, name, sizeof(name))) noodle_fs_remove(name);
    }
    for (uint16_t l = 0; l < MAX_LAYERS; ++l) {
        char canonical[8];
        snprintf(canonical, sizeof(canonical), "w%02u.bin", l);
        if (slotFilename(slot, canonical, name, sizeof(name))) noodle_fs_remove(name);
        snprintf(canonical, sizeof(canonical), "b%02u.bin", l);
        if (slotFilename(slot, canonical, name, sizeof(name))) noodle_fs_remove(name);
        snprintf(canonical, sizeof(canonical), "p%02u.bin", l);
        if (slotFilename(slot, canonical, name, sizeof(name))) noodle_fs_remove(name);
    }
    return ok;
}

bool NoodleAIModel::eraseAll() const {
    eraseSlot('A');
    eraseSlot('B');
    noodle_fs_remove(ACTIVE_FILE);
    return true;
}

bool NoodleAIModel::predict(const float *x, uint16_t &classIndex, float &confidence) {
    if (!_loaded || !x || !_tensorsInitialized) return false;

#if defined(NOODLE_USE_INT8)
    int8_t *input = noodle_i8_tensor_require_vector(&_actA, _inputDim, _inputScale, _inputZeroPoint);
    if (!input) return false;
    for (uint16_t i = 0; i < _inputDim; ++i) {
        const float value = (x[i] - _mean[i]) / _scale[i];
        if (!isfinite(value)) return false;
        input[i] = static_cast<int8_t>(noodle_quantize_float(value, _inputScale, _inputZeroPoint));
    }
#else
    float *input = noodle_tensor_require_vector(&_actA, _inputDim);
    if (!input) return false;
    for (uint16_t i = 0; i < _inputDim; ++i) input[i] = (x[i] - _mean[i]) / _scale[i];
#endif

#if defined(NAI_BENCH_FCN) && NAI_BENCH_FCN
    const uint32_t fcnStarted = micros();
#endif
    NoodleTensor *current = &_actA;
    NoodleTensor *next = &_actB;
    for (uint16_t l = 0; l < _nLayers; ++l) {
#if defined(SRAM_BACKEND) && SRAM_BACKEND && !defined(NOODLE_USE_INT8)
        const uint16_t wrote = noodle_fcn(current, next, _ramLayers[l]);
#else
        const uint16_t wrote = noodle_fcn(current, next, _layers[l]);
#endif
        if (wrote != _dims[l + 1]) return false;
        NoodleTensor *tmp = current;
        current = next;
        next = tmp;
    }
#if defined(NAI_BENCH_FCN) && NAI_BENCH_FCN
    _lastFcnUs = micros() - fcnStarted;
#endif

#if defined(NOODLE_USE_INT8)
    const int8_t *out = noodle_tensor_data(current);
#else
    const float *out = noodle_tensor_data(current);
#endif
    if (!out) return false;

    if (_binaryClassifier) {
        if (noodle_sigmoid(current) != 1) return false;
#if defined(NOODLE_USE_INT8)
        const float p1 = (static_cast<int32_t>(noodle_tensor_data(current)[0]) + 128) / 256.0f;
#else
        const float p1 = noodle_tensor_data(current)[0];
#endif
        if (p1 >= 0.5f) { classIndex = 1; confidence = p1; }
        else { classIndex = 0; confidence = 1.0f - p1; }
        return true;
    }

#if defined(NOODLE_USE_INT8)
    if (noodle_soft_max(current) != _nClasses) return false;
    out = noodle_tensor_data(current);
    if (!out) return false;
    classIndex = 0;
    confidence = (static_cast<int32_t>(out[0]) + 128) / 256.0f;
    for (uint16_t i = 1; i < _nClasses; ++i) {
        const float p = (static_cast<int32_t>(out[i]) + 128) / 256.0f;
        if (p > confidence) { confidence = p; classIndex = i; }
    }
    return true;
#else
    classIndex = 0;
    confidence = out[0];
    for (uint16_t i = 1; i < _nClasses; ++i) {
        if (out[i] > confidence) { confidence = out[i]; classIndex = i; }
    }
    return true;
#endif
}

uint16_t NoodleAIModel::representationChannelCount(InputRepresentation rep) {
    switch (rep) {
    case REP_ACCEL: return 3;
    case REP_GYRO: return 3;
    case REP_ACCEL_GYRO: return 6;
    case REP_QUATERNION: return 4;
    case REP_VELOCITY: return 3;
    case REP_VELOCITY_QUATERNION: return 7;
    default: return 0;
    }
}

uint16_t NoodleAIModel::channelCount() const {
    return representationChannelCount(_representation);
}

const char *NoodleAIModel::representationName() const {
    switch (_representation) {
    case REP_ACCEL: return "accel";
    case REP_GYRO: return "gyro";
    case REP_ACCEL_GYRO: return "accel+gyro";
    case REP_QUATERNION: return "quaternion";
    case REP_VELOCITY: return "velocity";
    case REP_VELOCITY_QUATERNION: return "velocity+quaternion";
    default: return "unknown";
    }
}

const String &NoodleAIModel::label(uint16_t index) const {
    static const String empty = "";
    if (!_loaded || index >= _nClasses) return empty;
    return _labels[index];
}

size_t NoodleAIModel::tensorArenaBytes() const {
    return noodle_buffer_arena_used_bytes();
}
