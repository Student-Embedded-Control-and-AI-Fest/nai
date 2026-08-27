#include "NoodleAIModel.h"

#include <math.h>
#include <new>
#include <string.h>

namespace {
constexpr uint16_t MODEL_VERSION = 4;
constexpr uint16_t FLAG_BINARY_CLASSIFIER = 0x0001;
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
    if ((name[0] != 'w' && name[0] != 'b') || name[3] != '.' || strcmp(name + 4, "bin") != 0)
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

    const bool noTrailingBytes = !f.available();
    f.close();
    return noTrailingBytes;
}

bool NoodleAIModel::validateParameterFiles(char slot) {
    char name[NOODLE_MAX_FILENAME + 1];
    if (!slotFilename(slot, MEAN_FILE, name, sizeof(name)) || !fileHasSize(name, sizeof(float) * _inputDim)) return false;
    if (!slotFilename(slot, SCALE_FILE, name, sizeof(name)) || !fileHasSize(name, sizeof(float) * _inputDim)) return false;

    for (uint16_t l = 0; l < _nLayers; ++l) {
        char canonical[8];
        snprintf(canonical, sizeof(canonical), "w%02u.bin", l);
        if (!slotFilename(slot, canonical, name, sizeof(name)) ||
            !fileHasSize(name, sizeof(float) * static_cast<size_t>(_dims[l]) * _dims[l + 1])) return false;
        snprintf(canonical, sizeof(canonical), "b%02u.bin", l);
        if (!slotFilename(slot, canonical, name, sizeof(name)) ||
            !fileHasSize(name, sizeof(float) * _dims[l + 1])) return false;
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
        snprintf(canonical, sizeof(canonical), "b%02u.bin", l);
        slotFilename(slot, canonical, _biasNames[l], sizeof(_biasNames[l]));

        _layers[l] = FCNFile{};
        _layers[l].weight_fn = _weightNames[l];
        _layers[l].bias_fn = _biasNames[l];
        _layers[l].O = _dims[l + 1];
        _layers[l].act = (l + 1 < _nLayers)
            ? ACT_RELU
            : (_binaryClassifier ? ACT_NONE : ACT_SOFTMAX);
    }
}

bool NoodleAIModel::loadSlot(char slot) {
    clearRuntime();
    if (slot != 'A' && slot != 'B') return false;
    if (!readConfig(slot) || !validateParameterFiles(slot) || !loadScaler(slot)) {
        clearRuntime();
        return false;
    }

    configureLayers(slot);
    initTensorsIfNeeded();
    if (!noodle_tensor_require_vector(&_actA, _inputDim) || !noodle_tensor_require_vector(&_actB, _dims[1])) {
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

    float *input = noodle_tensor_require_vector(&_actA, _inputDim);
    if (!input) return false;
    for (uint16_t i = 0; i < _inputDim; ++i) input[i] = (x[i] - _mean[i]) / _scale[i];

    NoodleTensor *current = &_actA;
    NoodleTensor *next = &_actB;
    for (uint16_t l = 0; l < _nLayers; ++l) {
        if (noodle_fcn(current, next, _layers[l]) != _dims[l + 1]) return false;
        NoodleTensor *tmp = current;
        current = next;
        next = tmp;
    }

    float *out = noodle_tensor_data(current);
    if (!out) return false;

    if (_binaryClassifier) {
        if (noodle_sigmoid(current) != 1) return false;
        const float p1 = noodle_tensor_data(current)[0];
        if (p1 >= 0.5f) { classIndex = 1; confidence = p1; }
        else { classIndex = 0; confidence = 1.0f - p1; }
        return true;
    }

    classIndex = 0;
    confidence = out[0];
    for (uint16_t i = 1; i < _nClasses; ++i) {
        if (out[i] > confidence) { confidence = out[i]; classIndex = i; }
    }
    return true;
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
