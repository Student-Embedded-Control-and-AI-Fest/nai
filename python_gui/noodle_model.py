from __future__ import annotations

import io
import struct
import zipfile
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

MAGIC_V2 = b"NAI2"
MAGIC_V3 = b"NAI3"
MAGIC = b"NAI4"
VERSION = 4
FLAG_BINARY_CLASSIFIER = 0x0001
SAMPLE_RATE_HZ = 50
MAX_LABEL_BYTES = 31
MAX_LAYERS = 8
MAX_CLASSES = 10
MAX_DIM = 1536
MAX_PACKAGE_BYTES = 550 * 1024

# ConfigHeader.reserved is the input-representation code in NAI4.
REP_ACCEL = 1
REP_GYRO = 2
REP_ACCEL_GYRO = 3
REP_QUATERNION = 4
REP_VELOCITY = 5
REP_VELOCITY_QUATERNION = 6

REPRESENTATION_TO_CODE = {
    "accel": REP_ACCEL,
    "gyro": REP_GYRO,
    "accel+gyro": REP_ACCEL_GYRO,
    "quaternion": REP_QUATERNION,
    "velocity": REP_VELOCITY,
    "velocity+quaternion": REP_VELOCITY_QUATERNION,
}
CODE_TO_REPRESENTATION = {v: k for k, v in REPRESENTATION_TO_CODE.items()}
REPRESENTATION_CHANNELS = {
    "accel": 3,
    "gyro": 3,
    "accel+gyro": 6,
    "quaternion": 4,
    "velocity": 3,
    "velocity+quaternion": 7,
}

# Backward-compatible names used by NAI3-era code.
SENSOR_ACCEL = REP_ACCEL
SENSOR_GYRO = REP_GYRO
SENSOR_ACCEL_GYRO = REP_ACCEL_GYRO

HEADER = struct.Struct("<4s8H")


def normalize_representation(representation: str) -> str:
    s = str(representation).strip().lower().replace(" ", "")
    aliases = {
        "accel": "accel",
        "accelerometer": "accel",
        "gyro": "gyro",
        "gyroscope": "gyro",
        "accel+gyro": "accel+gyro",
        "accelerometer+gyroscope": "accel+gyro",
        "both": "accel+gyro",
        "6axis": "accel+gyro",
        "6-axis": "accel+gyro",
        "quaternion": "quaternion",
        "quat": "quaternion",
        "velocity": "velocity",
        "linearvelocity": "velocity",
        "estimatedvelocity": "velocity",
        "velocity+quaternion": "velocity+quaternion",
        "velocity+quat": "velocity+quaternion",
        "vel+quat": "velocity+quaternion",
    }
    if s not in aliases:
        raise ValueError(f"Unknown input representation: {representation!r}")
    return aliases[s]


def representation_channel_count(representation: str) -> int:
    return REPRESENTATION_CHANNELS[normalize_representation(representation)]


# NAI3 API compatibility -------------------------------------------------------
def normalize_sensor_mode(sensor_mode: str) -> str:
    rep = normalize_representation(sensor_mode)
    if rep not in ("accel", "gyro", "accel+gyro"):
        raise ValueError(f"Not a raw sensor mode: {sensor_mode!r}")
    return rep


def sensor_channel_count(sensor_mode: str) -> int:
    return representation_channel_count(normalize_sensor_mode(sensor_mode))


def _f32_bytes(a: np.ndarray) -> bytes:
    return np.asarray(a, dtype="<f4", order="C").tobytes(order="C")


def parameter_count(model) -> int:
    return int(sum(w.size + b.size for w, b in zip(model.coefs_, model.intercepts_)))


@dataclass
class NoodleAIFilePackage:
    files: dict[str, bytes]

    @property
    def total_bytes(self) -> int:
        return sum(len(v) for v in self.files.values())

    def to_archive_bytes(self) -> bytes:
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, data in self.files.items():
                zf.writestr(name, data)
        return out.getvalue()

    @classmethod
    def from_archive_bytes(cls, data: bytes) -> "NoodleAIFilePackage":
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            files = {name: zf.read(name) for name in zf.namelist()}
        return cls(files)


def export_sklearn_mlp(
    model,
    scaler,
    labels: Sequence[str],
    window_length: int,
    representation: str = "accel+gyro",
    *,
    sensor_mode: str | None = None,
) -> NoodleAIFilePackage:
    """Export a scikit MLPClassifier as a NoodleAI NAI4 package.

    NAI4 stores a runtime-selectable input representation:
      accel                 -> [ax ay az] * N
      gyro                  -> [gx gy gz] * N
      accel+gyro            -> [ax ay az gx gy gz] * N
      quaternion            -> relative Madgwick quaternion [qw qx qy qz] * N
      velocity              -> estimated world-frame linear velocity [vx vy vz] * N
      velocity+quaternion   -> [vx vy vz qw qx qy qz] * N

    Parameters remain float32 and file-backed in FFat.
    """
    if sensor_mode is not None:
        # Compatibility with older callers; explicit representation wins only
        # when the caller did not also supply sensor_mode.
        representation = sensor_mode

    labels = [str(x) for x in labels]
    rep = normalize_representation(representation)
    rep_code = REPRESENTATION_TO_CODE[rep]
    channels = representation_channel_count(rep)

    if len(labels) < 2 or len(labels) > MAX_CLASSES:
        raise ValueError(f"NoodleAI supports 2..{MAX_CLASSES} labels")
    if window_length <= 0:
        raise ValueError("window_length must be positive")

    expected_classes = np.arange(len(labels), dtype=int)
    if not np.array_equal(np.asarray(model.classes_, dtype=int), expected_classes):
        raise ValueError("Training labels must map to consecutive class indices 0..K-1")

    input_dim = int(window_length) * channels
    if input_dim > MAX_DIM:
        raise ValueError(f"Input dimension exceeds NoodleAI limit ({MAX_DIM})")
    if model.coefs_[0].shape[0] != input_dim:
        raise ValueError(
            f"Model input size {model.coefs_[0].shape[0]} does not match "
            f"{window_length} × {channels} for representation {rep}"
        )
    if scaler.mean_.shape[0] != input_dim or scaler.scale_.shape[0] != input_dim:
        raise ValueError("Scaler dimension does not match model input")

    n_layers = len(model.coefs_)
    if n_layers < 1 or n_layers > MAX_LAYERS:
        raise ValueError(f"NoodleAI supports 1..{MAX_LAYERS} Dense layers")

    binary = len(labels) == 2
    final_dim = model.coefs_[-1].shape[1]
    if binary and final_dim != 1:
        raise ValueError("Expected sklearn binary MLPClassifier to have one output logit")
    if not binary and final_dim != len(labels):
        raise ValueError("Multiclass output dimension must equal number of labels")

    dims = [input_dim] + [int(w.shape[1]) for w in model.coefs_]
    if any(d <= 0 or d > MAX_DIM for d in dims):
        raise ValueError(f"Every layer dimension must fit 1..{MAX_DIM}")

    flags = FLAG_BINARY_CLASSIFIER if binary else 0
    cfg = bytearray(
        HEADER.pack(
            MAGIC,
            VERSION,
            flags,
            int(window_length),
            SAMPLE_RATE_HZ,
            n_layers,
            len(labels),
            input_dim,
            rep_code,
        )
    )
    cfg += struct.pack("<" + "H" * len(dims), *dims)
    for label in labels:
        encoded = label.encode("utf-8")
        if not encoded or len(encoded) > MAX_LABEL_BYTES:
            raise ValueError(f"Label must be 1..{MAX_LABEL_BYTES} UTF-8 bytes: {label!r}")
        cfg += struct.pack("<B", len(encoded)) + encoded

    files: dict[str, bytes] = {
        "cfg.bin": bytes(cfg),
        "mean.bin": _f32_bytes(scaler.mean_),
        "scale.bin": _f32_bytes(scaler.scale_),
    }
    for i, (w, b) in enumerate(zip(model.coefs_, model.intercepts_)):
        # sklearn [I][O] -> Noodle sequential [O][I]
        files[f"w{i:02d}.bin"] = _f32_bytes(np.asarray(w, dtype=np.float32).T)
        files[f"b{i:02d}.bin"] = _f32_bytes(b)

    package = NoodleAIFilePackage(files)
    if package.total_bytes > MAX_PACKAGE_BYTES:
        raise ValueError(
            f"Model package is {package.total_bytes/1024:.1f} KiB; limit is "
            f"{MAX_PACKAGE_BYTES/1024:.0f} KiB so both transactional FFat slots fit."
        )
    return package


@dataclass
class NoodleAIReference:
    window_length: int
    sample_rate_hz: int
    representation: str
    labels: list[str]
    dims: list[int]
    mean: np.ndarray
    scale: np.ndarray
    weights_oi: list[np.ndarray]
    biases: list[np.ndarray]
    binary: bool

    @property
    def channel_count(self) -> int:
        return representation_channel_count(self.representation)

    @property
    def sensor_mode(self) -> str:
        # Compatibility for callers that only inspect this field.
        return self.representation

    @classmethod
    def from_package(cls, package: NoodleAIFilePackage | Mapping[str, bytes]) -> "NoodleAIReference":
        files = package.files if isinstance(package, NoodleAIFilePackage) else dict(package)
        cfg = io.BytesIO(files["cfg.bin"])
        raw = cfg.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError("cfg.bin header is truncated")

        magic, version, flags, window_length, sample_rate, n_layers, n_classes, input_dim, reserved = HEADER.unpack(raw)

        # Backward compatibility: NAI2 was accel-only; NAI3 supported the three
        # raw sensor selections. NAI4 extends the code to derived representations.
        if magic == MAGIC_V2 and version == 2:
            representation = "accel"
        elif magic == MAGIC_V3 and version == 3:
            if reserved not in (REP_ACCEL, REP_GYRO, REP_ACCEL_GYRO):
                raise ValueError("Invalid NAI3 sensor mode")
            representation = CODE_TO_REPRESENTATION[reserved]
        elif magic == MAGIC and version == VERSION:
            if reserved not in CODE_TO_REPRESENTATION:
                raise ValueError("Invalid NAI4 input representation")
            representation = CODE_TO_REPRESENTATION[reserved]
        else:
            raise ValueError("Not a supported NoodleAI model")

        channels = representation_channel_count(representation)
        if input_dim != window_length * channels:
            raise ValueError("Invalid input dimension for representation")

        dims_raw = cfg.read(2 * (n_layers + 1))
        if len(dims_raw) != 2 * (n_layers + 1):
            raise ValueError("Truncated dimensions")
        dims = list(struct.unpack("<" + "H" * (n_layers + 1), dims_raw))
        if not dims or dims[0] != input_dim:
            raise ValueError("Invalid first layer dimension")

        labels: list[str] = []
        for _ in range(n_classes):
            n_raw = cfg.read(1)
            if len(n_raw) != 1:
                raise ValueError("Truncated label length")
            n = n_raw[0]
            text = cfg.read(n)
            if len(text) != n:
                raise ValueError("Truncated label")
            labels.append(text.decode("utf-8"))
        if cfg.read(1):
            raise ValueError("Unexpected trailing cfg.bin bytes")

        def f32_file(name: str, count: int) -> np.ndarray:
            data = files[name]
            if len(data) != count * 4:
                raise ValueError(f"{name} has the wrong byte count")
            return np.frombuffer(data, dtype="<f4").astype(np.float32, copy=True)

        mean = f32_file("mean.bin", input_dim)
        scale = f32_file("scale.bin", input_dim)
        weights_oi: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        for i in range(n_layers):
            in_dim, out_dim = dims[i], dims[i + 1]
            weights_oi.append(f32_file(f"w{i:02d}.bin", in_dim * out_dim).reshape(out_dim, in_dim))
            biases.append(f32_file(f"b{i:02d}.bin", out_dim))

        return cls(
            window_length=window_length,
            sample_rate_hz=sample_rate,
            representation=representation,
            labels=labels,
            dims=dims,
            mean=mean,
            scale=scale,
            weights_oi=weights_oi,
            biases=biases,
            binary=bool(flags & FLAG_BINARY_CLASSIFIER),
        )

    def predict_proba(self, X_raw: np.ndarray) -> np.ndarray:
        X = np.asarray(X_raw, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.dims[0]:
            raise ValueError("Input dimension mismatch")
        a = (X - self.mean) / self.scale

        for i, (w_oi, b) in enumerate(zip(self.weights_oi, self.biases)):
            a = a @ w_oi.T + b
            if i != len(self.weights_oi) - 1:
                a = np.maximum(a, np.float32(0.0))

        if self.binary:
            p1 = np.empty_like(a, dtype=np.float32)
            pos = a >= 0
            p1[pos] = 1.0 / (1.0 + np.exp(-a[pos]))
            ea = np.exp(a[~pos])
            p1[~pos] = ea / (1.0 + ea)
            p1 = p1[:, 0]
            return np.column_stack((1.0 - p1, p1)).astype(np.float32)

        logits = a - np.max(a, axis=1, keepdims=True)
        e = np.exp(logits)
        return (e / np.sum(e, axis=1, keepdims=True)).astype(np.float32)

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X_raw), axis=1)


NoodleMLPReference = NoodleAIReference
