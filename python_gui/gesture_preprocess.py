from __future__ import annotations

import numpy as np

FEATURE_NAMES = ("ax", "ay", "az", "gx", "gy", "gz")
MODE_INDICES = {
    "accel": (0, 1, 2),
    "gyro": (3, 4, 5),
    "accel+gyro": (0, 1, 2, 3, 4, 5),
}


def resample_center_6axis(raw: np.ndarray, target_n: int) -> np.ndarray:
    """Variable-duration [T,6] gesture -> centered [target_n,6]."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 6 or len(raw) < 2:
        raise ValueError("raw must have shape [T,6] with T>=2")
    if target_n < 2:
        raise ValueError("target_n must be >=2")

    pos = np.linspace(0.0, float(len(raw) - 1), target_n, dtype=np.float32)
    i0 = np.floor(pos).astype(np.int32)
    i1 = np.minimum(i0 + 1, len(raw) - 1)
    alpha = (pos - i0.astype(np.float32))[:, None]
    out = raw[i0] + alpha * (raw[i1] - raw[i0])
    out -= np.mean(out, axis=0, keepdims=True, dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def project_sensor_mode(centered6: np.ndarray, sensor_mode: str) -> np.ndarray:
    """Select accel, gyro, or all six channels and flatten time-major."""
    x = np.asarray(centered6, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 6:
        raise ValueError("centered6 must have shape [N,6]")
    mode = str(sensor_mode).strip().lower()
    if mode not in MODE_INDICES:
        raise ValueError(f"Unknown sensor mode: {sensor_mode}")
    return np.asarray(x[:, MODE_INDICES[mode]], dtype=np.float32).reshape(-1)


def preprocess_gesture(raw: np.ndarray, target_n: int, sensor_mode: str = "accel+gyro") -> np.ndarray:
    return project_sensor_mode(resample_center_6axis(raw, target_n), sensor_mode)
