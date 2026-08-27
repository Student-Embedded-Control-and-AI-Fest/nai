import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_gui"))

from gesture_preprocess import preprocess_gesture, resample_center_6axis


def test_centering_removes_constant_offsets():
    t = np.linspace(0, 1, 31, dtype=np.float32)
    raw = np.column_stack([
        np.sin(2*np.pi*t), np.cos(2*np.pi*t), t,
        100*np.sin(2*np.pi*t), 80*np.cos(2*np.pi*t), 40*t,
    ]).astype(np.float32)
    offsets = np.array([0.4, -0.2, 1.0, 20.0, -50.0, 100.0], dtype=np.float32)
    a = resample_center_6axis(raw, 50)
    b = resample_center_6axis(raw + offsets, 50)
    np.testing.assert_allclose(a, b, atol=2e-5)


def test_sensor_projection_dimensions():
    raw = np.arange(20*6, dtype=np.float32).reshape(20, 6)
    assert preprocess_gesture(raw, 50, "accel").shape == (150,)
    assert preprocess_gesture(raw, 50, "gyro").shape == (150,)
    assert preprocess_gesture(raw, 50, "accel+gyro").shape == (300,)
