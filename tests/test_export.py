import sys
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_gui"))

from noodle_model import NoodleAIReference, export_sklearn_mlp, sensor_channel_count


def test_export_matches_sklearn_for_all_sensor_modes():
    rng = np.random.default_rng(42)
    labels = ["1", "2", "3"]
    n = 12

    for mode in ("accel", "gyro", "accel+gyro"):
        d = n * sensor_channel_count(mode)
        X = rng.normal(size=(60, d)).astype(np.float32)
        y = np.repeat(np.arange(3), 20)
        # Give classes a small deterministic separation.
        X[y == 1, 0] += 1.5
        X[y == 2, 1] -= 1.5

        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        clf = MLPClassifier(hidden_layer_sizes=(8,), random_state=1, max_iter=300).fit(Xs, y)

        pkg = export_sklearn_mlp(clf, scaler, labels, n, sensor_mode=mode)
        ref = NoodleAIReference.from_package(pkg)
        assert ref.sensor_mode == mode
        assert ref.dims[0] == d
        np.testing.assert_array_equal(ref.predict(X), clf.predict(Xs))
