from __future__ import annotations

import math
import numpy as np

SAMPLE_RATE_HZ = 50
MADGWICK_BETA = 0.10
GRAVITY_MPS2 = 9.80665

RAW_REPRESENTATIONS = {"accel", "gyro", "accel+gyro"}
DERIVED_REPRESENTATIONS = {"quaternion", "velocity", "velocity+quaternion"}


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n > eps, n, 1.0)
    return x / n


def resample_time_major(data: np.ndarray, target_n: int) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or len(data) < 2:
        raise ValueError("Gesture representation must be a 2-D array with at least two samples")
    if target_n < 2:
        raise ValueError("Normalized gesture length must be at least 2")
    pos = np.linspace(0.0, float(len(data) - 1), int(target_n), dtype=np.float32)
    i0 = np.floor(pos).astype(np.int32)
    i1 = np.minimum(i0 + 1, len(data) - 1)
    alpha = (pos - i0.astype(np.float32))[:, None]
    out = data[i0] + alpha * (data[i1] - data[i0])
    return np.asarray(out, dtype=np.float32)


def _q_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if not math.isfinite(n) or n < 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _q_conj(q: np.ndarray) -> np.ndarray:
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = [float(v) for v in a]
    bw, bx, by, bz = [float(v) for v in b]
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _q_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    p = np.asarray([0.0, float(v[0]), float(v[1]), float(v[2])], dtype=np.float64)
    return _q_mul(_q_mul(q, p), _q_conj(q))[1:]


def _quat_align_accel_to_world_z(accel_g: np.ndarray) -> np.ndarray:
    """Minimal-rotation quaternion that maps measured gravity to +world Z.

    This initializes roll/pitch directly from the first accelerometer sample.
    Yaw remains deliberately unreferenced (zero by convention for each gesture).
    """
    u = np.asarray(accel_g, dtype=np.float64)
    n = float(np.linalg.norm(u))
    if not math.isfinite(n) or n < 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    u /= n
    v = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    d = float(np.dot(u, v))
    if d < -0.999999:
        # 180-degree case: choose a stable axis perpendicular to u.
        base = np.asarray([1.0, 0.0, 0.0], dtype=np.float64) if abs(u[0]) < 0.9 else np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(u, base)
        axis /= np.linalg.norm(axis)
        return np.asarray([0.0, axis[0], axis[1], axis[2]], dtype=np.float64)
    return _q_normalize(np.asarray([1.0 + d, *np.cross(u, v)], dtype=np.float64))


def madgwick_update_imu(
    q: np.ndarray,
    accel_g: np.ndarray,
    gyro_dps: np.ndarray,
    dt: float,
    beta: float = MADGWICK_BETA,
) -> np.ndarray:
    """One 6-axis Madgwick IMU update; quaternion order is [w,x,y,z]."""
    q1, q2, q3, q4 = [float(x) for x in q]
    gx, gy, gz = [math.radians(float(x)) for x in gyro_dps]
    ax, ay, az = [float(x) for x in accel_g]

    qdot = np.asarray(
        [
            0.5 * (-q2 * gx - q3 * gy - q4 * gz),
            0.5 * (q1 * gx + q3 * gz - q4 * gy),
            0.5 * (q1 * gy - q2 * gz + q4 * gx),
            0.5 * (q1 * gz + q2 * gy - q3 * gx),
        ],
        dtype=np.float64,
    )

    an = math.sqrt(ax * ax + ay * ay + az * az)
    if math.isfinite(an) and an > 1e-8:
        ax /= an
        ay /= an
        az /= an

        _2q1, _2q2, _2q3, _2q4 = 2.0 * q1, 2.0 * q2, 2.0 * q3, 2.0 * q4
        _4q1, _4q2, _4q3 = 4.0 * q1, 4.0 * q2, 4.0 * q3
        _8q2, _8q3 = 8.0 * q2, 8.0 * q3
        q1q1, q2q2, q3q3, q4q4 = q1*q1, q2*q2, q3*q3, q4*q4

        s = np.asarray(
            [
                _4q1*q3q3 + _2q3*ax + _4q1*q2q2 - _2q2*ay,
                _4q2*q4q4 - _2q4*ax + 4.0*q1q1*q2 - _2q1*ay - _4q2 + _8q2*q2q2 + _8q2*q3q3 + _4q2*az,
                4.0*q1q1*q3 + _2q1*ax + _4q3*q4q4 - _2q4*ay - _4q3 + _8q3*q2q2 + _8q3*q3q3 + _4q3*az,
                4.0*q2q2*q4 - _2q2*ax + 4.0*q3q3*q4 - _2q3*ay,
            ],
            dtype=np.float64,
        )
        sn = float(np.linalg.norm(s))
        if math.isfinite(sn) and sn > 1e-12:
            qdot -= float(beta) * (s / sn)

    return _q_normalize(np.asarray(q, dtype=np.float64) + qdot * float(dt))


def derive_quaternion_velocity(
    raw6: np.ndarray,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    beta: float = MADGWICK_BETA,
) -> tuple[np.ndarray, np.ndarray]:
    """Return relative quaternion [T,4] and zero-endpoint velocity [T,3].

    The accelerometer is assumed to be expressed in g and gyro in deg/s.
    The first accelerometer sample defines world +Z; yaw is arbitrary. Velocity
    uses gravity-compensated world-frame acceleration and a linear endpoint drift
    correction so v(0)=v(T)=0 for each button-delimited gesture.
    """
    raw = np.asarray(raw6, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 6 or len(raw) < 2:
        raise ValueError("Derived motion features require at least two raw 6-axis samples")
    fs = float(sample_rate_hz)
    if not math.isfinite(fs) or fs <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    dt = 1.0 / fs

    q_abs = _quat_align_accel_to_world_z(raw[0, :3])
    q_ref = q_abs.copy()
    q_rel = np.empty((len(raw), 4), dtype=np.float64)
    vel = np.zeros((len(raw), 3), dtype=np.float64)

    q_rel[0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    a_world_prev = _q_rotate(q_abs, raw[0, :3])
    a_lin_prev = (a_world_prev - np.asarray([0.0, 0.0, 1.0])) * GRAVITY_MPS2

    for k in range(1, len(raw)):
        q_abs = madgwick_update_imu(q_abs, raw[k, :3], raw[k, 3:6], dt, beta=beta)
        qr = _q_normalize(_q_mul(_q_conj(q_ref), q_abs))
        if float(np.dot(qr, q_rel[k - 1])) < 0.0:
            qr = -qr
        q_rel[k] = qr

        a_world = _q_rotate(q_abs, raw[k, :3])
        a_lin = (a_world - np.asarray([0.0, 0.0, 1.0])) * GRAVITY_MPS2
        vel[k] = vel[k - 1] + 0.5 * (a_lin_prev + a_lin) * dt
        a_lin_prev = a_lin

    # BOOT-delimited gestures are assumed approximately at rest at both ends.
    # Remove the linearly accumulated residual so both velocity boundaries are 0.
    end_v = vel[-1].copy()
    alpha = np.linspace(0.0, 1.0, len(raw), dtype=np.float64)[:, None]
    vel -= alpha * end_v[None, :]
    vel[0] = 0.0
    vel[-1] = 0.0

    return np.asarray(q_rel, dtype=np.float32), np.asarray(vel, dtype=np.float32)


def build_representation(
    raw6: np.ndarray,
    target_n: int,
    representation: str,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    beta: float = MADGWICK_BETA,
) -> np.ndarray:
    """Build one fixed-length time-major NoodleAI representation vector."""
    rep = str(representation).strip().lower().replace(" ", "")
    aliases = {
        "accelerometer": "accel", "accel": "accel",
        "gyroscope": "gyro", "gyro": "gyro",
        "accel+gyro": "accel+gyro", "both": "accel+gyro", "6axis": "accel+gyro", "6-axis": "accel+gyro",
        "quaternion": "quaternion", "quat": "quaternion",
        "velocity": "velocity", "linearvelocity": "velocity", "estimatedvelocity": "velocity",
        "velocity+quaternion": "velocity+quaternion", "velocity+quat": "velocity+quaternion", "vel+quat": "velocity+quaternion",
    }
    if rep not in aliases:
        raise ValueError(f"Unknown input representation: {representation!r}")
    rep = aliases[rep]

    raw = np.asarray(raw6, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 6 or len(raw) < 2:
        raise ValueError("Gesture needs at least two raw 6-axis IMU samples")

    if rep in RAW_REPRESENTATIONS:
        idx = {
            "accel": (0, 1, 2),
            "gyro": (3, 4, 5),
            "accel+gyro": (0, 1, 2, 3, 4, 5),
        }[rep]
        out = resample_time_major(raw[:, idx], target_n)
        out -= np.mean(out, axis=0, dtype=np.float32, keepdims=True)
        return np.asarray(out, dtype=np.float32).reshape(-1)

    quat, vel = derive_quaternion_velocity(raw, sample_rate_hz=sample_rate_hz, beta=beta)
    if rep == "quaternion":
        out = resample_time_major(quat, target_n)
        out = _normalize_rows(out.astype(np.float64)).astype(np.float32)
    elif rep == "velocity":
        out = resample_time_major(vel, target_n)
    else:
        combined = np.concatenate((vel, quat), axis=1)
        out = resample_time_major(combined, target_n)
        out[:, 3:7] = _normalize_rows(out[:, 3:7].astype(np.float64)).astype(np.float32)
    return np.asarray(out, dtype=np.float32).reshape(-1)
