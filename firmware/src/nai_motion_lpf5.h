#pragma once
// 5 Hz first-order LPF on raw 50 Hz IMU samples, resetting at each BOOT gesture.
// LPF output starts at the first raw sample so gravity is preserved.
#include <math.h>
#include <stdint.h>

static inline void nai_filter_raw6_lpf5(float *raw6, uint16_t samples,
                                        float sampleRateHz,
                                        bool accel, bool gyro) {
    if (!raw6 || samples == 0 || sampleRateHz <= 0.0f) return;
    const float dt = 1.0f / sampleRateHz;
    const float tau = 1.0f / (2.0f * 3.14159265358979323846f * 5.0f);
    const float beta = dt / (tau + dt);
    for (uint8_t c = 0; c < 6; ++c) {
        if ((c < 3 && !accel) || (c >= 3 && !gyro)) continue;
        float lp = raw6[c];
        for (uint16_t t=1; t<samples; ++t) {
            const uint16_t offset = t*6u+c;
            const float x=raw6[offset];
            lp += beta * (x - lp);
            raw6[offset]=lp;
        }
    }
}

// NAI4 ID 9: unit direction + magnitude of the same LPF-filtered 3D vector.
// Near free-fall the direction is undefined; emit zero direction and true magnitude.
static inline void nai_accel_direction_magnitude(float ax, float ay, float az, float out[4]) {
    const float mag = sqrtf(ax*ax+ay*ay+az*az);
    if (!isfinite(mag)) { out[0]=out[1]=out[2]=out[3]=0.0f; return; }
    const float inv = mag > 0.001f ? (1.0f/mag) : 0.0f;
    out[0]=ax*inv;out[1]=ay*inv;out[2]=az*inv;out[3]=mag;
}
