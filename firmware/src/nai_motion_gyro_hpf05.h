#pragma once
// NoodleAI Mode 1: 0.5-Hz high-pass filter on CLASSIFICATION gyro inputs only.
// Applied to gX,gY,gZ on raw 50-Hz gesture samples, before the 5-Hz LPF.
// Reset at each BOOT-delimited gesture. Does NOT feed continuous Madgwick.
#include <math.h>
#include <stdint.h>

static inline void nai_filter_raw6_gyro_hpf05(float *raw6, uint16_t samples,
                                              float sampleRateHz) {
    if (!raw6 || samples == 0 || sampleRateHz <= 0.0f) return;
    const float dt = 1.0f / sampleRateHz;
    const float tau = 1.0f / (2.0f * 3.14159265358979323846f * 0.5f);
    const float alpha = tau / (tau + dt);
    for (uint8_t c = 3; c < 6; ++c) {
        float prevX = raw6[c];
        float hp = 0.0f;
        raw6[c] = 0.0f;
        for (uint16_t t = 1; t < samples; ++t) {
            const uint16_t index = t * 6u + c;
            const float x = raw6[index];
            hp = alpha * (hp + x - prevX);
            raw6[index] = hp;
            prevX = x;
        }
    }
}
