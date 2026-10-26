#pragma once
// NoodleAI Mode 1: the direct-feature filter uses Mode 2's first-order
// high-pass -> low-pass equations. State is reset at each BOOT gesture.
// The sample period is the RAW acquisition period, before temporal resampling.
#include <math.h>
#include <stdint.h>

static inline void nai_filter_raw6_hplp(float *raw6, uint16_t samples,
                                        float sampleRateHz,
                                        bool accel, bool gyro,
                                        float highpassHz=0.10f,
                                        float lowpassHz=8.0f) {
    if (!raw6 || samples == 0 || sampleRateHz <= 0 ||
        highpassHz <= 0 || lowpassHz <= 0) return;
    const float dt = 1.0f / sampleRateHz;
    const float tauHP = 1.0f / (2.0f * 3.14159265358979323846f * highpassHz);
    const float tauLP = 1.0f / (2.0f * 3.14159265358979323846f * lowpassHz);
    const float alpha = tauHP / (tauHP + dt);
    const float beta = dt / (tauLP + dt);
    for (uint8_t c = 0; c < 6; ++c) {
        if ((c < 3 && !accel) || (c >= 3 && !gyro)) continue;
        float prevX = raw6[c];
        float prevHP = 0.0f;
        float prevLP = 0.0f;
        raw6[c] = 0.0f; // explicitly initialize per-gesture filter state
        for (uint16_t t = 1; t < samples; ++t) {
            const uint16_t offset = t * 6u + c;
            const float x = raw6[offset];
            const float hp = alpha * (prevHP + x - prevX);
            const float lp = prevLP + beta * (hp - prevLP);
            raw6[offset] = lp;
            prevX = x;
            prevHP = hp;
            prevLP = lp;
        }
    }
}
