/* 6. demodulators with carried state. */
#include "internal.h"

void atkdsp_fm_demod(const atkdsp_cf32 *in, size_t n, float *out, atkdsp_cf32 *prev, float gain) {
    if (!in || !out || !prev) return;
    float pr = prev->re, pi = prev->im;
    for (size_t i = 0; i < n; ++i) {
        const float xr = in[i].re, xi = in[i].im;
        /* x * conj(prev) */
        const float qr = xr * pr + xi * pi;
        const float qi = xi * pr - xr * pi;
        out[i] = gain * atan2f(qi, qr);
        pr = xr; pi = xi;
    }
    prev->re = pr; prev->im = pi;
}

void atkdsp_am_demod(const atkdsp_cf32 *in, size_t n, float *out, float *dc_state, float alpha) {
    if (!in || !out) return;
    if (alpha > 0.0f && dc_state) {
        float s = *dc_state;
        for (size_t i = 0; i < n; ++i) {
            const float env = sqrtf(in[i].re * in[i].re + in[i].im * in[i].im);
            s += alpha * (env - s);
            out[i] = env - s;
        }
        *dc_state = s;
    } else {
        for (size_t i = 0; i < n; ++i)
            out[i] = sqrtf(in[i].re * in[i].re + in[i].im * in[i].im);
    }
}
