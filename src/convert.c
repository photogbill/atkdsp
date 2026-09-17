/* 1. unpack: raw SDR bytes -> cf32 in one pass, optional running DC block. */
#include "internal.h"

/* One-pole DC block applied in place: state <- state + alpha*(x - state);
 * y = x - state. Kept in a separate loop so the conversion loops above it
 * stay trivially vectorisable. */
static void dc_block(atkdsp_cf32 *ATK_RESTRICT x, size_t n, float alpha,
                     atkdsp_cf32 *ATK_RESTRICT st) {
    float sr = st->re, si = st->im;
    for (size_t i = 0; i < n; ++i) {
        sr += alpha * (x[i].re - sr);
        si += alpha * (x[i].im - si);
        x[i].re -= sr;
        x[i].im -= si;
    }
    st->re = sr; st->im = si;
}

ptrdiff_t atkdsp_unpack(const void *raw, size_t nbytes, int fmt,
                        atkdsp_cf32 *out, size_t out_cap,
                        float dc_alpha, atkdsp_cf32 *dc_state) {
    if (!raw || !out) return ATKDSP_E_ARG;
    int bps = atkdsp_bytes_per_sample(fmt);
    if (bps == 0) return ATKDSP_E_ARG;
    size_t n = nbytes / (size_t)bps;
    if (n > out_cap) n = out_cap;

    switch (fmt) {
    case ATKDSP_FMT_CU8: {
        const uint8_t *ATK_RESTRICT b = (const uint8_t *)raw;
        const float k = 1.0f / 127.5f;
        for (size_t i = 0; i < n; ++i) {
            out[i].re = ((float)b[2 * i]     - 127.5f) * k;
            out[i].im = ((float)b[2 * i + 1] - 127.5f) * k;
        }
        break;
    }
    case ATKDSP_FMT_CI8: {
        const int8_t *ATK_RESTRICT b = (const int8_t *)raw;
        const float k = 1.0f / 128.0f;
        for (size_t i = 0; i < n; ++i) {
            out[i].re = (float)b[2 * i]     * k;
            out[i].im = (float)b[2 * i + 1] * k;
        }
        break;
    }
    case ATKDSP_FMT_CI16:
    case ATKDSP_FMT_CI16Q11: {
        const int16_t *ATK_RESTRICT b = (const int16_t *)raw;
        const float k = fmt == ATKDSP_FMT_CI16 ? (1.0f / 32768.0f) : (1.0f / 2048.0f);
        for (size_t i = 0; i < n; ++i) {
            out[i].re = (float)b[2 * i]     * k;
            out[i].im = (float)b[2 * i + 1] * k;
        }
        break;
    }
    case ATKDSP_FMT_CF32:
        if ((const void *)out != raw)
            memcpy(out, raw, n * sizeof(atkdsp_cf32));
        break;
    default:
        return ATKDSP_E_ARG;
    }

    if (dc_alpha > 0.0f && dc_state)
        dc_block(out, n, dc_alpha, dc_state);
    return (ptrdiff_t)n;
}
