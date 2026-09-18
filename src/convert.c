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

/* The conversion, with a constant offset folded into the scale.
 *
 * `off` is subtracted from every sample. It costs NOTHING per sample: each
 * format already subtracts a constant and multiplies by a scale, so
 * (x - c)*k - off is (x - (c + off/k))*k — a different constant, the same
 * two operations. That is the whole point. A caller that estimates the DC
 * offset once per block would otherwise need a second full pass to take it
 * out, and at 40 MSPS that pass is 16 MB read and 16 MB written every 50 ms
 * on a path that measures as bandwidth-bound.
 *
 * The mean of the samples BEFORE `off` was removed is accumulated in the
 * same loop and returned, so the caller can update its estimate without a
 * third pass. Eight partial accumulators, because one is a dependency chain.
 */
static ptrdiff_t unpack_core(const void *raw, size_t nbytes, int fmt,
                             atkdsp_cf32 *out, size_t out_cap,
                             float off_re, float off_im,
                             double *mean_re, double *mean_im) {
    if (!raw || !out) return ATKDSP_E_ARG;
    int bps = atkdsp_bytes_per_sample(fmt);
    if (bps == 0) return ATKDSP_E_ARG;
    size_t n = nbytes / (size_t)bps;
    if (n > out_cap) n = out_cap;

    switch (fmt) {
    case ATKDSP_FMT_CU8: {
        const uint8_t *ATK_RESTRICT b = (const uint8_t *)raw;
        const float k = 1.0f / 127.5f;
        const float cre = 127.5f + off_re / k, cim = 127.5f + off_im / k;
        for (size_t i = 0; i < n; ++i) {
            out[i].re = ((float)b[2 * i]     - cre) * k;
            out[i].im = ((float)b[2 * i + 1] - cim) * k;
        }
        break;
    }
    case ATKDSP_FMT_CI8: {
        const int8_t *ATK_RESTRICT b = (const int8_t *)raw;
        const float k = 1.0f / 128.0f;
        const float cre = off_re / k, cim = off_im / k;
        for (size_t i = 0; i < n; ++i) {
            out[i].re = ((float)b[2 * i]     - cre) * k;
            out[i].im = ((float)b[2 * i + 1] - cim) * k;
        }
        break;
    }
    case ATKDSP_FMT_CI16:
    case ATKDSP_FMT_CI16Q11: {
        const int16_t *ATK_RESTRICT b = (const int16_t *)raw;
        const float k = fmt == ATKDSP_FMT_CI16 ? (1.0f / 32768.0f) : (1.0f / 2048.0f);
        const float cre = off_re / k, cim = off_im / k;
        for (size_t i = 0; i < n; ++i) {
            out[i].re = ((float)b[2 * i]     - cre) * k;
            out[i].im = ((float)b[2 * i + 1] - cim) * k;
        }
        break;
    }
    case ATKDSP_FMT_CF32: {
        const atkdsp_cf32 *ATK_RESTRICT b = (const atkdsp_cf32 *)raw;
        if (off_re == 0.0f && off_im == 0.0f) {
            if ((const void *)out != raw) memcpy(out, b, n * sizeof(atkdsp_cf32));
        } else {
            for (size_t i = 0; i < n; ++i) {
                out[i].re = b[i].re - off_re;
                out[i].im = b[i].im - off_im;
            }
        }
        break;
    }
    default:
        return ATKDSP_E_ARG;
    }

    if (mean_re || mean_im) {
        double ar[8], ai[8];
        int l;
        for (l = 0; l < 8; ++l) { ar[l] = 0.0; ai[l] = 0.0; }
        size_t i = 0;
        const size_t m = n & ~(size_t)7;
        for (; i < m; i += 8)
            for (l = 0; l < 8; ++l) {
                ar[l] += (double)out[i + l].re;
                ai[l] += (double)out[i + l].im;
            }
        for (; i < n; ++i) { ar[0] += (double)out[i].re; ai[0] += (double)out[i].im; }
        double sr = 0.0, si = 0.0;
        for (l = 0; l < 8; ++l) { sr += ar[l]; si += ai[l]; }
        /* the mean BEFORE `off` was taken out: the loops above wrote x-off */
        const double inv = n ? 1.0 / (double)n : 0.0;
        if (mean_re) *mean_re = sr * inv + (double)off_re;
        if (mean_im) *mean_im = si * inv + (double)off_im;
    }
    return (ptrdiff_t)n;
}

ptrdiff_t atkdsp_unpack(const void *raw, size_t nbytes, int fmt,
                        atkdsp_cf32 *out, size_t out_cap,
                        float dc_alpha, atkdsp_cf32 *dc_state) {
    ptrdiff_t n = unpack_core(raw, nbytes, fmt, out, out_cap, 0.0f, 0.0f, NULL, NULL);
    if (n > 0 && dc_alpha > 0.0f && dc_state)
        dc_block(out, (size_t)n, dc_alpha, dc_state);
    return n;
}

ptrdiff_t atkdsp_unpack_dc(const void *raw, size_t nbytes, int fmt,
                           atkdsp_cf32 *out, size_t out_cap,
                           float off_re, float off_im,
                           double *mean_re, double *mean_im) {
    return unpack_core(raw, nbytes, fmt, out, out_cap, off_re, off_im,
                       mean_re, mean_im);
}
