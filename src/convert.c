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

int atkdsp_iq_health(const atkdsp_cf32 *in, size_t n, float clip_level,
                     atkdsp_iq_report *out) {
    if (!in || !out) return ATKDSP_E_ARG;
    memset(out, 0, sizeof *out);
    out->n = n;
    if (n == 0) return ATKDSP_OK;
    const double thr = (clip_level > 0.0f) ? (double)clip_level : 0.99;
    /* one pass: the five second-order sums, and the clip count. Eight partial
     * accumulators so the adds are not one dependency chain. */
    double sI[8], sQ[8], sII[8], sQQ[8], sIQ[8];
    size_t clip = 0;
    int l;
    for (l = 0; l < 8; ++l) { sI[l]=sQ[l]=sII[l]=sQQ[l]=sIQ[l]=0.0; }
    size_t i = 0;
    const size_t m = n & ~(size_t)7;
    for (; i < m; i += 8)
        for (l = 0; l < 8; ++l) {
            const double re = (double)in[i+l].re, im = (double)in[i+l].im;
            sI[l]+=re; sQ[l]+=im; sII[l]+=re*re; sQQ[l]+=im*im; sIQ[l]+=re*im;
            if (re >= thr || re <= -thr || im >= thr || im <= -thr) ++clip;
        }
    double SI=0,SQ=0,SII=0,SQQ=0,SIQ=0;
    for (l = 0; l < 8; ++l) { SI+=sI[l]; SQ+=sQ[l]; SII+=sII[l]; SQQ+=sQQ[l]; SIQ+=sIQ[l]; }
    for (; i < n; ++i) {
        const double re = (double)in[i].re, im = (double)in[i].im;
        SI+=re; SQ+=im; SII+=re*re; SQQ+=im*im; SIQ+=re*im;
        if (re >= thr || re <= -thr || im >= thr || im <= -thr) ++clip;
    }
    const double N = (double)n;
    const double mI = SI/N, mQ = SQ/N;
    out->dc_re = mI; out->dc_im = mQ;
    out->rms = sqrt((SII + SQQ) / N);
    /* variances and covariance with the DC removed */
    double vI = SII/N - mI*mI;
    double vQ = SQQ/N - mQ*mQ;
    double cIQ = SIQ/N - mI*mQ;
    if (vI < 0) vI = 0; if (vQ < 0) vQ = 0;
    out->gain_imbalance_db = (vQ > 1e-30 && vI > 1e-30)
                             ? 10.0 * log10(vI / vQ) : 0.0;
    double denom = sqrt(vI * vQ);
    double s = (denom > 1e-30) ? cIQ / denom : 0.0;
    if (s > 1.0) s = 1.0; if (s < -1.0) s = -1.0;
    const double phi = asin(s);                       /* radians */
    out->phase_error_deg = phi * 180.0 / ATK_PI;
    /* image rejection from amplitude ratio a and phase phi:
     * IRR = (a^2 + 1 + 2a cos phi) / (a^2 + 1 - 2a cos phi); perfect -> inf. */
    if (vI > 1e-30 && vQ > 1e-30) {
        const double a = sqrt(vI / vQ);
        const double c = cos(phi);
        const double num = a*a + 1.0 + 2.0*a*c;
        const double den = a*a + 1.0 - 2.0*a*c;
        out->image_rejection_db = (den > 1e-30)
                                  ? 10.0 * log10(num / den) : 1000.0;
    } else {
        out->image_rejection_db = 0.0;
    }
    out->clip_fraction = (double)clip / N;
    return ATKDSP_OK;
}
