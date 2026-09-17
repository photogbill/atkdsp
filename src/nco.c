/* 2. NCO — phase-continuous complex mixer. */
#include "internal.h"

/* Renormalise the rotating phasor this often. Float error per multiply is
 * ~1e-7; 256 steps keeps |z| within 3e-5 of 1, and the block phase is
 * re-derived from the double accumulator at every call anyway. */
#define RENORM_EVERY 256

void atkdsp_nco_init(atkdsp_nco *nco, double freq_hz, double sample_rate) {
    if (!nco) return;
    nco->phase = 0.0;
    atkdsp_nco_set_freq(nco, freq_hz, sample_rate);
}

void atkdsp_nco_set_freq(atkdsp_nco *nco, double freq_hz, double sample_rate) {
    if (!nco) return;
    /* -2*pi*f/fs: a channel at +f mixes DOWN to DC (atk.core.dsp.frequency_shift) */
    nco->step = sample_rate > 0.0 ? -2.0 * ATK_PI * freq_hz / sample_rate : 0.0;
}

void atkdsp_nco_mix(atkdsp_nco *nco, const atkdsp_cf32 *in, atkdsp_cf32 *out, size_t n) {
    if (!nco || !in || !out) return;
    if (nco->step == 0.0) {
        if (out != in) memcpy(out, in, n * sizeof(atkdsp_cf32));
        return;
    }
    const double step = nco->step;
    /* The phasor and its rotation are kept in DOUBLE: a float step rounds to
     * ~3e-8 rad and after a few hundred samples that is a visible phase error
     * against the reference. The x*z multiply is float in, double z, float
     * out — still one complex multiply per sample. */
    double ph = nco->phase;
    double zr = cos(ph), zi = sin(ph);
    const double dr = cos(step), di = sin(step);

    size_t i = 0;
    while (i < n) {
        size_t stop = i + RENORM_EVERY;
        if (stop > n) stop = n;
        for (; i < stop; ++i) {
            const double xr = in[i].re, xi = in[i].im;
            out[i].re = (float)(xr * zr - xi * zi);
            out[i].im = (float)(xr * zi + xi * zr);
            const double nr = zr * dr - zi * di;
            const double ni = zr * di + zi * dr;
            zr = nr; zi = ni;
        }
        const double mag = sqrt(zr * zr + zi * zi);
        if (mag > 0.0) { zr /= mag; zi /= mag; }
    }
    nco->phase = atk_wrap(ph + step * (double)n);
}
