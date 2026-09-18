/* 2. NCO — phase-continuous complex mixer. */
#include "internal.h"

/* Renormalise the rotating phasors this often, IN SAMPLES. Float error per
 * multiply is ~1e-7; 256 steps keeps |z| within 3e-5 of 1, and the block
 * phase is re-derived from the double accumulator at every call anyway. */
#define RENORM_EVERY 256

/* How many phasors advance at once.
 *
 * WHY THIS IS NOT ONE. A single rotating phasor (z *= d, once per sample) is
 * a dependency chain: every sample waits on the previous multiply to retire,
 * so the loop cannot use a vector unit however wide it is, and it cannot be
 * unrolled into anything faster either. Measured on a two-core SSE2 box at
 * 40 MSPS on 2026-09-18: 3.24 ns/sample, which was 65 % of an ENTIRE DDC —
 * more than the three decimation stages and the resampler put together.
 *
 * LANES phasors, each stepping by LANES*step, are LANES independent chains
 * covering the same samples, which is exactly the shape a vector unit wants.
 * The per-phasor error also falls, because each one takes n/LANES steps
 * rather than n between renormalisations.
 *
 * 8 fills a 256-bit unit in float for the sample multiply and takes two
 * registers in double for the rotation. It must divide RENORM_EVERY. */
#define LANES 8

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
    const double ph0 = nco->phase;

    /* The phasors and their rotation stay in DOUBLE: a float step rounds to
     * ~3e-8 rad, and after a few hundred samples that is a phase error
     * visible against the reference. The sample multiply is float — the
     * output is float — so only the rotation carries the wider type. */
    double zr[LANES], zi[LANES];
    int l;
    for (l = 0; l < LANES; ++l) {
        zr[l] = cos(ph0 + step * (double)l);
        zi[l] = sin(ph0 + step * (double)l);
    }
    const double dr = cos(step * (double)LANES);
    const double di = sin(step * (double)LANES);

    size_t i = 0;
    const size_t vec_end = n - (n % LANES);
    while (i < vec_end) {
        size_t stop = i + RENORM_EVERY;          /* both are multiples of LANES */
        if (stop > vec_end) stop = vec_end;
        for (; i < stop; i += LANES) {
            for (l = 0; l < LANES; ++l) {
                const float zrf = (float)zr[l], zif = (float)zi[l];
                const float xr = in[i + l].re, xi = in[i + l].im;
                out[i + l].re = xr * zrf - xi * zif;
                out[i + l].im = xr * zif + xi * zrf;
            }
            for (l = 0; l < LANES; ++l) {
                const double nr = zr[l] * dr - zi[l] * di;
                const double ni = zr[l] * di + zi[l] * dr;
                zr[l] = nr; zi[l] = ni;
            }
        }
        for (l = 0; l < LANES; ++l) {
            const double mag = sqrt(zr[l] * zr[l] + zi[l] * zi[l]);
            if (mag > 0.0) { zr[l] /= mag; zi[l] /= mag; }
        }
    }

    /* Fewer than LANES samples left. Taken straight from the phase rather
     * than by continuing a recurrence: it is at most seven cos/sin pairs, and
     * it is exact, so a stream delivered in awkward block sizes cannot drift
     * away from one delivered whole. */
    for (; i < n; ++i) {
        const double ph = ph0 + step * (double)i;
        const double c = cos(ph), s = sin(ph);
        const float xr = in[i].re, xi = in[i].im;
        out[i].re = (float)(xr * c - xi * s);
        out[i].im = (float)(xr * s + xi * c);
    }

    nco->phase = atk_wrap(ph0 + step * (double)n);
}
