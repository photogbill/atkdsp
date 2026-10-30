/* ---- 27. The receiver pair and the consistency lattice (§B6c) ---------------
 * docs/ATKDSP_PLAN.md §B6c. The twin is reference.py §27.
 *
 * atkdsp_xspec: the second-order centre of Undertow. Segments of nfft on the
 * global index, Hann, FFT each stream; Sxx, Syy, Sxy averaged over the last
 * navg segments. The magnitude-squared coherence C = |Sxy|^2 / (Sxx Syy) is
 * the test: for independent inputs over navg averages P(C > c) = (1 - c)^
 * (navg - 1), so the per-bin threshold at the scan's Pfa is 1 - (Pfa / nbins)^
 * (1/(navg - 1)). A learned noise-only mean cross-spectrum is subtracted from
 * Sxy first (common-mode spurs and coupled noise). The detected band's TDOA
 * is the slope of the cross phase across it (a coherence-weighted linear fit);
 * the intercept is the LO phase offset, which the slope ignores.
 *
 * atkdsp_align: two free-running streams aligned from a common signal. The
 * cross-correlation peak is the time offset; its drift between the halves is
 * the sample-rate offset; the product at the lag has a tone at the LO offset;
 * its phase per block is the LO wander (VLBI phase referencing).
 *
 * atkdsp_lattice: the independent chip-rate, period and offset estimates must
 * agree (an integer L of a known family; the common-oscillator ppm pair). */
#include "internal.h"
#include <stdio.h>

typedef struct { double re, im; } cd;

double atkdsp_msc_threshold(int navg, double nbins, double pfa) {
    if (navg < 2 || !(nbins >= 1.0) || !(pfa > 0.0) || !(pfa < 1.0)) return 0.0;
    const double per_bin = pfa / nbins;
    double c = 1.0 - pow(per_bin, 1.0 / (double)(navg - 1));
    if (c < 0.0) c = 0.0;
    if (c > 1.0) c = 1.0;
    return c;
}

struct atkdsp_xspec {
    double fs, pfa;
    int nfft, navg, nbins_scan;
    double *w, wsum2;
    atkdsp_fft *fft;
    atkdsp_cf32 *w1, *w2, *X1, *X2;
    double *sxx, *syy;         /* nfft running means                             */
    double *sxy_re, *sxy_im;
    double *base_re, *base_im; /* learned noise cross-spectrum mean              */
    uint64_t base_n;
    int pos; atkdsp_cf32 *b1, *b2;
    uint64_t samples, segments, resets;
};

atkdsp_xspec *atkdsp_xspec_create(double fs, int nfft, int navg, double pfa) {
    if (!(fs > 0.0) || nfft < 16 || nfft > (1 << 16) || navg < 2 || navg > (1 << 16)) return NULL;
    if (!(pfa > 0.0) || !(pfa < 0.5)) return NULL;
    atkdsp_xspec *h = (atkdsp_xspec *)calloc(1, sizeof *h);
    if (!h) return NULL;
    h->fs = fs; h->pfa = pfa; h->nfft = nfft; h->navg = navg; h->nbins_scan = nfft;
    h->fft = atkdsp_fft_create((size_t)nfft);
    h->w = (double *)malloc((size_t)nfft * sizeof *h->w);
    h->w1 = (atkdsp_cf32 *)malloc((size_t)nfft * sizeof *h->w1);
    h->w2 = (atkdsp_cf32 *)malloc((size_t)nfft * sizeof *h->w2);
    h->X1 = (atkdsp_cf32 *)malloc((size_t)nfft * sizeof *h->X1);
    h->X2 = (atkdsp_cf32 *)malloc((size_t)nfft * sizeof *h->X2);
    h->sxx = (double *)calloc((size_t)nfft, sizeof *h->sxx);
    h->syy = (double *)calloc((size_t)nfft, sizeof *h->syy);
    h->sxy_re = (double *)calloc((size_t)nfft, sizeof *h->sxy_re);
    h->sxy_im = (double *)calloc((size_t)nfft, sizeof *h->sxy_im);
    h->base_re = (double *)calloc((size_t)nfft, sizeof *h->base_re);
    h->base_im = (double *)calloc((size_t)nfft, sizeof *h->base_im);
    h->b1 = (atkdsp_cf32 *)malloc((size_t)nfft * sizeof *h->b1);
    h->b2 = (atkdsp_cf32 *)malloc((size_t)nfft * sizeof *h->b2);
    if (!h->fft || !h->w || !h->w1 || !h->w2 || !h->X1 || !h->X2 || !h->sxx || !h->syy ||
        !h->sxy_re || !h->sxy_im || !h->base_re || !h->base_im || !h->b1 || !h->b2) {
        atkdsp_xspec_destroy(h); return NULL;
    }
    double s = 0.0;
    for (int i = 0; i < nfft; i++) { h->w[i] = 0.5 * (1.0 - cos(2.0 * ATK_PI * i / nfft)); s += h->w[i] * h->w[i]; }
    h->wsum2 = s;
    return h;
}

void atkdsp_xspec_destroy(atkdsp_xspec *h) {
    if (!h) return;
    atkdsp_fft_destroy(h->fft);
    free(h->w); free(h->w1); free(h->w2); free(h->X1); free(h->X2);
    free(h->sxx); free(h->syy); free(h->sxy_re); free(h->sxy_im);
    free(h->base_re); free(h->base_im); free(h->b1); free(h->b2);
    free(h);
}

void atkdsp_xspec_reset(atkdsp_xspec *h) {
    if (!h) return;
    h->pos = 0; h->segments = 0; h->resets++;
    memset(h->sxx, 0, (size_t)h->nfft * sizeof *h->sxx);
    memset(h->syy, 0, (size_t)h->nfft * sizeof *h->syy);
    memset(h->sxy_re, 0, (size_t)h->nfft * sizeof *h->sxy_re);
    memset(h->sxy_im, 0, (size_t)h->nfft * sizeof *h->sxy_im);
}

/* exponential running mean with weight 1/min(segments, navg) — the last navg
 * segments' average, approached from empty */
static void xspec_segment(atkdsp_xspec *h, int learn) {
    for (int i = 0; i < h->nfft; i++) {
        h->w1[i].re = (float)(h->b1[i].re * h->w[i]); h->w1[i].im = (float)(h->b1[i].im * h->w[i]);
        h->w2[i].re = (float)(h->b2[i].re * h->w[i]); h->w2[i].im = (float)(h->b2[i].im * h->w[i]);
    }
    atkdsp_fft_exec(h->fft, h->w1, h->X1, 0);
    atkdsp_fft_exec(h->fft, h->w2, h->X2, 0);
    if (learn) {
        const double nw = (double)(h->base_n + 1);
        for (int i = 0; i < h->nfft; i++) {
            const double cr = (double)h->X1[i].re * h->X2[i].re + (double)h->X1[i].im * h->X2[i].im;
            const double ci = (double)h->X1[i].im * h->X2[i].re - (double)h->X1[i].re * h->X2[i].im;
            h->base_re[i] += (cr - h->base_re[i]) / nw;
            h->base_im[i] += (ci - h->base_im[i]) / nw;
        }
        h->base_n++;
        return;
    }
    const double nw = (double)(h->segments + 1 < (uint64_t)h->navg ? h->segments + 1 : (uint64_t)h->navg);
    for (int i = 0; i < h->nfft; i++) {
        const double xx = (double)h->X1[i].re * h->X1[i].re + (double)h->X1[i].im * h->X1[i].im;
        const double yy = (double)h->X2[i].re * h->X2[i].re + (double)h->X2[i].im * h->X2[i].im;
        const double cr = (double)h->X1[i].re * h->X2[i].re + (double)h->X1[i].im * h->X2[i].im;
        const double ci = (double)h->X1[i].im * h->X2[i].re - (double)h->X1[i].re * h->X2[i].im;
        h->sxx[i] += (xx - h->sxx[i]) / nw;
        h->syy[i] += (yy - h->syy[i]) / nw;
        h->sxy_re[i] += (cr - h->sxy_re[i]) / nw;
        h->sxy_im[i] += (ci - h->sxy_im[i]) / nw;
    }
    h->segments++;
}

static int xspec_feed(atkdsp_xspec *h, const atkdsp_cf32 *x1, const atkdsp_cf32 *x2, size_t n, int learn) {
    if (!h || (!x1 && n) || (!x2 && n)) return ATKDSP_E_ARG;
    size_t i = 0;
    while (i < n) {
        size_t take = (size_t)(h->nfft - h->pos);
        if (take > n - i) take = n - i;
        memcpy(h->b1 + h->pos, x1 + i, take * sizeof *x1);
        memcpy(h->b2 + h->pos, x2 + i, take * sizeof *x2);
        h->pos += (int)take; i += take;
        if (h->pos == h->nfft) { xspec_segment(h, learn); h->pos = 0; }
    }
    if (!learn) h->samples += n;
    return ATKDSP_OK;
}

int atkdsp_xspec_process(atkdsp_xspec *h, const atkdsp_cf32 *x1, const atkdsp_cf32 *x2, size_t n) {
    return xspec_feed(h, x1, x2, n, 0);
}
int atkdsp_xspec_baseline(atkdsp_xspec *h, const atkdsp_cf32 *x1, const atkdsp_cf32 *x2, size_t n) {
    return xspec_feed(h, x1, x2, n, 1);
}

static void bin_coh(const atkdsp_xspec *h, int i, double *coh, double *phase) {
    const double cr = h->sxy_re[i] - h->base_re[i], ci = h->sxy_im[i] - h->base_im[i];
    const double num = cr * cr + ci * ci;
    const double den = h->sxx[i] * h->syy[i];
    *coh = den > 0.0 ? num / den : 0.0;
    if (*coh > 1.0) *coh = 1.0;
    if (phase) *phase = atan2(ci, cr);
}

int atkdsp_xspec_read(const atkdsp_xspec *h, atkdsp_xspec_result *out) {
    if (!h || !out) return ATKDSP_E_ARG;
    memset(out, 0, sizeof *out);
    out->navg = (int)(h->segments < (uint64_t)h->navg ? h->segments : (uint64_t)h->navg);
    out->method[0] = ATKDSP_METHOD_MSC; out->method[1] = ATKDSP_METHOD_PHASE_REF;
    if (h->segments < 2) { out->flags |= ATKDSP_XS_WARMUP; return ATKDSP_OK; }
    out->coh_thr = atkdsp_msc_threshold(out->navg, (double)h->nbins_scan, h->pfa);
    int mb = 0; double mc = 0.0;
    for (int i = 0; i < h->nfft; i++) { double c, ph; bin_coh(h, i, &c, &ph); if (c > mc) { mc = c; mb = i; } }
    out->max_coh = mc;
    /* fftshifted bin index for display */
    out->max_bin = (mb + h->nfft / 2) % h->nfft - h->nfft / 2;
    { double l10 = (double)(out->navg - 1) * log10(1.0 - (mc < 1.0 ? mc : 1.0 - 1e-12)) + log10((double)h->nbins_scan);
      out->log10_pfa = l10 > 0.0 ? 0.0 : l10; }
    if (mc >= out->coh_thr) {
        out->flags |= ATKDSP_XS_DETECT;
        /* the band: contiguous bins around mb above threshold */
        int lo = mb, hi = mb;
        while (lo > 0) { double c, ph; bin_coh(h, lo - 1, &c, &ph); if (c < out->coh_thr) break; lo--; }
        while (hi < h->nfft - 1) { double c, ph; bin_coh(h, hi + 1, &c, &ph); if (c < out->coh_thr) break; hi++; }
        out->band_lo = (lo + h->nfft / 2) % h->nfft - h->nfft / 2;
        out->band_hi = (hi + h->nfft / 2) % h->nfft - h->nfft / 2;
        out->band_bins = hi - lo + 1;
        /* coherence-weighted linear fit of unwrapped phase vs bin across [lo, hi] */
        double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, cmean = 0;
        double prev = 0; int first = 1;
        for (int i = lo; i <= hi; i++) {
            double c, ph; bin_coh(h, i, &c, &ph);
            if (!first) { while (ph - prev > ATK_PI) ph -= 2 * ATK_PI; while (ph - prev < -ATK_PI) ph += 2 * ATK_PI; }
            first = 0; prev = ph;
            const double xb = (double)i, wgt = c;
            sw += wgt; sx += wgt * xb; sy += wgt * ph; sxx += wgt * xb * xb; sxy += wgt * xb * ph;
            cmean += c;
        }
        out->band_coh_mean = cmean / out->band_bins;
        const double dd = sw * sxx - sx * sx;
        if (out->band_bins >= 4 && dd != 0.0) {
            const double slope = (sw * sxy - sx * sy) / dd;      /* rad per bin */
            const double icpt = (sy - slope * sx) / sw;
            /* phase slope per bin -> per Hz: df per bin = fs/nfft; tau = -dphase/domega */
            const double df = h->fs / h->nfft;
            out->tdoa_s = -slope / (2.0 * ATK_PI * df);
            out->phase0 = icpt;
            double resid = 0; first = 1; prev = 0;
            for (int i = lo; i <= hi; i++) {
                double c, ph; bin_coh(h, i, &c, &ph);
                if (!first) { while (ph - prev > ATK_PI) ph -= 2 * ATK_PI; while (ph - prev < -ATK_PI) ph += 2 * ATK_PI; }
                first = 0; prev = ph;
                const double e = ph - (icpt + slope * i); resid += e * e;
            }
            out->slope_resid = sqrt(resid / out->band_bins);
            out->tdoa_err_s = out->slope_resid / (2.0 * ATK_PI * df * sqrt((double)out->band_bins));
            out->flags |= ATKDSP_XS_TDOA;
        }
    }
    return ATKDSP_OK;
}

int atkdsp_xspec_bins(const atkdsp_xspec *h, float *coh, float *phase) {
    if (!h) return ATKDSP_E_ARG;
    for (int i = 0; i < h->nfft; i++) {
        double c, ph; bin_coh(h, i, &c, &ph);
        const int j = (i + h->nfft / 2) % h->nfft;
        if (coh) coh[j] = (float)c;
        if (phase) phase[j] = (float)ph;
    }
    return ATKDSP_OK;
}

int atkdsp_xspec_describe(const atkdsp_xspec *h, char *buf, size_t cap) {
    if (!h || !buf || !cap) return ATKDSP_E_ARG;
    int n = snprintf(buf, cap, "xspec: fs %.0f Hz, nfft %d (%.4g Hz/bin), navg %d, Pfa %.3g over the scan; "
                     "%llu segments in, baseline from %llu",
                     h->fs, h->nfft, h->fs / h->nfft, h->navg, h->pfa,
                     (unsigned long long)h->segments, (unsigned long long)h->base_n);
    return n < 0 ? ATKDSP_E_ARG : (n >= (int)cap ? ATKDSP_E_CAP : n);
}

int atkdsp_xspec_stats(const atkdsp_xspec *h, atkdsp_cyclic_counters *st) {
    if (!h || !st) return ATKDSP_E_ARG;
    st->samples = h->samples; st->segments = h->segments; st->resets = h->resets; st->segment_fill = (uint64_t)h->pos;
    return ATKDSP_OK;
}

/* ---- atkdsp_align ------------------------------------------------------------- */
int atkdsp_align_estimate(const atkdsp_cf32 *x1, const atkdsp_cf32 *x2, size_t n, double fs,
                          int max_lag, int block, atkdsp_align *out, float *phase, size_t cap) {
    if (!x1 || !x2 || !out || n < (size_t)(2 * max_lag + 4) || max_lag < 1) return ATKDSP_E_ARG;
    memset(out, 0, sizeof *out);
    out->method[0] = ATKDSP_METHOD_PHASE_REF; out->method[1] = ATKDSP_METHOD_MSC;
    /* cross-correlation over +-max_lag, NON-COHERENT over blocks of ALIGN_BLK:
       a large LO offset puts a phase ramp on x1 conj x2 that would decorrelate
       one long coherent sum (1e-3 cycle/sample wraps many times over a long
       record), so each lag's statistic is sum over blocks of |block sum|. This
       tolerates an offset up to ~1/ALIGN_BLK cycle/sample before the lag peak
       is lost; the offset itself is then read from the product at that lag. */
    const size_t ALIGN_BLK = 256;
    double best = -1.0; int blag = 0;
    double *mag = (double *)malloc((size_t)(2 * max_lag + 1) * sizeof *mag);
    if (!mag) return ATKDSP_E_NOMEM;
    double noise = 0.0;
    for (int lag = -max_lag; lag <= max_lag; lag++) {
        const size_t lo = lag < 0 ? (size_t)(-lag) : 0;
        const size_t hi = lag < 0 ? n : n - (size_t)lag;
        double m = 0.0;
        for (size_t i = lo; i < hi; i += ALIGN_BLK) {
            const size_t end = i + ALIGN_BLK < hi ? i + ALIGN_BLK : hi;
            double sr = 0.0, si = 0.0;
            for (size_t j = i; j < end; j++) {
                const atkdsp_cf32 *a = &x1[j], *b = &x2[j + lag];
                sr += (double)a->re * b->re + (double)a->im * b->im;
                si += (double)a->im * b->re - (double)a->re * b->im;
            }
            m += sqrt(sr * sr + si * si);
        }
        mag[lag + max_lag] = m;
        noise += m;
        if (m > best) { best = m; blag = lag; }
    }
    noise = (noise - best) / (double)(2 * max_lag);
    out->peak_ratio = noise > 0.0 ? best / noise : HUGE_VAL;
    out->snr_db = 10.0 * log10((best / noise > 1 ? best / noise - 1 : 1e-9));
    /* sub-sample peak */
    double offs = 0.0;
    if (blag > -max_lag && blag < max_lag) {
        const double a = mag[blag + max_lag - 1], b = mag[blag + max_lag], c = mag[blag + max_lag + 1];
        const double d = a - 2 * b + c;
        if (d < 0.0) offs = 0.5 * (a - c) / d;
    }
    out->lag = (double)blag + offs;
    out->lag_err = 1.0 / sqrt(out->peak_ratio > 1 ? out->peak_ratio : 1.0);
    free(mag);
    /* rate offset: the lag from each half of the capture (same non-coherent sum) */
    const size_t h2 = n / 2;
    double l1 = 0, l2 = 0;
    for (int half = 0; half < 2; half++) {
        const atkdsp_cf32 *a1 = x1 + (half ? h2 : 0), *a2 = x2 + (half ? h2 : 0);
        const size_t nn = half ? n - h2 : h2;
        double bb = -1; int bl = 0;
        for (int lag = -max_lag; lag <= max_lag; lag++) {
            const size_t lo = lag < 0 ? (size_t)(-lag) : 0;
            const size_t hi = lag < 0 ? nn : nn - (size_t)lag;
            double m = 0.0;
            for (size_t i = lo; i < hi; i += ALIGN_BLK) {
                const size_t end = i + ALIGN_BLK < hi ? i + ALIGN_BLK : hi;
                double sr = 0, si = 0;
                for (size_t j = i; j < end; j++) {
                    const atkdsp_cf32 *a = &a1[j], *b = &a2[j + lag];
                    sr += (double)a->re * b->re + (double)a->im * b->im;
                    si += (double)a->im * b->re - (double)a->re * b->im;
                }
                m += sqrt(sr * sr + si * si);
            }
            if (m > bb) { bb = m; bl = lag; }
        }
        if (half) l2 = bl; else l1 = bl;
    }
    out->rate_ppm = (l2 - l1) / ((double)h2) * 1e6;
    /* LO offset: the product at the peak lag has a tone at the offset */
    const int lag = blag;
    const size_t lo = lag < 0 ? (size_t)(-lag) : 0;
    const size_t hi = lag < 0 ? n : n - (size_t)lag;
    const size_t np = hi - lo;
    cd *prod = (cd *)malloc(np * sizeof *prod);
    if (!prod) return ATKDSP_E_NOMEM;
    for (size_t i = lo; i < hi; i++) {
        const atkdsp_cf32 *a = &x1[i], *b = &x2[i + lag];
        prod[i - lo].re = (double)a->re * b->re + (double)a->im * b->im;
        prod[i - lo].im = (double)a->im * b->re - (double)a->re * b->im;
    }
    /* LO offset: the product carries the LO tone at baseband (|common|^2 is
       real), but the signal x noise cross terms sit at the band's own carrier
       and would swamp a small offset in a raw lag-1 phase. Block-summing the
       product first (same ALIGN_BLK) is a low-pass + decimate that kills the
       carrier and the white noise×noise term, leaving the baseband tone; its
       lag-1 phase over the block sums is 2 pi f_off BLK / fs. */
    const size_t nb = np / ALIGN_BLK;
    double dr = 0, di = 0;
    if (nb >= 2) {
        cd *blk = (cd *)calloc(nb, sizeof *blk);
        if (blk) {
            for (size_t m = 0; m < nb; m++) {
                double sr = 0, si = 0;
                for (size_t j = 0; j < ALIGN_BLK; j++) { sr += prod[m * ALIGN_BLK + j].re; si += prod[m * ALIGN_BLK + j].im; }
                blk[m].re = sr; blk[m].im = si;
            }
            for (size_t m = 1; m < nb; m++) {
                dr += blk[m].re * blk[m - 1].re + blk[m].im * blk[m - 1].im;
                di += blk[m].im * blk[m - 1].re - blk[m].re * blk[m - 1].im;
            }
            free(blk);
        }
    }
    const double dphi = atan2(di, dr);
    out->lo_offset_hz = -dphi * fs / (2.0 * ATK_PI * (double)ALIGN_BLK);   /* +: x2 LO above x1 */
    /* LO wander: within each block of `block` samples, the product's mean phase
       after the constant LO offset is removed — this is what atkdsp_align_apply
       subtracts to phase-reference the pair (VLBI on a calibrator). One value per
       block over the WHOLE stream (indexed i/block), not just [lo,hi). */
    if (phase && block > 0 && cap > 0) {
        const size_t nb = (n + (size_t)block - 1) / (size_t)block;
        for (size_t b = 0; b < nb && b < cap; b++) {
            double sr = 0.0, si = 0.0;
            const size_t i0 = b * (size_t)block, i1 = i0 + (size_t)block < n ? i0 + (size_t)block : n;
            for (size_t i = i0; i < i1; i++) {
                /* the product at the found integer lag, offset-derotated */
                const long j = (long)i + lag;
                if (j < 0 || (size_t)j >= n) continue;
                const double pr = (double)x1[i].re * x2[j].re + (double)x1[i].im * x2[j].im;
                const double pi = (double)x1[i].im * x2[j].re - (double)x1[i].re * x2[j].im;
                const double ph = -2.0 * ATK_PI * out->lo_offset_hz * (double)i / fs;
                const double cr = cos(ph), ci = sin(ph);
                sr += pr * cr - pi * ci; si += pr * ci + pi * cr;
            }
            phase[b] = (float)atan2(si, sr);
        }
        out->nblocks = (int)(nb < cap ? nb : cap);
    }
    free(prod);
    out->flags |= ATKDSP_ALIGN_FOUND;
    return ATKDSP_OK;
}

int atkdsp_align_apply(const atkdsp_cf32 *x2, size_t n, double fs, const atkdsp_align *a,
                       const float *phase, int block, atkdsp_cf32 *out) {
    if (!x2 || !a || !out || !(fs > 0.0)) return ATKDSP_E_ARG;
    const double rate = 1.0 + a->rate_ppm * 1e-6;
    for (size_t i = 0; i < n; i++) {
        /* source position on x2's axis: shift by lag, stretch by rate */
        const double sp = ((double)i) * rate + a->lag;
        const long k = (long)floor(sp);
        const double f = sp - k;
        atkdsp_cf32 v = {0.0f, 0.0f};
        if (k >= 0 && (size_t)k + 1 < n) {
            v.re = (float)((1 - f) * x2[k].re + f * x2[k + 1].re);
            v.im = (float)((1 - f) * x2[k].im + f * x2[k + 1].im);
        }
        /* derotate the LO offset and the tracked wander */
        double ph = -2.0 * ATK_PI * a->lo_offset_hz * (double)i / fs;
        if (phase && block > 0) ph -= phase[i / (size_t)block];
        const double cr = cos(ph), ci = sin(ph);
        out[i].re = (float)(v.re * cr - v.im * ci);
        out[i].im = (float)(v.re * ci + v.im * cr);
    }
    return ATKDSP_OK;
}

/* ---- atkdsp_lattice ----------------------------------------------------------- */
static int known_family(int L, int *fam) {
    int r;
    for (r = 2; r <= 20; r++) if (L == (1 << r) - 1) { *fam = 1; return 1; }   /* m-seq / Gold */
    if (L == 7 || L == 11 || L == 13) { *fam = 2; return 1; }                   /* Barker */
    if (L == 32) { *fam = 3; return 1; }                                        /* 802.15.4 */
    if (L == 1023 || L == 511 || L == 4092 || L == 4096 || L == 10230) { *fam = 4; return 1; }
    if (L == 32768 || L == 38400) { *fam = 5; return 1; }
    *fam = 0; return 0;
}

int atkdsp_lattice(const atkdsp_lattice_in *in, atkdsp_lattice_out *out) {
    if (!in || !out) return ATKDSP_E_ARG;
    memset(out, 0, sizeof *out);
    out->method = ATKDSP_METHOD_LATTICE;
    const double tol = in->rc_tol > 0.0 ? in->rc_tol : 0.02;
    const double ppm_tol = in->ppm_tol > 0.0 ? in->ppm_tol : 2.0;
    /* agreed chip rate: the mean of the available estimates */
    double sum = 0.0; int k = 0; double mn = HUGE_VAL, mx = -HUGE_VAL;
    const double est[3] = {in->rc_acf_hz, in->rc_line_hz, in->rc_nominal_hz};
    for (int i = 0; i < 3; i++) if (est[i] > 0.0) { sum += est[i]; k++; if (est[i] < mn) mn = est[i]; if (est[i] > mx) mx = est[i]; }
    if (k == 0) return ATKDSP_E_EMPTY;
    out->rc_hz = sum / k;
    out->rc_spread = out->rc_hz > 0.0 ? (mx - mn) / out->rc_hz : 0.0;
    if (k < 2 || out->rc_spread <= tol) out->flags |= ATKDSP_LAT_AGREE_RC;
    /* integer L */
    if (in->period_s > 0.0 && out->rc_hz > 0.0) {
        out->L = in->period_s * out->rc_hz;
        out->L_int = (int)floor(out->L + 0.5);
        out->L_frac = fabs(out->L - out->L_int);
        if (out->L_frac <= 0.02 * (out->L_int > 1 ? 1.0 : 1.0) && out->L_int >= 2) {
            out->flags |= ATKDSP_LAT_INTEGER_L;
            int fam; if (known_family(out->L_int, &fam)) { out->family = fam; out->flags |= ATKDSP_LAT_FAMILY; }
        }
    }
    /* common oscillator: cfo/fc vs drc/rc */
    if (in->fc_hz > 0.0 && out->rc_hz > 0.0) {
        out->ppm_carrier = in->cfo_hz / in->fc_hz * 1e6;
        /* chip ppm from the spread of estimates against the nominal, if given */
        if (in->rc_nominal_hz > 0.0)
            out->ppm_chip = (out->rc_hz - in->rc_nominal_hz) / in->rc_nominal_hz * 1e6;
        out->ppm_diff = fabs(out->ppm_carrier - out->ppm_chip);
        if (out->ppm_diff <= ppm_tol + in->rx_ppm) out->flags |= ATKDSP_LAT_ONE_CLOCK;
    }
    /* the verdict: chip rates agree AND an integer L (family is a bonus; one-clock
       only when a carrier was given) */
    if ((out->flags & ATKDSP_LAT_AGREE_RC) && (out->flags & ATKDSP_LAT_INTEGER_L))
        out->flags |= ATKDSP_LAT_CONSISTENT;
    return ATKDSP_OK;
}
