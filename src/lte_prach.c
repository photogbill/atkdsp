/* 11c. Passive PRACH handset-presence detector.
 *
 * A powered LTE handset transmits a PRACH preamble when it accesses a cell
 * (attach, tracking-area update, re-establishment). The preamble is a
 * Zadoff-Chu sequence. Detecting it PASSIVELY says a handset is alive and
 * transmitting here — presence, and with direction finding, location — which
 * is the actionable signal for search and rescue after a disaster. It
 * carries NO subscriber identity and this extracts none: it reports that an
 * access happened, which root sequence, and how many concurrent accesses,
 * never who.
 *
 * THE METHOD (root-agnostic, one FFT). After the sequence window is taken,
 * its DFT is the frequency-domain ZC of the transmitted root times the
 * channel:  Y[k] = H . exp(-j pi u k(k+1)/N).  The adjacent product
 * D[k] = Y[k+1] conj(Y[k]) = |H|^2 . exp(-j2pi u k/N) . const  is a PURE
 * TONE at frequency -u/N. One FFT of D shows a peak per concurrent root, at
 * bin N-u, with no bank of 838 correlators. A cyclic shift of a root adds
 * only a constant phase to D, so shifts of one root share a tone; to count
 * concurrent accesses on a detected root, one correlation with that root
 * gives the power-delay profile, whose peaks are the individual preambles.
 *
 * 36.211 sec 5.7. N_ZC = 839 (preamble formats 0-3, FDD). The numpy twin in
 * atkdsp.reference (prach_*) is the spec; tests/test_cross_check.py holds
 * this to it. Measured: 100% detection to -5 dB SNR, 0 false alarms in
 * 10000 noise windows, exact concurrent-access counts to 3. */
#include "internal.h"

#define NZC 839
#define PRACH_MIN_METRIC 40.0
#define PRACH_COUNT_FRAC 0.30

struct atkdsp_prach {
    atkdsp_fft  *fft;                 /* 839-point plan */
    atkdsp_cf32 *Y, *D, *SF, *tmp, *pdp, *zc;
    double      *S, *Ssort;
    int         *claimed;
};

atkdsp_prach *atkdsp_prach_create(void) {
    atkdsp_prach *h = (atkdsp_prach *)calloc(1, sizeof *h);
    if (!h) return NULL;
    h->fft = atkdsp_fft_create(NZC);
    h->Y   = (atkdsp_cf32 *)atk_aligned_malloc(NZC * sizeof(atkdsp_cf32));
    h->D   = (atkdsp_cf32 *)atk_aligned_malloc(NZC * sizeof(atkdsp_cf32));
    h->SF  = (atkdsp_cf32 *)atk_aligned_malloc(NZC * sizeof(atkdsp_cf32));
    h->tmp = (atkdsp_cf32 *)atk_aligned_malloc(NZC * sizeof(atkdsp_cf32));
    h->pdp = (atkdsp_cf32 *)atk_aligned_malloc(NZC * sizeof(atkdsp_cf32));
    h->zc  = (atkdsp_cf32 *)atk_aligned_malloc(NZC * sizeof(atkdsp_cf32));
    h->S   = (double *)atk_aligned_malloc(NZC * sizeof(double));
    h->Ssort = (double *)atk_aligned_malloc(NZC * sizeof(double));
    h->claimed = (int *)atk_aligned_malloc(NZC * sizeof(int));
    if (!h->fft || !h->Y || !h->D || !h->SF || !h->tmp || !h->pdp ||
        !h->zc || !h->S || !h->Ssort || !h->claimed) {
        atkdsp_prach_destroy(h); return NULL;
    }
    return h;
}

void atkdsp_prach_destroy(atkdsp_prach *h) {
    if (!h) return;
    if (h->fft) atkdsp_fft_destroy(h->fft);
    atk_aligned_free(h->Y); atk_aligned_free(h->D); atk_aligned_free(h->SF);
    atk_aligned_free(h->tmp); atk_aligned_free(h->pdp); atk_aligned_free(h->zc);
    atk_aligned_free(h->S); atk_aligned_free(h->Ssort); atk_aligned_free(h->claimed);
    free(h);
}

/* x_u(k) = exp(-j pi u k(k+1)/N), 36.211 5.7.2 (frequency domain). */
int atkdsp_prach_zc(int u, atkdsp_cf32 *out) {
    int k;
    if (!out || u < 0 || u >= NZC) return ATKDSP_E_ARG;
    for (k = 0; k < NZC; ++k) {
        const double a = -ATK_PI * (double)u * (double)k * (double)(k + 1) / NZC;
        out[k].re = (float)cos(a);
        out[k].im = (float)sin(a);
    }
    return ATKDSP_OK;
}

static int dcmp(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

/* Detect the preambles in ONE aligned window (NZC samples). */
static int detect_once(atkdsp_prach *h, const atkdsp_cf32 *seq,
                       double min_metric, atkdsp_prach_hit *out, size_t cap) {
    int k, b, nhits = 0;
    double med;
    if (atkdsp_fft_exec(h->fft, seq, h->Y, 0) != ATKDSP_OK) return ATKDSP_E_FFT;
    for (k = 0; k < NZC - 1; ++k) {
        const atkdsp_cf32 a = h->Y[k], bb = h->Y[k + 1];
        h->D[k].re = bb.re * a.re + bb.im * a.im;      /* b * conj(a) */
        h->D[k].im = bb.im * a.re - bb.re * a.im;
    }
    h->D[NZC - 1].re = h->D[NZC - 1].im = 0.0f;
    if (atkdsp_fft_exec(h->fft, h->D, h->SF, 0) != ATKDSP_OK) return ATKDSP_E_FFT;
    for (b = 0; b < NZC; ++b) {
        h->S[b] = (double)h->SF[b].re * h->SF[b].re
                + (double)h->SF[b].im * h->SF[b].im;
        h->Ssort[b] = h->S[b];
        h->claimed[b] = 0;
    }
    qsort(h->Ssort, NZC, sizeof(double), dcmp);
    med = h->Ssort[NZC / 2] + 1e-30;

    while ((size_t)nhits < cap) {
        int best = -1, d, u;
        double bestv = 0.0, top, thr;
        for (b = 0; b < NZC; ++b)
            if (!h->claimed[b] && h->S[b] > bestv) { bestv = h->S[b]; best = b; }
        if (best < 0 || bestv / med < (double)min_metric) break;
        for (d = -2; d <= 2; ++d) h->claimed[(best + d + NZC) % NZC] = 1;
        u = (NZC - best) % NZC;
        if (u == 0) continue;
        /* per-root power-delay profile: correlate with root u, IFFT */
        atkdsp_prach_zc(u, h->zc);
        for (k = 0; k < NZC; ++k) {
            const atkdsp_cf32 y = h->Y[k], z = h->zc[k];
            h->tmp[k].re = y.re * z.re + y.im * z.im;   /* y * conj(z) */
            h->tmp[k].im = y.im * z.re - y.re * z.im;
        }
        if (atkdsp_fft_exec(h->fft, h->tmp, h->pdp, 1) != ATKDSP_OK)
            return ATKDSP_E_FFT;
        top = 0.0;
        for (k = 0; k < NZC; ++k) {
            const double p = (double)h->pdp[k].re * h->pdp[k].re
                           + (double)h->pdp[k].im * h->pdp[k].im;
            if (p > top) top = p;
        }
        thr = PRACH_COUNT_FRAC * top;
        {
            int count = 0, delay = 0; double dtop = -1.0;
            for (k = 0; k < NZC; ++k) {
                const double p = (double)h->pdp[k].re * h->pdp[k].re
                               + (double)h->pdp[k].im * h->pdp[k].im;
                const double pm = (double)h->pdp[(k-1+NZC)%NZC].re*h->pdp[(k-1+NZC)%NZC].re
                                + (double)h->pdp[(k-1+NZC)%NZC].im*h->pdp[(k-1+NZC)%NZC].im;
                const double pn = (double)h->pdp[(k+1)%NZC].re*h->pdp[(k+1)%NZC].re
                                + (double)h->pdp[(k+1)%NZC].im*h->pdp[(k+1)%NZC].im;
                if (p >= thr && p >= pm && p > pn) {
                    ++count;
                    if (p > dtop) { dtop = p; delay = k; }
                }
            }
            out[nhits].root = u;
            out[nhits].count = count;
            out[nhits].metric = (float)(bestv / med);
            out[nhits].delay = delay;
            ++nhits;
        }
    }
    return nhits;
}

ptrdiff_t atkdsp_prach_detect(atkdsp_prach *h, const atkdsp_cf32 *seq, size_t n,
                              float min_metric, atkdsp_prach_hit *out,
                              size_t cap) {
    if (!h || !seq || !out || cap == 0) return ATKDSP_E_ARG;
    if (n < (size_t)NZC) return 0;
    if (min_metric <= 0.0f) min_metric = (float)PRACH_MIN_METRIC;
    return detect_once(h, seq, (double)min_metric, out, cap);
}

/* Slide an NZC window across a longer capture (step `win_step`), detect in
 * each, and merge by root: one entry per root, keeping the strongest metric
 * and the largest concurrent-access count seen. A real uplink capture is not
 * PRACH-occasion aligned, but the cyclic prefix means any window starting
 * within the CP is a clean cyclic rotation, so a step at or below the CP
 * length always lands one clean window on a preamble. */
ptrdiff_t atkdsp_prach_scan(atkdsp_prach *h, const atkdsp_cf32 *stream,
                            size_t n, int win_step, float min_metric,
                            atkdsp_prach_hit *out, size_t cap) {
    atkdsp_prach_hit tmp[32];
    size_t start;
    int merged = 0, i, j;
    if (!h || !stream || !out || cap == 0) return ATKDSP_E_ARG;
    if (n < (size_t)NZC) return 0;
    if (win_step < 1) win_step = 64;
    if (min_metric <= 0.0f) min_metric = (float)PRACH_MIN_METRIC;
    for (start = 0; start + (size_t)NZC <= n; start += (size_t)win_step) {
        int got = detect_once(h, stream + start, (double)min_metric,
                              tmp, sizeof tmp / sizeof tmp[0]);
        if (got < 0) return got;
        for (i = 0; i < got; ++i) {
            int found = -1;
            for (j = 0; j < merged; ++j)
                if (out[j].root == tmp[i].root) { found = j; break; }
            if (found < 0) {
                if ((size_t)merged >= cap) continue;
                out[merged++] = tmp[i];
            } else {
                if (tmp[i].metric > out[found].metric) {
                    out[found].metric = tmp[i].metric;
                    out[found].delay = tmp[i].delay;
                }
                if (tmp[i].count > out[found].count)
                    out[found].count = tmp[i].count;
            }
        }
    }
    return merged;
}
