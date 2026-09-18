/* 11. LTE cell search — PSS correlation, SSS decode, PCI.
 *
 * Everything here works at 1.92 MSPS with a 128-point FFT, which is LTE's
 * own smallest numerology (15 kHz subcarriers). A cell's synchronisation
 * signals live in the middle 62 subcarriers whatever its channel bandwidth,
 * so this slice finds a 20 MHz cell as readily as a 1.4 MHz one.
 *
 * The correlation is done with ONE FFT of the whole buffer rather than a
 * sliding dot product: three references at 128 taps is 384 complex
 * multiply-accumulates per input sample, and at 1.92 MSPS that is a
 * gigaflop of work to answer a question asked once a second. Transforming
 * the buffer once, multiplying by each conjugated reference and inverting is
 * four transforms for the whole scan.
 *
 * Sequence definitions: 3GPP TS 36.211 §6.11.1 (PSS) and §6.11.2 (SSS). The
 * numpy twin in atkdsp.reference is the readable statement of both, and
 * tests/test_cross_check.py holds this to it. */
#include "internal.h"

#define NSYM     ATKDSP_LTE_SYM          /* 128                          */
#define NCARR    62                      /* subcarriers the sync signals use */
#define NSSS     336                     /* 168 cells x 2 subframes      */
/* Within a slot (normal cyclic prefix) the SSS is the symbol before the
 * PSS, so its data starts exactly one symbol (9 CP + 128) earlier. */
#define SSS_BACK 137

static const int PSS_ROOTS[3] = { 25, 29, 34 };

/* struct atkdsp_lte now lives in internal.h (shared with lte_pbch.c). */

/* ---- sequences --------------------------------------------------------- */

/* d_u(n), 36.211 6.11.1.1 — the DC gap is why the exponent jumps at n=31. */
static void pss_values(int nid2, atkdsp_cf32 *out /* NCARR */) {
    const double u = (double)PSS_ROOTS[nid2];
    int n;
    for (n = 0; n < NCARR; ++n) {
        const double k = (n < 31) ? (double)n * (n + 1)
                                  : (double)(n + 1) * (n + 2);
        const double a = -ATK_PI * u * k / 63.0;
        out[n].re = (float)cos(a);
        out[n].im = (float)sin(a);
    }
}

/* x(i+5) = sum of the tapped bits, mod 2; init 0,0,0,0,1; s(i) = 1-2x(i) */
static void mseq(const int *taps, int ntaps, signed char *out /* 31 */) {
    int x[31], i, j;
    for (i = 0; i < 5; ++i) x[i] = (i == 4) ? 1 : 0;
    for (i = 0; i + 5 < 31; ++i) {
        int acc = 0;
        for (j = 0; j < ntaps; ++j) acc += x[i + taps[j]];
        x[i + 5] = acc & 1;
    }
    for (i = 0; i < 31; ++i) out[i] = (signed char)(1 - 2 * x[i]);
}

int atkdsp_lte_sss_symbol(int nid1, int nid2, int subframe, float *out62) {
    static const int TS[2] = { 2, 0 }, TC[2] = { 3, 0 }, TZ[4] = { 4, 2, 1, 0 };
    signed char S[31], C[31], Z[31];
    int n;
    if (!out62 || nid1 < 0 || nid1 > 167 || nid2 < 0 || nid2 > 2) return ATKDSP_E_ARG;
    if (subframe != 0 && subframe != 5) return ATKDSP_E_ARG;
    mseq(TS, 2, S); mseq(TC, 2, C); mseq(TZ, 4, Z);
    {
        const int qp = nid1 / 30;
        const int q  = (nid1 + qp * (qp + 1) / 2) / 30;
        const int mp = nid1 + q * (q + 1) / 2;
        const int m0 = mp % 31;
        const int m1 = (m0 + mp / 31 + 1) % 31;
        for (n = 0; n < 31; ++n) {
            const int s0 = S[(n + m0) % 31], s1 = S[(n + m1) % 31];
            const int c0 = C[(n + nid2) % 31], c1 = C[(n + nid2 + 3) % 31];
            const int z0 = Z[(n + (m0 % 8)) % 31], z1 = Z[(n + (m1 % 8)) % 31];
            if (subframe == 0) {
                out62[2 * n]     = (float)(s0 * c0);
                out62[2 * n + 1] = (float)(s1 * c1 * z0);
            } else {
                out62[2 * n]     = (float)(s1 * c0);
                out62[2 * n + 1] = (float)(s0 * c1 * z1);
            }
        }
    }
    return ATKDSP_OK;
}

/* map 62 values onto the 128-point grid (-31..-1, DC skipped, +1..+31) and
 * take them to the time domain */
static int to_symbol(atkdsp_fft *plan, const atkdsp_cf32 *carr, atkdsp_cf32 *out) {
    atkdsp_cf32 grid[NSYM];
    int i, rc;
    memset(grid, 0, sizeof grid);
    for (i = 0; i < 31; ++i) grid[NSYM - 31 + i] = carr[i];
    for (i = 0; i < 31; ++i) grid[1 + i] = carr[31 + i];
    rc = atkdsp_fft_exec(plan, grid, out, 1);              /* inverse, /n */
    if (rc) return rc;
    {   /* the twin's scale: ifft(X) * n / sqrt(62) */
        const float k = (float)NSYM / (float)sqrt((double)NCARR);
        for (i = 0; i < NSYM; ++i) { out[i].re *= k; out[i].im *= k; }
    }
    return ATKDSP_OK;
}

int atkdsp_lte_pss_symbol(int nid2, atkdsp_cf32 *out) {
    atkdsp_cf32 carr[NCARR];
    atkdsp_fft *plan;
    int rc;
    if (!out || nid2 < 0 || nid2 > 2) return ATKDSP_E_ARG;
    plan = atkdsp_fft_create(NSYM);
    if (!plan) return ATKDSP_E_NOMEM;
    pss_values(nid2, carr);
    rc = to_symbol(plan, carr, out);
    atkdsp_fft_destroy(plan);
    return rc;
}

/* ---- the handle -------------------------------------------------------- */

atkdsp_lte *atkdsp_lte_create(void) {
    atkdsp_lte *h = (atkdsp_lte *)calloc(1, sizeof *h);
    int u, i, nid1, sf, slot;
    if (!h) return NULL;
    h->sym = atkdsp_fft_create(NSYM);
    h->pss = (atkdsp_cf32 *)atk_aligned_malloc(3 * NSYM * sizeof(atkdsp_cf32));
    h->pss_freq = (atkdsp_cf32 *)atk_aligned_malloc(3 * NCARR * sizeof(atkdsp_cf32));
    /* ONE TABLE PER N_ID^(2). The SSS scrambling sequences c0 and c1 are
     * cyclic shifts BY N_ID^(2) (36.211 6.11.2.1), so a table built for one
     * of them is wrong for the other two — which decodes a confident and
     * completely incorrect PCI rather than failing. 3 * 336 * 62 floats is
     * 250 kB, built once. */
    h->sss = (float *)atk_aligned_malloc((size_t)3 * NSSS * NCARR * sizeof(float));
    h->vit_bp = (int32_t *)atk_aligned_malloc(
        (size_t)ATK_LTE_VIT_LAPS * 40 * 64 * sizeof(int32_t));
    if (!h->sym || !h->pss || !h->pss_freq || !h->sss || !h->vit_bp) {
        atkdsp_lte_destroy(h); return NULL;
    }
    for (u = 0; u < 3; ++u) {
        pss_values(u, h->pss_freq + (size_t)u * NCARR);
        if (to_symbol(h->sym, h->pss_freq + (size_t)u * NCARR,
                      h->pss + (size_t)u * NSYM) != ATKDSP_OK) {
            atkdsp_lte_destroy(h); return NULL;
        }
        h->pss_energy[u] = 0.0f;
        for (i = 0; i < NSYM; ++i) {
            const atkdsp_cf32 v = h->pss[(size_t)u * NSYM + i];
            h->pss_energy[u] += v.re * v.re + v.im * v.im;
        }
    }
    /* Every SSS hypothesis, once. 336 x 62 floats is 83 kB and turns the
     * decode into a flat dot product instead of 336 sequence generations. */
    slot = 0;
    for (u = 0; u < 3; ++u)
        for (nid1 = 0; nid1 < 168; ++nid1)
            for (sf = 0; sf < 2; ++sf)
                atkdsp_lte_sss_symbol(nid1, u, sf ? 5 : 0,
                                      h->sss + (size_t)(slot++) * NCARR);
    return h;
}

void atkdsp_lte_destroy(atkdsp_lte *h) {
    if (!h) return;
    if (h->sym) atkdsp_fft_destroy(h->sym);
    if (h->scan) atkdsp_fft_destroy(h->scan);
    atk_aligned_free(h->pss);
    atk_aligned_free(h->pss_freq);
    atk_aligned_free(h->sss);
    atk_aligned_free(h->X);
    atk_aligned_free(h->P);
    atk_aligned_free(h->Z);
    atk_aligned_free(h->energy);
    atk_aligned_free(h->vit_bp);
    free(h);
}

static int ensure_scan(atkdsp_lte *h, size_t n) {
    if (h->scan && h->scan_n == n) return ATKDSP_OK;
    if (h->scan) atkdsp_fft_destroy(h->scan);
    atk_aligned_free(h->X); atk_aligned_free(h->P);
    atk_aligned_free(h->Z); atk_aligned_free(h->energy);
    h->scan = atkdsp_fft_create(n);
    h->X = (atkdsp_cf32 *)atk_aligned_malloc(n * sizeof(atkdsp_cf32));
    h->P = (atkdsp_cf32 *)atk_aligned_malloc(n * sizeof(atkdsp_cf32));
    h->Z = (atkdsp_cf32 *)atk_aligned_malloc(n * sizeof(atkdsp_cf32));
    h->energy = (double *)atk_aligned_malloc(n * sizeof(double));
    h->scan_n = n;
    if (!h->scan || !h->X || !h->P || !h->Z || !h->energy) {
        h->scan_n = 0; return ATKDSP_E_NOMEM;
    }
    return ATKDSP_OK;
}

/* ---- detection --------------------------------------------------------- */

ptrdiff_t atkdsp_lte_detect(atkdsp_lte *h, const atkdsp_cf32 *in, size_t n,
                            float min_metric, atkdsp_lte_cell *out, size_t cap) {
    size_t i, k, valid;
    int u, best_u = -1;
    double best_m = 0.0;
    size_t best_k = 0;
    if (!h || !in || !out || cap == 0) return ATKDSP_E_ARG;
    if (n < 2 * (size_t)ATKDSP_LTE_PSS_PERIOD) return 0;
    if (ensure_scan(h, n) != ATKDSP_OK) return ATKDSP_E_NOMEM;
    if (min_metric <= 0.0f) min_metric = 0.06f;
    valid = n - NSYM;                     /* beyond this the circular corr wraps */

    /* running energy of the 128-sample window starting at k */
    {
        double acc = 0.0;
        for (i = 0; i < (size_t)NSYM && i < n; ++i)
            acc += (double)in[i].re * in[i].re + (double)in[i].im * in[i].im;
        h->energy[0] = acc;
        for (k = 1; k <= valid; ++k) {
            const atkdsp_cf32 a = in[k - 1], b = in[k + NSYM - 1];
            acc += (double)b.re * b.re + (double)b.im * b.im;
            acc -= (double)a.re * a.re + (double)a.im * a.im;
            h->energy[k] = acc;
        }
    }
    if (atkdsp_fft_exec(h->scan, in, h->X, 0) != ATKDSP_OK) return ATKDSP_E_FFT;

    for (u = 0; u < 3; ++u) {
        const atkdsp_cf32 *ref = h->pss + (size_t)u * NSYM;
        double top = 0.0; size_t top_k = 0;
        memset(h->P, 0, n * sizeof(atkdsp_cf32));
        memcpy(h->P, ref, NSYM * sizeof(atkdsp_cf32));
        if (atkdsp_fft_exec(h->scan, h->P, h->Z, 0) != ATKDSP_OK) return ATKDSP_E_FFT;
        /* X * conj(P): IFFT of that is sum_j x[k+j] conj(p[j]) */
        for (i = 0; i < n; ++i) {
            const float xr = h->X[i].re, xi = h->X[i].im;
            const float pr = h->Z[i].re, pi = h->Z[i].im;
            h->P[i].re = xr * pr + xi * pi;
            h->P[i].im = xi * pr - xr * pi;
        }
        if (atkdsp_fft_exec(h->scan, h->P, h->Z, 1) != ATKDSP_OK) return ATKDSP_E_FFT;
        for (k = 0; k <= valid; ++k) {
            const double cr = h->Z[k].re, ci = h->Z[k].im;
            const double e = h->energy[k] * (double)h->pss_energy[u];
            const double m = (e > 1e-20) ? (cr * cr + ci * ci) / e : 0.0;
            if (m > top) { top = m; top_k = k; }
        }
        if (top < (double)min_metric) continue;
        /* CONFIRM THE 5 ms REPEAT. A single strong correlation is an impulse
         * or a coincidence; a cell does this twice every radio frame, for
         * ever. The mate is required to reach half of THIS peak rather than
         * half of the threshold: a real cell's two occurrences are within a
         * fade of each other, while noise that happens to clear an absolute
         * bar twice does not. Written the weak way it let one detection
         * through in eight seeds of pure noise. */
        {
            int mate = 0;
            long long d;
            for (d = -ATKDSP_LTE_PSS_PERIOD; d <= ATKDSP_LTE_PSS_PERIOD;
                 d += 2 * ATKDSP_LTE_PSS_PERIOD) {
                const long long kk = (long long)top_k + d;
                if (kk < 0 || (size_t)kk > valid) continue;
                {
                    const double cr = h->Z[kk].re, ci = h->Z[kk].im;
                    const double e = h->energy[kk] * (double)h->pss_energy[u];
                    if (e > 1e-20 && (cr * cr + ci * ci) / e > 0.5 * top)
                        mate = 1;
                }
            }
            if (!mate) continue;
        }
        if (top > best_m) { best_m = top; best_k = top_k; best_u = u; }
    }
    if (best_u < 0) return 0;

    memset(out, 0, sizeof *out);
    out->nid2 = best_u;
    out->nid1 = -1;
    out->pci = -1;
    out->subframe = -1;
    out->offset = (long long)best_k;
    out->metric = (float)best_m;
    out->sss_score = 0.0f;

    /* carrier offset: the phase that accumulates across the two halves of
     * the matched symbol, 64 samples apart */
    {
        const atkdsp_cf32 *ref = h->pss + (size_t)best_u * NSYM;
        double ar = 0, ai = 0, br = 0, bi = 0;
        for (i = 0; i < 64; ++i) {
            const atkdsp_cf32 x = in[best_k + i], p = ref[i];
            ar += (double)p.re * x.re + (double)p.im * x.im;
            ai += (double)p.re * x.im - (double)p.im * x.re;
        }
        for (i = 64; i < NSYM; ++i) {
            const atkdsp_cf32 x = in[best_k + i], p = ref[i];
            br += (double)p.re * x.re + (double)p.im * x.im;
            bi += (double)p.re * x.im - (double)p.im * x.re;
        }
        {
            const double pr = br * ar + bi * ai;      /* b * conj(a) */
            const double pi_ = bi * ar - br * ai;
            out->cfo_hz = (float)(atan2(pi_, pr) * ATKDSP_LTE_RATE
                                  / (2.0 * ATK_PI * 64.0));
        }
    }

    /* ---- SSS, one symbol earlier, with the PSS as the channel reference */
    {
        long long at = (long long)best_k - SSS_BACK;
        if (at < 0) at += ATKDSP_LTE_PSS_PERIOD;
        if (at >= 0 && (size_t)at + NSYM <= n) {
            atkdsp_cf32 yp[NSYM], ys[NSYM];
            float r[NCARR];
            const atkdsp_cf32 *pf = h->pss_freq + (size_t)best_u * NCARR;
            int idx, best_i = -1;
            double best_s = -1e30;
            if (atkdsp_fft_exec(h->sym, in + best_k, yp, 0) == ATKDSP_OK &&
                atkdsp_fft_exec(h->sym, in + at, ys, 0) == ATKDSP_OK) {
                for (i = 0; i < NCARR; ++i) {
                    /* bin for carrier i: -31..-1 then +1..+31 */
                    const size_t b = (i < 31) ? (NSYM - 31 + i) : (1 + (i - 31));
                    const atkdsp_cf32 P = pf[i], Y = yp[b], S = ys[b];
                    /* H = Y * conj(P);  eq = S * conj(H) / |H| */
                    const double hr = (double)Y.re * P.re + (double)Y.im * P.im;
                    const double hi = (double)Y.im * P.re - (double)Y.re * P.im;
                    const double mag = sqrt(hr * hr + hi * hi) + 1e-12;
                    r[i] = (float)(((double)S.re * hr + (double)S.im * hi) / mag);
                }
                for (idx = 0; idx < NSSS; ++idx) {
                    const float *seq = h->sss
                        + ((size_t)best_u * NSSS + (size_t)idx) * NCARR;
                    double sc = 0.0;
                    for (i = 0; i < NCARR; ++i) sc += (double)seq[i] * r[i];
                    if (sc > best_s) { best_s = sc; best_i = idx; }
                }
                if (best_i >= 0) {
                    out->nid1 = best_i / 2;
                    out->subframe = (best_i & 1) ? 5 : 0;
                    out->pci = 3 * out->nid1 + best_u;
                    out->sss_score = (float)(best_s / (double)NCARR);
                }
            }
        }
    }
    return 1;
}
