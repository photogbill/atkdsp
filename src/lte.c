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

/* Non-coherent PSS integration (weak-signal detection). A cell repeats its PSS
 * every 5 ms (PSS_PERIOD samples); over a long buffer there are dozens to
 * hundreds of occurrences. Folding the normalised correlation metric modulo the
 * period and averaging the occurrences lifts a cell that no single occurrence
 * shows above the noise — measured reliable to ~-18 dB per-PSS-sample SNR (the
 * single-shot path dies near -13 dB) with zero false alarms in noise. The fold
 * is a CFAR test: its peak must exceed the fold's own mean by PSS_NI_K standard
 * deviations. K=9 gave 0 false alarms over 150 noise buffers and 100% detection
 * to -18 dB (scratch/pssni2.py). It runs only as a FALLBACK when the proven
 * single-shot path finds nothing, so strong-cell behaviour is unchanged. */
#define PSS_NI_K 9.0

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
    h->fold   = (double *)atk_aligned_malloc(
        (size_t)ATKDSP_LTE_PSS_PERIOD * sizeof(double));
    h->fcount = (int *)atk_aligned_malloc(
        (size_t)ATKDSP_LTE_PSS_PERIOD * sizeof(int));
    if (!h->sym || !h->pss || !h->pss_freq || !h->sss || !h->vit_bp
            || !h->fold || !h->fcount) {
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
    atk_aligned_free(h->fold);
    atk_aligned_free(h->fcount);
    atk_aligned_free(h->work);
    atk_aligned_free(h->vit_bp);
    free(h);
}

static int ensure_scan(atkdsp_lte *h, size_t n) {
    if (h->scan && h->scan_n == n) return ATKDSP_OK;
    if (h->scan) atkdsp_fft_destroy(h->scan);
    atk_aligned_free(h->X); atk_aligned_free(h->P);
    atk_aligned_free(h->Z); atk_aligned_free(h->energy);
    atk_aligned_free(h->work);
    h->scan = atkdsp_fft_create(n);
    h->X = (atkdsp_cf32 *)atk_aligned_malloc(n * sizeof(atkdsp_cf32));
    h->P = (atkdsp_cf32 *)atk_aligned_malloc(n * sizeof(atkdsp_cf32));
    h->Z = (atkdsp_cf32 *)atk_aligned_malloc(n * sizeof(atkdsp_cf32));
    h->energy = (double *)atk_aligned_malloc(n * sizeof(double));
    h->work = (atkdsp_cf32 *)atk_aligned_malloc(n * sizeof(atkdsp_cf32));
    h->scan_n = n;
    if (!h->scan || !h->X || !h->P || !h->Z || !h->energy || !h->work) {
        h->scan_n = 0; return ATKDSP_E_NOMEM;
    }
    return ATKDSP_OK;
}

/* ---- detection --------------------------------------------------------- */

/* Fill one cell: the CFO across the matched symbol's two halves, then the SSS
 * one symbol earlier with the PSS as the channel reference. Extracted verbatim
 * from the single-cell path so the primary cell (result[0]) is byte-identical
 * to what this library returned before multi-cell — the multi-cell code only
 * adds MORE cells behind it. */
static void decode_cell(atkdsp_lte *h, const atkdsp_cf32 *in, size_t n,
                        int u, size_t k, double metric, atkdsp_lte_cell *out) {
    size_t i;
    memset(out, 0, sizeof *out);
    out->nid2 = u;
    out->nid1 = -1;
    out->pci = -1;
    out->subframe = -1;
    out->offset = (long long)k;
    out->metric = (float)metric;
    out->sss_score = 0.0f;
    {
        const atkdsp_cf32 *ref = h->pss + (size_t)u * NSYM;
        double ar = 0, ai = 0, br = 0, bi = 0;
        for (i = 0; i < 64; ++i) {
            const atkdsp_cf32 x = in[k + i], p = ref[i];
            ar += (double)p.re * x.re + (double)p.im * x.im;
            ai += (double)p.re * x.im - (double)p.im * x.re;
        }
        for (i = 64; i < NSYM; ++i) {
            const atkdsp_cf32 x = in[k + i], p = ref[i];
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
    {
        long long at = (long long)k - SSS_BACK;
        if (at < 0) at += ATKDSP_LTE_PSS_PERIOD;
        if (at >= 0 && (size_t)at + NSYM <= n) {
            atkdsp_cf32 yp[NSYM], ys[NSYM];
            float r[NCARR];
            const atkdsp_cf32 *pf = h->pss_freq + (size_t)u * NCARR;
            int idx, best_i = -1;
            double best_s = -1e30;
            if (atkdsp_fft_exec(h->sym, in + k, yp, 0) == ATKDSP_OK &&
                atkdsp_fft_exec(h->sym, in + (size_t)at, ys, 0) == ATKDSP_OK) {
                for (i = 0; i < NCARR; ++i) {
                    const size_t b = (i < 31) ? (NSYM - 31 + i) : (1 + (i - 31));
                    const atkdsp_cf32 P = pf[i], Y = yp[b], S = ys[b];
                    const double hr = (double)Y.re * P.re + (double)Y.im * P.im;
                    const double hi = (double)Y.im * P.re - (double)Y.re * P.im;
                    const double mag = sqrt(hr * hr + hi * hi) + 1e-12;
                    r[i] = (float)(((double)S.re * hr + (double)S.im * hi) / mag);
                }
                for (idx = 0; idx < NSSS; ++idx) {
                    const float *seq = h->sss
                        + ((size_t)u * NSSS + (size_t)idx) * NCARR;
                    double sc = 0.0;
                    for (i = 0; i < NCARR; ++i) sc += (double)seq[i] * r[i];
                    if (sc > best_s) { best_s = sc; best_i = idx; }
                }
                if (best_i >= 0) {
                    out->nid1 = best_i / 2;
                    out->subframe = (best_i & 1) ? 5 : 0;
                    out->pci = 3 * out->nid1 + u;
                    out->sss_score = (float)(best_s / (double)NCARR);
                }
            }
        }
    }
}

/* ONE cell from a buffer: the strongest PSS peak with a 5 ms mate (strong
 * cells), else the non-coherent PSS fold with its CFAR test (weak cells). This
 * is the PROVEN single-cell detector, unchanged — it just takes the buffer as
 * an argument so the multi-cell loop can run it again on a residual. Returns 1
 * and sets *u/*k/*metric, or 0. `buf` must already be in h->X as its FFT and
 * h->energy as its running window energy (fill_scan does both). */
static int detect_one(atkdsp_lte *h, size_t n, size_t valid, float min_metric,
                      int *out_u, size_t *out_k, double *out_m) {
    const int P = ATKDSP_LTE_PSS_PERIOD;
    int u, best_u = -1;
    double best_m = 0.0; size_t best_k = 0;
    int ni_u = -1; size_t ni_k = 0; double ni_margin = 0.0, ni_metric = 0.0;
    size_t i, k;
    for (u = 0; u < 3; ++u) {
        const atkdsp_cf32 *ref = h->pss + (size_t)u * NSYM;
        const double pe = (double)h->pss_energy[u];
        double top = 0.0; size_t top_k = 0;
        int r;
        memset(h->P, 0, n * sizeof(atkdsp_cf32));
        memcpy(h->P, ref, NSYM * sizeof(atkdsp_cf32));
        if (atkdsp_fft_exec(h->scan, h->P, h->Z, 0) != ATKDSP_OK) return -1;
        for (i = 0; i < n; ++i) {
            const float xr = h->X[i].re, xi = h->X[i].im;
            const float pr = h->Z[i].re, pi = h->Z[i].im;
            h->P[i].re = xr * pr + xi * pi;
            h->P[i].im = xi * pr - xr * pi;
        }
        if (atkdsp_fft_exec(h->scan, h->P, h->Z, 1) != ATKDSP_OK) return -1;
        memset(h->fold, 0, (size_t)P * sizeof(double));
        memset(h->fcount, 0, (size_t)P * sizeof(int));
        r = 0;
        for (k = 0; k <= valid; ++k) {
            const double cr = h->Z[k].re, ci = h->Z[k].im;
            const double e = h->energy[k] * pe;
            const double m = (e > 1e-20) ? (cr * cr + ci * ci) / e : 0.0;
            if (m > top) { top = m; top_k = k; }
            h->fold[r] += m; h->fcount[r]++;
            if (++r == P) r = 0;
        }
        if (top >= (double)min_metric) {
            int mate = 0;
            long long d;
            for (d = -P; d <= P; d += 2 * P) {
                const long long kk = (long long)top_k + d;
                if (kk < 0 || (size_t)kk > valid) continue;
                {
                    const double cr = h->Z[kk].re, ci = h->Z[kk].im;
                    const double e = h->energy[kk] * pe;
                    if (e > 1e-20 && (cr * cr + ci * ci) / e > 0.5 * top) mate = 1;
                }
            }
            if (mate && top > best_m) { best_m = top; best_k = top_k; best_u = u; }
        }
        {
            int rpk = 0; double pk = -1.0, sum = 0.0, sq = 0.0; int cnt = 0;
            for (r = 0; r < P; ++r)
                if (h->fcount[r]) h->fold[r] /= (double)h->fcount[r];
            for (r = 0; r < P; ++r)
                if (h->fold[r] > pk) { pk = h->fold[r]; rpk = r; }
            for (r = 0; r < P; ++r) {
                if (r == rpk) continue;
                sum += h->fold[r]; sq += h->fold[r] * h->fold[r]; ++cnt;
            }
            if (cnt > 1) {
                const double mean = sum / cnt;
                const double var = sq / cnt - mean * mean;
                const double sd = (var > 0.0) ? sqrt(var) : 0.0;
                const double margin = (sd > 1e-30) ? (pk - mean) / sd : 0.0;
                if (margin >= PSS_NI_K && margin > ni_margin) {
                    size_t bk = (size_t)rpk; double bm = -1.0;
                    long long kk;
                    for (kk = rpk; (size_t)kk <= valid; kk += P) {
                        const double cr = h->Z[kk].re, ci = h->Z[kk].im;
                        const double e = h->energy[kk] * pe;
                        const double m = (e > 1e-20) ? (cr * cr + ci * ci) / e : 0.0;
                        if (m > bm) { bm = m; bk = (size_t)kk; }
                    }
                    ni_margin = margin; ni_u = u; ni_k = bk; ni_metric = bm;
                }
            }
        }
    }
    if (best_u < 0) {
        if (ni_u < 0) return 0;
        best_u = ni_u; best_k = ni_k; best_m = ni_metric;
    }
    *out_u = best_u; *out_k = best_k; *out_m = best_m;
    return 1;
}

/* Fill h->X (FFT of buf) and h->energy (running 128-window power of buf). */
static int fill_scan(atkdsp_lte *h, const atkdsp_cf32 *buf, size_t n, size_t valid) {
    size_t i, k;
    double acc = 0.0;
    for (i = 0; i < (size_t)NSYM && i < n; ++i)
        acc += (double)buf[i].re * buf[i].re + (double)buf[i].im * buf[i].im;
    h->energy[0] = acc;
    for (k = 1; k <= valid; ++k) {
        const atkdsp_cf32 a = buf[k - 1], b = buf[k + NSYM - 1];
        acc += (double)b.re * b.re + (double)b.im * b.im;
        acc -= (double)a.re * a.re + (double)a.im * a.im;
        h->energy[k] = acc;
    }
    return atkdsp_fft_exec(h->scan, buf, h->X, 0);
}

/* Least-squares remove one reference symbol from buf[at..at+NSYM): estimate the
 * complex gain g = <ref, buf>/|ref|^2 and subtract g*ref. Removes the dominant
 * cell's sync-signal energy so a weaker co-channel cell underneath it can be
 * detected and its SSS read without corruption. */
static void project_out(atkdsp_cf32 *buf, const atkdsp_cf32 *ref, double energy,
                        size_t at) {
    double gr = 0.0, gi = 0.0;
    int i;
    if (energy <= 1e-20) return;
    for (i = 0; i < NSYM; ++i) {
        gr += (double)ref[i].re * buf[at + i].re + (double)ref[i].im * buf[at + i].im;
        gi += (double)ref[i].re * buf[at + i].im - (double)ref[i].im * buf[at + i].re;
    }
    gr /= energy; gi /= energy;
    for (i = 0; i < NSYM; ++i) {
        buf[at + i].re -= (float)(gr * ref[i].re - gi * ref[i].im);
        buf[at + i].im -= (float)(gr * ref[i].im + gi * ref[i].re);
    }
}

/* Subtract a decoded cell's PSS (every 5 ms) and SSS (137 samples before each
 * PSS, subframe alternating) from the residual buffer. */
static void subtract_cell(atkdsp_lte *h, atkdsp_cf32 *buf, size_t n,
                          const atkdsp_lte_cell *c) {
    const int P = ATKDSP_LTE_PSS_PERIOD;
    const atkdsp_cf32 *pss = h->pss + (size_t)c->nid2 * NSYM;
    const double pe = (double)h->pss_energy[c->nid2];
    atkdsp_cf32 sss0[NSYM], sss5[NSYM];
    double se0 = 0.0, se5 = 0.0;
    int have_sss = (c->nid1 >= 0);
    int i;
    long long base, kk;
    if (have_sss) {
        float v[NCARR]; atkdsp_cf32 carr[NCARR];
        atkdsp_lte_sss_symbol(c->nid1, c->nid2, 0, v);
        for (i = 0; i < NCARR; ++i) { carr[i].re = v[i]; carr[i].im = 0.0f; }
        to_symbol(h->sym, carr, sss0);
        atkdsp_lte_sss_symbol(c->nid1, c->nid2, 5, v);
        for (i = 0; i < NCARR; ++i) { carr[i].re = v[i]; carr[i].im = 0.0f; }
        to_symbol(h->sym, carr, sss5);
        for (i = 0; i < NSYM; ++i) {
            se0 += (double)sss0[i].re * sss0[i].re + (double)sss0[i].im * sss0[i].im;
            se5 += (double)sss5[i].re * sss5[i].re + (double)sss5[i].im * sss5[i].im;
        }
    }
    base = c->offset % P;
    {
        long long j0 = (c->offset - base) / P;
        for (kk = base; kk + NSYM <= (long long)n; kk += P) {
            long long j = (kk - base) / P;
            int sf = c->subframe;
            if (((j - j0) & 1)) sf = (sf == 0) ? 5 : 0;
            project_out(buf, pss, pe, (size_t)kk);
            if (have_sss) {
                long long a = kk - SSS_BACK;
                if (a >= 0 && a + NSYM <= (long long)n)
                    project_out(buf, sf ? sss5 : sss0, sf ? se5 : se0, (size_t)a);
            }
        }
    }
}

/* Gates a SECONDARY cell (any after the strongest); the primary is never held
 * to them, so result[0] is exactly the old single-cell answer.
 *
 * The decisive one is the METRIC. After the strong cell is cancelled, a real
 * co-channel cell DOMINATES the residual and its normalised PSS correlation
 * jumps to ~0.8-1.0 (measured 0.98 for the oracle's -6 dB cell). A cancellation
 * stub left by imperfect subtraction is just residual, metric ~0.07 — a wide,
 * clean gap. (An SSS floor alone is not enough: a high-SNR synthetic single
 * cell throws stubs whose SSS looks real, ~1.2, but whose metric is ~0.07.)
 * The SSS floor stays as a second, independent check. */
#define SEC_METRIC_MIN 0.30
#define SEC_SSS_MIN 0.5f

ptrdiff_t atkdsp_lte_detect(atkdsp_lte *h, const atkdsp_cf32 *in, size_t n,
                            float min_metric, atkdsp_lte_cell *out, size_t cap) {
    size_t valid, nout = 0;
    if (!h || !in || !out || cap == 0) return ATKDSP_E_ARG;
    if (n < 2 * (size_t)ATKDSP_LTE_PSS_PERIOD) return 0;
    if (ensure_scan(h, n) != ATKDSP_OK) return ATKDSP_E_NOMEM;
    if (min_metric <= 0.0f) min_metric = 0.06f;
    valid = n - NSYM;                     /* beyond this the circular corr wraps */

    /* SUCCESSIVE INTERFERENCE CANCELLATION. Detect the strongest cell on the
     * live buffer (pass 0 is byte-identical to the old single-cell result),
     * decode it, then SUBTRACT its PSS/SSS from a residual copy and detect
     * again. Only after the strong cell is removed does a weaker co-channel
     * cell rise above the interference floor with an uncorrupted SSS — which is
     * why a plain re-scan of the same buffer could not separate them. Stops
     * when a pass finds nothing, when the residual yields a PCI already seen
     * (cancellation left a stub), or at `cap`. */
    memcpy(h->work, in, n * sizeof(atkdsp_cf32));
    while (nout < cap) {
        int u; size_t k; double m; int rc;
        const atkdsp_cf32 *buf = (nout == 0) ? in : h->work;
        if (fill_scan(h, buf, n, valid) != ATKDSP_OK) return ATKDSP_E_FFT;
        rc = detect_one(h, n, valid, min_metric, &u, &k, &m);
        if (rc < 0) return ATKDSP_E_FFT;
        if (rc == 0) break;
        atkdsp_lte_cell cell;
        decode_cell(h, buf, n, u, k, m, &cell);
        if (nout > 0) {
            /* a residual pass is only believed with a real SSS; below the
             * floor it is a cancellation ghost and the search is done. A
             * repeated PCI means the same cell resurfaced — also done. */
            int dup = 0; size_t j;
            for (j = 0; j < nout; ++j)
                if (out[j].pci == cell.pci && cell.pci >= 0) { dup = 1; break; }
            if (dup || cell.pci < 0 || m < SEC_METRIC_MIN
                    || cell.sss_score < SEC_SSS_MIN) break;
        }
        out[nout++] = cell;
        if (nout >= cap) break;
        /* cancel this cell from the residual so the next pass sees past it.
         * pass 0 read `in`; every later pass reads and writes h->work. */
        if (nout == 1) memcpy(h->work, in, n * sizeof(atkdsp_cf32));
        subtract_cell(h, h->work, n, &cell);
    }
    return (ptrdiff_t)nout;
}
