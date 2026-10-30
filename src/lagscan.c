/* ---- 31. atkdsp_lagscan: the full lag scan (Undertow piece 5) -----------------
 * atkdsp.h §31; docs/ATKDSP_PLAN.md §B6c (piece 5). The twin is reference.py
 * §31 (lagscan), the spec.
 *
 * Below where the survey's candidates hold the period (~2 dB under its own
 * threshold), every period of a cell is searched at once:
 *
 *   c_d[k] = y[k] conj(y[k + d]) — a lag product. Both samples inside one code
 *   period carry the same data sign, so it cancels; a carrier offset becomes a
 *   constant phase per d. Folded at the true period, c_d is the code's +-1
 *   product pattern with every bin at full amplitude; anywhere else, noise.
 *
 * The fold at every trial period is the FAST FOLDING ALGORITHM (Staelin 1969,
 * the pulsar search): for a base period P it folds n periods at the n trial
 * periods P / (1 - x/P), x = s/(n-1), in n log2(n) P complex additions, ping-ponging two
 * buffers; at each stage a pair of half-size folds (A, B) combine as
 * out[s] = A[s/2] + roll(B[s/2], (s+1)/2), roll(x, h)[b] = x[(b + h) mod P],
 * the roll done as two contiguous runs (no modulo per element).
 *
 * Per base period: S[s] = sum over nlags lags of sum_b |F_d[s][b]|^2 /
 * (n var(c_d)). The null (the twin's words, §31): M copies of c x the cell's
 * lag-product spectrum over the P bins — a generalized chi-square, built from
 * the cell's autocorrelation (lags 0..16) and matched to the mean and the
 * variance of the base period's own trials (those above median + 5 x the
 * interquartile spread left out, and at least the `trim` largest); its tail by
 * Lugannani-Rice's saddlepoint. The best trial's one-trial tail, its period
 * refined by a parabola; every row's tail times every trial scanned
 * (Bonferroni), DETECT at cfg->pfa.
 *
 * Families: detections in order of significance seed them. A seed's period
 * is refined by direct folds over all the data; one the families already
 * found explain (CLEAN: their folded profiles taken off the lag products, it
 * no longer detects) joins one; otherwise nested folds step it up to a whole
 * multiple k p or down to a whole fraction p / k (k <= 8) where the power
 * that is not periodic in the shorter period says so, and a new family is
 * made; every detection at p/q (p, q <= 8) of it is marked.
 *
 * Threads: the base periods are independent and run in parallel (dynamic
 * schedule — the cost grows with P and the cores may be unequal); each fills
 * only its own row. The families' direct folds (the grid of a refinement,
 * the k of a step) run in parallel, each into its own buffers, each sum in
 * sample order; the result is the same at any thread count. The lag
 * products and the folds are single precision (the twin's are double:
 * measured agreement in tests/test_lagscan.py); the sums are double. Each
 * base period allocates its two fold buffers (n P complex each, at most the
 * input's size) — this is an offline call, not a per-block kernel. */
#include "internal.h"

#ifdef _OPENMP
#  include <omp.h>
#endif

#define LS_CLIP   5.0                   /* the twin's LAGSCAN_CLIP   */
#define LS_ACF    16                    /* LAGSCAN_ACF               */
#define LS_KMAX   8                     /* LAGSCAN_KMAX              */
#define LS_STEPS  4                     /* LAGSCAN_STEPS             */
#define LS_POWER  6.0                   /* LAGSCAN_POWER             */
#define LS_REFINE 16                    /* LAGSCAN_REFINE            */
#define LS_TWO_PI 6.283185307179586

void atkdsp_lagscan_defaults(atkdsp_lagscan_cfg *c, double fs, int pmin, int pmax) {
    if (!c) return;
    memset(c, 0, sizeof *c);
    c->fs = fs; c->pmin = pmin; c->pmax = pmax;
    c->nlags = 8; c->lag_min = 12; c->lag_frac = 0.5;
    c->nmax = 1 << 15; c->pfa = 1e-3; c->trim = 0.005;
}

/* the lags for base period P (the twin's lagscan_lags, integer for integer) */
static int ls_lags(int P, const atkdsp_lagscan_cfg *c, int *out) {
    int hi = (int)floor((double)P * c->lag_frac);
    if (hi < c->lag_min + 1) hi = c->lag_min + 1;
    const int span = hi - c->lag_min;
    int nd = 0;
    for (int i = 0; i < c->nlags; ++i) {
        const int d = (c->nlags == 1) ? c->lag_min
                    : c->lag_min + (2 * i * span + (c->nlags - 1)) / (2 * (c->nlags - 1));
        int dup = 0;
        for (int j = 0; j < nd; ++j) if (out[j] == d) { dup = 1; break; }
        if (!dup) out[nd++] = d;
    }
    return nd;
}

/* the FFA: src holds n rows of P; the n profiles end in whichever buffer the
 * last stage wrote, returned. */
static atkdsp_cf32 *ls_ffa(atkdsp_cf32 *src, atkdsp_cf32 *dst, size_t n, size_t P) {
    size_t g = 1;
    while (g < n) {
        const size_t groups = n / (2 * g);
        for (size_t j = 0; j < groups; ++j) {
            const atkdsp_cf32 *A = src + (2 * j) * g * P;
            const atkdsp_cf32 *B = src + (2 * j + 1) * g * P;
            atkdsp_cf32 *O = dst + j * 2 * g * P;
            for (size_t s = 0; s < 2 * g; ++s) {
                const atkdsp_cf32 *a = A + (s / 2) * P;
                const atkdsp_cf32 *b = B + (s / 2) * P;
                const size_t h = ((s + 1) / 2) % P;
                atkdsp_cf32 *o = O + s * P;
                const float *ATK_RESTRICT af = (const float *)a;
                const float *ATK_RESTRICT bf = (const float *)(b + h);
                float *ATK_RESTRICT of = (float *)o;
                const size_t r1 = 2 * (P - h);
                for (size_t q = 0; q < r1; ++q) of[q] = af[q] + bf[q];
                const float *ATK_RESTRICT af2 = (const float *)(a + (P - h));
                const float *ATK_RESTRICT bf2 = (const float *)b;
                float *ATK_RESTRICT of2 = (float *)(o + (P - h));
                const size_t r2 = 2 * h;
                for (size_t q = 0; q < r2; ++q) of2[q] = af2[q] + bf2[q];
            }
        }
        atkdsp_cf32 *t = src; src = dst; dst = t;
        g *= 2;
    }
    return src;
}

/* a/b ~ p/q with p, q <= 8 (2e-3 relative), q-major order as the twin */
static int ls_ratio(double a, double b, int *p, int *q) {
    const double ratio = a / b;
    for (int qq = 1; qq <= LS_KMAX; ++qq)
        for (int pp = 1; pp <= LS_KMAX; ++pp) {
            const double pq = (double)pp / (double)qq;
            if (fabs(ratio - pq) < 2e-3 * pq) { *p = pp; *q = qq; return 1; }
        }
    return 0;
}

static int ls_cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* ---- the null ------------------------------------------------------------- */

/* rho[m] = |r(m)|^2 / r(0)^2, m = 0..LS_ACF (the twin's lagscan_acf) */
static void ls_acf(const atkdsp_cf32 *y, size_t N, double *rho) {
    for (int m = 0; m <= LS_ACF; ++m) rho[m] = 0.0;
    rho[0] = 1.0;
    if (N < 2) return;
    double r0 = 0.0;
    for (size_t t = 0; t < N; ++t) r0 += (double)y[t].re * y[t].re + (double)y[t].im * y[t].im;
    r0 /= (double)N;
    if (!(r0 > 0.0)) return;
    const int mmax = (N - 1 < (size_t)LS_ACF) ? (int)(N - 1) : LS_ACF;
    for (int m = 1; m <= mmax; ++m) {
        double rr = 0.0, ri = 0.0;                     /* conj(y[t]) y[t + m] */
        for (size_t t = 0; t + (size_t)m < N; ++t) {
            const double ar = y[t].re, ai = y[t].im, br = y[t + m].re, bi = y[t + m].im;
            rr += ar * br + ai * bi;
            ri += ar * bi - ai * br;
        }
        rr /= (double)(N - (size_t)m); ri /= (double)(N - (size_t)m);
        rho[m] = (rr * rr + ri * ri) / (r0 * r0);
    }
}

/* the circulant's eigenvalues over nb bins, k from 0 (dc) or 1; returns the count */
static int ls_spectrum(const double *rho, int nb, int dc, double *lam) {
    int n = 0;
    for (int k = dc ? 0 : 1; k < nb; ++k) {
        double v = rho[0];
        for (int m = 1; m <= LS_ACF; ++m)
            v = v + 2.0 * rho[m] * cos(LS_TWO_PI * (double)((long long)k * m) / (double)nb);
        lam[n++] = v > 1e-6 ? v : 1e-6;
    }
    return n;
}

static double ls_sum(const double *v, int n) { double s = 0.0; for (int i = 0; i < n; ++i) s += v[i]; return s; }
static double ls_sum2(const double *v, int n) { double s = 0.0; for (int i = 0; i < n; ++i) s += v[i] * v[i]; return s; }

static double ls_mills(double w) {                  /* Phi-bar(w) / phi(w), w > 0 */
    if (w < 35.0) return sqrt(3.141592653589793 / 2.0) * exp(0.5 * w * w) * erfc(w / sqrt(2.0));
    const double w2 = w * w;
    return (1.0 - 1.0 / w2 + 3.0 / (w2 * w2) - 15.0 / (w2 * w2 * w2)) / w;
}

/* log10 P(Q >= x), Q = mult copies of sum_j (sc lam_j) E_j — Lugannani-Rice
 * (the twin's lagscan_tail, with the eigenvalues scaled by sc) */
static double ls_tail(double x, const double *lam0, int n, double sc, double mult) {
    double mean = 0.0, var = 0.0, lmax = 0.0;
    for (int j = 0; j < n; ++j) {
        const double l = sc * lam0[j];
        mean += l; var += l * l;
        if (l > lmax) lmax = l;
    }
    mean *= mult; var *= mult;
    if (!(var > 0.0)) return 0.0;
    if (!(x > mean)) return log10(0.5 * erfc((x - mean) / sqrt(var) / sqrt(2.0)));
    double lo = 0.0, hi = 1.0 / lmax, s = 0.0;
    for (int it = 0; it < 100; ++it) {
        double k1 = 0.0, k2 = 0.0;
        for (int j = 0; j < n; ++j) {
            const double l = sc * lam0[j], g = l / (1.0 - l * s);
            k1 += g; k2 += g * g;
        }
        k1 *= mult; k2 *= mult;
        if (k1 > x) hi = s; else lo = s;
        double sn = s - (k1 - x) / k2;
        if (!(lo < sn && sn < hi)) sn = 0.5 * (lo + hi);
        const int done = fabs(sn - s) <= 1e-13 * hi;
        s = sn;
        if (done) break;
    }
    double K = 0.0, k2 = 0.0;
    for (int j = 0; j < n; ++j) {
        const double l = sc * lam0[j], d = 1.0 - l * s, g = l / d;
        K += log(d); k2 += g * g;
    }
    K *= -mult; k2 *= mult;
    const double w2 = 2.0 * (s * x - K);
    const double w = sqrt(w2 > 0.0 ? w2 : 0.0), u = s * sqrt(k2);
    if (w < 1e-6 || !(u > 0.0)) return log10(0.5);
    double t = ls_mills(w) + 1.0 / u - 1.0 / w;
    if (!(t > 0.0)) t = ls_mills(w);
    return (-0.5 * w * w - 0.5 * log(LS_TWO_PI) + log(t)) / log(10.0);
}

/* one base period into its row; ATKDSP_OK, or ATKDSP_E_NOMEM */
static int ls_one(const atkdsp_cf32 *y, size_t N, const atkdsp_lagscan_cfg *c, const double *rho, int P,
                  atkdsp_lagscan_row *row) {
    int L[64];
    const int nd = ls_lags(P, c, L);
    int dmax = 0;
    for (int i = 0; i < nd; ++i) if (L[i] > dmax) dmax = L[i];
    memset(row, 0, sizeof *row);
    row->base = P;
    row->family = -1;
    row->method[0] = ATKDSP_METHOD_LAGSCAN;
    row->method[1] = ATKDSP_METHOD_SCALE_SURVEY;
    const size_t avail = (N > (size_t)dmax) ? (N - (size_t)dmax) / (size_t)P : 0;
    if (avail < 16) { row->flags = ATKDSP_LAGSCAN_SHORT; return ATKDSP_OK; }
    size_t n = 1;
    while (n * 2 <= avail && n * 2 <= (size_t)c->nmax) n *= 2;
    const size_t np = n * (size_t)P;
    atkdsp_cf32 *b1 = (atkdsp_cf32 *)atk_aligned_malloc(np * sizeof(atkdsp_cf32));
    atkdsp_cf32 *b2 = (atkdsp_cf32 *)atk_aligned_malloc(np * sizeof(atkdsp_cf32));
    double *S = (double *)calloc(n, sizeof(double));
    double *srt = (double *)malloc(n * sizeof(double));
    double *lam = (double *)malloc((size_t)P * sizeof(double));
    if (!b1 || !b2 || !S || !srt || !lam) {
        atk_aligned_free(b1); atk_aligned_free(b2); free(S); free(srt); free(lam);
        return ATKDSP_E_NOMEM;
    }
    for (int li = 0; li < nd; ++li) {
        const size_t d = (size_t)L[li];
        /* the lag product, its mean removed, its variance */
        double mr = 0.0, mi = 0.0;
        for (size_t k = 0; k < np; ++k) {
            const double ar = y[k].re, ai = y[k].im, br = y[k + d].re, bi = y[k + d].im;
            mr += ar * br + ai * bi;
            mi += ai * br - ar * bi;
        }
        mr /= (double)np; mi /= (double)np;
        double v = 0.0;
        for (size_t k = 0; k < np; ++k) {
            const double ar = y[k].re, ai = y[k].im, br = y[k + d].re, bi = y[k + d].im;
            const double cr = ar * br + ai * bi - mr, ci = ai * br - ar * bi - mi;
            b1[k].re = (float)cr; b1[k].im = (float)ci;
            v += cr * cr + ci * ci;
        }
        v /= (double)np;
        if (!(v > 0.0)) continue;
        const atkdsp_cf32 *F = ls_ffa(b1, b2, n, (size_t)P);
        const double norm = 1.0 / ((double)n * v);
        for (size_t s = 0; s < n; ++s) {
            const atkdsp_cf32 *r = F + s * (size_t)P;
            double e = 0.0;
            for (int b = 0; b < P; ++b) e += (double)r[b].re * r[b].re + (double)r[b].im * r[b].im;
            S[s] += e * norm;
        }
    }
    /* the null's moments (the twin's words): the median and the interquartile
     * spread; trials above median + LS_CLIP x spread are signal and left out
     * (a strong code lifts a broad stretch of trials), and never fewer than
     * `trim` of the largest */
    memcpy(srt, S, n * sizeof(double));
    qsort(srt, n, sizeof(double), ls_cmp_double);
    const double med = srt[(n - 1) / 2];
    const double spread = (srt[(3 * n) / 4] - srt[n / 4]) / 1.349;
    const double lim = med + LS_CLIP * spread;
    size_t keep = 0;
    while (keep < n && srt[keep] <= lim) ++keep;
    const size_t cap_keep = n - (size_t)floor((double)n * c->trim);
    if (keep > cap_keep) keep = cap_keep;
    if (keep < 8) keep = 8;
    if (keep > n) keep = n;
    double mu = 0.0;
    for (size_t i = 0; i < keep; ++i) mu += srt[i];
    mu /= (double)keep;
    double var = 0.0;
    for (size_t i = 0; i < keep; ++i) var += (srt[i] - mu) * (srt[i] - mu);
    var /= (double)keep;
    row->nfold = (int32_t)n;
    if (mu > 0.0 && var > 0.0) {
        /* M copies of cc x the spectrum, its mean and variance the trials' */
        const int nl = ls_spectrum(rho, P, 0, lam);
        const double sl = ls_sum(lam, nl), sl2 = ls_sum2(lam, nl);
        const double cc = (var / mu) * sl / sl2, M = mu / (cc * sl);
        size_t sb = 0;
        for (size_t s = 1; s < n; ++s) if (S[s] > S[sb]) sb = s;
        const double raw = ls_tail(S[sb], lam, nl, cc, M);
        double frac = 0.0;
        if (sb > 0 && sb < n - 1) {
            const double a = S[sb - 1], b = S[sb], e = S[sb + 1];
            const double den = a - 2.0 * b + e;
            if (den < 0.0) frac = 0.5 * (a - e) / den;
        }
        /* P bins over the period P / (1 - x/P), x the drift a row — not P + x
         * (the twin's words) */
        const double xr = ((double)sb + frac) / (double)(n - 1);
        const double T = (double)P / (1.0 - xr / (double)P);
        row->period_samples = T;
        row->period_s = T / c->fs;
        row->log10_pfa_raw = raw;
        row->stat = S[sb];
        row->shape = M;
        row->scale = cc;
        row->excess = (S[sb] - mu) / mu;
    }
    atk_aligned_free(b1); atk_aligned_free(b2); free(S); free(srt); free(lam);
    return ATKDSP_OK;
}

/* ---- families ------------------------------------------------------------- */

typedef struct {                        /* a family found: what CLEAN takes off */
    double T;
    int    Q;
    double **prof;                      /* [lag value] -> Q complex (re, im), or NULL */
} ls_sub;

typedef struct {
    const atkdsp_cf32 *y;
    size_t N;
    const double *rho;
    ls_sub *subs;
    int nsub, dcap;                     /* dcap: lag values 0 .. dcap - 1 */
} ls_ctx;

/* one lag's folded mean profile at (T, Q), its mean taken off (the twin's
 * lagscan_profile); out: Q complex */
static void ls_profile(const ls_ctx *x, int d, double T, int Q, double *out, double *cnt) {
    const size_t Tn = x->N - (size_t)d;
    const double f = (double)Q / T;
    for (int b = 0; b < Q; ++b) { out[2 * b] = out[2 * b + 1] = 0.0; cnt[b] = 0.0; }
    double sr = 0.0, si = 0.0;
    for (size_t t = 0; t < Tn; ++t) {
        const int b = (int)((long long)floor((double)t * f) % Q);
        const double ar = x->y[t].re, ai = x->y[t].im, br = x->y[t + d].re, bi = x->y[t + d].im;
        const double cr = ar * br + ai * bi, ci = ai * br - ar * bi;
        out[2 * b] += cr; out[2 * b + 1] += ci; cnt[b] += 1.0;
        sr += cr; si += ci;
    }
    sr /= (double)Tn; si /= (double)Tn;
    for (int b = 0; b < Q; ++b) {
        if (cnt[b] > 0.0) { out[2 * b] = out[2 * b] / cnt[b] - sr; out[2 * b + 1] = out[2 * b + 1] / cnt[b] - si; }
        else out[2 * b] = out[2 * b + 1] = 0.0;
    }
}

/* every family's profile for these lags, made before a parallel batch reads them */
static int ls_need(ls_ctx *x, const int *lags, int nl) {
    for (int s = 0; s < x->nsub; ++s) {
        ls_sub *u = &x->subs[s];
        for (int i = 0; i < nl; ++i) {
            const int d = lags[i];
            if (d >= x->dcap || u->prof[d]) continue;
            double *pr = (double *)malloc((size_t)u->Q * 2 * sizeof(double));
            double *cn = (double *)malloc((size_t)u->Q * sizeof(double));
            if (!pr || !cn) { free(pr); free(cn); return ATKDSP_E_NOMEM; }
            ls_profile(x, d, u->T, u->Q, pr, cn);
            free(cn);
            u->prof[d] = pr;
        }
    }
    return ATKDSP_OK;
}

/* the lag products folded directly at a real period into nb bins, the
 * families in `use` (a bitmask over subs; ~0: all) taken off first (the twin's
 * lagscan_fold). Z: nl x nb complex; cnt: nb. The bins are computed once per
 * block of LS_BLOCK samples and shared by every lag; each lag's sums still
 * run in sample order. ATKDSP_OK or ATKDSP_E_NOMEM. */
#define LS_BLOCK 4096
static int ls_fold(const ls_ctx *x, const int *lags, int nl, double period, int nb, unsigned long long use,
                   double *Z, double *cnt) {
    int dmax = 0;
    for (int i = 0; i < nl; ++i) if (lags[i] > dmax) dmax = lags[i];
    const size_t Tn = x->N - (size_t)dmax;
    const double f = (double)nb / period;
    int ns = 0, sidx[64];
    for (int s = 0; s < x->nsub && s < 64; ++s) if ((use >> s) & 1) sidx[ns++] = s;
    int *bins = (int *)malloc((size_t)(1 + ns) * LS_BLOCK * sizeof(int));
    if (!bins) return ATKDSP_E_NOMEM;
    double sr[64], si[64], sq[64];
    for (int i = 0; i < nl; ++i) {
        sr[i] = si[i] = sq[i] = 0.0;
        double *z = Z + (size_t)i * nb * 2;
        for (int b = 0; b < 2 * nb; ++b) z[b] = 0.0;
    }
    for (int b = 0; b < nb; ++b) cnt[b] = 0.0;
    for (size_t t0 = 0; t0 < Tn; t0 += LS_BLOCK) {
        const int m = (Tn - t0 < (size_t)LS_BLOCK) ? (int)(Tn - t0) : LS_BLOCK;
        for (int j = 0; j < m; ++j) {
            const int b = (int)((long long)floor((double)(t0 + (size_t)j) * f) % nb);
            bins[j] = b;
            cnt[b] += 1.0;
        }
        for (int s = 0; s < ns; ++s) {
            const ls_sub *u = &x->subs[sidx[s]];
            const double fq = (double)u->Q / u->T;
            int *fb = bins + (size_t)(1 + s) * LS_BLOCK;
            for (int j = 0; j < m; ++j) fb[j] = (int)((long long)floor((double)(t0 + (size_t)j) * fq) % u->Q);
        }
        for (int i = 0; i < nl; ++i) {
            const int d = lags[i];
            double *z = Z + (size_t)i * nb * 2;
            double a_r = sr[i], a_i = si[i], a_q = sq[i];
            for (int j = 0; j < m; ++j) {
                const size_t t = t0 + (size_t)j;
                const double ar = x->y[t].re, ai = x->y[t].im, br = x->y[t + d].re, bi = x->y[t + d].im;
                double cr = ar * br + ai * bi, ci = ai * br - ar * bi;
                for (int s = 0; s < ns; ++s) {
                    const double *pr = x->subs[sidx[s]].prof[d];
                    const int bf = bins[(size_t)(1 + s) * LS_BLOCK + j];
                    cr = cr - pr[2 * bf]; ci = ci - pr[2 * bf + 1];
                }
                const int b = bins[j];
                z[2 * b] += cr; z[2 * b + 1] += ci;
                a_r += cr; a_i += ci; a_q += cr * cr + ci * ci;
            }
            sr[i] = a_r; si[i] = a_i; sq[i] = a_q;
        }
    }
    free(bins);
    for (int i = 0; i < nl; ++i) {
        double *z = Z + (size_t)i * nb * 2;
        const double mr = sr[i] / (double)Tn, mi = si[i] / (double)Tn;
        const double v = sq[i] / (double)Tn - (mr * mr + mi * mi);
        for (int b = 0; b < nb; ++b) {
            if (!(v > 0.0) || !(cnt[b] > 0.0)) { z[2 * b] = z[2 * b + 1] = 0.0; continue; }
            const double sc = sqrt(cnt[b] * v);
            z[2 * b] = (z[2 * b] - cnt[b] * mr) / sc;
            z[2 * b + 1] = (z[2 * b + 1] - cnt[b] * mi) / sc;
        }
    }
    return ATKDSP_OK;
}

/* (R, X): the folded power not periodic in nb/k bins, and the total */
static void ls_resid(const double *Z, int nl, int nb, int k, double *R, double *X) {
    const int Q = nb / k;
    double r = 0.0, tot = 0.0;
    for (int i = 0; i < nl; ++i) {
        const double *z = Z + (size_t)i * nb * 2;
        for (int q = 0; q < Q; ++q) {
            double mr = 0.0, mi = 0.0;
            for (int j = 0; j < k; ++j) { mr += z[2 * (j * Q + q)]; mi += z[2 * (j * Q + q) + 1]; }
            mr /= (double)k; mi /= (double)k;
            for (int j = 0; j < k; ++j) {
                const double dr = z[2 * (j * Q + q)] - mr, di = z[2 * (j * Q + q) + 1] - mi;
                r += dr * dr + di * di;
            }
        }
        for (int b = 0; b < 2 * nb; ++b) tot += z[b] * z[b];
    }
    *R = r; *X = tot;
}

/* a batch of independent folds: job i folds at per[i] into nbs[i] bins and
 * leaves (R, X) split into ks[i] sub-periods; run in parallel, each in its
 * own buffers. ATKDSP_OK or ATKDSP_E_NOMEM. */
static int ls_batch(const ls_ctx *x, const int *lags, int nl, int nj, const double *per, const int *nbs,
                    const int *ks, const unsigned long long *use, double *R, double *X) {
    int fails[64], j;
    for (j = 0; j < nj; ++j) fails[j] = 0;
#ifdef _OPENMP
    int nth = atkdsp_get_threads();
    if (nth < 1) nth = 1;
#   pragma omp parallel for num_threads(nth) schedule(dynamic, 1)
#endif
    for (j = 0; j < nj; ++j) {
        double *Z = (double *)malloc((size_t)nl * nbs[j] * 2 * sizeof(double));
        double *cnt = (double *)malloc((size_t)nbs[j] * sizeof(double));
        R[j] = X[j] = 0.0;
        if (Z && cnt && ls_fold(x, lags, nl, per[j], nbs[j], use[j], Z, cnt) == ATKDSP_OK) {
            ls_resid(Z, nl, nbs[j], ks[j], &R[j], &X[j]);
        } else {
            fails[j] = 1;
        }
        free(Z); free(cnt);
    }
    for (j = 0; j < nj; ++j) if (fails[j]) return ATKDSP_E_NOMEM;
    return ATKDSP_OK;
}

static int ls_round(double v) { return (int)floor(v + 0.5); }

/* the seed's period over +-LS_REFINE samples of drift, a parabola at the
 * peak (the twin's _ls_refine) */
static int ls_refine(const ls_ctx *x, const int *lags, int nl, double p, double *pout, double *Xout) {
    enum { NJ = 2 * LS_REFINE + 1 };
    int dmax = 0;
    for (int i = 0; i < nl; ++i) if (lags[i] > dmax) dmax = lags[i];
    const double Tn = (double)(x->N - (size_t)dmax), dl = p / Tn;
    const int Q = ls_round(p);
    double per[NJ], R[NJ], X[NJ];
    int nbs[NJ], ks[NJ];
    unsigned long long use[NJ];
    for (int j = 0; j < NJ; ++j) {
        per[j] = p + (double)(j - LS_REFINE) * dl; nbs[j] = Q; ks[j] = 1; use[j] = ~0ULL;
    }
    if (ls_batch(x, lags, nl, NJ, per, nbs, ks, use, R, X) != ATKDSP_OK) return ATKDSP_E_NOMEM;
    int jb = 0;
    for (int j = 1; j < NJ; ++j) if (X[j] > X[jb]) jb = j;
    double frac = 0.0;
    if (jb > 0 && jb < NJ - 1) {
        const double a = X[jb - 1], b = X[jb], e = X[jb + 1], den = a - 2.0 * b + e;
        if (den < 0.0) frac = 0.5 * (a - e) / den;
    }
    *pout = p + ((double)(jb - LS_REFINE) + frac) * dl;
    *Xout = X[jb];
    return ATKDSP_OK;
}

/* the nested folds (the twin's _ls_resolve) */
static int ls_resolve(const ls_ctx *x, const int *lags, int nl, double p, double M, double cc, double pfa,
                      double *Tout, uint32_t *flout, double *evout) {
    int dmax = 0;
    for (int i = 0; i < nl; ++i) if (lags[i] > dmax) dmax = lags[i];
    const double Tn = (double)(x->N - (size_t)dmax);
    const double lstep = log10(pfa / (double)(LS_KMAX - 1));
    uint32_t flags = 0;
    double ev = 0.0;
    double per[LS_KMAX], R[LS_KMAX], X[LS_KMAX];
    int nbs[LS_KMAX], ks[LS_KMAX];
    unsigned long long use[LS_KMAX];
    for (int st = 0; st < LS_STEPS; ++st) {
        int moved = 0;
        const int Q = ls_round(p);
        if (Q >= 2) {
            int nj = 0;
            for (int k = 2; k <= LS_KMAX; ++k) {
                if (16.0 * k * p > Tn) break;
                per[nj] = k * p; nbs[nj] = k * Q; ks[nj] = k; use[nj] = ~0ULL; ++nj;
            }
            if (nj && ls_batch(x, lags, nl, nj, per, nbs, ks, use, R, X) != ATKDSP_OK) return ATKDSP_E_NOMEM;
            double *lamQ = (double *)malloc((size_t)Q * sizeof(double));
            double *lamF = (double *)malloc((size_t)LS_KMAX * Q * sizeof(double));
            if (!lamQ || !lamF) { free(lamQ); free(lamF); return ATKDSP_E_NOMEM; }
            const int nq = ls_spectrum(x->rho, Q, 1, lamQ);
            const double sQ = ls_sum(lamQ, nq);
            for (int j = 0; j < nj && !moved; ++j) {
                const int k = ks[j];
                const int nf = ls_spectrum(x->rho, k * Q, 0, lamF);
                const double esig = X[j] - M * cc * ls_sum(lamF, nf);
                const double r0 = M * (k - 1) * cc * sQ;
                const double t = ls_tail(R[j], lamQ, nq, cc, M * (k - 1));
                if (t <= lstep && R[j] - r0 >= 0.5 * (double)(k - 1) / (double)k * esig) {
                    p = k * p; flags |= ATKDSP_LAGSCAN_FAM_UP; ev = t; moved = 1;
                }
            }
            free(lamQ); free(lamF);
        }
        if (!moved) break;
    }
    for (int st = 0; st < LS_STEPS; ++st) {
        int moved = 0, nj = 0;
        for (int k = LS_KMAX; k >= 2; --k) {
            const int Q = ls_round(p / k);
            if (Q < 4) continue;
            per[nj] = p; nbs[nj] = k * Q; ks[nj] = k; use[nj] = ~0ULL; ++nj;
        }
        if (nj && ls_batch(x, lags, nl, nj, per, nbs, ks, use, R, X) != ATKDSP_OK) return ATKDSP_E_NOMEM;
        for (int j = 0; j < nj && !moved; ++j) {
            const int k = ks[j], Q = nbs[j] / k;
            double *lamQ = (double *)malloc((size_t)Q * sizeof(double));
            double *lamF = (double *)malloc((size_t)nbs[j] * sizeof(double));
            if (!lamQ || !lamF) { free(lamQ); free(lamF); return ATKDSP_E_NOMEM; }
            const int nq = ls_spectrum(x->rho, Q, 1, lamQ), nf = ls_spectrum(x->rho, k * Q, 0, lamF);
            const double h1 = (double)(k - 1) / (double)k * (X[j] - M * cc * ls_sum(lamF, nf));
            const double r0 = M * (k - 1) * cc * ls_sum(lamQ, nq);
            const double sd = sqrt(M * (k - 1)) * cc * sqrt(ls_sum2(lamQ, nq));
            if (h1 >= LS_POWER * sd && R[j] - r0 < 0.5 * h1 - 3.0 * sd) {
                p = p / k; flags |= ATKDSP_LAGSCAN_FAM_DOWN; moved = 1;
            }
            free(lamQ); free(lamF);
        }
        if (!moved) break;
    }
    *Tout = p; *flout = flags; *evout = ev;
    return ATKDSP_OK;
}

/* a row joins family fi with p/q */
static void ls_mark(atkdsp_lagscan_row *r, atkdsp_lagscan_family *fams, int fi, int p, int q) {
    r->family = fi; r->harmonic_p = p; r->harmonic_q = q;
    r->flags |= (p == 1 && q == 1) ? ATKDSP_LAGSCAN_FUNDAMENTAL : ATKDSP_LAGSCAN_HARMONIC;
    fams[fi].members++;
}

static int ls_families(ls_ctx *x, const atkdsp_lagscan_cfg *c, atkdsp_lagscan_row *rows, int nrow,
                       double ntrials, atkdsp_lagscan_family *fams, int fcap, int *nfam) {
    const double lt = ntrials > 0.0 ? log10(ntrials) : 0.0, lpfa = log10(c->pfa);
    int *ord = (int *)malloc((size_t)nrow * sizeof(int));
    if (!ord) return ATKDSP_E_NOMEM;
    int nd = 0, nf = 0, rc = ATKDSP_OK;
    for (int i = 0; i < nrow; ++i) if (rows[i].flags & ATKDSP_LAGSCAN_DETECT) ord[nd++] = i;
    for (int i = 1; i < nd; ++i) {                          /* by (log10_pfa, base) */
        const int v = ord[i];
        int j = i - 1;
        while (j >= 0 && (rows[ord[j]].log10_pfa > rows[v].log10_pfa
                          || (rows[ord[j]].log10_pfa == rows[v].log10_pfa && rows[ord[j]].base > rows[v].base))) {
            ord[j + 1] = ord[j]; --j;
        }
        ord[j + 1] = v;
    }
    for (int oi = 0; oi < nd && rc == ATKDSP_OK; ++oi) {
        atkdsp_lagscan_row *seed = &rows[ord[oi]];
        if (seed->family >= 0) continue;
        int joined = 0;
        for (int fi = 0; fi < nf && !joined; ++fi) {
            int p, q;
            if (ls_ratio(seed->period_samples, fams[fi].period_samples, &p, &q)) {
                ls_mark(seed, fams, fi, p, q); joined = 1;
            }
        }
        if (joined) continue;
        int lags[64];
        const int nl = ls_lags(seed->base, c, lags);
        const double M = seed->shape, cc = seed->scale;
        double p, Xp;
        if ((rc = ls_need(x, lags, nl)) != ATKDSP_OK) break;
        if (nf) {
            /* cheap first (the twin's words): detects at its own period before
             * the families come off, not after -> theirs, without refining */
            const double p0 = seed->period_samples;
            const int Q0 = ls_round(p0);
            double per2[2], R2[2], X2[2];
            int nb2[2], k2[2];
            unsigned long long u2[2];
            per2[0] = per2[1] = p0; nb2[0] = nb2[1] = Q0; k2[0] = k2[1] = 1;
            u2[0] = 0ULL; u2[1] = ~0ULL;            /* (assigned, not initialised: older MSVC C) */
            if ((rc = ls_batch(x, lags, nl, 2, per2, nb2, k2, u2, R2, X2)) != ATKDSP_OK) break;
            double *lam0 = (double *)malloc((size_t)Q0 * sizeof(double));
            if (!lam0) { rc = ATKDSP_E_NOMEM; break; }
            const int n0 = ls_spectrum(x->rho, Q0, 0, lam0);
            const double t0 = ls_tail(X2[0], lam0, n0, cc, M) + lt, t1 = ls_tail(X2[1], lam0, n0, cc, M) + lt;
            free(lam0);
            if (t0 <= lpfa && t1 > lpfa) {
                double per[64], R[64], X[64];
                int nbs[64], ks[64];
                unsigned long long use[64];
                for (int s = 0; s < nf; ++s) { per[s] = p0; nbs[s] = Q0; ks[s] = 1; use[s] = 1ULL << s; }
                if ((rc = ls_batch(x, lags, nl, nf, per, nbs, ks, use, R, X)) != ATKDSP_OK) break;
                int fb = 0;
                for (int s = 1; s < nf; ++s) if (X[s] < X[fb]) fb = s;
                int pp = 0, qq = 0;
                if (!ls_ratio(p0, fams[fb].period_samples, &pp, &qq)) pp = qq = 0;
                ls_mark(seed, fams, fb, pp, qq);
                continue;
            }
        }
        if ((rc = ls_refine(x, lags, nl, seed->period_samples, &p, &Xp)) != ATKDSP_OK) break;
        if (nf) {
            const int Q = ls_round(p);
            double *lam = (double *)malloc((size_t)Q * sizeof(double));
            if (!lam) { rc = ATKDSP_E_NOMEM; break; }
            const int nq = ls_spectrum(x->rho, Q, 0, lam);
            const double t = ls_tail(Xp, lam, nq, cc, M);
            free(lam);
            if (t + lt > lpfa) {
                /* the known families explain it: the one whose removal alone leaves the least */
                double per[64], R[64], X[64];
                int nbs[64], ks[64];
                unsigned long long use[64];
                for (int s = 0; s < nf; ++s) { per[s] = p; nbs[s] = Q; ks[s] = 1; use[s] = 1ULL << s; }
                if ((rc = ls_batch(x, lags, nl, nf, per, nbs, ks, use, R, X)) != ATKDSP_OK) break;
                int fb = 0;
                for (int s = 1; s < nf; ++s) if (X[s] < X[fb]) fb = s;
                int pp = 0, qq = 0;
                if (!ls_ratio(seed->period_samples, fams[fb].period_samples, &pp, &qq)) pp = qq = 0;
                ls_mark(seed, fams, fb, pp, qq);
                continue;
            }
        }
        if (nf >= fcap) break;
        double T, ev;
        uint32_t fl;
        if ((rc = ls_resolve(x, lags, nl, p, M, cc, c->pfa, &T, &fl, &ev)) != ATKDSP_OK) break;
        if (!(T >= (double)c->pmin && T < (double)c->pmax)) fl |= ATKDSP_LAGSCAN_FAM_OUTSIDE;
        const int Q = ls_round(T);
        double per1 = T, R1, X1;
        int nb1 = Q, k1 = 1;
        unsigned long long use1 = ~0ULL;
        if ((rc = ls_batch(x, lags, nl, 1, &per1, &nb1, &k1, &use1, &R1, &X1)) != ATKDSP_OK) break;
        double *lam = (double *)malloc((size_t)Q * sizeof(double));
        double **prof = (double **)calloc((size_t)x->dcap, sizeof(double *));
        if (!lam || !prof) { free(lam); free(prof); rc = ATKDSP_E_NOMEM; break; }
        const int nq = ls_spectrum(x->rho, Q, 0, lam);
        const double noise = M * cc * ls_sum(lam, nq);
        free(lam);
        atkdsp_lagscan_family *F = &fams[nf];
        memset(F, 0, sizeof *F);
        F->period_samples = T;
        F->period_s = T / c->fs;
        F->log10_pfa = seed->log10_pfa;
        F->log10_step = ev;
        F->fold_snr = (X1 - noise) / noise;
        F->seed_base = seed->base;
        if (!ls_ratio(seed->period_samples, T, &F->seed_p, &F->seed_q)) F->seed_p = F->seed_q = 0;
        F->flags = fl;
        F->method[0] = ATKDSP_METHOD_LAGSCAN;
        F->method[1] = ATKDSP_METHOD_SCALE_SURVEY;
        x->subs[nf].T = T; x->subs[nf].Q = Q; x->subs[nf].prof = prof;
        x->nsub = ++nf;
        ls_mark(seed, fams, nf - 1, F->seed_p, F->seed_q);
        for (int j = 0; j < nd; ++j) {
            atkdsp_lagscan_row *r = &rows[ord[j]];
            int p2, q2;
            if (r->family < 0 && ls_ratio(r->period_samples, T, &p2, &q2)) ls_mark(r, fams, nf - 1, p2, q2);
        }
    }
    free(ord);
    *nfam = nf;
    return rc;
}

ptrdiff_t atkdsp_lagscan(const atkdsp_cf32 *y, size_t n, const atkdsp_lagscan_cfg *c,
                         atkdsp_lagscan_row *rows, size_t cap,
                         atkdsp_lagscan_family *fams, size_t fcap, size_t *nfam, double *ntrials) {
    if (nfam) *nfam = 0;
    if (!y || !c || !rows || (fcap && (!fams || !nfam))) return ATKDSP_E_ARG;
    if (!(c->fs > 0.0) || c->pmin < 2 || c->pmin >= c->pmax || c->nlags < 1 || c->nlags > 64
        || c->lag_min < 1 || !(c->lag_frac > 0.0 && c->lag_frac <= 1.0)) return ATKDSP_E_ARG;
    if (c->nmax < 16 || (c->nmax & (c->nmax - 1)) || !(c->pfa > 0.0 && c->pfa < 1.0)
        || !(c->trim >= 0.0 && c->trim < 0.5)) return ATKDSP_E_ARG;
    const int nrow = c->pmax - c->pmin;
    if (cap < (size_t)nrow) return ATKDSP_E_CAP;
    if (fcap > 64) fcap = 64;                 /* the explained test's use mask */
    double rho[LS_ACF + 1];
    ls_acf(y, n, rho);
    int fail = 0;
    int ip;
#ifdef _OPENMP
    int nth = atkdsp_get_threads();
    if (nth < 1) nth = 1;
    int *fails = (int *)calloc((size_t)nrow, sizeof(int));
    if (!fails) return ATKDSP_E_NOMEM;
#   pragma omp parallel for num_threads(nth) schedule(dynamic, 1)
    for (ip = 0; ip < nrow; ++ip) {
        /* the longest base periods first: they cost the most, so the dynamic
         * schedule balances better when they start early */
        const int idx = nrow - 1 - ip;
        if (ls_one(y, n, c, rho, c->pmin + idx, &rows[idx]) != ATKDSP_OK) fails[idx] = 1;
    }
    for (ip = 0; ip < nrow; ++ip) if (fails[ip]) fail = 1;
    free(fails);
#else
    for (ip = 0; ip < nrow; ++ip)
        if (ls_one(y, n, c, rho, c->pmin + ip, &rows[ip]) != ATKDSP_OK) fail = 1;
#endif
    if (fail) return ATKDSP_E_NOMEM;
    double tot = 0.0;
    for (ip = 0; ip < nrow; ++ip) tot += (double)rows[ip].nfold;
    const double lt = tot > 0.0 ? log10(tot) : 0.0, thr = log10(c->pfa);
    for (ip = 0; ip < nrow; ++ip) {
        atkdsp_lagscan_row *r = &rows[ip];
        if (!r->nfold || !(r->shape > 0.0)) continue;
        r->log10_pfa = r->log10_pfa_raw + lt;
        if (r->log10_pfa > 0.0) r->log10_pfa = 0.0;
        if (r->log10_pfa <= thr) r->flags |= ATKDSP_LAGSCAN_DETECT;
    }
    if (ntrials) *ntrials = tot;
    if (!fcap) return (ptrdiff_t)nrow;
    /* the lag values any seed can use: below max(lag_min + 1, lag_frac pmax) + 1 */
    int dcap = (int)floor((double)c->pmax * c->lag_frac);
    if (dcap < c->lag_min + 1) dcap = c->lag_min + 1;
    ls_ctx x;
    x.y = y; x.N = n; x.rho = rho; x.nsub = 0; x.dcap = dcap + 1;
    x.subs = (ls_sub *)calloc(fcap, sizeof(ls_sub));
    if (!x.subs) return ATKDSP_E_NOMEM;
    int nf = 0;
    const int rc = ls_families(&x, c, rows, nrow, tot, fams, (int)fcap, &nf);
    for (size_t s = 0; s < fcap; ++s) {
        if (!x.subs[s].prof) continue;
        for (int d = 0; d < x.dcap; ++d) free(x.subs[s].prof[d]);
        free(x.subs[s].prof);
    }
    free(x.subs);
    if (rc != ATKDSP_OK) return rc;
    *nfam = (size_t)nf;
    return (ptrdiff_t)nrow;
}
