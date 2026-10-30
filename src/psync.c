/* ---- 30. Undertow, pieces 2, 4 and 6: the period-synchronous watch, the
 * bootstrap and the hit blob ----------------------------------------------
 * docs/ATKDSP_PLAN.md §B6c (piece 2's measured note); the contract is
 * atkdsp.h §30 and the twin reference.py §30 (psync, bootstrap,
 * hit_export / hit_import) — this file follows the twin step for step. The
 * candidates accessors live beside their structures (dsss_det.c,
 * undertow.c).
 *
 * OpenMP: MSVC's /openmp is version 2.0, so only 2.0 constructs (parallel
 * for with schedule / num_threads, critical, atomic on x op= expr; signed int
 * loop variables) — "atomic write" (3.1) broke the Windows build of v0.22.0;
 * tests/test_portability.py now refuses anything newer.
 *
 * Determinism: level 0's grid is cut into fixed chains of PS_CHUNK points
 * (each warm-started from the same cold vector), the references and the
 * refinement candidates are independent, and every parallel loop writes its
 * own slot, combined afterwards in index order with a strict ">" — so the
 * answer does not depend on the thread count. */
#include "internal.h"
#include <stdio.h>

typedef struct { double re, im; } pcd;

#define PS_CHUNK  64
#define PS_IT0     6
#define PS_ITP    30
#define PS_ITR    10
#define PS_ITREF  30

void atkdsp_psync_defaults(atkdsp_psync_cfg *c, double fs, double period_s) {
    if (!c) return;
    memset(c, 0, sizeof *c);
    c->fs = fs; c->period_s = period_s; c->tol = 1.0; c->rate_max = 0.0;
    c->pfa = 1e-3; c->esc_pfa = 1e-3;
    c->n0 = 64; c->n0_max = 4096; c->nref = 32; c->max_periods = 0;
}

/* s_i = floor(off + i T + 0.5 r T^2 / fs i^2 + 0.5) — the twin's expression,
 * evaluated in the same order. */
static int64_t ps_start(int64_t i, double T, double r, double fs, double off) {
    const double di = (double)i;
    return (int64_t)floor(off + di * T + 0.5 * r * T * T / fs * di * di + 0.5);
}

static void ps_starts(int64_t *s, int n, double T, double r, double fs, double off) {
    for (int i = 0; i < n; ++i) s[i] = ps_start(i, T, r, fs, off);
}

/* Fixed-count power iteration on the windows x[s_i .. s_i + W):
 * v <- (1/n) sum_i y_i (y_i^H v), deflated against d (unit) when given,
 * normalised; returns |R v| (0 and v untouched if it vanishes). */
static double ps_lam(const atkdsp_cf32 *x, const int64_t *s, int n, int W, pcd *v, pcd *u,
                     int iters, const pcd *d) {
    double lam = 0.0;
    for (int it = 0; it < iters; ++it) {
        for (int k = 0; k < W; ++k) { u[k].re = 0.0; u[k].im = 0.0; }
        for (int i = 0; i < n; ++i) {
            const atkdsp_cf32 *y = x + s[i];
            double cr = 0.0, ci = 0.0;                      /* y^H v */
            for (int k = 0; k < W; ++k) {
                const double yr = y[k].re, yi = y[k].im;
                cr += yr * v[k].re + yi * v[k].im;
                ci += yr * v[k].im - yi * v[k].re;
            }
            for (int k = 0; k < W; ++k) {
                const double yr = y[k].re, yi = y[k].im;
                u[k].re += yr * cr - yi * ci;
                u[k].im += yr * ci + yi * cr;
            }
        }
        const double inv = 1.0 / (double)n;
        for (int k = 0; k < W; ++k) { u[k].re *= inv; u[k].im *= inv; }
        if (d) {
            double pr = 0.0, pi = 0.0;                      /* d^H u */
            for (int k = 0; k < W; ++k) {
                pr += d[k].re * u[k].re + d[k].im * u[k].im;
                pi += d[k].re * u[k].im - d[k].im * u[k].re;
            }
            for (int k = 0; k < W; ++k) {
                u[k].re -= d[k].re * pr - d[k].im * pi;
                u[k].im -= d[k].re * pi + d[k].im * pr;
            }
        }
        double nu = 0.0;
        for (int k = 0; k < W; ++k) nu += u[k].re * u[k].re + u[k].im * u[k].im;
        nu = sqrt(nu);
        if (!(nu > 0.0)) return 0.0;
        for (int k = 0; k < W; ++k) { v[k].re = u[k].re / nu; v[k].im = u[k].im / nu; }
        lam = nu;
    }
    return lam;
}

static void ps_cold(const atkdsp_cf32 *x, int W, pcd *v) {
    double nv = 0.0;
    for (int k = 0; k < W; ++k) { v[k].re = x[k].re; v[k].im = x[k].im; nv += v[k].re * v[k].re + v[k].im * v[k].im; }
    nv = sqrt(nv);
    if (nv > 0.0) { for (int k = 0; k < W; ++k) { v[k].re /= nv; v[k].im /= nv; } }
    else { const double a = 1.0 / sqrt((double)W); for (int k = 0; k < W; ++k) { v[k].re = a; v[k].im = 0.0; } }
}

/* periods whose wide window fits, the period stretched by tol and the rate */
static int64_t ps_avail(size_t N, int W, double T, double tol, double r, double fs) {
    double q = floor(((double)N - (double)W - 2.0) / (T + tol));
    if (!(q > 0.0)) return 0;
    if (q > 2e9) q = 2e9;
    int64_t n = (int64_t)q;
    while (n > 0 && ps_start(n - 1, T + tol, fabs(r), fs, 0.0) + W > (int64_t)N) n--;
    return n > 0 ? n : 0;
}

typedef struct { double a, b, sd, xbar, sxx; int k; } ps_h0;

static double ps_eps(int idx) {                      /* idx = 2 j + (0: -, 1: +) */
    const int j = idx / 2;
    const double e = 0.010 + 0.003 * (double)j;
    return (idx & 1) ? e : -e;
}

static double ps_eps_max(int nref) { return 0.010 + 0.003 * (double)(nref / 2 - 1); }

/* H0 model around Tc: lambda1 at Tc (1 + eps), a least-squares line in T - Tc. */
static int ps_refs(const atkdsp_cf32 *x, size_t N, double fs, double Tc, int n, int W, int nref,
                   ps_h0 *h, int *evals) {
    double *xs = (double *)malloc((size_t)nref * sizeof *xs), *ls = (double *)malloc((size_t)nref * sizeof *ls);
    int *ok = (int *)calloc((size_t)nref, sizeof *ok);
    int fail = 0;
    if (!xs || !ls || !ok) { free(xs); free(ls); free(ok); return ATKDSP_E_NOMEM; }
    int j;
#pragma omp parallel for num_threads(atkdsp_get_threads()) schedule(dynamic, 1)
    for (j = 0; j < nref; ++j) {
        const double Tr = Tc * (1.0 + ps_eps(j));
        if (ps_start(n - 1, Tr, 0.0, fs, 0.0) + W > (int64_t)N) continue;
        int64_t *s = (int64_t *)malloc((size_t)n * sizeof *s);
        pcd *v = (pcd *)malloc((size_t)W * sizeof *v), *u = (pcd *)malloc((size_t)W * sizeof *u);
        if (!s || !v || !u) { free(s); free(v); free(u);
#pragma omp atomic
            fail |= 1;
            continue; }
        ps_starts(s, n, Tr, 0.0, fs, 0.0);
        ps_cold(x, W, v);
        ls[j] = ps_lam(x, s, n, W, v, u, PS_ITREF, NULL);
        xs[j] = Tr - Tc;
        ok[j] = 1;
        free(s); free(v); free(u);
    }
    if (fail) { free(xs); free(ls); free(ok); return ATKDSP_E_NOMEM; }
    memset(h, 0, sizeof *h);
    int k = 0;
    double sx = 0.0, sl = 0.0;
    for (j = 0; j < nref; ++j) if (ok[j]) { sx += xs[j]; sl += ls[j]; k++; }
    *evals += k;
    h->k = k;
    if (k >= 4) {
        const double xbar = sx / k, lbar = sl / k;
        double sxx = 0.0, sxl = 0.0;
        for (j = 0; j < nref; ++j) if (ok[j]) { sxx += (xs[j] - xbar) * (xs[j] - xbar); sxl += (xs[j] - xbar) * (ls[j] - lbar); }
        const double b = sxx > 0.0 ? sxl / sxx : 0.0, a = lbar - b * xbar;
        double ss = 0.0;
        for (j = 0; j < nref; ++j) if (ok[j]) { const double e = ls[j] - (a + b * xs[j]); ss += e * e; }
        h->a = a; h->b = b; h->xbar = xbar; h->sxx = sxx; h->sd = sqrt(ss / (double)(k - 2));
    }
    free(xs); free(ls); free(ok);
    return ATKDSP_OK;
}

/* one-sided predictive-t tail (k - 2 dof) of lam against the H0 line at T,
 * times nhyp, clamped at 0; *t_out the t */
static double ps_tail(double lam, double T, double Tc, const ps_h0 *h, double nhyp, double *t_out) {
    *t_out = 0.0;
    if (h->k < 4 || !(h->sd > 0.0)) return 0.0;
    const double x = T - Tc;
    const double lev = 1.0 / h->k + (h->sxx > 0.0 ? (x - h->xbar) * (x - h->xbar) / h->sxx : 0.0);
    const double t = (lam - (h->a + h->b * x)) / (h->sd * sqrt(1.0 + lev));
    *t_out = t;
    if (!(t > 0.0)) return 0.0;
    const double l10 = atkdsp_t_log10_sf2(t, (double)(h->k - 2)) - log10(2.0) + log10(nhyp > 1.0 ? nhyp : 1.0);
    return l10 < 0.0 ? l10 : 0.0;
}

/* psync with the eigenvector returned in double (vout, W values) — the
 * bootstrap works on it before any float rounding, as the twin does. */
static int ps_core(const atkdsp_cf32 *x, size_t N, const atkdsp_psync_cfg *c, pcd *vout, int Wcap,
                   atkdsp_psync_result *out) {
    memset(out, 0, sizeof *out);
    const double fs = c->fs, T0 = c->period_s * c->fs, tol = c->tol, rmax = c->rate_max;
    if (!(fs > 0.0) || !(T0 >= 4.0) || !(tol > 0.0) || !(tol <= 0.25 * T0) || !(rmax >= 0.0)
        || !(c->pfa > 0.0) || !(c->pfa < 0.5)) return ATKDSP_E_ARG;
    if (c->n0 < 8 || c->n0 > c->n0_max || c->nref < 8 || c->nref > 64 || (c->nref & 1)
        || !(c->esc_pfa > 0.0) || !(c->esc_pfa < 0.5) || c->max_periods < 0) return ATKDSP_E_ARG;
    const int W = 2 * (int)floor(T0 + 0.5);
    if (Wcap < W) return ATKDSP_E_CAP;
    const double em = ps_eps_max(c->nref);
    int64_t navail = ps_avail(N, W, (T0 + tol) * (1.0 + em), 0.0, 0.0, fs);
    if (c->max_periods && navail > c->max_periods) navail = c->max_periods;
    if (navail < c->n0) return ATKDSP_E_EMPTY;
    out->window = W; out->snr_db = -99.0;
    out->method[0] = ATKDSP_METHOD_PSYNC; out->method[1] = ATKDSP_METHOD_EIGEN_WIDE;

    int rc = ATKDSP_OK, evals = 0, levels = 0;
    int n = c->n0;
    const int nmax = (int)(navail < 2000000000 ? navail : 2000000000);
    int64_t *s = (int64_t *)malloc((size_t)nmax * sizeof *s);
    pcd *v = (pcd *)malloc((size_t)W * sizeof *v), *u = (pcd *)malloc((size_t)W * sizeof *u);
    pcd *v2 = (pcd *)malloc((size_t)W * sizeof *v2), *vb = (pcd *)malloc((size_t)W * sizeof *vb);
    if (!s || !v || !u || !v2 || !vb) { rc = ATKDSP_E_NOMEM; goto done; }

    ps_h0 h0;
    double lam = 0.0, z = 0.0, T = T0, l10_0 = 0.0;
    int ok0 = 0, edge = 0;
    for (;;) {                                           /* ---- level 0, escalating */
        const double step = 0.5 / (double)n;
        const int K = (int)floor(tol / step + 1e-9);
        if ((rc = ps_refs(x, N, fs, T0, n, W, c->nref, &h0, &evals)) != ATKDSP_OK) goto done;
        const double bslope = h0.b;
        const int npts = 2 * K + 1, nch = (npts + PS_CHUNK - 1) / PS_CHUNK;
        double *bsc = (double *)malloc((size_t)nch * sizeof *bsc);
        int *bkk = (int *)malloc((size_t)nch * sizeof *bkk);
        pcd *bv = (pcd *)malloc((size_t)nch * (size_t)W * sizeof *bv);
        int fail = 0;
        if (!bsc || !bkk || !bv) { free(bsc); free(bkk); free(bv); rc = ATKDSP_E_NOMEM; goto done; }
        int ch;
#pragma omp parallel for num_threads(atkdsp_get_threads()) schedule(dynamic, 1)
        for (ch = 0; ch < nch; ++ch) {
            int64_t *ss = (int64_t *)malloc((size_t)n * sizeof *ss);
            pcd *cv = (pcd *)malloc((size_t)W * sizeof *cv), *cu = (pcd *)malloc((size_t)W * sizeof *cu);
            bsc[ch] = -HUGE_VAL; bkk[ch] = 0;
            if (!ss || !cv || !cu) { free(ss); free(cv); free(cu);
#pragma omp atomic
                fail |= 1;
                continue; }
            ps_cold(x, W, cv);
            const int k0 = -K + ch * PS_CHUNK, k1 = (k0 + PS_CHUNK - 1 < K) ? k0 + PS_CHUNK - 1 : K;
            for (int kk = k0; kk <= k1; ++kk) {
                ps_starts(ss, n, T0 + kk * step, 0.0, fs, 0.0);
                const double l = ps_lam(x, ss, n, W, cv, cu, PS_IT0, NULL);
                const double sc = l - bslope * (kk * step);
                if (sc > bsc[ch]) { bsc[ch] = sc; bkk[ch] = kk; memcpy(bv + (size_t)ch * W, cv, (size_t)W * sizeof *cv); }
            }
            free(ss); free(cv); free(cu);
        }
        if (fail) { free(bsc); free(bkk); free(bv); rc = ATKDSP_E_NOMEM; goto done; }
        int bi = 0;
        for (ch = 1; ch < nch; ++ch) if (bsc[ch] > bsc[bi]) bi = ch;
        const int bestkk = bkk[bi];
        memcpy(v, bv + (size_t)bi * W, (size_t)W * sizeof *v);
        free(bsc); free(bkk); free(bv);
        evals += npts;
        T = T0 + bestkk * step;
        ps_starts(s, n, T, 0.0, fs, 0.0);
        lam = ps_lam(x, s, n, W, v, u, PS_ITP, NULL);
        evals += 1;
        l10_0 = ps_tail(lam, T, T0, &h0, (double)npts, &z);
        levels++;
        edge = (bestkk >= K || bestkk <= -K);
        ok0 = l10_0 <= log10(c->esc_pfa);
        const int lim = c->n0_max < nmax ? c->n0_max : nmax;
        if (ok0 || n >= lim) break;
        n = 2 * n < lim ? 2 * n : lim;
    }
    out->n0_used = n;
    if (edge) out->flags |= ATKDSP_PS_EDGE;
    double r = 0.0, nrate = 1.0, Tc = T0;
    if (!ok0) {
        out->flags |= ATKDSP_PS_BUDGET;
    } else {                                             /* ---- refine to all the data */
        int64_t nall = nmax;
        if (rmax > 0.0) { const int64_t a = ps_avail(N, W, (T + 0.5) * (1.0 + em), 0.0, rmax, fs); if (a < nall) nall = a; }
        while (n < nall) {
            n = 2 * (int64_t)n < nall ? 2 * n : (int)nall;
            const double step = 0.5 / (double)n;
            double rates[13]; int nr = 0;
            rates[nr++] = r;
            if (rmax > 0.0) {
                const double rstep = 0.5 * fs / (T * T * (double)n * (double)n);
                /* active once a rate of rmax could slip a quarter sample over the span
                   (rmax >= rstep); then always +-6 steps — each level covers +-1.5 of the
                   previous level's step, so a rate the coarse level chose by noise can
                   walk back (v0.22: capping the steps at rmax / rstep could not) */
                const int kr = rmax >= rstep ? 6 : 0;
                if (kr >= 1) {
                    nr = 0;
                    for (int jj = -kr; jj <= kr; ++jj) {
                        const double rr = r + jj * rstep;
                        if (fabs(rr) <= rmax * (1.0 + 1e-12)) rates[nr++] = rr;
                    }
                    out->flags |= ATKDSP_PS_RATE;
                    const double nrt = floor(1.0 + 2.0 * rmax / rstep);
                    if (nrt > nrate) nrate = nrt;
                }
            }
            const int ncand = nr * 13;                     /* 13 periods (+-6 steps) per rate */
            double *cl = (double *)malloc((size_t)ncand * sizeof *cl);
            pcd *cvv = (pcd *)malloc((size_t)ncand * (size_t)W * sizeof *cvv);
            int fail = 0;
            if (!cl || !cvv) { free(cl); free(cvv); rc = ATKDSP_E_NOMEM; goto done; }
            int ci;
#pragma omp parallel for num_threads(atkdsp_get_threads()) schedule(dynamic, 1)
            for (ci = 0; ci < ncand; ++ci) {
                const double rr = rates[ci / 13], Tq = T + (ci % 13 - 6) * step;
                int64_t *ss = (int64_t *)malloc((size_t)n * sizeof *ss);
                pcd *cu = (pcd *)malloc((size_t)W * sizeof *cu);
                pcd *cv = cvv + (size_t)ci * W;
                if (!ss || !cu) { free(ss); free(cu); cl[ci] = -HUGE_VAL;
#pragma omp atomic
                    fail |= 1;
                    continue; }
                memcpy(cv, v, (size_t)W * sizeof *cv);
                ps_starts(ss, n, Tq, rr, fs, 0.0);
                cl[ci] = ps_lam(x, ss, n, W, cv, cu, PS_ITR, NULL);
                free(ss); free(cu);
            }
            if (fail) { free(cl); free(cvv); rc = ATKDSP_E_NOMEM; goto done; }
            int bi = -1; double bl = -1.0;
            for (ci = 0; ci < ncand; ++ci) if (cl[ci] > bl) { bl = cl[ci]; bi = ci; }
            if (bi >= 0) {
                r = rates[bi / 13]; T = T + (bi % 13 - 6) * step;
                memcpy(v, cvv + (size_t)bi * W, (size_t)W * sizeof *v);
            }
            free(cl); free(cvv);
            evals += ncand;
            levels++;
        }
        ps_starts(s, n, T, r, fs, 0.0);
        lam = ps_lam(x, s, n, W, v, u, PS_ITP, NULL);
        if ((rc = ps_refs(x, N, fs, T, n, W, c->nref, &h0, &evals)) != ATKDSP_OK) goto done;
        evals += 1;
        Tc = T;
    }
    /* ---- lambda2, the re-cut and the dominance */
    {
        const double mu = h0.a + h0.b * (T - Tc);
        double off = 0.0;
        ps_starts(s, n, T, r, fs, 0.0);
        const int64_t it = (int64_t)floor(T);
        if ((int64_t)N > it + W) ps_cold(x + it, W, v2); else ps_cold(x, W, v2);
        double pr = 0.0, pi = 0.0, nv = 0.0;              /* v2 -= v (v^H v2) */
        for (int k = 0; k < W; ++k) { pr += v[k].re * v2[k].re + v[k].im * v2[k].im; pi += v[k].re * v2[k].im - v[k].im * v2[k].re; }
        for (int k = 0; k < W; ++k) { v2[k].re -= v[k].re * pr - v[k].im * pi; v2[k].im -= v[k].re * pi + v[k].im * pr; nv += v2[k].re * v2[k].re + v2[k].im * v2[k].im; }
        nv = sqrt(nv); if (!(nv > 1e-300)) nv = 1e-300;
        for (int k = 0; k < W; ++k) { v2[k].re /= nv; v2[k].im /= nv; }
        memcpy(vb, v2, (size_t)W * sizeof *vb);           /* the start vector, kept for the re-cut */
        double lam2 = ps_lam(x, s, n, W, vb, u, PS_ITP, v);
        evals += 1;
        double dom = lam > mu ? (lam2 - mu) / (lam - mu) : 0.0;
        if (ok0 && dom > 0.7 && ps_start(n - 1, T, r, fs, 0.5 * T) + W <= (int64_t)N) {
            int64_t *s2 = (int64_t *)malloc((size_t)n * sizeof *s2);
            pcd *w1 = (pcd *)malloc((size_t)W * sizeof *w1), *w2 = (pcd *)malloc((size_t)W * sizeof *w2);
            if (!s2 || !w1 || !w2) { free(s2); free(w1); free(w2); rc = ATKDSP_E_NOMEM; goto done; }
            ps_starts(s2, n, T, r, fs, 0.5 * T);
            memcpy(w1, v, (size_t)W * sizeof *w1);
            const double l1b = ps_lam(x, s2, n, W, w1, u, PS_ITP, NULL);
            double qr = 0.0, qi = 0.0, nq = 0.0;          /* w2 = v2 - w1 (w1^H v2) */
            for (int k = 0; k < W; ++k) { qr += w1[k].re * v2[k].re + w1[k].im * v2[k].im; qi += w1[k].re * v2[k].im - w1[k].im * v2[k].re; }
            for (int k = 0; k < W; ++k) { w2[k].re = v2[k].re - (w1[k].re * qr - w1[k].im * qi); w2[k].im = v2[k].im - (w1[k].re * qi + w1[k].im * qr); nq += w2[k].re * w2[k].re + w2[k].im * w2[k].im; }
            nq = sqrt(nq); if (!(nq > 1e-300)) nq = 1e-300;
            for (int k = 0; k < W; ++k) { w2[k].re /= nq; w2[k].im /= nq; }
            const double l2b = ps_lam(x, s2, n, W, w2, u, PS_ITP, w1);
            evals += 2;
            const double domb = l1b > mu ? (l2b - mu) / (l1b - mu) : 0.0;
            if (domb < dom) {
                lam = l1b; lam2 = l2b; memcpy(v, w1, (size_t)W * sizeof *v); dom = domb; off = 0.5 * T;
                memcpy(s, s2, (size_t)n * sizeof *s);
                out->flags |= ATKDSP_PS_RECUT;
            }
            free(s2); free(w1); free(w2);
        }
        double sig2 = 0.0;
        for (int i = 0; i < n; ++i) {
            const atkdsp_cf32 *y = x + s[i];
            for (int k = 0; k < W; ++k) sig2 += (double)y[k].re * y[k].re + (double)y[k].im * y[k].im;
        }
        sig2 /= (double)n * (double)W;
        double nh = 4.0 * tol * (double)n; if (nh < 1.0) nh = 1.0;
        const double l10 = ps_tail(lam, T, Tc, &h0, nh * nrate, &z);
        out->period_samples = T; out->period_s = T / fs; out->rate = r;
        out->lambda1 = lam; out->lambda2 = lam2; out->ref_mean = mu; out->ref_sd = h0.sd;
        out->z = z; out->log10_pfa = l10; out->sigma2 = sig2; out->dominance = dom;
        out->window_offset = off; out->nperiods = n; out->evals = evals; out->levels = levels;
        if (lam > mu && sig2 > 0.0) out->snr_db = 10.0 * log10((lam - mu) / (T * sig2));
        if (ok0 && l10 <= log10(c->pfa)) {
            out->flags |= ATKDSP_PS_DETECT;
            if (dom > 0.7) out->flags |= ATKDSP_PS_MULTI;
        }
        if (out->flags & ATKDSP_PS_RATE) out->method[2] = ATKDSP_METHOD_LAG_RATE;
        memcpy(vout, v, (size_t)W * sizeof *v);
    }
done:
    free(s); free(v); free(u); free(v2); free(vb);
    return rc;
}

int atkdsp_psync(const atkdsp_cf32 *x, size_t n, const atkdsp_psync_cfg *cfg,
                 atkdsp_cf32 *wave, size_t cap, atkdsp_psync_result *out) {
    if (!x || !cfg || !wave || !out) return ATKDSP_E_ARG;
    const double T0 = cfg->period_s * cfg->fs;
    if (!(T0 >= 4.0) || !(T0 < 1e8)) { memset(out, 0, sizeof *out); return ATKDSP_E_ARG; }
    const int W = 2 * (int)floor(T0 + 0.5);
    if (cap < (size_t)W) { memset(out, 0, sizeof *out); return ATKDSP_E_CAP; }
    pcd *v = (pcd *)malloc((size_t)W * sizeof *v);
    if (!v) return ATKDSP_E_NOMEM;
    const int rc = ps_core(x, n, cfg, v, W, out);
    if (rc == ATKDSP_OK)
        for (int k = 0; k < W; ++k) { wave[k].re = (float)v[k].re; wave[k].im = (float)v[k].im; }
    free(v);
    return rc;
}

/* ---- 30b. the bootstrap --------------------------------------------------- */
#define BS_LSLICE 1024
#define BS_NPHASE 16

void atkdsp_bootstrap_defaults(atkdsp_bootstrap_cfg *c, double fs, double period_s, double bw_hz) {
    if (!c) return;
    memset(c, 0, sizeof *c);
    atkdsp_psync_defaults(&c->ps, fs, period_s);
    c->bw_hz = bw_hz; c->rc_hz = 0.0;
    c->chip_mod = ATKDSP_CHIP_BPSK; c->vote_passes = 6; c->poly_max_degree = 14;
    c->code_pfa = 1e-3;
}

/* argmax_t of the energy of v[t .. t+m0) (cumulative sums, the twin's order) */
static int bs_support(const pcd *v, int W, int m0) {
    double *cs = (double *)malloc(((size_t)W + 1) * sizeof *cs);
    if (!cs) return 0;
    cs[0] = 0.0;
    for (int i = 0; i < W; ++i) cs[i + 1] = cs[i] + (v[i].re * v[i].re + v[i].im * v[i].im);
    int best = 0; double bv = cs[m0] - cs[0];
    for (int t = 1; t + m0 <= W; ++t) { const double e = cs[t + m0] - cs[t]; if (e > bv) { bv = e; best = t; } }
    free(cs);
    return best;
}

ptrdiff_t atkdsp_bootstrap(const atkdsp_cf32 *x, size_t n, const atkdsp_bootstrap_cfg *cfg,
                           int8_t *chips, size_t cap, atkdsp_bootstrap_result *out) {
    if (!x || !cfg || !out || (!chips && cap)) return ATKDSP_E_ARG;
    memset(out, 0, sizeof *out);
    const double fs = cfg->ps.fs, bw = cfg->bw_hz, rcg = cfg->rc_hz;
    if (!(bw > 0.0 || rcg > 0.0) || cfg->chip_mod < 0 || cfg->chip_mod > 2) return ATKDSP_E_ARG;
    const double T0 = cfg->ps.period_s * fs;
    if (!(T0 >= 4.0) || !(T0 < 1e8)) return ATKDSP_E_ARG;
    out->method[0] = ATKDSP_METHOD_BOOTSTRAP; out->method[1] = ATKDSP_METHOD_PSYNC;
    const int W = 2 * (int)floor(T0 + 0.5);
    pcd *v = (pcd *)malloc((size_t)W * sizeof *v);
    if (!v) return ATKDSP_E_NOMEM;
    int rc = ps_core(x, n, &cfg->ps, v, W, &out->ps);
    if (rc != ATKDSP_OK) { free(v); return rc; }
    out->harmonic = 1;
    if ((out->ps.flags & ATKDSP_PS_DETECT) && (out->ps.flags & ATKDSP_PS_MULTI)) {
        /* a multiple of the period? try T/2, T/3, T/4 — the first clean detection wins */
        const double Tm = out->ps.period_samples;
        for (int k = 2; k <= 4; ++k) {
            const double Tk = Tm / (double)k;
            if (Tk < 8.0) break;
            atkdsp_psync_cfg c2 = cfg->ps;
            c2.period_s = Tk / fs; c2.tol = 0.5;
            const int Wk = 2 * (int)floor(Tk + 0.5);
            pcd *vk = (pcd *)malloc((size_t)Wk * sizeof *vk);
            if (!vk) { free(v); return ATKDSP_E_NOMEM; }
            atkdsp_psync_result pk;
            const int rk = ps_core(x, n, &c2, vk, Wk, &pk);
            if (rk == ATKDSP_OK && (pk.flags & ATKDSP_PS_DETECT) && !(pk.flags & ATKDSP_PS_MULTI)) {
                free(v); v = vk; out->ps = pk; out->harmonic = k; out->flags |= ATKDSP_BS_HARMONIC;
                break;
            }
            free(vk);
            if (rk == ATKDSP_E_NOMEM) { free(v); return rk; }
        }
    }
    if (!(out->ps.flags & ATKDSP_PS_DETECT) || (out->ps.flags & ATKDSP_PS_MULTI)) {
        out->flags |= ATKDSP_BS_REFUSED; free(v); return 0;
    }
    out->flags |= ATKDSP_BS_DETECT;
    const double T = out->ps.period_samples, r = out->ps.rate;
    const int Wv = out->ps.window;                     /* the eigenvector's length (the harmonic's, if taken) */
    const int m0 = (int)floor(T + 0.5);
    if (m0 < 4 || m0 > Wv) { out->flags |= ATKDSP_BS_REFUSED; free(v); return 0; }
    const int t0 = bs_support(v, Wv, m0);
    const pcd *seg = v + t0;
    const int P = cfg->chip_mod == ATKDSP_CHIP_BPSK ? 2 : 4;
    /* the carrier: the P-th power of the support, one tone at P f */
    double f = 0.0;
    {
        pcd *q = (pcd *)malloc((size_t)m0 * sizeof *q);
        if (!q) { free(v); return ATKDSP_E_NOMEM; }
        for (int i = 0; i < m0; ++i) {
            double wr = seg[i].re, wi = seg[i].im;
            for (int p = 1; p < P; ++p) { const double tr = wr * seg[i].re - wi * seg[i].im; wi = wr * seg[i].im + wi * seg[i].re; wr = tr; }
            q[i].re = wr; q[i].im = wi;
        }
        rc = atk_code_tone((const double *)q, (size_t)m0, &f);
        free(q);
        if (rc != ATKDSP_OK) { free(v); return rc; }
        f /= (double)P;
    }
    /* derotate and resample so a period is exactly m0 samples */
    const double fs2 = fs * (double)m0 / T;
    atkdsp_cf32 *xd = (atkdsp_cf32 *)malloc((n ? n : 1) * sizeof *xd);
    atkdsp_arb_resampler *rs = atkdsp_arb_resampler_create(fs, fs2, 60.0, 64);
    if (!xd || !rs) { free(xd); atkdsp_arb_resampler_destroy(rs); free(v); return ATKDSP_E_NOMEM; }
    for (size_t i = 0; i < n; ++i) {
        const double ph = -2.0 * ATK_PI * f * (double)i, cr = cos(ph), ci = sin(ph);
        const double re = (double)x[i].re * cr - (double)x[i].im * ci, im = (double)x[i].re * ci + (double)x[i].im * cr;
        xd[i].re = (float)re; xd[i].im = (float)im;
    }
    const size_t zcap = atkdsp_arb_resampler_out_max(rs, n) + 64 + n / 16;
    atkdsp_cf32 *zb = (atkdsp_cf32 *)malloc(zcap * sizeof *zb);
    if (!zb) { free(xd); atkdsp_arb_resampler_destroy(rs); free(v); return ATKDSP_E_NOMEM; }
    size_t nz = 0;
    if (r == 0.0) {
        const ptrdiff_t got = atkdsp_arb_resampler_process(rs, xd, n, zb, zcap);
        if (got < 0) { rc = (int)got; goto fail; }
        nz = (size_t)got;
    } else {
        for (size_t b0 = 0; b0 < n; b0 += 4096) {
            const size_t len = n - b0 < 4096 ? n - b0 : 4096;
            const double tm = ((double)b0 + 2048.0) / fs;
            atkdsp_arb_resampler_set_ratio(rs, (double)m0 / (T * (1.0 + r * tm)));
            const ptrdiff_t got = atkdsp_arb_resampler_process(rs, xd + b0, len, zb + nz, zcap - nz);
            if (got < 0) { rc = (int)got; goto fail; }
            nz += (size_t)got;
        }
    }
    free(xd); xd = NULL;
    atkdsp_arb_resampler_destroy(rs); rs = NULL;
    {
        atkdsp_dsss_code_config cc;
        atkdsp_dsss_code_defaults(&cc, fs2, fs2 / (double)m0, (double)m0 / fs2);
        cc.chip_mod = cfg->chip_mod; cc.vote_passes = cfg->vote_passes;
        cc.poly_max_degree = cfg->poly_max_degree; cc.pfa = cfg->code_pfa;
        int L = 0; double line = 0.0;
        if (rcg > 0.0) {
            L = (int)floor(rcg * T / fs + 0.5);
            out->flags |= ATKDSP_BS_RC_GIVEN;
        } else {
            int lmin = (int)ceil(0.4 * bw * T / fs); if (lmin < 2) lmin = 2;
            int lmax = (int)floor(1.25 * bw * T / fs); if (lmax > m0) lmax = m0;
            if (lmin <= lmax) {
                /* two chip-count scores from the support (derotated, phase-aligned
                   real): the chip-centre envelope (band-limited chips) and the
                   transition line (sharp chips) — each works where the other fails */
                const int nL = lmax - lmin + 1;
                double *env = (double *)malloc((size_t)nL * sizeof *env), *lnv = (double *)malloc((size_t)nL * sizeof *lnv);
                double *dr = (double *)malloc(((size_t)m0 + 1) * sizeof *dr), *ta = (double *)malloc((size_t)m0 * sizeof *ta);
                pcd *d = (pcd *)malloc((size_t)m0 * sizeof *d);
                if (!env || !lnv || !dr || !ta || !d) { free(env); free(lnv); free(dr); free(ta); free(d); rc = ATKDSP_E_NOMEM; goto fail; }
                double s2r = 0.0, s2i = 0.0;
                for (int i = 0; i < m0; ++i) {
                    const double ph = -2.0 * ATK_PI * f * (double)i, cr = cos(ph), ci = sin(ph);
                    d[i].re = seg[i].re * cr - seg[i].im * ci; d[i].im = seg[i].re * ci + seg[i].im * cr;
                    s2r += d[i].re * d[i].re - d[i].im * d[i].im; s2i += 2.0 * d[i].re * d[i].im;
                }
                for (int i = 0; i < m0; ++i) {                 /* |d[t+1] - d[t]|, circular */
                    const int i1 = (i + 1) % m0;
                    const double er = d[i1].re - d[i].re, ei = d[i1].im - d[i].im;
                    ta[i] = sqrt(er * er + ei * ei);
                }
                const double th = 0.5 * atan2(s2i, s2r), ct = cos(-th), st = sin(-th);
                double ms = 0.0;
                dr[0] = 0.0;
                for (int i = 0; i < m0; ++i) { dr[i] = d[i].re * ct - d[i].im * st; ms += dr[i] * dr[i]; }
                dr[m0] = dr[0];
                double rms = sqrt(ms / (double)m0); if (!(rms > 0.0)) rms = 1e-300;
                for (int li = 0; li < nL; ++li) {
                    const int Lc = lmin + li;
                    const double sps = T / (double)Lc;
                    double best = 0.0;
                    for (int j = 0; j < BS_NPHASE; ++j) {
                        double acc = 0.0;
                        for (int k = 0; k < Lc; ++k) {
                            const double pos = fmod((double)j * sps / (double)BS_NPHASE + (double)k * sps, (double)m0);
                            const int i0 = (int)floor(pos);
                            const double fr = pos - (double)i0;
                            acc += fabs(dr[i0] * (1.0 - fr) + dr[i0 + 1] * fr);
                        }
                        const double sc = acc / (double)Lc / rms;
                        if (sc > best) best = sc;
                    }
                    env[li] = best;
                    double lr = 0.0, lim = 0.0;
                    for (int t = 0; t < m0; ++t) {
                        const double ph = -2.0 * ATK_PI * (double)Lc * (double)t / T;
                        lr += ta[t] * cos(ph); lim += ta[t] * sin(ph);
                    }
                    lnv[li] = sqrt(lr * lr + lim * lim);
                }
                free(dr); free(d); free(ta);
                /* the shortlist: env 1, line 1, env 2, line 2, env 3, line 3 (each L once) */
                int eo[3] = { -1, -1, -1 }, lo[3] = { -1, -1, -1 }, sh[6], nsh = 0;
                for (int q = 0; q < 3 && q < nL; ++q) {
                    int b = -1, c2 = -1;
                    for (int li = 0; li < nL; ++li) {
                        int used = 0, usedl = 0;
                        for (int u = 0; u < q; ++u) { used |= eo[u] == li; usedl |= lo[u] == li; }
                        if (!used && (b < 0 || env[li] > env[b])) b = li;
                        if (!usedl && (c2 < 0 || lnv[li] > lnv[c2])) c2 = li;
                    }
                    eo[q] = b; lo[q] = c2;
                }
                for (int q = 0; q < 3; ++q) {
                    const int cand[2] = { eo[q], lo[q] };
                    for (int u = 0; u < 2; ++u) {
                        if (cand[u] < 0) continue;
                        int dup = 0;
                        for (int w2 = 0; w2 < nsh; ++w2) dup |= sh[w2] == cand[u];
                        if (!dup) sh[nsh++] = cand[u];
                    }
                }
                const size_t nzs = nz < (size_t)BS_LSLICE * m0 + 2 * (size_t)m0 ? nz : (size_t)BS_LSLICE * m0 + 2 * (size_t)m0;
                int8_t *tch = (int8_t *)malloc(2 * (size_t)m0 + 2);
                if (!tch) { free(env); free(lnv); rc = ATKDSP_E_NOMEM; goto fail; }
                int pick = eo[0];
                for (int q = 0; q < nsh; ++q) {
                    const int Lq = lmin + sh[q];
                    atkdsp_dsss_code_config c2 = cc;
                    c2.rc = (double)Lq * fs2 / (double)m0;
                    atkdsp_dsss_code_result r2;
                    const ptrdiff_t got = atkdsp_dsss_code(zb, nzs, &c2, tch, 2 * (size_t)m0 + 2, &r2);
                    if (got < 0) continue;
                    if (r2.L == Lq && (r2.flags & (ATKDSP_CODE_NAMED | ATKDSP_CODE_POLY))) { pick = sh[q]; break; }
                }
                free(tch);
                if (pick >= 0) {
                    L = lmin + pick;
                    double mo = 0.0; int any = 0;
                    for (int li = 0; li < nL; ++li) if (li != pick) { if (!any || env[li] > mo) mo = env[li]; any = 1; }
                    line = (any && mo > 0.0) ? env[pick] / mo : 0.0;
                }
                free(lnv);
                free(env);
            }
        }
        out->cfo_hz = f * fs; out->fs_code = fs2; out->support_start = t0;
        if (L < 2 || L > m0) { out->flags |= ATKDSP_BS_REFUSED; free(zb); free(v); return 0; }
        cc.rc = (double)L * fs2 / (double)m0;
        const ptrdiff_t got = atkdsp_dsss_code(zb, nz, &cc, chips, cap, &out->code);
        out->rc_hz = (double)L * fs / T; out->line_ratio = line; out->L = L;
        if (got >= 0) out->cfo_hz = f * fs + out->code.cfo_hz;   /* coarse + dsss_code's fine (Hz survive the resampling) */
        free(zb); free(v);
        if (got < 0) return got;
        if (out->code.L > 0 && !(out->code.flags & ATKDSP_CODE_REFUSED) && got > 0) {
            out->flags |= ATKDSP_BS_CODE;
            return got;
        }
        return 0;
    }
fail:
    free(xd); atkdsp_arb_resampler_destroy(rs); free(zb); free(v);
    return rc;
}

/* ---- 30c. the hit blob ------------------------------------------------------
 * Version 1, little-endian, fixed field order (atkdsp.h §30). */
static const char UT_MAGIC[8] = { 'A', 'T', 'K', 'U', 'T', 'H', 'I', 'T' };

static uint32_t ut_crc32(const uint8_t *p, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return c ^ 0xFFFFFFFFu;
}

typedef struct { uint8_t *p; size_t n, cap; } ut_w;
static void w_raw(ut_w *w, const void *src, size_t k) {
    if (w->p && w->n + k <= w->cap) memcpy(w->p + w->n, src, k);
    w->n += k;
}
static void w_u8le(ut_w *w, uint64_t v, int bytes) {
    uint8_t b[8];
    for (int i = 0; i < bytes; ++i) b[i] = (uint8_t)(v >> (8 * i));
    w_raw(w, b, (size_t)bytes);
}
static void w_f64(ut_w *w, double d) { uint64_t u; memcpy(&u, &d, 8); w_u8le(w, u, 8); }
static void w_i32(ut_w *w, int32_t v) { w_u8le(w, (uint32_t)v, 4); }
static void w_u32(ut_w *w, uint32_t v) { w_u8le(w, v, 4); }

typedef struct { const uint8_t *p; size_t n, at; int bad; } ut_r;
static uint64_t r_le(ut_r *r, int bytes) {
    if (r->at + (size_t)bytes > r->n) { r->bad = 1; return 0; }
    uint64_t v = 0;
    for (int i = 0; i < bytes; ++i) v |= (uint64_t)r->p[r->at + i] << (8 * i);
    r->at += (size_t)bytes;
    return v;
}
static double r_f64(ut_r *r) { const uint64_t u = r_le(r, 8); double d; memcpy(&d, &u, 8); return d; }
static int32_t r_i32(ut_r *r) { return (int32_t)(uint32_t)r_le(r, 4); }
static uint32_t r_u32(ut_r *r) { return (uint32_t)r_le(r, 4); }
static void r_raw(ut_r *r, void *dst, size_t k) {
    if (r->at + k > r->n) { r->bad = 1; return; }
    memcpy(dst, r->p + r->at, k); r->at += k;
}

static void w_hit(ut_w *w, const atkdsp_undertow_cell *h) {
    const double d[12] = { h->fc_hz, h->bw_hz, h->fluct_log10_pfa, h->period_s, h->acf_log10_pfa, h->acf_z,
                           h->cyc_log10_pfa, h->cyc_alpha_hz, h->cyc_z, h->rc_hz, h->L, h->log10_pfa };
    for (int i = 0; i < 12; ++i) w_f64(w, d[i]);
    w_i32(w, h->scale); w_i32(w, h->chan); w_i32(w, h->nwin); w_i32(w, h->nseen);
    w_u32(w, h->flags);
    for (int i = 0; i < 4; ++i) w_i32(w, h->method[i]);
}
static void r_hit(ut_r *r, atkdsp_undertow_cell *h) {
    memset(h, 0, sizeof *h);
    double *d[12] = { &h->fc_hz, &h->bw_hz, &h->fluct_log10_pfa, &h->period_s, &h->acf_log10_pfa, &h->acf_z,
                      &h->cyc_log10_pfa, &h->cyc_alpha_hz, &h->cyc_z, &h->rc_hz, &h->L, &h->log10_pfa };
    for (int i = 0; i < 12; ++i) *d[i] = r_f64(r);
    h->scale = r_i32(r); h->chan = r_i32(r); h->nwin = r_i32(r); h->nseen = r_i32(r);
    h->flags = r_u32(r);
    for (int i = 0; i < 4; ++i) h->method[i] = r_i32(r);
}
static void w_bs(ut_w *w, const atkdsp_bootstrap_result *b) {
    const atkdsp_psync_result *p = &b->ps;
    const double pd[13] = { p->period_s, p->period_samples, p->rate, p->lambda1, p->lambda2, p->ref_mean, p->ref_sd,
                            p->z, p->log10_pfa, p->snr_db, p->sigma2, p->dominance, p->window_offset };
    for (int i = 0; i < 13; ++i) w_f64(w, pd[i]);
    w_i32(w, p->nperiods); w_i32(w, p->n0_used); w_i32(w, p->levels); w_i32(w, p->evals); w_i32(w, p->window);
    w_u32(w, p->flags);
    for (int i = 0; i < 4; ++i) w_i32(w, p->method[i]);
    const atkdsp_dsss_code_result *c = &b->code;
    w_i32(w, c->L); w_i32(w, c->nperiods); w_i32(w, c->periods_needed); w_i32(w, c->start);
    const double cd8[8] = { c->cfo_hz, c->lambda1, c->lambda2, c->noise_var, c->ident_margin, c->snr_db,
                            c->vote_min, c->vote_mean };
    for (int i = 0; i < 8; ++i) w_f64(w, cd8[i]);
    w_i32(w, c->poly_method); w_i32(w, c->poly_degree); w_u8le(w, c->poly, 8); w_f64(w, c->poly_z);
    w_i32(w, c->lc); w_i32(w, c->lc_plateau); w_u32(w, c->flags);
    for (int i = 0; i < 8; ++i) w_i32(w, c->method[i]);
    char name[48]; memset(name, 0, sizeof name); memcpy(name, c->name, sizeof name); name[47] = 0;
    w_raw(w, name, 48);
    w_f64(w, b->cfo_hz); w_f64(w, b->rc_hz); w_f64(w, b->fs_code); w_f64(w, b->line_ratio);
    w_i32(w, b->L); w_i32(w, b->support_start); w_i32(w, b->harmonic); w_u32(w, b->flags);
    for (int i = 0; i < 4; ++i) w_i32(w, b->method[i]);
}
static void r_bs(ut_r *r, atkdsp_bootstrap_result *b) {
    memset(b, 0, sizeof *b);
    atkdsp_psync_result *p = &b->ps;
    double *pd[13] = { &p->period_s, &p->period_samples, &p->rate, &p->lambda1, &p->lambda2, &p->ref_mean, &p->ref_sd,
                       &p->z, &p->log10_pfa, &p->snr_db, &p->sigma2, &p->dominance, &p->window_offset };
    for (int i = 0; i < 13; ++i) *pd[i] = r_f64(r);
    p->nperiods = r_i32(r); p->n0_used = r_i32(r); p->levels = r_i32(r); p->evals = r_i32(r); p->window = r_i32(r);
    p->flags = r_u32(r);
    for (int i = 0; i < 4; ++i) p->method[i] = r_i32(r);
    atkdsp_dsss_code_result *c = &b->code;
    c->L = r_i32(r); c->nperiods = r_i32(r); c->periods_needed = r_i32(r); c->start = r_i32(r);
    double *cd8[8] = { &c->cfo_hz, &c->lambda1, &c->lambda2, &c->noise_var, &c->ident_margin, &c->snr_db,
                       &c->vote_min, &c->vote_mean };
    for (int i = 0; i < 8; ++i) *cd8[i] = r_f64(r);
    c->poly_method = r_i32(r); c->poly_degree = r_i32(r); c->poly = r_le(r, 8); c->poly_z = r_f64(r);
    c->lc = r_i32(r); c->lc_plateau = r_i32(r); c->flags = r_u32(r);
    for (int i = 0; i < 8; ++i) c->method[i] = r_i32(r);
    r_raw(r, c->name, 48); c->name[47] = 0;
    b->cfo_hz = r_f64(r); b->rc_hz = r_f64(r); b->fs_code = r_f64(r); b->line_ratio = r_f64(r);
    b->L = r_i32(r); b->support_start = r_i32(r); b->harmonic = r_i32(r); b->flags = r_u32(r);
    for (int i = 0; i < 4; ++i) b->method[i] = r_i32(r);
}

ptrdiff_t atkdsp_undertow_hit_export(const atkdsp_undertow_ctx *ctx, const atkdsp_undertow_cell *hit,
                                     const atkdsp_bootstrap_result *bs, const int8_t *chips,
                                     size_t nchips, uint8_t *buf, size_t cap) {
    if (!ctx || !hit || (nchips && !chips) || nchips > 0x7FFFFFFFu) return ATKDSP_E_ARG;
    const uint16_t sections = (uint16_t)(1u | (bs ? 2u : 0u) | (nchips ? 4u : 0u));
    ut_w w = { buf, 0, cap };
    w_raw(&w, UT_MAGIC, 8);
    w_u8le(&w, ATKDSP_UT_BLOB_VERSION, 2);
    w_u8le(&w, sections, 2);
    const size_t at_len = w.n;
    w_u32(&w, 0);                                     /* the total, patched below */
    w_f64(&w, ctx->t_unix); w_f64(&w, ctx->rf_center_hz); w_f64(&w, ctx->fs_survey);
    char src[40]; memset(src, 0, sizeof src); memcpy(src, ctx->source, sizeof src); src[39] = 0;
    w_raw(&w, src, 40);
    w_hit(&w, hit);
    if (bs) w_bs(&w, bs);
    if (nchips) { w_u32(&w, (uint32_t)nchips); w_raw(&w, chips, nchips); }
    const size_t total = w.n + 4;
    if (!buf) return (ptrdiff_t)total;
    if (cap < total) return ATKDSP_E_CAP;
    for (int i = 0; i < 4; ++i) buf[at_len + (size_t)i] = (uint8_t)((uint32_t)total >> (8 * i));
    const uint32_t crc = ut_crc32(buf, w.n);
    for (int i = 0; i < 4; ++i) buf[w.n + (size_t)i] = (uint8_t)(crc >> (8 * i));
    return (ptrdiff_t)total;
}

int atkdsp_undertow_hit_import(const uint8_t *buf, size_t n, atkdsp_undertow_ctx *ctx,
                               atkdsp_undertow_cell *hit, atkdsp_bootstrap_result *bs,
                               int *has_bs, int8_t *chips, size_t cap, size_t *nchips) {
    if (!buf || !ctx || !hit || !has_bs || !nchips || (cap && !chips)) return ATKDSP_E_ARG;
    *has_bs = 0; *nchips = 0;
    if (n < 20 || memcmp(buf, UT_MAGIC, 8) != 0) return ATKDSP_E_DECODE;
    ut_r r = { buf, n, 8, 0 };
    const unsigned ver = (unsigned)r_le(&r, 2), sections = (unsigned)r_le(&r, 2);
    const uint32_t total = r_u32(&r);
    if (ver != ATKDSP_UT_BLOB_VERSION || total != n || !(sections & 1u) || (sections & ~7u)) return ATKDSP_E_DECODE;
    uint32_t crc = 0;
    for (int i = 0; i < 4; ++i) crc |= (uint32_t)buf[n - 4 + (size_t)i] << (8 * i);
    if (ut_crc32(buf, n - 4) != crc) return ATKDSP_E_DECODE;
    r.n = n - 4;
    memset(ctx, 0, sizeof *ctx);
    ctx->t_unix = r_f64(&r); ctx->rf_center_hz = r_f64(&r); ctx->fs_survey = r_f64(&r);
    r_raw(&r, ctx->source, 40); ctx->source[39] = 0;
    r_hit(&r, hit);
    if (sections & 2u) {
        atkdsp_bootstrap_result tmp;
        r_bs(&r, &tmp);
        if (bs) *bs = tmp;
        *has_bs = 1;
    }
    if (sections & 4u) {
        const uint32_t k = r_u32(&r);
        if (r.bad || r.at + k > r.n) return ATKDSP_E_DECODE;
        *nchips = k;
        const size_t take = k < cap ? k : cap;
        if (take) memcpy(chips, buf + r.at, take);
        r.at += k;
    }
    if (r.bad || r.at != r.n) return ATKDSP_E_DECODE;
    return ATKDSP_OK;
}
