/* UNDERTOW — the scale-space survey (§29; docs/ATKDSP_PLAN.md §B6c piece 1).
 *
 * A composition of tested kernels: one PFB (§28b) per dyadic scale, one
 * atkdsp_dsss_det (§25, the fluctuation handle only) per cell, and per cell a
 * chunk-invariant chip-shape ACF accumulator and a cyclic |y|^2 alpha-spectrum
 * accumulator. The verdicts are formed at read time: the ACF shape and the
 * cyclic line against the neighbouring cells at the same scale (predictive
 * Student-t), every handle corrected for cells x handles (survey-wide Pfa),
 * and the hits merged across scales.
 *
 * The twin is python/atkdsp/reference.py §29 (class Undertow); every number
 * here is the twin's, operation for operation (the PFB and the detector are
 * float32 kernels, so C == twin to float precision, not bit for bit).
 */
#include "internal.h"
#include <stdio.h>

/* ---- Student's t tail via the regularized incomplete beta ---------------- */
static double ut_betacf(double a, double b, double x) {
    const int MAXIT = 300; const double EPS = 3.0e-16, FPMIN = 1.0e-300;
    const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    double c = 1.0, d = 1.0 - qab * x / qap;
    if (fabs(d) < FPMIN) d = FPMIN;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= MAXIT; ++m) {
        const int m2 = 2 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d; if (fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c; if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d; h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d; if (fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c; if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        const double de = d * c;
        h *= de;
        if (fabs(de - 1.0) < EPS) break;
    }
    return h;
}

static double ut_log_betai(double a, double b, double x) {
    if (x <= 0.0) return -INFINITY;
    if (x >= 1.0) return 0.0;
    const double lbt = lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log1p(-x);
    if (x < (a + 1.0) / (a + b + 2.0))
        return lbt + log(ut_betacf(a, b, x)) - log(a);
    const double v = 1.0 - exp(lbt + log(ut_betacf(b, a, 1.0 - x)) - log(b));
    return v > 0.0 ? log(v) : -INFINITY;
}

double atkdsp_t_log10_sf2(double t, double nu) {
    if (nu < 1.0) return 0.0;
    t = fabs(t);
    return ut_log_betai(0.5 * nu, 0.5, nu / (nu + t * t)) / log(10.0);
}

/* ---- the handle ------------------------------------------------------------ */
#define UT_MAXSCALES 12

struct atkdsp_undertow {
    atkdsp_undertow_cfg cfg;
    int          ns;                   /* scales                                   */
    int          nch[UT_MAXSCALES];    /* channels per scale                       */
    int          off[UT_MAXSCALES];    /* first cell index of each scale           */
    int          ncells;
    atkdsp_pfb  *pfb[UT_MAXSCALES];
    atkdsp_dsss_det **det;             /* ncells                                   */
    double      *Rre, *Rim;            /* ncells * (K+1)  ACF sums                 */
    double      *tre, *tim;            /* ncells * K      ACF tails (last K)       */
    uint64_t     cnt[UT_MAXSCALES];
    atkdsp_cf32 *seg;                  /* ncells * seg_len  cyclic segment buffers  */
    double      *S;                    /* ncells * seg_len  alpha-spectrum sums     */
    int          fill[UT_MAXSCALES];
    uint64_t     nseg[UT_MAXSCALES];
    atkdsp_fft  *fft;                  /* seg_len (thread-safe; shared by cells)   */
    atkdsp_cf32 *fin, *fout;           /* ncells * seg_len  per-cell FFT scratch    */
    double      *e;                    /* ncells * seg_len                          */
    atkdsp_cf32 *pout[UT_MAXSCALES];   /* PFB output per scale (scales run in parallel) */
    ptrdiff_t    m[UT_MAXSCALES];      /* hops each scale produced this chunk       */
    size_t       pcap;
    int         *cs, *ck;              /* ncells: cell -> scale, channel            */
    uint64_t     samples;
};

void atkdsp_undertow_defaults(atkdsp_undertow_cfg *c, double fs, int nscales, int lmax) {
    if (!c) return;
    memset(c, 0, sizeof *c);
    c->fs = fs; c->nscales = nscales; c->nchan0 = 8;
    c->max_lag = 4 * lmax;
    int w = 1; while (w < 2 * c->max_lag) w <<= 1;
    c->win = w;
    c->period_min = 16; c->acf_lags = 4; c->seg_len = 256; c->taps_per_poly = 12;
    c->atten_db = 80.0; c->pfa = 1e-3; c->max_block = 65536;
}

static int ut_pow2(int v) { return v > 0 && (v & (v - 1)) == 0; }

atkdsp_undertow *atkdsp_undertow_create(const atkdsp_undertow_cfg *cfg) {
    if (!cfg) return NULL;
    const atkdsp_undertow_cfg *c = cfg;
    if (!(c->fs > 0.0) || c->nscales < 1 || c->nscales > UT_MAXSCALES) return NULL;
    if (c->nchan0 < 4 || !ut_pow2(c->nchan0)) return NULL;
    if (!(1 <= c->period_min && c->period_min < c->max_lag && c->max_lag < c->win) || !ut_pow2(c->win)) return NULL;
    if (c->acf_lags < 1 || c->acf_lags > 32 || c->seg_len < 32 || !ut_pow2(c->seg_len)) return NULL;
    if (!(c->pfa > 0.0 && c->pfa < 0.5) || c->taps_per_poly < 2) return NULL;
    atkdsp_undertow *u = (atkdsp_undertow *)calloc(1, sizeof *u);
    if (!u) return NULL;
    u->cfg = *c;
    if (u->cfg.max_block < 1024) u->cfg.max_block = 65536;
    u->ns = c->nscales;
    int total = 0, nmax = 0;
    for (int s = 0; s < u->ns; ++s) {
        u->nch[s] = c->nchan0 << s; u->off[s] = total; total += u->nch[s];
        if (u->nch[s] > nmax) nmax = u->nch[s];
    }
    u->ncells = total;
    const int K = c->acf_lags, L = c->seg_len;
    u->det = (atkdsp_dsss_det **)calloc((size_t)total, sizeof *u->det);
    u->Rre = (double *)calloc((size_t)total * (K + 1), sizeof(double));
    u->Rim = (double *)calloc((size_t)total * (K + 1), sizeof(double));
    u->tre = (double *)calloc((size_t)total * K, sizeof(double));
    u->tim = (double *)calloc((size_t)total * K, sizeof(double));
    u->seg = (atkdsp_cf32 *)calloc((size_t)total * L, sizeof(atkdsp_cf32));
    u->S = (double *)calloc((size_t)total * L, sizeof(double));
    u->fft = atkdsp_fft_create((size_t)L);
    u->fin = (atkdsp_cf32 *)malloc((size_t)total * L * sizeof(atkdsp_cf32));
    u->fout = (atkdsp_cf32 *)malloc((size_t)total * L * sizeof(atkdsp_cf32));
    u->e = (double *)malloc((size_t)total * L * sizeof(double));
    u->cs = (int *)malloc((size_t)total * sizeof(int));
    u->ck = (int *)malloc((size_t)total * sizeof(int));
    u->pcap = 2 * (size_t)u->cfg.max_block + (size_t)nmax + 16;
    if (!u->det || !u->Rre || !u->Rim || !u->tre || !u->tim || !u->seg || !u->S || !u->fft ||
        !u->fin || !u->fout || !u->e || !u->cs || !u->ck) { atkdsp_undertow_destroy(u); return NULL; }
    for (int s = 0; s < u->ns; ++s) {
        u->pout[s] = (atkdsp_cf32 *)atk_aligned_malloc(u->pcap * sizeof(atkdsp_cf32));
        if (!u->pout[s]) { atkdsp_undertow_destroy(u); return NULL; }
        for (int k = 0; k < u->nch[s]; ++k) { u->cs[u->off[s] + k] = s; u->ck[u->off[s] + k] = k; }
    }
    for (int s = 0; s < u->ns; ++s) {
        u->pfb[s] = atkdsp_pfb_create(c->fs, (unsigned)u->nch[s], 2, (unsigned)c->taps_per_poly,
                                      c->atten_db, 1);
        if (!u->pfb[s]) { atkdsp_undertow_destroy(u); return NULL; }
        const double B = c->fs / (double)u->nch[s];
        for (int k = 0; k < u->nch[s]; ++k) {
            atkdsp_dsss_det_config dc;
            atkdsp_dsss_det_defaults(&dc, 2.0 * B, c->win, c->max_lag);
            dc.period_min = c->period_min;
            dc.handles = (int32_t)ATKDSP_DSSS_H_FLUCT;
            dc.pfa = c->pfa;
            u->det[u->off[s] + k] = atkdsp_dsss_det_create(&dc);
            if (!u->det[u->off[s] + k]) { atkdsp_undertow_destroy(u); return NULL; }
        }
    }
    return u;
}

void atkdsp_undertow_destroy(atkdsp_undertow *u) {
    if (!u) return;
    for (int s = 0; s < u->ns; ++s) atkdsp_pfb_destroy(u->pfb[s]);
    if (u->det) for (int i = 0; i < u->ncells; ++i) atkdsp_dsss_det_destroy(u->det[i]);
    free(u->det); free(u->Rre); free(u->Rim); free(u->tre); free(u->tim);
    free(u->seg); free(u->S); atkdsp_fft_destroy(u->fft); free(u->fin); free(u->fout); free(u->e);
    for (int s = 0; s < UT_MAXSCALES; ++s) atk_aligned_free(u->pout[s]);
    free(u->cs); free(u->ck);
    free(u);
}

void atkdsp_undertow_reset(atkdsp_undertow *u) {
    if (!u) return;
    const int K = u->cfg.acf_lags, L = u->cfg.seg_len;
    for (int s = 0; s < u->ns; ++s) { atkdsp_pfb_reset(u->pfb[s]); u->cnt[s] = 0; u->fill[s] = 0; u->nseg[s] = 0; }
    for (int i = 0; i < u->ncells; ++i) atkdsp_dsss_det_reset(u->det[i]);
    memset(u->Rre, 0, (size_t)u->ncells * (K + 1) * sizeof(double));
    memset(u->Rim, 0, (size_t)u->ncells * (K + 1) * sizeof(double));
    memset(u->tre, 0, (size_t)u->ncells * K * sizeof(double));
    memset(u->tim, 0, (size_t)u->ncells * K * sizeof(double));
    memset(u->seg, 0, (size_t)u->ncells * L * sizeof(atkdsp_cf32));
    memset(u->S, 0, (size_t)u->ncells * L * sizeof(double));
    u->samples = 0;
}

int atkdsp_undertow_ncells(const atkdsp_undertow *u) { return u ? u->ncells : 0; }

/* One cell over its m new samples (channel k of scale s's channel-major
 * output): the detector, the chip-shape ACF, and the cyclic segments starting
 * from the scale's fill0. Touches only this cell's state and scratch, so all
 * cells run in parallel and the result is the same at any thread count. */
static void ut_cell(atkdsp_undertow *u, int ci, size_t m, int fill0) {
    const int s = u->cs[ci], k = u->ck[ci];
    const int K = u->cfg.acf_lags, L = u->cfg.seg_len;
    const atkdsp_cf32 *v = u->pout[s] + (size_t)k * m;
    atkdsp_dsss_det_process(u->det[ci], v, m);
    double *Rr = u->Rre + (size_t)ci * (K + 1), *Ri = u->Rim + (size_t)ci * (K + 1);
    double *tr = u->tre + (size_t)ci * K, *ti = u->tim + (size_t)ci * K;
    /* w = tail(K) ++ v(m); R[t] += sum_{j<m} w[K+j] * conj(w[K+j-t]) */
    for (int t = 0; t <= K; ++t) {
        double ar = 0.0, ai = 0.0;
        const size_t jt = (size_t)t < m ? (size_t)t : m;
        for (size_t j = 0; j < jt; ++j) {             /* reaches back into the tail */
            const double xr = v[j].re, xi = v[j].im;
            const double yr = tr[K + (ptrdiff_t)j - t], yi = ti[K + (ptrdiff_t)j - t];
            ar += xr * yr + xi * yi;
            ai += xi * yr - xr * yi;
        }
        for (size_t j = jt; j < m; ++j) {
            const double xr = v[j].re, xi = v[j].im, yr = v[j - t].re, yi = v[j - t].im;
            ar += xr * yr + xi * yi;
            ai += xi * yr - xr * yi;
        }
        Rr[t] += ar; Ri[t] += ai;
    }
    if (m >= (size_t)K) {
        for (int i = 0; i < K; ++i) { tr[i] = v[m - K + i].re; ti[i] = v[m - K + i].im; }
    } else {
        const int sh = (int)m;
        for (int i = 0; i < K - sh; ++i) { tr[i] = tr[i + sh]; ti[i] = ti[i + sh]; }
        for (int i = 0; i < sh; ++i) { tr[K - sh + i] = v[i].re; ti[K - sh + i] = v[i].im; }
    }
    /* cyclic segments (every cell of a scale fills in lock-step from fill0) */
    atkdsp_cf32 *g = u->seg + (size_t)ci * L;
    atkdsp_cf32 *fin = u->fin + (size_t)ci * L, *fout = u->fout + (size_t)ci * L;
    double *e = u->e + (size_t)ci * L, *S = u->S + (size_t)ci * L;
    size_t pos = 0; int f = fill0;
    while (pos < m) {
        size_t take = (size_t)(L - f);
        if (take > m - pos) take = m - pos;
        memcpy(g + f, v + pos, take * sizeof(atkdsp_cf32));
        f += (int)take; pos += take;
        if (f == L) {
            double mean = 0.0;
            for (int i = 0; i < L; ++i) { e[i] = (double)g[i].re * g[i].re + (double)g[i].im * g[i].im; mean += e[i]; }
            mean /= (double)L;
            for (int i = 0; i < L; ++i) { fin[i].re = (float)(e[i] - mean); fin[i].im = 0.0f; }
            atkdsp_fft_exec(u->fft, fin, fout, 0);
            for (int i = 0; i < L; ++i)
                S[i] += (double)fout[i].re * fout[i].re + (double)fout[i].im * fout[i].im;
            f = 0;
        }
    }
}

int atkdsp_undertow_process(atkdsp_undertow *u, const atkdsp_cf32 *x, size_t n) {
    if (!u || (!x && n)) return ATKDSP_E_ARG;
    const int L = u->cfg.seg_len;
    size_t done = 0;
    while (done < n) {
        size_t chunk = n - done;
        if (chunk > (size_t)u->cfg.max_block) chunk = (size_t)u->cfg.max_block;
        const atkdsp_cf32 *xc = x + done;
        /* phase 1: every scale's filterbank (independent handles) */
        int s;
#pragma omp parallel for num_threads(atkdsp_get_threads()) schedule(dynamic, 1)
        for (s = 0; s < u->ns; ++s)
            u->m[s] = atkdsp_pfb_process(u->pfb[s], xc, chunk, u->pout[s], u->pcap);
        for (s = 0; s < u->ns; ++s) if (u->m[s] < 0) return (int)u->m[s];
        /* phase 2: every cell of every scale */
        int ci;
#pragma omp parallel for num_threads(atkdsp_get_threads()) schedule(dynamic, 4)
        for (ci = 0; ci < u->ncells; ++ci) {
            const int sc = u->cs[ci];
            if (u->m[sc] > 0) ut_cell(u, ci, (size_t)u->m[sc], u->fill[sc]);
        }
        /* phase 3: advance each scale's counters exactly as the cells did */
        for (s = 0; s < u->ns; ++s) {
            const size_t m = u->m[s] > 0 ? (size_t)u->m[s] : 0;
            size_t pos = 0; int f = u->fill[s];
            while (pos < m) {
                size_t take = (size_t)(L - f);
                if (take > m - pos) take = m - pos;
                f += (int)take; pos += take;
                if (f == L) { u->nseg[s] += 1; f = 0; }
            }
            u->fill[s] = f;
            u->cnt[s] += m;
        }
        done += chunk;
    }
    u->samples += n;
    return ATKDSP_OK;
}

/* ---- verdicts -------------------------------------------------------------- */
static int ut_refs(int k, int nc, int *out) {
    int r = (nc - 3) / 2; if (r < 1) r = 1; if (r > 4) r = 4;
    int n = 0;
    for (int d = -1 - r; d < -1; ++d) out[n++] = ((k + d) % nc + nc) % nc;
    for (int d = 2; d < 2 + r; ++d) out[n++] = (k + d) % nc;
    return n;
}

/* predictive Student-t of v against the reference values */
static double ut_pred_t(double v, const double *ref, int n, double *mu_out, double *sd_out) {
    double mu = 0.0, sd = 0.0;
    if (n >= 1) { for (int i = 0; i < n; ++i) mu += ref[i]; mu /= n; }
    if (n >= 2) { for (int i = 0; i < n; ++i) sd += (ref[i] - mu) * (ref[i] - mu); sd = sqrt(sd / (n - 1)); }
    if (mu_out) *mu_out = mu;
    if (sd_out) *sd_out = sd;
    if (n < 3 || !(sd > 0.0)) return 0.0;
    const double t = (v - mu) / (sd * sqrt(1.0 + 1.0 / n));
    return atkdsp_t_log10_sf2(t, (double)(n - 1));
}

ptrdiff_t atkdsp_undertow_cells(const atkdsp_undertow *u, atkdsp_undertow_cell *out, size_t cap) {
    if (!u || !out) return ATKDSP_E_ARG;
    if (cap < (size_t)u->ncells) return ATKDSP_E_CAP;
    const int K = u->cfg.acf_lags, L = u->cfg.seg_len;
    const double corr = log10((double)u->ncells) + log10(3.0);
    const double lpfa = log10(u->cfg.pfa);
    int nmax = 0;
    for (int s = 0; s < u->ns; ++s) if (u->nch[s] > nmax) nmax = u->nch[s];
    double *shape = (double *)malloc((size_t)nmax * K * sizeof(double));
    double *p0 = (double *)malloc((size_t)nmax * sizeof(double));
    double *Sn = (double *)malloc((size_t)nmax * L * sizeof(double));
    double *rv = (double *)malloc(16 * sizeof(double));
    int *wb = (int *)malloc((size_t)L * sizeof(int));
    double *afreq = (double *)malloc((size_t)L * sizeof(double));
    if (!shape || !p0 || !Sn || !rv || !wb || !afreq) {
        free(shape); free(p0); free(Sn); free(rv); free(wb); free(afreq); return ATKDSP_E_NOMEM;
    }
    for (int s = 0; s < u->ns; ++s) {
        const int nc = u->nch[s];
        const double B = u->cfg.fs / (double)nc;
        const uint64_t cnt = u->cnt[s];
        for (int k = 0; k < nc; ++k) {
            const int ci = u->off[s] + k;
            const double r0 = u->Rre[(size_t)ci * (K + 1)];
            p0[k] = cnt ? r0 / (double)cnt : 0.0;
            for (int t = 0; t < K; ++t) {
                const double re = u->Rre[(size_t)ci * (K + 1) + t + 1], im = u->Rim[(size_t)ci * (K + 1) + t + 1];
                shape[(size_t)k * K + t] = r0 > 0.0 ? sqrt(re * re + im * im) / r0 : 0.0;
            }
        }
        const uint64_t ns = u->nseg[s];
        int nwb = 0;
        for (int j = 0; j < L; ++j) {
            const double a = (double)((j < L / 2) ? j : j - L) * (2.0 * B) / (double)L;
            afreq[j] = a;
            if (fabs(a) >= 0.5 * B && fabs(a) <= B) wb[nwb++] = j;
        }
        if (ns) {
            for (int k = 0; k < nc; ++k) {
                double den = p0[k] * p0[k]; if (den < 1e-300) den = 1e-300;
                const double *S = u->S + (size_t)(u->off[s] + k) * L;
                for (int j = 0; j < L; ++j) Sn[(size_t)k * L + j] = S[j] / (double)ns / den;
            }
        }
        for (int k = 0; k < nc; ++k) {
            const int ci = u->off[s] + k;
            atkdsp_undertow_cell *c = &out[ci];
            memset(c, 0, sizeof *c);
            atkdsp_dsss_det_result rd;
            atkdsp_dsss_det_read(u->det[ci], &rd);
            c->scale = s; c->chan = k; c->bw_hz = B;
            c->fc_hz = (k < nc / 2) ? (double)k * B : (double)(k - nc) * B;
            c->nwin = rd.nwin; c->nseen = 1;
            c->method[0] = ATKDSP_METHOD_SCALE_SURVEY; c->method[1] = ATKDSP_METHOD_BUREL_FLUCT;
            c->method[2] = ATKDSP_METHOD_ACF_CFAR;     c->method[3] = ATKDSP_METHOD_SCALE_CYC;
            if (cnt < (uint64_t)u->cfg.win || (rd.flags & ATKDSP_DSSS_WARMUP)) c->flags |= ATKDSP_UT_WARMUP;
            {
                double f = rd.fluct_log10_pfa + corr;
                c->fluct_log10_pfa = f < 0.0 ? f : 0.0;
            }
            if (rd.flags & ATKDSP_DSSS_PERIOD) { c->period_s = rd.period_s; c->flags |= ATKDSP_UT_PERIOD; }
            int refs[16];
            const int nr = ut_refs(k, nc, refs);
            /* ACF */
            double best = 0.0, bz = 0.0;
            for (int t = 0; t < K; ++t) {
                for (int i = 0; i < nr; ++i) rv[i] = shape[(size_t)refs[i] * K + t];
                double mu, sd;
                const double lp = ut_pred_t(shape[(size_t)k * K + t], rv, nr, &mu, &sd);
                if (lp < best) { best = lp; bz = sd > 0.0 ? (shape[(size_t)k * K + t] - mu) / sd : 0.0; }
            }
            {
                double a = best + log10((double)K) + corr;
                c->acf_log10_pfa = a < 0.0 ? a : 0.0;
            }
            c->acf_z = bz;
            /* CYC */
            if (ns) {
                double bc = 0.0; int bb = -1;
                for (int w = 0; w < nwb; ++w) {
                    const int j = wb[w];
                    for (int i = 0; i < nr; ++i) rv[i] = Sn[(size_t)refs[i] * L + j];
                    double mu, sd;
                    const double v = Sn[(size_t)k * L + j];
                    const double lp = ut_pred_t(v, rv, nr, &mu, &sd);
                    if (lp < bc && v > mu) { bc = lp; bb = j; }
                }
                double a = bc + log10((double)(nwb > 1 ? nwb : 1)) + corr;
                c->cyc_log10_pfa = a < 0.0 ? a : 0.0;
                if (bb >= 0) {
                    c->cyc_alpha_hz = fabs(afreq[bb]);
                    for (int i = 0; i < nr; ++i) rv[i] = Sn[(size_t)refs[i] * L + bb];
                    double mu, sd;
                    ut_pred_t(0.0, rv, nr, &mu, &sd);
                    c->cyc_z = sd > 0.0 ? (Sn[(size_t)k * L + bb] - mu) / sd : 0.0;
                }
            }
            if (c->fluct_log10_pfa < lpfa) c->flags |= ATKDSP_UT_FLUCT;
            if (c->acf_log10_pfa < lpfa)   c->flags |= ATKDSP_UT_ACF;
            if (c->cyc_log10_pfa < lpfa)   c->flags |= ATKDSP_UT_CYC;
            c->log10_pfa = c->fluct_log10_pfa;
            if (c->acf_log10_pfa < c->log10_pfa) c->log10_pfa = c->acf_log10_pfa;
            if (c->cyc_log10_pfa < c->log10_pfa) c->log10_pfa = c->cyc_log10_pfa;
            if (c->flags & ATKDSP_UT_CYC) {
                c->rc_hz = c->cyc_alpha_hz;
                if (c->period_s > 0.0) {
                    const double Lc = c->period_s * c->rc_hz;
                    c->L = Lc;
                    const double tol = 0.05 * sqrt(Lc > 1.0 ? Lc : 1.0);
                    if (Lc >= 2.0 && fabs(Lc - floor(Lc + 0.5)) <= tol) c->flags |= ATKDSP_UT_CONSISTENT;
                }
            }
        }
    }
    free(shape); free(p0); free(Sn); free(rv); free(wb); free(afreq);
    return (ptrdiff_t)u->ncells;
}

typedef struct { const atkdsp_undertow_cell *c; int idx; } ut_ord;
static int ut_cmp(const void *pa, const void *pb) {
    const ut_ord *a = (const ut_ord *)pa, *b = (const ut_ord *)pb;
    if (a->c->log10_pfa < b->c->log10_pfa) return -1;
    if (a->c->log10_pfa > b->c->log10_pfa) return 1;
    if (a->c->scale != b->c->scale) return b->c->scale - a->c->scale;   /* finer scale first */
    return a->idx - b->idx;                                              /* stable */
}

ptrdiff_t atkdsp_undertow_read(const atkdsp_undertow *u, atkdsp_undertow_cell *hits, size_t cap) {
    if (!u || (!hits && cap)) return ATKDSP_E_ARG;
    atkdsp_undertow_cell *cs = (atkdsp_undertow_cell *)malloc((size_t)u->ncells * sizeof *cs);
    ut_ord *ord = (ut_ord *)malloc((size_t)u->ncells * sizeof *ord);
    atkdsp_undertow_cell *hs = (atkdsp_undertow_cell *)malloc((size_t)u->ncells * sizeof *hs);
    if (!cs || !ord || !hs) { free(cs); free(ord); free(hs); return ATKDSP_E_NOMEM; }
    ptrdiff_t r = atkdsp_undertow_cells(u, cs, (size_t)u->ncells);
    if (r < 0) { free(cs); free(ord); free(hs); return r; }
    int nlit = 0;
    const uint32_t FIRED = ATKDSP_UT_FLUCT | ATKDSP_UT_ACF | ATKDSP_UT_CYC;
    for (int i = 0; i < u->ncells; ++i)
        if (cs[i].flags & FIRED) { ord[nlit].c = &cs[i]; ord[nlit].idx = i; ++nlit; }
    qsort(ord, (size_t)nlit, sizeof *ord, ut_cmp);
    int nh = 0;
    for (int i = 0; i < nlit; ++i) {
        const atkdsp_undertow_cell *c = ord[i].c;
        const double lo = c->fc_hz - c->bw_hz, hi = c->fc_hz + c->bw_hz;
        int merged = 0;
        for (int j = 0; j < nh; ++j) {
            atkdsp_undertow_cell *h = &hs[j];
            if (lo < h->fc_hz + h->bw_hz && h->fc_hz - h->bw_hz < hi) {
                h->nseen += 1;
                h->flags |= c->flags & (FIRED | ATKDSP_UT_CONSISTENT);
                if (c->fluct_log10_pfa < h->fluct_log10_pfa) h->fluct_log10_pfa = c->fluct_log10_pfa;
                if (c->acf_log10_pfa < h->acf_log10_pfa)     h->acf_log10_pfa = c->acf_log10_pfa;
                if (c->cyc_log10_pfa < h->cyc_log10_pfa)     h->cyc_log10_pfa = c->cyc_log10_pfa;
                if (c->cyc_log10_pfa <= h->cyc_log10_pfa && c->cyc_alpha_hz > 0.0) {
                    h->cyc_alpha_hz = c->cyc_alpha_hz; h->cyc_z = c->cyc_z;
                    if (c->flags & ATKDSP_UT_CYC) {
                        h->rc_hz = c->rc_hz;
                        if (h->period_s > 0.0) h->L = h->period_s * h->rc_hz;
                    }
                }
                if (c->acf_log10_pfa <= h->acf_log10_pfa) h->acf_z = c->acf_z;
                merged = 1;
                break;
            }
        }
        if (!merged) hs[nh++] = *c;
    }
    ptrdiff_t ret;
    if ((size_t)nh > cap) {
        if (cap) memcpy(hits, hs, cap * sizeof *hs);
        ret = ATKDSP_E_CAP;
    } else {
        if (nh) memcpy(hits, hs, (size_t)nh * sizeof *hs);
        ret = nh;
    }
    free(cs); free(ord); free(hs);
    return ret;
}

int atkdsp_undertow_describe(const atkdsp_undertow *u, char *buf, size_t cap) {
    if (!u || !buf || cap == 0) return ATKDSP_E_ARG;
    const double fs = u->cfg.fs;
    return snprintf(buf, cap,
        "Undertow %.6g MSPS: %d scales (%d..%d chan, B %.4g..%.4g kHz), %d cells, win %d, "
        "lags %d..%d, Pfa %g survey-wide",
        fs / 1e6, u->ns, u->nch[0], u->nch[u->ns - 1], fs / u->nch[u->ns - 1] / 1e3,
        fs / u->nch[0] / 1e3, u->ncells, u->cfg.win, u->cfg.period_min, u->cfg.max_lag, u->cfg.pfa);
}

/* ---- §30: the survey's candidate periods ------------------------------------
 * Every cell's per_cell strongest fluctuation lags (atkdsp_dsss_det_lags) whose
 * CELL-level tail is at or below max_log10, sorted by that tail, then the finer
 * scale, the cell, the lag — the seeds atkdsp_psync takes below the survey's own
 * threshold. Twin: Undertow.candidates. */
typedef struct { atkdsp_undertow_cand c; int idx; } ut_cand_k;

static int ut_cand_cmp(const void *pa, const void *pb) {
    const ut_cand_k *a = (const ut_cand_k *)pa, *b = (const ut_cand_k *)pb;
    if (a->c.log10_pfa < b->c.log10_pfa) return -1;
    if (a->c.log10_pfa > b->c.log10_pfa) return 1;
    if (a->c.scale != b->c.scale) return a->c.scale > b->c.scale ? -1 : 1;
    if (a->idx != b->idx) return a->idx < b->idx ? -1 : 1;
    return (a->c.lag > b->c.lag) - (a->c.lag < b->c.lag);
}

ptrdiff_t atkdsp_undertow_candidates(const atkdsp_undertow *u, double max_log10, int per_cell,
                                     atkdsp_undertow_cand *out, size_t cap) {
    if (!u || per_cell < 1 || per_cell > 8 || (!out && cap)) return ATKDSP_E_ARG;
    ut_cand_k *all = (ut_cand_k *)malloc((size_t)u->ncells * (size_t)per_cell * sizeof *all);
    if (!all) return ATKDSP_E_NOMEM;
    size_t n = 0;
    atkdsp_dsss_lag lg[8];
    for (int s = 0; s < u->cfg.nscales; ++s) {
        const int nc = u->nch[s];
        const double B = u->cfg.fs / (double)nc;
        for (int k = 0; k < nc; ++k) {
            const int ci = u->off[s] + k;
            const ptrdiff_t got = atkdsp_dsss_det_lags(u->det[ci], lg, (size_t)per_cell);
            for (ptrdiff_t j = 0; j < got; ++j) {
                if (!(lg[j].log10_pfa <= max_log10)) continue;
                ut_cand_k *e = &all[n++];
                memset(e, 0, sizeof *e);
                e->idx = ci;
                e->c.scale = s; e->c.chan = k; e->c.lag = lg[j].lag;
                e->c.bw_hz = B; e->c.fs_cell = 2.0 * B;
                e->c.fc_hz = (k < nc / 2) ? (double)k * B : (double)(k - nc) * B;
                e->c.period_s = lg[j].period / (2.0 * B);
                e->c.ratio = lg[j].ratio; e->c.log10_pfa = lg[j].log10_pfa;
            }
        }
    }
    qsort(all, n, sizeof *all, ut_cand_cmp);
    for (size_t i = 0; i < n && i < cap; ++i) out[i] = all[i].c;
    free(all);
    return (ptrdiff_t)n;
}
