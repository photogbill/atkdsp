/* 19. CFAR and the burst tracker (docs/ATKDSP_PLAN.md §B1).
 *
 * A 1-D CFAR across frequency on each spectrum frame and a tracker that
 * turns hit bins into one record per burst. The definitions are in
 * atkdsp.h §19; the twin is python/atkdsp/reference.py §19 (class Cfar),
 * and every sum here runs in the order the twin's does, so the two agree
 * hit for hit.
 *
 * THE THRESHOLD IS SOLVED, NOT LOOKED UP. Pfa(alpha) = integral of
 * Q(K, K alpha t) against the density of the reference statistic T, for
 * unit-mean K-look gamma cells; Simpson's rule in u = ln t over the range
 * outside which each tail of T holds less than 1e-9 x Pfa; regula falsi
 * (Illinois) in ln alpha. The same code serves CA, OS, GO and SO, and the
 * K = 1 closed forms are the test that it is right (tests/test_detection.py).
 */
#include "internal.h"

#include <stdio.h>

#define LN10_OVER_10 0.23025850929940458
#define MAX_GRID 131072

/* ---- the threshold solver -------------------------------------------------- */

static double ln_gpdf(double s, double t) {
    return s * log(s) + (s - 1.0) * log(t) - s * t - lgamma(s);
}

/* ln density of T at t (noise mean 1 per cell). */
static double log_density(int mode, double t, int N, double K, int k) {
    const int n = N / 2;
    double lp, lq;
    if (mode == ATKDSP_CFAR_CA) return ln_gpdf(N * K, t);
    if (mode == ATKDSP_CFAR_GO || mode == ATKDSP_CFAR_SO) {
        const double s = n * K;
        atk_log_igam_pq(s, s * t, &lp, &lq);
        return log(2.0) + (mode == ATKDSP_CFAR_GO ? lp : lq) + ln_gpdf(s, t);
    }
    atk_log_igam_pq(K, K * t, &lp, &lq);
    double out = log((double)N) + lgamma((double)N) - lgamma((double)k)
               - lgamma((double)(N - k + 1)) + ln_gpdf(K, t);
    if (k > 1) out = out + (k - 1) * lp;
    if (N > k) out = out + (N - k) * lq;
    return out;
}

/* ln P(T <= t) and ln P(T > t). */
static void log_cdf_sf(int mode, double t, int N, double K, int k, double *cdf, double *sf) {
    const int n = N / 2;
    double lp, lq;
    if (mode == ATKDSP_CFAR_CA) { atk_log_igam_pq(N * K, N * K * t, cdf, sf); return; }
    if (mode == ATKDSP_CFAR_GO || mode == ATKDSP_CFAR_SO) {
        const double s = n * K;
        atk_log_igam_pq(s, s * t, &lp, &lq);
        if (mode == ATKDSP_CFAR_GO) { *cdf = 2.0 * lp; *sf = lq + log1p(exp(lp)); }
        else                        { *cdf = lp + log1p(exp(lq)); *sf = 2.0 * lq; }
        return;
    }
    atk_log_igam_pq(K, K * t, &lp, &lq);
    const double ln10 = 2.302585092994045684;
    *cdf = lp == -INFINITY ? -INFINITY : atk_log10_binom_tail(N, k, lp) * ln10;
    *sf  = lq == -INFINITY ? -INFINITY : atk_log10_binom_tail(N, N - k + 1, lq) * ln10;
}

typedef struct { double *t, *G; int M; double et; } cfar_grid;

static int build_grid(int mode, int N, double K, int k, double pfa, cfar_grid *g) {
    const double ln_eps = log(1e-9 * pfa);
    const double lo = -700.0, hi = 7.0;
    double cdf, sf, u_lo, u_hi;
    log_cdf_sf(mode, exp(lo), N, K, k, &cdf, &sf);
    if (cdf >= ln_eps) u_lo = lo;
    else {
        double a = lo, b = hi;
        for (int i = 0; i < 60; ++i) {
            const double mid = 0.5 * (a + b);
            log_cdf_sf(mode, exp(mid), N, K, k, &cdf, &sf);
            if (cdf < ln_eps) a = mid; else b = mid;
        }
        u_lo = a;
    }
    log_cdf_sf(mode, exp(hi), N, K, k, &cdf, &sf);
    if (sf > ln_eps) u_hi = hi;
    else {
        double a = u_lo, b = hi;
        for (int i = 0; i < 60; ++i) {
            const double mid = 0.5 * (a + b);
            log_cdf_sf(mode, exp(mid), N, K, k, &cdf, &sf);
            if (sf > ln_eps) a = mid; else b = mid;
        }
        u_hi = b;
    }
    const double s_ref = (mode == ATKDSP_CFAR_GO || mode == ATKDSP_CFAR_SO) ? (N / 2) * K : N * K;
    double h0 = 0.2 / sqrt(s_ref);
    if (h0 > 0.01) h0 = 0.01;
    const double span = u_hi - u_lo;
    long M = 2 * (long)ceil(span / (2.0 * h0));
    if (M < 512) M = 512;
    if (M > MAX_GRID) M = MAX_GRID;
    const double h = span / (double)M;
    g->M = (int)M;
    g->t = (double *)malloc((size_t)(M + 1) * sizeof(double));
    g->G = (double *)malloc((size_t)(M + 1) * sizeof(double));
    if (!g->t || !g->G) { free(g->t); free(g->G); g->t = g->G = NULL; return ATKDSP_E_NOMEM; }
    double et = 0.0;
    for (long j = 0; j <= M; ++j) {
        const double u = u_lo + h * (double)j;
        const double t = exp(u);
        double w = (j == 0 || j == M) ? 1.0 : ((j & 1) ? 4.0 : 2.0);
        w *= h / 3.0;
        const double lg = log_density(mode, t, N, K, k) + u;
        g->t[j] = t;
        g->G[j] = isfinite(lg) ? exp(lg) * w : 0.0;
    }
    for (long j = 0; j <= M; ++j) et += g->G[j] * g->t[j];
    g->et = et;
    return ATKDSP_OK;
}

static double grid_pfa(const cfar_grid *g, double K, double alpha) {
    double s = 0.0;
    for (int j = 0; j <= g->M; ++j) {
        double lp, lq;
        atk_log_igam_pq(K, K * alpha * g->t[j], &lp, &lq);
        s += g->G[j] * exp(lq);
    }
    return s;
}

static double root_f(const cfar_grid *g, double K, double v, double lnpfa) {
    double p = grid_pfa(g, K, exp(v));
    if (p < 1e-300) p = 1e-300;
    return log(p) - lnpfa;
}

int atkdsp_cfar_alpha(int mode, int n_ref, double pfa, double looks, int os_k,
                      double *alpha, double *et) {
    if (mode < 0 || mode > 3 || n_ref < 1 || n_ref > 256 || !(pfa >= 1e-15 && pfa <= 0.5)
        || !(looks >= 0.25 && looks <= 64e6) || !alpha)
        return ATKDSP_E_ARG;
    const int N = 2 * n_ref;
    const int k = os_k ? os_k : (3 * N + 3) / 4;
    if (mode == ATKDSP_CFAR_OS && (k < 1 || k > N)) return ATKDSP_E_ARG;
    cfar_grid g;
    const int rc = build_grid(mode, N, looks, k, pfa, &g);
    if (rc) return rc;
    const double lnpfa = log(pfa);
    double a = log(1e-3), b = log(1e18);
    double fa = root_f(&g, looks, a, lnpfa), fb = root_f(&g, looks, b, lnpfa);
    if (!(fa > 0.0 && fb < 0.0)) { free(g.t); free(g.G); return ATKDSP_E_ARG; }
    int side = 0, have_prev = 0;
    double c = a, prev = 0.0;
    for (int it = 0; it < 300; ++it) {
        c = (a * fb - b * fa) / (fb - fa);
        const double fc = root_f(&g, looks, c, lnpfa);
        if (fc == 0.0) break;
        if (fc > 0.0) {
            a = c; fa = fc;
            if (side == 1) fb *= 0.5;
            side = 1;
        } else {
            b = c; fb = fc;
            if (side == -1) fa *= 0.5;
            side = -1;
        }
        if (have_prev && fabs(c - prev) <= 1e-14 * (fabs(c) > 1.0 ? fabs(c) : 1.0)) break;
        prev = c; have_prev = 1;
    }
    *alpha = exp(c);
    if (et) *et = g.et;
    free(g.t); free(g.G);
    return ATKDSP_OK;
}

/* beta with P(Gamma(nb looks) / (nb looks) > beta) = ATKDSP_CFAR_EDGE_PFA. */
static double edge_beta(int nb, double looks) {
    const double s = (double)nb * looks;
    const double target = log(ATKDSP_CFAR_EDGE_PFA);
    double a = log(1e-6), b = log(1e6);
    for (int i = 0; i < 100; ++i) {
        const double mid = 0.5 * (a + b);
        double lp, lq;
        atk_log_igam_pq(s, s * exp(mid), &lp, &lq);
        if (lq > target) a = mid; else b = mid;
    }
    return exp(b);
}

/* ---- the handle ---------------------------------------------------------------- */

typedef struct {
    uint32_t id;
    int      state;          /* 0 pending, 1 open                             */
    uint64_t hist;
    int      misses, is_new;
    int32_t  n_hits, cur_lo, cur_hi, ulo, uhi;
    double   sbe, se, px, pn, band;
    uint32_t flags;
    double   t0;
    int64_t  last_hit, last_raw;
    double   t_hit_end, dt_hit, t_raw_end;
    /* this frame's observation */
    int      has_obs;
    int32_t  olo, ohi;
    double   fsx, fsn;
} cfar_track;

typedef struct {
    int32_t lo, hi;
    double sbe, se, sx, sn, px, pn;
    int track;               /* index into tracks, -1 = none                  */
} cfar_obs;

struct atkdsp_cfar {
    atkdsp_cfar_config cfg;
    size_t nbins;
    int N;
    double K, alpha, alpha_ca, et;
    uint8_t *mask, *ok, *sided, *hits;
    int32_t *idx;
    double *hist, *ht, *hd;
    int64_t seen, frame;
    double *X, *Z, *noise, *thr, *sel;
    double *beta;
    int has_last;
    cfar_track *tracks;
    int ntracks;
    uint32_t next_id;
    cfar_obs *obs;
    atkdsp_cfar_counters st;
};

void atkdsp_cfar_defaults(atkdsp_cfar_config *c) {
    if (!c) return;
    memset(c, 0, sizeof *c);
    c->mode = ATKDSP_CFAR_CA; c->n_ref = 16; c->n_guard = 4; c->k_frames = 1;
    c->pfa = 1e-6; c->looks = 1.0; c->os_k = 0;
    c->open_m = 2; c->open_n = 3; c->close_k = 3;
    c->merge_gap = 1; c->min_bins = 1; c->max_tracks = 256;
}

static int config_ok(size_t nbins, const atkdsp_cfar_config *c) {
    const long N = 2L * c->n_ref;
    if (c->mode < 0 || c->mode > 3) return 0;
    if (c->n_ref < 1 || c->n_ref > 256 || c->n_guard < 0 || c->n_guard > 256) return 0;
    if (nbins < (size_t)(N + 2L * c->n_guard + 1) || nbins > ((size_t)1 << 24)
        || (double)nbins * (double)N > (double)((size_t)1 << 26)) return 0;
    if (c->k_frames < 1 || c->k_frames > 64 || !(c->looks >= 0.25 && c->looks <= 1e4)) return 0;
    if (!(c->pfa >= 1e-15 && c->pfa <= 0.5)) return 0;
    if (c->mode == ATKDSP_CFAR_OS && (c->os_k < 0 || c->os_k > N)) return 0;
    if (c->open_n < 1 || c->open_n > 64 || c->open_m < 1 || c->open_m > c->open_n) return 0;
    if (c->close_k < 1 || c->close_k > 1000 || c->merge_gap < 0
        || (size_t)c->merge_gap > nbins) return 0;
    if (c->min_bins < 1 || (size_t)c->min_bins > nbins
        || c->max_tracks < 1 || c->max_tracks > 65536) return 0;
    return 1;
}

/* The reference cells of every bin: left nearest-first, right nearest-first,
 * then any borrowed from the far side. Same walk as the twin's _tables. */
static void build_tables(atkdsp_cfar *c) {
    const long nb = (long)c->nbins, nr = c->cfg.n_ref, ng = c->cfg.n_guard, N = c->N;
    const uint8_t *m = c->mask;
    for (long i = 0; i < nb; ++i) {
        c->ok[i] = 0; c->sided[i] = 0;
        if (m[i]) continue;
        int32_t *cells = c->idx + (size_t)i * (size_t)N;
        long nl = 0, nrr = 0, ne = 0;
        int32_t left[256], right[256], ext[512];
        long j = i - ng - 1;
        while (j >= 0 && nl < nr) { if (!m[j]) left[nl++] = (int32_t)j; --j; }
        const long jl = j;
        j = i + ng + 1;
        while (j < nb && nrr < nr) { if (!m[j]) right[nrr++] = (int32_t)j; ++j; }
        const long jr = j;
        long short_l = nr - nl, short_r = nr - nrr;
        j = jr;
        while (short_l > 0 && j < nb) { if (!m[j]) { ext[ne++] = (int32_t)j; --short_l; } ++j; }
        j = jl;
        while (short_r > 0 && j >= 0) { if (!m[j]) { ext[ne++] = (int32_t)j; --short_r; } --j; }
        if (nl + nrr + ne < N) continue;
        long o = 0;
        for (long q = 0; q < nl; ++q) cells[o++] = left[q];
        for (long q = 0; q < nrr; ++q) cells[o++] = right[q];
        for (long q = 0; q < ne; ++q) cells[o++] = ext[q];
        c->ok[i] = 1;
        c->sided[i] = ne == 0;
    }
}

void atkdsp_cfar_destroy(atkdsp_cfar *c) {
    if (!c) return;
    free(c->mask); free(c->ok); free(c->sided); free(c->hits); free(c->idx);
    free(c->hist); free(c->ht); free(c->hd);
    free(c->X); free(c->Z); free(c->noise); free(c->thr); free(c->sel);
    free(c->beta); free(c->tracks); free(c->obs);
    free(c);
}

atkdsp_cfar *atkdsp_cfar_create(size_t nbins, const atkdsp_cfar_config *cfg) {
    if (!cfg || !config_ok(nbins, cfg)) return NULL;
    atkdsp_cfar *c = (atkdsp_cfar *)calloc(1, sizeof *c);
    if (!c) return NULL;
    c->cfg = *cfg;
    c->nbins = nbins;
    c->N = 2 * cfg->n_ref;
    if (c->cfg.mode == ATKDSP_CFAR_OS && c->cfg.os_k == 0) c->cfg.os_k = (3 * c->N + 3) / 4;
    c->K = c->cfg.looks * (double)c->cfg.k_frames;
    if (atkdsp_cfar_alpha(c->cfg.mode, c->cfg.n_ref, c->cfg.pfa, c->K, c->cfg.os_k,
                          &c->alpha, &c->et) != ATKDSP_OK) { free(c); return NULL; }
    if (c->cfg.mode == ATKDSP_CFAR_GO || c->cfg.mode == ATKDSP_CFAR_SO) {
        if (atkdsp_cfar_alpha(ATKDSP_CFAR_CA, c->cfg.n_ref, c->cfg.pfa, c->K, 0,
                              &c->alpha_ca, NULL) != ATKDSP_OK) { free(c); return NULL; }
    } else {
        c->alpha_ca = c->alpha;
    }
    const size_t k = (size_t)c->cfg.k_frames, mt = (size_t)c->cfg.max_tracks;
    c->mask = (uint8_t *)calloc(nbins, 1);
    c->ok = (uint8_t *)calloc(nbins, 1);
    c->sided = (uint8_t *)calloc(nbins, 1);
    c->hits = (uint8_t *)calloc(nbins, 1);
    c->idx = (int32_t *)calloc(nbins * (size_t)c->N, sizeof(int32_t));
    c->hist = (double *)calloc(k * nbins, sizeof(double));
    c->ht = (double *)calloc(k, sizeof(double));
    c->hd = (double *)calloc(k, sizeof(double));
    c->X = (double *)calloc(nbins, sizeof(double));
    c->Z = (double *)calloc(nbins, sizeof(double));
    c->noise = (double *)calloc(nbins, sizeof(double));
    c->thr = (double *)calloc(nbins, sizeof(double));
    c->sel = (double *)calloc((size_t)c->N, sizeof(double));
    c->beta = (double *)malloc((nbins + 1) * sizeof(double));
    c->tracks = (cfar_track *)calloc(mt, sizeof(cfar_track));
    c->obs = (cfar_obs *)calloc(nbins, sizeof(cfar_obs));
    if (!c->mask || !c->ok || !c->sided || !c->hits || !c->idx || !c->hist || !c->ht || !c->hd
        || !c->X || !c->Z || !c->noise || !c->thr || !c->sel || !c->beta || !c->tracks
        || !c->obs) { atkdsp_cfar_destroy(c); return NULL; }
    for (size_t i = 0; i <= nbins; ++i) c->beta[i] = NAN;
    c->next_id = 1;
    build_tables(c);
    return c;
}

void atkdsp_cfar_reset(atkdsp_cfar *c) {
    if (!c) return;
    c->ntracks = 0;
    c->seen = 0;
    c->frame = 0;
    memset(c->hist, 0, (size_t)c->cfg.k_frames * c->nbins * sizeof(double));
    c->has_last = 0;
    c->st.resets++;
}

int atkdsp_cfar_set_mask(atkdsp_cfar *c, const uint8_t *mask) {
    if (!c) return ATKDSP_E_ARG;
    for (size_t i = 0; i < c->nbins; ++i) c->mask[i] = mask ? (mask[i] != 0) : 0;
    build_tables(c);
    return ATKDSP_OK;
}

size_t atkdsp_cfar_events_max(const atkdsp_cfar *c, size_t nframes) {
    return c ? nframes * (size_t)c->cfg.max_tracks : 0;
}

/* k-th smallest (0-based) of a[0..n), Hoare quickselect (as detect.c's). */
static double select_kth_d(double *a, long n, long k) {
    long lo = 0, hi = n - 1;
    while (lo < hi) {
        const double pivot = a[lo + (hi - lo) / 2];
        long i = lo, j = hi;
        while (i <= j) {
            while (a[i] < pivot) ++i;
            while (a[j] > pivot) --j;
            if (i <= j) { const double t = a[i]; a[i] = a[j]; a[j] = t; ++i; --j; }
        }
        if (k <= j) hi = j;
        else if (k >= i) lo = i;
        else break;
    }
    return a[k];
}

static double beta_for(atkdsp_cfar *c, int nb) {
    if (isnan(c->beta[nb])) c->beta[nb] = edge_beta(nb, c->cfg.looks);
    return c->beta[nb];
}

static int band_pass(atkdsp_cfar *c, const double *raw, int32_t lo, int32_t hi) {
    double sr = 0.0, sz = 0.0;
    int nb = 0;
    for (int32_t b = lo; b <= hi; ++b) {
        if (c->ok[b]) { sr += raw[b]; sz += c->noise[b]; ++nb; }
    }
    return nb > 0 && sr > beta_for(c, nb) * sz;
}

static uint32_t event_flags(const atkdsp_cfar *c, const cfar_track *t) {
    uint32_t f = t->flags;
    for (int32_t b = t->ulo; b <= t->uhi; ++b)
        if (!c->sided[b] || !c->ok[b]) { f |= ATKDSP_CFAR_EDGE; break; }
    const long m = c->cfg.merge_gap + 1;
    long a = (long)t->ulo - m, z = (long)t->uhi + m + 1;
    if (a < 0) a = 0;
    if (z > (long)c->nbins) z = (long)c->nbins;
    for (long b = a; b < z; ++b) if (c->mask[b]) { f |= ATKDSP_CFAR_MASK_ADJ; break; }
    return f;
}

static void make_event(const atkdsp_cfar *c, const cfar_track *t, int type, double t1,
                       uint32_t extra, atkdsp_cfar_event *e) {
    memset(e, 0, sizeof *e);
    e->type = type;
    e->id = t->id;
    e->flags = event_flags(c, t) | extra;
    e->lo_bin = t->ulo; e->hi_bin = t->uhi;
    e->n_hits = t->n_hits;
    e->t0 = t->t0; e->t1 = t1;
    e->fc_bin = t->se > 0.0 ? t->sbe / t->se : 0.5 * ((double)t->ulo + (double)t->uhi);
    e->level_db = t->px > 0.0 ? (float)(10.0 * log10(t->px)) : -INFINITY;
    e->noise_db = t->pn > 0.0 ? (float)(10.0 * log10(t->pn)) : -INFINITY;
    e->snr_db = t->pn > 0.0 ? e->level_db - e->noise_db : INFINITY;
    e->band_snr_db = t->band > 0.0 ? (float)(10.0 * log10(t->band)) : -INFINITY;
}

static void close_event(atkdsp_cfar *c, cfar_track *t, uint32_t extra, atkdsp_cfar_event *e) {
    const int k = c->cfg.k_frames;
    double t1;
    if (k == 1) t1 = t->t_hit_end;
    else if (t->last_raw >= t->last_hit - (k - 1)) t1 = t->t_raw_end;
    else {
        t1 = t->t_hit_end - 0.5 * (double)(k - 1) * t->dt_hit;
        t->flags |= ATKDSP_CFAR_T1_COARSE;
    }
    c->st.closed++;
    make_event(c, t, ATKDSP_CFAR_CLOSE, t1, extra, e);
}

static int popcount64(uint64_t x) { return atk_popcount64(x); }

/* One frame. Returns events written. */
static size_t one_frame(atkdsp_cfar *c, const float *row, double tf, double dt,
                        atkdsp_cfar_event *ev) {
    const size_t nb = c->nbins;
    const int k = c->cfg.k_frames, N = c->N, n = c->cfg.n_ref, mode = c->cfg.mode;
    c->st.frames++;
    const int64_t F = c->frame;
    const size_t slot = (size_t)(F % k);
    double *raw = c->hist + slot * nb;
    for (size_t i = 0; i < nb; ++i) {
        double v = (double)row[i];
        if (v < -300.0) v = -300.0;
        raw[i] = exp(v * LN10_OVER_10);
    }
    c->ht[slot] = tf;
    c->hd[slot] = dt;
    c->seen++;
    c->frame++;
    if (c->seen < k) { c->st.warmup++; return 0; }

    /* the average, the reference, the test */
    for (size_t i = 0; i < nb; ++i) {
        double s = 0.0;
        for (int r = 0; r < k; ++r) s += c->hist[(size_t)((F + 1 + r) % k) * nb + i];
        c->X[i] = s / k;
    }
    const double *X = c->X;
    uint64_t tested = 0, nhit = 0, untest = 0;
    for (size_t i = 0; i < nb; ++i) {
        c->hits[i] = 0;
        if (!c->ok[i]) {
            c->thr[i] = NAN; c->Z[i] = NAN; c->noise[i] = NAN;
            if (!c->mask[i]) ++untest;
            continue;
        }
        const int32_t *cells = c->idx + i * (size_t)N;
        double Z, a = c->alpha, scale = c->et;
        if (mode == ATKDSP_CFAR_OS) {
            /* a NaN reference cell makes T NaN (no order holds it), as a NaN
             * makes CA's mean NaN: the bin is not hit — the header's rule */
            int has_nan = 0;
            for (int q = 0; q < N; ++q) {
                c->sel[q] = X[cells[q]];
                if (isnan(c->sel[q])) has_nan = 1;
            }
            Z = has_nan ? NAN : select_kth_d(c->sel, N, c->cfg.os_k - 1);
        } else if ((mode == ATKDSP_CFAR_GO || mode == ATKDSP_CFAR_SO) && c->sided[i]) {
            double L = 0.0, R = 0.0;
            for (int q = 0; q < n; ++q) L += X[cells[q]];
            for (int q = n; q < N; ++q) R += X[cells[q]];
            L /= n; R /= n;
            if (isnan(L) || isnan(R)) Z = NAN;          /* numpy's maximum / minimum */
            else Z = mode == ATKDSP_CFAR_GO ? (L > R ? L : R) : (L < R ? L : R);
        } else {
            double s = 0.0;
            for (int q = 0; q < N; ++q) s += X[cells[q]];
            Z = s / N;
            if (mode == ATKDSP_CFAR_GO || mode == ATKDSP_CFAR_SO) { a = c->alpha_ca; scale = 1.0; }
        }
        c->Z[i] = Z;
        c->thr[i] = a * Z;
        c->noise[i] = Z / scale;
        ++tested;
        if (X[i] > c->thr[i]) { c->hits[i] = 1; ++nhit; }
    }
    c->st.cells_tested += tested;
    c->st.cell_hits += nhit;
    c->st.untestable += untest;
    c->has_last = 1;

    /* blobs */
    const long g = c->cfg.merge_gap;
    size_t nobs = 0;
    long i = 0;
    while (i < (long)nb) {
        if (!c->hits[i]) { ++i; continue; }
        long lo = i, hi = i, prev = i;
        for (long j = i + 1; j < (long)nb; ++j) {
            if (j - prev - 1 > g) break;
            if (c->hits[j]) { hi = j; prev = j; }
        }
        i = hi + 1;
        if (hi - lo + 1 < c->cfg.min_bins) continue;
        cfar_obs *o = &c->obs[nobs++];
        o->lo = (int32_t)lo; o->hi = (int32_t)hi;
        o->sbe = o->se = o->sx = o->sn = 0.0;
        long pk = -1;
        for (long b = lo; b <= hi; ++b) {
            if (!c->hits[b]) continue;
            double e = X[b] - c->noise[b];
            if (e < 0.0) e = 0.0;
            o->sbe += (double)b * e;
            o->se += e;
            o->sx += X[b];
            o->sn += c->noise[b];
            if (pk < 0 || X[b] > X[pk]) pk = b;
        }
        o->px = X[pk]; o->pn = c->noise[pk];
        o->track = -1;
    }

    /* association: each blob to the existing track it overlaps most */
    const int nexist = c->ntracks;
    for (int t = 0; t < nexist; ++t) c->tracks[t].has_obs = 0;
    const long m = g + 1;
    for (size_t q = 0; q < nobs; ++q) {
        cfar_obs *o = &c->obs[q];
        int best = -1;
        long bov = 0;
        for (int t = 0; t < nexist; ++t) {
            const cfar_track *tr = &c->tracks[t];
            if ((long)o->lo <= (long)tr->cur_hi + m && (long)o->hi >= (long)tr->cur_lo - m) {
                const long ov = (long)(o->hi < tr->cur_hi ? o->hi : tr->cur_hi)
                              - (long)(o->lo > tr->cur_lo ? o->lo : tr->cur_lo) + 1;
                if (best < 0 || ov > bov) { best = t; bov = ov; }
            }
        }
        o->track = best;
    }
    for (size_t q = 0; q < nobs; ++q) {
        cfar_obs *o = &c->obs[q];
        if (o->track >= 0) continue;
        if (c->ntracks >= c->cfg.max_tracks) { c->st.dropped++; continue; }
        cfar_track *tr = &c->tracks[c->ntracks];
        memset(tr, 0, sizeof *tr);
        tr->id = c->next_id;
        c->next_id = c->next_id + 1 ? c->next_id + 1 : 1;
        tr->cur_lo = tr->ulo = o->lo;
        tr->cur_hi = tr->uhi = o->hi;
        tr->px = -1.0;
        tr->last_hit = -1; tr->last_raw = -1;
        tr->dt_hit = dt;
        tr->is_new = 1;
        o->track = c->ntracks++;
    }
    /* each track's observation, its blobs taken in blob order */
    for (size_t q = 0; q < nobs; ++q) {
        const cfar_obs *o = &c->obs[q];
        if (o->track < 0) continue;
        cfar_track *tr = &c->tracks[o->track];
        if (!tr->has_obs) {
            tr->has_obs = 1; tr->olo = o->lo; tr->ohi = o->hi; tr->fsx = 0.0; tr->fsn = 0.0;
        } else {
            if (o->lo < tr->olo) tr->olo = o->lo;
            if (o->hi > tr->ohi) tr->ohi = o->hi;
        }
        tr->sbe += o->sbe;
        tr->se += o->se;
        tr->fsx += o->sx;
        tr->fsn += o->sn;
        if (o->px > tr->px) { tr->px = o->px; tr->pn = o->pn; }
    }

    /* update, in id order; compact as we go */
    const uint64_t maskn = (c->cfg.open_n >= 64) ? ~(uint64_t)0
                         : (((uint64_t)1 << c->cfg.open_n) - 1);
    size_t nev = 0;
    int keep = 0;
    for (int t = 0; t < c->ntracks; ++t) {
        cfar_track *tr = &c->tracks[t];
        int grew = 0;
        if (tr->has_obs) {
            const int32_t lo = tr->olo, hi = tr->ohi;
            tr->hist = ((tr->hist << 1) | 1u) & maskn;
            tr->misses = 0;
            tr->n_hits++;
            tr->cur_lo = lo; tr->cur_hi = hi;
            if (!tr->is_new) grew = lo < tr->ulo || hi > tr->uhi;
            if (lo < tr->ulo) tr->ulo = lo;
            if (hi > tr->uhi) tr->uhi = hi;
            if (tr->fsn > 0.0 && tr->fsx / tr->fsn > tr->band) tr->band = tr->fsx / tr->fsn;
            tr->last_hit = F;
            tr->t_hit_end = tf + dt;
            tr->dt_hit = dt;
            if (tr->is_new) {
                tr->is_new = 0;
                if (k == 1) tr->t0 = tf;
                else {
                    int found = 0;
                    for (int r = 0; r < k; ++r) {
                        const size_t s = (size_t)((F + 1 + r) % k);
                        if (band_pass(c, c->hist + s * nb, lo, hi)) {
                            tr->t0 = c->ht[s]; found = 1; break;
                        }
                    }
                    if (!found) {
                        tr->t0 = tf - 0.5 * (double)(k - 1) * dt;
                        tr->flags |= ATKDSP_CFAR_T0_COARSE;
                    }
                }
            }
        } else {
            tr->hist = (tr->hist << 1) & maskn;
            tr->misses++;
        }
        tr->has_obs = 0;
        if (k > 1 && band_pass(c, raw, tr->cur_lo, tr->cur_hi)) {
            tr->last_raw = F;
            tr->t_raw_end = tf + dt;
        }
        if (tr->state == 0) {
            if (popcount64(tr->hist) >= c->cfg.open_m) {
                tr->state = 1;
                c->st.opened++;
                make_event(c, tr, ATKDSP_CFAR_OPEN, NAN, 0, &ev[nev++]);
            } else if (tr->hist == 0) {
                continue;
            }
        } else if (tr->misses >= c->cfg.close_k) {
            close_event(c, tr, 0, &ev[nev++]);
            continue;
        } else if (grew) {
            c->st.updates++;
            make_event(c, tr, ATKDSP_CFAR_UPDATE, NAN, 0, &ev[nev++]);
        }
        if (keep != t) c->tracks[keep] = *tr;
        ++keep;
    }
    c->ntracks = keep;
    return nev;
}

ptrdiff_t atkdsp_cfar_process(atkdsp_cfar *c, const float *frames_db, size_t nframes,
                              double frame_t0, double frame_dt, atkdsp_cfar_event *ev,
                              size_t cap) {
    if (!c || (!frames_db && nframes) || (!ev && nframes)) return ATKDSP_E_ARG;
    if (cap < atkdsp_cfar_events_max(c, nframes)) return ATKDSP_E_CAP;
    size_t nev = 0;
    for (size_t j = 0; j < nframes; ++j)
        nev += one_frame(c, frames_db + j * c->nbins, frame_t0 + (double)j * frame_dt,
                         frame_dt, ev + nev);
    return (ptrdiff_t)nev;
}

ptrdiff_t atkdsp_cfar_flush(atkdsp_cfar *c, atkdsp_cfar_event *ev, size_t cap) {
    if (!c) return ATKDSP_E_ARG;
    size_t open = 0;
    for (int t = 0; t < c->ntracks; ++t) open += c->tracks[t].state == 1;
    if (open && (!ev || cap < open)) return ATKDSP_E_CAP;
    size_t nev = 0;
    for (int t = 0; t < c->ntracks; ++t)
        if (c->tracks[t].state == 1)
            close_event(c, &c->tracks[t], ATKDSP_CFAR_FLUSHED, &ev[nev++]);
    c->ntracks = 0;
    return (ptrdiff_t)nev;
}

int atkdsp_cfar_last(const atkdsp_cfar *c, float *avg_db, float *thr_db, uint8_t *hits) {
    if (!c) return ATKDSP_E_ARG;
    if (!c->has_last) return ATKDSP_E_EMPTY;
    for (size_t i = 0; i < c->nbins; ++i) {
        if (avg_db) avg_db[i] = (float)(10.0 * log10(c->X[i]));
        if (thr_db) thr_db[i] = (float)(10.0 * log10(c->thr[i]));
        if (hits) hits[i] = c->hits[i];
    }
    return ATKDSP_OK;
}

int atkdsp_cfar_info(const atkdsp_cfar *c, double *alpha, double *alpha_ca, double *et,
                     atkdsp_cfar_config *cfg) {
    if (!c) return ATKDSP_E_ARG;
    if (alpha) *alpha = c->alpha;
    if (alpha_ca) *alpha_ca = c->alpha_ca;
    if (et) *et = c->et;
    if (cfg) *cfg = c->cfg;
    return ATKDSP_OK;
}

int atkdsp_cfar_describe(const atkdsp_cfar *c, char *buf, size_t cap) {
    if (!c || !buf || !cap) return ATKDSP_E_ARG;
    static const char *names[4] = {"CA", "OS", "GO", "SO"};
    char os_txt[48] = "";
    if (c->cfg.mode == ATKDSP_CFAR_OS)
        snprintf(os_txt, sizeof os_txt, " (rank %d of %d)", c->cfg.os_k, c->N);
    size_t masked = 0;
    for (size_t i = 0; i < c->nbins; ++i) masked += c->mask[i] != 0;
    snprintf(buf, cap,
             "CFAR %s%s: %d+%d reference cells, %d guard each side, Pfa %.3g per cell per "
             "frame, K = %g look(s) x %d frame(s), alpha %.4g (%.2f dB); open %d of %d, "
             "close after %d misses, merge gap %d, min %d bin(s); %zu bins, %zu masked, "
             "tracks %d/%d",
             names[c->cfg.mode], os_txt, c->cfg.n_ref, c->cfg.n_ref, c->cfg.n_guard, c->cfg.pfa,
             c->cfg.looks, c->cfg.k_frames, c->alpha, 10.0 * log10(c->alpha), c->cfg.open_m,
             c->cfg.open_n, c->cfg.close_k, c->cfg.merge_gap, c->cfg.min_bins, c->nbins, masked,
             c->ntracks, c->cfg.max_tracks);
    return ATKDSP_OK;
}

int atkdsp_cfar_stats(const atkdsp_cfar *c, atkdsp_cfar_counters *st) {
    if (!c || !st) return ATKDSP_E_ARG;
    *st = c->st;
    st->tracks = (uint64_t)c->ntracks;
    return ATKDSP_OK;
}
