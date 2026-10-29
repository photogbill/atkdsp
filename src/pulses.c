/* 22. Live pulse runs: radar PDWs (docs/ATKDSP_PLAN.md §B3).
 *
 * Envelope -> threshold with hysteresis -> runs carried across block edges,
 * one pulse descriptor word per run. The definitions are in atkdsp.h §22;
 * the twin is python/atkdsp/reference.py §22 (class Pulses), and with
 * hyst_db = 0 both reproduce ATK's atk.core.pulse.detect_pulses word for
 * word: the same floor, threshold, flank trimming and 2.77 sigma chirp.
 *
 * Arithmetic, so the twin can match it: |x| is sqrt(re^2 + im^2) in double
 * rounded to float;
 * the instantaneous frequency is atan2f of the float product x[n] x*[n-1],
 * times (float)(fs / 2 pi) — pulse.py's complex64 arithmetic — and every sum
 * over a run is a double accumulated in sample order.
 */
#include "internal.h"

#include <limits.h>
#include <stdio.h>

struct atkdsp_pulses {
    atkdsp_pulses_config cfg;
    double t0;
    double thr_open, thr_close;      /* linear magnitude                     */
    uint64_t idx;                    /* index of the next sample since reset  */
    float prev_mag;                  /* |x| of the previous sample            */
    int have_prev;
    /* the open run */
    int open;
    uint64_t start;                  /* index of its first sample             */
    double open_cross;               /* the opening crossing, in samples      */
    uint32_t flags;
    atkdsp_cf32 *buf;                /* its samples, max_samples at most      */
    int32_t nbuf;
    uint64_t nrun;                   /* its length so far (may exceed nbuf)   */
    double sum_mag, sum_pow, peak;
    atkdsp_pulses_counters st;
};

void atkdsp_pulses_defaults(atkdsp_pulses_config *c) {
    if (!c) return;
    memset(c, 0, sizeof *c);
    c->fs = 1.0; c->floor = 1.0; c->thresh_db = 10.0; c->hyst_db = 0.0;
    c->min_samples = 4; c->max_samples = 65536;
}

static int config_ok(const atkdsp_pulses_config *c) {
    if (!(c->fs > 0.0) || !isfinite(c->fs)) return 0;
    if (!(c->floor > 0.0) || !isfinite(c->floor)) return 0;
    if (!(c->thresh_db >= 0.0 && c->thresh_db <= 120.0)) return 0;
    if (!(c->hyst_db >= 0.0 && c->hyst_db <= c->thresh_db)) return 0;
    if (c->min_samples < 1 || c->min_samples > (1 << 20)) return 0;
    if (c->max_samples < 4 || c->max_samples > (1 << 24)) return 0;
    return 1;
}

static void set_thresholds(atkdsp_pulses *p) {
    p->thr_open = p->cfg.floor * pow(10.0, p->cfg.thresh_db / 20.0);
    p->thr_close = p->cfg.floor * pow(10.0, (p->cfg.thresh_db - p->cfg.hyst_db) / 20.0);
}

atkdsp_pulses *atkdsp_pulses_create(const atkdsp_pulses_config *cfg, double t0) {
    if (!cfg || !config_ok(cfg) || !isfinite(t0)) return NULL;
    atkdsp_pulses *p = (atkdsp_pulses *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->cfg = *cfg;
    p->buf = (atkdsp_cf32 *)malloc((size_t)cfg->max_samples * sizeof *p->buf);
    if (!p->buf) { free(p); return NULL; }
    p->t0 = t0;
    set_thresholds(p);
    return p;
}

void atkdsp_pulses_destroy(atkdsp_pulses *p) {
    if (!p) return;
    free(p->buf);
    free(p);
}

void atkdsp_pulses_reset(atkdsp_pulses *p, double t0) {
    if (!p) return;
    p->t0 = isfinite(t0) ? t0 : 0.0;
    p->idx = 0; p->have_prev = 0; p->prev_mag = 0.0f;
    p->open = 0; p->nbuf = 0; p->nrun = 0;
    p->st.open = 0;
    p->st.resets++;
}

int atkdsp_pulses_set_floor(atkdsp_pulses *p, double floor_) {
    if (!p || !(floor_ > 0.0) || !isfinite(floor_)) return ATKDSP_E_ARG;
    p->cfg.floor = floor_;
    set_thresholds(p);
    return ATKDSP_OK;
}

size_t atkdsp_pulses_max(const atkdsp_pulses *p, size_t n) {
    if (!p) return 0;
    return n / ((size_t)p->cfg.min_samples + 1) + 1;
}

ATK_INLINE float mag_of(const atkdsp_cf32 *v) {
    return sqrtf(v->re * v->re + v->im * v->im);
}

/* Measure the closed run into *w. close_cross is the closing crossing in
 * samples (absolute index space). */
static void measure(atkdsp_pulses *p, double close_cross, atkdsp_pdw *w) {
    const double fs = p->cfg.fs;
    const int32_t n = p->nbuf;
    memset(w, 0, sizeof *w);
    w->start = p->start;
    w->end = p->start + p->nrun;
    w->n = (int32_t)(p->nrun > (uint64_t)INT32_MAX ? INT32_MAX : p->nrun);
    w->flags = p->flags;
    w->toa = p->t0 + p->open_cross / fs;
    w->pw = (close_cross - p->open_cross) / fs;
    const double mean = p->sum_mag / (double)p->nrun;
    double r = mean / p->cfg.floor;
    w->pa_db = (float)(20.0 * log10(r > 1e-12 ? r : 1e-12));
    r = p->peak / p->cfg.floor;
    w->peak_db = (float)(20.0 * log10(r > 1e-12 ? r : 1e-12));
    w->energy = (float)p->sum_pow;
    if (n < 4) { w->flags |= ATKDSP_PDW_NO_FREQ; return; }
    /* instantaneous frequency inside the run: n - 1 readings; the core trims
     * max(1, (n - 1) / 10) off each end, or keeps all when too short */
    const int32_t n_inst = n - 1;
    int32_t trim = n_inst / 10;
    if (trim < 1) trim = 1;
    int32_t lo = trim, hi = n_inst - trim;          /* [lo, hi) in reading space */
    if (hi - lo < 1) { lo = 0; hi = n_inst; }
    const float scale = (float)(fs / (2.0 * ATK_PI));
    double s1 = 0.0, s2 = 0.0, sk = 0.0, skk = 0.0, skf = 0.0;
    for (int32_t i = lo; i < hi; ++i) {
        const atkdsp_cf32 a = p->buf[i + 1], b = p->buf[i];
        const float pr = a.re * b.re + a.im * b.im;
        const float pi = a.im * b.re - a.re * b.im;
        const float f = atan2f(pi, pr) * scale;
        const double k = (double)(i - lo);
        s1 += f; s2 += (double)f * (double)f;
        sk += k; skk += k * k; skf += k * f;
    }
    const double m = (double)(hi - lo);
    const double mean_f = s1 / m;
    double var = s2 / m - mean_f * mean_f;
    if (var < 0.0) var = 0.0;
    w->freq_hz = mean_f;
    w->chirp_hz = 2.77 * sqrt(var);
    if (m >= 3.0) {
        const double den = m * skk - sk * sk;
        if (den > 0.0) w->slope_hz_s = (m * skf - sk * s1) / den * fs;
    }
}

ptrdiff_t atkdsp_pulses_process(atkdsp_pulses *p, const atkdsp_cf32 *x, size_t n,
                                atkdsp_pdw *out, size_t cap) {
    if (!p || (!x && n) || (!out && cap)) return ATKDSP_E_ARG;
    size_t got = 0;
    for (size_t i = 0; i < n; ++i) {
        const float mag = mag_of(&x[i]);
        const double md = (double)mag;
        if (!p->open) {
            if (md > p->thr_open) {
                p->open = 1;
                p->start = p->idx;
                p->flags = 0;
                if (p->have_prev) {
                    const double m0 = (double)p->prev_mag, m1 = md;
                    p->open_cross = (double)(p->idx - 1) + (p->thr_open - m0) / (m1 - m0);
                } else {
                    p->open_cross = (double)p->idx;
                    p->flags |= ATKDSP_PDW_EDGE_START;
                }
                p->nbuf = 0; p->nrun = 0;
                p->sum_mag = p->sum_pow = 0.0; p->peak = 0.0;
                p->st.open = 1;
            }
        } else if (md <= p->thr_close) {
            /* the run closes here: the crossing between the previous sample
             * (above the closing level) and this one */
            const double m0 = (double)p->prev_mag, m1 = md;
            const double close_cross = (double)(p->idx - 1) + (m0 - p->thr_close) / (m0 - m1);
            p->open = 0;
            p->st.open = 0;
            if (p->nrun < (uint64_t)p->cfg.min_samples) {
                p->st.short_runs++;
            } else {
                p->st.pulses++;
                if (p->flags & ATKDSP_PDW_TRUNCATED) p->st.truncated++;
                if (got < cap) measure(p, close_cross, &out[got++]);
                else p->st.dropped++;
            }
        }
        if (p->open) {
            if (p->nbuf < p->cfg.max_samples) p->buf[p->nbuf++] = x[i];
            else p->flags |= ATKDSP_PDW_TRUNCATED;
            p->nrun++;
            p->sum_mag += md;
            p->sum_pow += (double)x[i].re * (double)x[i].re + (double)x[i].im * (double)x[i].im;
            if (md > p->peak) p->peak = md;
        }
        p->prev_mag = mag;
        p->have_prev = 1;
        p->idx++;
    }
    p->st.samples += n;
    return (ptrdiff_t)got;
}

ptrdiff_t atkdsp_pulses_flush(atkdsp_pulses *p, atkdsp_pdw *out, size_t cap) {
    if (!p || (!out && cap)) return ATKDSP_E_ARG;
    if (!p->open) return 0;
    p->open = 0;
    p->st.open = 0;
    if (p->nrun < (uint64_t)p->cfg.min_samples) { p->st.short_runs++; return 0; }
    p->st.pulses++;
    if (p->flags & ATKDSP_PDW_TRUNCATED) p->st.truncated++;
    if (cap < 1) { p->st.dropped++; return 0; }
    p->flags |= ATKDSP_PDW_FLUSHED;
    measure(p, (double)(p->idx - 1), out);
    return 1;
}

int atkdsp_pulses_describe(const atkdsp_pulses *p, char *buf, size_t cap) {
    if (!p || !buf || !cap) return ATKDSP_E_ARG;
    snprintf(buf, cap,
             "pulses: fs %.6g Hz, floor %.4g (%.1f dBFS), open at +%.1f dB, close at +%.1f dB, "
             "min %d samples, keep %d; %llu pulses, %llu dropped, %llu short, open: %s",
             p->cfg.fs, p->cfg.floor, 20.0 * log10(p->cfg.floor), p->cfg.thresh_db,
             p->cfg.thresh_db - p->cfg.hyst_db, p->cfg.min_samples, p->cfg.max_samples,
             (unsigned long long)p->st.pulses, (unsigned long long)p->st.dropped,
             (unsigned long long)p->st.short_runs, p->open ? "yes" : "no");
    return ATKDSP_OK;
}

int atkdsp_pulses_stats(const atkdsp_pulses *p, atkdsp_pulses_counters *st) {
    if (!p || !st) return ATKDSP_E_ARG;
    *st = p->st;
    return ATKDSP_OK;
}

/* ---- the median of |x| ------------------------------------------------------- */
/* Quickselect (Hoare partition, median-of-three pivot): a[k] ends with the
 * k-th smallest, everything before it no larger. Signed indices on purpose. */
static float select_kth(float *a, long n, long k) {
    long lo = 0, hi = n - 1;
    while (lo < hi) {
        const long mid = lo + (hi - lo) / 2;
        float p0 = a[lo], p1 = a[mid], p2 = a[hi], pivot;
        if (p0 > p1) { const float t = p0; p0 = p1; p1 = t; }
        if (p1 > p2) { p1 = p2; }
        pivot = p0 > p1 ? p0 : p1;             /* the median of three */
        long i = lo, j = hi;
        while (i <= j) {
            while (a[i] < pivot) ++i;
            while (a[j] > pivot) --j;
            if (i <= j) { const float t = a[i]; a[i] = a[j]; a[j] = t; ++i; --j; }
        }
        if (k <= j) hi = j;
        else if (k >= i) lo = i;
        else break;
    }
    return a[k];
}

int atkdsp_mag_median(const atkdsp_cf32 *x, size_t n, double *median) {
    if (!x || !n || !median || n > (size_t)LONG_MAX) return ATKDSP_E_ARG;
    float *m = (float *)malloc(n * sizeof *m);
    if (!m) return ATKDSP_E_NOMEM;
    for (size_t i = 0; i < n; ++i) m[i] = mag_of(&x[i]);
    const long k = (long)(n / 2);
    const float hi = select_kth(m, (long)n, k);
    if (n & 1) *median = (double)hi;
    else {
        /* numpy's median of an even count: the mean of the two middle values.
         * After the selection every a[i < k] <= a[k], so the lower one is the
         * largest of them. */
        float lo = m[0];
        for (long i = 1; i < k; ++i) if (m[i] > lo) lo = m[i];
        *median = ((double)lo + (double)hi) / 2.0;
    }
    free(m);
    return ATKDSP_OK;
}
