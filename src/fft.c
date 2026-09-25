/* 5. FFT, windows, power spectra and the 100 % POI reduction.
 *
 * Backend: pocketfft (C, double precision, BSD-3, vendored in
 * vendor/pocketfft). It is used through this file only, so a float32 kernel
 * can replace it later behind the same ABI. The float<->double copies cost
 * far less than the transform for the sizes a spectrum display uses. */
#include "internal.h"
#include "pocketfft.h"

#ifdef _OPENMP
#  include <omp.h>
#endif

#define MAX_SLOTS 64

struct atkdsp_fft {
    cfft_plan plan;
    size_t    n;
    int       slots;          /* scratch slots (>= worker threads, + headroom) */
    double   *scratch;        /* slots * 2n doubles */
    float    *acc;            /* slots * n floats: per-thread reduction lines */
    int      *busy;           /* slots claim flags, 0 free / 1 in use          */
    /* Lazily allocated the first time atkdsp_spectrum_stats runs, then kept:
     * four per-thread accumulator planes (max, min, sum P, sum P^2), so a
     * plan only ever used for the display reduction pays nothing for them.
     * `stats_init` is the one-time-alloc claim (0 -> 1 winner allocates). */
    double   *stats;          /* slots * 4 * n doubles, or NULL until needed   */
    int       stats_init;
};

atkdsp_fft *atkdsp_fft_create(size_t n) {
    if (n < 2) return NULL;
    atkdsp_fft *p = (atkdsp_fft *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->n = n;
    p->slots = 1;
#ifdef _OPENMP
    /* One slot per worker the batched reduction may spawn, plus a little
     * headroom so a thread sharing this plan for a single exec/power_db does
     * not have to wait out a whole spectrum_reduce to get a slot. */
    p->slots = omp_get_max_threads() + 2;
    if (p->slots < 1) p->slots = 1;
    if (p->slots > MAX_SLOTS) p->slots = MAX_SLOTS;
#endif
    p->plan = make_cfft_plan(n);
    p->scratch = (double *)atk_aligned_malloc((size_t)p->slots * 2 * n * sizeof(double));
    p->acc = (float *)atk_aligned_malloc((size_t)p->slots * n * sizeof(float));
    p->busy = (int *)atk_aligned_malloc((size_t)p->slots * sizeof(int));
    if (!p->plan || !p->scratch || !p->acc || !p->busy) {
        atkdsp_fft_destroy(p); return NULL;
    }
    memset(p->busy, 0, (size_t)p->slots * sizeof(int));
    return p;
}

void atkdsp_fft_destroy(atkdsp_fft *p) {
    if (!p) return;
    if (p->plan) destroy_cfft_plan(p->plan);
    atk_aligned_free(p->scratch);
    atk_aligned_free(p->acc);
    atk_aligned_free(p->busy);
    atk_aligned_free(p->stats);
    free(p);
}

size_t atkdsp_fft_length(const atkdsp_fft *p) { return p ? p->n : 0; }

/* Claim a free scratch slot for the calling thread. Spins if every slot is in
 * use — which needs another thread to be mid-call on this same plan, and each
 * holder releases within one transform, so this always makes progress. The
 * cast drops const: the plan's compute buffers are scratch, not state, and
 * claiming one does not change what the plan represents — which is what lets
 * exec/power_db keep the `const atkdsp_fft *` the ABI promises. */
static int claim_slot(const atkdsp_fft *p) {
    for (;;) {
        for (int s = 0; s < p->slots; ++s)
            if (atk_try_claim(&((atkdsp_fft *)p)->busy[s]))
                return s;
        /* every slot busy: yield the smallest amount and try again */
#if defined(_OPENMP)
        /* nothing to call portably; a short spin is fine for a sub-ms hold */
#endif
    }
}

static void free_slot(const atkdsp_fft *p, int s) {
    atk_release(&((atkdsp_fft *)p)->busy[s]);
}

/* transform `in` (windowed if window) into the slot's double scratch */
static int transform(const atkdsp_fft *p, const atkdsp_cf32 *in, const float *window,
                     int inverse, double *sc) {
    const size_t n = p->n;
    if (window) {
        for (size_t i = 0; i < n; ++i) {
            sc[2 * i]     = (double)in[i].re * (double)window[i];
            sc[2 * i + 1] = (double)in[i].im * (double)window[i];
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            sc[2 * i]     = (double)in[i].re;
            sc[2 * i + 1] = (double)in[i].im;
        }
    }
    int rc = inverse ? cfft_backward(p->plan, sc, 1.0 / (double)n)
                     : cfft_forward(p->plan, sc, 1.0);
    return rc == 0 ? ATKDSP_OK : ATKDSP_E_FFT;
}

int atkdsp_fft_exec(const atkdsp_fft *p, const atkdsp_cf32 *in, atkdsp_cf32 *out, int inverse) {
    if (!p || !in || !out) return ATKDSP_E_ARG;
    int s = claim_slot(p);
    double *sc = p->scratch + (size_t)s * 2 * p->n;
    int rc = transform(p, in, NULL, inverse, sc);
    if (!rc)
        for (size_t i = 0; i < p->n; ++i) {
            out[i].re = (float)sc[2 * i];
            out[i].im = (float)sc[2 * i + 1];
        }
    free_slot(p, s);
    return rc;
}

int atkdsp_window(int kind, size_t n, float *out) {
    if (!out || n == 0) return ATKDSP_E_ARG;
    if (n == 1) { out[0] = 1.0f; return ATKDSP_OK; }
    const double d = (double)(n - 1);
    for (size_t i = 0; i < n; ++i) {
        const double x = 2.0 * ATK_PI * (double)i / d;
        double w;
        switch (kind) {
        case ATKDSP_WIN_RECT:    w = 1.0; break;
        case ATKDSP_WIN_HANN:    w = 0.5 - 0.5 * cos(x); break;
        case ATKDSP_WIN_HAMMING: w = 0.54 - 0.46 * cos(x); break;
        case ATKDSP_WIN_BLACKMANHARRIS:
            w = 0.35875 - 0.48829 * cos(x) + 0.14128 * cos(2 * x) - 0.01168 * cos(3 * x);
            break;
        default: return ATKDSP_E_ARG;
        }
        out[i] = (float)w;
    }
    return ATKDSP_OK;
}

/* |X|/n in dB, fftshifted, from the slot scratch into out */
static void mag_db_shifted(const double *sc, size_t n, float *out) {
    const size_t half = n / 2;
    const float inv = 1.0f / (float)n;
    for (size_t i = 0; i < n; ++i) {
        const size_t k = (i + half) % n;           /* fftshift: bin i of output = bin k of X */
        const float re = (float)sc[2 * k], im = (float)sc[2 * k + 1];
        const float mag = sqrtf(re * re + im * im) * inv;
        out[i] = 20.0f * log10f(mag + 1e-12f);   /* identical to atk.core.dsp.spectrum_db */
    }
}

int atkdsp_power_db(const atkdsp_fft *p, const atkdsp_cf32 *in, const float *window, float *out_db) {
    if (!p || !in || !out_db) return ATKDSP_E_ARG;
    int s = claim_slot(p);
    double *sc = p->scratch + (size_t)s * 2 * p->n;
    int rc = transform(p, in, window, 0, sc);
    if (!rc) mag_db_shifted(sc, p->n, out_db);
    free_slot(p, s);
    return rc;
}

/* linear power |X|^2/n^2, fftshifted, ACCUMULATED into acc by detector */
static void accumulate(const double *sc, size_t n, int detector, float *acc, int first) {
    const size_t half = n / 2;
    const float inv2 = 1.0f / ((float)n * (float)n);
    for (size_t i = 0; i < n; ++i) {
        const size_t k = (i + half) % n;
        const float re = (float)sc[2 * k], im = (float)sc[2 * k + 1];
        const float pw = (re * re + im * im) * inv2;
        if (first) { acc[i] = pw; continue; }
        switch (detector) {
        case ATKDSP_DET_MAX: if (pw > acc[i]) acc[i] = pw; break;
        case ATKDSP_DET_MIN: if (pw < acc[i]) acc[i] = pw; break;
        default:             acc[i] += pw; break;
        }
    }
}

ptrdiff_t atkdsp_spectrum_reduce(const atkdsp_fft *p, const atkdsp_cf32 *in,
                                 size_t n_samples, size_t hop, const float *window,
                                 int detector, float *out_line) {
    if (!p || !in || !out_line) return ATKDSP_E_ARG;
    if (detector < ATKDSP_DET_MAX || detector > ATKDSP_DET_AVG) return ATKDSP_E_ARG;
    const size_t n = p->n;
    if (hop == 0 || hop > n) hop = n;
    if (n_samples < n) return 0;
    const long long frames = (long long)((n_samples - n) / hop + 1);

    /* Each worker CLAIMS a scratch slot and holds it across its frames and
     * the merge — the same discipline exec/power_db use, so a spectrum_reduce
     * and an exec sharing this plan on two threads take disjoint slots instead
     * of both landing on slot 0 (the omp_get_thread_num bug: that number is 0
     * for every thread OUTSIDE a parallel region, which is exactly where an
     * external exec caller runs). Slots are released after the merge reads
     * them, never before. */
    int used[MAX_SLOTS];
    int claimed[MAX_SLOTS];
    int nclaimed = 0;
    memset(used, 0, sizeof used);
    int rc = ATKDSP_OK;
    /* MSVC implements OpenMP 2.0, which rejects a loop variable DECLARED in
     * the for-initialiser (error C3015). Declared here, it is made private
     * to each thread by the pragma. Found on the first Windows build. */
    long long f;

#ifdef _OPENMP
    int nthreads = atkdsp_get_threads();
    if (nthreads > p->slots) nthreads = p->slots;
    if (nthreads < 1) nthreads = 1;
#   pragma omp parallel num_threads(nthreads)
    {
        int s = claim_slot(p);
        double *sc = p->scratch + (size_t)s * 2 * n;
        float  *acc = p->acc + (size_t)s * n;
        int first = 1;
#       pragma omp for schedule(static) nowait
        for (f = 0; f < frames; ++f) {
            if (transform(p, in + (size_t)f * hop, window, 0, sc) != ATKDSP_OK) {
                rc = ATKDSP_E_FFT;
                continue;
            }
            accumulate(sc, n, detector, acc, first);
            first = 0;
        }
#       pragma omp critical
        {
            claimed[nclaimed++] = s;
            if (!first) used[s] = 1;        /* this slot holds a partial line */
        }
    }
#else
    {
        int s = claim_slot(p);
        double *sc = p->scratch + (size_t)s * 2 * n;
        float  *acc = p->acc + (size_t)s * n;
        int first = 1;
        for (f = 0; f < frames; ++f) {
            if (transform(p, in + (size_t)f * hop, window, 0, sc) != ATKDSP_OK) {
                rc = ATKDSP_E_FFT;
                continue;
            }
            accumulate(sc, n, detector, acc, first);
            first = 0;
        }
        claimed[nclaimed++] = s;
        if (!first) used[s] = 1;
    }
#endif
    if (rc) {
        for (int i = 0; i < nclaimed; ++i) free_slot(p, claimed[i]);
        return rc;
    }

    /* merge the per-thread lines into the first used slot, then to dB */
    int base = -1;
    for (int i = 0; i < nclaimed; ++i) {
        int s = claimed[i];
        if (!used[s]) continue;
        if (base < 0) { base = s; continue; }
        float *acc = p->acc + (size_t)s * n;
        float *bacc = p->acc + (size_t)base * n;
        for (size_t k = 0; k < n; ++k) {
            switch (detector) {
            case ATKDSP_DET_MAX: if (acc[k] > bacc[k]) bacc[k] = acc[k]; break;
            case ATKDSP_DET_MIN: if (acc[k] < bacc[k]) bacc[k] = acc[k]; break;
            default:             bacc[k] += acc[k]; break;
            }
        }
    }
    if (base < 0) base = claimed[0];        /* frames >= 1, so this is safe */
    float *macc = p->acc + (size_t)base * n;
    const float scale = detector == ATKDSP_DET_AVG ? 1.0f / (float)frames : 1.0f;
    for (size_t i = 0; i < n; ++i) {
        /* 10*log10(power) == 20*log10(mag); keep the +1e-12 floor on the magnitude
         * scale so a silent bin reads -240 dB exactly as spectrum_db does */
        const float mag = sqrtf(macc[i] * scale);
        out_line[i] = 20.0f * log10f(mag + 1e-12f);
    }
    for (int i = 0; i < nclaimed; ++i) free_slot(p, claimed[i]);
    return (ptrdiff_t)frames;
}

/* Lazily allocate the four stats accumulator planes, once, the first time
 * stats is asked for. Returns 0 on success. The CAS makes the winner allocate
 * while any concurrent caller waits for the pointer to appear. */
static int ensure_stats(const atkdsp_fft *cp) {
    atkdsp_fft *p = (atkdsp_fft *)cp;
    if (p->stats) return 0;
    if (atk_try_claim(&p->stats_init)) {
        double *s = (double *)atk_aligned_malloc(
            (size_t)p->slots * 4 * p->n * sizeof(double));
        p->stats = s;            /* publish; NULL stays NULL on OOM */
        return s ? 0 : ATKDSP_E_NOMEM;
    }
    while (!p->stats) {          /* another thread is allocating */
        if (!p->stats_init && !p->stats) return ATKDSP_E_NOMEM;  /* it failed */
    }
    return 0;
}

/* per-frame accumulate into one slot's four planes: [0]=max [1]=min [2]=sumP
 * [3]=sumP^2, each n doubles, linear power |X|^2/n^2, fftshifted. */
static void accumulate_stats(const double *sc, size_t n, double *st, int first) {
    const size_t half = n / 2;
    const double inv2 = 1.0 / ((double)n * (double)n);
    double *mx = st, *mn = st + n, *su = st + 2 * n, *s2 = st + 3 * n;
    for (size_t i = 0; i < n; ++i) {
        const size_t k = (i + half) % n;
        const double re = sc[2 * k], im = sc[2 * k + 1];
        const double pw = (re * re + im * im) * inv2;
        if (first) { mx[i] = mn[i] = su[i] = pw; s2[i] = pw * pw; continue; }
        if (pw > mx[i]) mx[i] = pw;
        if (pw < mn[i]) mn[i] = pw;
        su[i] += pw;
        s2[i] += pw * pw;
    }
}

ptrdiff_t atkdsp_spectrum_stats(const atkdsp_fft *p, const atkdsp_cf32 *in,
                                size_t n_samples, size_t hop, const float *window,
                                float *max_db, float *avg_db, float *min_db,
                                float *sk) {
    if (!p || !in) return ATKDSP_E_ARG;
    if (!max_db && !avg_db && !min_db && !sk) return ATKDSP_E_ARG;
    const size_t n = p->n;
    if (hop == 0 || hop > n) hop = n;
    if (n_samples < n) return 0;
    if (ensure_stats(p) != 0) return ATKDSP_E_NOMEM;
    const long long frames = (long long)((n_samples - n) / hop + 1);

    int used[MAX_SLOTS];
    int claimed[MAX_SLOTS];
    int nclaimed = 0;
    memset(used, 0, sizeof used);
    int rc = ATKDSP_OK;
    long long f;

#ifdef _OPENMP
    int nthreads = atkdsp_get_threads();
    if (nthreads > p->slots) nthreads = p->slots;
    if (nthreads < 1) nthreads = 1;
#   pragma omp parallel num_threads(nthreads)
    {
        int s = claim_slot(p);
        double *sc = p->scratch + (size_t)s * 2 * n;
        double *st = p->stats + (size_t)s * 4 * n;
        int first = 1;
#       pragma omp for schedule(static) nowait
        for (f = 0; f < frames; ++f) {
            if (transform(p, in + (size_t)f * hop, window, 0, sc) != ATKDSP_OK) {
                rc = ATKDSP_E_FFT;
                continue;
            }
            accumulate_stats(sc, n, st, first);
            first = 0;
        }
#       pragma omp critical
        {
            claimed[nclaimed++] = s;
            if (!first) used[s] = 1;
        }
    }
#else
    {
        int s = claim_slot(p);
        double *sc = p->scratch + (size_t)s * 2 * n;
        double *st = p->stats + (size_t)s * 4 * n;
        int first = 1;
        for (f = 0; f < frames; ++f) {
            if (transform(p, in + (size_t)f * hop, window, 0, sc) != ATKDSP_OK) {
                rc = ATKDSP_E_FFT;
                continue;
            }
            accumulate_stats(sc, n, st, first);
            first = 0;
        }
        claimed[nclaimed++] = s;
        if (!first) used[s] = 1;
    }
#endif
    if (rc) { for (int i = 0; i < nclaimed; ++i) free_slot(p, claimed[i]); return rc; }

    /* merge the per-thread planes into the first used slot */
    int base = -1;
    for (int i = 0; i < nclaimed; ++i) {
        int s = claimed[i];
        if (!used[s]) continue;
        if (base < 0) { base = s; continue; }
        const double *st = p->stats + (size_t)s * 4 * n;
        double *bt = p->stats + (size_t)base * 4 * n;
        for (size_t k = 0; k < n; ++k) {
            if (st[k] > bt[k]) bt[k] = st[k];                 /* max */
            if (st[n + k] < bt[n + k]) bt[n + k] = st[n + k]; /* min */
            bt[2 * n + k] += st[2 * n + k];                   /* sumP  */
            bt[3 * n + k] += st[3 * n + k];                   /* sumP2 */
        }
    }
    if (base < 0) base = claimed[0];
    const double *bt = p->stats + (size_t)base * 4 * n;
    const double *mx = bt, *mn = bt + n, *su = bt + 2 * n, *s2 = bt + 3 * n;
    const double M = (double)frames;
    for (size_t i = 0; i < n; ++i) {
        if (max_db) max_db[i] = 10.0f * log10f((float)mx[i] + 1e-12f);
        if (min_db) min_db[i] = 10.0f * log10f((float)mn[i] + 1e-12f);
        if (avg_db) avg_db[i] = 10.0f * log10f((float)(su[i] / M) + 1e-12f);
        if (sk) {
            /* M * sum(P^2) / (sum P)^2 : 2 for exponential (Gaussian) noise,
             * 1 for a constant. One frame cannot show variation, so it is a
             * constant by definition. */
            const double denom = su[i] * su[i];
            sk[i] = (frames < 2 || denom <= 0.0)
                    ? 1.0f : (float)(M * s2[i] / denom);
        }
    }
    for (int i = 0; i < nclaimed; ++i) free_slot(p, claimed[i]);
    return (ptrdiff_t)frames;
}

int atkdsp_window_stats(const float *window, size_t n,
                        double *coherent_gain, double *enbw_bins) {
    if (!window || n == 0) return ATKDSP_E_ARG;
    double s = 0.0, s2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double w = (double)window[i];
        s += w; s2 += w * w;
    }
    if (coherent_gain) *coherent_gain = s / (double)n;
    if (enbw_bins) *enbw_bins = (s > 0.0) ? (double)n * s2 / (s * s) : 0.0;
    return ATKDSP_OK;
}
