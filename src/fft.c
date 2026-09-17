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
    int       slots;          /* per-thread scratch count */
    double   *scratch;        /* slots * 2n doubles */
    float    *acc;            /* slots * n floats: per-thread reduction lines */
};

atkdsp_fft *atkdsp_fft_create(size_t n) {
    if (n < 2) return NULL;
    atkdsp_fft *p = (atkdsp_fft *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->n = n;
    p->slots = 1;
#ifdef _OPENMP
    p->slots = omp_get_max_threads();
    if (p->slots < 1) p->slots = 1;
    if (p->slots > MAX_SLOTS) p->slots = MAX_SLOTS;
#endif
    p->plan = make_cfft_plan(n);
    p->scratch = (double *)atk_aligned_malloc((size_t)p->slots * 2 * n * sizeof(double));
    p->acc = (float *)atk_aligned_malloc((size_t)p->slots * n * sizeof(float));
    if (!p->plan || !p->scratch || !p->acc) { atkdsp_fft_destroy(p); return NULL; }
    return p;
}

void atkdsp_fft_destroy(atkdsp_fft *p) {
    if (!p) return;
    if (p->plan) destroy_cfft_plan(p->plan);
    atk_aligned_free(p->scratch);
    atk_aligned_free(p->acc);
    free(p);
}

size_t atkdsp_fft_length(const atkdsp_fft *p) { return p ? p->n : 0; }

static int slot_index(const atkdsp_fft *p) {
#ifdef _OPENMP
    int t = omp_get_thread_num();
    return t < p->slots ? t : 0;
#else
    (void)p;
    return 0;
#endif
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
    double *sc = p->scratch + (size_t)slot_index(p) * 2 * p->n;
    int rc = transform(p, in, NULL, inverse, sc);
    if (rc) return rc;
    for (size_t i = 0; i < p->n; ++i) {
        out[i].re = (float)sc[2 * i];
        out[i].im = (float)sc[2 * i + 1];
    }
    return ATKDSP_OK;
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
    double *sc = p->scratch + (size_t)slot_index(p) * 2 * p->n;
    int rc = transform(p, in, window, 0, sc);
    if (rc) return rc;
    mag_db_shifted(sc, p->n, out_db);
    return ATKDSP_OK;
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

    int used[MAX_SLOTS];
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
#   pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
    for (f = 0; f < frames; ++f) {
        const int s = slot_index(p);
        double *sc = p->scratch + (size_t)s * 2 * n;
        float  *acc = p->acc + (size_t)s * n;
        if (transform(p, in + (size_t)f * hop, window, 0, sc) != ATKDSP_OK) {
            rc = ATKDSP_E_FFT;
            continue;
        }
        accumulate(sc, n, detector, acc, !used[s]);
        used[s] = 1;
    }
    if (rc) return rc;

    /* merge the per-thread lines into slot 0, then to dB */
    int any = 0;
    for (int s = 0; s < p->slots; ++s) {
        if (!used[s]) continue;
        float *acc = p->acc + (size_t)s * n;
        if (!any) { if (s != 0) memcpy(p->acc, acc, n * sizeof(float)); any = 1; continue; }
        for (size_t i = 0; i < n; ++i) {
            switch (detector) {
            case ATKDSP_DET_MAX: if (acc[i] > p->acc[i]) p->acc[i] = acc[i]; break;
            case ATKDSP_DET_MIN: if (acc[i] < p->acc[i]) p->acc[i] = acc[i]; break;
            default:             p->acc[i] += acc[i]; break;
            }
        }
    }
    const float scale = detector == ATKDSP_DET_AVG ? 1.0f / (float)frames : 1.0f;
    for (size_t i = 0; i < n; ++i) {
        /* 10*log10(power) == 20*log10(mag); keep the +1e-12 floor on the magnitude
         * scale so a silent bin reads -240 dB exactly as spectrum_db does */
        const float mag = sqrtf(p->acc[i] * scale);
        out_line[i] = 20.0f * log10f(mag + 1e-12f);
    }
    return (ptrdiff_t)frames;
}
