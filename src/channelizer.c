/* THE CHANNELIZER (§28; docs/ATKDSP_PLAN.md §A4, docs/CHANNELIZER_DF_REQUIREMENTS.md).
 *
 * 0.23.1 — the PFB's process path rewritten for speed (Bill's three findings,
 * 2026-10-30, all confirmed in the code): threads over HOPS, not only
 * elements (one radio used one thread); a single-precision fold over
 * duplicated taps (the double-product-then-float mix did not vectorise); a
 * hop-major tile written out channel by channel (the stride-m store missed
 * the cache on nearly every value); the FFT in place on the fold's buffer.
 * Measured on a 2-core AVX2 box (bench/bench_pfb.c): one radio 2.3-2.8x
 * faster on one thread, 3.7-4.4x on two; e.g. 20 MS/s at ~100 kHz channels
 * from 1.1x to 4.7x real time. Identical at any thread count; ~1e-7 from
 * v0.23.0 and from the float64 twin.
 *
 * 28b. the polyphase filterbank (PFB) — a uniform 2x-oversampled grid, every
 *      channel at once, N element streams on one block clock (the DF path).
 * 28a. the overlap-save extractor (OSX) — arbitrary channels at arbitrary
 *      centres. (Added below the PFB.)
 * W1/W2 — the per-channel covariance and power DF accumulates on the output.
 *
 * The numpy twin (the spec) is python/atkdsp/reference.py §28; every result
 * here matches it to float precision. Deterministic double arithmetic for the
 * prototype so the taps reproduce exactly.
 */
#include "internal.h"
#include <stdio.h>
#ifdef _OPENMP
#  include <omp.h>
#endif

double atk_kaiser_beta(double atten_db);                 /* src/design.c */

/* modified Bessel I0 (power series); matches src/design.c and numpy.kaiser. */
static double chan_i0(double x) {
    double sum = 1.0, term = 1.0;
    const double y = x * 0.5;
    for (int k = 1; k < 500; ++k) {
        term *= (y / (double)k);
        const double t2 = term * term;
        sum += t2;
        if (t2 < sum * 1e-17) break;
    }
    return sum;
}

/* ===================================================================== 28b.
 * Polyphase filterbank.
 */
struct atkdsp_pfb {
    double       fs, atten;
    unsigned     nchan, os, K, max_nel;
    unsigned     hop, ov;              /* hop = nchan/os; ov = L - hop        */
    size_t       L;                   /* prototype length = nchan*K          */
    double      *h;                   /* prototype taps, L (double)          */
    atkdsp_fft  *fft;                 /* length-nchan forward plan           */
    atkdsp_cf32 *twiddle;             /* nchan: W_n^j = exp(2pi i j / nchan)  */
    /* per-element streaming state */
    atkdsp_cf32 *hist;                /* max_nel * ov, the overlap carried    */
    atkdsp_cf32 *leftover;            /* max_nel * hop, up to hop-1 used      */
    unsigned     lc;                  /* leftover count (same for all el)     */
    int64_t      b;                   /* global hop index (drives rotation)   */
    /* v0.24: the fold reads the taps in single precision, duplicated per
     * re/im (2L floats), so its inner loop is one flat float multiply-add
     * the compiler vectorises 8 wide; the rotation reads a double twiddle
     * table (2 nchan). */
    float       *hf;                  /* 2L: hf[2q] = hf[2q+1] = (float)h[q]  */
    double      *twd;                 /* 2 nchan: cos, sin of 2 pi j / nchan  */
    /* per-THREAD scratch (v0.24: hops run in parallel, not just elements):
     * slot t = [win: L cf32][fold: 2 nchan floats][fft: 2 nchan doubles]
     * [tile: PFB_TILE x nchan cf32], tslot_bytes apart; tslots grown
     * (serially) when the thread count grows. */
    unsigned char *tscratch;
    int          tslots;
    size_t       tslot_bytes;
    /* cal */
    atkdsp_cf32 *cal;                 /* max_nel * nchan, or NULL             */
    unsigned     cal_nel;
};

static void pfb_build_prototype(atkdsp_pfb *p) {
    const size_t L = p->L;
    const double fc = 1.0 / (2.0 * (double)p->nchan);
    const double M = (double)(L - 1) * 0.5;
    const double beta = atk_kaiser_beta(p->atten);
    const double i0b = chan_i0(beta);
    double sum = 0.0;
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t n = 0; n < L; ++n) {
            const double t = (double)n - M;
            const double x = 2.0 * fc * t;
            const double sinc = (t == 0.0) ? 1.0 : sin(ATK_PI * x) / (ATK_PI * x);
            const double r = (M > 0.0) ? t / M : 0.0;
            const double w = chan_i0(beta * sqrt(r * r < 1.0 ? 1.0 - r * r : 0.0)) / i0b;
            const double h = 2.0 * fc * sinc * w;
            if (pass == 0) sum += h;
            else p->h[n] = (h / sum) * (double)p->nchan;   /* unit DC gain, x nchan */
        }
        if (sum == 0.0) sum = 1.0;
    }
}

/* hops per output tile: each thread transforms PFB_TILE hops into a
 * hop-major tile, then writes the tile channel by channel — PFB_TILE
 * consecutive samples of each channel's output (256 bytes at 32) instead of
 * one 8-byte value per channel per hop at a stride of m (v0.23 and before:
 * nearly every store a cache miss). */
#define PFB_TILE 32
/* a task (one element's run of hops) is at least this many hops, so a short
 * block is not split into slivers that cost more to schedule than to run;
 * and there are up to PFB_TASKS_PER_THREAD tasks a thread, handed out
 * dynamically (0.24.0): on a hybrid CPU the efficiency cores are slower, and
 * with one equal share a thread everyone waited on them — Bill's 14-core /
 * 20-thread bench ran slower at 20 threads than at 16. */
#define PFB_MIN_TASK 8
#define PFB_TASKS_PER_THREAD 4

static size_t pfb_align64(size_t b) { return (b + 63) & ~(size_t)63; }

static size_t pfb_slot_bytes(const atkdsp_pfb *p) {
    return pfb_align64(p->L * sizeof(atkdsp_cf32)) + pfb_align64(2 * (size_t)p->nchan * sizeof(float))
         + pfb_align64(2 * (size_t)p->nchan * sizeof(double))
         + pfb_align64((size_t)PFB_TILE * p->nchan * sizeof(atkdsp_cf32));
}

/* Make room for `nt` threads' scratch. Called serially (create, and the top
 * of process_n before any parallel region) — so it only allocates when the
 * thread count has GROWN since the last call, never per block. */
static int pfb_ensure_slots(atkdsp_pfb *p, int nt) {
    if (nt < 1) nt = 1;
    if (nt <= p->tslots && p->tscratch) return ATKDSP_OK;
    unsigned char *t = (unsigned char *)atk_aligned_malloc((size_t)nt * p->tslot_bytes);
    if (!t) return ATKDSP_E_NOMEM;
    atk_aligned_free(p->tscratch);
    p->tscratch = t;
    p->tslots = nt;
    return ATKDSP_OK;
}

atkdsp_pfb *atkdsp_pfb_create(double fs, unsigned nchan, unsigned oversample,
                              unsigned taps_per_poly, double atten_db, unsigned max_nel) {
    if (!(fs > 0.0) || nchan < 2 || oversample < 1 || nchan % oversample != 0) return NULL;
    if (taps_per_poly < 2 || max_nel < 1) return NULL;
    if (atten_db < 20.0) atten_db = 20.0;
    atkdsp_pfb *p = (atkdsp_pfb *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->fs = fs; p->atten = atten_db;
    p->nchan = nchan; p->os = oversample; p->K = taps_per_poly; p->max_nel = max_nel;
    p->hop = nchan / oversample;
    p->L = (size_t)nchan * taps_per_poly;
    p->ov = (unsigned)(p->L - p->hop);
    p->h = (double *)malloc(p->L * sizeof(double));
    p->fft = atkdsp_fft_create(nchan);
    p->twiddle = (atkdsp_cf32 *)malloc(nchan * sizeof(atkdsp_cf32));
    p->hist = (atkdsp_cf32 *)calloc((size_t)max_nel * p->ov, sizeof(atkdsp_cf32));
    p->leftover = (atkdsp_cf32 *)calloc((size_t)max_nel * p->hop, sizeof(atkdsp_cf32));
    p->hf = (float *)atk_aligned_malloc(2 * p->L * sizeof(float));
    p->twd = (double *)atk_aligned_malloc(2 * (size_t)nchan * sizeof(double));
    p->tslot_bytes = pfb_slot_bytes(p);
    if (!p->h || !p->fft || !p->twiddle || !p->hist || !p->leftover || !p->hf || !p->twd ||
        pfb_ensure_slots(p, atkdsp_get_threads()) != ATKDSP_OK) { atkdsp_pfb_destroy(p); return NULL; }
    pfb_build_prototype(p);
    for (size_t q = 0; q < p->L; ++q) p->hf[2 * q] = p->hf[2 * q + 1] = (float)p->h[q];
    for (unsigned j = 0; j < nchan; ++j) {
        const double a = 2.0 * ATK_PI * (double)j / (double)nchan;
        p->twiddle[j].re = (float)cos(a);
        p->twiddle[j].im = (float)sin(a);
        p->twd[2 * j] = cos(a);
        p->twd[2 * j + 1] = sin(a);
    }
    p->lc = 0; p->b = 0; p->cal = NULL; p->cal_nel = 0;
    return p;
}

void atkdsp_pfb_destroy(atkdsp_pfb *p) {
    if (!p) return;
    free(p->h); atkdsp_fft_destroy(p->fft); free(p->twiddle);
    free(p->hist); free(p->leftover);
    atk_aligned_free(p->hf); atk_aligned_free(p->twd); atk_aligned_free(p->tscratch);
    free(p->cal);
    free(p);
}

void atkdsp_pfb_reset(atkdsp_pfb *p) {
    if (!p) return;
    memset(p->hist, 0, (size_t)p->max_nel * p->ov * sizeof(atkdsp_cf32));
    memset(p->leftover, 0, (size_t)p->max_nel * p->hop * sizeof(atkdsp_cf32));
    p->lc = 0; p->b = 0;
}

unsigned atkdsp_pfb_nchan(const atkdsp_pfb *p) { return p ? p->nchan : 0; }
double   atkdsp_pfb_out_rate(const atkdsp_pfb *p) {
    return p ? (double)p->os * p->fs / (double)p->nchan : 0.0;
}
size_t   atkdsp_pfb_out_hops(const atkdsp_pfb *p, size_t n) {
    if (!p) return 0;
    const size_t avail = (size_t)p->lc + n;
    return avail / p->hop;
}
double   atkdsp_pfb_group_delay(const atkdsp_pfb *p) { return p ? (double)(p->L - 1) * 0.5 : 0.0; }
int64_t  atkdsp_pfb_block_origin(const atkdsp_pfb *p, int64_t hop_index) {
    return p ? hop_index * (int64_t)p->hop - (int64_t)p->ov : 0;
}

int atkdsp_pfb_set_cal(atkdsp_pfb *p, const atkdsp_cf32 *table, unsigned nel) {
    if (!p) return ATKDSP_E_ARG;
    if (!table) { free(p->cal); p->cal = NULL; p->cal_nel = 0; return ATKDSP_OK; }
    if (nel < 1 || nel > p->max_nel) return ATKDSP_E_ARG;
    atkdsp_cf32 *t = (atkdsp_cf32 *)malloc((size_t)nel * p->nchan * sizeof(atkdsp_cf32));
    if (!t) return ATKDSP_E_NOMEM;
    memcpy(t, table, (size_t)nel * p->nchan * sizeof(atkdsp_cf32));
    free(p->cal); p->cal = t; p->cal_nel = nel;
    return ATKDSP_OK;
}

/* One element's hops [i0, i1) into dstbase (nchan x m, channel-major), with
 * thread slot `slot`. Every hop depends only on its own window, so how the
 * hops are split among threads cannot change a single output value: the
 * result is identical at any thread count and any block split. */
static int pfb_run_hops(const atkdsp_pfb *p, unsigned e, const atkdsp_cf32 *xin, size_t i0, size_t i1,
                        size_t m, int64_t b0, unsigned char *slot, atkdsp_cf32 *dstbase) {
    const unsigned N = p->nchan, hop = p->hop;
    const size_t L = p->L, ov = p->ov, K = p->K, N2 = 2 * (size_t)N;
    atkdsp_cf32 *win = (atkdsp_cf32 *)slot;
    unsigned char *at = slot + pfb_align64(L * sizeof(atkdsp_cf32));
    float *fold = (float *)at;
    at += pfb_align64(N2 * sizeof(float));
    double *acc = (double *)at;
    at += pfb_align64(N2 * sizeof(double));
    atkdsp_cf32 *tile = (atkdsp_cf32 *)at;
    const atkdsp_cf32 *hist = p->hist + (size_t)e * ov;
    const atkdsp_cf32 *lo = p->leftover + (size_t)e * hop;
    const atkdsp_cf32 *cal = (p->cal && e < p->cal_nel) ? p->cal + (size_t)e * N : NULL;
    const size_t head = ov + p->lc;                     /* s index where `in` begins */
    const float *hf = p->hf;
    const double *twd = p->twd;
    for (size_t t0 = i0; t0 < i1; t0 += PFB_TILE) {
        const size_t t1 = (t0 + PFB_TILE < i1) ? t0 + PFB_TILE : i1;
        for (size_t i = t0; i < t1; ++i) {
            /* the window: s[i*hop, i*hop + L), s = hist ++ leftover ++ in */
            const size_t st = i * hop;
            const atkdsp_cf32 *wp;
            if (st >= head) {
                wp = xin + (st - head);
            } else {
                for (size_t q = 0; q < L; ++q) {
                    const size_t pq = st + q;
                    win[q] = (pq < ov) ? hist[pq] : (pq < head) ? lo[pq - ov] : xin[pq - head];
                }
                wp = win;
            }
            /* the fold, in single precision: fold[q] = sum_k h[kN + q/2] *
             * w[kN + q/2] (re and im interleaved, taps duplicated), k
             * ascending — one flat float multiply-add over 2N values per tap,
             * which vectorises 8 wide. Until v0.24 each product was formed in
             * double and rounded to float before the float sum: the same
             * accuracy class (float accumulation either way), but a mix the
             * compiler would not vectorise. Measured against v0.23's outputs
             * and the float64 twin in tests/test_channelizer.py. */
            {
                const float *ATK_RESTRICT w = (const float *)wp;
                float *ATK_RESTRICT f = fold;
                for (size_t q = 0; q < N2; ++q) f[q] = hf[q] * w[q];
                for (size_t k = 1; k < K; ++k) {
                    const float *ATK_RESTRICT hk = hf + k * N2;
                    const float *ATK_RESTRICT wk = w + k * N2;
                    for (size_t q = 0; q < N2; ++q) f[q] += hk[q] * wk[q];
                }
                for (size_t q = 0; q < N2; ++q) acc[q] = (double)f[q];
            }
            if (atk_fft_forward_inplace(p->fft, acc) != ATKDSP_OK) return ATKDSP_E_FFT;
            /* reference the bins to input sample 0: the block starts at
             * (b+1)*hop - L and L = K*N, so remove exp(+2 pi i k ((b+1) hop mod N)/N)
             * — multiply by exp(-2 pi i k shift/N), twiddle index (-k*shift) mod N,
             * stepped (no per-bin multiply and modulo). Correct at every
             * oversampling (v0.19-0.20 used +b*hop: right only at 2x). */
            const int64_t b = b0 + (int64_t)i;
            const unsigned shift = (unsigned)((((b + 1) * (int64_t)hop) % (int64_t)N + N) % N);
            atkdsp_cf32 *row = tile + (i - t0) * N;
            unsigned idx = 0;
            for (unsigned k = 0; k < N; ++k) {
                const double xr = acc[2 * k], xi = acc[2 * k + 1];
                const double wr = twd[2 * idx], wi = twd[2 * idx + 1];
                double yr = xr * wr - xi * wi;
                double yi = xr * wi + xi * wr;
                if (cal) {
                    const double cr = cal[k].re, ci = cal[k].im;
                    const double zr = yr * cr - yi * ci, zi = yr * ci + yi * cr;
                    yr = zr; yi = zi;
                }
                row[k].re = (float)yr;
                row[k].im = (float)yi;
                idx = (idx >= shift) ? idx - shift : idx + N - shift;
            }
        }
        /* the tile, channel by channel: (t1 - t0) consecutive outputs each */
        const size_t nt = t1 - t0;
        for (unsigned k = 0; k < N; ++k) {
            atkdsp_cf32 *d = dstbase + (size_t)k * m + t0;
            const atkdsp_cf32 *src = tile + k;
            for (size_t r = 0; r < nt; ++r) d[r] = src[r * N];
        }
    }
    return ATKDSP_OK;
}

ptrdiff_t atkdsp_pfb_process_n(atkdsp_pfb *p, const atkdsp_cf32 *const *ins, unsigned nel,
                               size_t n, atkdsp_cf32 *out, size_t cap) {
    if (!p || !ins || nel < 1 || nel > p->max_nel) return ATKDSP_E_ARG;
    const unsigned N = p->nchan, hop = p->hop;
    const size_t avail = (size_t)p->lc + n;
    const size_t m = avail / hop;
    if (m == 0) {
        /* stash all into leftover */
        for (unsigned e = 0; e < nel; ++e)
            if (n) memcpy(p->leftover + (size_t)e * hop + p->lc, ins[e], n * sizeof(atkdsp_cf32));
        p->lc = (unsigned)avail;
        return 0;
    }
    if (!out || cap < (size_t)nel * N * m) return ATKDSP_E_CAP;
    const size_t consumed = m * hop;
    const size_t rem = avail - consumed;
    const int64_t b0 = p->b;
    const size_t ov = p->ov, L = p->L;
    /* Element e's stream is s = hist(ov) ++ leftover(lc) ++ in(n), and hop i's
     * window is s[i*hop, i*hop + L). Windows wholly inside `in` are folded in
     * place; the first few (reaching back into hist/leftover) are assembled
     * in the thread's scratch.
     *
     * v0.24 — THE PARALLEL SPLIT IS OVER HOPS, not only elements. Through
     * v0.23 the one parallel loop ran over elements, so a single radio
     * (nel = 1) ran on one thread however many the machine had (Bill,
     * 2026-10-30: "With one radio its work runs on 1 of your 20 threads").
     * Now each element's hops are cut into runs and every (element, run)
     * pair is a task. Hops are independent, so the output is the same at
     * any thread count (tests/test_channelizer.py holds that). */
    int nth = atkdsp_get_threads();
    if (nth < 1) nth = 1;
    if (pfb_ensure_slots(p, nth) != ATKDSP_OK) return ATKDSP_E_NOMEM;
    size_t runs = (m + PFB_MIN_TASK - 1) / PFB_MIN_TASK;           /* per element */
    size_t want = ((size_t)nth * PFB_TASKS_PER_THREAD + nel - 1) / nel;
    if (runs > want) runs = want;
    if (runs < 1) runs = 1;
    const int ntask = (int)(runs * nel);
    int rc = ATKDSP_OK;
    /* MSVC's OpenMP 2.0: a signed loop variable declared outside the pragma,
     * and no atomic write — a failing task sets its own flag (fail[] is one
     * int per task, so no two threads write the same int). */
    int ti;
    int fail_stack[256];
    int *fail = (ntask <= 256) ? fail_stack : (int *)malloc((size_t)ntask * sizeof(int));
    if (!fail) return ATKDSP_E_NOMEM;
    memset(fail, 0, (size_t)ntask * sizeof(int));
#ifdef _OPENMP
    if (nth > ntask) nth = ntask;
#   pragma omp parallel num_threads(nth)
    {
        unsigned char *slot = p->tscratch + (size_t)omp_get_thread_num() * p->tslot_bytes;
#       pragma omp for schedule(dynamic, 1)
        for (ti = 0; ti < ntask; ++ti) {
            const unsigned e = (unsigned)((size_t)ti / runs);
            const size_t r = (size_t)ti % runs;
            const size_t i0 = r * m / runs, i1 = (r + 1) * m / runs;
            if (pfb_run_hops(p, e, ins[e], i0, i1, m, b0, slot, out + (size_t)e * N * m) != ATKDSP_OK)
                fail[ti] = 1;
        }
    }
#else
    for (ti = 0; ti < ntask; ++ti) {
        const unsigned e = (unsigned)((size_t)ti / runs);
        const size_t r = (size_t)ti % runs;
        const size_t i0 = r * m / runs, i1 = (r + 1) * m / runs;
        if (pfb_run_hops(p, e, ins[e], i0, i1, m, b0, p->tscratch, out + (size_t)e * N * m) != ATKDSP_OK)
            fail[ti] = 1;
    }
#endif
    for (ti = 0; ti < ntask; ++ti) if (fail[ti]) rc = ATKDSP_E_FFT;
    if (fail != fail_stack) free(fail);
    if (rc != ATKDSP_OK) return rc;
    /* save each element's state, after every hop has read the old one: new
     * hist = s[m*hop, m*hop + ov) — assembled in scratch first, since it can
     * read the old hist and leftover — then the new leftover =
     * s[m*hop + ov, head + n), which (m >= 1, lc < hop) always lies inside
     * `in`. */
    const unsigned lc = p->lc;
    const size_t head = ov + lc;
    atkdsp_cf32 *tmp = (atkdsp_cf32 *)p->tscratch;                 /* slot 0's window, L >= ov */
    for (unsigned e = 0; e < nel; ++e) {
        atkdsp_cf32 *hist = p->hist + (size_t)e * ov;
        const atkdsp_cf32 *lo = p->leftover + (size_t)e * hop;
        const atkdsp_cf32 *xin = ins[e];
        for (size_t q = 0; q < ov; ++q) {
            const size_t pq = consumed + q;
            tmp[q] = (pq < ov) ? hist[pq] : (pq < head) ? lo[pq - ov] : xin[pq - head];
        }
        for (size_t s2 = 0; s2 < rem; ++s2)
            p->leftover[(size_t)e * hop + s2] = xin[consumed + ov + s2 - head];
        memcpy(hist, tmp, ov * sizeof(atkdsp_cf32));
    }
    (void)L;
    p->lc = (unsigned)rem;
    p->b = b0 + (int64_t)m;
    return (ptrdiff_t)m;
}

ptrdiff_t atkdsp_pfb_process(atkdsp_pfb *p, const atkdsp_cf32 *in, size_t n,
                             atkdsp_cf32 *out, size_t cap) {
    const atkdsp_cf32 *ins[1] = { in };
    return atkdsp_pfb_process_n(p, ins, 1, n, out, cap);
}

int atkdsp_pfb_describe(const atkdsp_pfb *p, char *buf, size_t cap) {
    if (!p || !buf || cap == 0) return ATKDSP_E_ARG;
    return snprintf(buf, cap,
        "PFB %.6g MSPS: %u chan x%u oversample, %.6g kHz spacing, %.6g kHz out, "
        "%zu-tap proto (%u/poly), %.0f dB, <=%u el",
        p->fs / 1e6, p->nchan, p->os, p->fs / p->nchan / 1e3,
        atkdsp_pfb_out_rate(p) / 1e3, p->L, p->K, p->atten, p->max_nel);
}

/* ---- W1: per-channel covariance ------------------------------------------ */
void atkdsp_pfb_cov_accumulate(const atkdsp_cf32 *out, unsigned nel, unsigned nchan,
                               size_t m, const uint8_t *gate, atkdsp_cf32 *R, uint32_t *counts) {
    if (!out || !R || !counts || nel < 1 || nchan < 1) return;
    for (unsigned k = 0; k < nchan; ++k) {
        if (gate && !gate[k]) continue;
        atkdsp_cf32 *Rk = R + (size_t)k * nel * nel;
        for (size_t i = 0; i < m; ++i) {
            /* gather y[el] = out[(el*nchan + k)*m + i] */
            for (unsigned a = 0; a < nel; ++a) {
                const atkdsp_cf32 ya = out[((size_t)a * nchan + k) * m + i];
                for (unsigned bb = 0; bb < nel; ++bb) {
                    const atkdsp_cf32 yb = out[((size_t)bb * nchan + k) * m + i];
                    /* R[a][b] += ya * conj(yb) */
                    Rk[(size_t)a * nel + bb].re += ya.re * yb.re + ya.im * yb.im;
                    Rk[(size_t)a * nel + bb].im += ya.im * yb.re - ya.re * yb.im;
                }
            }
        }
        counts[k] += (uint32_t)m;
    }
}

void atkdsp_pfb_cov_reset(atkdsp_cf32 *R, uint32_t *counts, unsigned nel, unsigned nchan,
                          const uint8_t *gate) {
    if (!R || !counts) return;
    for (unsigned k = 0; k < nchan; ++k) {
        if (gate && !gate[k]) continue;
        memset(R + (size_t)k * nel * nel, 0, (size_t)nel * nel * sizeof(atkdsp_cf32));
        counts[k] = 0;
    }
}

/* ---- W2: per-channel power ------------------------------------------------ */
void atkdsp_pfb_channel_power(const atkdsp_cf32 *out, unsigned nel, unsigned nchan,
                              size_t m, float *pwr, float *mean_pwr) {
    if (!out || nel < 1 || nchan < 1) return;
    const double inv = m ? 1.0 / (double)m : 0.0;
    for (unsigned k = 0; k < nchan; ++k) {
        double msum = 0.0;
        for (unsigned a = 0; a < nel; ++a) {
            double s = 0.0;
            for (size_t i = 0; i < m; ++i) {
                const atkdsp_cf32 y = out[((size_t)a * nchan + k) * m + i];
                s += (double)y.re * y.re + (double)y.im * y.im;
            }
            const float pe = (float)(s * inv);
            if (pwr) pwr[(size_t)a * nchan + k] = pe;
            msum += s * inv;
        }
        if (mean_pwr) mean_pwr[k] = (float)(msum / (double)nel);
    }
}

/* ===================================================================== 28a.
 * Overlap-save extractor. One forward FFT of size nfft per block (overlap
 * ov = nfft/2); per channel: roll to baseband, x the channel filter's freq
 * response, FOLD to M = nfft/decim and inverse-FFT (the small ifft IS the
 * integer decimation), discard the ov/decim transient, then an arbitrary
 * resampler lands it on out_rate. Channels share the one FFT; each owns its
 * filter, its M-point inverse plan and its per-element resamplers, so
 * add/retune/remove never disturbs the others. Twin: reference.py §28a.
 */
double atk_kaiser_beta(double atten_db);                      /* src/design.c */

typedef struct {
    int          active;
    double       offset, bw, out_rate, chan_rate;
    unsigned     decim, shift;        /* shift = s mod nfft, s = floor(offset/fs*nfft + 0.5) */
    double       delta;               /* offset - s*fs/nfft: removed after the fold (Hz) */
    size_t       M, ntaps, ovd;       /* M = nfft/decim; ovd = ov/decim         */
    atkdsp_cf32 *H;                   /* nfft, channel filter freq response      */
    atkdsp_fft  *ifft;                /* M-point inverse plan                    */
    atkdsp_arb_resampler **rs;        /* max_nel; NULL entry = no resample       */
} osx_chan;

struct atkdsp_osx {
    double       fs, atten;
    size_t       nfft, ov, hop;
    unsigned     max_nel;
    int64_t      blk;                 /* blocks so far (block i starts at i*hop - ov) */
    atkdsp_fft  *fft;                 /* nfft forward plan                       */
    atkdsp_cf32 *hist;                /* max_nel * ov                            */
    atkdsp_cf32 *leftover;            /* max_nel * hop                           */
    unsigned     lc;
    osx_chan    *chans;
    int          nslots, cap_slots;
    /* scratch */
    atkdsp_cf32 *Xall;                /* max_nel * nfft (per-element forward FFT) */
    atkdsp_cf32 *Yf;                  /* nfft (M used)                           */
    atkdsp_cf32 *ys;                  /* nfft (M used)                           */
    atkdsp_cf32 *valid;              /* nfft (pre-resample valid region)        */
};

atkdsp_osx *atkdsp_osx_create(double fs, size_t nfft, double atten_db, unsigned max_nel) {
    if (!(fs > 0.0) || nfft < 64 || (nfft & (nfft - 1)) || max_nel < 1) return NULL;
    if (atten_db < 20.0) atten_db = 20.0;
    atkdsp_osx *x = (atkdsp_osx *)calloc(1, sizeof *x);
    if (!x) return NULL;
    x->fs = fs; x->atten = atten_db; x->nfft = nfft; x->max_nel = max_nel;
    x->ov = nfft / 2; x->hop = nfft - x->ov;
    x->fft = atkdsp_fft_create(nfft);
    x->hist = (atkdsp_cf32 *)calloc((size_t)max_nel * x->ov, sizeof(atkdsp_cf32));
    x->leftover = (atkdsp_cf32 *)calloc((size_t)max_nel * x->hop, sizeof(atkdsp_cf32));
    x->Xall = (atkdsp_cf32 *)atk_aligned_malloc((size_t)max_nel * nfft * sizeof(atkdsp_cf32));
    x->Yf = (atkdsp_cf32 *)atk_aligned_malloc(nfft * sizeof(atkdsp_cf32));
    x->ys = (atkdsp_cf32 *)atk_aligned_malloc(nfft * sizeof(atkdsp_cf32));
    x->valid = (atkdsp_cf32 *)atk_aligned_malloc(nfft * sizeof(atkdsp_cf32));
    if (!x->fft || !x->hist || !x->leftover || !x->Xall || !x->Yf || !x->ys || !x->valid) {
        atkdsp_osx_destroy(x); return NULL;
    }
    x->lc = 0; x->blk = 0; x->nslots = 0; x->cap_slots = 0; x->chans = NULL;
    return x;
}

static void osx_free_chan(osx_chan *c, unsigned max_nel) {
    if (!c->active) return;
    atk_aligned_free(c->H); atkdsp_fft_destroy(c->ifft);
    if (c->rs) { for (unsigned e = 0; e < max_nel; ++e) atkdsp_arb_resampler_destroy(c->rs[e]); free(c->rs); }
    c->H = NULL; c->ifft = NULL; c->rs = NULL; c->active = 0;
}

void atkdsp_osx_destroy(atkdsp_osx *x) {
    if (!x) return;
    for (int i = 0; i < x->nslots; ++i) osx_free_chan(&x->chans[i], x->max_nel);
    free(x->chans);
    atkdsp_fft_destroy(x->fft); free(x->hist); free(x->leftover);
    atk_aligned_free(x->Xall); atk_aligned_free(x->Yf); atk_aligned_free(x->ys); atk_aligned_free(x->valid);
    free(x);
}

void atkdsp_osx_reset(atkdsp_osx *x) {
    if (!x) return;
    memset(x->hist, 0, (size_t)x->max_nel * x->ov * sizeof(atkdsp_cf32));
    memset(x->leftover, 0, (size_t)x->max_nel * x->hop * sizeof(atkdsp_cf32));
    x->lc = 0; x->blk = 0;
    for (int i = 0; i < x->nslots; ++i)
        if (x->chans[i].active && x->chans[i].rs)
            for (unsigned e = 0; e < x->max_nel; ++e)
                if (x->chans[i].rs[e]) atkdsp_arb_resampler_reset(x->chans[i].rs[e]);
}

static unsigned osx_pick_decim(const atkdsp_osx *x, double bw, double out_rate) {
    const double target = (out_rate > 1.25 * bw) ? out_rate : 1.25 * bw;
    unsigned decim = 1;
    for (;;) {
        unsigned nd = decim * 2;
        if (x->fs / (double)nd < target) break;
        if (x->ov % nd != 0 || x->nfft % nd != 0) break;
        decim = nd;
    }
    return decim;
}

/* the integer bin the roll moves by, and what is left over (Hz) */
static void osx_tune(const atkdsp_osx *x, osx_chan *c, double offset_hz) {
    const long s = (long)floor(offset_hz / x->fs * (double)x->nfft + 0.5);
    long sh = s % (long)x->nfft; if (sh < 0) sh += (long)x->nfft;
    c->offset = offset_hz;
    c->shift = (unsigned)sh;
    c->delta = offset_hz - (double)s * x->fs / (double)x->nfft;
}

int atkdsp_osx_add(atkdsp_osx *x, double offset_hz, double bw_hz, double out_rate) {
    if (!x || !(bw_hz > 0.0) || !(out_rate > 0.0)) return ATKDSP_E_ARG;
    const unsigned decim = osx_pick_decim(x, bw_hz, out_rate);
    const double chan_rate = x->fs / (double)decim;
    /* channel prototype: lowpass fp=bw/2, stop=chan_rate-bw/2, capped to ov+1 */
    double fp = 0.5 * bw_hz;
    double fstop = chan_rate - 0.5 * bw_hz;
    if (fstop > 0.5 * x->fs) fstop = 0.5 * x->fs;
    if (fstop <= fp) return ATKDSP_E_ARG;
    ptrdiff_t nt = atkdsp_design_lowpass(fp, fstop, x->atten, x->fs, NULL, 0);
    if (nt <= 0) return ATKDSP_E_ARG;
    float *taps = (float *)malloc((size_t)nt * sizeof(float));
    if (!taps) return ATKDSP_E_NOMEM;
    atkdsp_design_lowpass(fp, fstop, x->atten, x->fs, taps, (size_t)nt);
    size_t ntaps = (size_t)nt;
    /* cap to ov+1 and renormalise (DC gain 1) if the filter is longer */
    if (ntaps > x->ov + 1) {
        ntaps = x->ov + 1;
        double s = 0.0; for (size_t i = 0; i < ntaps; ++i) s += taps[i];
        if (s == 0.0) s = 1.0;
        for (size_t i = 0; i < ntaps; ++i) taps[i] = (float)(taps[i] / s);
    }
    /* find or grow a slot */
    int slot = -1;
    for (int i = 0; i < x->nslots; ++i) if (!x->chans[i].active) { slot = i; break; }
    if (slot < 0) {
        if (x->nslots >= x->cap_slots) {
            int ncap = x->cap_slots ? x->cap_slots * 2 : 8;
            osx_chan *nc = (osx_chan *)realloc(x->chans, (size_t)ncap * sizeof(osx_chan));
            if (!nc) { free(taps); return ATKDSP_E_NOMEM; }
            memset(nc + x->cap_slots, 0, (size_t)(ncap - x->cap_slots) * sizeof(osx_chan));
            x->chans = nc; x->cap_slots = ncap;
        }
        slot = x->nslots++;
    }
    osx_chan *c = &x->chans[slot];
    memset(c, 0, sizeof *c);
    c->offset = offset_hz; c->bw = bw_hz; c->out_rate = out_rate;
    c->decim = decim; c->chan_rate = chan_rate; c->ntaps = ntaps;
    c->M = x->nfft / decim; c->ovd = x->ov / decim;
    osx_tune(x, c, offset_hz);
    /* H = FFT(taps zero-padded to nfft) */
    c->H = (atkdsp_cf32 *)atk_aligned_malloc(x->nfft * sizeof(atkdsp_cf32));
    atkdsp_cf32 *hin = (atkdsp_cf32 *)calloc(x->nfft, sizeof(atkdsp_cf32));
    if (!c->H || !hin) { free(taps); free(hin); osx_free_chan(c, x->max_nel); return ATKDSP_E_NOMEM; }
    for (size_t i = 0; i < ntaps; ++i) { hin[i].re = taps[i]; hin[i].im = 0.0f; }
    atkdsp_fft_exec(x->fft, hin, c->H, 0);
    free(hin); free(taps);
    c->ifft = atkdsp_fft_create(c->M);
    c->rs = (atkdsp_arb_resampler **)calloc(x->max_nel, sizeof(atkdsp_arb_resampler *));
    if (!c->ifft || !c->rs) { osx_free_chan(c, x->max_nel); return ATKDSP_E_NOMEM; }
    if (fabs(chan_rate - out_rate) > 1e-6 * out_rate) {
        for (unsigned e = 0; e < x->max_nel; ++e) {
            c->rs[e] = atkdsp_arb_resampler_create(chan_rate, out_rate, x->atten, 64);
            if (!c->rs[e]) { osx_free_chan(c, x->max_nel); return ATKDSP_E_NOMEM; }
        }
    }
    c->active = 1;
    return slot;
}

int atkdsp_osx_retune(atkdsp_osx *x, int ch, double offset_hz) {
    if (!x || ch < 0 || ch >= x->nslots || !x->chans[ch].active) return ATKDSP_E_ARG;
    osx_tune(x, &x->chans[ch], offset_hz);
    return ATKDSP_OK;
}

int atkdsp_osx_remove(atkdsp_osx *x, int ch) {
    if (!x || ch < 0 || ch >= x->nslots || !x->chans[ch].active) return ATKDSP_E_ARG;
    osx_free_chan(&x->chans[ch], x->max_nel);
    return ATKDSP_OK;
}

int    atkdsp_osx_channels(const atkdsp_osx *x) { return x ? x->nslots : 0; }
double atkdsp_osx_chan_rate(const atkdsp_osx *x, int ch) {
    if (!x || ch < 0 || ch >= x->nslots || !x->chans[ch].active) return 0.0;
    return x->chans[ch].out_rate;
}
size_t atkdsp_osx_out_max(const atkdsp_osx *x, int ch, size_t n) {
    if (!x || ch < 0 || ch >= x->nslots || !x->chans[ch].active) return 0;
    return (size_t)((double)n * x->chans[ch].out_rate / x->fs) + 64;
}

/* process nel elements (ins[e], each n samples) on one block clock. outs is
 * [nel][nslots]; out_counts is [nslots]. */
ptrdiff_t atkdsp_osx_process_n(atkdsp_osx *x, const atkdsp_cf32 *const *ins, unsigned nel,
                               size_t n, atkdsp_cf32 *const *outs, size_t *out_counts,
                               size_t cap_each) {
    if (!x || !ins || nel < 1 || nel > x->max_nel || !out_counts) return ATKDSP_E_ARG;
    const size_t nfft = x->nfft, ov = x->ov, hop = x->hop;
    const int ns = x->nslots;
    for (int ci = 0; ci < ns; ++ci) out_counts[ci] = 0;
    const size_t avail = (size_t)x->lc + n;
    const size_t m = avail / hop;
    if (m == 0) {
        for (unsigned e = 0; e < nel; ++e)
            if (n) memcpy(x->leftover + (size_t)e * hop + x->lc, ins[e], n * sizeof(atkdsp_cf32));
        x->lc = (unsigned)avail;
        return (ptrdiff_t)ns;
    }
    const size_t consumed = m * hop, rem = avail - consumed;
    const unsigned lc = x->lc;
    for (size_t i = 0; i < m; ++i) {
        /* forward FFT per element for this hop; slide each hist */
        for (unsigned e = 0; e < nel; ++e) {
            atkdsp_cf32 *hist = x->hist + (size_t)e * ov;
            atkdsp_cf32 *Xe = x->Xall + (size_t)e * nfft;    /* this element's forward FFT */
            /* build the block (hist(ov) ++ hop samples from leftover ++ in) in ys,
             * forward-FFT it into Xall[e], then slide hist */
            memcpy(x->ys, hist, ov * sizeof(atkdsp_cf32));
            const atkdsp_cf32 *lo = x->leftover + (size_t)e * hop;
            for (size_t s = 0; s < hop; ++s) {
                const size_t j = i * hop + s;
                x->ys[ov + s] = (j < lc) ? lo[j] : ins[e][j - lc];
            }
            atkdsp_fft_exec(x->fft, x->ys, Xe, 0);            /* X_e into Xall[e] */
            memcpy(hist, x->ys + hop, ov * sizeof(atkdsp_cf32));   /* slide hist = block[hop:] */
        }
        /* each channel, each element */
        for (int ci = 0; ci < ns; ++ci) {
            osx_chan *c = &x->chans[ci];
            if (!c->active) continue;
            const size_t M = c->M, ovd = c->ovd, nvalid = M - ovd;
            const unsigned sh = c->shift, decim = c->decim;
            const double invd = 1.0 / (double)decim;
            size_t emit = 0;
            for (unsigned e = 0; e < nel; ++e) {
                const atkdsp_cf32 *X = x->Xall + (size_t)e * nfft;
                /* fold: Yf[k] = sum_j X[(k+j*M+sh) mod nfft] * H[k+j*M] */
                for (size_t k = 0; k < M; ++k) {
                    float ar = 0.0f, ai = 0.0f;
                    for (unsigned j = 0; j < decim; ++j) {
                        const size_t bin = k + (size_t)j * M;
                        const size_t xi = (bin + sh) & (nfft - 1);   /* nfft power of 2 */
                        const atkdsp_cf32 xv = X[xi], hv = c->H[bin];
                        ar += xv.re * hv.re - xv.im * hv.im;
                        ai += xv.re * hv.im + xv.im * hv.re;
                    }
                    x->Yf[k].re = ar; x->Yf[k].im = ai;
                }
                atkdsp_fft_exec(c->ifft, x->Yf, x->ys, 1);    /* inverse, 1/M applied */
                /* The roll mixed by s bins relative to THIS block's start; take
                 * it to absolute time, exp(-2 pi i (s*start mod nfft)/nfft), and
                 * remove the residual delta at each sample's absolute input
                 * index n = blk*hop + k*decim: one phase per sample in absolute
                 * time, continuous across blocks, calls and chunkings. (v0.20.0
                 * had neither: an odd s flipped the sign every block and the
                 * channel sat up to half a bin off its centre.) */
                const int64_t bi = x->blk + (int64_t)i;
                const int64_t start = bi * (int64_t)hop - (int64_t)ov;
                int64_t smod = ((int64_t)sh * start) % (int64_t)nfft;
                if (smod < 0) smod += (int64_t)nfft;
                const double a0 = (double)smod / (double)nfft;
                for (size_t k = 0; k < nvalid; ++k) {
                    const int64_t nabs = bi * (int64_t)hop + (int64_t)k * (int64_t)decim;
                    const double ph = -2.0 * ATK_PI * (a0 + c->delta * (double)nabs / x->fs);
                    const double cr = cos(ph), si = sin(ph);
                    const double yr = x->ys[ovd + k].re * invd, yi = x->ys[ovd + k].im * invd;
                    x->valid[k].re = (float)(yr * cr - yi * si);
                    x->valid[k].im = (float)(yr * si + yi * cr);
                }
                atkdsp_cf32 *dst = outs ? outs[(size_t)e * ns + ci] : NULL;
                const size_t wpos = out_counts[ci];
                size_t got;
                if (c->rs && c->rs[e]) {
                    if (!dst) return ATKDSP_E_ARG;
                    ptrdiff_t g = atkdsp_arb_resampler_process(c->rs[e], x->valid, nvalid,
                                                               dst + wpos, cap_each - wpos);
                    if (g < 0) return g;
                    got = (size_t)g;
                } else {
                    if (!dst) return ATKDSP_E_ARG;
                    if (wpos + nvalid > cap_each) return ATKDSP_E_CAP;
                    memcpy(dst + wpos, x->valid, nvalid * sizeof(atkdsp_cf32));
                    got = nvalid;
                }
                emit = got;   /* identical across elements */
            }
            out_counts[ci] += emit;
        }
    }
    /* new leftover */
    for (unsigned e = 0; e < nel; ++e) {
        /* the new leftover is the stream tail [consumed, avail); since
         * consumed = m*hop >= hop > lc, it always comes from ins (no overlap
         * with the old leftover being overwritten). */
        for (size_t s = 0; s < rem; ++s)
            x->leftover[(size_t)e * hop + s] = ins[e][consumed + s - lc];
    }
    x->lc = (unsigned)rem;
    x->blk += (int64_t)m;
    return (ptrdiff_t)ns;
}

ptrdiff_t atkdsp_osx_process(atkdsp_osx *x, const atkdsp_cf32 *in, size_t n,
                             atkdsp_cf32 *const *outs, size_t *out_counts, size_t cap_each) {
    const atkdsp_cf32 *ins[1] = { in };
    return atkdsp_osx_process_n(x, ins, 1, n, outs, out_counts, cap_each);
}

int atkdsp_osx_describe(const atkdsp_osx *x, char *buf, size_t cap) {
    if (!x || !buf || cap == 0) return ATKDSP_E_ARG;
    int active = 0;
    for (int i = 0; i < x->nslots; ++i) if (x->chans[i].active) ++active;
    return snprintf(buf, cap, "OSX %.6g MSPS: %zu-pt FFT, ov %zu, %d channel(s), %.0f dB, <=%u el",
                    x->fs / 1e6, x->nfft, x->ov, active, x->atten, x->max_nel);
}
