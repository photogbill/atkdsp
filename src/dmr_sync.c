/* 13. 4FSK sync correlator: DMR / P25 pilots and burst timing.
 *
 * A complex-baseband matched filter for the 24-symbol sync words of DMR (ETSI
 * TS 102 361-1 Table 9.2) and the P25 Phase 1 frame sync, with a bank of
 * carrier-offset hypotheses, across one or more phase-coherent channels at
 * once (the KrakenSDR's five elements). It exists for ATK's KRAKEN_DF_PLAN:
 * §6.5 pilot-aided steering vectors (each hit's per-channel complex
 * amplitudes ARE the array's steering vector for that emitter, with the
 * processing gain of 24 coherent symbols) and §7.1 decoder-clocked DoA
 * (sample-exact sync timing, and the sync type — BS/MS/direct-mode slot,
 * voice/data — that is the repeater-vs-handset path tag).
 *
 * The specification is the twin, python/atkdsp/reference.py §13, which also
 * carries the long explanation of the model. In one paragraph: the reference
 * is the sync word frequency-modulated exactly as a transmitter does it (each
 * symbol a/3 * deviation, root-raised-cosine frequency pulses or rectangles,
 * phase zero at the window centre); per window and pattern, each channel gets
 * one partial correlation per SYMBOL, and a carrier offset is then a phasor per
 * symbol, so the offset bank costs 24 MACs per hypothesis instead of L. The
 * score is sum_c |C_c|^2 / (L * sum_c E_c), in [0,1]. Peaks are picked by a
 * streamed rule (a candidate is replaced only by a better one, and emitted
 * once `guard` samples pass with nothing better), then refined: parabolas in
 * time and in offset, amplitudes by exact per-sample derotation at the refined
 * offset, and the DSD-style count of sync symbols whose frequency sign reads
 * wrong.
 *
 * Streaming: the last L + guard + 2 samples of every channel are carried, the
 * input is consumed in chunks of CHUNK samples into one scratch buffer sized at
 * create (rule 2), and every number depends only on sample values — so a hit
 * is bit-identical however the stream is chunked (rule 3; the cross-check test
 * asserts it).
 *
 * Cost, per sample per channel: patterns * L complex MACs for the partials,
 * plus patterns * bins * 24 for the offset bank — but a pattern whose partials
 * cannot reach the threshold at ANY offset (by the triangle inequality,
 * (sum_m |p_m|)^2 bounds every |C|^2) skips its bank, which in noise is almost
 * every pattern. */
#include "internal.h"

#include <stdio.h>

#define NSYM        ATKDSP_SYNC_SYMBOLS
#define PULSE_SPAN  4                       /* RRC truncated to +-4 symbols   */
#define PULSE_RES   64                      /* table points per symbol         */
#define PULSE_N     (2 * PULSE_SPAN * PULSE_RES)
#define FS_MIN      9600.0
#define FS_MAX      384000.0
#define MAX_CH      64
#define SPAN_MAX    5000.0
#define MAX_BINS    201
#define GUARD_SYMBOLS 66
#define CHUNK       4096
#define MAX_PAT     9
#define PRUNE_MARGIN 1e-4                   /* float error in the bound        */

/* ---- the sync words -------------------------------------------------------
 * ETSI TS 102 361-1 Table 9.2 and TIA-102.BAAA, checked symbol for symbol
 * against dsd-neo's sync_patterns.h (an independent source) on 2026-10-25.
 * The names, order and tags are the twin's (reference.py _DMR_SYNC). */
typedef struct { const char *name; unsigned long long word; } sync_word;

static const sync_word DMR_WORDS[MAX_PAT] = {
    {"BS voice",     0x755FD7DF75F7ULL},
    {"BS data",      0xDFF57D75DF5DULL},
    {"MS voice",     0x7F7D5DD57DFDULL},
    {"MS data",      0xD5D7F77FD757ULL},
    {"MS RC",        0x77D55F7DFD77ULL},
    {"DM TS1 voice", 0x5D577F7757FFULL},
    {"DM TS1 data",  0xF7FDD5DDFD55ULL},
    {"DM TS2 voice", 0x7DFFD5F55D5FULL},
    {"DM TS2 data",  0xD7557F5FF7F5ULL},
};
static const sync_word P25_WORDS[1] = {
    {"P25 frame sync", 0x5575F5FF77FFULL},
};

typedef struct {
    const char      *name;
    const sync_word *words;
    int              n;
    double           dev_hz;      /* deviation of a +3 symbol */
    double           alpha;       /* default pulse roll-off   */
} sync_set;

static const sync_set SETS[2] = {
    {"DMR", DMR_WORDS, 9, 1944.0, 0.2},
    {"P25", P25_WORDS, 1, 1800.0, 0.2},
};

/* Dibit -> symbol (ETSI TS 102 361-1 Table 10.3): 01 +3, 00 +1, 10 -1, 11 -3. */
static int dibit_symbol(int d) {
    switch (d & 3) {
    case 1:  return 3;
    case 0:  return 1;
    case 2:  return -1;
    default: return -3;
    }
}

static void word_symbols(unsigned long long w, signed char *out) {
    for (int i = 0; i < NSYM; ++i)
        out[i] = (signed char)dibit_symbol((int)((w >> (46 - 2 * i)) & 3ULL));
}

int atkdsp_dmr_sync_patterns(int set) {
    if (set < 0 || set > 1) return ATKDSP_E_ARG;
    return SETS[set].n;
}

int atkdsp_dmr_sync_pattern(int set, int idx, signed char *sym, size_t cap) {
    if (set < 0 || set > 1 || idx < 0 || idx >= SETS[set].n) return ATKDSP_E_ARG;
    if (!sym || cap < NSYM) return ATKDSP_E_CAP;
    word_symbols(SETS[set].words[idx].word, sym);
    return NSYM;
}

const char *atkdsp_dmr_sync_pattern_name(int set, int idx) {
    if (set < 0 || set > 1 || idx < 0 || idx >= SETS[set].n) return NULL;
    return SETS[set].words[idx].name;
}

/* ---- the reference waveform ------------------------------------------------ */
static double rrc(double u, double a) {
    if (u == 0.0) return 1.0 - a + 4.0 * a / ATK_PI;
    if (a > 0.0 && fabs(1.0 - 16.0 * a * a * u * u) < 1e-9) {
        double q = ATK_PI / (4.0 * a);
        return (a / sqrt(2.0)) * ((1.0 + 2.0 / ATK_PI) * sin(q) + (1.0 - 2.0 / ATK_PI) * cos(q));
    }
    double num = sin(ATK_PI * u * (1.0 - a)) + 4.0 * a * u * cos(ATK_PI * u * (1.0 + a));
    double den = ATK_PI * u * (1.0 - 16.0 * a * a * u * u);
    return num / den;
}

/* Running integral of the pulse on -4..+4 symbols, trapezoid, total exactly 1. */
static void pulse_table(double a, double *tab) {
    const double du = 1.0 / PULSE_RES;
    double prev = rrc(-PULSE_SPAN, a), acc = 0.0;
    tab[0] = 0.0;
    for (int k = 1; k <= PULSE_N; ++k) {
        double h = rrc(-PULSE_SPAN + k * du, a);
        acc += 0.5 * (prev + h) * du;
        tab[k] = acc;
        prev = h;
    }
    double total = tab[PULSE_N];
    for (int k = 0; k <= PULSE_N; ++k) tab[k] /= total;
}

static double pulse_cum(const double *tab, double u) {
    if (!tab) {
        double g = u + 0.5;
        return g < 0.0 ? 0.0 : (g > 1.0 ? 1.0 : g);
    }
    if (u >= PULSE_SPAN) return 1.0;
    if (u <= -PULSE_SPAN) return 0.0;
    double x = (u + PULSE_SPAN) * PULSE_RES;
    int k = (int)floor(x);
    if (k > PULSE_N - 1) k = PULSE_N - 1;
    double f = x - k;
    return tab[k] + f * (tab[k + 1] - tab[k]);
}

static double sync_phase(const signed char *sym, double dev, const double *tab, double u) {
    double acc = 0.0;
    for (int m = 0; m < NSYM; ++m)
        acc += ((double)sym[m] / 3.0) * pulse_cum(tab, u - m - 0.5);
    return 2.0 * ATK_PI * (dev / ATKDSP_SYNC_RATE) * acc;
}

/* ---- the handle --------------------------------------------------------------
 * Patterns are processed TOGETHER, one SIMD lane each: the searched patterns'
 * references are stored transposed (sample-major, NP lanes per sample, NP =
 * the pattern count rounded up to 8), so the partial correlations of every
 * pattern come out of one pass over the window with independent per-lane
 * accumulators — which vectorises without reassociating anything (rule 7). */
struct atkdsp_dmr_sync {
    int    set, n_ch, n_pat, NP, L, K, kmax, guard, H, W, flags;
    int    pat[MAX_PAT];          /* set indices of the searched patterns       */
    unsigned mask;
    double fs, sps, step, alpha;
    float  thr;
    int    seg[NSYM + 1];         /* symbol m is window samples [seg[m], seg[m+1]) */
    signed char sym[MAX_PAT][NSYM];
    float *rr, *ri;               /* set->n * L: every pattern's reference      */
    float *tr, *ti;               /* L * NP: the SEARCHED patterns, transposed  */
    float *rot_re, *rot_im;       /* (K+2) * NSYM: exp(-j 2 pi df_k tau_m / fs),
                                     bins -1..K: the outer two only refine an
                                     edge hit, detection uses 0..K-1           */
    float *bre, *bim;             /* n_ch * W: carried history + one chunk      */
    int    hist;                  /* valid samples at the front of each row     */
    long long total;              /* absolute index of buffer position `hist`   */
    float *best;                  /* CHUNK: best score per new sample (-1 none) */
    int   *best_pk;               /* CHUNK: active pattern * K + bin            */
    float *pre, *pim;             /* n_ch * NSYM * NP partials (one window)     */
    float *sc;                    /* K * NP scores (one window)                 */
    double *qre, *qim;            /* L: conj(ref) * derotation, at an emission  */
    int    cand;                  /* a candidate is pending                     */
    long long cand_n;             /* its window's last sample, absolute         */
    float  cand_s;
    int    cand_pa, cand_k;
};

void atkdsp_dmr_sync_destroy(atkdsp_dmr_sync *s) {
    if (!s) return;
    atk_aligned_free(s->rr);     atk_aligned_free(s->ri);
    atk_aligned_free(s->tr);     atk_aligned_free(s->ti);
    atk_aligned_free(s->rot_re); atk_aligned_free(s->rot_im);
    atk_aligned_free(s->bre);    atk_aligned_free(s->bim);
    atk_aligned_free(s->best);   atk_aligned_free(s->best_pk);
    atk_aligned_free(s->pre);    atk_aligned_free(s->pim);
    atk_aligned_free(s->sc);
    atk_aligned_free(s->qre);    atk_aligned_free(s->qim);
    free(s);
}

void atkdsp_dmr_sync_reset(atkdsp_dmr_sync *s) {
    if (!s) return;
    s->hist = 0;
    s->total = 0;
    s->cand = 0;
}

atkdsp_dmr_sync *atkdsp_dmr_sync_create(int set, double sample_rate, int n_channels,
                                        unsigned pattern_mask, double cfo_span_hz,
                                        double cfo_step_hz, float threshold, int guard,
                                        double alpha, int flags) {
    if (set < 0 || set > 1) return NULL;
    if (!(sample_rate >= FS_MIN && sample_rate <= FS_MAX)) return NULL;
    if (n_channels < 1 || n_channels > MAX_CH) return NULL;
    const sync_set *st = &SETS[set];
    unsigned full = (1u << st->n) - 1u;
    unsigned mask = pattern_mask & full;
    if (mask == 0) return NULL;
    if (isnan(alpha)) alpha = st->alpha;
    if (alpha > 1.0) return NULL;
    if (!(cfo_span_hz >= 0.0 && cfo_span_hz <= SPAN_MAX)) return NULL;
    if (!(cfo_step_hz >= 0.0)) return NULL;
    double step = cfo_step_hz > 0.0 ? cfo_step_hz : 0.5 * ATKDSP_SYNC_RATE / NSYM;
    int kmax = cfo_span_hz > 0.0 ? (int)ceil(cfo_span_hz / step - 1e-9) : 0;
    if (2 * kmax + 1 > MAX_BINS) return NULL;
    if (!(threshold > 0.0f && threshold <= 1.0f)) return NULL;

    double sps = sample_rate / ATKDSP_SYNC_RATE;
    int L = (int)floor(NSYM * sps + 0.5);
    int g = guard > 0 ? guard : (int)floor(GUARD_SYMBOLS * sps + 0.5);
    if (g > 16 * L) return NULL;

    atkdsp_dmr_sync *s = (atkdsp_dmr_sync *)calloc(1, sizeof *s);
    if (!s) return NULL;
    s->set = set; s->n_ch = n_channels; s->mask = mask; s->flags = flags;
    s->fs = sample_rate; s->sps = sps; s->step = step; s->alpha = alpha;
    s->thr = threshold; s->L = L; s->kmax = kmax; s->K = 2 * kmax + 1; s->guard = g;
    s->H = L + g + 2;
    s->W = s->H + CHUNK;
    for (int p = 0; p < st->n; ++p) {
        word_symbols(st->words[p].word, s->sym[p]);
        if (mask >> p & 1u) s->pat[s->n_pat++] = p;
    }
    s->NP = (s->n_pat + 7) / 8 * 8;
    for (int m = 0; m < NSYM; ++m) {
        int b = (int)ceil(m * sps - 0.5);
        s->seg[m] = b < 0 ? 0 : (b > L ? L : b);
    }
    s->seg[NSYM] = L;

    size_t nref = (size_t)st->n * (size_t)L;
    size_t ntr = (size_t)L * (size_t)s->NP;
    size_t nbuf = (size_t)n_channels * (size_t)s->W;
    size_t npart = (size_t)n_channels * NSYM * (size_t)s->NP;
    s->rr = (float *)atk_aligned_malloc(nref * sizeof(float));
    s->ri = (float *)atk_aligned_malloc(nref * sizeof(float));
    s->tr = (float *)atk_aligned_malloc(ntr * sizeof(float));
    s->ti = (float *)atk_aligned_malloc(ntr * sizeof(float));
    s->rot_re = (float *)atk_aligned_malloc((size_t)(s->K + 2) * NSYM * sizeof(float));
    s->rot_im = (float *)atk_aligned_malloc((size_t)(s->K + 2) * NSYM * sizeof(float));
    s->bre = (float *)atk_aligned_malloc(nbuf * sizeof(float));
    s->bim = (float *)atk_aligned_malloc(nbuf * sizeof(float));
    s->best = (float *)atk_aligned_malloc(CHUNK * sizeof(float));
    s->best_pk = (int *)atk_aligned_malloc(CHUNK * sizeof(int));
    s->pre = (float *)atk_aligned_malloc(npart * sizeof(float));
    s->pim = (float *)atk_aligned_malloc(npart * sizeof(float));
    s->sc = (float *)atk_aligned_malloc((size_t)s->K * (size_t)s->NP * sizeof(float));
    s->qre = (double *)atk_aligned_malloc((size_t)L * sizeof(double));
    s->qim = (double *)atk_aligned_malloc((size_t)L * sizeof(double));
    if (!s->rr || !s->ri || !s->tr || !s->ti || !s->rot_re || !s->rot_im || !s->bre ||
        !s->bim || !s->best || !s->best_pk || !s->pre || !s->pim || !s->sc ||
        !s->qre || !s->qim) {
        atkdsp_dmr_sync_destroy(s);
        return NULL;
    }

    /* The reference, phase zero at the window centre u = L / (2 sps). */
    double tab[PULSE_N + 1];
    const double *tp = NULL;
    if (alpha >= 0.0) { pulse_table(alpha, tab); tp = tab; }
    double uc = (double)L / (2.0 * sps);
    for (int p = 0; p < st->n; ++p) {
        double ph0 = sync_phase(s->sym[p], st->dev_hz, tp, uc);
        for (int i = 0; i < L; ++i) {
            double ph = sync_phase(s->sym[p], st->dev_hz, tp, (i + 0.5) / sps) - ph0;
            s->rr[(size_t)p * L + i] = (float)cos(ph);
            s->ri[(size_t)p * L + i] = (float)((flags & ATKDSP_SYNC_INVERT) ? -sin(ph) : sin(ph));
        }
    }
    memset(s->tr, 0, ntr * sizeof(float));      /* padding lanes stay zero */
    memset(s->ti, 0, ntr * sizeof(float));
    for (int pa = 0; pa < s->n_pat; ++pa)
        for (int i = 0; i < L; ++i) {
            s->tr[(size_t)i * s->NP + pa] = s->rr[(size_t)s->pat[pa] * L + i];
            s->ti[(size_t)i * s->NP + pa] = s->ri[(size_t)s->pat[pa] * L + i];
        }
    /* The offset bank: one phasor per symbol, at the symbol's centre; one
     * extra bin beyond each edge so an edge hit still has two neighbours. */
    for (int k = -1; k <= s->K; ++k) {
        double df = (k - kmax) * step;
        for (int m = 0; m < NSYM; ++m) {
            double tau = (s->seg[m] + s->seg[m + 1] - 1) / 2.0 - (L - 1) / 2.0;
            double a = -2.0 * ATK_PI * df * tau / sample_rate;
            s->rot_re[(k + 1) * NSYM + m] = (float)cos(a);
            s->rot_im[(k + 1) * NSYM + m] = (float)sin(a);
        }
    }
    atkdsp_dmr_sync_reset(s);
    return s;
}

int atkdsp_dmr_sync_info(const atkdsp_dmr_sync *s, int *window, int *n_cfo,
                         double *cfo_step_hz, int *guard) {
    if (!s) return ATKDSP_E_ARG;
    if (window) *window = s->L;
    if (n_cfo) *n_cfo = s->K;
    if (cfo_step_hz) *cfo_step_hz = s->step;
    if (guard) *guard = s->guard;
    return ATKDSP_OK;
}

size_t atkdsp_dmr_sync_hits_max(const atkdsp_dmr_sync *s, size_t n_in) {
    if (!s) return 0;
    return n_in / (size_t)s->guard + 2;
}

int atkdsp_dmr_sync_reference(const atkdsp_dmr_sync *s, int pattern, atkdsp_cf32 *out,
                              size_t cap) {
    if (!s || pattern < 0 || pattern >= SETS[s->set].n) return ATKDSP_E_ARG;
    if (!out || cap < (size_t)s->L) return ATKDSP_E_CAP;
    for (int i = 0; i < s->L; ++i) {
        out[i].re = s->rr[(size_t)pattern * s->L + i];
        out[i].im = s->ri[(size_t)pattern * s->L + i];
    }
    return s->L;
}

int atkdsp_dmr_sync_describe(const atkdsp_dmr_sync *s, char *buf, size_t cap) {
    if (!s || !buf || cap == 0) return ATKDSP_E_ARG;
    char pulse[32];
    if (s->alpha < 0.0) snprintf(pulse, sizeof pulse, "rect");
    else snprintf(pulse, sizeof pulse, "RRC a=%.2f", s->alpha);
    int n = snprintf(buf, cap,
        "%s sync: %.0f Hz (%.2f sps), %d-sample window, %d pattern%s, "
        "CFO +-%.0f Hz in %d bin%s of %.0f Hz, threshold %.2f, guard %d, %s, "
        "%d channel%s%s",
        SETS[s->set].name, s->fs, s->sps, s->L, s->n_pat, s->n_pat != 1 ? "s" : "",
        s->kmax * s->step, s->K, s->K != 1 ? "s" : "", s->step, (double)s->thr, s->guard,
        pulse, s->n_ch, s->n_ch != 1 ? "s" : "",
        (s->flags & ATKDSP_SYNC_INVERT) ? ", inverted" : "");
    if (n < 0) return ATKDSP_E_ARG;
    return (size_t)n < cap ? n : (int)cap - 1;
}

/* ---- the arithmetic ------------------------------------------------------------
 * `e` is a buffer position: the window is bre/bim[c*W + e-L+1 .. c*W + e]. */
static double window_energy(const atkdsp_dmr_sync *s, int e) {
    double E = 0.0;
    for (int c = 0; c < s->n_ch; ++c) {
        const float *ATK_RESTRICT xr = s->bre + (size_t)c * s->W + (e - s->L + 1);
        const float *ATK_RESTRICT xi = s->bim + (size_t)c * s->W + (e - s->L + 1);
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};    /* 8 lanes: vectorises as is */
        int i = 0;
        for (; i + 8 <= s->L; i += 8)
            for (int l = 0; l < 8; ++l) acc[l] += xr[i + l] * xr[i + l] + xi[i + l] * xi[i + l];
        float tail = 0.0f;
        for (; i < s->L; ++i) tail += xr[i] * xr[i] + xi[i] * xi[i];
        double sum = tail;
        for (int l = 0; l < 8; ++l) sum += acc[l];
        E += sum;
    }
    return E;
}

/* Partial correlations of every searched pattern at once: for channel c,
 * symbol m and lane pa, pre/pim[(c*NSYM + m)*NP + pa] = sum over the symbol's
 * samples of x[i] * conj(r_pa[i]), accumulated as two independent sums,
 * sum(a*r) + sum(b*q) and sum(b*r) - sum(a*q), so no accumulator waits on
 * another. This is the hot loop: per sample and channel, L * NP lanes.
 *
 * AVX2 (both compilers define __AVX2__ for it: gcc -mavx2, MSVC /arch:AVX2) is
 * written out, because left to themselves the auto-vectorisers did not keep
 * these accumulators in registers (measured: scalar code, 4-8x slower). The
 * fallback is the same arithmetic in plain C. */
#if defined(__AVX2__)
#  include <immintrin.h>
/* Eight independent accumulators in flight either way — FMA latency is four
 * cycles and the core retires two a cycle, so fewer than eight chains leave it
 * waiting: 16 lanes run both halves in one pass, 8 lanes split even and odd
 * samples. (Measured: the 4-chain version ran at exactly the latency bound.) */
static void partials(atkdsp_dmr_sync *s, int e) {
    const int NP = s->NP;
    for (int c = 0; c < s->n_ch; ++c) {
        const float *xr = s->bre + (size_t)c * s->W + (e - s->L + 1);
        const float *xi = s->bim + (size_t)c * s->W + (e - s->L + 1);
        for (int m = 0; m < NSYM; ++m) {
            float *dr = s->pre + ((size_t)c * NSYM + m) * NP;
            float *di = s->pim + ((size_t)c * NSYM + m) * NP;
            const int i0 = s->seg[m], i1 = s->seg[m + 1];
            if (NP == 16) {
                __m256 ar0 = _mm256_setzero_ps(), br0 = _mm256_setzero_ps();
                __m256 ai0 = _mm256_setzero_ps(), bi0 = _mm256_setzero_ps();
                __m256 ar1 = _mm256_setzero_ps(), br1 = _mm256_setzero_ps();
                __m256 ai1 = _mm256_setzero_ps(), bi1 = _mm256_setzero_ps();
                for (int i = i0; i < i1; ++i) {
                    const __m256 a = _mm256_set1_ps(xr[i]), b = _mm256_set1_ps(xi[i]);
                    const float *rp = s->tr + (size_t)i * 16, *qp = s->ti + (size_t)i * 16;
                    const __m256 r0 = _mm256_loadu_ps(rp), q0 = _mm256_loadu_ps(qp);
                    const __m256 r1 = _mm256_loadu_ps(rp + 8), q1 = _mm256_loadu_ps(qp + 8);
                    ar0 = _mm256_fmadd_ps(a, r0, ar0); br0 = _mm256_fmadd_ps(b, q0, br0);
                    ai0 = _mm256_fmadd_ps(b, r0, ai0); bi0 = _mm256_fmadd_ps(a, q0, bi0);
                    ar1 = _mm256_fmadd_ps(a, r1, ar1); br1 = _mm256_fmadd_ps(b, q1, br1);
                    ai1 = _mm256_fmadd_ps(b, r1, ai1); bi1 = _mm256_fmadd_ps(a, q1, bi1);
                }
                _mm256_storeu_ps(dr, _mm256_add_ps(ar0, br0));
                _mm256_storeu_ps(di, _mm256_sub_ps(ai0, bi0));
                _mm256_storeu_ps(dr + 8, _mm256_add_ps(ar1, br1));
                _mm256_storeu_ps(di + 8, _mm256_sub_ps(ai1, bi1));
            } else {                                         /* NP == 8 */
                __m256 ar0 = _mm256_setzero_ps(), br0 = _mm256_setzero_ps();
                __m256 ai0 = _mm256_setzero_ps(), bi0 = _mm256_setzero_ps();
                __m256 ar1 = _mm256_setzero_ps(), br1 = _mm256_setzero_ps();
                __m256 ai1 = _mm256_setzero_ps(), bi1 = _mm256_setzero_ps();
                int i = i0;
                for (; i + 1 < i1; i += 2) {
                    const __m256 a0 = _mm256_set1_ps(xr[i]), b0 = _mm256_set1_ps(xi[i]);
                    const __m256 a1 = _mm256_set1_ps(xr[i + 1]), b1 = _mm256_set1_ps(xi[i + 1]);
                    const __m256 r0 = _mm256_loadu_ps(s->tr + (size_t)i * 8);
                    const __m256 q0 = _mm256_loadu_ps(s->ti + (size_t)i * 8);
                    const __m256 r1 = _mm256_loadu_ps(s->tr + (size_t)(i + 1) * 8);
                    const __m256 q1 = _mm256_loadu_ps(s->ti + (size_t)(i + 1) * 8);
                    ar0 = _mm256_fmadd_ps(a0, r0, ar0); br0 = _mm256_fmadd_ps(b0, q0, br0);
                    ai0 = _mm256_fmadd_ps(b0, r0, ai0); bi0 = _mm256_fmadd_ps(a0, q0, bi0);
                    ar1 = _mm256_fmadd_ps(a1, r1, ar1); br1 = _mm256_fmadd_ps(b1, q1, br1);
                    ai1 = _mm256_fmadd_ps(b1, r1, ai1); bi1 = _mm256_fmadd_ps(a1, q1, bi1);
                }
                if (i < i1) {
                    const __m256 a0 = _mm256_set1_ps(xr[i]), b0 = _mm256_set1_ps(xi[i]);
                    const __m256 r0 = _mm256_loadu_ps(s->tr + (size_t)i * 8);
                    const __m256 q0 = _mm256_loadu_ps(s->ti + (size_t)i * 8);
                    ar0 = _mm256_fmadd_ps(a0, r0, ar0); br0 = _mm256_fmadd_ps(b0, q0, br0);
                    ai0 = _mm256_fmadd_ps(b0, r0, ai0); bi0 = _mm256_fmadd_ps(a0, q0, bi0);
                }
                _mm256_storeu_ps(dr, _mm256_add_ps(_mm256_add_ps(ar0, ar1), _mm256_add_ps(br0, br1)));
                _mm256_storeu_ps(di, _mm256_sub_ps(_mm256_add_ps(ai0, ai1), _mm256_add_ps(bi0, bi1)));
            }
        }
    }
}
#else
static void partials(atkdsp_dmr_sync *s, int e) {
    const int NP = s->NP;
    for (int c = 0; c < s->n_ch; ++c) {
        const float *xr = s->bre + (size_t)c * s->W + (e - s->L + 1);
        const float *xi = s->bim + (size_t)c * s->W + (e - s->L + 1);
        for (int m = 0; m < NSYM; ++m) {
            float ar[16], br[16], ai[16], bi[16];
            for (int p = 0; p < NP; ++p) { ar[p] = br[p] = ai[p] = bi[p] = 0.0f; }
            for (int i = s->seg[m]; i < s->seg[m + 1]; ++i) {
                const float a = xr[i], b = xi[i];
                const float *r = s->tr + (size_t)i * NP, *q = s->ti + (size_t)i * NP;
                for (int p = 0; p < NP; ++p) {
                    ar[p] += a * r[p];
                    br[p] += b * q[p];
                    ai[p] += b * r[p];
                    bi[p] += a * q[p];
                }
            }
            float *dr = s->pre + ((size_t)c * NSYM + m) * NP;
            float *di = s->pim + ((size_t)c * NSYM + m) * NP;
            for (int p = 0; p < NP; ++p) { dr[p] = ar[p] + br[p]; di[p] = ai[p] - bi[p]; }
        }
    }
}
#endif

/* Scores of every lane at bin k into out[NP]: sum_c |sum_m p_cm * rot_km|^2,
 * divided by L*E. One function for detection and for the refinement's grid
 * scores, so the two are the same arithmetic. */
static void combine(atkdsp_dmr_sync *s, int k, double LE, float *ATK_RESTRICT out) {
    const int NP = s->NP;
    const float *qr = s->rot_re + (k + 1) * NSYM, *qi = s->rot_im + (k + 1) * NSYM;
    float acc[16 * ((MAX_PAT + 15) / 16)];
    for (int p = 0; p < NP; ++p) acc[p] = 0.0f;
    for (int c = 0; c < s->n_ch; ++c) {
        float cr[16 * ((MAX_PAT + 15) / 16)], ci[16 * ((MAX_PAT + 15) / 16)];
        for (int p = 0; p < NP; ++p) { cr[p] = 0.0f; ci[p] = 0.0f; }
        for (int m = 0; m < NSYM; ++m) {
            const float *ATK_RESTRICT a = s->pre + ((size_t)c * NSYM + m) * NP;
            const float *ATK_RESTRICT b = s->pim + ((size_t)c * NSYM + m) * NP;
            const float u = qr[m], v = qi[m];
            for (int p = 0; p < NP; ++p) {
                cr[p] += a[p] * u - b[p] * v;
                ci[p] += a[p] * v + b[p] * u;
            }
        }
        for (int p = 0; p < NP; ++p) acc[p] += cr[p] * cr[p] + ci[p] * ci[p];
    }
    const float inv = (float)(1.0 / LE);
    for (int p = 0; p < NP; ++p) out[p] = acc[p] * inv;
}

/* The best (score, pattern, bin) of the window ending at e; -1 when nothing can
 * reach the threshold. A pattern whose partials cannot reach it at ANY offset —
 * (sum_m |p_cm|)^2 summed over channels bounds every bin's |C|^2 — is out, and
 * when every pattern is out the offset bank is skipped: in noise, nearly every
 * sample. Ties keep the first (pattern ascending, then bin), as numpy's argmax
 * does in the twin. */
static void detect_one(atkdsp_dmr_sync *s, int e, float *best, int *best_pk) {
    const int NP = s->NP;
    *best = -1.0f;
    *best_pk = 0;
    double E = window_energy(s, e);
    if (!(E > 1e-30)) return;
    double LE = (double)s->L * E;
    partials(s, e);
    float bound[16 * ((MAX_PAT + 15) / 16)];
    for (int p = 0; p < NP; ++p) bound[p] = 0.0f;
    for (int c = 0; c < s->n_ch; ++c) {
        float sum[16 * ((MAX_PAT + 15) / 16)];
        for (int p = 0; p < NP; ++p) sum[p] = 0.0f;
        for (int m = 0; m < NSYM; ++m) {
            const float *ATK_RESTRICT a = s->pre + ((size_t)c * NSYM + m) * NP;
            const float *ATK_RESTRICT b = s->pim + ((size_t)c * NSYM + m) * NP;
            for (int p = 0; p < NP; ++p) sum[p] += sqrtf(a[p] * a[p] + b[p] * b[p]);
        }
        for (int p = 0; p < NP; ++p) bound[p] += sum[p] * sum[p];
    }
    const float floor_bound = (float)((double)s->thr * LE * (1.0 - PRUNE_MARGIN));
    int any = 0;
    for (int pa = 0; pa < s->n_pat; ++pa) any |= bound[pa] >= floor_bound;
    if (!any) return;
    for (int k = 0; k < s->K; ++k) combine(s, k, LE, s->sc + (size_t)k * NP);
    float b = -1.0f;
    int bpk = 0;
    for (int pa = 0; pa < s->n_pat; ++pa) {
        if (bound[pa] < floor_bound) continue;  /* every bin of it is below thr */
        for (int k = 0; k < s->K; ++k) {
            float sc = s->sc[(size_t)k * NP + pa];
            if (sc > b) { b = sc; bpk = pa * s->K + k; }
        }
    }
    *best = b;
    *best_pk = bpk;
}

/* One grid score, by exactly the detection arithmetic. */
static double score_at(atkdsp_dmr_sync *s, int e, int pa, int k) {
    double E = window_energy(s, e);
    if (!(E > 1e-30)) return 0.0;
    partials(s, e);
    float out[16 * ((MAX_PAT + 15) / 16)];
    combine(s, k, (double)s->L * E, out);
    return out[pa];
}

/* The exact score at a refined offset: conj(ref) * derotation is in qre/qim
 * (set by emit); amplitudes go to amps when it is not NULL. */
static double exact_at(atkdsp_dmr_sync *s, int e, atkdsp_cf32 *amps) {
    const int L = s->L;
    double E = 0.0, A = 0.0;
    for (int c = 0; c < s->n_ch; ++c) {
        const float *xr = s->bre + (size_t)c * s->W + (e - L + 1);
        const float *xi = s->bim + (size_t)c * s->W + (e - L + 1);
        double ar = 0.0, ai = 0.0;
        for (int i = 0; i < L; ++i) {
            ar += xr[i] * s->qre[i] - xi[i] * s->qim[i];
            ai += xr[i] * s->qim[i] + xi[i] * s->qre[i];
            E += (double)xr[i] * xr[i] + (double)xi[i] * xi[i];
        }
        ar /= L;
        ai /= L;
        A += ar * ar + ai * ai;
        if (amps) { amps[c].re = (float)ar; amps[c].im = (float)ai; }
    }
    return E > 1e-30 ? (double)L * A / E : 0.0;
}

static double parab(int have_a, double a, double b, int have_c, double c) {
    if (!have_a || !have_c) return 0.0;
    double den = a - 2.0 * b + c;
    if (!(den < 0.0)) return 0.0;
    double d = 0.5 * (a - c) / den;
    return d < -0.5 ? -0.5 : (d > 0.5 ? 0.5 : d);
}

/* Refine and write the pending candidate. `base` is the absolute index of
 * buffer position 0; `last` the last valid buffer position. The offset is
 * refined first (a parabola through the neighbouring bins), THEN the timing,
 * from exact scores at that offset: for FSK a residual offset looks like a
 * timing error, so the other order biases the timing. */
static void emit(atkdsp_dmr_sync *s, long long base, int last,
                 atkdsp_dmr_sync_hit *hit, atkdsp_cf32 *amps) {
    const int L = s->L, pa = s->cand_pa, k = s->cand_k, p = s->pat[pa];
    const long long n = s->cand_n;
    const int e = (int)(n - base);
    double g0 = score_at(s, e, pa, k);
    double gm = score_at(s, e, pa, k - 1);      /* bins -1 and K exist for this */
    double gp = score_at(s, e, pa, k + 1);
    double cfo = (k - s->kmax + parab(1, gm, g0, 1, gp)) * s->step;

    const float *rr = s->rr + (size_t)p * L, *ri = s->ri + (size_t)p * L;
    double w = -2.0 * ATK_PI * cfo / s->fs, ic = (L - 1) / 2.0;
    for (int i = 0; i < L; ++i) {
        double ang = w * (i - ic), cr = cos(ang), ci = sin(ang);
        s->qre[i] = rr[i] * cr + ri[i] * ci;      /* conj(r) * exp(j ang) */
        s->qim[i] = rr[i] * ci - ri[i] * cr;
    }
    double s0 = exact_at(s, e, amps);
    int have_m = n - 1 >= L - 1, have_p = e + 1 <= last;
    double sm = have_m ? exact_at(s, e - 1, NULL) : 0.0;
    double sp = have_p ? exact_at(s, e + 1, NULL) : 0.0;
    double dt = parab(have_m, sm, s0, have_p, sp);

    /* Sync symbols with the wrong frequency sign: the phase advance inside each
     * symbol, x[i] * conj(x[i-1]) over its own sample pairs, summed over
     * channels (their phases cancel in the product), offset removed. */
    int errors = 0;
    double rc = cos(w), rs = sin(w);
    for (int m = 0; m < NSYM; ++m) {
        double dr = 0.0, di = 0.0;
        for (int c = 0; c < s->n_ch; ++c) {
            const float *xr = s->bre + (size_t)c * s->W + (e - L + 1);
            const float *xi = s->bim + (size_t)c * s->W + (e - L + 1);
            for (int i = s->seg[m] + 1; i < s->seg[m + 1]; ++i) {
                dr += (double)xr[i] * xr[i - 1] + (double)xi[i] * xi[i - 1];
                di += (double)xi[i] * xr[i - 1] - (double)xr[i] * xi[i - 1];
            }
        }
        double im = dr * rs + di * rc;          /* imag of (dr + j di) * e^{jw} */
        int want = s->sym[p][m] > 0 ? 1 : -1;
        if (s->flags & ATKDSP_SYNC_INVERT) want = -want;
        int got = im > 0.0 ? 1 : (im < 0.0 ? -1 : 0);
        errors += got != want;
    }

    hit->start = n - L + 1;
    hit->frac = dt;
    hit->cfo_hz = cfo;
    hit->score = (float)s0;
    hit->errors = errors;
    hit->pattern = p;
    hit->reserved = 0;
}

ptrdiff_t atkdsp_dmr_sync_process(atkdsp_dmr_sync *s, const atkdsp_cf32 *in, size_t n,
                                  size_t ch_stride, atkdsp_dmr_sync_hit *hits,
                                  atkdsp_cf32 *amps, size_t cap) {
    if (!s || (!in && n)) return ATKDSP_E_ARG;
    if (s->n_ch > 1 && n && ch_stride < n) return ATKDSP_E_ARG;
    if (cap < atkdsp_dmr_sync_hits_max(s, n) || (!hits && cap)) return ATKDSP_E_CAP;
    const int L = s->L, W = s->W;
    size_t nh = 0;
    for (size_t off = 0; off < n; ) {
        int m = (int)((n - off) < (size_t)CHUNK ? (n - off) : (size_t)CHUNK);
        for (int c = 0; c < s->n_ch; ++c) {
            const atkdsp_cf32 *src = in + (size_t)c * ch_stride + off;
            float *dr = s->bre + (size_t)c * W + s->hist;
            float *di = s->bim + (size_t)c * W + s->hist;
            for (int j = 0; j < m; ++j) { dr[j] = src[j].re; di[j] = src[j].im; }
        }
        for (int j = 0; j < m; ++j) {
            int e = s->hist + j;
            if (e >= L - 1) detect_one(s, e, &s->best[j], &s->best_pk[j]);
            else { s->best[j] = -1.0f; s->best_pk[j] = 0; }
        }
        const long long base = s->total - s->hist;
        const int last = s->hist + m - 1;
        for (int j = 0; j < m; ++j) {
            long long a = s->total + j;
            if (s->cand && a - s->cand_n >= s->guard) {
                emit(s, base, last, &hits[nh], amps ? amps + nh * (size_t)s->n_ch : NULL);
                ++nh;
                s->cand = 0;
            }
            if (s->best[j] >= s->thr && (!s->cand || s->best[j] > s->cand_s)) {
                s->cand = 1;
                s->cand_n = a;
                s->cand_s = s->best[j];
                s->cand_pa = s->best_pk[j] / s->K;
                s->cand_k = s->best_pk[j] % s->K;
            }
        }
        s->total += m;
        s->hist += m;
        if (s->hist > s->H) {
            int drop = s->hist - s->H;
            for (int c = 0; c < s->n_ch; ++c) {
                memmove(s->bre + (size_t)c * W, s->bre + (size_t)c * W + drop, (size_t)s->H * sizeof(float));
                memmove(s->bim + (size_t)c * W, s->bim + (size_t)c * W + drop, (size_t)s->H * sizeof(float));
            }
            s->hist = s->H;
        }
        off += (size_t)m;
    }
    return (ptrdiff_t)nh;
}

ptrdiff_t atkdsp_dmr_sync_flush(atkdsp_dmr_sync *s, atkdsp_dmr_sync_hit *hits,
                                atkdsp_cf32 *amps, size_t cap) {
    if (!s) return ATKDSP_E_ARG;
    if (!s->cand) return 0;
    if (!hits || cap < 1) return ATKDSP_E_CAP;
    emit(s, s->total - s->hist, s->hist - 1, &hits[0], amps);
    s->cand = 0;
    return 1;
}
