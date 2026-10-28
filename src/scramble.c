/* Scramblers and linear complexity (header §15).
 *
 * Berlekamp-Massey runs on packed words: the discrepancy at step N is the
 * parity of C(x) AND the last L+1 bits read backwards, so the stream is kept
 * reversed and each step is a handful of popcounts.
 *
 * The scrambler mask convention is ATK siga.bits' (lfsr_sequence): degree D
 * = the highest set bit, bit 0 set, r[t+D] = XOR_{i<D, bit i} r[t+i]. In the
 * BM connection polynomial c_0..c_L that is mask bit i = c_{L-i}; the binding
 * converts (atkdsp.bm_mask).
 *
 * The search applies each candidate as the feed-forward filter
 *   y[t] = b[t] XOR (XOR_{i<D, bit i} b[t-D+i])
 * — the multiplicative descrambler, and also the annihilator of an additive
 * scrambler's sequence whatever its seed — and scores the structure left in
 * y with the G statistic of its words (reference.py §15 is the same search,
 * in the same order, with the same arithmetic). */
#include "internal.h"

#ifdef _OPENMP
#  include <omp.h>
#endif

#define LN10 2.302585092994045684

/* C ^= B << m, both packed (bit i = coefficient of x^i), C has nwc words. */
static void xor_shifted(uint64_t *C, const uint64_t *B, size_t nwb, size_t m, size_t nwc) {
    const size_t q = m >> 6;
    const unsigned r = (unsigned)(m & 63u);
    for (size_t j = 0; j < nwb; ++j) {
        const uint64_t v = B[j];
        if (!v) continue;
        if (j + q < nwc) C[j + q] ^= v << r;
        if (r && j + q + 1 < nwc) C[j + q + 1] ^= v >> (64u - r);
    }
}

/* The shared body of bm / bm_profile. */
static ptrdiff_t bm_run(const uint8_t *b, size_t n, uint8_t *poly, size_t cap, int32_t *prof) {
    const size_t nw = (n + 2 + 63) / 64 + 1;
    uint64_t *C = (uint64_t *)calloc(nw, sizeof *C);
    uint64_t *B = (uint64_t *)calloc(nw, sizeof *B);
    uint64_t *T = (uint64_t *)calloc(nw, sizeof *T);
    uint64_t *rs = (uint64_t *)calloc((n + 63) / 64 + 3, sizeof *rs);
    if (!C || !B || !T || !rs) { free(C); free(B); free(T); free(rs); return ATKDSP_E_NOMEM; }
    for (size_t i = 0; i < n; ++i)               /* rs[j] = b[n-1-j] */
        if (b[i] & 1u) { const size_t j = n - 1 - i; rs[j >> 6] |= 1ULL << (j & 63u); }
    C[0] = B[0] = 1;
    size_t L = 0, m = 1;
    for (size_t N = 0; N < n; ++N) {
        /* d = XOR_{i=0..L} c_i b[N-i]; b[N-i] = rs[n-1-N+i] */
        const size_t p0 = n - 1 - N;
        const size_t words = (L + 1 + 63) / 64;
        uint64_t acc = 0;
        for (size_t k = 0; k < words; ++k) {
            uint64_t cw = C[k];
            if (k == words - 1 && ((L + 1) & 63u)) cw &= (1ULL << ((L + 1) & 63u)) - 1;
            acc ^= cw & atk_bits_get64(rs, p0 + 64 * k);
        }
        const int d = atk_parity64(acc);
        if (!d) {
            ++m;
        } else if (2 * L <= N) {
            memcpy(T, C, nw * sizeof *C);
            xor_shifted(C, B, (L + 1 + 63) / 64 + 1, m, nw);
            L = N + 1 - L;
            memcpy(B, T, nw * sizeof *B);
            m = 1;
        } else {
            xor_shifted(C, B, (L + 1 + 63) / 64 + 1, m, nw);
            ++m;
        }
        if (prof) prof[N] = (int32_t)L;
    }
    ptrdiff_t ret = (ptrdiff_t)L;
    if (poly) {
        if (cap < L + 1) ret = ATKDSP_E_CAP;
        else for (size_t i = 0; i <= L; ++i) poly[i] = (uint8_t)((C[i >> 6] >> (i & 63u)) & 1u);
    }
    free(C); free(B); free(T); free(rs);
    return ret;
}

ptrdiff_t atkdsp_bm(const uint8_t *b, size_t n, uint8_t *poly, size_t cap) {
    if ((!b && n) || !poly) return ATKDSP_E_ARG;
    return bm_run(b, n, poly, cap, NULL);
}

ptrdiff_t atkdsp_bm_profile(const uint8_t *b, size_t n, int32_t *L_of_n) {
    if ((!b && n) || (!L_of_n && n)) return ATKDSP_E_ARG;
    return bm_run(b, n, NULL, 0, L_of_n);
}

/* ---- the registers -------------------------------------------------------- */

static int mask_degree(uint64_t poly) {
    int d = 63;                    /* never shifts by 64 (UB, and a hang on gcc) */
    while (d >= 0 && !((poly >> d) & 1u)) --d;
    return d;
}

static int check_mask(uint64_t poly, int *D, uint64_t *low) {
    const int d = mask_degree(poly);
    if (d < 2 || d > 63 || !(poly & 1u)) return 0;
    *D = d;
    *low = poly & ((1ULL << d) - 1);
    return 1;
}

int atkdsp_descramble_add(const uint8_t *b, size_t n, uint64_t poly, uint64_t *state,
                          uint8_t *out) {
    int D; uint64_t low;
    if ((!b && n) || (!out && n) || !state || !check_mask(poly, &D, &low)) return ATKDSP_E_ARG;
    const uint64_t smask = (1ULL << D) - 1;
    uint64_t s = *state & smask;
    if (!s) return ATKDSP_E_ARG;
    for (size_t t = 0; t < n; ++t) {
        out[t] = (uint8_t)((b[t] & 1u) ^ (unsigned)(s & 1u));
        const uint64_t fb = (uint64_t)atk_parity64(s & low);
        s = (s >> 1) | (fb << (D - 1));
    }
    *state = s;
    return ATKDSP_OK;
}

int atkdsp_descramble_mult(const uint8_t *b, size_t n, uint64_t poly, uint64_t *state,
                           uint8_t *out) {
    int D; uint64_t low;
    if ((!b && n) || (!out && n) || !state || !check_mask(poly, &D, &low)) return ATKDSP_E_ARG;
    const uint64_t smask = (1ULL << D) - 1;
    uint64_t s = *state & smask;
    for (size_t t = 0; t < n; ++t) {
        const uint64_t x = b[t] & 1u;                 /* scrambled bit in */
        out[t] = (uint8_t)(x ^ (uint64_t)atk_parity64(s & low));
        s = (s >> 1) | (x << (D - 1));
    }
    *state = s;
    return ATKDSP_OK;
}

int atkdsp_scramble_mult(const uint8_t *b, size_t n, uint64_t poly, uint64_t *state,
                         uint8_t *out) {
    int D; uint64_t low;
    if ((!b && n) || (!out && n) || !state || !check_mask(poly, &D, &low)) return ATKDSP_E_ARG;
    const uint64_t smask = (1ULL << D) - 1;
    uint64_t s = *state & smask;
    for (size_t t = 0; t < n; ++t) {
        const uint64_t x = (uint64_t)(b[t] & 1u) ^ (uint64_t)atk_parity64(s & low);
        out[t] = (uint8_t)x;
        s = (s >> 1) | (x << (D - 1));
    }
    *state = s;
    return ATKDSP_OK;
}

/* ---- the search -------------------------------------------------------------- */

/* G statistic of w-bit words of the packed stream y (bits 0..nbits), the word
 * value MSB-first (the twin's order), cells summed in ascending order. */
static double gstat_packed(const uint64_t *y, size_t nbits, int w, const uint8_t *rev) {
    uint64_t cnt[256];
    memset(cnt, 0, sizeof cnt);
    const size_t nw = nbits / (size_t)w;
    const uint64_t mask = (1ULL << w) - 1;
    for (size_t j = 0; j < nw; ++j) cnt[rev[atk_bits_get64(y, j * (size_t)w) & mask]]++;
    const int cells = 1 << w;
    const double e = (double)nw / (double)cells;
    double g = 0.0;
    for (int v = 0; v < cells; ++v)
        if (cnt[v]) g += (double)cnt[v] * log((double)cnt[v] / e);
    return 2.0 * g;
}

/* y over t in [t0, n) for mask (degree D, low terms): packed into yw. */
static void filter_packed(const uint64_t *bw, size_t n, size_t t0, int D, uint64_t low,
                          uint64_t *yw) {
    const size_t m = n - t0, nyw = (m + 63) / 64;
    int delay[64], nd = 0;
    for (int i = 0; i < D; ++i) if ((low >> i) & 1u) delay[nd++] = D - i;   /* b[t - (D - i)] */
    for (size_t k = 0; k < nyw; ++k) {
        const size_t t = t0 + 64 * k;
        uint64_t v = atk_bits_get64(bw, t);
        for (int j = 0; j < nd; ++j) v ^= atk_bits_get64(bw, t - (size_t)delay[j]);
        yw[k] = v;
    }
    if (m & 63u) yw[nyw - 1] &= (1ULL << (m & 63u)) - 1;
    yw[nyw] = 0;
}

typedef struct { uint64_t poly; double g; int D, terms; } scand;

static int scand_cmp(const void *x, const void *y) {
    const scand *a = (const scand *)x, *b = (const scand *)y;
    if (a->g != b->g) return a->g > b->g ? -1 : 1;
    if (a->D != b->D) return a->D < b->D ? -1 : 1;
    if (a->terms != b->terms) return a->terms < b->terms ? -1 : 1;
    return a->poly < b->poly ? -1 : (a->poly > b->poly);
}

/* Enumerate the candidate masks in the documented order. Returns the count;
 * fills out when non-NULL. */
static size_t enum_masks(int maxD, int max_terms, scand *out) {
    size_t c = 0;
    for (int D = 2; D <= maxD; ++D) {
        const uint64_t ends = (1ULL << D) | 1ULL;
        for (int a = 1; a < D; ++a) {                     /* trinomials */
            if (out) { out[c].poly = ends | (1ULL << a); out[c].D = D; out[c].terms = 3; }
            ++c;
        }
        if (max_terms >= 5)
            for (int a = 1; a < D; ++a)
                for (int bb = a + 1; bb < D; ++bb)
                    for (int cc = bb + 1; cc < D; ++cc) {
                        if (out) {
                            out[c].poly = ends | (1ULL << a) | (1ULL << bb) | (1ULL << cc);
                            out[c].D = D; out[c].terms = 5;
                        }
                        ++c;
                    }
    }
    return c;
}

ptrdiff_t atkdsp_scrambler_search(const uint8_t *b, size_t n, int kind, int max_degree,
                                  int max_terms, double pfa, double *g_identity,
                                  atkdsp_scramcand *out, size_t cap) {
    if ((!b && n) || (kind != ATKDSP_SCRAM_ADD && kind != ATKDSP_SCRAM_MULT)
        || max_degree < 2 || max_degree > 32 || (max_terms != 3 && max_terms != 5)
        || (!out && cap)) return ATKDSP_E_ARG;
    if (!(pfa > 0.0)) pfa = 1e-3;
    const size_t t0 = (size_t)max_degree;
    if (g_identity) *g_identity = NAN;
    if (n <= t0) return 0;
    const size_t m = n - t0;
    const int w = atk_chi_width(m);
    if (!w) return 0;
    uint8_t rev[256];
    for (int v = 0; v < (1 << w); ++v) {
        int r = 0;
        for (int k = 0; k < w; ++k) r |= ((v >> k) & 1) << (w - 1 - k);
        rev[v] = (uint8_t)r;
    }
    size_t nbw;
    uint64_t *bw = atk_bits_pack(b, n, &nbw);
    if (!bw) return ATKDSP_E_NOMEM;
    const size_t ncand = enum_masks(max_degree, max_terms, NULL);
    scand *cs = (scand *)malloc(ncand * sizeof *cs);
    if (!cs) { free(bw); return ATKDSP_E_NOMEM; }
    enum_masks(max_degree, max_terms, cs);
    const size_t nyw = (m + 63) / 64 + 1;

    /* doing nothing: the same statistic on b over the same bits */
    double g_id;
    {
        uint64_t *yw = (uint64_t *)calloc(nyw + 1, sizeof *yw);
        if (!yw) { free(bw); free(cs); return ATKDSP_E_NOMEM; }
        for (size_t k = 0; k + 1 < nyw; ++k) yw[k] = atk_bits_get64(bw, t0 + 64 * k);
        if (m & 63u) yw[(m + 63) / 64 - 1] &= (1ULL << (m & 63u)) - 1;
        g_id = gstat_packed(yw, m, w, rev);
        free(yw);
    }
    if (g_identity) *g_identity = g_id;

    int err = 0;
    const long long nc = (long long)ncand;
    long long i;               /* outside the for: MSVC's OpenMP 2.0 (C3015) */
#pragma omp parallel num_threads(atkdsp_get_threads())
    {
        uint64_t *yw = (uint64_t *)calloc(nyw + 1, sizeof *yw);
        if (!yw) {
#pragma omp critical
            err = 1;
        }
#pragma omp for schedule(dynamic, 64)
        for (i = 0; i < nc; ++i) {
            if (!yw) { cs[i].g = -1.0; continue; }
            const int D = cs[i].D;
            filter_packed(bw, n, t0, D, cs[i].poly & ((1ULL << D) - 1), yw);
            cs[i].g = gstat_packed(yw, m, w, rev);
        }
        free(yw);
    }
    if (err) { free(bw); free(cs); return ATKDSP_E_NOMEM; }

    qsort(cs, ncand, sizeof *cs, scand_cmp);
    const double dof = (double)((1 << w) - 1);
    const double trials = (double)ncand + 1.0;
    const double log10_thr = log10(pfa);
    const double margin = atk_norm_isf(pfa / trials) * sqrt(4.0 * dof);
    size_t nout = 0;
    uint8_t *tmp = NULL;
    for (size_t i = 0; i < ncand && nout < cap; ++i) {
        const double g = cs[i].g;
        const double l10 = atk_log_igamc(dof / 2.0, g / 2.0) / LN10 + log10(trials);
        if (l10 > log10_thr || !(g - g_id > margin)) break;   /* sorted: nothing later passes */
        atkdsp_scramcand *c = &out[nout++];
        memset(c, 0, sizeof *c);
        c->poly = cs[i].poly; c->degree = cs[i].D; c->terms = cs[i].terms;
        c->g = g; c->log10_pfa = l10; c->block_bits = w; c->g_data = NAN;
        if (kind != ATKDSP_SCRAM_ADD) continue;
        /* seed: the longest constant run of y over [D, n), 2D bits or more */
        const int D = cs[i].D;
        const uint64_t low = cs[i].poly & ((1ULL << D) - 1);
        size_t best_start = 0, best_len = 0, run_start = (size_t)D, run_len = 0;
        int run_val = -1, best_val = 0;
        for (size_t t = (size_t)D; t < n; ++t) {
            int yv = b[t] & 1;
            for (int k = 0; k < D; ++k) if ((low >> k) & 1u) yv ^= b[t - (size_t)D + (size_t)k] & 1;
            if (yv == run_val) ++run_len;
            else { run_val = yv; run_start = t; run_len = 1; }
            if (run_len > best_len) { best_len = run_len; best_start = run_start; best_val = yv; }
        }
        if (best_len < 2 * (size_t)D) continue;
        /* the data was best_val over [best_start - D, best_start + best_len):
         * the register there is b XOR best_val; take its state at best_start - D */
        const size_t ts = best_start - (size_t)D;
        uint64_t s = 0;
        for (int k = 0; k < D; ++k)
            s |= (uint64_t)((b[ts + (size_t)k] & 1u) ^ (unsigned)best_val) << k;
        const uint64_t smask = (1ULL << D) - 1;
        const uint64_t lowsh = low >> 1;
        for (size_t t = ts; t > 0; --t) {            /* step back to bit 0 */
            const uint64_t prev = ((s >> (D - 1)) & 1u) ^ (uint64_t)atk_parity64(s & lowsh);
            s = ((s << 1) | prev) & smask;
        }
        if (!s) continue;
        if (!tmp) { tmp = (uint8_t *)malloc(n); if (!tmp) continue; }
        uint64_t st = s;
        atkdsp_descramble_add(b, n, cs[i].poly, &st, tmp);
        c->seed = s; c->seeded = 1;
        c->g_data = atk_gstat(tmp + t0, m, w, NULL);
    }
    free(tmp); free(bw); free(cs);
    return (ptrdiff_t)nout;
}
