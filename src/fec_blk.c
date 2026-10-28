/* Small binary linear block codes by syndrome table (header §16d).
 *
 * Words are uint32 with bit j = position j (position 0 sent first). The
 * generator is row-reduced (tracking the row operations, so information bits
 * can be read back), its null space gives the parity-check matrix, and the
 * coset leaders are found by enumerating error patterns in order of weight:
 * the first pattern to reach a syndrome is its leader; a second pattern of
 * the SAME weight reaching it marks the syndrome ambiguous (the word is
 * equidistant from two codewords, and decode says so). dmin is the weight of
 * the lightest nonzero codeword: by enumerating codewords when k <= 20,
 * otherwise as the first nonzero pattern with a zero syndrome. */
#include "internal.h"
#include <stdio.h>

#define ATK_BLK_MAX_PATTERNS (1u << 25)

struct atkdsp_blk {
    int n, k, r, dmin;
    uint32_t G[32];              /* rows as given */
    uint32_t R[32];              /* reduced rows, identity on the pivot columns */
    uint32_t M[32];              /* R[i] = XOR of G rows in M[i] (bit = row) */
    int      piv[32];
    uint32_t H[20];              /* parity checks */
    uint32_t *leader;            /* [2^r]: leader pattern, or ~0u = ambiguous */
    uint8_t  *wt;                /* [2^r]: leader weight, 255 = unset */
    atkdsp_fec_stats st;
};

ATK_INLINE uint32_t syndrome(const atkdsp_blk *c, uint32_t v) {
    uint32_t s = 0;
    for (int i = 0; i < c->r; ++i) s |= (uint32_t)atk_parity64((uint64_t)(v & c->H[i])) << i;
    return s;
}

/* next combination of n bits with the same popcount (Gosper) */
ATK_INLINE uint32_t next_comb(uint32_t x) {
    const uint32_t u = x & (0u - x), v = u + x;
    return v + (((v ^ x) / u) >> 2);
}

atkdsp_blk *atkdsp_blk_create(int n, int k, const uint8_t *G) {
    if (n < 2 || n > 32 || k < 1 || k >= n || n - k > 20 || !G) return NULL;
    atkdsp_blk *c = (atkdsp_blk *)calloc(1, sizeof *c);
    if (!c) return NULL;
    c->n = n; c->k = k; c->r = n - k;
    for (int i = 0; i < k; ++i) {
        uint32_t row = 0;
        for (int j = 0; j < n; ++j) if (G[i * n + j] & 1u) row |= 1u << j;
        c->G[i] = c->R[i] = row;
        c->M[i] = 1u << i;
    }
    /* reduced row echelon form */
    int rank = 0;
    for (int col = 0; col < n && rank < k; ++col) {
        int p = -1;
        for (int i = rank; i < k; ++i) if ((c->R[i] >> col) & 1u) { p = i; break; }
        if (p < 0) continue;
        uint32_t t = c->R[p]; c->R[p] = c->R[rank]; c->R[rank] = t;
        t = c->M[p]; c->M[p] = c->M[rank]; c->M[rank] = t;
        for (int i = 0; i < k; ++i)
            if (i != rank && ((c->R[i] >> col) & 1u)) { c->R[i] ^= c->R[rank]; c->M[i] ^= c->M[rank]; }
        c->piv[rank++] = col;
    }
    if (rank < k) { free(c); return NULL; }
    /* null space: for each free column f, e_f + sum_i R[i]_f e_piv[i] */
    int hr = 0;
    for (int col = 0; col < n; ++col) {
        int is_piv = 0;
        for (int i = 0; i < k; ++i) if (c->piv[i] == col) { is_piv = 1; break; }
        if (is_piv) continue;
        uint32_t h = 1u << col;
        for (int i = 0; i < k; ++i) if ((c->R[i] >> col) & 1u) h |= 1u << c->piv[i];
        c->H[hr++] = h;
    }
    const size_t ns = (size_t)1 << c->r;
    c->leader = (uint32_t *)malloc(ns * sizeof *c->leader);
    c->wt = (uint8_t *)malloc(ns);
    if (!c->leader || !c->wt) { atkdsp_blk_destroy(c); return NULL; }
    memset(c->wt, 255, ns);
    c->leader[0] = 0; c->wt[0] = 0;
    size_t filled = 1;
    c->dmin = 0;
    if (k <= 20) {                      /* dmin from the codewords themselves (Gray code) */
        uint32_t cw = 0;
        int best = n + 1;
        for (uint32_t i = 1; i < (1u << k); ++i) {
            int bit = 0;
            while (!((i >> bit) & 1u)) ++bit;
            cw ^= c->G[bit];
            const int w = atk_popcount64(cw);
            if (w < best) best = w;
        }
        c->dmin = best;
    }
    const uint32_t full = (n == 32) ? 0xffffffffu : ((1u << n) - 1u);
    /* bounded work: a weight level that would take the total past 2^25
     * patterns is not started (a code like [I | 0] would otherwise need
     * billions). Syndromes it would have reached decode as uncorrectable;
     * for k > 20 an unfound dmin stays 0 (unknown). */
    double done = 0.0, comb = 1.0;            /* comb = C(n, w) */
    for (int w = 1; w <= n && (filled < ns || (c->dmin == 0)); ++w) {
        comb = comb * (double)(n - w + 1) / (double)w;
        if (done + comb > (double)ATK_BLK_MAX_PATTERNS) break;
        done += comb;
        uint32_t x = (w == 32) ? 0xffffffffu : ((1u << w) - 1u);
        while (1) {
            const uint32_t s = syndrome(c, x);
            if (s == 0) { if (!c->dmin) c->dmin = w; }
            else if (c->wt[s] == 255) { c->wt[s] = (uint8_t)w; c->leader[s] = x; ++filled; }
            else if (c->wt[s] == w && c->leader[s] != x) c->leader[s] = ~0u;
            if (x == (full & ~((1u << (n - w)) - 1u)) || w == n) break;   /* the last combination */
            x = next_comb(x);
            if (x > full) break;
        }
    }
    return c;
}

void atkdsp_blk_destroy(atkdsp_blk *c) {
    if (!c) return;
    free(c->leader); free(c->wt);
    free(c);
}

int atkdsp_blk_params(const atkdsp_blk *c, int *n, int *k, int *dmin) {
    if (!c) return ATKDSP_E_ARG;
    if (n) *n = c->n;
    if (k) *k = c->k;
    if (dmin) *dmin = c->dmin;
    return ATKDSP_OK;
}

int atkdsp_blk_encode(const atkdsp_blk *c, const uint8_t *data, uint8_t *code) {
    if (!c || !data || !code) return ATKDSP_E_ARG;
    uint32_t v = 0;
    for (int i = 0; i < c->k; ++i) if (data[i] & 1u) v ^= c->G[i];
    for (int j = 0; j < c->n; ++j) code[j] = (uint8_t)((v >> j) & 1u);
    return ATKDSP_OK;
}

int atkdsp_blk_decode(atkdsp_blk *c, uint8_t *code, uint8_t *data) {
    if (!c || !code) return ATKDSP_E_ARG;
    c->st.calls++; c->st.in++;
    uint32_t v = 0;
    for (int j = 0; j < c->n; ++j) if (code[j] & 1u) v |= 1u << j;
    const uint32_t s = syndrome(c, v);
    if (c->wt[s] == 255 || c->leader[s] == ~0u) { c->st.failures++; return ATKDSP_E_DECODE; }
    const uint32_t e = c->leader[s];
    v ^= e;
    for (int j = 0; j < c->n; ++j) code[j] = (uint8_t)((v >> j) & 1u);
    if (data) {
        /* v = a . R with a = v on the pivot columns; data = a . M */
        uint32_t d = 0;
        for (int i = 0; i < c->k; ++i) if ((v >> c->piv[i]) & 1u) d ^= c->M[i];
        for (int i = 0; i < c->k; ++i) data[i] = (uint8_t)((d >> i) & 1u);
    }
    const int w = atk_popcount64(e);
    c->st.out++; c->st.corrected += (uint64_t)w;
    return w;
}

int atkdsp_blk_describe(const atkdsp_blk *c, char *buf, size_t cap) {
    if (!c || !buf || !cap) return ATKDSP_E_ARG;
    snprintf(buf, cap, "block code (%d,%d) dmin %d, corrects %d, syndrome table of %u",
             c->n, c->k, c->dmin, c->dmin > 0 ? (c->dmin - 1) / 2 : 0, 1u << c->r);
    return ATKDSP_OK;
}

int atkdsp_blk_stats(const atkdsp_blk *c, atkdsp_fec_stats *st) {
    if (!c || !st) return ATKDSP_E_ARG;
    *st = c->st;
    return ATKDSP_OK;
}
