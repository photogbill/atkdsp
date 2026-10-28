/* GF(2) linear algebra for naming the code (header §18): packed Gaussian
 * elimination, null spaces in a canonical form, the deterministic random
 * generator the robust search draws subsets with, the robust search itself,
 * and atkdsp_gf2_rank_scan.
 *
 * A matrix is R rows of nw = ceil(w/64) words; bit j of a row (word j/64,
 * bit j%64) is column j. reference.py §18 does the same with Python integers
 * (bit j = column j), the same pivot order and the same random draws. */
#include "internal.h"

#ifdef _OPENMP
#  include <omp.h>
#endif

#define LN2 0.693147180559945309417

/* ---- the random generator: xorshift64* (Vigna), the twin's too ---------- */
void atk_rng_seed(uint64_t *s, uint64_t seed) {
    *s = seed ? seed : 0x9E3779B97F4A7C15ULL;
}
uint64_t atk_rng(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

ATK_INLINE int topbit_w(const uint64_t *v, int nw) {
    for (int q = nw - 1; q >= 0; --q)
        if (v[q]) { int b = 63; while (!((v[q] >> b) & 1u)) --b; return 64 * q + b; }
    return -1;
}
ATK_INLINE int getbit(const uint64_t *v, int j) { return (int)((v[j >> 6] >> (j & 63)) & 1u); }
ATK_INLINE void xorv(uint64_t *a, const uint64_t *b, int nw) { for (int q = 0; q < nw; ++q) a[q] ^= b[q]; }

/* Reduce m vectors (in place) to canonical form: reduced row echelon, each
 * pivot the vector's highest set bit, other vectors zero in pivot columns,
 * highest pivot first. Zero vectors are dropped. Returns the rank. */
int atk_gf2_canon(uint64_t *v, int m, int w) {
    const int nw = (w + 63) / 64;
    int r = 0;
    for (int col = w - 1; col >= 0 && r < m; --col) {
        /* rows r.. are zero above col (each higher column was either a pivot,
         * cleared, or empty in them), so the pivot row ends at col and an
         * XOR touches only words 0 .. col/64 */
        const int qn = (col >> 6) + 1;
        const uint64_t bit = 1ULL << (col & 63);
        int p = -1;
        for (int i = r; i < m; ++i) if (v[(size_t)i * nw + (col >> 6)] & bit) { p = i; break; }
        if (p < 0) continue;
        if (p != r)
            for (int q = 0; q < qn; ++q) {
                const uint64_t t = v[(size_t)p * nw + q];
                v[(size_t)p * nw + q] = v[(size_t)r * nw + q];
                v[(size_t)r * nw + q] = t;
            }
        const uint64_t *pr = v + (size_t)r * nw;
        for (int i = 0; i < m; ++i) {
            uint64_t *ri = v + (size_t)i * nw;
            if (i != r && (ri[col >> 6] & bit)) for (int q = 0; q < qn; ++q) ri[q] ^= pr[q];
        }
        ++r;
    }
    return r;
}

/* Rank of R rows (destroys them). */
int atk_gf2_rank(uint64_t *M, int R, int w) {
    const int nw = (w + 63) / 64;
    int r = 0;
    for (int col = w - 1; col >= 0 && r < R; --col) {
        const int qn = (col >> 6) + 1;
        const uint64_t bit = 1ULL << (col & 63);
        int p = -1;
        for (int i = r; i < R; ++i) if (M[(size_t)i * nw + (col >> 6)] & bit) { p = i; break; }
        if (p < 0) continue;
        if (p != r)
            for (int q = 0; q < qn; ++q) {
                const uint64_t t = M[(size_t)p * nw + q];
                M[(size_t)p * nw + q] = M[(size_t)r * nw + q];
                M[(size_t)r * nw + q] = t;
            }
        const uint64_t *pr = M + (size_t)r * nw;
        for (int i = r + 1; i < R; ++i) {
            uint64_t *ri = M + (size_t)i * nw;
            if (ri[col >> 6] & bit) for (int q = 0; q < qn; ++q) ri[q] ^= pr[q];
        }
        ++r;
    }
    return r;
}

/* The null space of R rows: every h with parity(row & h) = 0 for all rows,
 * in canonical form, into basis (cap vectors of nw words). Returns the
 * dimension (capped at cap), or -1 when out of memory. Forward elimination
 * first, so a full-rank matrix (the usual random subset) stops there; then
 * back-substitution to reduced form, and the free columns read off. */
int atk_gf2_nullspace(const uint64_t *rows, int R, int w, uint64_t *basis, int cap) {
    const int nw = (w + 63) / 64;
    uint64_t *M = (uint64_t *)malloc(((size_t)R + 1) * (size_t)nw * sizeof *M);
    int *piv = (int *)malloc(((size_t)w + 1) * sizeof *piv);
    uint8_t *ispiv = (uint8_t *)calloc((size_t)w + 1, 1);
    if (!M || !piv || !ispiv) { free(M); free(piv); free(ispiv); return -1; }
    if (R) memcpy(M, rows, (size_t)R * (size_t)nw * sizeof *M);
    int r = 0;
    for (int col = w - 1; col >= 0 && r < R; --col) {
        const int qn = (col >> 6) + 1;
        const uint64_t bit = 1ULL << (col & 63);
        int p = -1;
        for (int i = r; i < R; ++i) if (M[(size_t)i * nw + (col >> 6)] & bit) { p = i; break; }
        if (p < 0) continue;
        if (p != r)
            for (int q = 0; q < qn; ++q) {
                const uint64_t t = M[(size_t)p * nw + q];
                M[(size_t)p * nw + q] = M[(size_t)r * nw + q];
                M[(size_t)r * nw + q] = t;
            }
        const uint64_t *pr = M + (size_t)r * nw;
        for (int i = r + 1; i < R; ++i) {
            uint64_t *ri = M + (size_t)i * nw;
            if (ri[col >> 6] & bit) for (int q = 0; q < qn; ++q) ri[q] ^= pr[q];
        }
        piv[r] = col;
        ispiv[col] = 1;
        ++r;
    }
    int d = 0;
    if (r < w) {
        /* back-substitution: clear each pivot column above its row (row i
         * ends at its pivot, so words 0 .. piv/64) */
        for (int i = r - 1; i >= 0; --i) {
            const int col = piv[i], qn = (col >> 6) + 1;
            const uint64_t bit = 1ULL << (col & 63);
            const uint64_t *pr = M + (size_t)i * nw;
            for (int j = 0; j < i; ++j) {
                uint64_t *rj = M + (size_t)j * nw;
                if (rj[col >> 6] & bit) for (int q = 0; q < qn; ++q) rj[q] ^= pr[q];
            }
        }
        for (int f = w - 1; f >= 0 && d < cap; --f) {      /* free columns, high to low */
            if (ispiv[f]) continue;
            uint64_t *v = basis + (size_t)d * nw;
            memset(v, 0, (size_t)nw * sizeof *v);
            v[f >> 6] |= 1ULL << (f & 63);
            for (int i = 0; i < r; ++i)
                if ((M[(size_t)i * nw + (f >> 6)] >> (f & 63)) & 1u) v[piv[i] >> 6] |= 1ULL << (piv[i] & 63);
            ++d;
        }
    }
    free(M); free(piv); free(ispiv);
    if (d) atk_gf2_canon(basis, d, w);
    return d;
}

/* Rows of the matrix whose check h fails (parity 1). */
int atk_gf2_unsat(const uint64_t *rows, int R, int nw, const uint64_t *h) {
    int u = 0;
    for (int i = 0; i < R; ++i) {
        uint64_t acc = 0;
        const uint64_t *r = rows + (size_t)i * nw;
        for (int q = 0; q < nw; ++q) acc ^= r[q] & h[q];
        u += atk_parity64(acc);
    }
    return u;
}

static int popcount_w(const uint64_t *v, int nw) {
    int c = 0;
    for (int q = 0; q < nw; ++q) c += atk_popcount64(v[q]);
    return c;
}

/* Is v in the span of the canonical set s[0..m)? (reduces a copy) */
static int in_span(const uint64_t *s, int m, const uint64_t *v, int nw, uint64_t *tmp) {
    memcpy(tmp, v, (size_t)nw * sizeof *tmp);
    for (int i = 0; i < m; ++i) {
        const int p = topbit_w(s + (size_t)i * nw, nw);
        if (getbit(tmp, p)) xorv(tmp, s + (size_t)i * nw, nw);
    }
    for (int q = 0; q < nw; ++q) if (tmp[q]) return 0;
    return 1;
}

static int dbl_cmp(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y);
}

/* Rows idx[0..V) whose check h fails. */
static int unsat_idx(const uint64_t *rows, const int *idx, int V, int nw, const uint64_t *h) {
    int u = 0;
    for (int i = 0; i < V; ++i) {
        uint64_t acc = 0;
        const uint64_t *r = rows + (size_t)idx[i] * nw;
        for (int q = 0; q < nw; ++q) acc ^= r[q] & h[q];
        u += atk_parity64(acc);
    }
    return u;
}

/* Does check h hold on rows in a pattern by row index? log10 of the smallest
 * G-test p-value, over row periods 2..32, of "h holds" across the residues
 * of the row index (times the periods tried), over all R rows (the subset's
 * own rows hold by construction, but they are a random draw, so they add no
 * pattern). Channel noise breaks a true check on random rows; a width
 * that is not a multiple of the block length cuts codewords at an alignment
 * that cycles from row to row, so a relation that fits one phase of that
 * cycle holds on the rows of that phase only. */
static double period_l10(const uint64_t *rows, int R, int nw, const uint64_t *h) {
    const int cmax = R / 16 < 32 ? R / 16 : 32;
    if (cmax < 2) return 0.0;
    int nn[31][32], kk[31][32], ph[31];
    memset(nn, 0, sizeof nn);
    memset(kk, 0, sizeof kk);
    memset(ph, 0, sizeof ph);
    int K = 0;
    for (int i = 0; i < R; ++i) {
        uint64_t acc = 0;
        const uint64_t *r = rows + (size_t)i * nw;
        for (int q = 0; q < nw; ++q) acc ^= r[q] & h[q];
        const int ok = !atk_parity64(acc);
        K += ok;
        for (int c = 0; c < cmax - 1; ++c) {          /* period c + 2, phase i mod (c + 2) */
            ++nn[c][ph[c]]; kk[c][ph[c]] += ok;
            if (++ph[c] == c + 2) ph[c] = 0;
        }
    }
    if (K == 0 || K == R) return 0.0;
    const double pr = (double)K / (double)R;
    double best = 0.0;
    for (int c = 2; c <= cmax; ++c) {
        double G = 0.0;
        for (int j = 0; j < c; ++j) {
            const double a = kk[c - 2][j], b = nn[c - 2][j] - kk[c - 2][j];
            if (a > 0) G += a * log(a / (nn[c - 2][j] * pr));
            if (b > 0) G += b * log(b / (nn[c - 2][j] * (1.0 - pr)));
        }
        /* G above is half the G statistic: p = Q(dof / 2, G_stat / 2) */
        const double l10 = atk_log_igamc(0.5 * (c - 1), G) / log(10.0);
        if (l10 < best) best = l10;
    }
    return best + log10((double)(cmax - 1));
}

/* The same test on a row indicator ok[i] (1 = row i holds every check):
 * periods 2..pmax, at most R / 16. Used once per code found, so periods to 64
 * (a wrong length's alignment cycle can be longer than 32 rows). */
double atk_ok_period_l10(const uint8_t *ok, int R, int pmax) {
    int cmax = R / 16 < pmax ? R / 16 : pmax;
    if (cmax > 64) cmax = 64;
    if (cmax < 2) return 0.0;
    int K = 0;
    for (int i = 0; i < R; ++i) K += ok[i] & 1;
    if (K == 0 || K == R) return 0.0;
    const double pr = (double)K / (double)R;
    double best = 0.0;
    int nn[64], kk[64];
    for (int c = 2; c <= cmax; ++c) {
        memset(nn, 0, sizeof nn);
        memset(kk, 0, sizeof kk);
        for (int i = 0, j = 0; i < R; ++i) {
            ++nn[j]; kk[j] += ok[i] & 1;
            if (++j == c) j = 0;
        }
        double G = 0.0;
        for (int j = 0; j < c; ++j) {
            const double a = kk[j], b = nn[j] - kk[j];
            if (a > 0) G += a * log(a / (nn[j] * pr));
            if (b > 0) G += b * log(b / (nn[j] * (1.0 - pr)));
        }
        const double l10 = atk_log_igamc(0.5 * (c - 1), G) / log(10.0);
        if (l10 < best) best = l10;
    }
    return best + log10((double)(cmax - 1));
}

/* The robust null space (header §18): random subsets of S = w + 2 rows,
 * each subset's canonical null space, each vector kept if it holds on enough
 * of the OTHER R - S rows (the subset's own rows satisfy it by construction,
 * so they carry no evidence). Stops after ATK_ROBUST_TMAX trials, after 16
 * with nothing, or after 8 without growth. Writes the canonical span of the
 * kept checks. info: the weakest kept check's log10_pfa, the median BER
 * estimate, trials, completeness. Needs R >= S + ATK_ROBUST_VMIN. */
int atk_gf2_robust(const uint64_t *rows, int R, int w, uint64_t seed, double pfa,
                   uint64_t *basis, int cap, atk_robust_info *info) {
    return atk_gf2_robust_q(rows, R, w, seed, pfa, basis, cap, info, 16);
}

/* The same, giving up after `quit` trials that find nothing (a detection
 * pass: a code the method can see at all shows in the first few). */
int atk_gf2_robust_q(const uint64_t *rows, int R, int w, uint64_t seed, double pfa,
                     uint64_t *basis, int cap, atk_robust_info *info, int quit) {
    const int nw = (w + 63) / 64;
    const int S = w + 2;
    if (info) { info->log10_pfa = 0.0; info->ber = NAN; info->trials = 0; info->complete = 0; }
    if (R < S + ATK_ROBUST_VMIN || cap < 1) return 0;
    const int V = R - S;                          /* validation rows per trial */
    const double log10_thr = log10(pfa > 0.0 ? pfa : 1e-3);
    const double log10_trials = log10((double)ATK_ROBUST_TMAX * (double)w);
    int *idx = (int *)malloc((size_t)R * sizeof *idx);
    uint64_t *sub = (uint64_t *)malloc((size_t)S * (size_t)nw * sizeof *sub);
    uint64_t *nb = (uint64_t *)malloc((size_t)w * (size_t)nw * sizeof *nb);
    uint64_t *tmp = (uint64_t *)malloc((size_t)nw * sizeof *tmp);
    double *pest = (double *)malloc((size_t)(ATK_ROBUST_TMAX * w + 1) * sizeof *pest);
    if (!idx || !sub || !nb || !tmp || !pest) {
        free(idx); free(sub); free(nb); free(tmp); free(pest);
        return -1;
    }
    uint64_t st;
    atk_rng_seed(&st, seed);
    int m = 0, np = 0, still = 0, t, best_full = 0, prod = 0;
    double weakest = -INFINITY;
    for (t = 0; t < ATK_ROBUST_TMAX; ++t) {
        if (m == 0 && t >= quit) break;
        if (m > 0 && still >= 8) break;
        for (int i = 0; i < R; ++i) idx[i] = i;
        for (int i = 0; i < S; ++i) {             /* partial Fisher-Yates */
            const int j = i + (int)(atk_rng(&st) % (uint64_t)(R - i));
            const int x = idx[i]; idx[i] = idx[j]; idx[j] = x;
            memcpy(sub + (size_t)i * nw, rows + (size_t)idx[i] * nw, (size_t)nw * sizeof *sub);
        }
        const int d = atk_gf2_nullspace(sub, S, w, nb, w);
        if (d < 0) { m = -1; break; }
        int grew = 0, kept = 0;
        for (int k = 0; k < d; ++k) {
            const uint64_t *h = nb + (size_t)k * nw;
            const int u = unsat_idx(rows, idx + S, V, nw, h);
            const double l10 = atk_log10_binom_tail((double)V, (double)(V - u), -LN2) + log10_trials;
            if (l10 > log10_thr) continue;
            /* and it must fail no more often than channel noise could make a
             * true check fail (BER <= ATK_ROBUST_PMAX): a relation that holds
             * on 70 % of rows because a short check sometimes falls inside one
             * codeword is structure, but not a parity check at this width */
            const int wt = popcount_w(h, nw);
            const double fmax = 0.5 * (1.0 - pow(1.0 - 2.0 * ATK_ROBUST_PMAX, (double)wt));
            if ((double)u > (double)V * fmax + 3.0 * sqrt((double)V * fmax * (1.0 - fmax))) continue;
            const int fresh = m < cap && !in_span(basis, m, h, nw, tmp);
            /* a new check must not hold in a row-index pattern (a check
             * already in the span was tested when its span was built) */
            if (fresh && period_l10(rows, R, nw, h) <= ATK_PERIODIC_L10) continue;
            ++kept;
            const double f = (double)u / (double)V;
            pest[np++] = (f < 0.5) ? 0.5 * (1.0 - pow(1.0 - 2.0 * f, 1.0 / (double)wt)) : 0.5;
            if (l10 > weakest) weakest = l10;
            if (fresh) {
                memcpy(basis + (size_t)m * nw, h, (size_t)nw * sizeof *basis);
                ++m;
                m = atk_gf2_canon(basis, m, w);
                grew = 1;
            }
        }
        if (d > 0 && kept == d && d > best_full) best_full = d;
        still = grew ? 0 : still + 1;
        prod = grew ? 0 : prod + (kept > 0);
    }
    if (info) {
        info->trials = t;
        info->log10_pfa = m > 0 ? weakest : 0.0;
        if (np > 0) { qsort(pest, (size_t)np, sizeof *pest, dbl_cmp); info->ber = pest[np / 2]; }
        /* complete: a clean subset gave the whole span, or the span stopped
         * growing while 8 trials in a row each found checks. A trial with b
         * bad rows yields the true checks orthogonal to their b error
         * patterns, a random subspace of the dual; while the span misses
         * part of the dual, such a subspace falls inside it with
         * probability <= 1/2, so 8 productive trials that add nothing
         * leave it short with probability <= 2^-8 */
        info->complete = m > 0 && (best_full == m || prod >= 8);
    }
    free(idx); free(sub); free(nb); free(tmp); free(pest);
    return m;
}

/* Rows of width w from bit `off`, R of them, packed. */
void atk_gf2_rows(const uint64_t *bw, size_t off, int w, int R, int stride, uint64_t *M) {
    const int nw = (w + 63) / 64;
    for (int r = 0; r < R; ++r) {
        uint64_t *row = M + (size_t)r * nw;
        const size_t s = off + (size_t)r * (size_t)stride;
        for (int q = 0; q < nw; ++q) row[q] = atk_bits_get64(bw, s + 64 * (size_t)q);
        if (w & 63) row[nw - 1] &= (1ULL << (w & 63)) - 1;
    }
}

int atkdsp_gf2_rank_scan(const uint8_t *b, size_t n, size_t w_lo, size_t w_hi,
                         size_t n_offsets, size_t rows, int16_t *def) {
    if ((!b && n) || !def || w_lo < 2 || w_hi < w_lo || w_hi > 4096 || n_offsets < 1
        || n_offsets > 4096 || n > (size_t)INT32_MAX) return ATKDSP_E_ARG;
    /* rows per matrix: `rows` (0 = w + 32), never more than the data holds,
     * so the buffer is sized by what can be used, not by what was asked */
    size_t rmax = rows ? rows : w_hi + 32;
    if (rmax > n / w_lo) rmax = n / w_lo;
    const size_t nwh = (w_hi + 63) / 64;
    if (rmax > (SIZE_MAX / sizeof(uint64_t)) / nwh - 1) return ATKDSP_E_ARG;
    size_t nbw;
    uint64_t *bw = atk_bits_pack(b, n, &nbw);
    if (!bw) return ATKDSP_E_NOMEM;
    const long long nwid = (long long)(w_hi - w_lo + 1);
    int err = 0;
    long long iw;
#pragma omp parallel num_threads(atkdsp_get_threads())
    {
        uint64_t *M = (uint64_t *)malloc((rmax + 1) * nwh * sizeof *M);
        if (!M) {
#pragma omp critical
            err = 1;
        }
#pragma omp for schedule(dynamic, 1)
        for (iw = 0; iw < nwid; ++iw) {
            const size_t w = w_lo + (size_t)iw;
            for (size_t o = 0; o < n_offsets; ++o) {
                int16_t *cell = def + (size_t)iw * n_offsets + o;
                *cell = -1;
                if (!M || o >= w || o >= n) continue;
                size_t R = rows ? rows : w + 32;
                const size_t fit = (n - o) / w;
                if (R > fit) R = fit;
                if (R > rmax) R = rmax;
                if (R < w + 16) continue;
                atk_gf2_rows(bw, o, (int)w, (int)R, (int)w, M);
                *cell = (int16_t)((int)w - atk_gf2_rank(M, (int)R, (int)w));
            }
        }
        free(M);
    }
    free(bw);
    return err ? ATKDSP_E_NOMEM : ATKDSP_OK;
}
