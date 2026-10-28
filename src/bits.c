/* Bit primitives (header §14): correlation, search, the raster's sums, the
 * width scan, sync discovery, bit statistics, Walsh-Hadamard relations, and
 * the analyzer-strip transforms.
 *
 * Bits come in one per byte (bit 0 read) and are packed to 64-bit words here,
 * so a correlation is a popcount: 64 comparisons per instruction. Every result
 * is integer arithmetic until the final division, so the numpy twins in
 * reference.py §14 reproduce it exactly (bits_stats' p-values and the
 * transforms' spectra to float precision). */
#include "internal.h"
#include "pocketfft.h"

#ifdef _OPENMP
#  include <omp.h>
#endif

#define LN2 0.693147180559945309417

uint64_t *atk_bits_pack(const uint8_t *b, size_t n, size_t *nw_out) {
    const size_t nw = (n + 63) / 64;
    uint64_t *w = (uint64_t *)calloc(nw + 2, sizeof *w);
    if (!w) return NULL;
    for (size_t i = 0; i < n; ++i)
        if (b[i] & 1u) w[i >> 6] |= 1ULL << (i & 63u);
    if (nw_out) *nw_out = nw;
    return w;
}

/* Disagreements between stream bits [p, p+m) and [q, q+m). */
static uint64_t mismatches(const uint64_t *w, size_t p, size_t q, size_t m) {
    uint64_t d = 0;
    size_t i = 0;
    for (; i + 64 <= m; i += 64)
        d += (uint64_t)atk_popcount64(atk_bits_get64(w, p + i) ^ atk_bits_get64(w, q + i));
    if (i < m) {
        const uint64_t mask = (1ULL << (m - i)) - 1;
        d += (uint64_t)atk_popcount64((atk_bits_get64(w, p + i) ^ atk_bits_get64(w, q + i)) & mask);
    }
    return d;
}

/* Disagreements between packed pattern words pw (m bits) and stream bits at p. */
static uint64_t mismatches_pat(const uint64_t *w, size_t p, const uint64_t *pw, size_t m) {
    uint64_t d = 0;
    size_t i = 0, j = 0;
    for (; i + 64 <= m; i += 64, ++j)
        d += (uint64_t)atk_popcount64(atk_bits_get64(w, p + i) ^ pw[j]);
    if (i < m) {
        const uint64_t mask = (1ULL << (m - i)) - 1;
        d += (uint64_t)atk_popcount64((atk_bits_get64(w, p + i) ^ pw[j]) & mask);
    }
    return d;
}

int atkdsp_bits_autocorr(const uint8_t *b, size_t n, size_t max_lag, float *out) {
    if ((!b && n) || !out) return ATKDSP_E_ARG;
    uint64_t *w = atk_bits_pack(b, n, NULL);
    if (!w) return ATKDSP_E_NOMEM;
    const long long L = (long long)max_lag;
    long long k;               /* outside the for: MSVC's OpenMP 2.0 (C3015) */
#pragma omp parallel for num_threads(atkdsp_get_threads()) schedule(dynamic, 16)
    for (k = 0; k <= L; ++k) {
        if ((size_t)k >= n) { out[k] = NAN; continue; }
        const size_t m = n - (size_t)k;
        const uint64_t d = mismatches(w, 0, (size_t)k, m);
        out[k] = (float)(((double)m - 2.0 * (double)d) / (double)m);
    }
    free(w);
    return ATKDSP_OK;
}

int atkdsp_bits_xcorr(const uint8_t *a, size_t na, const uint8_t *p, size_t nb, float *out) {
    if (!a || !p || !out || nb < 1 || nb > na) return ATKDSP_E_ARG;
    uint64_t *w = atk_bits_pack(a, na, NULL);
    uint64_t *pw = atk_bits_pack(p, nb, NULL);
    if (!w || !pw) { free(w); free(pw); return ATKDSP_E_NOMEM; }
    const long long K = (long long)(na - nb);
    long long k;
#pragma omp parallel for num_threads(atkdsp_get_threads()) schedule(static)
    for (k = 0; k <= K; ++k) {
        const uint64_t d = mismatches_pat(w, (size_t)k, pw, nb);
        out[k] = (float)(((double)nb - 2.0 * (double)d) / (double)nb);
    }
    free(w); free(pw);
    return ATKDSP_OK;
}

ptrdiff_t atkdsp_bits_search(const uint8_t *b, size_t n, const uint8_t *pat, size_t plen,
                             int max_err, unsigned flags, atkdsp_bithit *hits, size_t cap) {
    if ((!b && n) || !pat || plen < 1 || plen > 4096 || max_err < 0) return ATKDSP_E_ARG;
    if (!hits && cap) return ATKDSP_E_ARG;
    const int counting = (hits == NULL);
    if (plen > n) return 0;
    uint8_t *rev = (uint8_t *)malloc(plen);
    uint64_t *w = atk_bits_pack(b, n, NULL);
    uint64_t *pw = atk_bits_pack(pat, plen, NULL);
    uint64_t *rw = NULL;
    if (rev) {
        for (size_t i = 0; i < plen; ++i) rev[i] = pat[plen - 1 - i];
        rw = atk_bits_pack(rev, plen, NULL);
    }
    if (!rev || !w || !pw || !rw) { free(rev); free(w); free(pw); free(rw); return ATKDSP_E_NOMEM; }
    ptrdiff_t count = 0;
    const size_t last = n - plen;
    for (size_t pos = 0; pos <= last; ++pos) {
        const long long d0 = (long long)mismatches_pat(w, pos, pw, plen);
        long long best = d0;
        int var = 0;
        if (flags & ATKDSP_BITS_INVERTED) {
            const long long d1 = (long long)plen - d0;
            if (d1 < best) { best = d1; var = 1; }
        }
        if (flags & ATKDSP_BITS_REVERSED) {
            const long long d2 = (long long)mismatches_pat(w, pos, rw, plen);
            if (d2 < best) { best = d2; var = 2; }
            if (flags & ATKDSP_BITS_INVERTED) {
                const long long d3 = (long long)plen - d2;
                if (d3 < best) { best = d3; var = 3; }
            }
        }
        if (best <= max_err) {
            if (!counting) {
                if ((size_t)count >= cap) { count = ATKDSP_E_CAP; break; }
                hits[count].pos = (long long)pos;
                hits[count].errors = (int)best;
                hits[count].variant = var;
            }
            ++count;
        }
    }
    free(rev); free(w); free(pw); free(rw);
    return count;
}

ptrdiff_t atkdsp_bits_columns(const uint8_t *b, size_t n, size_t width, size_t offset,
                              uint32_t *col_ones, uint32_t *row_ones, uint8_t *col_parity,
                              uint8_t *row_parity) {
    if ((!b && n) || width < 1) return ATKDSP_E_ARG;
    const size_t rows = offset < n ? (n - offset) / width : 0;
    if (col_ones) memset(col_ones, 0, width * sizeof *col_ones);
    uint32_t *cols = col_ones;
    uint32_t *tmp = NULL;
    if (!cols && col_parity) {
        tmp = (uint32_t *)calloc(width, sizeof *tmp);
        if (!tmp) return ATKDSP_E_NOMEM;
        cols = tmp;
    }
    for (size_t r = 0; r < rows; ++r) {
        const uint8_t *row = b + offset + r * width;
        uint32_t s = 0;
        for (size_t c = 0; c < width; ++c) {
            const uint32_t v = row[c] & 1u;
            s += v;
            if (cols) cols[c] += v;
        }
        if (row_ones) row_ones[r] = s;
        if (row_parity) row_parity[r] = (uint8_t)(s & 1u);
    }
    if (col_parity)
        for (size_t c = 0; c < width; ++c) col_parity[c] = (uint8_t)(cols[c] & 1u);
    free(tmp);
    return (ptrdiff_t)rows;
}

/* E|2K/R - 1| for K ~ Binomial(R, 1/2): C(2h, h) / 4^h with h = floor(R/2). */
static double width_mu(size_t R) {
    const double h = (double)(R / 2);
    if (h < 1.0) return 1.0;
    return exp(lgamma(2.0 * h + 1.0) - 2.0 * lgamma(h + 1.0) - 2.0 * h * LN2);
}

int atkdsp_bits_width_scan(const uint8_t *b, size_t n, size_t w_lo, size_t w_hi,
                           float *z, float *raw) {
    if ((!b && n) || !z || w_lo < 2 || w_hi < w_lo) return ATKDSP_E_ARG;
    const long long nwidth = (long long)(w_hi - w_lo + 1);
    int err = 0;
    long long iw;
#pragma omp parallel num_threads(atkdsp_get_threads())
    {
        uint32_t *cnt = (uint32_t *)malloc(w_hi * sizeof *cnt);
        if (!cnt) {
#pragma omp critical
            err = 1;
        }
#pragma omp for schedule(dynamic, 4)
        for (iw = 0; iw < nwidth; ++iw) {
            const size_t w = w_lo + (size_t)iw;
            const size_t R = n / w;
            if (!cnt || R < 2) {
                z[iw] = NAN;
                if (raw) raw[iw] = NAN;
                continue;
            }
            memset(cnt, 0, w * sizeof *cnt);
            for (size_t r = 0; r < R; ++r) {
                const uint8_t *row = b + r * w;
                for (size_t c = 0; c < w; ++c) cnt[c] += row[c] & 1u;
            }
            long long S = 0;
            for (size_t c = 0; c < w; ++c) {
                const long long d = 2LL * (long long)cnt[c] - (long long)R;
                S += d < 0 ? -d : d;
            }
            const double rv = (double)S / ((double)w * (double)R);
            const double mu = width_mu(R);
            const double var = 1.0 / (double)R - mu * mu;
            z[iw] = var > 0.0 ? (float)((rv - mu) * sqrt((double)w) / sqrt(var)) : NAN;
            if (raw) raw[iw] = (float)rv;
        }
        free(cnt);
    }
    return err ? ATKDSP_E_NOMEM : ATKDSP_OK;
}

/* ---- sync discovery ------------------------------------------------------ */

/* LSD radix sort of 64-bit keys, 16 bits per pass. */
static int radix_sort_u64(uint64_t *a, size_t n) {
    uint64_t *t = (uint64_t *)malloc(n * sizeof *t);
    size_t *cnt = (size_t *)malloc(65536 * sizeof *cnt);
    if (!t || !cnt) { free(t); free(cnt); return 0; }
    for (int pass = 0; pass < 4; ++pass) {
        const int sh = 16 * pass;
        memset(cnt, 0, 65536 * sizeof *cnt);
        for (size_t i = 0; i < n; ++i) cnt[(a[i] >> sh) & 0xffffu]++;
        size_t s = 0;
        for (size_t d = 0; d < 65536; ++d) { const size_t c = cnt[d]; cnt[d] = s; s += c; }
        for (size_t i = 0; i < n; ++i) t[cnt[(a[i] >> sh) & 0xffffu]++] = a[i];
        uint64_t *x = a; a = t; t = x;     /* four passes: ends back in the caller's array */
    }
    free(t); free(cnt);
    return 1;
}

typedef struct {
    uint32_t key;
    int      period, phase, count;
    long long first;
    int succ, pred, used;
} sync_node;

static int node_cmp(const void *x, const void *y) {
    const sync_node *a = (const sync_node *)x, *b = (const sync_node *)y;
    if (a->period != b->period) return a->period < b->period ? -1 : 1;
    if (a->phase != b->phase) return a->phase < b->phase ? -1 : 1;
    if (a->key != b->key) return a->key < b->key ? -1 : 1;
    return 0;
}

static int cand_cmp(const void *x, const void *y) {
    const atkdsp_synccand *a = (const atkdsp_synccand *)x, *b = (const atkdsp_synccand *)y;
    if (a->log10_pfa != b->log10_pfa) return a->log10_pfa < b->log10_pfa ? -1 : 1;
    if (a->count != b->count) return a->count > b->count ? -1 : 1;
    if (a->first != b->first) return a->first < b->first ? -1 : 1;
    if (a->period != b->period) return a->period < b->period ? -1 : 1;
    return 0;
}

static int u32_cmp(const void *x, const void *y) {
    const uint32_t a = *(const uint32_t *)x, b = *(const uint32_t *)y;
    return a < b ? -1 : (a > b);
}

/* Find the node at (period, phase) continuing word `key` (its first L-1 bits
 * equal key's last L-1) with an agreeing count. Best count wins, ties the
 * smaller key. Nodes are sorted by (period, phase, key). */
static int find_succ(const sync_node *nd, int nn, int i, int L) {
    const int P = nd[i].period;
    const int ph = (nd[i].phase + 1) % P;
    const uint32_t lowmask = (L == 32) ? 0x7fffffffu : ((1u << (L - 1)) - 1u);
    const uint32_t want = nd[i].key & lowmask;
    int lo = 0, hi = nn;          /* first node with (period, phase) >= (P, ph) */
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (nd[mid].period < P || (nd[mid].period == P && nd[mid].phase < ph)) lo = mid + 1;
        else hi = mid;
    }
    int best = -1;
    for (int j = lo; j < nn && nd[j].period == P && nd[j].phase == ph; ++j) {
        if (j == i) continue;
        if ((nd[j].key >> 1) != want) continue;
        /* the counts must agree: one random extra bit would leave about half
         * (sigma sqrt(mx)/2), so the smaller must sit 3 sigma above mx/2 */
        const int a = nd[i].count, c = nd[j].count;
        const int mn = a < c ? a : c, mx = a < c ? c : a;
        if (mn != mx && !((double)mn > (double)mx / 2.0 + 1.5 * sqrt((double)mx))) continue;
        if (best < 0 || nd[j].count > nd[best].count) best = j;
    }
    return best;
}

typedef struct { long long first; int count; double expected, log10_pfa; } sync_score;

/* Exact matches of pat[0, len) at positions ph + kP, and their significance. */
static sync_score score_pattern(const uint8_t *b, size_t n, const uint8_t *pat, int len,
                                int ph, int P, double log10_T) {
    sync_score r = {-1, 0, 0.0, 0.0};
    double slots = 0.0;
    for (size_t pos = (size_t)ph; pos + (size_t)len <= n; pos += (size_t)P) {
        slots += 1.0;
        int ok = 1;
        for (int j = 0; j < len; ++j)
            if ((b[pos + (size_t)j] & 1u) != pat[j]) { ok = 0; break; }
        if (ok) { if (r.first < 0) r.first = (long long)pos; ++r.count; }
    }
    const double ln_q = -(double)len * LN2;
    r.expected = slots * exp(ln_q);
    r.log10_pfa = atk_log10_binom_tail(slots, (double)r.count, ln_q) + log10_T + log10((double)P);
    return r;
}

ptrdiff_t atkdsp_bits_sync_discover(const uint8_t *b, size_t n, int L, int min_count,
                                    double pfa, atkdsp_synccand *out, size_t cap) {
    if ((!b && n) || L < 8 || L > 32 || (!out && cap)) return ATKDSP_E_ARG;
    if (n >= ((size_t)1 << 31)) return ATKDSP_E_ARG;     /* positions, periods held as 31-bit */
    if (min_count < 2) min_count = 3;
    if (!(pfa > 0.0)) pfa = 1e-3;
    if (n < (size_t)(2 * L)) return 0;
    const double log10_thr = log10(pfa);
    const size_t np = n - (size_t)L + 1;
    const uint32_t kmask = (L == 32) ? 0xffffffffu : ((1u << L) - 1u);

    uint64_t *kp = (uint64_t *)malloc(np * sizeof *kp);
    if (!kp) return ATKDSP_E_NOMEM;
    {
        uint32_t key = 0;
        for (int j = 0; j < L - 1; ++j) key = (key << 1) | (b[j] & 1u);
        for (size_t p = 0; p < np; ++p) {
            key = ((key << 1) | (b[p + (size_t)L - 1] & 1u)) & kmask;
            kp[p] = ((uint64_t)key << 32) | (uint64_t)p;
        }
    }
    if (!radix_sort_u64(kp, np)) { free(kp); return ATKDSP_E_NOMEM; }

    size_t ncap = 1024, nn = 0;
    sync_node *nd = (sync_node *)malloc(ncap * sizeof *nd);
    uint32_t *kept = (uint32_t *)malloc(np * sizeof *kept);
    uint32_t *diff = (uint32_t *)malloc(np * sizeof *diff);
    if (!nd || !kept || !diff) { free(kp); free(nd); free(kept); free(diff); return ATKDSP_E_NOMEM; }

    for (size_t g0 = 0; g0 < np;) {
        const uint32_t key = (uint32_t)(kp[g0] >> 32);
        size_t g1 = g0;
        while (g1 < np && (uint32_t)(kp[g1] >> 32) == key) ++g1;
        const size_t c = g1 - g0;
        if (c >= (size_t)min_count) {
            /* thin: keep an occurrence only L or more bits after the last kept */
            size_t nk = 0;
            for (size_t i = g0; i < g1; ++i) {
                const uint32_t p = (uint32_t)(kp[i] & 0xffffffffu);
                if (nk == 0 || p - kept[nk - 1] >= (uint32_t)L) kept[nk++] = p;
            }
            if (nk >= (size_t)min_count) {
                for (size_t i = 0; i + 1 < nk; ++i) diff[i] = kept[i + 1] - kept[i];
                qsort(diff, nk - 1, sizeof *diff, u32_cmp);
                uint32_t P = diff[0];
                size_t bestrun = 0;
                for (size_t i = 0; i < nk - 1;) {
                    size_t j = i;
                    while (j < nk - 1 && diff[j] == diff[i]) ++j;
                    if (j - i > bestrun) { bestrun = j - i; P = diff[i]; }
                    i = j;
                }
                /* the most populated residue mod P (ties: the smallest) */
                for (size_t i = 0; i < nk; ++i) diff[i] = kept[i] % P;
                qsort(diff, nk, sizeof *diff, u32_cmp);
                int bestph = 0, bestc = 0;
                for (size_t i = 0; i < nk;) {
                    size_t j = i;
                    while (j < nk && diff[j] == diff[i]) ++j;
                    if ((int)(j - i) > bestc) { bestc = (int)(j - i); bestph = (int)diff[i]; }
                    i = j;
                }
                long long firstp = 0;
                for (size_t i = 0; i < nk; ++i)
                    if ((int)(kept[i] % P) == bestph) { firstp = kept[i]; break; }
                if (bestc >= min_count) {
                    if (nn == ncap) {
                        ncap *= 2;
                        sync_node *t = (sync_node *)realloc(nd, ncap * sizeof *nd);
                        if (!t) { free(kp); free(nd); free(kept); free(diff); return ATKDSP_E_NOMEM; }
                        nd = t;
                    }
                    nd[nn].key = key; nd[nn].period = (int)P; nd[nn].phase = bestph;
                    nd[nn].count = bestc; nd[nn].first = firstp;
                    nd[nn].succ = nd[nn].pred = -1; nd[nn].used = 0;
                    ++nn;
                }
            }
        }
        g0 = g1;
    }
    free(kp); free(kept); free(diff);
    if (nn == 0) { free(nd); return 0; }

    qsort(nd, nn, sizeof *nd, node_cmp);
    const int inn = (int)nn;
    for (int i = 0; i < inn; ++i) nd[i].succ = find_succ(nd, inn, i, L);
    for (int i = 0; i < inn; ++i) {          /* one predecessor each: the best count */
        const int s = nd[i].succ;
        if (s < 0) continue;
        const int p = nd[s].pred;
        if (p < 0) nd[s].pred = i;
        else if (nd[i].count > nd[p].count) { nd[p].succ = -1; nd[s].pred = i; }
        else nd[i].succ = -1;
    }

    size_t ccap = 64, nc = 0;
    atkdsp_synccand *cand = (atkdsp_synccand *)malloc(ccap * sizeof *cand);
    if (!cand) { free(nd); return ATKDSP_E_NOMEM; }
    const double log10_T = (double)L * log10(2.0) + log10((double)n / (double)L);

    for (int pass = 0; pass < 2; ++pass) {   /* chains from their starts, then cycles */
        for (int s = 0; s < inn; ++s) {
            if (nd[s].used || (pass == 0 && nd[s].pred >= 0)) continue;
            uint8_t chain[ATKDSP_SYNC_MAXBITS];
            int len = 0;
            for (int j = 0; j < L; ++j) chain[len++] = (uint8_t)((nd[s].key >> (L - 1 - j)) & 1u);
            nd[s].used = 1;
            for (int cur = nd[s].succ; cur >= 0 && !nd[cur].used && len < ATKDSP_SYNC_MAXBITS;
                 cur = nd[cur].succ) {
                chain[len++] = (uint8_t)(nd[cur].key & 1u);
                nd[cur].used = 1;
            }
            const int P = nd[s].period, ph = nd[s].phase;
            /* trim either end while that makes the pattern more significant: a
             * field that never changes stays, a counter's slow bit goes */
            int lo = 0, hi = len;                       /* pattern = chain[lo, hi) */
            sync_score sc = score_pattern(b, n, chain + lo, hi - lo, (ph + lo) % P, P, log10_T);
            while (hi - lo > L) {
                const sync_score a = score_pattern(b, n, chain + lo + 1, hi - lo - 1, (ph + lo + 1) % P, P, log10_T);
                const sync_score z = score_pattern(b, n, chain + lo, hi - lo - 1, (ph + lo) % P, P, log10_T);
                if (z.log10_pfa <= a.log10_pfa && z.log10_pfa < sc.log10_pfa) { --hi; sc = z; }
                else if (a.log10_pfa < sc.log10_pfa) { ++lo; sc = a; }
                else break;
            }
            if (sc.count < min_count || sc.log10_pfa > log10_thr) continue;
            atkdsp_synccand c;
            memset(&c, 0, sizeof c);
            len = hi - lo;
            memcpy(c.pattern, chain + lo, (size_t)len);
            c.period = P; c.length = len;
            c.first = sc.first; c.count = sc.count;
            c.expected = sc.expected; c.log10_pfa = sc.log10_pfa;
            for (int p0 = 1; p0 <= len / 2; ++p0) {
                int per = 1;
                for (int j = 0; j + p0 < len; ++j)
                    if (c.pattern[j] != c.pattern[j + p0]) { per = 0; break; }
                if (per) { c.flags |= ATKDSP_SYNC_REPETITIVE; break; }
            }
            if (nc == ccap) {
                ccap *= 2;
                atkdsp_synccand *t = (atkdsp_synccand *)realloc(cand, ccap * sizeof *cand);
                if (!t) { free(nd); free(cand); return ATKDSP_E_NOMEM; }
                cand = t;
            }
            cand[nc++] = c;
        }
    }
    free(nd);
    qsort(cand, nc, sizeof *cand, cand_cmp);
    const size_t m = nc < cap ? nc : cap;
    if (m) memcpy(out, cand, m * sizeof *cand);
    free(cand);
    return (ptrdiff_t)m;
}

/* ---- statistics ------------------------------------------------------------ */

/* NIST SP 800-22 rev 1a, §2.9: expected value and variance of fn per L. */
static const double MAURER_E[17] = {0, 0.7326495, 1.5374383, 2.4016068, 3.3112247,
    4.2534266, 5.2177052, 6.1962507, 7.1836656, 8.1764248, 9.1723243, 10.170032,
    11.168765, 12.168070, 13.167693, 14.167488, 15.167379};
static const double MAURER_V[17] = {0, 0.690, 1.338, 1.901, 2.358, 2.705, 2.954, 3.125,
    3.238, 3.311, 3.356, 3.384, 3.401, 3.410, 3.416, 3.419, 3.421};

static double apen_phi(const uint8_t *b, size_t n, int m, uint64_t *cnt) {
    const size_t cells = (size_t)1 << m;
    memset(cnt, 0, cells * sizeof *cnt);
    const uint32_t mask = (uint32_t)(cells - 1);
    uint32_t v = 0;
    for (int j = 0; j < m - 1; ++j) v = (v << 1) | (b[j] & 1u);
    for (size_t i = 0; i < n; ++i) {   /* circular: the sequence followed by its first m-1 bits */
        v = ((v << 1) | (b[(i + (size_t)m - 1) % n] & 1u)) & mask;
        cnt[v]++;
    }
    double phi = 0.0;
    for (size_t c = 0; c < cells; ++c)
        if (cnt[c]) { const double p = (double)cnt[c] / (double)n; phi += p * log(p); }
    return phi;
}

int atkdsp_bits_stats(const uint8_t *b, size_t n, atkdsp_bitstats *o) {
    if ((!b && n) || !o) return ATKDSP_E_ARG;
    memset(o, 0, sizeof *o);
    o->n = n;
    for (size_t i = 0; i < n; ++i) o->ones += b[i] & 1u;
    o->density = n ? (double)o->ones / (double)n : NAN;

    for (size_t i = 0; i < n;) {   /* runs */
        const uint8_t v = b[i] & 1u;
        size_t j = i + 1;
        while (j < n && (b[j] & 1u) == v) ++j;
        const size_t len = j - i;
        const size_t k = len >= 32 ? 32 : len;
        if (v) o->run1[k]++; else o->run0[k]++;
        i = j;
    }
    for (int k = 1; k <= 8; ++k) {   /* overlapping n-grams */
        if (n < (size_t)k) break;
        unsigned v = 0;
        const unsigned mask = (1u << k) - 1u;
        for (int j = 0; j < k - 1; ++j) v = (v << 1) | (b[j] & 1u);
        for (size_t p = (size_t)k - 1; p < n; ++p) {
            v = ((v << 1) | (b[p] & 1u)) & mask;
            o->ngram[k - 1][v]++;
        }
    }
    /* chi-square of non-overlapping words */
    o->chi2_bits = atk_chi_width(n);
    if (o->chi2_bits) {
        const int w = o->chi2_bits;
        const size_t cells = (size_t)1 << w, nw = n / (size_t)w;
        uint64_t cnt[256];
        memset(cnt, 0, sizeof cnt);
        for (size_t i = 0; i < nw; ++i) {
            unsigned v = 0;
            for (int k = 0; k < w; ++k) v = (v << 1) | (b[i * (size_t)w + (size_t)k] & 1u);
            cnt[v]++;
        }
        const double e = (double)nw / (double)cells;
        double x2 = 0.0;
        for (size_t c = 0; c < cells; ++c) { const double d = (double)cnt[c] - e; x2 += d * d / e; }
        o->chi2 = x2;
        o->chi2_p = atk_igamc((double)(cells - 1) / 2.0, x2 / 2.0);
    } else {
        o->chi2 = o->chi2_p = NAN;
    }

    uint64_t *cnt = (uint64_t *)malloc(((size_t)1 << 16) * sizeof *cnt);
    if (!cnt) return ATKDSP_E_NOMEM;

    /* Maurer's universal statistic */
    int L = 0;
    for (int l = 16; l >= 1; --l)
        if ((double)n >= 1010.0 * (double)l * (double)((size_t)1 << l)) { L = l; break; }
    o->maurer_L = L;
    if (L) {
        const size_t Q = (size_t)10 << L;
        const size_t K = n / (size_t)L - Q;
        uint64_t *T = cnt;   /* last block index (1-based) at which each value was seen */
        memset(T, 0, ((size_t)1 << L) * sizeof *T);
        double sum = 0.0;
        for (size_t i = 1; i <= Q + K; ++i) {
            unsigned v = 0;
            const uint8_t *p = b + (i - 1) * (size_t)L;
            for (int k = 0; k < L; ++k) v = (v << 1) | (p[k] & 1u);
            if (i > Q) sum += log((double)(i - T[v])) / LN2;
            T[v] = i;
        }
        const double fn = sum / (double)K;
        const double c = 0.7 - 0.8 / L + (4.0 + 32.0 / L) * pow((double)K, -3.0 / L) / 15.0;
        const double sigma = c * sqrt(MAURER_V[L] / (double)K);
        o->maurer = fn;
        o->maurer_p = erfc(fabs(fn - MAURER_E[L]) / (sqrt(2.0) * sigma));
    } else {
        o->maurer = o->maurer_p = NAN;
    }

    /* approximate entropy */
    if (n >= 128) {
        int lg = 0;
        while (((size_t)2 << lg) <= n) ++lg;    /* floor(log2 n) */
        int m = lg - 6;
        if (m > 10) m = 10;
        o->apen_m = m;
        const double ap = apen_phi(b, n, m, cnt) - apen_phi(b, n, m + 1, cnt);
        o->apen = ap;
        const double x2 = 2.0 * (double)n * (LN2 - ap);
        o->apen_p = atk_igamc((double)((size_t)1 << (m - 1)), x2 / 2.0);
    } else {
        o->apen = o->apen_p = NAN;
    }

    /* block entropy by width */
    for (int w = 1; w <= 16; ++w) {
        const size_t cells = (size_t)1 << w, nw = n / (size_t)w;
        if (nw < cells) { o->block_entropy[w - 1] = NAN; continue; }
        memset(cnt, 0, cells * sizeof *cnt);
        for (size_t i = 0; i < nw; ++i) {
            unsigned v = 0;
            const uint8_t *p = b + i * (size_t)w;
            for (int k = 0; k < w; ++k) v = (v << 1) | (p[k] & 1u);
            cnt[v]++;
        }
        double h = 0.0;
        for (size_t c = 0; c < cells; ++c)
            if (cnt[c]) { const double p = (double)cnt[c] / (double)nw; h -= p * log(p); }
        o->block_entropy[w - 1] = h / LN2 / (double)w;
    }
    free(cnt);
    return ATKDSP_OK;
}

/* ---- Walsh-Hadamard ---------------------------------------------------------- */

ptrdiff_t atkdsp_bits_wht(const uint8_t *b, size_t n, size_t width, const uint32_t *cols,
                          int ncols, int32_t *spectrum) {
    if ((!b && n) || !cols || !spectrum || width < 1 || ncols < 1 || ncols > 24) return ATKDSP_E_ARG;
    for (int j = 0; j < ncols; ++j) if (cols[j] >= width) return ATKDSP_E_ARG;
    const size_t size = (size_t)1 << ncols;
    const size_t rows = n / width;
    if (rows > 0x7fffffffu) return ATKDSP_E_ARG;            /* the spectrum is int32 */
    memset(spectrum, 0, size * sizeof *spectrum);
    for (size_t r = 0; r < rows; ++r) {
        uint32_t v = 0;
        for (int j = 0; j < ncols; ++j) v |= (uint32_t)(b[r * width + cols[j]] & 1u) << j;
        spectrum[v]++;
    }
    for (size_t h = 1; h < size; h <<= 1)
        for (size_t i = 0; i < size; i += 2 * h)
            for (size_t j = i; j < i + h; ++j) {
                const int32_t x = spectrum[j], y = spectrum[j + h];
                spectrum[j] = x + y; spectrum[j + h] = x - y;
            }
    return (ptrdiff_t)rows;
}

static int is_pow2(size_t v) { return v && !(v & (v - 1)); }

ptrdiff_t atkdsp_bits_walsh(const uint8_t *b, size_t n, size_t size, int navg,
                            float *sequency_power, float *dyadic_acf) {
    if ((!b && n) || !is_pow2(size) || size < 4 || size > 65536) return ATKDSP_E_ARG;
    size_t blocks = n / size;
    if (navg > 0 && (size_t)navg < blocks) blocks = (size_t)navg;
    /* keep sum_blocks sum_k X^2 * size (the dyadic transform) inside int64 */
    const double lim = 4.0e18 / ((double)size * (double)size * (double)size);
    if ((double)blocks > lim) blocks = (size_t)lim;
    if (blocks == 0) {
        for (size_t i = 0; i < size; ++i) {
            if (sequency_power) sequency_power[i] = NAN;
            if (dyadic_acf) dyadic_acf[i] = NAN;
        }
        return 0;
    }
    int32_t *X = (int32_t *)malloc(size * sizeof *X);
    int64_t *P = (int64_t *)calloc(size, sizeof *P);
    if (!X || !P) { free(X); free(P); return ATKDSP_E_NOMEM; }
    for (size_t blk = 0; blk < blocks; ++blk) {
        const uint8_t *p = b + blk * size;
        for (size_t i = 0; i < size; ++i) X[i] = (p[i] & 1u) ? -1 : 1;
        for (size_t h = 1; h < size; h <<= 1)
            for (size_t i = 0; i < size; i += 2 * h)
                for (size_t j = i; j < i + h; ++j) {
                    const int32_t x = X[j], y = X[j + h];
                    X[j] = x + y; X[j + h] = x - y;
                }
        for (size_t k = 0; k < size; ++k) P[k] += (int64_t)X[k] * (int64_t)X[k];
    }
    int lg = 0;
    while (((size_t)1 << lg) < size) ++lg;
    const double norm = (double)size * (double)blocks;
    if (sequency_power)
        for (size_t s = 0; s < size; ++s) {
            size_t g = s ^ (s >> 1), h = 0;
            for (int k = 0; k < lg; ++k) h |= ((g >> k) & 1u) << (lg - 1 - k);
            sequency_power[s] = (float)((double)P[h] / norm);
        }
    if (dyadic_acf) {
        for (size_t h = 1; h < size; h <<= 1)
            for (size_t i = 0; i < size; i += 2 * h)
                for (size_t j = i; j < i + h; ++j) {
                    const int64_t x = P[j], y = P[j + h];
                    P[j] = x + y; P[j + h] = x - y;
                }
        const double dn = (double)size * (double)size * (double)blocks;
        for (size_t t = 0; t < size; ++t) dyadic_acf[t] = (float)((double)P[t] / dn);
    }
    free(X); free(P);
    return (ptrdiff_t)blocks;
}

ptrdiff_t atkdsp_bits_cyclic(const uint8_t *b, size_t n, size_t size, int navg,
                             float *cyclic_time, float *cyclic_freq) {
    if ((!b && n) || !is_pow2(size) || size < 4 || size > 65536) return ATKDSP_E_ARG;
    size_t blocks = n / size;
    if (navg > 0 && (size_t)navg < blocks) blocks = (size_t)navg;
    const size_t nf = size / 2 + 1;
    if (blocks == 0) {
        if (cyclic_time) for (size_t i = 0; i < size; ++i) cyclic_time[i] = NAN;
        if (cyclic_freq) for (size_t i = 0; i < nf; ++i) cyclic_freq[i] = NAN;
        return 0;
    }
    rfft_plan plan = make_rfft_plan(size);
    double *c = (double *)malloc(size * sizeof *c);
    double *pw = (double *)malloc(nf * sizeof *pw);
    double *psum = (double *)calloc(nf, sizeof *psum);
    int64_t *rsum = (int64_t *)calloc(size, sizeof *rsum);
    if (!plan || !c || !pw || !psum || !rsum) {
        if (plan) destroy_rfft_plan(plan);
        free(c); free(pw); free(psum); free(rsum);
        return ATKDSP_E_NOMEM;
    }
    int rc = 0;
    for (size_t blk = 0; blk < blocks && !rc; ++blk) {
        const uint8_t *p = b + blk * size;
        for (size_t i = 0; i < size; ++i) c[i] = (p[i] & 1u) ? -1.0 : 1.0;
        if (rfft_forward(plan, c, 1.0) != 0) { rc = 1; break; }
        /* halfcomplex: r0, r1, i1, r2, i2, ..., r_{size/2} */
        pw[0] = c[0] * c[0];
        for (size_t k = 1; k < size / 2; ++k) pw[k] = c[2 * k - 1] * c[2 * k - 1] + c[2 * k] * c[2 * k];
        pw[size / 2] = c[size - 1] * c[size - 1];
        for (size_t k = 0; k < nf; ++k) psum[k] += pw[k];
        /* circular autocorrelation = inverse transform of the power: integers */
        c[0] = pw[0];
        for (size_t k = 1; k < size / 2; ++k) { c[2 * k - 1] = pw[k]; c[2 * k] = 0.0; }
        c[size - 1] = pw[size / 2];
        if (rfft_backward(plan, c, 1.0 / (double)size) != 0) { rc = 1; break; }
        for (size_t t = 0; t < size; ++t) rsum[t] += (int64_t)llround(c[t]);
    }
    destroy_rfft_plan(plan);
    if (!rc) {
        const double norm = (double)size * (double)blocks;
        if (cyclic_freq) for (size_t k = 0; k < nf; ++k) cyclic_freq[k] = (float)(psum[k] / norm);
        if (cyclic_time) for (size_t t = 0; t < size; ++t) cyclic_time[t] = (float)((double)rsum[t] / norm);
    }
    free(c); free(pw); free(psum); free(rsum);
    return rc ? ATKDSP_E_FFT : (ptrdiff_t)blocks;
}
