/* Naming the code (header §18): block codes, convolutional codes,
 * Reed-Solomon, block interleavers. The linear algebra is src/gf2.c; the
 * decoders that supply the decode -> re-encode evidence are §16's. The twin
 * is reference.py §18, the same searches in the same order. */
#include "internal.h"
#include <stdarg.h>
#include <stdio.h>

#define LOG10_2 0.30102999566398119521
#define LN2     0.693147180559945309417
#define ATK_ID_ROWS 4096            /* codewords examined at most */
#define ATK_ROBUST_NMAX 512         /* the robust block search's longest length */
#ifdef ATKDSP_CODEREC_TRACE
#  define TRACE(...) fprintf(stderr, __VA_ARGS__)
#else
#  define TRACE(...) ((void)0)
#endif

/* A name, truncated to the field (names are short; this keeps the bound in one place). */
static void set_name(char *dst, size_t cap, const char *fmt, ...) {
    char tmp[160];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    size_t L = strlen(tmp);
    if (L >= cap) L = cap - 1;
    memcpy(dst, tmp, L);
    dst[L] = '\0';
}

/* ---- polynomials over GF(2) up to degree 639, bit i = x^i ---------------- */
#define PW 10
typedef struct { uint64_t w[PW]; } poly_t;
static int pdeg(const poly_t *p) {
    for (int q = PW - 1; q >= 0; --q)
        if (p->w[q]) { int b = 63; while (!((p->w[q] >> b) & 1u)) --b; return 64 * q + b; }
    return -1;
}
static void pxor_shift(poly_t *a, const poly_t *b, int k) {
    const int q = k / 64, r = k % 64;
    for (int j = 0; j < PW; ++j) {
        if (!b->w[j]) continue;
        if (j + q < PW) a->w[j + q] ^= b->w[j] << r;
        if (r && j + q + 1 < PW) a->w[j + q + 1] ^= b->w[j] >> (64 - r);
    }
}
static void pmod(poly_t *a, const poly_t *g) {
    const int dg = pdeg(g);
    int da;
    while ((da = pdeg(a)) >= dg && da >= 0) pxor_shift(a, g, da - dg);
}
static void pgcd(poly_t *a, poly_t *b) {
    while (pdeg(b) >= 0) { pmod(a, b); poly_t t = *a; *a = *b; *b = t; }
}
/* a codeword (bit j = position j, n positions) as a polynomial: x^(n-1-j) */
static void word_poly(const uint64_t *v, int n, poly_t *p) {
    memset(p, 0, sizeof *p);
    for (int j = 0; j < n; ++j)
        if ((v[j >> 6] >> (j & 63)) & 1u) { const int e = n - 1 - j; p->w[e / 64] |= 1ULL << (e % 64); }
}

static int is_primitive(uint32_t f, int m) {
    if ((f >> m) != 1u || !(f & 1u)) return 0;
    const uint32_t nn = (1u << m) - 1u;
    uint32_t x = 1;
    for (uint32_t i = 1; i <= nn; ++i) {
        x <<= 1;
        if (x >> m) x ^= f;
        if (x == 1u) return i == nn;
    }
    return 0;
}

/* ---- the block search, shared with the interleaver search ---------------- */
#define ATK_TIES_MAX 8
typedef struct {
    int N, offset, robust, w0, d0, complete;
    int slide;              /* > 0: the deficiency repeats every `slide` bits of offset */
    double log10_pfa;
    int nties;              /* offsets sharing the largest deficiency, offset first: an */
    int ties[ATK_TIES_MAX]; /* interleaver's block start can be any of them             */
} blk_find;

/* The smallest shift P (a divisor of N) under which the deficiency by offset
 * repeats. A block code of length N has its full deficiency at one offset
 * only (P = N); a convolutional code repeats every n bits, a linear
 * recurrence (an LFSR, a constant or periodic stream) at every offset. */
static int def_period(const int *dv, int N) {
    for (int P = 1; P < N; ++P) {
        if (N % P) continue;
        int same = 1;
        for (int o = 0; o < N && same; ++o) if (dv[o] != dv[(o + P) % N]) same = 0;
        if (same) return P;
    }
    return N;
}

static int nw_of(int w) { return (w + 63) / 64; }

/* log10 P(an R x w random GF(2) matrix has deficiency >= d): 2^-d(R-w+d)
 * times a constant below 4 (the product of the (1 - 2^-i) terms), so the
 * bound is 4 x 2^-d(R-w+d) */
static double def_l10(int d, int R, int w) {
    return 0.6020599913279624 - (double)d * (double)(R - w + d) * LOG10_2;
}

/* ---- fixed bits (v0.17.0; plan §D4) ------------------------------------------
 * A payload position that never varies — ASCII's high bit, a framing field —
 * is constant in every codeword. In the AFFINE space (each row with a
 * constant-1 column appended at bit N) that is the check (e_p, value), of
 * weight 1 in the N data bits whatever the value; in the plain null space a
 * constant 0 is the weight-1 check e_p and a constant 1 is no check at all.
 * These helpers judge extra structure in the affine space, so "a constant
 * position" has one shape. The twin: reference.py _quotient, _reduce,
 * _coset_is_constant. */
static int vec_topbit(const uint64_t *v, int nw) {
    for (int q = nw - 1; q >= 0; --q)
        if (v[q]) { int b = 63; while (!((v[q] >> b) & 1u)) --b; return 64 * q + b; }
    return -1;
}
static int vec_iszero(const uint64_t *v, int nw) {
    for (int q = 0; q < nw; ++q) if (v[q]) return 0;
    return 1;
}
static int vec_weight(const uint64_t *v, int nw) {
    int w = 0;
    for (int q = 0; q < nw; ++q) w += atk_popcount64(v[q]);
    return w;
}
/* v reduced (in place) against a canonical basis, highest pivot first. */
static void vec_reduce(const uint64_t *base, int db, int nw, uint64_t *v) {
    for (int i = 0; i < db; ++i) {
        const uint64_t *bi = base + (size_t)i * nw;
        const int t = vec_topbit(bi, nw);
        if (t >= 0 && ((v[t >> 6] >> (t & 63)) & 1u))
            for (int q = 0; q < nw; ++q) v[q] ^= bi[q];
    }
}
/* R rows of width N (nw_of(N) words) -> the same rows with bit N set, in
 * nw_of(N + 1) words. */
static void aug_rows(const uint64_t *rows, int R, int N, uint64_t *out) {
    const int nw = nw_of(N), nwa = nw_of(N + 1);
    for (int i = 0; i < R; ++i) {
        uint64_t *o = out + (size_t)i * nwa;
        memset(o, 0, (size_t)nwa * sizeof *o);
        memcpy(o, rows + (size_t)i * nw, (size_t)nw * sizeof *o);
        o[N >> 6] |= 1ULL << (N & 63);
    }
}
/* A canonical basis of span(Hs) modulo span(H0) (H0 canonical): each Hs
 * vector reduced against H0, the nonzero remainders canonicalised into out
 * (ds vectors of nwa words). Returns their count. */
static int aug_quotient(const uint64_t *H0, int d0, const uint64_t *Hs, int ds, int nwa,
                        uint64_t *out) {
    int m = 0;
    for (int i = 0; i < ds; ++i) {
        uint64_t *o = out + (size_t)m * nwa;
        memcpy(o, Hs + (size_t)i * nwa, (size_t)nwa * sizeof *o);
        vec_reduce(H0, d0, nwa, o);
        if (!vec_iszero(o, nwa)) ++m;
    }
    return m ? atk_gf2_canon(out, m, 64 * nwa) : 0;
}
/* True when v + span(H0) holds a vector of weight 1 in the N data bits (any
 * value of the constant-1 bit N): the extra structure v is a constant
 * position, not a code check. */
static int coset_is_constant(const uint64_t *H0, int d0, int N, const uint64_t *v, uint64_t *tmp) {
    const int nwa = nw_of(N + 1);
    for (int p = 0; p < N; ++p) {
        for (int bN = 0; bN < 2; ++bN) {
            memcpy(tmp, v, (size_t)nwa * sizeof *tmp);
            tmp[p >> 6] ^= 1ULL << (p & 63);
            if (bN) tmp[N >> 6] ^= 1ULL << (N & 63);
            vec_reduce(H0, d0, nwa, tmp);
            if (vec_iszero(tmp, nwa)) return 1;
        }
    }
    return 0;
}
/* The affine null space of the aligned rows at (off, N): R rows -> basis of
 * nwa words each (canonical). Returns its dimension, or -1 on memory. */
static int aug_nullspace(const uint64_t *bw, size_t off, int N, int R, uint64_t *basis) {
    const int nw = nw_of(N), nwa = nw_of(N + 1);
    uint64_t *rows = (uint64_t *)malloc(((size_t)R + 1) * (size_t)nw * sizeof *rows);
    uint64_t *ar = (uint64_t *)malloc(((size_t)R + 1) * (size_t)nwa * sizeof *ar);
    if (!rows || !ar) { free(rows); free(ar); return -1; }
    atk_gf2_rows(bw, off, N, R, N, rows);
    aug_rows(rows, R, N, ar);
    const int d = atk_gf2_nullspace(ar, R, N + 1, basis, N + 1);
    free(rows); free(ar);
    return d;
}
/* Any of the d checks (nwa words) failing on any of the R augmented rows? */
static int aug_any_unsat(const uint64_t *ar, int R, int nwa, const uint64_t *H, int d) {
    for (int i = 0; i < d; ++i)
        if (atk_gf2_unsat(ar, R, nwa, H + (size_t)i * nwa)) return 1;
    return 0;
}
/* Every vector of the extra structure a stretch shows beyond H0 is a constant
 * position: 1 when so (or when there is no extra), 0 when some extra is a
 * code check (noise broke it in the rows H0 came from), -1 on memory. */
static int extras_are_constant(const uint64_t *H0, int d0, const uint64_t *Hs, int ds, int N) {
    const int nwa = nw_of(N + 1);
    uint64_t *ex = (uint64_t *)malloc(((size_t)ds + 1) * (size_t)nwa * sizeof *ex);
    uint64_t *tmp = (uint64_t *)malloc((size_t)nwa * sizeof *tmp);
    if (!ex || !tmp) { free(ex); free(tmp); return -1; }
    const int m = aug_quotient(H0, d0, Hs, ds, nwa, ex);
    int ok = 1;
    for (int i = 0; i < m && ok; ++i)
        if (!coset_is_constant(H0, d0, N, ex + (size_t)i * nwa, tmp)) ok = 0;
    free(ex); free(tmp);
    return ok;
}

/* deficiency of the matrix at width w, offset o (exact, or robust) */
static int def_at(const uint64_t *bw, size_t n, size_t o, int w, int robust, uint64_t seed,
                  double pfa, double *l10, uint64_t *M, uint64_t *basis, int *complete) {
    const size_t fit = o < n ? (n - o) / (size_t)w : 0;
    *l10 = 0.0;
    if (complete) *complete = 1;
    if (!robust) {
        int R = w + 32;
        if ((size_t)R > fit) R = (int)fit;
        if (R < w + 16) return -2;
        atk_gf2_rows(bw, o, w, R, w, M);
        const int d = w - atk_gf2_rank(M, R, w);
        if (d > 0) *l10 = def_l10(d, R, w);
        return d;
    }
    const size_t rcap = (size_t)w + 2 + 1024;       /* the subset plus 1024 to validate on */
    const int R = (int)(fit < rcap ? fit : rcap);
    if (R < w + 2 + ATK_ROBUST_VMIN) return -2;
    atk_gf2_rows(bw, o, w, R, w, M);
    atk_robust_info info;
    /* detection (robust == 2) gives up after 6 empty trials, 3 beyond 512
     * bits: where the robust search can see a code at all, it does at once */
    const int quit = robust == 2 ? (w > 512 ? 3 : 6) : 16;
    const int d = atk_gf2_robust_q(M, R, w, seed, pfa, basis, w, &info, quit);
    if (d < 0) return d;
    if (d > 0) *l10 = info.log10_pfa;
    if (complete) *complete = info.complete;
    return d;
}

/* Block length and alignment. Exact elimination at widths n_lo..2 n_hi from
 * bit 0 finds the first deficient width w0 (a multiple of n: 2n always shows,
 * n only when bit 0 happens to be aligned); the smallest divisor N of w0 in
 * range whose best offset at width N is deficient is n, that offset the
 * alignment (at width n the aligned rows carry all n - k checks; a misaligned
 * row only the checks that fit in a codeword's head or tail). When nothing is
 * deficient, the same with the robust search at widths 2N. */
#define BS_EXACT  1
#define BS_ROBUST 2
static int block_search(const uint64_t *bw, size_t n, int n_lo, int n_hi, double pfa,
                        uint64_t seed, blk_find *f, int modes) {
    memset(f, 0, sizeof *f);
    const double log10_thr = log10(pfa);
    const int wlo = n_lo, whi = 2 * n_hi;
    const double log10_widths = log10((double)(whi - wlo + 1));
    const size_t rmax = (size_t)whi + 2 + 1024;
    uint64_t *basis = (uint64_t *)malloc((size_t)whi * (size_t)nw_of(whi) * sizeof *basis);
    uint64_t *M = (uint64_t *)malloc(rmax * (size_t)nw_of(whi) * sizeof *M);
    int *dv = (int *)malloc(((size_t)whi + 1) * sizeof *dv);
    if (!basis || !M || !dv) { free(basis); free(M); free(dv); return -1; }
    int found = 0;
    /* exact: the first deficient width, then its divisors' offsets */
    if (modes & BS_EXACT) {
        int w0 = 0;
        double l0 = 0.0;
        for (int w = wlo; w <= whi && !w0; ++w) {
            double l;
            const int d = def_at(bw, n, 0, w, 0, 0, pfa, &l, M, basis, NULL);
            if (d == -2) break;
            if (d > 0 && l + log10_widths <= log10_thr) { w0 = w; l0 = l + log10_widths; }
        }
        for (int j = w0 ? w0 / n_lo : 0; j >= 1 && !found; --j) {
            if (w0 % j) continue;
            const int N = w0 / j;
            if (N < n_lo || N > n_hi) continue;
            int bo = -1, bd = 0;
            double bl = 0.0;
            for (int o = 0; o < N; ++o) {
                double l;
                const int d = def_at(bw, n, (size_t)o, N, 0, 0, pfa, &l, M, basis, NULL);
                dv[o] = d;
                if (d > bd) { bd = d; bo = o; bl = l; }
            }
            if (bo < 0 || bl + log10((double)N) > log10_thr) continue;
            const int P = def_period(dv, N);
            TRACE("exact: N=%d deficiency repeats every %d\n", N, P);
            if (P < N) {                                 /* sliding structure: no block code */
                f->N = N; f->offset = bo; f->d0 = bd; f->slide = P; f->w0 = w0;
                f->log10_pfa = bl + log10((double)N);
                found = 1;
                break;
            }
            /* exact only if it holds on every codeword AND on three more
             * stretches spread over the stream: noise only lowers a
             * deficiency, so an error among the first rows lowers bd and the
             * all-rows count alike, while a clean stretch elsewhere shows the
             * true value. Any disagreement is noise: a job for the noisy paths */
            /* A stretch may carry MORE structure than the whole for an
             * innocent reason — a payload bit constant within it (ASCII's
             * high bit, a field) — so the extra is judged in the affine
             * space: a constant position is allowed, a code check the first
             * rows broke is noise (v0.17.0; before, any difference was noise
             * and a structured payload failed 1 seed in 20). The same
             * allowance covers the N + 32 rows the scan used, whose count
             * then yields to the whole stream's. */
            const size_t fit = (n - (size_t)bo) / (size_t)N;
            const int Rb = (int)(fit < ATK_ID_ROWS ? fit : ATK_ID_ROWS);
            const int nwa = nw_of(N + 1);
            uint64_t *Mb = (uint64_t *)malloc((size_t)Rb * (size_t)nw_of(N) * sizeof *Mb);
            uint64_t *Ha = (uint64_t *)malloc(((size_t)N + 1) * (size_t)nwa * sizeof *Ha);
            uint64_t *Hs = (uint64_t *)malloc(((size_t)N + 1) * (size_t)nwa * sizeof *Hs);
            if (!Mb || !Ha || !Hs) { free(Mb); free(Ha); free(Hs); free(basis); free(M); free(dv); return -1; }
            atk_gf2_rows(bw, (size_t)bo, N, Rb, N, Mb);
            const int dall = N - atk_gf2_rank(Mb, Rb, N);
            int same = dall == bd, da = -1, oom = 0;
            if (!same && dall > 0 && dall < bd) {
                const int Rs = (int)(fit < (size_t)N + 32 ? fit : (size_t)N + 32);
                da = aug_nullspace(bw, (size_t)bo, N, Rb, Ha);
                const int db = aug_nullspace(bw, (size_t)bo, N, Rs, Hs);
                if (da < 0 || db < 0) oom = 1;
                else if (db > da) {
                    const int e = extras_are_constant(Ha, da, Hs, db, N);
                    if (e < 0) oom = 1;
                    else if (e) { same = 1; bd = dall; }
                }
            }
            if (same && !oom && da < 0) { da = aug_nullspace(bw, (size_t)bo, N, Rb, Ha); if (da < 0) oom = 1; }
            const size_t R1 = (size_t)N + 32;
            uint64_t *ar = oom ? NULL : (uint64_t *)malloc((R1 + 1) * (size_t)nwa * sizeof *ar);
            if (!oom && !ar) oom = 1;
            for (int j = 1; j <= 3 && same && !oom && fit >= 2 * R1; ++j) {
                const size_t r0 = (size_t)j * (fit - R1) / 3;
                atk_gf2_rows(bw, (size_t)bo + r0 * (size_t)N, N, (int)R1, N, Mb);
                aug_rows(Mb, (int)R1, N, ar);
                if (aug_any_unsat(ar, (int)R1, nwa, Ha, da)) { same = 0; break; }
                const int ds = atk_gf2_nullspace(ar, (int)R1, N + 1, Hs, N + 1);
                if (ds < 0) { oom = 1; break; }
                if (ds > da) {
                    const int e = extras_are_constant(Ha, da, Hs, ds, N);
                    if (e < 0) { oom = 1; break; }
                    if (!e) same = 0;
                }
            }
            free(Mb); free(Ha); free(Hs); free(ar);
            if (oom) { free(basis); free(M); free(dv); return -1; }
            TRACE("exact: w0=%d N=%d bo=%d bd=%d dall=%d same=%d\n", w0, N, bo, bd, dall, same);
            if (!same) break;                            /* noisy */
            f->N = N; f->offset = bo; f->robust = 0; f->w0 = w0; f->d0 = bd; f->complete = 1;
            f->log10_pfa = (l0 > bl + log10((double)N)) ? l0 : bl + log10((double)N);
            f->nties = 0;
            for (int o = 0; o < N && f->nties < ATK_TIES_MAX; ++o)
                if (dv[o] == bd) f->ties[f->nties++] = o;
            found = 1;
        }
    }
    /* robust: structure at 2N, confirmed by the offsets at N. Widths to
     * 2N = 1024 only: the work grows as N^4 summed over the lengths, and a
     * subset of 2N rows of 2N bits is rarely clean enough beyond that */
    if (!found && (modes & BS_ROBUST)) {
        const int nrob = n_hi < ATK_ROBUST_NMAX ? n_hi : ATK_ROBUST_NMAX;
        const double log10_n = log10((double)(n_hi - n_lo + 1));
        for (int N = n_lo; N <= nrob && !found; ++N) {
            double l;
            int comp;
            const int d2 = def_at(bw, n, 0, 2 * N, 2, seed + (uint64_t)N, pfa, &l, M, basis, &comp);
            if (d2 == -2) break;
            if (d2 <= 0 || l + log10_n > log10_thr) continue;
            const double l0 = l + log10_n;
            TRACE("robust: N=%d fires at 2N d2=%d l10=%.1f\n", N, d2, l);
            int bo = -1, bd = 0, bc = 0, nt = 0, ties[ATK_TIES_MAX];
            double bl = 0.0;
            for (int o = 0; o < N; ++o) {
                const int d = def_at(bw, n, (size_t)o, N, 1, seed + 7919u * (uint64_t)(o + 1),
                                     pfa, &l, M, basis, &comp);
                if (d > bd) { bd = d; bo = o; bl = l; bc = comp; nt = 0; ties[nt++] = o; }
                else if (d > 0 && d == bd && nt < ATK_TIES_MAX) ties[nt++] = o;
            }
            const double lo = bl + log10((double)N) + log10_n;
            TRACE("robust: N=%d offsets best o=%d d=%d l10=%.1f\n", N, bo, bd, lo);
            if (bo < 0 || lo > log10_thr) continue;
            f->N = N; f->offset = bo; f->robust = 1; f->w0 = 2 * N; f->d0 = bd; f->complete = bc;
            f->log10_pfa = l0 > lo ? l0 : lo;
            f->nties = nt;
            memcpy(f->ties, ties, (size_t)nt * sizeof *ties);
            found = 1;
        }
    }
    free(basis); free(M); free(dv);
    return found;
}

/* ---- block codes ---------------------------------------------------------------- */

/* G rows (k of nw words) cyclic under H? */
static int is_cyclic(const uint64_t *G, int k, const uint64_t *H, int d, int n) {
    const int nw = nw_of(n);
    uint64_t *s = (uint64_t *)calloc((size_t)nw, sizeof *s);
    if (!s) return 0;
    int ok = 1;
    for (int i = 0; i < k && ok; ++i) {
        const uint64_t *c = G + (size_t)i * nw;
        memset(s, 0, (size_t)nw * sizeof *s);
        for (int j = 0; j < n; ++j) {          /* s[j] = c[(j + 1) mod n] */
            const int src = (j + 1) % n;
            if ((c[src >> 6] >> (src & 63)) & 1u) s[j >> 6] |= 1ULL << (j & 63);
        }
        for (int r = 0; r < d && ok; ++r) {
            uint64_t acc = 0;
            for (int q = 0; q < nw; ++q) acc ^= s[q] & H[(size_t)r * nw + q];
            if (atk_parity64(acc)) ok = 0;
        }
    }
    free(s);
    return ok;
}

/* generator polynomial of a cyclic code: gcd(codewords, x^n - 1) */
static void cyclic_gen(const uint64_t *G, int k, int n, poly_t *g) {
    poly_t a, b;
    memset(&a, 0, sizeof a);
    a.w[n / 64] |= 1ULL << (n % 64);
    a.w[0] ^= 1u;                                   /* x^n + 1 */
    for (int i = 0; i < k; ++i) {
        word_poly(G + (size_t)i * nw_of(n), n, &b);
        pgcd(&a, &b);
    }
    *g = a;
}

/* The order of x modulo g: the smallest e <= emax with g | x^e - 1, else 0. */
static int poly_order(const poly_t *g, int emax) {
    const int dg = pdeg(g);
    if (dg < 1 || !(g->w[0] & 1u)) return 0;
    poly_t y;
    memset(&y, 0, sizeof y);
    y.w[0] = 1u;
    for (int e = 1; e <= emax; ++e) {
        /* y *= x mod g */
        int carry = 0;
        for (int q = 0; q < PW; ++q) {
            const int c2 = (int)(y.w[q] >> 63);
            y.w[q] = (y.w[q] << 1) | (uint64_t)carry;
            carry = c2;
        }
        if ((y.w[dg / 64] >> (dg % 64)) & 1u) {
            for (int q = 0; q < PW; ++q) y.w[q] ^= g->w[q];
        }
        int one = (y.w[0] == 1u);
        for (int q = 1; q < PW && one; ++q) if (y.w[q]) one = 0;
        if (one) return e;
    }
    return 0;
}

/* gcd of the code's generator rows as polynomials: a shortened cyclic code
 * is every multiple of g below degree N, so its gcd is g (degree N - k) */
static void rows_gcd(const uint64_t *G, int k, int n, poly_t *g) {
    poly_t b;
    memset(g, 0, sizeof *g);
    for (int i = 0; i < k; ++i) {
        word_poly(G + (size_t)i * nw_of(n), n, &b);
        if (i == 0) *g = b; else pgcd(g, &b);
    }
}

/* Name a cyclic code of length L with generator g. Writes buf; returns
 * 1 for a code the §16c decoder handles (BCH/Hamming, t in *bch_t, field in
 * *bch_f), else 0. */
static int name_cyclic(const poly_t *g, int L, int k, char *buf, size_t cap, int *bch_t,
                       uint32_t *bch_f, int *bch_m) {
    const int dg = pdeg(g);
    if (dg == 1) { snprintf(buf, cap, "single parity check (%d,%d)", L, k); return 0; }
    if (k == 1) { snprintf(buf, cap, "repetition (%d,1)", L); return 0; }
    if (L == 23 && dg == 11 && (g->w[0] == 0xC75 || g->w[0] == 0xAE3)) {
        snprintf(buf, cap, "Golay(23,12)"); return 0;
    }
    for (int m = 3; m <= 9; ++m) {
        if (L != (1 << m) - 1) continue;
        for (uint32_t f = (1u << m) | 1u; f < (2u << m); f += 2) {
            if (!is_primitive(f, m)) continue;
            /* several designed t can share one generator (BCH(31,11) is t = 4
             * and t = 5): the largest is the code's guaranteed distance */
            int tbest = 0;
            for (int t = 1; 2 * t < L; ++t) {
                atkdsp_bch *b = atkdsp_bch_create(m, t, 0, f);
                if (!b) break;
                int bn, bk;
                atkdsp_bch_params(b, &bn, &bk, NULL);
                if (bn - bk > dg) { atkdsp_bch_destroy(b); break; }
                if (bn - bk == dg) {
                    uint8_t gen[640];
                    atkdsp_bch_generator(b, gen, sizeof gen);
                    poly_t q;
                    memset(&q, 0, sizeof q);
                    for (int i = 0; i <= dg; ++i) if (gen[i]) q.w[(dg - i) / 64] |= 1ULL << ((dg - i) % 64);
                    if (!memcmp(&q, g, sizeof q)) tbest = t;
                }
                atkdsp_bch_destroy(b);
            }
            if (tbest) {
                if (tbest == 1) snprintf(buf, cap, "Hamming(%d,%d)", L, k);
                else snprintf(buf, cap, "BCH(%d,%d) t=%d", L, k, tbest);
                *bch_t = tbest; *bch_f = f; *bch_m = m;
                return 1;
            }
        }
    }
    snprintf(buf, cap, "cyclic (%d,%d)", L, k);
    return 0;
}

/* ---- cyclic structure: which irreducible polynomials divide the codewords ----
 * The noisy-stream search for cyclic codes (and their extended and shortened
 * forms). Every codeword of a cyclic code is a multiple of its generator g,
 * so each irreducible factor q of g divides every error-free codeword, while
 * it divides a random word with probability 2^-deg(q). Counting divisible
 * rows needs no elimination, so one bad row costs one row, not the whole
 * subset: it works while a fair share of codewords are clean, (1 - p)^n,
 * which reaches 1e-3 even at n = 511.
 *
 * Divisibility of every window at once. With P_t the polynomial of bits
 * 0..t-1 (first bit highest degree), the window W of bits [t, t + L) is
 * P_{t+L} - x^L P_t, so q | W exactly when Q_{t+L} = Q_t, where
 * Q_t = x^-t P_t mod q = sum_{i<t} b_i x^-(i+1) mod q: a running XOR of
 * precomputed powers of x^-1 (q(0) = 1, so x is invertible). One pass per
 * factor gives every window's test as one comparison. */
#define CYC_DMAX 11                 /* factors up to degree 11 (Golay's)          */
#define CYC_R0   24                 /* rows per (length, offset) in the scan      */
#define CYC_RMIN 8
#define CYC_SMIN 3.0                /* a factor adds to the score from 10^-3      */
#define ATK_CYC_PMAX 0.08           /* the clean-codeword floor: what BER 8 % leaves */
#define ATK_CYC_CMIN 0.1            /* ... and never below 10 % of the codewords      */
/* The clean share of length-L codewords the root counts must imply: what BER
 * ATK_CYC_PMAX leaves, (1 - p)^L, and at least ATK_CYC_CMIN. Below that the
 * counts cannot tell a code from structure that fits a few rows only — one
 * phase of a wrong length's alignment cycle (Golay(24,12) seen at length 47
 * fits 5 phases of 24: 7 % of the rows). At the limits the header states the
 * clean share is 23 % or more. */
static double cyc_floor(int L) {
    const double f = pow(1.0 - ATK_CYC_PMAX, (double)L);
    return f > ATK_CYC_CMIN ? f : ATK_CYC_CMIN;
}
typedef struct { uint16_t q, e; uint8_t d, prim; } irr_t;

static uint32_t pmod32(uint32_t a, uint32_t q) {
    int dq = 31; while (!((q >> dq) & 1u)) --dq;
    for (int i = 31; i >= dq; --i) if ((a >> i) & 1u) a ^= q << (i - dq);
    return a;
}
/* every irreducible polynomial of degree 1..CYC_DMAX except x, by degree */
static int irr_list(irr_t *out, int cap) {
    int m = 0;
    for (int d = 1; d <= CYC_DMAX; ++d)
        for (uint32_t f = (1u << d) | 1u; f < (2u << d); f += 2) {
            int irr = 1;
            for (int i = 0; i < m && irr; ++i) {
                if (2 * out[i].d > d) break;
                if (pmod32(f, out[i].q) == 0) irr = 0;
            }
            if (!irr || m >= cap) continue;
            uint32_t y = pmod32(2u, f);
            int e = 1;
            while (y != 1u && e < (1 << d)) { y <<= 1; if (y >> d) y ^= f; ++e; }
            out[m].q = (uint16_t)f; out[m].d = (uint8_t)d; out[m].e = (uint16_t)e;
            out[m].prim = (uint8_t)(e == (1 << d) - 1);
            ++m;
        }
    return m;
}
/* Q_t for t = 0..T (T + 1 values) over b[0..T), for factor q */
static void cyc_q(const uint8_t *b, size_t T, const irr_t *f, uint16_t *v, uint16_t *Q) {
    const uint32_t q = f->q;
    uint32_t y = (q ^ 1u) >> 1;                       /* x^-1 */
    for (int i = 0; i < f->e; ++i) {
        v[i] = (uint16_t)y;
        y = (y & 1u) ? (y ^ q) >> 1 : y >> 1;         /* times x^-1 */
    }
    uint16_t acc = 0;
    int r = 0;
    Q[0] = 0;
    for (size_t t = 0; t < T; ++t) {
        if (b[t] & 1u) acc ^= v[r];
        Q[t + 1] = acc;
        if (++r == f->e) r = 0;
    }
}
static int field_m(int L) { int m = 1; while ((1 << m) - 1 < L) ++m; return m; }
/* Is factor f tried for windows of length L? A factor of x^L - 1 (cyclic
 * codes of length L) or, for codes shortened from 2^m - 1, a primitive
 * polynomial of the field that first holds L or the next one. var 1 (the
 * first L = N - 1 bits of each N: extended codes) only the cyclic case. */
static int cyc_relevant(const irr_t *f, int L, int var) {
    if (L % f->e == 0) return 1;
    if (var) return 0;
    const int m = field_m(L);
    return f->prim && (f->d == m || f->d == m + 1) && f->e > L;
}

typedef struct {
    int N, offset, var;             /* var 1: the cyclic part is the first N - 1 bits */
    double score, smax, log10_pfa;
} cyc_find;

/* Scan lengths n_lo..n_hi and their offsets; the best (N, offset, var) by the
 * summed significance of the factors that divide its rows. Returns 1 when
 * the strongest factor there beats pfa after counting every test, else 0. */
static int cyclic_scan(const uint8_t *b, size_t n, int n_lo, int n_hi, double pfa, cyc_find *cf) {
    memset(cf, 0, sizeof *cf);
    irr_t irr[512];
    const int ni = irr_list(irr, 512);
    if (n_lo < 3) n_lo = 3;
    if (n_hi < n_lo) return 0;
    /* rows per length; the largest prefix any length needs */
    int *R0 = (int *)malloc((size_t)(n_hi + 1) * sizeof *R0);
    size_t *base = (size_t *)malloc((size_t)(n_hi + 2) * sizeof *base);
    if (!R0 || !base) { free(R0); free(base); return -1; }
    size_t T = 0, tot = 0;
    for (int N = 0; N <= n_hi; ++N) {
        R0[N] = 0;
        base[N] = tot;
        if (N < n_lo || (size_t)N > n) continue;
        size_t r = (n - (size_t)N + 1) / (size_t)N;
        if (r > CYC_R0) r = CYC_R0;
        if (r < CYC_RMIN) continue;
        R0[N] = (int)r;
        tot += 2 * (size_t)N;                          /* score per offset, var 0 and 1 */
        const size_t need = (r + 1) * (size_t)N;
        if (need > T) T = need;
    }
    base[n_hi + 1] = tot;
    if (T > n) T = n;
    double *score = (double *)calloc(tot ? tot : 1, sizeof *score);
    double *smax = (double *)calloc(tot ? tot : 1, sizeof *smax);
    uint16_t *Q = (uint16_t *)malloc((T + 1) * sizeof *Q);
    uint16_t *v = (uint16_t *)malloc(2048 * sizeof *v);
    uint8_t *cnt = (uint8_t *)malloc((size_t)n_hi + 1);
    /* significance -log10 P(Bin(r, 2^-d) >= c), for r = CYC_RMIN..CYC_R0 */
    double (*sig)[CYC_DMAX + 1][CYC_R0 + 1] =
        (double (*)[CYC_DMAX + 1][CYC_R0 + 1])malloc((CYC_R0 + 1) * sizeof *sig);
    if (!score || !smax || !Q || !v || !cnt || !sig || !tot) {
        free(R0); free(base); free(score); free(smax); free(Q); free(v); free(cnt); free(sig);
        return tot ? -1 : 0;
    }
    for (int r = CYC_RMIN; r <= CYC_R0; ++r)
        for (int d = 1; d <= CYC_DMAX; ++d)
            for (int c = 0; c <= r; ++c)
                sig[r][d][c] = -atk_log10_binom_tail((double)r, (double)c, -(double)d * LN2);
    double ntests = 0.0;
    for (int i = 0; i < ni; ++i) {
        const irr_t *f = &irr[i];
        int any = 0;
        for (int N = n_lo; N <= n_hi && !any; ++N)
            if (R0[N] && (cyc_relevant(f, N, 0) || (N > 3 && cyc_relevant(f, N - 1, 1)))) any = 1;
        if (!any) continue;
        cyc_q(b, T, f, v, Q);
        for (int N = n_lo; N <= n_hi; ++N) {
            const int r = R0[N];
            if (!r) continue;
            for (int var = 0; var < 2; ++var) {
                const int L = N - var;
                if (L < 3 || !cyc_relevant(f, L, var)) continue;
                memset(cnt, 0, (size_t)N);
                for (int k = 0; k < r; ++k) {
                    const uint16_t *a = Q + (size_t)k * (size_t)N, *c = a + L;
                    for (int o = 0; o < N; ++o) cnt[o] = (uint8_t)(cnt[o] + (a[o] == c[o]));
                }
                double *sc = score + base[N] + (size_t)var * (size_t)N;
                double *sm = smax + base[N] + (size_t)var * (size_t)N;
                for (int o = 0; o < N; ++o) {
                    const double s = sig[r][f->d][cnt[o]];
                    if (s >= CYC_SMIN) sc[o] += s;
                    if (s > sm[o]) sm[o] = s;
                }
                ntests += (double)N;
            }
        }
    }
    /* the best: highest score; a longer length must beat the incumbent by a
     * quarter (a multiple of the true length scores nearly as well) */
    double best = 0.0;
    for (int N = n_lo; N <= n_hi; ++N) {
        if (!R0[N]) continue;
        int bo = -1, bv = 0;
        double bs = 0.0;
        for (int var = 0; var < 2; ++var)
            for (int o = 0; o < N; ++o) {
                const double s = score[base[N] + (size_t)var * (size_t)N + (size_t)o];
                if (s > bs) { bs = s; bo = o; bv = var; }
            }
        if (bo < 0) continue;
        if (best == 0.0 || bs > 1.25 * best) {
            best = bs;
            cf->N = N; cf->offset = bo; cf->var = bv; cf->score = bs;
            cf->smax = smax[base[N] + (size_t)bv * (size_t)N + (size_t)bo];
        }
    }
    cf->log10_pfa = -cf->smax + log10(ntests > 1.0 ? ntests : 1.0);
    TRACE("cyc scan: N=%d off=%d var=%d score=%.1f smax=%.1f l10=%.1f ntests=%.0f\n", cf->N, cf->offset, cf->var, cf->score, cf->smax, cf->log10_pfa, ntests);
    free(R0); free(base); free(score); free(smax); free(Q); free(v); free(cnt); free(sig);
    return best > 0.0 && cf->log10_pfa <= log10(pfa);
}

/* The generator at a found (N, offset, var): every irreducible polynomial
 * counted over up to 1024 aligned rows; q is a factor of g when its rate is
 * nearer the clean-row rate c + (1 - c) 2^-d than chance 2^-d, and
 * significant. c is estimated from the strongest factor. Returns deg g (0 =
 * nothing) and g (bit i = x^i); *clean the clean-row fraction. */
static int cyclic_generator(const uint8_t *b, size_t n, const cyc_find *cf, poly_t *g, double *clean) {
    const int N = cf->N, L = N - cf->var;
    memset(g, 0, sizeof *g);
    *clean = 0.0;
    size_t R = (n - (size_t)cf->offset) / (size_t)N;
    if (R > 1024) R = 1024;
    if (R * (size_t)N > 262144) R = 262144 / (size_t)N;    /* bounds the 411 passes */
    if (R < CYC_RMIN) return 0;
    irr_t irr[512];
    const int ni = irr_list(irr, 512);
    const size_t T = R * (size_t)N;
    uint16_t *Q = (uint16_t *)malloc((T + 1) * sizeof *Q);
    uint16_t *v = (uint16_t *)malloc(2048 * sizeof *v);
    int *cnt = (int *)malloc((size_t)ni * sizeof *cnt);
    if (!Q || !v || !cnt) { free(Q); free(v); free(cnt); return -1; }
    const uint8_t *s = b + cf->offset;
    /* each factor's implied clean-row rate c_q = (f_q - 2^-d) / (1 - 2^-d);
     * every factor of g sees the same clean rows, so the same c */
    double cbest = 0.0;
    double *cq = (double *)malloc((size_t)ni * sizeof *cq), *sq = (double *)malloc((size_t)ni * sizeof *sq);
    if (!cq || !sq) { free(Q); free(v); free(cnt); free(cq); free(sq); return -1; }
    for (int i = 0; i < ni; ++i) {
        cyc_q(s, T, &irr[i], v, Q);
        int c = 0;
        for (size_t k = 0; k < R; ++k) c += Q[k * (size_t)N + (size_t)L] == Q[k * (size_t)N];
        cnt[i] = c;
        const double pi = ldexp(1.0, -irr[i].d), fr = (double)c / (double)R;
        cq[i] = (fr - pi) / (1.0 - pi);
        sq[i] = sqrt((fr * (1.0 - fr) + 1.0 / (double)R) / (double)R) / (1.0 - pi);
        if (cq[i] - 2.0 * sq[i] > cbest) cbest = cq[i] - 2.0 * sq[i];   /* a lower bound */
    }
    const double thr = -3.0 - log10((double)ni);
    poly_t one;
    memset(&one, 0, sizeof one);
    *g = one;
    g->w[0] = 1u;
    int dg = 0;
    for (int i = 0; i < ni; ++i) {
        if (cq[i] + 3.0 * sq[i] < cbest) continue;          /* divides too few: not a factor */
        if (atk_log10_binom_tail((double)R, (double)cnt[i], -(double)irr[i].d * LN2) > thr) continue;
        if (dg + irr[i].d >= L) continue;
        TRACE("cyc gen: q=0x%x d=%d e=%d cnt=%d/%d cbest=%.3f\n", irr[i].q, irr[i].d, irr[i].e, cnt[i], (int)R, cbest);
        /* g *= q */
        poly_t qq, acc;
        memset(&qq, 0, sizeof qq);
        memset(&acc, 0, sizeof acc);
        qq.w[0] = irr[i].q;
        for (int e = 0; e <= dg; ++e)
            if ((g->w[e / 64] >> (e % 64)) & 1u) pxor_shift(&acc, &qq, e);
        *g = acc;
        dg += irr[i].d;
    }
    free(Q); free(v); free(cnt); free(cq); free(sq);
    *clean = cbest;
    return dg;
}

/* Checks (canonical, width N) of the cyclic code found: generator rows
 * x^i g, and for var 1 the N-th bit as the overall parity when the clean rows
 * say so (else it is left free). Returns the number of checks. */
static int cyclic_checks(const uint8_t *b, size_t n, cyc_find *cf, const poly_t *g, int dg,
                         uint64_t *H, int cap) {
    const int N = cf->N, L = N - cf->var, k = L - dg, nw = nw_of(N);
    if (k < 1) return 0;
    uint64_t *Gm = (uint64_t *)calloc(((size_t)k + 1) * (size_t)nw, sizeof *Gm);
    if (!Gm) return -1;
    int ext = 0;                    /* 1: parity last, 2: parity first */
    if (cf->var) {
        /* over rows divisible by g (clean): does the extra bit close the parity? */
        size_t R = (n - (size_t)cf->offset) / (size_t)N;
        if (R > 1024) R = 1024;
        long last = 0, first = 0, tot = 0;
        for (size_t r = 0; r < R; ++r) {
            const uint8_t *w = b + cf->offset + r * (size_t)N;
            poly_t p, gg = *g;
            memset(&p, 0, sizeof p);
            int par = 0;
            for (int j = 0; j < L; ++j) if (w[j] & 1u) { const int e = L - 1 - j; p.w[e / 64] |= 1ULL << (e % 64); par ^= 1; }
            pmod(&p, &gg);
            if (pdeg(&p) >= 0) continue;
            ++tot;
            last += (w[L] & 1u) == (uint8_t)par;
            if (cf->offset + r * (size_t)N >= 1) first += (w[-1] & 1u) == (uint8_t)par;
        }
        if (tot >= CYC_RMIN && last >= 0.9 * tot) ext = 1;
        else if (tot >= CYC_RMIN && first >= 0.9 * tot) ext = 2;
    }
    const int sh = (ext == 2) ? 1 : 0;              /* cyclic part starts at bit sh */
    for (int i = 0; i < k; ++i) {
        uint64_t *row = Gm + (size_t)i * nw;
        int par = 0;
        for (int e = 0; e <= dg; ++e)
            if ((g->w[e / 64] >> (e % 64)) & 1u) {
                const int deg = e + i, j = L - 1 - deg + sh;   /* coefficient of x^deg */
                row[j >> 6] |= 1ULL << (j & 63);
                par ^= 1;
            }
        if (ext) { const int j = (ext == 1) ? L : 0; if (par) row[j >> 6] |= 1ULL << (j & 63); }
    }
    if (ext == 2) cf->offset = (cf->offset + N - 1) % N;
    int kr = k;
    if (!ext && cf->var) {          /* the N-th bit is free: one more information bit */
        Gm[(size_t)k * nw + (L >> 6)] |= 1ULL << (L & 63);
        kr = k + 1;
    }
    const int d = atk_gf2_nullspace(Gm, kr, N, H, cap);
    free(Gm);
    return d;
}

/* ---- what the statistics assume ---------------------------------------------
 * Every log10_pfa here is computed for uniform, memoryless bits apart from
 * the code. Two cheap tests refuse a stream that is not: a bias in the count
 * of ones, and a short linear recurrence — Berlekamp-Massey over the first
 * 8192 bits: an LFSR, a constant or a periodic stream has a linear
 * complexity far below half the bits, while coded random data reaches it. */
#define ATK_BIAS_MAX 0.1            /* |P(1) - 1/2| beyond which nothing is named */
#define ATK_LC_BITS  8192
static int stream_unfit(const uint8_t *b, size_t n, double *p1, int *lc) {
    *p1 = 0.5;
    *lc = -1;
    /* the recurrence first: a constant stream is the shortest recurrence */
    const size_t m = n < ATK_LC_BITS ? n : ATK_LC_BITS;
    if (m >= 256) {
        uint8_t *poly = (uint8_t *)malloc(m + 1);
        if (poly) {
            const ptrdiff_t L = atkdsp_bm(b, m, poly, m + 1);
            free(poly);
            if (L >= 0) {
                *lc = (int)L;
                if (2 * (size_t)L + 64 < m) return 2;
            }
        }
    }
    if (n >= 2000) {
        size_t ones = 0;
        for (size_t i = 0; i < n; ++i) ones += b[i] & 1u;
        *p1 = (double)ones / (double)n;
        if (fabs(*p1 - 0.5) > ATK_BIAS_MAX) return 1;
    }
    return 0;
}

/* bits [start, start + len) of a packed vector of nsrc words, into dst */
static void bits_slice(const uint64_t *src, int nsrc, int start, int len, uint64_t *dst) {
    const int nwd = nw_of(len);
    for (int q = 0; q < nwd; ++q) {
        const int p = start + 64 * q, wq = p >> 6, r = p & 63;
        uint64_t v = wq < nsrc ? src[wq] >> r : 0;
        if (r && wq + 1 < nsrc) v |= src[wq + 1] << (64 - r);
        dst[q] = v;
    }
    if (len & 63) dst[nwd - 1] &= (1ULL << (len & 63)) - 1;
}
static int top_w(const uint64_t *v, int nw) {
    for (int q = nw - 1; q >= 0; --q)
        if (v[q]) { int b = 63; while (!((v[q] >> b) & 1u)) --b; return 64 * q + b; }
    return -1;
}
static int low_w(const uint64_t *v, int nw) {
    for (int q = 0; q < nw; ++q)
        if (v[q]) { int b = 0; while (!((v[q] >> b) & 1u)) ++b; return 64 * q + b; }
    return -1;
}

/* Is the code m = N / L copies of one length-L code? Then every canonical
 * check lies inside one L-block, and every block carries the same checks
 * (canonical order keeps a block's checks together, highest block first).
 * A search can land on a multiple of the true length; this undoes it.
 * Returns the block's check count, its checks in Hb (canonical, width L). */
static int block_copies(const uint64_t *H, int d, int N, int L, uint64_t *Hb) {
    const int nw = nw_of(N), nwl = nw_of(L), m = N / L;
    if (m < 2 || N % L || d % m) return 0;
    const int d0 = d / m;
    if (d0 < 1) return 0;
    uint64_t *t = (uint64_t *)malloc((size_t)nwl * sizeof *t);
    if (!t) return 0;
    int ok = 1;
    for (int i = 0; i < d && ok; ++i) {
        const uint64_t *h = H + (size_t)i * nw;
        const int top = top_w(h, nw), low = low_w(h, nw), blk = top / L;
        if (low / L != blk || blk != m - 1 - i / d0) { ok = 0; break; }
        bits_slice(h, nw, blk * L, L, t);
        if (blk == 0) memcpy(Hb + (size_t)(i % d0) * nwl, t, (size_t)nwl * sizeof *t);
    }
    /* every block's checks equal block 0's */
    for (int i = 0; i < d && ok; ++i) {
        const uint64_t *h = H + (size_t)i * nw;
        bits_slice(h, nw, (m - 1 - i / d0) * L, L, t);
        if (memcmp(t, Hb + (size_t)(i % d0) * nwl, (size_t)nwl * sizeof *t)) ok = 0;
    }
    free(t);
    return ok ? d0 : 0;
}

/* fraction of R rows of width N from bit `off` (stride N) that satisfy every check */
static double frac_valid(const uint64_t *bw, size_t off, int N, int R, const uint64_t *H, int d, uint64_t *row) {
    const int nw = nw_of(N);
    int v = 0;
    for (int r = 0; r < R; ++r) {
        atk_gf2_rows(bw, off + (size_t)r * (size_t)N, N, 1, N, row);
        int ok = 1;
        for (int c = 0; c < d && ok; ++c) {
            uint64_t acc = 0;
            for (int q = 0; q < nw; ++q) acc ^= row[q] & H[(size_t)c * nw + q];
            if (atk_parity64(acc)) ok = 0;
        }
        v += ok;
    }
    return R ? (double)v / (double)R : 0.0;
}

/* Checks that hold as well P bits along (P a proper divisor of N) are not a
 * block code's: a convolutional stream, or a recurrence. The smallest such
 * P, or 0. "As well": the shifted rows keep 80 % of the aligned rows'
 * excess over chance (a repetition code, the nearest block case, keeps 50 %). */
static int sliding_period(const uint64_t *bw, size_t n, size_t off, int N, const uint64_t *H, int d) {
    if (off + (size_t)N - 1 >= n) return 0;
    size_t fit = (n - off - (size_t)(N - 1)) / (size_t)N;
    const int R = (int)(fit < 1024 ? fit : 1024);
    if (R < 16) return 0;
    uint64_t *row = (uint64_t *)malloc((size_t)nw_of(N) * sizeof *row);
    if (!row) return 0;
    const double chance = ldexp(1.0, -(d < 60 ? d : 60));
    const double e0 = frac_valid(bw, off, N, R, H, d, row) - chance;
    int P = 0;
    if (e0 > 0.1)
        for (int p = 1; p < N && !P; ++p) {
            if (N % p) continue;
            const double ep = frac_valid(bw, off + (size_t)p, N, R, H, d, row) - chance;
            if (ep >= 0.8 * e0) P = p;
        }
    free(row);
    return P;
}

/* Do the failing codewords fail the way channel noise makes them? Two tests.
 * Noise strikes rows at random, so which rows fail has no pattern by row
 * index; structure that fits some rows only (a wrong length, a code outside
 * the range) fails by the phase of an alignment that cycles from row to row:
 * the G-test of periods 2..64 (ATK_PERIODIC_L10). And at a BER low enough to
 * name a code, most failing words carry ONE error, so their syndrome is a
 * column of H, where partial structure fails with arbitrary syndromes. With
 * V the valid fraction, p = 1 - V^(1/N) and the expected share of
 * single-error syndromes among failures q = N p (1-p)^(N-1) / (1 - (1-p)^N)
 * (more, never less, when other patterns share a column's syndrome). Refused
 * when far fewer are seen: log10 P(Bin(F, q) <= S) <= -4 and S < 0.8 q F.
 * Returns 1 = noise-like (or too few failures to tell), 0 = not. */
static int noise_like(const uint64_t *rows, int R, int N, const uint64_t *H, int d) {
    const int nw = nw_of(N), ns = nw_of(d);
    uint64_t *col = (uint64_t *)calloc((size_t)N * (size_t)ns, sizeof *col);
    uint64_t *sy = (uint64_t *)malloc((size_t)ns * sizeof *sy);
    uint8_t *ok = (uint8_t *)malloc((size_t)R + 1);
    if (!col || !sy || !ok) { free(col); free(sy); free(ok); return 1; }
    for (int c = 0; c < d; ++c)
        for (int j = 0; j < N; ++j)
            if ((H[(size_t)c * nw + (j >> 6)] >> (j & 63)) & 1u) col[(size_t)j * ns + (c >> 6)] |= 1ULL << (c & 63);
    long F = 0, S = 0;
    for (int i = 0; i < R; ++i) {
        memset(sy, 0, (size_t)ns * sizeof *sy);
        int any = 0;
        for (int c = 0; c < d; ++c) {
            uint64_t acc = 0;
            for (int q = 0; q < nw; ++q) acc ^= rows[(size_t)i * nw + q] & H[(size_t)c * nw + q];
            if (atk_parity64(acc)) { sy[c >> 6] |= 1ULL << (c & 63); any = 1; }
        }
        ok[i] = (uint8_t)!any;
        if (!any) continue;
        ++F;
        for (int j = 0; j < N; ++j)
            if (col[(size_t)j * ns] == sy[0] && !memcmp(col + (size_t)j * ns, sy, (size_t)ns * sizeof *sy)) { ++S; break; }
    }
    const double lp = atk_ok_period_l10(ok, R, 64);
    free(col); free(sy); free(ok);
    TRACE("noise_like: row-pattern l10=%.1f\n", lp);
    if (lp <= ATK_PERIODIC_L10) return 0;
    if (F < 8) return 1;
    const double V = (double)(R - F) / (double)R;
    const double p = V > 0.0 ? 1.0 - pow(V, 1.0 / (double)N) : 1.0;
    const double q = p < 1.0 ? (double)N * p * pow(1.0 - p, (double)(N - 1)) / (1.0 - pow(1.0 - p, (double)N)) : 0.0;
    if (!(q > 0.0) || q >= 1.0) return q >= 1.0 ? 1 : (S >= F / 2);
    const double l10 = atk_log10_binom_tail((double)F, (double)(F - S), log(1.0 - q));
    TRACE("noise_like: F=%ld S=%ld q=%.3f l10=%.1f\n", F, S, q, l10);
    return !(l10 <= -4.0 && (double)S < 0.8 * q * (double)F);
}

/* name an extended code: N bits whose all-ones vector is a check, cyclic
 * after the parity bit (last, or first) is punctured */
static int name_extended(const uint64_t *Gm, int k, int N, char *name, size_t cap, poly_t *g) {
    const int nw = nw_of(N), L = N - 1, nwl = nw_of(L);
    if (L < 2) return 0;
    uint64_t *Gp = (uint64_t *)calloc((size_t)k * (size_t)nwl, sizeof *Gp);
    uint64_t *Hp = (uint64_t *)malloc((size_t)L * (size_t)nwl * sizeof *Hp);
    int done = 0;
    for (int first = 0; first < 2 && Gp && Hp && !done; ++first) {
        memset(Gp, 0, (size_t)k * (size_t)nwl * sizeof *Gp);
        for (int i = 0; i < k; ++i) bits_slice(Gm + (size_t)i * nw, nw, first, L, Gp + (size_t)i * nwl);
        const int rk = atk_gf2_canon(Gp, k, L);
        const int dp = atk_gf2_nullspace(Gp, rk, L, Hp, L);
        if (rk == k && dp == L - k && is_cyclic(Gp, k, Hp, dp, L)) {
            char inner[40];
            int bt, bm;
            uint32_t bf;
            cyclic_gen(Gp, k, L, g);
            name_cyclic(g, L, k, inner, sizeof inner, &bt, &bf, &bm);
            /* restate the length, keep the rest ("BCH(31,21) t=2" ->
             * "extended BCH(32,21) t=2") */
            const char *lp = strchr(inner, '('), *rp = lp ? strchr(lp, ')') : NULL;
            if (lp && rp) set_name(name, cap, "extended %.*s(%d,%d)%s%s", (int)(lp - inner), inner, N, k,
                                   rp + 1, first ? " parity first" : "");
            else set_name(name, cap, "extended %s", inner);
            done = 1;
        }
    }
    free(Gp); free(Hp);
    return done;
}

/* The d checks in H (canonical, nw words of width N): the single-position
 * ones are a payload's fixed bits (a position always 0), not the code's,
 * and left in they refused true codes on short captures as "sliding" or
 * "fails unlike noise" (plan §D4, 2026-10-28). Compacted out of H into
 * `fixed` (a mask, nw words); returns the checks kept. Twin: _set_aside_fixed. */
static int set_aside_fixed(uint64_t *H, int d, int N, uint64_t *fixed) {
    const int nw = nw_of(N);
    memset(fixed, 0, (size_t)nw * sizeof *fixed);
    d = atk_gf2_canon(H, d, N);
    int kept = 0;
    for (int r = 0; r < d; ++r) {
        uint64_t *row = H + (size_t)r * nw;
        if (vec_weight(row, nw) == 1) { for (int q = 0; q < nw; ++q) fixed[q] |= row[q]; continue; }
        if (kept != r) memcpy(H + (size_t)kept * nw, row, (size_t)nw * sizeof *row);
        ++kept;
    }
    return kept;
}

/* dmin over the nonzero codewords of k generators (N <= 64 here, one word). */
static int gm_dmin(const uint64_t *Gm, int k, int nw, int N) {
    int best = N + 1;
    uint64_t cw = 0;
    for (uint32_t i = 1; i < (1u << k); ++i) {
        int bit = 0;
        while (!((i >> bit) & 1u)) ++bit;
        cw ^= Gm[(size_t)bit * nw];
        const int w = atk_popcount64(cw);
        if (w < best) best = w;
    }
    return best;
}

/* The stream cannot say whether a check includes a fixed position (its bit
 * is always 0 there), so the checks come back without them and a Hamming
 * code reads as "linear" with a smaller dmin. Every way of putting the fixed
 * positions back into the d checks is tried, up to 4096: the first that names
 * a cyclic or extended-cyclic code (not a generic "cyclic (n,k)") is taken;
 * failing that, for a small code (N <= 32, k <= 12, up to 256 ways) the way
 * with the largest dmin (the first of equals); else H is left as it is.
 * Twin: _restore_fixed. Returns 0, or ATKDSP_E_NOMEM. */
static int restore_fixed(uint64_t *H, int d, int N, const uint64_t *fixed) {
    const int nw = nw_of(N);
    int pos[512], np = 0;
    for (int j = 0; j < N; ++j) if ((fixed[j >> 6] >> (j & 63)) & 1u) pos[np++] = j;
    if (!np || d <= 0 || d * np > 12) return 0;
    const int k = N - d, ncell = d * np;
    const int small = N <= 32 && k <= 12 && ncell <= 8;
    uint64_t *Hc = (uint64_t *)malloc((size_t)d * (size_t)nw * sizeof *Hc);
    uint64_t *Gm = (uint64_t *)malloc((size_t)N * (size_t)nw * sizeof *Gm);
    uint64_t *best = (uint64_t *)malloc((size_t)d * (size_t)nw * sizeof *best);
    if (!Hc || !Gm || !best) { free(Hc); free(Gm); free(best); return ATKDSP_E_NOMEM; }
    int best_dmin = -1, have_best = 0;
    if (small) {
        const int kk = atk_gf2_nullspace(H, d, N, Gm, N);
        if (kk == k) { best_dmin = gm_dmin(Gm, k, nw, N); memcpy(best, H, (size_t)d * nw * sizeof *best); have_best = 1; }
    }
    for (int combo = 1; combo < (1 << ncell); ++combo) {
        memcpy(Hc, H, (size_t)d * (size_t)nw * sizeof *Hc);
        for (int c = 0; c < ncell; ++c)
            if ((combo >> c) & 1) {
                const int i = c / np, j = pos[c % np];
                Hc[(size_t)i * nw + (j >> 6)] ^= 1ULL << (j & 63);
            }
        if (atk_gf2_canon(Hc, d, N) != d) continue;
        const int kk = atk_gf2_nullspace(Hc, d, N, Gm, N);
        if (kk != k) continue;
        char nm[48] = "";
        if (is_cyclic(Gm, k, Hc, d, N)) {
            poly_t g;
            int bt = 0, bm = 0;
            uint32_t bf = 0;
            cyclic_gen(Gm, k, N, &g);
            name_cyclic(&g, N, k, nm, sizeof nm, &bt, &bf, &bm);
        } else {
            poly_t g;
            if (!name_extended(Gm, k, N, nm, sizeof nm, &g)) nm[0] = 0;
        }
        if (nm[0] && strncmp(nm, "cyclic (", 8) && strncmp(nm, "extended cyclic (", 17)) {
            memcpy(H, Hc, (size_t)d * (size_t)nw * sizeof *H);
            free(Hc); free(Gm); free(best);
            return 0;
        }
        if (small) {
            const int dm = gm_dmin(Gm, k, nw, N);
            if (dm > best_dmin) { best_dmin = dm; memcpy(best, Hc, (size_t)d * nw * sizeof *best); have_best = 1; }
        }
    }
    if (have_best) memcpy(H, best, (size_t)d * (size_t)nw * sizeof *H);
    free(Hc); free(Gm); free(best);
    return 0;
}

int atkdsp_block_code_id(const uint8_t *b, size_t n, int n_lo, int n_hi, double pfa,
                         uint64_t seed, atkdsp_blockid *out, uint8_t *G, size_t gcap,
                         uint8_t *gpoly, size_t pcap) {
    if ((!b && n) || !out || n_lo < 2 || n_hi < n_lo || n_hi > 512 || n > (size_t)INT32_MAX) return ATKDSP_E_ARG;
    if (!(pfa > 0.0)) pfa = 1e-3;
    memset(out, 0, sizeof *out);
    out->ber_est = out->agreement = out->valid_words = NAN;
    snprintf(out->name, sizeof out->name, "no code found");
    {
        double p1;
        int lc;
        const int u = stream_unfit(b, n, &p1, &lc);
        if (u == 1) { set_name(out->name, sizeof out->name, "no code found (biased: P(1) = %.2f)", p1); return 0; }
        if (u == 2) { snprintf(out->name, sizeof out->name, "no block code (linear recurrence)"); return 0; }
    }
    size_t nbw;
    uint64_t *bw = atk_bits_pack(b, n, &nbw);
    if (!bw) return ATKDSP_E_NOMEM;
    /* exact elimination first (noiseless streams), then the cyclic-root
     * count (cyclic codes in noise, any length), then the robust search */
    blk_find f;
    cyc_find cf;
    poly_t cg;
    int cyc = 0, cdg = 0;
    int rc = block_search(bw, n, n_lo, n_hi, pfa, seed, &f, BS_EXACT);
    if (rc < 0) { free(bw); return ATKDSP_E_NOMEM; }
    if (rc == 1 && f.slide) {
        /* every offset (or every few) carries the same checks: the stream is
         * a linear recurrence or a convolutional code, not a block code */
        out->n = f.N; out->offset = f.offset; out->deficiency = f.d0; out->log10_pfa = f.log10_pfa;
        if (f.slide == 1) snprintf(out->name, sizeof out->name, "no block code (linear recurrence)");
        else snprintf(out->name, sizeof out->name, "no block code (sliding, period %d)", f.slide);
        free(bw);
        return 0;
    }
    if (rc == 0) {
        const int c = cyclic_scan(b, n, n_lo, n_hi, pfa, &cf);
        if (c < 0) { free(bw); return ATKDSP_E_NOMEM; }
        if (c == 1) {
            double clean;
            cdg = cyclic_generator(b, n, &cf, &cg, &clean);
            if (cdg < 0) { free(bw); return ATKDSP_E_NOMEM; }
            /* the clean codewords must be as many as BER ATK_CYC_PMAX would
             * leave, (1 - p)^L: structure that divides only a few windows
             * (a length outside the range, a code seen at a wrong length)
             * is not a code at this length */
            if (cdg > 0 && clean < cyc_floor(cf.N - cf.var)) {
                TRACE("cyc: clean %.3f below the floor %.3f\n", clean, cyc_floor(cf.N - cf.var));
                cdg = 0;
            }
            /* a multiple of the true length scores nearly as well: when g's
             * order divides a shorter length in range, and the count there
             * finds a generator too, that length is the code's */
            const int e = cdg > 0 && !cf.var ? poly_order(&cg, cf.N) : 0;
            if (e > 0 && e < cf.N && cf.N % e == 0) {
                for (int L = e; L < cf.N; L += e) {
                    if (cf.N % L || L < n_lo || L < 3) continue;
                    cyc_find c2 = cf;
                    poly_t g2;
                    double cl2;
                    c2.N = L;
                    c2.offset = cf.offset % L;
                    const int d2 = cyclic_generator(b, n, &c2, &g2, &cl2);
                    if (d2 < 0) { free(bw); return ATKDSP_E_NOMEM; }
                    TRACE("cyc: order %d, length %d instead of %d: dg=%d\n", e, L, cf.N, d2);
                    if (d2 > 0 && cl2 >= cyc_floor(L)) { cf = c2; cg = g2; cdg = d2; break; }
                }
            }
            cyc = cdg > 0;
        }
        if (!cyc) {
            rc = block_search(bw, n, n_lo, n_hi, pfa, seed, &f, BS_ROBUST);
            if (rc <= 0) { free(bw); return rc < 0 ? ATKDSP_E_NOMEM : 0; }
        }
    }
    int N = cyc ? cf.N : f.N, nw = nw_of(N);
    uint64_t *H = (uint64_t *)malloc(2 * (size_t)N * (size_t)nw * sizeof *H);
    uint64_t *Gm = (uint64_t *)malloc((size_t)N * (size_t)nw * sizeof *Gm);
    if (!H || !Gm) { free(bw); free(H); free(Gm); return ATKDSP_E_NOMEM; }
    int d = 0, robust = 0, complete = 1;
    double rob_ber = NAN;
    if (cyc) {
        d = cyclic_checks(b, n, &cf, &cg, cdg, H, N);   /* may move the offset by one */
        if (d < 0) { free(bw); free(H); free(Gm); return ATKDSP_E_NOMEM; }
        f.N = N; f.offset = cf.offset; f.robust = 2; f.log10_pfa = cf.log10_pfa; f.d0 = d;
        robust = 2;
    }
    size_t fit = (n - (size_t)f.offset) / (size_t)N;
    int R = (int)(fit < ATK_ID_ROWS ? fit : ATK_ID_ROWS);
    uint64_t *rows = (uint64_t *)malloc(((size_t)R + 1) * (size_t)nw * sizeof *rows);
    if (!rows) { free(bw); free(H); free(Gm); return ATKDSP_E_NOMEM; }
    atk_gf2_rows(bw, (size_t)f.offset, N, R, N, rows);
    if (cyc) {
        /* a non-cyclic subcode of the cyclic code has more checks: the robust
         * search on the aligned rows adds any it can verify. Checks it adds
         * from an incomplete span make the dimension a lower bound */
        atk_robust_info info;
        const int dr = atk_gf2_robust(rows, R, N, seed ^ 0xC1Cull, pfa, H + (size_t)d * nw, N, &info);
        if (dr > 0) {
            const int d1 = atk_gf2_canon(H, d + dr, N);
            TRACE("cyc: d=%d robust adds dr=%d (new %d) complete=%d\n", d, dr, d1 - d, info.complete);
            if (d1 > d && !info.complete) { complete = 0; rob_ber = info.ber; }
            d = d1;
        }
    } else {
        d = f.robust ? 0 : atk_gf2_nullspace(rows, R, N, H, N);
        if (!f.robust && d != f.d0) d = 0;
        robust = f.robust;
        if (d == 0) {
            atk_robust_info info;
            d = atk_gf2_robust(rows, R, N, seed ^ 0xB10Cull, pfa, H, N, &info);
            robust = 1;
            complete = info.complete;
            rob_ber = info.ber;
        }
    }
    /* m copies of a shorter code (L | N, smallest first): name that code */
    if (d > 0 && complete) {
        for (int L = 2; L < N; ++L) {
            if (N % L || L < n_lo) continue;
            uint64_t *Hb = (uint64_t *)malloc((size_t)d * (size_t)nw_of(L) * sizeof *Hb);
            if (!Hb) break;
            const int d0 = block_copies(H, d, N, L, Hb);
            if (d0 > 0) {
                TRACE("block: %d copies of length %d\n", N / L, L);
                N = L; nw = nw_of(N); d = d0;
                memcpy(H, Hb, (size_t)d0 * (size_t)nw * sizeof *H);
                fit = (n - (size_t)f.offset) / (size_t)N;
                R = (int)(fit < ATK_ID_ROWS ? fit : ATK_ID_ROWS);
                free(rows);
                rows = (uint64_t *)malloc(((size_t)R + 1) * (size_t)nw * sizeof *rows);
                free(Hb);
                if (!rows) { free(bw); free(H); free(Gm); return ATKDSP_E_NOMEM; }
                atk_gf2_rows(bw, (size_t)f.offset, N, R, N, rows);
                break;
            }
            free(Hb);
        }
    }
    rc = 0;
    uint64_t fixed[9] = {0};                 /* nw_of(512) = 8 words, +1 */
    if (d > 0) {
        d = set_aside_fixed(H, d, N, fixed);
        out->fixed_bits = vec_weight(fixed, nw);
    }
    if (d > 0 && !complete) {
        /* checks verified, but no subset was ever clean: the dimension is a
         * lower bound, so the code is not named (plan §D4) */
        out->n = N; out->offset = f.offset; out->deficiency = d; out->robust = robust;
        out->rows_used = R; out->log10_pfa = f.log10_pfa; out->ber_est = rob_ber;
        snprintf(out->name, sizeof out->name, "no code found (n=%d too noisy)", N);
        d = 0;
    }
    if (d > 0 && d < N) {
        const int P = sliding_period(bw, n, (size_t)f.offset, N, H, d);
        if (P) {
            out->n = N; out->offset = f.offset; out->deficiency = d; out->log10_pfa = f.log10_pfa;
            if (P == 1) snprintf(out->name, sizeof out->name, "no block code (same checks at every offset)");
            else snprintf(out->name, sizeof out->name, "no block code (sliding, period %d)", P);
            d = 0;
        }
    }
    if (d > 0 && d < N && !noise_like(rows, R, N, H, d)) {
        out->n = N; out->offset = f.offset; out->deficiency = d; out->log10_pfa = f.log10_pfa;
        out->rows_used = R;
        set_name(out->name, sizeof out->name, "no code found (n=%d fails unlike noise)", N);
        d = 0;
    }
    if (d > 0 && d < N && out->fixed_bits) {
        const int rr = restore_fixed(H, d, N, fixed);
        if (rr) { free(bw); free(H); free(Gm); free(rows); return rr; }
    }
    if (d > 0 && d < N) {
        const int k = N - d;
        const int kk = atk_gf2_nullspace(H, d, N, Gm, N);
        if (kk == k) {
            out->n = N; out->k = k; out->offset = f.offset; out->deficiency = d;
            out->rows_used = R; out->robust = robust; out->log10_pfa = f.log10_pfa;
            /* the channel BER from the checks, and the valid words */
            {
                double *pe = (double *)malloc((size_t)d * sizeof *pe);
                int np = 0, valid = 0;
                for (int r = 0; r < d && pe; ++r) {
                    const int u = atk_gf2_unsat(rows, R, nw, H + (size_t)r * nw);
                    int wt = 0;
                    for (int q = 0; q < nw; ++q) wt += atk_popcount64(H[(size_t)r * nw + q]);
                    const double fr = (double)u / (double)R;
                    pe[np++] = fr < 0.5 ? 0.5 * (1.0 - pow(1.0 - 2.0 * fr, 1.0 / (double)wt)) : 0.5;
                }
                if (pe && np) {
                    for (int a = 1; a < np; ++a) {          /* insertion sort: tiny */
                        const double x = pe[a]; int c = a;
                        while (c > 0 && pe[c - 1] > x) { pe[c] = pe[c - 1]; --c; }
                        pe[c] = x;
                    }
                    out->ber_est = pe[np / 2];
                }
                free(pe);
                for (int i = 0; i < R; ++i) {
                    int ok = 1;
                    for (int r = 0; r < d && ok; ++r) {
                        uint64_t acc = 0;
                        for (int q = 0; q < nw; ++q) acc ^= rows[(size_t)i * nw + q] & H[(size_t)r * nw + q];
                        if (atk_parity64(acc)) ok = 0;
                    }
                    valid += ok;
                }
                out->valid_words = (double)valid / (double)R;
            }
            /* cyclic, shortened cyclic or extended cyclic, and the name */
            poly_t g;
            memset(&g, 0, sizeof g);
            int named_bch = 0, bt = 0, bm = 0, generic = 0;
            uint32_t bf = 0;
            out->cyclic = is_cyclic(Gm, k, H, d, N);
            if (out->cyclic) {
                cyclic_gen(Gm, k, N, &g);
                named_bch = name_cyclic(&g, N, k, out->name, sizeof out->name, &bt, &bf, &bm);
                generic = !strncmp(out->name, "cyclic (", 8);
            } else {
                /* a shortened cyclic code: the rows' gcd has degree N - k and
                 * its order (the mother length) exceeds N */
                poly_t gs;
                rows_gcd(Gm, k, N, &gs);
                const int e = pdeg(&gs) == N - k ? poly_order(&gs, 1023) : 0;
                if (e > N) {
                    char inner[40];
                    g = gs;
                    named_bch = name_cyclic(&gs, e, e - (N - k), inner, sizeof inner, &bt, &bf, &bm);
                    out->shortened = e - N;
                    set_name(out->name, sizeof out->name, "shortened %s as (%d,%d)", inner, N, k);
                }
            }
            if (!out->cyclic && !out->shortened) {
                /* the all-ones check closes an extended code */
                uint64_t *tmp = (uint64_t *)malloc(((size_t)d + 1) * (size_t)nw * sizeof *tmp);
                int ext = 0;
                if (tmp) {
                    memcpy(tmp, H, (size_t)d * (size_t)nw * sizeof *tmp);
                    uint64_t *ones = tmp + (size_t)d * nw;
                    memset(ones, 0, (size_t)nw * sizeof *ones);
                    for (int j = 0; j < N; ++j) ones[j >> 6] |= 1ULL << (j & 63);
                    ext = atk_gf2_canon(tmp, d + 1, N) == d;     /* ones in the dual */
                }
                free(tmp);
                if (ext) out->extended = name_extended(Gm, k, N, out->name, sizeof out->name, &g);
                if (!out->extended) snprintf(out->name, sizeof out->name, "linear (%d,%d)", N, k);
            }
            /* dmin and decode -> re-encode */
            uint8_t *Gb = (uint8_t *)malloc((size_t)k * (size_t)N);
            if (Gb) {
                for (int i = 0; i < k; ++i)
                    for (int j = 0; j < N; ++j)
                        Gb[(size_t)i * N + j] = (uint8_t)((Gm[(size_t)i * nw + (j >> 6)] >> (j & 63)) & 1u);
                if (G && gcap >= (size_t)k * (size_t)N) memcpy(G, Gb, (size_t)k * (size_t)N);
                uint8_t *word = (uint8_t *)malloc((size_t)N);
                long long changed = 0, decoded = 0;
                if (word && N <= 32 && N - k <= 20) {
                    atkdsp_blk *c = atkdsp_blk_create(N, k, Gb);
                    if (c) {
                        int dm = 0;
                        atkdsp_blk_params(c, NULL, NULL, &dm);
                        out->dmin = dm;
                        for (int i = 0; i < R; ++i) {
                            for (int j = 0; j < N; ++j)
                                word[j] = (uint8_t)((rows[(size_t)i * nw + (j >> 6)] >> (j & 63)) & 1u);
                            const int cc = atkdsp_blk_decode(c, word, NULL);
                            if (cc >= 0) { changed += cc; decoded += N; }
                        }
                        atkdsp_blk_destroy(c);
                    }
                } else if (word && named_bch && (out->cyclic || out->shortened)) {
                    atkdsp_bch *bc = atkdsp_bch_create(bm, bt, out->shortened, bf);
                    if (bc) {
                        for (int i = 0; i < R; ++i) {
                            for (int j = 0; j < N; ++j)
                                word[j] = (uint8_t)((rows[(size_t)i * nw + (j >> 6)] >> (j & 63)) & 1u);
                            const int cc = atkdsp_bch_decode(bc, word, NULL);
                            if (cc >= 0) { changed += cc; decoded += N; }
                        }
                        atkdsp_bch_destroy(bc);
                    }
                }
                if (decoded) out->agreement = 1.0 - (double)changed / (double)decoded;
                free(word);
                free(Gb);
            }
            /* a linear (or unnamed cyclic) code whose length is m symbols of
             * 3..8 bits may be a Reed-Solomon code's binary image: ask the
             * root test at this alignment, and take its name and evidence if
             * every row of G is an RS codeword */
            if ((!out->cyclic || generic) && !out->extended && !out->shortened) {
                for (int mm = 3; mm <= 8; ++mm) {
                    if (N % mm || (N - k) % mm || N / mm > (1 << mm) - 1 || N / mm < 3) continue;
                    atkdsp_rsid rs;
                    if (atkdsp_rs_code_id(b, n, (size_t)out->offset, mm, N / mm, pfa, &rs) != 1
                        || rs.nroots * mm != N - k) continue;
                    const int L = N / mm, NNr = (1 << mm) - 1;
                    atkdsp_rs *rc2 = atkdsp_rs_create(mm, rs.gfpoly, rs.fcr, rs.prim, rs.nroots, NNr - L,
                                                      rs.dual_basis ? ATKDSP_RS_DUAL_BASIS : 0);
                    int same = rc2 != NULL;
                    for (int i = 0; i < k && same; ++i) {
                        uint16_t sy[256];
                        for (int j = 0; j < L; ++j) {
                            unsigned v = 0;
                            for (int q = 0; q < mm; ++q) {
                                const int col = j * mm + q;
                                const unsigned bit = (unsigned)((Gm[(size_t)i * nw + (col >> 6)] >> (col & 63)) & 1u);
                                if (rs.lsb_first) v |= bit << q; else v = (v << 1) | bit;
                            }
                            sy[j] = (uint16_t)v;
                        }
                        if (atkdsp_rs_decode(rc2, sy, NULL, 0, NULL) != 0) same = 0;
                    }
                    if (rc2) atkdsp_rs_destroy(rc2);
                    if (!same) continue;
                    snprintf(out->name, sizeof out->name, "RS(%d,%d) GF(2^%d) binary image", rs.n, rs.k, mm);
                    if (rs.agreement == rs.agreement) out->agreement = rs.agreement;
                    break;
                }
            }
            if (out->fixed_bits) {
                char base[48];
                memcpy(base, out->name, sizeof base);
                set_name(out->name, sizeof out->name, "%s, %d fixed bit%s", base, out->fixed_bits,
                         out->fixed_bits > 1 ? "s" : "");
            }
            if (out->cyclic || out->extended || out->shortened) {
                const int dg = pdeg(&g);
                if (gpoly && dg >= 0 && pcap >= (size_t)dg + 1)
                    for (int i = 0; i <= dg; ++i) gpoly[i] = (uint8_t)((g.w[(dg - i) / 64] >> ((dg - i) % 64)) & 1u);
            }
            rc = 1;
        }
    }
    free(bw); free(H); free(Gm); free(rows);
    return rc;
}

/* ---- convolutional codes --------------------------------------------------------- */
typedef struct { int K; uint32_t p[4]; int n; const char *name; } conv_cat;
static const conv_cat CONV_CATALOG[] = {
    {7, {0171, 0133}, 2, "NASA/CCSDS, 802.11, DVB-S"},
    {7, {0133, 0171, 0165}, 3, "LTE (36.212)"},
    {9, {0561, 0753}, 2, "IS-95 / cdma2000 / 3GPP r=1/2"},
    {9, {0557, 0663, 0711}, 3, "IS-95 / 3GPP r=1/3"},
    {5, {023, 033}, 2, "GSM"},
    {7, {0133, 0171, 0145, 0133}, 4, "DAB / DRM mother code"},
    {3, {07, 05}, 2, "textbook K=3"},
};

/* Convolutional streams are searched over SHORT CONTIGUOUS STRETCHES rather
 * than random row subsets. Sliding windows share bits, so a stretch of R
 * windows touches only about n0 R + w bits and is error-free far more often
 * than R independent rows would be (K = 15, r = 1/2 at BER 1e-2: about a
 * third of generator stretches, against 1e-4 for a random subset). Noise only
 * ever lowers an exact deficiency, so the largest over the stretches is the
 * true one once any stretch is clean. Stretches start at multiples of n0,
 * spread evenly over the stream, in a fixed order. */
#define CONV_T 64                   /* stretches per width                        */
#define CONV_XROWS 32               /* rows beyond the width in a stretch         */
static int win_best(const uint64_t *bw, size_t n, int n0, int L, int *hits, int *lower,
                    double *l10, uint64_t *M) {
    const int w = n0 * (L + 1), R = w + CONV_XROWS;
    const size_t span = (size_t)n0 * (size_t)(R - 1) + (size_t)w;
    *hits = 0; *lower = 0; *l10 = 0.0;
    if (span > n) return -2;
    const size_t room = (n - span) / (size_t)n0;
    const int T = (size_t)CONV_T > room + 1 ? (int)(room + 1) : CONV_T;
    int best = 0;
    for (int j = 0; j < T; ++j) {
        const size_t st = (size_t)n0 * (T > 1 ? (room * (size_t)j) / (size_t)(T - 1) : 0);
        atk_gf2_rows(bw, st, w, R, n0, M);
        const int d = w - atk_gf2_rank(M, R, w);
        if (d > best) { *lower += *hits; best = d; *hits = 1; }
        else if (d == best) ++*hits;
        else ++*lower;
    }
    if (best > 0) *l10 = def_l10(best, R, w);
    return best;
}

/* rows of the joint generator system for phase ph, constraint length K */
static int gen_rows(const uint8_t *b, size_t n, int nn, int ph, int K, int t0, int steps, uint64_t *M) {
    const int w = nn * K, nw = nw_of(w);
    int r = 0;
    for (int t = t0 + K - 1; t < t0 + K - 1 + steps; ++t)
        for (int i = 1; i < nn; ++i) {
            uint64_t *row = M + (size_t)r * nw;
            memset(row, 0, (size_t)nw * sizeof *row);
            for (int j = 0; j < K; ++j) {
                const size_t c1 = (size_t)ph + (size_t)(t - j) * (size_t)nn;          /* stream 1 */
                const size_t ci = c1 + (size_t)i;                                   /* stream i+1 */
                if (b[c1] & 1u) { const int col = i * K + j; row[col >> 6] |= 1ULL << (col & 63); }
                if (b[ci] & 1u) { const int col = j; row[col >> 6] |= 1ULL << (col & 63); }
            }
            ++r;
        }
    (void)n;
    return r;
}

/* The generators of a rate 1/nn code: the smallest K, over the nn bit
 * alignments, at which c_1 g_i = c_i g_1 (each output stream is the input
 * times its generator) has a solution: a null space of nn K unknowns. A
 * wrong alignment is the same code with one generator delayed, so it needs a
 * larger K. Each K and alignment: exact null spaces over CONV_T short
 * stretches, and a candidate is kept when it holds on up to 4096 rows of the
 * whole stream (binomial against 1/2, every stretch, K and alignment
 * counted, and no more failures than BER ATK_ROBUST_PMAX allows). Returns 1
 * with *Kout, *phout and sol (bit s K + j = tap j of generator s), 0, or -1. */
static int conv_generators(const uint8_t *b, size_t n, int nn, int max_K, double pfa,
                           int *Kout, int *phout, uint64_t sol[2], int *noisy) {
    const double log10_thr = log10(pfa);
    const double log10_tests = log10((double)CONV_T * (double)nn * (double)max_K);
    *noisy = 0;
    for (int K = 2; K <= max_K; ++K) {
        const int w = nn * K, nw = nw_of(w);
        const int ss = (w + CONV_XROWS + nn - 2) / (nn - 1);          /* steps per stretch */
        for (int ph = 0; ph < nn; ++ph) {
            const size_t avail = (n - (size_t)ph) / (size_t)nn;
            if (avail < (size_t)(K + ss)) continue;
            const size_t room = avail - (size_t)(K - 1) - (size_t)ss;  /* first-step choices */
            int vs = (int)(avail - (size_t)(K - 1));
            if (vs > 4096 / (nn - 1)) vs = 4096 / (nn - 1);
            const int V = vs * (nn - 1), Rs = ss * (nn - 1);
            uint64_t *Mv = (uint64_t *)malloc((size_t)V * (size_t)nw * sizeof *Mv);
            uint64_t *Ms = (uint64_t *)malloc((size_t)Rs * (size_t)nw * sizeof *Ms);
            uint64_t *basis = (uint64_t *)malloc((size_t)w * (size_t)nw * sizeof *basis);
            if (!Mv || !Ms || !basis) { free(Mv); free(Ms); free(basis); return -1; }
            gen_rows(b, n, nn, ph, K, 0, vs, Mv);
            const int T = (size_t)CONV_T > room + 1 ? (int)(room + 1) : CONV_T;
            int found = 0;
            for (int j = 0; j < T && !found; ++j) {
                const int t0 = (int)(T > 1 ? (room * (size_t)j) / (size_t)(T - 1) : 0);
                gen_rows(b, n, nn, ph, K, t0, ss, Ms);
                const int d = atk_gf2_nullspace(Ms, Rs, w, basis, w);
                if (d < 0) { free(Mv); free(Ms); free(basis); return -1; }
                for (int q = 0; q < d && q < 4 && !found; ++q) {
                    const uint64_t *h = basis + (size_t)q * nw;
                    /* every generator must be nonzero: with g_1 = 0 (or a g_i
                     * = 0) whole equation types read 0 = 0, and a chance
                     * relation of the rest would pass a pooled test */
                    int wts[8], zero = 0;
                    for (int st = 0; st < nn; ++st) {
                        wts[st] = 0;
                        for (int j = 0; j < K; ++j) { const int col = st * K + j; wts[st] += (int)((h[col >> 6] >> (col & 63)) & 1u); }
                        if (!wts[st]) zero = 1;
                    }
                    if (zero) continue;
                    const int u = atk_gf2_unsat(Mv, V, nw, h);
                    const double l10 = atk_log10_binom_tail((double)V, (double)(V - u), -LN2) + log10_tests;
                    TRACE("conv gen K=%d ph=%d stretch %d: d=%d u=%d/%d l10=%.1f\n", K, ph, j, d, u, V, l10);
                    if (l10 > log10_thr) continue;
                    /* and each equation type (output i against output 1) fails
                     * no more often than BER ATK_ROBUST_PMAX allows */
                    int bad = 0;
                    for (int i = 1; i < nn && !bad; ++i) {
                        int ui = 0, vi = 0;
                        for (int r = i - 1; r < V; r += nn - 1) {
                            uint64_t acc = 0;
                            for (int x = 0; x < nw; ++x) acc ^= Mv[(size_t)r * nw + x] & h[x];
                            ui += atk_parity64(acc);
                            ++vi;
                        }
                        const double fmax = 0.5 * (1.0 - pow(1.0 - 2.0 * ATK_ROBUST_PMAX, (double)(wts[0] + wts[i])));
                        if ((double)ui > (double)vi * fmax + 3.0 * sqrt((double)vi * fmax * (1.0 - fmax))) bad = 1;
                    }
                    if (bad) continue;
                    sol[0] = h[0]; sol[1] = nw > 1 ? h[1] : 0;
                    *noisy = u > 0 || j > 0;
                    found = 1;
                }
            }
            free(Mv); free(Ms); free(basis);
            if (found) { *Kout = K; *phout = ph; return 1; }
        }
    }
    return 0;
}

int atkdsp_conv_code_id(const uint8_t *b, size_t n, int max_n, int max_K, double pfa,
                        uint64_t seed, atkdsp_convid *out) {
    (void)seed;     /* the stretches are fixed, so no random draws: kept for the ABI */
    if ((!b && n) || !out || max_n < 2 || max_n > 8 || max_K < 2 || max_K > 15
        || n > (size_t)INT32_MAX) return ATKDSP_E_ARG;
    if (!(pfa > 0.0)) pfa = 1e-3;
    memset(out, 0, sizeof *out);
    out->ber_est = out->agreement = out->density = NAN;
    snprintf(out->name, sizeof out->name, "no code found");
    {
        double p1;
        int lc;
        const int u = stream_unfit(b, n, &p1, &lc);
        if (u == 1) { set_name(out->name, sizeof out->name, "no code found (biased: P(1) = %.2f)", p1); return 0; }
        if (u == 2) { snprintf(out->name, sizeof out->name, "no code found (linear recurrence)"); return 0; }
    }
    size_t nbw;
    uint64_t *bw = atk_bits_pack(b, n, &nbw);
    if (!bw) return ATKDSP_E_NOMEM;
    const double log10_thr = log10(pfa);
    const int Lmax = max_K + 2;
    const double log10_tests = log10((double)CONV_T * (double)Lmax * (double)(max_n - 1));
    double dens[9], l10n[9];
    int rob[9], Rn[9], unsure[9];
    double best = 0.0;
    uint64_t *M = (uint64_t *)malloc(((size_t)max_n * (size_t)(Lmax + 5) + CONV_XROWS)
                                     * (size_t)nw_of(max_n * (Lmax + 5)) * sizeof *M);
    if (!M) { free(bw); return ATKDSP_E_NOMEM; }
    /* for each n0: the onset L0 (first window length with a significant
     * deficiency), then the growth over 4 more steps; the redundancy density
     * is that growth per bit. Best-of-stretches deficiencies throughout. */
    for (int n0 = 2; n0 <= max_n; ++n0) {
        dens[n0] = 0.0; rob[n0] = 0; l10n[n0] = 0.0; Rn[n0] = 0; unsure[n0] = 0;
        int L0 = 0, d0 = 0, h0 = 0, lo0 = 0;
        double l0 = 0.0;
        for (int L = 1; L <= Lmax; ++L) {
            int h, lo;
            double l;
            const int d = win_best(bw, n, n0, L, &h, &lo, &l, M);
            if (d == -2) break;
            if (d > 0 && l + log10_tests <= log10_thr) { L0 = L; d0 = d; h0 = h; lo0 = lo; l0 = l; break; }
        }
        if (!L0) continue;
        int h4, lo4;
        double l4;
        const int d4 = win_best(bw, n, n0, L0 + 4, &h4, &lo4, &l4, M);
        TRACE("conv n0=%d: onset L0=%d d=%d (hits %d, lower %d), L0+4: d=%d (hits %d, lower %d)\n",
              n0, L0, d0, h0, lo0, d4, h4, lo4);
        if (d4 < d0) continue;
        dens[n0] = (double)(d4 - d0) / (4.0 * (double)n0);
        l10n[n0] = l0 + log10_tests;
        Rn[n0] = n0 * (L0 + 1) + CONV_XROWS;
        rob[n0] = lo0 > 0 || lo4 > 0;                /* some stretches were hit by noise */
        unsure[n0] = h0 < 2 || h4 < 2;               /* one clean stretch: a thin margin */
        if (dens[n0] > best + 1e-9) best = dens[n0];
    }
    free(M);
    int robust = 0, rc = 0;
    double l10best = 0.0;
    if (best > 0.0)
        for (int n0 = 2; n0 <= max_n; ++n0)
            if (dens[n0] >= best - 0.05) {
                out->n = n0; robust = rob[n0]; l10best = l10n[n0]; out->rows_used = Rn[n0];
                break;
            }
    const int dens_unsure = out->n ? unsure[out->n] : 0;
    if (best <= 0.0 || !out->n) { free(bw); return 0; }
    int nn = out->n;
    out->density = best;
    out->robust = robust;
    out->log10_pfa = l10best;
    int k = nn - (int)floor(best * (double)nn + 0.5);
    if (k < 1) k = 1;
    if (k >= nn) {                  /* under half a check per step: no code */
        memset(out, 0, sizeof *out);
        out->ber_est = out->agreement = out->density = NAN;
        snprintf(out->name, sizeof out->name, "no code found");
        free(bw);
        return 0;
    }
    out->k = k;
    rc = 1;
    /* generators (rate 1/n0). A k/n with k | n is a 1/(n/k) code seen in
     * blocks of k steps: try n/k. A density from an incomplete robust count
     * is only a hint (it can pick a multiple of the true n): try every
     * divisor n0 >= 2 of n, smallest first. The generators, with the Viterbi
     * evidence, decide. A true k > 1 code (punctured) keeps its rate. */
    int bestK = 0, bestph = 0, gn = 0;
    uint64_t sol[2] = {0, 0};
    for (int n0 = 2; n0 <= nn && !bestK; ++n0) {
        if (nn % n0) continue;
        if (!dens_unsure && n0 * k != nn && n0 != nn) continue;
        int gnoisy = 0;
        const int g = conv_generators(b, n, n0, max_K, pfa, &bestK, &bestph, sol, &gnoisy);
        if (g < 0) { free(bw); return ATKDSP_E_NOMEM; }
        if (g) { gn = n0; if (gnoisy) robust = 1; }
    }
    /* a rate 1/(m n0) code also satisfies a longer rate 1/n0 description (its
     * streams paired up, K about m K'); a 1/n0 code in blocks of m steps is
     * rate m/(m n0). So a multiple whose density also showed, whose
     * generators are found AND whose K is no larger is the code. (A code
     * with a repeated output can fit a longer description at a multiple:
     * the K rule keeps the short one.) */
    for (int m = 2; bestK && gn * m <= max_n; ++m) {
        const int n0 = gn * m;
        if (!(dens[n0] > 0.0)) continue;
        int K2 = 0, ph2 = 0, gnoisy = 0;
        uint64_t sol2[2] = {0, 0};
        const int g = conv_generators(b, n, n0, max_K, pfa, &K2, &ph2, sol2, &gnoisy);
        if (g < 0) { free(bw); return ATKDSP_E_NOMEM; }
        TRACE("conv: multiple n0=%d of %d: %s (K %d vs %d)\n", n0, gn, g ? "found" : "none", K2, bestK);
        /* the multiple's description must be no longer: a rate 1/(m n0)
         * code needs K at n0 m but about m K at n0 */
        if (g && K2 <= bestK) {
            gn = n0; bestK = K2; bestph = ph2; sol[0] = sol2[0]; sol[1] = sol2[1];
            out->density = dens[n0]; l10best = l10n[n0];
            if (gnoisy) robust = 1;
        }
    }
    if (!bestK) {
        if (out->k > 1) {
            char rate[12];
            snprintf(rate, sizeof rate, "%d/%d", out->k & 7, nn & 15);
            snprintf(out->name, sizeof out->name, dens_unsure ? "rate %.11s convolutional? (density unsure)"
                                                               : "rate %.11s convolutional (k > 1: no generators)", rate);
        } else {
            snprintf(out->name, sizeof out->name, "convolutional, rate 1/%d (generators not found)", nn);
        }
        free(bw);
        return rc;
    }
    nn = gn;
    out->n = nn;
    out->k = 1;
    out->robust = robust;
    out->log10_pfa = l10best;
    out->K = bestK; out->phase = bestph;
    for (int s = 0; s < nn; ++s) {
        uint32_t p = 0;
        for (int j = 0; j < bestK; ++j) {
            const int col = s * bestK + j;
            if ((sol[col >> 6] >> (col & 63)) & 1u) p |= 1u << (bestK - 1 - j);
        }
        out->polys[s] = p;
    }
    /* the shortest form: taps no generator uses, at either end, are dropped
     * (a solution at a larger K than needed is the code times x^j) */
    for (;;) {
        uint32_t any_lo = 0, any_hi = 0;
        for (int s = 0; s < nn; ++s) { any_lo |= out->polys[s] & 1u; any_hi |= (out->polys[s] >> (bestK - 1)) & 1u; }
        if (bestK <= 2 || (any_lo && any_hi)) break;
        if (!any_lo) for (int s = 0; s < nn; ++s) out->polys[s] >>= 1;
        --bestK;
    }
    out->K = bestK;
    /* evidence: hard-decision Viterbi, re-encoded */
    {
        size_t steps = (n - (size_t)bestph) / (size_t)nn;
        if (steps > 20000) steps = 20000;
        const size_t smax = ((size_t)1 << 25) >> (bestK - 1);    /* bounds the trellis work */
        if (steps > smax) steps = smax;
        atkdsp_viterbi *v = atkdsp_viterbi_create(bestK, nn, out->polys, NULL, 0, ATKDSP_CONV_STREAM, 0, 0);
        const size_t dcap = v ? atkdsp_viterbi_out_max(v, steps * (size_t)nn) + 64 : 0;
        float *llr = (float *)malloc(steps * (size_t)nn * sizeof *llr);
        uint8_t *dec = (uint8_t *)malloc(dcap + 1);
        uint8_t *re = (uint8_t *)malloc(steps * (size_t)nn + 64);
        if (llr && dec && re && v) {
            for (size_t i = 0; i < steps * (size_t)nn; ++i) llr[i] = (b[(size_t)bestph + i] & 1u) ? -1.0f : 1.0f;
            ptrdiff_t m1 = atkdsp_viterbi_decode(v, llr, steps * (size_t)nn, dec, dcap, NULL);
            ptrdiff_t m2 = m1 >= 0 ? atkdsp_viterbi_flush(v, dec + m1, dcap - (size_t)m1) : -1;
            if (m1 >= 0 && m2 >= 0) {
                const size_t nd = (size_t)(m1 + m2);
                const ptrdiff_t nc = atkdsp_conv_encode(bestK, nn, out->polys, NULL, 0, ATKDSP_CONV_STREAM,
                                                        dec, nd, re, steps * (size_t)nn + 64);
                long long agree = 0, tot = 0;
                for (ptrdiff_t i = (ptrdiff_t)nn * (bestK - 1); i < nc; ++i) {
                    agree += (re[i] == (b[(size_t)bestph + (size_t)i] & 1u));
                    ++tot;
                }
                if (tot) { out->agreement = (double)agree / (double)tot; out->ber_est = 1.0 - out->agreement; }
            }
        }
        if (v) atkdsp_viterbi_destroy(v);
        free(llr); free(dec); free(re);
    }
    /* the name */
    {
        char polys[96];            /* 8 generators of <= 5 octal digits and commas: 47 */
        size_t o = 0;
        for (int s = 0; s < nn && o < sizeof polys - 1; ++s) {
            const int w = snprintf(polys + o, sizeof polys - o, "%s%o", s ? "," : "", (unsigned)out->polys[s]);
            if (w < 0) break;
            o += (size_t)w;
            if (o >= sizeof polys) o = sizeof polys - 1;
        }
        const char *cat = NULL;
        for (size_t c = 0; c < sizeof CONV_CATALOG / sizeof CONV_CATALOG[0]; ++c) {
            const conv_cat *e = &CONV_CATALOG[c];
            if (e->K != bestK || e->n != nn) continue;
            int used[8] = {0}, ok = 1;
            for (int s = 0; s < nn && ok; ++s) {
                int hit = 0;
                for (int q = 0; q < nn; ++q)
                    if (!used[q] && e->p[q] == out->polys[s]) { used[q] = 1; hit = 1; break; }
                ok = hit;
            }
            if (ok) { cat = e->name; break; }
        }
        if (cat) set_name(out->name, sizeof out->name, "K=%d r=1/%d (%s) %s", bestK, nn, polys, cat);
        else set_name(out->name, sizeof out->name, "K=%d r=1/%d (%s)", bestK, nn, polys);
    }
    free(bw);
    return rc;
}

/* ---- Reed-Solomon ------------------------------------------------------------------ */
static const uint8_t CCSDS_TAL[8] = {0x8d, 0xef, 0xec, 0x86, 0xfa, 0x99, 0xaf, 0x7b};

/* Codewords (of R from bit `off`) that vanish at alpha^e for each of ne
 * exponents, in one representation. -1 on no memory. */
static int rs_root_counts(const uint8_t *b, size_t off, int R, int L, int mm, uint32_t f, int dual,
                          int lsb, const uint8_t *tal1, const int *ex, int ne, int *cnt) {
    const int NN = (1 << mm) - 1;
    uint16_t exp_t[512];
    int16_t log_t[256];
    uint32_t x = 1;
    for (int i = 0; i < NN; ++i) { exp_t[i] = (uint16_t)x; log_t[x] = (int16_t)i; x <<= 1; if (x >> mm) x ^= f; }
    for (int i = NN; i < 2 * NN; ++i) exp_t[i] = exp_t[i - NN];
    log_t[0] = -1;
    uint16_t *sy = (uint16_t *)malloc((size_t)L * sizeof *sy);
    if (!sy) return -1;
    for (int i = 0; i < ne; ++i) cnt[i] = 0;
    for (int r = 0; r < R; ++r) {
        for (int j = 0; j < L; ++j) {
            const size_t base = off + ((size_t)r * (size_t)L + (size_t)j) * (size_t)mm;
            unsigned v = 0;
            for (int q = 0; q < mm; ++q) {
                const unsigned bit = b[base + (size_t)q] & 1u;
                if (lsb) v |= bit << q; else v = (v << 1) | bit;
            }
            sy[j] = (uint16_t)(dual ? tal1[v] : v);
        }
        for (int i = 0; i < ne; ++i) {
            unsigned acc = 0;
            for (int j = 0; j < L; ++j) {
                acc = acc ? exp_t[log_t[acc] + ex[i]] : 0;
                acc ^= sy[j];
            }
            cnt[i] += acc == 0;
        }
    }
    free(sy);
    return 0;
}

int atkdsp_rs_code_id(const uint8_t *b, size_t n, size_t offset, int m, int nsym, double pfa,
                      atkdsp_rsid *out) {
    if ((!b && n) || !out || m < 0 || m > 8 || (m && m < 3) || nsym < 0
        || n > (size_t)INT32_MAX) return ATKDSP_E_ARG;
    if (!(pfa > 0.0)) pfa = 1e-3;
    memset(out, 0, sizeof *out);
    out->valid_words = out->agreement = NAN;
    snprintf(out->name, sizeof out->name, "no code found");
    if (offset >= n) return 0;
    {
        double p1;
        int lc;
        const int u = stream_unfit(b + offset, n - offset, &p1, &lc);
        if (u == 1) { set_name(out->name, sizeof out->name, "no code found (biased: P(1) = %.2f)", p1); return 0; }
        if (u == 2) { snprintf(out->name, sizeof out->name, "no code found (linear recurrence)"); return 0; }
    }
    const double log10_thr = log10(pfa);
    /* the representations tried, counted first for the Bonferroni factor */
    double nrep = 0.0;
    for (int mm = m ? m : 3; mm <= (m ? m : 8); ++mm) {
        for (uint32_t f = (1u << mm) | 1u; f < (2u << mm); f += 2) if (is_primitive(f, mm)) nrep += 2.0;
        if (mm == 8) nrep += 2.0;
    }
    /* Stage 1, the representation: field polynomial, bit order, basis. For
     * each, every element is tried as a root over up to 64 codewords, and the
     * representation scores the summed significance of its elements' root
     * counts — every root of g adds, so the right one stands out even when
     * few codewords are clean. Stage 2 recounts that representation over up
     * to 1024 codewords for the root set itself. */
    uint16_t exp_t[512];
    int16_t log_t[256];
    uint8_t tal1[256];
    {
        uint8_t tal[256];
        for (int x = 0; x < 256; ++x) {
            uint8_t v = 0;
            for (int j = 0; j < 8; ++j) if ((x >> j) & 1) v ^= CCSDS_TAL[7 - j];
            tal[x] = v;
        }
        for (int x = 0; x < 256; ++x) tal1[tal[x]] = (uint8_t)x;
    }
    double best_score = 0.0;
    int bm = 0, bdual = 0, blsb = 0;
    uint32_t bf = 0;
    double ntests = 0.0;
    /* a representation and its twin: bits reversed within a symbol under the
     * reciprocal polynomial is the same code (the element times a constant,
     * roots negated). Both go to stage 2; the smaller first root is reported. */
    double twin_score = 0.0;
    uint32_t twin_f = 0;
    double (*stab)[512] = (double (*)[512])calloc(2, 512 * sizeof(double));
    if (!stab) return ATKDSP_E_NOMEM;
    atkdsp_rsid cand[2];
    int ncand = 0;
    memset(cand, 0, sizeof cand);
    for (int pass = 0; pass < 3; ++pass) {
        if (pass == 1) {
            /* the twin of the best: reciprocal polynomial, other bit order */
            twin_f = 0;
            if (!bdual) {
                uint32_t r = 0;
                for (int i = 0; i <= bm; ++i) if ((bf >> i) & 1u) r |= 1u << (bm - i);
                twin_f = r;
                twin_score = stab[1 - blsb][r & 511u];
            }
        }
        for (int mm = m ? m : 3; mm <= (m ? m : 8); ++mm) {
            if (pass && mm != bm) continue;
            if (pass == 2 && (!twin_f || twin_score < 0.5 * best_score)) continue;
            const int NN = (1 << mm) - 1;
            const int L = nsym ? nsym : NN;
            if (L > NN || L < 3) continue;
            const size_t fit = (n - offset) / ((size_t)L * (size_t)mm);
            const size_t rcap = pass ? 1024 : 64;
            const int R = (int)(fit < rcap ? fit : rcap);
            if (R < 4) continue;
            uint16_t *sym = (uint16_t *)malloc((size_t)R * (size_t)L * sizeof *sym);
            int *cnt = (int *)malloc((size_t)NN * sizeof *cnt);
            if (!sym || !cnt) { free(sym); free(cnt); free(stab); return ATKDSP_E_NOMEM; }
            for (uint32_t f = (1u << mm) | 1u; f < (2u << mm); f += 2) {
                if (pass == 1 && f != bf) continue;
                if (pass == 2 && f != twin_f) continue;
                if (!is_primitive(f, mm)) continue;
                uint32_t x = 1;
                for (int i = 0; i < NN; ++i) { exp_t[i] = (uint16_t)x; log_t[x] = (int16_t)i; x <<= 1; if (x >> mm) x ^= f; }
                for (int i = NN; i < 2 * NN; ++i) exp_t[i] = exp_t[i - NN];
                log_t[0] = -1;
                const int ndual = (mm == 8 && f == 0x187) ? 2 : 1;
                for (int dual = 0; dual < ndual; ++dual)
                    for (int lsb = 0; lsb < 2; ++lsb) {
                        if (pass == 1 && (dual != bdual || lsb != blsb)) continue;
                        if (pass == 2 && (dual || lsb == blsb)) continue;
                        for (int r = 0; r < R; ++r)
                            for (int j = 0; j < L; ++j) {
                                const size_t base = offset + ((size_t)r * (size_t)L + (size_t)j) * (size_t)mm;
                                unsigned v = 0;
                                for (int q = 0; q < mm; ++q) {
                                    const unsigned bit = b[base + (size_t)q] & 1u;
                                    if (lsb) v |= bit << q; else v = (v << 1) | bit;
                                }
                                sym[(size_t)r * L + j] = (uint16_t)(dual ? tal1[v] : v);
                            }
                        for (int e = 0; e < NN; ++e) {
                            cnt[e] = 0;
                            for (int r = 0; r < R; ++r) {
                                unsigned acc = 0;
                                const uint16_t *sy = sym + (size_t)r * L;
                                for (int j = 0; j < L; ++j) {
                                    acc = acc ? exp_t[log_t[acc] + e] : 0;
                                    acc ^= sy[j];
                                }
                                cnt[e] += (acc == 0);
                            }
                        }
                        if (!pass) {
                            double sc = 0.0;
                            for (int e = 0; e < NN; ++e) {
                                if (cnt[e] < 2) continue;
                                const double l = atk_log10_binom_tail((double)R, (double)cnt[e], -(double)mm * LN2);
                                if (l < -2.0) sc -= l;
                            }
                            ntests += (double)NN;
                            TRACE("rs rep m=%d f=0x%x dual=%d lsb=%d score=%.1f\n", mm, f, dual, lsb, sc);
                            if (!dual) stab[lsb][f & 511u] = sc;
                            /* reciprocal twins score the same up to rounding:
                             * a new best must win by more than that, so the
                             * first in the fixed order keeps a tie */
                            if (sc > best_score * (1.0 + 1e-9) + 1e-12) { best_score = sc; bm = mm; bf = f; bdual = dual; blsb = lsb; }
                            continue;
                        }
                        /* stage 2: the roots, their run, fcr and prim */
                        const double log10_trials = log10((double)NN * nrep);
                        int nroots = 0, roots[256];
                        double weakest = -INFINITY;
                        for (int e = 0; e < NN; ++e) {
                            const double l = atk_log10_binom_tail((double)R, (double)cnt[e], -(double)mm * LN2) + log10_trials;
                            if (l <= log10_thr) { roots[nroots++] = e; if (l > weakest) weakest = l; }
                        }
                        TRACE("rs stage 2: R=%d nroots=%d weakest=%.1f\n", R, nroots, weakest);
                        if (nroots < 2 || nroots >= L) continue;
                        int prim = 0, fcr = -1;
                        for (int pp = 1; pp < NN && !prim; ++pp) {
                            int a = pp, bb = NN;
                            while (bb) { const int t = a % bb; a = bb; bb = t; }
                            if (a != 1) continue;
                            int pinv = 1;
                            while ((pinv * pp) % NN != 1) ++pinv;
                            uint8_t mark[256];
                            memset(mark, 0, sizeof mark);
                            for (int i = 0; i < nroots; ++i) mark[(roots[i] * pinv) % NN] = 1;
                            for (int st = 0; st < NN; ++st) {        /* a start whose run covers them */
                                if (!mark[st] || mark[(st + NN - 1) % NN]) continue;
                                int len = 0;
                                while (len < nroots && mark[(st + len) % NN]) ++len;
                                if (len == nroots) { prim = pp; fcr = st; }
                                break;
                            }
                        }
                        if (!prim) continue;
                        atkdsp_rsid *c = &cand[ncand++];
                        c->m = mm; c->n = L; c->nroots = nroots; c->k = L - nroots;
                        c->fcr = fcr; c->prim = prim; c->gfpoly = f;
                        c->dual_basis = dual; c->lsb_first = lsb;
                        c->log10_pfa = weakest;
                    }
            }
            free(sym); free(cnt);
        }
        /* stage 1 must beat chance: its best score, against every element of
         * every representation tried */
        if (!pass && best_score + log10_thr - log10(ntests > 1.0 ? ntests : 1.0) < 0.0) { free(stab); return 0; }
    }
    free(stab);
    /* the twin wins on more roots, then a smaller fcr, then MSB first */
    if (ncand == 2 && (cand[1].nroots > cand[0].nroots
                       || (cand[1].nroots == cand[0].nroots && cand[1].fcr < cand[0].fcr)
                       || (cand[1].nroots == cand[0].nroots && cand[1].fcr == cand[0].fcr
                           && cand[1].lsb_first < cand[0].lsb_first))) cand[0] = cand[1];
    if (ncand) {
        out->m = cand[0].m; out->n = cand[0].n; out->nroots = cand[0].nroots; out->k = cand[0].k;
        out->fcr = cand[0].fcr; out->prim = cand[0].prim; out->gfpoly = cand[0].gfpoly;
        out->dual_basis = cand[0].dual_basis; out->lsb_first = cand[0].lsb_first;
        out->log10_pfa = cand[0].log10_pfa;
    }
    const int best_roots = ncand ? out->nroots : 0;
    if (!best_roots) { memset(out, 0, sizeof *out); out->valid_words = out->agreement = NAN;
                       snprintf(out->name, sizeof out->name, "no code found"); return 0; }
    /* half a codeword along, a code's roots vanish (a window across two
     * codewords is no codeword); structure that keeps them — a periodic
     * stream — is not codeword-aligned */
    {
        const int mm = out->m, L = out->n, NN = (1 << mm) - 1;
        const size_t off2 = offset + (size_t)(L / 2) * (size_t)mm;
        const size_t fit2 = off2 < n ? (n - off2) / ((size_t)L * (size_t)mm) : 0;
        const int R2 = (int)(fit2 < 1024 ? fit2 : 1024);
        int ex[256], cnt[256], kept = 0;
        for (int i = 0; i < out->nroots; ++i) ex[i] = (out->prim * (out->fcr + i)) % NN;
        if (R2 >= 4) {
            if (rs_root_counts(b, off2, R2, L, mm, out->gfpoly, out->dual_basis, out->lsb_first, tal1,
                               ex, out->nroots, cnt) < 0) return ATKDSP_E_NOMEM;
            const double ltr = log10((double)NN * nrep);
            for (int i = 0; i < out->nroots; ++i)
                if (atk_log10_binom_tail((double)R2, (double)cnt[i], -(double)mm * LN2) + ltr <= log10_thr) ++kept;
        }
        TRACE("rs shift test: R2=%d roots kept %d of %d\n", R2, kept, out->nroots);
        if (kept) {
            memset(out, 0, sizeof *out);
            out->valid_words = out->agreement = NAN;
            snprintf(out->name, sizeof out->name, "no code found (not codeword-aligned)");
            return 0;
        }
    }
    /* evidence: the §16b decoder over every codeword */
    {
        const int mm = out->m, L = out->n, NN = (1 << mm) - 1;
        atkdsp_rs *rs = atkdsp_rs_create(mm, out->gfpoly, out->fcr, out->prim, out->nroots, NN - L,
                                         out->dual_basis ? ATKDSP_RS_DUAL_BASIS : 0);
        const size_t fit = (n - offset) / ((size_t)L * (size_t)mm);
        const int R = (int)(fit < ATK_ID_ROWS ? fit : ATK_ID_ROWS);
        uint16_t *w = (uint16_t *)malloc((size_t)L * sizeof *w);
        if (rs && w) {
            int ok = 0;
            long long corr = 0;
            for (int r = 0; r < R; ++r) {
                for (int j = 0; j < L; ++j) {
                    const size_t base = offset + ((size_t)r * (size_t)L + (size_t)j) * (size_t)mm;
                    unsigned v = 0;
                    for (int q = 0; q < mm; ++q) {
                        const unsigned bit = b[base + (size_t)q] & 1u;
                        if (out->lsb_first) v |= bit << q; else v = (v << 1) | bit;
                    }
                    w[j] = (uint16_t)v;
                }
                uint16_t orig[256];
                memcpy(orig, w, (size_t)L * sizeof *w);
                const int c = atkdsp_rs_decode(rs, w, NULL, 0, NULL);
                if (c >= 0) {
                    ++ok;
                    for (int j = 0; j < L; ++j) corr += (w[j] != orig[j]);
                }
            }
            out->rows_used = R;
            out->valid_words = R ? (double)ok / (double)R : NAN;
            if (ok) out->agreement = 1.0 - (double)corr / ((double)ok * (double)L);
        }
        if (rs) atkdsp_rs_destroy(rs);
        free(w);
    }
    if (out->m == 8 && out->gfpoly == 0x187 && out->fcr == 112 && out->prim == 11 && out->nroots == 32
        && out->dual_basis)
        snprintf(out->name, sizeof out->name, "CCSDS RS(%d,%d) dual basis", out->n, out->k);
    else
        snprintf(out->name, sizeof out->name, "RS(%d,%d) GF(2^%d) 0x%X fcr %d prim %d%s", out->n, out->k,
                 out->m, (unsigned)out->gfpoly, out->fcr, out->prim, out->dual_basis ? " dual basis" : "");
    return 1;
}

/* ---- block interleavers ---------------------------------------------------------------- */
/* blocks to take: enough for every width the inner scan tries (exact:
 * (w + 33) w bits plus an offset; robust: the subset and 256 more rows) */
static size_t ilv_blocks(size_t N, int wmax, int robust) {
    const size_t need = robust ? ((size_t)wmax + 2 + 256 + 1) * (size_t)wmax
                               : ((size_t)wmax + 34) * (size_t)wmax;
    const size_t b = (need + N - 1) / N + 1;
    return b < 64 ? 64 : b;
}

/* The smallest width (2 .. wmax, below wstop) at which ny bits show a
 * significant deficiency, rows from bit 0 at stride w; exact or robust as
 * the block search was. 0 = none. -1 on no memory. */
static int inner_width(const uint8_t *y, size_t ny, int wmax, int wstop, int robust, uint64_t seed,
                       double pfa, int *dout) {
    const double log10_thr = log10(pfa);
    size_t ybw;
    *dout = 0;
    uint64_t *yw = atk_bits_pack(y, ny, &ybw);
    const size_t mrows = ny / 2 + 64;
    uint64_t *M = (uint64_t *)malloc(mrows * (size_t)nw_of(wmax) * sizeof *M);
    uint64_t *basis = (uint64_t *)malloc((size_t)wmax * (size_t)nw_of(wmax) * sizeof *basis);
    if (!yw || !M || !basis) { free(yw); free(M); free(basis); return -1; }
    int found = 0;
    for (int w = 2; w <= wmax && w < wstop && !found; ++w) {
        const size_t fit = ny / (size_t)w;
        int d = 0;
        double l = 0.0;
        if (!robust) {
            /* four offsets, so some row start is within w/8 of a codeword
             * start: a code of rate below 7/8 shows at its own length */
            int R = w + 32;
            if ((size_t)R + 1 > fit) R = (int)fit - 1;
            if (R < w + 16) break;
            for (int k4 = 0; k4 < 4 && !(d > 0 && l <= log10_thr); ++k4) {
                const int o = (k4 * w) / 4;
                if (k4 && o == (((k4 - 1) * w) / 4)) continue;
                atk_gf2_rows(yw, (size_t)o, w, R, w, M);
                d = w - atk_gf2_rank(M, R, w);
                l = d > 0 ? def_l10(d, R, w) + log10(4.0 * (double)wmax) : 0.0;
            }
        } else {
            const size_t rcap = (size_t)w + 2 + 1024;
            const int Rr = (int)(fit < rcap ? fit : rcap);
            atk_robust_info info;
            if (Rr < w + 2 + ATK_ROBUST_VMIN) break;
            atk_gf2_rows(yw, 0, w, Rr, w, M);
            /* a detection scan, as in the block search: a code the method
             * can see at all shows in the first few trials */
            d = atk_gf2_robust_q(M, Rr, w, seed + (uint64_t)w, pfa, basis, w, &info, 6);
            if (d < 0) { free(yw); free(M); free(basis); return -1; }
            l = d > 0 ? info.log10_pfa + log10((double)wmax) : 0.0;
        }
        if (d > 0 && l <= log10_thr) { found = w; *dout = d; }
    }
    free(yw); free(M); free(basis);
    return found;
}

int atkdsp_interleaver_id(const uint8_t *b, size_t n, int n_lo, int n_hi, double pfa,
                          uint64_t seed, atkdsp_ilvid *out) {
    if ((!b && n) || !out || n_lo < 4 || n_hi < n_lo || n_hi > 1024 || n > (size_t)INT32_MAX)
        return ATKDSP_E_ARG;
    if (!(pfa > 0.0)) pfa = 1e-3;
    memset(out, 0, sizeof *out);
    {
        double p1;
        int lc;
        if (stream_unfit(b, n, &p1, &lc)) return 0;
    }
    size_t nbw;
    uint64_t *bw = atk_bits_pack(b, n, &nbw);
    if (!bw) return ATKDSP_E_NOMEM;
    blk_find f;
    int rc = block_search(bw, n, n_lo, n_hi, pfa, seed, &f, BS_EXACT | BS_ROBUST);
    free(bw);
    if (rc <= 0) return rc;
    if (f.slide) return 0;                      /* a recurrence or a convolutional stream */
    out->period = f.N; out->offset = f.offset; out->log10_pfa = f.log10_pfa;
    /* the stream as it stands: an interleaver must make the inner code show
     * at a SMALLER width than the raw stream does (a plain code is not an
     * interleaver) */
    int w_raw = 0, d_raw = 0;
    {
        const int wmax = f.N < 128 ? f.N : 128;
        const size_t blocks = (n - (size_t)f.offset) / (size_t)f.N;
        size_t cap = ilv_blocks((size_t)f.N, wmax, f.robust);
        if (cap > blocks) cap = blocks;
        if (cap >= 2) {
            w_raw = inner_width(b + f.offset, cap * (size_t)f.N, wmax, wmax + 1, f.robust, seed ^ 0x1D, pfa, &d_raw);
            if (w_raw < 0) return ATKDSP_E_NOMEM;
        }
    }
    /* the de-interleaved code must be at least twice as local as the raw
     * stream's structure: a cyclic code's decimations have structure too */
    const int wlimit = w_raw ? w_raw / 2 + 1 : 1 << 20;
    int best_w = 0, best_d = 0;
    /* block sizes: the period's divisors (an interleaver block that is not a
     * multiple of the codeword makes the period their common multiple),
     * the period, and its multiples to 4x; each block start the period allows */
    /* the period and its multiples first, then the divisors — which then
     * need only scan the widths below the best already found */
    for (int pp = 0; pp < 2; ++pp) {
    const int pass = 1 - pp;
    for (int N = pass ? f.N : 4; pass ? (N <= n_hi && N <= 4 * f.N) : N < f.N; N += pass ? f.N : 1) {
        if (!pass && (f.N % N || N < n_lo)) continue;
        /* the period's alignment fixes a block start only to a multiple of
         * the period; a divisor block's own start is searched (exact mode:
         * every start, and the inner scan's four offsets find the codewords).
         * Robust mode takes the period's own start: there the interleaver
         * blocks and the inner codewords begin together (the period is their
         * common multiple), so the undone stream starts on a codeword — an
         * earlier block of the same alignment need not (Golay(24,12) in 6 x 10
         * blocks: the block before starts 12 bits into a codeword). Every
         * offset that ties for the period's largest deficiency is a start:
         * the deficiency alone can leave the alignment ambiguous (Hamming(8,4)
         * in 5 x 7 blocks, period 280: three offsets reach 140, one a block
         * start) */
        const int nt = f.nties > 0 ? f.nties : 1;
        const int nst = pass ? nt * (N / f.N) : (f.robust ? nt : N);
        for (int j = 0; j < nst; ++j) {
            const int t0 = f.nties > 0 ? f.ties[pass ? j / (N / f.N) : (f.robust ? j : 0)] : f.offset;
            const int st = pass ? t0 + (j % (N / f.N)) * f.N : (f.robust ? t0 : j);
            /* the inner codeword can be longer than a divisor block (63-bit
             * BCH words in 5 x 7 blocks: period 315): widths to the period */
            const int wmax = (N > f.N ? N : f.N) < 128 ? (N > f.N ? N : f.N) : 128;
            const size_t blocks = (size_t)st < n ? (n - (size_t)st) / (size_t)N : 0;
            size_t cap = ilv_blocks((size_t)N, wmax, f.robust);
            if (cap > blocks) cap = blocks;
            if (cap < 2) break;
            uint8_t *y = (uint8_t *)malloc(cap * (size_t)N + 1);
            if (!y) return ATKDSP_E_NOMEM;
            for (int r = 2; r <= N / 2; ++r) {
                if (N % r) continue;
                const int c = N / r;
                if (c < 2) continue;
                /* written by rows (r x c), read by columns: out index col*r + row = in index row*c + col */
                for (size_t q = 0; q < cap; ++q) {
                    const uint8_t *src = b + (size_t)st + q * (size_t)N;
                    uint8_t *dst = y + q * (size_t)N;
                    for (int row = 0; row < r; ++row)
                        for (int col = 0; col < c; ++col)
                            dst[row * c + col] = (uint8_t)(src[col * r + row] & 1u);
                }
                /* a smaller width wins; the same width with a larger
                 * deficiency too (a wrong block start can leave part of the
                 * inner code at its own width) */
                const int stop = best_w ? (best_w + 1 < wlimit ? best_w + 1 : wlimit) : wlimit;
                int dd;
                const int w = inner_width(y, cap * (size_t)N, wmax, stop, f.robust,
                                          seed + (uint64_t)(r * 4096), pfa, &dd);
                if (w < 0) { free(y); return ATKDSP_E_NOMEM; }
                if (w > 0 && (!best_w || w < best_w || dd > best_d)) {
                    best_w = w; best_d = dd; out->rows = r; out->cols = c; out->period = N;
                    out->offset = st % N;
                }
            }
            free(y);
        }
    }
    }
    out->inner_width = best_w;
    out->inner_deficiency = best_d;
    if (!best_w) { out->rows = out->cols = 0; return 0; }
    return 1;
}
