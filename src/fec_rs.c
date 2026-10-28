/* Reed-Solomon over GF(2^m) and binary BCH (header §16b, §16c).
 *
 * Both are decoded the classical way, written from the textbooks (Lin &
 * Costello; Blahut): syndromes, Berlekamp-Massey for the error(-and-erasure)
 * locator, a Chien search over the positions actually sent, and for RS the
 * Forney formula for the values. A codeword is position j = 0..n-1 with the
 * first symbol the HIGHEST degree, x^(n-1); a shortened code is the full code
 * with its leading positions zero, so nothing else changes. A decode is
 * accepted only when the locator's roots are all found among the positions
 * sent and the corrected word's syndromes are zero — a miscorrection beyond
 * t can still pass (that is what "beyond t" means), and the tests measure
 * how often rather than assume.
 *
 * CCSDS dual basis: Berlekamp's representation, by the conversion matrix of
 * CCSDS 131.0-B (rows 0x8d 0xef 0xec 0x86 0xfa 0x99 0xaf 0x7b). Symbols are
 * converted dual -> conventional on the way in and back on the way out. */
#include "internal.h"
#include <stdio.h>

/* ---- GF(2^m) -------------------------------------------------------------- */
typedef struct {
    int m, nn;            /* nn = 2^m - 1 */
    uint32_t poly;
    uint16_t *exp;        /* exp[i] = alpha^i, i in [0, 2nn) */
    int32_t  *log;        /* log[x], log[0] = -1 */
} gf_t;

static int gf_init(gf_t *g, int m, uint32_t poly) {
    memset(g, 0, sizeof *g);
    if (m < 3 || m > 16 || (poly >> m) != 1u) return 0;
    g->m = m; g->nn = (1 << m) - 1; g->poly = poly;
    g->exp = (uint16_t *)malloc(2 * (size_t)g->nn * sizeof *g->exp);
    g->log = (int32_t *)malloc(((size_t)g->nn + 1) * sizeof *g->log);
    if (!g->exp || !g->log) return 0;
    for (int i = 0; i <= g->nn; ++i) g->log[i] = -1;
    uint32_t x = 1;
    for (int i = 0; i < g->nn; ++i) {
        if (g->log[x] != -1) return 0;            /* not primitive: alpha repeats early */
        g->exp[i] = (uint16_t)x;
        g->log[x] = i;
        x <<= 1;
        if (x >> m) x ^= poly;
    }
    if (x != 1) return 0;
    for (int i = g->nn; i < 2 * g->nn; ++i) g->exp[i] = g->exp[i - g->nn];
    return 1;
}

static void gf_free(gf_t *g) { free(g->exp); free(g->log); g->exp = NULL; g->log = NULL; }

ATK_INLINE int gf_mod(const gf_t *g, long long e) {
    long long r = e % g->nn;
    return (int)(r < 0 ? r + g->nn : r);
}
ATK_INLINE uint32_t gf_mul(const gf_t *g, uint32_t a, uint32_t b) {
    if (!a || !b) return 0;
    return g->exp[g->log[a] + g->log[b]];
}
ATK_INLINE uint32_t gf_inv(const gf_t *g, uint32_t a) {
    return g->exp[(g->nn - g->log[a]) % g->nn];
}
ATK_INLINE uint32_t gf_pow_alpha(const gf_t *g, long long e) { return g->exp[gf_mod(g, e)]; }

/* Default primitive polynomials for m = 3..16 (the standard tables). */
static const uint32_t GF_DEFAULT[17] = {0, 0, 0, 0xB, 0x13, 0x25, 0x43, 0x89, 0x11D,
    0x211, 0x409, 0x805, 0x1053, 0x201B, 0x4443, 0x8003, 0x1100B};

/* Evaluate a polynomial given by position (sym[j] is the coefficient of
 * x^(n-1-j)) at x = beta, Horner. */
static uint32_t poly_eval_pos(const gf_t *g, const uint16_t *sym, int n, uint32_t beta) {
    uint32_t acc = 0;
    for (int j = 0; j < n; ++j) acc = gf_mul(g, acc, beta) ^ sym[j];
    return acc;
}

/* Berlekamp-Massey over GF(2^m) for nsyn syndromes S[0..nsyn) with the
 * locator seeded by an erasure locator lam[0..L0] (L0 erasures). lam gets
 * the errata locator (cap nsyn + 1 coefficients). Returns its degree, or -1. */
static int gf_bm(const gf_t *g, const uint32_t *S, int nsyn, uint32_t *lam, int L0, uint32_t *B, uint32_t *T) {
    const int cap = nsyn + 1;
    for (int i = 0; i < cap; ++i) B[i] = lam[i];
    int L = L0, mshift = 1;
    uint32_t bdisc = 1;
    for (int r = L0; r < nsyn; ++r) {
        uint32_t d = 0;
        for (int i = 0; i <= L && i <= r; ++i) d ^= gf_mul(g, lam[i], S[r - i]);
        if (!d) { ++mshift; continue; }
        const uint32_t coef = gf_mul(g, d, gf_inv(g, bdisc));
        if (2 * L <= r + L0) {
            for (int i = 0; i < cap; ++i) T[i] = lam[i];
            for (int i = 0; i + mshift < cap; ++i) lam[i + mshift] ^= gf_mul(g, coef, B[i]);
            L = r + 1 + L0 - L;
            for (int i = 0; i < cap; ++i) B[i] = T[i];
            bdisc = d; mshift = 1;
        } else {
            for (int i = 0; i + mshift < cap; ++i) lam[i + mshift] ^= gf_mul(g, coef, B[i]);
            ++mshift;
        }
    }
    int deg = 0;
    for (int i = 0; i < cap; ++i) if (lam[i]) deg = i;
    return deg == L ? L : -1;      /* a locator short of its length: too many errors */
}

/* ---- Reed-Solomon ---------------------------------------------------------- */
struct atkdsp_rs {
    gf_t gf;
    int m, fcr, prim, nroots, pad, n, k, flags;
    uint32_t gfpoly;
    uint32_t *gen;                 /* generator, gen[0..nroots], gen[nroots] = 1 (x^nroots) */
    uint32_t *S, *lam, *B, *T, *om;/* decoder scratch */
    uint16_t *work;                /* n symbols, conventional basis */
    int *loc;                      /* located positions */
    uint8_t tal[256], tal1[256];   /* dual basis tables (flags) */
    atkdsp_fec_stats st;
};

static const uint8_t CCSDS_TAL[8] = {0x8d, 0xef, 0xec, 0x86, 0xfa, 0x99, 0xaf, 0x7b};

atkdsp_rs *atkdsp_rs_create(int m, uint32_t gfpoly, int fcr, int prim, int nroots, int pad, int flags) {
    if (m < 3 || m > 16 || nroots < 1 || pad < 0 || prim < 1) return NULL;
    const int nn = (1 << m) - 1;
    if (nroots >= nn - pad || fcr < 0 || fcr >= nn || prim >= nn) return NULL;
    if (flags & ~ATKDSP_RS_DUAL_BASIS) return NULL;
    if ((flags & ATKDSP_RS_DUAL_BASIS) && !(m == 8 && gfpoly == 0x187)) return NULL;
    atkdsp_rs *r = (atkdsp_rs *)calloc(1, sizeof *r);
    if (!r) return NULL;
    if (!gf_init(&r->gf, m, gfpoly)) { gf_free(&r->gf); free(r); return NULL; }
    /* prim must generate the field too, or two positions share a locator */
    { long long a = prim, b = nn; while (b) { const long long t = a % b; a = b; b = t; } if (a != 1) { gf_free(&r->gf); free(r); return NULL; } }
    r->m = m; r->gfpoly = gfpoly; r->fcr = fcr; r->prim = prim; r->nroots = nroots; r->pad = pad;
    r->n = nn - pad; r->k = r->n - nroots; r->flags = flags;
    const size_t c = (size_t)nroots + 2;
    r->gen = (uint32_t *)calloc(c, sizeof *r->gen);
    r->S = (uint32_t *)calloc(c, sizeof *r->S);
    r->lam = (uint32_t *)calloc(c, sizeof *r->lam);
    r->B = (uint32_t *)calloc(c, sizeof *r->B);
    r->T = (uint32_t *)calloc(c, sizeof *r->T);
    r->om = (uint32_t *)calloc(c, sizeof *r->om);
    r->work = (uint16_t *)calloc((size_t)r->n, sizeof *r->work);
    r->loc = (int *)calloc(c, sizeof *r->loc);
    if (!r->gen || !r->S || !r->lam || !r->B || !r->T || !r->om || !r->work || !r->loc) {
        atkdsp_rs_destroy(r); return NULL;
    }
    /* gen(x) = prod_{i<nroots} (x - alpha^(prim*(fcr+i))), gen[d] = coefficient of x^d */
    r->gen[0] = 1;
    for (int i = 0; i < nroots; ++i) {
        const uint32_t root = gf_pow_alpha(&r->gf, (long long)prim * (fcr + i));
        for (int d = i + 1; d > 0; --d) r->gen[d] = r->gen[d - 1] ^ gf_mul(&r->gf, r->gen[d], root);
        r->gen[0] = gf_mul(&r->gf, r->gen[0], root);
    }
    if (flags & ATKDSP_RS_DUAL_BASIS) {
        for (int x = 0; x < 256; ++x) {
            uint8_t v = 0;
            for (int j = 0; j < 8; ++j) if ((x >> j) & 1) v ^= CCSDS_TAL[7 - j];
            r->tal[x] = v;               /* conventional -> dual */
        }
        for (int x = 0; x < 256; ++x) r->tal1[r->tal[x]] = (uint8_t)x;   /* dual -> conventional */
    }
    return r;
}

void atkdsp_rs_destroy(atkdsp_rs *r) {
    if (!r) return;
    gf_free(&r->gf);
    free(r->gen); free(r->S); free(r->lam); free(r->B); free(r->T); free(r->om);
    free(r->work); free(r->loc);
    free(r);
}

int atkdsp_rs_params(const atkdsp_rs *r, int *n, int *k) {
    if (!r) return ATKDSP_E_ARG;
    if (n) *n = r->n;
    if (k) *k = r->k;
    return ATKDSP_OK;
}

/* parity = data(x) x^nroots mod gen(x), conventional basis; parity[0] is the
 * coefficient of x^(nroots-1). */
static void rs_encode_conv(const atkdsp_rs *r, const uint16_t *data, uint16_t *parity) {
    const int nr = r->nroots;
    uint32_t *reg = r->T;               /* reg[0..nr), reg[0] = highest */
    for (int i = 0; i < nr; ++i) reg[i] = 0;
    for (int j = 0; j < r->k; ++j) {
        const uint32_t fb = (uint32_t)data[j] ^ reg[0];
        for (int i = 0; i < nr - 1; ++i) reg[i] = reg[i + 1] ^ gf_mul(&r->gf, fb, r->gen[nr - 1 - i]);
        reg[nr - 1] = gf_mul(&r->gf, fb, r->gen[0]);
    }
    for (int i = 0; i < nr; ++i) parity[i] = (uint16_t)reg[i];
}

int atkdsp_rs_encode(atkdsp_rs *r, const uint16_t *data, uint16_t *parity) {
    if (!r || !data || !parity) return ATKDSP_E_ARG;
    const uint32_t lim = (uint32_t)r->gf.nn;
    for (int j = 0; j < r->k; ++j) if (data[j] > lim) return ATKDSP_E_ARG;
    if (r->flags & ATKDSP_RS_DUAL_BASIS) {
        for (int j = 0; j < r->k; ++j) r->work[j] = r->tal1[data[j]];
        rs_encode_conv(r, r->work, parity);
        for (int i = 0; i < r->nroots; ++i) parity[i] = r->tal[parity[i]];
    } else {
        rs_encode_conv(r, data, parity);
    }
    return ATKDSP_OK;
}

int atkdsp_rs_decode(atkdsp_rs *r, uint16_t *sym, const int *erasures, int nerase, int *err_pos) {
    if (!r || !sym || nerase < 0 || (nerase && !erasures)) return ATKDSP_E_ARG;
    if (nerase > r->nroots) { r->st.calls++; r->st.in++; r->st.failures++; return ATKDSP_E_DECODE; }
    const gf_t *g = &r->gf;
    const int n = r->n, nr = r->nroots;
    for (int j = 0; j < n; ++j) if (sym[j] > (uint32_t)g->nn) return ATKDSP_E_ARG;
    for (int e = 0; e < nerase; ++e) if (erasures[e] < 0 || erasures[e] >= n) return ATKDSP_E_ARG;
    r->st.calls++; r->st.in++;
    uint16_t *w = r->work;
    for (int j = 0; j < n; ++j) w[j] = (r->flags & ATKDSP_RS_DUAL_BASIS) ? r->tal1[sym[j]] : sym[j];
    int any = 0;
    for (int i = 0; i < nr; ++i) {
        r->S[i] = poly_eval_pos(g, w, n, gf_pow_alpha(g, (long long)r->prim * (r->fcr + i)));
        any |= r->S[i] != 0;
    }
    if (!any && nerase == 0) { r->st.out++; return 0; }
    /* erasure locator prod (1 - X_e x), X_e = beta^deg = alpha^(prim*deg) */
    for (int i = 0; i <= nr; ++i) r->lam[i] = 0;
    r->lam[0] = 1;
    for (int e = 0; e < nerase; ++e) {
        const int deg = n - 1 - erasures[e];
        const uint32_t X = gf_pow_alpha(g, (long long)r->prim * deg);
        for (int d = e + 1; d > 0; --d) r->lam[d] ^= gf_mul(g, r->lam[d - 1], X);
    }
    const int L = gf_bm(g, r->S, nr, r->lam, nerase, r->B, r->T);
    if (L < 0 || 2 * L - nerase > nr) goto fail;
    /* Chien over the positions sent */
    int nloc = 0;
    for (int j = 0; j < n; ++j) {
        const int deg = n - 1 - j;
        const long long xinv = -(long long)r->prim * deg;   /* X^-1 */
        uint32_t v = 0;
        for (int i = 0; i <= L; ++i) if (r->lam[i]) v ^= gf_mul(g, r->lam[i], gf_pow_alpha(g, xinv * i));
        if (!v) { if (nloc >= nr) goto fail; r->loc[nloc++] = j; }
    }
    if (nloc != L) goto fail;
    /* omega = S(x) lam(x) mod x^nr */
    for (int i = 0; i < nr; ++i) {
        uint32_t a = 0;
        for (int k = 0; k <= i && k <= L; ++k) a ^= gf_mul(g, r->lam[k], r->S[i - k]);
        r->om[i] = a;
    }
    for (int q = 0; q < nloc; ++q) {
        const int j = r->loc[q], deg = n - 1 - j;
        const long long xi = (long long)r->prim * deg;     /* index of X */
        uint32_t num = 0, den = 0;
        for (int i = 0; i < nr; ++i) if (r->om[i]) num ^= gf_mul(g, r->om[i], gf_pow_alpha(g, -xi * i));
        for (int i = 1; i <= L; i += 2) if (r->lam[i]) den ^= gf_mul(g, r->lam[i], gf_pow_alpha(g, -xi * (i - 1)));
        if (!den) goto fail;
        /* Y = X^(1-fcr) * omega(X^-1) / lam'(X^-1) */
        const uint32_t y = gf_mul(g, gf_mul(g, num, gf_inv(g, den)), gf_pow_alpha(g, xi * (1 - r->fcr)));
        w[j] ^= (uint16_t)y;
    }
    for (int i = 0; i < nr; ++i)
        if (poly_eval_pos(g, w, n, gf_pow_alpha(g, (long long)r->prim * (r->fcr + i)))) goto fail;
    int changed = 0;
    for (int q = 0; q < nloc; ++q) {
        const int j = r->loc[q];
        const uint16_t v = (r->flags & ATKDSP_RS_DUAL_BASIS) ? r->tal[w[j]] : w[j];
        if (v != sym[j]) ++changed;
        sym[j] = v;
        if (err_pos) err_pos[q] = j;
    }
    r->st.out++; r->st.corrected += (uint64_t)changed;
    return nloc;
fail:
    r->st.failures++;
    return ATKDSP_E_DECODE;
}

int atkdsp_rs_describe(const atkdsp_rs *r, char *buf, size_t cap) {
    if (!r || !buf || !cap) return ATKDSP_E_ARG;
    snprintf(buf, cap, "RS(%d,%d) over GF(2^%d) poly 0x%X, fcr %d, prim %d, corrects %d%s",
             r->n, r->k, r->m, (unsigned)r->gfpoly, r->fcr, r->prim, r->nroots / 2,
             (r->flags & ATKDSP_RS_DUAL_BASIS) ? ", CCSDS dual basis" : "");
    return ATKDSP_OK;
}

int atkdsp_rs_stats(const atkdsp_rs *r, atkdsp_fec_stats *st) {
    if (!r || !st) return ATKDSP_E_ARG;
    *st = r->st;
    return ATKDSP_OK;
}

/* ---- binary BCH ---------------------------------------------------------------- */
struct atkdsp_bch {
    gf_t gf;
    int m, t, n, k, shorten;
    uint32_t gfpoly;
    uint8_t *gen;                  /* generator, gen[d] = coefficient of x^d, d = 0..n-k */
    uint32_t *S, *lam, *B, *T;
    int *loc;
    uint8_t *reg;
    atkdsp_fec_stats st;
};

atkdsp_bch *atkdsp_bch_create(int m, int t, int shorten, uint32_t gfpoly) {
    if (m < 3 || m > 16 || t < 1 || shorten < 0) return NULL;
    if (!gfpoly) gfpoly = GF_DEFAULT[m];
    const int nn = (1 << m) - 1;
    if (2 * t >= nn || shorten >= nn) return NULL;
    atkdsp_bch *b = (atkdsp_bch *)calloc(1, sizeof *b);
    if (!b) return NULL;
    if (!gf_init(&b->gf, m, gfpoly)) { gf_free(&b->gf); free(b); return NULL; }
    b->m = m; b->t = t; b->shorten = shorten; b->gfpoly = gfpoly;
    /* generator: the product of the distinct minimal polynomials of alpha^1..alpha^2t */
    uint8_t *used = (uint8_t *)calloc((size_t)nn, 1);
    uint32_t *g = (uint32_t *)calloc((size_t)nn + 1, sizeof *g);   /* GF coefficients during the product */
    uint32_t *mp = (uint32_t *)calloc((size_t)m + 2, sizeof *mp);
    if (!used || !g || !mp) { free(used); free(g); free(mp); atkdsp_bch_destroy(b); return NULL; }
    int deg = 0;
    g[0] = 1;
    int ok = 1;
    for (int i = 1; i <= 2 * t && ok; ++i) {
        if (used[i % nn]) continue;
        /* the cyclotomic coset of i and its minimal polynomial prod (x - alpha^e) */
        int md = 0;
        mp[0] = 1;
        int e = i % nn;
        do {
            used[e] = 1;
            const uint32_t root = b->gf.exp[e];
            mp[md + 1] = 0;
            for (int d = md + 1; d > 0; --d) mp[d] = mp[d - 1] ^ gf_mul(&b->gf, mp[d], root);
            mp[0] = gf_mul(&b->gf, mp[0], root);
            ++md;
            e = (int)(((long long)e * 2) % nn);
        } while (e != i % nn && md <= m);
        for (int d = 0; d <= md; ++d) if (mp[d] > 1) ok = 0;     /* must be binary */
        if (!ok || deg + md > nn) { ok = 0; break; }
        /* g *= mp (binary) */
        for (int d = deg + md; d >= 0; --d) {
            uint32_t acc = 0;
            for (int a = 0; a <= md; ++a) {
                const int bi = d - a;
                if (bi >= 0 && bi <= deg && mp[a]) acc ^= g[bi];
            }
            g[d] = acc;
        }
        deg += md;
    }
    free(used); free(mp);
    b->n = nn - shorten;
    b->k = b->n - deg;
    if (!ok || b->k < 1) { free(g); atkdsp_bch_destroy(b); return NULL; }
    b->gen = (uint8_t *)malloc((size_t)deg + 1);
    b->S = (uint32_t *)calloc(2 * (size_t)t + 2, sizeof *b->S);
    b->lam = (uint32_t *)calloc(2 * (size_t)t + 2, sizeof *b->lam);
    b->B = (uint32_t *)calloc(2 * (size_t)t + 2, sizeof *b->B);
    b->T = (uint32_t *)calloc(2 * (size_t)t + 2, sizeof *b->T);
    b->loc = (int *)calloc(2 * (size_t)t + 2, sizeof *b->loc);
    b->reg = (uint8_t *)calloc((size_t)deg + 1, 1);
    if (!b->gen || !b->S || !b->lam || !b->B || !b->T || !b->loc || !b->reg) { free(g); atkdsp_bch_destroy(b); return NULL; }
    for (int d = 0; d <= deg; ++d) b->gen[d] = (uint8_t)g[d];
    free(g);
    return b;
}

void atkdsp_bch_destroy(atkdsp_bch *b) {
    if (!b) return;
    gf_free(&b->gf);
    free(b->gen); free(b->S); free(b->lam); free(b->B); free(b->T); free(b->loc); free(b->reg);
    free(b);
}

int atkdsp_bch_params(const atkdsp_bch *b, int *n, int *k, int *t) {
    if (!b) return ATKDSP_E_ARG;
    if (n) *n = b->n;
    if (k) *k = b->k;
    if (t) *t = b->t;
    return ATKDSP_OK;
}

int atkdsp_bch_generator(const atkdsp_bch *b, uint8_t *g, size_t cap) {
    if (!b || !g) return ATKDSP_E_ARG;
    const int r = b->n - b->k;
    if (cap < (size_t)r + 1) return ATKDSP_E_CAP;
    for (int d = 0; d <= r; ++d) g[d] = b->gen[r - d];      /* highest degree first */
    return r;
}

int atkdsp_bch_encode(atkdsp_bch *b, const uint8_t *data, uint8_t *code) {
    if (!b || !data || !code) return ATKDSP_E_ARG;
    const int r = b->n - b->k;
    uint8_t *reg = b->reg;                 /* reg[0] = coefficient of x^(r-1) */
    memset(reg, 0, (size_t)r);
    for (int j = 0; j < b->k; ++j) {
        const uint8_t fb = (uint8_t)((data[j] & 1u) ^ reg[0]);
        for (int i = 0; i < r - 1; ++i) reg[i] = (uint8_t)(reg[i + 1] ^ (fb & b->gen[r - 1 - i]));
        reg[r - 1] = (uint8_t)(fb & b->gen[0]);
        code[j] = (uint8_t)(data[j] & 1u);
    }
    for (int i = 0; i < r; ++i) code[b->k + i] = reg[i];
    return ATKDSP_OK;
}

int atkdsp_bch_decode(atkdsp_bch *b, uint8_t *code, int *err_pos) {
    if (!b || !code) return ATKDSP_E_ARG;
    const gf_t *g = &b->gf;
    const int n = b->n, ns = 2 * b->t;
    b->st.calls++; b->st.in++;
    int any = 0;
    for (int i = 0; i < ns; ++i) {         /* S_i = r(alpha^(i+1)) */
        uint32_t acc = 0;
        const uint32_t beta = g->exp[(i + 1) % g->nn];
        for (int j = 0; j < n; ++j) acc = gf_mul(g, acc, beta) ^ (uint32_t)(code[j] & 1u);
        b->S[i] = acc;
        any |= acc != 0;
    }
    if (!any) { b->st.out++; return 0; }
    for (int i = 0; i <= ns; ++i) b->lam[i] = 0;
    b->lam[0] = 1;
    const int L = gf_bm(g, b->S, ns, b->lam, 0, b->B, b->T);
    if (L < 1 || L > b->t) goto fail;
    int nloc = 0;
    for (int j = 0; j < n; ++j) {
        const int deg = n - 1 - j;
        uint32_t v = 0;
        for (int i = 0; i <= L; ++i) if (b->lam[i]) v ^= gf_mul(g, b->lam[i], gf_pow_alpha(g, -(long long)deg * i));
        if (!v) { if (nloc >= L) goto fail; b->loc[nloc++] = j; }
    }
    if (nloc != L) goto fail;
    for (int q = 0; q < nloc; ++q) code[b->loc[q]] ^= 1u;
    /* the corrected word must be a codeword */
    for (int i = 0; i < ns; ++i) {
        uint32_t acc = 0;
        const uint32_t beta = g->exp[(i + 1) % g->nn];
        for (int j = 0; j < n; ++j) acc = gf_mul(g, acc, beta) ^ (uint32_t)(code[j] & 1u);
        if (acc) { for (int q = 0; q < nloc; ++q) code[b->loc[q]] ^= 1u; goto fail; }
    }
    if (err_pos) for (int q = 0; q < nloc; ++q) err_pos[q] = b->loc[q];
    b->st.out++; b->st.corrected += (uint64_t)nloc;
    return nloc;
fail:
    b->st.failures++;
    return ATKDSP_E_DECODE;
}

int atkdsp_bch_describe(const atkdsp_bch *b, char *buf, size_t cap) {
    if (!b || !buf || !cap) return ATKDSP_E_ARG;
    snprintf(buf, cap, "BCH(%d,%d) t=%d over GF(2^%d) poly 0x%X%s", b->n, b->k, b->t, b->m,
             (unsigned)b->gfpoly, b->shorten ? ", shortened" : "");
    return ATKDSP_OK;
}

int atkdsp_bch_stats(const atkdsp_bch *b, atkdsp_fec_stats *st) {
    if (!b || !st) return ATKDSP_E_ARG;
    *st = b->st;
    return ATKDSP_OK;
}
