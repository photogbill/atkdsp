/* CRC recovery (header §17): an unknown CRC worked out from frames that
 * carry it. The arithmetic, written from the algebra rather than from any
 * tool:
 *
 *   For a frame whose covered bits are message M (L bits) then the w-bit
 *   field, with the field mapped back to the register value R (undo the
 *   little-endian byte layout, then undo refout) and the message bytes
 *   reflected when refin, the register identity of src/crc.c gives
 *       p = M' x^w + R  ==  I x^L + X'     (mod g)
 *   with I = init and X' = refout ? reflect(xorout) : xorout.
 *
 *   - two frames of one length: p_i + p_j == 0 (mod g)       I, X' cancel
 *   - three lengths, base i:    (p_i+p_j)(x^Li+x^Lk) + (p_i+p_k)(x^Li+x^Lj)
 *                               == 0 (mod g)                  I, X' cancel
 *     (this one also carries 1 + x^Gd, Gd the gcd of the length gaps, and a
 *     power of x; both are divided out)
 *   - an assumed init I0 (0 or all-ones): (p_i + I0 x^Li) + (p_j + I0 x^Lj)
 *     == 0 (mod g) for any lengths                            X' cancels
 *
 * The GCD of such polynomials is g times whatever spurious factor the frames
 * have not yet ruled out; g is G itself when its degree is w, else a degree-w
 * divisor of G (up to 8 spare degrees). init then comes from a w x w linear
 * solve, X' from one frame, and the set is checked by recomputing EVERY
 * frame's CRC with the engine. The search order is fixed and documented in
 * the header; reference.py §17 runs the same search with Python integers. */
#include "internal.h"

#define LOG10_2 0.30102999566398119521

/* ---- GF(2)[x] polynomials, bit i = coefficient of x^i ---------------------- */
typedef struct { uint64_t *w; int cap; int deg; } gp;

static int gp_init(gp *p, int maxdeg) {
    p->cap = maxdeg / 64 + 2;
    p->w = (uint64_t *)calloc((size_t)p->cap, sizeof *p->w);
    p->deg = -1;
    return p->w != NULL;
}
static void gp_free(gp *p) { free(p->w); p->w = NULL; }
static void gp_zero(gp *p) { memset(p->w, 0, (size_t)p->cap * sizeof *p->w); p->deg = -1; }
static void gp_fixdeg(gp *p, int from) {
    int q = from >= 0 ? from / 64 : p->cap - 1;
    if (q >= p->cap) q = p->cap - 1;
    for (; q >= 0; --q)
        if (p->w[q]) {
            int b = 63;
            while (!((p->w[q] >> b) & 1u)) --b;
            p->deg = 64 * q + b;
            return;
        }
    p->deg = -1;
}
static void gp_copy(gp *d, const gp *s) {
    memset(d->w, 0, (size_t)d->cap * sizeof *d->w);
    const int n = (s->deg >= 0) ? s->deg / 64 + 1 : 0;
    memcpy(d->w, s->w, (size_t)n * sizeof *d->w);
    d->deg = s->deg;
}
/* a ^= b << k (a must have room) */
static void gp_xor_shift(gp *a, const gp *b, int k) {
    if (b->deg < 0) return;
    const int q = k / 64, r = k % 64, nb = b->deg / 64 + 1;
    for (int j = 0; j < nb; ++j) {
        const uint64_t v = b->w[j];
        if (!v) continue;
        if (j + q < a->cap) a->w[j + q] ^= v << r;
        if (r && j + q + 1 < a->cap) a->w[j + q + 1] ^= v >> (64 - r);
    }
    const int top = b->deg + k;
    gp_fixdeg(a, top > a->deg ? top : a->deg);
}
static void gp_mod(gp *a, const gp *g) {
    while (a->deg >= g->deg && a->deg >= 0) {
        const int k = a->deg - g->deg;
        const int q = k / 64, r = k % 64, ng = g->deg / 64 + 1;
        for (int j = 0; j < ng; ++j) {
            const uint64_t v = g->w[j];
            if (j + q < a->cap) a->w[j + q] ^= v << r;
            if (r && j + q + 1 < a->cap) a->w[j + q + 1] ^= v >> (64 - r);
        }
        gp_fixdeg(a, a->deg);
    }
}
/* gcd into a; b is clobbered */
static void gp_gcd(gp *a, gp *b) {
    while (b->deg >= 0) {
        gp_mod(a, b);
        gp t = *a; *a = *b; *b = t;
    }
}
static int gp_tz(const gp *a) {
    if (a->deg < 0) return 0;
    for (int q = 0; q < a->cap; ++q)
        if (a->w[q]) { int b = 0; while (!((a->w[q] >> b) & 1u)) ++b; return 64 * q + b; }
    return 0;
}
static void gp_shr(gp *a, int k) {
    if (k <= 0 || a->deg < 0) return;
    const int q = k / 64, r = k % 64;
    for (int j = 0; j < a->cap; ++j) {
        uint64_t v = 0;
        if (j + q < a->cap) v = a->w[j + q] >> r;
        if (r && j + q + 1 < a->cap) v |= a->w[j + q + 1] << (64 - r);
        a->w[j] = v;
    }
    gp_fixdeg(a, a->deg);
}
/* q = a / g exactly; returns 1 when the remainder is zero. a is clobbered. */
static int gp_divexact(gp *q, gp *a, const gp *g) {
    gp_zero(q);
    while (a->deg >= g->deg && a->deg >= 0) {
        const int k = a->deg - g->deg;
        q->w[k / 64] |= 1ULL << (k % 64);
        gp_xor_shift(a, g, k);
    }
    gp_fixdeg(q, q->cap * 64 - 1);
    return a->deg < 0;
}
static void gp_from_bits(gp *p, const uint8_t *b, size_t n) {
    gp_zero(p);
    for (size_t i = 0; i < n; ++i)
        if (b[i] & 1u) { const size_t e = n - 1 - i; p->w[e / 64] |= 1ULL << (e % 64); }
    gp_fixdeg(p, (int)n);
}

/* ---- w-bit residues mod g (g = x^w + glow) --------------------------------- */
ATK_INLINE uint64_t wm(int w) { return w >= 64 ? ~0ULL : ((1ULL << w) - 1); }
ATK_INLINE uint64_t mulx(uint64_t v, uint64_t glow, int w) {
    const uint64_t top = (v >> (w - 1)) & 1u;
    return ((v << 1) & wm(w)) ^ (top ? glow : 0);
}
static uint64_t mulmod(uint64_t u, uint64_t v, uint64_t glow, int w) {
    uint64_t r = 0, t = u;
    for (int k = 0; k < w; ++k) { if ((v >> k) & 1u) r ^= t; t = mulx(t, glow, w); }
    return r;
}
static uint64_t xpow(size_t L, uint64_t glow, int w) {   /* x^L mod g, square-and-multiply */
    uint64_t r = 1, base = mulx(1, glow, w);              /* x (w >= 2) */
    if (w == 1) base = glow & 1u;
    while (L) { if (L & 1u) r = mulmod(r, base, glow, w); base = mulmod(base, base, glow, w); L >>= 1; }
    return r;
}
static uint64_t gp_resid(const gp *p, const gp *g, gp *tmp) {
    gp_copy(tmp, p);
    gp_mod(tmp, g);
    return tmp->w[0] & wm(g->deg);
}

/* Solve I * a == rhs (mod g) for w-bit I. Writes up to cap solutions. */
static int solve_init(uint64_t a, uint64_t rhs, uint64_t glow, int w, uint64_t *sol, int cap) {
    uint64_t col[64], rows[64];
    uint64_t t = a;
    for (int k = 0; k < w; ++k) { col[k] = t; t = mulx(t, glow, w); }
    /* row-major augmented matrix: row r = equation for bit r; bits 0..w-1 the unknowns, bit w (in aug[]) the rhs */
    uint8_t aug[64];
    for (int r = 0; r < w; ++r) {
        uint64_t row = 0;
        for (int k = 0; k < w; ++k) row |= ((col[k] >> r) & 1u) << k;
        rows[r] = row;
        aug[r] = (uint8_t)((rhs >> r) & 1u);
    }
    int piv[64], np = 0;
    uint64_t pivmask = 0;
    for (int c = 0, r0 = 0; c < w && r0 < w; ++c) {
        int p = -1;
        for (int r = r0; r < w; ++r) if ((rows[r] >> c) & 1u) { p = r; break; }
        if (p < 0) continue;
        uint64_t tr = rows[p]; rows[p] = rows[r0]; rows[r0] = tr;
        uint8_t ta = aug[p]; aug[p] = aug[r0]; aug[r0] = ta;
        for (int r = 0; r < w; ++r)
            if (r != r0 && ((rows[r] >> c) & 1u)) { rows[r] ^= rows[r0]; aug[r] ^= aug[r0]; }
        piv[np++] = c; pivmask |= 1ULL << c;
        ++r0;
    }
    for (int r = np; r < w; ++r) if (aug[r]) return 0;       /* inconsistent */
    int freec[64], nf = 0;
    for (int c = 0; c < w; ++c) if (!((pivmask >> c) & 1u)) freec[nf++] = c;
    if (nf > 8) nf = 8;                                       /* at most 256 solutions */
    int out = 0;
    for (uint32_t m = 0; m < (1u << nf) && out < cap; ++m) {
        uint64_t x = 0;
        for (int f = 0; f < nf; ++f) if ((m >> f) & 1u) x |= 1ULL << freec[f];
        for (int i = 0; i < np; ++i) {
            uint64_t v = aug[i];
            v ^= (uint64_t)atk_parity64(rows[i] & x & ~(1ULL << piv[i]));
            if (v) x |= 1ULL << piv[i];
        }
        sol[out++] = x;
    }
    return out;
}

/* ---- the search -------------------------------------------------------------- */
typedef struct {
    int refin, refout, layout;
} combo_t;
static const combo_t COMBOS[8] = {
    {0, 0, 0}, {1, 1, 0}, {0, 1, 0}, {1, 0, 0}, {0, 0, 1}, {1, 1, 1}, {0, 1, 1}, {1, 0, 1}};

typedef struct {
    const uint8_t *bits;
    const size_t *len, *start;
    int nf;
    size_t maxlen;
    /* per evaluation */
    gp *p;                      /* nf polynomials */
    size_t *L;                  /* message lengths */
    uint8_t *buf;               /* scratch bits */
    gp t1, t2, t3, t4, acc, ge;
} ctx_t;

static uint64_t field_value(const uint8_t *f, int w, int layout) {
    uint64_t v = 0;
    if (!layout) { for (int i = 0; i < w; ++i) v = (v << 1) | (f[i] & 1u); return v; }
    for (int k = 0; k < w / 8; ++k) {
        uint64_t byte = 0;
        for (int i = 0; i < 8; ++i) byte = (byte << 1) | (f[8 * k + i] & 1u);
        v |= byte << (8 * k);
    }
    return v;
}

/* Build p_i for frame i: coverage [h, F - t), field its last w bits. */
static int build_p(ctx_t *c, int i, int w, const combo_t *cb, size_t h, size_t t) {
    const size_t F = c->len[i];
    if (F < h + t + (size_t)w + 1) return 0;
    const uint8_t *cov = c->bits + c->start[i] + h;
    const size_t C = F - h - t, L = C - (size_t)w;
    c->L[i] = L;
    if (cb->refin) {
        if (L % 8) return 0;
        for (size_t k = 0; k < L; k += 8)
            for (int j = 0; j < 8; ++j) c->buf[k + (size_t)j] = cov[k + 7 - (size_t)j];
    } else {
        memcpy(c->buf, cov, L);
    }
    uint64_t R = field_value(cov + L, w, cb->layout);
    if (cb->refout) R = atk_reflect(R, w);
    for (int j = 0; j < w; ++j) c->buf[L + (size_t)j] = (uint8_t)((R >> (w - 1 - j)) & 1u);
    gp_from_bits(&c->p[i], c->buf, C);
    return 1;
}

static long long llgcd(long long a, long long b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b) { const long long t = a % b; a = b; b = t; }
    return a;
}

/* G for one (offset, w, combo, hyp); 0 when the frames give nothing usable. */
static int compute_G(ctx_t *c, int w, int hyp, gp *G) {
    const int nf = c->nf;
    gp_zero(G);
    int have = 0;
    if (hyp == 0) {
        /* equal-length pairs */
        gp *GD = &c->t3;
        gp_zero(GD);
        int nd = 0;
        for (int i = 0; i < nf; ++i) {
            int first = -1;
            for (int k = 0; k < i; ++k) if (c->L[k] == c->L[i]) { first = k; break; }
            if (first < 0) continue;
            gp_copy(&c->t1, &c->p[i]);
            gp_xor_shift(&c->t1, &c->p[first], 0);
            if (c->t1.deg < 0) continue;
            if (!nd) gp_copy(GD, &c->t1);
            else { gp_copy(&c->t2, &c->t1); gp_gcd(GD, &c->t2); }
            ++nd;
        }
        /* three-length combinations, base i < j < k, at most 16 */
        gp *GE = &c->ge;
        gp_zero(GE);
        int ne = 0;
        long long gd = 0;
        for (int i = 0; i < nf && ne < 16; ++i)
            for (int j = i + 1; j < nf && ne < 16; ++j) {
                if (c->L[j] == c->L[i]) continue;
                for (int k = j + 1; k < nf && ne < 16; ++k) {
                    if (c->L[k] == c->L[i]) continue;
                    /* E = (p_i+p_j)(x^Li + x^Lk) + (p_i+p_k)(x^Li + x^Lj) */
                    gp_copy(&c->t1, &c->p[i]); gp_xor_shift(&c->t1, &c->p[j], 0);   /* d_ij */
                    gp_copy(&c->t2, &c->p[i]); gp_xor_shift(&c->t2, &c->p[k], 0);   /* d_ik */
                    gp_zero(&c->acc);
                    gp_xor_shift(&c->acc, &c->t1, (int)c->L[i]);
                    gp_xor_shift(&c->acc, &c->t1, (int)c->L[k]);
                    gp_xor_shift(&c->acc, &c->t2, (int)c->L[i]);
                    gp_xor_shift(&c->acc, &c->t2, (int)c->L[j]);
                    if (c->acc.deg < 0) continue;
                    gd = llgcd(gd, llgcd((long long)c->L[i] - (long long)c->L[j],
                                         (long long)c->L[i] - (long long)c->L[k]));
                    if (!ne) gp_copy(GE, &c->acc);
                    else { gp_copy(&c->t4, &c->acc); gp_gcd(GE, &c->t4); }
                    ++ne;
                }
            }
        if (ne >= 2) {
            gp_shr(GE, gp_tz(GE));
            /* divide out 1 + x^gd */
            gp_zero(&c->t4);
            c->t4.w[0] = 1; c->t4.w[gd / 64] |= 1ULL << (gd % 64);
            gp_fixdeg(&c->t4, (int)gd);
            gp_copy(&c->t1, GE);
            if (gp_divexact(&c->t2, &c->t1, &c->t4)) gp_copy(GE, &c->t2);
        }
        if (nd && ne >= 2) { gp_copy(G, GD); gp_copy(&c->t1, GE); gp_gcd(G, &c->t1); have = 1; }
        else if (nd) { gp_copy(G, GD); have = 1; }
        else if (ne >= 2) { gp_copy(G, GE); have = 1; }
    } else {
        /* assumed init: f_i = p_i + I0 x^Li, polys f_0 + f_j */
        const uint64_t I0 = (hyp == 2) ? wm(w) : 0;
        int np = 0;
        for (int j = 1; j < nf; ++j) {
            gp_copy(&c->t1, &c->p[0]);
            gp_xor_shift(&c->t1, &c->p[j], 0);
            if (I0) {
                gp_zero(&c->t2); c->t2.w[0] = I0; gp_fixdeg(&c->t2, 63);
                gp_xor_shift(&c->t1, &c->t2, (int)c->L[0]);
                gp_xor_shift(&c->t1, &c->t2, (int)c->L[j]);
            }
            if (c->t1.deg < 0) continue;
            if (!np) gp_copy(G, &c->t1);
            else { gp_copy(&c->t2, &c->t1); gp_gcd(G, &c->t2); }
            ++np;
        }
        have = np >= 2;
    }
    if (!have || G->deg < 0) return 0;
    gp_shr(G, gp_tz(G));
    return G->deg >= w;
}

/* Does the set reproduce every frame (the engine's own answer)? */
static int verify(ctx_t *c, const atkdsp_crcspec *s, int layout, size_t h, size_t t) {
    for (int i = 0; i < c->nf; ++i) {
        const size_t F = c->len[i];
        const uint8_t *cov = c->bits + c->start[i] + h;
        const size_t L = F - h - t - (size_t)s->width;
        uint64_t v;
        if (atkdsp_crc(cov, L, s, &v) != ATKDSP_OK) return 0;
        if (field_value(cov + L, s->width, layout) != v) return 0;
    }
    return 1;
}

static int same_cand(const atkdsp_crccand *a, const atkdsp_crccand *b) {
    return a->spec.width == b->spec.width && a->spec.poly == b->spec.poly
        && a->spec.init == b->spec.init && a->spec.xorout == b->spec.xorout
        && a->spec.refin == b->spec.refin && a->spec.refout == b->spec.refout
        && a->layout == b->layout && a->header_bits == b->header_bits
        && a->trailer_bits == b->trailer_bits;
}

static int simple_set(const atkdsp_crcspec *s) {
    const uint64_t ones = wm(s->width);
    return (s->init == 0 || s->init == ones) && (s->xorout == 0 || s->xorout == ones);
}

/* the ranking: catalogue sets, then simple ones, then log10_pfa, then a
 * byte-aligned offset (a prior from real protocols, not a deduction), then
 * the smaller offset */
static int cand_before(const atkdsp_crccand *a, const atkdsp_crccand *b) {
    const int ca = a->catalog >= 0, cb = b->catalog >= 0;
    if (ca != cb) return ca > cb;
    const int sa = simple_set(&a->spec), sb = simple_set(&b->spec);
    if (sa != sb) return sa > sb;
    if (a->log10_pfa != b->log10_pfa) return a->log10_pfa < b->log10_pfa;
    const int oa = a->header_bits + a->trailer_bits, ob = b->header_bits + b->trailer_bits;
    if ((oa % 8 == 0) != (ob % 8 == 0)) return oa % 8 == 0;    /* a prior: byte-aligned fields */
    return oa < ob;
}

ptrdiff_t atkdsp_crc_recover(const uint8_t *bits, const size_t *nbits, int nframes, int width,
                             int pos, int max_header, double pfa, atkdsp_crccand *out, size_t cap) {
    if (!bits || !nbits || nframes < 1 || width < 0 || width > 64 || (width && width < 3)
        || (pos != ATKDSP_CRC_END && pos != ATKDSP_CRC_SEARCH) || max_header < 0 || (!out && cap))
        return ATKDSP_E_ARG;
    if (!(pfa > 0.0)) pfa = 1e-3;
    if (nframes < 3) return 0;
    ctx_t c;
    memset(&c, 0, sizeof c);
    c.bits = bits; c.len = nbits; c.nf = nframes;
    size_t *start = (size_t *)malloc((size_t)nframes * sizeof *start);
    if (!start) return ATKDSP_E_NOMEM;
    size_t minlen = (size_t)-1, at = 0;
    for (int i = 0; i < nframes; ++i) {
        if (nbits[i] > ((size_t)1 << 24)) { free(start); return ATKDSP_E_ARG; }  /* 16 Mbit a frame */
        start[i] = at; at += nbits[i];
        if (nbits[i] > c.maxlen) c.maxlen = nbits[i];
        if (nbits[i] < minlen) minlen = nbits[i];
    }
    c.start = start;
    const int maxdeg = 2 * (int)c.maxlen + 192;
    c.p = (gp *)calloc((size_t)nframes, sizeof *c.p);
    c.L = (size_t *)calloc((size_t)nframes, sizeof *c.L);
    c.buf = (uint8_t *)malloc(c.maxlen + 64);
    int ok = c.p && c.L && c.buf;
    for (int i = 0; ok && i < nframes; ++i) ok = gp_init(&c.p[i], maxdeg);
    ok = ok && gp_init(&c.t1, maxdeg) && gp_init(&c.t2, maxdeg) && gp_init(&c.t3, maxdeg)
            && gp_init(&c.t4, maxdeg) && gp_init(&c.acc, maxdeg) && gp_init(&c.ge, maxdeg);
    gp G, Gc, gq, tmp;
    memset(&G, 0, sizeof G); memset(&Gc, 0, sizeof Gc); memset(&gq, 0, sizeof gq); memset(&tmp, 0, sizeof tmp);
    ok = ok && gp_init(&G, maxdeg) && gp_init(&Gc, maxdeg) && gp_init(&gq, maxdeg) && gp_init(&tmp, maxdeg);
    size_t ncand = 0, ccap = 16;
    atkdsp_crccand *cand = ok ? (atkdsp_crccand *)malloc(ccap * sizeof *cand) : NULL;
    ptrdiff_t ret = 0;
    if (!cand) { ret = ATKDSP_E_NOMEM; goto done; }

    int distinct = 0;
    for (int i = 1; i < nframes; ++i) if (nbits[i] != nbits[0]) distinct = 1;
    const size_t max_off = (pos == ATKDSP_CRC_END) ? (size_t)max_header
                         : (minlen > 7 ? minlen - 7 : 0);
    double trials = 0.0;
    const double log10_thr = log10(pfa);

    for (size_t off = 0; off <= max_off; ++off) {
        const size_t h = (pos == ATKDSP_CRC_END) ? off : 0;
        const size_t t = (pos == ATKDSP_CRC_END) ? 0 : off;
        const int wlo = width ? width : 3, whi = width ? width : 64;
        for (int w = wlo; w <= whi; ++w) {
            for (int ci = 0; ci < 8; ++ci) {
                const combo_t *cb = &COMBOS[ci];
                if (cb->layout && (w % 8)) continue;
                if (!width && off > 0 && ci > 0) continue;   /* offsets scanned for the plain form only */
                int built = 1;
                for (int i = 0; i < nframes && built; ++i) built = build_p(&c, i, w, cb, h, t);
                if (!built) continue;
                for (int hyp = 0; hyp < 3; ++hyp) {
                    trials += 1.0;
                    if (!compute_G(&c, w, hyp, &G)) continue;
                    const int excess_deg = G.deg - w;
                    if (excess_deg > 8) continue;
                    /* candidate g: G, or G / h for each degree-excess h with both end terms */
                    const uint32_t nh = excess_deg ? (1u << (excess_deg > 1 ? excess_deg - 1 : 0)) : 1u;
                    for (uint32_t hm = 0; hm < nh; ++hm) {
                        if (excess_deg == 0) {
                            gp_copy(&gq, &G);
                        } else {
                            gp_zero(&tmp);
                            uint64_t hv = 1ULL | (1ULL << excess_deg) | ((uint64_t)hm << 1);
                            tmp.w[0] = hv; gp_fixdeg(&tmp, 63);
                            gp_copy(&Gc, &G);
                            if (!gp_divexact(&gq, &Gc, &tmp)) continue;
                        }
                        if (gq.deg != w || !(gq.w[0] & 1u)) continue;
                        const uint64_t glow = (w == 64) ? gq.w[0] : (gq.w[0] & wm(w));
                        /* residues and powers */
                        uint64_t r[3] = {0, 0, 0}, xl[3] = {0, 0, 0};
                        int ia = 0, ib = -1;
                        for (int i = 1; i < nframes; ++i) if (c.L[i] != c.L[0]) { ib = i; break; }
                        r[0] = gp_resid(&c.p[ia], &gq, &tmp); xl[0] = xpow(c.L[ia], glow, w);
                        uint64_t sols[256];
                        int nsol;
                        int init_known = 1;
                        if (hyp == 1) { sols[0] = 0; nsol = 1; }
                        else if (hyp == 2) { sols[0] = wm(w); nsol = 1; }
                        else if (ib >= 0) {
                            r[1] = gp_resid(&c.p[ib], &gq, &tmp); xl[1] = xpow(c.L[ib], glow, w);
                            nsol = solve_init(xl[0] ^ xl[1], r[0] ^ r[1], glow, w, sols, 256);
                        } else { sols[0] = 0; nsol = 1; init_known = 0; }
                        int nver = 0;
                        atkdsp_crccand best;
                        memset(&best, 0, sizeof best);
                        for (int si = 0; si < nsol; ++si) {
                            const uint64_t I = sols[si];
                            const uint64_t Xp = r[0] ^ mulmod(I, xl[0], glow, w);
                            atkdsp_crcspec s;
                            memset(&s, 0, sizeof s);
                            s.width = w; s.poly = glow; s.init = I; s.refin = cb->refin; s.refout = cb->refout;
                            s.xorout = cb->refout ? atk_reflect(Xp, w) : Xp;
                            if (!verify(&c, &s, cb->layout, h, t)) continue;
                            if (!nver) { best.spec = s; }
                            ++nver;
                        }
                        if (!nver) continue;
                        if (nver > 1) init_known = 0;
                        /* an ASSUMED init is known only if it is the one init the frames allow */
                        if (hyp != 0) {
                            if (ib < 0) init_known = 0;
                            else {
                                uint64_t alt[256];
                                r[1] = gp_resid(&c.p[ib], &gq, &tmp); xl[1] = xpow(c.L[ib], glow, w);
                                const int na = solve_init(xl[0] ^ xl[1], r[0] ^ r[1], glow, w, alt, 256);
                                int nfit = 0;
                                for (int ai = 0; ai < na && nfit < 2; ++ai) {
                                    const uint64_t Xa = r[0] ^ mulmod(alt[ai], xl[0], glow, w);
                                    atkdsp_crcspec s2 = best.spec;
                                    s2.init = alt[ai];
                                    s2.xorout = cb->refout ? atk_reflect(Xa, w) : Xa;
                                    if (verify(&c, &s2, cb->layout, h, t)) ++nfit;
                                }
                                if (nfit != 1) init_known = 0;
                            }
                        }
                        const int u = (hyp == 0 && distinct) ? 3 : 2;
                        best.layout = cb->layout; best.header_bits = (int)h; best.trailer_bits = (int)t;
                        best.init_known = init_known; best.frames_ok = nframes;
                        best.catalog = -1;
                        best.log10_pfa = (double)(nframes - u) * (double)w;   /* finished below */
                        if (!init_known) {   /* a catalogue set of this shape that fits */
                            const int nc = atkdsp_crc_catalog_count();
                            for (int k = 0; k < nc; ++k) {
                                atkdsp_crcspec cs;
                                atkdsp_crc_catalog(k, &cs);
                                if (cs.width != w || cs.poly != glow || cs.refin != cb->refin
                                    || cs.refout != cb->refout) continue;
                                if (verify(&c, &cs, cb->layout, h, t)) {
                                    best.spec.init = cs.init; best.spec.xorout = cs.xorout;
                                    best.catalog = k;
                                    break;
                                }
                            }
                        } else {
                            best.catalog = atk_crc_catalog_find(&best.spec);
                        }
                        best.spec.check = atk_crc_check_value(&best.spec);
                        int dup = 0;
                        for (size_t q = 0; q < ncand; ++q) if (same_cand(&cand[q], &best)) { dup = 1; break; }
                        if (dup) continue;
                        if (ncand == ccap) {
                            ccap *= 2;
                            atkdsp_crccand *nc2 = (atkdsp_crccand *)realloc(cand, ccap * sizeof *cand);
                            if (!nc2) { ret = ATKDSP_E_NOMEM; goto done; }
                            cand = nc2;
                        }
                        cand[ncand++] = best;
                    }
                }
            }
        }
    }
    /* every offset scanned: figures with the whole search's trials, then keep
     * what passes. A bit constant across all frames — a header before the
     * coverage, a trailer after the field — is absorbed into init or xorout
     * by an equally valid set at another offset, and a polynomial that
     * factors is also exactly a CRC of its factors: the frames cannot break
     * those ties, so every set that reproduces them is reported, and ranked
     * catalogue first, then simple (init and xorout each 0 or all ones), then
     * by log10_pfa, then byte-aligned offsets, then the smallest offset. */
    {
        size_t kept = 0;
        for (size_t q = 0; q < ncand; ++q) {
            const double eq_bits = cand[q].log10_pfa;
            cand[q].log10_pfa = log10(trials) - eq_bits * LOG10_2;
            if (eq_bits > 0.0 && cand[q].log10_pfa <= log10_thr) cand[kept++] = cand[q];
        }
        ncand = kept;
    }
    for (size_t q = 0; q < ncand; ++q) {
        int alt = 0;
        for (size_t r = 0; r < ncand; ++r)
            if (r != q && cand[r].spec.width == cand[q].spec.width) ++alt;
        cand[q].alternatives = alt;
    }
    for (size_t a = 1; a < ncand; ++a) {        /* stable insertion sort */
        atkdsp_crccand x = cand[a];
        size_t b = a;
        while (b > 0 && cand_before(&x, &cand[b - 1])) { cand[b] = cand[b - 1]; --b; }
        cand[b] = x;
    }
    {
        const size_t m = ncand < cap ? ncand : cap;
        if (m) memcpy(out, cand, m * sizeof *cand);
        ret = (ptrdiff_t)m;
    }
done:
    if (c.p) for (int i = 0; i < nframes; ++i) gp_free(&c.p[i]);
    free(c.p); free(c.L); free(c.buf); free(start);
    gp_free(&c.t1); gp_free(&c.t2); gp_free(&c.t3); gp_free(&c.t4); gp_free(&c.acc); gp_free(&c.ge);
    gp_free(&G); gp_free(&Gc); gp_free(&gq); gp_free(&tmp);
    free(cand);
    return ret;
}
