/* 23. Cyclic probes (docs/ATKDSP_PLAN.md §B4): atkdsp_cyclic (a spectral
 * line at a candidate symbol rate after a nonlinearity — siga's
 * line-versus-hump test, natively) and atkdsp_cp_probe (OFDM: the lag
 * correlation a cyclic prefix leaves at the FFT length, and the symbol-rate
 * line of its windowed magnitude that gives the CP length). Definitions in
 * atkdsp.h §23; the twin is python/atkdsp/reference.py §23.
 *
 * BOTH ARE SEGMENT-BUFFERED. Segments of seg_len samples are cut on the
 * global sample index since reset, never on the caller's blocks, so every
 * number depends only on sample values and the stream's position (header
 * rule 3): a kernel fed in blocks of 1 and of 8192 reports the same. The
 * input is copied into a segment buffer (plus the history the lag products
 * need) and every candidate is measured when the segment completes: the
 * nonlinearity, its mean removed (DC — |x|^2's pedestal, a carrier's m-th
 * power — would otherwise leak into the line), a periodic Hann window
 * (sidelobes fall as 1/d^3, so a candidate 10 bins from DC sees ~1e-3 of it
 * and one 100 bins away nothing), and the DFT at the candidate frequency by
 * Goertzel (X = (s1 - e^{-jw} s2) e^{-jw(L-1)}), referred to the global
 * index with a per-segment rotation. The last nseg segments' phasors p_k
 * and unwindowed energies e_k are kept per candidate:
 *   level     |sum p_k|^2 / (sum e_k * sum w^2 / L): for white input (the
 *             nonlinearity of white noise is white) an exponential of mean 1,
 *             so a line of level T exceeds the noise with probability e^-T
 *   coherence |sum p_k| / sum |p_k|: 1 for a line, ~0.89 / sqrt K for noise
 *             or a hump (a continuum's phasors have random phases however
 *             strong they are — this is what tells a line from a hump)
 *   delta_hz  arg(sum p_k conj p_{k-1}) / (2 pi T_seg): the line's offset
 *             from the candidate (valid when there is a line); the aligned
 *             level and coherence have the phasors derotated by it, so a
 *             rate a few hundred ppm off still reads
 * THE THRESHOLDS ARE DERIVED, THEN MEASURED (atkdsp_cyclic_thresholds):
 * level_thr = -ln(pfa / ncand); coh_thr = sqrt(-ln(pfa / ncand) / (0.785 K))
 * from |sum p|^2 ~ Exp(K) against (sum |p|)^2 ~ (0.886 K)^2 for K Rayleigh
 * phasors. Measured on 4e5 draws (tests/test_phase3.py keeps the check):
 * at K = 16 the 1e-3 point sits at 0.68 against the formula's 0.74, at
 * K = 32 at 0.50 against 0.52, at K = 50 at 0.41 against 0.42 — the formula
 * is conservative by a few tens of percent and converges; below K = 16 it
 * exceeds 1 at 1e-3 and create refuses the configuration. The level's 1e-3
 * point measured 6.6 against 6.9 at K = 50 (conservative too). The verdict
 * LINE needs both; it is withheld (WARMUP) until nseg segments are in.
 *
 * cp_probe's score is |sum x[m] x*[m-N]| / sum (|x[m]|^2 + |x[m-N]|^2) / 2
 * over every pair since reset: for white noise a Rayleigh of scale
 * 1 / sqrt(M pairs), so score_thr = sqrt(-ln(pfa / ncand) / M); an OFDM
 * signal with a prefix of fraction rho at SNR g scores about
 * rho g / (1 + g). Its CP stage feeds |z[m]|, z the sum of the lag product
 * over the last cp samples, to the line machinery at fs / period (period =
 * lag + cp unless given): z peaks once a symbol when the window sits on the
 * prefix. |z| of noise is coloured over cp samples, so its level is relative
 * only; the coherence is the test, and LINE confirms the length.
 */
#include "internal.h"

#include <stdio.h>

#define CYC_MAX_CAND 64
#define CYC_SEG_MIN 16
#define CYC_SEG_MAX (1 << 16)
#define CYC_NSEG_MAX 4096
#define CP_LAG_MAX (1 << 16)
#define CP_CP_MAX (1 << 14)

/* ---- the line accumulator ------------------------------------------------------ */
typedef struct {
    double omega;              /* radians per sample                            */
    double c2, cw_re, cw_im;   /* 2 cos w, e^{-jw}                              */
    double fin_re, fin_im;     /* e^{-jw(L-1)}                                  */
    double rot_re, rot_im;     /* e^{-jw * (first index of the next segment)}   */
    double step_re, step_im;   /* e^{-jwL}                                      */
    double *p_re, *p_im, *e;   /* the last K segments (ring)                    */
    int K, head, count;
} line_acc;

static int line_init(line_acc *a, double omega, int L, int K) {
    memset(a, 0, sizeof *a);
    a->omega = omega;
    a->c2 = 2.0 * cos(omega);
    a->cw_re = cos(omega); a->cw_im = -sin(omega);
    a->fin_re = cos(omega * (double)(L - 1)); a->fin_im = -sin(omega * (double)(L - 1));
    a->step_re = cos(omega * (double)L); a->step_im = -sin(omega * (double)L);
    a->rot_re = 1.0; a->rot_im = 0.0;
    a->K = K;
    a->p_re = (double *)calloc((size_t)K, sizeof(double));
    a->p_im = (double *)calloc((size_t)K, sizeof(double));
    a->e = (double *)calloc((size_t)K, sizeof(double));
    return a->p_re && a->p_im && a->e;
}
static void line_free(line_acc *a) { free(a->p_re); free(a->p_im); free(a->e); }
static void line_reset(line_acc *a) {
    a->rot_re = 1.0; a->rot_im = 0.0; a->head = a->count = 0;
}

/* One complete segment of the (mean-removed) input v (L complex values, as
 * re/im arrays) with window w: push its phasor and energy. */
static void line_segment(line_acc *a, const double *vre, const double *vim, const double *w, int L) {
    double s1r = 0.0, s1i = 0.0, s2r = 0.0, s2i = 0.0, e = 0.0;
    for (int n = 0; n < L; ++n) {
        const double xr = w[n] * vre[n], xi = w[n] * vim[n];
        const double s0r = xr + a->c2 * s1r - s2r, s0i = xi + a->c2 * s1i - s2i;
        s2r = s1r; s2i = s1i; s1r = s0r; s1i = s0i;
        e += vre[n] * vre[n] + vim[n] * vim[n];
    }
    /* X = (s1 - e^{-jw} s2) e^{-jw(L-1)}, then the global rotation */
    double xr = s1r - (a->cw_re * s2r - a->cw_im * s2i);
    double xi = s1i - (a->cw_re * s2i + a->cw_im * s2r);
    double yr = xr * a->fin_re - xi * a->fin_im, yi = xr * a->fin_im + xi * a->fin_re;
    const double pr = yr * a->rot_re - yi * a->rot_im, pi = yr * a->rot_im + yi * a->rot_re;
    a->p_re[a->head] = pr; a->p_im[a->head] = pi; a->e[a->head] = e;
    a->head = (a->head + 1) % a->K;
    if (a->count < a->K) a->count++;
    const double nr = a->rot_re * a->step_re - a->rot_im * a->step_im;
    const double ni = a->rot_re * a->step_im + a->rot_im * a->step_re;
    a->rot_re = nr; a->rot_im = ni;
}

/* The statistics over the segments held, in the twin's order (oldest first). */
typedef struct {
    double level, coherence, delta_hz, level_aligned, coherence_aligned;
    int nseg;
} line_stats;

static void line_read(const line_acc *a, double wsum2_over_L, double t_seg, line_stats *s) {
    memset(s, 0, sizeof *s);
    const int K = a->count;
    s->nseg = K;
    if (!K) return;
    const int first = (a->head - K + a->K) % a->K;
    double sr = 0.0, si = 0.0, sa = 0.0, se = 0.0, dr = 0.0, di = 0.0;
    double lpr = 0.0, lpi = 0.0;
    for (int k = 0; k < K; ++k) {
        const int j = (first + k) % a->K;
        const double pr = a->p_re[j], pi = a->p_im[j];
        sr += pr; si += pi;
        sa += sqrt(pr * pr + pi * pi);
        se += a->e[j];
        if (k) { dr += pr * lpr + pi * lpi; di += pi * lpr - pr * lpi; }
        lpr = pr; lpi = pi;
    }
    const double den = se * wsum2_over_L;
    s->level = den > 0.0 ? (sr * sr + si * si) / den : 0.0;
    s->coherence = sa > 0.0 ? sqrt(sr * sr + si * si) / sa : 0.0;
    if (K >= 2 && (dr != 0.0 || di != 0.0)) {
        const double dphi = atan2(di, dr);           /* radians per segment */
        s->delta_hz = dphi / (2.0 * ATK_PI * t_seg);
        double qr = 0.0, qi = 0.0;
        for (int k = 0; k < K; ++k) {
            const int j = (first + k) % a->K;
            const double ang = -dphi * (double)k;
            const double cr = cos(ang), ci = sin(ang);
            qr += a->p_re[j] * cr - a->p_im[j] * ci;
            qi += a->p_re[j] * ci + a->p_im[j] * cr;
        }
        s->level_aligned = den > 0.0 ? (qr * qr + qi * qi) / den : 0.0;
        s->coherence_aligned = sa > 0.0 ? sqrt(qr * qr + qi * qi) / sa : 0.0;
    } else {
        s->level_aligned = s->level;
        s->coherence_aligned = s->coherence;
    }
}

static double *hann_periodic(int L, double *wsum2) {
    double *w = (double *)malloc((size_t)L * sizeof *w);
    if (!w) return NULL;
    double s = 0.0;
    for (int n = 0; n < L; ++n) {
        w[n] = 0.5 * (1.0 - cos(2.0 * ATK_PI * (double)n / (double)L));
        s += w[n] * w[n];
    }
    *wsum2 = s;
    return w;
}

void atkdsp_cyclic_thresholds(int nseg, double pfa_each, double *level_thr, double *coh_thr) {
    const double l = -log(pfa_each);
    if (level_thr) *level_thr = l;
    if (coh_thr) *coh_thr = nseg > 0 ? sqrt(l / (0.785 * (double)nseg)) : 2.0;
}

/* ---- 23a. atkdsp_cyclic --------------------------------------------------------- */
struct atkdsp_cyclic {
    double fs, pfa, pfa_each, level_thr, coh_thr, wsum2;
    int L, K, ncand;
    atkdsp_cyc_cand cand[CYC_MAX_CAND];
    line_acc acc[CYC_MAX_CAND];
    double *w;
    atkdsp_cf32 *seg;          /* L + 2 samples: two of history (DM, FDIFF)      */
    int pos;                   /* samples in the current segment               */
    int have_prev;             /* history samples valid: 0, 1 or 2             */
    double *vre, *vim;         /* scratch, L                                   */
    uint64_t samples, segments, resets;
};

static int cyc_cand_ok(const atkdsp_cyc_cand *c, double fs) {
    if (!(c->rate_hz > 0.0) || !(c->rate_hz < 0.5 * fs)) return 0;
    if (c->nonlin < 0 || c->nonlin > 5) return 0;
    return 1;
}

atkdsp_cyclic *atkdsp_cyclic_create(double fs, const atkdsp_cyc_cand *cands, int ncand,
                                    int seg_len, int nseg, double pfa) {
    if (!(fs > 0.0) || !isfinite(fs) || !cands || ncand < 1 || ncand > CYC_MAX_CAND) return NULL;
    if (seg_len < CYC_SEG_MIN || seg_len > CYC_SEG_MAX || nseg < 2 || nseg > CYC_NSEG_MAX) return NULL;
    if (!(pfa > 0.0 && pfa < 0.5)) return NULL;
    for (int i = 0; i < ncand; ++i) if (!cyc_cand_ok(&cands[i], fs)) return NULL;
    double lt, ct;
    atkdsp_cyclic_thresholds(nseg, pfa / (double)ncand, &lt, &ct);
    if (ct > 1.0) return NULL;               /* the coherence test cannot be met: more segments */
    atkdsp_cyclic *c = (atkdsp_cyclic *)calloc(1, sizeof *c);
    if (!c) return NULL;
    c->fs = fs; c->pfa = pfa; c->pfa_each = pfa / (double)ncand; c->level_thr = lt; c->coh_thr = ct;
    c->L = seg_len; c->K = nseg; c->ncand = ncand;
    memcpy(c->cand, cands, (size_t)ncand * sizeof *cands);
    c->w = hann_periodic(seg_len, &c->wsum2);
    c->seg = (atkdsp_cf32 *)malloc(((size_t)seg_len + 2) * sizeof *c->seg);
    c->vre = (double *)malloc((size_t)seg_len * sizeof(double));
    c->vim = (double *)malloc((size_t)seg_len * sizeof(double));
    int ok = c->w && c->seg && c->vre && c->vim;
    for (int i = 0; i < ncand && ok; ++i)
        ok = line_init(&c->acc[i], 2.0 * ATK_PI * cands[i].rate_hz / fs, seg_len, nseg);
    if (!ok) { atkdsp_cyclic_destroy(c); return NULL; }
    return c;
}

void atkdsp_cyclic_destroy(atkdsp_cyclic *c) {
    if (!c) return;
    for (int i = 0; i < c->ncand; ++i) line_free(&c->acc[i]);
    free(c->w); free(c->seg); free(c->vre); free(c->vim);
    free(c);
}

void atkdsp_cyclic_reset(atkdsp_cyclic *c) {
    if (!c) return;
    for (int i = 0; i < c->ncand; ++i) line_reset(&c->acc[i]);
    c->pos = 0; c->have_prev = 0;
    c->resets++;
}

/* The nonlinearity of the segment (with the two samples before it for DM
 * and FDIFF) into vre/vim, mean removed. */
static double arg_prod(const atkdsp_cf32 *a, const atkdsp_cf32 *b) {   /* arg(a conj b) */
    const double re = (double)a->re * b->re + (double)a->im * b->im;
    const double im = (double)a->im * b->re - (double)a->re * b->im;
    return atan2(im, re);
}
static void nonlin_segment(const atkdsp_cf32 *seg, int L, int have_prev, const atkdsp_cyc_cand *cd,
                           double *vre, double *vim) {
    const atkdsp_cf32 *x = seg + 2;                 /* seg[0..1] are the two previous samples */
    for (int n = 0; n < L; ++n) {
        double re = x[n].re, im = x[n].im;
        if (cd->hard_limit && cd->nonlin != ATKDSP_NL_ENV2) {
            const double m = sqrt(re * re + im * im);
            if (m > 0.0) { re /= m; im /= m; } else { re = im = 0.0; }
        }
        double yr, yi;
        switch (cd->nonlin) {
        case ATKDSP_NL_ENV2: yr = re * re + im * im; yi = 0.0; break;
        case ATKDSP_NL_POW2: yr = re * re - im * im; yi = 2.0 * re * im; break;
        case ATKDSP_NL_POW4: {
            const double ar = re * re - im * im, ai = 2.0 * re * im;
            yr = ar * ar - ai * ai; yi = 2.0 * ar * ai; break; }
        case ATKDSP_NL_POW8: {
            const double ar = re * re - im * im, ai = 2.0 * re * im;
            const double br = ar * ar - ai * ai, bi = 2.0 * ar * ai;
            yr = br * br - bi * bi; yi = 2.0 * br * bi; break; }
        case ATKDSP_NL_DM: {                         /* |x[n] - x[n-1]|^2 */
            double pr = seg[n + 1].re, pi = seg[n + 1].im;
            if (n == 0 && have_prev < 1) { pr = re; pi = im; }
            if (cd->hard_limit) {
                const double m = sqrt(pr * pr + pi * pi);
                if (m > 0.0) { pr /= m; pi /= m; } else { pr = pi = 0.0; }
            }
            const double dr = re - pr, di = im - pi;
            yr = dr * dr + di * di; yi = 0.0; break; }
        default: {                                   /* FDIFF: (f[n] - f[n-1])^2, f = arg(x[n] conj x[n-1]) */
            const int h = have_prev;                 /* x[n] = seg[n+2], x[n-1] = seg[n+1], x[n-2] = seg[n] */
            double f1 = 0.0, f0 = 0.0;
            if (n + h >= 1) f1 = arg_prod(&seg[n + 2], &seg[n + 1]);
            if (n + h >= 2) f0 = arg_prod(&seg[n + 1], &seg[n]);
            const double d = f1 - f0;
            yr = d * d; yi = 0.0; break; }
        }
        vre[n] = yr; vim[n] = yi;
    }
    double mr = 0.0, mi = 0.0;
    for (int n = 0; n < L; ++n) { mr += vre[n]; mi += vim[n]; }
    mr /= (double)L; mi /= (double)L;
    for (int n = 0; n < L; ++n) { vre[n] -= mr; vim[n] -= mi; }
}

static void cyc_finish_segment(atkdsp_cyclic *c) {
    for (int i = 0; i < c->ncand; ++i) {
        nonlin_segment(c->seg, c->L, c->have_prev, &c->cand[i], c->vre, c->vim);
        line_segment(&c->acc[i], c->vre, c->vim, c->w, c->L);
    }
    c->seg[0] = c->seg[c->L];                        /* the last two samples become the history */
    c->seg[1] = c->seg[c->L + 1];
    c->have_prev = c->have_prev + 2 > 2 ? 2 : c->have_prev + 2;
    c->pos = 0;
    c->segments++;
}

int atkdsp_cyclic_process(atkdsp_cyclic *c, const atkdsp_cf32 *x, size_t n) {
    if (!c || (!x && n)) return ATKDSP_E_ARG;
    size_t i = 0;
    while (i < n) {
        size_t take = (size_t)(c->L - c->pos);
        if (take > n - i) take = n - i;
        memcpy(c->seg + 2 + c->pos, x + i, take * sizeof *x);
        c->pos += (int)take;
        i += take;
        if (c->pos == c->L) cyc_finish_segment(c);
    }
    c->samples += n;
    return ATKDSP_OK;
}

int atkdsp_cyclic_read(const atkdsp_cyclic *c, atkdsp_cyc_result *out) {
    if (!c || !out) return ATKDSP_E_ARG;
    const double t_seg = (double)c->L / c->fs, w2 = c->wsum2 / (double)c->L;
    int nseg = 0;
    for (int i = 0; i < c->ncand; ++i) {
        line_stats s;
        line_read(&c->acc[i], w2, t_seg, &s);
        atkdsp_cyc_result *r = &out[i];
        memset(r, 0, sizeof *r);
        r->level = s.level; r->coherence = s.coherence; r->delta_hz = s.delta_hz;
        r->level_aligned = s.level_aligned; r->coherence_aligned = s.coherence_aligned;
        r->line_db = s.level > 1e-30 ? 10.0 * log10(s.level) : -300.0;
        r->level_thr = c->level_thr; r->coh_thr = c->coh_thr;
        r->nseg = s.nseg;
        if (s.nseg < c->K) r->flags |= ATKDSP_CYC_WARMUP;
        else if (s.level >= c->level_thr && s.coherence >= c->coh_thr) r->flags |= ATKDSP_CYC_LINE;
        nseg = s.nseg;
    }
    return nseg;
}

int atkdsp_cyclic_describe(const atkdsp_cyclic *c, char *buf, size_t cap) {
    if (!c || !buf || !cap) return ATKDSP_E_ARG;
    static const char *nl[6] = {"|x|^2", "x^2", "x^4", "x^8", "|x - x[-1]|^2", "(f - f[-1])^2"};
    int n = snprintf(buf, cap, "cyclic: fs %.6g Hz, %d candidate(s), segments of %d samples (%.4g s) x %d "
                     "(%.4g s), Pfa %.3g each -> level >= %.2f and coherence >= %.3f; %llu segments in:",
                     c->fs, c->ncand, c->L, (double)c->L / c->fs, c->K, (double)c->L * c->K / c->fs,
                     c->pfa_each, c->level_thr, c->coh_thr, (unsigned long long)c->segments);
    for (int i = 0; i < c->ncand && n >= 0 && (size_t)n < cap; ++i)
        n += snprintf(buf + n, cap - (size_t)n, "%s %.6g Bd %s%s", i ? "," : "", c->cand[i].rate_hz,
                      nl[c->cand[i].nonlin], c->cand[i].hard_limit && c->cand[i].nonlin ? " (limited)" : "");
    return ATKDSP_OK;
}

int atkdsp_cyclic_stats(const atkdsp_cyclic *c, atkdsp_cyclic_counters *st) {
    if (!c || !st) return ATKDSP_E_ARG;
    st->samples = c->samples; st->segments = c->segments; st->resets = c->resets;
    st->segment_fill = (uint64_t)c->pos;
    return ATKDSP_OK;
}

/* ---- 23b. atkdsp_cp_probe ------------------------------------------------------- */
typedef struct {
    atkdsp_cp_cand cand;
    double r_re, r_im, e;       /* since reset                                  */
    double npairs;
    line_acc acc;               /* the symbol-rate line of |z|                  */
} cp_slot;

struct atkdsp_cp_probe {
    double fs, pfa, pfa_each, wsum2, coh_thr, level_thr;
    int L, K, ncand, hist;      /* hist = the samples kept before the segment  */
    cp_slot *s;
    double *w;
    atkdsp_cf32 *buf;           /* hist + L samples                             */
    int pos;                    /* samples in the current segment               */
    uint64_t idx;               /* global index of the next sample              */
    double *zre, *zim, *vre, *vim;   /* scratch: hist + L lag products, L line input */
    uint64_t samples, segments, resets;
};

static int cp_cand_ok(const atkdsp_cp_cand *c) {
    if (c->lag < 1 || c->lag > CP_LAG_MAX) return 0;
    if (c->cp < 0 || c->cp > CP_CP_MAX) return 0;
    if (c->period < 0 || c->period > CP_LAG_MAX + CP_CP_MAX) return 0;
    return 1;
}

atkdsp_cp_probe *atkdsp_cp_probe_create(double fs, const atkdsp_cp_cand *cands, int ncand,
                                        int seg_len, int nseg, double pfa) {
    if (!(fs > 0.0) || !isfinite(fs) || !cands || ncand < 1 || ncand > CYC_MAX_CAND) return NULL;
    if (seg_len < CYC_SEG_MIN || seg_len > CYC_SEG_MAX || nseg < 2 || nseg > CYC_NSEG_MAX) return NULL;
    if (!(pfa > 0.0 && pfa < 0.5)) return NULL;
    int hist = 0, any_cp = 0;
    for (int i = 0; i < ncand; ++i) {
        if (!cp_cand_ok(&cands[i])) return NULL;
        const int need = cands[i].lag + cands[i].cp;
        if (need > hist) hist = need;
        if (cands[i].cp) {
            const int P = cands[i].period ? cands[i].period : cands[i].lag + cands[i].cp;
            if (P < 2) return NULL;
            any_cp = 1;
        }
    }
    double lt, ct;
    atkdsp_cyclic_thresholds(nseg, pfa / (double)ncand, &lt, &ct);
    if (ct > 1.0 && any_cp) return NULL;      /* the prefix stage's coherence test needs more segments */
    atkdsp_cp_probe *p = (atkdsp_cp_probe *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->fs = fs; p->pfa = pfa; p->pfa_each = pfa / (double)ncand; p->level_thr = lt; p->coh_thr = ct;
    p->L = seg_len; p->K = nseg; p->ncand = ncand; p->hist = hist;
    p->s = (cp_slot *)calloc((size_t)ncand, sizeof *p->s);
    p->w = hann_periodic(seg_len, &p->wsum2);
    p->buf = (atkdsp_cf32 *)calloc((size_t)hist + (size_t)seg_len, sizeof *p->buf);
    p->zre = (double *)malloc(((size_t)hist + (size_t)seg_len) * sizeof(double));
    p->zim = (double *)malloc(((size_t)hist + (size_t)seg_len) * sizeof(double));
    p->vre = (double *)malloc((size_t)seg_len * sizeof(double));
    p->vim = (double *)malloc((size_t)seg_len * sizeof(double));
    int ok = p->s && p->w && p->buf && p->zre && p->zim && p->vre && p->vim;
    for (int i = 0; i < ncand && ok; ++i) {
        p->s[i].cand = cands[i];
        const int P = cands[i].period ? cands[i].period : cands[i].lag + cands[i].cp;
        ok = line_init(&p->s[i].acc, 2.0 * ATK_PI / (double)P, seg_len, nseg);
    }
    if (!ok) { atkdsp_cp_probe_destroy(p); return NULL; }
    return p;
}

void atkdsp_cp_probe_destroy(atkdsp_cp_probe *p) {
    if (!p) return;
    if (p->s) for (int i = 0; i < p->ncand; ++i) line_free(&p->s[i].acc);
    free(p->s); free(p->w); free(p->buf); free(p->zre); free(p->zim); free(p->vre); free(p->vim);
    free(p);
}

void atkdsp_cp_probe_reset(atkdsp_cp_probe *p) {
    if (!p) return;
    for (int i = 0; i < p->ncand; ++i) {
        cp_slot *s = &p->s[i];
        s->r_re = s->r_im = s->e = s->npairs = 0.0;
        line_reset(&s->acc);
    }
    memset(p->buf, 0, ((size_t)p->hist + (size_t)p->L) * sizeof *p->buf);
    p->pos = 0; p->idx = 0;
    p->resets++;
}

/* Measure every candidate on the complete segment buf[hist .. hist + L). A
 * pair (m, m - lag) counts when m - lag >= 0 in the stream (idx is the
 * global index of the segment's first sample). */
static void cp_finish_segment(atkdsp_cp_probe *p) {
    const int L = p->L, H = p->hist;
    const atkdsp_cf32 *b = p->buf;
    for (int i = 0; i < p->ncand; ++i) {
        cp_slot *s = &p->s[i];
        const int N = s->cand.lag, W = s->cand.cp;
        /* lag products over [H - W, H + L): y[m] = x[m] conj x[m - N], zero
         * where the older sample is before the stream */
        const int m0 = H - W;
        for (int m = m0; m < H + L; ++m) {
            const long long g = (long long)p->idx + (m - H);          /* global index */
            if (m - N < 0 || g - N < 0) { p->zre[m - m0] = p->zim[m - m0] = 0.0; continue; }
            const double ar = b[m].re, ai = b[m].im, br = b[m - N].re, bi = b[m - N].im;
            p->zre[m - m0] = ar * br + ai * bi;
            p->zim[m - m0] = ai * br - ar * bi;
            if (m >= H) {
                s->r_re += p->zre[m - m0]; s->r_im += p->zim[m - m0];
                s->e += 0.5 * (ar * ar + ai * ai + br * br + bi * bi);
                s->npairs += 1.0;
            }
        }
        if (!W) continue;
        /* |z[m]|, z the moving sum over the last W products, m in the
         * segment: z[H + n] = sum of zre[n + 1 .. n + W] (zre[j] is the
         * product at m = H - W + j) */
        double zr = 0.0, zi = 0.0;
        for (int j = 1; j <= W; ++j) { zr += p->zre[j]; zi += p->zim[j]; }
        for (int n = 0; n < L; ++n) {
            /* z[m] for m = H + n: products (m - W + 1 .. m) = zre[n + 1 .. n + W] */
            if (n) {
                zr += p->zre[n + W] - p->zre[n];
                zi += p->zim[n + W] - p->zim[n];
            }
            p->vre[n] = sqrt(zr * zr + zi * zi);
            p->vim[n] = 0.0;
        }
        double mean = 0.0;
        for (int n = 0; n < L; ++n) mean += p->vre[n];
        mean /= (double)L;
        for (int n = 0; n < L; ++n) p->vre[n] -= mean;
        line_segment(&s->acc, p->vre, p->vim, p->w, L);
    }
    /* the last hist samples become the history */
    memmove(p->buf, p->buf + L, (size_t)H * sizeof *p->buf);
    p->idx += (uint64_t)L;
    p->pos = 0;
    p->segments++;
}

int atkdsp_cp_probe_process(atkdsp_cp_probe *p, const atkdsp_cf32 *x, size_t n) {
    if (!p || (!x && n)) return ATKDSP_E_ARG;
    size_t i = 0;
    while (i < n) {
        size_t take = (size_t)(p->L - p->pos);
        if (take > n - i) take = n - i;
        memcpy(p->buf + p->hist + p->pos, x + i, take * sizeof *x);
        p->pos += (int)take;
        i += take;
        if (p->pos == p->L) cp_finish_segment(p);
    }
    p->samples += n;
    return ATKDSP_OK;
}

int atkdsp_cp_probe_read(const atkdsp_cp_probe *p, atkdsp_cp_result *out) {
    if (!p || !out) return ATKDSP_E_ARG;
    const double t_seg = (double)p->L / p->fs, w2 = p->wsum2 / (double)p->L;
    int nseg = 0;
    for (int i = 0; i < p->ncand; ++i) {
        const cp_slot *s = &p->s[i];
        atkdsp_cp_result *r = &out[i];
        memset(r, 0, sizeof *r);
        r->npairs = s->npairs;
        r->score = s->e > 0.0 ? sqrt(s->r_re * s->r_re + s->r_im * s->r_im) / s->e : 0.0;
        r->score_thr = s->npairs > 0.0 ? sqrt(-log(p->pfa_each) / s->npairs) : 2.0;
        if (s->npairs > 0.0 && r->score >= r->score_thr) r->flags |= ATKDSP_CP_DETECT;
        if (s->cand.cp) {
            line_stats ls;
            line_read(&s->acc, w2, t_seg, &ls);
            r->line_level = ls.level; r->line_coherence = ls.coherence;
            r->line_delta_hz = ls.delta_hz; r->line_coh_thr = p->coh_thr;
            r->nseg = ls.nseg;
            if (ls.nseg < p->K) r->flags |= ATKDSP_CP_WARMUP;
            else if (ls.coherence >= p->coh_thr) r->flags |= ATKDSP_CP_LINE;
            nseg = ls.nseg;
        }
    }
    return nseg;
}

int atkdsp_cp_probe_describe(const atkdsp_cp_probe *p, char *buf, size_t cap) {
    if (!p || !buf || !cap) return ATKDSP_E_ARG;
    int n = snprintf(buf, cap, "cp_probe: fs %.6g Hz, %d candidate(s), segments of %d samples x %d, "
                     "Pfa %.3g each (coherence >= %.3f); %llu segments in:",
                     p->fs, p->ncand, p->L, p->K, p->pfa_each, p->coh_thr,
                     (unsigned long long)p->segments);
    for (int i = 0; i < p->ncand && n >= 0 && (size_t)n < cap; ++i) {
        const atkdsp_cp_cand *c = &p->s[i].cand;
        const int P = c->period ? c->period : c->lag + c->cp;
        if (c->cp) n += snprintf(buf + n, cap - (size_t)n, "%s lag %d, cp %d, period %d", i ? "," : "", c->lag, c->cp, P);
        else n += snprintf(buf + n, cap - (size_t)n, "%s lag %d", i ? "," : "", c->lag);
    }
    return ATKDSP_OK;
}

int atkdsp_cp_probe_stats(const atkdsp_cp_probe *p, atkdsp_cyclic_counters *st) {
    if (!p || !st) return ATKDSP_E_ARG;
    st->samples = p->samples; st->segments = p->segments; st->resets = p->resets;
    st->segment_fill = (uint64_t)p->pos;
    return ATKDSP_OK;
}

/* ---- 25b. atkdsp_cyclic_scan: every line in a band (§B6 step 2) ---------------
 * The same segmentation, nonlinearity, mean removal and periodic Hann as
 * atkdsp_cyclic, but one FFT per segment instead of one Goertzel per
 * candidate, and the last nseg phasors kept for every bin of [f_lo, f_hi).
 * Per bin the §23 statistics hold exactly (the FFT bin IS the Goertzel at
 * that frequency): level is Exp(1) for white input, coherence the line-
 * versus-hump test, delta the bin-to-line offset from the segment-to-segment
 * phase. The Pfa is the scan's: each bin at pfa / nbins (the max statistic
 * over independent bins — Lunden et al.'s form; the bins of one FFT of a
 * mean-removed, Hann-windowed segment are close to independent beyond the
 * window's main lobe, and the measured rate in tests/test_phase6.py is what
 * the threshold is judged by). A line is a bin passing both thresholds that
 * is a local maximum of level; lines come out strongest first; HARMONIC when
 * the bin at twice the line's frequency passes the level threshold too. */
struct atkdsp_cyclic_scan {
    double fs, pfa, pfa_each, level_thr, coh_thr, wsum2, f_lo, f_hi;
    int L, K, nonlin, hard_limit, k_lo, k_hi, nbins;
    double *w;
    atkdsp_cf32 *seg, *X, *Y;
    int pos, have_prev;
    double *vre, *vim;
    double *p_re, *p_im;       /* nbins x K ring                                 */
    double *e;                 /* K ring of segment energies                     */
    int head, count;
    atkdsp_fft *fft;
    uint64_t samples, segments, resets;
};

atkdsp_cyclic_scan *atkdsp_cyclic_scan_create(double fs, int nonlin, int hard_limit, int seg_len,
                                              int nseg, double f_lo, double f_hi, double pfa) {
    if (!(fs > 0.0) || !isfinite(fs) || nonlin < 0 || nonlin > 5) return NULL;
    if (seg_len < 64 || seg_len > CYC_SEG_MAX || nseg < 16 || nseg > CYC_NSEG_MAX) return NULL;
    if (!(f_lo >= 0.0) || !(f_hi > f_lo) || !(f_hi <= 0.5 * fs)) return NULL;
    if (!(pfa > 0.0 && pfa < 0.5)) return NULL;
    const int k_lo = (int)ceil(f_lo * seg_len / fs), k_hi = (int)floor(f_hi * seg_len / fs);
    if (k_hi <= k_lo) return NULL;
    const int nbins = k_hi - k_lo;
    double lt, ct;
    atkdsp_cyclic_thresholds(nseg, pfa / (double)nbins, &lt, &ct);
    if (ct > 1.0) return NULL;
    atkdsp_cyclic_scan *s = (atkdsp_cyclic_scan *)calloc(1, sizeof *s);
    if (!s) return NULL;
    s->fs = fs; s->pfa = pfa; s->pfa_each = pfa / (double)nbins; s->level_thr = lt; s->coh_thr = ct;
    s->f_lo = f_lo; s->f_hi = f_hi; s->L = seg_len; s->K = nseg; s->nonlin = nonlin; s->hard_limit = hard_limit;
    s->k_lo = k_lo; s->k_hi = k_hi; s->nbins = nbins;
    s->w = hann_periodic(seg_len, &s->wsum2);
    s->seg = (atkdsp_cf32 *)malloc(((size_t)seg_len + 2) * sizeof *s->seg);
    s->X = (atkdsp_cf32 *)malloc((size_t)seg_len * sizeof *s->X);
    s->Y = (atkdsp_cf32 *)malloc((size_t)seg_len * sizeof *s->Y);
    s->vre = (double *)malloc((size_t)seg_len * sizeof(double));
    s->vim = (double *)malloc((size_t)seg_len * sizeof(double));
    s->p_re = (double *)calloc((size_t)nbins * (size_t)nseg, sizeof(double));
    s->p_im = (double *)calloc((size_t)nbins * (size_t)nseg, sizeof(double));
    s->e = (double *)calloc((size_t)nseg, sizeof(double));
    s->fft = atkdsp_fft_create((size_t)seg_len);
    if (!s->w || !s->seg || !s->X || !s->Y || !s->vre || !s->vim || !s->p_re || !s->p_im || !s->e || !s->fft) {
        atkdsp_cyclic_scan_destroy(s); return NULL;
    }
    return s;
}

void atkdsp_cyclic_scan_destroy(atkdsp_cyclic_scan *s) {
    if (!s) return;
    atkdsp_fft_destroy(s->fft);
    free(s->w); free(s->seg); free(s->X); free(s->Y); free(s->vre); free(s->vim);
    free(s->p_re); free(s->p_im); free(s->e);
    free(s);
}

void atkdsp_cyclic_scan_reset(atkdsp_cyclic_scan *s) {
    if (!s) return;
    s->pos = 0; s->have_prev = 0; s->head = 0; s->count = 0; s->resets++;
}

static void scan_finish_segment(atkdsp_cyclic_scan *s) {
    atkdsp_cyc_cand cd; cd.rate_hz = 1.0; cd.nonlin = s->nonlin; cd.hard_limit = s->hard_limit;
    nonlin_segment(s->seg, s->L, s->have_prev, &cd, s->vre, s->vim);
    double e = 0.0;
    for (int n = 0; n < s->L; ++n) {
        e += s->vre[n] * s->vre[n] + s->vim[n] * s->vim[n];
        s->X[n].re = (float)(s->w[n] * s->vre[n]); s->X[n].im = (float)(s->w[n] * s->vim[n]);
    }
    atkdsp_fft_exec(s->fft, s->X, s->Y, 0);
    double *pr = s->p_re + (size_t)s->head * (size_t)s->nbins;
    double *pi = s->p_im + (size_t)s->head * (size_t)s->nbins;
    for (int b = 0; b < s->nbins; ++b) { pr[b] = s->Y[s->k_lo + b].re; pi[b] = s->Y[s->k_lo + b].im; }
    s->e[s->head] = e;
    s->head = (s->head + 1) % s->K;
    if (s->count < s->K) s->count++;
    s->seg[0] = s->seg[s->L]; s->seg[1] = s->seg[s->L + 1];
    s->have_prev = s->have_prev + 2 > 2 ? 2 : s->have_prev + 2;
    s->pos = 0;
    s->segments++;
}

int atkdsp_cyclic_scan_process(atkdsp_cyclic_scan *s, const atkdsp_cf32 *x, size_t n) {
    if (!s || (!x && n)) return ATKDSP_E_ARG;
    size_t i = 0;
    while (i < n) {
        size_t take = (size_t)(s->L - s->pos);
        if (take > n - i) take = n - i;
        memcpy(s->seg + 2 + s->pos, x + i, take * sizeof *x);
        s->pos += (int)take; i += take;
        if (s->pos == s->L) scan_finish_segment(s);
    }
    s->samples += n;
    return ATKDSP_OK;
}

/* per-bin statistics over the segments held (the §23 reading, FFT bins) */
static void scan_bin(const atkdsp_cyclic_scan *s, int b, double *level, double *coh, double *delta_hz) {
    const int K = s->count;
    const int first = (s->head - K + s->K) % s->K;
    double sr = 0.0, si = 0.0, sa = 0.0, se = 0.0, dr = 0.0, di = 0.0, lpr = 0.0, lpi = 0.0;
    for (int k = 0; k < K; ++k) {
        const int j = (first + k) % s->K;
        const double pr = s->p_re[(size_t)j * s->nbins + b], pi = s->p_im[(size_t)j * s->nbins + b];
        sr += pr; si += pi; sa += sqrt(pr * pr + pi * pi); se += s->e[j];
        if (k) { dr += pr * lpr + pi * lpi; di += pi * lpr - pr * lpi; }
        lpr = pr; lpi = pi;
    }
    const double den = se * (s->wsum2 / (double)s->L);
    *level = den > 0.0 ? (sr * sr + si * si) / den : 0.0;
    *coh = sa > 0.0 ? sqrt(sr * sr + si * si) / sa : 0.0;
    *delta_hz = (K >= 2 && (dr != 0.0 || di != 0.0)) ? atan2(di, dr) / (2.0 * ATK_PI * (double)s->L / s->fs) : 0.0;
}

int atkdsp_cyclic_scan_nbins(const atkdsp_cyclic_scan *s) { return s ? s->nbins : ATKDSP_E_ARG; }

ptrdiff_t atkdsp_cyclic_scan_level(const atkdsp_cyclic_scan *s, float *level, size_t cap) {
    if (!s || !level) return ATKDSP_E_ARG;
    if (cap < (size_t)s->nbins) return ATKDSP_E_CAP;
    for (int b = 0; b < s->nbins; ++b) {
        double lv = 0.0, co, dh;
        if (s->count) scan_bin(s, b, &lv, &co, &dh);
        level[b] = (float)lv;
    }
    return s->nbins;
}

static int cmp_line_desc(const void *a, const void *b) {
    const double x = ((const atkdsp_scan_line *)a)->level, y = ((const atkdsp_scan_line *)b)->level;
    return (x < y) - (x > y);
}

ptrdiff_t atkdsp_cyclic_scan_read(const atkdsp_cyclic_scan *s, atkdsp_scan_line *out, size_t cap) {
    if (!s || (!out && cap)) return ATKDSP_E_ARG;
    if (s->count < s->K) return 0;
    const int nb = s->nbins;
    double *lv = (double *)malloc((size_t)nb * 3 * sizeof(double));
    if (!lv) return ATKDSP_E_NOMEM;
    double *co = lv + nb, *dh = lv + 2 * nb;
    for (int b = 0; b < nb; ++b) scan_bin(s, b, &lv[b], &co[b], &dh[b]);
    size_t n = 0;
    for (int b = 0; b < nb; ++b) {
        if (lv[b] < s->level_thr || co[b] < s->coh_thr) continue;
        if (b > 0 && lv[b - 1] > lv[b]) continue;
        if (b + 1 < nb && lv[b + 1] > lv[b]) continue;
        if (n < cap) {
            atkdsp_scan_line *l = &out[n];
            memset(l, 0, sizeof *l);
            l->bin = s->k_lo + b;
            l->freq_hz = (double)(s->k_lo + b) * s->fs / (double)s->L + dh[b];
            l->level = lv[b]; l->coherence = co[b]; l->level_thr = s->level_thr; l->coh_thr = s->coh_thr;
            {
                double l10 = -lv[b] / log(10.0) + log10((double)nb);
                if (l10 > 0.0) l10 = 0.0;
                l->log10_pfa = l10;
            }
            l->flags = ATKDSP_SCAN_LINE;
            const int b2 = (int)floor(2.0 * l->freq_hz * s->L / s->fs + 0.5) - s->k_lo;
            if (b2 >= 0 && b2 < nb && lv[b2] >= s->level_thr) l->flags |= ATKDSP_SCAN_HARMONIC;
        }
        n++;
    }
    free(lv);
    const size_t m = n < cap ? n : cap;
    if (m > 1) qsort(out, m, sizeof *out, cmp_line_desc);
    return (ptrdiff_t)m;
}

int atkdsp_cyclic_scan_describe(const atkdsp_cyclic_scan *s, char *buf, size_t cap) {
    if (!s || !buf || !cap) return ATKDSP_E_ARG;
    static const char *nl[6] = {"|x|^2", "x^2", "x^4", "x^8", "|x - x[-1]|^2", "(f - f[-1])^2"};
    int n = snprintf(buf, cap, "cyclic_scan: fs %.6g Hz, %s%s over %.6g..%.6g Hz (%d bins of %.4g Hz), "
                     "segments of %d samples (%.4g s) x %d, Pfa %.3g over the scan (%.3g per bin) -> "
                     "level >= %.2f and coherence >= %.3f; %llu segments in",
                     s->fs, nl[s->nonlin], s->hard_limit && s->nonlin ? " (limited)" : "", s->f_lo, s->f_hi,
                     s->nbins, s->fs / (double)s->L, s->L, (double)s->L / s->fs, s->K, s->pfa, s->pfa_each,
                     s->level_thr, s->coh_thr, (unsigned long long)s->segments);
    return n < 0 ? ATKDSP_E_ARG : (n >= (int)cap ? ATKDSP_E_CAP : n);
}

int atkdsp_cyclic_scan_stats(const atkdsp_cyclic_scan *s, atkdsp_cyclic_counters *st) {
    if (!s || !st) return ATKDSP_E_ARG;
    st->samples = s->samples; st->segments = s->segments; st->resets = s->resets;
    st->segment_fill = (uint64_t)s->pos;
    return ATKDSP_OK;
}
