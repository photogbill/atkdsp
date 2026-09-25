/* 4. rational resampler — polyphase L/M with carried state.
 *
 * Reference semantics (the numpy twin): scipy.signal.upfirdn(h * up, x, up,
 * down) truncated to the outputs whose time lies inside the input so far,
 * i.e. output m exists once input index floor(m*down/up) has arrived. No
 * centring, no group-delay compensation: the filter's own delay is the
 * stream's delay, which is what a real-time path wants. */
#include "internal.h"

struct atkdsp_resampler {
    float       *taps;     /* ntaps, scaled by `up` */
    atkdsp_cf32 *hist;     /* last Q inputs, oldest first */
    size_t       ntaps;
    size_t       Q;        /* taps per phase = ceil(ntaps / up) = history depth */
    unsigned     up, down;
    uint64_t     t_rel;    /* next output time (in up-rate ticks) relative to the
                              first sample of the NEXT block; always < down */
};

atkdsp_resampler *atkdsp_resampler_create(unsigned up, unsigned down,
                                          const float *taps, size_t ntaps) {
    if (!taps || ntaps == 0 || up == 0 || down == 0) return NULL;
    atkdsp_resampler *r = (atkdsp_resampler *)calloc(1, sizeof *r);
    if (!r) return NULL;
    r->ntaps = ntaps; r->up = up; r->down = down;
    r->Q = (ntaps + up - 1) / up;
    r->taps = (float *)atk_aligned_malloc(ntaps * sizeof(float));
    r->hist = (atkdsp_cf32 *)atk_aligned_malloc(r->Q * sizeof(atkdsp_cf32));
    if (!r->taps || !r->hist) { atkdsp_resampler_destroy(r); return NULL; }
    for (size_t j = 0; j < ntaps; ++j) r->taps[j] = taps[j] * (float)up;
    atkdsp_resampler_reset(r);
    return r;
}

void atkdsp_resampler_destroy(atkdsp_resampler *r) {
    if (!r) return;
    atk_aligned_free(r->taps);
    atk_aligned_free(r->hist);
    free(r);
}

void atkdsp_resampler_reset(atkdsp_resampler *r) {
    if (!r) return;
    memset(r->hist, 0, r->Q * sizeof(atkdsp_cf32));
    r->t_rel = 0;
}

size_t atkdsp_resampler_out_max(const atkdsp_resampler *r, size_t n_in) {
    if (!r) return 0;
    uint64_t span = (uint64_t)n_in * r->up;
    if (span <= r->t_rel) return 0;
    return (size_t)((span - r->t_rel + r->down - 1) / r->down);
}

ptrdiff_t atkdsp_resampler_process(atkdsp_resampler *r, const atkdsp_cf32 *in, size_t n,
                                   atkdsp_cf32 *out, size_t out_cap) {
    if (!r || (!in && n) || !out) return ATKDSP_E_ARG;
    const size_t want = atkdsp_resampler_out_max(r, n);
    if (want > out_cap) return ATKDSP_E_CAP;
    const unsigned L = r->up, M = r->down;
    const size_t Q = r->Q, T = r->ntaps;
    const float *h = r->taps;
    const uint64_t span = (uint64_t)n * L;

    size_t k = 0;
    uint64_t t = r->t_rel;
    for (; t < span; t += M) {
        const size_t n0 = (size_t)(t / L);   /* local index of the newest input used */
        const unsigned p = (unsigned)(t % L);
        float ar = 0.0f, ai = 0.0f;
        /* q-th tap of this phase is h[p + q*L], applied to x[n0 - q] */
        for (size_t q = 0, j = p; j < T; ++q, j += L) {
            atkdsp_cf32 x;
            if (q <= n0) x = in[n0 - q];
            else {
                size_t back = q - n0;             /* 1 = last of hist */
                if (back > Q) break;              /* older than we keep -> zero */
                x = r->hist[Q - back];
            }
            ar += h[j] * x.re;
            ai += h[j] * x.im;
        }
        out[k].re = ar; out[k].im = ai; ++k;
    }
    r->t_rel = t - span;

    if (n >= Q) {
        memcpy(r->hist, in + n - Q, Q * sizeof(atkdsp_cf32));
    } else if (n) {
        memmove(r->hist, r->hist + n, (Q - n) * sizeof(atkdsp_cf32));
        memcpy(r->hist + Q - n, in, n * sizeof(atkdsp_cf32));
    }
    return (ptrdiff_t)k;
}

/* 4b. arbitrary (fractional) resampler — polyphase + linear interpolation.
 *
 * The bank has P branches, so it interpolates the input by P; branch b is the
 * sub-phase b/P and its taps are the prototype's every-P-th sample, p[b+q*P],
 * applied to x[i-q] (the same newest-first convention the rational one uses).
 * An output wanted at continuous input position `pos` takes i=floor(pos) as the
 * newest input and frac=pos-i as the sub-sample delay; the branch is b=floor
 * (frac*P) and the remainder mu=frac*P-b is filled by linear interpolation
 * toward the next branch. That interpolation is done with a forward-difference
 * bank d[k]=p[k+1]-p[k] (p[PQ]=0): d[b+q*P] is exactly branch(b+1)-branch(b),
 * and at the seam b=P-1 the flat index b+1 lands on p[(q+1)*P] = branch 0 of
 * the NEXT input, so the wrap is correct, not approximated. Each output then
 * costs one FIR over 2*Q taps. `pos` advances by in/out per output and is
 * carried across blocks, so a block boundary is invisible. */
struct atkdsp_arb_resampler {
    float       *p;        /* flat prototype,       length PQ, scaled by P */
    float       *d;        /* flat forward-diff,    length PQ                */
    atkdsp_cf32 *hist;     /* last Q inputs, oldest first (hist[Q-1] = x[-1]) */
    size_t       P, Q;     /* phases, taps-per-phase */
    double       in_rate, out_rate;
    double       step;     /* input samples per output = in_rate/out_rate */
    /* Position bookkeeping is BLOCKING-INVARIANT: output K sits at input
     * position base + K*step (since the last anchor), and the block that holds
     * it starts at the exact integer input offset in_off. Its position within
     * the block is base + K*step - in_off. Because in_off is an integer and
     * K*step is a pure function of K, the same output lands identically however
     * the stream is cut into blocks — the old `pos += step` per output drifted
     * by ~1e-4 across different chunkings. `base` is 0 until set_ratio, which
     * re-anchors to preserve the next output's position across a ratio change. */
    double       base;
    uint64_t     K;        /* outputs emitted since the anchor */
    uint64_t     in_off;   /* inputs consumed since the anchor */
};

atkdsp_arb_resampler *atkdsp_arb_resampler_create(double in_rate, double out_rate,
                                                  double atten_db, unsigned nphase) {
    if (in_rate <= 0.0 || out_rate <= 0.0) return NULL;
    if (atten_db <= 0.0) atten_db = 60.0;
    const size_t P = nphase ? nphase : 64u;

    /* prototype at the interpolated rate; cut at the lower of the two Nyquists
     * so it anti-aliases going down and does not widen the band going up. */
    const double fs_proto = in_rate * (double)P;
    const double nyq = 0.5 * (in_rate < out_rate ? in_rate : out_rate);
    const double fp = 0.8 * nyq, fstop = nyq;
    ptrdiff_t nt = atkdsp_design_lowpass(fp, fstop, atten_db, fs_proto, NULL, 0);
    if (nt <= 0) return NULL;

    atkdsp_arb_resampler *r = (atkdsp_arb_resampler *)calloc(1, sizeof *r);
    if (!r) return NULL;
    r->P = P;
    r->Q = ((size_t)nt + P - 1) / P;                 /* taps per phase */
    const size_t PQ = r->P * r->Q;
    r->p = (float *)atk_aligned_malloc(PQ * sizeof(float));
    r->d = (float *)atk_aligned_malloc(PQ * sizeof(float));
    r->hist = (atkdsp_cf32 *)atk_aligned_malloc(r->Q * sizeof(atkdsp_cf32));
    if (!r->p || !r->d || !r->hist) { atkdsp_arb_resampler_destroy(r); return NULL; }

    /* design into the front of p, zero-pad the tail to PQ, fold in the gain P */
    memset(r->p, 0, PQ * sizeof(float));
    if (atkdsp_design_lowpass(fp, fstop, atten_db, fs_proto, r->p, (size_t)nt) != nt) {
        atkdsp_arb_resampler_destroy(r); return NULL;
    }
    for (size_t j = 0; j < PQ; ++j) r->p[j] *= (float)P;
    for (size_t j = 0; j + 1 < PQ; ++j) r->d[j] = r->p[j + 1] - r->p[j];
    r->d[PQ - 1] = -r->p[PQ - 1];                    /* p[PQ] == 0 */

    r->in_rate = in_rate; r->out_rate = out_rate;
    r->step = in_rate / out_rate;
    atkdsp_arb_resampler_reset(r);
    return r;
}

void atkdsp_arb_resampler_destroy(atkdsp_arb_resampler *r) {
    if (!r) return;
    atk_aligned_free(r->p);
    atk_aligned_free(r->d);
    atk_aligned_free(r->hist);
    free(r);
}

void atkdsp_arb_resampler_reset(atkdsp_arb_resampler *r) {
    if (!r) return;
    memset(r->hist, 0, r->Q * sizeof(atkdsp_cf32));
    r->base = 0.0;
    r->K = 0;
    r->in_off = 0;
}

void atkdsp_arb_resampler_set_ratio(atkdsp_arb_resampler *r, double ratio) {
    if (!r || ratio <= 0.0) return;
    /* re-anchor so the NEXT output stays where it is, then change the step */
    r->base = r->base + (double)r->K * r->step;
    r->K = 0;
    r->out_rate = r->in_rate * ratio;
    r->step = 1.0 / ratio;                           /* in/out */
}

double atkdsp_arb_resampler_ratio(const atkdsp_arb_resampler *r) {
    return (r && r->step > 0.0) ? 1.0 / r->step : 0.0;       /* out/in */
}

size_t atkdsp_arb_resampler_out_max(const atkdsp_arb_resampler *r, size_t n_in) {
    if (!r || r->step <= 0.0) return 0;
    const double local = r->base + (double)r->K * r->step - (double)r->in_off;
    if (local >= (double)n_in) return 0;
    /* outputs at local, local+step, ... < n_in; a safe upper bound */
    return (size_t)(((double)n_in - local) / r->step) + 2u;
}

ptrdiff_t atkdsp_arb_resampler_process(atkdsp_arb_resampler *r, const atkdsp_cf32 *in,
                                       size_t n, atkdsp_cf32 *out, size_t out_cap) {
    if (!r || (!in && n) || !out) return ATKDSP_E_ARG;
    const size_t P = r->P, Q = r->Q;
    const float *pp = r->p, *dd = r->d;
    const atkdsp_cf32 *hist = r->hist;
    const double step = r->step, base = r->base, inoff = (double)r->in_off;

    size_t k = 0;
    uint64_t K = r->K;
    double local = base + (double)K * step - inoff;   /* pos of output K in THIS block */
    while (local < (double)n) {
        const ptrdiff_t i = (ptrdiff_t)floor(local);
        const double frac = local - (double)i;       /* [0,1) */
        double ph = frac * (double)P;
        size_t b = (size_t)ph;                       /* [0,P) */
        if (b >= P) b = P - 1;                        /* guard fp rounding */
        const float mu = (float)(ph - (double)b);
        float ar = 0.0f, ai = 0.0f;
        for (size_t q = 0; q < Q; ++q) {
            const ptrdiff_t m = i - (ptrdiff_t)q;    /* input index */
            atkdsp_cf32 x;
            if (m >= 0) x = in[m];
            else {
                const ptrdiff_t hb = (ptrdiff_t)Q + m;   /* -1 -> Q-1 (x[-1]) */
                if (hb < 0) continue;                    /* older than kept -> 0 */
                x = hist[hb];
            }
            const float c = pp[b + q * P] + mu * dd[b + q * P];
            ar += c * x.re; ai += c * x.im;
        }
        if (k >= out_cap) return ATKDSP_E_CAP;
        out[k].re = ar; out[k].im = ai; ++k;
        ++K;
        local = base + (double)K * step - inoff;
    }
    r->K = K;
    r->in_off += n;                                  /* exact: keeps the position invariant */

    if (n >= Q) {
        memcpy(r->hist, in + n - Q, Q * sizeof(atkdsp_cf32));
    } else if (n) {
        memmove(r->hist, r->hist + n, (Q - n) * sizeof(atkdsp_cf32));
        memcpy(r->hist + Q - n, in, n * sizeof(atkdsp_cf32));
    }
    return (ptrdiff_t)k;
}
