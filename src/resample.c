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
