/* 3. decimating FIR with carried history.
 *
 * Reference semantics (the numpy twin):  y = convolve(x, h)[:len(x)][::decim]
 * with x the whole stream since reset — i.e. output k is the filter evaluated
 * at input index k*decim, and the first ntaps-1 inputs see zeros before them.
 * Only surviving outputs are computed, so cost is ntaps/decim per sample. */
#include "internal.h"

struct atkdsp_fir {
    float       *taps;    /* ntaps, copied */
    atkdsp_cf32 *hist;    /* last ntaps-1 inputs seen, oldest first */
    atkdsp_cf32 *tmp;     /* ntaps scratch for windows straddling hist/in */
    size_t       ntaps;
    unsigned     decim;
    unsigned     pos;     /* (number of inputs consumed) mod decim */
};

atkdsp_fir *atkdsp_fir_create(const float *taps, size_t ntaps, unsigned decim) {
    if (!taps || ntaps == 0 || decim == 0) return NULL;
    atkdsp_fir *f = (atkdsp_fir *)calloc(1, sizeof *f);
    if (!f) return NULL;
    f->taps = (float *)atk_aligned_malloc(ntaps * sizeof(float));
    f->hist = (atkdsp_cf32 *)atk_aligned_malloc((ntaps > 1 ? ntaps - 1 : 1) * sizeof(atkdsp_cf32));
    f->tmp  = (atkdsp_cf32 *)atk_aligned_malloc(ntaps * sizeof(atkdsp_cf32));
    if (!f->taps || !f->hist || !f->tmp) { atkdsp_fir_destroy(f); return NULL; }
    memcpy(f->taps, taps, ntaps * sizeof(float));
    f->ntaps = ntaps;
    f->decim = decim;
    atkdsp_fir_reset(f);
    return f;
}

void atkdsp_fir_destroy(atkdsp_fir *f) {
    if (!f) return;
    atk_aligned_free(f->taps);
    atk_aligned_free(f->hist);
    atk_aligned_free(f->tmp);
    free(f);
}

void atkdsp_fir_reset(atkdsp_fir *f) {
    if (!f) return;
    if (f->ntaps > 1) memset(f->hist, 0, (f->ntaps - 1) * sizeof(atkdsp_cf32));
    f->pos = 0;
}

static size_t out_count(unsigned pos, unsigned decim, size_t n) {
    size_t first = (decim - pos) % decim;          /* first local index that is a multiple */
    if (first >= n) return 0;
    return 1 + (n - 1 - first) / decim;
}

size_t atkdsp_fir_out_max(const atkdsp_fir *f, size_t n_in) {
    if (!f) return 0;
    return out_count(f->pos, f->decim, n_in);
}

/* dot product of taps against a window ending at w[T-1]: sum_j h[j]*w[T-1-j] */
ATK_INLINE atkdsp_cf32 dot(const float *ATK_RESTRICT h, const atkdsp_cf32 *ATK_RESTRICT w,
                           size_t T) {
    float ar = 0.0f, ai = 0.0f;
    for (size_t j = 0; j < T; ++j) {
        const float hj = h[j];
        ar += hj * w[T - 1 - j].re;
        ai += hj * w[T - 1 - j].im;
    }
    atkdsp_cf32 y; y.re = ar; y.im = ai;
    return y;
}

ptrdiff_t atkdsp_fir_process(atkdsp_fir *f, const atkdsp_cf32 *in, size_t n,
                             atkdsp_cf32 *out, size_t out_cap) {
    if (!f || (!in && n) || !out) return ATKDSP_E_ARG;
    const size_t T = f->ntaps, H = T - 1;
    const unsigned D = f->decim;
    const size_t want = out_count(f->pos, D, n);
    if (want > out_cap) return ATKDSP_E_CAP;

    size_t k = 0;
    size_t i = (D - f->pos) % D;
    for (; i < n; i += D) {
        if (i >= H) {
            out[k++] = dot(f->taps, in + i - H, T);
        } else {
            /* window = hist[H-(H-i) .. H) ++ in[0..i] : H-i old samples then i+1 new */
            const size_t old = H - i;
            memcpy(f->tmp, f->hist + (H - old), old * sizeof(atkdsp_cf32));
            memcpy(f->tmp + old, in, (i + 1) * sizeof(atkdsp_cf32));
            out[k++] = dot(f->taps, f->tmp, T);
        }
    }

    /* carry the last H inputs */
    if (H) {
        if (n >= H) {
            memcpy(f->hist, in + n - H, H * sizeof(atkdsp_cf32));
        } else {
            memmove(f->hist, f->hist + n, (H - n) * sizeof(atkdsp_cf32));
            memcpy(f->hist + H - n, in, n * sizeof(atkdsp_cf32));
        }
    }
    f->pos = (unsigned)((f->pos + n) % D);
    return (ptrdiff_t)k;
}
