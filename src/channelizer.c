/* THE CHANNELIZER (§28; docs/ATKDSP_PLAN.md §A4, docs/CHANNELIZER_DF_REQUIREMENTS.md).
 *
 * 28b. the polyphase filterbank (PFB) — a uniform 2x-oversampled grid, every
 *      channel at once, N element streams on one block clock (the DF path).
 * 28a. the overlap-save extractor (OSX) — arbitrary channels at arbitrary
 *      centres. (Added below the PFB.)
 * W1/W2 — the per-channel covariance and power DF accumulates on the output.
 *
 * The numpy twin (the spec) is python/atkdsp/reference.py §28; every result
 * here matches it to float precision. Deterministic double arithmetic for the
 * prototype so the taps reproduce exactly.
 */
#include "internal.h"
#include <stdio.h>

double atk_kaiser_beta(double atten_db);                 /* src/design.c */

/* modified Bessel I0 (power series); matches src/design.c and numpy.kaiser. */
static double chan_i0(double x) {
    double sum = 1.0, term = 1.0;
    const double y = x * 0.5;
    for (int k = 1; k < 500; ++k) {
        term *= (y / (double)k);
        const double t2 = term * term;
        sum += t2;
        if (t2 < sum * 1e-17) break;
    }
    return sum;
}

/* ===================================================================== 28b.
 * Polyphase filterbank.
 */
struct atkdsp_pfb {
    double       fs, atten;
    unsigned     nchan, os, K, max_nel;
    unsigned     hop, ov;              /* hop = nchan/os; ov = L - hop        */
    size_t       L;                   /* prototype length = nchan*K          */
    double      *h;                   /* prototype taps, L (double)          */
    atkdsp_fft  *fft;                 /* length-nchan forward plan           */
    atkdsp_cf32 *twiddle;             /* nchan: W_n^j = exp(2pi i j / nchan)  */
    /* per-element streaming state */
    atkdsp_cf32 *hist;                /* max_nel * ov, the overlap carried    */
    atkdsp_cf32 *leftover;            /* max_nel * hop, up to hop-1 used      */
    unsigned     lc;                  /* leftover count (same for all el)     */
    int64_t      b;                   /* global hop index (drives rotation)   */
    /* shared per-hop scratch (one element at a time) */
    atkdsp_cf32 *win;                 /* L                                    */
    atkdsp_cf32 *folded;              /* nchan                                */
    atkdsp_cf32 *X;                   /* nchan                                */
    /* cal */
    atkdsp_cf32 *cal;                 /* max_nel * nchan, or NULL             */
    unsigned     cal_nel;
};

static void pfb_build_prototype(atkdsp_pfb *p) {
    const size_t L = p->L;
    const double fc = 1.0 / (2.0 * (double)p->nchan);
    const double M = (double)(L - 1) * 0.5;
    const double beta = atk_kaiser_beta(p->atten);
    const double i0b = chan_i0(beta);
    double sum = 0.0;
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t n = 0; n < L; ++n) {
            const double t = (double)n - M;
            const double x = 2.0 * fc * t;
            const double sinc = (t == 0.0) ? 1.0 : sin(ATK_PI * x) / (ATK_PI * x);
            const double r = (M > 0.0) ? t / M : 0.0;
            const double w = chan_i0(beta * sqrt(r * r < 1.0 ? 1.0 - r * r : 0.0)) / i0b;
            const double h = 2.0 * fc * sinc * w;
            if (pass == 0) sum += h;
            else p->h[n] = (h / sum) * (double)p->nchan;   /* unit DC gain, x nchan */
        }
        if (sum == 0.0) sum = 1.0;
    }
}

atkdsp_pfb *atkdsp_pfb_create(double fs, unsigned nchan, unsigned oversample,
                              unsigned taps_per_poly, double atten_db, unsigned max_nel) {
    if (!(fs > 0.0) || nchan < 2 || oversample < 1 || nchan % oversample != 0) return NULL;
    if (taps_per_poly < 2 || max_nel < 1) return NULL;
    if (atten_db < 20.0) atten_db = 20.0;
    atkdsp_pfb *p = (atkdsp_pfb *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->fs = fs; p->atten = atten_db;
    p->nchan = nchan; p->os = oversample; p->K = taps_per_poly; p->max_nel = max_nel;
    p->hop = nchan / oversample;
    p->L = (size_t)nchan * taps_per_poly;
    p->ov = (unsigned)(p->L - p->hop);
    p->h = (double *)malloc(p->L * sizeof(double));
    p->fft = atkdsp_fft_create(nchan);
    p->twiddle = (atkdsp_cf32 *)malloc(nchan * sizeof(atkdsp_cf32));
    p->hist = (atkdsp_cf32 *)calloc((size_t)max_nel * p->ov, sizeof(atkdsp_cf32));
    p->leftover = (atkdsp_cf32 *)calloc((size_t)max_nel * p->hop, sizeof(atkdsp_cf32));
    p->win = (atkdsp_cf32 *)atk_aligned_malloc(p->L * sizeof(atkdsp_cf32));
    p->folded = (atkdsp_cf32 *)atk_aligned_malloc(nchan * sizeof(atkdsp_cf32));
    p->X = (atkdsp_cf32 *)atk_aligned_malloc(nchan * sizeof(atkdsp_cf32));
    if (!p->h || !p->fft || !p->twiddle || !p->hist || !p->leftover || !p->win ||
        !p->folded || !p->X) { atkdsp_pfb_destroy(p); return NULL; }
    pfb_build_prototype(p);
    for (unsigned j = 0; j < nchan; ++j) {
        const double a = 2.0 * ATK_PI * (double)j / (double)nchan;
        p->twiddle[j].re = (float)cos(a);
        p->twiddle[j].im = (float)sin(a);
    }
    p->lc = 0; p->b = 0; p->cal = NULL; p->cal_nel = 0;
    return p;
}

void atkdsp_pfb_destroy(atkdsp_pfb *p) {
    if (!p) return;
    free(p->h); atkdsp_fft_destroy(p->fft); free(p->twiddle);
    free(p->hist); free(p->leftover);
    atk_aligned_free(p->win); atk_aligned_free(p->folded); atk_aligned_free(p->X);
    free(p->cal);
    free(p);
}

void atkdsp_pfb_reset(atkdsp_pfb *p) {
    if (!p) return;
    memset(p->hist, 0, (size_t)p->max_nel * p->ov * sizeof(atkdsp_cf32));
    memset(p->leftover, 0, (size_t)p->max_nel * p->hop * sizeof(atkdsp_cf32));
    p->lc = 0; p->b = 0;
}

unsigned atkdsp_pfb_nchan(const atkdsp_pfb *p) { return p ? p->nchan : 0; }
double   atkdsp_pfb_out_rate(const atkdsp_pfb *p) {
    return p ? (double)p->os * p->fs / (double)p->nchan : 0.0;
}
size_t   atkdsp_pfb_out_hops(const atkdsp_pfb *p, size_t n) {
    if (!p) return 0;
    const size_t avail = (size_t)p->lc + n;
    return avail / p->hop;
}
double   atkdsp_pfb_group_delay(const atkdsp_pfb *p) { return p ? (double)(p->L - 1) * 0.5 : 0.0; }
int64_t  atkdsp_pfb_block_origin(const atkdsp_pfb *p, int64_t hop_index) {
    return p ? hop_index * (int64_t)p->hop - (int64_t)p->ov : 0;
}

int atkdsp_pfb_set_cal(atkdsp_pfb *p, const atkdsp_cf32 *table, unsigned nel) {
    if (!p) return ATKDSP_E_ARG;
    if (!table) { free(p->cal); p->cal = NULL; p->cal_nel = 0; return ATKDSP_OK; }
    if (nel < 1 || nel > p->max_nel) return ATKDSP_E_ARG;
    atkdsp_cf32 *t = (atkdsp_cf32 *)malloc((size_t)nel * p->nchan * sizeof(atkdsp_cf32));
    if (!t) return ATKDSP_E_NOMEM;
    memcpy(t, table, (size_t)nel * p->nchan * sizeof(atkdsp_cf32));
    free(p->cal); p->cal = t; p->cal_nel = nel;
    return ATKDSP_OK;
}

ptrdiff_t atkdsp_pfb_process_n(atkdsp_pfb *p, const atkdsp_cf32 *const *ins, unsigned nel,
                               size_t n, atkdsp_cf32 *out, size_t cap) {
    if (!p || !ins || nel < 1 || nel > p->max_nel) return ATKDSP_E_ARG;
    const unsigned N = p->nchan, hop = p->hop;
    const size_t avail = (size_t)p->lc + n;
    const size_t m = avail / hop;
    if (m == 0) {
        /* stash all into leftover */
        for (unsigned e = 0; e < nel; ++e)
            if (n) memcpy(p->leftover + (size_t)e * hop + p->lc, ins[e], n * sizeof(atkdsp_cf32));
        p->lc = (unsigned)avail;
        return 0;
    }
    if (!out || cap < (size_t)nel * N * m) return ATKDSP_E_CAP;
    const size_t consumed = m * hop;
    const size_t rem = avail - consumed;
    const int64_t b0 = p->b;
    /* per element: build its (leftover ++ in) stream contiguously in `win`? no —
     * we need random access across hops, so index src via a small helper that
     * reads leftover then in. To keep one contiguous src, copy into a scratch
     * stream per element is O(n) extra memory; instead index on the fly. */
    for (unsigned e = 0; e < nel; ++e) {
        atkdsp_cf32 *hist = p->hist + (size_t)e * p->ov;
        const atkdsp_cf32 *cal = (p->cal && e < p->cal_nel) ? p->cal + (size_t)e * N : NULL;
        atkdsp_cf32 *dstbase = out + (size_t)e * N * m;
        const atkdsp_cf32 *lo = p->leftover + (size_t)e * hop;
        const unsigned lc = p->lc;
        const atkdsp_cf32 *xin = ins[e];
        for (size_t i = 0; i < m; ++i) {
            const size_t off = i * hop;                 /* start in (leftover ++ in) */
            /* assemble this hop's hop samples into win[ov..ov+hop) directly here,
             * reading across the leftover/in boundary, then let pfb_hop reuse
             * win's head for hist. We inline the assembly to handle the split. */
            memcpy(p->win, hist, p->ov * sizeof(atkdsp_cf32));
            for (unsigned s = 0; s < hop; ++s) {
                const size_t j = off + s;
                p->win[p->ov + s] = (j < lc) ? lo[j] : xin[j - lc];
            }
            /* fold + fft + rotate + cal, write stride m */
            const unsigned Nn = N;
            for (unsigned j = 0; j < Nn; ++j) { p->folded[j].re = 0.0f; p->folded[j].im = 0.0f; }
            for (size_t k = 0; k < p->K; ++k) {
                const size_t base = k * Nn;
                for (unsigned j = 0; j < Nn; ++j) {
                    const double hc = p->h[base + j];
                    p->folded[j].re += (float)(hc * p->win[base + j].re);
                    p->folded[j].im += (float)(hc * p->win[base + j].im);
                }
            }
            atkdsp_fft_exec(p->fft, p->folded, p->X, 0);
            const int64_t b = b0 + (int64_t)i;
            const unsigned shift = (unsigned)(((b * (int64_t)hop) % (int64_t)Nn + Nn) % Nn);
            for (unsigned k = 0; k < Nn; ++k) {
                const unsigned idx = (unsigned)(((size_t)k * shift) % Nn);
                const float wr = p->twiddle[idx].re, wi = p->twiddle[idx].im;
                float yr = p->X[k].re * wr - p->X[k].im * wi;
                float yi = p->X[k].re * wi + p->X[k].im * wr;
                if (cal) {
                    const float cr = cal[k].re, ci = cal[k].im;
                    const float zr = yr * cr - yi * ci, zi = yr * ci + yi * cr;
                    yr = zr; yi = zi;
                }
                dstbase[(size_t)k * m + i].re = yr;
                dstbase[(size_t)k * m + i].im = yi;
            }
            /* slide hist by hop */
            memmove(hist, p->win + hop, p->ov * sizeof(atkdsp_cf32));
        }
        /* new leftover = last `rem` samples of (leftover ++ in) */
        for (size_t s = 0; s < rem; ++s) {
            const size_t j = consumed + s;
            p->leftover[(size_t)e * hop + s] = (j < lc) ? lo[j] : xin[j - lc];
        }
    }
    p->lc = (unsigned)rem;
    p->b = b0 + (int64_t)m;
    return (ptrdiff_t)m;
}

ptrdiff_t atkdsp_pfb_process(atkdsp_pfb *p, const atkdsp_cf32 *in, size_t n,
                             atkdsp_cf32 *out, size_t cap) {
    const atkdsp_cf32 *ins[1] = { in };
    return atkdsp_pfb_process_n(p, ins, 1, n, out, cap);
}

int atkdsp_pfb_describe(const atkdsp_pfb *p, char *buf, size_t cap) {
    if (!p || !buf || cap == 0) return ATKDSP_E_ARG;
    return snprintf(buf, cap,
        "PFB %.6g MSPS: %u chan x%u oversample, %.6g kHz spacing, %.6g kHz out, "
        "%zu-tap proto (%u/poly), %.0f dB, <=%u el",
        p->fs / 1e6, p->nchan, p->os, p->fs / p->nchan / 1e3,
        atkdsp_pfb_out_rate(p) / 1e3, p->L, p->K, p->atten, p->max_nel);
}

/* ---- W1: per-channel covariance ------------------------------------------ */
void atkdsp_pfb_cov_accumulate(const atkdsp_cf32 *out, unsigned nel, unsigned nchan,
                               size_t m, const uint8_t *gate, atkdsp_cf32 *R, uint32_t *counts) {
    if (!out || !R || !counts || nel < 1 || nchan < 1) return;
    for (unsigned k = 0; k < nchan; ++k) {
        if (gate && !gate[k]) continue;
        atkdsp_cf32 *Rk = R + (size_t)k * nel * nel;
        for (size_t i = 0; i < m; ++i) {
            /* gather y[el] = out[(el*nchan + k)*m + i] */
            for (unsigned a = 0; a < nel; ++a) {
                const atkdsp_cf32 ya = out[((size_t)a * nchan + k) * m + i];
                for (unsigned bb = 0; bb < nel; ++bb) {
                    const atkdsp_cf32 yb = out[((size_t)bb * nchan + k) * m + i];
                    /* R[a][b] += ya * conj(yb) */
                    Rk[(size_t)a * nel + bb].re += ya.re * yb.re + ya.im * yb.im;
                    Rk[(size_t)a * nel + bb].im += ya.im * yb.re - ya.re * yb.im;
                }
            }
        }
        counts[k] += (uint32_t)m;
    }
}

void atkdsp_pfb_cov_reset(atkdsp_cf32 *R, uint32_t *counts, unsigned nel, unsigned nchan,
                          const uint8_t *gate) {
    if (!R || !counts) return;
    for (unsigned k = 0; k < nchan; ++k) {
        if (gate && !gate[k]) continue;
        memset(R + (size_t)k * nel * nel, 0, (size_t)nel * nel * sizeof(atkdsp_cf32));
        counts[k] = 0;
    }
}

/* ---- W2: per-channel power ------------------------------------------------ */
void atkdsp_pfb_channel_power(const atkdsp_cf32 *out, unsigned nel, unsigned nchan,
                              size_t m, float *pwr, float *mean_pwr) {
    if (!out || nel < 1 || nchan < 1) return;
    const double inv = m ? 1.0 / (double)m : 0.0;
    for (unsigned k = 0; k < nchan; ++k) {
        double msum = 0.0;
        for (unsigned a = 0; a < nel; ++a) {
            double s = 0.0;
            for (size_t i = 0; i < m; ++i) {
                const atkdsp_cf32 y = out[((size_t)a * nchan + k) * m + i];
                s += (double)y.re * y.re + (double)y.im * y.im;
            }
            const float pe = (float)(s * inv);
            if (pwr) pwr[(size_t)a * nchan + k] = pe;
            msum += s * inv;
        }
        if (mean_pwr) mean_pwr[k] = (float)(msum / (double)nel);
    }
}
