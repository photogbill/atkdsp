/* 10. DDC — NCO -> staged decimation -> exact-rate resampler, in one object.
 *
 * Why staged: a single 65-tap filter at the INPUT rate cannot be 15 kHz wide
 * at 10 MSPS (its transition is ~500 kHz), so everything within that folds
 * into the channel on decimation — ATK's decoder was being fed ~20 adjacent
 * channels at once (FUTURE_PLANS §RF-F §2). Each stage here is designed so
 * that whatever would alias into [-bw/2, +bw/2] is `atten_db` down, at the
 * rate that stage actually runs at; early stages have wide transitions and
 * few taps, the last one shapes the channel. */
#include "internal.h"
#include <stdio.h>

#define MAX_STAGES 12

double atk_kaiser_beta(double atten_db);
size_t atk_kaiser_length(double atten_db, double transition_hz, double rate);

struct atkdsp_ddc {
    double       fs, bw, out_rate, atten;
    atkdsp_nco   nco;
    int          nstages;
    unsigned     factor[MAX_STAGES];
    unsigned     ntaps[MAX_STAGES];
    double       rate_out[MAX_STAGES];      /* rate after stage i */
    atkdsp_fir  *fir[MAX_STAGES];
    atkdsp_resampler *rs;                    /* rational: r -> out_rate, small L/M */
    atkdsp_arb_resampler *arb;               /* arbitrary: r -> out_rate, any ratio */
    unsigned     up, down;
    unsigned     rs_taps;
    double       r;                          /* rate after the integer stages */
    size_t       max_block;
    atkdsp_cf32 *buf[MAX_STAGES + 1];        /* buf[0] = mixed input, buf[i+1] = stage i out */
    size_t       cap[MAX_STAGES + 1];
};

static unsigned gcd_u(unsigned a, unsigned b) {
    while (b) { unsigned t = a % b; a = b; b = t; }
    return a;
}

/* split D into stage factors, largest first, each <= 8 where the factors allow */
static int plan_factors(unsigned D, unsigned *out) {
    int n = 0;
    while (D > 1 && n < MAX_STAGES) {
        unsigned f = 0;
        for (unsigned c = 8; c >= 2; --c)
            if (D % c == 0) { f = c; break; }
        if (!f) {                            /* a prime > 8 remains */
            for (unsigned c = 9; c <= D; ++c)
                if (D % c == 0) { f = c; break; }
        }
        out[n++] = f;
        D /= f;
    }
    return n;
}

atkdsp_ddc *atkdsp_ddc_create(double fs, double offset_hz, double bw, double out_rate,
                              double atten_db, size_t max_block) {
    if (fs <= 0.0 || bw <= 0.0 || out_rate <= 0.0 || max_block == 0) return NULL;
    if (out_rate > fs || bw > fs) return NULL;
    if (atten_db < 20.0) atten_db = 20.0;
    atkdsp_ddc *d = (atkdsp_ddc *)calloc(1, sizeof *d);
    if (!d) return NULL;
    d->fs = fs; d->bw = bw; d->out_rate = out_rate; d->atten = atten_db;
    d->max_block = max_block;
    atkdsp_nco_init(&d->nco, offset_hz, fs);

    /* integer decimation to the lowest rate that still holds the channel and
     * is at least the requested output rate */
    const double floor_rate = out_rate > 1.25 * bw ? out_rate : 1.25 * bw;
    unsigned D = (unsigned)floor(fs / floor_rate);
    if (D < 1) D = 1;
    d->r = fs / (double)D;
    d->nstages = plan_factors(D, d->factor);

    double R = fs;
    float *taps = NULL; size_t tcap = 0;
    for (int i = 0; i < d->nstages; ++i) {
        const double Rn = R / (double)d->factor[i];
        const double fp = 0.5 * bw;
        double fstop = Rn - 0.5 * bw;             /* first alias into the channel */
        if (i == d->nstages - 1 && fstop > bw) fstop = bw;   /* last stage shapes the channel */
        if (fstop > 0.5 * R) fstop = 0.5 * R;
        if (fstop <= fp) { atkdsp_ddc_destroy(d); free(taps); return NULL; }
        ptrdiff_t n = atkdsp_design_lowpass(fp, fstop, atten_db, R, NULL, 0);
        if (n <= 0) { atkdsp_ddc_destroy(d); free(taps); return NULL; }
        if ((size_t)n > tcap) { free(taps); taps = (float *)malloc((size_t)n * sizeof(float)); tcap = (size_t)n; }
        if (!taps) { atkdsp_ddc_destroy(d); return NULL; }
        atkdsp_design_lowpass(fp, fstop, atten_db, R, taps, tcap);
        d->fir[i] = atkdsp_fir_create(taps, (size_t)n, d->factor[i]);
        d->ntaps[i] = (unsigned)n;
        if (!d->fir[i]) { atkdsp_ddc_destroy(d); free(taps); return NULL; }
        R = Rn;
        d->rate_out[i] = R;
    }
    if (d->nstages == 0) {
        /* no decimation at all: still shape the channel */
        const double fp = 0.5 * bw;
        double fstop = bw < 0.5 * fs ? bw : 0.5 * fs;
        ptrdiff_t n = atkdsp_design_lowpass(fp, fstop, atten_db, fs, NULL, 0);
        if (n > 0) {
            taps = (float *)malloc((size_t)n * sizeof(float));
            if (taps && atkdsp_design_lowpass(fp, fstop, atten_db, fs, taps, (size_t)n) == n) {
                d->fir[0] = atkdsp_fir_create(taps, (size_t)n, 1);
                d->factor[0] = 1; d->ntaps[0] = (unsigned)n; d->rate_out[0] = fs;
                d->nstages = 1;
            }
        }
    }

    /* final rate step: r -> out_rate. A small integer ratio is done exactly by
     * the rational resampler; a real-number ratio (an odd sample clock like
     * 2.457600 MHz, or a rate whose reduced denominator is huge) goes to the
     * arbitrary resampler instead of being refused. */
    d->up = 1; d->down = 1;
    if (fabs(d->r - out_rate) > 1e-9 * out_rate) {
        const double num = out_rate * (double)D, den = fs;    /* out/r = out*D/fs */
        if (num > 4e9 || den > 4e9 || floor(num) != num || floor(den) != den) {
            d->arb = atkdsp_arb_resampler_create(d->r, out_rate, atten_db, 0);
            if (!d->arb) { atkdsp_ddc_destroy(d); free(taps); return NULL; }
        } else {
            unsigned g = gcd_u((unsigned)num, (unsigned)den);
            d->up = (unsigned)num / g; d->down = (unsigned)den / g;
            const double hi = d->r * (double)d->up;            /* the polyphase rate */
            const double nyq = 0.5 * (d->r < out_rate ? d->r : out_rate);
            const double fp = nyq * 0.8, fstop = nyq;
            ptrdiff_t n = atkdsp_design_lowpass(fp, fstop, atten_db, hi, NULL, 0);
            if (n <= 0) { atkdsp_ddc_destroy(d); free(taps); return NULL; }
            free(taps); taps = (float *)malloc((size_t)n * sizeof(float));
            if (!taps) { atkdsp_ddc_destroy(d); return NULL; }
            atkdsp_design_lowpass(fp, fstop, atten_db, hi, taps, (size_t)n);
            d->rs = atkdsp_resampler_create(d->up, d->down, taps, (size_t)n);
            d->rs_taps = (unsigned)n;
            if (!d->rs) { atkdsp_ddc_destroy(d); free(taps); return NULL; }
        }
    }
    free(taps);

    /* scratch, sized once for max_block (rule 2) */
    size_t n = max_block;
    d->cap[0] = n;
    d->buf[0] = (atkdsp_cf32 *)atk_aligned_malloc(n * sizeof(atkdsp_cf32));
    if (!d->buf[0]) { atkdsp_ddc_destroy(d); return NULL; }
    for (int i = 0; i < d->nstages; ++i) {
        n = n / d->factor[i] + 2;
        d->cap[i + 1] = n;
        d->buf[i + 1] = (atkdsp_cf32 *)atk_aligned_malloc(n * sizeof(atkdsp_cf32));
        if (!d->buf[i + 1]) { atkdsp_ddc_destroy(d); return NULL; }
    }
    return d;
}

void atkdsp_ddc_destroy(atkdsp_ddc *d) {
    if (!d) return;
    for (int i = 0; i < MAX_STAGES; ++i) atkdsp_fir_destroy(d->fir[i]);
    atkdsp_resampler_destroy(d->rs);
    atkdsp_arb_resampler_destroy(d->arb);
    for (int i = 0; i <= MAX_STAGES; ++i) atk_aligned_free(d->buf[i]);
    free(d);
}

void atkdsp_ddc_reset(atkdsp_ddc *d) {
    if (!d) return;
    d->nco.phase = 0.0;
    for (int i = 0; i < d->nstages; ++i) atkdsp_fir_reset(d->fir[i]);
    if (d->rs) atkdsp_resampler_reset(d->rs);
    if (d->arb) atkdsp_arb_resampler_reset(d->arb);
}

void atkdsp_ddc_set_offset(atkdsp_ddc *d, double offset_hz) {
    if (d) atkdsp_nco_set_freq(&d->nco, offset_hz, d->fs);
}

double atkdsp_ddc_out_rate(const atkdsp_ddc *d) {
    return d ? ((d->rs || d->arb) ? d->out_rate : d->r) : 0.0;
}

size_t atkdsp_ddc_out_max(const atkdsp_ddc *d, size_t n_in) {
    if (!d) return 0;
    size_t n = n_in;
    for (int i = 0; i < d->nstages; ++i) n = atkdsp_fir_out_max(d->fir[i], n);
    if (d->rs) n = atkdsp_resampler_out_max(d->rs, n);
    else if (d->arb) n = atkdsp_arb_resampler_out_max(d->arb, n);
    return n;
}

ptrdiff_t atkdsp_ddc_process(atkdsp_ddc *d, const atkdsp_cf32 *in, size_t n,
                             atkdsp_cf32 *out, size_t out_cap) {
    if (!d || (!in && n) || !out) return ATKDSP_E_ARG;
    if (n > d->max_block) return ATKDSP_E_CAP;
    atkdsp_nco_mix(&d->nco, in, d->buf[0], n);
    size_t cur = n;
    for (int i = 0; i < d->nstages; ++i) {
        ptrdiff_t got = atkdsp_fir_process(d->fir[i], d->buf[i], cur, d->buf[i + 1], d->cap[i + 1]);
        if (got < 0) return got;
        cur = (size_t)got;
    }
    const atkdsp_cf32 *last = d->buf[d->nstages];
    if (d->rs)
        return atkdsp_resampler_process(d->rs, last, cur, out, out_cap);
    if (d->arb)
        return atkdsp_arb_resampler_process(d->arb, last, cur, out, out_cap);
    if (cur > out_cap) return ATKDSP_E_CAP;
    memcpy(out, last, cur * sizeof(atkdsp_cf32));
    return (ptrdiff_t)cur;
}

int atkdsp_ddc_plan(const atkdsp_ddc *d, unsigned *factors, unsigned *ntaps,
                    size_t cap, unsigned *up, unsigned *down) {
    if (!d) return ATKDSP_E_ARG;
    for (int i = 0; i < d->nstages && (size_t)i < cap; ++i) {
        if (factors) factors[i] = d->factor[i];
        if (ntaps) ntaps[i] = d->ntaps[i];
    }
    if (up) *up = d->up;
    if (down) *down = d->down;
    return d->nstages;
}

int atkdsp_ddc_describe(const atkdsp_ddc *d, char *buf, size_t cap) {
    if (!d || !buf || cap == 0) return ATKDSP_E_ARG;
    int w = snprintf(buf, cap, "%.6g MSPS: NCO", d->fs / 1e6);
    for (int i = 0; i < d->nstages && w >= 0 && (size_t)w < cap; ++i)
        w += snprintf(buf + w, cap - (size_t)w, " -> /%u (%u taps)", d->factor[i], d->ntaps[i]);
    if (w >= 0 && (size_t)w < cap)
        w += snprintf(buf + w, cap - (size_t)w, " -> %.6g Hz", d->r);
    if (d->rs && w >= 0 && (size_t)w < cap)
        w += snprintf(buf + w, cap - (size_t)w, " -> x%u/%u (%u taps) -> %.6g Hz",
                      d->up, d->down, d->rs_taps, d->out_rate);
    else if (d->arb && w >= 0 && (size_t)w < cap)
        w += snprintf(buf + w, cap - (size_t)w, " -> arb -> %.6g Hz", d->out_rate);
    return w;
}
