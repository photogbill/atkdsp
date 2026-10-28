/* 20. The burst descriptor word (docs/ATKDSP_PLAN.md §B2).
 *
 * One record per burst from channelised I/Q and a coarse [i0, i1) (the
 * CFAR's t0 / t1 in samples): edges refined in the time domain, occupied
 * bandwidth, centre, drift, levels, rise and fall. Definitions in atkdsp.h
 * §20; the twin is python/atkdsp/reference.py §20 (burst_measure), and the
 * arithmetic below runs in its order so the two agree to the last decision.
 *
 * A one-shot analysis call: it allocates scratch for the length of the call
 * (the §14 convention) and frees it before returning.
 */
#include "internal.h"
#include "pocketfft.h"

typedef struct {
    const atkdsp_cf32 *x;
    double *p, *cs, *ps, *pr;      /* power, its cumulative sum, two smoothings */
    double *fbuf, *acc;            /* FFT scratch (2 x 65536 max), PSD          */
    size_t n;
} bscratch;

static double seq_mean2(const double *a, size_t na, const double *b, size_t nb) {
    double s = 0.0;
    for (size_t i = 0; i < na; ++i) s += a[i];
    for (size_t i = 0; i < nb; ++i) s += b[i];
    return s / (double)(na + nb);
}

static double seq_mean(const double *a, size_t n) { return seq_mean2(a, n, NULL, 0); }

static double hann(size_t j, size_t n) {
    return 0.5 - 0.5 * cos(2.0 * ATK_PI * (double)j / (double)n);
}

/* Centred moving average of width W (odd), shrinking at the ends. */
static void smooth(const double *cs, size_t n, size_t W, double *out) {
    const size_t h = W / 2;
    for (size_t j = 0; j < n; ++j) {
        const size_t a = j >= h ? j - h : 0;
        const size_t b = j + h + 1 < n ? j + h + 1 : n;
        out[j] = (cs[b] - cs[a]) / (double)(b - a);
    }
}

static size_t auto_nfft(size_t L) {
    size_t m = 16;
    while (m * 2 <= L / 8 && m * 2 <= 4096) m *= 2;
    return m;
}

/* Welch: Hann, hop nfft/2, |FFT|^2 averaged, shifted (index 0 = -fs/2). */
static int welch(const atkdsp_cf32 *x, size_t len, size_t nfft, double *fbuf, double *acc,
                 int *nseg_out, double *u_out) {
    cfft_plan plan = make_cfft_plan(nfft);
    if (!plan) return ATKDSP_E_NOMEM;
    double u = 0.0;
    for (size_t j = 0; j < nfft; ++j) { const double w = hann(j, nfft); u += w * w; }
    double *tmp = fbuf + 2 * nfft;          /* nfft doubles of unshifted sums */
    for (size_t k = 0; k < nfft; ++k) tmp[k] = 0.0;
    int nseg = 0;
    for (size_t s = 0; s + nfft <= len; s += nfft / 2) {
        for (size_t j = 0; j < nfft; ++j) {
            const double w = hann(j, nfft);
            fbuf[2 * j] = (double)x[s + j].re * w;
            fbuf[2 * j + 1] = (double)x[s + j].im * w;
        }
        if (cfft_forward(plan, fbuf, 1.0) != 0) { destroy_cfft_plan(plan); return ATKDSP_E_FFT; }
        for (size_t k = 0; k < nfft; ++k)
            tmp[k] += fbuf[2 * k] * fbuf[2 * k] + fbuf[2 * k + 1] * fbuf[2 * k + 1];
        ++nseg;
    }
    destroy_cfft_plan(plan);
    const double d = nseg > 0 ? (double)nseg : 1.0;
    for (size_t k = 0; k < nfft; ++k) acc[k] = tmp[(k + nfft / 2) % nfft] / d;
    *nseg_out = nseg;
    *u_out = u;
    return ATKDSP_OK;
}

/* The occupied band of a SIGNED noise-subtracted PSD; 0 when none. */
static int obw(const double *D, double fs, size_t nfft, double frac, double *f_lo,
               double *f_hi, size_t *k_lo, size_t *k_hi) {
    const double df = fs / (double)nfft;
    double T = 0.0;
    for (size_t k = 0; k < nfft; ++k) T += D[k];
    if (!(T > 0.0)) return 0;
    const double tail = 0.5 * (1.0 - frac) * T;
    const ptrdiff_t h = (ptrdiff_t)(nfft / 2);
    double cum = 0.0;
    int got_lo = 0, got_hi = 0;
    for (size_t k = 0; k < nfft; ++k) {
        const double prev = cum;
        cum += D[k];
        if (cum >= tail && D[k] > 0.0) {
            *f_lo = (double)((ptrdiff_t)k - h) * df - 0.5 * df + df * (tail - prev) / D[k];
            *k_lo = k; got_lo = 1; break;
        }
    }
    cum = 0.0;
    for (size_t kk = nfft; kk-- > 0;) {
        const double prev = cum;
        cum += D[kk];
        if (cum >= tail && D[kk] > 0.0) {
            *f_hi = (double)((ptrdiff_t)kk - h) * df + 0.5 * df - df * (tail - prev) / D[kk];
            *k_hi = kk; got_hi = 1; break;
        }
    }
    return got_lo && got_hi && *k_hi >= *k_lo;
}

static ptrdiff_t cusum_start(const double *p, double lvl, ptrdiff_t a, ptrdiff_t b) {
    double best = 0.0, acc = 0.0;
    ptrdiff_t tau = b + 1;
    for (ptrdiff_t j = b; j >= a; --j) {
        acc += p[j] - lvl;
        if (acc > best) { best = acc; tau = j; }
    }
    return tau;
}

static ptrdiff_t cusum_end(const double *p, double lvl, ptrdiff_t a, ptrdiff_t b) {
    double best = 0.0, acc = 0.0;
    ptrdiff_t tau = a - 1;
    for (ptrdiff_t j = a; j <= b; ++j) {
        acc += p[j] - lvl;
        if (acc > best) { best = acc; tau = j; }
    }
    return tau;
}

typedef struct {
    double lvl, t0, t1;
    uint32_t f;
} edge_out;

/* Pass of the edge search: the core run (quarter-power, merged across gaps
 * of <= W) that holds the coarse midpoint (else the one overlapping the
 * middle half most), then a least-squares step at `lvl` around each end. */
static int edges(const bscratch *S, double N, double P, double thr, int gauss, size_t W,
                 ptrdiff_t mid, ptrdiff_t m0, ptrdiff_t m1, edge_out *eo) {
    const ptrdiff_t n = (ptrdiff_t)S->n;
    double lvl;
    if (thr > 0.0 || !(N > 0.0) || !gauss) {
        const double t = thr > 0.0 ? thr : ATKDSP_BDW_HALF_POWER_DB;
        lvl = N + (P - N) * pow(10.0, -t / 10.0);
    } else {
        lvl = log(P / N) * N * P / (P - N);
    }
    const double core_lvl = N + 0.25 * (P - N);
    const double *ps = S->ps;
    /* walk the runs in order, keeping the selection rule's answer */
    ptrdiff_t sel_a = -1, sel_b = -1, best_ov = 0, ov_a = -1, ov_b = -1;
    ptrdiff_t ra = -1, rb = -1;
    int found_mid = 0;
    ptrdiff_t j = 0;
    while (j < n && !found_mid) {
        if (ps[j] >= core_lvl) {
            const ptrdiff_t a = j;
            while (j < n && ps[j] >= core_lvl) ++j;
            const ptrdiff_t b = j - 1;
            if (ra >= 0 && a - rb - 1 <= (ptrdiff_t)W) {
                rb = b;
            } else {
                if (ra >= 0) {                 /* close the previous run */
                    if (ra <= mid && mid <= rb) { sel_a = ra; sel_b = rb; found_mid = 1; break; }
                    const ptrdiff_t ov = (rb < m1 - 1 ? rb : m1 - 1) - (ra > m0 ? ra : m0) + 1;
                    if (ov > best_ov) { best_ov = ov; ov_a = ra; ov_b = rb; }
                }
                ra = a; rb = b;
            }
        } else {
            ++j;
        }
    }
    if (!found_mid && ra >= 0) {
        if (ra <= mid && mid <= rb) { sel_a = ra; sel_b = rb; found_mid = 1; }
        else {
            const ptrdiff_t ov = (rb < m1 - 1 ? rb : m1 - 1) - (ra > m0 ? ra : m0) + 1;
            if (ov > best_ov) { best_ov = ov; ov_a = ra; ov_b = rb; }
        }
    }
    if (!found_mid) { sel_a = ov_a; sel_b = ov_b; }
    if (sel_a < 0) return ATKDSP_E_EMPTY;
    const ptrdiff_t e0 = sel_a, e1 = sel_b;
    const ptrdiff_t R = (ptrdiff_t)W + 8, half = (e1 - e0) / 2;
    const ptrdiff_t Rh = R < half ? R : half;
    eo->f = 0;
    /* the step is searched for even when the smoothed core reaches the
     * buffer's end (smoothing spreads a burst by half a window, which can
     * exceed a short margin); truncated means the step is AT the end */
    {
        const ptrdiff_t a = e0 - R > 0 ? e0 - R : 0, b = e0 + Rh < n - 1 ? e0 + Rh : n - 1;
        const ptrdiff_t tau = cusum_start(S->p, lvl, a, b);
        eo->t0 = (double)tau - 0.5;
        if (tau == 0) eo->f |= ATKDSP_BDW_TRUNC_START;
    }
    {
        const ptrdiff_t a = e1 - Rh > 0 ? e1 - Rh : 0, b = e1 + R < n - 1 ? e1 + R : n - 1;
        const ptrdiff_t tau = cusum_end(S->p, lvl, a, b);
        eo->t1 = (double)tau + 0.5;
        if (tau == n - 1) eo->f |= ATKDSP_BDW_TRUNC_END;
    }
    eo->lvl = lvl;
    return ATKDSP_OK;
}

/* First crossing of `level` by pr walking from j_from to j_to, interpolated. */
static int cross(const double *pr, ptrdiff_t j_from, ptrdiff_t j_to, double level, double *out) {
    const ptrdiff_t step = j_to >= j_from ? 1 : -1;
    ptrdiff_t j = j_from;
    while (j != j_to) {
        const ptrdiff_t k = j + step;
        const double a = pr[j], b = pr[k];
        if ((a < level) != (b < level)) {
            *out = (double)j + (double)step * (level - a) / (b - a);
            return 1;
        }
        j = k;
    }
    return 0;
}

static float dbf(double v) { return v > 0.0 ? (float)(10.0 * log10(v)) : -INFINITY; }

static int measure(bscratch *S, double fs, ptrdiff_t i0, ptrdiff_t i1, double thr, int nfft_req,
                   atkdsp_bdw *out) {
    const ptrdiff_t n = (ptrdiff_t)S->n;
    const atkdsp_cf32 *x = S->x;
    double *p = S->p;
    uint32_t flags = 0;
    const ptrdiff_t L = i1 - i0, g = L / 8 + 8;
    const ptrdiff_t m0 = i0 + L / 4, m1 = i1 - L / 4;
    double P = seq_mean(p + m0, (size_t)(m1 - m0));
    double N;
    {
        ptrdiff_t a = i0 - g > 0 ? i0 - g : 0;
        ptrdiff_t b = i1 + g < n ? i1 + g : n;
        const int guarded = a + (n - b) >= 32;
        if (!guarded) { a = i0; b = i1; }          /* no room for the guard */
        if (a + (n - b) >= 32) {
            N = seq_mean2(p, (size_t)a, p + b, (size_t)(n - b));
            /* unguarded samples within 1 dB of the plateau are the burst
             * itself (it fills the buffer), not noise */
            if (!guarded && !(P > 1.25 * N)) { N = 0.0; flags |= ATKDSP_BDW_NO_NOISE; }
        } else { N = 0.0; flags |= ATKDSP_BDW_NO_NOISE; }
    }
    if (!(P > N)) return ATKDSP_E_EMPTY;

    /* the preliminary bandwidth, for the smoothing */
    const size_t nf0 = auto_nfft((size_t)(m1 - m0));
    int nseg;
    double u;
    int rc = welch(x + m0, (size_t)(m1 - m0), nf0, S->fbuf, S->acc, &nseg, &u);
    if (rc) return rc;
    for (size_t k = 0; k < nf0; ++k) S->acc[k] -= N * u;
    double flo, fhi;
    size_t klo, khi;
    double b_est = fs / (double)nf0;
    if (obw(S->acc, fs, nf0, 0.99, &flo, &fhi, &klo, &khi) && fhi - flo > b_est) b_est = fhi - flo;
    const ptrdiff_t wmax = L / 4 > 1 ? L / 4 : 1;
    ptrdiff_t W = (ptrdiff_t)floor(32.0 * fs / b_est + 0.5);
    if (W < 3) W = 3;
    if (W > wmax) W = wmax;
    if (W % 2 == 0) W = W > 1 ? W - 1 : 1;
    ptrdiff_t Wr = (ptrdiff_t)floor(fs / b_est + 0.5);
    if (Wr < 1) Wr = 1;
    if (Wr > W) Wr = W;
    if (Wr > 63) Wr = 63;
    if (Wr % 2 == 0) Wr -= 1;
    const ptrdiff_t mid = (i0 + i1) / 2;

    smooth(S->cs, S->n, (size_t)W, S->ps);
    edge_out eo;
    rc = edges(S, N, P, thr, 0, (size_t)W, mid, m0, m1, &eo);
    if (rc) return rc;
    /* second pass: noise from outside the measured burst, plateau from inside */
    const ptrdiff_t R2 = 2 * W + 8;
    {
        ptrdiff_t lo_cut = (ptrdiff_t)floor(eo.t0) - R2, hi_cut = (ptrdiff_t)ceil(eo.t1) + R2;
        const ptrdiff_t a = lo_cut > 0 ? lo_cut : 0;
        ptrdiff_t b = hi_cut > 0 ? hi_cut : 0;
        if (b > n) b = n;
        if (a + (n - b) >= 32) {
            N = seq_mean2(p, (size_t)a, p + b, (size_t)(n - b));
            flags &= ~ATKDSP_BDW_NO_NOISE;
        }
    }
    ptrdiff_t a = (ptrdiff_t)ceil(eo.t0) + W, b = (ptrdiff_t)floor(eo.t1) - W;
    if (b - a + 1 < 16) {
        a = (ptrdiff_t)ceil(eo.t0) > 0 ? (ptrdiff_t)ceil(eo.t0) : 0;
        b = (ptrdiff_t)floor(eo.t1) < n - 1 ? (ptrdiff_t)floor(eo.t1) : n - 1;
    }
    if (b < a) return ATKDSP_E_EMPTY;
    P = seq_mean(p + a, (size_t)(b - a + 1));
    if (!(P > N)) return ATKDSP_E_EMPTY;
    double var = 0.0;
    for (ptrdiff_t j = a; j <= b; ++j) var += (p[j] - P) * (p[j] - P);
    var /= (double)(b - a + 1);
    const double Sg = P - N;
    const double kappa = (var - N * N - 2.0 * Sg * N) / (Sg * Sg);
    const int gauss = kappa > 0.5;
    if (gauss && thr == 0.0 && N > 0.0) flags |= ATKDSP_BDW_GAUSSIAN;
    rc = edges(S, N, P, thr, gauss, (size_t)W, mid, m0, m1, &eo);
    if (rc) return rc;
    flags |= eo.f;
    const double t0 = eo.t0, t1 = eo.t1;
    const ptrdiff_t s0 = (ptrdiff_t)ceil(t0) > 0 ? (ptrdiff_t)ceil(t0) : 0;
    const ptrdiff_t s1 = (ptrdiff_t)floor(t1) < n - 1 ? (ptrdiff_t)floor(t1) : n - 1;
    if (s1 < s0) return ATKDSP_E_EMPTY;
    double pk = p[s0];
    for (ptrdiff_t j = s0 + 1; j <= s1; ++j) if (p[j] > pk) pk = p[j];
    const double mean_p = seq_mean(p + s0, (size_t)(s1 - s0 + 1));
    const double snr = N > 0.0 ? 10.0 * log10((P - N) / N) : INFINITY;
    if (N > 0.0 && snr < 6.0) flags |= ATKDSP_BDW_LOW_SNR;

    /* rise and fall: 10-90 % amplitude (1 % and 81 % of the power above noise) */
    smooth(S->cs, S->n, (size_t)Wr, S->pr);
    const double l10 = N + 0.01 * (P - N), l90 = N + 0.81 * (P - N);
    const double noise_sd = N > 0.0 ? N / sqrt((double)Wr) : 0.0;
    if (gauss) flags |= ATKDSP_BDW_RISE_UNRESOLVED | ATKDSP_BDW_FALL_UNRESOLVED;
    double rise = NAN, fall = NAN;
    if (!(eo.f & ATKDSP_BDW_TRUNC_START)) {
        const ptrdiff_t c0 = (ptrdiff_t)floor(t0);
        const ptrdiff_t lim = c0 - R2 > 0 ? c0 - R2 : 0;
        double t10, t90;
        const int h10 = cross(S->pr, c0 + 1 < n - 1 ? c0 + 1 : n - 1, lim, l10, &t10);
        const int h90 = cross(S->pr, c0 > 0 ? c0 : 0, s1 < n - 1 ? s1 : n - 1, l90, &t90);
        if (h10 && h90) rise = (t90 - t10) / fs;
        if (0.01 * (P - N) < 3.0 * noise_sd || !(rise > 2.0 * (double)Wr / fs))
            flags |= ATKDSP_BDW_RISE_UNRESOLVED;
    } else {
        flags |= ATKDSP_BDW_RISE_UNRESOLVED;
    }
    if (!(eo.f & ATKDSP_BDW_TRUNC_END)) {
        const ptrdiff_t c1 = (ptrdiff_t)ceil(t1);
        const ptrdiff_t lim = c1 + R2 < n - 1 ? c1 + R2 : n - 1;
        double t10, t90;
        const int h10 = cross(S->pr, c1 - 1 > 0 ? c1 - 1 : 0, lim, l10, &t10);
        const int h90 = cross(S->pr, c1 < n - 1 ? c1 : n - 1, s0 > 0 ? s0 : 0, l90, &t90);
        if (h10 && h90) fall = (t10 - t90) / fs;
        if (0.01 * (P - N) < 3.0 * noise_sd || !(fall > 2.0 * (double)Wr / fs))
            flags |= ATKDSP_BDW_FALL_UNRESOLVED;
    } else {
        flags |= ATKDSP_BDW_FALL_UNRESOLVED;
    }

    /* the spectrum of the burst, widened by half a segment each side */
    const size_t nb_len = (size_t)(s1 - s0 + 1);
    size_t nf = nfft_req ? (size_t)nfft_req : auto_nfft(nb_len);
    if (nf < 16 || (nf & (nf - 1)) || nf > 65536) nf = auto_nfft(nb_len);
    const ptrdiff_t A = s0 - (ptrdiff_t)(nf / 2) > 0 ? s0 - (ptrdiff_t)(nf / 2) : 0;
    const ptrdiff_t B = s1 + 1 + (ptrdiff_t)(nf / 2) < n ? s1 + 1 + (ptrdiff_t)(nf / 2) : n;
    if ((size_t)(B - A) < nf) {
        if (B - A < 16) return ATKDSP_E_EMPTY;
        nf = 16;
        while (nf * 2 <= (size_t)(B - A)) nf *= 2;
    }
    rc = welch(x + A, (size_t)(B - A), nf, S->fbuf, S->acc, &nseg, &u);
    if (rc) return rc;
    const double nbin = N * u;
    double *D = S->acc;
    for (size_t k = 0; k < nf; ++k) D[k] = D[k] - nbin;
    if (!obw(D, fs, nf, 0.99, &flo, &fhi, &klo, &khi)) return ATKDSP_E_EMPTY;
    const double df = fs / (double)nf;
    const ptrdiff_t h = (ptrdiff_t)(nf / 2);
    double sw = 0.0, sf = 0.0;
    for (size_t k = klo; k <= khi; ++k) { sw += D[k]; sf += (double)((ptrdiff_t)k - h) * df * D[k]; }
    const double fc = sw > 0.0 ? sf / sw : 0.5 * (flo + fhi);
    double dmax = D[0];
    for (size_t k = 1; k < nf; ++k) if (D[k] > dmax) dmax = D[k];
    const double line = dmax * pow(10.0, -2.6);
    double bw26 = NAN;
    if (nbin * (1.0 + 3.0 / sqrt((double)nseg)) > line) flags |= ATKDSP_BDW_BW26_FLOOR;
    else {
        size_t ka = 0, kb = nf - 1;
        while (ka < nf && !(D[ka] >= line)) ++ka;
        while (kb > 0 && !(D[kb] >= line)) --kb;
        bw26 = (double)(kb - ka + 1) * df;
    }

    /* drift: each segment's centroid inside the occupied band, a line fitted */
    ptrdiff_t c0 = s0 + Wr, c1 = s1 - Wr + 1;
    if (c1 - c0 < 16) { c0 = s0; c1 = s1 + 1; }
    ptrdiff_t nsegd = (c1 - c0) / 64;
    if (nsegd > 8) nsegd = 8;
    double drift = NAN;
    if (nsegd >= 3) {
        const ptrdiff_t step = (c1 - c0) / nsegd;        /* spread over the whole burst, */
        const ptrdiff_t Ls = step < 32768 ? step : 32768; /* each at most 32768 samples  */
        size_t nf2 = 16;
        while (nf2 < (size_t)(2 * Ls) && nf2 < 65536) nf2 *= 2;
        double us = 0.0;
        for (ptrdiff_t j = 0; j < Ls; ++j) { const double w = hann((size_t)j, (size_t)Ls); us += w * w; }
        const double dres = fs / (double)Ls;
        const double lo_f = flo - 2.0 * dres, hi_f = fhi + 2.0 * dres;
        cfft_plan plan = make_cfft_plan(nf2);
        if (!plan) return ATKDSP_E_NOMEM;
        double ts[8], fcs[8];
        int nv = 0;
        const ptrdiff_t h2 = (ptrdiff_t)(nf2 / 2);
        for (ptrdiff_t sg = 0; sg < nsegd; ++sg) {
            const ptrdiff_t a0 = c0 + sg * step + (step - Ls) / 2;
            for (size_t j = 0; j < 2 * nf2; ++j) S->fbuf[j] = 0.0;
            for (ptrdiff_t j = 0; j < Ls; ++j) {
                const double w = hann((size_t)j, (size_t)Ls);
                S->fbuf[2 * j] = (double)x[a0 + j].re * w;
                S->fbuf[2 * j + 1] = (double)x[a0 + j].im * w;
            }
            if (cfft_forward(plan, S->fbuf, 1.0) != 0) { destroy_cfft_plan(plan); return ATKDSP_E_FFT; }
            double ssw = 0.0, ssf = 0.0;
            for (size_t k = 0; k < nf2; ++k) {
                const size_t ku = (k + nf2 / 2) % nf2;          /* shifted index k */
                const double fk = (double)((ptrdiff_t)k - h2) * (fs / (double)nf2);
                if (lo_f <= fk && fk <= hi_f) {
                    const double re = S->fbuf[2 * ku], im = S->fbuf[2 * ku + 1];
                    const double d = (re * re + im * im) - N * us;
                    ssw += d;
                    ssf += fk * d;
                }
            }
            if (ssw > 0.0) {
                ts[nv] = ((double)a0 + 0.5 * (double)Ls - (double)c0) / fs;
                fcs[nv] = ssf / ssw;
                ++nv;
            }
        }
        destroy_cfft_plan(plan);
        if (nv >= 3) {
            const double mt = seq_mean(ts, (size_t)nv), mf = seq_mean(fcs, (size_t)nv);
            double sxy = 0.0, sxx = 0.0;
            for (int q = 0; q < nv; ++q) {
                sxy += (ts[q] - mt) * (fcs[q] - mf);
                sxx += (ts[q] - mt) * (ts[q] - mt);
            }
            drift = sxx > 0.0 ? sxy / sxx : NAN;
        }
    }
    if (isnan(drift)) flags |= ATKDSP_BDW_SHORT;

    memset(out, 0, sizeof *out);
    out->t0 = t0; out->t1 = t1;
    out->duration_s = (t1 - t0) / fs;
    out->fc_hz = fc;
    out->drift_hz_s = drift;
    out->obw99_hz = fhi - flo;
    out->bw26_hz = bw26;
    out->rise_s = rise; out->fall_s = fall;
    out->peak_dbfs = dbf(pk);
    out->mean_dbfs = dbf(mean_p);
    out->noise_dbfs = (flags & ATKDSP_BDW_NO_NOISE) ? NAN : dbf(N);
    out->snr_db = (flags & ATKDSP_BDW_NO_NOISE) ? NAN : (float)snr;
    out->mod_hint = 0;
    out->flags = flags;
    out->nfft = (int32_t)nf;
    out->window = (int32_t)W;
    return ATKDSP_OK;
}

int atkdsp_burst_measure(const atkdsp_cf32 *iq, size_t n, double fs, size_t i0_coarse,
                         size_t i1_coarse, float thresh_db, int nfft, atkdsp_bdw *out) {
    if (!iq || !out || !(fs > 0.0) || n < 32 || i0_coarse >= i1_coarse || i1_coarse > n
        || i1_coarse - i0_coarse < 16 || !(thresh_db <= 40.0f) || nfft < 0 || n > ((size_t)1 << 40))
        return ATKDSP_E_ARG;
    bscratch S;
    memset(&S, 0, sizeof S);
    S.x = iq;
    S.n = n;
    S.p = (double *)malloc(n * sizeof(double));
    S.cs = (double *)malloc((n + 1) * sizeof(double));
    S.ps = (double *)malloc(n * sizeof(double));
    S.pr = (double *)malloc(n * sizeof(double));
    S.fbuf = (double *)malloc(3 * 65536 * sizeof(double) * 2);
    S.acc = (double *)malloc(65536 * sizeof(double));
    int rc = ATKDSP_E_NOMEM;
    if (S.p && S.cs && S.ps && S.pr && S.fbuf && S.acc) {
        S.cs[0] = 0.0;
        for (size_t j = 0; j < n; ++j) {
            const double re = (double)iq[j].re, im = (double)iq[j].im;
            S.p[j] = re * re + im * im;
            S.cs[j + 1] = S.cs[j] + S.p[j];
        }
        const double thr = thresh_db > 0.0f ? (double)thresh_db : 0.0;
        rc = measure(&S, fs, (ptrdiff_t)i0_coarse, (ptrdiff_t)i1_coarse, thr, nfft, out);
    }
    free(S.p); free(S.cs); free(S.ps); free(S.pr); free(S.fbuf); free(S.acc);
    return rc;
}
