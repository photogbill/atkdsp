/* ---- 25. Spread spectrum: the blind detector and the method table ----------
 * docs/ATKDSP_PLAN.md §B6 (the chain, step 1) and docs/DSSS_LITERATURE.md
 * §A1 (Burel), §A2 (CAV). The twin is reference.py §25 (DsssDet).
 *
 * The fluctuation statistic, exactly as the detector computes it:
 *   window m of T = win samples (cut on the global index),
 *   r_m(tau) = sum_{t < T - tau} y(t) conj y(t + tau),  tau = 0..max_lag
 *   rho_m(tau) = |r_m(tau)|^2 / (T - tau)
 *   rho(tau)   = mean over windows of rho_m(tau)
 * For white noise of power s2, r_m(tau) for tau >= 1 is a sum of T - tau
 * zero-mean terms of variance s2^2, so E|r_m|^2 = s2^2 (T - tau) and
 * E rho_m = s2^2 at every lag: a flat floor, whatever the window. The
 * (T - tau) normalisation is what makes it flat (a 1/T^2 normalisation
 * slopes down by (T - tau)/T across the lags and a 1/(T - tau)^2 one slopes
 * up). A signal repeating every P samples with per-period symbols a_k adds
 * E|sum a a* |c|^2|^2 ~ (T - tau)^2 Ps^2 / (T - tau) = (T - tau) Ps^2 at
 * tau = kP — the peaks fall off linearly across the lags, as they must.
 *
 * The floor is read off the lags themselves (median and MAD over
 * period_min..max_lag) rather than from s2: a receiver's own filter colours
 * the first lags and a wrong s2 would make the threshold a guess. Each
 * lag's z-score is (rho - median) / (1.4826 MAD); with nwin windows
 * averaged, rho(tau) at a noise lag is close to Gaussian, and the
 * Bonferroni threshold over the nlags lags tested is z = Phi^-1(1 - pfa /
 * nlags). tests/test_phase6.py measures the rate.
 *
 * CAV (Zeng & Liang, complex form): the same r_m(l) for l < L, averaged over
 * windows with a 1/(T - l) normalisation so each mean r(l) estimates the
 * autocorrelation at lag l; T1 = (1/L) sum_{n,m} |r(|n - m|)| over the L x L
 * Toeplitz covariance = |r(0)| + (2/L) sum_{l>=1} (L - l) |r(l)|; T2 = |r(0)|.
 * For complex white noise over Ns samples, |r(l)|, l >= 1, is Rayleigh with
 * mean s2 sqrt(pi / 4) / sqrt(Ns) = 0.8862 s2 / sqrt(Ns), and r(0) has mean
 * s2 and standard deviation s2 / sqrt(Ns). Following the paper's derivation
 * (T1 at its mean, T2's fluctuation carrying the tail):
 *   gamma = (1 + (L - 1) 0.8862 / sqrt(Ns)) / (1 - z_pfa / sqrt(Ns))
 * — no noise power anywhere in it. The z-score reported is
 *   z = (T1/T2 - 1 - (L - 1) 0.8862 / sqrt(Ns)) sqrt(Ns)
 * and its tail is the CAV Pfa-equivalent used by the fusion.
 *
 * ACF chip estimate: the coherent mean r(tau) (complex, over windows) is the
 * chip waveform's autocorrelation times Ps at small lags: for rectangular
 * chips |r(tau)| = Ps (1 - tau / Tc) for tau < Tc. A line fitted to |r(tau)|
 * over the first decreasing lags gives Ps (the intercept) and Tc (where it
 * reaches zero). Reported only when the fit is clean (R^2 >= 0.95 over at
 * least 3 lags and a negative slope) and flagged ACF; a carrier offset
 * rotates r by e^{j 2 pi df tau / fs}, which |.| ignores. */
#include "internal.h"
#include <stdio.h>

/* ---- the provenance table ------------------------------------------------- */
static const atkdsp_method_info METHODS[ATKDSP_METHOD_COUNT] = {
    {"none", "", "", "", ""},
    {"fluctuation-of-correlation detector", "G. Burel", "2000",
     "SCI 2000 (reproduced: METU thesis; Le Nir & Scheers, RMA 2015)", "FR"},
    {"covariance absolute-value test (CAV)", "Y. Zeng, Y.-C. Liang", "2008",
     "IEEE Trans. Veh. Technol. / arXiv 0808.2562", "SG"},
    {"cyclostationarity test, max statistic over a scan", "J. Lunden, V. Koivunen, A. Huttunen, H. V. Poor",
     "2007", "arXiv 0707.0909", "FI/US"},
    {"wide-window eigenvector of period-long windows", "G. Burel, C. Bouder; (SVD form) Tongxin Xuebao",
     "2000 / 2012", "ICASSP 2000; Journal on Communications 2012", "FR / CN"},
    {"eigenvalue identifiability rule", "N. R. Rao, A. Edelman", "2007", "arXiv 0705.2605", "US"},
    {"carrier offset estimated on the eigenvector", "V. Le Nir, B. Scheers", "2015",
     "Royal Military Academy, Belgium", "BE"},
    {"alternating sign-despread / k-means refinement", "Ye et al.", "2021",
     "Journal of Signal Processing (信号处理)", "CN"},
    {"per-chip vote margin", "ATK", "2026", "atkdsp", "ATK"},
    {"Berlekamp-Massey with the linear-complexity profile", "E. Berlekamp, J. Massey", "1968 / 1969",
     "IEEE Trans. Inf. Theory", "US"},
    {"parity-check counting over primitive polynomials", "N. Mitton; M. Cluzeau; W. Meier, O. Staffelbach",
     "2004 / 2007 / 1989", "DCSSI; INRIA; J. Cryptology", "FR / CH"},
    {"parallel code-phase search with the GLRT threshold", "GNSS-SDR (CTTC); K. Borre et al.", "2007 / 2015",
     "GNSS-SDR documentation; Springer", "ES / DK"},
    {"best-bit-pattern post-detection integration (MBAPDI)", "D. Gomez-Casco, J. A. Lopez-Salcedo, G. Seco-Granados",
     "2018", "IEEE Trans. Aerosp. Electron. Syst. / arXiv 1810.04085", "ES"},
    {"Doppler bin width rule", "B. Geiger, C. Vogel", "2011 / 2014", "arXiv 1107.1972 / IEEE TAES", "DE"},
    {"M2M4 moments C/N0 estimator", "D. R. Pauluzzi, N. C. Beaulieu", "2000", "IEEE Trans. Commun.", "CA"},
    {"narrowband-wideband power ratio C/N0 (NWPR)", "A. J. Van Dierendonck", "1992 / 1996",
     "Global Positioning System: Theory and Applications", "US"},
    {"power-distortion correlation-peak monitor", "K. D. Wesson, J. N. Gross, T. E. Humphreys, B. L. Evans",
     "2017 / 2018", "IEEE TAES / arXiv 1702.06554, 1712.04501", "US"},
    {"three-handle fusion", "ATK", "2026", "atkdsp", "ATK"},
    {"chip length from the autocorrelation triangle", "ATK (textbook)", "2026", "atkdsp", "ATK"},
    {"quadrature split of QPSK chips by the fourth-power phase", "ATK (after the two-source idea of MDPI Information 2023)",
     "2026", "atkdsp", "ATK"},
    {"magnitude-squared coherence test", "classic (Carter 1987)", "1987", "Proc. IEEE", "US"},
    {"phase referencing on a calibrator", "radio astronomy (VLBI)", "—", "classic", "—"},
    {"consistency lattice and common-oscillator test", "ATK", "2026", "atkdsp", "ATK"},
    {"dechirp detection of chirp spread spectrum", "Semtech (LoRa); open descriptions (gr-lora, LoRa-SDR)",
     "2013 / 2017", "patent / open source", "US/CH"},
};

const atkdsp_method_info *atkdsp_method(int method_id) {
    if (method_id < 0 || method_id >= ATKDSP_METHOD_COUNT) return NULL;
    return &METHODS[method_id];
}

/* ---- Gaussian tails ------------------------------------------------------- */
static double log10_q(double z) {            /* log10 P(N(0,1) > z) */
    if (z <= 0.0) return log10(1.0 - 0.5 * atk_igamc(0.5, z * z * 0.5));   /* Q(z) = 1 - Q(-z) */
    return atk_log_igamc(0.5, z * z * 0.5) / log(10.0) + log10(0.5);        /* Q(z) = Q(1/2, z^2/2) / 2 */
}

/* ---- the handle ----------------------------------------------------------- */
struct atkdsp_dsss_det {
    atkdsp_dsss_det_config cfg;
    size_t       nfft;
    atkdsp_fft  *fft;
    atkdsp_cf32 *buf;         /* win samples being filled                      */
    atkdsp_cf32 *X, *P;       /* nfft work                                      */
    int          fill;
    double      *rho;         /* max_lag + 1 running means of |r|^2/(T-tau)     */
    double      *r_re, *r_im; /* max_lag + 1 running means of r/(T-tau)         */
    double      *scratch;     /* max_lag + 1, for the median                     */
    uint64_t     samples, windows, resets;
};

void atkdsp_dsss_det_defaults(atkdsp_dsss_det_config *cfg, double fs, int win, int max_lag) {
    if (!cfg) return;
    memset(cfg, 0, sizeof *cfg);
    cfg->fs = fs; cfg->win = win; cfg->max_lag = max_lag;
    cfg->nwin_min = 16; cfg->handles = 0; cfg->pfa = 1e-3;
    cfg->cav_lags = 8; cfg->min_peaks = 3; cfg->period_min = 4;
}

static int cfg_ok(const atkdsp_dsss_det_config *c) {
    if (!c || !(c->fs > 0.0)) return 0;
    if (c->win < 16 || c->win > (1 << 22)) return 0;
    if (c->max_lag < 2 || c->max_lag > c->win / 2) return 0;
    if (c->nwin_min < 1 || c->nwin_min > (1 << 24)) return 0;
    if (!(c->pfa > 0.0) || !(c->pfa < 0.5)) return 0;
    if (c->cav_lags < 2 || c->cav_lags > 64 || c->cav_lags > c->max_lag) return 0;
    if (c->min_peaks < 1 || c->min_peaks > 64) return 0;
    if (c->period_min < 1 || c->period_min > c->max_lag) return 0;
    return 1;
}

static size_t next_pow2(size_t n) { size_t p = 1; while (p < n) p <<= 1; return p; }

atkdsp_dsss_det *atkdsp_dsss_det_create(const atkdsp_dsss_det_config *cfg) {
    if (!cfg_ok(cfg)) return NULL;
    atkdsp_dsss_det *h = (atkdsp_dsss_det *)calloc(1, sizeof *h);
    if (!h) return NULL;
    h->cfg = *cfg;
    if (h->cfg.handles == 0) h->cfg.handles = ATKDSP_DSSS_H_FLUCT | ATKDSP_DSSS_H_CAV | ATKDSP_DSSS_H_ACF;
    h->nfft = next_pow2((size_t)cfg->win + (size_t)cfg->max_lag);
    h->fft = atkdsp_fft_create(h->nfft);
    h->buf = (atkdsp_cf32 *)calloc((size_t)cfg->win, sizeof *h->buf);
    h->X = (atkdsp_cf32 *)calloc(h->nfft, sizeof *h->X);
    h->P = (atkdsp_cf32 *)calloc(h->nfft, sizeof *h->P);
    h->rho = (double *)calloc((size_t)cfg->max_lag + 1, sizeof *h->rho);
    h->r_re = (double *)calloc((size_t)cfg->max_lag + 1, sizeof *h->r_re);
    h->r_im = (double *)calloc((size_t)cfg->max_lag + 1, sizeof *h->r_im);
    h->scratch = (double *)calloc((size_t)cfg->max_lag + 1, sizeof *h->scratch);
    if (!h->fft || !h->buf || !h->X || !h->P || !h->rho || !h->r_re || !h->r_im || !h->scratch) {
        atkdsp_dsss_det_destroy(h); return NULL;
    }
    return h;
}

void atkdsp_dsss_det_destroy(atkdsp_dsss_det *h) {
    if (!h) return;
    atkdsp_fft_destroy(h->fft);
    free(h->buf); free(h->X); free(h->P); free(h->rho); free(h->r_re); free(h->r_im); free(h->scratch);
    free(h);
}

void atkdsp_dsss_det_reset(atkdsp_dsss_det *h) {
    if (!h) return;
    h->fill = 0; h->samples = 0; h->windows = 0; h->resets++;
    memset(h->rho, 0, ((size_t)h->cfg.max_lag + 1) * sizeof *h->rho);
    memset(h->r_re, 0, ((size_t)h->cfg.max_lag + 1) * sizeof *h->r_re);
    memset(h->r_im, 0, ((size_t)h->cfg.max_lag + 1) * sizeof *h->r_im);
}

/* One full window: its autocorrelation by FFT, folded into the running means. */
static void det_window(atkdsp_dsss_det *h) {
    const int T = h->cfg.win, ML = h->cfg.max_lag;
    memcpy(h->X, h->buf, (size_t)T * sizeof *h->X);
    memset(h->X + T, 0, (h->nfft - (size_t)T) * sizeof *h->X);
    atkdsp_fft_exec(h->fft, h->X, h->P, 0);
    for (size_t k = 0; k < h->nfft; k++) {
        const double re = h->P[k].re, im = h->P[k].im;
        h->P[k].re = (float)(re * re + im * im); h->P[k].im = 0.0f;
    }
    atkdsp_fft_exec(h->fft, h->P, h->X, 1);         /* X[tau] = (1/nfft) sum_k |Y_k|^2 e^{...}: the autocorrelation, already normalised by the inverse FFT's 1/n */
    const double nw = (double)(h->windows + 1);
    for (int tau = 0; tau <= ML; tau++) {
        /* conj because the product X conj X then IFFT gives sum y(t+tau) conj y(t)
           — the conjugate of what the header states; |.|^2 does not care, the
           coherent mean (r, for CAV and the ACF) does. */
        const double re = h->X[tau].re, im = -h->X[tau].im;
        const double norm = 1.0 / (double)(T - tau);
        const double rm = (re * re + im * im) * norm;
        h->rho[tau]  += (rm - h->rho[tau]) / nw;
        h->r_re[tau] += (re * norm - h->r_re[tau]) / nw;
        h->r_im[tau] += (im * norm - h->r_im[tau]) / nw;
    }
    h->windows++;
}

int atkdsp_dsss_det_process(atkdsp_dsss_det *h, const atkdsp_cf32 *x, size_t n) {
    if (!h || (!x && n)) return ATKDSP_E_ARG;
    const int T = h->cfg.win;
    size_t i = 0;
    while (i < n) {
        size_t take = (size_t)(T - h->fill);
        if (take > n - i) take = n - i;
        memcpy(h->buf + h->fill, x + i, take * sizeof *x);
        h->fill += (int)take; i += take;
        if (h->fill == T) { det_window(h); h->fill = 0; }
    }
    h->samples += n;
    return ATKDSP_OK;
}

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* parabolic peak refinement around index i of v (i-1, i, i+1 valid) */
static double parabola(const double *v, int i) {
    const double a = v[i - 1], b = v[i], c = v[i + 1];
    const double d = a - 2.0 * b + c;
    if (!(d < 0.0)) return (double)i;
    double off = 0.5 * (a - c) / d;
    if (off > 0.5) off = 0.5;
    if (off < -0.5) off = -0.5;
    return (double)i + off;
}

double atkdsp_cav_threshold(int lags, double ns, double pfa) {
    if (lags < 2 || !(ns > 1.0) || !(pfa > 0.0) || !(pfa < 0.5)) return 0.0;
    const double z = atk_norm_isf(pfa);
    const double s = sqrt(ns);
    const double den = 1.0 - z / s;
    if (!(den > 0.0)) return HUGE_VAL;
    return (1.0 + (double)(lags - 1) * 0.886226925452758 / s) / den;
}

int atkdsp_dsss_det_read(const atkdsp_dsss_det *h, atkdsp_dsss_det_result *out) {
    if (!h || !out) return ATKDSP_E_ARG;
    memset(out, 0, sizeof *out);
    const atkdsp_dsss_det_config *c = &h->cfg;
    const int ML = c->max_lag, PM = c->period_min;
    out->nwin = (int32_t)h->windows;
    out->method[0] = ATKDSP_METHOD_BUREL_FLUCT; out->method[1] = ATKDSP_METHOD_CAV;
    out->method[2] = ATKDSP_METHOD_ACF_CHIP;    out->method[3] = ATKDSP_METHOD_FUSION;
    out->fluct_log10_pfa = 0.0; out->cav_log10_pfa = 0.0;
    if (h->windows < (uint64_t)c->nwin_min || h->windows < 2) { out->flags |= ATKDSP_DSSS_WARMUP; return ATKDSP_OK; }
    double llr = 0.0;

    /* ---- FLUCT ----
       rho(tau) is the mean over nwin windows of |r_m(tau)|^2/(T-tau); for white
       noise r_m is complex Gaussian, so |r_m|^2/(T-tau) is exponential and rho
       is Gamma(nwin) with mean the per-cell noise power. Hence 2 nwin rho/E ~
       chi^2(2 nwin): the threshold is that chi-square tail, NOT a Gaussian — at
       nwin ~ 16 the Gamma tail is far heavier than Gaussian and a z-threshold
       gives tens of percent false alarms. E is the mean over lags (sparse
       signal peaks shift it by a few percent, slightly conservative); the
       median / MAD are kept for display only. */
    if (c->handles & ATKDSP_DSSS_H_FLUCT) {
        const int nl = ML - PM + 1;
        const double nw = (double)h->windows;
        double mean = 0.0;
        for (int tau = PM; tau <= ML; tau++) mean += h->rho[tau];
        mean /= (double)nl;
        memcpy(h->scratch, h->rho + PM, (size_t)nl * sizeof(double));
        qsort(h->scratch, (size_t)nl, sizeof(double), cmp_double);
        const double med = (nl & 1) ? h->scratch[nl / 2] : 0.5 * (h->scratch[nl / 2 - 1] + h->scratch[nl / 2]);
        for (int i = 0; i < nl; i++) h->scratch[i] = fabs(h->rho[PM + i] - med);
        qsort(h->scratch, (size_t)nl, sizeof(double), cmp_double);
        const double mad = (nl & 1) ? h->scratch[nl / 2] : 0.5 * (h->scratch[nl / 2 - 1] + h->scratch[nl / 2]);
        out->floor_mean = med; out->floor_std = 1.4826 * mad;
        if (!(mean > 0.0)) mean = 1e-300;
        /* the ratio whose chi^2(2 nwin) Bonferroni tail equals pfa: solve
           Q(nwin, nwin*r) = pfa/nl by bisection (atk_log_igamc is ln Q). */
        const double target = log(c->pfa / (double)nl);
        double rlo = 1.0, rhi = 50.0;
        while (atk_log_igamc(nw, nw * rhi) > target && rhi < 1e6) rhi *= 2.0;
        for (int it = 0; it < 80; it++) {
            const double rm = 0.5 * (rlo + rhi);
            if (atk_log_igamc(nw, nw * rm) > target) rlo = rm; else rhi = rm;
        }
        const double thr_ratio = 0.5 * (rlo + rhi);
        /* the strongest lag */
        int best = PM; double rbest = h->rho[PM];
        for (int tau = PM; tau <= ML; tau++) if (h->rho[tau] > rbest) { rbest = h->rho[tau]; best = tau; }
        out->best_lag = best;
        out->peak_ratio = mean > 0.0 ? h->rho[best] / mean : 0.0;
        out->peak_z = out->floor_std > 0.0 ? (h->rho[best] - med) / out->floor_std : 0.0;   /* display */
        {
            double l10 = atk_log_igamc(nw, nw * h->rho[best] / mean) / log(10.0) + log10((double)nl);
            if (l10 > 0.0) l10 = 0.0;
            out->fluct_log10_pfa = l10;
        }
        if (h->rho[best] / mean > thr_ratio) {
            out->flags |= ATKDSP_DSSS_FLUCT | ATKDSP_DSSS_DETECT;
            llr += -out->fluct_log10_pfa * log(10.0);
            /* the period is the FUNDAMENTAL, found among INTERIOR local maxima
               above thr_ratio; the strongest such peak sets the gate, and the
               fundamental must be at least half of it, so a sub-harmonic whose
               own bump is near the noise does not win just because its 2x lands
               on the real peak (and the central chip lobe, which only descends,
               is not an interior maximum). */
            const int plo = PM >= 1 ? PM : 1;
            double rref = 0.0;
            for (int tau = plo; tau < ML; tau++)
                if (h->rho[tau] > h->rho[tau - 1] && h->rho[tau] > h->rho[tau + 1] && h->rho[tau] > rref)
                    rref = h->rho[tau];
            const double rfund = 0.5 * rref;
            const double strong = thr_ratio * mean;
            int tried = 0;
            for (int tau = plo; tau < ML && tried < 8 && rref > strong; tau++) {
                if (h->rho[tau] < strong || h->rho[tau] < rfund) continue;
                if (!(h->rho[tau] > h->rho[tau - 1] && h->rho[tau] > h->rho[tau + 1])) continue;
                tried++;
                const double t1 = (tau > 0 && tau < ML) ? parabola(h->rho, tau) : (double)tau;
                const int K = ML / tau;
                int npk = 1; double sk2 = 1.0, skt = t1;
                for (int k = 2; k <= K; k++) {
                    const int centre = (int)floor(t1 * k + 0.5);
                    const int tol = (int)floor(0.005 * t1 * k) + 1;
                    int found = -1; double bestv = strong;
                    for (int u = centre - tol; u <= centre + tol; u++) {
                        if (u < PM || u > ML) continue;
                        if (h->rho[u] > bestv) { bestv = h->rho[u]; found = u; }
                    }
                    if (found < 0) continue;
                    const double tk = (found > 0 && found < ML) ? parabola(h->rho, found) : (double)found;
                    npk++; sk2 += (double)k * k; skt += (double)k * tk;
                }
                const int need = c->min_peaks < K ? c->min_peaks : K;
                if (npk >= need && npk >= 1) {
                    out->period = skt / sk2;
                    out->period_s = out->period / c->fs;
                    out->npeaks = npk;
                    if (npk >= 2 || need == 1) out->flags |= ATKDSP_DSSS_PERIOD;
                    break;
                }
            }
        }
    }

    /* ---- CAV ---- */
    if (c->handles & ATKDSP_DSSS_H_CAV) {
        const int L = c->cav_lags;
        const double ns = (double)h->windows * (double)c->win;
        const double r0 = fabs(h->r_re[0]);
        double t1 = r0;
        for (int l = 1; l < L; l++)
            t1 += 2.0 * (double)(L - l) / (double)L * sqrt(h->r_re[l] * h->r_re[l] + h->r_im[l] * h->r_im[l]);
        const double stat = r0 > 0.0 ? t1 / r0 : 0.0;
        out->cav_t = stat;
        out->cav_thr = atkdsp_cav_threshold(L, ns, c->pfa);
        const double zc = (stat - 1.0 - (double)(L - 1) * 0.886226925452758 / sqrt(ns)) * sqrt(ns);
        {
            double l10 = log10_q(zc);
            if (l10 > 0.0) l10 = 0.0;
            out->cav_log10_pfa = l10;
        }
        if (stat > out->cav_thr) {
            out->flags |= ATKDSP_DSSS_CAV | ATKDSP_DSSS_DETECT;
            llr += -out->cav_log10_pfa * log(10.0);
        }
    }

    /* ---- ACF chip estimate ---- */
    if (c->handles & ATKDSP_DSSS_H_ACF) {
        /* |r(tau)| over the first decreasing lags */
        const int kmax = ML < 64 ? ML : 64;
        double a[65]; int k = 0;
        for (int tau = 1; tau <= kmax; tau++) {
            a[tau] = sqrt(h->r_re[tau] * h->r_re[tau] + h->r_im[tau] * h->r_im[tau]);
            if (tau >= 2 && a[tau] >= a[tau - 1]) break;
            k = tau;
            if (a[tau] < 0.1 * a[1]) break;
        }
        if (k >= 3) {
            double sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
            for (int tau = 1; tau <= k; tau++) { sx += tau; sy += a[tau]; sxx += (double)tau * tau; sxy += tau * a[tau]; syy += a[tau] * a[tau]; }
            const double n = (double)k;
            const double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
            const double icpt = (sy - slope * sx) / n;
            const double ssr = syy - sy * sy / n;
            double sse = 0.0;
            for (int tau = 1; tau <= k; tau++) { const double e = a[tau] - (icpt + slope * tau); sse += e * e; }
            const double r2 = ssr > 0.0 ? 1.0 - sse / ssr : 0.0;
            if (slope < 0.0 && icpt > 0.0 && r2 >= 0.95) {
                out->tc_acf = -icpt / slope;
                out->ps_acf = icpt;
                out->flags |= ATKDSP_DSSS_ACF;
            }
        }
    }
    out->llr = llr;
    return ATKDSP_OK;
}

ptrdiff_t atkdsp_dsss_det_rho(const atkdsp_dsss_det *h, double *rho, size_t cap) {
    if (!h || !rho) return ATKDSP_E_ARG;
    const size_t n = (size_t)h->cfg.max_lag + 1;
    if (cap < n) return ATKDSP_E_CAP;
    memcpy(rho, h->rho, n * sizeof *rho);
    return (ptrdiff_t)n;
}

int atkdsp_dsss_det_describe(const atkdsp_dsss_det *h, char *buf, size_t cap) {
    if (!h || !buf || !cap) return ATKDSP_E_ARG;
    const atkdsp_dsss_det_config *c = &h->cfg;
    int n = snprintf(buf, cap, "dsss_det: fs %.0f Hz, windows of %d samples (%.4g s), lags %d..%d, "
                     "verdict after %d windows, Pfa %g per handle; handles:%s%s%s; CAV %d lags; "
                     "period needs %d equally spaced peaks; %llu windows in",
                     c->fs, c->win, c->win / c->fs, c->period_min, c->max_lag, c->nwin_min, c->pfa,
                     (c->handles & ATKDSP_DSSS_H_FLUCT) ? " fluctuation(Burel FR 2000)" : "",
                     (c->handles & ATKDSP_DSSS_H_CAV) ? " CAV(Zeng-Liang SG 2008)" : "",
                     (c->handles & ATKDSP_DSSS_H_ACF) ? " ACF-chip(ATK)" : "",
                     c->cav_lags, c->min_peaks, (unsigned long long)h->windows);
    return n < 0 ? ATKDSP_E_ARG : (n >= (int)cap ? ATKDSP_E_CAP : n);
}

int atkdsp_dsss_det_stats(const atkdsp_dsss_det *h, atkdsp_dsss_counters *st) {
    if (!h || !st) return ATKDSP_E_ARG;
    st->samples = h->samples; st->windows = h->windows; st->resets = h->resets;
    st->window_fill = (uint64_t)h->fill;
    return ATKDSP_OK;
}
