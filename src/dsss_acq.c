/* ---- 25e. atkdsp_dsss_acquire / atkdsp_dsss_trk (§B6 step 5) ----------------
 * docs/ATKDSP_PLAN.md §B6 step 5; docs/DSSS_LITERATURE.md §E1, §E2, §E4, §E5.
 * The twin is reference.py §25 (dsss_acquire, DsssTrk).
 *
 * Acquisition: the parallel code-phase search (GNSS-SDR / Borre). Per Doppler
 * bin f, derotate ncoh periods, FFT, multiply by the conjugate FFT of the
 * code replica (one period, zero-padded to ncoh periods), inverse FFT, |.|^2
 * — the correlation at every code phase at once; sum the chosen combiner over
 * nnoncoh dwells. NPDI sums the powers; MBAPDI (ES 2018) takes, per code
 * phase, the best sign pattern over the dwells (max over 2^(d-1) patterns of
 * |sum +- z_k|), which survives the data-bit flips NPDI pays for. Gamma =
 * 2 K S_max / P_in is chi-square(2 d) under white noise, so the threshold is
 * its quantile at 1 - (1 - Pfa)^(1/ncells). The Doppler step defaults to
 * 2/(3 T_coh) (DE 2011). code phase refined by a parabola.
 *
 * Tracking: a local replica per sample from the fractional code phase (the
 * code is held oversampled once) and the carrier NCO; early / prompt / late
 * at +-spacing chips and the quality pair at +-quality_spacing; a
 * non-coherent early-minus-late DLL and an arctan (Costas) PLL, second-order
 * loops, carried across the caller's blocks. One prompt per period is the
 * despread symbol; C/N0 by M2M4 and NWPR over cn0_periods; the quality
 * monitor's distortion |xi(-d)-xi(+d)|/sigma is Rayleigh under a clean peak
 * (US 2017/2018), reported with the received-power track. */
#include "internal.h"
#include <stdio.h>

typedef struct { double re, im; } cd;

double atkdsp_acq_threshold(int dwells, double ncells, double pfa) {
    if (dwells < 1 || !(ncells >= 1.0) || !(pfa > 0.0) || !(pfa < 1.0)) return 0.0;
    const double per_cell = 1.0 - pow(1.0 - pfa, 1.0 / ncells);
    /* chi-square(2 d): P(X > x) = Q(d, x/2); invert by bisection */
    const double a = (double)dwells;
    double lo = 0.0, hi = 2.0 * a + 40.0 * sqrt(2.0 * a) + 40.0;
    for (int it = 0; it < 200; it++) {
        const double mid = 0.5 * (lo + hi);
        if (atk_igamc(a, mid * 0.5) > per_cell) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

void atkdsp_dsss_acq_defaults(atkdsp_dsss_acq_config *cfg, double fs, double rc, int L) {
    if (!cfg) return;
    memset(cfg, 0, sizeof *cfg);
    cfg->fs = fs; cfg->rc = rc; cfg->L = L;
    cfg->ncoh = 1; cfg->nnoncoh = 1; cfg->combiner = ATKDSP_ACQ_NPDI;
    cfg->dop_lo = -5000.0; cfg->dop_hi = 5000.0; cfg->dop_step = 0.0; cfg->pfa = 1e-3;
}

/* the code sampled to Np = round(L*sps) samples, one period, as cf32 */
static int sample_code(const int8_t *code, int L, double sps, atkdsp_cf32 *rep, int Np) {
    for (int i = 0; i < Np; i++) {
        int k = (int)floor((double)i / sps);
        if (k >= L) k = L - 1;
        rep[i].re = (float)code[k]; rep[i].im = 0.0f;
    }
    return Np;
}

int atkdsp_dsss_acquire(const atkdsp_cf32 *iq, size_t n, const int8_t *code,
                        const atkdsp_dsss_acq_config *cfg, atkdsp_dsss_acq *out) {
    if (!iq || !code || !cfg || !out) return ATKDSP_E_ARG;
    memset(out, 0, sizeof *out);
    if (!(cfg->fs > 0.0) || !(cfg->rc > 0.0) || cfg->L < 2) return ATKDSP_E_ARG;
    if (cfg->ncoh < 1 || cfg->ncoh > 64 || cfg->nnoncoh < 1 || cfg->nnoncoh > 64) return ATKDSP_E_ARG;
    if (cfg->combiner == ATKDSP_ACQ_MBAPDI && cfg->nnoncoh > 10) return ATKDSP_E_ARG;
    if (!(cfg->dop_hi > cfg->dop_lo) || !(cfg->pfa > 0.0) || !(cfg->pfa < 1.0)) return ATKDSP_E_ARG;
    const double sps = cfg->fs / cfg->rc;
    const int Np = (int)floor((double)cfg->L * sps + 0.5);
    if (Np < 2) return ATKDSP_E_ARG;
    const int ncoh = cfg->ncoh, nnon = cfg->nnoncoh;
    const size_t need = (size_t)Np * (size_t)ncoh * (size_t)nnon;   /* periods used */
    if (n < need) return ATKDSP_E_EMPTY;
    const double tcoh = (double)Np * ncoh / cfg->fs;                /* coherent time */
    double dstep = cfg->dop_step > 0.0 ? cfg->dop_step : 2.0 / (3.0 * tcoh);
    const int ndop = (int)floor((cfg->dop_hi - cfg->dop_lo) / dstep) + 1;
    if (ndop < 1 || ndop > 100000) return ATKDSP_E_ARG;

    atkdsp_fft *fft = atkdsp_fft_create((size_t)Np);
    atkdsp_cf32 *rep = (atkdsp_cf32 *)calloc((size_t)Np, sizeof *rep);
    atkdsp_cf32 *Rf = (atkdsp_cf32 *)calloc((size_t)Np, sizeof *Rf);
    atkdsp_cf32 *buf = (atkdsp_cf32 *)calloc((size_t)Np, sizeof *buf);
    atkdsp_cf32 *Xf = (atkdsp_cf32 *)calloc((size_t)Np, sizeof *Xf);
    atkdsp_cf32 *corr = (atkdsp_cf32 *)calloc((size_t)Np, sizeof *corr);
    cd *coh = (cd *)calloc((size_t)Np, sizeof *coh);               /* coherent sum over ncoh, one phase line */
    double *acc = (double *)calloc((size_t)Np, sizeof *acc);       /* NPDI accumulator over phases */
    cd *zk = (cd *)calloc((size_t)Np * (size_t)nnon, sizeof *zk);  /* MBAPDI: the nnon coherent dwells */
    if (!fft || !rep || !Rf || !buf || !Xf || !corr || !coh || !acc || !zk) {
        atkdsp_fft_destroy(fft); free(rep); free(Rf); free(buf); free(Xf); free(corr); free(coh); free(acc); free(zk);
        return ATKDSP_E_NOMEM;
    }
    sample_code(code, cfg->L, sps, rep, Np);
    atkdsp_fft_exec(fft, rep, Rf, 0);
    for (int i = 0; i < Np; i++) Rf[i].im = -Rf[i].im;             /* conjugate FFT of one period */

    double pin = 0.0;
    for (size_t i = 0; i < n; i++) pin += (double)iq[i].re * iq[i].re + (double)iq[i].im * iq[i].im;
    pin /= (double)n;

    double gbest = -HUGE_VAL; int dbest = 0; size_t pbest = 0;
    for (int di = 0; di < ndop; di++) {
        const double f = cfg->dop_lo + di * dstep;
        memset(acc, 0, (size_t)Np * sizeof *acc);
        for (int d = 0; d < nnon; d++) {
            memset(coh, 0, (size_t)Np * sizeof *coh);
            for (int cq = 0; cq < ncoh; cq++) {
                const size_t base = ((size_t)d * ncoh + cq) * (size_t)Np;
                const atkdsp_cf32 *seg = iq + base;
                for (int i = 0; i < Np; i++) {
                    const double ph = -2.0 * ATK_PI * f * (double)(base + i) / cfg->fs;
                    const double cr = cos(ph), ci = sin(ph);
                    buf[i].re = (float)(seg[i].re * cr - seg[i].im * ci);
                    buf[i].im = (float)(seg[i].re * ci + seg[i].im * cr);
                }
                atkdsp_fft_exec(fft, buf, Xf, 0);
                for (int i = 0; i < Np; i++) {
                    corr[i].re = Xf[i].re * Rf[i].re - Xf[i].im * Rf[i].im;
                    corr[i].im = Xf[i].re * Rf[i].im + Xf[i].im * Rf[i].re;
                }
                atkdsp_fft_exec(fft, corr, buf, 1);               /* one period's correlation, already /Np */
                for (int i = 0; i < Np; i++) { coh[i].re += buf[i].re; coh[i].im += buf[i].im; }
            }
            if (cfg->combiner == ATKDSP_ACQ_NPDI)
                for (int i = 0; i < Np; i++) acc[i] += coh[i].re * coh[i].re + coh[i].im * coh[i].im;
            else
                for (int i = 0; i < Np; i++) zk[(size_t)d * Np + i] = coh[i];
        }
        if (cfg->combiner == ATKDSP_ACQ_MBAPDI) {
            const int patterns = 1 << (nnon - 1);
            for (int i = 0; i < Np; i++) {
                double best = 0.0;
                for (int mask = 0; mask < patterns; mask++) {
                    double sr = zk[i].re, si = zk[i].im;
                    for (int b = 1; b < nnon; b++) {
                        const double s = (mask >> (b - 1)) & 1 ? -1.0 : 1.0;
                        sr += s * zk[(size_t)b * Np + i].re; si += s * zk[(size_t)b * Np + i].im;
                    }
                    const double mag = sr * sr + si * si;
                    if (mag > best) best = mag;
                }
                acc[i] = best;
            }
        }
        for (int i = 0; i < Np; i++)
            if (acc[i] > gbest) { gbest = acc[i]; dbest = di; pbest = (size_t)i; }
    }
    /* peak-to-second-peak and C/N0 from the winning Doppler's phase line */
    {
        const double f = cfg->dop_lo + dbest * dstep;
        memset(acc, 0, (size_t)Np * sizeof *acc);
        for (int d = 0; d < nnon; d++) {
            memset(coh, 0, (size_t)Np * sizeof *coh);
            for (int cq = 0; cq < ncoh; cq++) {
                const size_t base = ((size_t)d * ncoh + cq) * (size_t)Np;
                const atkdsp_cf32 *seg = iq + base;
                for (int i = 0; i < Np; i++) {
                    const double ph = -2.0 * ATK_PI * f * (double)(base + i) / cfg->fs;
                    const double cr = cos(ph), ci = sin(ph);
                    buf[i].re = (float)(seg[i].re * cr - seg[i].im * ci);
                    buf[i].im = (float)(seg[i].re * ci + seg[i].im * cr);
                }
                atkdsp_fft_exec(fft, buf, Xf, 0);
                for (int i = 0; i < Np; i++) {
                    corr[i].re = Xf[i].re * Rf[i].re - Xf[i].im * Rf[i].im;
                    corr[i].im = Xf[i].re * Rf[i].im + Xf[i].im * Rf[i].re;
                }
                atkdsp_fft_exec(fft, corr, buf, 1);
                for (int i = 0; i < Np; i++) { coh[i].re += buf[i].re; coh[i].im += buf[i].im; }
            }
            for (int i = 0; i < Np; i++) acc[i] += coh[i].re * coh[i].re + coh[i].im * coh[i].im;
        }
        const int guard = (int)ceil(sps);
        double s2 = 0.0, cellmean = 0.0;
        for (int i = 0; i < Np; i++) {
            long dd = (long)i - (long)pbest; if (dd < 0) dd = -dd;
            if (dd <= guard || (long)Np - dd <= guard) { cellmean += acc[i]; continue; }
            if (acc[i] > s2) s2 = acc[i];
            cellmean += acc[i];
        }
        cellmean /= (double)Np;
        out->peak_ratio = s2 > 0.0 ? acc[pbest] / s2 : HUGE_VAL;
        out->cn0_dbhz = cellmean > 0.0 ? 10.0 * log10((acc[pbest] / cellmean - 1.0 > 0 ? acc[pbest] / cellmean - 1.0 : 1e-9) / tcoh) : 0.0;
        const size_t im1 = (pbest + Np - 1) % Np, ip1 = (pbest + 1) % Np;
        double offs = 0.0;
        const double aa = acc[im1], bb = acc[pbest], cc = acc[ip1], den = aa - 2 * bb + cc;
        if (den < 0.0) offs = 0.5 * (aa - cc) / den;
        out->code_phase = fmod(((double)pbest + offs) / sps, (double)cfg->L);
        if (out->code_phase < 0) out->code_phase += cfg->L;
    }
    out->doppler_hz = cfg->dop_lo + dbest * dstep;
    out->ndop = ndop;
    out->ncells = (int)((size_t)Np * (size_t)ndop);
    out->gamma = pin > 0.0 ? 2.0 * (double)Np * gbest / pin : 0.0;
    out->gamma_thr = atkdsp_acq_threshold(cfg->nnoncoh, (double)out->ncells, cfg->pfa);
    {
        /* log10 Pfa of the global peak: per-cell Q(d, gamma/2), times ncells */
        const double q = atk_log_igamc((double)cfg->nnoncoh, out->gamma * 0.5) / log(10.0) + log10((double)out->ncells);
        out->log10_pfa = q > 0.0 ? 0.0 : q;
    }
    if (out->gamma > out->gamma_thr) out->flags |= ATKDSP_ACQ_DETECT;
    out->method[0] = ATKDSP_METHOD_PCPS;
    out->method[1] = cfg->combiner == ATKDSP_ACQ_MBAPDI ? ATKDSP_METHOD_MBAPDI : ATKDSP_METHOD_PCPS;
    out->method[2] = ATKDSP_METHOD_DOPPLER_GRID;
    atkdsp_fft_destroy(fft); free(rep); free(Rf); free(buf); free(Xf); free(corr); free(coh); free(acc); free(zk);
    return ATKDSP_OK;
}

/* ---- tracking ------------------------------------------------------------------ */
struct atkdsp_dsss_trk {
    atkdsp_dsss_trk_config cfg;
    int8_t  *code;             /* L chips                                         */
    double   code_phase;       /* chips                                           */
    double   code_rate;        /* chips per sample = rc/fs x (1 + doppler/fc?) -> rc/fs here */
    double   carrier_phase;    /* radians                                         */
    double   carrier_rate;     /* radians per sample                              */
    double   code_hz;          /* code-rate error estimate (chips/s)             */
    double   pll_integ;        /* carrier frequency integrator (rad/s)            */
    double   wn_pll, dll_c1, Tupd;
    /* per-period correlator accumulators */
    cd       E, P, Lx, Q_e, Q_l;   /* early, prompt, late, quality early/late     */
    double   chip_in_period;   /* chips accumulated this period                   */
    /* C/N0 */
    double  *pr_hist;          /* |P|^2 ring over cn0_periods                     */
    cd      *p_hist;           /* P ring                                          */
    int      cn0_fill, cn0_head;
    double   power, power_start;
    uint64_t periods, samples, resets, dropped;
};

void atkdsp_dsss_trk_defaults(atkdsp_dsss_trk_config *cfg, double fs, double rc, int L,
                              double code_phase, double doppler_hz) {
    if (!cfg) return;
    memset(cfg, 0, sizeof *cfg);
    cfg->fs = fs; cfg->rc = rc; cfg->L = L; cfg->code_phase = code_phase; cfg->doppler_hz = doppler_hz;
    cfg->dll_bw_hz = 2.0; cfg->pll_bw_hz = 15.0; cfg->spacing = 0.5; cfg->quality_spacing = 0.15;
    cfg->cn0_periods = 20; cfg->power_window_s = 0.2; cfg->distortion_pfa = 1e-3; cfg->power_high_db = 3.0;
}

atkdsp_dsss_trk *atkdsp_dsss_trk_create(const int8_t *code, const atkdsp_dsss_trk_config *cfg) {
    if (!code || !cfg || !(cfg->fs > 0.0) || !(cfg->rc > 0.0) || cfg->L < 2) return NULL;
    if (!(cfg->spacing > 0.0) || cfg->spacing >= 1.0 || !(cfg->quality_spacing > 0.0)) return NULL;
    if (cfg->cn0_periods < 2 || cfg->cn0_periods > 4096) return NULL;
    atkdsp_dsss_trk *h = (atkdsp_dsss_trk *)calloc(1, sizeof *h);
    if (!h) return NULL;
    h->cfg = *cfg;
    h->code = (int8_t *)malloc((size_t)cfg->L);
    h->pr_hist = (double *)calloc((size_t)cfg->cn0_periods, sizeof *h->pr_hist);
    h->p_hist = (cd *)calloc((size_t)cfg->cn0_periods, sizeof *h->p_hist);
    if (!h->code || !h->pr_hist || !h->p_hist) { atkdsp_dsss_trk_destroy(h); return NULL; }
    memcpy(h->code, code, (size_t)cfg->L);
    h->code_phase = cfg->code_phase;
    h->code_rate = cfg->rc / cfg->fs;
    h->carrier_rate = 2.0 * ATK_PI * cfg->doppler_hz / cfg->fs;
    h->pll_integ = 2.0 * ATK_PI * cfg->doppler_hz;              /* start the integrator at the given Doppler (rad/s) */
    const double T = (double)cfg->L / cfg->rc;                  /* the update interval = one code period */
    h->Tupd = T;
    /* 2nd-order PLL: a velocity integrator plus a proportional term, both in
       rad/s; wn in rad/s with Bn (Hz) = 0.53 wn / (2 pi), so wn = 2 pi Bn / 0.53. */
    h->wn_pll = 2.0 * ATK_PI * cfg->pll_bw_hz / 0.53;
    /* 1st-order DLL: Bn = wn/4, wn (1/s) = 4 Bn; code_hz += wn * dll (chips/s). */
    h->dll_c1 = 4.0 * cfg->dll_bw_hz;
    return h;
}

void atkdsp_dsss_trk_destroy(atkdsp_dsss_trk *h) {
    if (!h) return;
    free(h->code); free(h->pr_hist); free(h->p_hist); free(h);
}

static int8_t code_at(const atkdsp_dsss_trk *h, double chip) {
    int k = (int)floor(chip);
    k %= h->cfg.L; if (k < 0) k += h->cfg.L;
    return h->code[k];
}

static double cn0_m2m4(const double *pr, const cd *p, int n) {
    /* M2 = mean |P|^2; M4 = mean |P|^4; for A^2 signal + noise:
       Pd = sqrt(2 M2^2 - M4), Pn = M2 - Pd; C/N0 = Pd/Pn / T is caller's. */
    double m2 = 0.0, m4 = 0.0;
    for (int i = 0; i < n; i++) { m2 += pr[i]; m4 += pr[i] * pr[i]; (void)p; }
    m2 /= n; m4 /= n;
    const double disc = 2.0 * m2 * m2 - m4;
    const double pd = disc > 0.0 ? sqrt(disc) : 0.0;
    const double pn = m2 - pd;
    return pn > 0.0 ? pd / pn : 1e9;
}

ptrdiff_t atkdsp_dsss_trk_process(atkdsp_dsss_trk *h, const atkdsp_cf32 *x, size_t n,
                                  atkdsp_cf32 *prompt, size_t cap) {
    if (!h || (!x && n)) return ATKDSP_E_ARG;
    size_t got = 0;
    const double sp = h->cfg.spacing, qs = h->cfg.quality_spacing;
    for (size_t i = 0; i < n; i++) {
        /* carrier wipe-off */
        const double cr = cos(-h->carrier_phase), ci = sin(-h->carrier_phase);
        const double xr = x[i].re * cr - x[i].im * ci, xi = x[i].re * ci + x[i].im * cr;
        /* the five correlators at the current code phase */
        const double cp = h->code_phase;
        const double ce = (double)code_at(h, cp - sp), cpr = (double)code_at(h, cp),
                     cl = (double)code_at(h, cp + sp),
                     cqe = (double)code_at(h, cp - qs), cql = (double)code_at(h, cp + qs);
        h->E.re += xr * ce; h->E.im += xi * ce;
        h->P.re += xr * cpr; h->P.im += xi * cpr;
        h->Lx.re += xr * cl; h->Lx.im += xi * cl;
        h->Q_e.re += xr * cqe; h->Q_e.im += xi * cqe;
        h->Q_l.re += xr * cql; h->Q_l.im += xi * cql;
        h->power += (double)x[i].re * x[i].re + (double)x[i].im * x[i].im;
        /* advance */
        h->carrier_phase += h->carrier_rate;
        h->code_phase += h->code_rate;
        h->chip_in_period += h->code_rate;
        if (h->chip_in_period >= (double)h->cfg.L) {
            /* close the period: discriminators and loop updates */
            const double e = sqrt(h->E.re * h->E.re + h->E.im * h->E.im);
            const double l = sqrt(h->Lx.re * h->Lx.re + h->Lx.im * h->Lx.im);
            const double den = e + l;
            const double dll = den > 0.0 ? 0.5 * (e - l) / den : 0.0;       /* chips */
            h->code_hz += h->dll_c1 * dll * h->Tupd;                        /* chips/s: pull the replica toward the signal */
            h->code_rate = (h->cfg.rc + h->code_hz) / h->cfg.fs;
            const double pll = h->P.re != 0.0 ? atan(h->P.im / h->P.re) : 0.0;  /* Costas 2-quadrant: data-bit (180 deg) insensitive */
            h->pll_integ += h->wn_pll * h->wn_pll * pll * h->Tupd;          /* rad/s velocity integrator */
            const double freq = 1.414213562373095 * h->wn_pll * pll + h->pll_integ;  /* rad/s */
            h->carrier_rate = freq / h->cfg.fs;
            /* output */
            if (got < cap) { prompt[got].re = (float)h->P.re; prompt[got].im = (float)h->P.im; }
            else h->dropped++;
            got++;
            /* C/N0 history */
            const double pr2 = h->P.re * h->P.re + h->P.im * h->P.im;
            h->pr_hist[h->cn0_head] = pr2; h->p_hist[h->cn0_head] = h->P;
            h->cn0_head = (h->cn0_head + 1) % h->cfg.cn0_periods;
            if (h->cn0_fill < h->cfg.cn0_periods) h->cn0_fill++;
            if (h->periods == 0) h->power_start = h->power;
            h->periods++;
            h->E.re = h->E.im = h->P.re = h->P.im = h->Lx.re = h->Lx.im = 0.0;
            h->Q_e.re = h->Q_e.im = h->Q_l.re = h->Q_l.im = 0.0;
            h->chip_in_period -= (double)h->cfg.L;
            h->power = 0.0;
        }
    }
    h->samples += n;
    return (ptrdiff_t)got;
}

int atkdsp_dsss_trk_read(const atkdsp_dsss_trk *h, atkdsp_dsss_trk_state *st) {
    if (!h || !st) return ATKDSP_E_ARG;
    memset(st, 0, sizeof *st);
    st->code_phase = fmod(h->code_phase, (double)h->cfg.L);
    st->doppler_hz = h->carrier_rate * h->cfg.fs / (2.0 * ATK_PI);
    st->periods = h->periods;
    const double T = (double)h->cfg.L / h->cfg.rc;
    if (h->cn0_fill >= 2) {
        const double snr = cn0_m2m4(h->pr_hist, h->p_hist, h->cn0_fill);
        st->cn0_m2m4 = 10.0 * log10((snr > 0 ? snr : 1e-9) / T);
        /* NWPR: narrowband/wideband power ratio over the window */
        double nbp_r = 0, nbp_i = 0, wbp = 0;
        for (int i = 0; i < h->cn0_fill; i++) { nbp_r += h->p_hist[i].re; nbp_i += h->p_hist[i].im; wbp += h->pr_hist[i]; }
        const double mu = wbp > 0 ? (nbp_r * nbp_r + nbp_i * nbp_i) / wbp : 0.0;
        const double ratio = (h->cn0_fill - mu) > 0 ? (mu - 1.0) / (h->cn0_fill - mu) : 0.0;
        st->cn0_nwpr = 10.0 * log10((ratio > 0 ? ratio : 1e-9) / T);
        if (st->cn0_m2m4 >= 25.0) st->flags |= ATKDSP_TRK_VALID_CN0;
        /* lock: |mean I| over mean |P| */
        double si = 0, sa = 0;
        for (int i = 0; i < h->cn0_fill; i++) { si += h->p_hist[i].re; sa += sqrt(h->pr_hist[i]); }
        st->lock = sa > 0 ? fabs(si) / sa : 0.0;
        if (st->lock > 0.5) st->flags |= ATKDSP_TRK_LOCK;
    }
    st->method[0] = ATKDSP_METHOD_M2M4; st->method[1] = ATKDSP_METHOD_NWPR; st->method[2] = ATKDSP_METHOD_PD_MONITOR;
    return ATKDSP_OK;
}

int atkdsp_dsss_trk_describe(const atkdsp_dsss_trk *h, char *buf, size_t cap) {
    if (!h || !buf || !cap) return ATKDSP_E_ARG;
    int n = snprintf(buf, cap, "dsss_trk: fs %.0f Hz, %d chips at %.6g chip/s, DLL %.3g Hz / PLL %.3g Hz, "
                     "E-L +-%.2f chip, quality +-%.2f chip, C/N0 over %d periods; %llu periods",
                     h->cfg.fs, h->cfg.L, h->cfg.rc, h->cfg.dll_bw_hz, h->cfg.pll_bw_hz, h->cfg.spacing,
                     h->cfg.quality_spacing, h->cfg.cn0_periods, (unsigned long long)h->periods);
    return n < 0 ? ATKDSP_E_ARG : (n >= (int)cap ? ATKDSP_E_CAP : n);
}

int atkdsp_dsss_trk_stats(const atkdsp_dsss_trk *h, atkdsp_dsss_counters *st) {
    if (!h || !st) return ATKDSP_E_ARG;
    st->samples = h->samples; st->windows = h->periods; st->resets = h->resets; st->window_fill = h->dropped;
    return ATKDSP_OK;
}
