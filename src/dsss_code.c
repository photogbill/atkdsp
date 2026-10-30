/* ---- 25d. atkdsp_dsss_code: the spreading code from a capture (§B6 step 4) ----
 * docs/ATKDSP_PLAN.md §B6 step 4; docs/DSSS_LITERATURE.md §C1-C5, §D1, §A3.
 * The twin is reference.py §25 (dsss_code). The order of operations, each
 * with its provenance:
 *
 *  1. Wide windows of 2N samples at hop N (N = the period in samples). The
 *     principal eigenvector of their covariance by power iteration ON THE
 *     DATA — v <- (1/M) sum_m y_m (y_m^H v) — never forming the 2N x 2N
 *     matrix; a second vector by deflation. For a code with one symbol per
 *     period the window holds a_m (a tail piece), a_{m+1} (a full period)
 *     and a_{m+2} (a head piece), orthogonal supports, so the principal
 *     vector is the full period and its support [t0, t0 + N) IS the code
 *     start — the wide window's gift (FR 2000 / CN 2012). A carrier offset
 *     only rotates the vector elementwise (the theorem of BE 2015), so this
 *     stage needs no offset removed.
 *     Degeneracy: when t0 = 0 the two halves are both full periods and the
 *     top two eigenvalues tie with DIFFERENT supports — the windows are
 *     re-cut N/2 later. QPSK chips tie with the SAME support (two codes of
 *     equal energy), which is the normal case there.
 *  2. Identifiability (US 2007): with m = 2N and n = M the noise bulk's
 *     edge is sigma^2 (1 + sqrt(m/n))^2 (Marchenko-Pastur, complex); the
 *     principal eigenvalue must clear it, or the vector is noise and the
 *     call refuses — with the number of periods that would clear it at the
 *     signal power the excess implies.
 *  3. Aligned periods y_m = x[t0 + mN .. +N). The carrier offset from the
 *     period-averaged P-th power (P = 2 BPSK, 4 QPSK): q(i) = sum_m y_m(i)^P
 *     is one tone at P x alpha cycles per sample at the chip centres (the
 *     symbols' signs vanish under the power), and Kay's weighted phase
 *     estimate reads it; the data are derotated. Then the constellation
 *     phase theta from the same power's argument (sum over chip centres:
 *     for BPSK c^2 e^{2j theta} > 0 real; for QPSK (a c_I + j b c_Q)^4 with
 *     |c| = 1 is -4 e^{4j theta}), and the data are rotated real.
 *  4. The code waveform by REAL power iteration on Re(y_m) (and Im(y_m)
 *     for the Q code), then vote_passes of the decision-directed step —
 *     a_m = sign(y_m . c), c <- sum_m a_m y_m (CN 2021) — and the chips are
 *     the signs at the chip instants t0 + (k + 1/2) sps. Each chip's MARGIN
 *     is |c(i_k)| over the noise of a mean of M real samples (ATK).
 *  5. The generator: Berlekamp-Massey (§15) on the chips with its profile;
 *     accepted when the complexity r satisfies 2r <= L/2 and the profile sat
 *     at r for the last half (a wrong chip makes the profile jump — the
 *     tell). Named: Barker (the table, any cyclic shift and sign);
 *     m-sequence when L = 2^r - 1 and the polynomial is primitive; Gold when
 *     the complexity is 2r', L = 2^r' - 1 and the polynomial factors into
 *     two primitive polynomials of degree r'. When BM is not believed and
 *     L + 1 is a power of two, parity-check counting over every primitive
 *     polynomial of that degree (FR 2004 / CH 1989): Z > k sqrt(L), k =
 *     Phi^-1(1 - pfa / candidates) — the polynomial through chip errors. */
#include "internal.h"
#include <stdio.h>

typedef struct { double re, im; } cd;

/* ---- power iteration on windows of the data ---------------------------------- */
/* windows: w_m = x + off + m*hop, length len; v in/out (unit norm); returns the
 * Rayleigh quotient. deflate against d (unit) when given. */
static double power_iter(const atkdsp_cf32 *x, size_t off, size_t hop, size_t len, int M,
                         cd *v, cd *u, const cd *d, int iters) {
    double lam = 0.0;
    for (int it = 0; it < iters; it++) {
        memset(u, 0, len * sizeof *u);
        for (int m = 0; m < M; m++) {
            const atkdsp_cf32 *y = x + off + (size_t)m * hop;
            double pr = 0.0, pi = 0.0;                      /* y^H v */
            for (size_t i = 0; i < len; i++) {
                pr += (double)y[i].re * v[i].re + (double)y[i].im * v[i].im;
                pi += (double)y[i].re * v[i].im - (double)y[i].im * v[i].re;
            }
            for (size_t i = 0; i < len; i++) {
                u[i].re += (double)y[i].re * pr - (double)y[i].im * pi;
                u[i].im += (double)y[i].re * pi + (double)y[i].im * pr;
            }
        }
        for (size_t i = 0; i < len; i++) { u[i].re /= (double)M; u[i].im /= (double)M; }
        if (d) {                                            /* u -= d (d^H u) */
            double pr = 0.0, pi = 0.0;
            for (size_t i = 0; i < len; i++) {
                pr += d[i].re * u[i].re + d[i].im * u[i].im;
                pi += d[i].re * u[i].im - d[i].im * u[i].re;
            }
            for (size_t i = 0; i < len; i++) {
                u[i].re -= d[i].re * pr - d[i].im * pi;
                u[i].im -= d[i].re * pi + d[i].im * pr;
            }
        }
        double nrm = 0.0, dot = 0.0;
        for (size_t i = 0; i < len; i++) {
            nrm += u[i].re * u[i].re + u[i].im * u[i].im;
            dot += v[i].re * u[i].re + v[i].im * u[i].im;
        }
        nrm = sqrt(nrm);
        if (!(nrm > 0.0)) { lam = 0.0; break; }
        const double newlam = nrm;                          /* |R v| for a unit v ~ eigenvalue */
        for (size_t i = 0; i < len; i++) { v[i].re = u[i].re / nrm; v[i].im = u[i].im / nrm; }
        const int converged = fabs(newlam - lam) <= 1e-7 * newlam && fabs(dot / nrm) > 1.0 - 1e-9;
        lam = newlam;
        if (converged) break;
    }
    return lam;
}

static size_t support_start(const cd *v, size_t len, size_t N) {
    /* argmax_k sum_{i=k}^{k+N-1} |v_i|^2, k = 0..len-N */
    double e = 0.0;
    for (size_t i = 0; i < N; i++) e += v[i].re * v[i].re + v[i].im * v[i].im;
    double best = e; size_t kb = 0;
    for (size_t k = 1; k + N <= len; k++) {
        e += v[k + N - 1].re * v[k + N - 1].re + v[k + N - 1].im * v[k + N - 1].im
           - v[k - 1].re * v[k - 1].re - v[k - 1].im * v[k - 1].im;
        if (e > best) { best = e; kb = k; }
    }
    return kb;
}

/* Kay's weighted phase average on a tone q (len values): cycles per sample. */
static double kay_freq(const cd *q, size_t len) {
    if (len < 3) return 0.0;
    const double Nn = (double)len;
    double sum = 0.0;
    for (size_t i = 0; i + 1 < len; i++) {
        const double pr = q[i + 1].re * q[i].re + q[i + 1].im * q[i].im;
        const double pi = q[i + 1].im * q[i].re - q[i + 1].re * q[i].im;
        const double t = (double)i - (Nn / 2.0 - 1.0);
        const double w = 1.5 * Nn / (Nn * Nn - 1.0) * (1.0 - (t / (Nn / 2.0)) * (t / (Nn / 2.0)));
        sum += w * atan2(pi, pr);
    }
    return sum / (2.0 * ATK_PI);
}

/* ---- real power iteration and the vote ---------------------------------------- */
/* yr: M aligned periods of real samples (stride N). c (N) in/out unit; a_m out. */
static void real_power(const double *yr, int M, size_t N, double *c, double *u, int iters) {
    for (int it = 0; it < iters; it++) {
        memset(u, 0, N * sizeof *u);
        for (int m = 0; m < M; m++) {
            const double *y = yr + (size_t)m * N;
            double p = 0.0;
            for (size_t i = 0; i < N; i++) p += y[i] * c[i];
            for (size_t i = 0; i < N; i++) u[i] += y[i] * p;
        }
        double nrm = 0.0;
        for (size_t i = 0; i < N; i++) nrm += u[i] * u[i];
        nrm = sqrt(nrm);
        if (!(nrm > 0.0)) return;
        for (size_t i = 0; i < N; i++) c[i] = u[i] / nrm;
    }
}
/* decision-directed passes; c comes out as the MEAN of the signed periods
 * (so its amplitude is the chip amplitude); returns the noise variance of a
 * real sample from the residual. */
static double vote(const double *yr, int M, size_t N, double *c, double *u, int passes, int8_t *a) {
    for (int p = 0; p < passes; p++) {
        memset(u, 0, N * sizeof *u);
        for (int m = 0; m < M; m++) {
            const double *y = yr + (size_t)m * N;
            double s = 0.0;
            for (size_t i = 0; i < N; i++) s += y[i] * c[i];
            a[m] = (int8_t)(s >= 0.0 ? 1 : -1);
            for (size_t i = 0; i < N; i++) u[i] += (double)a[m] * y[i];
        }
        for (size_t i = 0; i < N; i++) c[i] = u[i] / (double)M;
    }
    double res = 0.0;
    for (int m = 0; m < M; m++) {
        const double *y = yr + (size_t)m * N;
        for (size_t i = 0; i < N; i++) { const double e = y[i] - (double)a[m] * c[i]; res += e * e; }
    }
    return res / ((double)M * (double)N);
}

/* ---- naming ------------------------------------------------------------------ */
static int bm_poly_to_mask(const uint8_t *poly, int lc, uint64_t *mask) {
    if (lc < 1 || lc > 63) return 0;
    uint64_t m = 1ull << lc;
    for (int i = 0; i < lc; i++) if (poly[lc - i]) m |= 1ull << i;
    *mask = m;
    return 1;
}
static int is_pow2m1(int L, int *r) {
    for (int k = 2; k <= 16; k++) if (L == (1 << k) - 1) { *r = k; return 1; }
    return 0;
}
static int mask_primitive(uint64_t mask, int r) {
    uint64_t polys[4096];
    const ptrdiff_t n = atkdsp_primitive_polys(r, polys, 4096);
    for (ptrdiff_t i = 0; i < n && i < 4096; i++) if (polys[i] == mask) return 1;
    return 0;
}
/* does f (degree 2r) factor into two primitive polynomials of degree r? */
static int gold_factor(uint64_t f, int r, uint64_t *f1, uint64_t *f2) {
    uint64_t polys[4096];
    const ptrdiff_t n = atkdsp_primitive_polys(r, polys, 4096);
    for (ptrdiff_t i = 0; i < n && i < 4096; i++) {
        /* polynomial long division f / polys[i] over GF(2) */
        uint64_t rem = f, q = 0;
        const uint64_t d = polys[i];
        for (int k = 2 * r; k >= r; k--) {
            if ((rem >> k) & 1ull) { rem ^= d << (k - r); q |= 1ull << (k - r); }
        }
        if (rem == 0 && mask_primitive(q, r)) { *f1 = d; *f2 = q; return 1; }
    }
    return 0;
}
static void poly_string(uint64_t mask, int r, char *buf, size_t cap) {
    size_t n = 0;
    for (int i = r; i >= 0; i--) {
        if (!((mask >> i) & 1ull)) continue;
        int k;
        if (i == 0) k = snprintf(buf + n, cap > n ? cap - n : 0, "%s1", n ? "+" : "");
        else if (i == 1) k = snprintf(buf + n, cap > n ? cap - n : 0, "%sx", n ? "+" : "");
        else k = snprintf(buf + n, cap > n ? cap - n : 0, "%sx^%d", n ? "+" : "", i);
        if (k < 0) break;
        n += (size_t)k;
    }
}
static void set_name(atkdsp_dsss_code_result *out, const char *tmp) {
    size_t n = strlen(tmp);
    if (n >= sizeof out->name) n = sizeof out->name - 1;
    memcpy(out->name, tmp, n); out->name[n] = 0;
}
static int barker_match(const int8_t *chips, int L) {
    int8_t b[13];
    if (atkdsp_barker(L, b, 13) != L) return 0;
    for (int sgn = 1; sgn >= -1; sgn -= 2)
        for (int sh = 0; sh < L; sh++) {
            int ok = 1;
            for (int i = 0; i < L && ok; i++) ok = chips[(i + sh) % L] == (int8_t)(sgn * b[i]);
            if (ok) return 1;
        }
    return 0;
}

void atkdsp_dsss_code_defaults(atkdsp_dsss_code_config *cfg, double fs, double rc, double period_s) {
    if (!cfg) return;
    memset(cfg, 0, sizeof *cfg);
    cfg->fs = fs; cfg->rc = rc; cfg->period_s = period_s;
    cfg->chip_mod = ATKDSP_CHIP_BPSK; cfg->vote_passes = 6; cfg->poly_max_degree = 14;
    cfg->pfa = 1e-3;
}

ptrdiff_t atkdsp_dsss_code(const atkdsp_cf32 *iq, size_t n, const atkdsp_dsss_code_config *cfg,
                           int8_t *chips, size_t cap, atkdsp_dsss_code_result *out) {
    if (!iq || !cfg || !chips || !out) return ATKDSP_E_ARG;
    memset(out, 0, sizeof *out);
    if (!(cfg->fs > 0.0) || !(cfg->rc > 0.0) || !(cfg->period_s > 0.0)) return ATKDSP_E_ARG;
    if (cfg->chip_mod < 0 || cfg->chip_mod > 2) return ATKDSP_E_ARG;
    if (cfg->vote_passes < 0 || cfg->vote_passes > 64) return ATKDSP_E_ARG;
    if (cfg->poly_max_degree < 3 || cfg->poly_max_degree > 16) return ATKDSP_E_ARG;
    if (!(cfg->pfa > 0.0) || !(cfg->pfa < 0.5)) return ATKDSP_E_ARG;
    const double Nd = cfg->period_s * cfg->fs;
    const size_t N = (size_t)floor(Nd + 0.5);
    if (N < 2 || fabs(Nd - (double)N) > 0.02) return ATKDSP_E_ARG;
    const double sps = cfg->fs / cfg->rc;
    const int L = (int)floor((double)N / sps + 0.5);
    if (L < 2 || sps < 1.0 - 1e-9) return ATKDSP_E_ARG;
    const int two = cfg->chip_mod != ATKDSP_CHIP_BPSK;
    if (cap < (size_t)L * (two ? 2u : 1u)) return ATKDSP_E_CAP;
    if (n < 4 * N) return ATKDSP_E_EMPTY;
    out->L = L;
    int nm = 0;
    out->method[nm++] = ATKDSP_METHOD_EIGEN_WIDE;
    out->method[nm++] = ATKDSP_METHOD_RAO_EDELMAN;

    /* ---- 1. the wide-window eigenvector --------------------------------------- */
    const size_t W = 2 * N;
    cd *v1 = (cd *)calloc(W, sizeof *v1), *v2 = (cd *)calloc(W, sizeof *v2), *u = (cd *)calloc(W, sizeof *u);
    if (!v1 || !v2 || !u) { free(v1); free(v2); free(u); return ATKDSP_E_NOMEM; }
    size_t off = 0, t0 = 0;
    double lam1 = 0.0, lam2 = 0.0, trace = 0.0;
    int M = 0;
    for (int attempt = 0; attempt < 2; attempt++) {
        M = (int)((n - off) / N) - 1;
        if (M < 3) { free(v1); free(v2); free(u); return ATKDSP_E_EMPTY; }
        /* start from the first window (a signal-bearing guess), normalised */
        double nrm = 0.0;
        for (size_t i = 0; i < W; i++) {
            v1[i].re = iq[off + i].re; v1[i].im = iq[off + i].im;
            nrm += v1[i].re * v1[i].re + v1[i].im * v1[i].im;
        }
        nrm = sqrt(nrm); if (!(nrm > 0.0)) nrm = 1.0;
        for (size_t i = 0; i < W; i++) { v1[i].re /= nrm; v1[i].im /= nrm; }
        lam1 = power_iter(iq, off, N, W, M, v1, u, NULL, 60);
        /* second: start from a window orthogonalised against v1 */
        {
            double pr = 0.0, pi = 0.0;
            for (size_t i = 0; i < W; i++) {
                v2[i].re = iq[off + N + i].re; v2[i].im = iq[off + N + i].im;
                pr += v1[i].re * v2[i].re + v1[i].im * v2[i].im;
                pi += v1[i].re * v2[i].im - v1[i].im * v2[i].re;
            }
            nrm = 0.0;
            for (size_t i = 0; i < W; i++) {
                v2[i].re -= v1[i].re * pr - v1[i].im * pi;
                v2[i].im -= v1[i].re * pi + v1[i].im * pr;
                nrm += v2[i].re * v2[i].re + v2[i].im * v2[i].im;
            }
            nrm = sqrt(nrm); if (!(nrm > 0.0)) nrm = 1.0;
            for (size_t i = 0; i < W; i++) { v2[i].re /= nrm; v2[i].im /= nrm; }
        }
        lam2 = power_iter(iq, off, N, W, M, v2, u, v1, 60);
        trace = 0.0;
        for (int m = 0; m < M; m++) {
            const atkdsp_cf32 *y = iq + off + (size_t)m * N;
            for (size_t i = 0; i < W; i++) trace += (double)y[i].re * y[i].re + (double)y[i].im * y[i].im;
        }
        trace /= (double)M;
        t0 = support_start(v1, W, N);
        if (attempt == 0 && lam2 > 0.7 * lam1) {
            const size_t t1 = support_start(v2, W, N);
            const double dd = fabs((double)t1 - (double)t0);
            if (dd > 0.5 * (double)N) { off = N / 2; continue; }   /* the window-boundary tie */
        }
        break;
    }
    const double m_dim = (double)W, n_dim = (double)M;
    double sig2 = (trace - lam1 - lam2) / (m_dim - 2.0);
    if (!(sig2 > 0.0)) sig2 = 1e-30;
    const double edge = sig2 * (1.0 + sqrt(m_dim / n_dim)) * (1.0 + sqrt(m_dim / n_dim));
    out->lambda1 = lam1; out->lambda2 = lam2; out->noise_var = sig2;
    out->ident_margin = lam1 / edge;
    out->nperiods = M;
    out->start = (int32_t)(off + t0);
    {
        const double ps_n = lam1 - edge;                   /* a lower bound on Ps N */
        if (ps_n > 0.0) {
            const double need = m_dim * (sig2 / ps_n) * (sig2 / ps_n) * 1.5;
            out->periods_needed = need > 2e9 ? 2000000000 : (int32_t)ceil(need);
        } else out->periods_needed = -1;
    }
    out->snr_db = 10.0 * log10(((lam1 - sig2) > 0.0 ? (lam1 - sig2) : 1e-30) / (double)N * sps / sig2);
    if (out->ident_margin < 1.15) {
        out->flags |= ATKDSP_CODE_REFUSED;
        { char tmp[64]; snprintf(tmp, sizeof tmp, "not identifiable (margin %.2f)", out->ident_margin); set_name(out, tmp); }
        free(v1); free(v2); free(u);
        return 0;
    }
    out->flags |= ATKDSP_CODE_IDENTIFIABLE;
    free(v1); free(v2); free(u);

    /* ---- 3. aligned periods, the carrier offset, the constellation phase ------- */
    const size_t a0 = off + t0;
    const int Mp = (int)((n - a0) / N);
    if (Mp < 3) return ATKDSP_E_EMPTY;
    cd *q = (cd *)calloc(N, sizeof *q);
    atkdsp_cf32 *xd = (atkdsp_cf32 *)malloc((size_t)Mp * N * sizeof *xd);
    if (!q || !xd) { free(q); free(xd); return ATKDSP_E_NOMEM; }
    memcpy(xd, iq + a0, (size_t)Mp * N * sizeof *xd);
    const int P = two ? 4 : 2;
    double alpha = 0.0;
    /* chip instants within the period */
    int *inst = (int *)malloc((size_t)L * 2 * sizeof *inst);
    if (!inst) { free(q); free(xd); return ATKDSP_E_NOMEM; }
    for (int k = 0; k < L; k++) {
        inst[k] = (int)floor((k + 0.5) * sps);
        if (inst[k] >= (int)N) inst[k] = (int)N - 1;
        double qpos = (k + 0.5) * sps + (cfg->chip_mod == ATKDSP_CHIP_OQPSK ? 0.5 * sps : 0.0);
        inst[L + k] = (int)floor(qpos);
        if (inst[L + k] >= (int)N) inst[L + k] -= (int)N;        /* the late Q chip wraps */
    }
    for (int pass = 0; pass < 2; pass++) {
        /* q(i) = sum_m y_m(i)^P */
        for (size_t i = 0; i < N; i++) {
            double sr = 0.0, si = 0.0;
            for (int m = 0; m < Mp; m++) {
                double zr = xd[(size_t)m * N + i].re, zi = xd[(size_t)m * N + i].im;
                double wr = zr, wi = zi;
                for (int p = 1; p < P; p++) { const double tr = wr * zr - wi * zi; wi = wr * zi + wi * zr; wr = tr; }
                sr += wr; si += wi;
            }
            q[i].re = sr; q[i].im = si;
        }
        if (pass == 0) {
            if (!cfg->no_cfo) {
                alpha = kay_freq(q, N) / (double)P;
                if (fabs(alpha) > 1e-12) {
                    for (int m = 0; m < Mp; m++)
                        for (size_t i = 0; i < N; i++) {
                            const double ph = -2.0 * ATK_PI * alpha * (double)((size_t)m * N + i);
                            const double cr = cos(ph), ci = sin(ph);
                            atkdsp_cf32 *z = &xd[(size_t)m * N + i];
                            const double re = z->re * cr - z->im * ci, im = z->re * ci + z->im * cr;
                            z->re = (float)re; z->im = (float)im;
                        }
                    out->flags |= ATKDSP_CODE_CFO;
                    out->cfo_hz = alpha * cfg->fs;
                    out->method[nm++] = ATKDSP_METHOD_CFO_EIGEN;
                }
                continue;                                     /* recompute q on the derotated data */
            }
        }
        /* theta from the chip-centre samples of q */
        double sr = 0.0, si = 0.0;
        for (int k = 0; k < L; k++) { sr += q[inst[k]].re; si += q[inst[k]].im; }
        double theta = atan2(si, sr);
        if (two) theta -= ATK_PI;                              /* (a + jb)^4 = -4 */
        theta /= (double)P;
        const double cr = cos(-theta), ci = sin(-theta);
        for (size_t i = 0; i < (size_t)Mp * N; i++) {
            const double re = xd[i].re * cr - xd[i].im * ci, im = xd[i].re * ci + xd[i].im * cr;
            xd[i].re = (float)re; xd[i].im = (float)im;
        }
        break;
    }
    free(q);

    /* ---- 4. the code waveform(s) and the vote ----------------------------------- */
    double *yr = (double *)malloc((size_t)Mp * N * sizeof *yr);
    double *c = (double *)malloc(N * sizeof *c), *uu = (double *)malloc(N * sizeof *uu);
    int8_t *a = (int8_t *)malloc((size_t)Mp);
    if (!yr || !c || !uu || !a) { free(yr); free(c); free(uu); free(a); free(xd); free(inst); return ATKDSP_E_NOMEM; }
    out->method[nm++] = ATKDSP_METHOD_VOTE;
    out->method[nm++] = ATKDSP_METHOD_VOTE_MARGIN;
    if (two) out->method[nm++] = ATKDSP_METHOD_QPSK_SPLIT;
    double vmin = HUGE_VAL, vsum = 0.0;
    for (int branch = 0; branch < (two ? 2 : 1); branch++) {
        for (int m = 0; m < Mp; m++)
            for (size_t i = 0; i < N; i++)
                yr[(size_t)m * N + i] = branch ? xd[(size_t)m * N + i].im : xd[(size_t)m * N + i].re;
        /* start from the mean |.| shape: the first period */
        double nrm = 0.0;
        for (size_t i = 0; i < N; i++) { c[i] = yr[i]; nrm += c[i] * c[i]; }
        nrm = sqrt(nrm); if (!(nrm > 0.0)) nrm = 1.0;
        for (size_t i = 0; i < N; i++) c[i] /= nrm;
        real_power(yr, Mp, N, c, uu, 30);
        const double s2r = vote(yr, Mp, N, c, uu, cfg->vote_passes > 0 ? cfg->vote_passes : 1, a);
        const double sd = sqrt(s2r / (double)Mp);             /* noise of the mean at one sample */
        for (int k = 0; k < L; k++) {
            const double ck = c[inst[branch ? L + k : k]];
            chips[branch * L + k] = (int8_t)(ck >= 0.0 ? 1 : -1);
            const double z = sd > 0.0 ? fabs(ck) / sd : HUGE_VAL;
            if (z < vmin) vmin = z;
            vsum += z;
        }
    }
    out->vote_min = vmin; out->vote_mean = vsum / (double)(L * (two ? 2 : 1));
    if (two) out->flags |= ATKDSP_CODE_Q;
    free(yr); free(c); free(uu); free(a); free(xd); free(inst);

    /* ---- 5. the generator and the name ------------------------------------------ */
    char tmp[160];
    snprintf(tmp, sizeof tmp, "%d chips, generator not confirmed", L); set_name(out, tmp);
    if (L == 7 || L == 11 || L == 13) {
        if (barker_match(chips, L)) {
            snprintf(tmp, sizeof tmp, "Barker-%d", L);
            set_name(out, tmp);
            out->flags |= ATKDSP_CODE_NAMED;
        }
    }
    {
        uint8_t *bits = (uint8_t *)malloc((size_t)L);
        uint8_t *poly = (uint8_t *)malloc((size_t)L + 1);
        int32_t *prof = (int32_t *)malloc((size_t)L * sizeof *prof);
        uint8_t *bits1 = (uint8_t *)malloc((size_t)L);
        uint8_t *poly1 = (uint8_t *)malloc((size_t)L + 1);
        if (bits && poly && prof && bits1 && poly1) {
            for (int k = 0; k < L; k++) { bits[k] = (uint8_t)(chips[k] < 0); bits1[k] = (uint8_t)(chips[k] >= 0); }
            const ptrdiff_t lc = atkdsp_bm(bits, (size_t)L, poly, (size_t)L + 1);
            atkdsp_bm_profile(bits, (size_t)L, prof);
            out->lc = (int32_t)lc;
            int plateau = 0;
            if (lc > 0) { int i = L - 1; while (i >= 0 && prof[i] == lc) { plateau++; i--; } }
            out->lc_plateau = plateau;
            out->method[nm < 8 ? nm++ : 7] = ATKDSP_METHOD_BM;
            int r = is_pow2m1(L, &r) ? r : 0;
            /* The code is recovered up to a global sign; (chips<0) on -code is
               the m-sequence's complement, of linear complexity r+1 whose parity
               checks fail. So BM and the parity counting are tried on BOTH sign
               conventions (bits and its complement bits1). */
            const ptrdiff_t lc1 = atkdsp_bm(bits1, (size_t)L, poly1, (size_t)L + 1);
            int plat1 = 0;
            { atkdsp_bm_profile(bits1, (size_t)L, prof); int i = L - 1; while (i >= 0 && prof[i] == lc1) { plat1++; i--; } }
            /* 1. a CLEAN m-sequence or Gold on either sign. */
            for (int s = 0; s < 2 && !(out->flags & ATKDSP_CODE_NAMED); s++) {
                const ptrdiff_t clc = s ? lc1 : lc;
                const int cplat = s ? plat1 : plateau;
                const uint8_t *cpoly = s ? poly1 : poly;
                uint64_t mask = 0;
                if (!(clc >= 2 && 4 * clc <= L && 2 * cplat >= L && bm_poly_to_mask(cpoly, (int)clc, &mask))) continue;
                if (r && (int)clc == r && mask_primitive(mask, r)) {
                    char ps[40]; poly_string(mask, r, ps, sizeof ps);
                    snprintf(tmp, sizeof tmp, "m-sequence (%d) %s", L, ps); set_name(out, tmp);
                    out->poly = mask; out->poly_degree = r; out->poly_method = ATKDSP_POLY_BM;
                    out->lc = (int32_t)clc; out->lc_plateau = cplat;
                    out->flags |= ATKDSP_CODE_POLY | ATKDSP_CODE_NAMED;
                } else if (r && (int)clc == 2 * r) {
                    uint64_t f1, f2;
                    if (gold_factor(mask, r, &f1, &f2)) {
                        char p1[20], p2[20]; poly_string(f1, r, p1, sizeof p1); poly_string(f2, r, p2, sizeof p2);
                        snprintf(tmp, sizeof tmp, "Gold (%d) %s x %s", L, p1, p2); set_name(out, tmp);
                        out->poly = mask; out->poly_degree = (int32_t)clc; out->poly_method = ATKDSP_POLY_BM;
                        out->lc = (int32_t)clc; out->lc_plateau = cplat;
                        out->flags |= ATKDSP_CODE_POLY | ATKDSP_CODE_NAMED;
                    }
                }
            }
            /* 2. parity-check counting over the degree-r primitives, on the sign
               that gives the stronger statistic (error-tolerant, FR 2004 / CH 1989). */
            if (!(out->flags & ATKDSP_CODE_NAMED) && r && r <= cfg->poly_max_degree) {
                uint64_t polys[4096];
                const ptrdiff_t np = atkdsp_primitive_polys(r, polys, 4096);
                double zbest = -HUGE_VAL; uint64_t pbest = 0;
                int8_t *neg = (int8_t *)malloc((size_t)L);
                if (neg) for (int k = 0; k < L; k++) neg[k] = (int8_t)(-chips[k]);
                for (ptrdiff_t i = 0; i < np && i < 4096; i++) {
                    double z = atkdsp_parity_z(chips, L, polys[i], r);
                    if (neg) { const double zn = atkdsp_parity_z(neg, L, polys[i], r); if (zn > z) z = zn; }
                    if (z > zbest) { zbest = z; pbest = polys[i]; }
                }
                free(neg);
                const double k = atk_norm_isf(cfg->pfa / (double)(np > 0 ? np : 1));
                out->poly_z = zbest / sqrt((double)L);
                out->method[nm < 8 ? nm++ : 7] = ATKDSP_METHOD_PARITY_COUNT;
                if (zbest > k * sqrt((double)L)) {
                    out->poly = pbest; out->poly_degree = r; out->poly_method = ATKDSP_POLY_PARITY;
                    out->flags |= ATKDSP_CODE_POLY | ATKDSP_CODE_NAMED;
                    char ps[40]; poly_string(pbest, r, ps, sizeof ps);
                    snprintf(tmp, sizeof tmp, "m-sequence (%d) %s, by parity (%d checks failed)", L, ps,
                             (int)((L - zbest) / 2.0 + 0.5));
                    set_name(out, tmp);
                }
            }
            /* 3. Still unnamed: report the linear complexity and the chips. */
            uint64_t mask3 = 0;
            if (!(out->flags & ATKDSP_CODE_NAMED) && lc >= 2 && 4 * lc <= L && 2 * plateau >= L
                && bm_poly_to_mask(poly, (int)lc, &mask3)) {
                char ps[40]; poly_string(mask3, (int)lc, ps, sizeof ps);
                snprintf(tmp, sizeof tmp, "LFSR degree %d (%d) %s", (int)lc, L, ps); set_name(out, tmp);
                out->poly = mask3; out->poly_degree = (int32_t)lc; out->poly_method = ATKDSP_POLY_BM;
                out->flags |= ATKDSP_CODE_POLY;
            }
        }
        free(bits); free(poly); free(prof); free(bits1); free(poly1);
    }
    return L;
}
