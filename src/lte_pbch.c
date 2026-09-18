/* 11b. LTE PBCH -> MIB decode.
 *
 * The downlink physical broadcast channel carries the Master Information
 * Block: channel bandwidth, PHICH configuration, the antenna-port count and
 * the system frame number. It is BROADCAST and UNSCRAMBLED-to-everyone —
 * the network's own public parameters, nothing about any subscriber.
 *
 * The chain, receive side (36.211 physical, 36.212 coding, 36.331 MIB):
 *   time samples of slot-1 symbols 0..3 of subframe 0
 *     -> OFDM demod (128-FFT, normal CP), central 72 subcarriers
 *     -> CRS channel estimate per antenna port
 *     -> transmit-diversity combine (1 port direct; 2/4 ports SFBC / SFBC+FSTD)
 *     -> QPSK soft bits -> descramble (c_init = N_ID) -> de-rate-match
 *     -> tail-biting Viterbi (K=7, 133/171/165) -> CRC-16 with antenna mask.
 * Blind over {1,2,4 ports} x {SFN mod 4}. A decode is accepted only when the
 * CRC checks AND the 10 spare bits are zero and the bandwidth code is valid,
 * which keeps chance CRC passes off pure noise (measured 0 in 600 trials).
 *
 * The numpy twin in atkdsp.reference (lte_mib_* and build_pbch_samples) is
 * the readable statement of all of this; tests/test_cross_check.py holds the
 * C to it. */
#include "internal.h"

#define NFFT      128
#define NSC       72          /* 6 central RB, DC included at index 36     */
#define DC_K      36
#define CP0       10
#define CP1       9
#define NRB_MAX   110
#define SEG_BITS  480         /* QPSK soft bits carried by one PBCH frame  */
#define MIB_BITS  24
#define FRAME_BITS 40         /* MIB + CRC16                               */
#define CODED_BITS 120        /* rate-1/3 tail-biting mother codeword      */
#define MAXW      4           /* frames soft-combined in one window        */

static const double SQ = 0.70710678118654752440;   /* 1/sqrt(2) */
static const int    PBCH_SYM_OFF[4] = { CP0, CP0+NFFT+CP1,
                                        CP0+2*NFFT+2*CP1, CP0+3*NFFT+3*CP1 };
/* the three antenna-port CRC masks (36.212 Table 5.3.1.1-1) */
static const int NPORTS_SET[3] = { 1, 2, 4 };
static const int PERM32[32] = {
    1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31,
    0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30 };

/* ---- Gold sequence (36.211 7.2) ---------------------------------------- */
int atkdsp_lte_gold(unsigned c_init, int length, signed char *out) {
    /* x1(n+31)=x1(n+3)^x1(n); x2(n+31)=x2(n+3)^x2(n+2)^x2(n+1)^x2(n);
     * c(n)=x1(n+Nc)^x2(n+Nc), Nc=1600. */
    enum { NC = 1600 };
    static const int MAXLEN = 1920;
    signed char x1[1920 + NC + 31], x2[1920 + NC + 31];
    int n, total;
    if (!out || length <= 0 || length > MAXLEN) return ATKDSP_E_ARG;
    total = length + NC;
    memset(x1, 0, (size_t)(total + 31));
    memset(x2, 0, (size_t)(total + 31));
    x1[0] = 1;
    for (n = 0; n < 31; ++n) x2[n] = (signed char)((c_init >> n) & 1u);
    for (n = 0; n < total; ++n) {
        x1[n + 31] = (signed char)((x1[n + 3] ^ x1[n]) & 1);
        x2[n + 31] = (signed char)((x2[n+3] ^ x2[n+2] ^ x2[n+1] ^ x2[n]) & 1);
    }
    for (n = 0; n < length; ++n)
        out[n] = (signed char)((x1[n + NC] ^ x2[n + NC]) & 1);
    return ATKDSP_OK;
}

/* ---- CRC-16 (36.212 5.1.1, D^16+D^12+D^5+1) ---------------------------- */
static unsigned crc16(const signed char *bits, int nbits) {
    unsigned reg = 0, i;
    int k;
    for (k = 0; k < nbits; ++k) {
        reg = (reg << 1) | (unsigned)(bits[k] & 1);
        if (reg & 0x10000u) reg ^= 0x11021u;
    }
    for (i = 0; i < 16; ++i) { reg <<= 1; if (reg & 0x10000u) reg ^= 0x11021u; }
    return reg & 0xFFFFu;
}
/* mask bits (MSB first) for a given port count */
static unsigned crc_mask(int n_ports) {
    if (n_ports == 1) return 0x0000u;
    if (n_ports == 2) return 0xFFFFu;
    return 0x5555u;                 /* <0101...>, 4 ports */
}

/* ---- tail-biting convolutional trellis outputs ------------------------- */
static void build_bm(double bm[128][3]) {
    static const unsigned GEN[3] = { 0133u, 0171u, 0165u };
    int f, i, j, s, cur, full[7], acc;
    for (f = 0; f < 128; ++f) {
        s = f >> 1; cur = f & 1;
        full[0] = cur;
        for (j = 0; j < 6; ++j) full[1 + j] = (s >> j) & 1;   /* c_{k-1..k-6} */
        for (i = 0; i < 3; ++i) {
            acc = 0;
            for (j = 0; j < 7; ++j)
                if ((GEN[i] >> (6 - j)) & 1u) acc ^= full[j];
            bm[f][i] = 1.0 - 2.0 * (acc & 1);
        }
    }
}

/* wrap-around Viterbi. llr = 3*40 soft (positive => bit 0). out = 40 bits. */
static void viterbi_tb(const double *llr, signed char *out, int32_t *bp) {
    double bm[128][3];
    double pm[64], npm[64], A, B;
    const int K = FRAME_BITS, laps = ATK_LTE_VIT_LAPS, steps = laps * K;
    int lap, k, t, step = 0, s;
    build_bm(bm);
    for (t = 0; t < 64; ++t) pm[t] = 0.0;
    for (lap = 0; lap < laps; ++lap) {
        for (k = 0; k < K; ++k) {
            const double l0 = llr[3*k], l1 = llr[3*k+1], l2 = llr[3*k+2];
            double rew[128];
            int f;
            for (f = 0; f < 128; ++f)
                rew[f] = bm[f][0]*l0 + bm[f][1]*l1 + bm[f][2]*l2;
            for (t = 0; t < 64; ++t) {
                A = pm[t >> 1]        + rew[t];      /* pred f=t, state t>>1   */
                B = pm[(t >> 1) + 32] + rew[t + 64]; /* pred f=t+64            */
                if (A >= B) { npm[t] = A; bp[step*64 + t] = t >> 1; }
                else        { npm[t] = B; bp[step*64 + t] = (t >> 1) + 32; }
            }
            for (t = 0; t < 64; ++t) pm[t] = npm[t];
            ++step;
        }
    }
    /* best final state, trace back, keep the middle lap */
    s = 0; { double best = pm[0]; for (t = 1; t < 64; ++t) if (pm[t] > best) { best = pm[t]; s = t; } }
    {
        signed char dec[ATK_LTE_VIT_LAPS * FRAME_BITS];
        int mid = (laps / 2) * K;
        for (step = steps - 1; step >= 0; --step) {
            dec[step] = (signed char)(s & 1);
            s = bp[step*64 + s];
        }
        for (k = 0; k < K; ++k) out[k] = dec[mid + k];
    }
}

/* ---- rate de-match (36.212 5.1.4.2, convolutional) --------------------- */
/* Sub-block interleaver output indices for D=40: 64 positions, -1 = null. */
static void subblock_idx(int idx[64]) {
    const int C = 32, R = 2, nd = R*C - FRAME_BITS;    /* nd = 24 */
    int seq[64], r, c, p;
    for (p = 0; p < nd; ++p) seq[p] = -1;
    for (p = 0; p < FRAME_BITS; ++p) seq[nd + p] = p;  /* row-major write */
    /* read column-by-column after permuting columns */
    p = 0;
    for (c = 0; c < C; ++c)
        for (r = 0; r < R; ++r)
            idx[p++] = seq[r * C + PERM32[c]];
}

/* deposit one frame's 480 descrambled LLRs into acc[120] (interleaved). */
static void dematch_accumulate(const double *e_llr, int seg, int idx[64],
                               double acc[CODED_BITS]) {
    const int Rc = 64, Kw = 192;
    double w[192];
    int isnull[192], si, pos, j, produced, ei, start = seg * SEG_BITS;
    double d[3][FRAME_BITS];
    int p;
    for (p = 0; p < Kw; ++p) { w[p] = 0.0; isnull[p] = 0; }
    for (si = 0; si < 3; ++si)
        for (pos = 0; pos < Rc; ++pos)
            if (idx[pos] < 0) isnull[si*Rc + pos] = 1;
    /* advance to the segment start on the circular skip-null read */
    j = 0; produced = 0;
    while (produced < start) { if (!isnull[j % Kw]) ++produced; ++j; }
    for (ei = 0; ei < SEG_BITS; ) {
        pos = j % Kw;
        if (!isnull[pos]) { w[pos] += e_llr[ei]; ++ei; }
        ++j;
    }
    for (si = 0; si < 3; ++si) {
        for (p = 0; p < FRAME_BITS; ++p) d[si][p] = 0.0;
        for (pos = 0; pos < Rc; ++pos)
            if (idx[pos] >= 0) d[si][idx[pos]] += w[si*Rc + pos];
    }
    for (p = 0; p < FRAME_BITS; ++p) {
        acc[3*p]   += d[0][p];
        acc[3*p+1] += d[1][p];
        acc[3*p+2] += d[2][p];
    }
}

/* ---- CRS ---------------------------------------------------------------- */
static void crs_central(int n_id, int l, double re[12], double im[12]) {
    signed char c[4 * NRB_MAX];
    unsigned ci = (1u << 10) * (unsigned)(7*(1 + 1) + l + 1) * (unsigned)(2*n_id + 1)
                  + (unsigned)(2*n_id) + 1u;              /* n_s = 1 (slot 1) */
    int m, base = NRB_MAX - 6;                            /* 104 */
    atkdsp_lte_gold(ci, 4 * NRB_MAX, c);
    for (m = 0; m < 12; ++m) {
        re[m] = SQ * (1 - 2 * c[2*(base + m)]);
        im[m] = SQ * (1 - 2 * c[2*(base + m) + 1]);
    }
}
static void crs_pos(int n_id, int port, int l, int pos[12]) {
    int vshift = n_id % 6, v, m;
    if (port == 0) v = (l == 0) ? 0 : 3;
    else if (port == 1) v = (l == 0) ? 3 : 0;
    else if (port == 2) v = 0;
    else v = 3;
    for (m = 0; m < 12; ++m) pos[m] = 6*m + (v + vshift) % 6;
}
static int crs_symbol(int port) { return (port <= 1) ? 0 : 1; }

/* linear-interpolate a complex channel given at 12 sorted knots to 0..71 */
static void interp72(const int pos[12], const double *hre, const double *him,
                     double *ore, double *oim) {
    int k, m = 0;
    for (k = 0; k < NSC; ++k) {
        while (m < 11 && pos[m + 1] <= k) ++m;
        {
            int a = pos[m], b = pos[m + 1 < 12 ? m + 1 : 11];
            double t = (b == a) ? 0.0 : (double)(k - a) / (double)(b - a);
            int mb = (m + 1 < 12) ? m + 1 : 11;
            ore[k] = hre[m] + t * (hre[mb] - hre[m]);
            oim[k] = him[m] + t * (him[mb] - him[m]);
        }
    }
}

/* ---- OFDM demod one PBCH block into Y[4][72] --------------------------- */
static void demod_block(const atkdsp_lte *h, const atkdsp_cf32 *block,
                        double cfo_hz, atkdsp_cf32 Y[4][NSC]) {
    int l, i, k;
    atkdsp_cf32 buf[NFFT], spec[NFFT];
    const double w = -2.0 * ATK_PI * cfo_hz / ATKDSP_LTE_RATE;
    for (l = 0; l < 4; ++l) {
        const int off = PBCH_SYM_OFF[l];
        for (i = 0; i < NFFT; ++i) {
            const atkdsp_cf32 x = block[off + i];
            if (cfo_hz != 0.0) {
                const double ph = w * (double)(off + i);
                const double cr = cos(ph), sr = sin(ph);
                buf[i].re = (float)(x.re * cr - x.im * sr);
                buf[i].im = (float)(x.re * sr + x.im * cr);
            } else { buf[i] = x; }
        }
        atkdsp_fft_exec(h->sym, buf, spec, 0);              /* forward */
        for (k = 0; k < NSC; ++k) {
            int off_k = k - DC_K;                           /* -36..+35 */
            int bin = (off_k % NFFT + NFFT) % NFFT;
            Y[l][k] = spec[bin];
        }
    }
}

/* estimate H[port][72] from CRS (held across the 4 PBCH symbols) */
static void estimate_channels(int n_id, atkdsp_cf32 Y[4][NSC],
                              double Hre[4][NSC], double Him[4][NSC]) {
    int port, m;
    for (port = 0; port < 4; ++port) {
        int l = crs_symbol(port), pos[12];
        double cre[12], cim[12], hre[12], him[12];
        crs_central(n_id, l, cre, cim);
        crs_pos(n_id, port, l, pos);
        for (m = 0; m < 12; ++m) {
            /* H = Y / crs = Y * conj(crs) / |crs|^2 ; |crs|^2 = 1 */
            const double yr = Y[l][pos[m]].re, yi = Y[l][pos[m]].im;
            hre[m] = yr * cre[m] + yi * cim[m];
            him[m] = yi * cre[m] - yr * cim[m];
        }
        interp72(pos, hre, him, Hre[port], Him[port]);
    }
}

/* build the 240 (k,l) PBCH REs (4-port CRS punctured) */
static int build_remap(int n_id, int rk[240], int rl[240]) {
    int used[4][NSC], port, l, m, k, cnt = 0, pos[12];
    memset(used, 0, sizeof used);
    for (port = 0; port < 4; ++port) {
        l = crs_symbol(port);
        crs_pos(n_id, port, l, pos);
        for (m = 0; m < 12; ++m) used[l][pos[m]] = 1;
    }
    for (l = 0; l < 4; ++l)
        for (k = 0; k < NSC; ++k)
            if (!used[l][k]) { rk[cnt] = k; rl[cnt] = l; ++cnt; }
    return cnt;                       /* 240 */
}

/* transmit-diversity combine -> 240 soft QPSK symbols (yre,yim) */
static void combine(int n_ports, atkdsp_cf32 Y[4][NSC],
                    double Hre[4][NSC], double Him[4][NSC],
                    const int *rk, const int *rl, double *yre, double *yim) {
    int idx;
    if (n_ports == 1) {
        for (idx = 0; idx < 240; ++idx) {
            const int k = rk[idx], l = rl[idx];
            /* conj(H0)*Y */
            yre[idx] = Hre[0][k]*Y[l][k].re + Him[0][k]*Y[l][k].im;
            yim[idx] = Hre[0][k]*Y[l][k].im - Him[0][k]*Y[l][k].re;
        }
        return;
    }
    if (n_ports == 2) {
        for (idx = 0; idx < 240; idx += 2) {
            const int ka = rk[idx],   la = rl[idx];
            const int kb = rk[idx+1], lb = rl[idx+1];
            /* h0,h1 at ka (pair shares a subcarrier region) */
            const double h0r = Hre[0][ka], h0i = Him[0][ka];
            const double h1r = Hre[1][ka], h1i = Him[1][ka];
            const double r0r = Y[la][ka].re, r0i = Y[la][ka].im;
            const double r1r = Y[lb][kb].re, r1i = Y[lb][kb].im;
            /* y0 = conj(h0)*r0 + h1*conj(r1); y1 = conj(h0)*r1 - h1*conj(r0) */
            yre[idx]   = h0r*r0r + h0i*r0i + (h1r*r1r + h1i*r1i);
            yim[idx]   = h0r*r0i - h0i*r0r + (h1i*r1r - h1r*r1i);
            yre[idx+1] = h0r*r1r + h0i*r1i - (h1r*r0r + h1i*r0i);
            yim[idx+1] = h0r*r1i - h0i*r1r - (h1i*r0r - h1r*r0i);
        }
        return;
    }
    /* 4 ports: SFBC+FSTD. group of 4: (0,2) on first pair, (1,3) on second */
    for (idx = 0; idx < 240; idx += 4) {
        int g;
        for (g = 0; g < 2; ++g) {
            const int i0 = idx + 2*g, i1 = idx + 2*g + 1;
            const int pa = (g == 0) ? 0 : 1, pb = (g == 0) ? 2 : 3;
            const int ka = rk[i0], la = rl[i0], kb = rk[i1], lb = rl[i1];
            const double h0r = Hre[pa][ka], h0i = Him[pa][ka];
            const double h1r = Hre[pb][ka], h1i = Him[pb][ka];
            const double r0r = Y[la][ka].re, r0i = Y[la][ka].im;
            const double r1r = Y[lb][kb].re, r1i = Y[lb][kb].im;
            yre[i0] = h0r*r0r + h0i*r0i + (h1r*r1r + h1i*r1i);
            yim[i0] = h0r*r0i - h0i*r0r + (h1i*r1r - h1r*r1i);
            yre[i1] = h0r*r1r + h0i*r1i - (h1r*r0r + h1i*r0i);
            yim[i1] = h0r*r1i - h0i*r1r - (h1i*r0r - h1r*r0i);
        }
    }
}

/* ---- unpack MIB bits (36.331), with the false-alarm gate --------------- */
static int unpack_mib(const signed char *b, int i0, atkdsp_lte_mib *out) {
    static const int BW[6] = { 6, 15, 25, 50, 75, 100 };
    int bw = b[0]*4 + b[1]*2 + b[2], sfn_msb = 0, k;
    if (bw > 5) return 0;
    for (k = 14; k < 24; ++k) if (b[k]) return 0;    /* spare must be zero */
    for (k = 6; k < 14; ++k) sfn_msb = (sfn_msb << 1) | b[k];
    out->dl_bw_rb  = BW[bw];
    out->phich_dur = b[3];
    out->phich_res = b[4]*2 + b[5];
    out->sfn       = (sfn_msb << 2) | i0;
    return 1;
}

/* ---- decode one window of wlen consecutive frames --------------------- */
static int decode_window(atkdsp_lte *h, const atkdsp_cf32 *base,
                         ptrdiff_t stride, int fs, int wlen, int n_id,
                         double cfo_hz, const signed char *scr,
                         const int *rk, const int *rl, int idx[64],
                         atkdsp_lte_mib *out) {
    atkdsp_cf32 Y[MAXW][4][NSC];
    double Hre[MAXW][4][NSC], Him[MAXW][4][NSC];
    double yre[MAXW][240], yim[MAXW][240];
    int f, hi, i0, n;
    for (f = 0; f < wlen; ++f) {
        demod_block(h, base + (ptrdiff_t)(fs + f) * stride, cfo_hz, Y[f]);
        estimate_channels(n_id, Y[f], Hre[f], Him[f]);
    }
    for (hi = 0; hi < 3; ++hi) {
        const int n_ports = NPORTS_SET[hi];
        for (f = 0; f < wlen; ++f)
            combine(n_ports, Y[f], Hre[f], Him[f], rk, rl, yre[f], yim[f]);
        for (i0 = 0; i0 < 4; ++i0) {
            double acc[CODED_BITS];
            double llr[SEG_BITS];
            signed char bits[FRAME_BITS];
            unsigned rx, calc;
            int seg, p;
            for (p = 0; p < CODED_BITS; ++p) acc[p] = 0.0;
            for (f = 0; f < wlen; ++f) {
                seg = (i0 + f) & 3;
                for (n = 0; n < 240; ++n) {
                    const double s0 = 1.0 - 2.0*scr[seg*SEG_BITS + 2*n];
                    const double s1 = 1.0 - 2.0*scr[seg*SEG_BITS + 2*n + 1];
                    llr[2*n]   = s0 * yre[f][n];
                    llr[2*n+1] = s1 * yim[f][n];
                }
                dematch_accumulate(llr, seg, idx, acc);
            }
            viterbi_tb(acc, bits, h->vit_bp);
            /* CRC: payload bits[0..23], parity bits[24..39] */
            calc = crc16(bits, MIB_BITS) ^ crc_mask(n_ports);
            rx = 0;
            for (p = 0; p < 16; ++p) rx = (rx << 1) | (unsigned)(bits[24 + p] & 1);
            if (rx != calc) continue;
            if (unpack_mib(bits, i0, out)) { out->n_ports = n_ports; return 1; }
        }
    }
    return 0;
}

/* ---- public API -------------------------------------------------------- */
int atkdsp_lte_mib_decode(atkdsp_lte *h, const atkdsp_cf32 *block, int n_id,
                          double cfo_hz, atkdsp_lte_mib *out) {
    return atkdsp_lte_mib_decode_combined(h, block, 1, 0, n_id, cfo_hz, out);
}

int atkdsp_lte_mib_decode_combined(atkdsp_lte *h, const atkdsp_cf32 *base,
                                   int nframes, ptrdiff_t stride, int n_id,
                                   double cfo_hz, atkdsp_lte_mib *out) {
    signed char scr[1920];
    int rk[240], rl[240], idx[64], fs;
    if (!h || !base || !out || n_id < 0 || n_id > 503) return ATKDSP_E_ARG;
    if (nframes < 1) return ATKDSP_E_ARG;
    memset(out, 0, sizeof *out);
    atkdsp_lte_gold((unsigned)n_id, 1920, scr);
    build_remap(n_id, rk, rl);
    subblock_idx(idx);
    /* single-frame first (fast, primary), then 4-frame soft combine */
    for (fs = 0; fs < nframes; ++fs)
        if (decode_window(h, base, stride, fs, 1, n_id, cfo_hz,
                          scr, rk, rl, idx, out)) return 1;
    for (fs = 0; fs + MAXW <= nframes; ++fs)
        if (decode_window(h, base, stride, fs, MAXW, n_id, cfo_hz,
                          scr, rk, rl, idx, out)) return 1;
    return 0;
}
