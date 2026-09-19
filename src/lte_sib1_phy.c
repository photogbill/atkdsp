/* 11f. SIB1 physical layer: full-bandwidth subframe -> tower identity.
 *
 * Ties the downlink receive chain together for SystemInformationBlockType1:
 *   subframe I/Q at the cell numerology
 *     -> OFDM demod (N_FFT 128..2048 for 6..100 RB)
 *     -> port-0 CRS channel estimate + single-tap equalisation
 *     -> PCFICH (control-format indicator, CFI)
 *     -> blind SI-RNTI PDCCH search (DCI 1A -> RB allocation via RIV)
 *     -> PDSCH RE extraction + descramble -> turbo (atkdsp_lte_turbo_decode,
 *        the transport-block size gated by CRC-24) -> ASN.1 identity parse.
 * The result is the cell's own broadcast identity: operator PLMN, TAC, ECI.
 * Nothing about any subscriber is read.
 *
 * GEOMETRY CAVEAT: the REG/CCE numbering and the PCFICH/PDCCH RE choice here
 * are a self-consistent statement of 36.211 6.2.4/6.7/6.8. A transmit/receive
 * round trip (tests/test_cross_check.py, against the atkdsp.reference twin)
 * proves the coding, scrambling, interleaving and RNTI-masked CRC — NOT the
 * spec's exact RE positions on a real air capture. Confirm the control-region
 * geometry against a live signal before trusting a live decode. Not a
 * per-sample hot path (one SIB every 80 ms), so this call allocates scratch.
 *
 * The numpy twin is atkdsp.reference.lte_sib1_decode_iq and the lte_* PHY
 * helpers around it. 36.211 / 36.212 / 36.331. */
#include "internal.h"

#define SC_PER_RB   12
#define SYMS_SF     14
#define SYMS_SLOT   7
#define NRB_MAX     110
#define SI_RNTI     0xFFFF
#define DCI_D       40          /* DCI 1A (24) + CRC-16 = 40-bit conv frame  */
#define CONV_PERM32 32

static const double SQ = 0.70710678118654752440;
static const int PERM32[32] = {
    1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31,
    0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30 };

/* provided by src/lte_turbo.c (internal.h), for the CRC-gated K search */
extern int atk_qpp_count(void);
extern int atk_qpp_k(int i);

/* ---- gold (36.211 7.2), arbitrary length (PDSCH needs > 1920) ---------- */
static void gold(unsigned c_init, int length, signed char *out) {
    enum { NC = 1600 };
    int total = length + NC, n;
    signed char *x1 = (signed char *)malloc((size_t)(total + 31));
    signed char *x2 = (signed char *)malloc((size_t)(total + 31));
    if (!x1 || !x2) { free(x1); free(x2); return; }
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
    free(x1); free(x2);
}

/* ---- grid params ------------------------------------------------------- */
static int nfft_for(int n_rb) {
    switch (n_rb) {
        case 6: return 128; case 15: return 256; case 25: return 512;
        case 50: return 1024; case 75: return 1536; case 100: return 2048;
    }
    return 0;
}
static int sc_to_bin(int k, int n_sc, int nfft) {
    int off = k - n_sc / 2;
    if (off >= 0) off += 1;               /* skip DC */
    return ((off % nfft) + nfft) % nfft;
}

/* ---- OFDM demod one subframe into grid[n_sc*14] (row-major k, then l) --- */
static int ofdm_demod(const atkdsp_cf32 *s, size_t n, int n_rb,
                      atkdsp_cf32 *grid) {
    const int nfft = nfft_for(n_rb), n_sc = n_rb * SC_PER_RB;
    const int cp_long = nfft * 160 / 2048, cp_short = nfft * 144 / 2048;
    atkdsp_fft *plan = atkdsp_fft_create((size_t)nfft);
    atkdsp_cf32 *buf, *spec;
    int l, k; size_t pos = 0;
    if (!plan) return ATKDSP_E_FFT;
    buf  = (atkdsp_cf32 *)malloc(sizeof(atkdsp_cf32) * (size_t)nfft);
    spec = (atkdsp_cf32 *)malloc(sizeof(atkdsp_cf32) * (size_t)nfft);
    if (!buf || !spec) { free(buf); free(spec); atkdsp_fft_destroy(plan); return ATKDSP_E_NOMEM; }
    for (l = 0; l < SYMS_SF; ++l) {
        const int cp = (l % SYMS_SLOT == 0) ? cp_long : cp_short;
        pos += (size_t)cp;
        if (pos + (size_t)nfft > n) {
            free(buf); free(spec); atkdsp_fft_destroy(plan); return ATKDSP_E_ARG;
        }
        memcpy(buf, s + pos, sizeof(atkdsp_cf32) * (size_t)nfft);
        pos += (size_t)nfft;
        atkdsp_fft_exec(plan, buf, spec, 0);              /* forward */
        for (k = 0; k < n_sc; ++k) {
            const double sc = sqrt((double)n_sc) / (double)nfft;
            const atkdsp_cf32 v = spec[sc_to_bin(k, n_sc, nfft)];
            grid[(size_t)l * n_sc + k].re = (float)(v.re * sc);
            grid[(size_t)l * n_sc + k].im = (float)(v.im * sc);
        }
    }
    free(buf); free(spec); atkdsp_fft_destroy(plan);
    return ATKDSP_OK;
}

/* ---- CRS (port 0) reference values + positions for one CRS symbol ------ */
static int crs_re(int n_id, int n_rb, int n_s, int l, int v,
                  int *pos, double *rre, double *rim) {
    signed char *c = (signed char *)malloc((size_t)(4 * NRB_MAX));
    unsigned ci;
    int vshift = n_id % 6, m0 = NRB_MAX - n_rb, m, cnt = 2 * n_rb;
    if (!c) return 0;
    ci = (unsigned)((1 << 10) * (7 * (n_s + 1) + l + 1) * (2 * n_id + 1)
                    + 2 * n_id + 1);
    gold(ci, 4 * NRB_MAX, c);
    for (m = 0; m < cnt; ++m) {
        int idx = m0 + m;
        pos[m] = 6 * m + (v + vshift) % 6;
        rre[m] = SQ * (1 - 2 * c[2 * idx]);
        rim[m] = SQ * (1 - 2 * c[2 * idx + 1]);
    }
    free(c);
    return cnt;
}

/* linear interpolation matching numpy.interp (clamped at the ends). */
static void interp_clamp(const int *xp, const double *fpr, const double *fpi,
                         int npnt, int n_sc, double *ore, double *oim) {
    int k, m = 0;
    for (k = 0; k < n_sc; ++k) {
        if (k <= xp[0]) { ore[k] = fpr[0]; oim[k] = fpi[0]; continue; }
        if (k >= xp[npnt - 1]) { ore[k] = fpr[npnt - 1]; oim[k] = fpi[npnt - 1]; continue; }
        while (m < npnt - 1 && xp[m + 1] <= k) ++m;
        {
            int a = xp[m], b = xp[m + 1];
            double t = (b == a) ? 0.0 : (double)(k - a) / (double)(b - a);
            ore[k] = fpr[m] + t * (fpr[m + 1] - fpr[m]);
            oim[k] = fpi[m] + t * (fpi[m + 1] - fpi[m]);
        }
    }
}

/* Port-0 single-tap ZF equalisation of the whole grid, in place. CRS symbols
 * are {0,4,7,11}; each data symbol uses the nearest CRS symbol's H (ties to
 * the earlier). */
static int equalize(atkdsp_cf32 *grid, int n_id, int n_rb) {
    const int n_sc = n_rb * SC_PER_RB;
    const int crs_sym[4] = {0, 4, 7, 11};
    const int crs_ns[4]  = {0, 0, 1, 1};
    const int crs_l[4]   = {0, 4, 0, 4};
    double *Hre = (double *)malloc(sizeof(double) * (size_t)(4 * n_sc));
    double *Him = (double *)malloc(sizeof(double) * (size_t)(4 * n_sc));
    int *pos = (int *)malloc(sizeof(int) * (size_t)(2 * n_rb));
    double *hkr = (double *)malloc(sizeof(double) * (size_t)(2 * n_rb));
    double *hki = (double *)malloc(sizeof(double) * (size_t)(2 * n_rb));
    double *cre = (double *)malloc(sizeof(double) * (size_t)(2 * n_rb));
    double *cim = (double *)malloc(sizeof(double) * (size_t)(2 * n_rb));
    int ci, l, k, m, cnt;
    if (!Hre || !Him || !pos || !hkr || !hki || !cre || !cim) {
        free(Hre); free(Him); free(pos); free(hkr); free(hki); free(cre); free(cim);
        return ATKDSP_E_NOMEM;
    }
    for (ci = 0; ci < 4; ++ci) {
        int v = (crs_l[ci] == 0) ? 0 : 3;             /* port-0 v-shift base */
        cnt = crs_re(n_id, n_rb, crs_ns[ci], crs_l[ci], v, pos, cre, cim);
        for (m = 0; m < cnt; ++m) {
            const atkdsp_cf32 y = grid[(size_t)crs_sym[ci] * n_sc + pos[m]];
            /* H = Y * conj(crs), |crs| = 1 */
            hkr[m] = y.re * cre[m] + y.im * cim[m];
            hki[m] = y.im * cre[m] - y.re * cim[m];
        }
        interp_clamp(pos, hkr, hki, cnt, n_sc,
                     Hre + (size_t)ci * n_sc, Him + (size_t)ci * n_sc);
    }
    for (l = 0; l < SYMS_SF; ++l) {
        int best = 0, bd = 100, i;
        for (i = 0; i < 4; ++i) {
            int d = crs_sym[i] - l; if (d < 0) d = -d;
            if (d < bd) { bd = d; best = i; }
        }
        for (k = 0; k < n_sc; ++k) {
            const double hr = Hre[(size_t)best * n_sc + k];
            const double hi = Him[(size_t)best * n_sc + k];
            const double den = hr * hr + hi * hi + 1e-9;
            const atkdsp_cf32 x = grid[(size_t)l * n_sc + k];
            atkdsp_cf32 *o = &grid[(size_t)l * n_sc + k];
            o->re = (float)((x.re * hr + x.im * hi) / den);
            o->im = (float)((x.im * hr - x.re * hi) / den);
        }
    }
    free(Hre); free(Him); free(pos); free(hkr); free(hki); free(cre); free(cim);
    return ATKDSP_OK;
}

/* ---- CRS RE occupancy set (for PDSCH), as a per-(k,l) bytemap ---------- */
static void crs_re_set(int n_id, int n_rb, int n_ports, signed char *occ) {
    const int n_sc = n_rb * SC_PER_RB, vshift = n_id % 6;
    int n_s, port, m;
    memset(occ, 0, (size_t)(n_sc * SYMS_SF));
    for (n_s = 0; n_s < 2; ++n_s) {
        int base = n_s * SYMS_SLOT;
        for (port = 0; port < n_ports; ++port) {
            struct { int l, v; } sy[2]; int ns = 0;
            if (port <= 1) {
                sy[0].l = 0; sy[0].v = (port == 0) ? 0 : 3;
                sy[1].l = 4; sy[1].v = (port == 0) ? 3 : 0; ns = 2;
            } else {
                sy[0].l = 1; sy[0].v = (port == 2) ? 0 : 3; ns = 1;
            }
            for (m = 0; m < ns; ++m) {
                int mm; for (mm = 0; mm < 2 * n_rb; ++mm) {
                    int k = 6 * mm + (sy[m].v + vshift) % 6;
                    occ[(size_t)(base + sy[m].l) * n_sc + k] = 1;
                }
            }
        }
    }
}

/* ---- PCFICH ------------------------------------------------------------ */
static const signed char CFI_WORD[4][32] = {
    {0}, /* unused index 0 */
    {0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0},
    {1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1},
    {1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1},
};
static unsigned pcfich_cinit(int n_id, int n_s) {
    return (unsigned)(((n_s / 2 + 1) * (2 * n_id + 1)) << 9) + (unsigned)n_id;
}
static int pcfich_decode(const atkdsp_cf32 *sym16, int n_id, int n_s) {
    signed char c[32]; double llr[32]; int cfi, i, best_cfi = 1; double best = -1e30;
    gold(pcfich_cinit(n_id, n_s), 32, c);
    for (i = 0; i < 16; ++i) { llr[2*i] = sym16[i].re; llr[2*i+1] = sym16[i].im; }
    for (i = 0; i < 32; ++i) llr[i] = (1 - 2 * c[i]) * llr[i];
    for (cfi = 1; cfi <= 3; ++cfi) {
        double sc = 0.0; for (i = 0; i < 32; ++i) sc += (1 - 2 * CFI_WORD[cfi][i]) * llr[i];
        if (sc > best) { best = sc; best_cfi = cfi; }
    }
    return best_cfi;
}

/* ---- REG geometry: PCFICH REG indices + a symbol's REGs ---------------- */
static void pcfich_reg_idx(int n_id, int n_rb, int idx[4]) {
    int n_regs0 = 2 * n_rb, k_bar = n_id % (2 * n_rb), i;
    for (i = 0; i < 4; ++i) idx[i] = (k_bar + i * (n_regs0 / 4)) % n_regs0;
}

/* Build one control symbol's REGs. Each REG = up to 4 (k,l). Returns count of
 * REGs; fills rk[reg*4+j], rl[reg*4+j] (j<len), and len[reg]. */
static int sym_regs(int n_id, int n_rb, int l, int n_ports,
                    int *rk, int *rl, int *rlen) {
    int vshift = n_id % 6, has_crs = (l == 0) || (l == 1 && n_ports == 4);
    int g, i, nreg = 0;
    if (has_crs) {
        for (g = 0; g < 2 * n_rb; ++g) {
            int cnt = 0;
            for (i = 0; i < 6; ++i) {
                int k = g * 6 + i;
                int is_crs = ((k % 6) == (vshift % 6)) || ((k % 6) == ((vshift + 3) % 6));
                if (!is_crs && cnt < 4) { rk[nreg*4 + cnt] = k; rl[nreg*4 + cnt] = l; ++cnt; }
            }
            rlen[nreg] = cnt; ++nreg;      /* 4 data REs per group */
        }
    } else {
        for (g = 0; g < 3 * n_rb; ++g) {
            for (i = 0; i < 4; ++i) { rk[nreg*4 + i] = g * 4 + i; rl[nreg*4 + i] = l; }
            rlen[nreg] = 4; ++nreg;
        }
    }
    return nreg;
}

/* Ordered, interleaved PDCCH REG list (control region minus PCFICH). Fills
 * ck[]/cl[] with the REs, 4 per REG, in REG order; returns REG count. */
static int control_regs(int n_id, int n_rb, int cfi, int n_ports,
                        int *ck, int *cl) {
    /* per-symbol REG store */
    int maxreg = 3 * n_rb;
    int *rk = (int *)malloc(sizeof(int) * (size_t)(cfi * maxreg * 4));
    int *rl = (int *)malloc(sizeof(int) * (size_t)(cfi * maxreg * 4));
    int *rlen = (int *)malloc(sizeof(int) * (size_t)(cfi * maxreg));
    int *nreg = (int *)malloc(sizeof(int) * (size_t)cfi);
    int pc[4], l, i, most = 0, seqn = 0, out_n;
    /* interleaver working set: the REG REs in frequency-first order */
    int *seq_k, *seq_l;
    if (!rk || !rl || !rlen || !nreg) { free(rk); free(rl); free(rlen); free(nreg); return -1; }
    for (l = 0; l < cfi; ++l) {
        nreg[l] = sym_regs(n_id, n_rb, l, n_ports,
                           rk + (size_t)l * maxreg * 4,
                           rl + (size_t)l * maxreg * 4,
                           rlen + (size_t)l * maxreg);
        if (nreg[l] > most) most = nreg[l];
    }
    pcfich_reg_idx(n_id, n_rb, pc);
    /* frequency-first order: for each REG position i, walk symbols */
    seq_k = (int *)malloc(sizeof(int) * (size_t)(cfi * maxreg * 4));
    seq_l = (int *)malloc(sizeof(int) * (size_t)(cfi * maxreg * 4));
    if (!seq_k || !seq_l) { free(rk); free(rl); free(rlen); free(nreg); free(seq_k); free(seq_l); return -1; }
    for (i = 0; i < most; ++i) {
        for (l = 0; l < cfi; ++l) {
            if (i < nreg[l]) {
                int j;
                if (l == 0) {
                    int skip = 0, q; for (q = 0; q < 4; ++q) if (pc[q] == i) skip = 1;
                    if (skip) continue;                 /* PCFICH REG */
                }
                for (j = 0; j < 4; ++j) {
                    seq_k[seqn*4 + j] = rk[((size_t)l*maxreg + i)*4 + j];
                    seq_l[seqn*4 + j] = rl[((size_t)l*maxreg + i)*4 + j];
                }
                ++seqn;
            }
        }
    }
    /* REG-level sub-block interleave (32-col) + cyclic shift by N_ID */
    {
        int C = 32, R = (seqn + C - 1) / C, KPi = R * C, nd = KPi - seqn;
        int cix, rix, p, sh;
        int *perm_reg = (int *)malloc(sizeof(int) * (size_t)KPi);
        int pn = 0;
        for (cix = 0; cix < C; ++cix) {
            for (rix = 0; rix < R; ++rix) {
                int padpos = rix * C + PERM32[cix];      /* into null-padded seq */
                int src = padpos - nd;                   /* seq index or <0 null */
                perm_reg[pn++] = (src < 0) ? -1 : src;
            }
        }
        /* drop nulls, then cyclic shift by n_id */
        {
            int *kept = (int *)malloc(sizeof(int) * (size_t)seqn);
            int kn = 0;
            for (p = 0; p < KPi; ++p) if (perm_reg[p] >= 0) kept[kn++] = perm_reg[p];
            sh = n_id % kn;
            out_n = kn;
            for (p = 0; p < kn; ++p) {
                int reg = kept[(p + sh) % kn], j;
                for (j = 0; j < 4; ++j) {
                    ck[p*4 + j] = seq_k[reg*4 + j];
                    cl[p*4 + j] = seq_l[reg*4 + j];
                }
            }
            free(kept);
        }
        free(perm_reg);
    }
    free(rk); free(rl); free(rlen); free(nreg); free(seq_k); free(seq_l);
    return out_n;                                        /* number of REGs */
}

/* PCFICH REs (16), from symbol-0 REGs at the PCFICH indices. */
static void pcfich_res(int n_id, int n_rb, int *pk, int *pl) {
    int maxreg = 3 * n_rb;
    int *rk = (int *)malloc(sizeof(int) * (size_t)(maxreg * 4));
    int *rl = (int *)malloc(sizeof(int) * (size_t)(maxreg * 4));
    int *rlen = (int *)malloc(sizeof(int) * (size_t)maxreg);
    int idx[4], i, j, o = 0;
    sym_regs(n_id, n_rb, 0, 2, rk, rl, rlen);
    pcfich_reg_idx(n_id, n_rb, idx);
    for (i = 0; i < 4; ++i)
        for (j = 0; j < 4; ++j) { pk[o] = rk[idx[i]*4 + j]; pl[o] = rl[idx[i]*4 + j]; ++o; }
    free(rk); free(rl); free(rlen);
}

/* ---- generic conv-code (K=7, tail-biting) for the DCI ------------------ */
static unsigned crc16(const signed char *bits, int nbits) {
    unsigned reg = 0; int k, i;
    for (k = 0; k < nbits; ++k) { reg = (reg << 1) | (unsigned)(bits[k] & 1); if (reg & 0x10000u) reg ^= 0x11021u; }
    for (i = 0; i < 16; ++i) { reg <<= 1; if (reg & 0x10000u) reg ^= 0x11021u; }
    return reg & 0xFFFFu;
}
static void conv_bm(double bm[128][3]) {
    static const unsigned GEN[3] = { 0133u, 0171u, 0165u };
    int f, i, j, s, cur, full[7], acc;
    for (f = 0; f < 128; ++f) {
        s = f >> 1; cur = f & 1; full[0] = cur;
        for (j = 0; j < 6; ++j) full[1 + j] = (s >> j) & 1;
        for (i = 0; i < 3; ++i) { acc = 0; for (j = 0; j < 7; ++j) if ((GEN[i] >> (6 - j)) & 1u) acc ^= full[j]; bm[f][i] = 1.0 - 2.0 * (acc & 1); }
    }
}
/* wrap-around Viterbi, generic K. llr = 3*K, out = K bits. */
static void viterbi_tb(const double *llr, int K, signed char *out) {
    double bm[128][3], pm[64], npm[64], A, B; const int laps = 3, steps = laps * K;
    int lap, k, t, step = 0, s; int32_t *bp = (int32_t *)malloc(sizeof(int32_t) * (size_t)steps * 64);
    signed char *dec;
    if (!bp) return;
    conv_bm(bm);
    for (t = 0; t < 64; ++t) pm[t] = 0.0;
    for (lap = 0; lap < laps; ++lap) {
        for (k = 0; k < K; ++k) {
            const double l0 = llr[3*k], l1 = llr[3*k+1], l2 = llr[3*k+2];
            double rew[128]; int f;
            for (f = 0; f < 128; ++f) rew[f] = bm[f][0]*l0 + bm[f][1]*l1 + bm[f][2]*l2;
            for (t = 0; t < 64; ++t) {
                A = pm[t >> 1] + rew[t]; B = pm[(t >> 1) + 32] + rew[t + 64];
                if (A >= B) { npm[t] = A; bp[(size_t)step*64 + t] = t >> 1; }
                else        { npm[t] = B; bp[(size_t)step*64 + t] = (t >> 1) + 32; }
            }
            for (t = 0; t < 64; ++t) pm[t] = npm[t];
            ++step;
        }
    }
    s = 0; { double best = pm[0]; for (t = 1; t < 64; ++t) if (pm[t] > best) { best = pm[t]; s = t; } }
    dec = (signed char *)malloc((size_t)steps);
    for (step = steps - 1; step >= 0; --step) { dec[step] = (signed char)(s & 1); s = bp[(size_t)step*64 + s]; }
    { int mid = (laps / 2) * K; for (k = 0; k < K; ++k) out[k] = dec[mid + k]; }
    free(bp); free(dec);
}
/* conv sub-block interleaver indices for D: length KPi = 32*ceil(D/32). */
static void conv_subblock_idx(int D, int *idx, int *KPi_out) {
    int C = 32, R = (D + C - 1) / C, KPi = R * C, nd = KPi - D, c, r, p = 0;
    for (c = 0; c < C; ++c) for (r = 0; r < R; ++r) {
        int src = r * C + PERM32[c];             /* into null-padded [nd..] */
        idx[p++] = (src < nd) ? -1 : (src - nd);
    }
    *KPi_out = KPi;
}
/* conv rate de-match: E LLRs -> 3*D LLRs (interleaved d[3*p+si]). seg=(0,E). */
static void conv_rate_dematch(const double *e_llr, int E, int D, double *out3D) {
    int idx[64], KPi, Kw, si, pos, j, ei, p; double *w;
    conv_subblock_idx(D, idx, &KPi);            /* KPi<=64 for D<=64 */
    Kw = 3 * KPi;
    w = (double *)calloc((size_t)Kw, sizeof(double));
    /* circular read from j=0, skipping nulls */
    j = 0; ei = 0;
    while (ei < E) {
        pos = j % Kw;
        {
            int local = pos % KPi;             /* which interleaver position */
            int isnull = (idx[local] < 0);
            if (!isnull) { w[pos] += e_llr[ei]; ++ei; }
        }
        ++j;
    }
    for (si = 0; si < 3; ++si)
        for (p = 0; p < D; ++p) out3D[3*p + si] = 0.0;
    for (si = 0; si < 3; ++si)
        for (pos = 0; pos < KPi; ++pos)
            if (idx[pos] >= 0) out3D[3*idx[pos] + si] += w[si*KPi + pos];
    free(w);
}

/* Decode one PDCCH candidate (AL*36 REs already gathered as soft syms).
 * Returns 1 with the DCI payload (D-16 bits) in dci, else 0. */
static int pdcch_candidate(const atkdsp_cf32 *sym, int nsym, int n_id, int n_s,
                           int cce_offset, signed char *dci) {
    int E = nsym * 2, glen = cce_offset * 72 + E, i;
    signed char *c = (signed char *)malloc((size_t)glen);
    double *llr = (double *)malloc(sizeof(double) * (size_t)E);
    double *d = (double *)malloc(sizeof(double) * (size_t)(3 * DCI_D));
    signed char bits[DCI_D]; unsigned calc, rx, mask = SI_RNTI; int ok;
    if (!c || !llr || !d) { free(c); free(llr); free(d); return 0; }
    gold((unsigned)((n_s / 2 << 9) + n_id), glen, c);
    for (i = 0; i < nsym; ++i) { llr[2*i] = sym[i].re; llr[2*i+1] = sym[i].im; }
    for (i = 0; i < E; ++i) llr[i] = (1 - 2 * c[cce_offset * 72 + i]) * llr[i];
    conv_rate_dematch(llr, E, DCI_D, d);
    viterbi_tb(d, DCI_D, bits);
    calc = crc16(bits, DCI_D - 16) ^ mask;
    rx = 0; for (i = 0; i < 16; ++i) rx = (rx << 1) | (unsigned)(bits[DCI_D - 16 + i] & 1);
    ok = (rx == calc);
    if (ok) for (i = 0; i < DCI_D - 16; ++i) dci[i] = bits[i];
    free(c); free(llr); free(d);
    return ok;
}

/* ---- DCI 1A RIV -> (rb_start, L) --------------------------------------- */
static int riv_bits(int n_rb) {
    int tot = n_rb * (n_rb + 1) / 2 - 1, n = 0;
    while (tot > 0) { ++n; tot >>= 1; }
    return n;
}
static int riv_value(int rb_start, int L, int n_rb) {
    if ((L - 1) <= n_rb / 2) return n_rb * (L - 1) + rb_start;
    return n_rb * (n_rb - L + 1) + (n_rb - 1 - rb_start);
}
static int riv_inv(int v, int n_rb, int *st, int *L) {
    int LL, s;
    for (LL = 1; LL <= n_rb; ++LL)
        for (s = 0; s <= n_rb - LL; ++s)
            if (riv_value(s, LL, n_rb) == v) { *st = s; *L = LL; return 1; }
    return 0;
}

/* ---- public API -------------------------------------------------------- */
int atkdsp_lte_sib1_decode(const atkdsp_cf32 *samples, size_t n, int n_rb,
                           int n_id, int n_ports, int subframe, double cfo_hz,
                           int equalize_on, atkdsp_lte_sib1 *out) {
    const int nfft = nfft_for(n_rb), n_sc = n_rb * SC_PER_RB, n_s = subframe * 2;
    atkdsp_cf32 *grid = NULL, *s = NULL;
    signed char *occ = NULL;
    int *ck = NULL, *cl = NULL;
    int rc = 0, i, cfi, n_cce, found = 0, dci_cce = 0;
    signed char dci[DCI_D];
    static const int SS_AL[2] = {4, 8}, SS_NC[2] = {4, 2};
    if (!samples || !out || !nfft || n_id < 0 || n_id > 503) return ATKDSP_E_ARG;
    memset(out, 0, sizeof *out); out->csg_id = -1;

    /* optional CFO removal at the full-BW sample rate */
    if (cfo_hz != 0.0) {
        const double w = -2.0 * ATK_PI * cfo_hz / ((double)nfft * 15000.0);
        s = (atkdsp_cf32 *)malloc(sizeof(atkdsp_cf32) * n);
        if (!s) return ATKDSP_E_NOMEM;
        for (i = 0; i < (int)n; ++i) {
            double ph = w * i, cr = cos(ph), sr = sin(ph);
            s[i].re = (float)(samples[i].re * cr - samples[i].im * sr);
            s[i].im = (float)(samples[i].re * sr + samples[i].im * cr);
        }
    }
    grid = (atkdsp_cf32 *)malloc(sizeof(atkdsp_cf32) * (size_t)(n_sc * SYMS_SF));
    occ  = (signed char *)malloc((size_t)(n_sc * SYMS_SF));
    if (!grid || !occ) { rc = ATKDSP_E_NOMEM; goto done; }
    rc = ofdm_demod(s ? s : samples, n, n_rb, grid);
    if (rc != ATKDSP_OK) goto done;
    if (equalize_on) { rc = equalize(grid, n_id, n_rb); if (rc != ATKDSP_OK) goto done; }
    rc = 0;

    /* 1) PCFICH -> CFI */
    {
        int pk[16], pl[16]; atkdsp_cf32 pc[16];
        pcfich_res(n_id, n_rb, pk, pl);
        for (i = 0; i < 16; ++i) pc[i] = grid[(size_t)pl[i] * n_sc + pk[i]];
        cfi = pcfich_decode(pc, n_id, n_s);
    }

    /* 2) blind SI-RNTI PDCCH search over the common search space */
    ck = (int *)malloc(sizeof(int) * (size_t)(3 * n_rb * cfi * 4));
    cl = (int *)malloc(sizeof(int) * (size_t)(3 * n_rb * cfi * 4));
    if (!ck || !cl) { rc = ATKDSP_E_NOMEM; goto done; }
    {
        int nreg = control_regs(n_id, n_rb, cfi, n_ports, ck, cl);
        int total_re, ss;
        if (nreg < 0) { rc = ATKDSP_E_NOMEM; goto done; }
        total_re = nreg * 4;
        n_cce = total_re / 36;
        for (ss = 0; ss < 2 && !found; ++ss) {
            int al = SS_AL[ss], m;
            for (m = 0; m < SS_NC[ss]; ++m) {
                int cce = m * al, nsym = al * 36, j;
                atkdsp_cf32 *seg;
                if (cce + al > n_cce) continue;
                seg = (atkdsp_cf32 *)malloc(sizeof(atkdsp_cf32) * (size_t)nsym);
                for (j = 0; j < nsym; ++j) {
                    int reI = cce * 36 + j;
                    seg[j] = grid[(size_t)cl[reI] * n_sc + ck[reI]];
                }
                if (pdcch_candidate(seg, nsym, n_id, n_s, cce, dci)) {
                    found = 1; dci_cce = cce; free(seg); break;
                }
                free(seg);
            }
        }
    }
    if (!found) { rc = 0; goto done; }
    (void)dci_cce;

    /* 3) DCI 1A -> RIV -> allocation */
    {
        int rb = riv_bits(n_rb), v = 0, st, L, K, kidx, decoded = 0;
        int *alloc, na, sym, sub, ecount;
        double *dllr; signed char *cbits; atkdsp_cf32 *pd; signed char *scr;
        for (i = 0; i < rb; ++i) v = (v << 1) | (dci[2 + i] & 1);
        if (!riv_inv(v, n_rb, &st, &L)) { rc = 0; goto done; }

        /* PDSCH RE list for the allocation */
        crs_re_set(n_id, n_rb, n_ports, occ);
        alloc = (int *)malloc(sizeof(int) * (size_t)L);
        for (i = 0; i < L; ++i) alloc[i] = st + i;
        /* count REs */
        ecount = 0;
        for (sym = cfi; sym < SYMS_SF; ++sym)
            for (na = 0; na < L; ++na)
                for (sub = 0; sub < SC_PER_RB; ++sub) {
                    int k = alloc[na] * SC_PER_RB + sub;
                    if (!occ[(size_t)sym * n_sc + k]) ++ecount;
                }
        {
            int E = ecount * 2, e = 0;
            pd = (atkdsp_cf32 *)malloc(sizeof(atkdsp_cf32) * (size_t)ecount);
            for (sym = cfi; sym < SYMS_SF; ++sym)
                for (na = 0; na < L; ++na)
                    for (sub = 0; sub < SC_PER_RB; ++sub) {
                        int k = alloc[na] * SC_PER_RB + sub;
                        if (!occ[(size_t)sym * n_sc + k]) pd[e++] = grid[(size_t)sym * n_sc + k];
                    }
            /* descramble */
            scr = (signed char *)malloc((size_t)E);
            dllr = (double *)malloc(sizeof(double) * (size_t)E);
            gold((unsigned)(((unsigned)SI_RNTI << 14) | ((subframe) << 9) | (unsigned)n_id), E, scr);
            for (i = 0; i < ecount; ++i) {
                dllr[2*i]   = (1 - 2 * scr[2*i])   * pd[i].re;
                dllr[2*i+1] = (1 - 2 * scr[2*i+1]) * pd[i].im;
            }
            /* 4) turbo: CRC-24 gates the transport-block size (ascending K) */
            cbits = NULL;
            {
                float *fllr = (float *)malloc(sizeof(float) * (size_t)E);
                int nq = atk_qpp_count(), hi = (E < 6144) ? E : 6144;
                for (i = 0; i < E; ++i) fllr[i] = (float)dllr[i];
                for (kidx = 0; kidx < nq && !decoded; ++kidx) {
                    K = atk_qpp_k(kidx);
                    if (K < 40 || K > hi) continue;
                    cbits = (signed char *)malloc((size_t)(K - 24));
                    if (atkdsp_lte_turbo_decode(fllr, (size_t)E, K, 0, 8, cbits) == 1) {
                        if (atkdsp_lte_sib1_parse(cbits, K - 24, out) == 1) decoded = 1;
                    }
                    free(cbits); cbits = NULL;
                }
                free(fllr);
            }
            free(pd); free(scr); free(dllr);
        }
        free(alloc);
        rc = decoded ? 1 : 0;
    }

done:
    free(grid); free(occ); free(ck); free(cl); free(s);
    return rc;
}
