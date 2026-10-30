/* ---- 25c. Known codes, primitive polynomials and the parity statistic ------
 * GPS C/A (IS-GPS-200): G1 = x^10 + x^3 + 1, G2 = x^10 + x^9 + x^8 + x^6 +
 * x^3 + x^2 + 1, both seeded with ones; chip = G1 out XOR (G2 taps for the
 * PRN); 1023 chips, +1 for bit 0 (the usual receiver mapping: chip = 1 - 2b).
 * Barker 7, 11, 13 as +-1. IEEE 802.15.4-2006 2.4 GHz O-QPSK: the sixteen
 * 32-chip sequences (table 73), chip 0 first, +1 for a 1.
 *
 * Primitive polynomials of degree r: every polynomial with bits r and 0 set
 * and odd weight whose x has order 2^r - 1 in GF(2)[x]/f — x^(2^r - 1) = 1 mod
 * f and x^((2^r - 1)/q) != 1 for every prime q | 2^r - 1 (r <= 16, so the
 * factorisation is by trial division). The order is the §15 mask order:
 * ascending mask value.
 *
 * The parity statistic Z for a candidate mask over L cyclic chips: the check
 * bits are the mask's set positions; Z = sum over n of (-1)^(parity of the
 * bits at n + those positions mod L). For the true polynomial of weight w
 * and chip error rate p, E Z = L (1 - 2p)^w; for a wrong one E Z = 0, Var Z
 * = L. Chips +1 map to bit 0. */
#include "internal.h"

/* ---- GPS C/A ---------------------------------------------------------------- */
/* G2 tap pairs (1-based register positions) for PRN 1..37 (33..37 are reserved
 * / ground; kept for completeness), and the WAAS/EGNOS PRNs 120..138 given as
 * the G2 delay in chips (IS-GPS-200 table 3-IIA). */
static const int G2_TAPS[37][2] = {
    {2, 6}, {3, 7}, {4, 8}, {5, 9}, {1, 9}, {2, 10}, {1, 8}, {2, 9}, {3, 10}, {2, 3},
    {3, 4}, {5, 6}, {6, 7}, {7, 8}, {8, 9}, {9, 10}, {1, 4}, {2, 5}, {3, 6}, {4, 7},
    {5, 8}, {6, 9}, {1, 3}, {4, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 6}, {2, 7},
    {3, 8}, {4, 9}, {5, 10}, {4, 10}, {1, 7}, {2, 8}, {4, 10}};
static const int SBAS_DELAY[19] = {   /* PRN 120..138: G2 delay, chips */
    145, 175, 52, 21, 237, 235, 886, 657, 634, 762, 355, 1012, 176, 603, 130, 359, 595, 68, 386};

ptrdiff_t atkdsp_gps_ca(int prn, int8_t *chips, size_t cap) {
    if (!chips) return ATKDSP_E_ARG;
    if (cap < 1023) return ATKDSP_E_CAP;
    int g1[10], g2[10];
    for (int i = 0; i < 10; i++) g1[i] = g2[i] = 1;
    int tap1 = 0, tap2 = 0, delay = -1;
    if (prn >= 1 && prn <= 37) { tap1 = G2_TAPS[prn - 1][0]; tap2 = G2_TAPS[prn - 1][1]; }
    else if (prn >= 120 && prn <= 138) delay = SBAS_DELAY[prn - 120];
    else return ATKDSP_E_ARG;
    /* with a delay, generate G2's full sequence first and read it delayed */
    int8_t g2seq[1023];
    if (delay >= 0) {
        int r[10]; for (int i = 0; i < 10; i++) r[i] = 1;
        for (int n = 0; n < 1023; n++) {
            g2seq[n] = (int8_t)r[9];
            const int fb = r[1] ^ r[2] ^ r[5] ^ r[7] ^ r[8] ^ r[9];     /* x^10+x^9+x^8+x^6+x^3+x^2+1 */
            for (int i = 9; i > 0; i--) r[i] = r[i - 1];
            r[0] = fb;
        }
    }
    for (int n = 0; n < 1023; n++) {
        const int g1out = g1[9];
        int g2out;
        if (delay >= 0) g2out = g2seq[(n - delay + 1023) % 1023];
        else g2out = g2[tap1 - 1] ^ g2[tap2 - 1];
        chips[n] = (int8_t)((g1out ^ g2out) ? -1 : 1);
        const int f1 = g1[2] ^ g1[9];                                     /* x^10 + x^3 + 1 */
        const int f2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9];
        for (int i = 9; i > 0; i--) { g1[i] = g1[i - 1]; g2[i] = g2[i - 1]; }
        g1[0] = f1; g2[0] = f2;
    }
    return 1023;
}

/* ---- Barker ----------------------------------------------------------------- */
ptrdiff_t atkdsp_barker(int length, int8_t *chips, size_t cap) {
    static const int8_t b7[7] = {1, 1, 1, -1, -1, 1, -1};
    static const int8_t b11[11] = {1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1};
    static const int8_t b13[13] = {1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1};
    const int8_t *b;
    if (!chips) return ATKDSP_E_ARG;
    if (length == 7) b = b7; else if (length == 11) b = b11; else if (length == 13) b = b13;
    else return ATKDSP_E_ARG;
    if (cap < (size_t)length) return ATKDSP_E_CAP;
    memcpy(chips, b, (size_t)length);
    return length;
}

/* ---- IEEE 802.15.4 ----------------------------------------------------------- */
/* Table 73, chip (c0 c1 ... c31) as a 32-bit value with c0 in bit 0. */
static const uint32_t IEEE_CHIPS[16] = {
    0x744AC39Bu, 0x44AC39B7u, 0x4AC39B74u, 0xAC39B744u, 0xC39B744Au, 0x39B744ACu, 0x9B744AC3u, 0xB744AC39u,
    0xDEE06931u, 0xEE06931Du, 0xE06931DEu, 0x06931DEEu, 0x6931DEE0u, 0x931DEE06u, 0x31DEE069u, 0x1DEE0693u};

ptrdiff_t atkdsp_ieee802154_chips(int symbol, int8_t *chips, size_t cap) {
    if (!chips || symbol < 0 || symbol > 15) return ATKDSP_E_ARG;
    if (cap < 32) return ATKDSP_E_CAP;
    const uint32_t v = IEEE_CHIPS[symbol];
    for (int i = 0; i < 32; i++) chips[i] = (int8_t)(((v >> i) & 1u) ? 1 : -1);
    return 32;
}

/* ---- GF(2)[x] arithmetic for primitivity -------------------------------------- */
static uint32_t pmulmod(uint32_t a, uint32_t b, uint32_t f, int r) {
    uint32_t res = 0;
    for (int i = 0; i < r; i++) {
        if ((b >> i) & 1u) res ^= a;
        a <<= 1;
        if ((a >> r) & 1u) a ^= f;
    }
    return res;
}
static uint32_t ppowmod(uint32_t base, uint32_t e, uint32_t f, int r) {
    uint32_t res = 1;
    while (e) {
        if (e & 1u) res = pmulmod(res, base, f, r);
        base = pmulmod(base, base, f, r);
        e >>= 1;
    }
    return res;
}
static int is_primitive(uint32_t f, int r) {
    const uint32_t order = (1u << r) - 1u;
    if (ppowmod(2u, order, f, r) != 1u) return 0;         /* x^(2^r-1) must be 1 */
    uint32_t m = order;
    for (uint32_t q = 2; q * q <= m; q++) {
        if (m % q) continue;
        while (m % q == 0) m /= q;
        if (ppowmod(2u, order / q, f, r) == 1u) return 0;
    }
    if (m > 1 && ppowmod(2u, order / m, f, r) == 1u) return 0;
    return 1;
}

ptrdiff_t atkdsp_primitive_polys(int degree, uint64_t *out, size_t cap) {
    if (degree < 2 || degree > 16) return ATKDSP_E_ARG;
    const uint32_t hi = 1u << degree;
    size_t n = 0;
    for (uint32_t f = hi | 1u; f < (hi << 1); f += 2u) {
        if (!(atk_popcount64(f) & 1)) continue;             /* even weight is divisible by x + 1 */
        if (!is_primitive(f, degree)) continue;
        if (out && n < cap) out[n] = f;
        n++;
    }
    return (ptrdiff_t)n;
}

double atkdsp_parity_z(const int8_t *chips, int L, uint64_t poly, int degree) {
    if (!chips || L < 2 || degree < 1 || degree > 63) return 0.0;
    int taps[64], nt = 0;
    for (int i = 0; i <= degree; i++) if ((poly >> i) & 1u) taps[nt++] = i;
    double z = 0.0;
    for (int n = 0; n < L; n++) {
        int par = 0;
        for (int t = 0; t < nt; t++) par ^= (chips[(n + taps[t]) % L] < 0);
        z += par ? -1.0 : 1.0;
    }
    return z;
}
