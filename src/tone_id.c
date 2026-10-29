/* 24. Analog radio IDs (docs/ATKDSP_PLAN.md §C3): CTCSS, DCS, DTMF and a
 * tone tracker on demodulated FM audio. Definitions in atkdsp.h §24; the
 * twin is python/atkdsp/reference.py §24 (class ToneId).
 *
 * FOUR DETECTORS, ONE CLOCK. Every detector works on frames cut on the
 * input sample index since reset (never on the caller's blocks), so the
 * events come out the same however the audio is chunked (header rule 3):
 *   - the sub-audio chain: a Kaiser low-pass (300 Hz pass, 500 Hz stop,
 *     60 dB; atkdsp_design_lowpass, so the twin designs the same taps)
 *     decimated by floor(fs / 1000) to about 1 kHz; CTCSS reads it in
 *     100 ms Hann segments, DCS keeps its last second;
 *   - DTMF: 25 ms Hann frames every 10 ms at the full rate;
 *   - the tone tracker: 20 ms Hann frames every 10 ms, an FFT.
 * A DFT at a nominal frequency is refined by the phase advance between
 * consecutive frames (unambiguous to +-1 / (2 hop): 5 Hz for CTCSS's 100 ms,
 * 50 Hz for DTMF's 10 ms), which is what makes the measured frequency a
 * fingerprint feature rather than a bin number.
 *
 * DCS. The code is sent as a 23-bit Golay(23,12) word at 134.4 Bd, bit 0
 * first, repeated without a gap: the 9 code bits, the fixed bits 1,0,0
 * (bits 9..11, "001" in transmission order), then 11 check bits from the
 * generator 0xC75 (x^11 + x^10 + x^6 + x^5 + x^4 + x^2 + 1 — Hank Wallace's
 * routine, the one every DCS implementation uses). The Golay code is
 * cyclic, so every rotation of a valid word is valid and so is its
 * complement: the marker bits and the standard table settle the rotation
 * (no two of the 104 standard codes collide in the same polarity — checked
 * over all 23 rotations), but the polarity cannot be settled from the air,
 * because each standard code's complement is another standard code rotated
 * (023 normal IS 047 inverted). The decoder reads normal polarity and
 * reports the inverted alias beside it. Up to three errors are correctable
 * (the code is perfect: every 11-bit syndrome is a unique pattern of weight
 * <= 3); dcs_max_errors says how many to accept.
 *
 * The thresholds are stated with their reasons beside each one below.
 */
#include "internal.h"

#include <stdio.h>

/* ---- tables ----------------------------------------------------------------- */
#define N_CTCSS 50
static const double CTCSS_TONES[N_CTCSS] = {
    67.0, 69.3, 71.9, 74.4, 77.0, 79.7, 82.5, 85.4, 88.5, 91.5, 94.8, 97.4, 100.0, 103.5,
    107.2, 110.9, 114.8, 118.8, 123.0, 127.3, 131.8, 136.5, 141.3, 146.2, 151.4, 156.7,
    159.8, 162.2, 165.5, 167.9, 171.3, 173.8, 177.3, 179.9, 183.5, 186.2, 189.9, 192.8,
    196.6, 199.5, 203.5, 206.5, 210.7, 218.1, 225.7, 229.1, 233.6, 241.8, 250.3, 254.1};
#define DCS_TURNOFF_HZ 134.4
#define N_DCS 104
static const int DCS_CODES[N_DCS] = {   /* octal */
    023, 025, 026, 031, 032, 036, 043, 047, 051, 053, 054, 065, 071, 072, 073, 074, 0114, 0115,
    0116, 0122, 0125, 0131, 0132, 0134, 0143, 0145, 0152, 0155, 0156, 0162, 0165, 0172, 0174,
    0205, 0212, 0223, 0225, 0226, 0243, 0244, 0245, 0246, 0251, 0252, 0255, 0261, 0263, 0265,
    0266, 0271, 0274, 0306, 0311, 0315, 0325, 0331, 0332, 0343, 0346, 0351, 0356, 0364, 0365,
    0371, 0411, 0412, 0413, 0423, 0431, 0432, 0445, 0446, 0452, 0454, 0455, 0462, 0464, 0465,
    0466, 0503, 0506, 0516, 0523, 0526, 0532, 0546, 0565, 0606, 0612, 0624, 0627, 0631, 0632,
    0654, 0662, 0664, 0703, 0712, 0723, 0731, 0732, 0734, 0743, 0754};
#define DCS_POLY 0xC75u
#define DCS_RATE 134.4
static const double DTMF_LOW[4] = {697.0, 770.0, 852.0, 941.0};
static const double DTMF_HIGH[4] = {1209.0, 1336.0, 1477.0, 1633.0};
static const char DTMF_KEYS[17] = "123A456B789C*0#D";

int atkdsp_ctcss_tone(int index, double *hz) {
    if (index < 0 || index >= N_CTCSS || !hz) return ATKDSP_E_ARG;
    *hz = CTCSS_TONES[index];
    return ATKDSP_OK;
}
int atkdsp_dcs_code(int index, int *octal) {
    if (index < 0 || index >= N_DCS || !octal) return ATKDSP_E_ARG;
    *octal = DCS_CODES[index];
    return ATKDSP_OK;
}

/* ---- Golay(23,12), the DCS way ------------------------------------------------- */
static uint32_t golay_syndrome(uint32_t w, int nbits) {
    uint32_t c = w;
    for (int i = 0; i < nbits; ++i) {
        if (c & 1u) c ^= DCS_POLY;
        c >>= 1;
    }
    return c & 0x7FFu;
}

uint32_t atkdsp_dcs_encode(int octal) {
    const uint32_t data = 04000u | ((uint32_t)octal & 0x1FFu);
    return (golay_syndrome(data, 12) << 12) | data;
}

static int dcs_in_table(int code) {
    for (int i = 0; i < N_DCS; ++i) if (DCS_CODES[i] == code) return 1;
    return 0;
}

/* syndrome -> error pattern of weight <= 3 (2047 patterns + the zero one) */
static uint32_t *dcs_table(void) {
    static uint32_t tab[2048];
    static int built = 0;
    if (!built) {
        memset(tab, 0, sizeof tab);
        for (int a = 0; a < 23; ++a) {
            tab[golay_syndrome(1u << a, 23)] = 1u << a;
            for (int b = a + 1; b < 23; ++b) {
                tab[golay_syndrome((1u << a) | (1u << b), 23)] = (1u << a) | (1u << b);
                for (int c = b + 1; c < 23; ++c) {
                    const uint32_t e = (1u << a) | (1u << b) | (1u << c);
                    tab[golay_syndrome(e, 23)] = e;
                }
            }
        }
        built = 1;
    }
    return tab;
}

/* The standard code a 23-bit word reads as (after correction), or -1. */
static int dcs_read(uint32_t w) {
    if ((w & 0xE00u) != 0x800u) return -1;
    const int code = (int)(w & 0x1FFu);
    return dcs_in_table(code) ? code : -1;
}

int atkdsp_dcs_decode(uint32_t word, int max_errors, int *code, int *alias) {
    if (max_errors < 0 || max_errors > 3) return ATKDSP_E_ARG;
    const uint32_t *tab = dcs_table();
    word &= 0x7FFFFFu;
    const uint32_t s = golay_syndrome(word, 23);
    const uint32_t e = s ? tab[s] : 0u;
    const int nerr = atk_popcount64(e);
    if (code) *code = -1;
    if (alias) *alias = -1;
    if (s && !e) return ATKDSP_E_DECODE;            /* cannot happen for a perfect code */
    if (nerr > max_errors) return ATKDSP_E_DECODE;
    const uint32_t w = word ^ e;
    /* the normal reading: some rotation of w carries the marker and a
     * standard code (at most one does); the inverted reading likewise of ~w */
    int c = -1, a = -1;
    for (int k = 0; k < 23 && c < 0; ++k) {
        const uint32_t r = ((w << k) | (w >> (23 - k))) & 0x7FFFFFu;
        c = dcs_read(r);
    }
    const uint32_t wi = ~w & 0x7FFFFFu;
    for (int k = 0; k < 23 && a < 0; ++k) {
        const uint32_t r = ((wi << k) | (wi >> (23 - k))) & 0x7FFFFFu;
        a = dcs_read(r);
    }
    if (code) *code = c;
    if (alias) *alias = a;
    return c >= 0 ? nerr : ATKDSP_E_DECODE;
}

/* ---- the handle ------------------------------------------------------------------ */
typedef struct { double re, im; } cplx;

struct atkdsp_tone_id {
    atkdsp_tone_config cfg;
    double fs, t0;
    uint64_t idx;                    /* input samples since reset                  */
    /* the sub-audio chain */
    float *lp;  int lp_n;            /* taps                                        */
    float *lp_hist; int lp_pos;      /* ring of lp_n inputs                         */
    unsigned decim; int lp_phase;    /* outputs every decim inputs                  */
    double fs_sub;
    uint64_t sub_idx;                /* sub-audio samples produced                  */
    /* CTCSS */
    int cs_L; float *cs_seg; int cs_pos; double *cs_w; double cs_wsum;
    double cs_c2[N_CTCSS + 1], cs_cw_re[N_CTCSS + 1], cs_cw_im[N_CTCSS + 1], cs_fin_re[N_CTCSS + 1], cs_fin_im[N_CTCSS + 1];
    cplx cs_prev[N_CTCSS + 1];       /* the previous segment's phasors              */
    int cs_have_prev;
    uint64_t cs_segs;                /* segments completed                          */
    /* CTCSS tracking */
    int cs_state;                    /* 0 none, 1 pending, 2 open                   */
    int cs_count, cs_miss;
    double cs_f_sum, cs_lvl_sum, cs_snr_sum, cs_f_last;
    uint64_t cs_seg0;                /* first segment of the run                    */
    uint64_t cs_seg_last;
    int cs_reverse, cs_turnoff, cs_bin;
    double cs_resid_prev;            /* the previous segment's phase residual      */
    uint64_t turnoff_seg;            /* 1 + the segment the 134.4 Hz tone was last seen in */
    /* DCS */
    float *dcs_ring; int dcs_M, dcs_head; uint64_t dcs_filled;
    double dcs_dc, dcs_alpha;
    int dcs_step, dcs_count;         /* search every dcs_step sub samples           */
    int dcs_state;                   /* 0 none, 1 open                              */
    int dcs_code, dcs_alias, dcs_miss, dcs_corrected;
    uint64_t dcs_t0_sub, dcs_tlast_sub;
    /* DTMF */
    int dt_W, dt_H; float *dt_buf; int dt_pos; double *dt_w; double dt_wsum, dt_wsum2;
    double dt_c2[16], dt_cw_re[16], dt_cw_im[16], dt_fin_re[16], dt_fin_im[16];
    cplx dt_prev[8]; int dt_have_prev;
    int dt_state, dt_key, dt_count, dt_miss;
    double dt_f1_sum, dt_f2_sum, dt_lvl_sum, dt_snr_sum;
    uint64_t dt_frame0, dt_frame_last, dt_frames;
    /* tone tracker */
    int tk_W, tk_H, tk_nfft; float *tk_buf; int tk_pos; double *tk_w; double tk_wsum;
    atkdsp_cf32 *tk_in, *tk_out; atkdsp_fft *tk_plan;
    int tk_state, tk_count;
    double tk_f_run, tk_f_sum, tk_lvl_sum, tk_snr_sum;
    uint64_t tk_frame0, tk_frame_last, tk_frames;
    /* output */
    atkdsp_tone_event *ev; size_t cap, got;
    atkdsp_tone_counters st;
};

void atkdsp_tone_id_defaults(atkdsp_tone_config *c) {
    if (!c) return;
    memset(c, 0, sizeof *c);
    c->fs = 8000.0; c->enable = ATKDSP_TONE_EN_ALL; c->dcs_max_errors = 1;
    c->ctcss_min_snr_db = 10.0; c->dtmf_min_dbfs = -45.0; c->tone_min_dbfs = -45.0;
    c->tone_min_snr_db = 10.0;
}

static int config_ok(const atkdsp_tone_config *c) {
    if (!(c->fs >= 6000.0 && c->fs <= 96000.0)) return 0;
    if (c->enable < 1 || c->enable > ATKDSP_TONE_EN_ALL) return 0;
    if (c->dcs_max_errors < 0 || c->dcs_max_errors > 3) return 0;
    if (!(c->ctcss_min_snr_db >= 0.0 && c->ctcss_min_snr_db <= 60.0)) return 0;
    if (!(c->dtmf_min_dbfs >= -120.0 && c->dtmf_min_dbfs <= 0.0)) return 0;
    if (!(c->tone_min_dbfs >= -120.0 && c->tone_min_dbfs <= 0.0)) return 0;
    if (!(c->tone_min_snr_db >= 0.0 && c->tone_min_snr_db <= 60.0)) return 0;
    return 1;
}

static void goertzel_coefs(double omega, int L, double *c2, double *cw_re, double *cw_im,
                           double *fin_re, double *fin_im) {
    *c2 = 2.0 * cos(omega);
    *cw_re = cos(omega); *cw_im = -sin(omega);
    *fin_re = cos(omega * (double)(L - 1)); *fin_im = -sin(omega * (double)(L - 1));
}

/* The DFT of v (L reals, already windowed) at a Goertzel coefficient set. */
static cplx goertzel(const double *v, int L, double c2, double cw_re, double cw_im,
                     double fin_re, double fin_im) {
    double s1 = 0.0, s2 = 0.0;
    for (int n = 0; n < L; ++n) {
        const double s0 = v[n] + c2 * s1 - s2;
        s2 = s1; s1 = s0;
    }
    const double xr = s1 - cw_re * s2, xi = -cw_im * s2;
    cplx out = { xr * fin_re - xi * fin_im, xr * fin_im + xi * fin_re };
    return out;
}

static double *hann_periodic(int L, double *wsum, double *wsum2) {
    double *w = (double *)malloc((size_t)L * sizeof *w);
    if (!w) return NULL;
    double s = 0.0, s2 = 0.0;
    for (int n = 0; n < L; ++n) {
        w[n] = 0.5 * (1.0 - cos(2.0 * ATK_PI * (double)n / (double)L));
        s += w[n]; s2 += w[n] * w[n];
    }
    *wsum = s;
    if (wsum2) *wsum2 = s2;
    return w;
}

static double median_of(double *a, int n) {         /* destroys a; n >= 1 */
    for (int i = 1; i < n; ++i) {                    /* insertion sort: n <= 64 here */
        const double x = a[i]; int j = i;
        while (j > 0 && a[j - 1] > x) { a[j] = a[j - 1]; --j; }
        a[j] = x;
    }
    return (n & 1) ? a[n / 2] : 0.5 * (a[n / 2 - 1] + a[n / 2]);
}

static double wrap_pi(double x) {
    while (x > ATK_PI) x -= 2.0 * ATK_PI;
    while (x < -ATK_PI) x += 2.0 * ATK_PI;
    return x;
}

static void emit(atkdsp_tone_id *h, int kind, int type, double t0, double t1, double f1, double f2,
                 double level_db, double snr_db, int code, int alias, uint32_t flags) {
    h->st.events++;
    if (h->got >= h->cap) { h->st.dropped++; return; }
    atkdsp_tone_event *e = &h->ev[h->got++];
    memset(e, 0, sizeof *e);
    e->t0 = t0; e->t1 = t1; e->freq_hz = f1; e->freq2_hz = f2;
    e->level_db = (float)level_db; e->snr_db = (float)snr_db;
    e->kind = kind; e->type = type; e->code = code; e->alias = alias; e->flags = flags;
    switch (kind) {
    case ATKDSP_TONE_CTCSS: snprintf(e->text, sizeof e->text, "CTCSS %.1f", f1); break;
    case ATKDSP_TONE_DCS:   snprintf(e->text, sizeof e->text, "DCS %03oN", code); break;
    case ATKDSP_TONE_DTMF:  snprintf(e->text, sizeof e->text, "DTMF %c", (char)code); break;
    default:                snprintf(e->text, sizeof e->text, "TONE %.0f", f1); break;
    }
}

/* ---- create / destroy ---------------------------------------------------------- */
void atkdsp_tone_id_destroy(atkdsp_tone_id *h) {
    if (!h) return;
    free(h->lp); free(h->lp_hist);
    free(h->cs_seg); free(h->cs_w);
    free(h->dcs_ring);
    free(h->dt_buf); free(h->dt_w);
    free(h->tk_buf); free(h->tk_w); free(h->tk_in); free(h->tk_out);
    if (h->tk_plan) atkdsp_fft_destroy(h->tk_plan);
    free(h);
}

atkdsp_tone_id *atkdsp_tone_id_create(const atkdsp_tone_config *cfg, double t0) {
    if (!cfg || !config_ok(cfg) || !isfinite(t0)) return NULL;
    atkdsp_tone_id *h = (atkdsp_tone_id *)calloc(1, sizeof *h);
    if (!h) return NULL;
    h->cfg = *cfg; h->fs = cfg->fs; h->t0 = t0;
    const double fs = cfg->fs;
    /* the sub-audio chain */
    h->decim = (unsigned)floor(fs / 1000.0);
    h->fs_sub = fs / (double)h->decim;
    const ptrdiff_t nt = atkdsp_design_lowpass(270.0, 330.0, 60.0, fs, NULL, 0);
    if (nt <= 0) { atkdsp_tone_id_destroy(h); return NULL; }
    h->lp_n = (int)nt;
    h->lp = (float *)malloc((size_t)nt * sizeof(float));
    h->lp_hist = (float *)calloc((size_t)nt, sizeof(float));
    if (!h->lp || !h->lp_hist || atkdsp_design_lowpass(270.0, 330.0, 60.0, fs, h->lp, (size_t)nt) != nt) {
        atkdsp_tone_id_destroy(h); return NULL;
    }
    /* CTCSS: 100 ms segments */
    h->cs_L = (int)floor(0.1 * h->fs_sub + 0.5);
    h->cs_seg = (float *)malloc((size_t)h->cs_L * sizeof(float));
    h->cs_w = hann_periodic(h->cs_L, &h->cs_wsum, NULL);
    if (!h->cs_seg || !h->cs_w) { atkdsp_tone_id_destroy(h); return NULL; }
    for (int i = 0; i <= N_CTCSS; ++i) {
        const double f = i < N_CTCSS ? CTCSS_TONES[i] : DCS_TURNOFF_HZ;
        goertzel_coefs(2.0 * ATK_PI * f / h->fs_sub, h->cs_L, &h->cs_c2[i], &h->cs_cw_re[i],
                       &h->cs_cw_im[i], &h->cs_fin_re[i], &h->cs_fin_im[i]);
    }
    /* DCS: the last second, a search every 250 ms */
    h->dcs_M = (int)floor(1.0 * h->fs_sub + 0.5);
    h->dcs_step = (int)floor(0.25 * h->fs_sub + 0.5);
    h->dcs_ring = (float *)calloc((size_t)h->dcs_M, sizeof(float));
    if (!h->dcs_ring) { atkdsp_tone_id_destroy(h); return NULL; }
    h->dcs_alpha = 1.0 / (2.0 * h->fs_sub);            /* a 2 s DC tracker: the code has runs of 11 bits */
    h->dcs_code = h->dcs_alias = -1;
    /* DTMF: 20 ms frames every 10 ms (a 40 ms digit, Q.24's minimum, holds
     * three whole frames) */
    h->dt_W = (int)floor(0.02 * fs + 0.5);
    h->dt_H = (int)floor(0.01 * fs + 0.5);
    h->dt_buf = (float *)calloc((size_t)h->dt_W, sizeof(float));
    h->dt_w = hann_periodic(h->dt_W, &h->dt_wsum, &h->dt_wsum2);
    if (!h->dt_buf || !h->dt_w) { atkdsp_tone_id_destroy(h); return NULL; }
    for (int i = 0; i < 16; ++i) {
        const double f = (i < 4 ? DTMF_LOW[i] : i < 8 ? DTMF_HIGH[i - 4]
                          : i < 12 ? 2.0 * DTMF_LOW[i - 8] : 2.0 * DTMF_HIGH[i - 12]);
        goertzel_coefs(2.0 * ATK_PI * f / fs, h->dt_W, &h->dt_c2[i], &h->dt_cw_re[i], &h->dt_cw_im[i],
                       &h->dt_fin_re[i], &h->dt_fin_im[i]);
    }
    h->dt_key = -1;
    /* the tone tracker: 20 ms frames every 10 ms, an FFT */
    h->tk_W = (int)floor(0.02 * fs + 0.5);
    h->tk_H = h->dt_H;
    int nf = 1; while (nf < h->tk_W) nf <<= 1;
    h->tk_nfft = nf;
    h->tk_buf = (float *)calloc((size_t)h->tk_W, sizeof(float));
    h->tk_w = hann_periodic(h->tk_W, &h->tk_wsum, NULL);
    h->tk_in = (atkdsp_cf32 *)calloc((size_t)nf, sizeof *h->tk_in);
    h->tk_out = (atkdsp_cf32 *)calloc((size_t)nf, sizeof *h->tk_out);
    h->tk_plan = atkdsp_fft_create((size_t)nf);
    if (!h->tk_buf || !h->tk_w || !h->tk_in || !h->tk_out || !h->tk_plan) { atkdsp_tone_id_destroy(h); return NULL; }
    return h;
}

void atkdsp_tone_id_reset(atkdsp_tone_id *h, double t0) {
    if (!h) return;
    h->t0 = isfinite(t0) ? t0 : 0.0;
    h->idx = 0; h->sub_idx = 0;
    memset(h->lp_hist, 0, (size_t)h->lp_n * sizeof(float)); h->lp_pos = 0; h->lp_phase = 0;
    h->cs_pos = 0; h->cs_have_prev = 0; h->cs_segs = 0; h->cs_state = 0; h->cs_count = h->cs_miss = 0;
    h->cs_reverse = h->cs_turnoff = 0; h->cs_resid_prev = 0.0; h->turnoff_seg = 0;
    memset(h->dcs_ring, 0, (size_t)h->dcs_M * sizeof(float)); h->dcs_head = 0; h->dcs_filled = 0;
    h->dcs_dc = 0.0; h->dcs_count = 0; h->dcs_state = 0; h->dcs_code = h->dcs_alias = -1; h->dcs_miss = 0;
    h->dt_pos = 0; h->dt_have_prev = 0; h->dt_state = 0; h->dt_key = -1; h->dt_count = h->dt_miss = 0; h->dt_frames = 0;
    h->tk_pos = 0; h->tk_state = 0; h->tk_count = 0; h->tk_frames = 0;
    h->st.resets++;
}

size_t atkdsp_tone_id_events_max(const atkdsp_tone_id *h, size_t n) {
    if (!h) return 0;
    const size_t frames = n / (size_t)h->dt_H + 2;
    const size_t segs = n / ((size_t)h->cs_L * h->decim) + 2;
    const size_t srch = n / ((size_t)h->dcs_step * h->decim) + 2;
    return 4 * frames + 2 * segs + 2 * srch;
}

/* ---- CTCSS ------------------------------------------------------------------------ */
/* Sub-audio sample m is the filter's output at input (m + 1) decim - 1 and
 * represents the input (lp_n - 1) / 2 samples earlier (the FIR's delay). */
static double sub_time(const atkdsp_tone_id *h, double m) {
    const double in_idx = (m + 1.0) * (double)h->decim - 1.0 - 0.5 * (double)(h->lp_n - 1);
    return h->t0 + in_idx / h->fs;
}
static double cs_seg_time(const atkdsp_tone_id *h, uint64_t seg) {
    return sub_time(h, (double)seg * (double)h->cs_L);
}

static void cs_end(atkdsp_tone_id *h, uint64_t end_seg, uint32_t flags) {
    const double f = h->cs_f_sum / (double)h->cs_count;
    int code = -1;
    double best = 1.0;                                /* within 1 Hz of a standard tone */
    for (int i = 0; i < N_CTCSS; ++i) {
        const double d = fabs(f - CTCSS_TONES[i]);
        if (d < best) { best = d; code = i; }
    }
    if (code < 0) flags |= ATKDSP_TONE_F_NONSTANDARD;
    if (h->cs_reverse) flags |= ATKDSP_TONE_F_REVERSE_BURST;
    if (h->cs_turnoff || (h->turnoff_seg && h->turnoff_seg - 1 >= h->cs_seg_last
                          && h->turnoff_seg - 1 <= h->cs_seg_last + 5)) flags |= ATKDSP_TONE_F_TURNOFF;
    emit(h, ATKDSP_TONE_CTCSS, ATKDSP_TONE_END, cs_seg_time(h, h->cs_seg0), cs_seg_time(h, end_seg),
         f, 0.0, h->cs_lvl_sum / h->cs_count, h->cs_snr_sum / h->cs_count, code, -1, flags);
    h->cs_state = 0; h->cs_count = 0; h->cs_miss = 0; h->cs_reverse = h->cs_turnoff = 0;
    h->cs_resid_prev = 0.0;
}

static void cs_segment(atkdsp_tone_id *h) {
    const int L = h->cs_L;
    double v[1024];                                  /* L <= ~117 at fs_sub <= 1166 */
    double mean = 0.0;
    for (int n = 0; n < L; ++n) mean += h->cs_seg[n];
    mean /= (double)L;
    for (int n = 0; n < L; ++n) v[n] = (h->cs_seg[n] - mean) * h->cs_w[n];
    cplx p[N_CTCSS + 1];
    double mag2[N_CTCSS + 1];
    int peak = -1;
    double pm = -1.0;
    for (int i = 0; i <= N_CTCSS; ++i) {
        p[i] = goertzel(v, L, h->cs_c2[i], h->cs_cw_re[i], h->cs_cw_im[i], h->cs_fin_re[i], h->cs_fin_im[i]);
        mag2[i] = p[i].re * p[i].re + p[i].im * p[i].im;
        if (mag2[i] > pm) { pm = mag2[i]; peak = i; }
    }
    /* the floor: the median of the other bins' power, the peak's +-2 left out */
    double tmp[N_CTCSS + 1];
    int nt = 0;
    for (int i = 0; i <= N_CTCSS; ++i) if (i < peak - 2 || i > peak + 2) tmp[nt++] = mag2[i];
    const double floor_ = median_of(tmp, nt);
    const double snr_db = 10.0 * log10((pm + 1e-30) / (floor_ + 1e-30));
    const double level_db = 20.0 * log10(2.0 * sqrt(pm) / h->cs_wsum + 1e-30);
    const double t_seg = (double)L / h->fs_sub;
    /* a tone is a LINE: the bins within 12 Hz of the peak (the Hann lobe)
     * must hold 70 % of the power of the bins within 40 Hz of it. Voice is
     * broadband and a DCS stream is a comb of lines 5.8 Hz apart: both fail
     * this; a voice fundamental 50 Hz away (radios high-pass their audio
     * at 300 Hz, but not all) does not count against the tone. */
    double near = 0.0, total = 0.0;
    const double fpk = peak < N_CTCSS ? CTCSS_TONES[peak] : DCS_TURNOFF_HZ;
    for (int i = 0; i <= N_CTCSS; ++i) {
        const double fi = i < N_CTCSS ? CTCSS_TONES[i] : DCS_TURNOFF_HZ;
        if (fabs(fi - fpk) <= 40.0) total += mag2[i];
        if (fabs(fi - fpk) <= 12.0) near += mag2[i];
    }
    const int line = total > 0.0 && near >= 0.7 * total;
    double f_meas = 0.0;
    int have = 0;
    if (line && snr_db >= h->cfg.ctcss_min_snr_db && h->cs_have_prev) {
        const double f0 = peak < N_CTCSS ? CTCSS_TONES[peak] : DCS_TURNOFF_HZ;
        const cplx q = h->cs_prev[peak];
        const double dr = p[peak].re * q.re + p[peak].im * q.im, di = p[peak].im * q.re - p[peak].re * q.im;
        const double adv = wrap_pi(atan2(di, dr) - 2.0 * ATK_PI * f0 * t_seg);
        f_meas = f0 + adv / (2.0 * ATK_PI * t_seg);
        have = 1;
    }
    const uint64_t seg = h->cs_segs;                 /* this segment's index */
    /* the DCS turn-off tone: the 134.4 Hz bin, read within a hertz of it (a
     * CTCSS tone changing to 131.8 or 136.5 peaks there for a segment too,
     * with a frequency reading that is off) */
    const int turnoff = have && peak == N_CTCSS && fabs(f_meas - DCS_TURNOFF_HZ) <= 1.0;
    if (have && peak == N_CTCSS) have = 0;           /* that bin is not a CTCSS tone */
    if (turnoff) h->turnoff_seg = seg + 1;
    /* a radio sends CTCSS or DCS, not both: while a DCS code is being read
     * (its comb has lines near the low tones) no CTCSS tone opens */
    if (h->dcs_state == 1 && h->cs_state != 2) have = 0;
    if (h->cs_state == 2) {
        if (turnoff) h->cs_turnoff = 1;
        const double f_run = h->cs_f_sum / (double)h->cs_count;
        if (have && fabs(f_meas - f_run) <= 1.5 && fabs(f_meas - h->cs_f_last) <= 0.75) {
            /* the squelch-tail reversal: a phase step the frequency does not explain */
            const cplx q = h->cs_prev[peak];
            const double dr = p[peak].re * q.re + p[peak].im * q.im, di = p[peak].im * q.re - p[peak].re * q.im;
            const double resid = wrap_pi(atan2(di, dr) - 2.0 * ATK_PI * f_run * t_seg);
            /* a 120 degree step (Motorola) landing inside a segment shows as
             * two steps of about 60: the sum over two transitions catches it */
            if (fabs(resid) > 0.5 * ATK_PI || fabs(wrap_pi(resid + h->cs_resid_prev)) > 0.5 * ATK_PI)
                h->cs_reverse = 1;
            h->cs_resid_prev = resid;
            h->cs_count++; h->cs_f_sum += f_meas; h->cs_lvl_sum += level_db; h->cs_snr_sum += snr_db;
            h->cs_seg_last = seg; h->cs_miss = 0; h->cs_f_last = f_meas;
        } else if (have && fabs(f_meas - f_run) <= 12.0) {
            /* a line within the Hann lobe of the tone whose frequency reading
             * jumped: a phase step inside the segment — the squelch-tail
             * reversal (120, 180 or 240 degrees read as +-3.3 or 5 Hz over a
             * 100 ms segment, often with the peak on a neighbouring bin).
             * Counted as a miss, so a tone that really changed ends after
             * the hang; a tone that returns to its frequency goes on. */
            h->cs_reverse = 1;
            h->cs_f_last = f_run;
            if (++h->cs_miss >= 3) cs_end(h, h->cs_seg_last + 1, 0);
        } else {
            h->cs_miss++;
            if (h->cs_miss >= 3 || (have && fabs(f_meas - f_run) > 12.0)) {  /* 0.3 s hang */
                cs_end(h, h->cs_seg_last + 1, 0);
                if (have) { h->cs_state = 1; h->cs_count = 1; h->cs_f_sum = f_meas; h->cs_lvl_sum = level_db;
                            h->cs_snr_sum = snr_db; h->cs_seg0 = seg; h->cs_seg_last = seg; h->cs_f_last = f_meas;
                            h->cs_bin = peak; }
            }
        }
    } else if (h->cs_state == 1) {
        if (have && fabs(f_meas - h->cs_f_last) <= 0.75) {
            h->cs_count++; h->cs_f_sum += f_meas; h->cs_lvl_sum += level_db; h->cs_snr_sum += snr_db;
            h->cs_seg_last = seg; h->cs_f_last = f_meas;
            if (h->cs_count >= 3) {                  /* 0.3 s of a steady line: voice pitch wanders */
                h->cs_state = 2;
                const double f = h->cs_f_sum / h->cs_count;
                int code = -1; double best = 1.0;
                for (int i = 0; i < N_CTCSS; ++i) { const double d = fabs(f - CTCSS_TONES[i]); if (d < best) { best = d; code = i; } }
                emit(h, ATKDSP_TONE_CTCSS, ATKDSP_TONE_START, cs_seg_time(h, h->cs_seg0), NAN, f, 0.0,
                     h->cs_lvl_sum / h->cs_count, h->cs_snr_sum / h->cs_count, code, -1,
                     code < 0 ? ATKDSP_TONE_F_NONSTANDARD : 0u);
                h->st.ctcss++;
            }
        } else if (have) {
            h->cs_count = 1; h->cs_f_sum = f_meas; h->cs_lvl_sum = level_db; h->cs_snr_sum = snr_db;
            h->cs_seg0 = seg; h->cs_seg_last = seg; h->cs_f_last = f_meas; h->cs_bin = peak;
        } else {
            h->cs_state = 0; h->cs_count = 0;
        }
    } else if (have) {
        h->cs_state = 1; h->cs_count = 1; h->cs_f_sum = f_meas; h->cs_lvl_sum = level_db;
        h->cs_snr_sum = snr_db; h->cs_seg0 = seg; h->cs_seg_last = seg; h->cs_f_last = f_meas; h->cs_bin = peak;
    }
    memcpy(h->cs_prev, p, sizeof p);
    h->cs_have_prev = 1;
    h->cs_segs++;
}

/* ---- DCS --------------------------------------------------------------------------- */
static void dcs_end(atkdsp_tone_id *h, uint32_t flags) {
    if (h->dcs_corrected) flags |= ATKDSP_TONE_F_CORRECTED;
    /* a turn-off tone (134.4 Hz) within half a second after the last word */
    if (h->turnoff_seg) {
        const double t_off = cs_seg_time(h, h->turnoff_seg - 1);
        const double t_last = sub_time(h, (double)h->dcs_tlast_sub);
        if (t_off >= t_last - 0.15 && t_off <= t_last + 0.5) flags |= ATKDSP_TONE_F_TURNOFF;
    }
    if (!dcs_in_table(h->dcs_code)) flags |= ATKDSP_TONE_F_NONSTANDARD;
    emit(h, ATKDSP_TONE_DCS, ATKDSP_TONE_END, sub_time(h, (double)h->dcs_t0_sub),
         sub_time(h, (double)h->dcs_tlast_sub), DCS_RATE, 0.0, 0.0, 0.0, h->dcs_code, h->dcs_alias, flags);
    h->dcs_state = 0; h->dcs_code = h->dcs_alias = -1; h->dcs_miss = 0; h->dcs_corrected = 0;
}

/* The search over the last second: bits at eight timing phases, every
 * 23-bit alignment; the reading is the (phase, alignment) whose run of
 * consecutive valid words ending at the newest is longest. */
static void dcs_search(atkdsp_tone_id *h) {
    const int M = h->dcs_M;
    /* a radio sends CTCSS or DCS, not both: a confirmed tone's sign pattern
     * at 134.4 Bd is periodic, and a periodic pattern can pass the Golay
     * check, so no code opens while a tone is being read */
    if (h->cs_state == 2 && h->dcs_state == 0) return;
    const double sps = h->fs_sub / DCS_RATE;
    const uint64_t avail = h->dcs_filled < (uint64_t)M ? h->dcs_filled : (uint64_t)M;
    if (avail < (uint64_t)(2.5 * 23.0 * sps)) return;   /* fewer than two and a half words */
    const int n = (int)avail;
    const int start = (h->dcs_head - n + M) % M;        /* oldest sample in the ring */
    int best_run = 0, best_code = -1, best_alias = -1, best_corr = 0;
    int best_first_bit = 0, best_phase = 0, best_nb = 0;
    uint8_t bits[2048];
    for (int ph = 0; ph < 8; ++ph) {
        const double off = (double)ph / 8.0 * sps;
        int nb = 0;
        for (;;) {
            const int a = (int)floor(off + nb * sps + 0.5), b = (int)floor(off + (nb + 1) * sps + 0.5);
            if (b > n || nb >= 2048) break;
            double s = 0.0;
            for (int i = a; i < b; ++i) s += h->dcs_ring[(start + i) % M];
            bits[nb++] = s > 0.0 ? 1 : 0;
        }
        if (nb < 46) continue;
        for (int al = 0; al < 23; ++al) {
            /* words ending at the newest: the last complete word at this alignment */
            const int nwords = (nb - al) / 23;
            if (nwords < 2) continue;
            int run = 0, code = -1, alias = -1, corr = 0;
            for (int wi = nwords - 1; wi >= 0; --wi) {
                uint32_t w = 0;
                const int b0 = al + wi * 23;
                for (int i = 0; i < 23; ++i) if (bits[b0 + i]) w |= 1u << i;
                int c, a;
                const int rc = atkdsp_dcs_decode(w, h->cfg.dcs_max_errors, &c, &a);
                if (rc < 0 || c < 0) break;
                if (run == 0) { code = c; alias = a; }
                else if (c != code) break;
                if (rc > 0) corr = 1;
                run++;
            }
            if (run > best_run || (run == best_run && run > 0 && corr < best_corr)) {
                best_run = run; best_code = code; best_alias = alias; best_corr = corr;
                best_first_bit = al + (nwords - run) * 23; best_phase = ph; best_nb = nwords;
            }
        }
    }
    (void)best_nb;
    const double sps_ = sps;
    if (best_run >= 2) {
        /* sub-audio index of the run's first bit and of the newest word's end */
        const double first = (double)(h->sub_idx - avail) + (double)best_phase / 8.0 * sps_ + best_first_bit * sps_;
        const double last = (double)(h->sub_idx - avail) + (double)best_phase / 8.0 * sps_
                            + (best_first_bit + best_run * 23) * sps_;
        if (h->dcs_state == 1 && best_code != h->dcs_code) dcs_end(h, 0);
        if (h->dcs_state == 0) {
            h->dcs_state = 1; h->dcs_code = best_code; h->dcs_alias = best_alias;
            h->dcs_t0_sub = (uint64_t)(first > 0.0 ? first : 0.0);
            if (sub_time(h, (double)h->dcs_t0_sub) < h->t0) h->dcs_t0_sub = (uint64_t)ceil(0.5 * (double)(h->lp_n - 1) / (double)h->decim);
            h->dcs_corrected = best_corr;
            emit(h, ATKDSP_TONE_DCS, ATKDSP_TONE_START, sub_time(h, (double)h->dcs_t0_sub), NAN,
                 DCS_RATE, 0.0, 0.0, 0.0, best_code, best_alias,
                 (best_corr ? ATKDSP_TONE_F_CORRECTED : 0u) | (dcs_in_table(best_code) ? 0u : ATKDSP_TONE_F_NONSTANDARD));
            h->st.dcs++;
        }
        if (best_corr) h->dcs_corrected = 1;
        h->dcs_tlast_sub = (uint64_t)last;
        h->dcs_miss = 0;
    } else if (h->dcs_state == 1) {
        if (++h->dcs_miss >= 2) dcs_end(h, 0);
    }
}

static void dcs_push(atkdsp_tone_id *h, float y) {
    h->dcs_dc += h->dcs_alpha * ((double)y - h->dcs_dc);
    h->dcs_ring[h->dcs_head] = (float)((double)y - h->dcs_dc);
    h->dcs_head = (h->dcs_head + 1) % h->dcs_M;
    h->dcs_filled++;
    if (++h->dcs_count >= h->dcs_step) { h->dcs_count = 0; dcs_search(h); }
}

/* ---- DTMF ---------------------------------------------------------------------------- */
/* A frame is identified by the input index of its first sample. */
static double in_time(const atkdsp_tone_id *h, uint64_t in_idx) {
    return h->t0 + (double)in_idx / h->fs;
}

static void dt_end(atkdsp_tone_id *h, uint64_t end_frame, uint32_t flags) {
    emit(h, ATKDSP_TONE_DTMF, ATKDSP_TONE_END, in_time(h, h->dt_frame0),
         in_time(h, end_frame + (uint64_t)h->dt_W), h->dt_f1_sum / h->dt_count,
         h->dt_f2_sum / h->dt_count, h->dt_lvl_sum / h->dt_count, h->dt_snr_sum / h->dt_count,
         DTMF_KEYS[h->dt_key], -1, flags);
    h->dt_state = 0; h->dt_key = -1; h->dt_count = 0; h->dt_miss = 0;
}

static void dt_frame(atkdsp_tone_id *h, uint64_t frame) {   /* frame = its first input index */
    const int W = h->dt_W;
    double v[4096];                                  /* W <= 2400 at 96 kHz */
    double e = 0.0;
    for (int n = 0; n < W; ++n) {
        const int j = (h->dt_pos + n) % W;           /* oldest first */
        v[n] = (double)h->dt_buf[j] * h->dt_w[n];
        e += v[n] * v[n];
    }
    cplx p[16];
    double m2[16];
    for (int i = 0; i < 16; ++i) {
        p[i] = goertzel(v, W, h->dt_c2[i], h->dt_cw_re[i], h->dt_cw_im[i], h->dt_fin_re[i], h->dt_fin_im[i]);
        m2[i] = p[i].re * p[i].re + p[i].im * p[i].im;
    }
    int lo = 0, hi = 4;
    for (int i = 1; i < 4; ++i) if (m2[i] > m2[lo]) lo = i;
    for (int i = 5; i < 8; ++i) if (m2[i] > m2[hi]) hi = i;
    const double a_lo = 2.0 * sqrt(m2[lo]) / h->dt_wsum, a_hi = 2.0 * sqrt(m2[hi]) / h->dt_wsum;
    const double lvl_lo = 20.0 * log10(a_lo + 1e-30), lvl_hi = 20.0 * log10(a_hi + 1e-30);
    int key = -1;
    double f1 = 0.0, f2 = 0.0, snr = 0.0;
    const double t_hop = (double)h->dt_H / h->fs;
    if (lvl_lo >= h->cfg.dtmf_min_dbfs && lvl_hi >= h->cfg.dtmf_min_dbfs && h->dt_have_prev) {
        /* the pair must be the frame: two sinusoids carry (sum w)^2 / (2 sum w^2) of
         * the windowed energy per unit of bin power */
        const double norm = h->dt_wsum * h->dt_wsum / (2.0 * h->dt_wsum2);
        const double share = (m2[lo] + m2[hi]) / (e * norm + 1e-30);
        const double twist = lvl_hi - lvl_lo;
        /* second harmonics within 20 dB of their fundamentals are voice, not
         * DTMF. The low tone's harmonic sits near the high tone for some keys
         * (2 x 697 = 1394 beside 1336; 2 x 770 = 1540 beside 1477) and reads
         * the high tone's leakage, so it is judged only when 150 Hz clear. */
        const int lo_harm_clear = fabs(2.0 * DTMF_LOW[lo] - DTMF_HIGH[hi - 4]) > 150.0;
        const int harm_ok = (!lo_harm_clear || m2[lo + 8] < 0.01 * m2[lo]) && m2[hi + 8] < 0.01 * m2[hi];
        /* each tone's frequency from the frame-to-frame phase advance */
        double fm[2];
        const int bins[2] = {lo, hi};
        const double nom[2] = {DTMF_LOW[lo], DTMF_HIGH[hi - 4]};
        for (int k = 0; k < 2; ++k) {
            const cplx q = h->dt_prev[bins[k]];
            const cplx c = p[bins[k]];
            const double dr = c.re * q.re + c.im * q.im, di = c.im * q.re - c.re * q.im;
            const double adv = wrap_pi(atan2(di, dr) - 2.0 * ATK_PI * nom[k] * t_hop);
            fm[k] = nom[k] + adv / (2.0 * ATK_PI * t_hop);
        }
        const int freq_ok = fabs(fm[0] - nom[0]) <= 0.015 * nom[0] && fabs(fm[1] - nom[1]) <= 0.015 * nom[1];
        if (share >= 0.6 && fabs(twist) <= 8.0 && harm_ok && freq_ok) {
            key = lo * 4 + (hi - 4);
            f1 = fm[0]; f2 = fm[1];
            /* snr: the pair over the other six fundamentals' mean */
            double rest = 0.0;
            for (int i = 0; i < 8; ++i) if (i != lo && i != hi) rest += m2[i];
            snr = 10.0 * log10((m2[lo] + m2[hi]) / (rest / 3.0 + 1e-30));
        }
    }
    const double lvl = lvl_lo < lvl_hi ? lvl_lo : lvl_hi;
    if (h->dt_state == 2) {
        if (key == h->dt_key) {
            h->dt_count++; h->dt_f1_sum += f1; h->dt_f2_sum += f2; h->dt_lvl_sum += lvl; h->dt_snr_sum += snr;
            h->dt_frame_last = frame; h->dt_miss = 0;
        } else if (key < 0 && h->dt_miss == 0) {
            h->dt_miss = 1;                          /* one dropped frame is forgiven (Q.24: 10 ms) */
        } else {
            const int newkey = key;
            dt_end(h, h->dt_frame_last, 0);
            if (newkey >= 0) { h->dt_state = 1; h->dt_key = newkey; h->dt_count = 1; h->dt_f1_sum = f1; h->dt_f2_sum = f2;
                               h->dt_lvl_sum = lvl; h->dt_snr_sum = snr; h->dt_frame0 = frame; h->dt_frame_last = frame; }
        }
    } else if (h->dt_state == 1) {
        if (key == h->dt_key) {
            h->dt_count++; h->dt_f1_sum += f1; h->dt_f2_sum += f2; h->dt_lvl_sum += lvl; h->dt_snr_sum += snr;
            h->dt_frame_last = frame;
            if (h->dt_count >= 3) {                  /* 30 ms of whole frames: a 40 ms digit */
                h->dt_state = 2;
                emit(h, ATKDSP_TONE_DTMF, ATKDSP_TONE_START, in_time(h, h->dt_frame0), NAN,
                     h->dt_f1_sum / h->dt_count, h->dt_f2_sum / h->dt_count, h->dt_lvl_sum / h->dt_count,
                     h->dt_snr_sum / h->dt_count, DTMF_KEYS[h->dt_key], -1, 0u);
                h->st.dtmf++;
            }
        } else if (key >= 0) {
            h->dt_key = key; h->dt_count = 1; h->dt_f1_sum = f1; h->dt_f2_sum = f2; h->dt_lvl_sum = lvl;
            h->dt_snr_sum = snr; h->dt_frame0 = frame; h->dt_frame_last = frame;
        } else {
            h->dt_state = 0; h->dt_key = -1; h->dt_count = 0;
        }
    } else if (key >= 0) {
        h->dt_state = 1; h->dt_key = key; h->dt_count = 1; h->dt_f1_sum = f1; h->dt_f2_sum = f2;
        h->dt_lvl_sum = lvl; h->dt_snr_sum = snr; h->dt_frame0 = frame; h->dt_frame_last = frame;
    }
    for (int i = 0; i < 8; ++i) h->dt_prev[i] = p[i];
    h->dt_have_prev = 1;
    h->dt_frames++;
}

/* ---- the tone tracker ------------------------------------------------------------------ */
static void tk_end(atkdsp_tone_id *h, uint64_t end_frame, uint32_t flags) {
    emit(h, ATKDSP_TONE_TONE, ATKDSP_TONE_END, in_time(h, h->tk_frame0),
         in_time(h, end_frame + (uint64_t)h->tk_W), h->tk_f_sum / h->tk_count, 0.0,
         h->tk_lvl_sum / h->tk_count, h->tk_snr_sum / h->tk_count, -1, -1, flags);
    h->tk_state = 0; h->tk_count = 0;
}

static void tk_frame(atkdsp_tone_id *h, uint64_t frame) {   /* frame = its first input index */
    const int W = h->tk_W, N = h->tk_nfft;
    for (int n = 0; n < W; ++n) {
        const int j = (h->tk_pos + n) % W;
        h->tk_in[n].re = (float)((double)h->tk_buf[j] * h->tk_w[n]);
        h->tk_in[n].im = 0.0f;
    }
    for (int n = W; n < N; ++n) h->tk_in[n].re = h->tk_in[n].im = 0.0f;
    atkdsp_fft_exec(h->tk_plan, h->tk_in, h->tk_out, 0);
    const double bin_hz = h->fs / (double)N;
    int k0 = (int)ceil(250.0 / bin_hz), k1 = (int)floor(3500.0 / bin_hz);
    if (k1 > N / 2 - 2) k1 = N / 2 - 2;
    if (k0 < 2) k0 = 2;
    double m2[4096 / 2 + 1];
    int peak = k0;
    double band = 0.0;
    for (int k = k0; k <= k1; ++k) {
        m2[k] = (double)h->tk_out[k].re * h->tk_out[k].re + (double)h->tk_out[k].im * h->tk_out[k].im;
        band += m2[k];
        if (m2[k] > m2[peak]) peak = k;
    }
    double f = 0.0, lvl = -200.0, snr = 0.0;
    int have = 0;
    if (band > 0.0 && peak > k0 && peak < k1) {
        const double la = 0.5 * log(m2[peak - 1] + 1e-300), lb = 0.5 * log(m2[peak] + 1e-300),
                     lc = 0.5 * log(m2[peak + 1] + 1e-300);
        const double den = la - 2.0 * lb + lc;
        const double d = den != 0.0 ? 0.5 * (la - lc) / den : 0.0;
        const double lpk = lb - 0.25 * (la - lc) * d;      /* log amplitude at the true peak */
        f = ((double)peak + d) * bin_hz;
        lvl = 20.0 * log10(2.0 * exp(lpk) / h->tk_wsum + 1e-30);
        double share = 0.0;
        for (int k = peak - 2; k <= peak + 2; ++k) if (k >= k0 && k <= k1) share += m2[k];
        share /= band;
        /* the floor: the median of the band's bins outside the peak's +-3 */
        double tmp[4096 / 2 + 1];
        int nt = 0;
        for (int k = k0; k <= k1; ++k) if (k < peak - 3 || k > peak + 3) tmp[nt++] = m2[k];
        const double fl = nt ? median_of(tmp, nt) : 0.0;
        snr = 10.0 * log10((m2[peak] + 1e-30) / (fl + 1e-30));
        have = share >= 0.6 && lvl >= h->cfg.tone_min_dbfs && snr >= h->cfg.tone_min_snr_db;
    }
    if (h->tk_state == 2) {
        const double tol = 0.015 * h->tk_f_run > 10.0 ? 0.015 * h->tk_f_run : 10.0;
        if (have && fabs(f - h->tk_f_run) <= tol) {
            h->tk_count++; h->tk_f_sum += f; h->tk_lvl_sum += lvl; h->tk_snr_sum += snr;
            h->tk_f_run = h->tk_f_sum / h->tk_count; h->tk_frame_last = frame;
        } else {
            tk_end(h, h->tk_frame_last, 0);
            if (have) { h->tk_state = 1; h->tk_count = 1; h->tk_f_run = f; h->tk_f_sum = f; h->tk_lvl_sum = lvl;
                        h->tk_snr_sum = snr; h->tk_frame0 = frame; h->tk_frame_last = frame; }
        }
    } else if (h->tk_state == 1) {
        const double tol = 0.015 * h->tk_f_run > 10.0 ? 0.015 * h->tk_f_run : 10.0;
        if (have && fabs(f - h->tk_f_run) <= tol) {
            h->tk_count++; h->tk_f_sum += f; h->tk_lvl_sum += lvl; h->tk_snr_sum += snr;
            h->tk_f_run = h->tk_f_sum / h->tk_count; h->tk_frame_last = frame;
            h->tk_state = 2;
            emit(h, ATKDSP_TONE_TONE, ATKDSP_TONE_START, in_time(h, h->tk_frame0), NAN, h->tk_f_run, 0.0,
                 h->tk_lvl_sum / h->tk_count, h->tk_snr_sum / h->tk_count, -1, -1, 0u);
            h->st.tones++;
        } else if (have) {
            h->tk_count = 1; h->tk_f_run = f; h->tk_f_sum = f; h->tk_lvl_sum = lvl; h->tk_snr_sum = snr;
            h->tk_frame0 = frame; h->tk_frame_last = frame;
        } else {
            h->tk_state = 0; h->tk_count = 0;
        }
    } else if (have) {
        h->tk_state = 1; h->tk_count = 1; h->tk_f_run = f; h->tk_f_sum = f; h->tk_lvl_sum = lvl;
        h->tk_snr_sum = snr; h->tk_frame0 = frame; h->tk_frame_last = frame;
    }
    h->tk_frames++;
}

/* ---- process ---------------------------------------------------------------------------- */
ptrdiff_t atkdsp_tone_id_process(atkdsp_tone_id *h, const float *x, size_t n, atkdsp_tone_event *ev,
                                 size_t cap) {
    if (!h || (!x && n) || (!ev && cap)) return ATKDSP_E_ARG;
    h->ev = ev; h->cap = cap; h->got = 0;
    const int en = h->cfg.enable;
    const int sub = en & (ATKDSP_TONE_EN_CTCSS | ATKDSP_TONE_EN_DCS);
    for (size_t i = 0; i < n; ++i) {
        const float s = x[i];
        /* full-rate frames */
        if (en & ATKDSP_TONE_EN_DTMF) {
            h->dt_buf[h->dt_pos] = s;
            h->dt_pos = (h->dt_pos + 1) % h->dt_W;
        }
        if (en & ATKDSP_TONE_EN_TONE) {
            h->tk_buf[h->tk_pos] = s;
            h->tk_pos = (h->tk_pos + 1) % h->tk_W;
        }
        h->idx++;
        if (h->idx % (uint64_t)h->dt_H == 0) {
            if ((en & ATKDSP_TONE_EN_DTMF) && h->idx >= (uint64_t)h->dt_W) dt_frame(h, h->idx - (uint64_t)h->dt_W);
            if ((en & ATKDSP_TONE_EN_TONE) && h->idx >= (uint64_t)h->tk_W) tk_frame(h, h->idx - (uint64_t)h->tk_W);
        }
        /* the sub-audio chain */
        if (sub) {
            h->lp_hist[h->lp_pos] = s;
            h->lp_pos = (h->lp_pos + 1) % h->lp_n;
            if (++h->lp_phase == (int)h->decim) {
                h->lp_phase = 0;
                double y = 0.0;
                int j = h->lp_pos - 1;
                for (int k = 0; k < h->lp_n; ++k) {
                    if (j < 0) j += h->lp_n;
                    y += (double)h->lp[k] * (double)h->lp_hist[j];
                    --j;
                }
                h->sub_idx++;
                if (en & ATKDSP_TONE_EN_CTCSS) {
                    h->cs_seg[h->cs_pos++] = (float)y;
                    if (h->cs_pos == h->cs_L) { cs_segment(h); h->cs_pos = 0; }
                }
                if (en & ATKDSP_TONE_EN_DCS) dcs_push(h, (float)y);
            }
        }
    }
    h->st.samples += n;
    h->ev = NULL; h->cap = 0;
    return (ptrdiff_t)h->got;
}

ptrdiff_t atkdsp_tone_id_flush(atkdsp_tone_id *h, atkdsp_tone_event *ev, size_t cap) {
    if (!h || (!ev && cap)) return ATKDSP_E_ARG;
    h->ev = ev; h->cap = cap; h->got = 0;
    if (h->cs_state == 2) cs_end(h, h->cs_seg_last + 1, ATKDSP_TONE_F_FLUSHED);
    h->cs_state = 0; h->cs_count = 0;
    if (h->dcs_state == 1) dcs_end(h, ATKDSP_TONE_F_FLUSHED);
    if (h->dt_state == 2) dt_end(h, h->dt_frame_last, ATKDSP_TONE_F_FLUSHED);
    h->dt_state = 0; h->dt_key = -1; h->dt_count = 0;
    if (h->tk_state == 2) tk_end(h, h->tk_frame_last, ATKDSP_TONE_F_FLUSHED);
    h->tk_state = 0; h->tk_count = 0;
    h->ev = NULL; h->cap = 0;
    return (ptrdiff_t)h->got;
}

int atkdsp_tone_id_describe(const atkdsp_tone_id *h, char *buf, size_t cap) {
    if (!h || !buf || !cap) return ATKDSP_E_ARG;
    snprintf(buf, cap,
             "tone_id: fs %.6g Hz; sub-audio /%u to %.6g Hz (%d taps), CTCSS %d-sample segments "
             "(%.3g s) SNR >= %.1f dB, DCS 134.4 Bd over %d samples every %d, max %d error(s); DTMF %d/%d "
             "samples >= %.1f dBFS; tones %d/%d samples FFT %d >= %.1f dBFS SNR >= %.1f dB; enabled %s%s%s%s; "
             "%llu events (%llu dropped)",
             h->fs, h->decim, h->fs_sub, h->lp_n, h->cs_L, (double)h->cs_L / h->fs_sub, h->cfg.ctcss_min_snr_db,
             h->dcs_M, h->dcs_step, h->cfg.dcs_max_errors, h->dt_W, h->dt_H, h->cfg.dtmf_min_dbfs,
             h->tk_W, h->tk_H, h->tk_nfft, h->cfg.tone_min_dbfs, h->cfg.tone_min_snr_db,
             (h->cfg.enable & ATKDSP_TONE_EN_CTCSS) ? "CTCSS " : "", (h->cfg.enable & ATKDSP_TONE_EN_DCS) ? "DCS " : "",
             (h->cfg.enable & ATKDSP_TONE_EN_DTMF) ? "DTMF " : "", (h->cfg.enable & ATKDSP_TONE_EN_TONE) ? "TONE" : "",
             (unsigned long long)h->st.events, (unsigned long long)h->st.dropped);
    return ATKDSP_OK;
}

int atkdsp_tone_id_stats(const atkdsp_tone_id *h, atkdsp_tone_counters *st) {
    if (!h || !st) return ATKDSP_E_ARG;
    *st = h->st;
    return ATKDSP_OK;
}
