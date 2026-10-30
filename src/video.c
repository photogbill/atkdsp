/* 32. Analog video: composite samples to pictures (docs/VIDEO_PLAN.md, §C6).
 *
 * The twin is python/atkdsp/reference.py §32 (class Video); the definitions
 * are in atkdsp.h §32. Every stage below is a static function with the same
 * name as the twin's method, doing the same arithmetic in the same order, so
 * the cross-check holds decision for decision.
 *
 * CLEAN ROOM. Written from the video standards — SMPTE ST 170 (the 525-line
 * composite signal), ITU-R BT.1700 and BT.470 (the 525- and 625-line
 * systems), ITU-R BT.601 (the active-line sample counts) — and textbook loop
 * design (Rice, "Digital Communications: A Discrete-Time Approach", the
 * second-order loop gains). sigutils' tvproc.c was not read: rule 7 makes LGPL
 * code a test oracle only, and sigutils' repo licence (GPL-3) contradicts its
 * file headers (LGPL-3).
 *
 * THE NUMBERS AND WHERE THEY COME FROM (nominal; to be checked against the
 * Recommendation texts before release — a wrong framing number shifts the
 * picture by pixels, it does not break lock):
 *   525-line: fH = 4.5 MHz / 286 (ST 170); 262.5 lines a field; H sync 4.7 us,
 *     equalizing 2.3 us, broad 27.1 us; 0H to the first active sample 122 and
 *     an active line of 720 samples at 13.5 MHz (BT.601); 18 lines from the
 *     field's first broad pulse to the first visible line, 240 visible a
 *     field; black 7.5 IRE over blank (setup), white 100 IRE, sync 40 IRE;
 *     fsc = 315/88 MHz.
 *   625-line: fH = 15,625 Hz; 312.5 lines a field; H 4.7 us, equalizing
 *     2.35 us, broad 27.3 us; 132 / 720 samples at 13.5 MHz (BT.601); 22 lines
 *     to the first visible, 288 visible; black = blank, white 700 mV over a
 *     300 mV sync; fsc = 4,433,618.75 Hz.
 *   custom: the 525-line fractions scaled by the measured period; 7 % of the
 *     field for the vertical blanking; black = blank, white 2.5 sync amplitudes.
 *
 * THRESHOLDS, DERIVED (rule 4):
 *   - Before lock the slicer sits at q02 + 0.15 d with hysteresis 0.05 d, where
 *     q02 is the 2nd percentile and d = median - q02 of the last two segments
 *     of the smoothed signal. Sync is ~7.4 % of a line (4.7 / 63.6 us), so q02
 *     lies in the sync tip. For any picture d <= (1 + white_rel) A (A the sync
 *     amplitude): 0.15 d + 0.05 d <= 0.70 A keeps the release level below
 *     blank; for a picture at blank (d ~ A) the entry level 0.10 A is above
 *     the tip.
 *   - Locked, the slicer sits at the 50 % point between the measured tip and
 *     blank — the standard's own definition of 0H — with hysteresis
 *     3 sigma (the back porch's noise), clamped to 5..20 % of A.
 *   - Pulse classes split at the midpoints between nominal widths:
 *     3.5 us (equalizing | H) and 15.9 us (H | broad); under 1.0 us or over
 *     54.6 us is noise.
 *   - Line loop: noise bandwidths 0.05 (acquire, the first 32 lines) and
 *     0.005 cycles a line (track), damping 0.707. At 0.005, edge noise of
 *     100 ns rms leaves 100 ns x sqrt(2 x 0.005) = 10 ns of line jitter, under
 *     a third of a 73 ns pixel at 720 px. Windows 3.0 / 1.5 us.
 *   - Acquisition: 8 consecutive H spacings within 1 % of their mean, in
 *     58..70 us (AUTO: 14.3..17.2 kHz) or +-2 % of a forced standard's period.
 *   - Field lock: two field intervals within one line of each other, 100..420
 *     lines each (and within 2 lines of a forced standard's). After a loss,
 *     one interval that matches the field length seen before is enough.
 *   - Line lock is lost after 64 consecutive missing lines (4 ms): a short
 *     RF dropout coasts through without losing a field.
 *   - Stacking: per 8x8 block, the mean |frame - average| is tested against
 *     its noise expectation sigma_d (sqrt(2/pi) + 4 sqrt(1 - 2/pi) / sqrt(m)),
 *     sigma_d = sigma_pixel sqrt(1 + 1/n): a four-sigma test on the mean of m
 *     half-normal values.
 *
 * ARITHMETIC (so the twin can match): samples are float; everything else is
 * double; a pixel is rounded to float once, at the end. Medians are order
 * statistics (the element at (m - 1) / 2 of the sorted window). Sums run in
 * index order. The box filter is a running sum restarted every segment.
 */
#include "internal.h"

#include <stdio.h>

#define VID_T525        (286.0 / 4.5e6)
#define VID_SEG_S       64e-6
#define VID_BOX_S       0.5e-6
#define VID_LOOK_BACK   32
#define VID_HIST_SEGS   4
#define VID_QUEUE       4
#define VID_VIS_MAX     420
#define VID_H_MAX       (2 * VID_VIS_MAX)
#define VID_RQ          256
#define VID_TRAP_TAPS   31
#define VID_W_MIN       1.0e-6
#define VID_B_EQ_H      3.5e-6
#define VID_B_H_BROAD   15.9e-6
#define VID_W_MAX       54.6e-6
#define VID_ACQ_LO      58e-6
#define VID_ACQ_HI      70e-6
#define VID_ACQ_N       8
#define VID_WIN_WIDE    3.0e-6
#define VID_WIN_TRACK   1.5e-6
#define VID_BL_WIDE     0.05
#define VID_BL_TRACK    0.005
#define VID_ZETA        0.70710678118654752
#define VID_WIDE_LINES  32
#define VID_LOSE        64
#define VID_LPF_MIN     100.0
#define VID_LPF_MAX     420.0
#define VID_TIP_A       0.6e-6
#define VID_TIP_B       4.1e-6
#define VID_BLANK_A     5.7e-6
#define VID_BLANK_B     8.7e-6
#define VID_SIG_A       8.0e-6
#define VID_SIG_B       9.0e-6
#define VID_TRAP_HALF   1.0e6

typedef struct {
    double line_hz, lpf, a0, la, black_rel, white_rel, fsc;
    int vbi, visible, comb_lag;
} vid_nominal;

static const vid_nominal VID_NOM_525 = {
    4.5e6 / 286.0, 262.5, 122.0 / 13.5e6, 720.0 / 13.5e6, 7.5 / 40.0, 100.0 / 40.0,
    315e6 / 88.0, 18, 240, 1 };
static const vid_nominal VID_NOM_625 = {
    15625.0, 312.5, 132.0 / 13.5e6, 720.0 / 13.5e6, 0.0, 700.0 / 300.0,
    4433618.75, 22, 288, 2 };

typedef struct {                     /* the standard in force, once known */
    double a0f, laf, black_rel, white_rel, fsc, la_s;
    int vbi, visible, comb_lag;
} vid_std;

typedef struct {                     /* a line waiting to be rendered */
    double t0, y, T;
    int64_t fseq;
    int acc, pcls, cls, vl;
} vid_line;

typedef struct {
    int valid;
    int64_t seq;
    int cls;
    float *rows;                     /* VID_VIS_MAX x width */
    unsigned char *filled;           /* VID_VIS_MAX */
    int any_filled;
    int have_t0;
    double t0;
} vid_field;

typedef struct vid_core {
    struct atkdsp_video *o;
    double sign;
    /* slicer */
    int have_thr;
    double thr_s, thr_h;
    double *sm_prev, *sm_cur;        /* S each: the last two segments, smoothed */
    int have_sm_prev, have_sm_cur;
    int low;
    double lead;
    /* line loop */
    int track;
    int have_last_h;
    double last_h;
    int acq_n;
    double acq_d;
    double T, t_last;
    int wide, consec, miss_run;
    unsigned char lk_acc[64];
    double lk_e[64];
    int lk_n, lk_pos;
    double lpf_mem;
    /* fields */
    int have_v, v_lock;
    double t_v;
    int cls_v;
    int64_t field_seq;
    double ivs[2];
    int n_ivs;
    int v_coast, progressive;
    double lpf;
    /* render queue */
    vid_line rq[VID_RQ];
    int rq_head, rq_n;
    /* levels */
    int64_t n_lv;
    double tip_s, blank_s, sig_s;
    /* standard */
    int have_std, std_id, std_key_visible, std_key_prog;
    vid_std std;
    int have_trap;
    double trap[VID_TRAP_TAPS];
    /* fields being rendered */
    vid_field fbuf[2];
    int cur, last;                   /* index into fbuf, or -1 */
    int64_t comb_seq;
    int comb_n;
    double *comb_rows;               /* 2 x width */
    /* scratch */
    double *work;                    /* the segment's smoothed values + look-back */
    double *cs;                      /* running sums */
    double *scr;                     /* order statistics and level windows */
    double *row, *row2;              /* width each */
} vid_core;

struct atkdsp_video {
    atkdsp_video_config cfg;
    atkdsp_video_config pend_cfg;
    int has_pend;
    int S, L;
    double Ld;
    /* history: global sample index hist_g0 + i lives at hist[i] */
    float *hist;
    size_t hist_cap, hist_n;
    int64_t hist_g0;
    int64_t log_base;                /* the twin's hist_base: the oldest sample it may use */
    int64_t nseg;
    float *pend;
    int pend_n;
    vid_core core[2];
    int ncores;
    int committed;                   /* -1, or a core index */
    int have_last_sign;
    double last_sign;
    /* stacking */
    float *avg;
    int32_t *nb;
    int have_avg, avg_h, avg_w;
    /* the frame queue */
    float *qpix[VID_QUEUE];
    atkdsp_video_meta qmeta[VID_QUEUE];
    int q_head, q_n;
    float *frame;                    /* assembly buffer, VID_H_MAX x width */
    int alloc_w;                     /* the width every buffer is sized for (>= cfg.width) */
    int32_t *nb_scr;
    atkdsp_video_counters st;
};

/* ---- small helpers ------------------------------------------------------- */
static void vid_loop_gains(double bl, double *kp, double *ki) {
    const double z = VID_ZETA;
    const double th = bl / (z + 0.25 / z);
    const double d = 1.0 + 2.0 * z * th + th * th;
    *kp = 4.0 * z * th / d;
    *ki = 4.0 * th * th / d;
}

/* The element at index k of the sorted a[0..n) (a is reordered). */
static double vid_select(double *a, size_t n, size_t k) {
    size_t lo = 0, hi = n - 1;
    while (lo < hi) {
        const double piv = a[lo + (hi - lo) / 2];
        size_t i = lo, j = hi;
        for (;;) {
            while (a[i] < piv) i++;
            while (a[j] > piv) j--;
            if (i >= j) break;
            const double t = a[i]; a[i] = a[j]; a[j] = t;
            i++; j--;
        }
        if (k <= j) hi = j; else lo = j + 1;
    }
    return a[k];
}

static int config_ok(const atkdsp_video_config *c) {
    if (!(c->fs >= 4e6 && c->fs <= 200e6) || !isfinite(c->fs)) return 0;
    if (c->standard < 0 || c->standard > 3) return 0;
    if (c->standard == ATKDSP_VID_STD_CUSTOM) {
        if (!(c->custom_line_hz >= 5000.0 && c->custom_line_hz <= 40000.0)) return 0;
        if (c->custom_lines < 2 * (int)VID_LPF_MIN || c->custom_lines > 2 * (int)VID_LPF_MAX) return 0;
    }
    if (c->polarity < 0 || c->polarity > 2) return 0;
    if (c->yc < 0 || c->yc > 2) return 0;
    if (c->fields < 0 || c->fields > 2) return 0;
    if (c->swap_fields < 0 || c->swap_fields > 1) return 0;
    if (c->width < 64 || c->width > 1920) return 0;
    if (c->stack < 1 || c->stack > 64) return 0;
    if (c->motion_aware < 0 || c->motion_aware > 1) return 0;
    return 1;
}

void atkdsp_video_defaults(atkdsp_video_config *c) {
    if (!c) return;
    memset(c, 0, sizeof *c);
    c->fs = 13.5e6;
    c->standard = ATKDSP_VID_STD_AUTO;
    c->polarity = ATKDSP_VID_POL_AUTO;
    c->yc = ATKDSP_VID_YC_TRAP;
    c->fields = ATKDSP_VID_INTERLACED;
    c->width = 720;
    c->stack = 1;
    c->motion_aware = 1;
}

/* ---- forward declarations ------------------------------------------------ */
static void core_reset(vid_core *c);
static void core_segment(vid_core *c, int64_t k);
static void stack_reset(atkdsp_video *v);

static void trap_taps(double fsc, double pixel_rate, double *h, int *ok) {
    const double f1 = (fsc - VID_TRAP_HALF) / pixel_rate;
    const double f2 = (fsc + VID_TRAP_HALF) / pixel_rate;
    *ok = 0;
    if (!(0.0 < f1 && f1 < f2 && f2 < 0.5)) return;
    const int n = VID_TRAP_TAPS, m = (n - 1) / 2;
    double w[VID_TRAP_TAPS], l1[VID_TRAP_TAPS], l2[VID_TRAP_TAPS];
    for (int k = 0; k < n; k++) w[k] = 0.54 - 0.46 * cos(2.0 * ATK_PI * (double)k / (double)(n - 1));
    for (int pass = 0; pass < 2; pass++) {
        const double fc = pass ? f2 : f1;
        double *l = pass ? l2 : l1;
        double s = 0.0;
        for (int k = 0; k < n; k++) {
            const double x = (double)(k - m);
            l[k] = (k == m) ? 2.0 * fc : sin(2.0 * ATK_PI * fc * x) / (ATK_PI * x);
            l[k] *= w[k];
            s += l[k];
        }
        for (int k = 0; k < n; k++) l[k] /= s;
    }
    for (int k = 0; k < n; k++) h[k] = -(l2[k] - l1[k]);
    h[m] += 1.0;
    *ok = 1;
}

/* ---- allocation ------------------------------------------------------------ */
static void core_free(vid_core *c) {
    free(c->sm_prev); free(c->sm_cur); free(c->work); free(c->cs); free(c->scr);
    free(c->row); free(c->row2); free(c->comb_rows);
    for (int i = 0; i < 2; i++) { free(c->fbuf[i].rows); free(c->fbuf[i].filled); }
    memset(c, 0, sizeof *c);
}

static int core_alloc_width(vid_core *c, int width) {
    for (int i = 0; i < 2; i++) {
        free(c->fbuf[i].rows);
        c->fbuf[i].rows = (float *)calloc((size_t)VID_VIS_MAX * (size_t)width, sizeof(float));
        if (!c->fbuf[i].rows) return 0;
        if (!c->fbuf[i].filled) {
            c->fbuf[i].filled = (unsigned char *)calloc(VID_VIS_MAX, 1);
            if (!c->fbuf[i].filled) return 0;
        }
    }
    free(c->row); free(c->row2); free(c->comb_rows);
    c->row = (double *)malloc((size_t)width * sizeof(double));
    c->row2 = (double *)malloc((size_t)width * sizeof(double));
    c->comb_rows = (double *)malloc(2 * (size_t)width * sizeof(double));
    return c->row && c->row2 && c->comb_rows;
}

static int core_alloc(vid_core *c, atkdsp_video *v, double sign) {
    memset(c, 0, sizeof *c);
    c->o = v;
    c->sign = sign;
    const size_t S = (size_t)v->S, B = VID_LOOK_BACK + 1, L = (size_t)v->L;
    c->sm_prev = (double *)malloc(S * sizeof(double));
    c->sm_cur = (double *)malloc(S * sizeof(double));
    c->work = (double *)malloc((S + B) * sizeof(double));
    c->cs = (double *)malloc((S + B + L + 1) * sizeof(double));
    /* order statistics over two segments; level windows are shorter */
    c->scr = (double *)malloc(2 * S * sizeof(double));
    if (!c->sm_prev || !c->sm_cur || !c->work || !c->cs || !c->scr) return 0;
    if (!core_alloc_width(c, v->cfg.width)) return 0;
    core_reset(c);
    return 1;
}

static int video_alloc_width(atkdsp_video *v, int width) {
    const size_t fsz = (size_t)VID_H_MAX * (size_t)width;
    for (int i = 0; i < VID_QUEUE; i++) {
        free(v->qpix[i]);
        v->qpix[i] = (float *)malloc(fsz * sizeof(float));
        if (!v->qpix[i]) return 0;
    }
    free(v->avg); free(v->frame); free(v->nb); free(v->nb_scr);
    v->avg = (float *)malloc(fsz * sizeof(float));
    v->frame = (float *)malloc(fsz * sizeof(float));
    const size_t nbl = (size_t)((VID_H_MAX + 7) / 8) * (size_t)((width + 7) / 8);
    v->nb = (int32_t *)malloc(nbl * sizeof(int32_t));
    v->nb_scr = (int32_t *)malloc(65 * sizeof(int32_t));
    return v->avg && v->frame && v->nb && v->nb_scr;
}

static void video_free(atkdsp_video *v) {
    for (int i = 0; i < 2; i++) core_free(&v->core[i]);
    for (int i = 0; i < VID_QUEUE; i++) free(v->qpix[i]);
    free(v->hist); free(v->pend); free(v->avg); free(v->frame); free(v->nb); free(v->nb_scr);
    free(v);
}

/* The twin's _full_reset, minus the stream position when keep_stream. */
static int video_full_reset(atkdsp_video *v, int keep_stream) {
    if (!keep_stream) {
        v->hist_n = 0;
        v->hist_g0 = 0;
        v->log_base = 0;
        v->nseg = 0;
        v->pend_n = 0;
        v->q_head = v->q_n = 0;
    }
    stack_reset(v);
    v->committed = -1;
    v->have_last_sign = 0;
    v->last_sign = 0.0;
    const int pol = v->cfg.polarity;
    v->ncores = pol == ATKDSP_VID_POL_AUTO ? 2 : 1;
    for (int i = 0; i < 2; i++) {
        vid_core *c = &v->core[i];
        const double s = (pol == ATKDSP_VID_POL_INVERTED) ? -1.0 : (i == 0 ? 1.0 : -1.0);
        if (i < v->ncores) {
            c->sign = s;
            core_reset(c);
        }
    }
    if (pol != ATKDSP_VID_POL_AUTO) v->committed = 0;
    return 1;
}

atkdsp_video *atkdsp_video_create(const atkdsp_video_config *cfg) {
    if (!cfg || !config_ok(cfg)) return NULL;
    atkdsp_video *v = (atkdsp_video *)calloc(1, sizeof *v);
    if (!v) return NULL;
    v->cfg = *cfg;
    v->S = (int)floor(cfg->fs * VID_SEG_S + 0.5);
    v->L = (int)floor(cfg->fs * VID_BOX_S + 0.5);
    if (v->L < 1) v->L = 1;
    v->Ld = 0.5 * (double)(v->L - 1);
    v->hist_cap = (size_t)(2 * VID_HIST_SEGS) * (size_t)v->S;
    v->hist = (float *)malloc(v->hist_cap * sizeof(float));
    v->pend = (float *)malloc((size_t)v->S * sizeof(float));
    if (!v->hist || !v->pend || !video_alloc_width(v, cfg->width)) { video_free(v); return NULL; }
    v->alloc_w = cfg->width;
    for (int i = 0; i < 2; i++) {
        if (!core_alloc(&v->core[i], v, i == 0 ? 1.0 : -1.0)) { video_free(v); return NULL; }
    }
    video_full_reset(v, 0);
    return v;
}

void atkdsp_video_destroy(atkdsp_video *v) {
    if (v) video_free(v);
}

void atkdsp_video_reset(atkdsp_video *v) {
    if (!v) return;
    video_full_reset(v, 0);
}

/* ---- the core: state ------------------------------------------------------ */
static void v_reset(vid_core *c) {
    c->have_v = 0; c->v_lock = 0; c->t_v = 0.0; c->cls_v = 0; c->field_seq = 0;
    c->n_ivs = 0; c->v_coast = 0; c->progressive = 0; c->lpf = 0.0;
}

static void field_reset(vid_core *c) {
    c->cur = c->last = -1;
    c->fbuf[0].valid = c->fbuf[1].valid = 0;
    c->comb_seq = -1;
    c->comb_n = 0;
    stack_reset(c->o);
}

static void core_reset(vid_core *c) {
    c->have_thr = 0; c->thr_s = c->thr_h = 0.0;
    c->have_sm_prev = c->have_sm_cur = 0;
    c->low = 0; c->lead = 0.0;
    c->track = 0; c->have_last_h = 0; c->last_h = 0.0; c->acq_n = 0; c->acq_d = 0.0;
    c->T = 0.0; c->t_last = 0.0; c->wide = 1; c->consec = 0; c->miss_run = 0;
    c->lk_n = 0; c->lk_pos = 0;
    c->lpf_mem = 0.0;
    v_reset(c);
    c->rq_head = c->rq_n = 0;
    c->n_lv = 0; c->tip_s = c->blank_s = c->sig_s = 0.0;
    c->have_std = 0; c->std_id = 0; c->have_trap = 0;
    field_reset(c);
}

/* ---- helpers --------------------------------------------------------------- */
static double core_k(const vid_core *c) {
    return c->track ? c->T / VID_T525 : c->o->cfg.fs;
}

static void acq_range(const vid_core *c, double *lo, double *hi) {
    const atkdsp_video_config *g = &c->o->cfg;
    const double fs = g->fs;
    if (g->standard == ATKDSP_VID_STD_AUTO) { *lo = VID_ACQ_LO * fs; *hi = VID_ACQ_HI * fs; return; }
    const double lh = g->standard == ATKDSP_VID_STD_CUSTOM ? g->custom_line_hz
                     : (g->standard == ATKDSP_VID_STD_525 ? VID_NOM_525.line_hz : VID_NOM_625.line_hz);
    const double t = fs / lh;
    *lo = 0.98 * t; *hi = 1.02 * t;
}

static int expected_lpf(const vid_core *c, double *e) {
    const atkdsp_video_config *g = &c->o->cfg;
    if (g->standard == ATKDSP_VID_STD_CUSTOM) { *e = g->custom_lines / 2.0; return 1; }
    if (g->standard == ATKDSP_VID_STD_525) { *e = VID_NOM_525.lpf; return 1; }
    if (g->standard == ATKDSP_VID_STD_625) { *e = VID_NOM_625.lpf; return 1; }
    return 0;
}

ATK_INLINE double hval(const atkdsp_video *v, int64_t g) {
    return (double)v->hist[g - v->hist_g0];
}

/* ---- forward ---------------------------------------------------------------- */
static void core_pulse(vid_core *c, double lead, double trail);
static void coast_until(vid_core *c, double t, int segment_end);
static void v_flywheel(vid_core *c, double end);
static void render(vid_core *c, int64_t last_idx);
static void next_thresholds(vid_core *c);

/* ---- the slicer -------------------------------------------------------------
 * work[] holds the smoothed values for global indices [i0 - B, i1). */
static double cross(const vid_core *c, int64_t base, int64_t i, double s, int down) {
    int64_t lo = i - VID_LOOK_BACK;
    if (lo < base + 1) lo = base + 1;
    for (int64_t j = i; j >= lo; j--) {
        const double a = c->work[j - 1 - base];
        const double b = c->work[j - base];
        if (down && a >= s && s > b) return (double)(j - 1) + (a - s) / (a - b);
        if (!down && a <= s && s < b) return (double)(j - 1) + (s - a) / (b - a);
    }
    return (double)i;
}

static void core_segment(vid_core *c, int64_t k) {
    atkdsp_video *v = c->o;
    const int64_t S = v->S;
    const int64_t i0 = k * S, i1 = i0 + S;
    if (k >= 1) {
        const int64_t B = VID_LOOK_BACK + 1, L = v->L;
        const int64_t org = i0 - B - L;
        /* cs[m] = sum of sign * x[org + 1 .. org + m] */
        double run = 0.0;
        c->cs[0] = 0.0;
        const int64_t ncs = i1 - (org + 1);
        for (int64_t m = 0; m < ncs; m++) {
            run += c->sign * hval(v, org + 1 + m);
            c->cs[m + 1] = run;
        }
        const int64_t base = i0 - B;
        for (int64_t n = base; n < i1; n++)
            c->work[n - base] = (c->cs[n - org] - c->cs[n - L - org]) / (double)L;
        /* keep the last two segments' smoothed values (acquisition levels) */
        double *t = c->sm_prev; c->sm_prev = c->sm_cur; c->sm_cur = t;
        c->have_sm_prev = c->have_sm_cur;
        memcpy(c->sm_cur, c->work + (i0 - base), (size_t)S * sizeof(double));
        c->have_sm_cur = 1;
        if (c->have_thr) {
            const double s = c->thr_s, h = c->thr_h, d = v->Ld;
            const double lo_thr = s - h, hi_thr = s + h;
            for (int64_t j = 0; j < S; j++) {
                const double x = c->work[i0 - base + j];
                if (!c->low && x < lo_thr) {
                    c->low = 1;
                    c->lead = cross(c, base, i0 + j, s, 1) - d;
                } else if (c->low && x > hi_thr) {
                    c->low = 0;
                    const double trail = cross(c, base, i0 + j, s, 0) - d;
                    core_pulse(c, c->lead, trail);
                }
            }
        }
    }
    const double end = (double)(i1 - 1) - v->Ld;
    if (c->track) {
        coast_until(c, end, 1);
        v_flywheel(c, end);
    }
    render(c, i1 - 1);
    next_thresholds(c);
}

static void next_thresholds(vid_core *c) {
    if (c->track && c->n_lv >= 8) {
        const double a = c->blank_s - c->tip_s;
        if (a > 0.0) {
            c->thr_s = 0.5 * (c->tip_s + c->blank_s);
            double h = 3.0 * c->sig_s;
            if (h < 0.05 * a) h = 0.05 * a;
            if (h > 0.2 * a) h = 0.2 * a;
            c->thr_h = h;
            c->have_thr = 1;
            return;
        }
    }
    if (!c->have_sm_prev || !c->have_sm_cur) { c->have_thr = 0; return; }
    const size_t S = (size_t)c->o->S, m = 2 * S;
    memcpy(c->scr, c->sm_prev, S * sizeof(double));
    memcpy(c->scr + S, c->sm_cur, S * sizeof(double));
    const double q02 = vid_select(c->scr, m, (size_t)(0.02 * (double)(m - 1)));
    const double q50 = vid_select(c->scr, m, (m - 1) / 2);
    const double d = q50 - q02;
    if (!(d > 1e-12)) { c->have_thr = 0; return; }
    c->thr_s = q02 + 0.15 * d;
    c->thr_h = 0.05 * d;
    c->have_thr = 1;
}

/* ---- pulses and the line loop ------------------------------------------------ */
static int classify(const vid_core *c, double w) {
    const double k = core_k(c);
    if (w < VID_W_MIN * k || w > VID_W_MAX * k) return -1;
    if (w < VID_B_EQ_H * k) return 0;
    if (w < VID_B_H_BROAD * k) return 1;
    return 2;
}

static void lock_push(vid_core *c, int acc, double e) {
    c->lk_acc[c->lk_pos] = (unsigned char)acc;
    c->lk_e[c->lk_pos] = e;
    c->lk_pos = (c->lk_pos + 1) & 63;
    if (c->lk_n < 64) c->lk_n++;
}

static double h_lock(const vid_core *c) {
    if (c->lk_n == 0) return 0.0;
    double s2 = 0.0;
    int na = 0;
    const int start = (c->lk_pos - c->lk_n + 64) & 63;
    for (int i = 0; i < c->lk_n; i++) {          /* oldest first, as the twin's list */
        const int p = (start + i) & 63;
        if (c->lk_acc[p]) { s2 += c->lk_e[p] * c->lk_e[p]; na++; }
    }
    if (na == 0) return 0.0;
    const double rms = sqrt(s2 / (double)na);
    const double win = VID_WIN_TRACK * core_k(c);
    double q = 1.0 - rms / win;
    if (q < 0.0) q = 0.0;
    return ((double)na / 64.0) * q;
}

static void emit_line(vid_core *c, double t0, int acc, int pcls) {
    if (c->rq_n == VID_RQ) {                  /* cannot happen: render drains each segment */
        c->rq_head = (c->rq_head + 1) % VID_RQ;
        c->rq_n--;
    }
    vid_line *l = &c->rq[(c->rq_head + c->rq_n) % VID_RQ];
    l->t0 = t0; l->acc = acc; l->pcls = pcls; l->T = c->T;
    if (c->have_v) {
        l->y = (t0 - c->t_v) / c->T;
        l->fseq = c->field_seq; l->cls = c->cls_v; l->vl = c->v_lock;
    } else {
        l->y = 0.0; l->fseq = -1; l->cls = 0; l->vl = 0;
    }
    c->rq_n++;
}

static void lose_h(vid_core *c) {
    c->track = 0;
    c->have_last_h = 0; c->last_h = 0.0;
    c->acq_n = 0; c->acq_d = 0.0;
    v_reset(c);
    c->have_std = 0; c->std_id = 0;
    c->cur = c->last = -1;
    c->fbuf[0].valid = c->fbuf[1].valid = 0;
    c->comb_seq = -1; c->comb_n = 0;
    stack_reset(c->o);
}

static void acquire(vid_core *c, double lead) {
    double lo, hi;
    acq_range(c, &lo, &hi);
    if (c->have_last_h) {
        const double d = lead - c->last_h;
        if (lo <= d && d <= hi) {
            if (c->acq_n == 0 || fabs(d - c->acq_d) <= 0.01 * c->acq_d) {
                c->acq_d = (c->acq_d * c->acq_n + d) / (double)(c->acq_n + 1);
                c->acq_n++;
            } else {
                c->acq_n = 1;
                c->acq_d = d;
            }
        } else {
            c->acq_n = 0;
        }
    }
    c->have_last_h = 1;
    c->last_h = lead;
    if (c->acq_n >= VID_ACQ_N) {
        c->track = 1;
        c->T = c->acq_d;
        c->t_last = lead;
        c->wide = 1; c->consec = 0; c->miss_run = 0;
        c->lk_n = 0; c->lk_pos = 0;
        c->o->st.h_relocks++;
        emit_line(c, lead, 1, 1);
    }
}

static void accept(vid_core *c, double lead, int cls) {
    const double t_pred = c->t_last + c->T;
    const double e = lead - t_pred;
    double kp, ki, lo, hi;
    vid_loop_gains(c->wide ? VID_BL_WIDE : VID_BL_TRACK, &kp, &ki);
    acq_range(c, &lo, &hi);
    double T = c->T + ki * e;
    if (T < lo) T = lo;
    if (T > hi) T = hi;
    c->T = T;
    const double t0 = t_pred + kp * e;
    c->t_last = t0;
    c->consec++;
    if (c->wide && c->consec >= VID_WIDE_LINES) c->wide = 0;
    c->miss_run = 0;
    lock_push(c, 1, e);
    c->o->st.lines_locked++;
    emit_line(c, t0, 1, cls);
}

static void coast_until(vid_core *c, double t, int segment_end) {
    while (c->track) {
        const double win = (c->wide ? VID_WIN_WIDE : VID_WIN_TRACK) * core_k(c);
        const double t_pred = c->t_last + c->T;
        if (!(t_pred + win < t)) return;
        if (segment_end && c->low && c->lead <= t_pred + win) return;
        c->t_last = t_pred;
        c->consec = 0;
        c->miss_run++;
        lock_push(c, 0, 0.0);
        c->o->st.lines_coasted++;
        emit_line(c, t_pred, 0, -1);
        if (c->miss_run >= VID_LOSE) lose_h(c);
    }
}

/* ---- fields ------------------------------------------------------------------ */
static void decide_std(vid_core *c);

static void v_judge(vid_core *c) {
    const int was = c->v_lock;
    int ok = 0;
    if (c->n_ivs == 1 && c->lpf_mem > 0.0) {
        ok = fabs(c->ivs[0] - c->lpf_mem) <= 1.0;
        if (ok) { c->ivs[1] = c->lpf_mem; c->n_ivs = 2; }
    }
    if (c->n_ivs == 2 && !ok) {
        const double a = c->ivs[0], b = c->ivs[1];
        ok = fabs(a - b) <= 1.0 && a >= VID_LPF_MIN && a <= VID_LPF_MAX &&
             b >= VID_LPF_MIN && b <= VID_LPF_MAX;
        double ex;
        if (ok && expected_lpf(c, &ex)) ok = fabs(0.5 * (a + b) - ex) <= 2.0;
    }
    c->v_lock = ok;
    if (ok) {
        c->lpf = 0.5 * (c->ivs[0] + c->ivs[1]);
        c->lpf_mem = c->lpf;
        decide_std(c);
        if (!was) c->o->st.v_relocks++;
    }
}

static void v_event(vid_core *c, double tb) {
    if (!c->track) return;
    const double lmin = c->v_lock ? 0.75 * c->lpf : 150.0;
    if (c->have_v && (tb - c->t_v) < lmin * c->T) return;
    const double q = (tb - c->t_last) / c->T;
    const double phi = q - floor(q);
    const int cls = (phi < 0.25 || phi >= 0.75) ? 0 : 1;
    const double n_half = floor(2.0 * q + 0.5);
    const double t_v = c->t_last + n_half * 0.5 * c->T;
    if (c->have_v) {
        const double iv = (t_v - c->t_v) / c->T;
        if (c->n_ivs < 2) c->ivs[c->n_ivs++] = iv;
        else { c->ivs[0] = c->ivs[1]; c->ivs[1] = iv; }
        c->progressive = (cls == c->cls_v);
    }
    c->t_v = t_v;
    c->cls_v = cls;
    c->field_seq++;
    c->have_v = 1;
    c->v_coast = 0;
    c->o->st.fields++;
    v_judge(c);
}

static void v_flywheel(vid_core *c, double end) {
    while (c->v_lock && c->have_v) {
        const double dead = c->t_v + (c->lpf + 3.0) * c->T;
        if (!(end > dead)) return;
        c->t_v = c->t_v + c->lpf * c->T;
        if (!c->progressive) c->cls_v ^= 1;
        c->field_seq++;
        c->v_coast++;
        c->o->st.fields++;
        if (c->v_coast > 4) { c->v_lock = 0; c->have_v = 0; c->n_ivs = 0; }
    }
}

static void decide_std(vid_core *c) {
    const atkdsp_video_config *g = &c->o->cfg;
    const double fs = g->fs;
    const double line_hz = fs / c->T;
    int sid = g->standard;
    if (sid == ATKDSP_VID_STD_AUTO) {
        sid = ATKDSP_VID_STD_CUSTOM;
        if (fabs(line_hz - VID_NOM_525.line_hz) <= 0.005 * VID_NOM_525.line_hz &&
            fabs(c->lpf - VID_NOM_525.lpf) <= 2.0) sid = ATKDSP_VID_STD_525;
        else if (fabs(line_hz - VID_NOM_625.line_hz) <= 0.005 * VID_NOM_625.line_hz &&
                 fabs(c->lpf - VID_NOM_625.lpf) <= 2.0) sid = ATKDSP_VID_STD_625;
    }
    vid_std s;
    if (sid == ATKDSP_VID_STD_525 || sid == ATKDSP_VID_STD_625) {
        const vid_nominal *n = sid == ATKDSP_VID_STD_525 ? &VID_NOM_525 : &VID_NOM_625;
        const double tn = 1.0 / n->line_hz;
        s.a0f = n->a0 / tn; s.laf = n->la / tn; s.vbi = n->vbi; s.visible = n->visible;
        s.black_rel = n->black_rel; s.white_rel = n->white_rel; s.fsc = n->fsc;
        s.comb_lag = n->comb_lag; s.la_s = n->la;
    } else {
        const int vbi = (int)floor(0.07 * c->lpf + 0.5);
        s.a0f = (122.0 / 13.5e6) / VID_T525; s.laf = (720.0 / 13.5e6) / VID_T525;
        s.vbi = vbi; s.visible = (int)floor(c->lpf) - vbi - 2;
        s.black_rel = 0.0; s.white_rel = 2.5; s.fsc = 0.0; s.comb_lag = 1;
        s.la_s = (720.0 / 13.5e6) / VID_T525 / line_hz;
    }
    const int prog = c->progressive || g->fields == ATKDSP_VID_PROGRESSIVE;
    if (!c->have_std || c->std_id != sid || c->std_key_visible != s.visible ||
        c->std_key_prog != prog) {
        c->std = s;
        c->have_std = 1;
        c->std_id = sid;
        c->std_key_visible = s.visible;
        c->std_key_prog = prog;
        c->cur = c->last = -1;
        c->fbuf[0].valid = c->fbuf[1].valid = 0;
        c->comb_seq = -1; c->comb_n = 0;
        stack_reset(c->o);
        c->have_trap = 0;
        if (g->yc == ATKDSP_VID_YC_TRAP && s.fsc > 0.0)
            trap_taps(s.fsc, (double)g->width / s.la_s, c->trap, &c->have_trap);
    }
}

static void core_pulse(vid_core *c, double lead, double trail) {
    if (c->track) coast_until(c, lead, 0);
    const int cls = classify(c, trail - lead);
    if (cls < 0) { c->o->st.pulses_rejected++; return; }
    if (!c->track) {
        if (cls == 1) acquire(c, lead);
        return;
    }
    if (cls == 2) { v_event(c, lead); return; }
    const double t_pred = c->t_last + c->T;
    const double win = (c->wide ? VID_WIN_WIDE : VID_WIN_TRACK) * core_k(c);
    if (fabs(lead - t_pred) <= win) accept(c, lead, cls);
}

/* ---- rendering ----------------------------------------------------------------- */
static void emit_frame(vid_core *c, const float *frame, int H, int W, double t0,
                       int have_t0, int prog);

static void finish_field(vid_core *c, int fi) {
    atkdsp_video *v = c->o;
    vid_field *f = &c->fbuf[fi];
    v->st.fields_rendered++;
    if (!f->any_filled) { c->last = fi; return; }
    const int mode = v->cfg.fields;
    const int prog = c->progressive || mode == ATKDSP_VID_PROGRESSIVE;
    const int W = v->cfg.width, vis = c->std.visible;
    if (prog) {
        emit_frame(c, f->rows, vis, W, f->t0, f->have_t0, 1);
    } else if (mode == ATKDSP_VID_ONE_FIELD) {
        for (int r = 0; r < vis; r++) {
            memcpy(v->frame + (size_t)(2 * r) * W, f->rows + (size_t)r * W, (size_t)W * sizeof(float));
            memcpy(v->frame + (size_t)(2 * r + 1) * W, f->rows + (size_t)r * W, (size_t)W * sizeof(float));
        }
        emit_frame(c, v->frame, 2 * vis, W, f->t0, f->have_t0, 0);
    } else {
        const int li = c->last;
        if (f->cls == 1 && li >= 0 && c->fbuf[li].valid && c->fbuf[li].cls == 0 &&
            c->fbuf[li].seq == f->seq - 1) {
            const vid_field *last = &c->fbuf[li];
            const vid_field *top = v->cfg.swap_fields ? f : last;
            const vid_field *bot = v->cfg.swap_fields ? last : f;
            for (int r = 0; r < vis; r++) {
                memcpy(v->frame + (size_t)(2 * r) * W, top->rows + (size_t)r * W, (size_t)W * sizeof(float));
                memcpy(v->frame + (size_t)(2 * r + 1) * W, bot->rows + (size_t)r * W, (size_t)W * sizeof(float));
            }
            if (last->have_t0) emit_frame(c, v->frame, 2 * vis, W, last->t0, 1, 0);
            else emit_frame(c, v->frame, 2 * vis, W, f->t0, f->have_t0, 0);
        }
    }
    c->last = fi;
}

static int new_field(vid_core *c, int64_t fseq, int cls) {
    /* the buffer that is neither the last finished field */
    const int fi = (c->last == 0) ? 1 : 0;
    vid_field *f = &c->fbuf[fi];
    const int W = c->o->cfg.width, vis = c->std.visible;
    memset(f->rows, 0, (size_t)vis * (size_t)W * sizeof(float));
    memset(f->filled, 0, (size_t)vis);
    f->any_filled = 0;
    f->valid = 1;
    f->seq = fseq;
    f->cls = cls;
    f->have_t0 = 0;
    f->t0 = 0.0;
    return fi;
}

/* window of sign * x over [ceil(t0 + a k), floor(t0 + b k)] into scr; returns its length */
static size_t level_window(vid_core *c, double t0, double a, double b, double T, double *dst) {
    const double k = T / VID_T525;
    const int64_t ia = (int64_t)ceil(t0 + a * k);
    const int64_t ib = (int64_t)floor(t0 + b * k);
    if (ib < ia) return 0;
    size_t n = 0;
    for (int64_t i = ia; i <= ib; i++) dst[n++] = c->sign * hval(c->o, i);
    return n;
}

static void render_line(vid_core *c, const vid_line *l) {
    atkdsp_video *v = c->o;
    int have_lb = 0;
    double line_blank = 0.0;
    if (l->acc && l->pcls == 1) {
        double *tw = c->scr;
        const size_t nt = level_window(c, l->t0, VID_TIP_A, VID_TIP_B, l->T, tw);
        double *bw = c->scr + nt;
        const size_t nb = level_window(c, l->t0, VID_BLANK_A, VID_BLANK_B, l->T, bw);
        double *sw = c->scr + nt + nb;
        const size_t ns = level_window(c, l->t0, VID_SIG_A, VID_SIG_B, l->T, sw);
        if (nt && nb && ns >= 2) {
            double s1 = 0.0, s2 = 0.0;
            for (size_t i = 0; i < ns; i++) s1 += sw[i];
            for (size_t i = 0; i < ns; i++) s2 += sw[i] * sw[i];
            const double tip = vid_select(tw, nt, (nt - 1) / 2);
            const double blank = vid_select(bw, nb, (nb - 1) / 2);
            const double m = (double)ns;
            const double var = s2 / m - (s1 / m) * (s1 / m);
            const double sig = var > 0.0 ? sqrt(var) : 0.0;
            if (c->n_lv == 0) { c->tip_s = tip; c->blank_s = blank; c->sig_s = sig; }
            else {
                c->tip_s += (tip - c->tip_s) / 8.0;
                c->blank_s += (blank - c->blank_s) / 8.0;
                c->sig_s += (sig - c->sig_s) / 64.0;
            }
            c->n_lv++;
            have_lb = 1;
            line_blank = blank;
        }
    }
    if (l->fseq < 0 || !l->vl || !c->have_std) return;
    if (c->cur < 0 || c->fbuf[c->cur].seq != l->fseq) {
        if (c->cur >= 0) finish_field(c, c->cur);
        c->cur = new_field(c, l->fseq, l->cls);
    }
    const vid_std *s = &c->std;
    const double off = l->cls == 1 ? 0.5 : 0.0;
    const int kk = (int)floor(l->y - off - (double)s->vbi + 0.5);
    if (kk < 0 || kk >= s->visible) return;
    const double a = c->blank_s - c->tip_s;
    if (!(a > 0.0)) return;
    const int W = v->cfg.width;
    const double blank = have_lb ? line_blank : c->blank_s;
    /* Keys (a = -1/2) cubic at t0 + a0f T + (p + 1/2) laf T / W */
    const double ts = l->t0 + s->a0f * l->T, step = s->laf * l->T / (double)W;
    double *row = c->row;
    for (int p = 0; p < W; p++) {
        const double pos = ts + ((double)p + 0.5) * step;
        const double fi = floor(pos);
        const int64_t i = (int64_t)fi;
        const double f = pos - fi;
        const double xm1 = hval(v, i - 1), x0 = hval(v, i), x1 = hval(v, i + 1), x2 = hval(v, i + 2);
        const double wm1 = ((-0.5 * f + 1.0) * f - 0.5) * f;
        const double w0 = (1.5 * f - 2.5) * f * f + 1.0;
        const double w1 = ((-1.5 * f + 2.0) * f + 0.5) * f;
        const double w2 = (0.5 * f - 0.5) * f * f;
        double r = ((wm1 * xm1 + w0 * x0) + w1 * x1) + w2 * x2;
        if (c->sign < 0.0) r = -r;
        row[p] = r;
    }
    const int yc = v->cfg.yc;
    if (yc == ATKDSP_VID_YC_COMB) {
        const int lag = s->comb_lag;
        if (c->comb_seq != l->fseq) { c->comb_seq = l->fseq; c->comb_n = 0; }
        double *raw = c->row2;
        memcpy(raw, row, (size_t)W * sizeof(double));
        if (c->comb_n >= lag) {
            const double *prev = c->comb_rows + (size_t)(c->comb_n - lag) * W;
            for (int p = 0; p < W; p++) row[p] = 0.5 * (row[p] + prev[p]);
        }
        if (c->comb_n < 2) {
            memcpy(c->comb_rows + (size_t)c->comb_n * W, raw, (size_t)W * sizeof(double));
            c->comb_n++;
        } else {
            memcpy(c->comb_rows, c->comb_rows + W, (size_t)W * sizeof(double));
            memcpy(c->comb_rows + W, raw, (size_t)W * sizeof(double));
        }
    } else if (yc == ATKDSP_VID_YC_TRAP && c->have_trap) {
        const int m = (VID_TRAP_TAPS - 1) / 2;
        double *acc = c->row2;
        for (int p = 0; p < W; p++) acc[p] = 0.0;
        for (int t = 0; t < VID_TRAP_TAPS; t++) {
            /* the same sum per pixel, in tap order, as the twin's clipped
             * gather — split so the interior loop vectorises */
            const double h = c->trap[t];
            const int sh = t - m;
            const int p0 = sh < 0 ? -sh : 0;
            const int p1 = sh > 0 ? W - 1 - sh : W - 1;
            for (int p = 0; p < p0; p++) acc[p] = acc[p] + h * row[0];
            for (int p = p0; p <= p1; p++) acc[p] = acc[p] + h * row[p + sh];
            for (int p = p1 + 1; p < W; p++) acc[p] = acc[p] + h * row[W - 1];
        }
        row = acc;
    }
    const double black = blank + a * s->black_rel;
    const double white = blank + a * s->white_rel;
    vid_field *fd = &c->fbuf[c->cur];
    float *dst = fd->rows + (size_t)kk * W;
    for (int p = 0; p < W; p++) {
        double q = (row[p] - black) / (white - black);
        if (q < 0.0) q = 0.0;
        if (q > 1.0) q = 1.0;
        dst[p] = (float)q;
    }
    fd->filled[kk] = 1;
    fd->any_filled = 1;
    if (!fd->have_t0 || l->t0 < fd->t0) { fd->t0 = l->t0; fd->have_t0 = 1; }
}

static void render(vid_core *c, int64_t last_idx) {
    atkdsp_video *v = c->o;
    while (c->rq_n > 0) {
        const vid_line *l = &c->rq[c->rq_head];
        const double laf = c->have_std ? c->std.laf : (720.0 / 13.5e6) / VID_T525;
        const double a0f = c->have_std ? c->std.a0f : (122.0 / 13.5e6) / VID_T525;
        const int64_t need = (int64_t)ceil(l->t0 + (a0f + laf) * l->T) + 3;
        if (need > last_idx) break;
        vid_line cp = *l;
        c->rq_head = (c->rq_head + 1) % VID_RQ;
        c->rq_n--;
        if ((int64_t)floor(cp.t0) - 2 < v->log_base) continue;   /* fell out of history */
        render_line(c, &cp);
    }
}

/* ---- stacking and the queue -------------------------------------------------- */
static void stack_reset(atkdsp_video *v) {
    v->have_avg = 0;
}

/* stack into v->avg; returns the median block depth; out = v->avg (or frame when N = 1) */
static int stack(atkdsp_video *v, const float *f, int H, int W, double sig_p, const float **out) {
    const int N = v->cfg.stack;
    if (N <= 1) { *out = f; return 1; }
    const int bh = (H + 7) / 8, bw = (W + 7) / 8;
    if (!v->have_avg || v->avg_h != H || v->avg_w != W) {
        memcpy(v->avg, f, (size_t)H * W * sizeof(float));
        for (int i = 0; i < bh * bw; i++) v->nb[i] = 1;
        v->have_avg = 1; v->avg_h = H; v->avg_w = W;
        *out = v->avg;
        return 1;
    }
    const double c1 = sqrt(2.0 / ATK_PI), c2 = sqrt(1.0 - 2.0 / ATK_PI);
    for (int by = 0; by < bh; by++) {
        const int y0 = by * 8, y1 = (y0 + 8 < H) ? y0 + 8 : H;
        for (int bx = 0; bx < bw; bx++) {
            const int x0 = bx * 8, x1 = (x0 + 8 < W) ? x0 + 8 : W;
            int n = v->nb[by * bw + bx];
            int reset = 0;
            if (v->cfg.motion_aware && sig_p > 0.0) {
                const int m = (y1 - y0) * (x1 - x0);
                double sad = 0.0;
                for (int y = y0; y < y1; y++)
                    for (int x = x0; x < x1; x++) {
                        const size_t i = (size_t)y * W + x;
                        sad += fabs((double)f[i] - (double)v->avg[i]);
                    }
                const double mad = sad / (double)m;
                const double sd = sig_p * sqrt(1.0 + 1.0 / (double)n);
                const double thr = sd * (c1 + 4.0 * c2 / sqrt((double)m));
                reset = mad > thr;
            }
            if (reset) {
                v->nb[by * bw + bx] = 1;
                for (int y = y0; y < y1; y++)
                    memcpy(v->avg + (size_t)y * W + x0, f + (size_t)y * W + x0,
                           (size_t)(x1 - x0) * sizeof(float));
            } else {
                n = n + 1 < N ? n + 1 : N;
                v->nb[by * bw + bx] = n;
                for (int y = y0; y < y1; y++)
                    for (int x = x0; x < x1; x++) {
                        const size_t i = (size_t)y * W + x;
                        const double ad = (double)v->avg[i];
                        const double d = (double)f[i] - ad;
                        v->avg[i] = (float)(ad + d / (double)n);
                    }
            }
        }
    }
    /* the median block depth: counts 1..64, so a counting pass */
    const int nbl = bh * bw, k = (nbl - 1) / 2;
    for (int i = 0; i <= 64; i++) v->nb_scr[i] = 0;
    for (int i = 0; i < nbl; i++) v->nb_scr[v->nb[i]]++;
    int acc = 0, med = 1;
    for (int i = 0; i <= 64; i++) { acc += v->nb_scr[i]; if (acc > k) { med = i; break; } }
    *out = v->avg;
    return med;
}

static void emit_frame(vid_core *c, const float *frame, int H, int W, double t0,
                       int have_t0, int prog) {
    atkdsp_video *v = c->o;
    if (v->committed < 0 || &v->core[v->committed] != c) return;
    atkdsp_video_meta m;
    memset(&m, 0, sizeof m);
    m.sample0 = have_t0 ? (int64_t)floor(t0) : 0;
    m.standard = c->std_id;
    m.polarity = c->sign > 0.0 ? 1 : 2;
    m.progressive = prog;
    m.field_order = v->cfg.swap_fields;
    m.width = W; m.lines = H;
    m.line_hz = v->cfg.fs / c->T;
    m.lines_per_field = c->lpf;
    m.h_lock = (float)h_lock(c);
    m.v_lock = c->v_lock ? 1 : 0;
    m.snr_db = (float)(c->sig_s > 0.0 ? 20.0 * log10((c->blank_s - c->tip_s) / c->sig_s) : 99.0);
    const double a = c->blank_s - c->tip_s;
    const double sig_p = a > 0.0 ? c->sig_s / (a * (c->std.white_rel - c->std.black_rel)) : 0.0;
    const float *out;
    m.stacked = stack(v, frame, H, W, sig_p, &out);
    v->st.frames_out++;
    int slot;
    if (v->q_n == VID_QUEUE) {                       /* drop the oldest */
        slot = v->q_head;
        v->q_head = (v->q_head + 1) % VID_QUEUE;
        v->st.frames_dropped++;
    } else {
        slot = (v->q_head + v->q_n) % VID_QUEUE;
        v->q_n++;
    }
    memcpy(v->qpix[slot], out, (size_t)H * W * sizeof(float));
    v->qmeta[slot] = m;
}

/* ---- the stream ---------------------------------------------------------------- */
static void apply_pending(atkdsp_video *v) {
    const atkdsp_video_config c = v->pend_cfg;
    const atkdsp_video_config old = v->cfg;
    v->has_pend = 0;
    v->cfg = c;
    const int reacq = c.standard != old.standard || c.custom_line_hz != old.custom_line_hz ||
                      c.custom_lines != old.custom_lines || c.polarity != old.polarity;
    if (reacq) { video_full_reset(v, 1); return; }
    const int geom = c.width != old.width || c.fields != old.fields || c.yc != old.yc ||
                     c.swap_fields != old.swap_fields;
    if (c.width != old.width && v->q_n) {
        v->st.frames_dropped += (uint64_t)v->q_n;
        v->q_head = v->q_n = 0;
    }
    for (int i = 0; i < v->ncores; i++) {
        vid_core *cc = &v->core[i];
        if (geom && cc->have_std) {
            cc->have_std = 0;
            cc->cur = cc->last = -1;
            cc->fbuf[0].valid = cc->fbuf[1].valid = 0;
            cc->comb_seq = -1; cc->comb_n = 0;
            if (cc->v_lock) decide_std(cc);
        }
    }
    if (geom || c.stack != old.stack || c.motion_aware != old.motion_aware) stack_reset(v);
}

static void commit(atkdsp_video *v) {
    if (v->cfg.polarity != ATKDSP_VID_POL_AUTO) return;
    if (v->committed >= 0) {
        if (!v->core[v->committed].track) {
            const int lost = v->committed;
            v->committed = -1;
            for (int i = 0; i < v->ncores; i++)
                if (i != lost) core_reset(&v->core[i]);
        }
        return;
    }
    for (int i = 0; i < v->ncores; i++) {
        if (v->core[i].v_lock) {
            if (v->have_last_sign && v->last_sign != v->core[i].sign) v->st.polarity_flips++;
            v->have_last_sign = 1;
            v->last_sign = v->core[i].sign;
            v->committed = i;
            return;
        }
    }
}

static void segment(atkdsp_video *v, const float *seg) {
    const int64_t S = v->S;
    const int64_t k = v->nseg;
    /* append, keeping at least the last VID_HIST_SEGS segments contiguous */
    if (v->hist_n + (size_t)S > v->hist_cap) {
        const size_t keep = (size_t)(VID_HIST_SEGS - 1) * (size_t)S;
        const size_t drop = v->hist_n - keep;
        memmove(v->hist, v->hist + drop, keep * sizeof(float));
        v->hist_n = keep;
        v->hist_g0 += (int64_t)drop;
    }
    memcpy(v->hist + v->hist_n, seg, (size_t)S * sizeof(float));
    v->hist_n += (size_t)S;
    const int64_t total = (k + 1) * S;
    v->log_base = total > VID_HIST_SEGS * S ? total - VID_HIST_SEGS * S : 0;
    if (v->committed >= 0) core_segment(&v->core[v->committed], k);
    else for (int i = 0; i < v->ncores; i++) core_segment(&v->core[i], k);
    v->nseg++;
    commit(v);
}

ptrdiff_t atkdsp_video_process(atkdsp_video *v, const float *x, size_t n) {
    if (!v || (!x && n)) return ATKDSP_E_ARG;
    v->st.samples_in += n;
    const int S = v->S;
    size_t i = 0;
    while (i < n) {
        size_t take = (size_t)(S - v->pend_n);
        if (take > n - i) take = n - i;
        memcpy(v->pend + v->pend_n, x + i, take * sizeof(float));
        v->pend_n += (int)take;
        i += take;
        if (v->pend_n == S) {
            if (v->has_pend) apply_pending(v);
            segment(v, v->pend);
            v->pend_n = 0;
        }
    }
    return (ptrdiff_t)v->q_n;
}

/* Grow every width-dependent buffer to hold `w` pixels a line, keeping its
 * contents (realloc): queued frames, the fields being built and the comb rows
 * stay valid, since their layout follows cfg.width, which changes only when
 * the pending settings are applied. Buffers never shrink. */
static int grow_width(atkdsp_video *v, int w) {
    if (w <= v->alloc_w) return 1;
    const size_t fsz = (size_t)VID_H_MAX * (size_t)w;
    for (int i = 0; i < VID_QUEUE; i++) {
        float *p = (float *)realloc(v->qpix[i], fsz * sizeof(float));
        if (!p) return 0;
        v->qpix[i] = p;
    }
    float *a = (float *)realloc(v->avg, fsz * sizeof(float));
    if (!a) return 0;
    v->avg = a;
    a = (float *)realloc(v->frame, fsz * sizeof(float));
    if (!a) return 0;
    v->frame = a;
    const size_t nbl = (size_t)((VID_H_MAX + 7) / 8) * (size_t)((w + 7) / 8);
    int32_t *nb = (int32_t *)realloc(v->nb, nbl * sizeof(int32_t));
    if (!nb) return 0;
    v->nb = nb;
    for (int i = 0; i < 2; i++) {
        vid_core *c = &v->core[i];
        for (int f = 0; f < 2; f++) {
            float *r = (float *)realloc(c->fbuf[f].rows, (size_t)VID_VIS_MAX * (size_t)w * sizeof(float));
            if (!r) return 0;
            c->fbuf[f].rows = r;
        }
        double *d = (double *)realloc(c->row, (size_t)w * sizeof(double));
        if (!d) return 0;
        c->row = d;
        d = (double *)realloc(c->row2, (size_t)w * sizeof(double));
        if (!d) return 0;
        c->row2 = d;
        d = (double *)realloc(c->comb_rows, 2 * (size_t)w * sizeof(double));
        if (!d) return 0;
        c->comb_rows = d;
    }
    v->alloc_w = w;
    return 1;
}

int atkdsp_video_set(atkdsp_video *v, const atkdsp_video_config *cfg) {
    if (!v || !cfg || !config_ok(cfg) || cfg->fs != v->cfg.fs) return ATKDSP_E_ARG;
    if (!grow_width(v, cfg->width)) return ATKDSP_E_NOMEM;
    v->pend_cfg = *cfg;
    v->has_pend = 1;
    return ATKDSP_OK;
}

int atkdsp_video_frame(atkdsp_video *v, float *pix, size_t cap, atkdsp_video_meta *meta) {
    if (!v) return ATKDSP_E_ARG;
    if (v->q_n == 0) return 0;
    const atkdsp_video_meta *m = &v->qmeta[v->q_head];
    if (meta) *meta = *m;
    const size_t need = (size_t)m->width * (size_t)m->lines;
    if (!pix || cap < need) return ATKDSP_E_CAP;
    memcpy(pix, v->qpix[v->q_head], need * sizeof(float));
    v->q_head = (v->q_head + 1) % VID_QUEUE;
    v->q_n--;
    return 1;
}

int atkdsp_video_stats(const atkdsp_video *v, atkdsp_video_counters *st) {
    if (!v || !st) return ATKDSP_E_ARG;
    *st = v->st;
    return ATKDSP_OK;
}

int atkdsp_video_describe(const atkdsp_video *v, char *buf, size_t cap) {
    if (!v || !buf || !cap) return ATKDSP_E_ARG;
    const atkdsp_video_config *g = &v->cfg;
    const vid_core *c = v->committed >= 0 ? &v->core[v->committed] : &v->core[0];
    char head[256];
    if (!c->track) {
        snprintf(head, sizeof head, "searching for line sync");
    } else {
        const double h = h_lock(c);
        if (c->v_lock) {
            const char *nm = c->std_id == ATKDSP_VID_STD_525 ? "525-line"
                           : c->std_id == ATKDSP_VID_STD_625 ? "625-line"
                           : c->std_id == ATKDSP_VID_STD_CUSTOM ? "custom" : "standard unknown";
            snprintf(head, sizeof head, "%s, %s, locked (H %.2f, V yes), %.1f Hz line rate, "
                     "%.1f lines a field", nm, c->progressive ? "progressive" : "interlaced", h,
                     g->fs / c->T, c->lpf);
        } else {
            snprintf(head, sizeof head, "line-rate pulses at %.1f Hz but no fields yet (H %.2f, V no)",
                     g->fs / c->T, h);
        }
    }
    int real = (int)(g->fs * (720.0 / 13.5e6));
    if (real > g->width) real = g->width;
    const char *yc = g->yc == ATKDSP_VID_YC_OFF ? "chroma passed"
                   : g->yc == ATKDSP_VID_YC_COMB ? "line comb" : "chroma trap";
    if (g->yc == ATKDSP_VID_YC_TRAP && c->have_std && !c->have_trap)
        yc = "chroma trap OFF (no subcarrier for this standard, or too few pixels)";
    const char *pol = g->polarity == ATKDSP_VID_POL_AUTO ? "auto" : "set";
    const int n = snprintf(buf, cap, "video: %s, %d px of which about %d are real at %.3g MS/s, "
                           "%s, stack %d%s, %s polarity (%s).", head, g->width, real, g->fs / 1e6,
                           yc, g->stack, (g->motion_aware && g->stack > 1) ? " motion-aware" : "",
                           c->sign > 0.0 ? "normal" : "inverted", pol);
    return n < 0 ? ATKDSP_E_ARG : ATKDSP_OK;
}
