/* ---- 26. Chirp spread spectrum: the LoRa probe (§B6b) ------------------------
 * docs/ATKDSP_PLAN.md §B6b. The twin is reference.py §26 (CssProbe, css_demod).
 * A LoRa up-chirp over bw sweeps bw Hz in T_sym = 2^sf / bw s; a symbol is
 * that chirp cyclically shifted by its value. Multiply a symbol-length window
 * by the conjugate base up-chirp (de-chirp) and FFT: a LoRa symbol collapses
 * to one bin at its value, noise does not. The probe slides a symbol window
 * at hop T_sym/4 on the global index; a window passes when its de-chirped
 * peak stands above the mean bin power by the per-window threshold (Exp(1),
 * so -ln(pfa / (bins x candidates))); PREAMBLE when preamble_min consecutive
 * passing windows agree on the bin within +-1. */
#include "internal.h"
#include <stdio.h>

#define CSS_MAX_CAND 32

#define CSS_RING 256

typedef struct {
    int sf; double bw;
    int Nsym;                  /* samples per symbol at fs = 2^sf * fs / bw     */
    atkdsp_fft *fft;
    atkdsp_cf32 *dechirp;      /* conj base up-chirp, Nsym                       */
    atkdsp_cf32 *win, *X;
    double level_thr;
    int last_bin, run, nwin;
    int ring[CSS_RING];        /* recent peak bins (-1 = no detection)          */
    int rhead;
    double last_level, last_offset;
    unsigned flags;
} css_c;

struct atkdsp_css_probe {
    double fs, pfa;
    int ncand, preamble_min;
    css_c c[CSS_MAX_CAND];
    atkdsp_cf32 *hist;         /* the longest symbol, for the sliding window     */
    int maxN, fill;
    size_t pos;                /* global sample index                            */
    uint64_t samples, segments, resets;
};

static int make_cand(css_c *c, double fs, int sf, double bw, double pfa_each) {
    if (sf < 5 || sf > 12 || !(bw > 0.0) || !(bw <= fs)) return 0;
    const double os = fs / bw;                 /* samples per chip = oversample  */
    const int Nsym = (int)floor((double)(1 << sf) * os + 0.5);
    if (Nsym < 8 || Nsym > (1 << 20)) return 0;
    c->sf = sf; c->bw = bw; c->Nsym = Nsym;
    c->fft = atkdsp_fft_create((size_t)Nsym);
    c->dechirp = (atkdsp_cf32 *)malloc((size_t)Nsym * sizeof *c->dechirp);
    c->win = (atkdsp_cf32 *)malloc((size_t)Nsym * sizeof *c->win);
    c->X = (atkdsp_cf32 *)malloc((size_t)Nsym * sizeof *c->X);
    if (!c->fft || !c->dechirp || !c->win || !c->X) return 0;
    /* base up-chirp: f(t) = -bw/2 + bw t / Tsym, phase = 2 pi (-bw/2 t + bw t^2/(2 Tsym)) */
    const double Tsym = (double)(1 << sf) / bw;
    for (int i = 0; i < Nsym; i++) {
        const double t = (double)i / fs;
        const double ph = 2.0 * ATK_PI * (-0.5 * bw * t + 0.5 * bw / Tsym * t * t);
        c->dechirp[i].re = (float)cos(-ph); c->dechirp[i].im = (float)sin(-ph);   /* conjugate */
    }
    c->level_thr = -log(pfa_each);
    c->last_bin = -1; c->run = 0; c->nwin = 0; c->rhead = 0;
    for (int i = 0; i < CSS_RING; i++) c->ring[i] = -1;
    return 1;
}

atkdsp_css_probe *atkdsp_css_probe_create(double fs, const atkdsp_css_cand *cands, int ncand,
                                          int preamble_min, double pfa) {
    if (!(fs > 0.0) || !cands || ncand < 1 || ncand > CSS_MAX_CAND) return NULL;
    if (preamble_min < 1 || preamble_min > 64 || !(pfa > 0.0) || !(pfa < 0.5)) return NULL;
    atkdsp_css_probe *p = (atkdsp_css_probe *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->fs = fs; p->pfa = pfa; p->ncand = ncand; p->preamble_min = preamble_min;
    int maxN = 0;
    for (int i = 0; i < ncand; i++) {
        /* per-window Pfa split over bins and candidates */
        const int bins = 1 << cands[i].sf;
        if (!make_cand(&p->c[i], fs, cands[i].sf, cands[i].bw_hz, pfa / (double)bins / (double)ncand)) {
            atkdsp_css_probe_destroy(p); return NULL;
        }
        if (p->c[i].Nsym > maxN) maxN = p->c[i].Nsym;
    }
    p->maxN = maxN;
    p->hist = (atkdsp_cf32 *)calloc((size_t)maxN, sizeof *p->hist);
    if (!p->hist) { atkdsp_css_probe_destroy(p); return NULL; }
    return p;
}

void atkdsp_css_probe_destroy(atkdsp_css_probe *p) {
    if (!p) return;
    for (int i = 0; i < p->ncand; i++) {
        atkdsp_fft_destroy(p->c[i].fft); free(p->c[i].dechirp); free(p->c[i].win); free(p->c[i].X);
    }
    free(p->hist); free(p);
}

void atkdsp_css_probe_reset(atkdsp_css_probe *p) {
    if (!p) return;
    p->fill = 0; p->pos = 0; p->resets++;
    for (int i = 0; i < p->ncand; i++) {
        p->c[i].last_bin = -1; p->c[i].run = 0; p->c[i].nwin = 0; p->c[i].flags = 0; p->c[i].rhead = 0;
        for (int j = 0; j < CSS_RING; j++) p->c[i].ring[j] = -1;
    }
}

/* one de-chirp + FFT of the last Nsym samples ending at hist[fill-1] */
static void css_eval(atkdsp_css_probe *p, css_c *c) {
    const int N = c->Nsym;
    /* the window is the last N samples of hist (circular not needed: hist holds maxN, we keep it linear) */
    const int base = p->fill - N;
    if (base < 0) return;
    for (int i = 0; i < N; i++) {
        const atkdsp_cf32 *s = &p->hist[base + i];
        c->win[i].re = s->re * c->dechirp[i].re - s->im * c->dechirp[i].im;
        c->win[i].im = s->re * c->dechirp[i].im + s->im * c->dechirp[i].re;
    }
    atkdsp_fft_exec(c->fft, c->win, c->X, 0);
    const int bins = 1 << c->sf;               /* after de-chirp a symbol sits in one of 2^sf bins */
    /* fold the Nsym FFT bins into 2^sf symbol bins (oversampling spreads each symbol bin by os) */
    int peak = 0; double pmax = 0.0, psum = 0.0;
    for (int b = 0; b < bins; b++) {
        /* symbol value b maps to FFT bin round(b * N / bins) and its alias near N */
        const int k = (int)((long)b * N / bins);
        const double m = (double)c->X[k].re * c->X[k].re + (double)c->X[k].im * c->X[k].im;
        psum += m;
        if (m > pmax) { pmax = m; peak = b; }
    }
    const double mean = psum / bins;
    const double level = mean > 0.0 ? pmax / mean : 0.0;
    c->last_level = level; c->nwin++;
    /* push this window's peak bin (or -1) onto the ring */
    const int det = level >= c->level_thr;
    c->ring[c->rhead] = det ? peak : -1;
    c->rhead = (c->rhead + 1) % CSS_RING;
    if (det) {
        c->flags = ATKDSP_CSS_DETECT;
        c->last_bin = peak;
        /* PREAMBLE: the same bin (+-1) recurs over the recent windows. At a
           hop of Nsym/4 three of four windows straddle a symbol boundary and
           dechirp elsewhere, so a run of CONSECUTIVE equal bins never forms on
           the 8 identical up-chirps; the symbol-aligned repeats do. Count how
           many of the last 4*preamble_min windows share this bin. */
        const int span = 4 * p->preamble_min < CSS_RING ? 4 * p->preamble_min : CSS_RING;
        int cnt = 0;
        for (int j = 0; j < span; j++) {
            const int idx = (c->rhead - 1 - j + CSS_RING) % CSS_RING;
            const int b = c->ring[idx];
            if (b >= 0 && abs(b - peak) <= 1) cnt++;
        }
        c->run = cnt;
        if (cnt >= p->preamble_min) c->flags |= ATKDSP_CSS_PREAMBLE;
        c->last_offset = 0.0;
    } else {
        c->flags = 0; c->run = 0; c->last_bin = -1;
    }
}

int atkdsp_css_probe_process(atkdsp_css_probe *p, const atkdsp_cf32 *x, size_t n) {
    if (!p || (!x && n)) return ATKDSP_E_ARG;
    for (size_t i = 0; i < n; i++) {
        if (p->fill == p->maxN) {
            memmove(p->hist, p->hist + p->maxN / 2, (size_t)(p->maxN - p->maxN / 2) * sizeof *p->hist);
            p->fill -= p->maxN / 2;
        }
        p->hist[p->fill++] = x[i];
        p->pos++;
        /* evaluate each candidate every Nsym/4 global samples once enough history is in */
        for (int c = 0; c < p->ncand; c++) {
            const int hop = p->c[c].Nsym / 4 > 0 ? p->c[c].Nsym / 4 : 1;
            if ((p->pos % (size_t)hop) == 0 && p->fill >= p->c[c].Nsym) { css_eval(p, &p->c[c]); p->segments++; }
        }
    }
    p->samples += n;
    return ATKDSP_OK;
}

int atkdsp_css_probe_read(const atkdsp_css_probe *p, atkdsp_css_result *out) {
    if (!p || !out) return ATKDSP_E_ARG;
    for (int i = 0; i < p->ncand; i++) {
        atkdsp_css_result *r = &out[i];
        memset(r, 0, sizeof *r);
        r->level = p->c[i].last_level; r->level_thr = p->c[i].level_thr;
        r->offset_hz = p->c[i].last_offset; r->bin = p->c[i].last_bin;
        r->run = p->c[i].run; r->nwin = p->c[i].nwin;
        r->flags = p->c[i].flags;
        if (p->c[i].nwin == 0) r->flags |= ATKDSP_CSS_WARMUP;
    }
    return p->ncand;
}

int atkdsp_css_probe_describe(const atkdsp_css_probe *p, char *buf, size_t cap) {
    if (!p || !buf || !cap) return ATKDSP_E_ARG;
    int n = snprintf(buf, cap, "css_probe: fs %.0f Hz, %d candidate(s), preamble %d windows, Pfa %.3g:",
                     p->fs, p->ncand, p->preamble_min, p->pfa);
    for (int i = 0; i < p->ncand && n >= 0 && (size_t)n < cap; i++)
        n += snprintf(buf + n, cap - (size_t)n, "%s SF%d/%.6g Hz", i ? "," : "", p->c[i].sf, p->c[i].bw);
    return ATKDSP_OK;
}

int atkdsp_css_probe_stats(const atkdsp_css_probe *p, atkdsp_cyclic_counters *st) {
    if (!p || !st) return ATKDSP_E_ARG;
    st->samples = p->samples; st->segments = p->segments; st->resets = p->resets; st->segment_fill = (uint64_t)p->fill;
    return ATKDSP_OK;
}

ptrdiff_t atkdsp_css_demod(const atkdsp_cf32 *iq, size_t n, double fs, int sf, double bw_hz,
                           double offset_hz, double t0, uint16_t *symbols, size_t cap) {
    if (!iq || !symbols || sf < 5 || sf > 12 || !(bw_hz > 0.0) || !(bw_hz <= fs)) return ATKDSP_E_ARG;
    const double os = fs / bw_hz;
    const int N = (int)floor((double)(1 << sf) * os + 0.5);
    if (N < 8) return ATKDSP_E_ARG;
    const int bins = 1 << sf;
    const double Tsym = (double)(1 << sf) / bw_hz;
    atkdsp_fft *fft = atkdsp_fft_create((size_t)N);
    atkdsp_cf32 *dc = (atkdsp_cf32 *)malloc((size_t)N * sizeof *dc);
    atkdsp_cf32 *w = (atkdsp_cf32 *)malloc((size_t)N * sizeof *w);
    atkdsp_cf32 *X = (atkdsp_cf32 *)malloc((size_t)N * sizeof *X);
    if (!fft || !dc || !w || !X) { atkdsp_fft_destroy(fft); free(dc); free(w); free(X); return ATKDSP_E_NOMEM; }
    for (int i = 0; i < N; i++) {
        const double t = (double)i / fs;
        const double ph = 2.0 * ATK_PI * (-0.5 * bw_hz * t + 0.5 * bw_hz / Tsym * t * t + offset_hz * t);
        dc[i].re = (float)cos(-ph); dc[i].im = (float)sin(-ph);
    }
    size_t got = 0;
    for (double start = t0; ; start += N) {
        const long s0 = (long)floor(start + 0.5);
        if (s0 < 0 || (size_t)s0 + N > n) break;
        for (int i = 0; i < N; i++) {
            const atkdsp_cf32 *s = &iq[s0 + i];
            w[i].re = s->re * dc[i].re - s->im * dc[i].im;
            w[i].im = s->re * dc[i].im + s->im * dc[i].re;
        }
        atkdsp_fft_exec(fft, w, X, 0);
        int peak = 0; double pmax = 0.0;
        for (int b = 0; b < bins; b++) {
            const int k = (int)((long)b * N / bins);
            const double m = (double)X[k].re * X[k].re + (double)X[k].im * X[k].im;
            if (m > pmax) { pmax = m; peak = b; }
        }
        if (got < cap) symbols[got] = (uint16_t)peak;
        got++;
    }
    atkdsp_fft_destroy(fft); free(dc); free(w); free(X);
    return (ptrdiff_t)got;
}
