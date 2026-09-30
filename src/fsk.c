/* 12. FSK front end: symbol integrate-and-dump + sync-word search.
 *
 * The reusable spine every FSK/AFSK decoder shares (POCSAG, FLEX, and later
 * APRS/ACARS/AIS): turn a real baseband stream (an FM discriminator output)
 * into one soft value per symbol, and find a known sync pattern in those
 * symbols. Not the 40-MSPS hot path — a pager channel is 48 kHz in, a few
 * thousand symbols/s out — but it is on every decoder's critical path and is
 * the piece they share, so it lives here with a numpy twin and a cross-check.
 *
 * atkdsp_symdump is an INTEGRATE-AND-DUMP: it averages each symbol's worth of
 * samples into one soft value — the matched filter for the near-rectangular
 * output an FM discriminator produces from CPFSK. `sps` may be FRACTIONAL
 * (POCSAG-512 at 48 kHz is 93.75 samples/symbol), so the symbol boundary is
 * tracked with a real accumulator; `phase` positions the first boundary
 * (data-aided timing — the caller picks the phase that maximises symbol
 * amplitude). It is open-loop and deterministic: all state is carried, so a
 * block boundary is invisible and the stream chunked any way yields the same
 * symbols. (A tracked M&M/Gardner loop for a drifting clock is a later
 * increment; a per-burst phase estimate covers the pager case today.)
 *
 * Everything but the input samples and the emitted symbols is double.
 */
#include "internal.h"

/* ---- symbol integrate-and-dump ------------------------------------------ */
struct atkdsp_symdump {
    double sps;      /* samples per symbol (>= 1, may be fractional)         */
    double acc;      /* samples accumulated toward the next symbol boundary  */
    double sum;      /* running sum of the current symbol's samples          */
    long   cnt;      /* samples in the current symbol so far                 */
};

atkdsp_symdump *atkdsp_symdump_create(double sps, double phase) {
    if (!(sps >= 1.0)) return NULL;
    atkdsp_symdump *s = (atkdsp_symdump *)calloc(1, sizeof *s);
    if (!s) return NULL;
    s->sps = sps;
    if (phase < 0.0) phase = 0.0;
    if (phase >= sps) phase = 0.0;
    s->acc = phase;
    s->sum = 0.0;
    s->cnt = 0;
    return s;
}

void atkdsp_symdump_destroy(atkdsp_symdump *s) { free(s); }

void atkdsp_symdump_reset(atkdsp_symdump *s) {
    if (!s) return;
    s->acc = 0.0;
    s->sum = 0.0;
    s->cnt = 0;
}

size_t atkdsp_symdump_out_max(const atkdsp_symdump *s, size_t n_in) {
    if (!s) return 0;
    return (size_t)((double)n_in / s->sps) + 2;
}

ptrdiff_t atkdsp_symdump_process(atkdsp_symdump *s, const float *in, size_t n,
                                 float *out, size_t out_cap) {
    if (!s || (!in && n) || (!out && out_cap)) return ATKDSP_E_ARG;
    size_t k = 0;
    for (size_t i = 0; i < n; ++i) {
        s->sum += (double)in[i];
        s->cnt += 1;
        s->acc += 1.0;
        if (s->acc >= s->sps) {
            s->acc -= s->sps;
            if (k >= out_cap) return ATKDSP_E_CAP;
            out[k++] = (float)(s->sum / (double)s->cnt);
            s->sum = 0.0;
            s->cnt = 0;
        }
    }
    return (ptrdiff_t)k;
}

/* ---- sync-word search --------------------------------------------------- *
 * Slide `pattern` (plen bits, 0/1) over the sign of the soft symbols `sym`
 * (soft < 0 => bit 1, matching the decoder's slicer) and report every position
 * whose bits equal the pattern (polarity 0) or its complement (polarity 1) in
 * all but <= max_err places. `pos` is the index of the first symbol AFTER the
 * matched pattern — where a frame's payload begins. Writes up to `cap` hits;
 * returns the number found, or ATKDSP_E_CAP if more than `cap` exist. Pure;
 * the twin is reference.sync_search. */
ptrdiff_t atkdsp_sync_search(const float *sym, size_t n,
                             const signed char *pattern, size_t plen,
                             int max_err, atkdsp_sync_hit *out, size_t cap) {
    if ((!sym && n) || !pattern || plen == 0 || (!out && cap)) return ATKDSP_E_ARG;
    if (n < plen) return 0;
    ptrdiff_t cnt = 0;
    const size_t last = n - plen;
    for (size_t i = 0; i <= last; ++i) {
        int err = 0;
        for (size_t j = 0; j < plen; ++j) {
            const int b = sym[i + j] < 0.0f ? 1 : 0;
            const int p = pattern[j] ? 1 : 0;
            err += (b != p);
        }
        const int norm = err <= max_err;
        const int inv = ((int)plen - err) <= max_err;
        if (norm || inv) {
            if ((size_t)cnt >= cap) return ATKDSP_E_CAP;
            out[cnt].pos = (long long)(i + plen);
            out[cnt].polarity = norm ? 0 : 1;
            out[cnt].errors = norm ? err : ((int)plen - err);
            ++cnt;
        }
    }
    return cnt;
}

/* ---- adaptive front-end conditioning (off-centre / sub-optimal RF) ------ *
 * A carrier offset on an FM-discriminator output is a constant pedestal; cheap
 * gear also drifts and varies in level. dc_track removes the slow pedestal so a
 * zero-threshold slicer stays correct however far off-centre the channel sits;
 * agc normalises the amplitude so soft symbols (and a 4-level slicer) see one
 * scale. Both are leaky integrators — deterministic, streaming, in/out may
 * alias; the accumulator is double so the C matches its numpy twin tightly. */
void atkdsp_dc_track(const float *in, size_t n, float *out, float *state, float alpha) {
    if (!in || !out) return;
    if (!(alpha > 0.0f)) { if (out != in) memmove(out, in, n * sizeof(float)); return; }
    double s = state ? (double)*state : 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double x = (double)in[i];
        s += (double)alpha * (x - s);
        out[i] = (float)(x - s);
    }
    if (state) *state = (float)s;
}

void atkdsp_agc(const float *in, size_t n, float *out, float *state, float alpha, float ref) {
    if (!in || !out) return;
    if (!(ref > 0.0f)) ref = 1.0f;
    if (!(alpha > 0.0f)) { if (out != in) memmove(out, in, n * sizeof(float)); return; }
    double a = state ? (double)*state : 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double x = (double)in[i];
        const double m = x < 0.0 ? -x : x;
        a += (double)alpha * (m - a);
        out[i] = (float)(x * ((double)ref / (a > 1e-9 ? a : 1e-9)));
    }
    if (state) *state = (float)a;
}
