/* Convolutional codes: the encoder and a general Viterbi decoder (header §16a).
 *
 * State convention. The state is the last K-1 input bits, the most recent in
 * the TOP bit (K-2). The encoder register for input u from state s is
 * reg = (u << (K-1)) | s, so a generator's most significant bit multiplies the
 * current input — the usual octal convention (K=7: 0171, 0133). The next state
 * is reg >> 1. The two predecessors of state s' are p0 = (s' << 1) & (S-1) and
 * p1 = p0 | 1, both entered with input u = the top bit of s': the butterfly
 * (2j, 2j+1) -> (j, j + S/2).
 *
 * Add-compare-select keeps the p0 path on a tie (the predecessor whose oldest
 * bit is 0), as reference.lte_viterbi_tb does. Metrics are float32 and each
 * step subtracts the best one, so they never grow and the twin
 * (reference.py §16a) reproduces every decision bit for bit.
 *
 * Tail-biting is decoded wrap-around: the block extended by `depth` steps of
 * itself on each side (depth = the block: three laps), all start metrics
 * equal, traced back from the best end state, the middle copy kept. */
#include "internal.h"
#include <stdio.h>

#define MAXN 8

struct atkdsp_viterbi {
    int K, n, S, mode, depth, punc_len;
    uint32_t polys[MAXN];
    uint8_t  keep[MAXN * 64];      /* puncture matrix n x punc_len, 1 = sent */
    int      kp[64], kp_sum;       /* bits sent per period column, per period */
    uint8_t *outw;                 /* [2S]: codeword (bit i = output i) for (p, u) */
    float   *pm, *pm2;             /* path metrics */
    uint64_t *dec;                 /* decisions: steps x words */
    size_t   *off;                 /* TERM/TB: LLR offset per step (max_steps + 1) */
    uint8_t  *tbits;               /* traceback output (max_steps) */
    uint8_t  *code;                /* TERM/TB: re-encode for the metric */
    size_t    code_cap;
    int      words;                /* 64-bit words per step */
    size_t   max_steps;            /* TERM/TB: decision rows allocated */
    size_t   max_bits;
    /* STREAM state */
    size_t   ring;                 /* decision rows in the ring (2 * depth) */
    unsigned long long steps_done, released;
    float    pend[MAXN];           /* the step being assembled */
    int      pend_i;               /* outputs of that step filled so far */
    atkdsp_fec_stats st;
};

ATK_INLINE int parity32(uint32_t x) { return atk_parity64((uint64_t)x); }

static void build_outw(atkdsp_viterbi *v) {
    for (int p = 0; p < v->S; ++p)
        for (int u = 0; u < 2; ++u) {
            const uint32_t reg = ((uint32_t)u << (v->K - 1)) | (uint32_t)p;
            unsigned cw = 0;
            for (int i = 0; i < v->n; ++i) cw |= (unsigned)parity32(reg & v->polys[i]) << i;
            v->outw[2 * p + u] = (uint8_t)cw;
        }
}

atkdsp_viterbi *atkdsp_viterbi_create(int K, int n, const uint32_t *polys,
                                      const uint8_t *puncture, int punc_len,
                                      int mode, int depth, size_t max_bits) {
    if (K < 2 || K > 15 || n < 1 || n > MAXN || !polys) return NULL;
    if (mode < ATKDSP_CONV_TERM || mode > ATKDSP_CONV_STREAM || depth < 0) return NULL;
    if (puncture && (punc_len < 1 || punc_len > 64)) return NULL;
    if (mode != ATKDSP_CONV_STREAM && max_bits < 1) return NULL;
    /* bounded so no size computed below can wrap (2^28 data bits per block) */
    if (max_bits > ((size_t)1 << 28) || depth > (1 << 24)) return NULL;
    for (int i = 0; i < n; ++i)
        if (polys[i] == 0 || polys[i] >> K) return NULL;
    atkdsp_viterbi *v = (atkdsp_viterbi *)calloc(1, sizeof *v);
    if (!v) return NULL;
    v->K = K; v->n = n; v->S = 1 << (K - 1); v->mode = mode;
    for (int i = 0; i < n; ++i) v->polys[i] = polys[i];
    v->punc_len = puncture ? punc_len : 1;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < v->punc_len; ++j)
            v->keep[i * v->punc_len + j] = puncture ? (uint8_t)(puncture[i * punc_len + j] ? 1 : 0) : 1;
    v->kp_sum = 0;
    for (int j = 0; j < v->punc_len; ++j) {
        int s = 0;
        for (int i = 0; i < n; ++i) s += v->keep[i * v->punc_len + j];
        if (s == 0) { free(v); return NULL; }      /* every input bit must send something */
        v->kp[j] = s; v->kp_sum += s;
    }
    if (depth == 0 && mode == ATKDSP_CONV_STREAM) depth = (puncture ? 12 : 6) * K;
    v->depth = depth;
    v->max_bits = max_bits;
    v->words = (v->S + 63) / 64;
    if (mode == ATKDSP_CONV_TERM) v->max_steps = max_bits + (size_t)(K - 1);
    else if (mode == ATKDSP_CONV_TAILBITE)
        v->max_steps = max_bits + 2 * (depth ? ((size_t)depth < max_bits ? (size_t)depth : max_bits) : max_bits);
    else v->max_steps = v->ring = 2 * (size_t)depth;
    v->outw = (uint8_t *)malloc(2 * (size_t)v->S);
    v->pm = (float *)atk_aligned_malloc((size_t)v->S * sizeof(float));
    v->pm2 = (float *)atk_aligned_malloc((size_t)v->S * sizeof(float));
    if (v->max_steps > ((size_t)-1) / ((size_t)v->words * sizeof(uint64_t))) { atkdsp_viterbi_destroy(v); return NULL; }
    v->dec = (uint64_t *)malloc(v->max_steps * (size_t)v->words * sizeof(uint64_t));
    v->tbits = (uint8_t *)malloc(v->max_steps ? v->max_steps : 1);
    if (mode != ATKDSP_CONV_STREAM) {
        v->off = (size_t *)malloc((v->max_steps + 1) * sizeof *v->off);
        v->code_cap = (size_t)n * (max_bits + (size_t)(K - 1));
        v->code = (uint8_t *)malloc(v->code_cap);
    }
    if (!v->outw || !v->pm || !v->pm2 || !v->dec || !v->tbits
        || (mode != ATKDSP_CONV_STREAM && (!v->off || !v->code))) { atkdsp_viterbi_destroy(v); return NULL; }
    build_outw(v);
    atkdsp_viterbi_reset(v);
    v->st.resets = 0;
    return v;
}

void atkdsp_viterbi_destroy(atkdsp_viterbi *v) {
    if (!v) return;
    free(v->outw); atk_aligned_free(v->pm); atk_aligned_free(v->pm2); free(v->dec);
    free(v->off); free(v->tbits); free(v->code);
    free(v);
}

void atkdsp_viterbi_reset(atkdsp_viterbi *v) {
    if (!v) return;
    for (int s = 0; s < v->S; ++s) v->pm[s] = 0.0f;
    v->steps_done = v->released = 0;
    v->pend_i = 0;
    v->st.resets++;
}

size_t atkdsp_viterbi_out_max(const atkdsp_viterbi *v, size_t nllr) {
    if (!v) return 0;
    /* a call completes at most nllr + 1 steps (every input bit sends one LLR
     * or more; the +1 is a pending step whose LLRs all arrived earlier), and
     * releases at most those plus depth - 1 already held */
    if (v->mode == ATKDSP_CONV_STREAM) return nllr + (size_t)v->depth;
    return v->max_bits;
}

/* One trellis step: metrics for the 2^n codewords from the step's n LLRs,
 * then add-compare-select into pm2, decisions into row d, renormalise. */
static void acs_step(atkdsp_viterbi *v, const float *llr, uint64_t *d) {
    float met[256];
    const int ncw = 1 << v->n;
    for (int cw = 0; cw < ncw; ++cw) {
        float m = 0.0f;
        for (int i = 0; i < v->n; ++i) m = ((cw >> i) & 1) ? m - llr[i] : m + llr[i];
        met[cw] = m;
    }
    const int half = v->S / 2;
    const float *pm = v->pm;
    float *nm = v->pm2;
    const uint8_t *ow = v->outw;
    memset(d, 0, (size_t)v->words * sizeof *d);
    for (int j = 0; j < half; ++j) {
        const int p0 = 2 * j, p1 = p0 + 1;
        for (int u = 0; u < 2; ++u) {
            const int s = j + u * half;
            const float m0 = pm[p0] + met[ow[2 * p0 + u]];
            const float m1 = pm[p1] + met[ow[2 * p1 + u]];
            if (m0 >= m1) nm[s] = m0;
            else { nm[s] = m1; d[s >> 6] |= 1ULL << (s & 63); }
        }
    }
    float best = nm[0];
    for (int s = 1; s < v->S; ++s) if (nm[s] > best) best = nm[s];
    for (int s = 0; s < v->S; ++s) nm[s] = nm[s] - best;
    v->pm2 = v->pm; v->pm = nm;
}

static int best_state(const atkdsp_viterbi *v) {
    int b = 0;
    for (int s = 1; s < v->S; ++s) if (v->pm[s] > v->pm[b]) b = s;
    return b;
}

/* Trace back from state s through rows [row_lo, row_hi) (row_hi exclusive,
 * rows given by a mapper for the ring), writing input bits for those rows. */
static void traceback(const atkdsp_viterbi *v, int s, long long hi, long long lo,
                      uint8_t *bits_for_row0, int ring) {
    const int top = v->K - 2;
    for (long long t = hi - 1; t >= lo; --t) {
        const size_t row = ring ? (size_t)((unsigned long long)t % v->ring) : (size_t)t;
        const uint64_t *d = v->dec + row * (size_t)v->words;
        if (bits_for_row0) bits_for_row0[t - lo] = (uint8_t)((s >> top) & 1);
        const int x = (int)((d[s >> 6] >> (s & 63)) & 1u);
        s = ((s << 1) & (v->S - 1)) | x;
    }
}

/* Steps a TERM/TB block of nllr LLRs spans, or -1. */
static long long block_steps(const atkdsp_viterbi *v, size_t nllr) {
    const size_t full = nllr / (size_t)v->kp_sum;
    size_t steps = full * (size_t)v->punc_len;
    size_t rem = nllr - full * (size_t)v->kp_sum;
    while (rem > 0) {
        const size_t k = (size_t)v->kp[steps % (size_t)v->punc_len];
        if (k > rem) return -1;
        rem -= k; ++steps;
    }
    return (long long)steps;
}

/* Gather step t's n LLRs (zeros where punctured) from a block whose first
 * step's LLRs start at llr[off[t]]. */
static void gather(const atkdsp_viterbi *v, const float *llr, const size_t *off, size_t t, float *x) {
    const size_t j = t % (size_t)v->punc_len;
    size_t o = off[t];
    for (int i = 0; i < v->n; ++i)
        x[i] = v->keep[i * v->punc_len + (int)j] ? llr[o++] : 0.0f;
}

ptrdiff_t atkdsp_conv_encode(int K, int n, const uint32_t *polys, const uint8_t *puncture,
                             int punc_len, int mode, const uint8_t *bits, size_t nbits,
                             uint8_t *out, size_t cap) {
    if (K < 2 || K > 15 || n < 1 || n > MAXN || !polys || (!bits && nbits) || (!out && cap)) return ATKDSP_E_ARG;
    if (mode < ATKDSP_CONV_TERM || mode > ATKDSP_CONV_STREAM) return ATKDSP_E_ARG;
    if (puncture && (punc_len < 1 || punc_len > 64)) return ATKDSP_E_ARG;
    if (mode == ATKDSP_CONV_TAILBITE && nbits == 0) return ATKDSP_E_ARG;
    for (int i = 0; i < n; ++i) if (polys[i] == 0 || polys[i] >> K) return ATKDSP_E_ARG;
    const int P = puncture ? punc_len : 1;
    const uint32_t smask = (1u << (K - 1)) - 1u;
    uint32_t s = 0;
    if (mode == ATKDSP_CONV_TAILBITE)
        for (size_t j = 0; j < (size_t)(K - 1); ++j) {
            const size_t idx = (nbits * (size_t)K + nbits - (size_t)(K - 1) + j) % nbits;
            s = (((uint32_t)(bits[idx] & 1u)) << (K - 2)) | (s >> 1);
        }
    const size_t steps = nbits + (mode == ATKDSP_CONV_TERM ? (size_t)(K - 1) : 0);
    size_t w = 0;
    for (size_t t = 0; t < steps; ++t) {
        const uint32_t u = t < nbits ? (uint32_t)(bits[t] & 1u) : 0u;
        const uint32_t reg = (u << (K - 1)) | s;
        for (int i = 0; i < n; ++i) {
            if (puncture && !puncture[i * punc_len + (int)(t % (size_t)P)]) continue;
            if (w >= cap) return ATKDSP_E_CAP;
            out[w++] = (uint8_t)parity32(reg & polys[i]);
        }
        s = (reg >> 1) & smask;
    }
    return (ptrdiff_t)w;
}

/* The decoded path's correlation with the block: re-encode and compare. */
static float block_metric(atkdsp_viterbi *v, const float *llr, size_t nllr,
                          const uint8_t *data, size_t nbits) {
    uint8_t *code = v->code;
    if (nllr > v->code_cap) return NAN;
    uint8_t punc[MAXN * 64];
    for (int i = 0; i < v->n * v->punc_len; ++i) punc[i] = v->keep[i];
    const ptrdiff_t m = atkdsp_conv_encode(v->K, v->n, v->polys, punc, v->punc_len,
                                           v->mode, data, nbits, code, nllr);
    double num = 0.0, den = 0.0;
    if (m == (ptrdiff_t)nllr)
        for (size_t i = 0; i < nllr; ++i) {
            num += code[i] ? -(double)llr[i] : (double)llr[i];
            den += fabs((double)llr[i]);
        }
    return (m == (ptrdiff_t)nllr && den > 0.0) ? (float)(num / den) : NAN;
}

static ptrdiff_t decode_block(atkdsp_viterbi *v, const float *llr, size_t nllr,
                              uint8_t *bits, size_t cap, float *metric) {
    const long long steps = block_steps(v, nllr);
    if (steps < 0) return ATKDSP_E_ARG;
    const int term = (v->mode == ATKDSP_CONV_TERM);
    if (term && steps < v->K - 1) return ATKDSP_E_ARG;
    const size_t nbits = term ? (size_t)steps - (size_t)(v->K - 1) : (size_t)steps;
    if (nbits < 1 || nbits > v->max_bits) return ATKDSP_E_ARG;
    if (cap < nbits) return ATKDSP_E_CAP;
    size_t *off = v->off;
    {
        size_t o = 0;
        for (long long t = 0; t < steps; ++t) { off[t] = o; o += (size_t)v->kp[t % v->punc_len]; }
        off[steps] = o;
    }
    float x[MAXN];
    if (term) {
        for (int s = 0; s < v->S; ++s) v->pm[s] = -INFINITY;
        v->pm[0] = 0.0f;
        for (long long t = 0; t < steps; ++t) {
            gather(v, llr, off, (size_t)t, x);
            acs_step(v, x, v->dec + (size_t)t * (size_t)v->words);
        }
        traceback(v, 0, steps, 0, v->tbits, 0);
        memcpy(bits, v->tbits, nbits);
    } else {
        const size_t W = (v->depth && (size_t)v->depth < (size_t)steps) ? (size_t)v->depth : (size_t)steps;
        const size_t total = W + (size_t)steps + W;
        for (int s = 0; s < v->S; ++s) v->pm[s] = 0.0f;
        for (size_t e = 0; e < total; ++e) {
            const size_t t = (e + (size_t)steps - W) % (size_t)steps;   /* block step index */
            gather(v, llr, off, t, x);
            acs_step(v, x, v->dec + e * (size_t)v->words);
        }
        traceback(v, best_state(v), (long long)total, 0, v->tbits, 0);
        memcpy(bits, v->tbits + W, nbits);
    }
    if (metric) *metric = block_metric(v, llr, nllr, bits, nbits);
    for (int s = 0; s < v->S; ++s) v->pm[s] = 0.0f;
    v->st.calls++; v->st.in += nllr; v->st.out += nbits;
    return (ptrdiff_t)nbits;
}

/* STREAM: release the oldest `depth` undecided bits once 2*depth are held. */
static ptrdiff_t stream_release(atkdsp_viterbi *v, uint8_t *bits, size_t cap, size_t *w, int final) {
    const unsigned long long D = (unsigned long long)v->depth;
    if (!final) {
        if (v->steps_done < v->released + 2 * D) return 0;
        if (*w + D > cap) return ATKDSP_E_CAP;
        traceback(v, best_state(v), (long long)v->steps_done, (long long)v->released, v->tbits, 1);
        memcpy(bits + *w, v->tbits, (size_t)D);
        *w += (size_t)D;
        v->released += D;
        return 0;
    }
    const unsigned long long left = v->steps_done - v->released;
    if (!left) return 0;
    if (*w + left > cap) return ATKDSP_E_CAP;
    traceback(v, best_state(v), (long long)v->steps_done, (long long)v->released, bits + *w, 1);
    *w += (size_t)left;
    v->released = v->steps_done;
    return 0;
}

ptrdiff_t atkdsp_viterbi_decode(atkdsp_viterbi *v, const float *llr, size_t nllr,
                                uint8_t *bits, size_t cap, float *metric) {
    if (!v || (!llr && nllr) || (!bits && cap)) return ATKDSP_E_ARG;
    if (v->mode != ATKDSP_CONV_STREAM) return decode_block(v, llr, nllr, bits, cap, metric);
    if (metric) *metric = NAN;
    if (cap < atkdsp_viterbi_out_max(v, nllr)) return ATKDSP_E_CAP;
    size_t w = 0, k = 0;
    while (1) {
        /* fill the pending step */
        const size_t j = (size_t)(v->steps_done % (unsigned long long)v->punc_len);
        while (v->pend_i < v->n) {
            if (v->keep[v->pend_i * v->punc_len + (int)j]) {
                if (k >= nllr) break;
                v->pend[v->pend_i] = llr[k++];
            } else {
                v->pend[v->pend_i] = 0.0f;
            }
            v->pend_i++;
        }
        if (v->pend_i < v->n) break;
        const size_t row = (size_t)(v->steps_done % (unsigned long long)v->ring);
        acs_step(v, v->pend, v->dec + row * (size_t)v->words);
        v->steps_done++;
        v->pend_i = 0;
        const ptrdiff_t rc = stream_release(v, bits, cap, &w, 0);
        if (rc < 0) return rc;
    }
    v->st.calls++; v->st.in += nllr; v->st.out += w;
    return (ptrdiff_t)w;
}

ptrdiff_t atkdsp_viterbi_flush(atkdsp_viterbi *v, uint8_t *bits, size_t cap) {
    if (!v || (!bits && cap)) return ATKDSP_E_ARG;
    if (v->mode != ATKDSP_CONV_STREAM) return 0;
    size_t w = 0;
    const ptrdiff_t rc = stream_release(v, bits, cap, &w, 1);
    if (rc < 0) return rc;
    v->pend_i = 0;
    v->st.out += w;
    return (ptrdiff_t)w;
}

int atkdsp_viterbi_describe(const atkdsp_viterbi *v, char *buf, size_t cap) {
    if (!v || !buf || !cap) return ATKDSP_E_ARG;
    char polys[96] = "", punc[MAXN * 65 + 48] = "";
    size_t o = 0;
    for (int i = 0; i < v->n; ++i)
        o += (size_t)snprintf(polys + o, sizeof polys - o, "%s0%o", i ? ", " : "", (unsigned)v->polys[i]);
    int punctured = 0;
    for (int i = 0; i < v->n * v->punc_len; ++i) if (!v->keep[i]) punctured = 1;
    if (punctured) {
        o = (size_t)snprintf(punc, sizeof punc, ", punctured %d/%d [", v->punc_len, v->kp_sum);
        for (int i = 0; i < v->n && o < sizeof punc; ++i) {
            if (i) punc[o++] = ' ';
            for (int j = 0; j < v->punc_len && o + 2 < sizeof punc; ++j) punc[o++] = v->keep[i * v->punc_len + j] ? '1' : '0';
        }
        if (o + 2 < sizeof punc) punc[o++] = ']';
        punc[o < sizeof punc ? o : sizeof punc - 1] = 0;
    }
    const char *mode = v->mode == ATKDSP_CONV_TERM ? "terminated"
                     : v->mode == ATKDSP_CONV_TAILBITE ? "tail-biting" : "streaming";
    if (v->mode == ATKDSP_CONV_STREAM)
        snprintf(buf, cap, "Viterbi K=%d r=1/%d (%s)%s, %s, depth %d", v->K, v->n, polys, punc, mode, v->depth);
    else if (v->mode == ATKDSP_CONV_TAILBITE)
        snprintf(buf, cap, "Viterbi K=%d r=1/%d (%s)%s, %s, wrap %s, max %zu bits", v->K, v->n,
                 polys, punc, mode, v->depth ? "depth" : "three laps", v->max_bits);
    else
        snprintf(buf, cap, "Viterbi K=%d r=1/%d (%s)%s, %s, max %zu bits", v->K, v->n, polys, punc,
                 mode, v->max_bits);
    return ATKDSP_OK;
}

int atkdsp_viterbi_stats(const atkdsp_viterbi *v, atkdsp_fec_stats *st) {
    if (!v || !st) return ATKDSP_E_ARG;
    *st = v->st;
    return ATKDSP_OK;
}
