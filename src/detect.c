/* 8. detection on a spectrum line, and the sweep stitch. */
#include "internal.h"

/* quickselect: k-th smallest of a[0..n) with signed indices (Hoare) */
static float select_kth(float *a, ptrdiff_t n, ptrdiff_t k) {
    ptrdiff_t lo = 0, hi = n - 1;
    while (lo < hi) {
        const float pivot = a[lo + (hi - lo) / 2];
        ptrdiff_t i = lo, j = hi;
        while (i <= j) {
            while (a[i] < pivot) ++i;
            while (a[j] > pivot) --j;
            if (i <= j) {
                const float t = a[i]; a[i] = a[j]; a[j] = t;
                ++i; --j;
            }
        }
        if (k <= j) hi = j;
        else if (k >= i) lo = i;
        else break;
    }
    return a[k];
}

float atkdsp_median(const float *x, size_t n, float *scratch) {
    if (!x || !scratch || n == 0) return -120.0f;
    memcpy(scratch, x, n * sizeof(float));
    const ptrdiff_t m = (ptrdiff_t)n;
    if (n & 1) return select_kth(scratch, m, m / 2);
    /* numpy: mean of the two middle values */
    const float hi = select_kth(scratch, m, m / 2);
    const float lo = select_kth(scratch, m, m / 2 - 1);
    return 0.5f * (lo + hi);
}

/* 5-bin moving average with edge padding: identical to
 * np.convolve(np.pad(x, 2, mode="edge"), ones(5)/5, "valid") */
static void smooth5(const float *x, size_t n, float *out) {
    if (n < 5) { memcpy(out, x, n * sizeof(float)); return; }
    for (size_t i = 0; i < n; ++i) {
        float s = 0.0f;
        for (int d = -2; d <= 2; ++d) {
            ptrdiff_t j = (ptrdiff_t)i + d;
            if (j < 0) j = 0;
            if (j >= (ptrdiff_t)n) j = (ptrdiff_t)n - 1;
            s += x[j];
        }
        out[i] = s * 0.2f;
    }
}

static int cmp_snr_desc(const void *a, const void *b) {
    const float x = ((const atkdsp_channel *)a)->snr_db;
    const float y = ((const atkdsp_channel *)b)->snr_db;
    return (x < y) - (x > y);
}

ptrdiff_t atkdsp_detect_channels(const float *line, size_t n, float floor_db,
                                 float threshold_db, int gap_bins, int min_run,
                                 float *smooth_scratch, atkdsp_channel *out, size_t out_cap) {
    if (!line || !smooth_scratch || (!out && out_cap)) return ATKDSP_E_ARG;
    if (n == 0) return 0;
    if (floor_db != floor_db) return ATKDSP_E_ARG;   /* NaN: caller computes it (atkdsp_median) */
    float *sm = smooth_scratch;
    smooth5(line, n, sm);
    const float thr = floor_db + threshold_db;
    if (gap_bins < 2) gap_bins = 2;
    if (min_run < 2) min_run = 2;

    /* mask lives in the scratch as 0/1 floats after this point */
    for (size_t i = 0; i < n; ++i) sm[i] = sm[i] > thr ? 1.0f : 0.0f;

    /* bridge interior gaps shorter than gap_bins */
    size_t i = 0;
    while (i < n) {
        if (sm[i] == 0.0f) {
            size_t j = i;
            while (j < n && sm[j] == 0.0f) ++j;
            if (i > 0 && j < n && (j - i) < (size_t)gap_bins)
                for (size_t k = i; k < j; ++k) sm[k] = 1.0f;
            i = j;
        } else ++i;
    }

    size_t found = 0, kept = 0;
    i = 0;
    while (i < n) {
        if (sm[i] == 0.0f) { ++i; continue; }
        size_t j = i;
        while (j < n && sm[j] != 0.0f) ++j;
        const size_t width = j - i;
        if (width >= (size_t)min_run) {
            ++found;
            if (kept < out_cap) {
                double wsum = 0.0, csum = 0.0;
                float peak = line[i];
                for (size_t k = i; k < j; ++k) {
                    const double w = pow(10.0, (double)line[k] / 10.0);
                    wsum += w; csum += (double)(k - i) * w;
                    if (line[k] > peak) peak = line[k];
                }
                atkdsp_channel *c = &out[kept++];
                c->start_bin = (int)i; c->end_bin = (int)j;
                c->centroid_bin = (float)((double)i + csum / (wsum + 1e-12));
                c->peak_db = peak;
                c->snr_db = peak - floor_db;
            }
        }
        i = j;
    }
    if (found > out_cap) return ATKDSP_E_CAP;
    if (kept > 1) qsort(out, kept, sizeof *out, cmp_snr_desc);
    return (ptrdiff_t)kept;
}

void atkdsp_stitch_max(const float *seg, size_t nseg,
                       double seg_lo_hz, double seg_hz_per_bin,
                       double keep_lo_hz, double keep_hi_hz,
                       double out_lo_hz, double out_hz_per_bin,
                       float *out, size_t nout) {
    if (!seg || !out || nout == 0 || seg_hz_per_bin <= 0.0 || out_hz_per_bin <= 0.0) return;
    const double out_hi_hz = out_lo_hz + out_hz_per_bin * (double)nout;
    for (size_t i = 0; i < nseg; ++i) {
        const double f = seg_lo_hz + ((double)i + 0.5) * seg_hz_per_bin;
        if (f < keep_lo_hz || f > keep_hi_hz) continue;
        if (f < out_lo_hz || f > out_hi_hz) continue;
        ptrdiff_t k = (ptrdiff_t)((f - out_lo_hz) / out_hz_per_bin);
        if (k < 0) k = 0;
        if (k >= (ptrdiff_t)nout) k = (ptrdiff_t)nout - 1;
        const float v = seg[i];
        const float cur = out[k];
        if (cur != cur || v > cur) out[k] = v;   /* NaN = unmeasured */
    }
}
