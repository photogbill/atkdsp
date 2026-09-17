/* 7. display helpers: LUT mapping and peak-preserving column reduction. */
#include "internal.h"

void atkdsp_db_to_pixels(const float *db, size_t n, float lo, float hi,
                         const uint32_t *lut256, uint32_t *out) {
    if (!db || !lut256 || !out) return;
    const float span = hi - lo;
    const float k = span > 0.0f ? 255.0f / span : 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float v = db[i];
        if (v != v) { out[i] = lut256[0]; continue; }      /* NaN */
        float t = (v - lo) * k;
        if (t < 0.0f) t = 0.0f;
        if (t > 255.0f) t = 255.0f;
        out[i] = lut256[(int)(t + 0.5f)];
    }
}

void atkdsp_decimate_max(const float *in, size_t n, float *out, size_t m) {
    if (!in || !out || m == 0 || n == 0) return;
    if (m >= n) {
        memcpy(out, in, n * sizeof(float));
        for (size_t i = n; i < m; ++i) out[i] = in[n - 1];
        return;
    }
    /* Same partition as atk.ui.spectrum_view.decimate_max: `per = n // m`
     * bins per column, and any remainder folded into the LAST column. */
    const size_t per = n / m;
    for (size_t c = 0; c < m; ++c) {
        const size_t a = c * per;
        const size_t b = (c + 1 == m) ? n : a + per;
        float best = in[a];
        for (size_t i = a + 1; i < b; ++i)
            if (in[i] > best) best = in[i];
        out[c] = best;
    }
}
