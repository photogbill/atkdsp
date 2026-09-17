/* 9. filter design — Kaiser-windowed sinc low-pass. Deterministic double
 * arithmetic so the numpy twin reproduces the taps to float precision. */
#include "internal.h"

/* modified Bessel I0 by its power series (converges fast for beta <= ~20) */
static double bessel_i0(double x) {
    double sum = 1.0, term = 1.0;
    const double y = x * 0.5;
    for (int k = 1; k < 500; ++k) {
        term *= (y / (double)k);
        const double t2 = term * term;
        sum += t2;
        if (t2 < sum * 1e-17) break;
    }
    return sum;
}

double atk_kaiser_beta(double atten_db) {
    if (atten_db > 50.0) return 0.1102 * (atten_db - 8.7);
    if (atten_db >= 21.0) return 0.5842 * pow(atten_db - 21.0, 0.4) + 0.07886 * (atten_db - 21.0);
    return 0.0;
}

size_t atk_kaiser_length(double atten_db, double transition_hz, double rate) {
    if (transition_hz <= 0.0 || rate <= 0.0) return 0;
    const double dw = 2.0 * ATK_PI * transition_hz / rate;
    double n = (atten_db > 8.0 ? (atten_db - 8.0) : 1.0) / (2.285 * dw);
    size_t N = (size_t)ceil(n) + 1;
    if (N < 3) N = 3;
    if ((N & 1) == 0) ++N;                    /* odd: symmetric about a tap */
    return N;
}

ptrdiff_t atkdsp_design_lowpass(double fp_hz, double fs_hz, double atten_db,
                                double rate, float *out, size_t out_cap) {
    if (rate <= 0.0 || fp_hz < 0.0 || fs_hz <= fp_hz || fs_hz > rate * 0.5 + 1e-9)
        return ATKDSP_E_ARG;
    const size_t N = atk_kaiser_length(atten_db, fs_hz - fp_hz, rate);
    if (!out) return (ptrdiff_t)N;
    if (out_cap < N) return ATKDSP_E_CAP;
    const double beta = atk_kaiser_beta(atten_db);
    const double i0b = bessel_i0(beta);
    const double fc = 0.5 * (fp_hz + fs_hz) / rate;     /* normalised cutoff (cycles/sample) */
    const double M = (double)(N - 1) * 0.5;
    /* two passes, no scratch: the sum first, then the normalised taps, each
     * tap computed in double and cast ONCE (the twin does exactly this) */
    double sum = 0.0;
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t n = 0; n < N; ++n) {
            const double t = (double)n - M;
            const double x = 2.0 * fc * t;
            const double sinc = (t == 0.0) ? 1.0 : sin(ATK_PI * x) / (ATK_PI * x);
            const double r = t / M;
            const double w = bessel_i0(beta * sqrt(r * r < 1.0 ? 1.0 - r * r : 0.0)) / i0b;
            const double h = 2.0 * fc * sinc * w;
            if (pass == 0) sum += h;
            else out[n] = (float)(h / sum);
        }
        if (sum == 0.0) sum = 1.0;
    }
    return (ptrdiff_t)N;
}
