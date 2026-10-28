/* Statistics the detectors share: the incomplete gamma (chi-square and G
 * tail probabilities), a binomial upper tail in log space, the normal
 * quantile, and the G statistic of a bit stream's words. Each has a twin of
 * the same arithmetic in python/atkdsp/reference.py (§14 helpers), so the
 * thresholds C and the twin derive are the same numbers. */
#include "internal.h"

#define LN10 2.302585092994045684

/* Series for the lower regularised gamma P(a, x), valid for x < a + 1. */
static double gamma_p_series(double a, double x) {
    double ap = a, del = 1.0 / a, sum = del;
    for (int i = 0; i < 1000; ++i) {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if (fabs(del) < fabs(sum) * 1e-16) break;
    }
    return sum * exp(-x + a * log(x) - lgamma(a));
}

/* ln of the continued fraction part of Q(a, x), x >= a + 1 (modified Lentz). */
static double gamma_q_cf_log(double a, double x) {
    const double tiny = 1e-300;
    double b = x + 1.0 - a, c = 1.0 / tiny, d = 1.0 / b, h = d;
    for (int i = 1; i < 1000; ++i) {
        const double an = -(double)i * ((double)i - a);
        b += 2.0;
        d = an * d + b;
        if (fabs(d) < tiny) d = tiny;
        c = b + an / c;
        if (fabs(c) < tiny) c = tiny;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (fabs(del - 1.0) < 1e-16) break;
    }
    return log(h);
}

double atk_log_igamc(double a, double x) {
    if (!(a > 0.0)) return 0.0;
    if (!(x > 0.0)) return 0.0;
    if (x < a + 1.0) {
        const double p = gamma_p_series(a, x);
        return p >= 1.0 ? -INFINITY : log1p(-p);
    }
    return -x + a * log(x) - lgamma(a) + gamma_q_cf_log(a, x);
}

double atk_igamc(double a, double x) {
    return exp(atk_log_igamc(a, x));
}

static double ln_binom_term(double n, double k, double ln_q, double ln_1mq) {
    return lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0) + k * ln_q + (n - k) * ln_1mq;
}

double atk_log10_binom_tail(double n, double c, double ln_q) {
    if (c <= 0.0) return 0.0;
    if (c > n) return -INFINITY;
    const double q = exp(ln_q);
    if (c <= n * q) return 0.0;            /* at or below the mean: no evidence */
    const double ln_1mq = log1p(-q);
    const double t0 = ln_binom_term(n, c, ln_q, ln_1mq);
    double s = 1.0;
    for (double k = c + 1.0; k <= n; k += 1.0) {
        const double d = ln_binom_term(n, k, ln_q, ln_1mq) - t0;
        if (d < -40.0) break;
        s += exp(d);
    }
    const double r = (t0 + log(s)) / LN10;
    return r > 0.0 ? 0.0 : r;
}

/* Acklam's rational approximation to the normal quantile, with one Halley
 * step against erfc. Used for thresholds only. */
static double norm_ppf(double q) {
    static const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01,
        2.506628277459239e+00};
    static const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01};
    static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00,
        2.938163982698783e+00};
    static const double d[4] = {7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00};
    const double plow = 0.02425;
    double x;
    if (q <= 0.0) return -INFINITY;
    if (q >= 1.0) return INFINITY;
    if (q < plow) {
        const double t = sqrt(-2.0 * log(q));
        x = (((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5])
          / ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1.0);
    } else if (q <= 1.0 - plow) {
        const double r = q - 0.5, s = r * r;
        x = (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r
          / (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1.0);
    } else {
        const double t = sqrt(-2.0 * log(1.0 - q));
        x = -(((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5])
          / ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1.0);
    }
    const double e = 0.5 * erfc(-x / sqrt(2.0)) - q;
    const double u = e * sqrt(2.0 * ATK_PI) * exp(x * x / 2.0);
    return x - u / (1.0 + x * u / 2.0);
}

double atk_norm_isf(double p) { return -norm_ppf(p); }

int atk_chi_width(size_t nbits) {
    for (int w = 8; w >= 1; --w)
        if (nbits / (size_t)w >= (size_t)5 << w) return w;
    return 0;
}

double atk_gstat(const uint8_t *b, size_t n, int w, double *dof) {
    if (w < 1 || w > 8) { if (dof) *dof = 0.0; return NAN; }
    uint64_t cnt[256];
    memset(cnt, 0, sizeof cnt);
    const size_t nw = n / (size_t)w;
    for (size_t i = 0; i < nw; ++i) {
        unsigned v = 0;
        const uint8_t *p = b + i * (size_t)w;
        for (int k = 0; k < w; ++k) v = (v << 1) | (p[k] & 1u);
        cnt[v]++;
    }
    const int cells = 1 << w;
    if (dof) *dof = (double)(cells - 1);
    if (nw == 0) return NAN;
    const double e = (double)nw / (double)cells;
    double g = 0.0;
    for (int v = 0; v < cells; ++v)
        if (cnt[v]) g += (double)cnt[v] * log((double)cnt[v] / e);
    return 2.0 * g;
}
