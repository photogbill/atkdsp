/* The polyphase filterbank (atkdsp_pfb) on its own: how many times real time
 * it runs, one radio and the Kraken DF case, at 1, 2, ... threads.
 *
 *   build:  build.sh / build.bat build it as bin/atkdsp_bench_pfb
 *   run:    bin/atkdsp_bench_pfb [max_threads]      (default: every core)
 *
 * The configurations are ATK's: the trunking channelizer's ~100 kHz grid at
 * 2.4 to 56 MS/s (fs/100 kHz channels, 2x oversampled, 12 taps a phase,
 * 80 dB), the survey's 64-channel scale, and the Kraken DF bank (192
 * channels x 5 elements). Each is 2 s of noise fed in 64k-sample blocks (the
 * size ATK's worker hands over), best of three, the output buffer reused.
 * "x real time" is seconds of signal per second of wall clock: 1.0 means the
 * worker is exactly saturated. */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200112L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "atkdsp.h"
#ifdef _WIN32
#include <windows.h>
static double now(void) { LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
                          return (double)c.QuadPart / (double)f.QuadPart; }
#else
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
#endif

static double run(double fs, unsigned N, unsigned nel, const atkdsp_cf32 *const *x, size_t n,
                  size_t blk, atkdsp_cf32 *out, size_t cap) {
    double best = 1e30;
    for (int r = 0; r < 3; ++r) {
        atkdsp_pfb *p = atkdsp_pfb_create(fs, N, 2, 12, 80.0, nel);
        if (!p) { printf("create failed\n"); exit(1); }
        double t0 = now();
        for (size_t i = 0; i < n; i += blk) {
            size_t m = (n - i < blk) ? n - i : blk;
            const atkdsp_cf32 *in[8];
            for (unsigned e = 0; e < nel; ++e) in[e] = x[e] + i;
            if (atkdsp_pfb_process_n(p, in, nel, m, out, cap) < 0) { printf("process failed\n"); exit(1); }
        }
        double dt = now() - t0;
        if (dt < best) best = dt;
        atkdsp_pfb_destroy(p);
    }
    return best;
}

int main(int argc, char **argv) {
    int maxth = argc > 1 ? atoi(argv[1]) : atkdsp_get_threads();
    if (maxth < 1) maxth = 1;
    struct { double fs; unsigned N, nel; const char *what; } cfg[] = {
        {2.4e6, 24, 1, "trunking grid, RTL 2.4 MS/s"},
        {2.4e6, 64, 1, "survey scale, 64 channels"},
        {10e6, 100, 1, "trunking grid, 10 MS/s"},
        {20e6, 200, 1, "trunking grid, 20 MS/s"},
        {40e6, 400, 1, "trunking grid, 40 MS/s"},
        {56e6, 560, 1, "trunking grid, 56 MS/s"},
        {2.4e6, 192, 5, "Kraken DF, 192 ch x 5 elements"},
    };
    const size_t blk = 65536;
    printf("atkdsp %s - %s\n", atkdsp_version_string(), atkdsp_build_info());
    printf("%-34s %8s", "configuration", "MS/s");
    int ths[16], nth = 0;
    for (int t = 1; t <= maxth && nth < 16; t = (t < 2 ? t + 1 : t * 2)) ths[nth++] = t;
    if (ths[nth - 1] != maxth && nth < 16) ths[nth++] = maxth;
    for (int i = 0; i < nth; ++i) printf("  %3d thr", ths[i]);
    printf("   (x real time)\n");
    for (size_t c = 0; c < sizeof cfg / sizeof cfg[0]; ++c) {
        const double secs = cfg[c].fs >= 20e6 ? 0.5 : 2.0;
        const size_t n = (size_t)(cfg[c].fs * secs);
        atkdsp_cf32 *x[8];
        unsigned seed = 1;
        for (unsigned e = 0; e < cfg[c].nel; ++e) {
            x[e] = (atkdsp_cf32 *)malloc(n * sizeof *x[e]);
            for (size_t i = 0; i < n; ++i) {
                seed = seed * 1103515245u + 12345u; x[e][i].re = (float)((seed >> 8) & 0xffff) / 65536.0f - 0.5f;
                seed = seed * 1103515245u + 12345u; x[e][i].im = (float)((seed >> 8) & 0xffff) / 65536.0f - 0.5f;
            }
        }
        size_t cap = (size_t)cfg[c].nel * cfg[c].N * (blk / (cfg[c].N / 2) + 2);
        atkdsp_cf32 *out = (atkdsp_cf32 *)calloc(cap, sizeof *out);
        printf("%-34s %8.1f", cfg[c].what, cfg[c].fs / 1e6);
        for (int i = 0; i < nth; ++i) {
            atkdsp_set_threads(ths[i]);
            double dt = run(cfg[c].fs, cfg[c].N, cfg[c].nel, (const atkdsp_cf32 *const *)x, n, blk, out, cap);
            printf("  %7.2f", secs / dt);
            fflush(stdout);
        }
        printf("\n");
        for (unsigned e = 0; e < cfg[c].nel; ++e) free(x[e]);
        free(out);
    }
    return 0;
}
