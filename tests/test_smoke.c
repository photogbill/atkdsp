#define _USE_MATH_DEFINES
/* C smoke test: the library loads, the ABI answers, every kernel runs on a
 * known signal and produces the expected shape of answer. The NUMERICAL
 * checks against numpy live in tests/test_cross_check.py; this file exists so
 * a build with no Python can still prove the DLL is sane. Exit 0 = pass. */
#include "atkdsp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); ++fails; } \
                              else printf("ok:   %s\n", msg); } while (0)

int main(void) {
    printf("atkdsp %s (%s)\n", atkdsp_version_string(), atkdsp_build_info());
    CHECK(atkdsp_abi_version() == ATKDSP_ABI_VERSION, "abi version matches header");
    CHECK(atkdsp_bytes_per_sample(ATKDSP_FMT_CI16Q11) == 4, "ci16q11 is 4 bytes");
    CHECK(atkdsp_bytes_per_sample(99) == 0, "unknown format is 0");

    /* -- unpack ci16q11 ------------------------------------------------- */
    int16_t raw[8] = { 2048, 0, 0, -2048, 1024, 1024, 7, 7 };  /* 4 samples + torn? no: 8 halves */
    atkdsp_cf32 s[4];
    ptrdiff_t n = atkdsp_unpack(raw, sizeof raw, ATKDSP_FMT_CI16Q11, s, 4, 0.0f, NULL);
    CHECK(n == 4, "unpack count");
    CHECK(fabsf(s[0].re - 1.0f) < 1e-6f && fabsf(s[1].im + 1.0f) < 1e-6f, "unpack q11 scale");
    n = atkdsp_unpack(raw, 7, ATKDSP_FMT_CI16Q11, s, 4, 0.0f, NULL);
    CHECK(n == 1, "torn tail ignored");

    /* -- NCO: mixing a tone at f down to DC leaves a constant ------------ */
    const size_t N = 4096;
    atkdsp_cf32 *x = (atkdsp_cf32 *)malloc(N * sizeof *x);
    atkdsp_cf32 *y = (atkdsp_cf32 *)malloc(N * sizeof *y);
    const double fs = 1e6, f0 = 12345.0;
    for (size_t i = 0; i < N; ++i) {
        x[i].re = (float)cos(2 * M_PI * f0 * (double)i / fs);
        x[i].im = (float)sin(2 * M_PI * f0 * (double)i / fs);
    }
    atkdsp_nco nco; atkdsp_nco_init(&nco, f0, fs);
    atkdsp_nco_mix(&nco, x, y, N / 2);
    atkdsp_nco_mix(&nco, x + N / 2, y + N / 2, N / 2);     /* second block: continuous? */
    float worst = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        float d = fabsf(y[i].re - 1.0f) + fabsf(y[i].im);
        if (d > worst) worst = d;
    }
    CHECK(worst < 2e-4f, "nco mixes a tone to DC continuously across blocks");

    /* -- FIR: DC through a unity-gain low-pass, decimated ---------------- */
    float taps[33];
    for (int i = 0; i < 33; ++i) taps[i] = 1.0f / 33.0f;
    atkdsp_fir *fir = atkdsp_fir_create(taps, 33, 8);
    CHECK(fir != NULL, "fir create");
    for (size_t i = 0; i < N; ++i) { x[i].re = 1.0f; x[i].im = 0.5f; }
    size_t cap = atkdsp_fir_out_max(fir, N);
    n = atkdsp_fir_process(fir, x, N, y, cap);
    CHECK(n == (ptrdiff_t)(N / 8), "fir output count");
    CHECK(fabsf(y[n - 1].re - 1.0f) < 1e-5f && fabsf(y[n - 1].im - 0.5f) < 1e-5f, "fir dc gain");
    atkdsp_fir_destroy(fir);

    /* -- resampler: 3/2 on N inputs gives N*3/2 outputs over two calls --- */
    float rt[48];
    for (int i = 0; i < 48; ++i) rt[i] = 1.0f / 48.0f;
    atkdsp_resampler *rs = atkdsp_resampler_create(3, 2, rt, 48);
    CHECK(rs != NULL, "resampler create");
    size_t total = 0;
    atkdsp_cf32 *z = (atkdsp_cf32 *)malloc(2 * N * sizeof *z);
    n = atkdsp_resampler_process(rs, x, 1000, z, 2 * N); total += (size_t)n;
    n = atkdsp_resampler_process(rs, x, N - 1000, z + total, 2 * N - total); total += (size_t)n;
    CHECK(total == N * 3 / 2, "resampler exact count across blocks");
    atkdsp_resampler_destroy(rs);

    /* -- FFT + POI reduce: a tone lands in the right bin ----------------- */
    const size_t nfft = 1024;
    atkdsp_fft *p = atkdsp_fft_create(nfft);
    CHECK(p != NULL && atkdsp_fft_length(p) == nfft, "fft create");
    for (size_t i = 0; i < N; ++i) {
        x[i].re = (float)cos(2 * M_PI * 100.0 * (double)i / (double)nfft);   /* bin +100 */
        x[i].im = (float)sin(2 * M_PI * 100.0 * (double)i / (double)nfft);
    }
    float *win = (float *)malloc(nfft * sizeof(float));
    float *line = (float *)malloc(nfft * sizeof(float));
    atkdsp_window(ATKDSP_WIN_HANN, nfft, win);
    ptrdiff_t frames = atkdsp_spectrum_reduce(p, x, N, nfft, win, ATKDSP_DET_MAX, line);
    CHECK(frames == (ptrdiff_t)(N / nfft), "reduce frame count");
    size_t best = 0;
    for (size_t i = 1; i < nfft; ++i) if (line[i] > line[best]) best = i;
    CHECK(best == nfft / 2 + 100, "tone in the right fftshifted bin");
    float pk[1024];
    CHECK(atkdsp_power_db(p, x, win, pk) == ATKDSP_OK, "power_db runs");
    CHECK(fabsf(pk[best] - line[best]) < 1e-3f, "reduce(max) of identical frames == one frame");
    atkdsp_fft_destroy(p);

    /* -- FM demod: constant tone -> constant frequency ------------------- */
    atkdsp_cf32 prev = { 0.0f, 0.0f };
    float *fm = (float *)malloc(N * sizeof(float));
    atkdsp_fm_demod(x, N, fm, &prev, 1.0f);
    const float expect = (float)(2 * M_PI * 100.0 / (double)nfft);
    CHECK(fabsf(fm[10] - expect) < 1e-4f && fabsf(fm[N - 1] - expect) < 1e-4f, "fm demod frequency");

    /* -- detect_channels: one carrier in noise-free floor ---------------- */
    float *scratch = (float *)malloc(nfft * sizeof(float));
    for (size_t i = 0; i < nfft; ++i) line[i] = -100.0f;
    for (size_t i = 500; i < 520; ++i) line[i] = -40.0f;
    float floor_db = atkdsp_median(line, nfft, scratch);
    atkdsp_channel ch[4];
    n = atkdsp_detect_channels(line, nfft, floor_db, 6.0f, 2, 2, scratch, ch, 4);
    CHECK(n == 1, "one channel found");
    CHECK(n == 1 && ch[0].start_bin <= 500 && ch[0].end_bin >= 520, "channel spans the carrier");
    CHECK(n == 1 && fabsf(ch[0].snr_db - 60.0f) < 0.01f, "channel snr");

    /* -- decimate_max / pixels / stitch ---------------------------------- */
    float cols[4];
    atkdsp_decimate_max(line, nfft, cols, 4);
    CHECK(cols[1] == -40.0f && cols[0] == -100.0f, "decimate_max keeps the peak in its column");
    uint32_t lut[256]; for (int i = 0; i < 256; ++i) lut[i] = (uint32_t)i;
    uint32_t px[4];
    atkdsp_db_to_pixels(cols, 4, -100.0f, -40.0f, lut, px);
    CHECK(px[0] == 0 && px[1] == 255, "db_to_pixels maps lo/hi to 0/255");
    float out[10];
    for (int i = 0; i < 10; ++i) out[i] = NAN;
    float seg[4] = { 1, 2, 3, 4 };
    atkdsp_stitch_max(seg, 4, 0.0, 1.0, 0.0, 4.0, 0.0, 0.5, out, 10);
    CHECK(out[1] == 1.0f && out[3] == 2.0f && out[7] == 4.0f && out[0] != out[0], "stitch_max places bins");

    /* -- design + DDC: a 100 kHz interferer is 60 dB down at 10 MSPS ------ */
    {
        ptrdiff_t nt = atkdsp_design_lowpass(7500.0, 15000.0, 60.0, 48000.0, NULL, 0);
        CHECK(nt > 0 && (nt & 1), "design_lowpass sizes an odd filter");
        const size_t nn = 1000000;   /* 0.1 s at 10 MSPS */
        atkdsp_cf32 *big = (atkdsp_cf32 *)malloc(nn * sizeof *big);
        atkdsp_ddc *d = atkdsp_ddc_create(10e6, 0.0, 15000.0, 48000.0, 60.0, nn);
        CHECK(d != NULL, "ddc create");
        char desc[256]; atkdsp_ddc_describe(d, desc, sizeof desc);
        CHECK(strstr(desc, "48000 Hz") != NULL, "ddc lands on exactly 48000 Hz");
        double lv[2];
        for (int k = 0; k < 2; ++k) {
            const double off = k ? 100000.0 : 0.0;
            for (size_t i = 0; i < nn; ++i) {
                const double ph = 2 * M_PI * off * (double)i / 10e6;
                big[i].re = (float)cos(ph); big[i].im = (float)sin(ph);
            }
            atkdsp_ddc_reset(d);
            size_t capd = atkdsp_ddc_out_max(d, nn);
            atkdsp_cf32 *yo = (atkdsp_cf32 *)malloc(capd * sizeof *yo);
            ptrdiff_t m = atkdsp_ddc_process(d, big, nn, yo, capd);
            double acc = 0.0;
            for (ptrdiff_t i = m / 2; i < m; ++i) acc += yo[i].re * yo[i].re + yo[i].im * yo[i].im;
            lv[k] = 10.0 * log10(acc / (double)(m - m / 2) + 1e-30);
            free(yo);
        }
        CHECK(fabs(lv[0]) < 0.5, "ddc passes the wanted channel at 0 dB");
        CHECK(lv[1] - lv[0] < -60.0, "ddc puts a 100 kHz interferer 60 dB down");
        atkdsp_ddc_destroy(d); free(big);
    }

    free(x); free(y); free(z); free(win); free(line); free(fm); free(scratch);
    printf("%d failure(s)\n", fails);
    return fails ? 1 : 0;
}
