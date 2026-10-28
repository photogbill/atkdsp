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
    const float s0re = s[0].re;
    n = atkdsp_unpack(raw, 7, ATKDSP_FMT_CI16Q11, s, 4, 0.0f, NULL);
    CHECK(n == 1, "torn tail ignored");

    /* -- unpack_dc: the offset folded into the scale, the mean returned --- */
    {
        atkdsp_cf32 t[4];
        double mr = 0.0, mi = 0.0;
        n = atkdsp_unpack_dc(raw, sizeof raw, ATKDSP_FMT_CI16Q11, t, 4,
                             0.25f, -0.5f, &mr, &mi);
        CHECK(n == 4, "unpack_dc count");
        /* same conversion as above, minus the offset */
        CHECK(fabsf(t[0].re - (s0re - 0.25f)) < 1e-6f, "unpack_dc removes the offset");
        /* the mean is of the samples BEFORE the offset came out */
        double want_r = 0.0, want_i = 0.0;
        for (int q = 0; q < 4; ++q) {
            want_r += (double)raw[2 * q] / 2048.0;
            want_i += (double)raw[2 * q + 1] / 2048.0;
        }
        CHECK(fabs(mr - want_r / 4.0) < 1e-6 && fabs(mi - want_i / 4.0) < 1e-6,
              "unpack_dc mean is taken before the subtraction");
        n = atkdsp_unpack_dc(raw, sizeof raw, ATKDSP_FMT_CI16Q11, t, 4,
                             0.0f, 0.0f, NULL, NULL);
        CHECK(n == 4 && fabsf(t[0].re - s0re) < 1e-7f,
              "unpack_dc with no offset is unpack");
    }

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

    /* -- the bitstream engine (v0.14.0): one line per kernel --------------- */
    {
        /* CRC: the catalogue's check values, and the engine on a bit message */
        const char *msg = "123456789";
        uint8_t mb[72];
        for (int i = 0; i < 9; ++i)
            for (int k = 0; k < 8; ++k) mb[8 * i + k] = (uint8_t)(((unsigned char)msg[i] >> (7 - k)) & 1u);
        int okc = atkdsp_crc_catalog_count() >= 100;
        for (int i = 0; i < atkdsp_crc_catalog_count(); ++i) {
            atkdsp_crcspec cs; uint64_t v = 0;
            atkdsp_crc_catalog(i, &cs);
            if (atkdsp_crc(mb, 72, &cs, &v) != ATKDSP_OK || v != cs.check) okc = 0;
        }
        CHECK(okc, "crc: every catalogue set reproduces its check value");

        /* an LFSR, x^15 + x^14 + 1 (siga.bits mask 0xC001) */
        enum { NB = 4096 };
        uint8_t *zb = (uint8_t *)calloc(NB, 1), *seq = (uint8_t *)malloc(NB);
        uint64_t st = 0x2A5D;
        atkdsp_descramble_add(zb, NB, 0xC001, &st, seq);
        uint8_t poly[64];
        ptrdiff_t Lc = atkdsp_bm(seq, 30, poly, sizeof poly);
        CHECK(Lc == 15 && poly[0] == 1 && poly[1] == 1 && poly[15] == 1, "bm: 30 bits give the 15-stage register");
        int32_t prof[NB];
        atkdsp_bm_profile(seq, NB, prof);
        CHECK(prof[NB - 1] == 15, "bm_profile plateaus at 15");
        uint8_t *sc = (uint8_t *)malloc(NB), *ds = (uint8_t *)malloc(NB);
        uint64_t s1 = 0x123, s2 = 0x777;
        atkdsp_scramble_mult(seq, NB, 0x8011, &s1, sc);
        atkdsp_descramble_mult(sc, NB, 0x8011, &s2, ds);
        CHECK(memcmp(ds + 15, seq + 15, NB - 15) == 0, "multiplicative descrambler self-synchronises");

        /* bit primitives on a period-37 stream */
        uint8_t *pb = (uint8_t *)malloc(NB);
        for (int i = 0; i < NB; ++i) pb[i] = seq[i % 37];
        float ac[80];
        atkdsp_bits_autocorr(pb, NB, 79, ac);
        CHECK(ac[37] == 1.0f && ac[74] == 1.0f, "bits_autocorr sees the period");
        float zw[120];
        atkdsp_bits_width_scan(pb, NB, 2, 121, zw, NULL);
        int arg = 0; for (int i = 1; i < 120; ++i) if (zw[i] > zw[arg]) arg = i;
        CHECK(arg + 2 == 37 || (arg + 2) % 37 == 0, "bits_width_scan peaks at the period");
        atkdsp_bithit hit[8];
        ptrdiff_t nh = atkdsp_bits_search(pb, 200, pb + 10, 20, 0, 0, hit, 8);
        CHECK(nh >= 1 && hit[0].pos == 10, "bits_search finds a planted word");
        atkdsp_bitstats bs;
        CHECK(atkdsp_bits_stats(seq, NB, &bs) == ATKDSP_OK && bs.n == NB, "bits_stats runs");
        uint32_t cols[2] = {0, 1}; int32_t spec[4];
        CHECK(atkdsp_bits_wht(pb, NB, 37, cols, 2, spec) == NB / 37 && spec[0] == NB / 37, "bits_wht counts rows");
        float sq[256], dy[256], ct[256], cf[129];
        CHECK(atkdsp_bits_walsh(seq, NB, 256, 0, sq, dy) == 16 && fabsf(dy[0] - 1.0f) < 1e-6f, "bits_walsh");
        CHECK(atkdsp_bits_cyclic(seq, NB, 256, 0, ct, cf) == 16 && ct[0] == 1.0f, "bits_cyclic");
        atkdsp_synccand sy[4];
        CHECK(atkdsp_bits_sync_discover(seq, NB, 16, 3, 0.0, sy, 4) == 0, "sync_discover: an m-sequence holds no recurring word");

        /* Viterbi K=7 r=1/2 round trip, terminated */
        const uint32_t g7[2] = {0171, 0133};
        uint8_t code[2 * (200 + 6)], dec[200];
        float llr[2 * (200 + 6)], met = 0.0f;
        ptrdiff_t nc = atkdsp_conv_encode(7, 2, g7, NULL, 0, ATKDSP_CONV_TERM, seq, 200, code, sizeof code);
        for (ptrdiff_t i = 0; i < nc; ++i) llr[i] = code[i] ? -1.0f : 1.0f;
        llr[17] = -llr[17]; llr[90] = -llr[90];              /* two channel errors */
        atkdsp_viterbi *vt = atkdsp_viterbi_create(7, 2, g7, NULL, 0, ATKDSP_CONV_TERM, 0, 200);
        ptrdiff_t nd = atkdsp_viterbi_decode(vt, llr, (size_t)nc, dec, sizeof dec, &met);
        CHECK(nd == 200 && memcmp(dec, seq, 200) == 0 && met > 0.97f, "viterbi corrects two errors");
        atkdsp_viterbi_destroy(vt);

        /* CCSDS RS(255,223) in the dual basis: 16 errors */
        atkdsp_rs *rs = atkdsp_rs_create(8, 0x187, 112, 11, 32, 0, ATKDSP_RS_DUAL_BASIS);
        uint16_t cw[255], bad[255];
        for (int i = 0; i < 223; ++i) cw[i] = (uint16_t)((i * 37 + 11) & 0xff);
        atkdsp_rs_encode(rs, cw, cw + 223);
        memcpy(bad, cw, sizeof cw);
        for (int i = 0; i < 16; ++i) bad[i * 15] ^= (uint16_t)(1 + i);
        CHECK(rs && atkdsp_rs_decode(rs, bad, NULL, 0, NULL) == 16 && memcmp(bad, cw, sizeof cw) == 0,
              "rs: CCSDS RS(255,223) corrects 16 errors");
        atkdsp_rs_destroy(rs);

        /* BCH(31,21): POCSAG's generator 0x769 */
        atkdsp_bch *bch = atkdsp_bch_create(5, 2, 0, 0);
        uint8_t gen[11]; unsigned gv = 0;
        atkdsp_bch_generator(bch, gen, sizeof gen);
        for (int i = 0; i < 11; ++i) gv = (gv << 1) | gen[i];
        uint8_t bc[31];
        atkdsp_bch_encode(bch, seq, bc);
        bc[3] ^= 1; bc[29] ^= 1;
        CHECK(gv == 0x769 && atkdsp_bch_decode(bch, bc, NULL) == 2 && memcmp(bc, seq, 21) == 0,
              "bch: BCH(31,21) is POCSAG's code and corrects two");
        atkdsp_bch_destroy(bch);

        /* Hamming(7,4) by syndrome table */
        const uint8_t G74[28] = {0,0,0,1,0,1,1, 0,0,1,0,1,1,0, 0,1,0,1,1,0,0, 1,0,1,1,0,0,0};
        atkdsp_blk *blk = atkdsp_blk_create(7, 4, G74);
        uint8_t d4[4] = {1, 0, 1, 1}, c7[7], o4[4]; int dmin = 0;
        atkdsp_blk_encode(blk, d4, c7);
        c7[2] ^= 1;
        atkdsp_blk_params(blk, NULL, NULL, &dmin);
        CHECK(dmin == 3 && atkdsp_blk_decode(blk, c7, o4) == 1 && memcmp(o4, d4, 4) == 0,
              "blk: Hamming(7,4) dmin 3, corrects one");
        atkdsp_blk_destroy(blk);

        /* CRC recovery: four XMODEM frames of different lengths */
        atkdsp_crcspec xm; atkdsp_crc_catalog(0, &xm);
        for (int i = 0; i < atkdsp_crc_catalog_count(); ++i) {
            if (strcmp(atkdsp_crc_catalog_name(i), "CRC-16/XMODEM") == 0) { atkdsp_crc_catalog(i, &xm); break; }
        }
        uint8_t fr[4 * 400]; size_t lens[4]; size_t o = 0;
        for (int f = 0; f < 4; ++f) {
            const size_t L = 80 + 37 * (size_t)f;
            memcpy(fr + o, seq + 100 * f, L);
            uint64_t v = 0; atkdsp_crc(fr + o, L, &xm, &v);
            for (int k = 0; k < 16; ++k) fr[o + L + k] = (uint8_t)((v >> (15 - k)) & 1u);
            lens[f] = L + 16; o += L + 16;
        }
        atkdsp_crccand cc[4];
        ptrdiff_t ncr = atkdsp_crc_recover(fr, lens, 4, 16, ATKDSP_CRC_END, 0, 0.0, cc, 4);
        CHECK(ncr >= 1 && cc[0].spec.poly == 0x1021 && cc[0].catalog >= 0, "crc_recover finds XMODEM from four frames");
        free(zb); free(seq); free(sc); free(ds); free(pb);
    }

    free(x); free(y); free(z); free(win); free(line); free(fm); free(scratch);
    printf("%d failure(s)\n", fails);
    return fails ? 1 : 0;
}
