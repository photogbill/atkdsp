#define _USE_MATH_DEFINES
/* C smoke test: the library loads, the ABI answers, every kernel runs on a
 * known signal and produces the expected shape of answer. The NUMERICAL
 * checks against numpy live in tests/test_cross_check.py; this file exists so
 * a build with no Python can still prove the DLL is sane. Exit 0 = pass. */
#include "atkdsp.h"
#define ATK_SMOKE_PI 3.14159265358979323846
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

        /* -- naming the code (v0.15.0): one line per recogniser ------------- */
        {
            /* data bits from splitmix64 (an LFSR's bits would add a linear
             * recurrence of their own) */
            uint8_t *rd = (uint8_t *)malloc(NB);
            uint64_t sm = 1;
            for (int i = 0; i < NB; ++i) {
                uint64_t z = (sm += 0x9E3779B97F4A7C15ULL);
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
                rd[i] = (uint8_t)((z ^ (z >> 31)) >> 63);
            }
            /* Hamming(7,4) codewords from the table encoder, 600 of them */
            const size_t NW = 600;
            uint8_t *hs = (uint8_t *)malloc(NW * 7 + 7);
            atkdsp_blk *h74 = atkdsp_blk_create(7, 4, G74);
            for (size_t w = 0; w < NW; ++w) atkdsp_blk_encode(h74, rd + 4 * w, hs + 7 * w);
            atkdsp_blk_destroy(h74);
            atkdsp_blockid bi;
            int rb = atkdsp_block_code_id(hs + 3, NW * 7 - 3, 4, 64, 1e-3, 0, &bi, NULL, 0, NULL, 0);
            CHECK(rb == 1 && bi.n == 7 && bi.k == 4 && strcmp(bi.name, "Hamming(7,4)") == 0 && bi.agreement == 1.0,
                  "block_code_id names Hamming(7,4)");
            int16_t def[19 * 8];
            CHECK(atkdsp_gf2_rank_scan(hs, NW * 7, 2, 20, 8, 0, def) == ATKDSP_OK && def[(7 - 2) * 8] == 3,
                  "gf2_rank_scan: deficiency 3 at width 7, aligned");
            /* K = 7 r = 1/2 (171,133) */
            const uint32_t p2[2] = {0171, 0133};
            uint8_t *cs = (uint8_t *)malloc(2 * NB + 64);
            ptrdiff_t nc = atkdsp_conv_encode(7, 2, p2, NULL, 0, ATKDSP_CONV_STREAM, rd, NB, cs, 2 * NB + 64);
            atkdsp_convid ci;
            int rcv = atkdsp_conv_code_id(cs + 1, (size_t)nc - 1, 8, 15, 1e-3, 0, &ci);
            CHECK(rcv == 1 && ci.n == 2 && ci.K == 7 && strstr(ci.name, "NASA/CCSDS") != NULL,
                  "conv_code_id names K=7 (171,133)");
            /* RS(15,11) over GF(16), 200 codewords, MSB first */
            atkdsp_rs *r15 = atkdsp_rs_create(4, 0x13, 0, 1, 4, 0, 0);
            uint8_t *rb2 = (uint8_t *)malloc(200 * 60);
            for (int w = 0; w < 200; ++w) {
                uint16_t sy[15];
                for (int j = 0; j < 11; ++j) {
                    const int o2 = (w * 44 + j * 4) % (NB - 4);
                    sy[j] = (uint16_t)(rd[o2] << 3 | rd[o2 + 1] << 2 | rd[o2 + 2] << 1 | rd[o2 + 3]);
                }
                atkdsp_rs_encode(r15, sy, sy + 11);
                for (int j = 0; j < 15; ++j) for (int q = 0; q < 4; ++q) rb2[w * 60 + j * 4 + q] = (uint8_t)((sy[j] >> (3 - q)) & 1);
            }
            atkdsp_rs_destroy(r15);
            atkdsp_rsid ri;
            CHECK(atkdsp_rs_code_id(rb2, 200 * 60, 0, 4, 15, 1e-3, &ri) == 1 && ri.k == 11 && ri.gfpoly == 0x13 && ri.fcr == 0,
                  "rs_code_id names RS(15,11) 0x13 fcr 0");
            /* the Hamming stream interleaved 8 x 7 */
            uint8_t *il = (uint8_t *)malloc(NW * 7);
            const size_t nb8 = (NW * 7) / 56;
            for (size_t q = 0; q < nb8; ++q)
                for (int row = 0; row < 8; ++row)
                    for (int col = 0; col < 7; ++col) il[q * 56 + (size_t)col * 8 + (size_t)row] = hs[q * 56 + (size_t)row * 7 + (size_t)col];
            atkdsp_ilvid ii;
            CHECK(atkdsp_interleaver_id(il + 5, nb8 * 56 - 5, 8, 256, 1e-3, 0, &ii) == 1 && ii.period == 56
                  && ii.rows == 8 && ii.cols == 7, "interleaver_id finds 8 x 7");
            /* the negative control */
            CHECK(atkdsp_block_code_id(rd, NB, 4, 64, 1e-3, 0, &bi, NULL, 0, NULL, 0) == 0
                  && strcmp(bi.name, "no code found") == 0, "block_code_id: no code in random bits");
            free(rd); free(hs); free(cs); free(rb2); free(il);
        }
        free(zb); free(seq); free(sc); free(ds); free(pb);
    }

    /* ---- the detection track (v0.16.0) ---- */
    {
        /* CFAR: alpha reproduces CA's closed form; a planted burst gives one track */
        double al = 0.0, et = 0.0;
        CHECK(atkdsp_cfar_alpha(ATKDSP_CFAR_CA, 16, 1e-6, 1.0, 0, NULL, &al, &et) == ATKDSP_OK
              && fabs(al / (32.0 * (pow(1e-6, -1.0 / 32.0) - 1.0)) - 1.0) < 1e-7,
              "cfar_alpha: CA alpha = N (Pfa^(-1/N) - 1)");
        /* v0.17.0: with the Hann window's correlation the equivalent count is
         * about 17 of 32 cells, and the threshold rises accordingly */
        {
            const double hann[4] = {4.0 / 9.0, 1.0 / 36.0, 0.0, 0.0};
            const double ne = atkdsp_cfar_n_eff(16, hann);
            double al2 = 0.0;
            CHECK(ne > 17.0 && ne < 18.0, "cfar_n_eff: Hann, 16 a side -> ~17.4 of 32");
            CHECK(atkdsp_cfar_alpha(ATKDSP_CFAR_CA, 16, 1e-6, 1.0, 0, hann, &al2, NULL) == ATKDSP_OK
                  && al2 > al, "cfar_alpha: correlated cells raise the threshold");
            float w[4096];
            double r[4];
            atkdsp_window(ATKDSP_WIN_HANN, 4096, w);
            CHECK(atkdsp_cfar_window_rho2(w, 4096, r) == ATKDSP_OK
                  && fabs(r[0] - 4.0 / 9.0) < 1e-3 && fabs(r[1] - 1.0 / 36.0) < 1e-3
                  && r[2] < 1e-3 && r[3] < 1e-3, "cfar_window_rho2: Hann -> 4/9, 1/36, 0, 0");
        }
        atkdsp_cfar_config cc;
        atkdsp_cfar_defaults(&cc);
        enum { NBC = 256, NF = 40 };
        atkdsp_cfar *cf = atkdsp_cfar_create(NBC, &cc);
        CHECK(cf != NULL, "cfar create");
        float *fr = (float *)malloc(sizeof(float) * NBC * NF);
        uint64_t st = 12345;
        for (int f = 0; f < NF; ++f)
            for (int b = 0; b < NBC; ++b) {
                st = st * 6364136223846793005ULL + 1442695040888963407ULL;
                const double u = ((double)(st >> 11) + 0.5) / 9007199254740992.0;
                double p = -log(u);                              /* unit exponential noise */
                if (b == 100 && f >= 10 && f < 20) p += 1000.0;  /* a 30 dB burst */
                fr[f * NBC + b] = (float)(10.0 * log10(p));
            }
        const size_t evcap = atkdsp_cfar_events_max(cf, NF) + 256;
        atkdsp_cfar_event *ev = (atkdsp_cfar_event *)malloc(evcap * sizeof *ev);
        ptrdiff_t ne = atkdsp_cfar_process(cf, fr, NF, 0.0, 0.001, ev, evcap);
        ne += atkdsp_cfar_flush(cf, ev + (ne > 0 ? ne : 0), 256);
        int closes = 0, good = 0;
        for (ptrdiff_t i = 0; i < ne; ++i)
            if (ev[i].type == ATKDSP_CFAR_CLOSE) {
                ++closes;
                good = fabs(ev[i].fc_bin - 100.0) < 0.5 && fabs(ev[i].t0 - 0.010) < 1e-9
                       && fabs(ev[i].t1 - 0.020) < 1e-9;
            }
        CHECK(closes == 1 && good, "cfar: one track for one burst, fc and t0/t1 exact");
        char cdesc[512];
        CHECK(atkdsp_cfar_describe(cf, cdesc, sizeof cdesc) == ATKDSP_OK
              && strstr(cdesc, "CFAR CA") != NULL, "cfar describe");
        atkdsp_cfar_destroy(cf);
        free(fr);
        free(ev);

        /* burst_measure: a carrier keyed on at sample 3000, off after 6999 */
        enum { NBM = 10000 };
        atkdsp_cf32 *bx = (atkdsp_cf32 *)malloc(sizeof(atkdsp_cf32) * NBM);
        for (int i = 0; i < NBM; ++i) {
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            const double u1 = ((double)(st >> 11) + 0.5) / 9007199254740992.0;
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            const double u2 = ((double)(st >> 11) + 0.5) / 9007199254740992.0;
            const double rr = sqrt(-log(u1)), th = 2.0 * ATK_SMOKE_PI * u2;   /* unit-power noise */
            double re = rr * cos(th), im = rr * sin(th);
            if (i >= 3000 && i < 7000) {
                re += 10.0 * cos(2.0 * ATK_SMOKE_PI * 0.05 * i);
                im += 10.0 * sin(2.0 * ATK_SMOKE_PI * 0.05 * i);
            }
            bx[i].re = (float)re; bx[i].im = (float)im;
        }
        atkdsp_bdw bd;
        CHECK(atkdsp_burst_measure(bx, NBM, 100000.0, 2900, 7100, 0.0f, 0, &bd) == ATKDSP_OK
              && fabs(bd.t0 - 2999.5) <= 2.0 && fabs(bd.t1 - 6999.5) <= 2.0
              && fabs(bd.fc_hz - 5000.0) < 100.0 && fabs(bd.snr_db - 20.0) < 1.0,
              "burst_measure: edges within 2 samples, fc, SNR");
        free(bx);

        /* ring: lap a consumer and count exactly what it lost */
        const size_t rb = atkdsp_ring_bytes(100, 4, 2);
        void *rm = NULL;
        rm = malloc(rb + 64);
        void *ra = (void *)(((uintptr_t)rm + 63) & ~(uintptr_t)63);
        atkdsp_ring *ring = atkdsp_ring_init(ra, rb, 100, 4, 2);
        CHECK(ring != NULL && atkdsp_ring_attach(ra, rb) == ring, "ring init / attach");
        const int cid = atkdsp_ring_open(ring, 0);
        for (uint64_t q = 0; q < 10; ++q) {
            uint8_t *blkp = (uint8_t *)atkdsp_ring_claim(ring);
            memset(blkp, (int)q, 100);
            atkdsp_ring_meta mm;
            memset(&mm, 0, sizeof mm);
            mm.nbytes = 100;
            atkdsp_ring_commit(ring, &mm);
        }
        uint8_t got[128];
        atkdsp_ring_meta gm;
        uint64_t lostb = 0;
        const ptrdiff_t gn = atkdsp_ring_read(ring, cid, got, sizeof got, &gm, &lostb);
        CHECK(gn == 100 && lostb == 6 && gm.seq == 6 && got[0] == 6, "ring: lapped reader told it lost exactly 6");
        free(rm);
    }

    /* ---- phase 3 (v0.17.0) ---- */
    {
        /* pulses: three 80-sample pulses 20 dB over a unit-power noise floor,
         * split across two calls, come back as three words at their samples */
        enum { NP = 20000, PW = 80 };
        atkdsp_cf32 *pz = (atkdsp_cf32 *)malloc(NP * sizeof *pz);
        uint64_t st = 777;
        for (int i = 0; i < NP; ++i) {
            /* Box-Muller from the same LCG as above */
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            const double u1 = ((double)(st >> 11) + 0.5) / 9007199254740992.0;
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            const double u2 = ((double)(st >> 11) + 0.5) / 9007199254740992.0;
            const double r = sqrt(-log(u1));          /* unit power: E|x|^2 = 1 */
            pz[i].re = (float)(r * cos(2 * M_PI * u2));
            pz[i].im = (float)(r * sin(2 * M_PI * u2));
        }
        const int starts[3] = {1000, 7000, 15000};
        for (int k = 0; k < 3; ++k)
            for (int i = 0; i < PW; ++i) {
                const double ph = 2 * M_PI * 0.05 * i;
                pz[starts[k] + i].re += (float)(10.0 * cos(ph));
                pz[starts[k] + i].im += (float)(10.0 * sin(ph));
            }
        double med = 0.0;
        CHECK(atkdsp_mag_median(pz, NP, &med) == ATKDSP_OK && med > 0.7 && med < 1.0,
              "mag_median: the Rayleigh median of unit-power noise (0.83)");
        atkdsp_pulses_config pc;
        atkdsp_pulses_defaults(&pc);
        pc.fs = 1e6; pc.floor = med;
        atkdsp_pulses *pp = atkdsp_pulses_create(&pc, 0.0);
        CHECK(pp != NULL, "pulses create");
        atkdsp_pdw pw[8];
        ptrdiff_t np_ = atkdsp_pulses_process(pp, pz, 7040, pw, 8);        /* cuts the 2nd pulse */
        np_ += atkdsp_pulses_process(pp, pz + 7040, NP - 7040, pw + (np_ > 0 ? np_ : 0), 8 - (size_t)(np_ > 0 ? np_ : 0));
        np_ += atkdsp_pulses_flush(pp, pw + (np_ > 0 ? np_ : 0), 1);
        int okp = np_ == 3;
        for (int k = 0; k < 3 && okp; ++k)
            okp = pw[k].start == (uint64_t)starts[k] && pw[k].n == PW
                  && fabs(pw[k].freq_hz - 0.05e6) < 2e3 && pw[k].pa_db > 18.0f && pw[k].flags == 0;
        CHECK(okp, "pulses: three pulses, one cut by a block edge, at their samples and frequency");
        char pd[256];
        CHECK(atkdsp_pulses_describe(pp, pd, sizeof pd) == ATKDSP_OK && strstr(pd, "3 pulses") != NULL,
              "pulses describe");
        atkdsp_pulses_destroy(pp);
        free(pz);
    }
    {
        /* cyclic: random on-off keying at 1000 Bd over unit noise, with
         * rectangular symbols, so its clock shows in the transition energy
         * |x[n] - x[n-1]|^2 at 1000 and not at 1300 (and NOT in |x|^2: a
         * rectangular pulse's envelope has no clock line — shaping is what
         * puts it there); then noise alone with random blocks of 64 and a
         * 16-sample prefix correlates at lag 64 and not at 100 */
        enum { NC = 48000 };
        atkdsp_cf32 *cz = (atkdsp_cf32 *)malloc(NC * sizeof *cz);
        uint64_t st = 4242;
        int on = 0;
        for (int i = 0; i < NC; ++i) {
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            const double u1 = ((double)(st >> 11) + 0.5) / 9007199254740992.0;
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            const double u2 = ((double)(st >> 11) + 0.5) / 9007199254740992.0;
            const double r = sqrt(-log(u1));
            if (i % 48 == 0) on = (int)(st >> 40) & 1;             /* a new symbol every 48 samples */
            const double a = on ? 3.0 : 0.0;
            cz[i].re = (float)(r * cos(2 * M_PI * u2) + a * cos(0.3 * i));
            cz[i].im = (float)(r * sin(2 * M_PI * u2) + a * sin(0.3 * i));
        }
        atkdsp_cyc_cand cc[3] = {{1000.0, ATKDSP_NL_DM, 0}, {1300.0, ATKDSP_NL_DM, 0}, {1000.0, ATKDSP_NL_ENV2, 0}};
        atkdsp_cyclic *cy = atkdsp_cyclic_create(48000.0, cc, 3, 960, 50, 1e-3);
        CHECK(cy != NULL, "cyclic create");
        CHECK(atkdsp_cyclic_process(cy, cz, NC) == ATKDSP_OK, "cyclic process");
        atkdsp_cyc_result cr[3];
        CHECK(atkdsp_cyclic_read(cy, cr) == 50 && (cr[0].flags & ATKDSP_CYC_LINE)
              && !(cr[1].flags & ATKDSP_CYC_LINE) && !(cr[2].flags & ATKDSP_CYC_LINE)
              && cr[0].coherence > 0.7,
              "cyclic: the 1000 Bd clock line in the transitions, nothing at 1300 or in |x|^2");
        atkdsp_cyclic_destroy(cy);
        /* a prefixed block stream: 64 fresh noise samples per symbol, the
         * last 16 copied in front (no tone: a carrier correlates at EVERY lag) */
        for (int i = 0; i < NC; ++i) {
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            const double u1 = ((double)(st >> 11) + 0.5) / 9007199254740992.0;
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            const double u2 = ((double)(st >> 11) + 0.5) / 9007199254740992.0;
            const double r = sqrt(-log(u1));
            cz[i].re = (float)(r * cos(2 * M_PI * u2));
            cz[i].im = (float)(r * sin(2 * M_PI * u2));
        }
        for (int i = 0; i + 80 <= NC; i += 80)
            for (int j = 0; j < 16; ++j) cz[i + j] = cz[i + 64 + j];
        atkdsp_cp_cand pc2[2] = {{64, 16, 0, 0}, {100, 16, 0, 0}};
        atkdsp_cp_probe *cp = atkdsp_cp_probe_create(48000.0, pc2, 2, 2400, 20, 1e-3);
        CHECK(cp != NULL, "cp_probe create");
        CHECK(atkdsp_cp_probe_process(cp, cz, NC) == ATKDSP_OK, "cp_probe process");
        atkdsp_cp_result pr2[2];
        CHECK(atkdsp_cp_probe_read(cp, pr2) == 20 && (pr2[0].flags & ATKDSP_CP_DETECT)
              && (pr2[0].flags & ATKDSP_CP_LINE) && !(pr2[1].flags & ATKDSP_CP_DETECT),
              "cp_probe: the prefix correlates at lag 64 with its period, not at 100");
        atkdsp_cp_probe_destroy(cp);
        free(cz);
    }
    {
        /* tone_id: a 127.3 Hz CTCSS tone under 1.2 s, then a DTMF '5' for
         * 80 ms, at 8 kHz; the DCS codec round-trips every standard code */
        enum { NT = 8000 * 2 };
        float *au = (float *)calloc(NT, sizeof(float));
        for (int i = 0; i < 9600; ++i) au[i] = (float)(0.15 * sin(2 * M_PI * 127.3 * i / 8000.0));
        for (int i = 10400; i < 11040; ++i)
            au[i] = (float)(0.3 * (sin(2 * M_PI * 770.0 * i / 8000.0) + sin(2 * M_PI * 1336.0 * i / 8000.0)));
        atkdsp_tone_config tc;
        atkdsp_tone_id_defaults(&tc);
        tc.fs = 8000.0;
        atkdsp_tone_id *ti = atkdsp_tone_id_create(&tc, 0.0);
        CHECK(ti != NULL, "tone_id create");
        atkdsp_tone_event tev[64];
        ptrdiff_t nt = atkdsp_tone_id_process(ti, au, NT / 2, tev, 64);
        nt += atkdsp_tone_id_process(ti, au + NT / 2, NT / 2, tev + nt, 64 - (size_t)nt);
        nt += atkdsp_tone_id_flush(ti, tev + nt, 64 - (size_t)nt);
        int saw_ctcss = 0, saw_dtmf = 0, other = 0;
        for (ptrdiff_t i = 0; i < nt; ++i) {
            if (tev[i].kind == ATKDSP_TONE_CTCSS && tev[i].type == ATKDSP_TONE_END)
                saw_ctcss = tev[i].code == 19 && fabs(tev[i].freq_hz - 127.3) < 0.2 && tev[i].t1 > 1.1;
            else if (tev[i].kind == ATKDSP_TONE_DTMF && tev[i].type == ATKDSP_TONE_END)
                saw_dtmf = tev[i].code == '5' && fabs(tev[i].t0 - 1.3) < 0.02;
            else if (tev[i].type == ATKDSP_TONE_END) ++other;
        }
        CHECK(saw_ctcss && saw_dtmf && !other, "tone_id: CTCSS 127.3 and DTMF 5, nothing else");
        int ok_dcs = 1;
        for (int i = 0; i < 104 && ok_dcs; ++i) {
            int oc = 0, code = -1, alias = -1;
            atkdsp_dcs_code(i, &oc);
            ok_dcs = atkdsp_dcs_decode(atkdsp_dcs_encode(oc), 1, &code, &alias) == 0 && code == oc && alias >= 0;
        }
        CHECK(ok_dcs, "dcs: every standard code encodes and reads back as itself, with its inverted alias");
        atkdsp_tone_id_destroy(ti);
        free(au);
    }

    /* ---- §25-27 spread spectrum (v0.18.0, ABI 19) ---------------------------- */
    {
        /* a 31-chip m-sequence (x^5+x^2+1), 4 samples/chip, BPSK data, at -6 dB */
        const int r = 5, sps = 4, L = 31, N = L * sps, nsym = 4000;
        int8_t code[31]; int st5[5] = {1, 0, 0, 0, 0};
        for (int k = 0; k < L; ++k) {           /* taps bit0,bit2 -> x^5+x^2+1 */
            code[k] = (int8_t)(1 - 2 * st5[0]);
            int fb = st5[0] ^ st5[2];
            for (int i = 0; i < 4; ++i) st5[i] = st5[i + 1];
            st5[4] = fb;
        }
        atkdsp_cf32 *sx = (atkdsp_cf32 *)malloc((size_t)N * nsym * sizeof *sx);
        unsigned rs = 12345u;
        #define URAND (rs = rs * 1103515245u + 12345u, ((double)(rs >> 8) / 16777216.0))
        for (int s = 0; s < nsym; ++s) {
            const double d = (URAND > 0.5) ? 1.0 : -1.0;
            for (int c = 0; c < L; ++c)
                for (int j = 0; j < sps; ++j) {
                    atkdsp_cf32 *v = &sx[(size_t)(s * L + c) * sps + j];
                    v->re = (float)(d * code[c]); v->im = 0.0f;
                }
        }
        double pw = 0.0; const size_t ns = (size_t)N * nsym;
        for (size_t i = 0; i < ns; ++i) pw += (double)sx[i].re * sx[i].re;
        pw /= ns;
        const double sg = sqrt(pw / (2.0 * pow(10.0, -6.0 / 10.0)));
        for (size_t i = 0; i < ns; ++i) {       /* Box-Muller noise */
            const double u1 = URAND + 1e-12, u2 = URAND;
            const double rr = sqrt(-2.0 * log(u1));
            sx[i].re += (float)(sg * rr * cos(2.0 * ATK_SMOKE_PI * u2));
            sx[i].im += (float)(sg * rr * sin(2.0 * ATK_SMOKE_PI * u2));
        }
        /* detector finds the period */
        atkdsp_dsss_det_config dc; atkdsp_dsss_det_defaults(&dc, 1.0, 16 * N, 3 * N);
        atkdsp_dsss_det *det = atkdsp_dsss_det_create(&dc);
        atkdsp_dsss_det_result dr;
        atkdsp_dsss_det_process(det, sx, ns);
        atkdsp_dsss_det_read(det, &dr);
        CHECK(det != NULL && (dr.flags & ATKDSP_DSSS_DETECT) && fabs(dr.period - N) <= 1.0,
              "dsss_det: the 31-chip code's period found at -6 dB");
        atkdsp_dsss_det_destroy(det);
        /* code recovered and named */
        atkdsp_dsss_code_config cc; atkdsp_dsss_code_defaults(&cc, 1.0, 1.0 / sps, (double)N);
        int8_t chips[31]; atkdsp_dsss_code_result cres;
        ptrdiff_t gl = atkdsp_dsss_code(sx, ns, &cc, chips, 31, &cres);
        CHECK(gl == L && (cres.flags & ATKDSP_CODE_NAMED) && strstr(cres.name, "m-sequence (31)") != NULL,
              "dsss_code: the 31-chip m-sequence recovered and named");
        free(sx);
        /* the method table resolves, with a country tag */
        const atkdsp_method_info *mi = atkdsp_method(ATKDSP_METHOD_BUREL_FLUCT);
        CHECK(mi != NULL && mi->country != NULL && strcmp(mi->country, "FR") == 0,
              "method table: Burel's detector tagged FR");
        /* known codes: GPS PRN 1 balance and Barker-11 */
        int8_t ca[1023]; int ones = 0;
        CHECK(atkdsp_gps_ca(1, ca, 1023) == 1023, "gps_ca: PRN 1 generated");
        for (int i = 0; i < 1023; ++i) ones += ca[i] < 0;
        CHECK(ones == 512, "gps_ca: PRN 1 is balanced (512 ones)");
        #undef URAND
    }

    /* §28 the channelizer: a tone at a channel centre lands in that channel */
    {
        const unsigned nch = 64, K = 8;
        const double fs = 2.4e6, sp = fs / (double)nch;
        atkdsp_pfb *pf = atkdsp_pfb_create(fs, nch, 2, K, 80.0, 1);
        const size_t nn = (size_t)nch * 60;
        atkdsp_cf32 *px = (atkdsp_cf32 *)malloc(nn * sizeof *px);
        for (size_t i = 0; i < nn; ++i) {
            const double ph = 2.0 * ATK_SMOKE_PI * 10.0 * sp / fs * (double)i;   /* channel 10 */
            px[i].re = (float)cos(ph); px[i].im = (float)sin(ph);
        }
        const size_t hop = nch / 2, m = nn / hop;
        atkdsp_cf32 *po = (atkdsp_cf32 *)malloc((size_t)nch * m * sizeof *po);
        ptrdiff_t got = pf ? atkdsp_pfb_process(pf, px, nn, po, (size_t)nch * m) : -1;
        int best = -1; double bestp = -1.0;
        for (unsigned k = 0; got > 0 && k < nch; ++k) {
            double s = 0.0;
            for (ptrdiff_t i = got / 4; i < got; ++i) {
                const atkdsp_cf32 v = po[(size_t)k * got + i];
                s += (double)v.re * v.re + (double)v.im * v.im;
            }
            if (s > bestp) { bestp = s; best = (int)k; }
        }
        CHECK(pf != NULL && got > 0 && best == 10,
              "pfb: a tone at a channel centre lands in that channel");
        atkdsp_pfb_destroy(pf); free(px); free(po);
    }

    free(x); free(y); free(z); free(win); free(line); free(fm); free(scratch);
    printf("%d failure(s)\n", fails);
    return fails ? 1 : 0;
}
