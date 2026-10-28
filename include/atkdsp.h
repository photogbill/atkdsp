/*
 * atkdsp — the Analyst Toolkit's native DSP kernels.
 *
 * THIS HEADER IS THE ABI. Everything ATK's Python side relies on is here and
 * nowhere else. Rules, in order of importance:
 *
 *   1. Plain C99 with a C calling convention. No C++, no exceptions, no
 *      global mutable state, no callbacks into the caller. Loadable by
 *      ctypes, cffi, or any language that can call C.
 *   2. Nothing in a *_process / *_exec / *_mix / *_demod call allocates.
 *      Every buffer is caller-provided or was allocated once by the matching
 *      *_create. That is what makes a call's cost flat, which is the whole
 *      reason this library exists (see ATK FUTURE_PLANS §RF-F, the GIL
 *      convoy of 2026-09-15).
 *   3. Streaming state is EXPLICIT — a struct or an opaque handle the caller
 *      owns — and every kernel that has state is continuous across calls:
 *      the NCO carries its phase, the FIR its history, the resampler its
 *      fractional position, the FM discriminator its previous sample. A
 *      block boundary must be invisible in the output.
 *   4. Numbers in, numbers out. No I/O, no threads the caller cannot see
 *      (OpenMP inside a single call is the one exception, and
 *      atkdsp_set_threads controls it), no printing.
 *   5. ABI changes bump ATKDSP_ABI_VERSION. Changing or removing a function
 *      obviously does — and so does ADDING one, which this rule used to
 *      exempt. The exemption was wrong in practice: the Python binding binds
 *      every symbol eagerly at load, so a binding that knows a new function
 *      and a library that does not fails with an AttributeError about a
 *      missing symbol rather than the sentence the version check exists to
 *      print. A number that goes up is cheap; a mismatch that reports itself
 *      as something else is not. (Bumped 1 -> 2 on 2026-09-18 for
 *      atkdsp_unpack_dc.) 2 -> 3 for the LTE cell search. 3 -> 4 PBCH/MIB,
 *      4 -> 5 the PRACH detector, 5 -> 6 the turbo decoder + CRC-24A that the
 *      SIB1 transport block rides on, 6 -> 7 the SIB1 identity (ASN.1 UPER),
 *      7 -> 8 the SIB1 physical-layer receive chain (subframe -> identity),
 *      8 -> 9 SIB1 scheduling + the SIB2-5 network fingerprint.
 *
 * Sample convention: complex float32 as {re, im} pairs (same memory layout as
 * numpy complex64). All sizes are in SAMPLES unless the name says bytes.
 * Return codes: 0 = ok, negative = ATKDSP_E_* below. Size-returning
 * functions return the count produced.
 */

#ifndef ATKDSP_H
#define ATKDSP_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef ATKDSP_BUILD
#    define ATKDSP_API __declspec(dllexport)
#  else
#    define ATKDSP_API __declspec(dllimport)
#  endif
#else
#  define ATKDSP_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ATKDSP_ABI_VERSION 17
#define ATKDSP_VERSION_STRING "0.16.0"
/* 9 -> 10: atkdsp_spectrum_stats (per-bin max/avg/min + spectral kurtosis in
 * one pass) and atkdsp_window_stats (coherent gain + ENBW, so a level can be
 * read in true dBFS and a bin width in Hz). The FFT plan also became genuinely
 * safe to share across caller threads in this version (a per-slot claim
 * replaced omp_get_thread_num, which was 0 for every thread outside a parallel
 * region) — a fix, not an ABI change, but it rode in on this bump. */
/* 10 -> 11: atkdsp_arb_resampler — an ARBITRARY-ratio resampler (§4b). The
 * rational one (§4) needs the ratio to be a ratio of integers: it handles a
 * nominal 2.457600 MHz -> 48 kHz fine (255/256). What it CANNOT do is a ratio
 * that is a real number — a clock measured to a fractional hertz (2457603.1 Hz
 * after calibration), a non-integer output rate, or a live sample-clock
 * correction of a few ppm. A polyphase bank with linear interpolation between
 * phases (a first-order Farrow) resamples by any positive ratio and lets that
 * ratio be nudged per block. Output position is tracked from a global index, so
 * the same sample lands identically however the stream is chunked. The DDC
 * (§10) routes to it for exactly those non-integer rates it used to refuse (an
 * integer ratio still takes the exact rational path), so a channel off an
 * uncalibrated or fractional clock is native instead of returning NULL. */
/* 11 -> 12: atkdsp_symdump (symbol integrate-and-dump) and atkdsp_sync_search
 * (sync-word correlation) — the FSK/AFSK front-end spine shared by the pager
 * and data decoders (§12, src/fsk.c). */
/* 12 -> 13: atkdsp_dc_track (adaptive DC/baseline removal for an off-centre
 * channel) and atkdsp_agc (adaptive amplitude normalisation) — the flexible
 * front end for sub-optimal RF (§12, src/fsk.c). */
/* 13 -> 14: atkdsp_dmr_sync — the DMR / P25 sync correlator (§13,
 * src/dmr_sync.c): pilot-aided steering vectors and burst timing for the
 * KrakenSDR DF engine (ATK KRAKEN_DF_PLAN §6.5, §7.1). */
/* 14 -> 15: the bitstream engine (§14-17; docs/ATKDSP_PLAN.md §D phase 1a) —
 * bit primitives, sync discovery and bit statistics; Berlekamp-Massey and the
 * scramblers; Viterbi, Reed-Solomon, BCH and small block codes; the CRC engine,
 * its catalogue and CRC recovery. ATKDSP_E_DECODE (-5) is new. */
/* 15 -> 16: naming the code (§18; docs/ATKDSP_PLAN.md §D4, phase 1b) — the
 * rank-deficiency heat map, block / convolutional / Reed-Solomon / block
 * interleaver recognition, each with a robust (noisy-stream) search. */
/* 16 -> 17: the detection track (§19-21; docs/ATKDSP_PLAN.md §B1, §B2, §A3,
 * phase 2) — CFAR (CA / OS / GO / SO, threshold solved for K-look gamma
 * cells) with the burst tracker; the burst descriptor word; the ring buffer
 * in caller memory (one producer, up to 16 consumers, a seqlock per block,
 * attachable from another process). ATKDSP_E_OVERRUN (-6) and
 * ATKDSP_E_EMPTY (-7) are new. */

/* ---- errors ------------------------------------------------------------ */
#define ATKDSP_OK            0
#define ATKDSP_E_ARG        -1   /* null pointer, bad size, unknown enum      */
#define ATKDSP_E_NOMEM      -2   /* a *_create could not allocate            */
#define ATKDSP_E_CAP        -3   /* caller's output buffer too small         */
#define ATKDSP_E_FFT        -4   /* the FFT backend reported failure         */
#define ATKDSP_E_DECODE     -5   /* a decoder found the word uncorrectable   */
#define ATKDSP_E_OVERRUN    -6   /* a ring reader was lapped: the data is gone */
#define ATKDSP_E_EMPTY      -7   /* nothing there to return or to measure     */

/* ---- types ------------------------------------------------------------- */
typedef struct { float re, im; } atkdsp_cf32;

/* Raw sample formats, SigMF-style. Values are part of the ABI. */
enum atkdsp_format {
    ATKDSP_FMT_CU8     = 0,   /* RTL-SDR: unsigned 8-bit, centred on 127.5 */
    ATKDSP_FMT_CI8     = 1,   /* HackRF: signed 8-bit, /128                 */
    ATKDSP_FMT_CI16    = 2,   /* SpyServer et al: signed 16-bit LE, /32768  */
    ATKDSP_FMT_CI16Q11 = 3,   /* bladeRF SC16 Q11: signed 16-bit LE, /2048  */
    ATKDSP_FMT_CF32    = 4    /* float32 pairs, already +-1                 */
};

enum atkdsp_detector {
    ATKDSP_DET_MAX = 0,       /* peak hold across frames: 100 % POI          */
    ATKDSP_DET_MIN = 1,       /* what is ALWAYS there                        */
    ATKDSP_DET_AVG = 2        /* mean power (in linear power, then dB)      */
};

enum atkdsp_window {
    ATKDSP_WIN_RECT           = 0,
    ATKDSP_WIN_HANN           = 1,
    ATKDSP_WIN_HAMMING        = 2,
    ATKDSP_WIN_BLACKMANHARRIS = 3
};

/* ---- identity ---------------------------------------------------------- */
ATKDSP_API int         atkdsp_abi_version(void);
ATKDSP_API const char *atkdsp_version_string(void);   /* "0.1.0"            */
ATKDSP_API const char *atkdsp_build_info(void);       /* compiler, date, SIMD, openmp */
ATKDSP_API int         atkdsp_bytes_per_sample(int fmt);   /* 0 = unknown  */
/* OpenMP worker count for batched kernels; <=0 restores the default. Returns
 * the count in force. Always 1 in a build without OpenMP. */
ATKDSP_API int         atkdsp_set_threads(int n);
ATKDSP_API int         atkdsp_get_threads(void);

/* ---- 1. unpack: raw bytes -> cf32, with an optional running DC block ----
 * Converts as many whole samples as fit in min(nbytes, out_cap). A torn
 * trailing byte is ignored. dc_alpha in (0,1] applies a one-pole DC block
 * (state carried in *dc_state, which the caller zero-initialises once);
 * dc_alpha <= 0 disables it. Returns samples written, or ATKDSP_E_ARG. */
ATKDSP_API ptrdiff_t atkdsp_unpack(const void *raw, size_t nbytes, int fmt,
                                   atkdsp_cf32 *out, size_t out_cap,
                                   float dc_alpha, atkdsp_cf32 *dc_state);

/* ---- 1b. unpack, minus a constant, with the mean handed back -----------
 * Subtracts (off_re, off_im) from every sample and, if mean_re/mean_im are
 * given, returns the mean of the samples BEFORE that subtraction.
 *
 * Both are free: each format already subtracts a constant and multiplies by
 * a scale, so (x-c)*k - off is (x-(c+off/k))*k, and the mean accumulates in
 * the loop that is already reading. It exists because a caller that
 * estimates the offset once per block (ATK's RfChain) would otherwise make
 * two more full passes over the samples — at 40 MSPS, 48 MB per 50 ms block
 * on a path that measures as bandwidth-bound, for a notch and an average.
 *
 * This is the DC block to reach for when the caller has blocks.
 * atkdsp_unpack's per-sample pole is the one to reach for when it does not:
 * it is exact at every sample and costs five times the conversion. */
ATKDSP_API ptrdiff_t atkdsp_unpack_dc(const void *raw, size_t nbytes, int fmt,
                                      atkdsp_cf32 *out, size_t out_cap,
                                      float off_re, float off_im,
                                      double *mean_re, double *mean_im);

/* ---- 1c. front-end health, in one pass over a converted block ----------
 * The three things a wideband front end gets wrong that then look like real
 * signals on the display: a DC/LO-leakage spike at the centre, an I/Q
 * imbalance that puts a mirror image of every signal on the far side of
 * centre, and overload that folds intermodulation across the band. This
 * measures all of them from one block of cf32, cheaply (one pass, no FFT):
 *
 *   dc_re/dc_im         mean of I and Q — the centre spike, in the same units
 *                       as the samples (+-1 full scale)
 *   rms                 sqrt(mean |z|^2), the block's level
 *   gain_imbalance_db   10*log10(var I / var Q); 0 is balanced
 *   phase_error_deg     quadrature skew: asin(cov(I,Q)/sqrt(varI varQ))
 *   image_rejection_db  derived from the two above — how far below a signal
 *                       its mirror image sits. Large is good; below ~25 dB an
 *                       image is a "signal" that is not there. inf when perfect.
 *   clip_fraction       fraction of samples with |I| or |Q| >= clip_level
 *                       (<=0 uses 0.99): a proxy for a railed front end, on the
 *                       converted samples (a cu8 255 lands at ~+1.0)
 *
 * Returns ATKDSP_OK, or ATKDSP_E_ARG. The numpy twin is atkdsp.reference
 * .iq_health. This is an analysis call, not the per-sample hot path — run it
 * about once a second, not on every buffer. */
typedef struct {
    double dc_re, dc_im;
    double rms;
    double gain_imbalance_db;
    double phase_error_deg;
    double image_rejection_db;
    double clip_fraction;
    size_t n;
} atkdsp_iq_report;
ATKDSP_API int atkdsp_iq_health(const atkdsp_cf32 *in, size_t n, float clip_level,
                                atkdsp_iq_report *out);

/* ---- 2. NCO: phase-continuous complex mixer ----------------------------
 * out[n] = in[n] * exp(j * (phase + n*step)); phase advances by n*step and
 * is kept wrapped. step = -2*pi*f/fs shifts a channel at +f down to DC.
 * A rotating phasor is used with periodic renormalisation, so the cost is one
 * complex multiply per sample; the phase itself is tracked in double so a
 * ten-hour run does not drift. in and out may alias. */
typedef struct { double phase; double step; } atkdsp_nco;
ATKDSP_API void atkdsp_nco_init(atkdsp_nco *nco, double freq_hz, double sample_rate);
ATKDSP_API void atkdsp_nco_set_freq(atkdsp_nco *nco, double freq_hz, double sample_rate);
ATKDSP_API void atkdsp_nco_mix(atkdsp_nco *nco, const atkdsp_cf32 *in,
                               atkdsp_cf32 *out, size_t n);

/* ---- 3. decimating FIR with carried history ----------------------------
 * y[k] = sum_j h[j] * x[k*decim - j], computed ONLY for the outputs that
 * survive decimation (taps/decim multiplies per input sample). Taps are
 * copied at create. The filter is causal: output k corresponds to input
 * k*decim, with the first (ntaps-1) inputs of a fresh filter seen as zeros.
 * Output count for n inputs is floor((n + phase)/decim) with the phase
 * carried, so the total over many calls is exact. */
typedef struct atkdsp_fir atkdsp_fir;
ATKDSP_API atkdsp_fir *atkdsp_fir_create(const float *taps, size_t ntaps, unsigned decim);
ATKDSP_API void        atkdsp_fir_destroy(atkdsp_fir *f);
ATKDSP_API void        atkdsp_fir_reset(atkdsp_fir *f);
ATKDSP_API size_t      atkdsp_fir_out_max(const atkdsp_fir *f, size_t n_in);
ATKDSP_API ptrdiff_t   atkdsp_fir_process(atkdsp_fir *f, const atkdsp_cf32 *in, size_t n,
                                          atkdsp_cf32 *out, size_t out_cap);

/* ---- 4. rational resampler (polyphase, L up / M down) -------------------
 * Taps are the prototype low-pass designed at rate fs*up (cutoff <= min(fs,
 * fs*up/down)/2), and are applied with gain `up` folded in. The fractional
 * position is carried across calls, so N seconds in produce exactly
 * N*fs*up/down samples out (+-1 at the very end of a stream). */
typedef struct atkdsp_resampler atkdsp_resampler;
ATKDSP_API atkdsp_resampler *atkdsp_resampler_create(unsigned up, unsigned down,
                                                     const float *taps, size_t ntaps);
ATKDSP_API void      atkdsp_resampler_destroy(atkdsp_resampler *r);
ATKDSP_API void      atkdsp_resampler_reset(atkdsp_resampler *r);
ATKDSP_API size_t    atkdsp_resampler_out_max(const atkdsp_resampler *r, size_t n_in);
ATKDSP_API ptrdiff_t atkdsp_resampler_process(atkdsp_resampler *r, const atkdsp_cf32 *in,
                                              size_t n, atkdsp_cf32 *out, size_t out_cap);

/* ---- 4b. arbitrary (fractional) resampler --------------------------------
 * Resamples by ANY positive ratio out_rate/in_rate, not just a reduced L/M.
 *
 * A prototype low-pass is designed once at in_rate*nphase and split into
 * `nphase` polyphase branches (interpolate-by-nphase, gain nphase folded in);
 * for an output at a continuous input position the branch nearest its
 * fractional phase is applied and LINEARLY INTERPOLATED toward the next branch
 * (a first-order Farrow), so a finite bank still resolves a position between two
 * phases. `create` designs the prototype (an allocation, at create only — a
 * *_process call still allocates nothing, rule 2); pass atten_db<=0 for the
 * 60 dB default and nphase==0 for 64. The cutoff is 0.5*min(in_rate,out_rate),
 * so it anti-aliases on the way down and does not widen the band on the way up.
 *
 * The fractional position and the last taps-per-phase inputs are carried across
 * calls, so a block boundary is invisible and N seconds in produce ~N*ratio
 * samples out. set_ratio nudges the ratio live WITHOUT rebuilding the prototype
 * (for a few-ppm clock correction); a large change is better done by rebuilding,
 * because the prototype's cutoff was fixed at create. */
typedef struct atkdsp_arb_resampler atkdsp_arb_resampler;
ATKDSP_API atkdsp_arb_resampler *atkdsp_arb_resampler_create(double in_rate, double out_rate,
                                                             double atten_db, unsigned nphase);
ATKDSP_API void      atkdsp_arb_resampler_destroy(atkdsp_arb_resampler *r);
ATKDSP_API void      atkdsp_arb_resampler_reset(atkdsp_arb_resampler *r);
ATKDSP_API void      atkdsp_arb_resampler_set_ratio(atkdsp_arb_resampler *r, double ratio);
ATKDSP_API double    atkdsp_arb_resampler_ratio(const atkdsp_arb_resampler *r);
ATKDSP_API size_t    atkdsp_arb_resampler_out_max(const atkdsp_arb_resampler *r, size_t n_in);
ATKDSP_API ptrdiff_t atkdsp_arb_resampler_process(atkdsp_arb_resampler *r, const atkdsp_cf32 *in,
                                                  size_t n, atkdsp_cf32 *out, size_t out_cap);

/* ---- 5. FFT ------------------------------------------------------------
 * A plan for one length (any length; power-of-two and 5-smooth are fastest).
 * Forward is unscaled; inverse is scaled by 1/n (numpy convention).
 * A plan may be shared by several threads for *_exec and is used that way by
 * atkdsp_spectrum_reduce. */
typedef struct atkdsp_fft atkdsp_fft;
ATKDSP_API atkdsp_fft *atkdsp_fft_create(size_t n);
ATKDSP_API void        atkdsp_fft_destroy(atkdsp_fft *p);
ATKDSP_API size_t      atkdsp_fft_length(const atkdsp_fft *p);
ATKDSP_API int         atkdsp_fft_exec(const atkdsp_fft *p, const atkdsp_cf32 *in,
                                       atkdsp_cf32 *out, int inverse);

/* Window of `kind`, length n, into out (caller-provided). */
ATKDSP_API int atkdsp_window(int kind, size_t n, float *out);

/* One power spectrum: window (NULL = rectangular), FFT, |X|^2/n^2 in dB,
 * fftshifted so DC is the centre bin. out has n floats. Matches
 * atk.core.dsp.spectrum_db to float precision. */
ATKDSP_API int atkdsp_power_db(const atkdsp_fft *p, const atkdsp_cf32 *in,
                               const float *window, float *out_db);

/* 100 % probability of intercept: EVERY frame of n samples (stride `hop`,
 * hop <= n, hop == n for no overlap) inside `in[0..n_samples)` is transformed
 * and reduced per bin with the detector into ONE line of n floats (dB,
 * fftshifted). Frames run in parallel under OpenMP. Returns the number of
 * frames used (0 when n_samples < n) or an error. A short tail (< n) is not
 * used — the caller keeps it for the next block. */
ATKDSP_API ptrdiff_t atkdsp_spectrum_reduce(const atkdsp_fft *p, const atkdsp_cf32 *in,
                                            size_t n_samples, size_t hop,
                                            const float *window, int detector,
                                            float *out_line);

/* Every per-bin statistic a display's frames can give, in ONE pass over the
 * same frames spectrum_reduce would take — because the FFTs are the whole cost
 * and computing four things from each transform is nearly free once you have
 * it. Any of the four outputs may be NULL to skip it; each is n floats,
 * fftshifted like the reduce line.
 *
 *   max_db   peak hold  (10log10 of the largest |X|^2/n^2 per bin)
 *   avg_db   mean power (10log10 of the mean)   — the calibrated level
 *   min_db   the floor that is ALWAYS there     — what min-hold shows
 *   sk       SPECTRAL KURTOSIS, dimensionless: M*(sum P^2)/(sum P)^2 over the
 *            M frames, P=|X|^2. ~2.0 for stationary Gaussian noise, ~1.0 for a
 *            steady carrier, and >2 for an ON/OFF (bursty) emitter — so one
 *            block of the waterfall says, per bin, whether it is noise, a
 *            constant signal, or something keying, without a second capture.
 *            Needs M>=2; with one frame it is written as 1.0 (a constant).
 *
 * Returns the frame count (0 when n_samples < n) or an error. Frames run in
 * parallel under OpenMP, on the same claimed-slot scratch as the reduce. */
ATKDSP_API ptrdiff_t atkdsp_spectrum_stats(const atkdsp_fft *p, const atkdsp_cf32 *in,
                                           size_t n_samples, size_t hop,
                                           const float *window,
                                           float *max_db, float *avg_db,
                                           float *min_db, float *sk);

/* Coherent gain and equivalent noise bandwidth of a window, so a reading can
 * be turned into a true level. coherent_gain = (sum w)/n is what a full-scale
 * tone loses to the window (multiply a tone's linear magnitude by 1/cg to
 * recover dBFS); enbw_bins = n*(sum w^2)/(sum w)^2 is the window's noise
 * bandwidth in BINS (multiply by sample_rate/n for Hz), the width to divide a
 * noise-power reading by for a power spectral density. Either output may be
 * NULL. Matches atkdsp.reference.window_stats. */
ATKDSP_API int atkdsp_window_stats(const float *window, size_t n,
                                   double *coherent_gain, double *enbw_bins);

/* ---- 6. demodulators ---------------------------------------------------
 * FM: out[n] = gain * arg(in[n] * conj(prev)); *prev is carried so the first
 * sample of a block is demodulated against the last of the previous one.
 * Zero-initialise *prev once (a zero prev yields 0 for the first sample). */
ATKDSP_API void atkdsp_fm_demod(const atkdsp_cf32 *in, size_t n, float *out,
                                atkdsp_cf32 *prev, float gain);
/* AM: out[n] = |in[n]| - running mean (alpha as for unpack; <=0 = raw envelope). */
ATKDSP_API void atkdsp_am_demod(const atkdsp_cf32 *in, size_t n, float *out,
                                float *dc_state, float alpha);

/* ---- 7. display helpers ------------------------------------------------
 * dB line -> 32-bit pixels through a 256-entry LUT (caller's palette, any
 * byte order — the LUT is copied through untouched). lo..hi map to 0..255,
 * clamped; NaN maps to entry 0. */
ATKDSP_API void atkdsp_db_to_pixels(const float *db, size_t n, float lo, float hi,
                                    const uint32_t *lut256, uint32_t *out);
/* Peak-preserving reduction of n bins to m columns (m <= n): the max of each
 * bin group, so a narrow burst survives being drawn narrower than a pixel. */
ATKDSP_API void atkdsp_decimate_max(const float *in, size_t n, float *out, size_t m);

/* ---- 8. detection on a spectrum line -----------------------------------
 * Median of n floats (a robust noise floor). scratch holds n floats. */
ATKDSP_API float atkdsp_median(const float *x, size_t n, float *scratch);

typedef struct {
    int   start_bin, end_bin;   /* [start, end) */
    float centroid_bin;         /* power-weighted centre */
    float peak_db;
    float snr_db;               /* peak - floor */
} atkdsp_channel;

/* Contiguous runs of bins above floor+threshold (floor_db from atkdsp_median
 * of the smoothed line, as signal_id does; NaN is refused) in a 5-bin-smoothed copy of
 * the line, with gaps shorter than `gap_bins` bridged and runs narrower than
 * `min_run` dropped — the same policy as atk.core.signal_id.detect_channels,
 * without its per-bin Python loop. smooth_scratch holds n floats. Returns
 * channels written (strongest first), or ATKDSP_E_CAP when out_cap is too
 * small (out then holds the first out_cap found, unsorted). */
ATKDSP_API ptrdiff_t atkdsp_detect_channels(const float *line, size_t n, float floor_db,
                                            float threshold_db, int gap_bins, int min_run,
                                            float *smooth_scratch,
                                            atkdsp_channel *out, size_t out_cap);

/* Sweep stitch: max-accumulate one captured segment into a stitched trace.
 * Segment bin i sits at seg_lo_hz + (i + 0.5) * seg_hz_per_bin; only bins
 * inside [keep_lo_hz, keep_hi_hz] (the usable part of the capture) AND inside
 * the output span are used. Output bin k covers out_lo_hz + k*out_hz_per_bin.
 * NaN in out means "not yet measured" and is replaced by the first value. */
ATKDSP_API void atkdsp_stitch_max(const float *seg, size_t nseg,
                                  double seg_lo_hz, double seg_hz_per_bin,
                                  double keep_lo_hz, double keep_hi_hz,
                                  double out_lo_hz, double out_hz_per_bin,
                                  float *out, size_t nout);

/* ---- 9. filter design ---------------------------------------------------
 * Kaiser-windowed sinc low-pass: passband edge fp, stopband edge fs_, at
 * least `atten_db` of stopband attenuation, at sample rate `rate`. Writes up
 * to out_cap taps (unity DC gain, odd length) and returns the length needed;
 * call with out=NULL to size the buffer. Deterministic: the numpy twin
 * (atkdsp.reference.design_lowpass) produces the same taps to float precision. */
ATKDSP_API ptrdiff_t atkdsp_design_lowpass(double fp_hz, double fs_hz, double atten_db,
                                           double rate, float *out, size_t out_cap);

/* ---- 10. DDC: one call from wideband I/Q to an exact-rate channel --------
 * NCO (phase-continuous) -> a chain of decimating FIR stages, each designed
 * so that nothing aliases into the channel by less than `atten_db` -> a
 * final rational resampler to EXACTLY `out_rate` (so 10 MSPS gives 48 000
 * Hz, not 48 077). The filters are designed at create; `max_block` is the
 * largest input block process() will be given (scratch is sized then —
 * rule 2). set_offset retunes within the span with no discontinuity. */
typedef struct atkdsp_ddc atkdsp_ddc;
ATKDSP_API atkdsp_ddc *atkdsp_ddc_create(double sample_rate, double offset_hz,
                                         double channel_bw_hz, double out_rate,
                                         double atten_db, size_t max_block);
ATKDSP_API void      atkdsp_ddc_destroy(atkdsp_ddc *d);
ATKDSP_API void      atkdsp_ddc_reset(atkdsp_ddc *d);
ATKDSP_API void      atkdsp_ddc_set_offset(atkdsp_ddc *d, double offset_hz);
ATKDSP_API double    atkdsp_ddc_out_rate(const atkdsp_ddc *d);
ATKDSP_API size_t    atkdsp_ddc_out_max(const atkdsp_ddc *d, size_t n_in);
ATKDSP_API ptrdiff_t atkdsp_ddc_process(atkdsp_ddc *d, const atkdsp_cf32 *in, size_t n,
                                        atkdsp_cf32 *out, size_t out_cap);
/* The plan, for the operator: "10 MSPS: NCO -> /8 (23 taps) -> /2 (31) ->
 * /13 (211) -> 48076.9 -> x624/625 -> 48000". Returns chars written. */
ATKDSP_API int atkdsp_ddc_describe(const atkdsp_ddc *d, char *buf, size_t cap);
/* Stage factors and tap counts, for the twin and the tests: returns the
 * number of decimation stages, filling factors[i] / ntaps[i] up to cap;
 * *up / *down are the final resampler ratio (1/1 when none is needed). */
ATKDSP_API int atkdsp_ddc_plan(const atkdsp_ddc *d, unsigned *factors, unsigned *ntaps,
                               size_t cap, unsigned *up, unsigned *down);


/* ---- 11. LTE cell search: PSS, SSS, PCI --------------------------------
 * Finds LTE downlink cells in a stream that has been brought to
 * ATKDSP_LTE_RATE — 1.92 MSPS, the 128-point/15 kHz numerology. The caller
 * does that with a DDC; every LTE cell puts its synchronisation signals in
 * the middle 1.08 MHz whatever its channel bandwidth, so a 1.4 MHz slice of
 * a 20 MHz carrier is enough to find and identify it. That is what makes a
 * survey cheap.
 *
 * What it does NOT do, on purpose: decode anything. PBCH -> MIB, blind
 * PDCCH search and the SIBs are turbo and polar decoding, rate matching and
 * channel estimation — a different kind of project with mature open
 * implementations. This answers "is there a cell here, which one, how
 * strong, how far off frequency", which is the question a survey and a
 * fake-tower check actually ask.
 *
 * `metric` is a normalised correlation, 0..1, so it does not move with gain.
 * A cell is only reported when the 5 ms PSS repeat is also present, which is
 * what keeps noise and one-off impulses out (measured: nothing on eight
 * seeds of pure noise, correct PCI down to -3 dB SNR). */
#define ATKDSP_LTE_RATE      1920000.0   /* the rate the input must be at */
#define ATKDSP_LTE_SYM       128         /* samples in one OFDM symbol     */
#define ATKDSP_LTE_PSS_PERIOD 9600       /* 5 ms: PSS is in slots 0 and 10 */

typedef struct {
    int       nid2;       /* 0..2, from the PSS                            */
    int       nid1;       /* 0..167, from the SSS; -1 if it was not read   */
    int       pci;        /* 3*nid1 + nid2; -1 if nid1 is unknown          */
    int       subframe;   /* 0 or 5 (the SSS says which); -1 if unknown    */
    long long offset;     /* index of the first PSS sample in the input    */
    float     metric;     /* normalised PSS correlation peak, 0..1         */
    float     sss_score;  /* mean SSS correlation, 0..1                    */
    float     cfo_hz;     /* carrier frequency offset of the cell          */
} atkdsp_lte_cell;

typedef struct atkdsp_lte atkdsp_lte;
ATKDSP_API atkdsp_lte *atkdsp_lte_create(void);
ATKDSP_API void        atkdsp_lte_destroy(atkdsp_lte *h);
/* The reference PSS symbol for one N_ID^(2), 128 samples — for tests and
 * for a caller that wants to draw it. */
ATKDSP_API int         atkdsp_lte_pss_symbol(int nid2, atkdsp_cf32 *out);
/* The 62 SSS values (+1/-1) for one cell and subframe. */
ATKDSP_API int         atkdsp_lte_sss_symbol(int nid1, int nid2, int subframe,
                                             float *out62);
/* Scan `in` (at ATKDSP_LTE_RATE) for cells. Needs at least two PSS periods
 * to confirm the 5 ms repeat. Returns how many cells were written. */
ATKDSP_API ptrdiff_t   atkdsp_lte_detect(atkdsp_lte *h, const atkdsp_cf32 *in,
                                         size_t n, float min_metric,
                                         atkdsp_lte_cell *out, size_t cap);

/* ---- 11b. LTE PBCH -> MIB ----------------------------------------------
 * The Master Information Block: downlink bandwidth, PHICH config, antenna
 * count and system frame number. Broadcast and public — the network's own
 * parameters, nothing about any subscriber. Decoded from slot-1 symbols
 * 0..3 of subframe 0 (CRS channel estimate, transmit-diversity combine,
 * tail-biting Viterbi, CRC-16 with the antenna mask). */
#define ATKDSP_LTE_FRAME       19200  /* samples in a 10 ms radio frame     */
#define ATKDSP_LTE_PBCH_OFFSET 128    /* PSS data-start -> PBCH block start  */
#define ATKDSP_LTE_PBCH_BLOCK  549    /* samples: 4 OFDM symbols with CP     */

typedef struct {
    int dl_bw_rb;    /* downlink bandwidth in resource blocks: 6..100      */
    int phich_dur;   /* 0 normal, 1 extended                               */
    int phich_res;   /* 0..3: oneSixth / half / one / two                  */
    int sfn;         /* full 10-bit system frame number                    */
    int n_ports;     /* 1, 2 or 4 antenna ports (from the CRC mask)        */
} atkdsp_lte_mib;

/* Gold sequence (36.211 7.2); length <= 1920. Exposed for tests. */
ATKDSP_API int atkdsp_lte_gold(unsigned c_init, int length, signed char *out);

/* Decode the MIB from one PBCH block. `block` points at the first sample
 * (including its cyclic prefix) of the 4-symbol block, ATKDSP_LTE_PBCH_OFFSET
 * samples after the subframe-0 PSS offset atkdsp_lte_detect returns; it needs
 * ATKDSP_LTE_PBCH_BLOCK samples. `cfo_hz` is the cell offset from detect (0
 * to skip). Returns 1 decoded, 0 none, <0 error. */
ATKDSP_API int atkdsp_lte_mib_decode(atkdsp_lte *h, const atkdsp_cf32 *block,
                                     int n_id, double cfo_hz,
                                     atkdsp_lte_mib *out);
/* Soft-combine up to `nframes` consecutive PBCH blocks (stride
 * ATKDSP_LTE_FRAME apart) for weak cells: every single frame first, then
 * every aligned 4-frame window. `base` points at the first block. */
ATKDSP_API int atkdsp_lte_mib_decode_combined(atkdsp_lte *h,
                                     const atkdsp_cf32 *base, int nframes,
                                     ptrdiff_t stride, int n_id,
                                     double cfo_hz, atkdsp_lte_mib *out);

/* ---- 11c. Passive PRACH handset-presence detector ----------------------
 * Finds the Zadoff-Chu access preamble a powered handset sends when it
 * reaches a cell. PRESENCE and location only, never identity: it reports
 * that an access happened, which root sequence, and how many concurrent
 * accesses, nothing about the subscriber. Root-agnostic: one FFT of the
 * differential product D[k]=Y[k+1]conj(Y[k]) shows a tone per concurrent
 * root (no 838-way correlator bank); a per-root power-delay profile counts
 * the accesses. Feed the sequence window (>= ATKDSP_PRACH_NZC samples at the
 * PRACH rate) after tuning to the uplink. */
#define ATKDSP_PRACH_NZC 839          /* Zadoff-Chu length, formats 0-3 */

typedef struct {
    int   root;    /* Zadoff-Chu root index of a preamble present         */
    int   count;   /* concurrent accesses on this root (PDP peaks)        */
    float metric;  /* differential-tone peak / median (detection strength)*/
    int   delay;   /* strongest PDP peak index (timing / cyclic shift)    */
} atkdsp_prach_hit;

typedef struct atkdsp_prach atkdsp_prach;
ATKDSP_API atkdsp_prach *atkdsp_prach_create(void);
ATKDSP_API void          atkdsp_prach_destroy(atkdsp_prach *h);
/* The frequency-domain ZC root sequence (839 values); for tests and PDPs. */
ATKDSP_API int           atkdsp_prach_zc(int u, atkdsp_cf32 *out);
/* Detect preambles in `seq` (>= ATKDSP_PRACH_NZC samples). `min_metric` is
 * the tone peak/median to clear (<=0 uses the default 40). Returns how many
 * distinct roots were written, or <0 on error. */
ATKDSP_API ptrdiff_t     atkdsp_prach_detect(atkdsp_prach *h,
                                     const atkdsp_cf32 *seq, size_t n,
                                     float min_metric, atkdsp_prach_hit *out,
                                     size_t cap);
/* Slide an NZC window across a longer uplink capture (step `win_step`, use
 * 64) and merge detections by root — one entry per root with its best metric
 * and largest concurrent-access count. This is the call a survey uses: it
 * does not need PRACH-occasion timing. */
ATKDSP_API ptrdiff_t     atkdsp_prach_scan(atkdsp_prach *h,
                                     const atkdsp_cf32 *stream, size_t n,
                                     int win_step, float min_metric,
                                     atkdsp_prach_hit *out, size_t cap);

/* ---- 11d. LTE turbo decoder + CRC-24A ----------------------------------
 * The rate-1/3 turbo code (36.212 5.1.3.2) that protects the PDSCH transport
 * block a SIB rides on: two 8-state RSC constituents joined by a QPP
 * interleaver, rate-matched (5.1.4.1) into the E soft bits transmitted, and
 * decoded by iterative max-log-MAP. This is DOWNLINK BROADCAST data — a SIB1
 * transport block is the cell's own public identity, nothing about any
 * subscriber. Not a per-sample hot path (a SIB every 80 ms, a few hundred
 * bits), so this call allocates its own scratch and frees it before returning.
 *
 * K is the turbo code-block size (an entry of 36.212 Table 5.1.3-3, 40..6144);
 * the transport block is K-24 bits plus its CRC-24A. rv is the redundancy
 * version (0 for a first SIB transmission). iters<=0 uses 8. */

/* CRC-24A (36.212 5.1.1) of `nbits` bits into out24 (MSB first). For tests and
 * for a caller that attaches its own CRC. */
ATKDSP_API int atkdsp_lte_crc24a(const signed char *bits, int nbits,
                                 signed char *out24);

/* Decode E descrambled LLRs (>0 favours bit 0) for code-block size K. On a
 * CRC-24A pass, writes the K-24 payload bits to `out` and returns 1; returns 0
 * when the CRC fails, or <0 (ATKDSP_E_ARG for an unknown K, E_NOMEM). */
ATKDSP_API int atkdsp_lte_turbo_decode(const float *llr, size_t E, int K,
                                       int rv, int iters, signed char *out);

/* ---- 11e. SIB1 -> tower identity (ASN.1 UPER) --------------------------
 * The identity-bearing part of SystemInformationBlockType1: the operator
 * (PLMN = MCC/MNC), the trackingAreaCode and the 28-bit E-UTRAN cell identity
 * (ECI). Decoded from the transport-block bits a turbo decode produced (the
 * BCCH-DL-SCH-Message content, one bit per byte, MSB first). Broadcast, public
 * — the cell's own name for itself; the decode STOPS after
 * cellAccessRelatedInfo, so nothing about a subscriber is touched. */
#define ATKDSP_LTE_MAX_PLMN 6

typedef struct {
    int mcc[3];      /* three MCC digits; mcc[0] = -1 if this PLMN had none    */
    int mnc[3];      /* MNC digits; only the first mnc_len are valid           */
    int mnc_len;     /* 2 or 3                                                 */
    int reserved;    /* cellReservedForOperatorUse (1 = reserved)              */
} atkdsp_lte_plmn;

/* One entry of the SIB1 schedulingInfoList: an SI message's periodicity and
 * the SIB types it carries (SIB2 is implicit in the first entry). */
#define ATKDSP_LTE_MAX_SI      8
#define ATKDSP_LTE_MAX_SIBMAP  8
typedef struct {
    int periodicity_rf;                 /* 8..512 radio frames                */
    int n_sibs;
    int sibs[ATKDSP_LTE_MAX_SIBMAP];    /* SIB-Type numbers (3..)             */
} atkdsp_lte_sched;

typedef struct {
    int             n_plmn;                    /* 1..6                         */
    atkdsp_lte_plmn plmn[ATKDSP_LTE_MAX_PLMN];
    int             tac;                       /* trackingAreaCode, 16-bit     */
    unsigned        cell_id;                   /* cellIdentity / ECI, 28-bit   */
    int             cell_barred;               /* cellBarred ENUMERATED index  */
    int             csg_id;                    /* -1 if absent                 */
    /* scheduling (best-effort; 0 / empty if the bits stop after the identity) */
    int             freq_band;                 /* freqBandIndicator, 0 if none */
    int             si_window_ms;              /* si-WindowLength, 0 if none    */
    int             n_sched;                   /* schedulingInfoList entries    */
    atkdsp_lte_sched sched[ATKDSP_LTE_MAX_SI];
} atkdsp_lte_sib1;

/* Parse the tower identity (and, best-effort, the scheduling) from `nbits`
 * decoded transport-block bits. Returns 1 on a clean identity parse, 0 on a
 * structural mismatch (not a SIB1, or the bits ran out), <0 on a bad argument. */
ATKDSP_API int atkdsp_lte_sib1_parse(const signed char *bits, int nbits,
                                     atkdsp_lte_sib1 *out);

/* ---- 11f. SIB1 physical layer: subframe -> tower identity --------------
 * DO NOT USE (2026-10-27, docs/ATKDSP_PLAN.md §0): this chain still carries
 * six known physical-layer faults, and ATK decodes SIB1 with its own
 * lte_sib1_rx.py instead. Kept for the ABI only; a comment, not a change.
 * The whole downlink receive chain for a SIB1: OFDM demod at the cell
 * numerology (n_rb in {6,15,25,50,75,100}), a port-0 CRS channel estimate and
 * single-tap equalisation (equalize != 0), PCFICH, a blind SI-RNTI PDCCH
 * search (DCI 1A -> RB allocation), then PDSCH descramble + turbo + ASN.1.
 * `samples` covers one subframe (14 OFDM symbols at nfft*15 kHz); `subframe`
 * is which one (5 for SIB1). cfo_hz corrects a carrier offset (0 to skip).
 * Returns 1 with the identity in *out, 0 when no SIB1 is present, <0 on error.
 *
 * The control-region RE geometry is self-consistent (proved by the round-trip
 * cross-check) but not yet confirmed on a real air capture -- see the note in
 * src/lte_sib1_phy.c. */
ATKDSP_API int atkdsp_lte_sib1_decode(const atkdsp_cf32 *samples, size_t n,
                                      int n_rb, int n_id, int n_ports,
                                      int subframe, double cfo_hz,
                                      int equalize, atkdsp_lte_sib1 *out);

/* ---- 11g. SystemInformation: the SIB2-5 network fingerprint ------------
 * atkdsp_lte_si_decode: DO NOT USE (2026-10-27) -- it rides on the SIB1
 * physical layer above and inherits its faults. atkdsp_lte_si_parse (the
 * ASN.1 on bits) is unaffected.
 * The SIB2+ messages (carried in BCCH-DL-SCH SystemInformation, scheduled by
 * SIB1's schedulingInfoList) that make a cell's fingerprint: SIB2 (access
 * barring, PRACH config, uplink carrier freq/bandwidth, reference-signal
 * power), SIB3 (reselection), SIB4 (intra-frequency neighbour PCIs), SIB5
 * (inter-frequency carriers + neighbour PCIs). All BROADCAST and public.
 *
 * GEOMETRY/LAYOUT CAVEAT: the ASN.1 field layouts are 36.331 R8 and are proven
 * self-consistent by the round-trip cross-check, NOT against a real air
 * capture. The neighbour lists and access barring are simple and
 * high-confidence; the deep radioResourceConfig fields (PRACH, UL freq) most
 * need a live-capture check. */
#define ATKDSP_LTE_MAX_NEIGH 16
#define ATKDSP_LTE_MAX_FREQ  8

typedef struct { int pci; int q_off; } atkdsp_lte_neigh;

typedef struct {
    int present;
    int barring;              /* ac-BarringInfo present (0/1)                 */
    int barring_emergency;    /* ac-BarringForEmergency                      */
    int prach_root;           /* rootSequenceIndex, 0..837                   */
    int prach_config_index;   /* prach-ConfigIndex, 0..63                    */
    int prach_high_speed;     /* highSpeedFlag                               */
    int prach_zcc;            /* zeroCorrelationZoneConfig, 0..15            */
    int prach_freq_offset;    /* prach-FreqOffset, 0..94                     */
    int ref_sig_power;        /* referenceSignalPower, dBm (-60..50)         */
    int ul_earfcn;            /* ul-CarrierFreq EARFCN, -1 if absent         */
    int ul_bandwidth_rb;      /* 6/15/25/50/75/100, 0 if absent             */
    int time_align_timer;     /* timeAlignmentTimerCommon enum index 0..7    */
} atkdsp_lte_sib2;

typedef struct {
    int present;
    int q_rxlevmin;           /* dBm                                          */
    int s_intra_search;       /* -1 if absent                                */
    int resel_priority;       /* cellReselectionPriority 0..7                */
} atkdsp_lte_sib3;

typedef struct {
    int present;
    int n_neigh;
    atkdsp_lte_neigh neigh[ATKDSP_LTE_MAX_NEIGH];   /* intra-freq neighbours  */
    int n_black;
    int black_start[ATKDSP_LTE_MAX_NEIGH];          /* blacklist PCI starts   */
} atkdsp_lte_sib4;

typedef struct {
    int dl_earfcn;                                   /* inter-freq carrier     */
    int n_neigh;
    int neigh_pci[ATKDSP_LTE_MAX_NEIGH];
} atkdsp_lte_interfreq;

typedef struct {
    int present;
    int n_freq;
    atkdsp_lte_interfreq freq[ATKDSP_LTE_MAX_FREQ];
} atkdsp_lte_sib5;

typedef struct {
    atkdsp_lte_sib2 sib2;
    atkdsp_lte_sib3 sib3;
    atkdsp_lte_sib4 sib4;
    atkdsp_lte_sib5 sib5;
} atkdsp_lte_si;

/* Parse a SystemInformation message's transport-block bits into the SIBs it
 * carries. Returns 1 on a clean parse, 0 on a structural mismatch, <0 on a bad
 * argument. Each atkdsp_lte_sibN.present says whether that SIB was seen. */
ATKDSP_API int atkdsp_lte_si_parse(const signed char *bits, int nbits,
                                   atkdsp_lte_si *out);

/* The whole receive chain for a SystemInformation message: same PHY as
 * atkdsp_lte_sib1_decode (OFDM demod, CRS equalise, PCFICH, blind SI-RNTI
 * PDCCH, PDSCH + turbo), then atkdsp_lte_si_parse. `subframe` is the SI-window
 * subframe the caller scheduled from SIB1. Returns 1 with the SIBs in *out, 0
 * when nothing decodes, <0 on error. */
ATKDSP_API int atkdsp_lte_si_decode(const atkdsp_cf32 *samples, size_t n,
                                    int n_rb, int n_id, int n_ports,
                                    int subframe, double cfo_hz,
                                    int equalize, atkdsp_lte_si *out);

/* ---- 12. FSK front end: symbol integrate-and-dump + sync-word search -----
 * The shared spine for FSK/AFSK decoders (POCSAG, FLEX, ...). See src/fsk.c.
 *
 * atkdsp_symdump: integrate-and-dump matched filter. Averages each symbol's
 * worth of samples of a real baseband stream (an FM discriminator output) into
 * one soft value. `sps` may be fractional (POCSAG-512 at 48 kHz = 93.75); the
 * symbol boundary is tracked with a real accumulator and `phase` (0..sps)
 * positions the first boundary — data-aided timing, the caller picks the phase
 * that maximises symbol amplitude. Open-loop and deterministic; all state is
 * carried, so a block boundary is invisible. out_max bounds a call's symbol
 * count; process returns the exact count. A tracked timing loop (M&M/Gardner)
 * for a drifting clock is a later increment. */
typedef struct atkdsp_symdump atkdsp_symdump;
ATKDSP_API atkdsp_symdump *atkdsp_symdump_create(double sps, double phase);
ATKDSP_API void      atkdsp_symdump_destroy(atkdsp_symdump *s);
ATKDSP_API void      atkdsp_symdump_reset(atkdsp_symdump *s);
ATKDSP_API size_t    atkdsp_symdump_out_max(const atkdsp_symdump *s, size_t n_in);
ATKDSP_API ptrdiff_t atkdsp_symdump_process(atkdsp_symdump *s, const float *in, size_t n,
                                            float *out, size_t out_cap);

/* atkdsp_sync_search: slide a bit pattern (0/1) over the SIGN of soft symbols
 * (soft < 0 => bit 1) and report matches within max_err bits, each with a
 * polarity (0 = pattern, 1 = complement, an inverted FSK stream) and the
 * position of the first symbol after the match. Returns hits, or ATKDSP_E_CAP
 * when more than cap exist. */
typedef struct {
    long long pos;        /* index of the first symbol AFTER the match      */
    int       polarity;   /* 0 = pattern, 1 = complement (inverted stream)  */
    int       errors;     /* bit mismatches in the winning polarity         */
} atkdsp_sync_hit;
ATKDSP_API ptrdiff_t atkdsp_sync_search(const float *sym, size_t n,
                                        const signed char *pattern, size_t plen,
                                        int max_err, atkdsp_sync_hit *out, size_t cap);

/* atkdsp_dc_track: adaptive DC/baseline removal for a real stream. A carrier
 * offset on an FM-discriminator output is a constant pedestal; a leaky
 * integrator tracks the slow mean and subtracts it (out = in - s;
 * s += alpha*(in-s)), so a zero-threshold slicer is right however far
 * off-centre the channel sits. `alpha` in (0,1] is the tracking rate (corner
 * ~ alpha*fs/2pi) — set it WELL BELOW the symbol rate so it follows
 * offset/drift, not the data. alpha<=0 copies. *state (one float) is the
 * carried estimate; zero it once. in and out may alias. */
ATKDSP_API void atkdsp_dc_track(const float *in, size_t n, float *out,
                                float *state, float alpha);
/* atkdsp_agc: adaptive amplitude normalisation. Tracks the mean magnitude and
 * scales toward `ref`: a += alpha*(|in|-a); out = in*ref/max(a,eps). Makes
 * soft symbols' scale invariant to each transmitter's deviation/level, and
 * gives a 4-level slicer a normalised input. alpha<=0 copies; ref<=0 uses 1.
 * *state (one float) is the carried amplitude; zero it once. in/out alias. */
ATKDSP_API void atkdsp_agc(const float *in, size_t n, float *out,
                           float *state, float alpha, float ref);

/* ---- 13. 4FSK sync correlator: DMR / P25 pilots and burst timing --------
 * A complex-baseband matched filter for the 24-symbol sync words of DMR (ETSI
 * TS 102 361-1 Table 9.2: BS and MS voice/data, MS reverse channel, the four
 * direct-mode slot words) and the P25 Phase 1 frame sync, with a bank of
 * carrier-offset hypotheses, across 1..64 phase-coherent channels at once.
 * The model and the measurements are in src/dmr_sync.c and the twin
 * (python/atkdsp/reference.py §13). Two jobs (ATK KRAKEN_DF_PLAN):
 *   - pilot-aided DF (§6.5): a hit carries the per-channel complex amplitude
 *     of THAT emitter's sync, phase at the window centre; amps[c] / amps[0]
 *     is the steering vector, with 24 coherent symbols of processing gain,
 *     and a co-channel emitter that is not sending a sync does not pull it;
 *   - decoder-clocked DoA (§7.1): the sync's sample-exact position and type
 *     (BS / MS / direct-mode slot — the repeater-vs-handset path tag).
 * Input: complex baseband centred on the channel at 9.6..384 kHz (channelise
 * first; 24 or 48 kHz is right), channel c at in[c*ch_stride + i]. State is
 * carried: a hit is bit-identical however the stream is chunked. A peak is
 * reported once `guard` samples have passed it with nothing better (guard 0 =
 * 66 symbols, the rest of a DMR burst either side, which also swallows the
 * partial matches its own payload makes); _flush reports a pending one at
 * the end of a stream or of a one-shot buffer.
 *   score   normalised correlation 0..1, summed over channels: gain-free,
 *           about SNR/(1+SNR) for a true sync in its own channel bandwidth.
 *   errors  sync symbols whose frequency SIGN reads wrong, DSD's test, summed
 *           over channels. <= 3 means the pattern label can be trusted
 *           (measured: one channel at +3 dB, 84 % of true syncs; five channels
 *           at 0 dB, 99 %); a burst with no sync (DMR voice B-F) makes partial
 *           matches that read 5 or more — the label is wrong, but it is still
 *           that emitter, so the amplitudes are still its steering vector.
 * `pattern_mask` selects patterns by set index (bit p; 0 bits = refused);
 * `alpha` is the RRC roll-off of the frequency pulses (NaN = the set's 0.2,
 * negative = rectangular); `cfo_step_hz` 0 = 100 Hz (half the 5 ms sync's
 * resolution: <= 0.9 dB between bins). Returns NULL on a bad argument. */
#define ATKDSP_SYNC_DMR      0     /* 9 patterns; +3 = +1944 Hz              */
#define ATKDSP_SYNC_P25      1     /* 1 pattern;  +3 = +1800 Hz              */
#define ATKDSP_SYNC_INVERT   1     /* flags: a spectrally inverted stream    */
#define ATKDSP_SYNC_SYMBOLS  24
#define ATKDSP_SYNC_RATE     4800.0

typedef struct {
    long long start;     /* stream index of the best window's first sample   */
    double    frac;      /* sub-sample refinement, -0.5..0.5; the sync's
                            leading edge is at start + frac - 0.5            */
    double    cfo_hz;    /* carrier offset, refined between bins             */
    float     score;     /* normalised correlation 0..1 at the refined offset */
    int       errors;    /* sync symbols with the wrong frequency sign, 0..24 */
    int       pattern;   /* index in the set (atkdsp_dmr_sync_pattern_name)  */
    int       reserved;  /* 0                                                 */
} atkdsp_dmr_sync_hit;   /* 40 bytes */

typedef struct atkdsp_dmr_sync atkdsp_dmr_sync;
/* The tables, for the binding, the twin's cross-check and a caller's labels. */
ATKDSP_API int         atkdsp_dmr_sync_patterns(int set);
ATKDSP_API int         atkdsp_dmr_sync_pattern(int set, int idx, signed char *sym,
                                               size_t cap);          /* +-3, +-1 */
ATKDSP_API const char *atkdsp_dmr_sync_pattern_name(int set, int idx);
ATKDSP_API atkdsp_dmr_sync *atkdsp_dmr_sync_create(int set, double sample_rate,
                                    int n_channels, unsigned pattern_mask,
                                    double cfo_span_hz, double cfo_step_hz,
                                    float threshold, int guard, double alpha,
                                    int flags);
ATKDSP_API void      atkdsp_dmr_sync_destroy(atkdsp_dmr_sync *s);
ATKDSP_API void      atkdsp_dmr_sync_reset(atkdsp_dmr_sync *s);
/* Window length L (samples), offset bins, bin spacing, guard. Any may be NULL. */
ATKDSP_API int       atkdsp_dmr_sync_info(const atkdsp_dmr_sync *s, int *window,
                                          int *n_cfo, double *cfo_step_hz, int *guard);
/* A process() call of n_in samples writes at most this many hits; cap below
 * it is refused before anything is consumed (ATKDSP_E_CAP). */
ATKDSP_API size_t    atkdsp_dmr_sync_hits_max(const atkdsp_dmr_sync *s, size_t n_in);
/* `amps` (NULL to skip) receives n_channels values per hit, hit-major. */
ATKDSP_API ptrdiff_t atkdsp_dmr_sync_process(atkdsp_dmr_sync *s, const atkdsp_cf32 *in,
                                    size_t n, size_t ch_stride,
                                    atkdsp_dmr_sync_hit *hits, atkdsp_cf32 *amps,
                                    size_t cap);
/* Report the pending peak now (0 or 1 hit). The stream may continue. */
ATKDSP_API ptrdiff_t atkdsp_dmr_sync_flush(atkdsp_dmr_sync *s, atkdsp_dmr_sync_hit *hits,
                                    atkdsp_cf32 *amps, size_t cap);
/* The reference waveform of one pattern (any in the set): L unit-modulus
 * samples, phase zero at the window centre. Returns L. */
ATKDSP_API int       atkdsp_dmr_sync_reference(const atkdsp_dmr_sync *s, int pattern,
                                    atkdsp_cf32 *out, size_t cap);
/* "DMR sync: 48000 Hz (10.00 sps), 240-sample window, 9 patterns, CFO +-1000
 * Hz in 21 bins of 100 Hz, threshold 0.30, guard 660, RRC a=0.20, 5 channels" */
ATKDSP_API int       atkdsp_dmr_sync_describe(const atkdsp_dmr_sync *s, char *buf,
                                    size_t cap);

/* ==== THE BITSTREAM ENGINE (v0.14.0, ABI 15) ==============================
 * docs/ATKDSP_PLAN.md §D. Sections 14-17 below: bit primitives, scramblers
 * and linear complexity, FEC decoders, the CRC engine and CRC recovery.
 *
 * Conventions for every function in 14-17:
 *   - Bits are passed ONE PER BYTE (uint8; only bit 0 of each byte is read),
 *     the numpy form ATK's siga already uses. Bit 0 of the array is the first
 *     bit received. Kernels pack to 64-bit words inside.
 *   - Patterns, words and register values written as integers are MSB-FIRST:
 *     the first bit in time is the most significant.
 *   - The one-shot ANALYSIS calls (bits_*, bm*, scrambler_search, crc_recover)
 *     are not the real-time path. They may allocate scratch for the length of
 *     the call and return ATKDSP_E_NOMEM when they cannot. The decoders'
 *     *_decode calls on a created handle do not allocate (rule 2); a handle
 *     carries counters (*_stats) and so is used by one thread at a time.
 *   - Every detector or search reports log10_pfa: the base-10 log of the
 *     chance that random bits would produce the result, multiplied by the
 *     number of candidates the search tried (Bonferroni). -3 means "one in a
 *     thousand, searching this widely". The formula sits beside each one.
 *   - Recognise, never attack: these tools find structure. Encrypted or
 *     compressed data has none, and they say so; nothing here recovers keys.
 */

/* ---- 14. bit primitives -------------------------------------------------- */

/* out[k] = (agreements - disagreements) / (n - k) between b[i] and b[i+k],
 * k = 0..max_lag (max_lag + 1 values; NaN where k >= n). +1 = the stream
 * repeats at lag k, -1 = it repeats inverted, ~0 = unrelated (random bits
 * give a standard deviation of 1/sqrt(n - k)). */
ATKDSP_API int atkdsp_bits_autocorr(const uint8_t *b, size_t n, size_t max_lag, float *out);

/* out[k] = (agreements - disagreements) / nb between the pattern p and
 * a[k .. k+nb), for k = 0..na-nb (na - nb + 1 values). Needs 1 <= nb <= na. */
ATKDSP_API int atkdsp_bits_xcorr(const uint8_t *a, size_t na, const uint8_t *p, size_t nb,
                                 float *out);

/* Pattern search with allowed errors. Reports every position whose Hamming
 * distance to the pattern (or, per `flags`, to its complement and/or its
 * reversal) is <= max_err. One hit per position: the variant with the fewest
 * errors (ties: lowest variant number). pos is the index of the FIRST bit of
 * the match (atkdsp_sync_search reports the bit after it). plen 1..4096.
 * hits == NULL with cap == 0 counts the hits and returns the count; otherwise
 * more than cap hits is ATKDSP_E_CAP. */
#define ATKDSP_BITS_INVERTED 1u    /* flags / variant: the complement        */
#define ATKDSP_BITS_REVERSED 2u    /* flags / variant: the pattern backwards */
typedef struct {
    long long pos;       /* first bit of the match                         */
    int       errors;    /* bit mismatches in the winning variant           */
    int       variant;   /* 0 = as given, |1 inverted, |2 reversed           */
} atkdsp_bithit;         /* 16 bytes */
ATKDSP_API ptrdiff_t atkdsp_bits_search(const uint8_t *b, size_t n, const uint8_t *pat,
                                        size_t plen, int max_err, unsigned flags,
                                        atkdsp_bithit *hits, size_t cap);

/* The raster's column and row sums. Rows are `width` bits starting at
 * `offset`; only whole rows count (rows = (n - offset) / width). col_ones has
 * `width` entries, row_ones `rows`; the parities are the sums mod 2. Any
 * output may be NULL. Returns rows. */
ATKDSP_API ptrdiff_t atkdsp_bits_columns(const uint8_t *b, size_t n, size_t width,
                                         size_t offset, uint32_t *col_ones,
                                         uint32_t *row_ones, uint8_t *col_parity,
                                         uint8_t *row_parity);

/* Width scan: the raster width that makes the columns most consistent,
 * found numerically. For each width w in w_lo..w_hi (w >= 2) the stream is
 * cut into R = n / w whole rows from bit 0, and
 *   raw[w - w_lo] = mean over the w columns of |2 p_c - 1|   (p_c = ones fraction)
 *   z[w - w_lo]   = (raw - mu_R) / (sigma_R / sqrt(w))
 * where mu_R = E|2K/R - 1| and sigma_R^2 = 1/R - mu_R^2 for K ~ Binomial(R, 1/2)
 * — random bits give z ~ N(0, 1) at every width, so widths compare directly
 * and a threshold is a normal quantile: z > Phi^-1(1 - pfa / widths scanned).
 * A frame of length F peaks at F and at its multiples (the harmonic rule:
 * take the smallest width that reaches the peak). R < 2 gives NaN. raw may be
 * NULL. */
ATKDSP_API int atkdsp_bits_width_scan(const uint8_t *b, size_t n, size_t w_lo, size_t w_hi,
                                      float *z, float *raw);

/* Sync discovery: recurring words nobody put in a table.
 * Every word_len-bit window (8..32) is hashed; for each word seen at least
 * min_count times (after thinning occurrences closer than word_len; n < 2^31), the most
 * common gap between occurrences is its period P and the most populated
 * residue mod P its phase. Windows at consecutive phases of one period that
 * overlap consistently and whose counts agree (the smaller above
 * mx/2 + 1.5 sqrt(mx): 3 sigma above what one random extra bit would leave)
 * are merged into one longer pattern (up to 128 bits), counted exactly on its
 * grid, and trimmed from either end while that lowers log10_pfa (a field
 * that never changes stays in, a counter's slowly changing bit drops out).
 *   expected  = grid slots * 2^-length  (matches random bits would put there)
 *   log10_pfa = log10 P(Binomial(slots, 2^-length) >= count)
 *               + log10(2^word_len * (n / word_len) * P)   (words x periods x phases)
 * Reported when log10_pfa <= log10(pfa) (pfa <= 0 means 1e-3), best first;
 * returns the best `cap`. flags bit 0: the pattern is itself periodic in at
 * most half its length (idle fill or a preamble, not a sync word). */
#define ATKDSP_SYNC_MAXBITS 128
#define ATKDSP_SYNC_REPETITIVE 1
typedef struct {
    long long first;         /* position of the first on-grid match         */
    int       period;        /* bits from one occurrence to the next        */
    int       length;        /* pattern bits (word_len..128)                */
    int       count;         /* exact matches on the grid                   */
    int       flags;         /* ATKDSP_SYNC_REPETITIVE                      */
    double    expected;      /* matches random bits would give on the grid  */
    double    log10_pfa;     /* see above                                   */
    uint8_t   pattern[ATKDSP_SYNC_MAXBITS];   /* one bit per byte          */
} atkdsp_synccand;           /* 168 bytes */
ATKDSP_API ptrdiff_t atkdsp_bits_sync_discover(const uint8_t *b, size_t n, int word_len,
                                               int min_count, double pfa,
                                               atkdsp_synccand *out, size_t cap);

/* Bit statistics in one pass. NIST SP 800-22 rev 1a where named.
 *   run0 / run1   run-length histograms; [k] = runs of exactly k (1..31),
 *                 [32] = runs of 32 or more; the first and last runs count
 *   ngram[k-1][v] overlapping k-bit windows with value v (MSB first), k = 1..8
 *   chi2          non-overlapping chi2_bits-bit words against uniform; chi2_bits
 *                 is the widest (<= 8) with at least 5 expected per cell;
 *                 chi2_p = Q(dof/2, chi2/2)
 *   maurer        Maurer's universal statistic fn (§2.9) at the largest block L
 *                 with n >= 1010 * L * 2^L (NIST's own table); maurer_L = 0 and
 *                 NaN when n < 2020
 *   apen          approximate entropy ApEn(m) (§2.12), m = min(10, floor(log2 n) - 6);
 *                 apen_m = 0 and NaN when n < 128
 *   block_entropy non-overlapping w-bit words, bits per BIT (1.0 = maximal),
 *                 w = 1..16 at [w-1]; NaN while there are fewer words than 2^w
 *                 (a plug-in estimate below that cannot reach its maximum) */
typedef struct {
    uint64_t n, ones;
    double   density;
    uint64_t run0[33], run1[33];
    uint64_t ngram[8][256];
    double   chi2, chi2_p, maurer, maurer_p, apen, apen_p;
    double   block_entropy[16];
    int      chi2_bits, maurer_L, apen_m, reserved;
} atkdsp_bitstats;
ATKDSP_API int atkdsp_bits_stats(const uint8_t *b, size_t n, atkdsp_bitstats *out);

/* Walsh-Hadamard parity relations between columns. Rows of `width` bits from
 * bit 0 (rows = n / width); for each row the chosen columns form a value
 * v = sum_j b[row*width + cols[j]] << j (ncols 1..24), and
 *   spectrum[u] = sum over rows of (-1)^(parity of the columns selected by u)
 * for u = 0..2^ncols - 1. spectrum[0] = rows. A linear relation — a parity or
 * checksum bit, an XOR of fields — gives |spectrum[u]| = rows; random columns
 * give about sqrt(rows). Returns rows (at most 2^31 - 1, else ATKDSP_E_ARG). */
ATKDSP_API ptrdiff_t atkdsp_bits_wht(const uint8_t *b, size_t n, size_t width,
                                     const uint32_t *cols, int ncols, int32_t *spectrum);

/* The analyzer-strip transforms, over blocks of `size` bits (a power of two,
 * 4..65536), averaged over navg consecutive blocks (navg <= 0: every whole
 * block). x = 1 - 2b (+-1). Returns the blocks used; 0 (outputs NaN) when
 * n < size. Either output may be NULL.
 *  walsh:  sequency_power[s] = X[h(s)]^2 / size, X = Walsh-Hadamard transform
 *          of x, listed in SEQUENCY order (s = number of sign changes of the
 *          Walsh function; h(s) = bit-reverse(s ^ (s >> 1))), size entries;
 *          dyadic_acf[t] = (1/size) sum_i x[i] x[i XOR t], natural order.
 *  cyclic: cyclic_time[t] = (1/size) sum_i x[i] x[(i + t) mod size], size
 *          entries; cyclic_freq[k] = |DFT(x)[k]|^2 / size, k = 0..size/2. */
ATKDSP_API ptrdiff_t atkdsp_bits_walsh(const uint8_t *b, size_t n, size_t size, int navg,
                                       float *sequency_power, float *dyadic_acf);
ATKDSP_API ptrdiff_t atkdsp_bits_cyclic(const uint8_t *b, size_t n, size_t size, int navg,
                                        float *cyclic_time, float *cyclic_freq);

/* ---- 15. scramblers and linear complexity ------------------------------- */

/* Berlekamp-Massey over GF(2): the shortest LFSR that generates b[0..n).
 * poly receives the connection polynomial c_0..c_L (c_0 = 1) with
 *   b[k] = XOR_{j=1..L} c_j b[k-j]      for every k >= L.
 * Returns L (the linear complexity), ATKDSP_E_CAP when cap < L + 1.
 * An L-stage register needs only 2L bits to be found. */
ATKDSP_API ptrdiff_t atkdsp_bm(const uint8_t *b, size_t n, uint8_t *poly, size_t cap);
/* The linear complexity profile: L_of_n[i] = linear complexity of b[0..i].
 * An LFSR plateaus at its length; random bits track (i + 1) / 2. Returns the
 * final L. */
ATKDSP_API ptrdiff_t atkdsp_bm_profile(const uint8_t *b, size_t n, int32_t *L_of_n);

/* Scrambler registers. `poly` is the polynomial as a bitmask in ATK siga.bits'
 * convention (siga.bits.lfsr_sequence): degree D = highest set bit (2..63),
 * bit 0 must be set, and the register obeys
 *   r[t + D] = XOR over i < D with bit i set of r[t + i].
 * `*state` is the register, carried across calls (bit i = the i-th oldest of
 * the D bits it holds), so a stream may be fed in any blocks.
 *  descramble_add: additive (synchronous). out = b XOR the register output;
 *     state is the LFSR (seed it with the scrambler's start state, nonzero).
 *     Scrambling and descrambling are the same operation.
 *  descramble_mult / scramble_mult: multiplicative (self-synchronising):
 *     out[t] = b[t] XOR (XOR over i < D, bit i set, of s[t - D + i]) where s is
 *     the SCRAMBLED stream; state holds its last D bits. The descrambler
 *     locks after D bits from any start state; one channel error comes out as
 *     one error per term of the polynomial.
 * The same mask in both means the same register: a multiplicative scrambler
 * fed zeros emits the additive one's sequence. Returns 0, or ATKDSP_E_ARG for
 * a mask without bit 0 (a register that only ever emits zeros) or a zero seed. */
ATKDSP_API int atkdsp_descramble_add(const uint8_t *b, size_t n, uint64_t poly,
                                     uint64_t *state, uint8_t *out);
ATKDSP_API int atkdsp_descramble_mult(const uint8_t *b, size_t n, uint64_t poly,
                                      uint64_t *state, uint8_t *out);
ATKDSP_API int atkdsp_scramble_mult(const uint8_t *b, size_t n, uint64_t poly,
                                    uint64_t *state, uint8_t *out);

/* Scrambler search: which register, ranked by the structure it leaves.
 * Candidates are every trinomial (max_terms 3) or trinomial and pentanomial
 * (max_terms 5) of degree 2..max_degree (<= 32) — odd weight, as every
 * primitive polynomial has — in a fixed order: degree, then middle taps
 * lexicographic. Each is applied as the feed-forward filter
 *   y[t] = b[t] XOR (XOR over i < D, bit i set, of b[t - D + i])
 * which IS the multiplicative descrambler, and which also removes an additive
 * scrambler whatever its seed (the register sequence satisfies its own
 * recurrence), leaving the data smeared by the polynomial. The structure of y
 * over bits max_degree..n is the G statistic of its non-overlapping w-bit words
 * against uniform (w = widest <= 8 with 5 expected per cell; dof = 2^w - 1):
 *   G = 2 sum O ln(O / E) ~ chi2(dof) for random bits,
 *   log10_pfa = log10 Q(dof/2, G/2) + log10(candidates + 1).
 * *g_identity (may be NULL) is the same statistic on b itself — doing nothing.
 * A candidate is reported when log10_pfa <= log10(pfa) (pfa <= 0: 1e-3) AND
 * G - G_identity > z * sqrt(4 dof), z = Phi^-1(1 - pfa / (candidates + 1)):
 * a stream that already had its structure was not scrambled. Best G first
 * (near-ties: lower degree, fewer taps). ADD also recovers the seed when y
 * holds a constant run of 2D bits or more (the data was constant there, so
 * the register is b XOR that constant), steps the register back to bit 0, and
 * reports g_data, the statistic of the stream descrambled with that seed.
 * Returns the number reported (the best `cap`). */
#define ATKDSP_SCRAM_ADD  0
#define ATKDSP_SCRAM_MULT 1
typedef struct {
    uint64_t poly;        /* siga.bits mask                                    */
    uint64_t seed;        /* ADD, seeded: the register at bit 0                */
    double   g;           /* G statistic of the filtered stream y              */
    double   log10_pfa;
    double   g_data;      /* ADD, seeded: G of the seeded descramble; else NaN */
    int      degree;
    int      terms;       /* nonzero terms: 3 or 5                             */
    int      seeded;
    int      block_bits;  /* w                                                 */
} atkdsp_scramcand;       /* 56 bytes */
ATKDSP_API ptrdiff_t atkdsp_scrambler_search(const uint8_t *b, size_t n, int kind,
                                             int max_degree, int max_terms, double pfa,
                                             double *g_identity,
                                             atkdsp_scramcand *out, size_t cap);

/* ---- 16. FEC decoders ------------------------------------------------------
 * Soft input is a log-likelihood ratio per coded bit, LLR = ln P(0)/P(1):
 * positive means 0. Hard bits go in as +1 (bit 0) / -1 (bit 1); 0 is an
 * erasure. The decoders count what they do: */
typedef struct {
    uint64_t calls;       /* decode calls                                      */
    uint64_t in;          /* LLRs (Viterbi) or codewords (block codes) taken   */
    uint64_t out;         /* bits decided (Viterbi) or codewords returned      */
    uint64_t corrected;   /* bits / symbols changed by the block decoders      */
    uint64_t failures;    /* words found uncorrectable                         */
    uint64_t resets;
} atkdsp_fec_stats;       /* 48 bytes */

/* 16a. Convolutional codes. Rate 1/n (n 1..8), constraint length K (2..15),
 * generators in the usual octal form: the MOST significant of the K bits
 * multiplies the CURRENT input bit (K=7 NASA: 0171, 0133; LTE: 0133, 0171,
 * 0165). Optional puncturing: an n x punc_len matrix, row-major (row i =
 * output i, column j = input bit t with t mod punc_len = j), 1 = sent; the
 * coded stream interleaves the kept outputs of each input bit in order i.
 *   TERM      the encoder starts at zero and K-1 zero tail bits follow the
 *             data (they are coded and sent; decode returns the data only).
 *   TAILBITE  the encoder starts in the state its last K-1 data bits leave
 *             (LTE PBCH/PDCCH). Decoded wrap-around: the block is extended by
 *             `depth` steps of itself on both sides (0 = the whole block,
 *             i.e. three laps) and traced back from the best end state.
 *   STREAM    continuous; decisions are released `depth` steps late (0 =
 *             6K, 12K punctured), at fixed step indices, so the output is
 *             identical however the LLRs are chunked; _flush releases the rest.
 * max_bits bounds the data bits of one TERM or TAILBITE block (<= 2^28) — the
 * decision memory and every scratch buffer are allocated once, here.
 * Metrics are float32; each step subtracts the best metric so they never grow. */
#define ATKDSP_CONV_TERM     0
#define ATKDSP_CONV_TAILBITE 1
#define ATKDSP_CONV_STREAM   2
typedef struct atkdsp_viterbi atkdsp_viterbi;
ATKDSP_API atkdsp_viterbi *atkdsp_viterbi_create(int K, int n, const uint32_t *polys,
                                                 const uint8_t *puncture, int punc_len,
                                                 int mode, int depth, size_t max_bits);
ATKDSP_API void      atkdsp_viterbi_destroy(atkdsp_viterbi *v);
ATKDSP_API void      atkdsp_viterbi_reset(atkdsp_viterbi *v);
/* Most bits one decode of nllr LLRs can return. */
ATKDSP_API size_t    atkdsp_viterbi_out_max(const atkdsp_viterbi *v, size_t nllr);
/* TERM / TAILBITE: llr is exactly one block (its length fixes the step count;
 * a length no step count produces is ATKDSP_E_ARG). Returns the data bits.
 * *metric (may be NULL) = the decoded path's correlation with the input,
 * sum (1 - 2c) llr / sum |llr|: 1.0 when every received bit agrees with the
 * decoded codeword, ~0 for noise. STREAM: any number of LLRs; returns the bits
 * released by this call; *metric is NaN. */
ATKDSP_API ptrdiff_t atkdsp_viterbi_decode(atkdsp_viterbi *v, const float *llr, size_t nllr,
                                           uint8_t *bits, size_t cap, float *metric);
/* STREAM: trace back from the best state and release every undecided bit
 * (a partial step's LLRs are dropped). Other modes: 0. */
ATKDSP_API ptrdiff_t atkdsp_viterbi_flush(atkdsp_viterbi *v, uint8_t *bits, size_t cap);
/* "Viterbi K=7 r=1/2 (0171, 0133), punctured 3/4 [110 101], streaming, depth 84" */
ATKDSP_API int       atkdsp_viterbi_describe(const atkdsp_viterbi *v, char *buf, size_t cap);
ATKDSP_API int       atkdsp_viterbi_stats(const atkdsp_viterbi *v, atkdsp_fec_stats *st);
/* The encoder for every case above (TERM appends the tail; STREAM starts at
 * zero and adds none). Returns coded bits written. */
ATKDSP_API ptrdiff_t atkdsp_conv_encode(int K, int n, const uint32_t *polys,
                                        const uint8_t *puncture, int punc_len, int mode,
                                        const uint8_t *bits, size_t nbits,
                                        uint8_t *out, size_t cap);

/* 16b. Reed-Solomon over GF(2^m), m = 3..16, in the usual parameters:
 * gfpoly the field polynomial (bit m set), fcr the first consecutive root
 * (index form), prim the root spacing (the generator's roots are
 * alpha^(prim*(fcr + i)), i < nroots), pad the shortening (leading zero
 * symbols not sent). n = 2^m - 1 - pad, k = n - nroots. Codewords are
 * systematic: k data symbols then nroots parity. Decoding is Berlekamp-Massey
 * + Chien + Forney with erasures (2 errors + erasures <= nroots).
 * ATKDSP_RS_DUAL_BASIS: symbols on the wire are in Berlekamp's dual basis, the
 * CCSDS 131.0-B convention (m = 8 with gfpoly 0x187 only). CCSDS RS(255,223):
 * (8, 0x187, 112, 11, 32, 0, ATKDSP_RS_DUAL_BASIS). decode corrects sym in
 * place and returns the symbols corrected (err_pos, may be NULL, receives
 * their positions; nroots entries suffice), or ATKDSP_E_DECODE. */
#define ATKDSP_RS_DUAL_BASIS 1
typedef struct atkdsp_rs atkdsp_rs;
ATKDSP_API atkdsp_rs *atkdsp_rs_create(int m, uint32_t gfpoly, int fcr, int prim,
                                       int nroots, int pad, int flags);
ATKDSP_API void       atkdsp_rs_destroy(atkdsp_rs *r);
ATKDSP_API int        atkdsp_rs_params(const atkdsp_rs *r, int *n, int *k);
ATKDSP_API int        atkdsp_rs_encode(atkdsp_rs *r, const uint16_t *data, uint16_t *parity);
ATKDSP_API int        atkdsp_rs_decode(atkdsp_rs *r, uint16_t *sym, const int *erasures,
                                       int nerase, int *err_pos);
ATKDSP_API int        atkdsp_rs_describe(const atkdsp_rs *r, char *buf, size_t cap);
ATKDSP_API int        atkdsp_rs_stats(const atkdsp_rs *r, atkdsp_fec_stats *st);

/* 16c. Binary BCH, narrow sense: n = 2^m - 1 - shorten (m 3..16), designed
 * to correct t errors; the generator is the product of the distinct minimal
 * polynomials of alpha^1..alpha^2t over the field gfpoly (0 = the default
 * primitive polynomial for m). Codewords are data (k bits) then parity, the
 * first bit the highest-degree coefficient — POCSAG's BCH(31,21) is
 * (5, 2, 0, 0). decode corrects in place: returns bits corrected (err_pos,
 * may be NULL, gets their indices; t entries suffice) or ATKDSP_E_DECODE. */
typedef struct atkdsp_bch atkdsp_bch;
ATKDSP_API atkdsp_bch *atkdsp_bch_create(int m, int t, int shorten, uint32_t gfpoly);
ATKDSP_API void        atkdsp_bch_destroy(atkdsp_bch *b);
ATKDSP_API int         atkdsp_bch_params(const atkdsp_bch *b, int *n, int *k, int *t);
/* The generator, n - k + 1 coefficients, highest degree first. Returns n - k. */
ATKDSP_API int         atkdsp_bch_generator(const atkdsp_bch *b, uint8_t *g, size_t cap);
ATKDSP_API int         atkdsp_bch_encode(atkdsp_bch *b, const uint8_t *data, uint8_t *code);
ATKDSP_API int         atkdsp_bch_decode(atkdsp_bch *b, uint8_t *code, int *err_pos);
ATKDSP_API int         atkdsp_bch_describe(const atkdsp_bch *b, char *buf, size_t cap);
ATKDSP_API int         atkdsp_bch_stats(const atkdsp_bch *b, atkdsp_fec_stats *st);

/* 16d. Any small binary linear code from its generator matrix (k x n,
 * row-major 0/1, rank k; n <= 32, n - k <= 20), decoded by syndrome table:
 * the received word is moved to the nearest codeword (the minimum-weight
 * coset leader). When two error patterns of that weight share the syndrome
 * the word is equidistant from two codewords, and decode says so
 * (ATKDSP_E_DECODE) rather than pick one. dmin is found while the table is
 * built. The build is bounded: a weight level that would take it past 2^25
 * error patterns is not started, so a degenerate code (dmin 1, e.g. [I | 0])
 * builds in seconds and the syndromes it never reached decode as
 * uncorrectable (for k > 20 an unfound dmin is reported 0). data (k, may be
 * NULL) receives the information bits of the corrected codeword. Returns bits
 * corrected. */
typedef struct atkdsp_blk atkdsp_blk;
ATKDSP_API atkdsp_blk *atkdsp_blk_create(int n, int k, const uint8_t *G);
ATKDSP_API void        atkdsp_blk_destroy(atkdsp_blk *c);
ATKDSP_API int         atkdsp_blk_params(const atkdsp_blk *c, int *n, int *k, int *dmin);
ATKDSP_API int         atkdsp_blk_encode(const atkdsp_blk *c, const uint8_t *data, uint8_t *code);
ATKDSP_API int         atkdsp_blk_decode(atkdsp_blk *c, uint8_t *code, uint8_t *data);
ATKDSP_API int         atkdsp_blk_describe(const atkdsp_blk *c, char *buf, size_t cap);
ATKDSP_API int         atkdsp_blk_stats(const atkdsp_blk *c, atkdsp_fec_stats *st);

/* ---- 17. the CRC engine and CRC recovery --------------------------------
 * Parameters in the Rocksoft model (width 1..64; poly without its x^width
 * term; init the register preset; refin reflects each input BYTE; refout
 * reflects the register before xorout). Messages are bits and need not be
 * whole bytes. The register runs MSB-first over the bits in the order given,
 * so the check value over "123456789" is the CRC of its ASCII bytes written
 * out MSB-first. For an air stream sent LSB-first, the air order already IS
 * the reflected order: pass it with refin = 0. refin with a partial last byte
 * reflects the whole bytes only. */
typedef struct {
    uint64_t poly, init, xorout;
    uint64_t check;       /* catalogue: the CRC of "123456789"                 */
    int      width, refin, refout, reserved;
} atkdsp_crcspec;         /* 48 bytes */
ATKDSP_API int atkdsp_crc(const uint8_t *bits, size_t nbits, const atkdsp_crcspec *s,
                          uint64_t *out);
/* The catalogue of named parameter sets (facts, re-typed from the published
 * catalogues; each one's check value is tested). */
ATKDSP_API int         atkdsp_crc_catalog_count(void);
ATKDSP_API int         atkdsp_crc_catalog(int idx, atkdsp_crcspec *out);
ATKDSP_API const char *atkdsp_crc_catalog_name(int idx);

/* CRC recovery: an unknown CRC worked out from frames that carry it.
 * Frames are back to back in `bits`, frame i nbits[i] long (each <= 2^24). A CRC is affine:
 * with the coverage of frame i read as a polynomial p_i (message M_i, field
 * un-reflected / un-swapped to the register value),
 *   p_i = M_i x^w + field_i  ==  init x^(len M_i) + xorout'   (mod g)
 * so the XOR of two frames of one length is a multiple of g (init and xorout
 * cancel), and three frames of different lengths combine into one that is
 * too. The GCD of those polynomials is g (spurious (1+x) powers from the
 * length differences removed); init and xorout then come from a 2-unknown
 * linear solve, and every candidate must reproduce EVERY frame.
 * Searched: width (0 = 3..64), reflection (refin/refout) and field layout
 * (MSB first, or little-endian bytes for byte widths), a header the CRC does
 * not cover (0..max_header bits; ATKDSP_CRC_END), or bits after the field
 * (ATKDSP_CRC_SEARCH: the field need not end the frame).
 *   log10_pfa = log10(trials) - w * (equations beyond the unknowns) * log10 2
 * (each frame is w bits of equations; g, init and xorout cost w each). Needs
 * at least three frames (two of one length, or three lengths with four or
 * more frames when init is unknown). Reported when log10_pfa <= log10(pfa)
 * (pfa <= 0: 1e-3). Width unknown: offsets beyond 0 are scanned for the plain
 * form only (refin = refout = 0, MSB-first field).
 * What frames cannot decide, and how it is reported:
 *  - a bit constant across all frames (a header before the coverage, a
 *    trailer after the field) is absorbed into init or xorout by an equally
 *    valid set at another offset;
 *  - a polynomial that factors is also exactly a CRC of each factor's width,
 *    and with few frames a spurious factor can stand in for one of g's own;
 *  - for g with an (x+1) factor, h = g/(x+1) gives h x^L == h (mod g) for
 *    every L, so (init ^ h, xorout ^ h) is the same CRC: init_known = 0.
 * So every offset is scanned and every set that reproduces every frame is
 * returned, `alternatives` counting the others of its width, ranked: a
 * catalogue set first, then simple sets (init and xorout each 0 or all ones),
 * then log10_pfa, then a byte-aligned offset (a prior, not a deduction), then
 * the smaller offset. More frames remove the spurious alternatives; what the
 * protocol is known to be removes the rest. init_known = 0 when every frame has the
 * same length: init and xorout cannot be separated and init is reported 0
 * with xorout absorbing it — exact for that length (a catalogue set that
 * reproduces the frames is substituted and named). */
#define ATKDSP_CRC_END    0
#define ATKDSP_CRC_SEARCH 1
typedef struct {
    atkdsp_crcspec spec;   /* .check = the recovered set's CRC of "123456789" */
    int    layout;         /* 0 = field MSB first, 1 = little-endian bytes     */
    int    header_bits;    /* leading bits outside the coverage                */
    int    trailer_bits;   /* bits after the field                             */
    int    init_known;     /* see above                                        */
    int    frames_ok;      /* frames the set reproduces (= nframes)            */
    int    catalog;        /* index of the identical catalogue set, or -1      */
    double log10_pfa;
    int    alternatives;   /* other reported sets of this width that also fit  */
    int    reserved;
} atkdsp_crccand;          /* 88 bytes */
ATKDSP_API ptrdiff_t atkdsp_crc_recover(const uint8_t *bits, const size_t *nbits, int nframes,
                                        int width, int pos, int max_header, double pfa,
                                        atkdsp_crccand *out, size_t cap);

/* ---- 18. naming the code (v0.15.0, ABI 16; docs/ATKDSP_PLAN.md §D4) -------
 * Codes recognised from the bits alone. Every answer carries its evidence
 * (Bill, 2026-10-04): the structure found, a log10_pfa (how likely random
 * bits are to show it, every test counted) and decode -> re-encode
 * agreement. Random bits come back "no code found"; so does a stream beyond
 * the method's reach, with the estimated BER when there is one.
 *
 * Rank deficiency. Cut the stream into rows of w bits. Rows that are
 * codewords of a linear (n, k) code span k dimensions, so the GF(2) rank
 * falls n - k short of n: a RANK DEFICIENCY, and the null space of the row
 * matrix is the code's parity checks. A width of 2n holds one whole codeword
 * at the same place in every row whatever the stream's alignment, so the
 * deficiency shows at 2n without knowing where codewords start; at width n
 * the aligned offset carries all n - k checks and any other only those that
 * fit in a codeword's head or tail. R random rows of w bits lose d dimensions
 * with probability below 4 * 2^-(d (R - w + d)), so an exact deficiency d
 * scores log10_pfa = log10 4 - d (R - w + d) log10 2, plus log10 of the widths
 * and offsets tried.
 *
 * What the statistics assume. Every log10_pfa is computed for uniform,
 * memoryless bits apart from the code, so a stream that is not is refused
 * first, by every function here: a bias, |P(1) - 1/2| > 0.1 over 2000 bits or
 * more ("no code found (biased: P(1) = 0.10)"), and a short linear recurrence
 * — a Berlekamp-Massey linear complexity LC over the first 8192 bits with
 * 2 LC + 64 below the bits examined, which an LFSR, a constant or a periodic
 * stream has and coded random data does not ("no block code (linear
 * recurrence)" from block_code_id, "no code found (linear recurrence)" from
 * conv_code_id and rs_code_id; interleaver_id returns 0).
 *
 * Noise. One bit error in a row breaks plain elimination. After exact
 * elimination, three remedies:
 *  - Cyclic roots (block codes). Every codeword of a cyclic code is a
 *    multiple of its generator g, so each irreducible factor q of g divides
 *    every error-free codeword, and a random word with probability 2^-deg q.
 *    With P_t the polynomial of bits 0..t-1 (first bit highest degree), q
 *    divides the window of bits [t, t + L) exactly when x^-t P_t and
 *    x^-(t+L) P_(t+L) agree mod q: one comparison per window and factor.
 *    Counting divisible windows needs only a fair share of clean codewords,
 *    (1 - p)^n, so BCH(255,239) is named at BER 1e-3 from 24 codewords.
 *    Extended codes (n - 1 cyclic bits and a parity bit) and shortened codes
 *    are covered. The clean share the counts imply must be at least
 *    (1 - 0.08)^n, what BER 8 % would leave, and at least 10 %: structure
 *    that divides only a few windows (a code outside the length range, or
 *    one seen at a wrong length — Golay(24,12) at length 47 fits 5 phases of
 *    24, 7 % of the windows) is not a code at this length. At the limits
 *    stated below the clean share is 23 % or more.
 *  - The robust null space. Random subsets of w + 2 rows, each subset's null
 *    space (an error only SHRINKS it), and a candidate check h is kept when
 *    it holds on enough of the OTHER rows: with u failures among V,
 *        log10 P(Bin(V, 1/2) <= u) + log10(64 w) <= log10(pfa),
 *    no more failures than BER 5 % could cause on a check of its weight, and
 *    no pattern by row index (periods 2..32, G-test at 1e-6) — a wrong width
 *    cuts codewords at an alignment that cycles from row to row, so a
 *    relation fitting one phase holds on that phase's rows only. A check of
 *    weight m failing on a fraction f estimates the BER,
 *    p = (1 - (1 - 2f)^(1/m)) / 2. The span is complete when a clean subset
 *    gave all of it, or when 8 trials in a row found checks but nothing new
 *    (a trial with b bad rows yields a random subspace of the dual; while the
 *    span is short of the dual, such a subspace falls inside it with
 *    probability <= 1/2). It needs about (w + 34) w bits, and works while a
 *    subset of about w rows of w bits holds fewer errors than the code has
 *    checks: short codes to a few percent BER, n = 40 to about 1e-3.
 *  - Short contiguous stretches (convolutional codes). Sliding windows share
 *    bits, so a short stretch is error-free far more often than a random row
 *    subset, and noise only ever lowers an exact deficiency: the largest over
 *    64 stretches is the true one once any stretch is clean.
 * The random subsets come from a fixed xorshift64* generator (seed in `seed`,
 * 0 = the default), so a result repeats exactly and the twin reproduces it.
 *
 * Every null space reported is in CANONICAL form — reduced row echelon with
 * the pivot at each vector's highest set bit, highest pivot first — so the
 * same code always prints the same matrix. Bit j of a row (or of a check) is
 * the j-th bit of the row in time; a generator or check matrix is returned
 * one bit per byte, row-major. Polynomials follow §16: a codeword's first bit
 * is its highest-degree coefficient. */

/* The deficiency heat map. def[(w - w_lo) * n_offsets + o] = w - rank of the
 * matrix of `rows` consecutive w-bit rows starting at bit o (rows 0 = w + 32,
 * each limited by the data), for w in w_lo..w_hi (2..4096) and o in
 * 0..n_offsets-1 (see below for -1). Exact
 * elimination, no noise tolerance: random rows give 0 except with probability
 * about 2^-(d (R - w + d)) for a deficiency d. n_offsets = 1 is the quick scan
 * (offset 0, enough to find 2n); n_offsets = w_hi the full map. A cell is -1
 * where o >= w, or where fewer than w + 16 rows are used — whether the data
 * or a small `rows` limits them. */
ATKDSP_API int atkdsp_gf2_rank_scan(const uint8_t *b, size_t n, size_t w_lo, size_t w_hi,
                                    size_t n_offsets, size_t rows, int16_t *def);

/* Block codes, lengths n_lo..n_hi (2..512). After the stream check above, in
 * order:
 *   1. exact elimination at widths n_lo..2 n_hi from bit 0 (the first
 *      deficient width w0), then each divisor N of w0 in range, smallest
 *      first: the offset with the largest deficiency at width N is the
 *      alignment, accepted only if that deficiency holds over every codeword
 *      (up to 4096) and over three more stretches of the stream. A deficiency
 *      that repeats under a shift smaller than N is not a block code: "no
 *      block code (linear recurrence)" for an LFSR, a constant or a periodic
 *      stream, "(sliding, period P)" for a convolutional stream — returns 0;
 *   2. the cyclic-root count over every length (3 or more, with 8+ codewords)
 *      and offset (factors of x^L - 1 and the primitive polynomials of the
 *      field holding L, degree <= 11): the best (length, offset) by summed
 *      significance, the generator from up to 1024 codewords (a factor must
 *      divide as many as the strongest factor does), the clean floor above;
 *      when the generator's order divides a shorter length in range, that
 *      length is counted too and kept if it gives a generator (a multiple of
 *      the true length scores nearly as well). Then the robust search on the
 *      aligned codewords adds any checks a non-cyclic subcode has;
 *   3. the robust search at widths 2N, N ascending; where it finds checks,
 *      the robust search at width N over every offset gives the alignment.
 * Then one reduction and two refusals (a refusal returns 0 with n, offset,
 * deficiency and log10_pfa filled):
 *   - checks found at a multiple m L of a code's length (m copies of the
 *     length-L checks, block by block) are reduced to length L first;
 *   - checks that hold as well P bits along (P a proper divisor of N: the
 *     shifted rows keep 80 % of the aligned rows' excess over chance; a
 *     repetition code, the nearest block case, keeps 50 %) are not a block
 *     code of length N: "no block code (same checks at every offset)" for a
 *     recurrence, "no block code (sliding, period P)" for a convolutional
 *     stream — or a block code of length P below n_lo, seen at a multiple
 *     (the bits cannot tell those two apart; the step 1 names mean the same);
 *   - the failing codewords must fail the way channel noise makes them.
 *     Noise strikes rows at random, so which rows fail has no pattern by row
 *     index, where structure that fits some rows only (a wrong length, a code
 *     outside the range) fails by the phase of an alignment that cycles from
 *     row to row: refused when a G-test over row periods 2..64 (each residue
 *     with 16 rows or more) gives log10 p <= -6, every period counted. And at
 *     a BER low enough to name a code most failing words carry ONE error, so
 *     their syndrome is a column of the check matrix, where partial structure
 *     fails with arbitrary syndromes: with V the valid fraction, p = 1 -
 *     V^(1/n) and the expected share of single-error syndromes among the F
 *     failures q = n p (1-p)^(n-1) / (1 - (1-p)^n), S seen with
 *     log10 P(Bin(F, q) <= S) <= -4 and S < 0.8 q F is refused (fewer than 8
 *     failures pass). Either refusal is "no code found (n=N fails unlike
 *     noise)".
 * Then the code is named and its evidence gathered:
 *   cyclic   every cyclic shift of a codeword is a codeword; the generator
 *            polynomial is the GCD of the codewords' polynomials (gpoly)
 *   name     "Hamming(7,4)", "Golay(23,12)", "BCH(31,21) t=2" (the largest t
 *            whose consecutive roots g has), "extended Golay(24,12)" (with
 *            " parity first" when the parity bit leads the codeword),
 *            "shortened BCH(63,51) t=2 as (50,38)", "RS(15,11) GF(2^4) binary
 *            image" (a length of m-bit symbols is put to rs_code_id at the
 *            alignment found, and taken when every row of G is an RS
 *            codeword), "cyclic (n,k)" or "linear (n,k)" — named by generator
 *            polynomial, against every primitive polynomial of the field
 *   dmin     exact when n <= 32 and n - k <= 20 (the §16d table), else 0
 *   evidence agreement = 1 - (bits changed by decoding) / (bits decoded)
 *            with that table, the §16c decoder for a named BCH/Hamming code
 *            (shortened too), or the §16b decoder for RS; valid_words =
 *            codewords with a zero syndrome; ber_est from the checks' failures
 * G (k x n, may be NULL) receives the generator in canonical form (identity
 * on its pivot columns: systematic on those positions); gpoly (may be NULL)
 * the generator polynomial, highest degree first, when cyclic (n - k + 1
 * coefficients), shortened (n - k + 1) or extended (of the n - 1 cyclic bits:
 * n - k coefficients). gcap / pcap are their sizes in bytes. When the checks
 * found are verified but no subset was ever clean the dimension is only a
 * lower bound: out->name is "no code found (n=N too noisy)" with n, offset
 * and ber_est filled, and 0 returned.
 * Measured (tests/test_coderec.py; 3000 codewords, 6 trials): Hamming(7,4),
 * extended Hamming(8,4) and Golay(23,12) / (24,12) to BER 5e-2 (5 or 6 of
 * 6); Hamming(15,11) and extended (16,11) to 2e-2 (3e-2: 5 of 6);
 * Hamming(31,26) to 2e-2 (3e-2: 1 of 6); BCH(63) to 1e-2, BCH(127) to 5e-3, BCH(255) to
 * 3e-3 (what limits it is the count of clean codewords among 24); random
 * (n, n/2) codes to 1e-2 at n = 40. Cost on random bits (the worst case: no
 * early exit): 2 Mbit in 0.9 s at n_hi = 256, 5 s at 512.
 * Returns 1 when a code is found, 0 when not, or an error. */
typedef struct {
    int    n, k, offset;      /* block length, dimension, first aligned bit   */
    int    deficiency;        /* n - k as measured                            */
    int    cyclic, extended;  /* see above                                    */
    int    dmin;              /* 0 = not computed                             */
    int    rows_used;         /* aligned codewords examined                   */
    int    robust;            /* how: 0 exact, 1 robust search, 2 cyclic roots */
    int    shortened;         /* shortened cyclic: mother length - n, else 0  */
    double log10_pfa;         /* chance random bits show this structure       */
    double ber_est;           /* channel BER from the checks' failures        */
    double agreement;         /* decode -> re-encode, fraction of bits kept   */
    double valid_words;       /* fraction of codewords with a zero syndrome   */
    char   name[48];
} atkdsp_blockid;             /* 120 bytes */
ATKDSP_API int atkdsp_block_code_id(const uint8_t *b, size_t n, int n_lo, int n_hi,
                                    double pfa, uint64_t seed, atkdsp_blockid *out,
                                    uint8_t *G, size_t gcap, uint8_t *gpoly, size_t pcap);

/* Convolutional codes, rate k/n (n 2..max_n <= 8). The stream is cut into
 * windows of L + 1 steps of n0 bits, sliding one step; for the true n the
 * deficiency grows by n - k per step, so the REDUNDANCY DENSITY (growth per
 * step / n0) is 1 - k/n at the true n and its multiples, and less at any
 * other n0 (a wrong n0 sees a time-varying code). For each n0 the onset L0
 * (the first window length with a significant deficiency) and the growth
 * over 4 more steps give the density; the smallest n0 within 0.05 of the
 * largest is n, and k = n - round(density n). Deficiencies are the best of 64
 * short stretches (above). For k = 1 the generators are then solved directly:
 * c_1 g_i = c_i g_1 for every output i (each stream is the input times its
 * generator), a null space of n K unknowns, at the smallest K (2..max_K <=
 * 15) over the n bit alignments (a wrong alignment is the same code with one
 * generator delayed, so it needs a larger K) — exact null spaces over 64
 * stretches, a candidate kept only if every generator is nonzero, it holds
 * on up to 4096 equations from the whole stream (binomial against 1/2, every
 * stretch, K and alignment counted) and each output's equations fail no more
 * often than BER 5 % allows. A density below half a check per step (k = n)
 * is "no code found". A k / n with k | n is tried as rate 1 / (n / k); an
 * uncertain density (one clean stretch) tries every divisor of n. A rate
 * 1/(m n) code also fits a longer rate 1/n description, with K about m times
 * larger; so a multiple of n wins when its density showed, its generators are
 * found and its K is no larger. Generators in §16a's convention (MSB = the
 * current bit), in stream order, in their shortest form (taps no generator
 * uses dropped). Evidence: a hard-decision Viterbi decode (at most 2^25
 * trellis steps x states) re-encoded, agreement = fraction of coded bits
 * kept; ber_est = 1 - agreement. A true k > 1 code (punctured) is reported as
 * its rate with polys 0. Catalogued codes are named ("K=7 r=1/2 (171,133)
 * NASA/CCSDS, 802.11, DVB-S"). robust = 1 when noise was seen (a stretch
 * below the best, or the generators' equations failing somewhere). `seed` is
 * kept for the ABI: the stretches are fixed, so nothing is drawn. Measured:
 * the catalogue and K up to 15 at BER 1e-2 from 3000 data bits; K = 7 r=1/2
 * at 3e-2. Returns 1 / 0 / error as above. */
typedef struct {
    int      n, k, K;         /* outputs, inputs, constraint length           */
    int      phase;           /* stream bit where output 1 of a step falls    */
    uint32_t polys[8];        /* k = 1: generators in stream order            */
    int      robust, rows_used;
    double   density;         /* measured redundancy density (1 - k/n)        */
    double   log10_pfa, ber_est, agreement;
    char     name[64];
} atkdsp_convid;              /* 152 bytes */
ATKDSP_API int atkdsp_conv_code_id(const uint8_t *b, size_t n, int max_n, int max_K,
                                   double pfa, uint64_t seed, atkdsp_convid *out);

/* Reed-Solomon over GF(2^m), on codeword-ALIGNED input: the first codeword
 * starts at bit `offset` (from a frame sync, or block_code_id). m 3..8 (0 =
 * try them all), nsym symbols per codeword (0 = the full 2^m - 1; fewer =
 * shortened). The representations tried: every primitive field polynomial of
 * degree m, both bit orders within a symbol and, for m = 8 / 0x187, the CCSDS
 * dual basis. Stage 1: for each, every field element is evaluated on up to 64
 * codewords, and the representation scores the summed significance of its
 * elements' zero counts (a binomial test against 2^-m) — every root of the
 * generator adds, so the right one stands out even when few codewords are
 * clean; the best must beat chance over every element of every
 * representation. Stage 2 recounts it over up to 1024 codewords: an element
 * that is a zero of more codewords than chance allows (Bonferroni over
 * elements and representations) is a root of the generator. The roots must
 * be consecutive powers of some alpha^prim: that gives fcr and prim, nroots =
 * their number, k = nsym - nroots. A symbol's bits reversed under the
 * reciprocal polynomial is the same code (roots negated); both are counted
 * and the one with more roots, then the smaller fcr, then MSB first is
 * reported. Stage 1 is only a screen (it picks the representation); stage 2's
 * test is the one log10_pfa reports. Noise only removes codewords from the
 * count. Half a codeword along the roots must vanish (a window across two
 * codewords is no codeword): a stream that keeps them — periodic structure —
 * is "no code found (not codeword-aligned)". Evidence: the §16b decoder over
 * every codeword. Measured: RS(15,11) and
 * RS(63,55) at BER 3e-3; CCSDS RS(255,223) and RS(204,188) at 1e-3, which
 * needs a few hundred codewords (13 % of 2040-bit codewords are then clean).
 * Returns 1 / 0 / error. */
typedef struct {
    int      m, n, k, nroots, fcr, prim;
    uint32_t gfpoly;
    int      dual_basis;      /* CCSDS Berlekamp dual basis                    */
    int      lsb_first;       /* symbol bits sent LSB first                    */
    int      rows_used;
    double   log10_pfa;
    double   valid_words;     /* codewords the decoder accepted               */
    double   agreement;       /* 1 - symbols corrected / symbols decoded      */
    char     name[48];
} atkdsp_rsid;                /* 112 bytes */
ATKDSP_API int atkdsp_rs_code_id(const uint8_t *b, size_t n, size_t offset, int m, int nsym,
                                 double pfa, atkdsp_rsid *out);

/* Block interleavers over coded data. The interleaved stream is itself a
 * linear code of length N = rows x cols, so its period and alignment come
 * from the block search (exact, then robust; N in n_lo..n_hi <= 1024). The
 * period found is the smallest with consistent structure, so its multiples up
 * to 4x and each candidate block start are tried. Each factorisation of N is
 * undone (written by rows, read by columns; the transposed case is the other
 * factorisation), and the one whose de-interleaved stream shows structure at
 * the SMALLEST width (the inner code, local again) gives rows and cols; at
 * the same width the larger deficiency wins. The block sizes tried: the
 * period and its multiples to 4x, then the period's divisors (an interleaver
 * block that is not a multiple of the codeword makes the period their common
 * multiple; inner widths are then scanned to the period, since the codeword
 * can be longer than the block). Block starts: every start of a divisor
 * block in exact mode; otherwise each offset that ties for the period's
 * largest deficiency (the deficiency alone can leave the alignment ambiguous:
 * Hamming(8,4) in 5 x 7 blocks, period 280, has three such offsets, one of
 * them a block start). A plain code is not an interleaver: the inner width
 * must be at most half the width at which the stream as it stands shows
 * structure (a cyclic code's decimations have structure too). Exact widths
 * are scanned at four offsets, so a code of rate below 7/8 shows at its own
 * length. The smallest equivalent interleaver is reported: 3 x 14 over
 * Hamming(7,4) is 3 x 7 (the same output, the codewords kept whole).
 * Data: an interleaver spreads each codeword over its block, so from an
 * unknown start its structure shows at 2N only — about (2N + 32) 2N bits
 * (exact) or (2N + 34) 2N (robust, N <= 512), and 64 blocks or more. A
 * 16 x 7 Hamming(7,4) interleaver at BER 1e-3 is found from 600 blocks
 * (67 kbit; 6 of 6 trials), not from 520. In noise the robust block search
 * needs a subset of 2N rows to keep fewer errors than the deficiency: 63-bit
 * BCH codewords interleaved to a period of 252 or more are out of reach at
 * BER 1e-3. Cost: under 4 s for the cases in tests/test_coderec.py; 10-30 s
 * in noise when the period is a common multiple of 280-420 bits. Returns
 * 1 / 0 / error; on 0 after a period was found, period, offset and
 * log10_pfa still describe it, with rows = cols = 0. */
typedef struct {
    int    period, offset;    /* N and the first bit of an interleaver block  */
    int    rows, cols;        /* written by rows, read by columns             */
    int    inner_width;       /* width at which the undone stream first shows structure */
    int    inner_deficiency;
    double log10_pfa;
} atkdsp_ilvid;               /* 32 bytes */
ATKDSP_API int atkdsp_interleaver_id(const uint8_t *b, size_t n, int n_lo, int n_hi,
                                     double pfa, uint64_t seed, atkdsp_ilvid *out);

/* ==== THE DETECTION TRACK (v0.16.0, ABI 17) ===============================
 * docs/ATKDSP_PLAN.md §B1, §B2, §A3. Sections 19-21: CFAR and the burst
 * tracker, the burst descriptor word, the ring. Twins in reference.py §19-21.
 */

/* ---- 19. CFAR and the burst tracker (§B1) --------------------------------
 * A 1-D CFAR across frequency on each spectrum frame, and a tracker across
 * frames that turns hit bins into one record per burst (OPEN, UPDATE,
 * CLOSE) — the detector cued DF, Bug Sniffer and Drone Discovery wait on.
 *
 * INPUT. Frames are power spectra in dB, nbins per frame, row-major, AVERAGE
 * power per bin: a periodogram (looks = 1) or an average of several
 * (looks = the effective number averaged). A max-hold frame is not gamma
 * distributed and breaks the false-alarm rate: feed the AVERAGE line
 * (atkdsp_spectrum_reduce with the avg detector, or atkdsp_spectrum_stats'
 * average, which comes in the same pass as a display's max-hold), with
 * looks = the FFT frames averaged into it. Values below -300 dB are read as -300. A NaN bin never hits, and
 * a NaN among a bin's reference cells makes its statistic NaN in every mode,
 * so that bin is not hit either: mask NaN bins.
 *
 * THE TEST. k_frames frames are averaged (linear power, a sliding window),
 * so a cell's noise is gamma with K = looks x k_frames looks. Each bin's
 * reference is N = 2 n_ref cells, n_ref each side beyond n_guard guard cells
 * (siga.detect.cfar's `train` and `guard`). Masked bins are never tested and
 * never used as reference; the window walks past them. At a band edge (or
 * past a mask) a side that runs short borrows the shortfall from the other
 * side, so N stays N and the false-alarm rate holds there too — a bin that
 * cannot gather N cells is untestable. The statistic T:
 *   CA  the mean of the N cells (the default);
 *   OS  the os_k-th smallest (crowded bands: a neighbour does not raise it;
 *       os_k 0 = ceil(3N/4));
 *   GO  the greater of the two side means (clutter edges: fewer false alarms);
 *   SO  the smaller (closely spaced targets). A GO/SO bin whose window had to
 *       borrow falls back to CA, with CA's alpha.
 * A cell hits when its average > alpha x T. THE THRESHOLD IS DERIVED:
 *   Pfa(alpha) = integral Q(K, K alpha t) f_T(t) dt
 * with f_T the density of T for unit-mean K-look gamma cells (gamma for CA,
 * the max / min of two gammas for GO / SO, the order statistic for OS),
 * integrated by Simpson's rule in ln t over the range outside which each
 * tail holds less than 1e-9 x Pfa, and solved for alpha by regula falsi
 * (Illinois) in ln alpha. K = 1 reproduces the closed forms to 1e-9:
 * CA alpha = N (Pfa^(-1/N) - 1); OS Pfa = prod_{i<k} (N-i)/(N-i+alpha); GO and
 * SO Gandhi & Kassam (1988). Pfa is per cell per frame; with k_frames > 1
 * successive frames share cells, so hits are correlated in time (the rate
 * per frame is still Pfa). atkdsp_cfar_alpha exposes the solver.
 *
 * THE TRACKER. Hit bins closer than merge_gap non-hit bins form one blob
 * (min_bins wide or more). A blob joins the track whose last span it
 * overlaps most, within merge_gap + 1 bins (ties: the older track); a track
 * may take several blobs (a signal breaking up); a blob near no track
 * starts a pending one (beyond max_tracks it is dropped and counted). A
 * pending track OPENs after open_m hit frames in the last open_n, and is
 * forgotten if it has none; an open track CLOSEs after close_k frames in a
 * row without a hit, and reports UPDATE when its span grows. One event per
 * track per frame at most, so atkdsp_cfar_events_max = nframes x max_tracks.
 *   t0  the start of its first hit frame. With k_frames > 1 the average
 *       lags the burst, so the k raw frames up to the first hit are tested
 *       on their own (the band's raw power against its noise, a single-frame
 *       gamma test at CFAR_EDGE_PFA = 1e-3 for the band's width) and t0 is
 *       the start of the earliest that passes; if none does, t0 is the
 *       midpoint of that window and T0_COARSE is set.
 *   t1  the end of its last hit frame; with k_frames > 1 the end of the last
 *       frame whose raw band passed the same test, if it is within k - 1
 *       frames of the last hit (else the midpoint, and T1_COARSE). A frame
 *       spans [frame_t0 + j dt, frame_t0 + (j+1) dt).
 *   fc_bin   the excess-weighted centroid over the track's life (fractional
 *            bin; the caller knows Hz per bin and where bin 0 sits).
 *   lo/hi    the union of its spans; level_db the peak averaged bin power,
 *            noise_db the noise estimate there (T / E[T], unbiased in
 *            homogeneous noise), snr_db their difference, band_snr_db the
 *            best frame's sum over the hit bins against their noise.
 *   flags    T0_COARSE, T1_COARSE; EDGE (a bin of the span used a borrowed
 *            window); MASK_ADJ (a masked bin within merge_gap + 1); FLUSHED
 *            (closed by atkdsp_cfar_flush).
 *
 * The plan's create(nbins, mode, n_ref, ...) became a config struct:
 * thirteen parameters, and the next one does not reorder a call. */
#define ATKDSP_CFAR_CA 0
#define ATKDSP_CFAR_OS 1
#define ATKDSP_CFAR_GO 2
#define ATKDSP_CFAR_SO 3
#define ATKDSP_CFAR_OPEN   1
#define ATKDSP_CFAR_UPDATE 2
#define ATKDSP_CFAR_CLOSE  3
#define ATKDSP_CFAR_T0_COARSE 1u
#define ATKDSP_CFAR_T1_COARSE 2u
#define ATKDSP_CFAR_EDGE      4u
#define ATKDSP_CFAR_MASK_ADJ  8u
#define ATKDSP_CFAR_FLUSHED  16u
#define ATKDSP_CFAR_EDGE_PFA 1e-3

typedef struct {
    int32_t mode;        /* ATKDSP_CFAR_CA (default) / OS / GO / SO          */
    int32_t n_ref;       /* reference cells EACH side, 1..256 (16)           */
    int32_t n_guard;     /* guard cells each side, 0..256 (4)                */
    int32_t k_frames;    /* frames averaged before testing, 1..64 (1)        */
    double  pfa;         /* per cell per frame, 1e-15..0.5 (1e-6)            */
    double  looks;       /* periodograms averaged into one frame, 0.25..1e4 (1) */
    int32_t os_k;        /* OS rank 1..2 n_ref; 0 = ceil(3N/4)                */
    int32_t open_m;      /* open after open_m hit frames ... (2)             */
    int32_t open_n;      /* ... in the last open_n, 1..64 (3)                */
    int32_t close_k;     /* close after close_k misses in a row, 1..1000 (3) */
    int32_t merge_gap;   /* non-hit bins inside one detection (1)            */
    int32_t min_bins;    /* narrower blobs ignored (1)                       */
    int32_t max_tracks;  /* pending + open at once, 1..65536 (256)           */
    int32_t reserved[3];
} atkdsp_cfar_config;    /* 72 bytes */

typedef struct {
    double   t0, t1;     /* seconds; t1 NaN until CLOSE                      */
    double   fc_bin;     /* excess-weighted centroid, fractional bin         */
    int32_t  type;       /* OPEN / UPDATE / CLOSE                            */
    uint32_t id;         /* unique per handle (from 1)                       */
    uint32_t flags;
    int32_t  lo_bin, hi_bin;
    int32_t  n_hits;     /* frames with a hit                                */
    float    level_db, noise_db, snr_db, band_snr_db;
} atkdsp_cfar_event;     /* 64 bytes */

typedef struct {
    uint64_t frames, warmup;      /* taken; before k_frames had arrived     */
    uint64_t cells_tested, cell_hits, untestable;
    uint64_t opened, closed, updates;
    uint64_t dropped;             /* blobs refused at max_tracks             */
    uint64_t resets;
    uint64_t tracks;              /* pending + open now                      */
} atkdsp_cfar_counters;           /* 88 bytes */

typedef struct atkdsp_cfar atkdsp_cfar;
ATKDSP_API void         atkdsp_cfar_defaults(atkdsp_cfar_config *cfg);
/* The solver alone: alpha for (mode, n_ref, pfa) with `looks` total looks
 * (K) and the OS rank; *et (may be NULL) = E[T] in units of the noise mean. */
ATKDSP_API int          atkdsp_cfar_alpha(int mode, int n_ref, double pfa, double looks,
                                          int os_k, double *alpha, double *et);
/* NULL on an invalid configuration (the binding prints the ranges above). */
ATKDSP_API atkdsp_cfar *atkdsp_cfar_create(size_t nbins, const atkdsp_cfar_config *cfg);
ATKDSP_API void         atkdsp_cfar_destroy(atkdsp_cfar *c);
/* Forget the frames and the tracks (a retune); the mask stays. */
ATKDSP_API void         atkdsp_cfar_reset(atkdsp_cfar *c);
/* nbins bytes, nonzero = masked (known emitters, operator mutes); NULL
 * clears. Rebuilds the reference windows; open tracks carry on. */
ATKDSP_API int          atkdsp_cfar_set_mask(atkdsp_cfar *c, const uint8_t *mask);
ATKDSP_API size_t       atkdsp_cfar_events_max(const atkdsp_cfar *c, size_t nframes);
/* cap below events_max(nframes) is refused before anything is consumed. */
ATKDSP_API ptrdiff_t    atkdsp_cfar_process(atkdsp_cfar *c, const float *frames_db,
                                            size_t nframes, double frame_t0,
                                            double frame_dt, atkdsp_cfar_event *ev,
                                            size_t cap);
/* CLOSE every open track now (FLUSHED), forget the pending ones; the stream
 * may continue. cap >= max_tracks always suffices. */
ATKDSP_API ptrdiff_t    atkdsp_cfar_flush(atkdsp_cfar *c, atkdsp_cfar_event *ev, size_t cap);
/* The last tested frame: averaged power and threshold (dB; NaN where not
 * tested) and the hit mask, nbins each; any may be NULL. ATKDSP_E_EMPTY
 * before the first tested frame. For drawing the threshold on a display. */
ATKDSP_API int          atkdsp_cfar_last(const atkdsp_cfar *c, float *avg_db, float *thr_db,
                                         uint8_t *hits);
/* alpha, CA's alpha (GO / SO edge fallback), E[T], and the configuration as
 * resolved (os_k filled in). Any may be NULL. */
ATKDSP_API int          atkdsp_cfar_info(const atkdsp_cfar *c, double *alpha, double *alpha_ca,
                                         double *et, atkdsp_cfar_config *cfg);
/* "CFAR CA: 16+16 reference cells, 4 guard each side, Pfa 1e-06 per cell per
 * frame, K = 1 look(s) x 1 frame(s), alpha 17.28 (12.37 dB); open 2 of 3,
 * close after 3 misses, merge gap 1, min 1 bin(s); 4096 bins, 0 masked,
 * tracks 0/256" */
ATKDSP_API int          atkdsp_cfar_describe(const atkdsp_cfar *c, char *buf, size_t cap);
ATKDSP_API int          atkdsp_cfar_stats(const atkdsp_cfar *c, atkdsp_cfar_counters *st);

/* ---- 20. The burst descriptor word (§B2) ---------------------------------
 * Radar has the PDW; this is the comms equivalent — one record per burst,
 * measured on channelised I/Q (centred near 0 Hz, a few times the burst's
 * bandwidth) around a coarse [i0, i1) such as the CFAR's t0 / t1 converted
 * to samples. The buffer should hold noise on at least one side: the noise
 * level N comes from outside [i0, i1) — beyond a guard of (i1 - i0)/8 + 8
 * samples when the buffer has 32 or more there, else from all of what lies
 * outside (and if that is within 1 dB of the plateau it is the burst itself)
 * — and is re-measured outside the burst once its edges are found; with no
 * noise to be had, NO_NOISE (the levels that need N are NaN).
 * Truncated means the edge step is AT the buffer's end (TRUNC_START /
 * TRUNC_END), not that the smoothed core reaches it.
 *
 *   t0, t1   the burst's edges as sample POSITIONS in the buffer: a burst
 *            whose first sample is s and last e has t0 = s - 0.5 and t1 =
 *            e + 0.5 (the boundaries between samples). Found by a
 *            least-squares step between the noise level N and the plateau
 *            P around each end of the burst's core (the run where the power,
 *            smoothed over `window` samples, holds above a quarter of the
 *            way from N to P), at a level that depends on the envelope:
 *              - a constant envelope (FM, FSK, a carrier): the half-power
 *                crossing, N + (P - N)/2 (or thresh_db below P when
 *                thresh_db > 0 is given);
 *              - a Gaussian-like one (OFDM, spread spectrum; flag GAUSSIAN,
 *                from the burst's own power fluctuation with the noise's
 *                share removed, > 0.5): the likelihood boundary
 *                ln(P/N) N P / (P - N), because such a signal fades through
 *                its half-power level on its own.
 *            Measured (tests/test_detection.py): a hard-keyed constant-
 *            envelope burst within 2 samples at 10 dB SNR (at the channel
 *            rate, noise in the whole channel), 0 at 20-30 dB; a Gaussian-
 *            like burst 10 kHz wide at 100 kS/s within ~20 samples at 10 dB,
 *            ~7 at 20 dB, 2 at 30 dB (its own fades set that); on a slow
 *            ramp the edge is the ramp's half-power point.
 *   fc_hz    the power centroid of the burst's spectrum inside its 99 %
 *            band, relative to 0 Hz of the buffer.
 *   obw99_hz the 99 % occupied bandwidth: the noise-subtracted Welch
 *            spectrum (Hann, nfft, hop nfft/2, the burst widened by nfft/2
 *            each side so its whole length is under the windows' flat sum;
 *            nfft 0 = the largest power of two <= burst/8 in 16..4096), its
 *            residual left SIGNED so the noise cancels, edges where the
 *            power summed from each end reaches 0.5 %, interpolated in the
 *            bin. Within a bin at 10-30 dB.
 *   bw26_hz  bins within 26 dB of the peak, outermost to outermost; NaN and
 *            BW26_FLOOR when the noise floor (+3 sigma of the average) is
 *            above that line.
 *   drift_hz_s  a line through the centroids of 3-8 segments spread over
 *            the burst (each 64 to 32768 samples); NaN and SHORT below that. Within
 *            5 % for a carrier at 10-30 dB; on FSK the symbol mix moves each
 *            segment's centroid, so it is coarser (~10 % in the tests).
 *   rise_s, fall_s  10 % to 90 % of the amplitude above the noise, on the
 *            power smoothed over Wr = fs / bandwidth samples (<= 63);
 *            RISE_UNRESOLVED / FALL_UNRESOLVED when the 10 % level is
 *            inside 3 sigma of the smoothed noise, when the result is under
 *            2 Wr (smoothing-limited), when a truncated end has none, and
 *            always for a GAUSSIAN burst.
 *   peak_dbfs, mean_dbfs  10 log10 of the peak and mean |x|^2 over
 *            [t0, t1] (0 dBFS = a full-scale complex sinusoid, |x| = 1);
 *            noise_dbfs = N; snr_db = (P - N) / N; LOW_SNR below 6 dB.
 *   mod_hint 0: unknown. The §B4 probes will fill it.
 * A one-shot analysis call (it allocates scratch for the call).
 * ATKDSP_E_EMPTY when there is no burst at [i0, i1) (no power above the
 * noise there). */
#define ATKDSP_BDW_TRUNC_START       1u   /* the burst began before the buffer */
#define ATKDSP_BDW_TRUNC_END         2u
#define ATKDSP_BDW_NO_NOISE          4u
#define ATKDSP_BDW_LOW_SNR           8u
#define ATKDSP_BDW_RISE_UNRESOLVED  16u
#define ATKDSP_BDW_FALL_UNRESOLVED  32u
#define ATKDSP_BDW_BW26_FLOOR       64u
#define ATKDSP_BDW_SHORT           128u
#define ATKDSP_BDW_GAUSSIAN        256u
#define ATKDSP_BDW_HALF_POWER_DB   3.0102999566398120

typedef struct {
    double   t0, t1;          /* sample positions of the edges (s - 0.5, e + 0.5) */
    double   duration_s;
    double   fc_hz, drift_hz_s;
    double   obw99_hz, bw26_hz;
    double   rise_s, fall_s;
    double   reserved_d;
    float    peak_dbfs, mean_dbfs, noise_dbfs, snr_db;
    int32_t  mod_hint;
    uint32_t flags;
    int32_t  nfft;            /* the spectrum's segment length                 */
    int32_t  window;          /* the core smoothing, samples                   */
} atkdsp_bdw;                 /* 112 bytes */

/* thresh_db 0 = the edge rule above; > 0 = that many dB below the plateau
 * for every envelope. nfft 0 = automatic. */
ATKDSP_API int atkdsp_burst_measure(const atkdsp_cf32 *iq, size_t n, double fs,
                                    size_t i0_coarse, size_t i1_coarse, float thresh_db,
                                    int nfft, atkdsp_bdw *out);

/* ---- 21. The ring (§A3) ----------------------------------------------------
 * A block ring that lives in CALLER memory — a private buffer inside ATK, or
 * a named shared-memory mapping a program in another process attaches to
 * (atkpr's processor, the next native consumer). The §RF-F2 block ring and
 * the §RF-F9 DVR. The library allocates nothing and starts no thread.
 *   - ONE producer, up to 16 consumers, each with its own cursor.
 *   - THE PRODUCER NEVER WAITS (an SDR cannot): claim() hands it the next
 *     block to fill in place, commit() publishes it with its header. A slow
 *     consumer is lapped and told exactly how many blocks it lost.
 *   - Fixed-size blocks (block_bytes, rounded up to 64), each with an
 *     atkdsp_ring_meta: seq (set by commit), the global index of its first
 *     sample, UTC ns, centre and sample rate, gain, the sample format, flags
 *     (RETUNE, OVERFLOW, CAL — so a DVR cut never spans a retune without
 *     saying so) and nbytes (<= block_bytes).
 *   - A SEQLOCK per block: the lock word is 2 seq + 1 while seq is written
 *     and 2 seq + 2 once committed; a reader checks it before and after its
 *     copy, so a torn block is never returned — it is counted (torn) and
 *     reported lost.
 *   - read() copies; peek() / release() is zero-copy: peek returns a pointer
 *     into the ring, and release says whether the block survived the use
 *     (ATKDSP_E_OVERRUN: it was overwritten — discard what you computed).
 *   - copy_range(seq0, seq1) cuts a past span for the DVR (Export pre-D
 *     I/Q, DF the selection): whole blocks still in the ring, packed, with
 *     their headers; it returns the number copied and stops at cap / mcap.
 *   - 64-bit sequence numbers; the size is bounded by memory alone. 40 MSPS
 *     of ci16 is 160 MB/s, so 30 s of DVR is 4.8 GB: ATK's Setup states the
 *     RAM before it allocates.
 * The memory must be 8-byte aligned (64 is better) and atkdsp_ring_bytes()
 * long. init() formats it; attach() validates a ring another process made
 * (NULL if the magic, the version or the sizes are wrong). Producer calls
 * come from one thread; each consumer id from one thread at a time.
 * Layout: docs in src/ring.c (fixed offsets, so any language can read it). */
#define ATKDSP_RING_MAX_CONSUMERS 16
#define ATKDSP_RING_RETUNE   1u     /* meta flags: the first block after a retune */
#define ATKDSP_RING_OVERFLOW 2u     /* the device dropped samples before it        */
#define ATKDSP_RING_CAL      4u     /* the calibration state changed               */

typedef struct {
    uint64_t seq, first_sample;
    int64_t  t_utc_ns;
    double   fc_hz, fs_hz;
    float    gain_db;
    int32_t  format;               /* the caller's code (ATK: atkdsp.FMT)         */
    uint32_t flags, nbytes;
} atkdsp_ring_meta;                /* 56 bytes */

typedef struct {
    uint64_t committed, oldest;    /* blocks [oldest, committed) are held          */
    uint64_t block_bytes, nblocks, total_bytes;
    uint64_t torn;                 /* reads that caught a block being overwritten  */
    uint32_t max_consumers, active, active_mask, reserved;
    uint64_t lost[ATKDSP_RING_MAX_CONSUMERS];  /* per consumer id                   */
    uint64_t read[ATKDSP_RING_MAX_CONSUMERS];
    uint64_t lag[ATKDSP_RING_MAX_CONSUMERS];   /* committed - cursor               */
} atkdsp_ring_counters;            /* 448 bytes */

typedef struct atkdsp_ring atkdsp_ring;
/* Bytes of memory a ring needs (0: block_bytes >= 1, nblocks >= 2,
 * max_consumers 1..16). */
ATKDSP_API size_t       atkdsp_ring_bytes(size_t block_bytes, size_t nblocks, int max_consumers);
ATKDSP_API atkdsp_ring *atkdsp_ring_init(void *mem, size_t mem_bytes, size_t block_bytes,
                                         size_t nblocks, int max_consumers);
ATKDSP_API atkdsp_ring *atkdsp_ring_attach(void *mem, size_t mem_bytes);
/* Producer: the next block to fill (block_bytes writable), then commit it.
 * commit sets meta.seq; nbytes 0 or above block_bytes, or no claim, is E_ARG
 * (so read's 0 always means "nothing new"). */
ATKDSP_API void        *atkdsp_ring_claim(atkdsp_ring *r);
ATKDSP_API int          atkdsp_ring_commit(atkdsp_ring *r, const atkdsp_ring_meta *m);
/* A consumer id (0..15), reading from the next block committed or from the
 * oldest still held; E_CAP when all max_consumers are taken. close frees it
 * (a consumer in a process that died can be closed by anyone). */
ATKDSP_API int          atkdsp_ring_open(atkdsp_ring *r, int from_oldest);
ATKDSP_API int          atkdsp_ring_close(atkdsp_ring *r, int id);
/* The next block: its bytes (0 = nothing new), its header, and the blocks
 * lost since the last call (lapped or torn). A block larger than cap is
 * E_CAP and stays to be read again with a bigger buffer. */
ATKDSP_API ptrdiff_t    atkdsp_ring_read(atkdsp_ring *r, int id, void *out, size_t cap,
                                         atkdsp_ring_meta *m, uint64_t *lost);
ATKDSP_API const void  *atkdsp_ring_peek(atkdsp_ring *r, int id, atkdsp_ring_meta *m,
                                         uint64_t *lost);
ATKDSP_API int          atkdsp_ring_release(atkdsp_ring *r, int id);
ATKDSP_API ptrdiff_t    atkdsp_ring_copy_range(const atkdsp_ring *r, uint64_t seq0,
                                               uint64_t seq1, void *out, size_t cap,
                                               atkdsp_ring_meta *metas, size_t mcap);
ATKDSP_API int          atkdsp_ring_stats(const atkdsp_ring *r, atkdsp_ring_counters *st);
/* "ring: 256 blocks x 1048576 bytes (256.0 MiB of data), 3 of 16 consumers,
 * 1234 committed, 0 torn reads" */
ATKDSP_API int          atkdsp_ring_describe(const atkdsp_ring *r, char *buf, size_t cap);

#ifdef __cplusplus
}
#endif
#endif /* ATKDSP_H */
