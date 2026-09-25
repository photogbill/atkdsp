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

#define ATKDSP_ABI_VERSION 11
#define ATKDSP_VERSION_STRING "0.10.0"
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

/* ---- errors ------------------------------------------------------------ */
#define ATKDSP_OK            0
#define ATKDSP_E_ARG        -1   /* null pointer, bad size, unknown enum      */
#define ATKDSP_E_NOMEM      -2   /* a *_create could not allocate            */
#define ATKDSP_E_CAP        -3   /* caller's output buffer too small         */
#define ATKDSP_E_FFT        -4   /* the FFT backend reported failure         */

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

#ifdef __cplusplus
}
#endif
#endif /* ATKDSP_H */
