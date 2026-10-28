/* Internal helpers shared by the kernels. Not part of the ABI. */
#ifndef ATKDSP_INTERNAL_H
#define ATKDSP_INTERNAL_H

#if !defined(_MSC_VER) && !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200112L   /* posix_memalign */
#endif

#define ATKDSP_BUILD 1
#include "atkdsp.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _MSC_VER
#  define ATK_RESTRICT __restrict
#  define ATK_INLINE   static __inline
#else
#  define ATK_RESTRICT restrict
#  define ATK_INLINE   static inline
#endif

#ifndef ATK_PI
#  define ATK_PI 3.14159265358979323846
#endif

/* -120 dB floor so log10 never sees zero. Same constant as atk.core.dsp. */
#define ATK_DB_EPS 1e-12f

ATK_INLINE float atk_db10(float p) {
    return 10.0f * log10f(p + ATK_DB_EPS);
}

/* Wrap a phase into [-pi, pi). */
ATK_INLINE double atk_wrap(double ph) {
    ph = fmod(ph, 2.0 * ATK_PI);
    if (ph >= ATK_PI) ph -= 2.0 * ATK_PI;
    if (ph < -ATK_PI) ph += 2.0 * ATK_PI;
    return ph;
}

/* Aligned allocation for SIMD-friendly buffers. */
void *atk_aligned_malloc(size_t bytes);
void  atk_aligned_free(void *p);

/* ---- a tiny lock-free claim, for the FFT plan's scratch slots -----------
 * The header promises "a plan may be shared by several threads"; making that
 * true means two caller threads must never take the same scratch slot. These
 * are the whole of the machinery: try to claim a free flag (0 -> 1), and
 * release it (-> 0). Compare-and-swap on the two compilers atkdsp targets,
 * no OS threads, no OpenMP required — so it also protects callers that share
 * a plan WITHOUT OpenMP, which omp_get_thread_num never could. A flag is one
 * 32-bit int; on Windows `long` and `int` are both 32-bit, so the cast is
 * width-exact. */
#if defined(_MSC_VER)
#  include <intrin.h>
static __inline int atk_try_claim(volatile int *p) {
    return _InterlockedCompareExchange((volatile long *)p, 1, 0) == 0;
}
static __inline void atk_release(volatile int *p) {
    _InterlockedExchange((volatile long *)p, 0);
}
#else
static inline int atk_try_claim(volatile int *p) {
    return __sync_bool_compare_and_swap(p, 0, 1);
}
static inline void atk_release(volatile int *p) {
    __sync_lock_release(p);
}
#endif

/* ---- the bitstream engine (§14-17) ------------------------------------- */
#if defined(_MSC_VER) && defined(__AVX2__)
#  include <intrin.h>      /* an AVX2 CPU always has POPCNT */
ATK_INLINE int atk_popcount64(uint64_t x) { return (int)__popcnt64(x); }
#elif defined(_MSC_VER)
/* MSVC's __popcnt64 always emits POPCNT; an ATKDSP_AVX2=OFF build is for CPUs
 * that may lack it, so count in software there. */
ATK_INLINE int atk_popcount64(uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (int)((x * 0x0101010101010101ULL) >> 56);
}
#else
ATK_INLINE int atk_popcount64(uint64_t x) { return __builtin_popcountll(x); }
#endif
ATK_INLINE int atk_parity64(uint64_t x) { return atk_popcount64(x) & 1; }

/* Bits (one per byte, bit 0 read) -> 64-bit words, bit i of the stream at
 * word i/64 bit i%64. The array has nw + 1 words (a zero guard word), so
 * atk_bits_get64 may read one word past the last. Caller frees. */
uint64_t *atk_bits_pack(const uint8_t *b, size_t n, size_t *nw);
/* The 64 stream bits starting at bit p (bit 0 of the result = stream bit p). */
ATK_INLINE uint64_t atk_bits_get64(const uint64_t *w, size_t p) {
    const size_t q = p >> 6; const unsigned r = (unsigned)(p & 63u);
    return r ? (w[q] >> r) | (w[q + 1] << (64u - r)) : w[q];
}

/* Statistics shared by the detectors (src/stat.c). */
double atk_igamc(double a, double x);          /* regularised upper incomplete gamma Q(a,x) */
double atk_log_igamc(double a, double x);      /* ln Q(a,x), finite far into the tail       */
double atk_log10_binom_tail(double n, double c, double ln_q); /* log10 P(Bin(n,q) >= c)      */
double atk_norm_isf(double p);                 /* z with P(N(0,1) > z) = p                    */
double atk_gstat(const uint8_t *b, size_t n, int w, double *dof); /* G vs uniform, w-bit words */
int    atk_chi_width(size_t nbits);            /* widest w <= 8 with 5 expected per cell      */

/* CRC engine internals (src/crc.c), shared with src/crc_recover.c. */
uint64_t atk_reflect(uint64_t v, int w);
int      atk_crc_valid(const atkdsp_crcspec *s);
uint64_t atk_crc_register(const uint8_t *bits, size_t nbits, int width, uint64_t poly,
                          uint64_t init, int refin);
uint64_t atk_crc_check_value(const atkdsp_crcspec *s);
int      atk_crc_catalog_find(const atkdsp_crcspec *s);

/* Valid turbo code-block sizes (36.212 Table 5.1.3-3), for the SIB1 PHY's
 * CRC-gated K search. Defined in src/lte_turbo.c; not part of the ABI. */
int atk_qpp_count(void);
int atk_qpp_k(int i);

/* ---- LTE handle -------------------------------------------------------
 * Defined here, not in the ABI (atkdsp.h exposes only the opaque typedef),
 * so src/lte.c (PSS/SSS) and src/lte_pbch.c (PBCH->MIB) — and the SIB
 * decoders to come — share one handle and one cached 128-point FFT plan.
 * atkdsp_lte_create / _destroy live in src/lte.c and own every field. */
#define ATK_LTE_VIT_LAPS 3            /* wrap-around laps for the TB Viterbi */
struct atkdsp_lte {
    atkdsp_cf32 *pss;         /* 3 * 128  reference symbols, time domain     */
    atkdsp_cf32 *pss_freq;    /* 3 * 62   reference values, frequency        */
    float       *sss;         /* 3 * 336 * 62, +1/-1                         */
    float        pss_energy[3];
    atkdsp_fft  *sym;         /* a 128-point plan, reused by SSS and PBCH    */
    atkdsp_fft  *scan;        /* the scan plan and its scratch               */
    size_t       scan_n;
    atkdsp_cf32 *X, *P, *Z;   /* scan_n each                                 */
    double      *energy;      /* scan_n running |x|^2 over 128               */
    double      *fold;        /* PSS_PERIOD  non-coherent PSS fold accum     */
    int         *fcount;      /* PSS_PERIOD  occurrences per residue         */
    atkdsp_cf32 *work;        /* scan_n  residual buffer for interference    */
                              /* cancellation across detect passes           */
    int32_t     *vit_bp;      /* ATK_LTE_VIT_LAPS*40*64 Viterbi backpointers */
};

#endif
