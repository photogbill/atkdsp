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
    int32_t     *vit_bp;      /* ATK_LTE_VIT_LAPS*40*64 Viterbi backpointers */
};

#endif
