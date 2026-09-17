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

#endif
