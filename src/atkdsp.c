/* Identity, threads, allocation. */
#include "internal.h"

#ifdef _OPENMP
#  include <omp.h>
#endif

#if defined(__AVX2__)
#  define ATK_SIMD "avx2"
#elif defined(__AVX__)
#  define ATK_SIMD "avx"
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#  define ATK_SIMD "sse2"
#else
#  define ATK_SIMD "scalar"
#endif

#if defined(_MSC_VER)
#  define ATK_CC "msvc"
#elif defined(__clang__)
#  define ATK_CC "clang"
#elif defined(__GNUC__)
#  define ATK_CC "gcc"
#else
#  define ATK_CC "cc"
#endif

#ifdef _OPENMP
#  define ATK_OMP "openmp"
#else
#  define ATK_OMP "no-openmp"
#endif

/* The ABI number is STRINGIFIED from the macro, not typed in. It was typed
 * in — " abi" " 1" — and would have gone on saying 1 after the bump to 2,
 * which is the one line anybody reads to check exactly that. */
#define ATK_STR2(x) #x
#define ATK_STR(x)  ATK_STR2(x)

static const char BUILD_INFO[] =
    ATKDSP_VERSION_STRING " abi " ATK_STR(ATKDSP_ABI_VERSION)
    " " ATK_CC " " ATK_SIMD " " ATK_OMP " built " __DATE__ " " __TIME__;

int atkdsp_abi_version(void) { return ATKDSP_ABI_VERSION; }
const char *atkdsp_version_string(void) { return ATKDSP_VERSION_STRING; }
const char *atkdsp_build_info(void) { return BUILD_INFO; }

int atkdsp_bytes_per_sample(int fmt) {
    switch (fmt) {
    case ATKDSP_FMT_CU8:     return 2;
    case ATKDSP_FMT_CI8:     return 2;
    case ATKDSP_FMT_CI16:    return 4;
    case ATKDSP_FMT_CI16Q11: return 4;
    case ATKDSP_FMT_CF32:    return 8;
    default:                 return 0;
    }
}

static int g_threads = 0;   /* 0 = OpenMP default */

int atkdsp_set_threads(int n) {
#ifdef _OPENMP
    g_threads = n > 0 ? n : 0;
    return atkdsp_get_threads();
#else
    (void)n;
    return 1;
#endif
}

int atkdsp_get_threads(void) {
#ifdef _OPENMP
    return g_threads > 0 ? g_threads : omp_get_max_threads();
#else
    return 1;
#endif
}

void *atk_aligned_malloc(size_t bytes) {
    if (bytes == 0) bytes = 64;
#ifdef _MSC_VER
    return _aligned_malloc(bytes, 64);
#else
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    return p;
#endif
}

void atk_aligned_free(void *p) {
    if (!p) return;
#ifdef _MSC_VER
    _aligned_free(p);
#else
    free(p);
#endif
}
