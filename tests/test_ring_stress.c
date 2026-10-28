/* The ring's stress test and Bill's bench (docs/ATKDSP_PLAN.md §A3).
 *
 *   atkdsp_ring_stress [seconds] [MB/s] [block_KiB] [nblocks]
 *
 * One producer and three consumers on their own threads:
 *   A reads everything as fast as it can,
 *   B does three times the work per block (it falls behind and is lapped),
 *   C opens and then stalls until the producer stops, then drains.
 * The producer fills every 8-byte word of block `seq` with a value derived
 * from seq and the word's index, so a consumer can prove each block it gets
 * is whole (no torn block is ever returned). At the end every consumer
 * drains, and for each one   blocks read + blocks lost == blocks committed
 * since it opened   — the lost counts are exact to the block.
 *
 * MB/s 0 = as fast as the producer can go (the default, 2 s, 64 KiB x 256).
 * Bill's bench (40 MSPS of ci16 is 160 MB/s; ten minutes):
 *   bin\atkdsp_ring_stress.exe 600 160 1024 512
 * Exit status 0 = every check held. Prints the ring's stats as it goes.
 *
 * A data race on the block bytes is the seqlock's design (the second read of
 * the lock is what makes the copy safe to use), so this test is not one to
 * run under ThreadSanitizer. */
#include "atkdsp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <windows.h>
#  include <malloc.h>
typedef HANDLE thr_t;
static double now_s(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
static DWORD WINAPI thr_tramp(LPVOID a);
typedef struct { void *(*fn)(void *); void *arg; } tramp_arg;
static DWORD WINAPI thr_tramp(LPVOID a) {
    tramp_arg *t = (tramp_arg *)a;
    t->fn(t->arg);
    return 0;
}
static int thr_start(thr_t *t, void *(*fn)(void *), void *arg, tramp_arg *ta) {
    ta->fn = fn; ta->arg = arg;
    *t = CreateThread(NULL, 0, thr_tramp, ta, 0, NULL);
    return *t ? 0 : -1;
}
static void thr_join(thr_t t) { WaitForSingleObject(t, INFINITE); CloseHandle(t); }
static void *aligned_alloc64(size_t n) { return _aligned_malloc(n, 64); }
static void aligned_free64(void *p) { _aligned_free(p); }
static volatile LONG stop_flag = 0;
static int get_stop(void) { return InterlockedCompareExchange(&stop_flag, 0, 0) != 0; }
static void set_stop(void) { InterlockedExchange(&stop_flag, 1); }
#else
#  include <pthread.h>
#  include <time.h>
typedef pthread_t thr_t;
typedef struct { int unused; } tramp_arg;
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
static int thr_start(thr_t *t, void *(*fn)(void *), void *arg, tramp_arg *ta) {
    (void)ta;
    return pthread_create(t, NULL, fn, arg);
}
static void thr_join(thr_t t) { pthread_join(t, NULL); }
static void *aligned_alloc64(size_t n) {
    void *p = NULL;
    return posix_memalign(&p, 64, n) == 0 ? p : NULL;
}
static void aligned_free64(void *p) { free(p); }
static volatile int stop_flag = 0;
static int get_stop(void) { return __atomic_load_n(&stop_flag, __ATOMIC_ACQUIRE); }
static void set_stop(void) { __atomic_store_n(&stop_flag, 1, __ATOMIC_RELEASE); }
#endif

#define K_MIX 0x9E3779B97F4A7C15ULL

typedef struct {
    atkdsp_ring *r;
    size_t block_bytes;
    double rate;                 /* bytes / s, 0 = full speed */
    double seconds;
    uint64_t committed;
} prod_arg;

typedef struct {
    atkdsp_ring *r;
    int id, kind;                /* 0 fast, 1 slow, 2 stalled */
    size_t block_bytes;
    uint64_t start, nread, lost, torn_seen, order_bad, nbytes;
    int ok;
} cons_arg;

static void *producer(void *a) {
    prod_arg *p = (prod_arg *)a;
    const size_t words = p->block_bytes / 8;
    const double t0 = now_s();
    uint64_t seq = 0;
    for (;;) {
        const double t = now_s() - t0;
        if (t >= p->seconds) break;
        if (p->rate > 0.0) {                       /* hold the rate by the clock */
            const double due = (double)seq * (double)p->block_bytes / p->rate;
            if (t < due) continue;
        }
        uint64_t *w = (uint64_t *)atkdsp_ring_claim(p->r);
        for (size_t i = 0; i < words; ++i) w[i] = seq * K_MIX + i;
        atkdsp_ring_meta m;
        memset(&m, 0, sizeof m);
        m.first_sample = seq * words;
        m.nbytes = (uint32_t)(words * 8);
        m.fs_hz = 40e6;
        if (atkdsp_ring_commit(p->r, &m) != ATKDSP_OK) { fprintf(stderr, "commit failed\n"); break; }
        ++seq;
    }
    p->committed = seq;
    set_stop();
    return NULL;
}

static int check_block(const uint64_t *w, size_t words, const atkdsp_ring_meta *m) {
    if (m->first_sample != m->seq * words || m->nbytes != words * 8) return 0;
    for (size_t i = 0; i < words; ++i) if (w[i] != m->seq * K_MIX + i) return 0;
    return 1;
}

static void *consumer(void *a) {
    cons_arg *c = (cons_arg *)a;
    const size_t words = c->block_bytes / 8;
    uint64_t *buf = (uint64_t *)aligned_alloc64(c->block_bytes);
    uint64_t *junk = (uint64_t *)aligned_alloc64(c->block_bytes);
    uint64_t last = 0;
    int have_last = 0;
    c->ok = buf && junk;
    if (c->kind == 2) while (!get_stop()) { /* stalled */ }
    for (;;) {
        atkdsp_ring_meta m;
        uint64_t lost = 0;
        const int stopped = get_stop();
        const ptrdiff_t n = atkdsp_ring_read(c->r, c->id, buf, c->block_bytes, &m, &lost);
        c->lost += lost;
        if (n < 0) { c->ok = 0; break; }
        if (n == 0) {
            if (stopped) break;                    /* drained after the producer stopped */
            continue;
        }
        if (!check_block(buf, words, &m)) { c->ok = 0; c->torn_seen++; }
        if (have_last && m.seq != last + 1 + lost) c->order_bad++;
        last = m.seq; have_last = 1;
        c->nread++;
        c->nbytes += (uint64_t)n;
        if (c->kind == 1)                          /* three times the work */
            for (int k = 0; k < 3; ++k) memcpy(junk, buf, c->block_bytes);
    }
    aligned_free64(buf);
    aligned_free64(junk);
    return NULL;
}

int main(int argc, char **argv) {
    const double seconds = argc > 1 ? atof(argv[1]) : 2.0;
    const double mbps = argc > 2 ? atof(argv[2]) : 0.0;
    const size_t block = (size_t)(argc > 3 ? atof(argv[3]) : 64.0) * 1024;
    const size_t nblocks = (size_t)(argc > 4 ? atof(argv[4]) : 256.0);
    const size_t bytes = atkdsp_ring_bytes(block, nblocks, 4);
    if (!bytes) { fprintf(stderr, "bad ring shape\n"); return 2; }
    void *mem = aligned_alloc64(bytes);
    if (!mem) { fprintf(stderr, "cannot allocate %zu bytes\n", bytes); return 2; }
    atkdsp_ring *r = atkdsp_ring_init(mem, bytes, block, nblocks, 4);
    if (!r) { fprintf(stderr, "ring_init refused\n"); return 2; }
    char desc[256];
    atkdsp_ring_describe(r, desc, sizeof desc);
    printf("%s; %.0f s at %s\n", desc, seconds, mbps > 0 ? argv[2] : "full speed");

    cons_arg cs[3];
    for (int i = 0; i < 3; ++i) {
        memset(&cs[i], 0, sizeof cs[i]);
        cs[i].r = r; cs[i].kind = i; cs[i].block_bytes = block;
        cs[i].id = atkdsp_ring_open(r, 0);
        cs[i].start = 0;
    }
    prod_arg pa = {r, block, mbps * 1e6, seconds, 0};
    thr_t tp, tc[3];
    tramp_arg ta[4];
    for (int i = 0; i < 3; ++i) thr_start(&tc[i], consumer, &cs[i], &ta[i]);
    const double t0 = now_s();
    thr_start(&tp, producer, &pa, &ta[3]);
    thr_join(tp);
    const double el = now_s() - t0;
    for (int i = 0; i < 3; ++i) thr_join(tc[i]);

    atkdsp_ring_counters st;
    atkdsp_ring_stats(r, &st);
    int fail = 0;
    printf("committed %llu blocks in %.2f s (%.1f MB/s), torn reads caught %llu\n",
           (unsigned long long)st.committed, el,
           (double)st.committed * (double)block / el / 1e6, (unsigned long long)st.torn);
    static const char *names[3] = {"fast", "slow (3x work)", "stalled"};
    for (int i = 0; i < 3; ++i) {
        const cons_arg *c = &cs[i];
        const int exact = c->nread + c->lost == st.committed - c->start;
        const int agree = c->lost == st.lost[c->id] && c->nread == st.read[c->id];
        printf("  %-15s read %llu, lost %llu (%.1f %%), whole %s, order %s, counts exact %s\n",
               names[i], (unsigned long long)c->nread, (unsigned long long)c->lost,
               st.committed ? 100.0 * (double)c->lost / (double)st.committed : 0.0,
               c->ok && !c->torn_seen ? "yes" : "NO", c->order_bad ? "NO" : "yes",
               exact && agree ? "yes" : "NO");
        if (!c->ok || c->torn_seen || c->order_bad || !exact || !agree) fail = 1;
    }
    if (st.committed < 2) fail = 1;
    printf(fail ? "RING STRESS: FAILED\n" : "RING STRESS: ok\n");
    aligned_free64(mem);
    return fail;
}
