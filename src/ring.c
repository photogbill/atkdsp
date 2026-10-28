/* 21. The ring (docs/ATKDSP_PLAN.md §A3).
 *
 * A block ring in CALLER memory: a private buffer inside ATK, or a named
 * shared-memory mapping another process attaches to. One producer that never
 * waits, up to 16 consumers each with its own cursor, fixed-size blocks each
 * with a header, a seqlock per block. Definitions in atkdsp.h §21; the twin
 * (single process) is python/atkdsp/reference.py §21 (class Ring).
 *
 * LAYOUT (all offsets from the start of the caller's memory, which must be
 * 8-byte aligned; 64 is better):
 *   0     header, 256 bytes: magic, version, sizes, write_seq (the next seq
 *         to commit), the producer's claimed flag, the torn-read counter
 *   256   max_consumers records of 64 bytes: active, cursor, lost, read,
 *         the seq a peek holds
 *   ...   nblocks slots: a 64-byte slot header (the seqlock word and the
 *         block's atkdsp_ring_meta) then block_bytes (rounded up to 64)
 * Everything is a fixed-width integer at a fixed offset, so a second process
 * — or a program in another language — reads the same layout.
 *
 * THE SEQLOCK. A slot's lock word is 2 seq + 1 while block `seq` is being
 * written into it and 2 seq + 2 once it is committed. A reader that wants
 * seq s takes the lock (acquire), copies, fences, and takes it again: the
 * copy is good only if both reads were 2 s + 2. Anything larger means the
 * producer lapped the reader — the block is gone and is counted lost.
 * Copying memory the producer may be writing is a race the C standard does
 * not bless and every seqlock relies on; the second read of the lock is
 * what makes the result safe to use.
 *
 * ATOMICS through a small shim (gcc/clang __atomic, MSVC Interlocked and
 * barriers), so the code stays C99 and the library still starts no threads.
 */
#include "internal.h"

#include <stdio.h>

#if defined(_MSC_VER)
#  include <intrin.h>
#  if defined(_M_ARM64)
#    define ATK_HW_FENCE() __dmb(_ARM64_BARRIER_ISH)
#  elif defined(_M_ARM)
#    define ATK_HW_FENCE() __dmb(_ARM_BARRIER_ISH)
#  else
#    define ATK_HW_FENCE() _mm_mfence()
#  endif
#  if defined(_M_ARM64) || defined(_M_ARM)
#    define ATK_WEAK_ORDER 1      /* ARM: loads and stores reorder; real fences */
#  else
#    define ATK_WEAK_ORDER 0      /* x86 / x64 (TSO): a compiler barrier suffices */
#  endif
/* 64-bit loads and stores must be single accesses. On x64 and ARM64 an
 * aligned volatile access is; on 32-bit x86 the compiler splits it in two, so
 * there it goes through the 8-byte compare-exchange. */
#  if defined(_M_IX86)
static __inline uint64_t ld64(const volatile uint64_t *p) {
    return (uint64_t)_InterlockedCompareExchange64((volatile __int64 *)p, 0, 0);
}
static __inline void st64(volatile uint64_t *p, uint64_t v) {
    __int64 old = *(volatile __int64 *)p;
    for (;;) {
        const __int64 got = _InterlockedCompareExchange64((volatile __int64 *)p, (__int64)v, old);
        if (got == old) break;
        old = got;
    }
}
#  else
static __inline uint64_t ld64(const volatile uint64_t *p) { return *p; }
static __inline void st64(volatile uint64_t *p, uint64_t v) { *p = v; }
#  endif
static __inline uint64_t ld_acq(const volatile uint64_t *p) {
    const uint64_t v = ld64(p);
#  if ATK_WEAK_ORDER
    ATK_HW_FENCE();
#  endif
    _ReadWriteBarrier();
    return v;
}
static __inline uint64_t ld_rlx(const volatile uint64_t *p) { return ld64(p); }
static __inline void st_rel(volatile uint64_t *p, uint64_t v) {
#  if ATK_WEAK_ORDER
    ATK_HW_FENCE();
#  endif
    _ReadWriteBarrier();
    st64(p, v);
}
static __inline void st_rlx(volatile uint64_t *p, uint64_t v) { st64(p, v); }
static __inline void fence_acq(void) {
#  if ATK_WEAK_ORDER
    ATK_HW_FENCE();
#  endif
    _ReadWriteBarrier();
}
static __inline void fence_rel(void) { fence_acq(); }
static __inline int cas32(volatile uint32_t *p, uint32_t expect, uint32_t desired) {
    return (uint32_t)_InterlockedCompareExchange((volatile long *)p, (long)desired,
                                                 (long)expect) == expect;
}
static __inline void add64(volatile uint64_t *p, uint64_t v) {
    _InterlockedExchangeAdd64((volatile __int64 *)p, (__int64)v);
}
static __inline void st32_rel(volatile uint32_t *p, uint32_t v) {
    _InterlockedExchange((volatile long *)p, (long)v);
}
static __inline uint32_t ld32_acq(const volatile uint32_t *p) {
    const uint32_t v = *p;
#  if ATK_WEAK_ORDER
    ATK_HW_FENCE();
#  endif
    _ReadWriteBarrier();
    return v;
}
#else
static inline uint64_t ld_acq(const volatile uint64_t *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
static inline uint64_t ld_rlx(const volatile uint64_t *p) { return __atomic_load_n(p, __ATOMIC_RELAXED); }
static inline void st_rel(volatile uint64_t *p, uint64_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
static inline void st_rlx(volatile uint64_t *p, uint64_t v) { __atomic_store_n(p, v, __ATOMIC_RELAXED); }
static inline void fence_acq(void) { __atomic_thread_fence(__ATOMIC_ACQUIRE); }
static inline void fence_rel(void) { __atomic_thread_fence(__ATOMIC_RELEASE); }
static inline int cas32(volatile uint32_t *p, uint32_t expect, uint32_t desired) {
    return __atomic_compare_exchange_n(p, &expect, desired, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}
static inline void add64(volatile uint64_t *p, uint64_t v) { __atomic_fetch_add(p, v, __ATOMIC_RELAXED); }
static inline void st32_rel(volatile uint32_t *p, uint32_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
static inline uint32_t ld32_acq(const volatile uint32_t *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
#endif

#define RING_MAGIC   0x31474E49524B5441ULL      /* "ATKRING1" little-endian */
#define RING_VERSION 1u
#define HDR_BYTES    256u
#define CONS_BYTES   64u
#define SLOT_HDR     64u

typedef struct {
    uint64_t magic;
    uint32_t version, header_bytes;
    uint64_t block_bytes, nblocks, slot_bytes, total_bytes;
    uint32_t max_consumers, pad0;
    uint64_t pad1;
    volatile uint64_t write_seq;     /* offset 64 */
    volatile uint32_t claimed;
    uint32_t pad2;
    uint64_t pad3[6];
    volatile uint64_t torn;          /* offset 128 */
    uint64_t pad4[15];
} ring_hdr;                          /* 256 bytes */

typedef struct {
    volatile uint32_t active;
    uint32_t pad;
    volatile uint64_t cursor, lost, nread, peek;   /* peek: held seq + 1, 0 = none */
    uint64_t pad2[3];
} ring_cons;                         /* 64 bytes */

typedef struct {
    volatile uint64_t lock;
    atkdsp_ring_meta meta;           /* 56 bytes */
} ring_slot;                         /* 64 bytes */

/* The layout is the contract with other processes: hold it. */
typedef char ring_hdr_is_256[(sizeof(ring_hdr) == HDR_BYTES) ? 1 : -1];
typedef char ring_cons_is_64[(sizeof(ring_cons) == CONS_BYTES) ? 1 : -1];
typedef char ring_slot_is_64[(sizeof(ring_slot) == SLOT_HDR) ? 1 : -1];

static ring_hdr *H(const atkdsp_ring *r) { return (ring_hdr *)(void *)r; }
static ring_cons *CONS(const atkdsp_ring *r, int i) {
    return (ring_cons *)(void *)((char *)(void *)r + HDR_BYTES + (size_t)i * CONS_BYTES);
}
static ring_slot *SLOT(const atkdsp_ring *r, uint64_t seq) {
    const ring_hdr *h = H(r);
    const size_t base = HDR_BYTES + (size_t)h->max_consumers * CONS_BYTES;
    return (ring_slot *)(void *)((char *)(void *)r + base
                                 + (size_t)(seq % h->nblocks) * (size_t)h->slot_bytes);
}
static unsigned char *DATA(const atkdsp_ring *r, uint64_t seq) {
    return (unsigned char *)SLOT(r, seq) + SLOT_HDR;
}

static size_t r64(size_t v) { return (v + 63) / 64 * 64; }

size_t atkdsp_ring_bytes(size_t block_bytes, size_t nblocks, int max_consumers) {
    if (block_bytes < 1 || nblocks < 2 || max_consumers < 1
        || max_consumers > ATKDSP_RING_MAX_CONSUMERS) return 0;
    const size_t slot = SLOT_HDR + r64(block_bytes);
    if (nblocks > (SIZE_MAX - 8192) / slot) return 0;
    return HDR_BYTES + CONS_BYTES * (size_t)max_consumers + nblocks * slot;
}

atkdsp_ring *atkdsp_ring_init(void *mem, size_t mem_bytes, size_t block_bytes, size_t nblocks,
                              int max_consumers) {
    const size_t need = atkdsp_ring_bytes(block_bytes, nblocks, max_consumers);
    if (!mem || !need || mem_bytes < need || ((uintptr_t)mem & 7u)) return NULL;
    memset(mem, 0, HDR_BYTES + CONS_BYTES * (size_t)max_consumers);
    ring_hdr *h = (ring_hdr *)mem;
    h->version = RING_VERSION;
    h->header_bytes = HDR_BYTES;
    h->block_bytes = r64(block_bytes);
    h->nblocks = nblocks;
    h->slot_bytes = SLOT_HDR + r64(block_bytes);
    h->total_bytes = need;
    h->max_consumers = (uint32_t)max_consumers;
    atkdsp_ring *r = (atkdsp_ring *)mem;
    for (size_t i = 0; i < nblocks; ++i) {
        ring_slot *s = SLOT(r, i);
        memset(s, 0, SLOT_HDR);
    }
    fence_rel();
    st_rel((volatile uint64_t *)&h->magic, RING_MAGIC);   /* last: attach checks it */
    return r;
}

atkdsp_ring *atkdsp_ring_attach(void *mem, size_t mem_bytes) {
    if (!mem || mem_bytes < HDR_BYTES || ((uintptr_t)mem & 7u)) return NULL;
    ring_hdr *h = (ring_hdr *)mem;
    /* every size is checked before anything past the header is touched: a
     * mapping that is not a whole ring of this version is refused */
    if (ld_acq((volatile uint64_t *)&h->magic) != RING_MAGIC || h->version != RING_VERSION
        || h->header_bytes != HDR_BYTES || h->max_consumers < 1
        || h->max_consumers > ATKDSP_RING_MAX_CONSUMERS || h->nblocks < 2
        || h->block_bytes < 64 || (h->block_bytes & 63u) != 0
        || h->slot_bytes != SLOT_HDR + h->block_bytes || h->total_bytes == 0
        || h->total_bytes > mem_bytes
        || h->total_bytes != atkdsp_ring_bytes((size_t)h->block_bytes, (size_t)h->nblocks,
                                               (int)h->max_consumers))
        return NULL;
    return (atkdsp_ring *)mem;
}

void *atkdsp_ring_claim(atkdsp_ring *r) {
    if (!r) return NULL;
    ring_hdr *h = H(r);
    const uint64_t s = ld_rlx(&h->write_seq);
    ring_slot *sl = SLOT(r, s);
    st_rlx(&sl->lock, 2 * s + 1);
    fence_rel();                      /* the odd lock is seen before any data store */
    h->claimed = 1;
    return DATA(r, s);
}

int atkdsp_ring_commit(atkdsp_ring *r, const atkdsp_ring_meta *m) {
    if (!r || !m) return ATKDSP_E_ARG;
    ring_hdr *h = H(r);
    if (!h->claimed || m->nbytes < 1 || m->nbytes > h->block_bytes) return ATKDSP_E_ARG;
    const uint64_t s = ld_rlx(&h->write_seq);
    ring_slot *sl = SLOT(r, s);
    sl->meta = *m;
    sl->meta.seq = s;
    st_rel(&sl->lock, 2 * s + 2);
    st_rel(&h->write_seq, s + 1);
    h->claimed = 0;
    return ATKDSP_OK;
}

int atkdsp_ring_open(atkdsp_ring *r, int from_oldest) {
    if (!r) return ATKDSP_E_ARG;
    ring_hdr *h = H(r);
    for (int i = 0; i < (int)h->max_consumers; ++i) {
        ring_cons *c = CONS(r, i);
        if (cas32(&c->active, 0, 1)) {
            const uint64_t w = ld_acq(&h->write_seq);
            const uint64_t oldest = w > h->nblocks ? w - h->nblocks : 0;
            st_rlx(&c->lost, 0);
            st_rlx(&c->nread, 0);
            st_rlx(&c->peek, 0);
            st_rel(&c->cursor, from_oldest ? oldest : w);
            return i;
        }
    }
    return ATKDSP_E_CAP;
}

int atkdsp_ring_close(atkdsp_ring *r, int id) {
    if (!r || id < 0 || id >= (int)H(r)->max_consumers) return ATKDSP_E_ARG;
    st32_rel(&CONS(r, id)->active, 0);
    return ATKDSP_OK;
}

static ring_cons *consumer(atkdsp_ring *r, int id) {
    if (!r || id < 0 || id >= (int)H(r)->max_consumers) return NULL;
    ring_cons *c = CONS(r, id);
    return ld32_acq(&c->active) ? c : NULL;
}

/* Move c's cursor past what is gone. Returns 1 and the seq and the lock value
 * of the next committed block, 0 when there is nothing new; *lost counts the
 * blocks skipped. */
static int next_block(atkdsp_ring *r, ring_cons *c, uint64_t *seq, uint64_t *lockv,
                      uint64_t *lost) {
    ring_hdr *h = H(r);
    for (;;) {
        uint64_t s = ld_rlx(&c->cursor);
        const uint64_t w = ld_acq(&h->write_seq);
        if (s >= w) return 0;
        if (w - s > h->nblocks) {
            *lost += w - h->nblocks - s;
            s = w - h->nblocks;
            st_rel(&c->cursor, s);
        }
        const uint64_t v = ld_acq(&SLOT(r, s)->lock);
        if (v != 2 * s + 2) {
            if (v > 2 * s + 2) {                 /* lapped, or being rewritten */
                *lost += 1;
                st_rel(&c->cursor, s + 1);
                continue;
            }
            return 0;
        }
        *seq = s;
        *lockv = v;
        return 1;
    }
}

ptrdiff_t atkdsp_ring_read(atkdsp_ring *r, int id, void *out, size_t cap, atkdsp_ring_meta *m,
                           uint64_t *lost) {
    ring_cons *c = consumer(r, id);
    if (!c || (!out && cap)) return ATKDSP_E_ARG;
    uint64_t lostn = 0;
    for (;;) {
        uint64_t s, v;
        if (!next_block(r, c, &s, &v, &lostn)) {
            if (lostn) st_rel(&c->lost, ld_rlx(&c->lost) + lostn);
            if (lost) *lost = lostn;
            return 0;
        }
        ring_slot *sl = SLOT(r, s);
        atkdsp_ring_meta mm = sl->meta;
        const size_t nb = mm.nbytes;
        if (nb > cap) {
            fence_acq();
            if (ld_rlx(&sl->lock) == v) {        /* a real block, just too big: keep it */
                if (lostn) st_rel(&c->lost, ld_rlx(&c->lost) + lostn);
                if (lost) *lost = lostn;
                return ATKDSP_E_CAP;
            }
            add64(&H(r)->torn, 1);
            lostn += 1;
            st_rel(&c->cursor, s + 1);
            continue;
        }
        memcpy(out, DATA(r, s), nb);
        fence_acq();
        if (ld_rlx(&sl->lock) != v) {            /* overwritten while we copied */
            add64(&H(r)->torn, 1);
            lostn += 1;
            st_rel(&c->cursor, s + 1);
            continue;
        }
        st_rel(&c->cursor, s + 1);
        st_rel(&c->nread, ld_rlx(&c->nread) + 1);
        if (lostn) st_rel(&c->lost, ld_rlx(&c->lost) + lostn);
        if (lost) *lost = lostn;
        if (m) *m = mm;
        return (ptrdiff_t)nb;
    }
}

const void *atkdsp_ring_peek(atkdsp_ring *r, int id, atkdsp_ring_meta *m, uint64_t *lost) {
    ring_cons *c = consumer(r, id);
    if (!c) return NULL;
    uint64_t lostn = 0;
    for (;;) {
        uint64_t s, v;
        if (!next_block(r, c, &s, &v, &lostn)) {
            if (lostn) st_rel(&c->lost, ld_rlx(&c->lost) + lostn);
            if (lost) *lost = lostn;
            return NULL;
        }
        ring_slot *sl = SLOT(r, s);
        atkdsp_ring_meta mm = sl->meta;
        fence_acq();
        if (ld_rlx(&sl->lock) != v) {
            add64(&H(r)->torn, 1);
            lostn += 1;
            st_rel(&c->cursor, s + 1);
            continue;
        }
        st_rel(&c->peek, s + 1);
        if (lostn) st_rel(&c->lost, ld_rlx(&c->lost) + lostn);
        if (lost) *lost = lostn;
        if (m) *m = mm;
        return DATA(r, s);
    }
}

int atkdsp_ring_release(atkdsp_ring *r, int id) {
    ring_cons *c = consumer(r, id);
    if (!c) return ATKDSP_E_ARG;
    const uint64_t held = ld_rlx(&c->peek);
    if (!held) return ATKDSP_E_ARG;
    const uint64_t s = held - 1;
    st_rlx(&c->peek, 0);
    fence_acq();
    const uint64_t v = ld_rlx(&SLOT(r, s)->lock);
    st_rel(&c->cursor, s + 1);
    if (v == 2 * s + 2) {
        st_rel(&c->nread, ld_rlx(&c->nread) + 1);
        return ATKDSP_OK;
    }
    add64(&H(r)->torn, 1);
    st_rel(&c->lost, ld_rlx(&c->lost) + 1);
    return ATKDSP_E_OVERRUN;
}

ptrdiff_t atkdsp_ring_copy_range(const atkdsp_ring *r, uint64_t seq0, uint64_t seq1, void *out,
                                 size_t cap, atkdsp_ring_meta *metas, size_t mcap) {
    if (!r || (!out && cap) || (!metas && mcap)) return ATKDSP_E_ARG;
    ring_hdr *h = H(r);
    const uint64_t w = ld_acq(&h->write_seq);
    const uint64_t oldest = w > h->nblocks ? w - h->nblocks : 0;
    uint64_t s = seq0 > oldest ? seq0 : oldest;
    const uint64_t end = seq1 < w ? seq1 : w;
    size_t used = 0, nm = 0;
    for (; s < end; ++s) {
        ring_slot *sl = SLOT(r, s);
        const uint64_t v = ld_acq(&sl->lock);
        if (v != 2 * s + 2) continue;            /* gone, or being rewritten */
        const atkdsp_ring_meta mm = sl->meta;
        const size_t nb = mm.nbytes;
        if (used + nb > cap || nm >= mcap) break;
        memcpy((unsigned char *)out + used, DATA(r, s), nb);
        fence_acq();
        if (ld_rlx(&sl->lock) != v) { add64(&h->torn, 1); continue; }
        metas[nm++] = mm;
        used += nb;
    }
    return (ptrdiff_t)nm;
}

int atkdsp_ring_stats(const atkdsp_ring *r, atkdsp_ring_counters *st) {
    if (!r || !st) return ATKDSP_E_ARG;
    ring_hdr *h = H(r);
    memset(st, 0, sizeof *st);
    const uint64_t w = ld_acq(&h->write_seq);
    st->committed = w;
    st->oldest = w > h->nblocks ? w - h->nblocks : 0;
    st->block_bytes = h->block_bytes;
    st->nblocks = h->nblocks;
    st->total_bytes = h->total_bytes;
    st->max_consumers = h->max_consumers;
    st->torn = ld_rlx(&h->torn);
    for (int i = 0; i < (int)h->max_consumers; ++i) {
        ring_cons *c = CONS(r, i);
        if (!ld32_acq(&c->active)) continue;
        st->active++;
        st->active_mask |= 1u << i;
        st->lost[i] = ld_rlx(&c->lost);
        st->read[i] = ld_rlx(&c->nread);
        const uint64_t cur = ld_rlx(&c->cursor);
        st->lag[i] = w > cur ? w - cur : 0;
    }
    return ATKDSP_OK;
}

int atkdsp_ring_describe(const atkdsp_ring *r, char *buf, size_t cap) {
    if (!r || !buf || !cap) return ATKDSP_E_ARG;
    atkdsp_ring_counters st;
    atkdsp_ring_stats(r, &st);
    snprintf(buf, cap,
             "ring: %llu blocks x %llu bytes (%.1f MiB of data), %u of %u consumers, "
             "%llu committed, %llu torn reads",
             (unsigned long long)st.nblocks, (unsigned long long)st.block_bytes,
             (double)st.nblocks * (double)st.block_bytes / 1048576.0, st.active,
             st.max_consumers, (unsigned long long)st.committed, (unsigned long long)st.torn);
    return ATKDSP_OK;
}
