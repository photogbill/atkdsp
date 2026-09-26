/* 11e. SIB1 -> tower identity (ASN.1 UPER, the identity-bearing part).
 *
 * SystemInformationBlockType1 is the cell's OWN public broadcast: the operator
 * (PLMN = MCC/MNC), the trackingAreaCode (TAC) and the 28-bit E-UTRAN cell
 * identity (ECI). This decodes exactly that part of BCCH-DL-SCH-Message and
 * STOPS after cellAccessRelatedInfo — nothing about any subscriber is in SIB1,
 * and none is read. It is the network's own name for itself, the answer a
 * search-and-rescue survey and a fake-tower check actually need.
 *
 * Input is the decoded transport-block bits (from atkdsp_lte_turbo_decode, one
 * bit per byte, MSB first), i.e. the BCCH-DL-SCH-Message content. UNALIGNED
 * PER (36.331 R8 baseline structure). The numpy twin atkdsp.reference.
 * lte_sib1_decode / lte_sib1_encode is the readable statement and a matching
 * encoder, so the decoder is validated by a round trip before a real capture. */
#include "internal.h"

/* ---- UPER bit reader over a one-bit-per-byte array --------------------- */
typedef struct { const signed char *b; int n, pos, bad; } bitrd;

static unsigned br_u(bitrd *r, int nbits) {
    unsigned v = 0; int i;
    for (i = 0; i < nbits; ++i) {
        if (r->pos >= r->n) { r->bad = 1; return 0; }
        v = (v << 1) | (unsigned)(r->b[r->pos++] & 1);
    }
    return v;
}

/* bits to hold a constrained value with `rng` possibilities (ceil log2). */
static int nbits_for(int rng) {
    int n = 0, m;
    if (rng <= 1) return 0;
    m = rng - 1;
    while (m > 0) { ++n; m >>= 1; }
    return n;
}

int atkdsp_lte_sib1_parse(const signed char *bits, int nbits,
                          atkdsp_lte_sib1 *out) {
    static const int SI_PERIODICITY_RF[7] = {8, 16, 32, 64, 128, 256, 512};
    static const int SI_WINDOW_MS[7] = {1, 2, 5, 10, 15, 20, 40};
    bitrd r;
    int csg_present, i, d, last_mcc_valid = 0, last_mcc[3] = {0,0,0};
    int has_pmax, has_tdd, save, nsi, si_win;
    if (!bits || !out || nbits < 0) return ATKDSP_E_ARG;
    memset(out, 0, sizeof *out);
    out->csg_id = -1;
    r.b = bits; r.n = nbits; r.pos = 0; r.bad = 0;

    if (br_u(&r, 1) != 0) return 0;          /* BCCH-DL-SCH type CHOICE -> c1 */
    if (br_u(&r, 1) != 1) return 0;          /* c1 CHOICE -> SIB1             */
    /* SystemInformationBlockType1 has NO extension marker (it grows through
     * nonCriticalExtension); reading one here put every later field one bit
     * late, and no real SIB1 ever parsed (Bill's B13 capture, 2026-09-26). */
    has_pmax = (int)br_u(&r, 1);              /* p-Max present                */
    has_tdd  = (int)br_u(&r, 1);              /* tdd-Config present           */
    (void)br_u(&r, 1);                        /* nonCriticalExtension present */
    csg_present = (int)br_u(&r, 1);           /* cellAccessRelatedInfo csg    */

    out->n_plmn = (int)br_u(&r, nbits_for(6)) + 1;   /* SIZE(1..6), 3 bits    */
    if (out->n_plmn < 1 || out->n_plmn > ATKDSP_LTE_MAX_PLMN) return 0;
    for (i = 0; i < out->n_plmn; ++i) {
        atkdsp_lte_plmn *p = &out->plmn[i];
        int mcc_present = (int)br_u(&r, 1);
        if (mcc_present) {
            for (d = 0; d < 3; ++d) p->mcc[d] = (int)br_u(&r, 4);
            last_mcc[0] = p->mcc[0]; last_mcc[1] = p->mcc[1]; last_mcc[2] = p->mcc[2];
            last_mcc_valid = 1;
        } else if (last_mcc_valid) {
            p->mcc[0] = last_mcc[0]; p->mcc[1] = last_mcc[1]; p->mcc[2] = last_mcc[2];
        } else {
            p->mcc[0] = -1; p->mcc[1] = -1; p->mcc[2] = -1;   /* none to inherit */
        }
        p->mnc_len = (int)br_u(&r, 1) + 2;                   /* SIZE(2..3)      */
        for (d = 0; d < p->mnc_len; ++d) p->mnc[d] = (int)br_u(&r, 4);
        p->reserved = (br_u(&r, 1) == 0);      /* ENUMERATED{reserved,notReserved} */
    }
    out->tac     = (int)br_u(&r, 16);          /* trackingAreaCode BIT STRING(16) */
    out->cell_id = br_u(&r, 28);               /* cellIdentity BIT STRING(28)     */
    out->cell_barred = (int)br_u(&r, 1);
    (void)br_u(&r, 1);                          /* intraFreqReselection            */
    (void)br_u(&r, 1);                          /* csg-Indication BOOLEAN          */
    if (csg_present) out->csg_id = (int)br_u(&r, 27);

    if (r.bad) return 0;                        /* identity gates the return */

    /* -- scheduling (best-effort; identity above is what gates the return) -- */
    save = r.pos;
    {
        int has_qoff = (int)br_u(&r, 1);        /* q-RxLevMinOffset present   */
        (void)br_u(&r, nbits_for(49));          /* q-RxLevMin (-70..-22)      */
        if (has_qoff) (void)br_u(&r, 3);        /* q-RxLevMinOffset (1..8)    */
    }
    if (has_pmax) (void)br_u(&r, nbits_for(64));/* p-Max                      */
    out->freq_band = (int)br_u(&r, nbits_for(64)) + 1;   /* freqBandIndicator */
    nsi = (int)br_u(&r, nbits_for(32)) + 1;     /* schedulingInfoList 1..32   */
    out->n_sched = 0;
    for (i = 0; i < nsi; ++i) {
        int per = (int)br_u(&r, nbits_for(7));  /* si-Periodicity (7)         */
        int nmap = (int)br_u(&r, nbits_for(32)); /* sib-MappingInfo SIZE(0..31)*/
        int j;
        atkdsp_lte_sched *sc = (out->n_sched < ATKDSP_LTE_MAX_SI)
                               ? &out->sched[out->n_sched] : NULL;
        if (sc) {
            sc->periodicity_rf = (per < 7) ? SI_PERIODICITY_RF[per] : 0;
            sc->n_sibs = 0;
        }
        for (j = 0; j < nmap; ++j) {
            /* SIB-Type: 16 root values (sibType3..18), then an extension:
             * sibType19, 20, 21, 24, 25, 26 as a normally-small number */
            static const int EXT[6] = {19, 20, 21, 24, 25, 26};
            int sib;
            if (br_u(&r, 1)) {
                int big = (int)br_u(&r, 1), e = (int)br_u(&r, 6);
                sib = (!big && e < 6) ? EXT[e] : 0;
            } else {
                sib = (int)br_u(&r, 4) + 3;
            }
            if (sc && sc->n_sibs < ATKDSP_LTE_MAX_SIBMAP)
                sc->sibs[sc->n_sibs++] = sib;
        }
        if (sc) ++out->n_sched;
    }
    if (has_tdd) { (void)br_u(&r, nbits_for(7)); (void)br_u(&r, nbits_for(9)); }
    si_win = (int)br_u(&r, nbits_for(7));       /* si-WindowLength (7)        */
    if (r.bad) {                                /* identity-only stream       */
        r.pos = save;
        out->freq_band = 0; out->n_sched = 0; out->si_window_ms = 0;
    } else {
        out->si_window_ms = (si_win < 7) ? SI_WINDOW_MS[si_win] : 0;
    }
    return 1;
}
