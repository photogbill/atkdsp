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
    bitrd r;
    int csg_present, i, d, last_mcc_valid = 0, last_mcc[3] = {0,0,0};
    if (!bits || !out || nbits < 0) return ATKDSP_E_ARG;
    memset(out, 0, sizeof *out);
    out->csg_id = -1;
    r.b = bits; r.n = nbits; r.pos = 0; r.bad = 0;

    if (br_u(&r, 1) != 0) return 0;          /* BCCH-DL-SCH type CHOICE -> c1 */
    if (br_u(&r, 1) != 1) return 0;          /* c1 CHOICE -> SIB1             */
    (void)br_u(&r, 1);                        /* SIB1 extension bit           */
    (void)br_u(&r, 1);                        /* p-Max present                */
    (void)br_u(&r, 1);                        /* tdd-Config present           */
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

    return r.bad ? 0 : 1;
}
