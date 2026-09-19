/* 11g. SystemInformation -> SIB2-5 network fingerprint (ASN.1 UPER).
 *
 * The SIB2+ messages carried in BCCH-DL-SCH SystemInformation: SIB2 (access
 * barring, PRACH config, uplink carrier freq/bandwidth, reference-signal
 * power), SIB3 (reselection), SIB4 (intra-frequency neighbour PCIs), SIB5
 * (inter-frequency carriers + neighbour PCIs). All BROADCAST and public — the
 * cell's own configuration and neighbour topology, nothing about a subscriber.
 *
 * Input is the decoded transport-block bits (one bit per byte, MSB first) of a
 * SystemInformation message. 36.331 R8 ASN.1, unaligned PER. The numpy twin is
 * atkdsp.reference.lte_si_decode / lte_si_encode; a round trip proves the codec
 * self-consistent. LAYOUT CAVEAT (see atkdsp.h 11g): the neighbour lists and
 * barring are high-confidence, the deep radioResourceConfig fields (PRACH, UL
 * freq) most need a real-capture check. */
#include "internal.h"

typedef struct { const signed char *b; int n, pos, bad; } bitrd;

static unsigned br_u(bitrd *r, int nb) {
    unsigned v = 0; int i;
    for (i = 0; i < nb; ++i) {
        if (r->pos >= r->n) { r->bad = 1; return 0; }
        v = (v << 1) | (unsigned)(r->b[r->pos++] & 1);
    }
    return v;
}
static int br_b(bitrd *r) { return (int)br_u(r, 1); }

static int nbits_for(int rng) {
    int n = 0, m;
    if (rng <= 1) return 0;
    m = rng - 1;
    while (m > 0) { ++n; m >>= 1; }
    return n;
}
/* constrained INTEGER (lb..ub) */
static int di(bitrd *r, int lb, int ub) { return (int)br_u(r, nbits_for(ub - lb + 1)) + lb; }
/* extensible ENUMERATED/CHOICE index; -1 for an extension value not modelled */
static int ext_idx(bitrd *r, int root) {
    if (br_b(r)) return -1;
    return (int)br_u(r, nbits_for(root));
}

/* ---- PhysCellIdRange { start (0..503), range ENUM(16) OPTIONAL } -------- */
static int dec_pcid_start(bitrd *r) {
    int has = br_b(r), start = di(r, 0, 503);
    if (has) (void)br_u(r, nbits_for(16));
    return start;
}

/* ---- SIB4 (intra-frequency neighbours) --------------------------------- */
static void dec_sib4(bitrd *r, atkdsp_lte_sib4 *s) {
    int has_n, has_b, has_csg, i, n;
    s->present = 1; s->n_neigh = 0; s->n_black = 0;
    br_b(r);                                   /* SIB4 ext bit */
    has_n = br_b(r); has_b = br_b(r); has_csg = br_b(r);
    if (has_n) {
        n = di(r, 1, 16);
        for (i = 0; i < n; ++i) {
            int pci, q;
            br_b(r);                           /* IntraFreqNeighCellInfo ext bit */
            pci = di(r, 0, 503);
            q = (int)br_u(r, nbits_for(31));
            if (s->n_neigh < ATKDSP_LTE_MAX_NEIGH) {
                s->neigh[s->n_neigh].pci = pci;
                s->neigh[s->n_neigh].q_off = q;
                ++s->n_neigh;
            }
        }
    }
    if (has_b) {
        n = di(r, 1, 16);
        for (i = 0; i < n; ++i) {
            int start = dec_pcid_start(r);
            if (s->n_black < ATKDSP_LTE_MAX_NEIGH) s->black_start[s->n_black++] = start;
        }
    }
    if (has_csg) (void)dec_pcid_start(r);
}

/* ---- SIB5 (inter-frequency carriers) ----------------------------------- */
static void dec_sib5(bitrd *r, atkdsp_lte_sib5 *s) {
    int nfreq, f;
    s->present = 1; s->n_freq = 0;
    br_b(r);                                   /* SIB5 ext bit */
    nfreq = di(r, 1, 8);
    for (f = 0; f < nfreq; ++f) {
        int has_pmax, has_sf, has_crp, has_qoff, has_neigh, has_black, dl, i, n;
        atkdsp_lte_interfreq *fr = (s->n_freq < ATKDSP_LTE_MAX_FREQ)
                                   ? &s->freq[s->n_freq] : NULL;
        br_b(r);                               /* InterFreqCarrierFreqInfo ext */
        has_pmax = br_b(r); has_sf = br_b(r); has_crp = br_b(r);
        has_qoff = br_b(r); has_neigh = br_b(r); has_black = br_b(r);
        dl = di(r, 0, 65535);                   /* dl-CarrierFreq */
        (void)di(r, -70, -22);                  /* q-RxLevMin */
        if (has_pmax) (void)di(r, -30, 33);
        (void)di(r, 0, 7);                      /* t-ReselectionEUTRA */
        if (has_sf) { (void)br_u(r, 2); (void)br_u(r, 2); }
        (void)di(r, 0, 31); (void)di(r, 0, 31); /* threshX-High/Low */
        (void)br_u(r, nbits_for(6));            /* allowedMeasBandwidth */
        br_b(r);                                /* presenceAntennaPort1 */
        if (has_crp) (void)di(r, 0, 7);
        (void)br_u(r, 2);                       /* neighCellConfig */
        if (has_qoff) (void)br_u(r, nbits_for(31));
        if (fr) { fr->dl_earfcn = dl; fr->n_neigh = 0; }
        if (has_neigh) {
            n = di(r, 1, 16);
            for (i = 0; i < n; ++i) {
                int pci = di(r, 0, 503);
                (void)br_u(r, nbits_for(31));
                if (fr && fr->n_neigh < ATKDSP_LTE_MAX_NEIGH)
                    fr->neigh_pci[fr->n_neigh++] = pci;
            }
        }
        if (has_black) {
            n = di(r, 1, 16);
            for (i = 0; i < n; ++i) (void)dec_pcid_start(r);
        }
        if (fr) ++s->n_freq;
    }
}

/* ---- SIB3 (cell reselection) ------------------------------------------- */
static void dec_sib3(bitrd *r, atkdsp_lte_sib3 *s) {
    int has_speed, has_snon, has_pmax, has_sintra, has_amb, has_sf;
    s->present = 1;
    br_b(r);                                   /* SIB3 ext bit */
    has_speed = br_b(r); (void)br_u(r, nbits_for(16));   /* speedState opt, q-Hyst */
    if (has_speed) {                            /* MobilityStateParameters + q-HystSF */
        (void)br_u(r, 3); (void)br_u(r, 3); (void)di(r, 1, 16); (void)di(r, 1, 16);
        (void)br_u(r, 2); (void)br_u(r, 2);
    }
    has_snon = br_b(r);
    if (has_snon) (void)di(r, 0, 31);          /* s-NonIntraSearch */
    (void)di(r, 0, 31);                         /* threshServingLow */
    s->resel_priority = di(r, 0, 7);
    has_pmax = br_b(r); has_sintra = br_b(r); has_amb = br_b(r); has_sf = br_b(r);
    s->q_rxlevmin = di(r, -70, -22);
    if (has_pmax) (void)di(r, -30, 33);
    s->s_intra_search = has_sintra ? di(r, 0, 31) : -1;
    if (has_amb) (void)br_u(r, nbits_for(6));
    br_b(r); (void)br_u(r, 2); (void)di(r, 0, 7);   /* presAnt1, neighCellCfg, t-Resel */
    if (has_sf) { (void)br_u(r, 2); (void)br_u(r, 2); }
}

/* ---- SIB2 (barring + radioResourceConfigCommon + freqInfo) -------------- */
static void dec_acbc(bitrd *r) {               /* AC-BarringConfig */
    (void)br_u(r, nbits_for(16)); (void)br_u(r, nbits_for(8)); (void)br_u(r, 5);
}
static void dec_rrc(bitrd *r, atkdsp_lte_sib2 *s) {
    int has_grpA;
    br_b(r);                                   /* RRC-CommonSIB ext */
    br_b(r);                                   /* RACH-ConfigCommon ext */
    has_grpA = br_b(r); (void)br_u(r, nbits_for(16));   /* numberOfRA-Preambles */
    if (has_grpA) { (void)br_u(r, nbits_for(15)); (void)br_u(r, nbits_for(4)); (void)br_u(r, nbits_for(8)); }
    (void)br_u(r, nbits_for(4)); (void)br_u(r, nbits_for(16));      /* powerRamping */
    (void)br_u(r, nbits_for(11)); (void)br_u(r, nbits_for(8)); (void)br_u(r, nbits_for(8)); /* ra-Sup */
    (void)di(r, 1, 8);                          /* maxHARQ-Msg3Tx */
    br_b(r); (void)br_u(r, nbits_for(4));       /* bcch-Config */
    br_b(r); (void)br_u(r, nbits_for(4)); (void)br_u(r, nbits_for(8));  /* pcch-Config */
    s->prach_root = di(r, 0, 837);
    s->prach_config_index = di(r, 0, 63);
    s->prach_high_speed = br_b(r);
    s->prach_zcc = di(r, 0, 15);
    s->prach_freq_offset = di(r, 0, 94);
    s->ref_sig_power = di(r, -60, 50);
    (void)di(r, 0, 3);                          /* p-b */
    (void)di(r, 1, 4); (void)br_u(r, 1); (void)di(r, 0, 98); br_b(r);   /* pusch basic */
    br_b(r); (void)di(r, 0, 29); br_b(r); (void)di(r, 0, 7);            /* ul-RS */
    (void)br_u(r, nbits_for(3)); (void)di(r, 0, 98); (void)di(r, 0, 7); (void)di(r, 0, 2047); /* pucch */
    if (br_b(r)) {                              /* soundingRS CHOICE: 1 = setup */
        br_b(r);                                /* srs-MaxUpPts OPTIONAL */
        (void)br_u(r, nbits_for(8)); (void)br_u(r, nbits_for(16)); br_b(r);
    }
    (void)di(r, -126, 24); (void)br_u(r, nbits_for(8)); (void)di(r, -127, -96);  /* uplinkPowerControl */
    (void)br_u(r, nbits_for(3)); (void)br_u(r, nbits_for(3)); (void)br_u(r, nbits_for(4));
    (void)br_u(r, nbits_for(3)); (void)br_u(r, nbits_for(3)); (void)di(r, -1, 6);
    (void)br_u(r, nbits_for(2));                /* ul-CyclicPrefixLength */
}
static void dec_uetimers(bitrd *r) {
    static const int C[6] = {8, 8, 7, 8, 7, 8};
    int i;
    br_b(r);
    for (i = 0; i < 6; ++i) (void)br_u(r, nbits_for(C[i]));
}
static void dec_mbsfn(bitrd *r) {
    int n = di(r, 1, 8), i;
    for (i = 0; i < n; ++i) {
        (void)br_u(r, nbits_for(6)); (void)di(r, 0, 7);
        if (br_b(r) == 0) (void)br_u(r, 6); else (void)br_u(r, 24);
    }
}
static const int UL_BW_RB[6] = {6, 15, 25, 50, 75, 100};

static void dec_sib2(bitrd *r, atkdsp_lte_sib2 *s) {
    int has_ac, has_mbsfn, has_ulf, has_ulbw, bw;
    s->present = 1; s->ul_earfcn = -1; s->ul_bandwidth_rb = 0;
    br_b(r);                                    /* SIB2 ext bit */
    has_ac = br_b(r); has_mbsfn = br_b(r);
    if (has_ac) {
        int has_sig, has_dat;
        s->barring = 1;
        has_sig = br_b(r); has_dat = br_b(r);
        s->barring_emergency = br_b(r);
        if (has_sig) dec_acbc(r);
        if (has_dat) dec_acbc(r);
    }
    dec_rrc(r, s);
    dec_uetimers(r);
    has_ulf = br_b(r); has_ulbw = br_b(r);
    if (has_ulf) s->ul_earfcn = di(r, 0, 65535);
    if (has_ulbw) { bw = (int)br_u(r, nbits_for(6)); s->ul_bandwidth_rb = (bw < 6) ? UL_BW_RB[bw] : 0; }
    (void)di(r, 1, 32);                         /* additionalSpectrumEmission */
    if (has_mbsfn) dec_mbsfn(r);
    s->time_align_timer = (int)br_u(r, nbits_for(8));
}

/* ---- SystemInformation container --------------------------------------- */
#define SIB_CHOICE_ROOT 10

int atkdsp_lte_si_parse(const signed char *bits, int nbits, atkdsp_lte_si *out) {
    bitrd r;
    int n, i;
    if (!bits || !out || nbits < 0) return ATKDSP_E_ARG;
    memset(out, 0, sizeof *out);
    r.b = bits; r.n = nbits; r.pos = 0; r.bad = 0;
    if (br_b(&r) != 0) return 0;                /* BCCH-DL-SCH CHOICE -> c1        */
    if (br_b(&r) != 0) return 0;                /* c1 CHOICE -> systemInformation  */
    if (br_b(&r) != 0) return 0;                /* criticalExtensions -> r8        */
    br_b(&r);                                   /* nonCriticalExtension present    */
    n = di(&r, 1, 32);                          /* sib-TypeAndInfo SIZE(1..32)     */
    for (i = 0; i < n; ++i) {
        int idx = ext_idx(&r, SIB_CHOICE_ROOT);
        int st;
        if (idx < 0) break;                     /* an extension SIB not modelled   */
        st = idx + 2;
        if (st == 2)      dec_sib2(&r, &out->sib2);
        else if (st == 3) dec_sib3(&r, &out->sib3);
        else if (st == 4) dec_sib4(&r, &out->sib4);
        else if (st == 5) dec_sib5(&r, &out->sib5);
        else break;                             /* sib6..sib11 not modelled here   */
        if (r.bad) return 0;
    }
    return r.bad ? 0 : 1;
}
