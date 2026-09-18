"""LTE turbo codec — the critical PDSCH primitive. Prototype / spec.

Constituent code: 8-state RSC, g0(D)=1+D^2+D^3 (feedback), g1(D)=1+D+D^3
(parity), rate 1/3 (systematic + 2 parity). Trellis-terminated with 3 tail
bits per constituent. Internal interleaver: QPP, Pi(i)=(f1 i + f2 i^2) mod K.
Decoder: iterative max-log-MAP (BCJR) between the two constituents.

3GPP TS 36.212 5.1.3.2 (encoder), 5.1.3.2.3 (QPP table). This validates the
decoder by an encode->decode round trip across SNR before any C.
"""
import numpy as np

# ---- constituent RSC trellis ---------------------------------------------
# state = (r1,r2,r3), r1 newest; index = r1*4 + r2*2 + r3.
def _rsc_step(state, u):
    r1 = (state >> 2) & 1
    r2 = (state >> 1) & 1
    r3 = state & 1
    a = u ^ r2 ^ r3                 # feedback (g0 = 1 + D^2 + D^3)
    z = a ^ r1 ^ r3                 # parity   (g1 = 1 + D + D^3)
    ns = (a << 2) | (r1 << 1) | r2
    return ns, z

# next-state, parity, and the tail input that drives the state toward zero
_NS = np.zeros((8, 2), dtype=np.int32)
_PAR = np.zeros((8, 2), dtype=np.int32)
_TAIL_U = np.zeros(8, dtype=np.int32)      # u that makes a=0 (tail)
for _s in range(8):
    for _u in (0, 1):
        _NS[_s, _u], _PAR[_s, _u] = _rsc_step(_s, _u)
    r2 = (_s >> 1) & 1; r3 = _s & 1
    _TAIL_U[_s] = r2 ^ r3                   # a = u ^ r2 ^ r3 = 0


def rsc_encode(u):
    """Return (systematic incl. tail, parity incl. tail, final tail sys/par).
    Returns arrays of length K+3 for systematic and parity."""
    K = len(u)
    s = 0
    sysb = np.empty(K + 3, dtype=np.int8)
    par = np.empty(K + 3, dtype=np.int8)
    for k in range(K):
        sysb[k] = u[k]
        par[k] = _PAR[s, u[k]]
        s = _NS[s, u[k]]
    for j in range(3):                      # 3 tail bits back to state 0
        ut = _TAIL_U[s]
        sysb[K + j] = ut
        par[K + j] = _PAR[s, ut]
        s = _NS[s, ut]
    return sysb, par


# ---- QPP interleaver (36.212 Table 5.1.3-3, subset) ----------------------
_QPP = {40: (3, 10), 48: (7, 12), 64: (7, 16), 128: (15, 32), 256: (15, 32),
        512: (31, 64), 1024: (31, 64), 2048: (31, 64), 6144: (263, 480),
        96: (11, 24), 192: (23, 48), 384: (25, 96), 768: (23, 48)}


def qpp(K):
    f1, f2 = _QPP[K]
    i = np.arange(K)
    return (f1 * i + f2 * i * i) % K


# ---- turbo encode --------------------------------------------------------
def turbo_encode(u):
    """u (K bits) -> the transmitted pieces (rate 1/3, 12 tail bits):
       sys      : K systematic bits (transmitted once)
       par1     : K+3 parity from constituent 1 (last 3 are its tail parity)
       par2     : K+3 parity from constituent 2 (interleaved input)
       tail1_sys: 3 tail systematic bits of constituent 1
       tail2_sys: 3 tail systematic bits of constituent 2
    Total = K + (K+3) + (K+3) + 3 + 3 = 3K+12."""
    K = len(u)
    sys1, par1 = rsc_encode(u)
    pi = qpp(K)
    sys2, par2 = rsc_encode(u[pi])
    return {"sys": sys1[:K], "par1": par1, "par2": par2,
            "tail1_sys": sys1[K:K + 3], "tail2_sys": sys2[K:K + 3]}


# ---- max-log-MAP BCJR for one constituent --------------------------------
_NEG = -1e18
def _bcjr(sys_llr, par_llr, apri, terminated=True):
    """Return extrinsic LLR for the K info bits. Arrays length K+3 for
    sys/par (incl. tail); apri length K."""
    n = len(sys_llr)             # K+3
    K = n - 3
    # branch metric per (state,u): 0.5*(sys_llr*(1-2u_sys)+par_llr*(1-2*par)) + apri
    # forward alpha
    alpha = np.full((n + 1, 8), _NEG)
    alpha[0, 0] = 0.0
    for k in range(n):
        ap = apri[k] if k < K else 0.0
        for s in range(8):
            if alpha[k, s] <= _NEG:
                continue
            for u in (0, 1):
                ns = _NS[s, u]
                z = _PAR[s, u]
                g = 0.5 * (sys_llr[k] * (1 - 2 * u) + par_llr[k] * (1 - 2 * z)) \
                    + (ap * (1 - 2 * u) * 0.5 if k < K else 0.0)
                v = alpha[k, s] + g
                if v > alpha[k + 1, ns]:
                    alpha[k + 1, ns] = v
    # backward beta
    beta = np.full((n + 1, 8), _NEG)
    if terminated:
        beta[n, 0] = 0.0
    else:
        beta[n, :] = 0.0
    for k in range(n - 1, -1, -1):
        ap = apri[k] if k < K else 0.0
        for s in range(8):
            best = _NEG
            for u in (0, 1):
                ns = _NS[s, u]
                z = _PAR[s, u]
                g = 0.5 * (sys_llr[k] * (1 - 2 * u) + par_llr[k] * (1 - 2 * z)) \
                    + (ap * (1 - 2 * u) * 0.5 if k < K else 0.0)
                v = beta[k + 1, ns] + g
                if v > best:
                    best = v
            beta[k, s] = best
    # LLR over the K info bits, using the FULL branch metric (incl. apriori)
    ext = np.zeros(K)
    for k in range(K):
        ap = apri[k]
        m1 = _NEG; m0 = _NEG
        for s in range(8):
            if alpha[k, s] <= _NEG:
                continue
            for u in (0, 1):
                ns = _NS[s, u]
                z = _PAR[s, u]
                g = 0.5 * (sys_llr[k] * (1 - 2 * u) + par_llr[k] * (1 - 2 * z)
                           + ap * (1 - 2 * u))
                v = alpha[k, s] + g + beta[k + 1, ns]
                if u == 1:
                    m1 = max(m1, v)
                else:
                    m0 = max(m0, v)
        llr = m0 - m1                 # a posteriori LLR: >0 favours 0
        ext[k] = llr - sys_llr[k] - ap          # extrinsic
    return ext


def turbo_decode(sys, par1, par2, tail1_sys, tail2_sys, K, iters=8):
    """Iterative max-log-MAP. All inputs are LLRs (>0 favours 0):
       sys       : K,   par1/par2 : K+3,   tail1_sys/tail2_sys : 3.
    Returns K hard bits."""
    pi = qpp(K)
    inv = np.argsort(pi)
    sys1 = np.concatenate([sys, tail1_sys])          # decoder 1 systematic
    sys2 = np.concatenate([sys[pi], tail2_sys])      # decoder 2 systematic
    apri = np.zeros(K)
    for _ in range(iters):
        e1 = _bcjr(sys1, par1, apri)
        e2 = _bcjr(sys2, par2, e1[pi])
        apri = e2[inv]                               # deinterleave to dec 1
    e1 = _bcjr(sys1, par1, apri)
    llr = sys + apri + e1
    return (llr < 0).astype(np.int8)


# ---- turbo rate matching (36.212 5.1.4.1) --------------------------------
# turbo sub-block column permutation (Table 5.1.4-1)
_TPERM = np.array([0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,
                   1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31])


def turbo_encode_d(u):
    """u (K bits) -> the three rate-matcher input streams d0,d1,d2, each of
    length K+4, with the tail bits laid out per 36.212 5.1.3.2.2."""
    K = len(u)
    sys1, par1 = rsc_encode(u)              # length K+3 (K info + 3 tail)
    pi = qpp(K)
    sys2, par2 = rsc_encode(u[pi])
    xt = sys1[K:K+3]; zt = par1[K:K+3]      # enc1 tail (systematic, parity)
    xpt = sys2[K:K+3]; zpt = par2[K:K+3]    # enc2 tail
    d0 = np.empty(K+4, np.int8); d1 = np.empty(K+4, np.int8); d2 = np.empty(K+4, np.int8)
    d0[:K] = u; d1[:K] = par1[:K]; d2[:K] = par2[:K]
    # tail columns (5.1.3.2.2)
    d0[K], d1[K], d2[K]       = xt[0], zt[0], xt[1]
    d0[K+1], d1[K+1], d2[K+1] = zt[1], xt[2], zt[2]
    d0[K+2], d1[K+2], d2[K+2] = xpt[0], zpt[0], xpt[1]
    d0[K+3], d1[K+3], d2[K+3] = zpt[1], xpt[2], zpt[2]
    return d0, d1, d2


def _turbo_subblock_maps(D):
    """Return (idx01, idx2): for each of the K_Pi output positions, the source
    index into the null-padded length-D stream (or -1 for a null). idx01 is
    for streams 0 and 1, idx2 for stream 2 (36.212 5.1.4.1.1)."""
    C = 32
    R = -(-D // C)
    KPi = R * C
    ND = KPi - D
    padded = np.array([-1]*ND + list(range(D)))     # row-major null-padded
    M = padded.reshape(R, C)
    # streams 0,1: permute columns, read column by column
    idx01 = M[:, _TPERM].reshape(-1, order='F')
    # stream 2: v2[k] = y[pi(k)], pi(k) = (P[k//R] + C*(k mod R) + 1) mod KPi
    k = np.arange(KPi)
    pi2 = (_TPERM[k // R] + C * (k % R) + 1) % KPi
    idx2 = padded[pi2]
    return idx01, idx2, R, KPi


def _k0(rv, R, Ncb):
    return R * (2 * (-(-Ncb // (8 * R))) * rv + 2)


def rate_match_turbo(d0, d1, d2, E, rv=0, Ncb=None):
    D = len(d0)
    idx01, idx2, R, KPi = _turbo_subblock_maps(D)
    v0 = np.array([-1 if i < 0 else int(d0[i]) for i in idx01], np.int8)
    v1 = np.array([-1 if i < 0 else int(d1[i]) for i in idx01], np.int8)
    v2 = np.array([-1 if i < 0 else int(d2[i]) for i in idx2], np.int8)
    w = np.empty(3 * KPi, np.int8)
    w[:KPi] = v0
    w[KPi::2] = v1
    w[KPi+1::2] = v2
    Kw = 3 * KPi
    if Ncb is None:
        Ncb = Kw
    j = _k0(rv, R, Ncb)
    out = np.empty(E, np.int8); k = 0
    while k < E:
        val = w[j % Ncb]
        j += 1
        if val >= 0:
            out[k] = val; k += 1
    return out


def rate_dematch_turbo(e_llr, D, E, rv=0, Ncb=None):
    idx01, idx2, R, KPi = _turbo_subblock_maps(D)
    Kw = 3 * KPi
    if Ncb is None:
        Ncb = Kw
    isnull = np.zeros(Ncb, bool)
    # mark nulls in the circular buffer
    v0n = idx01 < 0; v1n = idx01 < 0; v2n = idx2 < 0
    isnull[:KPi] = v0n
    isnull[KPi::2] = v1n
    isnull[KPi+1::2] = v2n
    w = np.zeros(Ncb)
    j = _k0(rv, R, Ncb); k = 0
    while k < E:
        pos = j % Ncb
        if not isnull[pos]:
            w[pos] += e_llr[k]; k += 1
        j += 1
    v0 = w[:KPi]; v1 = w[KPi::2][:KPi]; v2 = w[KPi+1::2][:KPi]
    d0 = np.zeros(D); d1 = np.zeros(D); d2 = np.zeros(D)
    for kk in range(KPi):
        if idx01[kk] >= 0:
            d0[idx01[kk]] += v0[kk]; d1[idx01[kk]] += v1[kk]
        if idx2[kk] >= 0:
            d2[idx2[kk]] += v2[kk]
    return d0, d1, d2


def turbo_decode_from_d(d0, d1, d2, K, iters=8):
    """Decode from the three de-rate-matched streams (each length K+4)."""
    sys = d0[:K]
    par1 = np.concatenate([d1[:K], [d1[K], d0[K+1], d2[K+1]]])       # z_K,z_{K+1},z_{K+2}
    par2 = np.concatenate([d2[:K], [d1[K+2], d0[K+3], d2[K+3]]])     # z'_K,z'_{K+1},z'_{K+2}
    tail1_sys = np.array([d0[K], d2[K], d1[K+1]])                     # x_K,x_{K+1},x_{K+2}
    tail2_sys = np.array([d0[K+2], d2[K+2], d1[K+3]])                 # x'_K,x'_{K+1},x'_{K+2}
    return turbo_decode(sys, par1, par2, tail1_sys, tail2_sys, K, iters)
