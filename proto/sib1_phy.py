"""SIB1 physical layer — full-bandwidth downlink grid, control and shared
channels. Downlink broadcast only.

Builds the resource grid for a whole subframe at the cell's numerology, OFDM
(de)modulates it, estimates the channel from CRS, and lays out / recovers
PCFICH, PDCCH and PDSCH. A faithful transmit path is included so the receive
path is validated by an encode->decode round trip before any C.

3GPP TS 36.211 (physical channels). Reuses the validated coding primitives in
pbch_proto (Gold, CRC-16, conv+Viterbi) and turbo_proto (turbo + rate match).
"""
import numpy as np
import pbch_proto as B
import turbo_proto as T

# N_RB -> N_FFT
_NFFT = {6: 128, 15: 256, 25: 512, 50: 1024, 75: 1536, 100: 2048}
SC_PER_RB = 12
SYMS_PER_SUBFRAME = 14          # normal CP, 2 slots x 7
SYMS_PER_SLOT = 7


def grid_params(n_rb):
    nfft = _NFFT[n_rb]
    n_sc = n_rb * SC_PER_RB
    cp_long = nfft * 160 // 2048
    cp_short = nfft * 144 // 2048
    return {"n_rb": n_rb, "nfft": nfft, "n_sc": n_sc,
            "cp_long": cp_long, "cp_short": cp_short}


def _sc_to_bin(k, n_sc, nfft):
    """Resource-grid subcarrier k=0..n_sc-1 (DC in the middle, skipped) to an
    FFT bin. k=0 is the lowest subcarrier."""
    half = n_sc // 2
    off = k - half                      # -half .. +half-1
    if off >= 0:
        off += 1                        # skip DC
    return off % nfft


def cp_len(sym_in_slot, gp):
    return gp["cp_long"] if sym_in_slot == 0 else gp["cp_short"]


def ofdm_modulate(grid, gp):
    """grid: complex [n_sc, 14] -> time samples for the subframe."""
    n_sc, nfft = gp["n_sc"], gp["nfft"]
    out = []
    for l in range(SYMS_PER_SUBFRAME):
        X = np.zeros(nfft, dtype=complex)
        for k in range(n_sc):
            X[_sc_to_bin(k, n_sc, nfft)] = grid[k, l]
        xt = np.fft.ifft(X) * nfft / np.sqrt(n_sc)
        cp = cp_len(l % SYMS_PER_SLOT, gp)
        out.append(np.concatenate([xt[-cp:], xt]))
    return np.concatenate(out)


def ofdm_demodulate(samples, gp):
    """time samples -> grid [n_sc, 14]."""
    n_sc, nfft = gp["n_sc"], gp["nfft"]
    grid = np.zeros((n_sc, SYMS_PER_SUBFRAME), dtype=complex)
    pos = 0
    for l in range(SYMS_PER_SUBFRAME):
        cp = cp_len(l % SYMS_PER_SLOT, gp)
        pos += cp
        xt = samples[pos:pos + nfft]
        pos += nfft
        X = np.fft.fft(xt) * np.sqrt(n_sc) / nfft
        for k in range(n_sc):
            grid[k, l] = X[_sc_to_bin(k, n_sc, nfft)]
    return grid


# ---- CRS (36.211 6.10.1), port 0, whole band -----------------------------
NRB_MAX = 110
_SQ = 1.0 / np.sqrt(2.0)


def crs_symbols_in_subframe():
    """OFDM symbols (0..13) that carry port-0 CRS, with their slot/l/v."""
    out = []
    for n_s in range(2):                 # two slots in the subframe
        for l in (0, 4):                 # port-0 CRS symbols within a slot
            v = 0 if l == 0 else 3
            sym = n_s * SYMS_PER_SLOT + l
            out.append((sym, n_s, l, v))
    return out


def crs_seq(n_id, n_s, l):
    ci = (1 << 10) * (7 * (n_s + 1) + l + 1) * (2 * n_id + 1) + 2 * n_id + 1
    c = B.gold(ci, 4 * NRB_MAX)
    return _SQ * (1 - 2 * c[0::2]) + 1j * _SQ * (1 - 2 * c[1::2])


def crs_re(n_id, n_rb, n_s, l, v):
    """(subcarrier k in 0..n_sc-1, reference value) for port-0 CRS."""
    n_sc = n_rb * SC_PER_RB
    vshift = n_id % 6
    r = crs_seq(n_id, n_s, l)
    m0 = NRB_MAX - n_rb                  # first sequence index used by this BW
    out = []
    for m in range(2 * n_rb):
        k = 6 * m + (v + vshift) % 6     # subcarrier within the band
        out.append((k, r[m0 + m]))
    return out


def crs_positions(n_id, n_rb):
    """Set of (k, sym) REs occupied by port-0 CRS in the subframe."""
    occ = {}
    for sym, n_s, l, v in crs_symbols_in_subframe():
        for k, val in crs_re(n_id, n_rb, n_s, l, v):
            occ[(k, sym)] = val
    return occ


# ---- CRS RE positions for N ports (to skip when mapping data) -------------
def crs_re_set(n_id, n_rb, n_ports):
    """All (k, sym) REs occupied by CRS for the given antenna-port count."""
    n_sc = n_rb * SC_PER_RB
    vshift = n_id % 6
    occ = set()
    ports = range(n_ports)
    for n_s in range(2):
        base = n_s * SYMS_PER_SLOT
        for port in ports:
            if port <= 1:
                syms = ((0, 0 if port == 0 else 3), (4, 3 if port == 0 else 0))
            else:                                   # ports 2,3 -> symbol 1
                syms = ((1, 0 if port == 2 else 3),)
            for l, v in syms:
                for m in range(2 * n_rb):
                    occ.add((6 * m + (v + vshift) % 6, base + l))
    return occ


# ---- PDSCH -----------------------------------------------------------------
SI_RNTI = 0xFFFF


def pdsch_re_list(n_id, n_rb, alloc_rbs, cfi, n_ports):
    """Ordered (k, sym) REs for PDSCH: the allocated RBs across symbols
    cfi..13, frequency-first then symbol, skipping CRS. (PSS/SSS exclusion for
    subframe 5's centre RBs is applied by the caller's allocation here; the
    round trip shares this list so it validates the data path.)"""
    crs = crs_re_set(n_id, n_rb, n_ports)
    res = []
    for sym in range(cfi, SYMS_PER_SUBFRAME):
        for rb in alloc_rbs:
            for sub in range(SC_PER_RB):
                k = rb * SC_PER_RB + sub
                if (k, sym) not in crs:
                    res.append((k, sym))
    return res


def pdsch_scramble(n_id, n_rnti, subframe, length):
    n_s = subframe * 2
    c_init = (n_rnti << 14) | (0 << 13) | ((n_s // 2) << 9) | n_id
    return B.gold(c_init, length)


def _qpsk(bits):
    b = np.asarray(bits, np.int8).reshape(-1, 2)
    return _SQ * ((1 - 2 * b[:, 0]) + 1j * (1 - 2 * b[:, 1]))


def pdsch_encode(tb_bits, n_id, n_rnti, subframe, E):
    """Transport block -> E scrambled QPSK-ready bits (before RE mapping)."""
    frame = np.concatenate([tb_bits, __import__("sib1_asn1").crc24a(tb_bits)])
    K = len(frame)
    d0, d1, d2 = T.turbo_encode_d(frame)
    e = T.rate_match_turbo(d0, d1, d2, E)
    c = pdsch_scramble(n_id, n_rnti, subframe, E)
    return (e ^ c).astype(np.int8), K


def pdsch_decode(llr_scr, n_id, n_rnti, subframe, K):
    """E scrambled-bit LLRs -> transport block bits, or None on CRC fail."""
    import sib1_asn1 as A
    E = len(llr_scr)
    c = pdsch_scramble(n_id, n_rnti, subframe, E)
    llr = (1 - 2 * c) * llr_scr
    d0, d1, d2 = T.rate_dematch_turbo(llr, K + 4, E)
    dec = T.turbo_decode_from_d(d0, d1, d2, K, iters=8)
    if A.crc24a_check(dec):
        return dec[:-24]
    return None


# ---- REGs, PCFICH, PDCCH (36.211 6.2.4, 6.7, 6.8) ------------------------
# NOTE: the REG numbering and PCFICH/PHICH positions below follow the spec as
# implemented here for a self-consistent transmit/receive round trip. Their
# EXACT geometry must be confirmed against a real capture before trusting a
# live decode — a round trip proves the coding/interleaver, not the spec's
# precise RE choice.
def _sym_regs(n_id, n_rb, l, n_ports):
    """REGs in one control symbol: list of REGs, each a list of 4 (k, sym).
    Assumes 2-port CRS in symbol 0 (and 4-port CRS in symbol 1 when 4 ports),
    which is how the control region is laid out."""
    vshift = n_id % 6
    n_sc = n_rb * SC_PER_RB
    has_crs = (l == 0) or (l == 1 and n_ports == 4)
    regs = []
    if has_crs:
        crs_res = {(6 * m + (vshift + o) % 6) for m in range(2 * n_rb)
                   for o in (0, 3)}
        for g in range(2 * n_rb):                 # groups of 6 subcarriers
            group = [g * 6 + i for i in range(6)]
            data = [k for k in group if k not in crs_res]
            regs.append([(k, l) for k in data[:4]])
    else:
        for g in range(3 * n_rb):                 # groups of 4 subcarriers
            regs.append([(g * 4 + i, l) for i in range(4)])
    return regs


def pcfich_reg_idx(n_id, n_rb):
    """The 4 PCFICH REG indices within symbol 0's REG list."""
    n_regs0 = 2 * n_rb
    k_bar = (n_id % (2 * n_rb))                    # in units of REG groups
    return [(k_bar + i * (n_regs0 // 4)) % n_regs0 for i in range(4)]


def _perm_regs(seq, n_id):
    """REG-level sub-block interleave (conv 32-col permutation) + cyclic shift
    by N_ID, dropping nulls. `seq` is the ordered PDCCH REG list."""
    P = B._PERM if hasattr(B, "_PERM") else None
    perm = np.array([1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31,
                     0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30])
    D = len(seq)
    C = 32
    R = -(-D // C)
    nd = R * C - D
    padded = [None] * nd + list(seq)
    M = [padded[r * C:(r + 1) * C] for r in range(R)]
    out = []
    for c in range(C):
        for r in range(R):
            out.append(M[r][perm[c]])
    out = [x for x in out if x is not None]
    # cyclic shift by N_ID
    sh = n_id % len(out)
    return out[sh:] + out[:sh]


def control_regs(n_id, n_rb, cfi, n_ports):
    """Ordered PDCCH REG list (control region minus PCFICH), interleaved."""
    all_regs = []
    for l in range(cfi):
        all_regs.append(_sym_regs(n_id, n_rb, l, n_ports))
    # PCFICH REGs (symbol 0) to exclude
    pc = set(pcfich_reg_idx(n_id, n_rb))
    seq = []
    # frequency-first ordering: for each REG position, walk symbols
    maxregs = max(len(r) for r in all_regs)
    for i in range(maxregs):
        for l in range(cfi):
            if i < len(all_regs[l]):
                if l == 0 and i in pc:
                    continue                       # PCFICH REG
                seq.append(all_regs[l][i])
    return _perm_regs(seq, n_id)


def pcfich_res(n_id, n_rb):
    regs = _sym_regs(n_id, n_rb, 0, 2)
    out = []
    for idx in pcfich_reg_idx(n_id, n_rb):
        out.extend(regs[idx])
    return out                                     # 16 REs
