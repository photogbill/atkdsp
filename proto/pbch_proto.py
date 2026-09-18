"""PBCH -> MIB prototype. The readable spec the C in src/lte_pbch.c will be
held to. 3GPP TS 36.211 (physical), 36.212 (coding), 36.331 (MIB).

Chain, transmit side:
    MIB(24) -> +CRC16 masked by antenna count (40) -> tail-biting conv code
    rate 1/3 (120) -> rate match to 1920 -> scramble (c_init = N_ID) ->
    QPSK -> layer map + tx diversity precode (1/2/4 ports) ->
    map to 240 REs in slot-1 symbols 0..3 (CRS punctured for 4 ports) ->
    OFDM (128-FFT, normal CP) -> subframe-0 time samples at 1.92 MSPS.

Receive side inverts it, blind over {nPorts in 1,2,4} x {SFN mod 4 in 0..3},
using CRS for per-port channel estimates and CRC for confirmation.
"""
import numpy as np

RATE = 1_920_000
NFFT = 128
NRB_PBCH = 6
NSC = NRB_PBCH * 12          # 72 subcarriers of the 6 central RB (DC incl.)
DC_K = NSC // 2              # 36: the DC subcarrier, index within the 72
CP0, CP1 = 10, 9            # normal-CP lengths at 128-FFT (first / rest)

# ---- Gold sequence (36.211 7.2) ------------------------------------------
_NC = 1600
def gold(c_init: int, length: int) -> np.ndarray:
    n = length + _NC
    x1 = np.zeros(n + 31, dtype=np.int8)
    x2 = np.zeros(n + 31, dtype=np.int8)
    x1[0] = 1
    for i in range(31):
        x2[i] = (c_init >> i) & 1
    for i in range(n):
        x1[i + 31] = (x1[i + 3] ^ x1[i]) & 1
        x2[i + 31] = (x2[i + 3] ^ x2[i + 2] ^ x2[i + 1] ^ x2[i]) & 1
    c = np.empty(length, dtype=np.int8)
    for i in range(length):
        c[i] = (x1[i + _NC] ^ x2[i + _NC]) & 1
    return c

# ---- CRC-16 (36.212 5.1.1, gCRC16 = D^16+D^12+D^5+1) ---------------------
# MSB-first polynomial division in GF(2): parity = (msg << 16) mod g.
_G16 = 0x11021          # D^16+D^12+D^5+1
def crc16(bits: np.ndarray) -> np.ndarray:
    """Return the 16 parity bits for `bits` (36.212 CRC calculation)."""
    reg = 0
    for b in bits:
        reg = (reg << 1) | int(b) & 1
        if reg & (1 << 16):
            reg ^= _G16
    for _ in range(16):
        reg <<= 1
        if reg & (1 << 16):
            reg ^= _G16
    return np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=np.int8)

# The three antenna-port CRC masks (36.212 Table 5.3.1.1-1).
CRC_MASK = {
    1: np.zeros(16, dtype=np.int8),
    2: np.ones(16, dtype=np.int8),
    4: np.array([0,1]*8, dtype=np.int8),   # <0000...> for 1, <1111> for 2,
}                                          # <0101...0101> for 4 ports

def crc16_attach(mib: np.ndarray, n_ports: int) -> np.ndarray:
    p = crc16(mib) ^ CRC_MASK[n_ports]
    return np.concatenate([mib, p]).astype(np.int8)

def crc16_check(bits40: np.ndarray):
    """Return n_ports if the CRC checks for one of the masks, else None."""
    payload, rx = bits40[:24], bits40[24:]
    calc = crc16(payload)
    for n_ports, mask in CRC_MASK.items():
        if np.array_equal((calc ^ mask) & 1, rx & 1):
            return n_ports
    return None

# ---- tail-biting convolutional code (36.212 5.1.3.1) ---------------------
# G0=133, G1=171, G2=165 octal; register [c_k, c_{k-1}, ..., c_{k-6}]
_GEN = (0o133, 0o171, 0o165)
def _taps(g):  # 7-bit tap masks, MSB = current input
    return [(g >> (6 - j)) & 1 for j in range(7)]
_TAPS = [_taps(g) for g in _GEN]

def conv_encode(bits: np.ndarray) -> np.ndarray:
    """Tail-biting: initial state = last 6 input bits."""
    K = len(bits)
    reg = list(bits[-6:][::-1])   # reg[0]=c_{-1}=c_{K-1}, ... reg[5]=c_{K-6}
    out = np.empty(3 * K, dtype=np.int8)
    for k in range(K):
        cur = int(bits[k])
        full = [cur] + reg[:6]    # [c_k, c_{k-1}, ..., c_{k-6}]
        for i, t in enumerate(_TAPS):
            out[3 * k + i] = sum(full[j] * t[j] for j in range(7)) & 1
        reg = [cur] + reg[:5]
    return out

# 64-state trellis, precomputed
def _build_trellis():
    ns = np.zeros((64, 2), dtype=np.int32)
    ob = np.zeros((64, 2, 3), dtype=np.int8)
    for s in range(64):
        reg5 = [(s >> i) & 1 for i in range(6)]   # reg[0..5] = c_{k-1..k-6}
        for cur in (0, 1):
            full = [cur] + reg5
            for i, t in enumerate(_TAPS):
                ob[s, cur, i] = sum(full[j] * t[j] for j in range(7)) & 1
            ns[s, cur] = (cur | (s << 1)) & 0x3F
    return ns, ob
_NS, _OB = _build_trellis()

# branch reward table: bm[f,i] = (1 - 2*out_bit_i) for transition f = s*2+cur.
# next state t = f & 63; the two predecessors of t are f=t and f=t+64, both
# with input bit cur = t & 1 (the LSB of the state we move into).
_BM = (1 - 2 * _OB.reshape(64, 2, 3).reshape(128, 3)).astype(np.float64)
_PRED_A = np.arange(64) // 2          # from-state when predecessor f = t
_PRED_B = np.arange(64) // 2 + 32     # from-state when predecessor f = t+64
_BBSTEP = (np.arange(64) & 1).astype(np.int8)

def viterbi_tb(llr: np.ndarray, laps: int = 3) -> np.ndarray:
    """Wrap-around Viterbi for the tail-biting code. `llr` is 3*K soft values
    (LLR, positive => bit 0). Returns K hard bits. Vectorised over the 64
    states; ties keep the lower-indexed predecessor (matches the C loop)."""
    K = len(llr) // 3
    pm = np.zeros(64)                     # all start states equal (tail-biting)
    steps = laps * K
    bp = np.empty((steps, 64), dtype=np.int32)
    L = llr.reshape(K, 3)
    step = 0
    for lap in range(laps):
        for k in range(K):
            reward = _BM @ L[k]           # (128,)
            cand = np.repeat(pm, 2) + reward
            A = cand[:64]; B = cand[64:]
            takeA = A >= B
            pm = np.where(takeA, A, B)
            bp[step] = np.where(takeA, _PRED_A, _PRED_B)
            step += 1
    s = int(np.argmax(pm))
    dec = np.zeros(steps, dtype=np.int8)
    for step in range(steps - 1, -1, -1):
        dec[step] = s & 1                 # input bit = LSB of the state entered
        s = int(bp[step, s])
    mid = (laps // 2) * K
    return dec[mid:mid + K].copy()

# ---- rate matching (36.212 5.1.4.2, convolutional) -----------------------
_PERM = np.array([1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31,
                  0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30])
def _subblock_indices(D: int):
    """Return (perm_index_or_-1 for each of the R*32 output positions)."""
    C = 32
    R = -(-D // C)
    nd = R * C - D
    seq = [-1] * nd + list(range(D))        # nulls first, then data indices
    M = np.array(seq).reshape(R, C)
    Mp = M[:, _PERM]
    return Mp.reshape(-1, order='F')         # column-by-column read

def rate_match(streams, E: int) -> np.ndarray:
    D = len(streams[0])
    idx = _subblock_indices(D)
    v = []
    for s in streams:
        col = np.array([(-1 if i < 0 else int(s[i])) for i in idx], dtype=np.int8)
        v.append(col)
    w = np.concatenate(v)                    # length 3*R*32, -1 marks null
    out = np.empty(E, dtype=np.int8)
    Kw = len(w)
    j = 0
    k = 0
    while k < E:
        val = w[j % Kw]
        j += 1
        if val >= 0:
            out[k] = val
            k += 1
    return out

def rate_dematch(e_llr: np.ndarray, D: int, seg=None):
    """Accumulate soft bits back into the 3 mother streams.
    `seg` = (start,stop) restricts to that slice of the E-index space;
    None means e_llr already covers [0,len)."""
    idx = _subblock_indices(D)
    Rc = len(idx)                            # R*32 per stream
    Kw = 3 * Rc
    isnull = np.zeros(Kw, dtype=bool)
    for si in range(3):
        for pos, i in enumerate(idx):
            if i < 0:
                isnull[si * Rc + pos] = True
    w = np.zeros(Kw)
    start = 0 if seg is None else seg[0]
    # walk the same circular skip-null read, depositing LLRs
    j = 0
    produced = 0
    ei = 0
    E_here = len(e_llr)
    target = E_here if seg is None else (seg[1] - seg[0])
    # advance j to the produced-count == start
    while produced < start:
        if not isnull[j % Kw]:
            produced += 1
        j += 1
    while ei < target:
        pos = j % Kw
        if not isnull[pos]:
            w[pos] += e_llr[ei]
            ei += 1
        j += 1
    # split + de-interleave each stream
    d = []
    order = idx
    for si in range(3):
        col = w[si * Rc:(si + 1) * Rc]
        dd = np.zeros(D)
        for pos, i in enumerate(order):
            if i >= 0:
                dd[i] += col[pos]
        d.append(dd)
    # interleave d0,d1,d2 -> 3*D llr for viterbi
    out = np.empty(3 * D)
    out[0::3] = d[0]; out[1::3] = d[1]; out[2::3] = d[2]
    return out

# ---- QPSK ----------------------------------------------------------------
_S = 1.0 / np.sqrt(2.0)
def qpsk_mod(bits: np.ndarray) -> np.ndarray:
    b = bits.reshape(-1, 2)
    return _S * ((1 - 2*b[:, 0]) + 1j * (1 - 2*b[:, 1]))

# ---- CRS (36.211 6.10.1) -------------------------------------------------
NRB_MAX = 110
def crs_seq(n_id, n_s, l):
    ci = (1 << 10) * (7 * (n_s + 1) + l + 1) * (2 * n_id + 1) + 2 * n_id + 1
    c = gold(ci, 4 * NRB_MAX)
    r = _S * (1 - 2 * c[0::2]) + 1j * _S * (1 - 2 * c[1::2])
    return r      # length 2*NRB_MAX; central 6 RB use indices below

def crs_for_central(n_id, n_s, l):
    """CRS values for the 6 central RB: 12 complex values, in k=0..71 order."""
    r = crs_seq(n_id, n_s, l)
    m0 = NRB_MAX - NRB_PBCH             # 104: first of the central 12
    return r[m0:m0 + 2 * NRB_PBCH]      # 12 values, m'=0..11

def crs_positions(n_id, port, l):
    """subcarrier k (0..71) carrying CRS for `port` in symbol l of a slot."""
    vshift = n_id % 6
    if port == 0:
        v = 0 if l == 0 else 3
    elif port == 1:
        v = 3 if l == 0 else 0
    elif port == 2:
        v = 0
    else:
        v = 3
    return [6 * m + (v + vshift) % 6 for m in range(2 * NRB_PBCH)]

# CRS-bearing symbols within a slot for each port
def crs_symbols(port):
    return (0, 4) if port in (0, 1) else (1,)

# ---- RE map: the 240 PBCH REs (k,l) with l in 0..3, CRS(4 ports) removed --
def pbch_re_map(n_id):
    crs_k = {l: set() for l in range(4)}
    for port in range(4):
        for l in crs_symbols(port):
            if l < 4:
                for k in crs_positions(n_id, port, l):
                    crs_k[l].add(k)
    res = []
    for l in range(4):
        for k in range(NSC):
            if k not in crs_k[l]:
                res.append((k, l))
    assert len(res) == 240, len(res)
    return res

# ---- tx diversity precoding (36.211 6.3.4.3) -----------------------------
def precode(y, n_ports):
    n = len(y)
    x = np.zeros((n_ports, n), dtype=complex)
    if n_ports == 1:
        x[0] = y
    elif n_ports == 2:
        for i in range(0, n, 2):
            x[0, i]   = _S * y[i]
            x[1, i]   = -_S * np.conj(y[i+1])
            x[0, i+1] = _S * y[i+1]
            x[1, i+1] = _S * np.conj(y[i])
    else:  # 4 ports: SFBC + FSTD
        for i in range(0, n, 4):
            x[0, i]   = _S * y[i]
            x[2, i]   = -_S * np.conj(y[i+1])
            x[0, i+1] = _S * y[i+1]
            x[2, i+1] = _S * np.conj(y[i])
            x[1, i+2] = _S * y[i+2]
            x[3, i+2] = -_S * np.conj(y[i+3])
            x[1, i+3] = _S * y[i+3]
            x[3, i+3] = _S * np.conj(y[i+2])
    return x

def alamouti_combine(r0, r1, h0, h1):
    """Recover two symbols from an SFBC pair encoded as
        r0 = h0*(S y0) + h1*(-S conj(y1))
        r1 = h0*(S y1) + h1*( S conj(y0))
    (the 36.211 6.3.4.3 convention used by precode()). Returns (y0,y1,gain),
    each scaled by gain*S where gain = |h0|^2+|h1|^2."""
    g = (np.abs(h0)**2 + np.abs(h1)**2)
    y0 = (np.conj(h0) * r0 + h1 * np.conj(r1))
    y1 = (np.conj(h0) * r1 - h1 * np.conj(r0))
    return y0, y1, g

# ---- OFDM ----------------------------------------------------------------
def _k_to_bin(k):
    off = k - DC_K            # -36..+35
    return off % NFFT

def ofdm_mod_symbol(grid72, first):
    X = np.zeros(NFFT, dtype=complex)
    for k in range(NSC):
        X[_k_to_bin(k)] = grid72[k]
    xt = np.fft.ifft(X) * NFFT / np.sqrt(NSC)   # match to_symbol() scaling
    cp = CP0 if first else CP1
    return np.concatenate([xt[-cp:], xt])

def ofdm_demod_symbol(samples, first):
    cp = CP0 if first else CP1
    xt = samples[cp:cp + NFFT]
    X = np.fft.fft(xt) * np.sqrt(NSC) / NFFT
    return np.array([X[_k_to_bin(k)] for k in range(NSC)])

# ==== top level: MIB <-> subframe-0 PBCH time samples =====================
def mib_pack(dl_bw, phich_dur, phich_res, sfn):
    """dl_bw in {6,15,25,50,75,100}; sfn 0..1023. Returns 24 bits + segment i."""
    bw_map = {6:0,15:1,25:2,50:3,75:4,100:5}
    bits = []
    bits += [(bw_map[dl_bw] >> (2-i)) & 1 for i in range(3)]
    bits += [phich_dur & 1]
    bits += [(phich_res >> (1-i)) & 1 for i in range(2)]
    sfn_msb = sfn >> 2                       # the 8 MSBs
    bits += [(sfn_msb >> (7-i)) & 1 for i in range(8)]
    bits += [0]*10                           # spare
    return np.array(bits, dtype=np.int8), sfn & 3

def mib_unpack(bits24, i):
    bw_map = {0:6,1:15,2:25,3:50,4:75,5:100}
    bw = bw_map[int(bits24[0])*4 + int(bits24[1])*2 + int(bits24[2])]
    phich_dur = int(bits24[3])
    phich_res = int(bits24[4])*2 + int(bits24[5])
    sfn_msb = 0
    for b in bits24[6:14]:
        sfn_msb = (sfn_msb << 1) | int(b)
    sfn = (sfn_msb << 2) | i
    return dict(dl_bw=bw, phich_dur=phich_dur, phich_res=phich_res, sfn=sfn)

def pbch_encode_segment(mib24, n_id, n_ports, i):
    """Return the 240 QPSK symbols for segment i (SFN mod 4 = i)."""
    bits40 = crc16_attach(mib24, n_ports)
    coded = conv_encode(bits40)
    streams = [coded[0::3], coded[1::3], coded[2::3]]
    b1920 = rate_match(streams, 1920)
    c = gold(n_id, 1920)
    bscr = (b1920 ^ c).astype(np.int8)
    seg = bscr[i*480:(i+1)*480]
    return qpsk_mod(seg)

def build_pbch_samples(mib24, n_id, n_ports, sfn, H_ports, snr_db=100.0, seed=0):
    """Synthesize the 4 PBCH OFDM symbols (slot 1, subframe 0) as time samples.
    H_ports: list of n_ports complex per-subcarrier channels (len 72 each)."""
    i = sfn & 3
    y = pbch_encode_segment(mib24, n_id, n_ports, i)
    x = precode(y, n_ports)                       # [n_ports, 240]
    re_map = pbch_re_map(n_id)
    # frequency grids for the 4 symbols, channel applied per port
    grids = [np.zeros(NSC, dtype=complex) for _ in range(4)]
    for idx, (k, l) in enumerate(re_map):
        s = 0j
        for p in range(n_ports):
            s += H_ports[p][k] * x[p, idx]
        grids[l][k] = s
    # add CRS (through the same channels)
    for p in range(n_ports):
        for l in crs_symbols(p):
            if l >= 4:
                continue
            r = crs_for_central(n_id, 1, l)       # slot 1, n_s=1
            pos = crs_positions(n_id, p, l)
            for m, k in enumerate(pos):
                grids[l][k] += H_ports[p][k] * r[m]
    # OFDM modulate, symbol 0 is first-of-slot (CP0)
    parts = [ofdm_mod_symbol(grids[l], first=(l == 0)) for l in range(4)]
    sig = np.concatenate(parts)
    if snr_db < 90:
        rng = np.random.default_rng(seed)
        sp = np.mean(np.abs(sig)**2)
        npow = sp / (10**(snr_db/10))
        sig = sig + np.sqrt(npow/2)*(rng.standard_normal(len(sig))
                                     + 1j*rng.standard_normal(len(sig)))
    return sig

def _interp_channel(vals_at_k, ks):
    """Linear-interpolate complex channel samples given at subcarriers ks to 0..71."""
    ks = np.array(ks); order = np.argsort(ks)
    ks = ks[order]; v = np.array(vals_at_k)[order]
    out = np.interp(np.arange(NSC), ks, v.real) + 1j*np.interp(np.arange(NSC), ks, v.imag)
    return out

def pbch_decode(samples, n_id):
    """Blind decode over n_ports x segment. Returns dict or None."""
    # demod the 4 symbols; offs[l] = start (incl. CP) of PBCH symbol l
    offs = [0, CP0+NFFT, CP0+NFFT+CP1+NFFT, CP0+NFFT+2*(CP1+NFFT)]
    Y = [ofdm_demod_symbol(samples[offs[l]:], first=(l == 0)) for l in range(4)]
    # channel per port
    H = {}
    for p in range(4):
        est = None
        for l in crs_symbols(p):
            if l >= 4:
                continue
            r = crs_for_central(n_id, 1, l)
            pos = crs_positions(n_id, p, l)
            hk = [Y[l][k]/r[m] for m, k in enumerate(pos)]
            est = _interp_channel(hk, pos)
            break
        H[p] = est
    re_map = pbch_re_map(n_id)
    c = gold(n_id, 1920)
    for n_ports in (1, 2, 4):
        if any(H[p] is None for p in range(n_ports)):
            continue
        # equalize / combine -> 240 soft symbols
        yhat = np.zeros(240, dtype=complex)
        if n_ports == 1:
            for idx, (k, l) in enumerate(re_map):
                yhat[idx] = np.conj(H[0][k]) * Y[l][k]
        elif n_ports == 2:
            for idx in range(0, 240, 2):
                k0, l0 = re_map[idx]; k1, l1 = re_map[idx+1]
                y0, y1, g = alamouti_combine(Y[l0][k0], Y[l1][k1],
                                             H[0][k0], H[1][k0])
                yhat[idx] = y0; yhat[idx+1] = y1
        else:
            for idx in range(0, 240, 4):
                k0, l0 = re_map[idx]; k1, l1 = re_map[idx+1]
                k2, l2 = re_map[idx+2]; k3, l3 = re_map[idx+3]
                a0, a1, _ = alamouti_combine(Y[l0][k0], Y[l1][k1],
                                             H[0][k0], H[2][k0])
                b0, b1, _ = alamouti_combine(Y[l2][k2], Y[l3][k3],
                                             H[1][k2], H[3][k2])
                yhat[idx]=a0; yhat[idx+1]=a1; yhat[idx+2]=b0; yhat[idx+3]=b1
        # LLRs (scrambled), mapping order
        llr = np.empty(480)
        llr[0::2] = yhat.real
        llr[1::2] = yhat.imag
        for i in range(4):
            cc = c[i*480:(i+1)*480]
            dl = (1 - 2*cc) * llr
            d = rate_dematch(dl, 40, seg=(i*480, (i+1)*480))
            bits40 = viterbi_tb(d)
            np_ck = crc16_check(bits40)
            if np_ck != n_ports:
                continue
            # false-alarm gate: a real MIB has a valid bandwidth code and 10
            # spare bits that are all zero (36.331). This turns an occasional
            # chance CRC pass on noise into ~2^-20 further protection.
            bw_code = int(bits40[0])*4 + int(bits40[1])*2 + int(bits40[2])
            if bw_code > 5 or np.any(bits40[14:24] != 0):
                continue
            info = mib_unpack(bits40[:24], i)
            info['n_ports'] = n_ports
            return info
    return None

def _soft_symbols(Y, H, n_ports, re_map):
    """Return the 480 scrambled-bit LLRs (mapping order) for one frame."""
    yhat = np.zeros(240, dtype=complex)
    if n_ports == 1:
        for idx, (k, l) in enumerate(re_map):
            yhat[idx] = np.conj(H[0][k]) * Y[l][k]
    elif n_ports == 2:
        for idx in range(0, 240, 2):
            k0, l0 = re_map[idx]; k1, l1 = re_map[idx+1]
            y0, y1, _ = alamouti_combine(Y[l0][k0], Y[l1][k1], H[0][k0], H[1][k0])
            yhat[idx] = y0; yhat[idx+1] = y1
    else:
        for idx in range(0, 240, 4):
            k0, l0 = re_map[idx]; k1, l1 = re_map[idx+1]
            k2, l2 = re_map[idx+2]; k3, l3 = re_map[idx+3]
            a0, a1, _ = alamouti_combine(Y[l0][k0], Y[l1][k1], H[0][k0], H[2][k0])
            b0, b1, _ = alamouti_combine(Y[l2][k2], Y[l3][k3], H[1][k2], H[3][k2])
            yhat[idx]=a0; yhat[idx+1]=a1; yhat[idx+2]=b0; yhat[idx+3]=b1
    llr = np.empty(480)
    llr[0::2] = yhat.real
    llr[1::2] = yhat.imag
    return llr

def _channels(Y, n_id):
    H = {}
    for p in range(4):
        est = None
        for l in crs_symbols(p):
            if l >= 4:
                continue
            r = crs_for_central(n_id, 1, l)
            pos = crs_positions(n_id, p, l)
            hk = [Y[l][k]/r[m] for m, k in enumerate(pos)]
            est = _interp_channel(hk, pos)
            break
        H[p] = est
    return H

def pbch_decode_frames(sample_blocks, n_id):
    """Soft-combine 1..4 consecutive PBCH blocks (each the 4-symbol slot-1
    block of subframe 0). Returns MIB dict (with sfn of the FIRST block) or
    None. Recovers SFN mod 4 of the first frame as `i0`."""
    re_map = pbch_re_map(n_id)
    c = gold(n_id, 1920)
    frames = []
    for sb in sample_blocks:
        offs = [0, CP0+NFFT, CP0+NFFT+CP1+NFFT, CP0+NFFT+2*(CP1+NFFT)]
        Y = [ofdm_demod_symbol(sb[offs[l]:], first=(l == 0)) for l in range(4)]
        frames.append((Y, _channels(Y, n_id)))
    nfr = len(frames)
    for n_ports in (1, 2, 4):
        if any(H[p] is None for (_, H) in frames for p in range(n_ports)):
            continue
        raw = [_soft_symbols(Y, H, n_ports, re_map) for (Y, H) in frames]
        for i0 in range(4):
            acc = np.zeros(3 * 40)
            for f in range(nfr):
                seg = (i0 + f) % 4
                cc = c[seg*480:(seg+1)*480]
                dl = (1 - 2*cc) * raw[f]
                acc += rate_dematch(dl, 40, seg=(seg*480, (seg+1)*480))
            bits40 = viterbi_tb(acc)
            if crc16_check(bits40) != n_ports:
                continue
            bw_code = int(bits40[0])*4 + int(bits40[1])*2 + int(bits40[2])
            if bw_code > 5 or np.any(bits40[14:24] != 0):
                continue
            info = mib_unpack(bits40[:24], i0)
            info['n_ports'] = n_ports
            return info
    return None
