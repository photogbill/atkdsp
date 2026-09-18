"""SIB1 acquisition, stage 1: PCFICH (control-format indicator) and the
PDCCH DCI decode core. Downlink broadcast only.

Reuses the validated MIB primitives (Gold sequence, tail-biting Viterbi,
rate matching, CRC-16, QPSK) from pbch_proto. This stage proves the coding /
scrambling / RNTI-masked-CRC path that a blind SI-RNTI search rests on; the
full-bandwidth control-region RE mapping (REG/CCE quadruplet interleaving)
and the PDSCH turbo decode come in later stages.

3GPP TS 36.211 (6.7 PCFICH, 6.8 PDCCH), 36.212 (5.3.3 DCI, 5.3.4 CFI).
"""
import numpy as np
import pbch_proto as B          # gold, crc16, conv_encode, viterbi_tb, rate_*, qpsk

SI_RNTI = 0xFFFF                # the RNTI that scrambles a SIB DCI's CRC

# ---- PCFICH (36.212 5.3.4 Table) -----------------------------------------
# The CFI codeword: 32 bits, one per CFI value 1..3 (4 is reserved).
_CFI_WORD = {
    1: np.array([0, 1, 1, 0] * 8, dtype=np.int8),
    2: np.array([1, 0, 1, 1] * 8, dtype=np.int8),
    3: np.array([1, 1, 0, 1] * 8, dtype=np.int8),
    4: np.array([0, 0, 0, 0] * 8, dtype=np.int8),
}


def pcfich_cinit(n_id, n_s):
    # 36.211 6.7.1
    return ((n_s // 2 + 1) * (2 * n_id + 1) << 9) + n_id


def pcfich_encode(cfi, n_id, n_s):
    b = _CFI_WORD[cfi]
    c = B.gold(pcfich_cinit(n_id, n_s), 32)
    scr = (b ^ c).astype(np.int8)
    return B.qpsk_mod(scr)                 # 16 QPSK symbols -> 16 REs


def pcfich_decode(sym16, n_id, n_s):
    """Return the most likely CFI (1..3) from the 16 received QPSK symbols."""
    c = B.gold(pcfich_cinit(n_id, n_s), 32)
    llr = np.empty(32)
    llr[0::2] = np.asarray(sym16).real
    llr[1::2] = np.asarray(sym16).imag
    llr = (1 - 2 * c) * llr                # descramble
    best, best_cfi = -1e30, 1
    for cfi in (1, 2, 3):
        score = float(np.dot(1 - 2 * _CFI_WORD[cfi], llr))
        if score > best:
            best, best_cfi = score, cfi
    return best_cfi


# ---- PDCCH DCI (36.212 5.3.3) --------------------------------------------
def pdcch_cinit(n_id, n_s):
    return (n_s // 2 << 9) + n_id          # 36.211 6.8.2


def dci_crc_attach(dci_bits, rnti):
    """CRC-16 over the DCI, its 16 parity bits XORed with the RNTI (MSB
    first). 36.212 5.3.3.2."""
    p = B.crc16(dci_bits)
    mask = np.array([(rnti >> (15 - i)) & 1 for i in range(16)], dtype=np.int8)
    return np.concatenate([dci_bits, p ^ mask]).astype(np.int8)


def dci_crc_check(bits, rnti):
    payload, rx = bits[:-16], bits[-16:]
    calc = B.crc16(payload)
    mask = np.array([(rnti >> (15 - i)) & 1 for i in range(16)], dtype=np.int8)
    return np.array_equal((calc ^ mask) & 1, rx & 1)


def pdcch_encode(dci_bits, rnti, E, n_id, n_s, cce_offset=0):
    """One DCI -> E scrambled QPSK-ready bits (a candidate's payload).

    `cce_offset` is where the candidate sits; PDCCH scrambling runs over the
    whole region, so the candidate's bits are scrambled from that offset."""
    frame = dci_crc_attach(dci_bits, rnti)
    coded = B.conv_encode(frame)
    e = B.rate_match([coded[0::3], coded[1::3], coded[2::3]], E)
    c = B.gold(pdcch_cinit(n_id, n_s), cce_offset * 72 + E)[cce_offset * 72:]
    return (e ^ c).astype(np.int8)


def pdcch_decode_candidate(scr_bits, D, rnti, n_id, n_s, cce_offset=0):
    """Try to decode ONE candidate. Returns the DCI payload bits or None."""
    E = len(scr_bits)
    c = B.gold(pdcch_cinit(n_id, n_s), cce_offset * 72 + E)[cce_offset * 72:]
    llr = (1 - 2 * c) * scr_bits           # descramble (soft in, +/-1 scale)
    d = B.rate_dematch(llr, D, seg=(0, E))
    bits = B.viterbi_tb(d)
    if not dci_crc_check(bits, rnti):
        return None
    return bits[:-16]


# aggregation levels in the common search space and their candidate counts
COMMON_SS = [(4, 4), (8, 2)]               # (aggregation level, #candidates)


def pdcch_blind_search(soft_bits_by_cce, D, rnti, n_id, n_s, n_cce):
    """Blind-decode the common search space for `rnti`.

    `soft_bits_by_cce` is the descrambling-domain +/-1 soft bits laid out CCE
    by CCE (72 per CCE). Returns (dci, agg_level, cce_offset) or None."""
    for al, ncand in COMMON_SS:
        for m in range(ncand):
            cce = m * al
            if cce + al > n_cce:
                continue
            seg = soft_bits_by_cce[cce * 72:(cce + al) * 72]
            dci = pdcch_decode_candidate(seg, D, rnti, n_id, n_s, cce_offset=cce)
            if dci is not None:
                return dci, al, cce
    return None
