"""FLEX paging — decode machinery. Prototype / spec (proto-first, like pocsag_proto).

FLEX (Motorola, ARIB STD-T44) is what most modern paging channels actually run —
including the hospital/EMS 6400 bps/4-level net Bill's friend captured at
929.611 MHz. It shares POCSAG's BCH(31,21) with generator 0x769, so the FEC and
the atkdsp FSK front end carry straight over; what is new is the frame structure:
a phase of 88 words = 11 blocks x 8 bit-interleaved codewords, a Frame
Information Word, and address/vector/message words that assemble into pages.

============================ STATUS — READ THIS ============================
This mirrors POCSAG's proto discipline (validated by an encode->decode round
trip) BUT, like ATK's LTE SIB1 receiver, a self-consistent round trip proves the
LOGIC, not real-signal correctness. CONFIRMED against the de-facto reference
(multimon-ng demod_flex_next.c) and reproduced here as real constants:
  - sync/mode table (A-codes 0x870C/0x7B18/0xB068/0xDEA0), BCH(31,21) gen 0x769,
  - the 8x32 block interleave, the FIW checksum, the alphanumeric 3x7-bit word
    packing and its fragment header + K-checksum, the vector-type enum.
PENDING confirmation against the reference AND a real capture (Bill's friend's
929.611 channel is the corpus):
  1. On-wire codeword BIT-NUMBERING (info vs parity end) — chosen self-consistent
     here; must match the wire before a live decode is trusted.
  2. 4-LEVEL (6400) PHASE DE-MUX: a 4-level symbol carries 2 bits split across
     phases A/B/C/D. The 2-level path (1600/3200) is complete; the 4-level
     phase assignment is scaffolded (see split_4level) and needs the capture.
     The adaptive LEVEL slicer (slice_4level) IS implemented and validated
     offset/gain-invariant; only the level->dibit->phase mapping is pending.
  3. On-air SYNC ACQUISITION (bit-sync dotting, A/inv-A/C commas) — the proto
     works at the frame level; RF acquisition is atkdsp sync_search on the A-code.
  4. Numeric / short / binary vector types (only alphanumeric is assembled here).
Ref: multimon-ng demod_flex_next.c; ARIB STD-T44; sigidwiki FLEX thesis.
===========================================================================
"""
from __future__ import annotations

import numpy as np

# ---- constants (confirmed against multimon-ng) --------------------------
BCH_GEN = 0x769                      # x^10+x^9+x^8+x^6+x^5+x^3+1 — as POCSAG
PHASE_WORDS = 88                     # 11 blocks x 8 words
BLOCK_WORDS = 8
WORDS_PER_BLOCK_BITS = BLOCK_WORDS * 32   # 256 interleaved bits per block

#: Sync-1 "A" code -> (bps, level). ReFLEX (0x4C7C) is noted, not decoded.
MODES = {
    0x870C: (1600, 2),
    0x7B18: (3200, 2),
    0xB068: (3200, 4),   # 3200 bps, 4-level (1600 symbols/s)
    0xDEA0: (6400, 4),   # 6400 bps, 4-level (3200 symbols/s) — the hospital net
}

#: Vector (message) types — FLEX enum (3.10.1.3).
VEC_SECURE, VEC_SHORT_INSTR, VEC_SHORT_MSG, VEC_NUMERIC = 0, 1, 2, 3
VEC_SPECIAL_NUM, VEC_ALPHANUMERIC, VEC_BINARY, VEC_NUM_NUMBERED = 4, 5, 6, 7


# ---- BCH(31,21) + even parity -------------------------------------------
# Same generator as POCSAG. Bit layout here (self-consistent, PENDING #1):
# info in bits 0..20, BCH parity in bits 21..30, even parity in bit 31.
def _rem(v: int) -> int:
    """Remainder of v (info in low 21 bits, shifted) mod BCH_GEN, degree<10."""
    for i in range(30, 9, -1):
        if (v >> i) & 1:
            v ^= BCH_GEN << (i - 10)
    return v & 0x3FF


def bch_encode(info21: int) -> int:
    """21 info bits -> 32-bit FLEX codeword (info 0..20, parity 21..30, even 31)."""
    info21 &= 0x1FFFFF
    # systematic: parity = remainder of (info << 10) — computed on a reversed
    # register so the 31-bit code is a valid BCH codeword under 0x769.
    parity = _rem(info21 << 10)
    cw31 = info21 | (parity << 21)
    ones = bin(cw31).count("1") & 1
    return cw31 | (ones << 31)


def _syndrome31(cw31: int) -> int:
    return _rem(((cw31 & 0x1FFFFF) << 10) ^ ((cw31 >> 21) & 0x3FF))


def _err_table():
    tab = {0: 0}
    for i in range(31):
        e1 = 1 << i
        tab[_syndrome31(e1)] = e1
        for j in range(i + 1, 31):
            e2 = e1 | (1 << j)
            tab[_syndrome31(e2)] = e2
    return tab


_ERR = _err_table()


def bch_decode(cw32: int):
    """32-bit codeword -> (info21, n_corrected, ok). Bounded-distance t=2."""
    cw31 = cw32 & 0x7FFFFFFF
    syn = _syndrome31(cw31)
    if syn == 0:
        corr = 0
    elif syn in _ERR:
        cw31 ^= _ERR[syn]
        corr = bin(_ERR[syn]).count("1")
    else:
        return cw31 & 0x1FFFFF, -1, False
    return cw31 & 0x1FFFFF, corr, True


# ---- block interleave (8 words x 32 bits) -------------------------------
# Confirmed de-interleave: for bit index c in a 256-bit block,
#   codeword = c & 7 ; bit position = (c >> 3) & 31.
def interleave_block(words8) -> np.ndarray:
    """8 codewords (32-bit ints) -> 256 interleaved bits (transmit order)."""
    bits = np.empty(WORDS_PER_BLOCK_BITS, dtype=np.int8)
    for c in range(WORDS_PER_BLOCK_BITS):
        w = c & 7
        b = (c >> 3) & 31
        bits[c] = (words8[w] >> b) & 1
    return bits


def deinterleave_block(bits256) -> list:
    """256 interleaved bits -> 8 codewords (32-bit ints)."""
    words = [0] * BLOCK_WORDS
    for c in range(WORDS_PER_BLOCK_BITS):
        w = c & 7
        b = (c >> 3) & 31
        words[w] |= (int(bits256[c]) & 1) << b
    return words


# ---- Frame Information Word ----------------------------------------------
def fiw_build(cycle: int, frame: int, roaming=0, repeat=0, traffic=0) -> int:
    """Assemble a FIW with a valid checksum (nibble-sum + bit20 == 0xF)."""
    fiw = ((cycle & 0xF) << 4) | ((frame & 0x7F) << 8) | ((roaming & 1) << 15) \
        | ((repeat & 1) << 16) | ((traffic & 0xF) << 17)
    s = ((fiw & 0xF) + ((fiw >> 4) & 0xF) + ((fiw >> 8) & 0xF)
         + ((fiw >> 12) & 0xF) + ((fiw >> 16) & 0xF) + ((fiw >> 20) & 0x01))
    fiw |= (0xF - (s & 0xF)) & 0xF          # checksum nibble so total == 0xF
    return fiw & 0x1FFFFF


def fiw_parse(fiw: int):
    """-> dict, with 'valid' from the checksum. None fields on a bad checksum."""
    s = ((fiw & 0xF) + ((fiw >> 4) & 0xF) + ((fiw >> 8) & 0xF)
         + ((fiw >> 12) & 0xF) + ((fiw >> 16) & 0xF) + ((fiw >> 20) & 0x01)) & 0xF
    return {"valid": s == 0xF, "cycle": (fiw >> 4) & 0xF, "frame": (fiw >> 8) & 0x7F,
            "roaming": (fiw >> 15) & 1, "repeat": (fiw >> 16) & 1,
            "traffic": (fiw >> 17) & 0xF}


# ---- address + capcode ---------------------------------------------------
def capcode_short(addr_word: int) -> int:
    """Short-address capcode (the common case)."""
    return addr_word & 0x1FFFFF


# ---- alphanumeric message words (3 x 7-bit ASCII per 21-bit word) --------
def aln_header(msg_num: int, frag=0b11, cont=0, retrieval=0, maildrop=0) -> int:
    """Fragment header word body (bits 0..20); K checksum (bits0..9) filled by
    the block builder once the content words are known."""
    return ((cont & 1) << 10) | ((frag & 3) << 11) | ((msg_num & 0x3F) << 13) \
        | ((retrieval & 1) << 19) | ((maildrop & 1) << 20)


def aln_pack(text: str) -> list:
    """Text -> list of 21-bit words, 3 x 7-bit ASCII each (low char first)."""
    words = []
    data = [ord(c) & 0x7F for c in text]
    for i in range(0, len(data), 3):
        chunk = data[i:i + 3] + [0] * (3 - len(data[i:i + 3]))
        words.append(chunk[0] | (chunk[1] << 7) | (chunk[2] << 14))
    return words


def aln_unpack(words) -> str:
    """Inverse of aln_pack. Stops at a NUL, like a real terminated page."""
    out = []
    for w in words:
        for shift in (0, 7, 14):
            c = (w >> shift) & 0x7F
            if c == 0:
                return "".join(out)
            out.append(chr(c))
    return "".join(out)


# ---- 4-level symbol -> dibit (SCAFFOLD, PENDING #2) ----------------------
def split_4level(symbols):
    """4-level FSK symbols (0..3) -> the two bit-planes that feed the phase
    pairs. The MAPPING is the confirmed level->dibit order; the assignment of
    the two bits to phases A/B/C/D is what must be confirmed against a 6400
    capture before this is trusted. Returned as (msb_bits, lsb_bits)."""
    s = np.asarray(symbols, dtype=np.int64)
    # Gray-ish level -> dibit is the usual FLEX order; PENDING confirmation.
    msb = (s >> 1) & 1
    lsb = s & 1
    return msb.astype(np.int8), lsb.astype(np.int8)


def slice_4level(symbols, center=True):
    """Adaptive 4-level slicer (layer 3): recover each 4-level symbol's level index
    (0..3), invariant to carrier offset (median-centre) and deviation/gain (scale
    from the data: E|x| = 2u for equiprobable {-3,-1,1,3}u). This is the
    multi-threshold decision an off-centre, sub-optimal-RF 6400/4FSK channel needs
    and a fixed slicer can't. The level->dibit ordering that feeds the 4 phases is
    the piece still PENDING against a real 6400 capture (see split_4level/STATUS)."""
    x = np.asarray(symbols, dtype=np.float64)
    if x.size == 0:
        return np.zeros(0, dtype=np.int8)
    if center:
        x = x - np.median(x)
    u = float(np.mean(np.abs(x))) / 2.0
    t = 2.0 * (u if u > 1e-9 else 1e-9)
    out = np.zeros(x.size, dtype=np.int8)
    out[x >= -t] = 1
    out[x >= 0.0] = 2
    out[x >= t] = 3
    return out
