"""POCSAG (CCIR Radiopaging Code No. 1) - detector + receiver. Prototype / spec.

The readable, validated specification the atkdsp C kernels will be held to
(proto-first, like turbo_proto / pbch_proto). Whole receive chain, 512/1200/2400:

    2-FSK baseband -> FM discriminator -> preamble bit-timing -> per-symbol soft
    values -> frame-sync search (0x7CD215D8, both polarities) -> BCH(31,21)+parity
    correction (hard bounded-distance AND Chase-2 soft-decision) -> batch/frame
    parse -> 21-bit address + 2-bit function -> alpha (7-bit) / numeric (4-bit).

The DECODER (all of the above) and the DETECTOR (`detect`) share one front end.
The detector is the frame-sync correlation run in confirm-only mode: cheap,
near-certain, never has to decode a bit to say "POCSAG, this baud, this polarity".

Standardised constants are the real ones (sync, idle, BCH generator), so those
are correct against the air. The conventions the standard leaves to the
transmitter - FSK polarity, and the per-character bit order of the payload - are
handled by trying both polarities and are flagged for a bench test against
multimon / a live channel; a self-consistent round trip cannot settle those.

Refs: ITU-R M.584; RPC No. 1 (POCSAG).
"""
from __future__ import annotations
import numpy as np

# ---- standardised constants (correct against the air) --------------------
SYNC = 0x7CD215D8            # Frame Synchronisation Codeword (FSC)
IDLE = 0x7A89C197            # Idle codeword
BCH_GEN = 0x769              # x^10+x^9+x^8+x^6+x^5+x^3+1  (BCH(31,21))
IDLE_INFO = (IDLE >> 11) & 0x1FFFFF
PREAMBLE_BITS = 576          # min alternating 1010... for bit sync
BATCH_WORDS = 16             # codewords per batch (8 frames x 2)
BAUDS = (512, 1200, 2400)


def _word_bits(w: int):
    return [(w >> (31 - i)) & 1 for i in range(32)]


# ---- BCH(31,21) + even parity -------------------------------------------
def _reduce(reg: int) -> int:
    """Polynomial remainder mod BCH_GEN, reg's high bits down to degree <10."""
    for i in range(30, 9, -1):
        if (reg >> i) & 1:
            reg ^= BCH_GEN << (i - 10)
    return reg & 0x3FF


def _bch_parity10(info21: int) -> int:
    return _reduce(info21 << 10)


def bch_encode(info21: int) -> int:
    """21 info bits (flag + 20 payload, MSB=flag) -> full 32-bit codeword."""
    cw31 = (info21 << 10) | _bch_parity10(info21)
    return (cw31 << 1) | (bin(cw31).count("1") & 1)


def _syndrome(cw31: int) -> int:
    return _reduce(cw31)


def _build_error_table():
    # d=5 => every weight<=2 pattern over 31 bits has a distinct syndrome:
    # an exact, obviously-correct bounded-distance decoder, no GF(32) needed.
    tab = {0: 0}
    for i in range(31):
        e1 = 1 << i
        tab[_syndrome(e1)] = e1
        for j in range(i + 1, 31):
            e2 = e1 | (1 << j)
            tab[_syndrome(e2)] = e2
    return tab


_ERR_TABLE = _build_error_table()


def bch_hard_decode(cw32: int):
    """Full 32-bit word -> (info21, n_corrected, ok). Bounded-distance t=2."""
    cw31 = cw32 >> 1
    syn = _syndrome(cw31)
    if syn == 0:
        corr = 0
    elif syn in _ERR_TABLE:
        e = _ERR_TABLE[syn]
        cw31 ^= e
        corr = bin(e).count("1")
    else:
        return (cw31 >> 10) & 0x1FFFFF, -1, False
    return (cw31 >> 10) & 0x1FFFFF, corr, True


def bch_soft_decode(llr32, p: int = 5):
    """Chase-2 soft-decision. llr32: 32 LLRs, >0 favours bit 0, MSB first.
    Flip the p least-reliable of the 31 BCH bits, hard-decode each candidate,
    keep the valid codeword best matching the soft input. -> (info21, ok)."""
    llr = np.asarray(llr32, float)
    hard = (llr < 0).astype(np.int64)
    bch_llr = llr[:31]
    flips = np.argsort(np.abs(bch_llr))[:p]
    base = 0
    for b in hard:
        base = (base << 1) | int(b)
    best_info, best_metric = None, None
    for mask in range(1 << p):
        w = base
        for k in range(p):
            if (mask >> k) & 1:
                w ^= 1 << (31 - int(flips[k]))
        info21, _, ok = bch_hard_decode(w)
        if not ok:
            continue
        cw31 = bch_encode(info21) >> 1
        bits = np.array([(cw31 >> (30 - i)) & 1 for i in range(31)])
        metric = float(np.sum(bch_llr * (1 - 2 * bits)))
        if best_metric is None or metric > best_metric:
            best_metric, best_info = metric, info21
    if best_info is None:
        return (base >> 11) & 0x1FFFFF, False
    return best_info, True


# ---- codewords -----------------------------------------------------------
def frame_of_address(address21: int) -> int:
    return address21 & 0x7


def address_codeword(address21: int, function: int) -> int:
    addr18 = (address21 >> 3) & 0x3FFFF
    return bch_encode((0 << 20) | (addr18 << 2) | (function & 3))


def message_codeword(payload20: int) -> int:
    return bch_encode((1 << 20) | (payload20 & 0xFFFFF))


# ---- text packing --------------------------------------------------------
_NUM_MAP = "0123456789*U -)("     # 16 numeric symbols (0x0..0xF)
_NUM_REV = {c: i for i, c in enumerate(_NUM_MAP)}


def _pack20(payload_bits):
    """bits as transmitted (first bit first) -> list of 20-bit ints, 0-padded."""
    words = []
    for i in range(0, len(payload_bits), 20):
        chunk = list(payload_bits[i:i + 20])
        chunk += [0] * (20 - len(chunk))
        v = 0
        for b in chunk:
            v = (v << 1) | b
        words.append(v)
    return words


def encode_alpha(text: str):
    bits = []
    for ch in text:
        c = ord(ch) & 0x7F
        for i in range(7):            # LSB first
            bits.append((c >> i) & 1)
    return bits


def decode_alpha(bits):
    chars = []
    for i in range(0, len(bits) - 6, 7):
        c = 0
        for k in range(7):
            c |= bits[i + k] << k
        if c == 0:
            break
        chars.append(chr(c))
    return "".join(chars)


def encode_numeric(text: str):
    bits = []
    for ch in text:
        d = _NUM_REV.get(ch, 0)
        for i in range(4):            # LSB first
            bits.append((d >> i) & 1)
    return bits


def decode_numeric(bits):
    out = []
    for i in range(0, len(bits) - 3, 4):
        d = 0
        for k in range(4):
            d |= bits[i + k] << k
        out.append(_NUM_MAP[d])
    return "".join(out)


def build_message_codewords(kind: str, text: str):
    bits = encode_alpha(text) if kind == "alpha" else encode_numeric(text)
    return [message_codeword(w) for w in _pack20(bits)]


# ---- batch assembly + serialisation -------------------------------------
def build_stream_bits(messages):
    """messages: list of dict(address, function, kind, text) -> transmitted bits
    (preamble + one or more batches)."""
    batches, used = [[IDLE] * BATCH_WORDS], [[False] * BATCH_WORDS]

    def place(msg):
        f = frame_of_address(msg["address"])
        acw = address_codeword(msg["address"], msg["function"])
        mcws = build_message_codewords(msg["kind"], msg["text"])
        pos = 2 * f
        for bi in range(len(batches)):
            if used[bi][pos]:
                continue
            if pos + 1 + len(mcws) > BATCH_WORDS:
                continue
            if any(used[bi][pos + 1 + k] for k in range(len(mcws))):
                continue
            batches[bi][pos] = acw
            used[bi][pos] = True
            for k, m in enumerate(mcws):
                batches[bi][pos + 1 + k] = m
                used[bi][pos + 1 + k] = True
            return
        batches.append([IDLE] * BATCH_WORDS)
        used.append([False] * BATCH_WORDS)
        place(msg)

    for m in messages:
        place(m)

    bits = [1 if (i % 2 == 0) else 0 for i in range(PREAMBLE_BITS)]
    for b in batches:
        bits += _word_bits(SYNC)
        for w in b:
            bits += _word_bits(w)
    return bits


# ---- 2-FSK modem (baseband model) ----------------------------------------
def modulate(bits, fs, baud, fdev=4500.0):
    """CPFSK: bit 0 -> +fdev, bit 1 -> -fdev (POCSAG: 0 is the higher freq)."""
    sps = int(round(fs / baud))
    freqs = np.where(np.repeat(np.asarray(bits), sps) == 0, +fdev, -fdev)
    phase = 2 * np.pi * np.cumsum(freqs) / fs
    return np.exp(1j * phase).astype(np.complex128), sps


def add_awgn(x, snr_db, rng):
    p = np.mean(np.abs(x) ** 2)
    n0 = p / (10 ** (snr_db / 10))
    return x + np.sqrt(n0 / 2) * (rng.standard_normal(len(x))
                                  + 1j * rng.standard_normal(len(x)))


def discriminate(x):
    """FM discriminator: instantaneous frequency (rad/sample)."""
    return np.angle(x[1:] * np.conj(x[:-1]))


# ---- timing + per-symbol soft values -------------------------------------
def symbol_soft(disc, sps, phase):
    n = (len(disc) - phase) // sps
    seg = disc[phase:phase + n * sps].reshape(n, sps)
    return seg.mean(axis=1)           # >0 -> bit 0 (higher freq)


def estimate_phase(disc, sps):
    best_ph, best_e = 0, -1.0
    for ph in range(sps):
        e = float(np.mean(np.abs(symbol_soft(disc, sps, ph))))
        if e > best_e:
            best_e, best_ph = e, ph
    return best_ph


# ---- receiver ------------------------------------------------------------
_SYNCBITS = np.array(_word_bits(SYNC))


def _find_syncs(bits, max_err=2):
    out = []
    for i in range(0, len(bits) - 32):
        d = int(np.sum(bits[i:i + 32] != _SYNCBITS))
        if d <= max_err:
            out.append((i + 32, 0))
        elif 32 - d <= max_err:
            out.append((i + 32, 1))
    return out


def _read_word(bits, i):
    v = 0
    for k in range(32):
        v = (v << 1) | int(bits[i + k])
    return v


def decode(soft, use_soft=True, max_sync_err=2):
    """soft: per-symbol soft values (>0 -> bit 0). -> list of message dicts."""
    soft = np.asarray(soft, float)
    bits = (soft < 0).astype(np.int64)
    inv = 1 - bits
    msgs, seen = [], set()
    for start, pol in _find_syncs(bits, max_sync_err):
        if start in seen or start + BATCH_WORDS * 32 > len(bits):
            continue
        seen.add(start)
        cur = None
        for wi in range(BATCH_WORDS):
            i = start + wi * 32
            frame = wi // 2
            if use_soft:
                s = soft[i:i + 32]
                info21, ok = bch_soft_decode(-s if pol else s)
            else:
                info21, _, ok = bch_hard_decode(
                    _read_word(inv if pol else bits, i))
            if not ok:
                continue
            if info21 == IDLE_INFO:
                if cur:
                    msgs.append(_finish(cur, pol))
                    cur = None
                continue
            if ((info21 >> 20) & 1) == 0:           # address codeword
                if cur:
                    msgs.append(_finish(cur, pol))
                addr18 = (info21 >> 2) & 0x3FFFF
                cur = {"address": (addr18 << 3) | frame,
                       "function": info21 & 3, "bits": []}
            else:                                   # message codeword
                if cur is not None:
                    pay = info21 & 0xFFFFF
                    cur["bits"] += [(pay >> (19 - k)) & 1 for k in range(20)]
        if cur:
            msgs.append(_finish(cur, pol))
    return _dedup(msgs)


def _finish(cur, pol):
    b = cur["bits"]
    return {"address": cur["address"], "function": cur["function"],
            "alpha": decode_alpha(b), "numeric": decode_numeric(b),
            "polarity": pol}


def _dedup(msgs):
    out, seen = [], set()
    for m in msgs:
        key = (m["address"], m["function"], m["alpha"])
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out


# ---- detector (shared front end, confirm-only) ---------------------------
def detect(x, fs, bauds=BAUDS, max_err=2, min_syncs=1):
    """Is POCSAG present? Correlate for the frame-sync word at each candidate
    baud and both polarities. -> dict(present, baud, polarity, n_syncs, confidence)."""
    disc = discriminate(x)
    best = {"present": False, "baud": None, "polarity": None,
            "n_syncs": 0, "confidence": 0.0}
    for baud in bauds:
        sps = int(round(fs / baud))
        if sps < 2 or len(disc) < 64 * sps:
            continue
        soft = symbol_soft(disc, sps, estimate_phase(disc, sps))
        bits = (soft < 0).astype(np.int64)
        syncs = _find_syncs(bits, max_err)
        if len(syncs) > best["n_syncs"]:
            best = {"present": len(syncs) >= min_syncs, "baud": baud,
                    "polarity": syncs[0][1] if syncs else None,
                    "n_syncs": len(syncs),
                    "confidence": min(1.0, len(syncs) / 4.0)}
    return best
