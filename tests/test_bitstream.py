"""The bitstream engine (atkdsp.h §14-17, docs/ATKDSP_PLAN.md §D phase 1a).

Three kinds of test, the plan's rules 1-3:
  - C == twin, on the same data (and block by block where state is carried);
  - known answers: planted truth in synthetic streams, published constants
    (CRC check values, NASA's dual-basis matrix, ATK's own pager encoder, the
    LTE tail-biting code the PBCH decoder already uses);
  - negative controls: random bits must come back "nothing" at the derived rate.

Run from the repo root:  python -m pytest tests -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
import atkdsp                                  # noqa: E402
from atkdsp import reference as ref            # noqa: E402

pytestmark = pytest.mark.skipif(not atkdsp.available(),
                                reason="atkdsp library not built (run build.sh / build.bat)")

rng = np.random.default_rng(20261028)


def rbits(n, p=0.5):
    return (rng.random(n) < p).astype(np.uint8)


def frames_stream(nframes, F, sync, counter_bits=8, parity=True, payload=None):
    """Synthetic frames: sync | counter | payload | one parity bit over the
    payload — planted truth for the width scan, sync discovery and WHT."""
    out = []
    for i in range(nframes):
        cnt = np.array([(i >> (counter_bits - 1 - k)) & 1 for k in range(counter_bits)], np.uint8)
        body = F - len(sync) - counter_bits - (1 if parity else 0)
        pay = rbits(body) if payload is None else payload(i, body)
        fr = [np.asarray(sync, np.uint8), cnt, pay]
        if parity:
            fr.append(np.array([pay.sum() & 1], np.uint8))
        out.append(np.concatenate(fr))
    return np.concatenate(out)


SYNC32 = np.array([int(c) for c in format(0x1ACFFC1D, "032b")], np.uint8)   # CCSDS ASM


# ============================================================================
# §14 bit primitives
# ============================================================================
def test_version_and_symbols():
    assert atkdsp.ABI_VERSION == 15
    assert atkdsp.version() == "0.14.0"
    lib = atkdsp.load()
    for name in ("atkdsp_bits_autocorr", "atkdsp_bm", "atkdsp_scrambler_search",
                 "atkdsp_viterbi_decode", "atkdsp_rs_decode", "atkdsp_bch_decode",
                 "atkdsp_blk_decode", "atkdsp_crc", "atkdsp_crc_recover"):
        assert getattr(lib, name, None) is not None


@pytest.mark.parametrize("n", [1, 63, 64, 65, 1000, 4099])
def test_autocorr_matches_twin(n):
    b = rbits(n)
    assert np.array_equal(atkdsp.bits_autocorr(b, 70), ref.bits_autocorr(b, 70), equal_nan=True)


def test_autocorr_sees_the_period():
    base = (np.random.default_rng(5).random(37) < 0.5).astype(np.uint8)
    b = np.tile(base, 40)
    ac = atkdsp.bits_autocorr(b, 100)
    assert ac[37] == 1.0 and ac[74] == 1.0
    assert np.all(np.abs(np.delete(ac[1:75], [36, 73])) < 0.7)


def test_xcorr_matches_twin_and_finds_the_word():
    b = rbits(5000)
    b[1234:1266] = SYNC32
    x = atkdsp.bits_xcorr(b, SYNC32)
    assert np.array_equal(x, ref.bits_xcorr(b, SYNC32))
    assert int(np.argmax(x)) == 1234 and x[1234] == 1.0


@pytest.mark.parametrize("flags", [0, 1, 2, 3])
def test_search_matches_twin(flags):
    b = rbits(20000)
    b[100:132] = SYNC32
    b[5000:5032] = 1 - SYNC32
    b[9000:9032] = SYNC32[::-1]
    b[13000:13032] = SYNC32 ^ np.eye(32, dtype=np.uint8)[3] ^ np.eye(32, dtype=np.uint8)[17]
    got = atkdsp.bits_search(b, SYNC32, 2, flags)
    assert got == ref.bits_search(b, SYNC32, 2, flags)
    pos = {p: (e, v) for p, e, v in got}
    assert pos[100] == (0, 0) and pos[13000] == (2, 0)
    assert (5000 in pos) == bool(flags & 1)
    assert (9000 in pos) == bool(flags & 2)


def test_search_counting_and_growing_past_the_first_buffer():
    b = np.zeros(10000, np.uint8)
    hits = atkdsp.bits_search(b, np.zeros(8, np.uint8), 0)
    assert len(hits) == 10000 - 7                         # past the binding's first 4096


def test_columns_match_twin():
    b = rbits(10007)
    for w, off in [(7, 0), (64, 3), (100, 99)]:
        got = atkdsp.bits_columns(b, w, off)
        exp = ref.bits_columns(b, w, off)
        for g, e in zip(got, exp):
            assert np.array_equal(g, e)


def test_width_scan_matches_twin_and_finds_the_frame():
    F = 191
    b = frames_stream(120, F, SYNC32)
    z, raw = atkdsp.bits_width_scan(b, 2, 600)
    zt, rawt = ref.bits_width_scan(b, 2, 600)
    np.testing.assert_allclose(z, zt, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(raw, rawt, rtol=1e-7)
    widths = np.arange(2, 601)
    thr = ref.norm_isf(1e-3 / widths.size)
    peak = float(np.nanmax(z))
    fundamental = int(widths[np.nonzero(z >= 0.5 * peak)[0][0]])
    assert fundamental == F
    assert z[2 * F - 2] > thr and z[3 * F - 2] > thr        # the harmonic rule


def test_width_scan_negative_control():
    z, _ = atkdsp.bits_width_scan(rbits(50000), 2, 1000)
    thr = ref.norm_isf(1e-3 / 999)
    assert np.nanmax(z) < thr
    assert abs(float(np.nanmean(z))) < 0.15 and 0.85 < float(np.nanstd(z)) < 1.15


def test_sync_discover_finds_an_unlisted_sync_word():
    F = 256
    b = frames_stream(80, F, SYNC32)
    b = np.concatenate([rbits(77), b])                  # the stream starts mid-frame
    got = atkdsp.bits_sync_discover(b, 16, 3)
    exp = ref.bits_sync_discover(b, 16, 3)
    assert len(got) == len(exp)
    for g, e in zip(got, exp):
        for k in ("first", "period", "length", "count", "flags"):
            assert g[k] == e[k], k
        assert np.array_equal(g["pattern"], e["pattern"])
        assert g["log10_pfa"] == pytest.approx(e["log10_pfa"], rel=1e-9, abs=1e-9)
    best = got[0]
    assert best["period"] == F and best["first"] % F == 77 % F
    s = "".join(map(str, best["pattern"]))
    assert "".join(map(str, SYNC32)) in s and best["count"] >= 78
    assert best["log10_pfa"] < -100 and not best["flags"]


def test_sync_discover_negative_control_and_idle_flag():
    assert atkdsp.bits_sync_discover(rbits(200000), 16, 3) == []
    idle = np.concatenate([np.tile([1, 0], 500), rbits(3000)]).astype(np.uint8)
    got = atkdsp.bits_sync_discover(idle, 16, 3)
    assert got and all(c["flags"] & atkdsp.SYNC_REPETITIVE for c in got)


def test_bit_stats_match_twin():
    for n in (100, 3000, 70000, 410000):
        b = rbits(n, 0.47)
        s = atkdsp.bits_stats(b)
        t = ref.bits_stats(b)
        for k in ("n", "ones", "chi2_bits", "maurer_L", "apen_m"):
            assert s[k] == t[k], k
        for k in ("run0", "run1", "ngram"):
            assert np.array_equal(s[k], t[k]), k
        for k in ("density", "chi2", "chi2_p", "maurer", "maurer_p", "apen", "apen_p"):
            assert s[k] == pytest.approx(t[k], rel=1e-9, abs=1e-12, nan_ok=True), k
        np.testing.assert_allclose(s["block_entropy"], t["block_entropy"], rtol=1e-12)


def test_apen_is_nist_sp800_22_worked_example():
    """NIST SP 800-22 rev 1a §2.12.4: e = 0100110101, m = 3:
    phi(3) = -1.643418, phi(4) = -1.834372, ApEn = 0.190954."""
    e = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1], np.uint8)
    assert ref._apen_phi(e, 3) == pytest.approx(-1.643418, abs=1e-6)
    assert ref._apen_phi(e, 4) == pytest.approx(-1.834372, abs=1e-6)
    ap = ref._apen_phi(e, 3) - ref._apen_phi(e, 4)
    assert ap == pytest.approx(0.190954, abs=1e-6)
    assert math.log(2) - ap == pytest.approx(0.502193, abs=1e-6)
    chi2 = 2 * 10 * (math.log(2) - ap)                     # 10.043859
    assert ref.igamc(4.0, chi2 / 2) == pytest.approx(0.261961, abs=1e-6)   # NIST's P-value


def test_bit_stats_see_structure_and_not_random():
    r = np.random.default_rng(1)                       # fixed: p-values are random variables
    s = atkdsp.bits_stats((r.random(500000) < 0.5).astype(np.uint8))
    assert s["maurer_p"] > 1e-3 and s["apen_p"] > 1e-3 and s["chi2_p"] > 1e-3
    # a 1000-bit repeat: ApEn and the word chi-square see it; Maurer's test
    # (L = 6 blocks) barely does — block repeats 167 blocks apart look ordinary
    s2 = atkdsp.bits_stats(np.tile((r.random(1000) < 0.5).astype(np.uint8), 500))
    assert s2["apen_p"] < 1e-6 and s2["chi2_p"] < 1e-6
    # a 40-bit repeat: every block recurs within 7 blocks, and Maurer sees that
    s3 = atkdsp.bits_stats(np.tile((r.random(40) < 0.5).astype(np.uint8), 12500))
    assert s3["maurer_p"] < 1e-6


def test_wht_matches_twin_and_finds_the_parity_column():
    F = 64
    b = frames_stream(400, F, SYNC32[:16], counter_bits=8)   # 16 sync, 8 counter, 39 payload, 1 parity
    cols = list(range(30, 40)) + [F - 1]                      # payload bits 30..39 and the parity
    spec, rows = atkdsp.bits_wht(b, F, cols)
    st, rt = ref.bits_wht(b, F, cols)
    assert rows == rt == 400 and np.array_equal(spec, st)
    # the parity bit is the XOR of all 39 payload bits, so no 10-column subset
    # holds the relation: every coefficient but [0] stays near sqrt(rows)
    assert spec[0] == 400 and np.abs(spec[1:]).max() < 5 * math.sqrt(400)


def test_wht_sees_a_planted_xor_relation():
    F = 48
    rows = []
    for _ in range(500):
        r = rbits(F)
        r[40] = r[3] ^ r[7] ^ r[11]                            # a planted checksum bit
        rows.append(r)
    b = np.concatenate(rows)
    cols = [3, 7, 11, 40, 20, 21]
    spec, n = atkdsp.bits_wht(b, F, cols)
    assert spec[0b001111] == n                                 # cols 3^7^11^40 == 0 always
    others = np.delete(np.abs(spec), [0, 0b001111])
    assert others.max() < 5 * math.sqrt(n)


def test_walsh_matches_twin_and_orders_by_sequency():
    b = rbits(16 * 1024)
    seq, dy, k = atkdsp.bits_walsh(b, 1024, 0)
    st, dt, kt = ref.bits_walsh(b, 1024, 0)
    assert k == kt == 16 and np.array_equal(seq, st) and np.array_equal(dy, dt)
    # h(s) really is the Walsh function with s sign changes
    size = 64
    H = np.array([[(-1) ** bin(i & j).count("1") for j in range(size)] for i in range(size)])
    idx = ref._sequency_index(size)
    for s in range(size):
        assert int(np.count_nonzero(np.diff(H[idx[s]]) != 0)) == s
    assert dy[0] == pytest.approx(1.0)


def test_walsh_sees_a_dyadic_period():
    base = rbits(16)
    b = np.tile(base, 512)                    # x[i] = x[i XOR 16k] within each 256-block
    _, dy, _ = atkdsp.bits_walsh(b, 256, 0)
    assert dy[16] == pytest.approx(1.0) and dy[32] == pytest.approx(1.0)


def test_cyclic_matches_twin():
    b = np.tile(rbits(100), 200)
    tm, fq, k = atkdsp.bits_cyclic(b, 4096, 0)
    tt, ft, kt = ref.bits_cyclic(b, 4096, 0)
    assert k == kt == 4
    assert np.array_equal(tm, tt)
    np.testing.assert_allclose(fq, ft, rtol=1e-6, atol=1e-6)
    assert tm[0] == 1.0


def test_short_input_gives_nan_and_zero_blocks():
    seq, dy, k = atkdsp.bits_walsh(rbits(100), 256)
    assert k == 0 and np.all(np.isnan(seq))
    tm, fq, k = atkdsp.bits_cyclic(rbits(100), 256)
    assert k == 0 and np.all(np.isnan(fq))


# ============================================================================
# §15 scramblers and linear complexity
# ============================================================================
def atk_lfsr_sequence(taps, length, seed=1):
    """ATK siga.bits.lfsr_sequence, copied verbatim (minus its docstring) from
    D:\\Analyst_Toolkit\\ATK\\atk\\core\\siga\\bits.py so the mask convention is
    held to ATK's own code, not to a re-reading of it."""
    deg = max(1, int(taps).bit_length() - 1)
    state = int(seed) & ((1 << deg) - 1) or 1
    out = np.empty(int(length), dtype=np.uint8)
    for i in range(int(length)):
        bit = 0
        t = int(taps)
        s = state
        while t:
            if t & 1:
                bit ^= s & 1
            t >>= 1
            s >>= 1
        out[i] = state & 1
        state = ((state >> 1) | (bit << (deg - 1))) & ((1 << deg) - 1)
    return out


@pytest.mark.parametrize("mask,seed", [(0x89, 1), (0xC001, 0x1234), (0x800021, 0x5A5A5A),
                                       (0x100000009, 0xABCDEF01)])
def test_descramble_add_is_atk_lfsr(mask, seed):
    b = rbits(3000)
    st = np.array([seed], np.uint64)
    got = atkdsp.descramble_add(b, mask, st)
    assert np.array_equal(got, b ^ atk_lfsr_sequence(mask, b.size, seed))
    st2 = np.array([seed], np.uint64)
    assert np.array_equal(got, ref.descramble_add(b, mask, st2)) and st[0] == st2[0]


def test_registers_are_continuous_across_blocks():
    b = rbits(5000)
    for fn in ("descramble_add", "descramble_mult", "scramble_mult"):
        st = np.array([0x1F3], np.uint64)
        whole = getattr(atkdsp, fn)(b, 0x8E1, st)
        st = np.array([0x1F3], np.uint64)
        parts, i = [], 0
        for size in (1, 7, 239, 240, 241, 999, 4096):
            parts.append(getattr(atkdsp, fn)(b[i:i + size], 0x8E1, st))
            i += size
        assert np.array_equal(whole, np.concatenate(parts)[:b.size]), fn


def test_multiplicative_round_trip_self_sync_and_error_spread():
    mask = 0x800021                                   # x^23 + x^5 + 1 (delays 18, 23)
    d = rbits(4000)
    s = atkdsp.scramble_mult(d, mask, np.array([0x123], np.uint64))
    back = atkdsp.descramble_mult(s, mask, np.array([0x7777], np.uint64))   # wrong start state
    assert np.array_equal(back[23:], d[23:])          # locked after D bits
    s2 = s.copy()
    s2[2000] ^= 1
    back2 = atkdsp.descramble_mult(s2, mask, np.array([0], np.uint64))
    assert int(np.count_nonzero(back2[23:] != d[23:])) == 3     # one error per term
    assert np.array_equal(s, ref.scramble_mult(d, mask, np.array([0x123], np.uint64)))


def test_zero_register_and_dead_mask_are_refused():
    with pytest.raises(atkdsp.AtkDspError):
        atkdsp.descramble_add(rbits(10), 0x8E1, np.array([0], np.uint64))
    with pytest.raises(atkdsp.AtkDspError):
        atkdsp.descramble_add(rbits(10), 0x8E0, np.array([1], np.uint64))


# primitive polynomials (an LFSR's linear complexity is its degree only when
# its polynomial is irreducible): x^7+x^3+1, x^9+x^4+1, x^15+x^14+1,
# x^23+x^5+1, x^32+x^22+x^2+x+1
@pytest.mark.parametrize("mask", [0x89, 0x211, 0xC001, 0x800021, 0x100400007])
def test_berlekamp_massey_recovers_the_register_from_2L_bits(mask):
    D = mask.bit_length() - 1
    seed = int(rng.integers(1, 1 << D))
    seq = atk_lfsr_sequence(mask, 2 * D, seed)
    L, poly = atkdsp.bm(seq)
    assert (L, list(poly)) == tuple(x if i == 0 else list(x) for i, x in enumerate(ref.bm(seq)))
    assert L == D and atkdsp.bm_mask(poly) == mask
    # and the recovered register continues the sequence
    st = np.array([sum(int(seq[k]) << k for k in range(D))], np.uint64)
    cont = atkdsp.descramble_add(np.zeros(500, np.uint8), atkdsp.bm_mask(poly), st)
    assert np.array_equal(cont, atk_lfsr_sequence(mask, 500, seed))


def test_bm_profile_matches_twin_and_tracks_half_for_random():
    b = rbits(3000)
    prof = atkdsp.bm_profile(b)
    assert np.array_equal(prof, ref.bm_profile(b))
    assert abs(int(prof[-1]) - 1500) <= 3
    seq = atk_lfsr_sequence(0x800021, 3000, 77)
    assert atkdsp.bm_profile(seq)[-1] == 23


def _framed_payload(nframes, F=200):
    """Framed, structured plaintext: a sync, a counter and ASCII text — the
    kind of stream that has structure for a descrambler to reveal."""
    text = (b"ATK test frame: the quick brown fox jumps over the lazy dog. " * 4)
    out = []
    for i in range(nframes):
        body = np.unpackbits(np.frombuffer(text, np.uint8))[: F - 40]
        cnt = np.array([(i >> (7 - k)) & 1 for k in range(8)], np.uint8)
        out.append(np.concatenate([SYNC32, cnt, body]))
    return np.concatenate(out)


def test_scrambler_search_finds_a_multiplicative_scrambler():
    d = _framed_payload(120)
    mask = 0x8011                                         # x^15 + x^4 + 1
    s = atkdsp.scramble_mult(d, mask, np.array([0x4321], np.uint64))
    got, gid = atkdsp.scrambler_search(s, atkdsp.SCRAM_MULT, 16, 5)
    assert got and got[0]["poly"] == mask and got[0]["log10_pfa"] < -10
    exp, gidt = ref.scrambler_search(s, ref.SCRAM_MULT, 16, 5)
    assert gid == gidt
    assert [(c["poly"], c["g"], c["log10_pfa"]) for c in got] == \
        [(c["poly"], c["g"], c["log10_pfa"]) for c in exp]


def test_scrambler_search_finds_an_additive_scrambler_and_its_seed():
    d = np.concatenate([np.zeros(300, np.uint8), _framed_payload(60), np.zeros(200, np.uint8)])
    mask = 0xC001                                         # DVB PRBS x^15 + x^14 + 1
    seed = 0x2A5D
    s = atkdsp.descramble_add(d, mask, np.array([seed], np.uint64))
    got, gid = atkdsp.scrambler_search(s, atkdsp.SCRAM_ADD, 16, 3)
    assert got and got[0]["poly"] == mask and got[0]["seeded"] == 1
    st = np.array([got[0]["seed"]], np.uint64)
    assert np.array_equal(atkdsp.descramble_add(s, mask, st), d)
    exp, _ = ref.scrambler_search(s, ref.SCRAM_ADD, 16, 3)
    assert [(c["poly"], c["seed"], c["seeded"], c["g"]) for c in got] == \
        [(c["poly"], c["seed"], c["seeded"], c["g"]) for c in exp]


def test_scrambler_search_negative_controls():
    got, _ = atkdsp.scrambler_search(rbits(30000), atkdsp.SCRAM_MULT, 20, 5)
    assert got == []                                       # random bits: no scrambler
    got, _ = atkdsp.scrambler_search(_framed_payload(120), atkdsp.SCRAM_MULT, 16, 5)
    assert got == []                                       # never scrambled: doing nothing wins


# ============================================================================
# §16 FEC
# ============================================================================
LTE = (0o133, 0o171, 0o165)


def test_conv_encode_tailbiting_is_the_lte_encoder():
    for n in (40, 57, 100):
        b = rbits(n)
        assert np.array_equal(atkdsp.conv_encode(7, LTE, b, atkdsp.CONV_TAILBITE),
                              ref.lte_conv_encode(b).astype(np.uint8))


def test_viterbi_tailbiting_matches_the_lte_decoder():
    for _ in range(20):
        b = rbits(40)
        c = ref.lte_conv_encode(b).astype(np.float64)
        llr = (1 - 2 * c) * 2.0 + rng.standard_normal(c.size) * 1.1
        v = atkdsp.Viterbi(7, LTE, mode=atkdsp.CONV_TAILBITE, max_bits=64)
        got = v.decode(llr.astype(np.float32))
        assert np.array_equal(got, ref.lte_viterbi_tb(llr.astype(np.float32)))


CASES = [
    dict(K=7, polys=(0o171, 0o133), mode=0, punc=None, n=300),
    dict(K=7, polys=(0o171, 0o133), mode=0, punc=[[1, 1, 0], [1, 0, 1]], n=301),      # 3/4
    dict(K=9, polys=(0o557, 0o663, 0o711), mode=0, punc=None, n=200),
    dict(K=5, polys=(0o23, 0o35), mode=1, punc=None, n=77),
    dict(K=15, polys=(0o74000, 0o46321), mode=0, punc=None, n=60),
    dict(K=3, polys=(0o7, 0o5), mode=2, punc=None, n=500),
    dict(K=7, polys=(0o171, 0o133), mode=2, punc=[[1, 0, 1, 0, 1], [1, 1, 0, 1, 0]], n=900),  # 5/6
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"K{c['K']}m{c['mode']}p{c['punc'] is not None}")
def test_viterbi_matches_twin_and_round_trips(case):
    K, polys, mode, punc, n = case["K"], case["polys"], case["mode"], case["punc"], case["n"]
    b = rbits(n)
    code = atkdsp.conv_encode(K, polys, b, mode, punc)
    assert np.array_equal(code, ref.conv_encode(K, polys, b, mode, punc))
    llr = ((1 - 2 * code.astype(np.float32)) * 1.5
           + rng.standard_normal(code.size).astype(np.float32) * 0.9).astype(np.float32)
    v = atkdsp.Viterbi(K, polys, punc, mode, max_bits=n + 8)
    t = ref.Viterbi(K, polys, punc, mode, max_bits=n + 8)
    if mode == atkdsp.CONV_STREAM:
        got = np.concatenate([v.decode(llr[:333]), v.decode(llr[333:]), v.flush()])
        exp = np.concatenate([t.decode(llr), t.flush()])
        assert np.array_equal(got, exp)
        assert got.size == n
    else:
        got = v.decode(llr)
        exp = t.decode(llr)
        assert np.array_equal(got, exp)
        assert v.metric == pytest.approx(t.metric, abs=1e-6)
    clean = ((1 - 2 * code.astype(np.float32)) * 4).astype(np.float32)
    v.reset()
    dec = v.decode(clean)
    if mode == atkdsp.CONV_STREAM:
        dec = np.concatenate([dec, v.flush()])
    else:
        assert v.metric == pytest.approx(1.0)
    assert np.array_equal(dec, b)


def test_viterbi_stream_is_identical_however_it_is_chunked():
    b = rbits(4000)
    code = atkdsp.conv_encode(7, (0o171, 0o133), b, atkdsp.CONV_STREAM)
    llr = ((1 - 2 * code.astype(np.float32)) + rng.standard_normal(code.size).astype(np.float32) * 0.8)
    v = atkdsp.Viterbi(7, (0o171, 0o133), mode=atkdsp.CONV_STREAM)
    whole = np.concatenate([v.decode(llr), v.flush()])
    v.reset()
    parts, i = [], 0
    for size in (1, 7, 239, 240, 241, 999, 4096, 8192):
        parts.append(v.decode(llr[i:i + size]))
        i += size
    parts.append(v.flush())
    assert np.array_equal(whole, np.concatenate(parts))


def _ber(K, polys, ebn0_db, nbits, block=20000):
    """Soft-decision BER of a terminated code over BPSK/AWGN."""
    n = len(polys)
    rate = 1.0 / n
    sigma = math.sqrt(1.0 / (2 * rate * 10 ** (ebn0_db / 10)))
    v = atkdsp.Viterbi(K, polys, mode=atkdsp.CONV_TERM, max_bits=block)
    errs = tot = 0
    r = np.random.default_rng(7)
    while tot < nbits:
        b = (r.random(block) < 0.5).astype(np.uint8)
        c = atkdsp.conv_encode(K, polys, b, atkdsp.CONV_TERM)
        y = (1 - 2 * c.astype(np.float64)) + r.standard_normal(c.size) * sigma
        llr = (2 * y / sigma ** 2).astype(np.float32)
        errs += int(np.count_nonzero(v.decode(llr) != b))
        tot += block
    return errs / tot


def _union_bound_k7(ebn0_db):
    """Soft-decision union bound for K=7 r=1/2 (171, 133): sum B_d Q(sqrt(2 d R Eb/N0))
    with the code's textbook bit-weight spectrum (d_free = 10; B_10..B_20 =
    36, 211, 1404, 11633, 77433, 502690 at even d — Proakis)."""
    e = 10 ** (ebn0_db / 10)
    B = {10: 36, 12: 211, 14: 1404, 16: 11633, 18: 77433, 20: 502690}
    return sum(b * 0.5 * math.erfc(math.sqrt(2 * d * 0.5 * e) / math.sqrt(2)) for d, b in B.items())


def test_viterbi_ber_k7_rate_half_matches_theory_and_the_published_curve():
    """K=7 r=1/2, unquantised soft decisions. The literature puts BER 1e-5 at
    4.2-4.4 dB (uncoded BPSK: 9.6 dB); the crossing must lie inside that
    widened by the plan's 0.3 dB, and at 4.2 dB the measured BER must sit on
    the union bound (tight there; errors come in bursts of ~5, hence the band)."""
    assert _ber(7, (0o171, 0o133), 3.9, 2_000_000) > 1e-5
    assert _ber(7, (0o171, 0o133), 4.7, 4_000_000) < 1e-5
    ratio = _ber(7, (0o171, 0o133), 4.2, 4_000_000) / _union_bound_k7(4.2)
    assert 0.4 < ratio < 1.5, ratio


def test_viterbi_k9_rate_third_beats_k7_rate_half():
    """K=9 r=1/3 (557, 663, 711) has ~1 dB more coding gain than K=7 r=1/2."""
    e = 3.6
    assert _ber(9, (0o557, 0o663, 0o711), e, 600_000) < 0.2 * _ber(7, (0o171, 0o133), e, 600_000)


def test_viterbi_refuses_what_it_cannot_do():
    with pytest.raises(atkdsp.AtkDspError):
        atkdsp.Viterbi(16, (1, 3))
    with pytest.raises(atkdsp.AtkDspError):
        atkdsp.Viterbi(7, (0o171, 0o133), puncture=[[0, 1], [0, 1]])   # column 0 sends nothing
    v = atkdsp.Viterbi(7, (0o171, 0o133), max_bits=10)
    with pytest.raises(atkdsp.AtkDspError):
        v.decode(np.zeros(2 * (100 + 6), np.float32))      # longer than max_bits
    with pytest.raises(atkdsp.AtkDspError):
        v.decode(np.zeros(31, np.float32))                  # no step count makes 31


def test_viterbi_describe_and_stats():
    v = atkdsp.Viterbi(7, (0o171, 0o133), puncture=[[1, 1, 0], [1, 0, 1]], mode=atkdsp.CONV_STREAM)
    d = v.describe()
    assert "K=7" in d and "0171, 0133" in d and "punctured 3/4 [110 101]" in d and "depth 84" in d
    v.decode(np.ones(400, np.float32))
    s = v.stats()
    assert s["calls"] == 1 and s["in"] == 400


# ---- Reed-Solomon ------------------------------------------------------------------
CCSDS = dict(m=8, gfpoly=0x187, fcr=112, prim=11, nroots=32)


def test_ccsds_dual_basis_matrix_is_nasas():
    """The inverse of the conversion matrix the C uses must be the inverse
    transform NASA publishes (NTRS 19940004345, Eq. 6)."""
    T = ref.CCSDS_TAL
    def mul(x, M):
        v = 0
        for j in range(8):
            if (x >> j) & 1:
                v ^= M[7 - j]
        return v
    tal = [mul(x, T) for x in range(256)]
    inv = [0] * 256
    for x in range(256):
        inv[tal[x]] = x
    rows = [format(inv[1 << (7 - i)], "08b") for i in range(8)]
    assert rows == ["11000101", "01000010", "00101110", "11111101",
                    "11110000", "01111001", "10101100", "11001100"]


@pytest.mark.parametrize("dual", [False, True])
def test_ccsds_rs_corrects_t_errors_and_2t_erasures(dual):
    r = atkdsp.RS(dual_basis=dual, **CCSDS)
    t = ref.RS(dual_basis=dual, **CCSDS)
    assert (r.n, r.k) == (255, 223)
    for trial in range(30):
        data = rng.integers(0, 256, 223).astype(np.uint16)
        par = r.encode(data)
        assert np.array_equal(par, t.encode(data))
        cw = np.concatenate([data, par])
        bad = cw.copy()
        ne = trial % 17                                   # 0..16 errors
        pos = rng.choice(255, ne, replace=False)
        bad[pos] ^= rng.integers(1, 256, ne).astype(np.uint16)
        fixed, cnt, where = r.decode(bad)
        tf, tc, tw = t.decode(bad)
        assert cnt == tc == ne and np.array_equal(fixed, cw) and np.array_equal(tf, cw)
        assert sorted(where) == sorted(pos.tolist()) == sorted(tw)
    data = rng.integers(0, 256, 223).astype(np.uint16)
    cw = np.concatenate([data, r.encode(data)])
    er = rng.choice(255, 32, replace=False)
    bad = cw.copy()
    bad[er] = rng.integers(0, 256, 32).astype(np.uint16)
    fixed, cnt, _ = r.decode(bad, er)
    assert np.array_equal(fixed, cw) and cnt == 32
    er10 = er[:10]
    bad = cw.copy()
    bad[er10] ^= 0x55
    pos = [p for p in rng.choice(255, 40, replace=False) if p not in er10][:11]
    bad[pos] ^= 0x0F
    fixed, cnt, _ = r.decode(bad, er10)                   # 2*11 + 10 = 32
    assert np.array_equal(fixed, cw)


def test_rs_beyond_t_is_measured_not_assumed():
    """17 errors in CCSDS RS(255,223) exceed t = 16. The decoder must never
    hand back a 'correction' that is not a codeword, and how often it lands on
    a wrong codeword is measured (the bound is about 1/16!, effectively 0)."""
    r = atkdsp.RS(**CCSDS)
    wrong = failed = 0
    for _ in range(200):
        data = rng.integers(0, 256, 223).astype(np.uint16)
        cw = np.concatenate([data, r.encode(data)])
        bad = cw.copy()
        pos = rng.choice(255, 17, replace=False)
        bad[pos] ^= rng.integers(1, 256, 17).astype(np.uint16)
        fixed, cnt, _ = r.decode(bad)
        if cnt == atkdsp.E_DECODE:
            failed += 1
        elif not np.array_equal(fixed, cw):
            wrong += 1
    assert failed == 200 and wrong == 0


@pytest.mark.parametrize("params", [dict(m=4, gfpoly=0x13, fcr=1, prim=1, nroots=4),
                                    dict(m=8, gfpoly=0x11D, fcr=0, prim=1, nroots=16, pad=100),
                                    dict(m=10, gfpoly=0x409, fcr=1, prim=1, nroots=8, pad=900),
                                    dict(m=12, gfpoly=0x1053, fcr=5, prim=11, nroots=6, pad=4000)])
def test_rs_other_fields_match_twin(params):
    r = atkdsp.RS(**params)
    t = ref.RS(**params)
    tt = params["nroots"] // 2
    for _ in range(10):
        data = rng.integers(0, 1 << params["m"], r.k).astype(np.uint16)
        cw = np.concatenate([data, r.encode(data)])
        bad = cw.copy()
        pos = rng.choice(r.n, tt, replace=False)
        bad[pos] ^= rng.integers(1, 1 << params["m"], tt).astype(np.uint16)
        f1, c1, _ = r.decode(bad)
        f2, c2, _ = t.decode(bad)
        assert c1 == c2 == tt and np.array_equal(f1, cw) and np.array_equal(f2, cw)


# ---- BCH -------------------------------------------------------------------------------
def atk_pocsag_bch_encode(info21):
    """ATK's pager BCH encoder, copied from atk/core/pocsag_native.py."""
    BCH_GEN = 0x769
    def _reduce(reg):
        for i in range(30, 9, -1):
            if (reg >> i) & 1:
                reg ^= BCH_GEN << (i - 10)
        return reg & 0x3FF
    cw31 = (info21 << 10) | _reduce(info21 << 10)
    return (cw31 << 1) | (bin(cw31).count("1") & 1)


def test_bch_31_21_is_pocsags_code():
    b = atkdsp.BCH(5, 2)
    assert (b.n, b.k, b.t) == (31, 21, 2)
    g = int("".join(map(str, b.generator())), 2)
    assert g == 0x769
    for _ in range(50):
        info = int(rng.integers(0, 1 << 21))
        data = np.array([(info >> (20 - i)) & 1 for i in range(21)], np.uint8)
        code = b.encode(data)
        cw32 = atk_pocsag_bch_encode(info)
        assert int("".join(map(str, code)), 2) == cw32 >> 1
        bad = code.copy()
        pos = rng.choice(31, 2, replace=False)
        bad[pos] ^= 1
        fixed, cnt, where = b.decode(bad)
        assert cnt == 2 and np.array_equal(fixed, code) and sorted(where) == sorted(pos.tolist())


@pytest.mark.parametrize("m,t,sh", [(6, 2, 0), (7, 3, 10), (8, 4, 0), (10, 6, 300), (13, 5, 8000)])
def test_bch_corrects_t_and_matches_twin(m, t, sh):
    b = atkdsp.BCH(m, t, sh)
    r = ref.BCH(m, t, sh)
    assert (b.n, b.k) == (r.n, r.k)
    assert np.array_equal(b.generator(), r.generator())
    for _ in range(5):
        data = rbits(b.k)
        code = b.encode(data)
        assert np.array_equal(code, r.encode(data))
        bad = code.copy()
        pos = rng.choice(b.n, t, replace=False)
        bad[pos] ^= 1
        f1, c1, _ = b.decode(bad)
        f2, c2, _ = r.decode(bad)
        assert c1 == c2 == t and np.array_equal(f1, code) and np.array_equal(f2, code)


def test_bch_beyond_t_is_measured():
    """3 errors in BCH(31,21): about 1 in 6 such words decode to a wrong
    codeword (a sphere of radius 2 covers 466 of the 1024 syndromes, half of
    which are not 3-error words' own) — the rate is measured and the C and the
    twin must agree on every word."""
    b = atkdsp.BCH(5, 2)
    r = ref.BCH(5, 2)
    wrong = failed = 0
    for _ in range(400):
        data = rbits(21)
        code = b.encode(data)
        bad = code.copy()
        bad[rng.choice(31, 3, replace=False)] ^= 1
        f, c, _ = b.decode(bad)
        f2, c2, _ = r.decode(bad)
        assert c == c2 and np.array_equal(f, f2)
        if c == atkdsp.E_DECODE:
            failed += 1
        else:
            assert not np.array_equal(f, code)
            wrong += 1
    assert wrong + failed == 400 and 0.2 < wrong / 400 < 0.6


# ---- small block codes ---------------------------------------------------------------------
@pytest.mark.parametrize("name,dmin", [("hamming(7,4)", 3), ("hamming(15,11)", 3), ("hamming(31,26)", 3),
                                       ("hamming(8,4)", 4), ("hamming(16,11)", 4),
                                       ("golay(23,12)", 7), ("golay(24,12)", 8)])
def test_small_codes_dmin_and_correction(name, dmin):
    G = atkdsp.SMALL_CODES[name]()
    c = atkdsp.BlockCode(G)
    t = ref.BlockCode(G)
    assert c.dmin == t.dmin == dmin
    tc = (dmin - 1) // 2
    for _ in range(30):
        data = rbits(c.k)
        code = c.encode(data)
        assert np.array_equal(code, t.encode(data))
        bad = code.copy()
        bad[rng.choice(c.n, tc, replace=False)] ^= 1
        f, d, cnt = c.decode(bad)
        f2, d2, cnt2 = t.decode(bad)
        assert cnt == cnt2 == tc and np.array_equal(f, code) and np.array_equal(d, data)
        assert np.array_equal(d2, data)


def test_even_dmin_codes_say_when_a_word_is_equidistant():
    c = atkdsp.BlockCode(atkdsp.SMALL_CODES["golay(24,12)"]())
    data = rbits(12)
    code = c.encode(data)
    bad = code.copy()
    bad[[0, 5, 9, 17]] ^= 1                                # weight 4 = dmin/2: always a tie
    _, _, cnt = c.decode(bad)
    assert cnt == atkdsp.E_DECODE


def test_blk_handles_a_non_systematic_generator():
    G = atkdsp.SMALL_CODES["hamming(7,4)"]()
    G2 = G.copy()
    G2[0] ^= G2[1]; G2[2] ^= G2[3]; G2[3] ^= G2[0]
    c = atkdsp.BlockCode(G2)
    data = rbits(4)
    code = c.encode(data)
    bad = code.copy(); bad[4] ^= 1
    f, d, cnt = c.decode(bad)
    assert cnt == 1 and np.array_equal(d, data)


# ============================================================================
# §17 the CRC engine and CRC recovery
# ============================================================================
def test_every_catalogue_check_value():
    cat = atkdsp.crc_catalog()
    assert len(cat) == len(ref.CRC_CATALOG) >= 100
    msg = atkdsp.ascii_bits(b"123456789")
    for (name, spec), e in zip(cat, ref.CRC_CATALOG):
        assert name == e[0]
        assert (spec["width"], spec["poly"], spec["init"], spec["refin"], spec["refout"],
                spec["xorout"], spec["check"]) == e[1:]
        assert atkdsp.crc(msg, spec) == spec["check"], name
        assert ref.crc(msg, spec) == spec["check"], name


def test_crc_table_path_equals_bitwise_and_twin():
    for name in ("CRC-32/ISO-HDLC", "CRC-16/IBM-SDLC", "CRC-64/XZ", "CRC-24/BLE", "CRC-8/ROHC",
                 "CRC-12/UMTS", "CRC-5/USB"):
        spec = atkdsp.crc_spec(name)
        for n in (8 * 300, 8 * 300 + 5, 8 * 31, 8 * 32 + 1, 7):
            b = rbits(n)
            assert atkdsp.crc(b, spec) == ref.crc(b, spec), (name, n)


def test_crc24a_matches_the_lte_kernel():
    spec = atkdsp.crc_spec("CRC-24/LTE-A")
    for n in (40, 100, 1000):
        b = rbits(n)
        parity = atkdsp.lte_crc24a(b.astype(np.int8))
        assert int("".join(str(int(x)) for x in parity[-24:]), 2) == atkdsp.crc(b, spec)


def make_frames(spec, nframes, lengths, header=0, trailer=0, layout=0, same_header=False):
    w = spec["width"]
    hdr0 = rbits(header)
    frames = []
    for i in range(nframes):
        L = int(lengths[i])
        msg = rbits(L)
        v = ref.crc(msg, spec)
        if layout:
            fb = np.concatenate([np.unpackbits(np.array([(v >> (8 * k)) & 0xff], np.uint8))
                                 for k in range(w // 8)])
        else:
            fb = np.array([(v >> (w - 1 - j)) & 1 for j in range(w)], np.uint8)
        hdr = hdr0 if same_header else rbits(header)
        frames.append(np.concatenate([hdr, msg, fb, rbits(trailer)]))
    return frames


def frames_needed(w, trials=24.0, u=3):
    """The smallest frame count whose equations beat log10(pfa = 1e-3) with
    half a decade to spare — the arithmetic the header documents."""
    n = 4
    while (n - u) * w * math.log10(2) < 3.5 + math.log10(trials):
        n += 1
    return n


def _same(spec, got):
    return all(got[k] == spec[k] for k in ("width", "poly", "init", "refin", "refout", "xorout"))


@pytest.mark.parametrize("entry", ref.CRC_CATALOG, ids=lambda e: e[0])
def test_crc_recover_every_catalogue_crc_from_random_lengths(entry):
    spec = ref.crc_spec(entry[0])
    w = spec["width"]
    nf = frames_needed(w)
    lengths = 8 * rng.integers(6, 40, nf)                 # whole bytes: refin needs them
    while len(set(lengths.tolist())) < nf:
        lengths = 8 * rng.integers(6, 40, nf)
    frames = make_frames(spec, nf, lengths)
    got = atkdsp.crc_recover(frames, width=w)
    assert got, entry[0]
    best = got[0]
    assert _same(spec, best["spec"]), (entry[0], best)
    # g with an (x+1) factor: h = g/(x+1) has h x^L == h (mod g) for every L,
    # so (init ^ h, xorout ^ h) is the SAME CRC on every message — init cannot
    # be known from frames, and the catalogue's own set is named instead
    even = bin(spec["poly"] | (1 << w)).count("1") % 2 == 0
    assert best["init_known"] == (0 if even else 1), (entry[0], best)
    assert best["frames_ok"] == nf and best["log10_pfa"] <= -3
    assert best["spec"]["check"] == spec["check"]
    idx = [e[0] for e in ref.CRC_CATALOG].index(entry[0])
    assert best["catalog"] == idx


@pytest.mark.parametrize("name", ["CRC-16/XMODEM", "CRC-16/IBM-SDLC", "CRC-32/ISO-HDLC",
                                  "CRC-24/OPENPGP", "CRC-8/SMBUS", "CRC-15/CAN"])
def test_crc_recover_with_a_7_byte_header_excluded(name):
    spec = ref.crc_spec(name)
    w = spec["width"]
    nf = frames_needed(w, trials=24.0 * 57)
    lengths = 8 * rng.integers(6, 30, nf)
    frames = make_frames(spec, nf, lengths, header=56)
    got = atkdsp.crc_recover(frames, width=w, max_header=64)
    assert got and got[0]["header_bits"] == 56 and _same(spec, got[0]["spec"]), got[:1]


def test_crc_recover_little_endian_field_and_trailer():
    spec = ref.crc_spec("CRC-32/ISO-HDLC")
    frames = make_frames(spec, 5, 8 * rng.integers(8, 30, 5), layout=1)
    got = atkdsp.crc_recover(frames, width=32)
    assert got and got[0]["layout"] == 1 and _same(spec, got[0]["spec"])
    spec = ref.crc_spec("CRC-16/XMODEM")
    frames = make_frames(spec, 6, 8 * rng.integers(8, 30, 6), trailer=8)
    got = atkdsp.crc_recover(frames, width=16, pos=atkdsp.CRC_SEARCH)
    assert got and got[0]["trailer_bits"] == 8 and _same(spec, got[0]["spec"])


def test_crc_recover_unknown_width_and_bit_granular_frames():
    for name in ("CRC-16/XMODEM", "CRC-32/MPEG-2", "CRC-11/FLEXRAY"):
        spec = ref.crc_spec(name)
        w = spec["width"]
        nf = frames_needed(w, trials=62 * 24.0)
        lengths = rng.integers(50, 300, nf)                  # not whole bytes
        frames = make_frames(spec, nf, lengths)
        got = atkdsp.crc_recover(frames, width=0)
        assert got and _same(spec, got[0]["spec"]), (name, got[:1])


def test_crc_recover_equal_lengths_fold_init_and_name_the_catalogue_set():
    spec = ref.crc_spec("CRC-16/IBM-3740")
    frames = make_frames(spec, 5, [120] * 5)
    got = atkdsp.crc_recover(frames, width=16)
    assert got
    best = got[0]
    assert best["init_known"] == 0 and best["spec"]["poly"] == 0x1021
    assert best["catalog"] >= 0 and _same(spec, best["spec"])


def test_crc_recover_negative_control():
    frames = [rbits(int(n)) for n in rng.integers(100, 300, 8)]
    assert atkdsp.crc_recover(frames, width=0, max_header=16) == []
    assert atkdsp.crc_recover(frames, width=16, max_header=64) == []


@pytest.mark.parametrize("name,kw", [("CRC-16/KERMIT", dict(width=16)),
                                     ("CRC-8/AUTOSAR", dict(width=8)),
                                     ("CRC-16/XMODEM", dict(width=16, max_header=12)),
                                     ("CRC-24/OPENPGP", dict(width=24))])
def test_crc_recover_matches_twin(name, kw):
    spec = ref.crc_spec(name)
    nf = frames_needed(spec["width"], trials=24.0 * 13)
    hdr = 8 if kw.get("max_header") else 0
    frames = make_frames(spec, nf, 8 * rng.integers(3, 9, nf), header=hdr)
    got = atkdsp.crc_recover(frames, **kw)
    exp = ref.crc_recover(frames, **kw)
    assert len(got) == len(exp) >= 1
    for g, e in zip(got, exp):
        for k in ("layout", "header_bits", "trailer_bits", "init_known", "frames_ok", "catalog"):
            assert g[k] == e[k], k
        assert g["spec"] == e["spec"]
        assert g["log10_pfa"] == pytest.approx(e["log10_pfa"], abs=1e-9)


# ---- what frames cannot decide (atkdsp.h §17), pinned down -------------------------
def test_x_plus_one_factor_makes_init_and_xorout_trade_exactly():
    """g = x^16 + x^12 + x^5 + 1 = (x+1) q(x). With h = q as a 16-bit value,
    (init ^ h, xorout ^ h) is the same CRC as (init, xorout) for EVERY message
    length — the reason crc_recover reports init_known = 0 for such g."""
    g = (1 << 16) | 0x1021
    q, r = ref._pdiv(g, 0b11)
    assert r == 0
    base = atkdsp.crc_spec("CRC-16/XMODEM")
    twin = dict(base, init=base["init"] ^ q, xorout=base["xorout"] ^ q)
    for n in (0, 1, 7, 8, 13, 100, 1001):
        b = rbits(n)
        assert atkdsp.crc(b, base) == atkdsp.crc(b, twin)


def test_a_constant_trailer_is_absorbed_and_reported_as_an_alternative():
    """Bits after the field that are the same in every frame (a closing flag)
    make the frames ALSO a valid CRC whose field ends later, with an adjusted
    xorout (for an unreflected CRC the algebra is exact). Both are reported;
    the catalogue set ranks first."""
    flag = np.array([0, 1, 1, 1, 1, 1, 1, 0], np.uint8)
    spec = atkdsp.crc_spec("CRC-16/XMODEM")
    frames = [np.concatenate([f, flag]) for f in make_frames(spec, 6, 8 * rng.integers(8, 30, 6))]
    got = atkdsp.crc_recover(frames, width=16, pos=atkdsp.CRC_SEARCH, cap=64)
    assert got[0]["trailer_bits"] == 8 and _same(spec, got[0]["spec"])
    assert got[0]["alternatives"] >= 1 and any(c["trailer_bits"] < 8 for c in got)
    spec = atkdsp.crc_spec("CRC-16/IBM-SDLC")            # HDLC's own FCS, reflected
    frames = [np.concatenate([f, flag]) for f in make_frames(spec, 6, 8 * rng.integers(8, 30, 6))]
    got = atkdsp.crc_recover(frames, width=16, pos=atkdsp.CRC_SEARCH, cap=64)
    assert got[0]["trailer_bits"] == 8 and _same(spec, got[0]["spec"])


def test_unknown_width_reports_the_factor_widths_too():
    spec = atkdsp.crc_spec("CRC-16/XMODEM")              # (x+1) times a degree-15 factor
    nf = frames_needed(16, trials=62 * 24.0)
    frames = make_frames(spec, nf, rng.integers(60, 200, nf))
    got = atkdsp.crc_recover(frames, width=0, cap=32)
    assert _same(spec, got[0]["spec"]) and got[0]["catalog"] >= 0
    widths = {c["spec"]["width"] for c in got}
    assert 15 in widths                                  # the factor is a CRC of the frames too


# ---- regressions from the independent review (2026-10-28) ---------------------------
def test_viterbi_stream_short_call_after_a_long_one():
    """out_max was nllr + n, but a call can release nllr + depth - 1 bits: a
    2-LLR call right after 166 overflowed the caller's buffer, and the failed
    call left the ring corrupt for the next. Now out_max is the true bound."""
    g = (0o171, 0o133)
    b = rbits(3000)
    code = atkdsp.conv_encode(7, g, b, atkdsp.CONV_STREAM)
    llr = (1 - 2 * code.astype(np.float32))
    v = atkdsp.Viterbi(7, g, mode=atkdsp.CONV_STREAM)
    t = ref.Viterbi(7, g, mode=ref.CONV_STREAM)
    got, exp, i = [], [], 0
    for size in (166, 2, 1, 1, 83, 2, 500, 3, 2000, 1, 1, 1000, 10 ** 6):
        got.append(v.decode(llr[i:i + size]))
        exp.append(t.decode(llr[i:i + size]))
        i += size
    got.append(v.flush()); exp.append(t.flush())
    assert np.array_equal(np.concatenate(got), np.concatenate(exp))
    assert np.array_equal(np.concatenate(got), b)


def test_viterbi_create_refuses_sizes_that_would_wrap():
    with pytest.raises(atkdsp.AtkDspError):
        atkdsp.Viterbi(7, (0o171, 0o133), max_bits=2 ** 62)


def test_viterbi_describe_with_eight_outputs_and_a_64_column_puncture():
    p = np.ones((8, 64), np.uint8)
    p[3, 17] = 0
    v = atkdsp.Viterbi(5, [0o23, 0o35, 0o27, 0o31, 0o37, 0o25, 0o33, 0o21], puncture=p,
                       mode=atkdsp.CONV_STREAM)
    d = v.describe()
    assert "punctured 64/511" in d and d.count("1") > 400


def test_degree_63_scrambler_mask():
    mask = (1 << 63) | 0b11
    b = rbits(500)
    s1, s2 = np.array([0x123456789], np.uint64), np.array([0x123456789], np.uint64)
    assert np.array_equal(atkdsp.descramble_add(b, mask, s1), ref.descramble_add(b, mask, s2))
    assert s1[0] == s2[0]


def test_degenerate_block_code_builds_quickly_and_says_what_it_cannot_correct():
    import time
    G = np.concatenate([np.eye(12, dtype=np.uint8), np.zeros((12, 20), np.uint8)], axis=1)
    t0 = time.perf_counter()
    c = atkdsp.BlockCode(G)
    assert time.perf_counter() - t0 < 20 and c.dmin == 1
    data = rbits(12)
    code = c.encode(data)
    bad = code.copy()
    bad[12:30] = 1                                       # weight 18 in the zero part
    _, _, cnt = c.decode(bad)
    assert cnt == atkdsp.E_DECODE                        # beyond the bounded table
    _, d, cnt = c.decode(code)
    assert cnt == 0 and np.array_equal(d, data)
