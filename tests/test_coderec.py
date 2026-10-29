"""Naming the code (atkdsp.h §18, docs/ATKDSP_PLAN.md §D4, phase 1b).

The plan's acceptance: synthetic streams from every code in D3, at BER 0, 1e-4
and 1e-3, give back (n, k) — or rate, K and generators — and interleaved
variants their dimensions; random bits give nothing. The evidence rule (Bill,
2026-10-04): every named code carries its log10_pfa and decode -> re-encode
agreement, and beyond the method's reach the answer is "no code found", with
the estimated BER when there is one.

Three kinds of test, as in phase 1a:
  - C == twin (reference.py §18), field for field, on small cases;
  - known answers: planted codes at the plan's BERs, and the measured limits
    the header states;
  - negative controls: random bits, linear recurrences, a convolutional stream
    offered as a block code.

Run from the repo root:  python -m pytest tests -q
"""

from __future__ import annotations

import math
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
import atkdsp                                  # noqa: E402
from atkdsp import reference as ref            # noqa: E402

pytestmark = pytest.mark.skipif(not atkdsp.available(),
                                reason="atkdsp library not built (run build.sh / build.bat)")

BERS = (0.0, 1e-4, 1e-3)


def gen(seed):
    return np.random.default_rng(seed)


def bsc(x, p, r):
    return (x ^ (r.random(x.size) < p)).astype(np.uint8)


def block_stream(G, nwords, r, skip=5):
    k = G.shape[0]
    return (r.integers(0, 2, (nwords, k)).astype(np.uint8) @ G % 2).astype(np.uint8).ravel()[skip:]


def bch_stream(bc, nwords, r, skip=3):
    return np.concatenate([bc.encode(r.integers(0, 2, bc.k).astype(np.uint8))
                           for _ in range(nwords)])[skip:]


def sym_bits(sym, m, lsb=False):
    sh = np.arange(m) if lsb else np.arange(m)[::-1]
    return ((np.asarray(sym, np.int64)[:, None] >> sh) & 1).astype(np.uint8).ravel()


def rs_stream(rs, m, nwords, r, lsb=False):
    words = []
    for _ in range(nwords):
        d = r.integers(0, 1 << m, rs.k)
        words.append(np.concatenate([d, rs.encode(d)]))
    return np.concatenate([sym_bits(w, m, lsb) for w in words])


def interleave(bits, rows, cols):
    """Written by rows (rows x cols), read by columns."""
    nb = bits.size // (rows * cols)
    return bits[:nb * rows * cols].reshape(nb, rows, cols).transpose(0, 2, 1).reshape(-1)


def systematic(G):
    G = (np.asarray(G, np.uint8) & 1).copy()
    k, n = G.shape
    r = 0
    for c in range(n):
        p = [i for i in range(r, k) if G[i, c]]
        if not p:
            continue
        G[[r, p[0]]] = G[[p[0], r]]
        for i in range(k):
            if i != r and G[i, c]:
                G[i] ^= G[r]
        r += 1
        if r == k:
            break
    return G


def assert_same(c, t):
    """C dict == twin dict, every field (floats to 1e-9 relative)."""
    for key, x in c.items():
        y = t[key]
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            assert (x is None) == (y is None), key
            if x is not None:
                assert np.array_equal(np.asarray(x), np.asarray(y)), key
        elif isinstance(x, float) or isinstance(y, float):
            if math.isnan(x) or math.isnan(y):
                assert math.isnan(x) and math.isnan(y), (key, x, y)
            else:
                assert y == pytest.approx(x, rel=1e-9, abs=1e-12), key
        else:
            assert x == y, (key, x, y)


def assert_evidence(r, p, pfa=1e-3):
    """The evidence rule: significant, and decoding changes about what the
    channel did (bits kept >= 1 - 4 BER - slack)."""
    assert r["found"]
    assert r["log10_pfa"] <= math.log10(pfa)
    assert r["agreement"] >= 1.0 - 4.0 * p - 2e-4, r


# ============================================================================
# the library
# ============================================================================
def test_version_and_symbols():
    assert atkdsp.ABI_VERSION >= 16            # §18 arrived in 16 (v0.15.0)
    lib = atkdsp.load()
    for name in ("atkdsp_gf2_rank_scan", "atkdsp_block_code_id", "atkdsp_conv_code_id",
                 "atkdsp_rs_code_id", "atkdsp_interleaver_id"):
        assert getattr(lib, name, None) is not None


def test_argument_errors():
    b = np.zeros(1000, np.uint8)
    with pytest.raises(atkdsp.AtkDspError):
        atkdsp.block_code_id(b, 4, 600)
    with pytest.raises(atkdsp.AtkDspError):
        atkdsp.conv_code_id(b, 9, 7)
    with pytest.raises(atkdsp.AtkDspError):
        atkdsp.rs_code_id(b, 0, 9)
    with pytest.raises(atkdsp.AtkDspError):
        atkdsp.interleaver_id(b, 2, 64)


# ============================================================================
# C == twin
# ============================================================================
def test_rank_scan_matches_twin_and_shows_2n():
    r = gen(1)
    s = block_stream(atkdsp.SMALL_CODES["hamming(7,4)"](), 300, r, skip=2)
    c = atkdsp.gf2_rank_scan(s, 2, 20, 8)
    assert np.array_equal(c, ref.gf2_rank_scan(s, 2, 20, 8))
    # width 7: deficiency 3 on the aligned offset only; width 14 = 2n: at least
    # 3 at every offset (a whole codeword in every row), 6 when aligned
    assert c[7 - 2, 5] == 3 and sorted(c[7 - 2, :7])[-2] < 3
    assert (c[14 - 2, :8] >= 3).all() and c[14 - 2, 5] == 6


@pytest.mark.parametrize("case", ["h74-0", "h74-1e-3", "h74-1e-2", "g24-1e-3", "rnd16-1e-3",
                                  "short25-1e-3", "random", "lfsr"])
def test_block_matches_twin(case):
    r = gen(zlib.crc32(case.encode()) % 1000)      # stable across runs
    nhi = 32
    if case.startswith("h74"):
        s = block_stream(atkdsp.SMALL_CODES["hamming(7,4)"](), 400, r)
    elif case.startswith("g24"):
        s = block_stream(atkdsp.SMALL_CODES["golay(24,12)"](), 400, r)
    elif case.startswith("rnd16"):
        G = np.hstack([np.eye(8, dtype=np.uint8), r.integers(0, 2, (8, 8)).astype(np.uint8)])
        s, nhi = block_stream(G, 600, r, skip=3), 24
    elif case.startswith("short25"):
        s = bch_stream(atkdsp.BCH(5, 2, 6), 300, r, skip=2)
    elif case == "random":
        s, nhi = r.integers(0, 2, 6000).astype(np.uint8), 24
    else:
        st = [1] + [0] * 14
        while len(st) < 20000:
            st.append(st[-15] ^ st[-14])
        s, nhi = np.array(st, np.uint8), 24
    p = float(case.split("-", 1)[1]) if "-" in case and case[-1].isdigit() else 0.0
    x = bsc(s, p, r)
    assert_same(atkdsp.block_code_id(x, 4, nhi), ref.block_code_id(x, 4, nhi))


@pytest.mark.parametrize("K,polys,p", [(5, [0o23, 0o33], 0.0), (5, [0o23, 0o33], 1e-2),
                                       (3, [0o7, 0o5], 1e-2), (4, [0o13, 0o15, 0o17], 1e-2),
                                       (0, None, 0.0)])
def test_conv_matches_twin(K, polys, p):
    r = gen(K * 10 + int(p * 1000))
    if K:
        s = atkdsp.conv_encode(K, polys, r.integers(0, 2, 1500).astype(np.uint8), atkdsp.CONV_STREAM)[3:]
    else:
        s = r.integers(0, 2, 4000).astype(np.uint8)
    x = bsc(s, p, r)
    assert_same(atkdsp.conv_code_id(x, 3, 7), ref.conv_code_id(x, 3, 7))


@pytest.mark.parametrize("m,gf,fcr,prim,nr,lsb,p", [(4, 0x13, 0, 1, 4, False, 3e-3),
                                                    (4, 0x19, 1, 1, 4, True, 0.0),
                                                    (6, 0x43, 1, 1, 6, False, 3e-3)])
def test_rs_matches_twin(m, gf, fcr, prim, nr, lsb, p):
    r = gen(m * 100 + nr)
    rs = atkdsp.RS(m, gf, fcr, prim, nr, 0, False)
    x = bsc(rs_stream(rs, m, 120, r, lsb), p, r)
    assert_same(atkdsp.rs_code_id(x, 0, m, rs.n), ref.rs_code_id(x, 0, m, rs.n))
    y = r.integers(0, 2, 8000).astype(np.uint8)
    assert_same(atkdsp.rs_code_id(y, 0, m, rs.n), ref.rs_code_id(y, 0, m, rs.n))


@pytest.mark.parametrize("p", [0.0, 1e-3])
def test_interleaver_matches_twin(p):
    r = gen(77)
    coded = block_stream(atkdsp.SMALL_CODES["hamming(7,4)"](), 1200, r, skip=0)
    x = bsc(interleave(coded, 4, 7)[5:], p, r)
    c = atkdsp.interleaver_id(x, 8, 40)
    assert_same(c, ref.interleaver_id(x, 8, 40))
    assert (c["period"], c["rows"], c["cols"], c["inner_width"]) == (28, 4, 7, 7)


# ============================================================================
# block codes: every D3 code at BER 0, 1e-4, 1e-3
# ============================================================================
SMALL = [("hamming(7,4)", "Hamming(7,4)", 3), ("hamming(15,11)", "Hamming(15,11)", 3),
         ("hamming(31,26)", "Hamming(31,26)", 3), ("hamming(8,4)", "extended Hamming(8,4)", 4),
         ("hamming(16,11)", "extended Hamming(16,11)", 4), ("golay(23,12)", "Golay(23,12)", 7),
         ("golay(24,12)", "extended Golay(24,12)", 8)]


@pytest.mark.parametrize("p", BERS)
@pytest.mark.parametrize("code,name,dmin", SMALL)
def test_small_codes_named(code, name, dmin, p):
    r = gen(len(code) * 7 + int(p * 1e5))
    G = atkdsp.SMALL_CODES[code]()
    k, n = G.shape
    res = atkdsp.block_code_id(bsc(block_stream(G, 800, r), p, r), 4, 64)
    assert (res["n"], res["k"], res["name"], res["dmin"]) == (n, k, name, dmin)
    assert res["robust"] == (0 if p == 0.0 else res["robust"])
    assert_evidence(res, p)
    # the generator spans the planted code (canonical: same matrix whatever the data)
    Gc = res["G"]
    both = ref.gf2_rank([int("".join(map(str, row[::-1])), 2) for row in np.vstack([Gc, G])])
    assert both == k


@pytest.mark.parametrize("p", BERS)
@pytest.mark.parametrize("m,t,name", [(5, 2, "BCH(31,21) t=2"), (6, 3, "BCH(63,45) t=3"),
                                      (7, 4, "BCH(127,99) t=4"), (8, 2, "BCH(255,239) t=2")])
def test_bch_named(m, t, name, p):
    r = gen(m * 13 + t + int(p * 1e5))
    bc = atkdsp.BCH(m, t)
    res = atkdsp.block_code_id(bsc(bch_stream(bc, max(300, 3 * bc.n), r), p, r), 4, 256)
    assert (res["n"], res["k"], res["name"]) == (bc.n, bc.k, name)
    assert res["cyclic"] == 1
    gp = int("".join(map(str, res["gpoly"])), 2)
    assert gp == int("".join(map(str, bc.generator())), 2)
    assert_evidence(res, p)


def test_bch_31_21_is_pocsag():
    r = gen(5)
    res = atkdsp.block_code_id(bch_stream(atkdsp.BCH(5, 2), 200, r), 4, 64)
    assert int("".join(map(str, res["gpoly"])), 2) == 0x769


@pytest.mark.parametrize("p", BERS)
@pytest.mark.parametrize("m,t,sh,name", [(6, 2, 13, "shortened BCH(63,51) t=2 as (50,38)"),
                                         (8, 2, 55, "shortened BCH(255,239) t=2 as (200,184)"),
                                         (4, 1, 2, "shortened Hamming(15,11) as (13,9)")])
def test_shortened_named(m, t, sh, name, p):
    r = gen(m * 17 + sh + int(p * 1e5))
    bc = atkdsp.BCH(m, t, sh)
    res = atkdsp.block_code_id(bsc(bch_stream(bc, 500, r, skip=5), p, r), 4, 256)
    assert (res["n"], res["k"], res["shortened"], res["name"]) == (bc.n, bc.k, sh, name)
    assert_evidence(res, p)


@pytest.mark.parametrize("p", BERS)
@pytest.mark.parametrize("n,k", [(12, 6), (20, 10), (40, 20)])
def test_random_linear_codes(n, k, p):
    """Any small linear code (the §16d class): not cyclic, found by elimination."""
    r = gen(n * 31 + int(p * 1e5))
    G = np.hstack([np.eye(k, dtype=np.uint8), r.integers(0, 2, (k, n - k)).astype(np.uint8)])
    res = atkdsp.block_code_id(bsc(block_stream(G, max(1500, 40 * n), r, skip=3), p, r), 4, 128)
    assert (res["n"], res["k"], res["name"]) == (n, k, f"linear ({n},{k})")
    assert res["log10_pfa"] <= -3
    both = ref.gf2_rank([int("".join(map(str, row[::-1])), 2) for row in np.vstack([res["G"], G])])
    assert both == k


@pytest.mark.parametrize("p", BERS + (1e-2,))
def test_partial_checks_of_a_wrong_width_are_refused(p):
    """The robust search's periodicity test. The shortened Golay (20,8): rows
    of 10 bits alternate between a codeword's two halves, each with checks of
    its own, so width 10 shows "checks" holding on every other row; and a
    random (40,20) searched from 30 meets widths whose alignment cycles."""
    r = gen(2008 + int(p * 1e5))
    G = systematic(atkdsp.SMALL_CODES["golay(24,12)"]())[4:, 4:]
    res = atkdsp.block_code_id(bsc(block_stream(G, 1500, r, skip=3), p, r), 4, 128)
    assert (res["n"], res["k"], res["name"]) == (20, 8, "linear (20,8)")
    G = np.hstack([np.eye(20, dtype=np.uint8), r.integers(0, 2, (20, 20)).astype(np.uint8)])
    res = atkdsp.block_code_id(bsc(block_stream(G, 1600, r, skip=3), min(p, 1e-3), r), 30, 50)
    assert (res["n"], res["k"]) == (40, 20)


@pytest.mark.parametrize("p", BERS)
def test_rs_binary_image_named(p):
    r = gen(415 + int(p * 1e5))
    rs = atkdsp.RS(4, 0x13, 0, 1, 4)
    res = atkdsp.block_code_id(bsc(rs_stream(rs, 4, 800, r)[7:], p, r), 4, 128)
    assert (res["n"], res["k"], res["name"]) == (60, 44, "RS(15,11) GF(2^4) binary image")
    assert_evidence(res, p)


def test_block_noisy_limits_stated_in_the_header():
    """Hamming / Golay to 5e-2, BCH(255,239) to 3e-3 (header §18)."""
    r = gen(9)
    for code, name in (("hamming(7,4)", "Hamming(7,4)"), ("golay(24,12)", "extended Golay(24,12)")):
        res = atkdsp.block_code_id(bsc(block_stream(atkdsp.SMALL_CODES[code](), 3000, r), 0.05, r), 4, 64)
        assert res["name"] == name
        assert res["ber_est"] == pytest.approx(0.05, rel=0.25)
    bc = atkdsp.BCH(8, 2)
    res = atkdsp.block_code_id(bsc(bch_stream(bc, 400, r), 3e-3, r), 4, 256)
    assert res["name"] == "BCH(255,239) t=2"
    assert res["ber_est"] == pytest.approx(3e-3, rel=0.3)


def test_beyond_the_limit_says_so_with_the_ber():
    """Structure verified but never complete: 'too noisy', with the BER."""
    r = gen(11)
    G = np.hstack([np.eye(16, dtype=np.uint8), r.integers(0, 2, (16, 8)).astype(np.uint8)])
    for _ in range(5):                       # the margin is thin by design: find one
        res = atkdsp.block_code_id(bsc(block_stream(G, 4000, r, skip=3), 0.015, r), 4, 64)
        if res["name"].startswith("no code found (n=24"):
            break
    assert not res["found"]
    assert res["name"] == "no code found (n=24 too noisy)"
    assert res["ber_est"] == pytest.approx(0.015, rel=0.3)
    # far beyond: nothing at all
    bc = atkdsp.BCH(7, 4)
    res = atkdsp.block_code_id(bsc(bch_stream(bc, 400, r), 0.03, r), 4, 256)
    assert not res["found"] and res["name"].startswith("no code found")


# ============================================================================
# convolutional codes
# ============================================================================
CONV = [(7, [0o171, 0o133], "NASA/CCSDS"), (7, [0o133, 0o171, 0o165], "LTE"),
        (9, [0o561, 0o753], "IS-95 / cdma2000"), (9, [0o557, 0o663, 0o711], "IS-95 / 3GPP r=1/3"),
        (5, [0o23, 0o33], "GSM"), (7, [0o133, 0o171, 0o145, 0o133], "DAB / DRM"),
        (3, [0o7, 0o5], "textbook"), (15, [0o46321, 0o51271], None)]


@pytest.mark.parametrize("p", BERS + (1e-2,))
@pytest.mark.parametrize("K,polys,cat", CONV)
def test_conv_named(K, polys, cat, p):
    r = gen(K * 7 + len(polys) + int(p * 1e5))
    s = atkdsp.conv_encode(K, polys, r.integers(0, 2, 6000).astype(np.uint8),
                           atkdsp.CONV_STREAM)[int(r.integers(0, 8)):]
    res = atkdsp.conv_code_id(bsc(s, p, r))
    assert (res["n"], res["k"], res["K"]) == (len(polys), 1, K)
    assert sorted(res["polys"]) == sorted(polys)
    if cat:
        assert cat in res["name"]
    assert_evidence(res, p)
    if p <= 1e-3:                       # a measured density: noise can lower it
        assert res["density"] == pytest.approx(1 - 1 / len(polys), abs=0.05)


@pytest.mark.parametrize("punc,n,k", [([1, 1, 1, 0], 3, 2), ([1, 1, 0, 1, 1, 0], 4, 3)])
def test_punctured_reports_the_rate(punc, n, k):
    r = gen(n * 10 + k)
    s = atkdsp.conv_encode(7, [0o171, 0o133], r.integers(0, 2, 20000).astype(np.uint8),
                           atkdsp.CONV_STREAM, puncture=punc)[3:]
    res = atkdsp.conv_code_id(s)
    assert (res["n"], res["k"], res["polys"]) == (n, k, [])
    assert res["name"] == f"rate {k}/{n} convolutional (k > 1: no generators)"


def test_conv_high_noise_limit():
    r = gen(3)
    s = atkdsp.conv_encode(7, [0o171, 0o133], r.integers(0, 2, 20000).astype(np.uint8), atkdsp.CONV_STREAM)
    res = atkdsp.conv_code_id(bsc(s, 0.02, r))
    assert "NASA/CCSDS" in res["name"] and res["ber_est"] == pytest.approx(0.02, rel=0.3)


def test_conv_stream_is_not_a_block_code():
    r = gen(4)
    s = atkdsp.conv_encode(7, [0o171, 0o133], r.integers(0, 2, 20000).astype(np.uint8), atkdsp.CONV_STREAM)[3:]
    res = atkdsp.block_code_id(s, 4, 64)
    assert not res["found"] and res["name"] == "no block code (sliding, period 2)"


# ============================================================================
# Reed-Solomon
# ============================================================================
RSCASES = [("RS(15,11) GF(2^4) 0x13 fcr 0 prim 1", 4, 0x13, 0, 1, 4, 0, False, False, 80),
           ("RS(63,55) GF(2^6) 0x43 fcr 1 prim 1", 6, 0x43, 1, 1, 8, 0, False, False, 160),
           ("CCSDS RS(255,223) dual basis", 8, 0x187, 112, 11, 32, 0, True, False, 400),
           ("RS(204,188) GF(2^8) 0x11D fcr 0 prim 1", 8, 0x11D, 0, 1, 16, 51, False, False, 400),
           ("RS(255,223) GF(2^8) 0x11D fcr 1 prim 1", 8, 0x11D, 1, 1, 32, 0, False, True, 400)]


@pytest.mark.parametrize("p", BERS)
@pytest.mark.parametrize("name,m,gf,fcr,prim,nr,pad,dual,lsb,nw", RSCASES)
def test_rs_named(name, m, gf, fcr, prim, nr, pad, dual, lsb, nw, p):
    r = gen(m * 1000 + nr + pad + int(p * 1e5))
    rs = atkdsp.RS(m, gf, fcr, prim, nr, pad, dual)
    res = atkdsp.rs_code_id(bsc(rs_stream(rs, m, nw, r, lsb), p, r), 0, 0 if m == 4 else m, rs.n)
    assert res["name"] == name
    assert (res["n"], res["k"], res["lsb_first"], res["dual_basis"]) == (rs.n, rs.k, int(lsb), int(dual))
    assert res["log10_pfa"] <= -3
    assert res["agreement"] >= 1.0 - 4.0 * m * p - 2e-4
    assert res["valid_words"] >= 0.99


# ============================================================================
# interleaved variants
# ============================================================================
@pytest.mark.parametrize("code,rows,cols,inner,p", [
    ("hamming(7,4)", 16, 7, 7, 0.0), ("hamming(7,4)", 16, 7, 7, 1e-4), ("hamming(7,4)", 16, 7, 7, 1e-3),
    ("hamming(7,4)", 7, 16, 7, 0.0), ("golay(24,12)", 8, 24, 24, 0.0), ("golay(24,12)", 8, 24, 24, 1e-3),
    ("hamming(15,11)", 12, 15, 15, 1e-4)])
def test_interleaved_recovered(code, rows, cols, inner, p):
    r = gen(rows * cols + int(p * 1e5))
    G = atkdsp.SMALL_CODES[code]()
    k, n = G.shape
    coded = block_stream(G, 2500 * (rows * cols) // n, r, skip=0)
    res = atkdsp.interleaver_id(bsc(interleave(coded, rows, cols)[11:], p, r), 8, 512)
    assert (res["period"], res["rows"], res["cols"], res["inner_width"]) == (rows * cols, rows, cols, inner)
    assert res["log10_pfa"] <= -3


# ============================================================================
# negative controls
# ============================================================================
@pytest.mark.parametrize("n", [2000, 50000, 200000])
def test_random_bits_name_nothing(n):
    x = gen(n).integers(0, 2, n).astype(np.uint8)
    assert not atkdsp.block_code_id(x, 4, 256)["found"]
    assert not atkdsp.conv_code_id(x)["found"]
    assert not atkdsp.interleaver_id(x, 8, 256)["found"]
    assert not atkdsp.rs_code_id(x, 0, 0, 0)["found"]


def test_false_alarm_rate_on_many_short_streams():
    """pfa 1e-3 is a ceiling, every test counted: 40 streams give none."""
    r = gen(99)
    hits = sum(atkdsp.block_code_id(r.integers(0, 2, 8000).astype(np.uint8), 4, 64)["found"]
               or atkdsp.conv_code_id(r.integers(0, 2, 8000).astype(np.uint8), 4, 9)["found"]
               for _ in range(40))
    assert hits == 0


@pytest.mark.parametrize("kind", ["lfsr", "zeros", "periodic"])
def test_recurrences_are_not_block_codes(kind):
    r = gen(5)
    if kind == "lfsr":
        st = list(r.integers(0, 2, 31))
        st[0] = 1
        while len(st) < 50000:
            st.append(st[-31] ^ st[-28])
        x = np.array(st, np.uint8)
    elif kind == "zeros":
        x = np.zeros(50000, np.uint8)
    else:
        x = np.tile(r.integers(0, 2, 37).astype(np.uint8), 1400)
    res = atkdsp.block_code_id(x, 4, 128)
    assert not res["found"] and res["name"] == "no block code (linear recurrence)"
    assert not atkdsp.interleaver_id(x, 8, 128)["found"]


# ============================================================================
# properties
# ============================================================================
def test_repeatable_and_canonical():
    r = gen(21)
    G = atkdsp.SMALL_CODES["golay(24,12)"]()
    a = bsc(block_stream(G, 600, r), 1e-3, r)
    b = bsc(block_stream(G, 600, r, skip=9), 1e-3, r)
    ra, rb = atkdsp.block_code_id(a, 4, 64, seed=5), atkdsp.block_code_id(b, 4, 64, seed=5)
    assert np.array_equal(ra["G"], rb["G"])                  # the same code, the same matrix
    assert atkdsp.block_code_id(a, 4, 64, seed=5)["G"].tobytes() == ra["G"].tobytes()
    # canonical: each row's pivot is its highest set bit, highest pivot first, cleared elsewhere
    piv = [int(np.flatnonzero(row)[-1]) for row in ra["G"]]
    assert piv == sorted(piv, reverse=True)
    for i, pv in enumerate(piv):
        assert ra["G"][:, pv].sum() == 1


def test_golay_generator_polynomial():
    r = gen(23)
    res = atkdsp.block_code_id(block_stream(atkdsp.SMALL_CODES["golay(23,12)"](), 400, r), 4, 64)
    assert int("".join(map(str, res["gpoly"])), 2) in (0xC75, 0xAE3)


# ============================================================================
# held by test: the independent review (2026-10) and the stress sweeps after it
# ============================================================================
H74 = atkdsp.SMALL_CODES["hamming(7,4)"]()
G24 = atkdsp.SMALL_CODES["golay(24,12)"]()


def nullspace_of(G):
    """Check matrix of a generator matrix (rows of 0/1)."""
    G = systematic(G)
    k, n = G.shape
    piv = [int(np.flatnonzero(row)[0]) for row in G]
    free = [c for c in range(n) if c not in piv]
    H = np.zeros((len(free), n), np.uint8)
    for i, f in enumerate(free):
        H[i, f] = 1
        for r, c in enumerate(piv):
            H[i, c] = G[r, f]
    return H


def clean_rows_valid(clean, res):
    """Fraction of the noise-free stream's rows, at the alignment found, that
    satisfy every check found: 1 for a true description."""
    N, off = res["n"], res["offset"]
    H = nullspace_of(res["G"])
    R = (clean.size - off) // N
    rows = clean[off:off + R * N].reshape(R, N)
    return 1.0 - (rows @ H.T % 2).any(axis=1).mean()


def test_conv_name_of_eight_long_generators_fits():
    """8 generators of 5 octal digits: the name is cut to the field, never
    overrun (the review's crash), and the code is still named."""
    r = gen(31)
    polys = [0o40001 | (int(x) << 1) for x in r.integers(0, 1 << 13, 8)]
    s = atkdsp.conv_encode(15, polys, r.integers(0, 2, 3000).astype(np.uint8), atkdsp.CONV_STREAM)
    res = atkdsp.conv_code_id(s, 8, 15)
    assert res["polys"] == polys and res["K"] == 15 and len(res["name"]) <= 63


def test_rank_scan_rows_small_or_huge():
    s = block_stream(H74, 300, gen(32))
    assert (atkdsp.gf2_rank_scan(s, 2, 20, 4, 10) == -1).all()     # 10 rows < w + 16 everywhere
    huge = atkdsp.gf2_rank_scan(s, 2, 20, 1, 1 << 40)                # clamped to the data
    assert (huge >= 0).all()
    assert np.array_equal(huge, ref.gf2_rank_scan(s, 2, 20, 1, 1 << 40))


@pytest.mark.parametrize("seed", [0, 10007, 10068])
def test_exact_path_survives_an_early_error(seed):
    """One error among the first rows lowers the first deficiency; the exact
    path then hands over to the noisy paths instead of naming a shrunk code."""
    r = gen(seed)
    if seed == 0:
        x = block_stream(H74, 600, r, skip=0)
        x[23] ^= 1
    else:
        x = bsc(block_stream(H74, 800, r), 1e-3, r)
    c = atkdsp.block_code_id(x, 4, 64)
    assert c["name"] == "Hamming(7,4)"
    assert_same(c, ref.block_code_id(x, 4, 64))


def test_bch_named_by_its_largest_t():
    res = atkdsp.block_code_id(bch_stream(atkdsp.BCH(5, 5), 300, gen(7), skip=0), 4, 64)
    assert res["name"] == "BCH(31,11) t=5"


@pytest.mark.parametrize("code,name", [("hamming(8,4)", "extended Hamming(8,4) parity first"),
                                       ("hamming(16,11)", "extended Hamming(16,11) parity first"),
                                       ("golay(24,12)", "extended Golay(24,12) parity first")])
def test_parity_first_extended_codes(code, name):
    G = np.roll(atkdsp.SMALL_CODES[code](), 1, axis=1)
    s = block_stream(G, 800, gen(8), skip=3)
    c = atkdsp.block_code_id(s, 4, 64)
    assert c["name"] == name and c["extended"] == 1
    assert_same(c, ref.block_code_id(s, 4, 64))


def test_extended_bch_parity_first():
    bc, r = atkdsp.BCH(5, 2), gen(9)
    words = [bc.encode(r.integers(0, 2, bc.k).astype(np.uint8)) for _ in range(400)]
    s = np.concatenate([np.concatenate([[w.sum() & 1], w]) for w in words]).astype(np.uint8)
    res = atkdsp.block_code_id(s, 4, 64)
    assert res["name"] == "extended BCH(32,21) t=2 parity first"
    assert len(res["gpoly"]) == 32 - 21                     # of the 31 cyclic bits: n - k


@pytest.mark.parametrize("lo,hi,p", [(4, 23, 0.0), (25, 64, 0.0), (25, 64, 1e-3), (25, 53, 1e-2)])
def test_code_outside_the_range_is_not_named(lo, hi, p):
    """Golay(24,12) searched where 24 is not allowed. Before the 10 % clean
    floor and the row-pattern test, lengths 46 and 47 were named (a cyclic
    count fitting 5 of the alignment's 24 phases); the stress sweep that found
    it had 27 false descriptions in 2880 runs, now none."""
    for sd in range(6):
        r = gen(1000 * sd + 24 + lo)
        x = bsc(block_stream(G24, 800, r, skip=int(r.integers(0, 24))), p, r)
        res = atkdsp.block_code_id(x, lo, hi)
        assert not res["found"], (sd, res["name"])


def test_named_codes_are_true_descriptions():
    """Whatever is named must hold on the noise-free stream at the alignment
    reported (the stress oracle, on the cases that used to fail)."""
    bc = atkdsp.BCH(6, 2)
    for sd in range(4):
        r = gen(50 + sd)
        clean = bch_stream(bc, 320, r, skip=int(r.integers(0, 63)))
        for lo, hi in ((4, 128), (64, 64), (32, 70)):
            res = atkdsp.block_code_id(clean, lo, hi)
            if res["found"]:
                assert clean_rows_valid(clean, res) == 1.0, (sd, lo, hi, res["name"])


def test_streams_the_statistics_do_not_fit():
    """Biased bits, a recurrence, a noisy LFSR: refused before any test that
    assumes uniform memoryless bits, by every recogniser."""
    bern = (gen(11).random(50000) < 0.1).astype(np.uint8)
    tile = np.tile(gen(12).integers(0, 2, 100).astype(np.uint8), 500)
    for x, nb, nc in ((bern, "no code found (biased: P(1) = 0.10)", "no code found (biased: P(1) = 0.10)"),
                      (tile, "no block code (linear recurrence)", "no code found (linear recurrence)")):
        c = atkdsp.block_code_id(x, 4, 64)
        assert c["name"] == nb and not c["found"]
        assert_same(c, ref.block_code_id(x, 4, 64))
        for fn, args in ((atkdsp.conv_code_id, ()), (atkdsp.rs_code_id, (0, 0, 0))):
            res = fn(x, *args)
            assert res["name"] == nc and not res["found"]
        assert not atkdsp.interleaver_id(x, 8, 128)["found"]
    r = gen(13)
    st = list(r.integers(0, 2, 31))
    st[0] = 1
    while len(st) < 40000:
        st.append(st[-31] ^ st[-28])
    x = bsc(np.array(st, np.uint8), 1e-3, r)          # noise hides it from Berlekamp-Massey
    res = atkdsp.block_code_id(x, 4, 64)
    assert not res["found"] and res["name"].startswith("no block code")


def test_rs_periodic_stream_is_not_codeword_aligned():
    """One RS(15,11) codeword repeated: every window is a codeword, so the
    roots survive half a codeword along — structure, but not aligned
    codewords. (Seed 2: a codeword with no roots beyond the generator's;
    most seeds give this, a few have an extra root and so no run.)"""
    rs, r = atkdsp.RS(4, 0x13, 0, 1, 4), gen(2)
    d = r.integers(0, 16, 11)
    w = sym_bits(np.concatenate([d, rs.encode(d)]), 4)
    x = bsc(np.tile(w, 600), 1e-3, r)
    c = atkdsp.rs_code_id(x, 0, 4, 15)
    assert not c["found"] and c["name"] == "no code found (not codeword-aligned)"
    assert_same(c, ref.rs_code_id(x, 0, 4, 15))


def test_rs_reciprocal_tie_matches_twin():
    """A reversed-bit, reciprocal-polynomial twin scores the same up to
    rounding; C and twin must break the tie alike."""
    rs, r = atkdsp.RS(3, 0xB, 2, 1, 4), gen(12)
    words = [np.concatenate([d, rs.encode(d)]) for d in (r.integers(0, 8, 3) for _ in range(60))]
    x = np.concatenate([sym_bits(w, 3, lsb=True) for w in words])
    x = bsc(x, 0.03, r)
    assert_same(atkdsp.rs_code_id(x, 0, 3, 7), ref.rs_code_id(x, 0, 3, 7))


@pytest.mark.parametrize("K,polys,max_n,name", [
    (3, [7, 7, 5], 8, "K=3 r=1/3 (7,7,5)"), (2, [3, 3, 1], 8, "K=2 r=1/3 (3,3,1)"),
    (7, [0o171, 0o171, 0o133], 3, "K=7 r=1/3 (171,171,133)"),
    (5, [0o23, 0o23], 8, "K=2 r=1/2 (1,1)")])
def test_conv_repeated_outputs(K, polys, max_n, name):
    """A repeated output is not a longer code at a multiple of n (the K rule);
    (23,23) generates the same set of sequences as (1,1), its shortest form."""
    s = atkdsp.conv_encode(K, polys, gen(15).integers(0, 2, 4000).astype(np.uint8), atkdsp.CONV_STREAM)
    assert atkdsp.conv_code_id(s, max_n, 15)["name"] == name


@pytest.mark.parametrize("code", ["hamming(7,4)", "golay(24,12)"])
def test_plain_codes_are_not_interleavers(code):
    r = gen(16)
    x = block_stream(atkdsp.SMALL_CODES[code](), 3000, r, skip=3)
    assert not atkdsp.interleaver_id(x, 8, 512)["found"]
    bc = atkdsp.BCH(6, 2)
    assert not atkdsp.interleaver_id(bch_stream(bc, 1500, r, skip=0), 8, 512)["found"]


@pytest.mark.parametrize("code,rows,cols,inner,p,nwords", [
    ("hamming(7,4)", 4, 5, 7, 0.0, 6000), ("golay(24,12)", 6, 10, 24, 0.0, 3000),
    ("hamming(7,4)", 4, 5, 7, 1e-3, 14000), ("golay(24,12)", 6, 10, 24, 1e-3, 6300),
    ("hamming(8,4)", 5, 7, 8, 1e-3, 44000)])
def test_interleaver_block_not_a_multiple_of_the_codeword(code, rows, cols, inner, p, nwords):
    """The period is then a common multiple; the block is one of its
    divisors, and in noise its start one of the offsets tying for the
    period's deficiency (Hamming(8,4) in 5 x 7: three offsets, one right)."""
    r = gen(rows * cols + int(p * 1e4))
    G = atkdsp.SMALL_CODES[code]()
    skip = int(r.integers(0, rows * cols))
    x = bsc(interleave(block_stream(G, nwords, r, skip=0), rows, cols)[skip:], p, r)
    res = atkdsp.interleaver_id(x, 8, 512)
    assert (res["rows"], res["cols"], res["period"], res["inner_width"]) == (rows, cols, rows * cols, inner)
    assert res["offset"] == (-skip) % (rows * cols)


def test_interleaver_inner_code_longer_than_the_block():
    r = gen(17)
    bc = atkdsp.BCH(6, 2)
    clean = interleave(bch_stream(bc, 5500, r, skip=0), 5, 7)[11:]
    res = atkdsp.interleaver_id(clean, 8, 512)
    assert (res["rows"], res["cols"], res["inner_width"], res["offset"]) == (5, 7, 63, (-11) % 35)


def test_equivalent_interleaver_is_reported():
    """3 x 14 over Hamming(7,4) writes the same output as 3 x 7 (the codewords
    kept whole, their order permuted): the smaller one is the answer."""
    x = interleave(block_stream(H74, 6000, gen(18), skip=0), 3, 14)[5:]
    res = atkdsp.interleaver_id(x, 8, 512)
    assert (res["rows"], res["cols"], res["period"], res["inner_width"]) == (3, 7, 21, 7)


# ============================================================================
# fixed bits — a structured payload inside a block code (v0.17.0, plan §D4)
# ============================================================================
# The Bit Bench's harness found it: text under Hamming(7,4) gave the null
# space single-position checks (the payload's fixed bits — ASCII's high bit)
# and then refused the stream as "sliding" or "fails unlike noise" at 28 kbit,
# 1 seed in 20 at 42 kbit. Now the exact search allows a constant position
# as extra structure in a stretch, the single-position checks are set aside
# and reported, and the checks are restored to a named code where that
# names one.
LETTERS = np.frombuffer(b"abcdefghijklmnopqrstuvwxyz ", np.uint8)


def ascii_bits(r, nchars):
    by = LETTERS[r.integers(0, LETTERS.size, nchars)]
    return np.unpackbits(by[:, None], axis=1, bitorder="little").ravel()


def hamming_cyclic_encode(nibbles):
    out = []
    for m in nibbles:
        w = 0
        for i in range(4):
            if m[i]:
                w ^= 0b1011 << i
        out.append([(w >> j) & 1 for j in range(7)])
    return np.array(out, np.uint8).ravel()


@pytest.mark.parametrize("seed", range(6))
def test_text_under_a_systematic_hamming_code_is_named_with_its_fixed_bit(seed):
    """Two codewords per character at n_lo = 14: the high bit of ASCII is a
    fixed 0 in every block; k and dmin come out as Hamming's."""
    r = gen(100 + seed)
    G = atkdsp.SMALL_CODES["hamming(7,4)"]()
    coded = ((ascii_bits(r, 6000).reshape(-1, 4) @ G) & 1).astype(np.uint8).ravel()[:28000]
    t = ref.block_code_id(coded, 14, 32, 1e-3, 1)
    assert t["found"] and (t["n"], t["k"], t["fixed_bits"], t["dmin"]) == (14, 8, 1, 3), t["name"]
    assert t["name"].endswith(", 1 fixed bit") and t["agreement"] == 1.0
    assert_same(atkdsp.block_code_id(coded, 14, 32, 1e-3, 1), t)


@pytest.mark.parametrize("seed", range(3))
def test_a_three_bit_payload_under_cyclic_hamming_is_hamming_with_a_fixed_bit(seed):
    r = gen(200 + seed)
    nib = np.zeros((6000, 4), np.uint8)
    nib[:, :3] = r.integers(0, 2, (6000, 3))
    coded = hamming_cyclic_encode(nib)[:28000]
    t = ref.block_code_id(coded, 4, 32, 1e-3, 1)
    assert t["name"] == "Hamming(7,4), 1 fixed bit" and t["cyclic"] == 1 and t["dmin"] == 3
    assert t["fixed_bits"] == 1 and t["agreement"] == 1.0
    assert_same(atkdsp.block_code_id(coded, 4, 32, 1e-3, 1), t)


def test_text_under_hamming_behind_an_interleaver_decodes_by_the_chain():
    """The harness case end to end: interleaver_id finds the block, the
    de-interleaved stream names the code with its fixed bit, and the text
    comes back through the generator handed over (a systematic encoder, so
    the payload sits at known positions)."""
    r = gen(300)
    text = LETTERS[r.integers(0, LETTERS.size, 8000)]
    nib = np.unpackbits(text[:, None], axis=1, bitorder="little").reshape(-1, 4)
    Gs = systematic(atkdsp.SMALL_CODES["hamming(7,4)"]())
    coded = ((nib @ Gs) & 1).astype(np.uint8).ravel()
    x = interleave(coded, 16, 7)[:42000]
    il = atkdsp.interleaver_id(x, 8, 512)
    assert il["found"] and (il["rows"], il["cols"]) == (16, 7)
    y = ref._deinterleave(ref._bits(x), il["offset"], (x.size - il["offset"]) // il["period"],
                          il["period"], il["rows"])
    res = atkdsp.block_code_id(y, 14, 32, 1e-3, 1)
    assert res["found"] and res["fixed_bits"] == 1 and res["agreement"] == 1.0, res["name"]
    assert res["name"].endswith(", 1 fixed bit")
    # the bench decodes with the generator handed back; the payload is where
    # the systematic encoder put it (columns 0..3 of each 7-bit word)
    bc = ref.BlockCode(res["G"])
    words = np.asarray(y[res["offset"]:res["offset"] + 14 * 300]).reshape(-1, 14)
    got = []
    for w in words:
        corrected, _data, cc = bc.decode(w)
        assert cc == 0
        got.append(np.concatenate([corrected[0:4], corrected[7:11]]))
    chars = np.packbits(np.array(got, np.uint8), axis=1, bitorder="little").ravel()
    assert bytes(chars) in bytes(text)


def test_random_payloads_report_no_fixed_bits():
    r = gen(400)
    G = atkdsp.SMALL_CODES["hamming(7,4)"]()
    coded = block_stream(G, 4000, r)
    t = ref.block_code_id(coded, 4, 32, 1e-3, 1)
    assert t["found"] and t["fixed_bits"] == 0 and "fixed" not in t["name"]
    c = atkdsp.block_code_id(coded, 4, 32, 1e-3, 1)
    assert c["fixed_bits"] == 0
    assert_same(c, t)
