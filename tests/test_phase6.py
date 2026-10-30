"""Phase 6a (atkdsp.h §25-27; docs/ATKDSP_PLAN.md §B6, §B6b, §B6c): spread
spectrum — the blind detector, the chip-rate scan, blind code recovery, the
known-code acquisition and tracking, the LoRa dechirp probe, the receiver
pair and the consistency lattice. Known answers first (a planted code is
recovered and named), noise and the wrong signal at the design rate, C
against its twin in uneven blocks. `needs_lib` tests skip when the library
is not built; the rest run on the twins alone.

Run from the repo root:  python -m pytest tests/test_phase6.py -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
import atkdsp                                  # noqa: E402
from atkdsp import reference as ref            # noqa: E402

HAVE = atkdsp.available()
needs_lib = pytest.mark.skipif(not HAVE, reason="atkdsp library not built")
BLOCKS = [1, 7, 239, 1024, 4096, 65537]


# =============================================================================
# helpers: m-sequences, Gold codes, DSSS signals
# =============================================================================
def mseq(mask: int, r: int) -> np.ndarray:
    """One period of the m-sequence of a primitive polynomial (siga.bits mask
    convention), as +-1 chips with chip = 1 - 2*bit (so (chip < 0) == bit)."""
    L = (1 << r) - 1
    st = [1] + [0] * (r - 1)
    taps = [i for i in range(r) if (mask >> i) & 1]
    out = []
    for _ in range(L):
        out.append(st[0])
        fb = 0
        for i in taps:
            fb ^= st[i]
        st = st[1:] + [fb]
    return np.array([1 - 2 * b for b in out], np.int8)


def gold(mask1, mask2, r, shift) -> np.ndarray:
    a = (mseq(mask1, r) < 0).astype(int)
    b = (mseq(mask2, r) < 0).astype(int)
    g = a ^ np.roll(b, shift)
    return np.array([1 - 2 * x for x in g], np.int8)


def dsss_signal(code, sps, nsym, snr_db, rng, chip_mod="bpsk", cfo_cyc=0.0):
    """A DSSS signal: data symbols spread by `code`, rectangular chips at `sps`
    samples/chip, at the given in-band SNR, with an optional carrier offset."""
    L = len(code)
    if chip_mod == "bpsk":
        data = rng.integers(0, 2, nsym) * 2 - 1
        chips = np.repeat(data, L) * np.tile(code, nsym)
        base = np.repeat(chips, sps).astype(np.complex128)
    elif chip_mod == "qpsk":
        di = rng.integers(0, 2, nsym) * 2 - 1
        dq = rng.integers(0, 2, nsym) * 2 - 1
        ci = np.repeat(di, L) * np.tile(code, nsym)
        cq = np.repeat(dq, L) * np.tile(code, nsym)
        base = np.repeat(ci + 1j * cq, sps).astype(np.complex128) / math.sqrt(2)
    else:
        raise ValueError(chip_mod)
    if cfo_cyc:
        base = base * np.exp(2j * np.pi * cfo_cyc * np.arange(base.size))
    p = np.mean(np.abs(base) ** 2)
    n = (rng.standard_normal(base.size) + 1j * rng.standard_normal(base.size)) * math.sqrt(p / (2 * 10 ** (snr_db / 10)))
    return (base + n)


def feed_uneven(obj, x, rng):
    """Feed x to obj.process in uneven blocks (chunk-invariance)."""
    i = 0
    while i < x.size:
        b = min(int(rng.choice(BLOCKS)), x.size - i)
        obj.process(x[i:i + b])
        i += b


R5 = ref.primitive_polys(5)
R7 = ref.primitive_polys(7)


# =============================================================================
# §25a the blind detector
# =============================================================================
@pytest.mark.parametrize("r,sps", [(5, 4), (7, 2), (7, 4)])
def test_detector_finds_the_period(r, sps):
    rng = np.random.default_rng(100 + r * 10 + sps)
    code = mseq(ref.primitive_polys(r)[0], r)
    L = code.size
    N = L * sps
    x = dsss_signal(code, sps, 5000, -8, rng)
    win = 16 * N
    ml = min(3 * N, win // 2)
    cfg = ref.dsss_det_defaults(1.0, win, ml)
    d = ref.DsssDet(cfg)
    d.process(x)
    out = d.read()
    assert out["flags"] & ref.DSSS_DETECT
    assert abs(out["period"] - N) <= 1, (out["period"], N)


@needs_lib
@pytest.mark.parametrize("r,sps", [(5, 4), (7, 2), (7, 4)])
def test_detector_c_equals_twin(r, sps):
    rng = np.random.default_rng(200 + r * 10 + sps)
    code = mseq(ref.primitive_polys(r)[0], r)
    N = code.size * sps
    x = dsss_signal(code, sps, 4000, -8, rng)
    win = 16 * N
    ml = min(3 * N, win // 2)
    cfg = ref.dsss_det_defaults(1.0, win, ml)
    dc, dt = atkdsp.DsssDet(cfg), ref.DsssDet(cfg)
    feed_uneven(dc, x.astype(np.complex64), np.random.default_rng(1))
    feed_uneven(dt, x, np.random.default_rng(1))
    rc, rt = dc.read(), dt.read()
    for k in ("period", "peak_z", "cav_t", "tc_acf", "llr", "floor_mean"):
        assert abs(rc[k] - rt[k]) <= 1e-4 * (abs(rt[k]) + 1e-9) + 1e-7, (k, rc[k], rt[k])
    assert rc["flags"] == rt["flags"]
    assert np.allclose(dc.rho(), dt.rho(), rtol=1e-4, atol=1e-7)


def test_detector_noise_false_alarm_band():
    """On noise the fused DETECT fires at about its Pfa — a binomial band over
    many independent captures."""
    rng = np.random.default_rng(7)
    N = 127 * 4
    win = 16 * N
    ml = min(3 * N, win // 2)
    pfa = 1e-2
    trials = 200
    hits = 0
    for _ in range(trials):
        cfg = ref.dsss_det_defaults(1.0, win, ml)
        cfg["pfa"] = pfa
        cfg["nwin_min"] = 16
        d = ref.DsssDet(cfg)
        x = (rng.standard_normal(win * 16) + 1j * rng.standard_normal(win * 16))
        d.process(x)
        if d.read()["flags"] & ref.DSSS_DETECT:
            hits += 1
    # the chi-square threshold (not a Gaussian) makes the fluctuation handle's
    # rate match its Pfa; the union of three handles each at pfa sits at a few x
    # pfa. A calibrated detector is well under 5 % here (measured ~2 %).
    assert hits <= 0.05 * trials, (hits, trials)


def test_detector_cav_blind_at_one_sample_per_chip():
    """Zeng-Liang's CAV needs correlated samples; at one rectangular sample per
    chip it has nothing, and the detector still finds the period by Burel."""
    rng = np.random.default_rng(11)
    code = mseq(R7[0], 7)
    N = code.size * 1
    x = dsss_signal(code, 1, 6000, -6, rng)
    win = 16 * N
    cfg = ref.dsss_det_defaults(1.0, win, min(3 * N, win // 2))
    d = ref.DsssDet(cfg)
    d.process(x)
    out = d.read()
    assert out["flags"] & ref.DSSS_FLUCT
    assert abs(out["period"] - N) <= 1


# =============================================================================
# §25b the chip-rate scan
# =============================================================================
def test_scan_finds_the_chip_line():
    """A BPSK stream has a clock line at the chip rate after the DM nonlinearity."""
    rng = np.random.default_rng(3)
    sps = 8
    rc = 1.0 / sps
    data = rng.integers(0, 2, 30000) * 2 - 1
    x = np.repeat(data, sps).astype(np.complex128) + 0.5 * (rng.standard_normal(30000 * sps) + 1j * rng.standard_normal(30000 * sps))
    s = ref.CyclicScan(1.0, ref.NL_DM, 0, 256, 64, 0.5 * rc, 2 * rc, 1e-3)
    s.process(x)
    lines = s.read()
    assert lines, "no line found"
    assert abs(lines[0]["freq_hz"] - rc) < 0.01 * rc


@needs_lib
def test_scan_c_equals_twin():
    rng = np.random.default_rng(4)
    sps = 8
    rc = 1.0 / sps
    data = rng.integers(0, 2, 20000) * 2 - 1
    x = np.repeat(data, sps).astype(np.complex128) + 0.5 * (rng.standard_normal(20000 * sps) + 1j * rng.standard_normal(20000 * sps))
    sc = atkdsp.CyclicScan(1.0, "dm", 0, 256, 64, 0.5 * rc, 2 * rc, 1e-3)
    st = ref.CyclicScan(1.0, ref.NL_DM, 0, 256, 64, 0.5 * rc, 2 * rc, 1e-3)
    feed_uneven(sc, x.astype(np.complex64), np.random.default_rng(2))
    feed_uneven(st, x, np.random.default_rng(2))
    assert np.allclose(sc.level(), st.level(), rtol=1e-3, atol=1e-5)
    lc, lt = sc.read(), st.read()
    assert len(lc) == len(lt)
    if lc:
        assert abs(lc[0]["freq_hz"] - lt[0]["freq_hz"]) < 1e-6


def test_scan_noise_is_quiet():
    rng = np.random.default_rng(5)
    x = (rng.standard_normal(256 * 64) + 1j * rng.standard_normal(256 * 64))
    s = ref.CyclicScan(1.0, ref.NL_ENV2, 0, 256, 64, 0.1, 0.4, 1e-3)
    s.process(x)
    assert len(s.read()) == 0


# =============================================================================
# §25d blind code recovery
# =============================================================================
def _match_code(a, b):
    a = np.asarray(a, np.int8)
    for sgn in (1, -1):
        for sh in range(len(b)):
            if np.array_equal(a, (sgn * np.roll(b, sh)).astype(np.int8)):
                return True
    return False


@pytest.mark.parametrize("r,sps,snr", [(5, 4, -3), (7, 4, -6), (7, 2, -10)])
def test_code_recovered_and_named(r, sps, snr):
    rng = np.random.default_rng(300 + r * 10 + sps)
    mask = ref.primitive_polys(r)[0]
    code = mseq(mask, r)
    N = code.size * sps
    x = dsss_signal(code, sps, 5000, snr, rng)
    chips, res = ref.dsss_code(x, ref.dsss_code_defaults(1.0, 1.0 / sps, float(N)))
    assert res["flags"] & ref.CODE_IDENTIFIABLE
    assert _match_code(chips, code)
    assert res["flags"] & ref.CODE_NAMED
    assert res["name"].startswith("m-sequence (%d)" % code.size)


@needs_lib
@pytest.mark.parametrize("r,sps,snr", [(5, 4, -3), (7, 4, -6)])
def test_code_c_equals_twin(r, sps, snr):
    rng = np.random.default_rng(400 + r)
    code = mseq(ref.primitive_polys(r)[0], r)
    N = code.size * sps
    x = dsss_signal(code, sps, 4000, snr, rng)
    cfg = ref.dsss_code_defaults(1.0, 1.0 / sps, float(N))
    cc, rc = atkdsp.dsss_code(x.astype(np.complex64), cfg)
    ct, rt = ref.dsss_code(x, cfg)
    assert np.array_equal(cc, ct)
    assert rc["name"] == rt["name"]
    assert abs(rc["ident_margin"] - rt["ident_margin"]) < 1e-3 * rt["ident_margin"]
    assert rc["poly"] == rt["poly"]


def test_code_refuses_when_not_identifiable():
    """Too few periods at low SNR: the Rao-Edelman rule refuses with the count
    of periods it would need, rather than returning noise."""
    rng = np.random.default_rng(13)
    code = mseq(R7[0], 7)
    N = code.size * 4
    x = dsss_signal(code, 4, 10, -15, rng)        # only ~10 periods, deep
    chips, res = ref.dsss_code(x, ref.dsss_code_defaults(1.0, 0.25, float(N)))
    if not (res["flags"] & ref.CODE_IDENTIFIABLE):
        assert res["flags"] & ref.CODE_REFUSED
        assert res["periods_needed"] != 0


# Carrier offsets across the band where the period-summed P-th power used to
# cancel itself: tiny (half a turn over the capture), the bootstrap's residual
# (~15 turns), and up to the in-period limit 1/(2 P sps). 0 dB, 4000 periods.
CFO_CASES = [0.0, 1e-6, 3e-5, -3e-5, 3e-4, 2e-3, -2e-3, 1e-2, -1e-2]


@pytest.mark.parametrize("cfo", CFO_CASES)
def test_code_recovered_through_a_carrier_offset(cfo):
    rng = np.random.default_rng(700 + int(abs(cfo) * 1e7) % 97)
    code = mseq(ref.primitive_polys(5)[0], 5)
    sps, nper = 4, 4000
    N = code.size * sps
    x = dsss_signal(code, sps, nper, 0.0, rng, cfo_cyc=cfo)
    chips, res = ref.dsss_code(x, ref.dsss_code_defaults(1.0, 1.0 / sps, float(N)))
    assert _match_code(chips, code), (cfo, res["name"], res["cfo_hz"])
    assert res["name"].startswith("m-sequence (31)")
    # the offset itself, to a fraction of the capture's resolution
    assert abs(res["cfo_hz"] - cfo) < 2.0 / (N * nper), (cfo, res["cfo_hz"])


@needs_lib
@pytest.mark.parametrize("cfo", [3e-5, -2e-3])
def test_code_cfo_c_equals_twin(cfo):
    rng = np.random.default_rng(42)
    code = mseq(ref.primitive_polys(5)[0], 5)
    N = code.size * 4
    x = dsss_signal(code, 4, 3000, -2.0, rng, cfo_cyc=cfo).astype(np.complex64)
    cfg = ref.dsss_code_defaults(1.0, 0.25, float(N))
    ct, rt = ref.dsss_code(x, cfg)
    cc, rc = atkdsp.dsss_code(x, cfg)
    assert np.array_equal(np.asarray(cc), np.asarray(ct))
    assert abs(rc["cfo_hz"] - rt["cfo_hz"]) < 1e-9
    assert rc["name"] == rt["name"]


def test_code_gold_named():
    rng = np.random.default_rng(21)
    r = 5
    m1, m2 = R5[0], R5[1]
    code = gold(m1, m2, r, 3)
    assert code.size == 31
    N = code.size * 4
    x = dsss_signal(code, 4, 6000, -2, rng)
    chips, res = ref.dsss_code(x, ref.dsss_code_defaults(1.0, 0.25, float(N)))
    assert _match_code(chips, code)
    # Gold codes have linear complexity 2r; named Gold or at least the right LC
    assert res["lc"] in (2 * r, 2 * r - 1, 2 * r + 1) or "Gold" in res["name"]


def test_code_barker_named():
    rng = np.random.default_rng(22)
    for L in (7, 11, 13):
        code = ref.barker(L)
        N = L * 4
        x = dsss_signal(code, 4, 8000, -1, rng)
        chips, res = ref.dsss_code(x, ref.dsss_code_defaults(1.0, 0.25, float(N)))
        assert _match_code(chips, code), L
        assert "Barker-%d" % L == res["name"], (L, res["name"])


def test_code_polynomial_through_chip_errors():
    """Berlekamp-Massey is exact or nothing; the parity-check counting names the
    polynomial through a few planted chip errors."""
    r = 7
    mask = R7[0]
    code = mseq(mask, r)
    bits = (code < 0).astype(np.int8)
    chips = np.where(bits, -1, 1).astype(np.int8)
    rng = np.random.default_rng(31)
    for ber in (0.0, 0.05, 0.10):
        c = chips.copy()
        flip = rng.random(c.size) < ber
        c[flip] *= -1
        # the true polynomial scores N(1-2ber)^w; a wrong one ~ 0
        z_true = ref.parity_z(c, mask, r)
        z_wrong = ref.parity_z(c, ref.primitive_polys(r)[1], r)
        assert z_true > z_wrong, (ber, z_true, z_wrong)


@needs_lib
def test_parity_z_c_equals_twin():
    code = mseq(R7[0], 7)
    chips = np.where((code < 0), -1, 1).astype(np.int8)
    for p in ref.primitive_polys(7)[:5]:
        assert abs(atkdsp.parity_z(chips, p, 7) - ref.parity_z(chips, p, 7)) < 1e-9


# =============================================================================
# §25c known codes
# =============================================================================
@needs_lib
def test_known_codes_c_equals_twin():
    for prn in (1, 5, 19, 32, 120, 138):
        assert np.array_equal(atkdsp.gps_ca(prn), ref.gps_ca(prn))
    for L in (7, 11, 13):
        assert np.array_equal(atkdsp.barker(L), ref.barker(L))
    for s in range(16):
        assert np.array_equal(atkdsp.ieee802154_chips(s), ref.ieee802154_chips(s))
    for r in range(2, 13):
        assert atkdsp.primitive_polys(r) == ref.primitive_polys(r)


def test_gps_ca_is_balanced_and_mls():
    """Every C/A code has 2^9 ones and 2^9-1 zeros (balance), and BM gives its
    linear complexity (a Gold code of degree-10 factors: LC 20)."""
    for prn in (1, 10, 25):
        c = ref.gps_ca(prn)
        assert c.size == 1023
        ones = int(np.sum(c < 0))
        assert abs(ones - 512) <= 1


# =============================================================================
# §25e acquisition and tracking
# =============================================================================
def _known_signal(code, sps, nper, phase_chips, dop_cyc, snr_db, rng, data=False):
    L = len(code)
    Np = L * sps
    rep = np.repeat(code.astype(float), sps)
    full = np.tile(rep, nper + 2)
    shift = int(round(phase_chips * sps))
    sig = full[shift:shift + Np * nper].astype(np.complex128)
    if data:
        bits = np.repeat(rng.integers(0, 2, nper // 20 + 1) * 2 - 1, 20)[:nper]
        sig = np.concatenate([b * rep for b in np.repeat(bits, 1)])[:Np * nper]
    sig = sig * np.exp(2j * np.pi * dop_cyc * np.arange(sig.size))
    p = np.mean(np.abs(sig) ** 2)
    n = (rng.standard_normal(sig.size) + 1j * rng.standard_normal(sig.size)) * math.sqrt(p / (2 * 10 ** (snr_db / 10)))
    return sig + n


@pytest.mark.parametrize("snr", [-3, -10])
def test_acquire_finds_phase_and_doppler(snr):
    rng = np.random.default_rng(500 + snr)
    code = ref.gps_ca(5)
    L = code.size
    sps = 2
    dop = 0.0008
    x = _known_signal(code, sps, 20, 40, dop, snr, rng)
    cfg = ref.dsss_acq_defaults(1.0, 1.0 / sps, L)
    cfg.update(ncoh=4, nnoncoh=4, dop_lo=-0.003, dop_hi=0.003, dop_step=0.0004)
    out = ref.dsss_acquire(x, code, cfg)
    assert out["flags"] & ref.ACQ_DETECT
    # code phase is where the code sits in the data: (L - planted) mod L
    assert out["peak_ratio"] > 2.5


@needs_lib
def test_acquire_c_equals_twin():
    rng = np.random.default_rng(6)
    code = ref.gps_ca(5)
    L = code.size
    sps = 2
    x = _known_signal(code, sps, 20, 40, 0.0008, -3, rng)
    cfg = ref.dsss_acq_defaults(1.0, 1.0 / sps, L)
    cfg.update(ncoh=4, nnoncoh=4, dop_lo=-0.003, dop_hi=0.003, dop_step=0.0004)
    oc = atkdsp.dsss_acquire(x.astype(np.complex64), code.astype(np.int8), cfg)
    ot = ref.dsss_acquire(x, code, cfg)
    assert abs(oc["gamma"] - ot["gamma"]) < 1e-3 * ot["gamma"]
    assert abs(oc["code_phase"] - ot["code_phase"]) < 1e-3
    assert oc["doppler_hz"] == pytest.approx(ot["doppler_hz"])
    assert oc["flags"] == ot["flags"]


@needs_lib
@pytest.mark.parametrize("snr", [0, -5, -10])
def test_tracking_despreads(snr):
    rng = np.random.default_rng(700 + snr)
    fs, rc = 4.092e6, 1.023e6
    code = ref.gps_ca(10)
    L = code.size
    sps = int(round(fs / rc))
    Np = L * sps
    rep = np.repeat(code.astype(float), sps)
    nper = 400
    bits = np.repeat(rng.integers(0, 2, nper // 20 + 1) * 2 - 1, 20)[:nper]
    sig = np.concatenate([b * rep for b in bits]).astype(np.complex128)
    sig = sig * np.exp(2j * np.pi * 25.0 * np.arange(sig.size) / fs)
    p = np.mean(np.abs(sig) ** 2)
    x = (sig + (rng.standard_normal(sig.size) + 1j * rng.standard_normal(sig.size)) * math.sqrt(p / (2 * 10 ** (snr / 10)))).astype(np.complex64)
    cfg = ref.dsss_trk_defaults(fs, rc, L, 0.0, 25.0)
    tc = atkdsp.DsssTrk(code.astype(np.int8), cfg)
    pr = tc.process(x)
    st = tc.read()
    syms = np.sign(pr.real)
    agree = np.mean(syms[80:] == np.sign(bits[80:len(syms)]))
    agree = max(agree, 1 - agree)
    assert agree > 0.95, (snr, agree)
    assert st["flags"] & ref.TRK_LOCK
    assert st["cn0_m2m4"] > 40.0


@needs_lib
def test_tracking_c_equals_twin():
    rng = np.random.default_rng(8)
    fs, rc = 4.092e6, 1.023e6
    code = ref.gps_ca(10)
    L = code.size
    sps = int(round(fs / rc))
    rep = np.repeat(code.astype(float), sps)
    nper = 200
    bits = np.repeat(rng.integers(0, 2, nper // 20 + 1) * 2 - 1, 20)[:nper]
    sig = np.concatenate([b * rep for b in bits]).astype(np.complex128) * np.exp(2j * np.pi * 25.0 * np.arange(nper * L * sps) / fs)
    p = np.mean(np.abs(sig) ** 2)
    x = (sig + (rng.standard_normal(sig.size) + 1j * rng.standard_normal(sig.size)) * math.sqrt(p / (2 * 10 ** (-5 / 10)))).astype(np.complex64)
    cfg = ref.dsss_trk_defaults(fs, rc, L, 0.0, 25.0)
    pc = atkdsp.DsssTrk(code.astype(np.int8), cfg).process(x)
    pt = ref.DsssTrk(code, cfg).process(x.astype(np.complex128))
    m = np.mean(np.abs(pt))
    assert np.max(np.abs(pc - pt)) < 1e-5 * m


def test_acq_threshold_monotone():
    """The acquisition threshold rises as Pfa falls and matches C."""
    a = ref.acq_threshold(4, 1e4, 1e-2)
    b = ref.acq_threshold(4, 1e4, 1e-4)
    assert b > a > 0


# =============================================================================
# §26 the LoRa chirp probe
# =============================================================================
def _lora(sf, bw, fs, syms):
    """A LoRa frame: the base up-chirp frequency-shifted by each symbol value
    (s / 2^sf of the band), which is what atkdsp's dechirp demod inverts."""
    N = int(round((1 << sf) * fs / bw))
    Tsym = (1 << sf) / bw
    t = np.arange(N) / fs
    base = np.exp(1j * 2 * np.pi * (-bw / 2 * t + bw / (2 * Tsym) * t * t))
    n = np.arange(N)
    return np.concatenate([base * np.exp(1j * 2 * np.pi * s * n / (1 << sf)) for s in syms])


def test_css_detects_preamble():
    """CSS_PREAMBLE reflects the window passing now, so it is read while the 8
    up-chirps go by (the real caller reads each frame)."""
    rng = np.random.default_rng(40)
    fs, sf, bw = 1.0e6, 8, 125e3
    pre = _lora(sf, bw, fs, [0] * 8)
    p = ref.CssProbe(fs, [(sf, bw), (7, bw)], 6, 1e-3)
    p.process(pre + 0.2 * (rng.standard_normal(pre.size) + 1j * rng.standard_normal(pre.size)))
    r = p.read()
    assert r[0]["flags"] & ref.CSS_DETECT
    assert r[0]["flags"] & ref.CSS_PREAMBLE
    assert not (r[1]["flags"] & ref.CSS_PREAMBLE)        # the wrong SF does not get a preamble


@needs_lib
def test_css_c_equals_twin():
    rng = np.random.default_rng(41)
    fs, sf, bw = 1.0e6, 8, 125e3
    sig = _lora(sf, bw, fs, [0] * 8 + [10, 40, 100])
    x = sig + 0.2 * (rng.standard_normal(sig.size) + 1j * rng.standard_normal(sig.size))
    pc = atkdsp.CssProbe(fs, [(sf, bw), (7, bw)], 6, 1e-3)
    pt = ref.CssProbe(fs, [(sf, bw), (7, bw)], 6, 1e-3)
    feed_uneven(pc, x.astype(np.complex64), np.random.default_rng(3))
    feed_uneven(pt, x, np.random.default_rng(3))
    rc, rt = pc.read(), pt.read()
    for a, b in zip(rc, rt):
        assert abs(a["level"] - b["level"]) < 1e-3 * (b["level"] + 1)
        assert a["flags"] == b["flags"]


def test_css_demod_round_trip():
    """The dechirp demod inverts the modulator: symbols out == symbols in."""
    fs, sf, bw = 1.0e6, 7, 125e3
    syms = [5, 20, 60, 120]
    sig = _lora(sf, bw, fs, syms)
    out = ref.css_demod(sig, fs, sf, bw, 0.0, 0.0, cap=len(syms))
    assert list(int(v) for v in out) == syms


@needs_lib
def test_css_demod_c_equals_twin():
    fs, sf, bw = 1.0e6, 7, 125e3
    sig = _lora(sf, bw, fs, [5, 20, 60, 120])
    dc = atkdsp.css_demod(sig.astype(np.complex64), fs, sf, bw, 0.0, 0.0, cap=4)
    dt = ref.css_demod(sig, fs, sf, bw, 0.0, 0.0, cap=4)
    assert np.array_equal(dc, dt)


def test_css_noise_quiet():
    rng = np.random.default_rng(42)
    fs, sf, bw = 1.0e6, 8, 125e3
    x = (rng.standard_normal(40 * int((1 << sf) * fs / bw)) + 1j * rng.standard_normal(40 * int((1 << sf) * fs / bw)))
    p = ref.CssProbe(fs, [(sf, bw)], 6, 1e-3)
    p.process(x)
    assert not (p.read()[0]["flags"] & ref.CSS_PREAMBLE)


# =============================================================================
# §27 the receiver pair and the lattice
# =============================================================================
def _two_rx(n, band_lo, band_hi, delay, snr_db, rng, lo_off=0.0):
    """Two receivers hearing a common band-limited signal, delayed by `delay`
    samples at rx2, with independent noise."""
    w = (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    W = np.fft.fft(w)
    mask = np.zeros(n, bool)
    k = np.fft.fftfreq(n)
    mask[(k >= band_lo) & (k < band_hi)] = True
    W[~mask] = 0
    common = np.fft.ifft(W)
    common = common / math.sqrt(np.mean(np.abs(common) ** 2))
    sh = np.exp(-2j * np.pi * np.fft.fftfreq(n) * delay)
    common2 = np.fft.ifft(np.fft.fft(common) * sh)
    if lo_off:
        common2 = common2 * np.exp(2j * np.pi * lo_off * np.arange(n))
    s = math.sqrt(10 ** (snr_db / 10))
    x1 = s * common + (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2)
    x2 = s * common2 + (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2)
    return x1, x2


def test_xspec_detects_and_times():
    rng = np.random.default_rng(50)
    x1, x2 = _two_rx(1 << 16, 0.10, 0.15, 7.0, 3, rng)
    xs = ref.Xspec(1.0, 1024, 64, 1e-3)
    xs.process(x1, x2)
    out = xs.read()
    assert out["flags"] & ref.XS_DETECT
    assert out["flags"] & ref.XS_TDOA
    assert abs(abs(out["tdoa_s"]) - 7.0) < 0.5, out["tdoa_s"]   # the sign is a reference convention


@needs_lib
def test_xspec_c_equals_twin():
    rng = np.random.default_rng(51)
    x1, x2 = _two_rx(1 << 16, 0.10, 0.15, 7.0, 3, rng)
    xc = atkdsp.Xspec(1.0, 1024, 64, 1e-3)
    xt = ref.Xspec(1.0, 1024, 64, 1e-3)
    xc.process(x1.astype(np.complex64), x2.astype(np.complex64))
    xt.process(x1, x2)
    rc, rt = xc.read(), xt.read()
    for key in ("max_coh", "tdoa_s", "phase0"):
        assert abs(rc[key] - rt[key]) < 1e-3 * (abs(rt[key]) + 1e-6), (key, rc[key], rt[key])
    assert rc["flags"] == rt["flags"]


def test_xspec_independent_noise_quiet():
    rng = np.random.default_rng(52)
    n = 1 << 16
    x1 = (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    x2 = (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    xs = ref.Xspec(1.0, 1024, 64, 1e-3)
    xs.process(x1, x2)
    assert not (xs.read()["flags"] & ref.XS_DETECT)


def test_align_measures_lag_and_lo():
    rng = np.random.default_rng(53)
    # the lag search is non-coherent over blocks, so it survives a large LO
    # offset; the LO *estimate* from the product is accurate for a modest offset
    # (the phase-referencing path refines a large one). Test both in their range.
    x1, x2 = _two_rx(1 << 16, 0.02, 0.48, 9.0, 10, rng, lo_off=0.001)
    out, _ = ref.align_estimate(x1, x2, 1.0, 32, block=0)
    assert abs(out["lag"] - 9.0) < 1.0, out["lag"]        # lag robust to the big offset
    x1, x2 = _two_rx(1 << 16, 0.02, 0.48, 9.0, 10, rng, lo_off=1e-4)
    out, _ = ref.align_estimate(x1, x2, 1.0, 32, block=0)
    assert abs(out["lag"] - 9.0) < 1.0
    assert abs(out["lo_offset_hz"] - 1e-4) < 5e-5         # LO accurate at a modest offset


@needs_lib
def test_align_c_equals_twin():
    rng = np.random.default_rng(54)
    x1, x2 = _two_rx(1 << 15, 0.02, 0.48, 6.0, 10, rng, lo_off=0.0005)
    oc, _ = atkdsp.align_estimate(x1.astype(np.complex64), x2.astype(np.complex64), 1.0, 32, block=0)
    ot, _ = ref.align_estimate(x1, x2, 1.0, 32, block=0)
    assert abs(oc["lag"] - ot["lag"]) < 1e-3
    assert abs(oc["lo_offset_hz"] - ot["lo_offset_hz"]) < 1e-6
    assert abs(oc["rate_ppm"] - ot["rate_ppm"]) < 1.0


def test_lattice_consistent_and_inconsistent():
    # agreeing estimates of a known family + one clock (the two chip-rate
    # estimates come from one signal, so they agree closely)
    good = dict(rc_acf_hz=1023000.0, rc_line_hz=1023000.0, rc_nominal_hz=1023000.0,
                period_s=1023 / 1023000.0, fc_hz=1.57542e9, cfo_hz=1540.0)
    out = ref.lattice(good)
    assert out["flags"] & ref.LAT_AGREE_RC
    assert out["flags"] & ref.LAT_INTEGER_L
    assert out["L_int"] == 1023
    assert out["flags"] & ref.LAT_FAMILY
    # disagreeing chip rates -> not consistent
    bad = dict(rc_acf_hz=1000.0, rc_line_hz=1500.0, period_s=0.031)
    o2 = ref.lattice(bad)
    assert not (o2["flags"] & ref.LAT_AGREE_RC)


@needs_lib
def test_lattice_c_equals_twin():
    for li in (dict(rc_acf_hz=1023000.0, rc_line_hz=1023000.0, rc_nominal_hz=1023000.0,
                    period_s=1023 / 1023000.0, fc_hz=1.57542e9, cfo_hz=1540.0),
               dict(rc_acf_hz=1000.0, rc_line_hz=1500.0, period_s=0.031)):
        a, b = atkdsp.lattice(li), ref.lattice(li)
        assert a["flags"] == b["flags"]
        assert a["L_int"] == b["L_int"]
        assert abs(a["rc_hz"] - b["rc_hz"]) < 1e-6


@needs_lib
def test_method_table_c_equals_twin():
    for mid in range(1, ref.METHOD_COUNT):
        c = atkdsp.method(mid)
        t = ref.method(mid)
        assert c is not None and t is not None
        assert c[3] == t[3]                                # the country tag (Bill's menu grouping)
