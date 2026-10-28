"""The detection track (atkdsp.h §19-21; docs/ATKDSP_PLAN.md §B1, §B2, §A3).

CFAR and the burst tracker, the burst descriptor word, the ring — each
against known answers first (rule 2), against noise at the design rate
(rule 3), and C against its twin (rule 1). Tests marked `needs_lib` skip when
bin/atkdsp.(dll|so) is not built; the rest run on the twins alone.

Run from the repo root:  python -m pytest tests/test_detection.py -q
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
import atkdsp                                  # noqa: E402
from atkdsp import reference as ref            # noqa: E402

HAVE = atkdsp.available()
needs_lib = pytest.mark.skipif(not HAVE, reason="atkdsp library not built")


# ---- helpers ------------------------------------------------------------------
def binom_interval(n: int, p: float, conf: float = 0.99) -> tuple:
    """The exact central `conf` interval of Binomial(n, p): (lo, hi)."""
    mu = n * p
    sd = math.sqrt(n * p * (1 - p))
    k0, k1 = max(0, int(mu - 12 * sd) - 5), min(n, int(mu + 12 * sd) + 5)
    ks = np.arange(k0, k1 + 1)
    lp = np.array([math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                   + k * math.log(p) + (n - k) * math.log1p(-p) for k in ks])
    pm = np.exp(lp - lp.max())
    pm /= pm.sum()
    c = np.cumsum(pm)
    a = (1 - conf) / 2
    return int(ks[np.searchsorted(c, a)]), int(ks[np.searchsorted(c, 1 - a)])


def same_events(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        for k in x:
            u, v = x[k], y[k]
            if isinstance(u, float):
                if math.isnan(u) and math.isnan(v):
                    continue
                if not (u == v or abs(u - v) <= 1e-9 * max(1.0, abs(u))):
                    return False
            elif u != v:
                return False
    return True


def _cfar(impl, nbins, **cfg):
    return atkdsp.Cfar(nbins, **cfg) if impl == "c" else ref.Cfar(nbins, **cfg)


IMPLS = ["twin", pytest.param("c", marks=needs_lib)]


# =============================================================================
# §19 the threshold
# =============================================================================
@pytest.mark.parametrize("pfa", [1e-2, 1e-4, 1e-6, 1e-9, 1e-12])
def test_alpha_reproduces_the_one_look_closed_forms(pfa):
    N, n, k = 32, 16, 24
    a_ca, _ = ref.cfar_alpha(ref.CFAR_CA, 16, pfa)
    assert a_ca == pytest.approx(N * (pfa ** (-1 / N) - 1), rel=1e-8)
    a_os, et = ref.cfar_alpha(ref.CFAR_OS, 16, pfa)
    p = 1.0
    for i in range(k):
        p *= (N - i) / (N - i + a_os)
    assert p == pytest.approx(pfa, rel=1e-7)
    assert et == pytest.approx(sum(1.0 / (N - i) for i in range(k)), rel=1e-9)   # E[k-th of N]

    def so(a):
        x = 2 + a / n
        return 2 * x ** (-n) * sum(math.comb(n - 1 + j, j) * x ** (-j) for j in range(n))
    a_so, _ = ref.cfar_alpha(ref.CFAR_SO, 16, pfa)
    a_go, _ = ref.cfar_alpha(ref.CFAR_GO, 16, pfa)
    assert so(a_so) == pytest.approx(pfa, rel=1e-7)                  # Gandhi & Kassam
    assert 2 * (1 + a_go / n) ** (-n) - so(a_go) == pytest.approx(pfa, rel=1e-7)


@pytest.mark.parametrize("K", [2, 4, 8])
def test_ca_alpha_with_k_looks_matches_the_incomplete_beta(K):
    """CA on K-look gamma cells: Pfa = I_{N/(N+alpha)}(NK, K), a finite sum
    for integer K."""
    N = 32
    for pfa in (1e-3, 1e-6, 1e-10):
        a, _ = ref.cfar_alpha(ref.CFAR_CA, 16, pfa, float(K))
        x, A = N / (N + a), N * K
        p = sum(math.exp(math.lgamma(A + j) - math.lgamma(j + 1) - math.lgamma(A)
                         + A * math.log(x) + j * math.log1p(-x)) for j in range(K))
        assert p == pytest.approx(pfa, rel=1e-7)


@needs_lib
def test_the_c_solver_agrees_with_the_twin():
    for mode in range(4):
        for K in (1.0, 2.5, 16.0):
            a, et = atkdsp.cfar_alpha(mode, 12, 1e-7, K)
            b, eu = ref.cfar_alpha(mode, 12, 1e-7, K)
            assert a == pytest.approx(b, rel=1e-12) and et == pytest.approx(eu, rel=1e-12)


def test_bad_configurations_are_refused_with_the_reason():
    with pytest.raises(ValueError, match="nbins"):
        ref.Cfar(20, n_ref=16)
    with pytest.raises(ValueError, match="pfa"):
        ref.Cfar(256, pfa=0.9)
    with pytest.raises(ValueError, match="unknown"):
        ref.Cfar(256, colour="red")
    if HAVE:
        with pytest.raises(atkdsp.AtkDspError, match="open_m"):
            atkdsp.Cfar(256, open_m=4, open_n=3)


# =============================================================================
# §19 the false-alarm rate — the test that proves the threshold is derived
# =============================================================================
def _fa(impl, mode, pfa, nframes, looks=1.0, k_frames=1, nb=4096, seed=1, edges=False):
    """Hits / cells on unit noise, counted only at cells far enough apart to
    be independent (a shared reference window correlates neighbours, which
    would widen the spread beyond the binomial). `edges` counts only the
    first and last bins, whose whole windows are borrowed from one side."""
    rng = np.random.default_rng(seed)
    nr, ng = 8, 2
    c = _cfar(impl, nb, mode=mode, pfa=pfa, n_ref=nr, n_guard=ng, looks=looks,
              k_frames=k_frames, open_m=1, open_n=1, close_k=1, max_tracks=4096)
    W = 2 * (nr + ng) + 1
    if edges:      # the outermost bins only: each borrows its whole window
        sel = np.array([0, nb - 1])
    else:
        sel = np.arange(W, nb - W, W)
    hits = cells = 0
    for f in range(nframes):
        p = rng.gamma(looks, 1.0 / looks, nb)
        c.process(10 * np.log10(p), f * 1e-3, 1e-3)
        if f >= k_frames - 1 and (f - (k_frames - 1)) % k_frames == 0:
            _x, _t, h = c.last()
            hits += int(h[sel].sum())
            cells += sel.size
    return hits, cells


@needs_lib
@pytest.mark.parametrize("mode", [0, 1, 2, 3], ids=["CA", "OS", "GO", "SO"])
@pytest.mark.parametrize("pfa", [1e-2, 1e-3, 1e-4])
def test_false_alarm_rate_in_noise_at_three_rates(mode, pfa):
    want = 400 if pfa >= 1e-3 else 150
    nframes = int(math.ceil(want / pfa / 195))
    hits, cells = _fa("c", mode, pfa, nframes, seed=int(-math.log10(pfa)) * 10 + mode)
    lo, hi = binom_interval(cells, pfa)
    assert lo <= hits <= hi, (hits, cells, lo, hi)


@pytest.mark.parametrize("mode", [0, 1])
def test_the_twin_holds_the_rate_too(mode):
    hits, cells = _fa("twin", mode, 1e-2, 160, nb=2048, seed=7 + mode)
    lo, hi = binom_interval(cells, 1e-2)
    assert lo <= hits <= hi, (hits, cells, lo, hi)


@needs_lib
@pytest.mark.parametrize("looks", [4.0, 2.5])
def test_the_rate_holds_with_averaged_input_frames(looks):
    hits, cells = _fa("c", 0, 1e-3, 2400, looks=looks, seed=31)
    lo, hi = binom_interval(cells, 1e-3)
    assert lo <= hits <= hi, (hits, cells, lo, hi)


@needs_lib
def test_the_rate_holds_with_the_k_frame_average():
    hits, cells = _fa("c", 0, 1e-3, 4 * 2400, k_frames=4, seed=41)
    lo, hi = binom_interval(cells, 1e-3)
    assert lo <= hits <= hi, (hits, cells, lo, hi)


@needs_lib
@pytest.mark.parametrize("mode", [0, 1, 2], ids=["CA", "OS", "GO"])
def test_the_rate_holds_at_the_band_edges(mode):
    """The window borrows from the far side at an edge, so N stays N; a
    zero-padded window (siga.detect.cfar's 'same' convolution) would read
    low there and fire far more often."""
    hits, cells = _fa("c", mode, 1e-2, 20000, nb=128, seed=51 + mode, edges=True)
    lo, hi = binom_interval(cells, 1e-2)
    assert lo <= hits <= hi, (hits, cells, lo, hi)


# =============================================================================
# §19 planted bursts
# =============================================================================
FS20 = 20e6
NFFT = 1024


def _capture(seed=3):
    """50 ms of complex noise at 20 MSPS with planted bursts and a repeater."""
    rng = np.random.default_rng(seed)
    n = 1_000_000
    x = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2)).astype(np.complex64)
    t = np.arange(n) / FS20
    planted = []
    # (offset Hz, start s, duration s, kind, level)
    for f0, t0, dur, kind, lvl in [(3.1e6, 0.004, 0.002, "tone", 0.38),
                                   (-6.25e6, 0.011, 0.005, "fsk", 0.30),
                                   (1.0e6, 0.020, 0.010, "noise", 18.0),
                                   (7.77e6, 0.040, 0.001, "tone", 0.5)]:
        s, e = int(t0 * FS20), int((t0 + dur) * FS20)
        tt = t[s:e]
        if kind == "tone":
            sig = lvl * np.exp(2j * np.pi * f0 * tt)
        elif kind == "fsk":
            sym = np.repeat(rng.integers(0, 2, (e - s) // 400 + 1) * 2 - 1, 400)[:e - s]
            sig = lvl * np.exp(2j * np.pi * np.cumsum(f0 + 4e3 * sym) / FS20)
        else:
            w = rng.standard_normal(e - s) + 1j * rng.standard_normal(e - s)
            X = np.fft.fft(w)
            fr = np.fft.fftfreq(e - s, 1 / FS20)
            X[np.abs(fr - f0) > 100e3] = 0
            sig = np.fft.ifft(X)
            sig *= lvl / np.sqrt(np.mean(np.abs(sig) ** 2))
        x[s:e] += sig.astype(np.complex64)
        planted.append((f0, s / FS20, e / FS20))
    rep = -2.5e6
    x += (0.6 * np.exp(2j * np.pi * rep * t)).astype(np.complex64)     # the repeater
    w = np.hanning(NFFT + 1)[:NFFT].astype(np.float32)
    nf = n // NFFT
    X = np.fft.fftshift(np.fft.fft(x[:nf * NFFT].reshape(nf, NFFT) * w, axis=1), axes=1)
    frames = (10 * np.log10(np.abs(X) ** 2 / np.sum(w * w))).astype(np.float32)
    return frames, planted, rep


def _bin_of(f):
    return f / (FS20 / NFFT) + NFFT / 2


@pytest.mark.parametrize("impl", IMPLS)
def test_planted_bursts_give_one_detection_each_and_the_masked_repeater_none(impl):
    """CUED_DF_PLAN §5.1: each planted burst gives ONE detection, fc within a
    bin and t0 within a frame; the masked repeater gives none; noise gives
    nothing else."""
    frames, planted, rep = _capture()
    dt = NFFT / FS20
    c = _cfar(impl, NFFT, n_guard=8)
    mask = np.zeros(NFFT, bool)
    rb = int(round(_bin_of(rep)))
    mask[rb - 2:rb + 3] = True
    c.set_mask(mask)
    ev = c.process(frames, 0.0, dt) + c.flush()
    closes = [e for e in ev if e["type"] == ref.CFAR_CLOSE]
    assert len(closes) == len(planted), [(round(e["fc_bin"], 1), e["t0"]) for e in closes]
    for f0, t0, t1 in planted:
        m = [e for e in closes if abs(e["fc_bin"] - _bin_of(f0)) <= 1.0]
        assert len(m) == 1, (f0, [(e["fc_bin"], e["lo"], e["hi"]) for e in closes])
        assert abs(m[0]["t0"] - t0) <= dt, (f0, m[0]["t0"], t0)
        assert abs(m[0]["t1"] - t1) <= 2 * dt, (f0, m[0]["t1"], t1)
    assert not [e for e in closes if abs(e["fc_bin"] - _bin_of(rep)) < 4]
    assert c.stats()["opened"] == len(planted)


def test_os_cfar_sees_a_weak_signal_beside_a_strong_one():
    """A strong neighbour inside the reference window raises CA's threshold
    and hides a weak signal; OS's order statistic ignores it."""
    rng = np.random.default_rng(9)
    nb = 512
    lin = rng.exponential(1.0, (60, nb))
    lin[:, 250:253] += 10 ** 4.0          # strong, 40 dB
    lin[:, 262] += 10 ** 1.6              # weak, 16 dB, 10 bins away
    db = 10 * np.log10(lin)
    found = {}
    for name, mode in (("CA", 0), ("OS", 1)):
        c = ref.Cfar(nb, mode=mode, n_ref=16, n_guard=2, pfa=1e-6)
        ev = c.process(db, 0.0, 1.0) + c.flush()
        found[name] = any(abs(e["fc_bin"] - 262) < 1 for e in ev if e["type"] == 3)
    assert found == {"CA": False, "OS": True}


def test_a_masked_carrier_is_neither_tested_nor_used_as_reference():
    rng = np.random.default_rng(11)
    nb = 512
    base = rng.exponential(1.0, nb)
    loud = base.copy()
    loud[300:303] += 1e6
    mask = np.zeros(nb, bool)
    mask[300:303] = True
    thr = []
    for row in (base, loud):
        c = ref.Cfar(nb, pfa=1e-6)
        c.set_mask(mask)
        c.process(10 * np.log10(row), 0.0, 1.0)
        thr.append(c.last()[1])
    assert np.array_equal(thr[0], thr[1], equal_nan=True)
    assert np.isnan(thr[1][300:303]).all() and c.last()[2][300:303].sum() == 0


def test_the_k_frame_average_places_t0_within_a_frame_or_says_coarse():
    rng = np.random.default_rng(5)
    nb = 512
    lin = rng.exponential(1.0, (300, nb))
    lin[50:80, 99:102] += 10 ** 0.8       # 8 dB: found by the average, edges by the raw test
    lin[200:260, 399:402] += 10 ** 0.3    # 3 dB: too weak for one frame — coarse
    c = ref.Cfar(nb, pfa=1e-6, k_frames=8)
    ev = [e for e in c.process(10 * np.log10(lin), 0.0, 1.0) + c.flush() if e["type"] == 3]
    a = [e for e in ev if abs(e["fc_bin"] - 100) < 2]
    assert len(a) == 1 and abs(a[0]["t0"] - 50) <= 1 and abs(a[0]["t1"] - 80) <= 1
    assert not a[0]["flags"] & (ref.CFAR_T0_COARSE | ref.CFAR_T1_COARSE)
    b = [e for e in ev if abs(e["fc_bin"] - 400) < 3]
    assert b and any(e["flags"] & (ref.CFAR_T0_COARSE | ref.CFAR_T1_COARSE) for e in b)


@pytest.mark.parametrize("impl", IMPLS)
def test_tracks_beyond_max_tracks_are_dropped_and_counted(impl):
    nb = 600
    lin = np.ones((5, nb))
    lin[:, 50:550:10] = 1e5                # 50 separate carriers
    c = _cfar(impl, nb, max_tracks=20, n_ref=2, n_guard=1, open_m=1, open_n=1, merge_gap=0)
    ev = c.process(10 * np.log10(lin), 0.0, 1.0)
    st = c.stats()
    assert st["tracks"] == 20 and st["opened"] == 20 and st["dropped"] == 30 * 5
    assert len(ev) == 20


@needs_lib
def test_a_cap_below_events_max_is_refused_before_anything_is_consumed():
    c = atkdsp.Cfar(256)
    f = np.zeros((3, 256), np.float32)
    assert c.events_max(3) == 3 * 256
    buf = (atkdsp._CfarEvent * 10)()
    rc = c._lib.atkdsp_cfar_process(c._h, f.ctypes.data_as(atkdsp.C.POINTER(atkdsp.C.c_float)),
                                    3, 0.0, 1.0, buf, 10)
    assert rc == -3 and c.stats()["frames"] == 0


@needs_lib
@pytest.mark.parametrize("mode", [0, 1, 2, 3], ids=["CA", "OS", "GO", "SO"])
@pytest.mark.parametrize("k_frames", [1, 3])
def test_cfar_c_equals_the_twin_in_uneven_blocks(mode, k_frames):
    rng = np.random.default_rng(100 + mode * 7 + k_frames)
    nb = 777
    cfg = dict(mode=mode, pfa=1e-4, k_frames=k_frames, n_ref=8, n_guard=2, merge_gap=2,
               close_k=2)
    c, t = atkdsp.Cfar(nb, **cfg), ref.Cfar(nb, **cfg)
    mask = np.zeros(nb, bool)
    mask[100:110] = mask[5] = True
    mask[770:] = True
    c.set_mask(mask)
    t.set_mask(mask)
    lin = rng.exponential(1.0, (961, nb))
    for b, f0, f1, snr in [(50, 10, 60, 15), (300, 100, 103, 20), (500, 150, 250, 8),
                           (105, 20, 80, 30), (2, 40, 70, 20), (760, 0, 961, 25),
                           (640, 400, 900, 12)]:
        lin[f0:f1, max(b - 2, 0):b + 3] += 10 ** (snr / 10)
    db = (10 * np.log10(lin)).astype(np.float32)
    ec, et, i = [], [], 0
    for s in [1, 7, 239, 240, 241, 233]:
        ec += c.process(db[i:i + s], i * 0.01, 0.01)
        et += t.process(db[i:i + s], i * 0.01, 0.01)
        i += s
    ec += c.flush()
    et += t.flush()
    assert len(ec) > 5 and same_events(ec, et)
    lc, lt = c.last(), t.last()
    assert np.array_equal(lc[2], lt[2])
    assert np.allclose(lc[1], lt[1], rtol=1e-6, equal_nan=True)
    assert c.stats() == t.stats()
    assert c.describe() == t.describe()


@needs_lib
def test_the_binding_logs_its_plan_when_made(caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="atkdsp.detect"):
        atkdsp.Cfar(1024, pfa=1e-5)
    assert any("CFAR CA" in r.getMessage() and "alpha" in r.getMessage()
               for r in caplog.records)


# =============================================================================
# §20 the burst descriptor word
# =============================================================================
FS = 100e3


def _burst(kind, snr_db=20.0, n=20000, s=6000, e=14000, f0=5e3, drift=0.0, B=10e3,
           seed=0, ramp=0):
    rg = np.random.default_rng(seed)
    x = (rg.standard_normal(n) + 1j * rg.standard_normal(n)) / math.sqrt(2)
    A = 10 ** (snr_db / 20)
    m = e - s
    t = np.arange(m) / FS
    if kind == "cw":
        sig = A * np.exp(2j * np.pi * (f0 * t + 0.5 * drift * t * t))
    elif kind == "fsk":
        sym = np.repeat(rg.integers(0, 2, m // 20 + 1) * 2 - 1, 20)[:m]
        sig = A * np.exp(2j * np.pi * np.cumsum(f0 + 2400 * sym + drift * t) / FS)
    else:
        w = rg.standard_normal(m) + 1j * rg.standard_normal(m)
        X = np.fft.fft(w)
        X[np.abs(np.fft.fftfreq(m, 1 / FS) - f0) > B / 2] = 0
        sig = np.fft.ifft(X)
        sig *= A / np.sqrt(np.mean(np.abs(sig) ** 2))
    env = np.ones(m)
    if ramp:
        k = np.arange(ramp)
        rc = 0.5 - 0.5 * np.cos(np.pi * (k + 0.5) / ramp)
        env[:ramp] = rc
        env[m - ramp:] = rc[::-1]
    x[s:e] += sig * env
    return x.astype(np.complex64)


@pytest.mark.parametrize("snr", [10, 20, 30])
@pytest.mark.parametrize("kind", ["cw", "fsk"])
def test_bdw_edges_within_two_samples_for_a_keyed_constant_envelope(kind, snr):
    """§B2's acceptance: t0 within 2 samples at the channel rate, 10-30 dB."""
    for seed in range(6):
        d = ref.burst_measure(_burst(kind, snr, drift=30e3, seed=seed), FS, 5800, 14300)
        assert abs(d["t0"] - 5999.5) <= 2.0 and abs(d["t1"] - 13999.5) <= 2.0, d
        assert not d["flags"] & ref.BDW_GAUSSIAN
        assert d["snr_db"] == pytest.approx(snr, abs=0.6)


@pytest.mark.parametrize("snr,tol", [(10, 30), (20, 12), (30, 4)])
def test_bdw_gaussian_bursts_take_the_likelihood_edge(snr, tol):
    """A noise-like burst fades through its own half-power level; its edge is
    the likelihood boundary, and its own fades set how well it is placed."""
    for seed in range(6):
        d = ref.burst_measure(_burst("flat", snr, seed=seed), FS, 5800, 14300)
        assert d["flags"] & ref.BDW_GAUSSIAN
        assert abs(d["t0"] - 5999.5) <= tol and abs(d["t1"] - 13999.5) <= tol, d
        assert d["flags"] & ref.BDW_RISE_UNRESOLVED


@pytest.mark.parametrize("snr", [10, 20, 30])
def test_bdw_occupied_bandwidth_within_a_bin(snr):
    for seed in range(4):
        d = ref.burst_measure(_burst("flat", snr, B=10e3, seed=seed), FS, 5800, 14300)
        assert abs(d["obw99_hz"] - 0.99 * 10e3) <= FS / d["nfft"], d["obw99_hz"]
        assert abs(d["fc_hz"] - 5e3) <= FS / d["nfft"] * 2


@pytest.mark.parametrize("snr", [10, 20, 30])
def test_bdw_drift_within_five_percent_on_a_carrier(snr):
    for seed in range(4):
        d = ref.burst_measure(_burst("cw", snr, drift=50e3, seed=seed), FS, 5800, 14300)
        assert d["drift_hz_s"] == pytest.approx(50e3, rel=0.05)
        assert d["fc_hz"] == pytest.approx(5e3 + 50e3 * 0.04, abs=30)


def test_bdw_rise_time_on_a_raised_cosine_ramp():
    true = (math.acos(1 - 2 * 0.9) - math.acos(1 - 2 * 0.1)) / math.pi
    d = ref.burst_measure(_burst("cw", 30, ramp=400, seed=1), FS, 5800, 14300)
    assert d["rise_s"] == pytest.approx(true * 400 / FS, rel=0.10)
    assert d["fall_s"] == pytest.approx(true * 400 / FS, rel=0.10)
    assert not d["flags"] & (ref.BDW_RISE_UNRESOLVED | ref.BDW_FALL_UNRESOLVED)
    sharp = ref.burst_measure(_burst("cw", 30, ramp=20, seed=1), FS, 5800, 14300)
    assert sharp["flags"] & ref.BDW_RISE_UNRESOLVED       # shorter than the smoothing


def test_bdw_says_what_it_could_not_measure():
    x = _burst("cw", 20, s=0, e=20000)
    d = ref.burst_measure(x, FS, 100, 19000)
    assert d["flags"] & ref.BDW_TRUNC_START and d["flags"] & ref.BDW_TRUNC_END
    assert d["flags"] & ref.BDW_NO_NOISE and math.isnan(d["snr_db"])
    noise = _burst("cw", -200, seed=3)
    with pytest.raises(LookupError):
        ref.burst_measure(noise, FS, 6000, 7000)
    with pytest.raises(ValueError):
        ref.burst_measure(noise, FS, 7000, 6000)
    if HAVE:
        assert atkdsp.burst_measure(noise, FS, 6000, 7000) is None


@needs_lib
def test_bdw_c_equals_the_twin():
    bad = []
    for kind in ("cw", "fsk", "flat"):
        for snr in (6, 10, 20, 30):
            for seed in range(3):
                x = _burst(kind, snr, drift=30e3, seed=seed, ramp=60 if seed == 2 else 0)
                for i0, i1, th, nf in [(5800, 14300, 0.0, 0), (6100, 13800, 0.0, 256),
                                       (5800, 14300, 6.0, 0), (0, 20000, 0.0, 0)]:
                    try:
                        a = ref.burst_measure(x, FS, i0, i1, th, nf)
                    except LookupError:
                        a = None
                    b = atkdsp.burst_measure(x, FS, i0, i1, th, nf)
                    if (a is None) != (b is None):
                        bad.append((kind, snr, seed, i0, "presence"))
                        continue
                    if a is None:
                        continue
                    for k in a:
                        u, v = a[k], b[k]
                        if isinstance(u, float):
                            if math.isnan(u) or math.isnan(v):
                                if not (math.isnan(u) and math.isnan(v)):
                                    bad.append((kind, snr, k, u, v))
                            elif not abs(u - v) <= 1e-6 * max(1.0, abs(u)):
                                bad.append((kind, snr, k, u, v))
                        elif u != v:
                            bad.append((kind, snr, k, u, v))
    assert not bad, bad[:5]


# =============================================================================
# §21 the ring
# =============================================================================
def test_ring_twin_semantics():
    r = ref.Ring(100, 4, 2)
    assert r.block_bytes == 128
    a = r.open(False)
    for q in range(10):
        r.write(bytes([q]) * 100, first_sample=q * 25)
    data, meta, lost = r.read(a)
    assert lost == 6 and meta["seq"] == 6 and data[0] == 6 and meta["nbytes"] == 100
    b = r.open(True)                      # from the oldest held: seq 6
    assert r.read(b)[1]["seq"] == 6
    view, meta, lost = r.peek(a)
    assert meta["seq"] == 7
    for _ in range(4):                    # lapped while held
        r.write(b"x")
    assert r.release(a) == -6
    out, metas = r.copy_range(0, 100)
    assert [m["seq"] for m in metas] == [10, 11, 12, 13] and out.size == 4
    with pytest.raises(ValueError):
        r.commit(5)                        # no claim
    st = r.stats()
    assert st["committed"] == 14 and st["oldest"] == 10 and st["torn"] == 1


@needs_lib
def test_ring_c_equals_the_twin_on_a_scripted_run():
    rng = np.random.default_rng(1)
    for nb, bb, mc in [(4, 100, 3), (8, 64, 4), (5, 1, 2)]:
        c, t = atkdsp.Ring(bb, nb, mc), ref.Ring(bb, nb, mc)
        ic, it = [c.open(False), c.open(True)], [t.open(False), t.open(True)]
        for step in range(500):
            op = int(rng.integers(0, 10))
            if op < 5:
                n = int(rng.integers(1, t.block_bytes + 1))
                data = rng.integers(0, 256, n).astype(np.uint8).tobytes()
                kw = dict(first_sample=step * 7, t_utc_ns=step, fc_hz=1e6 + step, fs_hz=2e6,
                          gain_db=3.5, format=2, flags=op & 1)
                c.write(data, **kw)
                t.write(data, **kw)
            elif op < 8:
                k = int(rng.integers(0, 2))
                a, b = c.read(ic[k]), t.read(it[k])
                assert (a[0] is None) == (b[0] is None) and a[2] == b[2]
                if a[0] is not None:
                    assert np.array_equal(a[0], b[0]) and a[1] == b[1]
            elif op == 8:
                k = int(rng.integers(0, 2))
                a, b = c.peek(ic[k]), t.peek(it[k])
                assert a[2] == b[2] and (a[0] is None) == (b[0] is None)
                if a[0] is not None:
                    assert np.array_equal(a[0], b[0]) and a[1] == b[1]
                    if rng.integers(0, 2):
                        for _ in range(nb):
                            c.write(b"x")
                            t.write(b"x")
                    assert c.release(ic[k]) == (t.release(it[k]) == 0)
            else:
                s0 = int(rng.integers(0, t.write_seq + 1))
                s1 = s0 + int(rng.integers(0, 6))
                a, b = c.copy_range(s0, s1), t.copy_range(s0, s1)
                assert np.array_equal(a[0], b[0]) and a[1] == b[1]
        sc, st = c.stats(), t.stats()
        for k in ("committed", "oldest", "active", "torn", "lost", "read", "lag"):
            assert sc[k] == st[k], k
        assert c.describe() == t.describe()


@needs_lib
def test_ring_attach_refuses_memory_that_is_not_a_ring():
    lib = atkdsp.load()
    junk = np.zeros(4096, np.uint64)
    assert not lib.atkdsp_ring_attach(junk.ctypes.data, junk.nbytes)
    assert atkdsp.ring_bytes(0, 4) == 0 and atkdsp.ring_bytes(64, 1) == 0
    assert atkdsp.ring_bytes(64, 4, 17) == 0


@needs_lib
def test_a_second_process_attaches_to_a_named_ring_and_reads_the_same_seqs():
    """§A3's acceptance: from Python, another process attaches to a named
    mapping and reads the same seqs (the cross-process half the twin cannot
    do — its declared exception)."""
    name = f"atkdsp_test_{os.getpid()}"
    r = atkdsp.Ring(256, 16, 4, name=name)
    try:
        for q in range(10):
            r.write(bytes([q % 256]) * (100 + q), first_sample=1000 * q, fc_hz=433.92e6)
        code = textwrap.dedent(f"""
            import sys, json
            sys.path.insert(0, {str(ROOT / 'python')!r})
            import atkdsp
            r = atkdsp.Ring.attach({name!r})
            cid = r.open(from_oldest=True)
            out = []
            while True:
                d, m, lost = r.read(cid)
                if d is None:
                    break
                out.append([m["seq"], m["first_sample"], int(d.size), int(d[0]), m["fc_hz"]])
            r.close_consumer(cid)
            print(json.dumps({{"blocks": out, "describe": r.describe()}}))
            r.close()
        """)
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           timeout=60)
        assert p.returncode == 0, p.stderr
        import json
        got = json.loads(p.stdout.strip().splitlines()[-1])
        assert got["blocks"] == [[q, 1000 * q, 100 + q, q, 433.92e6] for q in range(10)]
        assert "10 committed" in got["describe"]
        assert r.stats()["active"] == 0          # the other process closed its id
    finally:
        r.close(unlink=True)


@needs_lib
def test_the_ring_stress_program_holds():
    exe = ROOT / "bin" / ("atkdsp_ring_stress.exe" if sys.platform == "win32"
                          else "atkdsp_ring_stress")
    if not exe.exists():
        pytest.skip("the stress program is built by build.sh / build.bat")
    p = subprocess.run([str(exe), "1"], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and "RING STRESS: ok" in p.stdout, p.stdout + p.stderr


# =============================================================================
# Regressions from the independent review (2026-10-28)
# =============================================================================
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("mode", [0, 1, 2, 3], ids=["CA", "OS", "GO", "SO"])
def test_a_nan_reference_cell_means_no_hit_in_every_mode(impl, mode):
    """GO / SO dropped a NaN side when it was the left one, and OS's
    quickselect had no order for NaN: a NaN anywhere in a bin's reference now
    makes its statistic NaN, so it is not hit — C and twin alike."""
    nb = 64
    row = np.zeros(nb, np.float32)
    row[20] = 40.0                         # a strong cell under test
    for side in (16, 24):                  # NaN in the left window, then the right
        r = row.copy()
        r[side] = np.nan
        c = _cfar(impl, nb, mode=mode, n_ref=4, n_guard=1, open_m=1, open_n=1)
        c.process(r, 0.0, 1.0)
        _x, thr, hits = c.last()
        assert hits[20] == 0 and np.isnan(thr[20]), (side, thr[20])


@pytest.mark.parametrize("a", [6.4e5, 2e4])
def test_the_incomplete_gamma_holds_for_large_shapes(a):
    """A fixed 1000 iterations stopped short past a ~ 5e4 (P(6.4e5, 6.4e5)
    came out 0.39). P(a, a) = 1/2 + 1/(3 sqrt(2 pi a)) + O(1/a)."""
    lp, _lq = ref.log_igam_pq(a, a)
    assert math.exp(lp) == pytest.approx(0.5 + 1 / (3 * math.sqrt(2 * math.pi * a)), abs=2e-4)
    lpv, _ = ref._v_log_igam_pq(a, np.array([a]))
    assert lpv[0] == pytest.approx(lp, abs=1e-12)


def test_ca_alpha_at_a_large_look_count_matches_its_closed_form():
    K, N = 2000, 32
    a, _ = ref.cfar_alpha(ref.CFAR_CA, 16, 1e-6, float(K))
    x, A = N / (N + a), N * K
    terms = [math.lgamma(A + j) - math.lgamma(j + 1) - math.lgamma(A) + A * math.log(x)
             + j * math.log1p(-x) for j in range(K)]
    m = max(terms)
    p = math.exp(m) * sum(math.exp(t - m) for t in terms)
    assert p == pytest.approx(1e-6, rel=1e-6)


@pytest.mark.parametrize("impl", ["twin", pytest.param("c", marks=needs_lib)])
def test_a_long_burst_measures_its_drift_without_overrunning(impl):
    """The drift segments were not capped: past 524 288 samples the C FFT ran
    on a truncated segment, past ~1.57 M it wrote beyond its buffer, and the
    twin raised. Segments are now at most 32768 samples, spread over the burst."""
    n, s, e = 900_000, 50_000, 850_000
    rg = np.random.default_rng(2)
    x = ((rg.standard_normal(n) + 1j * rg.standard_normal(n)) / math.sqrt(2)).astype(np.complex64)
    t = np.arange(e - s) / FS
    x[s:e] += (10 * np.exp(2j * np.pi * (5e3 * t + 0.5 * 250.0 * t * t))).astype(np.complex64)
    f = ref.burst_measure if impl == "twin" else atkdsp.burst_measure
    d = f(x, FS, s - 100, e + 100)
    assert d["drift_hz_s"] == pytest.approx(250.0, rel=0.02)
    assert abs(d["t0"] - (s - 0.5)) <= 2 and abs(d["t1"] - (e - 0.5)) <= 2


@pytest.mark.parametrize("margin", [300, 100])
def test_a_short_noise_margin_is_not_mistaken_for_truncation(margin):
    """With less noise than the L/8 guard, N fell to 0 and the smoothed core
    reached the buffer's end: TRUNC_START / TRUNC_END on a whole burst."""
    L = 4000
    x = _burst("cw", 40.0, n=L + 2 * margin, s=margin, e=margin + L, seed=1)
    for f in [ref.burst_measure] + ([atkdsp.burst_measure] if HAVE else []):
        d = f(x, FS, margin, margin + L)
        assert abs(d["t0"] - (margin - 0.5)) <= 2 and abs(d["t1"] - (margin + L - 0.5)) <= 2
        assert not d["flags"] & (ref.BDW_TRUNC_START | ref.BDW_TRUNC_END | ref.BDW_NO_NOISE)


def test_the_twin_refuses_a_nan_threshold_as_the_c_does():
    x = _burst("cw", 20.0)
    with pytest.raises(ValueError):
        ref.burst_measure(x, FS, 5800, 14300, thresh_db=float("nan"))
    if HAVE:
        with pytest.raises(atkdsp.AtkDspError):
            atkdsp.burst_measure(x, FS, 5800, 14300, thresh_db=float("nan"))


@needs_lib
def test_ring_views_keep_their_memory_after_close_and_collection():
    """claim() / peek() views were built from raw addresses: touching one
    after close() or after the Ring was collected read freed memory."""
    import gc
    r = atkdsp.Ring(64, 4, 2)
    blk = r.claim()
    del r
    gc.collect()
    blk[:4] = 7                                   # the view owns the buffer now
    assert int(blk[0]) == 7
    name = f"atkdsp_view_{os.getpid()}"
    r = atkdsp.Ring(64, 4, 2, name=name)
    cid = r.open(True)
    r.write(b"abcd")
    v, _m, _l = r.peek(cid)
    r.close(unlink=True)
    assert v.tobytes() == b"abcd"


@needs_lib
def test_ring_attach_refuses_a_zero_size_header():
    lib = atkdsp.load()
    mem = np.zeros(32, np.uint64)                  # 256 bytes: a header and nothing else
    mem[0] = 0x31474E49524B5441                    # the magic
    mem[1] = 1 | (256 << 32)                       # version 1, header 256
    mem[6] = 1                                     # max_consumers = 1, the rest zero
    assert not lib.atkdsp_ring_attach(mem.ctypes.data, mem.nbytes)
