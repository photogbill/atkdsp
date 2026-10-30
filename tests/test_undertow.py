"""Undertow, piece 1 — the scale-space survey (atkdsp.h §29; docs/ATKDSP_PLAN.md
§B6c). Twin first (the numpy spec), C against the twin, then the measured
behaviour the survey is judged by: no survey-wide false alarms on empty
coloured noise, a below-the-floor code found with its exact period in the right
cell, the same answer at any thread count, and the period_min rule (below the
channelized noise's correlation length the noise's own autocorrelation reads
as a period — the finding that rule encodes). `needs_lib` tests skip when the
library is not built.

Run from the repo root:  python -m pytest tests/test_undertow.py -q
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

TAPS = {31: (5, 2), 63: (6, 1), 127: (7, 1)}


def mseq(L):
    m, k = TAPS[L]
    reg = [1] + [0] * (m - 1)
    out = []
    for _ in range(L):
        out.append(reg[-1])
        fb = reg[m - 1] ^ reg[k - 1]
        reg = [fb] + reg[:-1]
    return np.array([1.0 if b else -1.0 for b in out])


def scene(fs, n, snr_db, rc, fc, L=31, seed=0, signal=True, coloured=True):
    """A data-modulated DSSS code (one bit per period) in noise; snr_db is
    in-band (signal over noise in a bandwidth of rc). Coloured = a gentle
    front-end tilt, the realistic case."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.zeros(n, complex)
    if signal:
        ci = np.floor(t / (fs / rc)).astype(np.int64)
        c = mseq(L)[ci % L] * rng.choice([-1.0, 1.0], size=ci.max() // L + 1)[ci // L]
        x = c * np.exp(2j * np.pi * fc / fs * t)
    s2 = (1.0 / 10 ** (snr_db / 10.0) / rc) * fs
    w = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * math.sqrt(s2 / 2)
    if coloured:
        h = np.array([1.0, 0.18, -0.06])
        w = np.convolve(w, h, mode="same") / np.sqrt(np.sum(h ** 2))
    return (x + w).astype(np.complex64)


# the twin runs a scaled scene (fs/10, rc/10, fc/10): identical in cell samples
FS_T, RC_T, FC_T = 240e3, 10e3, 31.2e3
FS, RC, FC = 2.4e6, 100e3, 312e3


# ---- the Student-t tail -------------------------------------------------------
def test_t_tail_known_values():
    assert abs(ref.t_log10_sf2(0.0, 5) - 0.0) < 1e-12               # P(|T|>0) = 1
    assert abs(ref.t_log10_sf2(1.0, 1) - math.log10(0.5)) < 1e-12   # Cauchy
    # large nu -> normal: P(|Z| > 1.959964) = 0.05
    assert abs(10 ** ref.t_log10_sf2(1.959964, 1e6) - 0.05) < 1e-4
    assert ref.t_log10_sf2(50.0, 7) < -7.0


@needs_lib
def test_t_tail_c_matches_twin():
    worst = 0.0
    for nu in (1, 2, 3, 5, 7, 30, 1000):
        for t in (0.0, 0.3, 1.0, 2.5, 6.0, 20.0, 300.0):
            worst = max(worst, abs(atkdsp.t_log10_sf2(t, nu) - ref.t_log10_sf2(t, nu)))
    assert worst < 1e-12


# ---- the twin -----------------------------------------------------------------
def _twin_run(x, nscales=4, lmax=64, blocks=50000):
    u = ref.Undertow(ref.undertow_defaults(FS_T, nscales=nscales, lmax=lmax))
    for i in range(0, x.size, blocks):
        u.process(x[i:i + blocks])
    return u


def test_twin_noise_only_has_no_hits():
    x = scene(FS_T, int(FS_T * 1.5), -6, RC_T, FC_T, signal=False, seed=3)
    assert _twin_run(x).read() == []


def test_twin_finds_the_code_with_its_period():
    x = scene(FS_T, int(FS_T * 2.0), 0.0, RC_T, FC_T, seed=42)
    hits = _twin_run(x).read()
    assert len(hits) == 1
    h = hits[0]
    assert h["flags"] & ref.UT_FLUCT and h["flags"] & ref.UT_PERIOD
    assert abs(h["period_s"] - 31 / RC_T) < 0.01 * 31 / RC_T
    assert abs(h["fc_hz"] - FC_T) <= h["bw_hz"]
    assert h["nseen"] >= 2                     # one emitter, several cells, merged


def test_twin_chunk_invariance():
    x = scene(FS_T, int(FS_T * 0.6), 0.0, RC_T, FC_T, seed=5)
    cfg = ref.undertow_defaults(FS_T, nscales=3, lmax=32)
    u1 = ref.Undertow(cfg); u1.process(x)
    u2 = ref.Undertow(cfg)
    for a, b in ((0, 1), (1, 7000), (7000, 7001), (7001, 50000), (50000, x.size)):
        u2.process(x[a:b])
    keys = ("fluct_log10_pfa", "period_s", "acf_log10_pfa", "cyc_log10_pfa", "cyc_alpha_hz", "flags", "nwin")
    worst = max(abs(float(a[k]) - float(b[k])) for a, b in zip(u1.cells(), u2.cells()) for k in keys)
    assert worst < 1e-9


def test_twin_refuses_bad_configs():
    good = ref.undertow_defaults(FS_T, 3, 32)
    for k, v in (("nchan0", 6), ("nscales", 0), ("period_min", 200), ("win", 300), ("pfa", 0.7),
                 ("seg_len", 100), ("acf_lags", 0)):
        bad = dict(good); bad[k] = v
        with pytest.raises(ValueError):
            ref.Undertow(bad)


# ---- C against the twin ---------------------------------------------------------
@needs_lib
def test_defaults_and_describe_match_the_twin():
    c = atkdsp.undertow_defaults(FS_T, 3, 32)
    t = ref.undertow_defaults(FS_T, 3, 32)
    assert all(c[k] == t[k] for k in t)
    assert atkdsp.Undertow(c).describe() == ref.Undertow(t).describe()


@needs_lib
def test_c_cells_and_hits_match_the_twin():
    x = scene(FS_T, int(FS_T * 1.5), -2.0, RC_T, FC_T, seed=11)
    uc = atkdsp.Undertow(atkdsp.undertow_defaults(FS_T, 3, 32))
    ut = ref.Undertow(ref.undertow_defaults(FS_T, 3, 32))
    for a, b in ((0, 7000), (7000, 7001), (7001, 90000), (90000, x.size)):
        uc.process(x[a:b]); ut.process(x[a:b])
    C, T = uc.cells(), ut.cells()
    assert len(C) == len(T) == 56
    for k in ("fluct_log10_pfa", "period_s", "acf_log10_pfa", "acf_z", "cyc_log10_pfa", "cyc_z",
              "cyc_alpha_hz", "fc_hz", "bw_hz"):
        d = max(abs(a[k] - b[k]) / max(1.0, abs(b[k])) for a, b in zip(C, T))
        assert d < 1e-4, f"{k}: {d:.2e}"
    for a, b in zip(C, T):
        for k in ("scale", "chan", "nwin", "flags", "method"):
            assert a[k] == b[k], (k, a[k], b[k])
    hc, ht = uc.read(), ut.read()
    assert [(h["scale"], h["chan"], h["nseen"], h["flags"]) for h in hc] == \
           [(h["scale"], h["chan"], h["nseen"], h["flags"]) for h in ht]


@needs_lib
def test_c_refuses_bad_configs():
    good = atkdsp.undertow_defaults(FS_T, 3, 32)
    for k, v in (("nchan0", 6), ("period_min", 200), ("pfa", 0.7)):
        bad = dict(good); bad[k] = v
        with pytest.raises(atkdsp.AtkDspError):
            atkdsp.Undertow(bad)


# ---- the measured behaviour, at the real rate ----------------------------------
def _c_run(x, lmax=64, nscales=4, **over):
    cfg = atkdsp.undertow_defaults(FS, nscales, lmax)
    cfg.update(over)
    u = atkdsp.Undertow(cfg)
    for i in range(0, x.size, 262144):
        u.process(x[i:i + 262144])
    return u


@needs_lib
def test_no_false_alarms_on_empty_coloured_noise():
    for seed in range(4):
        x = scene(FS, int(FS * 1.0), -6, RC, FC, signal=False, seed=100 + seed)
        assert _c_run(x).read() == [], f"false alarm on empty noise (seed {seed})"


@needs_lib
def test_finds_a_code_below_the_floor_with_its_period():
    x = scene(FS, int(FS * 2.0), -6.0, RC, FC, seed=7)
    hits = _c_run(x).read()
    assert len(hits) == 1
    h = hits[0]
    assert h["flags"] & atkdsp.UT_FLUCT and h["flags"] & atkdsp.UT_PERIOD
    assert abs(h["period_s"] - 31 / RC) < 0.01 * 31 / RC
    assert abs(h["fc_hz"] - FC) <= h["bw_hz"]
    assert h["log10_pfa"] < -3.0
    assert h["method"][0] == 24 and atkdsp.method(24)[3] == "ATK"


@needs_lib
def test_the_same_answer_at_any_thread_count():
    x = scene(FS, int(FS * 0.5), -3.0, RC, FC, seed=9)
    lib = atkdsp.load()
    old = lib.atkdsp_get_threads()
    try:
        out = []
        for th in (1, 2):
            lib.atkdsp_set_threads(th)
            out.append(_c_run(x).cells())
        assert out[0] == out[1]
    finally:
        lib.atkdsp_set_threads(old)


@needs_lib
def test_period_min_below_the_noise_correlation_false_alarms():
    """Why period_min defaults to 16: a 2x-oversampled cell's noise is
    correlated out to ~lag 10, and with hundreds of windows the Gamma model is
    so tight that the noise's own small-lag bump reads as a period. This
    documents the failure the default prevents."""
    x = scene(FS, int(FS * 1.0), -6, RC, FC, signal=False, seed=31)
    assert _c_run(x, period_min=4).read() != []
    assert _c_run(x).read() == []
