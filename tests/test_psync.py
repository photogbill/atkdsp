"""Undertow pieces 2, 4 and 6 (atkdsp.h §30; docs/ATKDSP_PLAN.md §B6c, piece 2's
measured note): the survey's candidate periods, the period-synchronous watch
(psync), the bootstrap (psync -> the code) and the hit blob. C against the twin
first, then the behaviour each piece is judged by: deeper than the fluctuation
handle, quiet on noise, refusing symbol-clocked data that is not a repeating
code, following a drifting period, recovering the code through a carrier
offset, and a blob that C and the twin write byte for byte. `needs_lib` tests
skip when the library is not built.

The scene is a scaled survey cell (240 kS/s input, 8 channels, the code at
channel 2 filling 0.75 of the cell spacing — the survey's matched scale), so
the twin runs in about a second.

Run from the repo root:  python -m pytest tests/test_psync.py -q
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

FS, NCH, CH = 240e3, 8, 2
B = FS / NCH
FC = CH * B
FCELL = 2.0 * B
RC = 0.75 * B
L = 31
CODE = np.array([1.0 if b else -1.0 for b in
                 [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0]])
T = L * FCELL / RC                              # 82.667 cell samples


def wide(seconds, snr_db, seed, df=0.0, signal=True, drift=0.0):
    """The input stream at FS: a data-modulated 31-chip code (one bit a
    period) at FC + df in white noise; snr_db in-band (over RC). drift: the
    chip rate's relative change per second (a linear ramp)."""
    rng = np.random.default_rng(seed)
    n = int(FS * seconds)
    t = np.arange(n)
    s2 = (1.0 / 10 ** (snr_db / 10.0) / RC) * FS
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * math.sqrt(s2 / 2)
    if signal:
        ts = t / FS
        ph = RC * (ts + 0.5 * drift * ts * ts)              # chips elapsed
        ci = np.floor(ph).astype(np.int64)
        sym = rng.choice([-1.0, 1.0], size=int(ci.max()) // L + 2)
        x = x + CODE[ci % L] * sym[ci // L] * np.exp(2j * np.pi * (FC + df) / FS * t)
    return x.astype(np.complex64)


def cell(x):
    return atkdsp.Pfb(FS, NCH).process(x)[CH].copy() if HAVE else ref.Pfb(FS, NCH).process(x)[CH].copy()


def _match(chips, code):
    c = np.asarray(chips, np.int8)
    if c.size != code.size:
        return False
    for sg in (1, -1):
        for sh in range(code.size):
            if np.array_equal(c, (sg * np.roll(code, sh)).astype(np.int8)):
                return True
    return False


# ---- candidates ---------------------------------------------------------------
def test_twin_lags_find_the_period_below_the_handle():
    y = ref.Pfb(FS, NCH).process(wide(1.0, -10.0, 102))[CH]          # the handle is silent here (measured)
    d = ref.DsssDet(dict(ref.dsss_det_defaults(FCELL, 512, 256), period_min=16, handles=1, nwin_min=2))
    d.process(y)
    lags = d.lags(3)
    assert len(lags) == 3
    assert not (d.read()["flags"] & ref.DSSS_FLUCT)
    assert lags[0]["ratio"] >= lags[1]["ratio"] >= lags[2]["ratio"]
    assert any(abs(g["period"] - T) < 1.0 for g in lags)


@needs_lib
def test_lags_c_matches_twin():
    y = cell(wide(1.0, -10.0, 102))
    cfg = dict(ref.dsss_det_defaults(FCELL, 512, 256), period_min=16, handles=1, nwin_min=2)
    dt, dc = ref.DsssDet(cfg), atkdsp.DsssDet(cfg)
    dt.process(y); dc.process(y)
    a, b = dt.lags(4), dc.lags(4)
    assert [g["lag"] for g in a] == [g["lag"] for g in b]
    for g, h in zip(a, b):
        for k in ("period", "ratio", "log10_pfa"):
            assert abs(g[k] - h[k]) <= 1e-6 * max(1.0, abs(g[k])), k


@needs_lib
def test_candidates_c_matches_twin_and_hold_the_period():
    x = wide(2.0, -8.0, 8)
    cfg = atkdsp.undertow_defaults(FS, 3, 32)
    uc, ut = atkdsp.Undertow(cfg), ref.Undertow(ref.undertow_defaults(FS, 3, 32))
    uc.process(x); ut.process(x)
    cc, ct = uc.candidates(-1.0, 3), ut.candidates(-1.0, 3)
    assert len(cc) == len(ct) > 0
    for a, b in zip(cc, ct):
        assert (a["scale"], a["chan"], a["lag"]) == (b["scale"], b["chan"], b["lag"])
        assert abs(a["log10_pfa"] - b["log10_pfa"]) <= 1e-6 * max(1.0, abs(b["log10_pfa"]))
    # the code's own cell (scale 0, channel 2) offers its period or a multiple
    mine = [c for c in cc if c["scale"] == 0 and c["chan"] == CH]
    per = [c["period_s"] * c["fs_cell"] / T for c in mine]
    assert any(abs(p - round(p)) < 0.02 and round(p) >= 1 for p in per), per


# ---- psync: C against the twin ------------------------------------------------
@needs_lib
def test_psync_c_matches_twin():
    y = cell(wide(1.0, -10.0, 3, df=4e3))
    cfg = ref.psync_defaults(FCELL, (T + 0.3) / FCELL)
    rt, wt = ref.psync(y, cfg)
    rc, wc = atkdsp.psync(y, cfg)
    for k in ("flags", "nperiods", "n0_used", "levels", "evals", "window", "method"):
        assert rt[k] == rc[k], k
    for k in ("period_samples", "z", "log10_pfa", "lambda1", "lambda2", "ref_mean", "ref_sd", "dominance"):
        assert abs(rt[k] - rc[k]) <= 1e-6 * max(1.0, abs(rt[k])), (k, rt[k], rc[k])
    assert abs(np.vdot(wt, wc)) / np.linalg.norm(wt) / np.linalg.norm(wc) > 1 - 1e-6


def test_psync_refuses_bad_configs():
    y = np.zeros(20000, np.complex64)
    good = ref.psync_defaults(FCELL, T / FCELL)
    for k, v in (("tol", 0.0), ("tol", T), ("nref", 7), ("nref", 9), ("n0", 4), ("pfa", 0.7), ("esc_pfa", 0.0)):
        bad = dict(good); bad[k] = v
        with pytest.raises(ValueError):
            ref.psync(y, bad)
        if HAVE:
            with pytest.raises(atkdsp.AtkDspError):
                atkdsp.psync(y, bad)


# ---- psync: the behaviour -----------------------------------------------------
@needs_lib
def test_psync_finds_the_period_where_the_fluctuation_handle_cannot():
    """At -15 dB in 4 s of this scaled cell (60 kS/s — a tenth of the real
    survey's rate) the fluctuation handle is silent; psync, seeded a third of
    a sample off, finds the period to a millionth of itself (measured: 4/4
    seeds, z 9-35)."""
    y = cell(wide(4.0, -15.0, 201))
    d = atkdsp.DsssDet(dict(atkdsp.dsss_det_defaults(FCELL, 512, 256), period_min=16, handles=1, nwin_min=2))
    d.process(y)
    assert not (d.read()["flags"] & atkdsp.DSSS_FLUCT)
    r, _ = atkdsp.psync(y, atkdsp.psync_defaults(FCELL, (T + 0.3) / FCELL))
    assert r["flags"] & atkdsp.PS_DETECT and not (r["flags"] & atkdsp.PS_MULTI)
    assert abs(r["period_samples"] - T) < 1e-3
    assert r["log10_pfa"] < -3.0


@needs_lib
def test_psync_is_quiet_on_noise():
    for seed in range(6):
        y = cell(wide(1.0, 0.0, 500 + seed, signal=False))
        r, _ = atkdsp.psync(y, atkdsp.psync_defaults(FCELL, (T + 0.3) / FCELL))
        assert not (r["flags"] & atkdsp.PS_DETECT), (seed, r["z"], r["log10_pfa"])


@needs_lib
def test_psync_same_answer_at_any_thread_count():
    y = cell(wide(1.0, -12.0, 13))
    lib = atkdsp.load()
    old = lib.atkdsp_get_threads()
    try:
        out = []
        for th in (1, 2):
            lib.atkdsp_set_threads(th)
            out.append(atkdsp.psync(y, atkdsp.psync_defaults(FCELL, (T + 0.3) / FCELL)))
        assert out[0][0] == out[1][0]
        assert np.array_equal(out[0][1], out[1][1])
    finally:
        lib.atkdsp_set_threads(old)


@needs_lib
def test_psync_refuses_symbol_clocked_data():
    """Plain QPSK bursts 20 dB up fire the fluctuation handle at the symbol
    clock (measured: log10 Pfa ~ -120); psync at those lags finds no repeating
    code — no detection, or MULTI — and the bootstrap refuses."""
    rng = np.random.default_rng(12345)
    n = int(FS * 1.0)
    w = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2)).astype(np.complex64)
    bl = int(0.005 * FS)
    for _ in range(max(1, int(0.01 * n / bl))):
        s = int(rng.integers(0, n - bl))
        k = np.floor(np.arange(bl) / (FS / (0.31 * B))).astype(np.int64)
        sy = np.exp(1j * np.pi / 2 * rng.integers(0, 4, k.max() + 1))[k]
        w[s:s + bl] += (10.0 * sy * np.exp(2j * np.pi * (FC + 0.1 * B) / FS * np.arange(bl))).astype(np.complex64)
    y = cell(w)
    d = atkdsp.DsssDet(dict(atkdsp.dsss_det_defaults(FCELL, 512, 256), period_min=16, handles=1, nwin_min=2))
    d.process(y)
    assert d.read()["flags"] & atkdsp.DSSS_FLUCT                 # the survey's false DSSS hit
    for g in d.lags(3):
        r, _ = atkdsp.psync(y, atkdsp.psync_defaults(FCELL, g["period"] / FCELL))
        assert not (r["flags"] & atkdsp.PS_DETECT) or (r["flags"] & atkdsp.PS_MULTI), (g, r["flags"], r["z"])
        chips, b = atkdsp.bootstrap(y, atkdsp.bootstrap_defaults(FCELL, g["period"] / FCELL, B))
        assert chips.size == 0 and b["flags"] & atkdsp.BS_REFUSED


@needs_lib
def test_psync_follows_a_drifting_period():
    """A chip clock ramping 20 ppm/s slips the period ~10 samples (4 chips)
    over 4 s — what a satellite's Doppler rate does over minutes: without the
    rate search the alignment smears; with it lambda1 rises (measured +9 %)
    and the rate comes back with its sign and to about +-30 % — the precision
    nearest-sample window starts allow at this SNR and watch (lambda1 over the
    rate is bumpy at the 1-2 % level; measured, the true rate is its peak).
    z is not the measure here: a strong drifting code leaks unevenly into the
    mistuned references and widens their spread. (At 4 ppm/s the slip is
    under a chip and the rate is not identifiable at all — measured flat.)"""
    drift = 2e-5
    y = cell(wide(4.0, -10.0, 21, drift=drift))
    c0 = atkdsp.psync_defaults(FCELL, (T + 0.3) / FCELL)
    r0, _ = atkdsp.psync(y, c0)
    c1 = dict(c0, rate_max=5e-5)
    r1, _ = atkdsp.psync(y, c1)
    assert r1["flags"] & atkdsp.PS_RATE and r1["flags"] & atkdsp.PS_DETECT
    assert r1["method"][2] == 28 and atkdsp.method(28)[3] == "ATK"
    assert -1.4 * drift < r1["rate"] < -0.6 * drift, r1["rate"]
    assert r1["lambda1"] > 1.05 * r0["lambda1"], (r0["lambda1"], r1["lambda1"])


# ---- the bootstrap ------------------------------------------------------------
@needs_lib
def test_bootstrap_resolves_a_multiple_of_the_period():
    """Candidates are often 2T or 3T (the fluctuation peaks there too); psync
    rightly says MULTI there, and the bootstrap finds T itself."""
    y = cell(wide(2.0, -10.0, 102))
    for k in (2, 3):
        r, _ = atkdsp.psync(y, atkdsp.psync_defaults(FCELL, (k * T + 0.3) / FCELL))
        assert r["flags"] & atkdsp.PS_DETECT and r["flags"] & atkdsp.PS_MULTI
        chips, b = atkdsp.bootstrap(y, atkdsp.bootstrap_defaults(FCELL, (k * T + 0.3) / FCELL, B))
        assert b["flags"] & atkdsp.BS_HARMONIC and b["harmonic"] == k
        assert b["flags"] & atkdsp.BS_CODE and _match(chips, CODE)
        assert abs(b["ps"]["period_samples"] - T) < 1e-3


@needs_lib
def test_bootstrap_c_matches_twin():
    y = cell(wide(1.0, -10.0, 3, df=4e3))
    cfg = ref.bootstrap_defaults(FCELL, (T + 0.3) / FCELL, B)
    ct, bt = ref.bootstrap(y, cfg)
    cc, bc = atkdsp.bootstrap(y, cfg)
    assert np.array_equal(ct, cc)
    assert bt["flags"] == bc["flags"] and bt["L"] == bc["L"]
    assert bt["code"]["name"] == bc["code"]["name"]
    assert abs(bt["cfo_hz"] - bc["cfo_hz"]) < 1e-6 * FCELL


@needs_lib
@pytest.mark.parametrize("snr,df", [(-10.0, 0.0), (-14.0, 9e3), (-16.0, -6e3)])
def test_bootstrap_recovers_and_names_the_code(snr, df):
    y = cell(wide(2.0, snr, 31, df=df))
    chips, b = atkdsp.bootstrap(y, atkdsp.bootstrap_defaults(FCELL, (T + 0.3) / FCELL, B))
    assert b["flags"] & atkdsp.BS_CODE, (b["flags"], b["ps"]["flags"])
    assert b["L"] == L and _match(chips, CODE)
    assert b["code"]["name"].startswith("m-sequence (31)")
    assert abs(b["rc_hz"] - RC) < 1e-3 * RC
    assert abs(b["cfo_hz"] - df) < 50.0
    assert b["method"][0] == 29


@needs_lib
def test_bootstrap_with_a_known_chip_rate():
    y = cell(wide(1.0, -12.0, 41))
    cfg = dict(atkdsp.bootstrap_defaults(FCELL, (T + 0.3) / FCELL, B), rc_hz=RC)
    chips, b = atkdsp.bootstrap(y, cfg)
    assert b["flags"] & atkdsp.BS_RC_GIVEN and b["flags"] & atkdsp.BS_CODE and _match(chips, CODE)


# ---- the hit blob -------------------------------------------------------------
def _a_hit():
    return dict(fc_hz=FC, bw_hz=B, fluct_log10_pfa=-7.5, period_s=T / FCELL, acf_log10_pfa=-2.0, acf_z=3.1,
                cyc_log10_pfa=0.0, cyc_alpha_hz=0.0, cyc_z=0.0, rc_hz=0.0, L=0.0, log10_pfa=-4.2,
                scale=0, chan=CH, nwin=1200, nseen=3, flags=9, method=[24, 1, 25, 26])


def test_twin_blob_round_trip_and_rejects_damage():
    ctx = dict(t_unix=1.7e9, rf_center_hz=915e6, fs_survey=FS, source="bug sweep, room 2")
    chips = CODE.astype(np.int8)
    b = ref.hit_export(ctx, _a_hit(), None, chips)
    c2, h2, bs2, ch2 = ref.hit_import(b)
    assert c2 == ctx and bs2 is None and np.array_equal(ch2, chips)
    assert all(h2[k] == v for k, v in _a_hit().items())
    bad = bytearray(b); bad[40] ^= 1
    with pytest.raises(ValueError):
        ref.hit_import(bytes(bad))
    with pytest.raises(ValueError):
        ref.hit_import(b[:-1])


@needs_lib
def test_blob_c_equals_twin_byte_for_byte():
    y = cell(wide(1.0, -10.0, 3, df=4e3))
    chips, bs = atkdsp.bootstrap(y, atkdsp.bootstrap_defaults(FCELL, (T + 0.3) / FCELL, B))
    ctx = dict(t_unix=1.7e9, rf_center_hz=915e6, fs_survey=FS, source="bug sweep")
    bc = atkdsp.hit_export(ctx, _a_hit(), bs, chips)
    bt = ref.hit_export(ctx, _a_hit(), bs, chips)
    assert bc == bt
    for imp in (atkdsp.hit_import, ref.hit_import):
        c2, h2, b2, ch2 = imp(bc)
        assert c2["source"] == "bug sweep" and h2["nseen"] == 3 and np.array_equal(ch2, chips)
        assert b2["code"]["name"] == bs["code"]["name"] and b2["L"] == bs["L"]
        assert b2["ps"]["period_samples"] == bs["ps"]["period_samples"]
    bad = bytearray(bc); bad[100] ^= 4
    with pytest.raises(atkdsp.AtkDspError):
        atkdsp.hit_import(bytes(bad))
