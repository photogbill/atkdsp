"""Undertow piece 5 — the full lag scan (atkdsp.h §31; docs/ATKDSP_PLAN.md
§B6c). The FFA and the null's saddlepoint first, then C against the twin row
for row and family for family, then what the scan is judged by: the period
of a code below where the survey's candidates hold it; a strong code — whose
partial folds detect in dozens of rows — as ONE family at its true period;
two codes in one cell as two; a period below the scanned range stepped down
to; noise quiet at the stated rate; the same answer at any thread count; its
refusals. `needs_lib` tests skip when the library is not built.

The scene is the scaled survey cell of tests/test_psync.py: 240 kS/s input,
8 channels, a 31-chip code with one data bit a period at channel 2, filling
0.75 of the spacing — period 82.667 cell samples.

Run from the repo root:  python -m pytest tests/test_lagscan.py -q
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
FCELL = 2.0 * B
RC = 0.75 * B
L = 31
CODE = np.array([1.0 if b else -1.0 for b in
                 [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0]])
T = L * FCELL / RC                              # 82.667 cell samples
C15 = np.array([1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1], float)
RC2 = 0.55 * B
T2 = 15 * FCELL / RC2                           # 54.545 cell samples


def wide(seconds, snr_db, seed, signal=True):
    rng = np.random.default_rng(seed)
    n = int(FS * seconds)
    t = np.arange(n)
    s2 = (1.0 / 10 ** (snr_db / 10.0) / RC) * FS
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * math.sqrt(s2 / 2)
    if signal:
        ci = np.floor(t * (RC / FS)).astype(np.int64)
        sym = rng.choice([-1.0, 1.0], size=int(ci.max()) // L + 2)
        x = x + CODE[ci % L] * sym[ci // L] * np.exp(2j * np.pi * (CH * B) / FS * t)
    return x.astype(np.complex64)


def two_codes(seconds, snr_db, seed):
    """The 31-chip code and a 15-chip one at 0.55 of the spacing, 3 kHz off,
    each snr_db in its own band."""
    rng = np.random.default_rng(seed)
    n = int(FS * seconds)
    t = np.arange(n)
    s2 = (1.0 / 10 ** (snr_db / 10.0) / RC) * FS
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * math.sqrt(s2 / 2)
    ci = np.floor(t * (RC / FS)).astype(np.int64)
    sym = rng.choice([-1.0, 1.0], size=int(ci.max()) // L + 2)
    x = x + CODE[ci % L] * sym[ci // L] * np.exp(2j * np.pi * (CH * B) / FS * t)
    cj = np.floor(t * (RC2 / FS)).astype(np.int64)
    sym2 = rng.choice([-1.0, 1.0], size=int(cj.max()) // 15 + 2)
    x = x + math.sqrt(RC2 / RC) * C15[cj % 15] * sym2[cj // 15] * np.exp(2j * np.pi * (CH * B + 3000.0) / FS * t)
    return x.astype(np.complex64)


def cell(x):
    return atkdsp.Pfb(FS, NCH).process(x)[CH].copy() if HAVE else ref.Pfb(FS, NCH).process(x)[CH].copy()


# ---- the FFA and the null ----------------------------------------------------------
def test_ffa_folds_every_trial_period():
    """Row s of the FFA, drift x = s/(n-1) a row, folds the period
    P / (1 - x/P) — P bins over a period a little longer than P. A pattern
    planted at a fractional period comes out within the tree's rounding of
    it (each of log2(n) stages rounds its shift; measured, at most a few
    trials). The first-order P + x is short by x^2/P: at n = 8,192 that was
    44 trials off at 82.67 and 80 at 82.9 (measured) — the families' direct
    folds over the whole record could not find the period again."""
    rng = np.random.default_rng(0)
    pat = rng.standard_normal(64)
    for P, n in ((50, 256), (82, 2048)):
        for per in np.linspace(P, P + 0.99, 12):
            t = np.arange(P * n)
            x = pat[np.floor((t % per) / per * 64).astype(int)].astype(complex)
            s = int(np.argmax(np.sum(np.abs(ref.ffa(x.reshape(n, P))) ** 2, axis=1)))
            got = P / (1.0 - (s / (n - 1)) / P)
            assert abs(got - per) * (n - 1) < math.log2(n) / 2 + 1, (P, n, per, s)


def test_lags_are_spread_and_distinct():
    assert ref.lagscan_lags(82, 12, 0.5, 8) == [12, 16, 20, 24, 29, 33, 37, 41]
    assert ref.lagscan_lags(16, 12, 0.5, 8) == [12, 13]
    assert ref.lagscan_lags(256, 12, 0.5, 1) == [12]


def test_the_saddlepoint_tail_is_exact_where_it_can_be_checked():
    """Distinct eigenvalues have a closed-form tail (a sum of exponentials);
    equal ones are a Gamma. Lugannani-Rice holds both to 0.02 in log10 down
    to 1e-16 (measured: 0.017 at the deepest), the Gamma to 0.002."""
    lam = np.array([1.0, 0.7, 0.45, 0.2])

    def exact(x):
        return sum(np.prod([lj / (lj - lk) for k, lk in enumerate(lam) if k != j]) * math.exp(-x / lj)
                   for j, lj in enumerate(lam))
    for x in (3.0, 6.0, 10.0, 15.0, 25.0, 40.0):
        assert abs(ref.lagscan_tail(x, lam, 1.0) - math.log10(exact(x))) < 0.02, x
    for a, x in ((40, 55.0), (40, 80.0), (200, 260.0), (200, 330.0), (5, 30.0)):
        assert abs(ref.lagscan_tail(x, np.ones(1), float(a)) - ref.log_igamc(a, x) / math.log(10)) < 0.002


def test_the_null_spectrum_is_the_cells():
    """White noise: lag products uncorrelated, the spectrum flat at 1. A PFB
    cell (2x oversampled): neighbouring lag products correlate (|r(1)|^2 about
    0.46, measured) and the spectrum spreads to about twice its mean — the
    spread that makes a moment-matched Gamma's far tail too thin."""
    rng = np.random.default_rng(2)
    w = rng.standard_normal(200000) + 1j * rng.standard_normal(200000)
    rw = ref.lagscan_acf(w)
    assert rw[0] == 1.0 and np.all(rw[1:] < 1e-3)
    lw = ref.lagscan_spectrum(rw, 64, False)
    assert np.all(np.abs(lw - 1.0) < 0.02)
    rc = ref.lagscan_acf(ref.Pfb(FS, NCH).process(w.astype(np.complex64))[CH])
    assert 0.35 < rc[1] < 0.6
    lc = ref.lagscan_spectrum(rc, 82, False)
    assert abs(lc.mean() - 1.0) < 0.05 and 1.7 < lc.max() / lc.mean() < 2.3


# ---- C against the twin ---------------------------------------------------------
@needs_lib
@pytest.mark.parametrize("snr,lo,hi", [(-12.0, 60, 110), (-6.0, 40, 120)])
def test_c_equals_twin_row_for_row(snr, lo, hi):
    y = cell(wide(1.0, snr, 3))
    cfg = atkdsp.lagscan_defaults(FCELL, lo, hi)
    rc, fc, nc = atkdsp.lagscan(y, cfg)
    rt, ft, nt = ref.lagscan(y, dict(cfg))
    assert nc == nt and len(rc) == len(rt) == hi - lo
    for a, b in zip(rc, rt):
        assert a["base"] == b["base"] and a["nfold"] == b["nfold"]
        assert (a["flags"], a["family"], a["harmonic_p"], a["harmonic_q"]) == \
               (b["flags"], b["family"], b["harmonic_p"], b["harmonic_q"]), a["base"]
        if a["nfold"]:
            assert abs(a["period_samples"] - b["period_samples"]) < 1e-6
            assert abs(a["log10_pfa_raw"] - b["log10_pfa_raw"]) < 1e-5 * max(1.0, abs(b["log10_pfa_raw"]))
            assert abs(a["excess"] - b["excess"]) < 1e-5 * max(1.0, abs(b["excess"]))
    assert len(fc) == len(ft) >= 1
    for a, b in zip(fc, ft):
        assert (a["members"], a["flags"], a["seed_base"], a["seed_p"], a["seed_q"]) == \
               (b["members"], b["flags"], b["seed_base"], b["seed_p"], b["seed_q"])
        assert abs(a["period_samples"] - b["period_samples"]) < 1e-6 * b["period_samples"]
        assert abs(a["fold_snr"] - b["fold_snr"]) < 1e-5 * max(1.0, abs(b["fold_snr"]))


# ---- what it is for ---------------------------------------------------------------
@needs_lib
def test_the_scan_finds_the_period_where_the_candidates_do_not():
    """-14 dB, 4 s: the survey's candidates no longer hold the period (the
    candidate rows of its cell are elsewhere — measured); the scan's first
    family holds it to a thousandth of a sample a period (measured: 4e-5)."""
    x = wide(4.0, -14.0, 41)
    u = atkdsp.Undertow(atkdsp.undertow_defaults(FS, 4, 64))
    u.process(x)
    held = [c for c in u.candidates(-1.0, 3) if c["scale"] == 0 and c["chan"] == CH
            and abs(c["period_s"] * c["fs_cell"] - T) < 1.0]
    assert not held
    rows, fams, nt = atkdsp.lagscan(cell(x), atkdsp.lagscan_defaults(FCELL, 16, 257))
    assert fams and abs(fams[0]["period_samples"] - T) < 1e-3, fams[:2]
    assert fams[0]["log10_pfa"] < -5 and fams[0]["fold_snr"] > 0.5 and nt > 1e5
    assert len(fams) == 1


@needs_lib
def test_a_strong_code_is_one_family():
    """-6 dB, 1 s: the code folds partly at 2T/3, 3T/4, 2T, 3T ... and 83 rows
    detect (measured); the most significant is 2T/3, not T. They are one
    family at T: nested folds step 2T/3 up to 2T and down to T, and every
    other detection is a p/q of it or explained by it (CLEAN)."""
    rows, fams, _ = atkdsp.lagscan(cell(wide(1.0, -6.0, 3)), atkdsp.lagscan_defaults(FCELL, 16, 257))
    det = [r for r in rows if r["flags"] & ref.LAGSCAN_DETECT]
    assert len(det) > 50 and len(fams) == 1
    f = fams[0]
    assert abs(f["period_samples"] - T) < 1e-3 and f["members"] == len(det)
    assert all(r["family"] == 0 for r in det)
    fund = [r for r in det if r["flags"] & ref.LAGSCAN_FUNDAMENTAL]
    assert fund and all(abs(r["period_samples"] - T) < 0.2 for r in fund)


@needs_lib
def test_two_codes_are_two_families():
    """A 31-chip code (period 82.667) and a 15-chip one at another rate
    (54.545), -8 dB each, 1 s: two families, each at its period, nothing
    else — the second is not explained by the first."""
    rows, fams, _ = atkdsp.lagscan(cell(two_codes(1.0, -8.0, 7)), atkdsp.lagscan_defaults(FCELL, 16, 257))
    assert len(fams) == 2
    got = sorted(f["period_samples"] for f in fams)
    assert abs(got[0] - T2) < 2e-3 and abs(got[1] - T) < 2e-3, got


@needs_lib
def test_a_period_below_the_scan_is_stepped_down_to():
    """Scanning 100..256 only, the code's 2T, 3T ... detect; the family steps
    down to T (82.667) and says it lies outside the scan."""
    rows, fams, _ = atkdsp.lagscan(cell(wide(1.0, -8.0, 9)), atkdsp.lagscan_defaults(FCELL, 100, 257))
    assert len(fams) == 1
    f = fams[0]
    assert abs(f["period_samples"] - T) < 2e-3
    assert f["flags"] & ref.LAGSCAN_FAM_DOWN and f["flags"] & ref.LAGSCAN_FAM_OUTSIDE


@needs_lib
def test_noise_is_quiet_at_the_stated_rate():
    """Noise-only cells, 1 s, every period 16..256: no family at pfa 1e-3,
    and each cell's best corrected tail spread as stated (P(best <= x) =
    10^x: about 0.32 of cells at or under -0.5). Measured over 316 cells
    (2 s, 4 s, real rate): 0.26-0.44 at -0.5, 0.05-0.13 at -1, 0.01 at -2."""
    best, fams_all = [], 0
    for seed in range(24):
        y = cell(wide(1.0, 0.0, 9000 + seed, signal=False))
        rows, fams, _ = atkdsp.lagscan(y, atkdsp.lagscan_defaults(FCELL, 16, 257))
        fams_all += len(fams)
        best.append(min(r["log10_pfa"] for r in rows if r["nfold"]))
    best = np.array(best)
    assert fams_all == 0
    assert 0.1 <= np.mean(best <= -0.5) <= 0.6, best


@needs_lib
def test_the_same_answer_at_any_thread_count():
    y = cell(wide(1.0, -10.0, 5))
    cfg = atkdsp.lagscan_defaults(FCELL, 30, 130)
    before = atkdsp.get_threads()
    try:
        outs = []
        for th in (1, 2, 3):
            atkdsp.set_threads(th)
            outs.append(atkdsp.lagscan(y, cfg))
    finally:
        atkdsp.set_threads(before)
    assert outs[0][1]                              # a family to compare
    for o in outs[1:]:
        assert o[0] == outs[0][0] and o[1] == outs[0][1]


@needs_lib
def test_short_data_is_marked_not_scanned():
    y = cell(wide(0.05, 0.0, 1, signal=False))          # 3,000 cell samples
    rows, fams, nt = atkdsp.lagscan(y, atkdsp.lagscan_defaults(FCELL, 100, 260))
    short = [r for r in rows if r["flags"] & ref.LAGSCAN_SHORT]
    assert short and all(r["nfold"] == 0 for r in short)
    assert all(r["base"] * 16 > y.size - 128 for r in short)


@needs_lib
def test_no_families_asked_none_made():
    y = cell(wide(1.0, -8.0, 9))
    rows, fams, _ = atkdsp.lagscan(y, atkdsp.lagscan_defaults(FCELL, 60, 110), fcap=0)
    assert fams == [] and all(r["family"] == -1 for r in rows)
    assert any(r["flags"] & ref.LAGSCAN_DETECT for r in rows)


def test_twin_refuses_bad_configs():
    y = np.zeros(1000, complex)
    for k, v in (("pmin", 1), ("pmax", 10), ("nlags", 0), ("lag_min", 0), ("lag_frac", 0.0),
                 ("nmax", 1000), ("pfa", 0.0), ("trim", 0.6)):
        cfg = dict(ref.lagscan_defaults(1.0, 16, 64), **{k: v})
        if k == "pmax":
            cfg["pmin"] = 16
        with pytest.raises(ValueError):
            ref.lagscan(y, cfg)
    with pytest.raises(ValueError):
        ref.lagscan(y, ref.lagscan_defaults(1.0, 16, 64), fcap=-1)


@needs_lib
def test_c_refuses_bad_configs():
    y = np.zeros(1000, np.complex64)
    for k, v in (("pmin", 1), ("nlags", 0), ("nmax", 1000), ("pfa", 1.5)):
        with pytest.raises(atkdsp.AtkDspError):
            atkdsp.lagscan(y, dict(atkdsp.lagscan_defaults(1.0, 16, 64), **{k: v}))


@needs_lib
def test_the_method_is_named():
    m = atkdsp.method(ref.METHOD_LAGSCAN)
    assert m and "fast folding" in m[0] and "Staelin" in m[1]
    assert ref.method(ref.METHOD_LAGSCAN)[0] == m[0]
