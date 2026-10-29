"""Phase 3 (atkdsp.h §22-24; docs/ATKDSP_PLAN.md §B3, §B4, §C3): live pulse
runs, the cyclic probes, the analog radio IDs — known answers first (rule 2),
noise and the wrong signal at the design rate (rule 3), C against its twin in
uneven blocks (rule 1). Tests marked `needs_lib` skip when the library is not
built; the rest run on the twins alone.

Run from the repo root:  python -m pytest tests/test_phase3.py -q
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
IMPLS = ["twin", pytest.param("c", marks=needs_lib)]
BLOCKS = [1, 7, 239, 240, 241, 999, 4096, 8192]


# =============================================================================
# §22 pulse runs
# =============================================================================
# ATK's atk.core.pulse.detect_pulses (2026-09-29 vectorised form), copied here
# as the oracle: the kernel must give its words for the same floor.
def _oracle_segment_sums(v, lo, hi):
    if lo.size == 0:
        return np.zeros(0, dtype=np.float64)
    idx = np.empty(2 * lo.size, dtype=np.int64)
    idx[0::2] = lo
    idx[1::2] = hi
    last = idx[-1] >= v.size
    if last:
        idx[-1] = v.size - 1
    sums = np.add.reduceat(v.astype(np.float64, copy=False), idx)[0::2]
    if last:
        sums[-1] += float(v[-1])
    return sums


def oracle_detect_pulses(iq, sample_rate, threshold_db=10.0, min_samples=4):
    x = np.asarray(iq)
    mag = np.abs(x).astype(np.float64)
    med = float(np.median(mag))
    floor = med if med > 0 else float(np.mean(mag))
    thresh = floor * (10.0 ** (threshold_db / 20.0))
    above = mag > thresh
    if not above.any():
        return floor, []
    edges = np.diff(above.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if above[0]:
        starts.insert(0, 0)
    if above[-1]:
        ends.append(mag.size)
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    keep = (ends - starts) >= int(min_samples)
    starts, ends = starts[keep], ends[keep]
    widths = ends - starts
    amp = _oracle_segment_sums(mag, starts, ends) / widths
    inst = np.angle(x[1:] * np.conj(x[:-1])).astype(np.float32)
    inst *= np.float32(sample_rate / (2.0 * np.pi))
    n_inst = widths - 1
    trim = np.maximum(1, n_inst // 10)
    core_lo = starts + trim
    core_hi = ends - 1 - trim
    short = core_hi - core_lo < 1
    core_lo = np.where(short, starts, core_lo)
    core_hi = np.where(short, starts + np.maximum(n_inst, 1), core_hi)
    core_hi = np.minimum(core_hi, inst.size)
    core_lo = np.minimum(core_lo, core_hi - 1)
    core_n = (core_hi - core_lo).astype(np.float64)
    s1 = _oracle_segment_sums(inst, core_lo, core_hi)
    s2 = _oracle_segment_sums(inst * inst, core_lo, core_hi)
    mean_f = s1 / core_n
    var = np.maximum(s2 / core_n - mean_f * mean_f, 0.0)
    chirp = 2.77 * np.sqrt(var)
    chirp = np.where(widths < 4, 0.0, chirp)
    mean_f = np.where(widths < 4, 0.0, mean_f)
    amp_db = 20.0 * np.log10(np.maximum(amp / floor, 1e-12))
    return floor, [{"start": int(s), "end": int(e), "toa_s": float(s / sample_rate),
                    "width_s": float(w / sample_rate), "amplitude_db": float(adb),
                    "freq_hz": float(f), "chirp_hz": float(c)}
                   for s, e, w, adb, f, c in zip(starts, ends, widths, amp_db, mean_f, chirp)]


def radar_scene(seed=1, fs=10e6, n=400_000, pri=2000, pw=80, snr_db=20.0, chirp_hz=0.0,
                f0=123_456.0, start=500):
    """Rectangular pulses on complex Gaussian noise of unit power, a carrier
    at f0 (plus a linear sweep of chirp_hz across the pulse). Returns the
    scene and the planted (start, width) pairs."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2)
    a = 10 ** (snr_db / 20)
    planted = []
    t = np.arange(pw) / fs
    for s in range(start, n - pw, pri):
        ph = 2 * np.pi * (f0 * t + 0.5 * chirp_hz / (pw / fs) * t * t)
        x[s:s + pw] += a * np.exp(1j * ph)
        planted.append((s, pw))
    return x.astype(np.complex64), planted


def _pulses(impl, **cfg):
    return atkdsp.Pulses(0.0, **cfg) if impl == "c" else ref.Pulses(0.0, **cfg)


def _run_blocks(p, x, blocks):
    out = []
    i = 0
    j = 0
    while i < x.size:
        b = blocks[j % len(blocks)]
        out += p.process(x[i:i + b])
        i += b
        j += 1
    out += p.flush()
    return out


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("chirp", [0.0, 2e6])
def test_the_words_are_pulse_pys_for_the_same_floor(impl, chirp):
    fs = 10e6
    x, planted = radar_scene(seed=2, fs=fs, chirp_hz=chirp)
    floor, want = oracle_detect_pulses(x, fs)
    got = (atkdsp.detect_pdws if impl == "c" else ref.detect_pdws)(x, fs)
    assert len(got) == len(want) == len(planted)
    for g, w, (s, pw) in zip(got, want, planted):
        assert g["start"] == w["start"] == s and g["end"] == w["end"] == s + pw
        assert g["toa"] == pytest.approx(w["toa_s"], abs=1.0 / fs)        # within a sample
        assert g["pw"] == pytest.approx(w["width_s"], abs=2.0 / fs)
        assert g["pa_db"] == pytest.approx(w["amplitude_db"], abs=1e-3)
        assert g["freq_hz"] == pytest.approx(w["freq_hz"], abs=1e-6 * fs)
        assert g["chirp_hz"] == pytest.approx(w["chirp_hz"], abs=1e-6 * fs)
        assert g["flags"] == 0


@pytest.mark.parametrize("impl", IMPLS)
def test_a_pulse_straddling_a_block_boundary_is_reported_once_at_every_chunk_size(impl):
    fs = 10e6
    x, planted = radar_scene(seed=3, fs=fs, n=60_000, pri=3000, pw=50)
    floor = ref.mag_median(x)
    whole = _run_blocks(_pulses(impl, fs=fs, floor=floor), x, [x.size])
    assert len(whole) == len(planted)
    for blocks in ([1], [7], [239, 240, 241], [999], [4096], [8192], [3001, 17, 2983]):
        got = _run_blocks(_pulses(impl, fs=fs, floor=floor), x, blocks)
        assert len(got) == len(whole), blocks
        for g, w in zip(got, whole):
            for k in ("start", "end", "n", "flags"):
                assert g[k] == w[k], (blocks, k)
            for k in ("toa", "pw", "freq_hz", "chirp_hz", "slope_hz_s", "pa_db", "peak_db", "energy"):
                assert g[k] == pytest.approx(w[k], rel=1e-9, abs=1e-12), (blocks, k)


@needs_lib
def test_c_equals_the_twin_in_uneven_blocks():
    fs = 2.5e6
    x, planted = radar_scene(seed=4, fs=fs, n=120_000, pri=2500, pw=120, chirp_hz=4e5, snr_db=14)
    floor = ref.mag_median(x)
    assert atkdsp.mag_median(x) == pytest.approx(floor, rel=1e-7)
    for blocks in ([1], BLOCKS, [120_000]):
        c = _run_blocks(atkdsp.Pulses(0.0, fs=fs, floor=floor, hyst_db=2.0), x, blocks)
        t = _run_blocks(ref.Pulses(0.0, fs=fs, floor=floor, hyst_db=2.0), x, blocks)
        assert len(c) == len(t) == len(planted)
        for g, w in zip(c, t):
            for k in ("start", "end", "n", "flags"):
                assert g[k] == w[k], k
            assert g["toa"] == pytest.approx(w["toa"], rel=1e-12)
            assert g["pw"] == pytest.approx(w["pw"], rel=1e-9)
            assert g["freq_hz"] == pytest.approx(w["freq_hz"], abs=1e-6 * fs)
            assert g["chirp_hz"] == pytest.approx(w["chirp_hz"], abs=1e-6 * fs)
            assert g["slope_hz_s"] == pytest.approx(w["slope_hz_s"], rel=1e-5, abs=1e-6 * fs * fs)
            assert g["pa_db"] == pytest.approx(w["pa_db"], abs=1e-4)
            assert g["peak_db"] == pytest.approx(w["peak_db"], abs=1e-4)
            assert g["energy"] == pytest.approx(w["energy"], rel=1e-6)


@pytest.mark.parametrize("impl", IMPLS)
def test_planted_truth_toa_pw_frequency_and_chirp(impl):
    """Known answers: TOA within a sample, PW within two, the carrier within
    the noise of the estimate, a chirp's span and sign."""
    fs = 20e6
    x, planted = radar_scene(seed=5, fs=fs, n=300_000, pri=5000, pw=200, snr_db=25,
                             chirp_hz=3e6, f0=-1.5e6)
    got = (atkdsp.detect_pdws if impl == "c" else ref.detect_pdws)(x, fs)
    assert len(got) == len(planted)
    for g, (s, pw) in zip(got, planted):
        assert abs(g["toa"] * fs - s) <= 1.0
        assert abs(g["pw"] * fs - pw) <= 2.0
        # the sweep runs from f0 to f0 + chirp over the pulse: its mean is the
        # middle; the core keeps 80 % of it, so the span reads about 0.8 chirp
        assert g["freq_hz"] == pytest.approx(-1.5e6 + 1.5e6, abs=0.05 * 3e6)
        assert 0.6 * 3e6 < g["chirp_hz"] < 1.0 * 3e6
        assert g["slope_hz_s"] == pytest.approx(3e6 / (pw / fs), rel=0.1)
        assert g["pa_db"] > 20.0


@pytest.mark.parametrize("impl", IMPLS)
def test_noise_alone_yields_nothing_at_the_design_rate(impl):
    """10 dB over the Rayleigh median is 3.72 sigma: a sample crosses with
    probability 9.8e-4, four in a row 1e-12 — nothing in 2 M samples."""
    rng = np.random.default_rng(6)
    x = ((rng.standard_normal(2_000_000) + 1j * rng.standard_normal(2_000_000)) / math.sqrt(2)).astype(np.complex64)
    got = (atkdsp.detect_pdws if impl == "c" else ref.detect_pdws)(x, 1e6)
    assert got == []
    # and with min_samples = 1 the single-sample crossings appear at about
    # the stated rate (they are counted as short runs, min 4 default)
    p = _pulses(impl, fs=1e6, floor=ref.mag_median(x), min_samples=1)
    one = p.process(x) + p.flush()
    rate = len(one) / x.size
    assert 0.6e-3 < rate < 1.4e-3, rate


@pytest.mark.parametrize("impl", IMPLS)
def test_hysteresis_keeps_a_dipping_pulse_whole(impl):
    fs = 1e6
    rng = np.random.default_rng(7)
    x = ((rng.standard_normal(20_000) + 1j * rng.standard_normal(20_000)) / math.sqrt(2))
    floor = float(np.median(np.abs(x)))
    a = floor * 10 ** (12 / 20)                     # 12 dB over the floor
    env = np.full(400, a)
    env[200:203] = floor * 10 ** (8.5 / 20)         # a dip 1.5 dB under the 10 dB line
    x[5000:5400] = env * np.exp(1j * 0.3 * np.arange(400))
    x = x.astype(np.complex64)
    split = _pulses(impl, fs=fs, floor=floor)
    whole = _pulses(impl, fs=fs, floor=floor, hyst_db=3.0)
    assert len(split.process(x) + split.flush()) == 2
    one = whole.process(x) + whole.flush()
    assert len(one) == 1 and one[0]["n"] == 400


@pytest.mark.parametrize("impl", IMPLS)
def test_a_long_run_is_truncated_but_timed_from_its_real_edges(impl):
    fs = 1e6
    rng = np.random.default_rng(8)
    x = ((rng.standard_normal(50_000) + 1j * rng.standard_normal(50_000)) / math.sqrt(2))
    floor = float(np.median(np.abs(x)))
    x[1000:31000] += 10 * floor * np.exp(1j * 0.1 * np.arange(30000))
    x = x.astype(np.complex64)
    p = _pulses(impl, fs=fs, floor=floor, max_samples=4096)
    got = p.process(x) + p.flush()
    assert len(got) == 1
    g = got[0]
    assert g["flags"] & atkdsp.PDW_TRUNCATED and g["n"] == 30000 and g["end"] - g["start"] == 30000
    assert g["pw"] == pytest.approx(30000 / fs, abs=3 / fs)
    assert g["freq_hz"] == pytest.approx(0.1 / (2 * np.pi) * fs, rel=0.02)
    assert p.stats()["truncated"] == 1


@pytest.mark.parametrize("impl", IMPLS)
def test_a_pulse_open_at_the_end_is_flushed_and_one_at_the_start_is_edge_marked(impl):
    fs = 1e6
    rng = np.random.default_rng(9)
    x = ((rng.standard_normal(3000) + 1j * rng.standard_normal(3000)) / math.sqrt(2))
    floor = float(np.median(np.abs(x)))
    x[:50] += 10 * floor
    x[2900:] += 10 * floor
    x = x.astype(np.complex64)
    p = _pulses(impl, fs=fs, floor=floor)
    got = p.process(x)
    assert len(got) == 1 and got[0]["flags"] & atkdsp.PDW_EDGE_START and got[0]["start"] == 0
    assert got[0]["toa"] == 0.0
    fl = p.flush()
    assert len(fl) == 1 and fl[0]["flags"] & atkdsp.PDW_FLUSHED and fl[0]["end"] == 3000
    assert p.flush() == []


@pytest.mark.parametrize("impl", IMPLS)
def test_the_output_cap_drops_and_counts_rather_than_losing_state(impl):
    fs = 1e6
    x, planted = radar_scene(seed=10, fs=fs, n=100_000, pri=1000, pw=20, snr_db=20)
    floor = ref.mag_median(x)
    p = _pulses(impl, fs=fs, floor=floor)
    got = p.process(x, cap=10)
    assert len(got) == 10 and p.stats()["dropped"] == len(planted) - 10
    assert p.stats()["pulses"] == len(planted)
    assert got[9]["start"] == planted[9][0]


def test_bad_configurations_are_refused_with_the_reason():
    with pytest.raises(ValueError, match="floor"):
        ref.Pulses(fs=1e6, floor=0.0)
    with pytest.raises(ValueError, match="hyst"):
        ref.Pulses(fs=1e6, floor=1.0, hyst_db=20.0, thresh_db=10.0)
    with pytest.raises(ValueError, match="unknown"):
        ref.Pulses(fs=1e6, floor=1.0, colour="red")
    if HAVE:
        with pytest.raises(atkdsp.AtkDspError, match="min_samples"):
            atkdsp.Pulses(fs=1e6, floor=1.0, min_samples=0)


@needs_lib
def test_describe_and_stats_say_the_plan_and_the_counts():
    x, planted = radar_scene(seed=11, fs=1e6, n=50_000, pri=1000, pw=20)
    p = atkdsp.Pulses(0.0, fs=1e6, floor=ref.mag_median(x), hyst_db=2.0)
    d = p.describe()
    assert d.startswith("pulses: fs 1e+06 Hz") and "open at +10.0 dB, close at +8.0 dB" in d
    p.process(x)
    s = p.stats()
    assert s["samples"] == 50_000 and s["pulses"] == len(planted) and s["dropped"] == 0
    assert ref.Pulses(0.0, fs=1e6, floor=p.cfg["floor"], hyst_db=2.0).describe() == d


# =============================================================================
# §23 cyclic probes
# =============================================================================
FS_AUDIO = 48000.0


def _rrc(beta, span, sps):
    t = np.arange(-span * sps, span * sps + 1) / sps
    h = np.zeros_like(t)
    for i, tt in enumerate(t):
        if abs(tt) < 1e-9:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(tt) - 1 / (4 * beta)) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = ((np.sin(np.pi * tt * (1 - beta)) + 4 * beta * tt * np.cos(np.pi * tt * (1 + beta)))
                    / (np.pi * tt * (1 - (4 * beta * tt) ** 2)))
    return h / np.sqrt(np.sum(h ** 2))


def _noise(rng, n, power=1.0):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * math.sqrt(power / 2)


def qpsk_rrc(rng, nsym, baud=4800.0, fs=FS_AUDIO, snr_db=10.0, beta=0.35, fc=0.0):
    sps = int(round(fs / baud))
    s = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, nsym)))
    up = np.zeros(nsym * sps, complex)
    up[::sps] = s
    x = np.convolve(up, _rrc(beta, 6, sps), mode="same")
    x /= np.sqrt(np.mean(np.abs(x) ** 2))
    x *= np.exp(2j * np.pi * fc * np.arange(x.size) / fs)
    return (x + _noise(rng, x.size, 10 ** (-snr_db / 10))).astype(np.complex64)


def fsk4(rng, nsym, baud=4800.0, fs=FS_AUDIO, snr_db=20.0, shaped=False, dev=648.0):
    sps = int(round(fs / baud))
    syms = rng.integers(0, 4, nsym) * 2 - 3
    up = np.repeat(syms, sps).astype(np.float64)
    if shaped:
        h = np.hanning(2 * sps)
        h /= h.sum()
        freq = np.convolve(up, h, mode="same") * dev
    else:
        freq = up * dev
    x = np.exp(1j * np.cumsum(2 * np.pi * freq / fs))
    return (x + _noise(rng, x.size, 10 ** (-snr_db / 10))).astype(np.complex64)


def analog_fm(rng, n, fs=FS_AUDIO, snr_db=20.0):
    """Voice-like FM: three tones whose frequencies wander (a random walk of
    +-25 % over a second, so no line stands still — steady tones WOULD put
    lines at their sums and differences after a square law, and the probe
    is right to report those), 2.5 kHz deviation, constant envelope."""
    m = np.zeros(n)
    for f0, a, ph in ((300.0, 1.0, 0.0), (1130.0, 0.6, 1.0), (2470.0, 0.3, 2.0)):
        walk = np.cumsum(rng.standard_normal(n)) / math.sqrt(fs)
        f = f0 * (1.0 + 0.25 * np.tanh(walk))
        m += a * np.sin(np.cumsum(2 * np.pi * f / fs) + ph)
    m /= np.max(np.abs(m))
    x = np.exp(1j * np.cumsum(2 * np.pi * 2500 * m / fs))
    return (x + _noise(rng, n, 10 ** (-snr_db / 10))).astype(np.complex64)


def _cyc(impl, fs, cands, seg, nseg, pfa=1e-3):
    if impl == "c":
        return atkdsp.Cyclic(fs, cands, seg, nseg, pfa)
    return ref.Cyclic(fs, [(r, atkdsp._nl_code(n)) + tuple(c[2:]) for c, r, n in
                           ((c, c[0], c[1]) for c in cands)], seg, nseg, pfa)


RATE_CANDS = [(4800, "env2"), (2400, "env2"), (1200, "env2"), (3600, "env2"),
              (4800, "dm"), (2400, "dm"), (4800, "fdiff"), (2400, "fdiff"), (3600, "fdiff"),
              (4800, "pow4"), (2400, "pow4")]


@pytest.mark.parametrize("impl", IMPLS)
def test_a_shaped_qpsk_gives_its_clock_line_and_no_other(impl):
    rng = np.random.default_rng(21)
    x = qpsk_rrc(rng, 9600)
    c = _cyc(impl, FS_AUDIO, RATE_CANDS, 960, 50)
    c.process(x)
    res = c.read()
    by = {(r, n): v for (r, n), v in zip(RATE_CANDS, res)}
    for key in ((4800, "env2"), (4800, "dm"), (4800, "pow4")):
        assert by[key]["flags"] & atkdsp.CYC_LINE, (key, by[key])
        assert by[key]["coherence"] > 0.6 and by[key]["level"] > 50
        assert abs(by[key]["delta_hz"]) < 2.5                    # the rate is exact (noise: a hertz or two)
    for key in ((2400, "env2"), (1200, "env2"), (3600, "env2"), (2400, "dm"), (3600, "fdiff")):
        assert not by[key]["flags"] & atkdsp.CYC_LINE, (key, by[key])


@pytest.mark.parametrize("impl", IMPLS)
def test_fsk_shows_its_clock_through_the_transition_nonlinearities(impl):
    """A rectangular-pulse 4FSK clocks through the frequency transitions
    (FDIFF), a shaped one through the sample-to-sample energy (DM) — and a
    clock line has harmonics, so 9600 may read with 4800."""
    rng = np.random.default_rng(22)
    cands = [(4800, "fdiff"), (2400, "fdiff"), (3600, "fdiff"), (4800, "dm"), (2400, "dm"), (3600, "dm"),
             (4800, "env2")]
    rect, shaped = fsk4(rng, 9600, shaped=False), fsk4(rng, 9600, shaped=True)
    for sig, nonlin in ((rect, "fdiff"), (shaped, "dm")):
        c = _cyc(impl, FS_AUDIO, cands, 960, 50)
        c.process(sig)
        by = {(r, n): v for (r, n), v in zip(cands, c.read())}
        assert by[(4800, nonlin)]["flags"] & atkdsp.CYC_LINE, (nonlin, by[(4800, nonlin)])
        for r in (2400, 3600):
            assert not by[(r, nonlin)]["flags"] & atkdsp.CYC_LINE, (r, nonlin)
        assert not by[(4800, "env2")]["flags"] & atkdsp.CYC_LINE     # constant envelope: no |x|^2 line


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("kind", ["noise", "fm"])
def test_noise_and_analog_fm_give_no_line(impl, kind):
    rng = np.random.default_rng(23)
    x = _noise(rng, 96000).astype(np.complex64) if kind == "noise" else analog_fm(rng, 96000)
    c = _cyc(impl, FS_AUDIO, RATE_CANDS, 960, 50)
    c.process(x)
    for r in c.read():
        assert not r["flags"] & atkdsp.CYC_LINE, r
        assert r["flags"] == 0 and r["nseg"] == 50


@needs_lib
def test_the_line_rate_on_noise_sits_at_or_below_the_design_rate():
    """Rule 3, measured: 60 candidates x 300 trials of noise at Pfa 0.3 for
    the search (0.005 each), 32 segments. The derived thresholds are
    conservative by design (the Rayleigh-walk bound), so the count must sit
    inside the binomial interval's top and above a quarter of the mean."""
    rng = np.random.default_rng(24)
    cands = [(300.0 + 73.0 * i, ["env2", "dm", "fdiff", "pow2"][i % 4]) for i in range(60)]
    lines = 0
    trials = 300
    for _ in range(trials):
        c = atkdsp.Cyclic(FS_AUDIO, cands, 240, 32, 0.3)
        c.process(_noise(rng, 240 * 32).astype(np.complex64))
        lines += sum(1 for r in c.read() if r["flags"] & atkdsp.CYC_LINE)
    n = trials * 60
    mu = n * 0.005
    sd = math.sqrt(mu)
    assert lines <= mu + 2.6 * sd, (lines, mu)
    assert lines >= 0.25 * mu, (lines, mu)


@needs_lib
def test_cyclic_c_equals_the_twin_in_uneven_blocks():
    rng = np.random.default_rng(25)
    x = qpsk_rrc(rng, 4000, snr_db=5.0, fc=1234.5)
    cands = [(4800, "env2"), (2400, "dm"), (4800, "fdiff"), (4800, "pow4", 1), (1200, "pow2", 1)]
    t = ref.Cyclic(FS_AUDIO, [(r, atkdsp._nl_code(n)) + tuple(c[2:]) for c in cands for r, n in [c[:2]]],
                   800, 20)
    t.process(x)
    want = t.read()
    for blocks in ([1] * 5 + [4096], [7], [239, 240, 241], [999], [x.size]):
        c = atkdsp.Cyclic(FS_AUDIO, cands, 800, 20)
        i = 0
        j = 0
        while i < x.size:
            b = blocks[j % len(blocks)]
            c.process(x[i:i + b])
            i += b
            j += 1
        got = c.read()
        for g, w in zip(got, want):
            assert g["nseg"] == w["nseg"] and g["flags"] == w["flags"]
            for k in ("level", "coherence", "delta_hz", "level_aligned", "coherence_aligned"):
                assert g[k] == pytest.approx(w[k], rel=1e-7, abs=1e-9), (blocks, k)
        assert c.stats()["segments"] == t.stats()["segments"]


def test_cyclic_refuses_a_search_it_cannot_decide():
    with pytest.raises(ValueError, match="coherence test"):
        ref.Cyclic(FS_AUDIO, [(4800, 0)], 960, 8, 1e-3)
    with pytest.raises(ValueError, match="rate_hz"):
        ref.Cyclic(FS_AUDIO, [(30000, 0)], 960, 50)
    if HAVE:
        with pytest.raises(atkdsp.AtkDspError, match="coherence test"):
            atkdsp.Cyclic(FS_AUDIO, [(4800, "env2")], 960, 8, 1e-3)
        c = atkdsp.Cyclic(FS_AUDIO, [(4800, "env2"), (2400, "dm", 1)], 960, 50)
        d = c.describe()
        assert d.startswith("cyclic: fs 48000 Hz, 2 candidate(s), segments of 960 samples (0.02 s) x 50 (1 s)")
        assert d.endswith("4800 Bd |x|^2, 2400 Bd |x - x[-1]|^2 (limited)")
        assert ref.Cyclic(FS_AUDIO, [(4800, 0), (2400, 4, 1)], 960, 50).describe() == d


# ---- the cyclic-prefix probe ----------------------------------------------------
def ofdm(rng, nsym, N=64, cp=16, used=52, snr_db=10.0):
    out = []
    idx = np.r_[1:used // 2 + 1, N - used // 2:N]
    for _ in range(nsym):
        X = np.zeros(N, complex)
        X[idx] = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, idx.size)))
        x = np.fft.ifft(X) * np.sqrt(N / used)
        out.append(np.concatenate([x[-cp:], x]))
    x = np.concatenate(out)
    x /= np.sqrt(np.mean(np.abs(x) ** 2))
    return (x + _noise(rng, x.size, 10 ** (-snr_db / 10))).astype(np.complex64)


CP_CANDS = [(64, 16), (64, 8), (64, 16, 72), (128, 32), (2048, 144), (32, 8), (256, 0)]


def _cp(impl, fs, cands, seg, nseg, pfa=1e-3):
    return (atkdsp.CpProbe if impl == "c" else ref.CpProbe)(fs, cands, seg, nseg, pfa)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("snr", [10.0, 0.0])
def test_ofdm_is_found_at_its_fft_length_and_its_prefix_is_confirmed(impl, snr):
    rng = np.random.default_rng(31)
    x = ofdm(rng, 2500, snr_db=snr)
    p = _cp(impl, 20e6, CP_CANDS, 4000, 50)
    p.process(x)
    res = p.read()
    by = {c: r for c, r in zip(CP_CANDS, res)}
    g = 10 ** (snr / 10)
    assert by[(64, 16)]["flags"] == atkdsp.CP_DETECT | atkdsp.CP_LINE
    assert by[(64, 16)]["score"] == pytest.approx(0.2 * g / (1 + g), rel=0.15)
    assert by[(64, 16)]["line_coherence"] > 0.95
    assert by[(64, 8)]["flags"] == atkdsp.CP_DETECT                 # right length, wrong prefix
    assert by[(64, 16, 72)]["flags"] == atkdsp.CP_DETECT            # right prefix, wrong period
    for c in ((128, 32), (2048, 144), (32, 8), (256, 0)):
        assert by[c]["flags"] == 0, (c, by[c])
    assert by[(256, 0)]["nseg"] == 0                                 # detection only: no line stage


@pytest.mark.parametrize("impl", IMPLS)
def test_noise_correlates_at_no_lag(impl):
    rng = np.random.default_rng(32)
    p = _cp(impl, 20e6, CP_CANDS, 4000, 50)
    p.process(_noise(rng, 200_000).astype(np.complex64))
    for r in p.read():
        assert r["flags"] == 0, r


@needs_lib
def test_the_score_rate_on_noise_is_the_rayleigh_rate():
    """64 lags on white noise at Pfa 0.32 for the search (0.005 each): the
    score is a Rayleigh of scale 1/sqrt(pairs), and the count of DETECTs
    lands in its binomial interval."""
    rng = np.random.default_rng(33)
    cands = [(lag, 0) for lag in range(1, 65)]
    hits = 0
    trials = 200
    for _ in range(trials):
        p = atkdsp.CpProbe(1e6, cands, 2048, 2, 0.32)
        p.process(_noise(rng, 4096).astype(np.complex64))
        hits += sum(1 for r in p.read() if r["flags"] & atkdsp.CP_DETECT)
    n, pfa = trials * 64, 0.005
    mu, sd = n * pfa, math.sqrt(n * pfa * (1 - pfa))
    assert mu - 2.6 * sd <= hits <= mu + 2.6 * sd, (hits, mu, sd)


@needs_lib
def test_cp_probe_c_equals_the_twin_in_uneven_blocks():
    rng = np.random.default_rng(34)
    x = ofdm(rng, 600, snr_db=3.0)
    t = ref.CpProbe(20e6, CP_CANDS[:4], 2000, 16)
    t.process(x)
    want = t.read()
    for blocks in ([1] * 3 + [4096], [7], [239, 240, 241], [999], [x.size]):
        p = atkdsp.CpProbe(20e6, CP_CANDS[:4], 2000, 16)
        i = j = 0
        while i < x.size:
            b = blocks[j % len(blocks)]
            p.process(x[i:i + b])
            i += b
            j += 1
        for g, w in zip(p.read(), want):
            assert g["nseg"] == w["nseg"] and g["flags"] == w["flags"] and g["npairs"] == w["npairs"]
            for k in ("score", "score_thr", "line_level", "line_coherence", "line_delta_hz"):
                assert g[k] == pytest.approx(w[k], rel=1e-7, abs=1e-9), (blocks, k)


@needs_lib
def test_cp_probe_describe_and_reset():
    p = atkdsp.CpProbe(20e6, [(64, 16), (2048, 144, 15360), (256, 0)], 4000, 20)
    d = p.describe()
    assert d.endswith("lag 64, cp 16, period 80, lag 2048, cp 144, period 15360, lag 256")
    assert ref.CpProbe(20e6, [(64, 16), (2048, 144, 15360), (256, 0)], 4000, 20).describe() == d
    rng = np.random.default_rng(35)
    p.process(ofdm(rng, 500))
    assert p.read()[0]["npairs"] > 0
    p.reset()
    assert p.read()[0]["npairs"] == 0 and p.stats()["resets"] == 1


# =============================================================================
# §24 analog radio IDs
# =============================================================================
FS_RADIO = 8000.0


def radio_voice(rng, n, fs=FS_RADIO, level=0.3, lo=300.0):
    """Voice as a radio passes it: 300..3000 Hz noise with a syllabic
    envelope (the transmitter's audio high-pass is what keeps the sub-audio
    band clear for CTCSS and DCS; a voice fundamental below 300 Hz at full
    level is not something a radio sends)."""
    w = rng.standard_normal(n)
    X = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1 / fs)
    X[(f < lo) | (f > 3000)] = 0
    v = np.fft.irfft(X, n)
    env = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 4 * np.arange(n) / fs))
    v = v * env
    v /= np.max(np.abs(v)) + 1e-9
    return level * v


def ctcss_tone(n, f, fs=FS_RADIO, amp=0.15):
    return amp * np.sin(2 * np.pi * f * np.arange(n) / fs)


def dcs_wave(n, code, fs=FS_RADIO, amp=0.15, invert=False, start_bit=0, flip_bit=None):
    """The 23-bit word repeated at 134.4 Bd as NRZ, through a 300 Hz
    one-pole (a radio's shaping); flip_bit inverts that bit of every word."""
    w = ref.dcs_encode(code)
    bits = [(w >> i) & 1 for i in range(23)]
    if flip_bit is not None:
        bits[flip_bit] ^= 1
    sps = fs / 134.4
    t = np.arange(n)
    bit_idx = ((t / sps).astype(int) + start_bit) % 23
    nrz = np.array([1.0 if bits[b] else -1.0 for b in bit_idx])
    if invert:
        nrz = -nrz
    a = math.exp(-2 * math.pi * 300 / fs)
    y = np.zeros(n)
    s = 0.0
    for i in range(n):
        s = a * s + (1 - a) * nrz[i]
        y[i] = s
    return amp * y


def dtmf_seq(keys, fs=FS_RADIO, on=0.08, off=0.08, amp=0.3, twist_db=0.0, detune=0.0):
    out = []
    for k in keys:
        i = ref.DTMF_KEYS.index(k)
        lo, hi = ref.DTMF_LOW[i // 4] * (1 + detune), ref.DTMF_HIGH[i % 4] * (1 + detune)
        t = np.arange(int(on * fs)) / fs
        out.append(amp * (np.sin(2 * np.pi * lo * t) + 10 ** (twist_db / 20) * np.sin(2 * np.pi * hi * t)))
        out.append(np.zeros(int(off * fs)))
    return np.concatenate(out)


def tone_seq(freqs, dur, fs=FS_RADIO, amp=0.4):
    return np.concatenate([amp * np.sin(2 * np.pi * f * np.arange(int(dur * fs)) / fs) for f in freqs])


def _tone_id(impl, **cfg):
    return atkdsp.ToneId(0.0, **cfg) if impl == "c" else ref.ToneId(0.0, **cfg)


def _run_tone(h, x, blocks=None):
    x = np.asarray(x, np.float32)
    if blocks is None:
        return h.process(x) + h.flush()
    out, i, j = [], 0, 0
    while i < x.size:
        b = blocks[j % len(blocks)]
        out += h.process(x[i:i + b])
        i += b
        j += 1
    return out + h.flush()


def _ends(ev, kind):
    return [e for e in ev if e["kind"] == kind and e["type"] == atkdsp.TONE_END]


@needs_lib
def test_every_ctcss_tone_is_read_under_voice_as_itself():
    rng = np.random.default_rng(41)
    n = int(3.0 * FS_RADIO)
    for i, tone in enumerate(ref.CTCSS_TONES):
        x = radio_voice(rng, n) + ctcss_tone(n, tone) + 0.01 * rng.standard_normal(n)
        ev = _run_tone(atkdsp.ToneId(0.0, fs=FS_RADIO), x)
        ends = _ends(ev, atkdsp.TONE_CTCSS)
        assert len(ends) == 1 and ends[0]["code"] == i, (tone, ev)
        assert abs(ends[0]["freq_hz"] - tone) < 0.2 and ends[0]["t1"] - ends[0]["t0"] > 2.7
        assert ends[0]["text"] == f"CTCSS {tone:.1f}" and ends[0]["level_db"] == pytest.approx(-16.5, abs=0.5)
        assert not [e for e in ev if e["kind"] != atkdsp.TONE_CTCSS]        # no DCS, DTMF or tone


@pytest.mark.parametrize("impl", IMPLS)
def test_a_nonstandard_tone_is_reported_with_its_frequency(impl):
    rng = np.random.default_rng(42)
    n = int(3.0 * FS_RADIO)
    x = radio_voice(rng, n) + ctcss_tone(n, 105.0) + 0.01 * rng.standard_normal(n)
    ends = _ends(_run_tone(_tone_id(impl, fs=FS_RADIO), x), atkdsp.TONE_CTCSS)
    assert len(ends) == 1 and ends[0]["code"] == -1 and ends[0]["flags"] & atkdsp.TONE_F_NONSTANDARD
    assert abs(ends[0]["freq_hz"] - 105.0) < 0.2


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("shift", [120, 180, 240])
def test_the_reverse_burst_is_flagged_and_a_plain_end_is_not(impl, shift):
    rng = np.random.default_rng(43)
    t = np.arange(int(1.68 * FS_RADIO)) / FS_RADIO
    ph = 2 * np.pi * 127.3 * t
    ph[t >= 1.5] += np.deg2rad(shift)
    x = np.concatenate([0.15 * np.sin(ph), np.zeros(int(0.5 * FS_RADIO))])
    x += 0.01 * rng.standard_normal(x.size)
    ends = _ends(_run_tone(_tone_id(impl, fs=FS_RADIO), x), atkdsp.TONE_CTCSS)
    assert len(ends) == 1 and ends[0]["flags"] & atkdsp.TONE_F_REVERSE_BURST and ends[0]["t1"] > 1.6
    plain = np.concatenate([0.15 * np.sin(2 * np.pi * 127.3 * t), np.zeros(int(0.5 * FS_RADIO))])
    plain += 0.01 * rng.standard_normal(plain.size)
    ends = _ends(_run_tone(_tone_id(impl, fs=FS_RADIO), plain), atkdsp.TONE_CTCSS)
    assert len(ends) == 1 and not ends[0]["flags"] & atkdsp.TONE_F_REVERSE_BURST


@needs_lib
def test_every_dcs_code_is_read_under_voice_as_itself_with_its_alias():
    rng = np.random.default_rng(44)
    n = int(2.0 * FS_RADIO)
    for i, code in enumerate(ref.DCS_CODES):
        x = radio_voice(rng, n) + dcs_wave(n, code, start_bit=i % 23) + 0.01 * rng.standard_normal(n)
        ev = _run_tone(atkdsp.ToneId(0.0, fs=FS_RADIO), x)
        d = [e for e in ev if e["kind"] == atkdsp.TONE_DCS]
        assert len(d) == 2 and d[0]["code"] == code and d[1]["code"] == code, (oct(code), ev)
        assert d[0]["alias"] in ref.DCS_CODES and d[0]["text"] == f"DCS {code:03o}N"
        assert not [e for e in ev if e["kind"] == atkdsp.TONE_CTCSS]       # the comb is not a tone


@pytest.mark.parametrize("impl", IMPLS)
def test_an_inverted_dcs_reads_as_its_alias_and_the_alias_table_closes(impl):
    """023 sent inverted is 047 in normal polarity — the same bits. The
    reading says both; the operator's convention picks."""
    n = int(2.0 * FS_RADIO)
    x = dcs_wave(n, 0o023, invert=True)
    d = _ends(_run_tone(_tone_id(impl, fs=FS_RADIO), x), atkdsp.TONE_DCS)
    assert len(d) == 1 and d[0]["code"] == 0o047 and d[0]["alias"] == 0o023
    rc, c, a = (atkdsp.dcs_decode if impl == "c" else ref.dcs_decode)(ref.dcs_encode(0o023), 0)
    assert (rc, c, a) == (0, 0o023, 0o047)
    for code in ref.DCS_CODES:
        _rc, c, a = ref.dcs_decode(ref.dcs_encode(code), 0)
        assert c == code and a in ref.DCS_CODES
        assert ref.dcs_decode(ref.dcs_encode(a), 0)[2] == code           # the alias's alias


@pytest.mark.parametrize("impl", IMPLS)
def test_a_corrected_bit_is_flagged_and_refused_at_zero(impl):
    n = int(2.0 * FS_RADIO)
    x = dcs_wave(n, 0o754, flip_bit=5)
    d1 = _ends(_run_tone(_tone_id(impl, fs=FS_RADIO, dcs_max_errors=1), x), atkdsp.TONE_DCS)
    assert len(d1) == 1 and d1[0]["code"] == 0o754 and d1[0]["flags"] & atkdsp.TONE_F_CORRECTED
    for e in _run_tone(_tone_id(impl, fs=FS_RADIO, dcs_max_errors=0), x):
        assert not e["flags"] & atkdsp.TONE_F_CORRECTED
    clean = _ends(_run_tone(_tone_id(impl, fs=FS_RADIO), dcs_wave(n, 0o754)), atkdsp.TONE_DCS)
    assert len(clean) == 1 and not clean[0]["flags"] & atkdsp.TONE_F_CORRECTED


@pytest.mark.parametrize("impl", IMPLS)
def test_the_dcs_turn_off_tone_is_flagged(impl):
    rng = np.random.default_rng(45)
    x = np.concatenate([dcs_wave(int(1.5 * FS_RADIO), 0o023),
                        0.15 * np.sin(2 * np.pi * 134.4 * np.arange(int(0.2 * FS_RADIO)) / FS_RADIO),
                        np.zeros(int(1.0 * FS_RADIO))])
    x += 0.01 * rng.standard_normal(x.size)
    d = _ends(_run_tone(_tone_id(impl, fs=FS_RADIO), x), atkdsp.TONE_DCS)
    assert len(d) == 1 and d[0]["flags"] & atkdsp.TONE_F_TURNOFF


@pytest.mark.parametrize("impl", IMPLS)
def test_every_dtmf_key_decodes_at_q24_timing_with_its_frequencies(impl):
    rng = np.random.default_rng(46)
    x = dtmf_seq(ref.DTMF_KEYS, on=0.05, off=0.05)                       # 50 ms on / 50 ms off
    x += 0.003 * rng.standard_normal(x.size)
    ends = _ends(_run_tone(_tone_id(impl, fs=FS_RADIO), x), atkdsp.TONE_DTMF)
    assert "".join(chr(e["code"]) for e in ends) == ref.DTMF_KEYS
    for e, k in zip(ends, ref.DTMF_KEYS):
        i = ref.DTMF_KEYS.index(k)
        assert e["freq_hz"] == pytest.approx(ref.DTMF_LOW[i // 4], abs=2.0)
        assert e["freq2_hz"] == pytest.approx(ref.DTMF_HIGH[i % 4], abs=2.0)
        assert 0.03 <= e["t1"] - e["t0"] <= 0.07 and e["text"] == f"DTMF {k}"


@pytest.mark.parametrize("impl", IMPLS)
def test_dtmf_tolerances_twist_detune_and_talk_off(impl):
    rng = np.random.default_rng(47)
    ok = dtmf_seq("7", twist_db=-6.0, detune=0.01)                     # within Q.24: accepted
    assert [chr(e["code"]) for e in _ends(_run_tone(_tone_id(impl, fs=FS_RADIO), ok), atkdsp.TONE_DTMF)] == ["7"]
    bad_twist = dtmf_seq("7", twist_db=-12.0)
    assert not _ends(_run_tone(_tone_id(impl, fs=FS_RADIO), bad_twist), atkdsp.TONE_DTMF)
    detuned = dtmf_seq("7", detune=0.035)                               # beyond 3.5 %: refused
    assert not _ends(_run_tone(_tone_id(impl, fs=FS_RADIO), detuned), atkdsp.TONE_DTMF)
    voice = radio_voice(rng, int(3.0 * FS_RADIO), level=0.6)
    assert not _ends(_run_tone(_tone_id(impl, fs=FS_RADIO), voice), atkdsp.TONE_DTMF)
    short = dtmf_seq("7", on=0.025)                                     # under 40 ms: not a digit
    assert not _ends(_run_tone(_tone_id(impl, fs=FS_RADIO), short), atkdsp.TONE_DTMF)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("table,digits,dur", [("ZVEI1", "12345", 0.07), ("CCIR", "90817", 0.10), ("EEA", "2468", 0.04)])
def test_selcall_tones_are_measured_and_matched(impl, table, digits, dur):
    rng = np.random.default_rng(48)
    freqs = [ref.SELCALL_TABLES[table][d] for d in digits]
    x = tone_seq(freqs, dur) + 0.005 * rng.standard_normal(int(len(freqs) * dur * FS_RADIO))
    ends = _ends(_run_tone(_tone_id(impl, fs=FS_RADIO), x), atkdsp.TONE_TONE)
    assert len(ends) == len(freqs)
    for e, f in zip(ends, freqs):
        assert e["freq_hz"] == pytest.approx(f, abs=1.5)
        assert e["t1"] - e["t0"] == pytest.approx(dur, abs=0.02)
    assert ref.selcall_match([e["freq_hz"] for e in ends], durations=[e["t1"] - e["t0"] for e in ends]) == (table, digits, 0)


@pytest.mark.parametrize("impl", IMPLS)
def test_voice_and_noise_alone_give_no_id(impl):
    rng = np.random.default_rng(49)
    n = int(4.0 * FS_RADIO)
    for x in (radio_voice(rng, n, level=0.8), 0.1 * rng.standard_normal(n), np.zeros(n)):
        assert _run_tone(_tone_id(impl, fs=FS_RADIO), x) == []


@needs_lib
@pytest.mark.parametrize("fs", [8000.0, 22050.0, 48000.0])
def test_tone_id_c_equals_the_twin_in_uneven_blocks(fs):
    rng = np.random.default_rng(50)
    n = int(2.0 * fs)
    x = np.concatenate([radio_voice(rng, n, fs) + ctcss_tone(n, 127.3, fs),
                        radio_voice(rng, n, fs) + dcs_wave(n, 0o023, fs),
                        dtmf_seq("15#", fs), tone_seq((1060, 1160, 1270), 0.07, fs)])
    x = (x + 0.01 * rng.standard_normal(x.size)).astype(np.float32)
    want = _run_tone(ref.ToneId(0.0, fs=fs), x)
    assert [e["kind"] for e in want].count(atkdsp.TONE_DTMF) == 6
    for blocks in ([1] * 3 + [8192], [7], [239, 240, 241], [999], [x.size]):
        got = _run_tone(atkdsp.ToneId(0.0, fs=fs), x, blocks)
        assert len(got) == len(want), blocks
        for g, w in zip(got, want):
            for k in ("kind", "type", "code", "alias", "flags", "text"):
                assert g[k] == w[k], (blocks, k, g, w)
            assert g["t0"] == pytest.approx(w["t0"], abs=1e-9)
            if g["t1"] == g["t1"]:
                assert g["t1"] == pytest.approx(w["t1"], abs=1e-9)
            else:
                assert w["t1"] != w["t1"]
            assert g["freq_hz"] == pytest.approx(w["freq_hz"], abs=1e-6)
            assert g["level_db"] == pytest.approx(w["level_db"], abs=1e-3)


@needs_lib
def test_tone_id_cap_describe_stats_and_refusals():
    x = dtmf_seq("123456789", on=0.05, off=0.05)
    h = atkdsp.ToneId(0.0, fs=FS_RADIO, enable=atkdsp.TONE_EN_DTMF)
    got = h.process(x, cap=3)
    assert len(got) == 3 and h.stats()["dropped"] == 15 and h.stats()["events"] == 18
    d = h.describe()
    assert d.startswith("tone_id: fs 8000 Hz; sub-audio /8 to 1000 Hz") and "enabled DTMF" in d
    assert ref.ToneId(0.0, fs=FS_RADIO, enable=atkdsp.TONE_EN_DTMF).describe().split(";")[:-1] == d.split(";")[:-1]
    with pytest.raises(atkdsp.AtkDspError, match="fs"):
        atkdsp.ToneId(0.0, fs=1000.0)
    with pytest.raises(ValueError, match="dcs_max_errors"):
        ref.ToneId(0.0, fs=FS_RADIO, dcs_max_errors=4)
    assert atkdsp.ctcss_tone if hasattr(atkdsp, "ctcss_tone") else True
