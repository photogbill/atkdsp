"""Every C kernel against its numpy twin, on the same data, block by block.

The twins in atkdsp.reference are the specification; a C kernel that
disagrees with its twin is wrong (or the twin is, and then the test says
exactly where). Streaming kernels are fed in UNEVEN blocks so a boundary
that leaks would show.

Run from the repo root:  python -m pytest tests -q
Needs bin/atkdsp.(dll|so) — see build.bat / build.sh.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
import atkdsp                                  # noqa: E402
from atkdsp import reference as ref            # noqa: E402

pytestmark = pytest.mark.skipif(not atkdsp.available(),
                                reason="atkdsp library not built (run build.sh / build.bat)")

rng = np.random.default_rng(20260917)


def noise(n, scale=0.3):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64) * scale


def blocks(x, sizes):
    i = 0
    for s in sizes:
        yield x[i:i + s]
        i += s
    if i < x.size:
        yield x[i:]


def lowpass(cutoff, fs, ntaps):
    """Windowed-sinc, same as atk.core.dsp.design_lowpass."""
    fc = max(1e-4, min(0.499, cutoff / fs))
    n = np.arange(ntaps) - (ntaps - 1) / 2.0
    h = np.sinc(2 * fc * n) * np.hamming(ntaps)
    return (h / np.sum(h)).astype(np.float32)


# ---- identity ---------------------------------------------------------------
def test_abi_and_version():
    assert atkdsp.load().atkdsp_abi_version() == atkdsp.ABI_VERSION
    assert atkdsp.version() == "0.4.0"
    assert "abi" in atkdsp.build_info()


def test_the_build_info_reports_the_abi_it_was_built_with():
    """It used to carry the number as a typed-in literal — `" abi" " 1"` —
    which would have kept saying 1 after the bump to 2, on the one line
    anybody reads to check exactly that."""
    assert f"abi {atkdsp.ABI_VERSION}" in atkdsp.build_info()


def test_every_bound_symbol_exists_in_this_library():
    """The reason adding a function now bumps the ABI (atkdsp.h rule 5).

    The binding binds eagerly, so a binding that knows a newer function and a
    library that does not used to fail with an AttributeError about a missing
    symbol instead of the version sentence. `load()` succeeding at all is that
    check; this names it so the next person does not re-earn it.
    """
    lib = atkdsp.load()
    for name in ("atkdsp_unpack", "atkdsp_unpack_dc", "atkdsp_nco_mix",
                 "atkdsp_fir_process", "atkdsp_ddc_process",
                 "atkdsp_spectrum_reduce", "atkdsp_design_lowpass"):
        assert getattr(lib, name, None) is not None, f"{name} is not in the library"


def test_bytes_per_sample_agree():
    for f in ("cu8", "ci8", "ci16", "ci16q11", "cf32"):
        assert atkdsp.bytes_per_sample(f) == ref.bytes_per_sample(f)


# ---- 1. unpack ----------------------------------------------------------------
@pytest.mark.parametrize("fmt,dtype", [("cu8", np.uint8), ("ci8", np.int8),
                                       ("ci16", "<i2"), ("ci16q11", "<i2"), ("cf32", "<f4")])
def test_unpack_matches(fmt, dtype):
    if fmt == "cu8":
        raw = rng.integers(0, 256, 4000, dtype=np.uint8).tobytes()
    elif fmt == "ci8":
        raw = rng.integers(-128, 128, 4000, dtype=np.int8).tobytes()
    elif fmt == "cf32":
        raw = rng.standard_normal(4000).astype("<f4").tobytes()
    else:
        raw = rng.integers(-2048, 2048, 4000, dtype="<i2").tobytes()
    raw = raw[:-1]                                   # torn tail
    a = atkdsp.unpack(raw, fmt)
    b = ref.unpack(raw, fmt)
    assert a.size == b.size == (len(raw) // ref.bytes_per_sample(fmt))
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-7)


def test_unpack_dc_block_is_continuous_across_blocks():
    raw = (rng.integers(-2048, 2048, 20000, dtype="<i2") + 300).astype("<i2").tobytes()
    sa = np.zeros(1, np.complex64); sb = np.zeros(1, np.complex64)
    outs_a, outs_b = [], []
    for chunk in (raw[:4000], raw[4000:4004], raw[4004:]):
        outs_a.append(atkdsp.unpack(chunk, "ci16q11", 0.01, sa))
        outs_b.append(ref.unpack(chunk, "ci16q11", 0.01, sb))
    a, b = np.concatenate(outs_a), np.concatenate(outs_b)
    np.testing.assert_allclose(a, b, rtol=0, atol=2e-6)
    assert abs(a[-500:].mean()) < 0.01, "the DC offset is gone"


# ---- 2. NCO -------------------------------------------------------------------
def test_nco_matches_and_is_phase_continuous():
    x = noise(30000, 1.0)
    fs, f0 = 2.4e6, 173_125.0
    a = atkdsp.Nco(f0, fs); b = ref.Nco(f0, fs)
    ya = np.concatenate([a.mix(blk) for blk in blocks(x, [1, 999, 4096, 7, 13000])])
    yb = np.concatenate([b.mix(blk) for blk in blocks(x, [1, 999, 4096, 7, 13000])])
    np.testing.assert_allclose(ya, yb, rtol=0, atol=1e-5)
    # and the whole thing equals one call — no boundary artefact
    y1 = ref.Nco(f0, fs).mix(x)
    np.testing.assert_allclose(ya, y1, rtol=0, atol=1e-5)


def test_nco_zero_frequency_is_identity():
    x = noise(100)
    assert np.array_equal(atkdsp.Nco(0.0, 1e6).mix(x), x)


# ---- 3. FIR -------------------------------------------------------------------
@pytest.mark.parametrize("decim", [1, 3, 8, 50])
def test_fir_matches_across_uneven_blocks(decim):
    x = noise(12345, 1.0)
    h = lowpass(10e3, 2.4e6, 65)
    a = atkdsp.Fir(h, decim); b = ref.Fir(h, decim)
    sizes = [5, 64, 1, 4000, 3, 8000]
    ya = np.concatenate([a.process(blk) for blk in blocks(x, sizes)])
    yb = np.concatenate([b.process(blk) for blk in blocks(x, sizes)])
    assert ya.size == yb.size == (x.size + decim - 1) // decim
    np.testing.assert_allclose(ya, yb, rtol=0, atol=1e-5)
    # equals the unblocked reference
    y1 = ref.Fir(h, decim).process(x)
    np.testing.assert_allclose(ya, y1, rtol=0, atol=1e-5)


def test_fir_out_max_is_exact():
    f = atkdsp.Fir(np.ones(4, np.float32) / 4, 7)
    total = 0
    for n in (1, 6, 7, 8, 100, 0, 3):
        cap = f.out_max(n)
        got = f.process(noise(n)).size if n else 0
        assert got == cap
        total += got
    assert total == (125 + 6) // 7


def test_fir_reset_forgets():
    h = lowpass(10e3, 2.4e6, 33)
    f = atkdsp.Fir(h, 2)
    x = noise(500, 1.0)
    y0 = f.process(x); f.reset(); y1 = f.process(x)
    np.testing.assert_array_equal(y0, y1)


# ---- 4. resampler -------------------------------------------------------------
@pytest.mark.parametrize("up,down", [(1, 1), (3, 2), (2, 3), (160, 147), (48000, 48077)])
def test_resampler_matches_and_counts_exactly(up, down):
    x = noise(6000, 1.0)
    ntaps = 8 * up if up < 100 else 4 * up
    h = lowpass(0.45 / max(up, down), 1.0, ntaps)     # normalised prototype at rate `up`
    a = atkdsp.Resampler(up, down, h); b = ref.Resampler(up, down, h)
    sizes = [1, 17, 2000, 3, 999]
    ya = np.concatenate([a.process(blk) for blk in blocks(x, sizes)])
    yb = np.concatenate([b.process(blk) for blk in blocks(x, sizes)])
    assert ya.size == yb.size
    expect = -(-x.size * up // down)                  # ceil(N*up/down)
    assert ya.size == expect
    np.testing.assert_allclose(ya, yb, rtol=0, atol=2e-5)


def test_resampler_48077_to_48000_over_ten_seconds():
    """The bladeRF-at-10-MSPS case: fs/208 = 48076.9 Hz into dsd-neo's 48 000."""
    up, down = 48000, 48077
    h = lowpass(0.45 / down, 1.0, 4 * up)
    r = atkdsp.Resampler(up, down, h)
    total = 0
    for _ in range(10):
        total += r.process(noise(48077)).size
    assert abs(total - 480_000) <= 1


# ---- 5. FFT -------------------------------------------------------------------
@pytest.mark.parametrize("n", [16, 1000, 1024, 4096, 65536])
def test_fft_matches_numpy(n):
    x = noise(n, 1.0)
    p = atkdsp.Fft(n)
    np.testing.assert_allclose(p.exec(x), np.fft.fft(x), rtol=1e-5, atol=1e-4 * np.sqrt(n))
    np.testing.assert_allclose(p.exec(x, inverse=True), np.fft.ifft(x), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kind", ["rectangular", "hann", "hamming", "blackmanharris"])
def test_window_matches(kind):
    np.testing.assert_allclose(atkdsp.window(kind, 2048), ref.window(kind, 2048), rtol=0, atol=1e-6)


def test_window_matches_numpy_hanning():
    np.testing.assert_allclose(atkdsp.window("hann", 4096), np.hanning(4096), rtol=0, atol=1e-6)


def test_power_db_matches_spectrum_db():
    x = noise(4096, 1.0) + 0.5 * np.exp(2j * np.pi * 300 / 4096 * np.arange(4096))
    w = atkdsp.window("hann", 4096)
    a = atkdsp.Fft(4096).power_db(x, w)
    b = ref.power_db(x, w)
    np.testing.assert_allclose(a, b, rtol=0, atol=2e-3)   # dB


@pytest.mark.parametrize("det", ["max", "min", "avg"])
@pytest.mark.parametrize("hop", [None, 512])
def test_spectrum_reduce_matches(det, hop):
    x = noise(40 * 1024 + 37, 1.0)
    x[5000:5300] += 3.0 * np.exp(2j * np.pi * 0.2 * np.arange(300))   # a burst
    w = atkdsp.window("hann", 1024)
    la, fa = atkdsp.Fft(1024).spectrum_reduce(x, hop, w, det)
    lb, fb = ref.spectrum_reduce(x, 1024, hop, w, det)
    assert fa == fb
    np.testing.assert_allclose(la, lb, rtol=0, atol=3e-3)


def test_spectrum_reduce_max_sees_a_burst_shorter_than_a_frame():
    """100 % POI: the burst is in the max line and NOT in a single trailing frame."""
    n = 2048
    x = noise(64 * n, 0.1)
    x[10 * n + 100: 10 * n + 400] += 5.0 * np.exp(2j * np.pi * 0.25 * np.arange(300))
    p = atkdsp.Fft(n)
    line, frames = p.spectrum_reduce(x, None, None, "max")
    assert frames == 64
    burst_bin = n // 2 + int(0.25 * n)
    assert line[burst_bin] > line.mean() + 30
    last = p.power_db(x[-n:])
    assert last[burst_bin] < last.mean() + 15


def test_spectrum_reduce_short_input_gives_zero_frames():
    line, frames = atkdsp.Fft(1024).spectrum_reduce(noise(100))
    assert frames == 0


def test_threads_can_be_set():
    n = atkdsp.set_threads(2)
    assert n >= 1
    atkdsp.set_threads(0)


# ---- 6. demod -----------------------------------------------------------------
def test_fm_demod_matches_and_carries_prev():
    x = noise(10000, 1.0)
    pa = np.zeros(1, np.complex64); pb = np.zeros(1, np.complex64)
    sizes = [1, 3, 4000, 5996]
    a = np.concatenate([atkdsp.fm_demod(blk, pa, 0.7) for blk in blocks(x, sizes)])
    b = np.concatenate([ref.fm_demod(blk, pb, 0.7) for blk in blocks(x, sizes)])
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-5)
    assert a[0] == 0.0, "first sample against a zero prev is 0"
    assert pa[0] == x[-1]


def test_am_demod_matches():
    x = noise(5000, 1.0) + 2.0
    sa = np.zeros(1, np.float32); sb = np.zeros(1, np.float32)
    a = np.concatenate([atkdsp.am_demod(blk, sa, 0.02) for blk in blocks(x, [7, 3000])])
    b = np.concatenate([ref.am_demod(blk, sb, 0.02) for blk in blocks(x, [7, 3000])])
    np.testing.assert_allclose(a, b, rtol=0, atol=2e-5)
    np.testing.assert_allclose(atkdsp.am_demod(x), ref.am_demod(x), rtol=0, atol=1e-6)


# ---- 7. display ---------------------------------------------------------------
def test_db_to_pixels_matches():
    db = (rng.standard_normal(5000) * 30 - 70).astype(np.float32)
    db[::97] = np.nan
    lut = rng.integers(0, 2**32, 256, dtype=np.uint32)
    np.testing.assert_array_equal(atkdsp.db_to_pixels(db, -110, -30, lut),
                                  ref.db_to_pixels(db, -110, -30, lut))


@pytest.mark.parametrize("n,m", [(8192, 4096), (8192, 1000), (131072, 4096), (100, 100), (10, 16)])
def test_decimate_max_matches(n, m):
    x = rng.standard_normal(n).astype(np.float32)
    np.testing.assert_array_equal(atkdsp.decimate_max(x, m), ref.decimate_max(x, m))


# ---- 8. detection -------------------------------------------------------------
@pytest.mark.parametrize("n", [5, 6, 1023, 1024, 131072])
def test_median_matches_numpy(n):
    x = rng.standard_normal(n).astype(np.float32)
    assert atkdsp.median(x) == pytest.approx(ref.median(x), abs=1e-6)


def test_detect_channels_matches_on_a_busy_band():
    n = 16384
    floor = -100.0
    # exponential-in-power noise, as atk-dsd-decoding-noise insists
    p = rng.exponential(1.0, n)
    line = (floor + 10 * np.log10(p + 1e-12)).astype(np.float32)
    for c, w, lvl in [(2000, 12, -60), (2030, 6, -70), (9000, 80, -50), (15000, 3, -65)]:
        line[c:c + w] = lvl + rng.standard_normal(w).astype(np.float32)
    fl = atkdsp.median(ref.smooth5(line))
    a = atkdsp.detect_channels(line, fl, 6.0, 4, 2, max_channels=4096)
    b = ref.detect_channels(line, fl, 6.0, 4, 2)
    assert len(a) == len(b) and len(a) >= 3
    for ca, cb in zip(a, b):
        assert (ca.start_bin, ca.end_bin) == (cb.start_bin, cb.end_bin)
        assert ca.centroid_bin == pytest.approx(cb.centroid_bin, abs=1e-2)
        assert ca.peak_db == pytest.approx(cb.peak_db, abs=1e-4)
        assert ca.snr_db == pytest.approx(cb.snr_db, abs=1e-4)


def test_detect_channels_negative_control_on_noise():
    """Stays quiet on an empty band at a 12 dB threshold — the entire problem."""
    n = 8192
    p = rng.exponential(1.0, n)
    line = (-100 + 10 * np.log10(p)).astype(np.float32)
    fl = atkdsp.median(ref.smooth5(line))
    assert atkdsp.detect_channels(line, fl, 12.0, 4, 3) == []


def test_detect_channels_refuses_nan_floor_and_small_capacity():
    line = np.full(100, -100.0, np.float32); line[40:60] = -40
    with pytest.raises(atkdsp.AtkDspError):
        atkdsp.detect_channels(line, float("nan"), 6, 2, 2)
    with pytest.raises(atkdsp.AtkDspError):
        atkdsp.detect_channels(line, -100.0, 6, 2, 2, max_channels=0)


def test_stitch_max_matches_sweep_stitch():
    out_a = np.full(4096, np.nan, np.float32); out_b = out_a.copy()
    for k in range(6):
        seg = (rng.standard_normal(1024) * 10 - 80).astype(np.float32)
        c = 100e6 + k * 1.8e6
        sr = 2.4e6
        args = (c - sr / 2, sr / 1024, c - 0.45 * sr, c + 0.45 * sr, 99e6, 12e6 / 4096)
        atkdsp.stitch_max(seg, *args, out_a)
        ref.stitch_max(seg, *args, out_b)
    np.testing.assert_array_equal(out_a, out_b)
    assert np.isnan(out_a).sum() > 0 and np.isfinite(out_a).sum() > 2000


# ---- 9. filter design ---------------------------------------------------------
@pytest.mark.parametrize("fp,fs_,att,rate", [(7500, 15000, 60, 48000), (7500, 3_000_000, 60, 10e6),
                                             (100, 200, 40, 1000), (0, 20000, 80, 48000)])
def test_design_lowpass_matches_twin_to_float_precision(fp, fs_, att, rate):
    a = atkdsp.design_lowpass(fp, fs_, att, rate)
    b = ref.design_lowpass(fp, fs_, att, rate)
    assert a.size == b.size and a.size % 2 == 1
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-7)
    assert abs(float(a.sum()) - 1.0) < 1e-5, "unity DC gain"


def test_design_lowpass_meets_its_stopband():
    h = atkdsp.design_lowpass(7500, 15000, 60, 48000)
    H = 20 * np.log10(np.abs(np.fft.rfft(h, 65536)) + 1e-12)
    f = np.fft.rfftfreq(65536, 1 / 48000)
    assert H[f >= 15000].max() < -58, "stopband is at least ~60 dB down"
    assert H[f <= 7500].min() > -0.5, "passband is flat"


def test_design_lowpass_refuses_bad_edges():
    with pytest.raises(atkdsp.AtkDspError):
        atkdsp.design_lowpass(100, 50, 60, 1000)


# ---- 10. DDC ------------------------------------------------------------------
@pytest.mark.parametrize("fs", [2_400_000, 10_000_000, 40_000_000])
def test_ddc_matches_twin_across_uneven_blocks(fs):
    x = noise(int(fs * 0.02), 1.0)
    a = atkdsp.Ddc(fs, 123_456.0, 15_000, 48_000, 60, max_block=x.size)
    b = ref.Ddc(fs, 123_456.0, 15_000, 48_000, 60)
    assert a.plan() == b.plan()
    assert a.out_rate == b.out_rate == 48_000
    sizes = [1, 7777, 3, x.size // 3]
    ya = np.concatenate([a.process(blk) for blk in blocks(x, sizes)])
    yb = np.concatenate([b.process(blk) for blk in blocks(x, sizes)])
    assert ya.size == yb.size
    np.testing.assert_allclose(ya, yb, rtol=0, atol=3e-5)


@pytest.mark.parametrize("fs,worst_db", [(2_400_000, -60), (10_000_000, -60), (40_000_000, -60)])
def test_ddc_rejects_an_interferer_100khz_away(fs, worst_db):
    """The §2 defect: today's channelize passes a tone 100 kHz off at -2.9 dB
    (10 MSPS) / -0.2 dB (40 MSPS). This must be 60 dB down."""
    n = int(fs * 0.05)
    t = np.arange(n)
    d = atkdsp.Ddc(fs, 0.0, 15_000, 48_000, 60, max_block=n)
    ref_level = None
    for off in (0.0, 100_000.0, 250_000.0):
        tone = np.exp(2j * np.pi * off / fs * t).astype(np.complex64)
        d.reset()
        y = d.process(tone)
        tail = y[y.size // 2:]
        lvl = 10 * np.log10(np.mean(np.abs(tail) ** 2) + 1e-30)
        if off == 0.0:
            ref_level = lvl
            assert abs(lvl) < 0.5
        else:
            assert lvl - ref_level < worst_db, f"{off} Hz off is only {lvl - ref_level:.1f} dB down"


def test_ddc_output_rate_is_exact_over_ten_seconds():
    fs = 10_000_000
    d = atkdsp.Ddc(fs, 0.0, 15_000, 48_000, 60, max_block=fs // 10)
    total = 0
    for _ in range(100):                       # 100 x 0.1 s
        total += d.process(noise(fs // 10)).size
    assert abs(total - 480_000) <= 2
    assert "48000" in d.describe() and "x624/625" in d.describe()


def test_ddc_set_offset_is_continuous():
    fs = 2_400_000
    n = 48_000
    t = np.arange(4 * n)
    tone = np.exp(2j * np.pi * 300_000 / fs * t).astype(np.complex64)
    d = atkdsp.Ddc(fs, 300_000.0, 15_000, 48_000, 60, max_block=n)
    d.process(tone[:n]); y1 = d.process(tone[n:2 * n])
    d.set_offset(300_000.0)                   # same offset again: nothing changes
    y2 = d.process(tone[2 * n:3 * n])
    assert np.abs(np.abs(y2) - np.abs(y1).mean()).max() < 0.05
    assert abs(np.angle(y2[-1] * np.conj(y2[-2]))) < 1e-3, "the mixed tone sits at DC, no phase step"


def test_ddc_refuses_what_it_cannot_do():
    with pytest.raises(atkdsp.AtkDspError):
        atkdsp.Ddc(2_400_000, 0.0, 15_000, 4_800_000, 60)      # out_rate > fs
    d = atkdsp.Ddc(2_400_000, 0.0, 15_000, 48_000, 60, max_block=1000)
    with pytest.raises(atkdsp.AtkDspError):
        d.process(noise(2000))                                   # block > max_block


# ---------------------------------------------------------------------------
# 1b. unpack_dc — the offset folded into the scale, the mean handed back
# ---------------------------------------------------------------------------

def _raw(fmt, n=4096, seed=77):
    r = np.random.default_rng(seed)
    if fmt == "cu8":
        return r.integers(0, 256, 2 * n, dtype=np.uint8).tobytes()
    if fmt == "ci8":
        return r.integers(-128, 128, 2 * n, dtype=np.int8).tobytes()
    if fmt == "cf32":
        return r.standard_normal(2 * n).astype("<f4").tobytes()
    return r.integers(-2000, 2000, 2 * n, dtype=np.int64).astype("<i2").tobytes()


@pytest.mark.parametrize("fmt", ["cu8", "ci8", "ci16", "ci16q11", "cf32"])
@pytest.mark.parametrize("offset", [0j, 0.123 - 0.045j])
def test_unpack_dc_matches_its_twin(fmt, offset):
    raw = _raw(fmt)
    a, ma = atkdsp.unpack_dc(raw, fmt, offset)
    b, mb = ref.unpack_dc(raw, fmt, offset)
    assert a.dtype == np.complex64 and a.size == b.size
    assert np.max(np.abs(a - b)) < 1e-6, f"{fmt}: samples differ"
    assert abs(ma - mb) < 1e-6, f"{fmt}: mean differs"


@pytest.mark.parametrize("fmt", ["cu8", "ci16q11", "cf32"])
def test_unpack_dc_is_unpack_when_the_offset_is_zero(fmt):
    """The new path must not quietly become a second, different converter."""
    raw = _raw(fmt, seed=5)
    a, _ = atkdsp.unpack_dc(raw, fmt, 0j)
    assert np.array_equal(a, atkdsp.unpack(raw, fmt))


def test_unpack_dc_returns_the_mean_from_before_the_subtraction():
    """That is what makes a running estimate possible without a third pass:
    the caller needs to know where the offset IS, not where it is after the
    last guess was removed."""
    n = 4096
    off = 0.25 - 0.1j
    x = (np.full(n, 0.4 + 0.2j) + 0.01 * noise(n)).astype(np.complex64)
    raw = x.tobytes()
    y, mean = atkdsp.unpack_dc(raw, "cf32", off)
    assert abs(mean - complex(np.mean(x, dtype=np.complex128))) < 1e-5
    assert abs(complex(np.mean(y, dtype=np.complex128)) - (mean - off)) < 1e-5


def test_the_offset_costs_nothing_the_conversion_was_not_already_doing():
    """The claim in atkdsp.h §1b. Not a timing assertion — those are flaky —
    but the arithmetic one underneath it: folding the offset into the
    constant gives the same answer as subtracting afterwards."""
    raw = _raw("ci16q11", seed=9)
    off = -0.031 + 0.017j
    folded, _ = atkdsp.unpack_dc(raw, "ci16q11", off)
    after = atkdsp.unpack(raw, "ci16q11") - np.complex64(off)
    assert np.max(np.abs(folded - after)) < 1e-6


# ---------------------------------------------------------------------------
# 11. LTE cell search
# ---------------------------------------------------------------------------

LTE_SLOT = 960
LTE_SYM5 = 10 + 128 + 4 * (9 + 128) + 9      # SSS data start within a slot
LTE_SYM6 = 10 + 128 + 5 * (9 + 128) + 9      # PSS data start


def lte_frame(nid1, nid2, snr_db=10.0, frames=2, cfo_hz=0.0, seed=0):
    """A synthetic FDD downlink: PSS and SSS in slots 0 and 10, with cyclic
    prefixes, noise at a stated SNR, and an optional carrier offset."""
    r = np.random.default_rng(abs(int(seed)) + 1)
    n = frames * 19200
    x = np.zeros(n, dtype=np.complex128)
    p = np.asarray(ref.lte_pss_symbol(nid2), dtype=np.complex128)
    for f in range(frames):
        for slot, sf in ((0, 0), (10, 5)):
            base = f * 19200 + slot * LTE_SLOT
            carr = ref.lte_sss_symbol(nid1, nid2, sf).astype(np.complex128)
            s = np.asarray(ref._lte_to_symbol(carr), dtype=np.complex128)
            for start, sym in ((base + LTE_SYM5, s), (base + LTE_SYM6, p)):
                x[start:start + 128] += sym
                x[start - 9:start] += sym[-9:]
    power = float(np.mean(np.abs(x[x != 0]) ** 2))
    noise = (r.standard_normal(n) + 1j * r.standard_normal(n)) / np.sqrt(2)
    x = x + noise * np.sqrt(power / (10 ** (snr_db / 10.0)))
    if cfo_hz:
        x = x * np.exp(2j * np.pi * cfo_hz * np.arange(n) / ref.LTE_RATE)
    return x.astype(np.complex64)


@pytest.mark.parametrize("nid2", [0, 1, 2])
def test_lte_pss_symbol_matches_its_twin(nid2):
    assert np.max(np.abs(atkdsp.lte_pss_symbol(nid2)
                         - ref.lte_pss_symbol(nid2))) < 1e-5


@pytest.mark.parametrize("nid1", [0, 1, 57, 100, 167])
@pytest.mark.parametrize("nid2", [0, 1, 2])
@pytest.mark.parametrize("sf", [0, 5])
def test_lte_sss_symbol_matches_its_twin(nid1, nid2, sf):
    a = atkdsp.lte_sss_symbol(nid1, nid2, sf)
    assert np.array_equal(a, ref.lte_sss_symbol(nid1, nid2, sf))
    assert set(np.unique(a)) <= {-1.0, 1.0}


def test_the_sss_depends_on_the_pss_identity():
    """c0 and c1 are cyclic shifts BY N_ID^(2), so one table cannot serve all
    three. Built the wrong way it decodes a confident, completely incorrect
    PCI rather than failing — which is the worst possible answer for a
    fake-tower check."""
    a = atkdsp.lte_sss_symbol(57, 0, 0)
    b = atkdsp.lte_sss_symbol(57, 1, 0)
    assert not np.array_equal(a, b)


def test_every_lte_cell_identity_is_distinct():
    seen = {}
    for nid1 in range(168):
        for sf in (0, 5):
            key = atkdsp.lte_sss_symbol(nid1, 1, sf).tobytes()
            assert key not in seen, f"collision {seen.get(key)} vs {(nid1, sf)}"
            seen[key] = (nid1, sf)
    assert len(seen) == 336


@pytest.mark.parametrize("cell", [(57, 1), (100, 2), (0, 0), (167, 2)])
@pytest.mark.parametrize("snr", [15.0, 5.0, 0.0])
def test_lte_detect_finds_the_right_cell(cell, snr):
    nid1, nid2 = cell
    got = atkdsp.Lte().detect(lte_frame(nid1, nid2, snr, seed=nid1 + int(snr)),
                              min_metric=0.15)
    assert got, f"PCI {3 * nid1 + nid2} at {snr} dB was missed"
    c = got[0]
    assert c["nid2"] == nid2
    assert c["nid1"] == nid1
    assert c["pci"] == 3 * nid1 + nid2
    assert c["subframe"] in (0, 5)
    assert 0.0 < c["metric"] <= 1.05


def test_lte_detect_agrees_with_its_twin():
    for snr in (15.0, 5.0, 0.0, -3.0):
        x = lte_frame(57, 1, snr, seed=int(snr) + 40)
        a = atkdsp.Lte().detect(x, min_metric=0.06)
        b = ref.lte_detect(x, min_metric=0.06)
        assert len(a) == len(b), f"{snr} dB: C found {len(a)}, twin {len(b)}"
        if not a:
            continue
        assert a[0]["nid2"] == b[0]["nid2"]
        assert a[0]["pci"] == b[0]["pci"]
        assert a[0]["offset"] == b[0]["offset"]
        assert a[0]["metric"] == pytest.approx(b[0]["metric"], abs=2e-3)
        assert a[0]["cfo_hz"] == pytest.approx(b[0]["cfo_hz"], abs=5.0)


def test_lte_detect_estimates_the_carrier_offset():
    for cfo in (-2000.0, -500.0, 0.0, 500.0, 2000.0):
        got = atkdsp.Lte().detect(lte_frame(100, 2, 15.0, cfo_hz=cfo, seed=7),
                                  min_metric=0.15)
        assert got, f"{cfo} Hz off frequency and the cell was lost"
        assert got[0]["cfo_hz"] == pytest.approx(cfo, abs=250.0)
        assert got[0]["pci"] == 302, "the offset broke the identification"


def test_lte_detect_does_not_invent_cells():
    """The 5 ms repeat is what makes this safe to trigger on. Without it a
    single strong correlation — an impulse, a coincidence — would report a
    cell that is not there."""
    r = np.random.default_rng(11)
    lte = atkdsp.Lte()
    false = 0
    for _ in range(20):
        z = (r.standard_normal(38400) + 1j * r.standard_normal(38400))
        false += len(lte.detect(z.astype(np.complex64), min_metric=0.15))
    assert false == 0, f"{false} cells found in pure noise"
    t = np.arange(38400) / ref.LTE_RATE
    assert not lte.detect(np.exp(2j * np.pi * 3e5 * t).astype(np.complex64),
                          min_metric=0.15), "a plain carrier reads as a cell"


def test_lte_detect_needs_two_periods_before_it_will_say_anything():
    short = lte_frame(57, 1, 20.0, seed=2)[:2 * 9600 - 1]
    assert atkdsp.Lte().detect(short, min_metric=0.15) == []


def test_lte_detect_survives_odd_buffer_lengths():
    x = lte_frame(57, 1, 15.0, seed=5)
    lte = atkdsp.Lte()
    for n in (2 * 9600, 2 * 9600 + 1, 30000, 38400, 38401):
        got = lte.detect(x[:n], min_metric=0.15)
        for c in got:
            assert 0 <= c["offset"] <= n - 128


# ---- 11b. LTE PBCH -> MIB ------------------------------------------------
def _flat(n_ports, seed):
    r = np.random.default_rng(seed)
    return [(r.standard_normal() + 1j * r.standard_normal()) * np.ones(72)
            for _ in range(n_ports)]


def _pbch_block(nid, n_ports, sfn, dl_bw=50, snr_db=100.0, seed=0):
    mib = ref.lte_mib_pack(dl_bw, sfn % 2, sfn % 4, sfn)
    H = _flat(n_ports, seed + nid + sfn)
    return ref.lte_build_pbch_samples(mib, nid, n_ports, sfn, H, snr_db, seed)


@pytest.mark.parametrize("c_init", [0, 1, 123, 65535, 500000])
@pytest.mark.parametrize("length", [64, 480, 1920])
def test_lte_gold_matches_its_twin(c_init, length):
    assert np.array_equal(atkdsp.lte_gold(c_init, length),
                          ref.lte_gold(c_init, length))


def test_lte_pbch_re_map_has_240_data_res():
    for nid in (0, 1, 123, 300, 503):
        assert len(ref.lte_pbch_re_map(nid)) == 240


@pytest.mark.parametrize("n_ports", [1, 2, 4])
@pytest.mark.parametrize("sfn", [0, 1, 2, 3, 132, 1023])
@pytest.mark.parametrize("nid", [0, 123, 503])
def test_lte_mib_decode_matches_its_twin(n_ports, sfn, nid):
    """C and twin must agree bit-for-bit on the decoded MIB (noiseless)."""
    blk = _pbch_block(nid, n_ports, sfn, seed=nid + sfn)
    c = atkdsp.Lte().mib_decode(blk, nid)
    t = ref.lte_mib_decode(blk, nid)
    assert c == t
    assert c is not None
    assert c["sfn"] == sfn and c["n_ports"] == n_ports and c["dl_bw_rb"] == 50


@pytest.mark.parametrize("n_ports", [1, 2, 4])
def test_lte_mib_decode_recovers_the_mib_at_operating_snr(n_ports):
    """At 3 dB every frame should decode; the survey works far above this."""
    for sfn in (7, 44, 260, 900):
        bw = [6, 15, 25, 50, 75, 100][sfn % 6]
        mib = ref.lte_mib_pack(bw, sfn % 2, sfn % 4, sfn)
        H = _flat(n_ports, sfn)
        sig = ref.lte_build_pbch_samples(mib, 123, n_ports, sfn, H,
                                         snr_db=3.0, seed=sfn)
        d = atkdsp.Lte().mib_decode(sig, 123)
        assert d is not None and d["sfn"] == sfn and d["n_ports"] == n_ports
        assert d["dl_bw_rb"] == bw and d["phich_res"] == sfn % 4


def test_lte_mib_decode_is_silent_on_noise():
    """The spare-bits gate keeps chance CRC passes off pure noise."""
    rng = np.random.default_rng(11)
    fa = 0
    for _ in range(400):
        sig = ((rng.standard_normal(549) + 1j * rng.standard_normal(549))
               / np.sqrt(2)).astype(np.complex64)
        if atkdsp.Lte().mib_decode(sig, int(rng.integers(0, 504))) is not None:
            fa += 1
    assert fa == 0, f"{fa} false MIB decodes on noise"


def test_lte_mib_combine_beats_a_single_frame():
    """Four soft-combined frames decode a cell a single frame cannot."""
    nid, n_ports, sfn0 = 123, 2, 400          # 40 ms-aligned (sfn0 % 4 == 0)
    got_single = got_combined = 0
    for s in range(20):
        H = _flat(n_ports, 900 + s)
        blocks = [ref.lte_build_pbch_samples(
                      ref.lte_mib_pack(50, 0, 0, sfn0 + f), nid, n_ports,
                      sfn0 + f, H, snr_db=-9.0, seed=s * 4 + f)
                  for f in range(4)]
        if atkdsp.Lte().mib_decode(blocks[0], nid):
            got_single += 1
        d = atkdsp.Lte().mib_decode_frames(blocks, nid)
        if d and d["sfn"] == sfn0:
            got_combined += 1
    assert got_combined > got_single


def test_lte_mib_combine_matches_its_twin():
    nid, n_ports, sfn0 = 57, 2, 100
    H = _flat(n_ports, 3)
    blocks = [ref.lte_build_pbch_samples(
                  ref.lte_mib_pack(25, 1, 2, sfn0 + f), nid, n_ports,
                  sfn0 + f, H, snr_db=-7.0, seed=10 + f) for f in range(4)]
    assert atkdsp.Lte().mib_decode_frames(blocks, nid) == \
        ref.lte_mib_decode_frames(blocks, nid)


def test_lte_mib_end_to_end_from_the_detector_offset():
    """Full frame: detect the PSS, then decode the MIB at the published
    offset (LTE_PBCH_OFFSET after the PSS). Proves the offset constant."""
    nid1, nid2, sfn = 57, 1, 132
    nid = 3 * nid1 + nid2
    x = np.asarray(lte_frame(nid1, nid2, snr_db=40.0, frames=2, seed=5),
                   dtype=np.complex128)
    for f in range(2):                       # same SFN in both frames
        blk = np.asarray(_pbch_block(nid, 1, sfn, seed=100 + f),
                         dtype=np.complex128)
        base = f * 19200 + 960               # slot 1 of subframe 0
        x[base:base + ref.LTE_PBCH_BLOCK] += blk
    r = np.random.default_rng(77)
    power = float(np.mean(np.abs(x[x != 0]) ** 2))
    x = x + (r.standard_normal(len(x)) + 1j * r.standard_normal(len(x))) \
        / np.sqrt(2) * np.sqrt(power / 10 ** (15.0 / 10.0))
    x = x.astype(np.complex64)

    got = atkdsp.Lte().detect(x, min_metric=0.15)
    assert got and got[0]["pci"] == nid
    off = got[0]["offset"]
    if got[0]["subframe"] == 5:
        off -= ref.LTE_PSS_PERIOD
    start = off + ref.LTE_PBCH_OFFSET
    blk = x[start:start + ref.LTE_PBCH_BLOCK]
    d = atkdsp.Lte().mib_decode(blk, nid, got[0]["cfo_hz"])
    assert d is not None and d["sfn"] == sfn and d["dl_bw_rb"] == 50


# ---- 11c. Passive PRACH detector ----------------------------------------
@pytest.mark.parametrize("u", [1, 25, 129, 400, 838])
def test_prach_zc_matches_its_twin(u):
    assert np.max(np.abs(atkdsp.prach_zc(u) - ref.prach_zc(u))) < 1e-4


def _roots(hits):
    return sorted(h["root"] for h in hits)


@pytest.mark.parametrize("snr", [10.0, 5.0, 0.0, -5.0])
def test_prach_detects_a_preamble_and_agrees_with_the_twin(snr):
    pr = atkdsp.Prach()
    for s in range(12):
        u = 1 + (s * 17) % 837
        seq = ref.prach_build_capture([(u, 0)], snr_db=snr, seed=s)
        c = pr.detect(seq)
        t = ref.prach_detect(seq)
        assert any(h["root"] == u for h in c), f"root {u} missed at {snr} dB"
        assert _roots(c) == _roots(t)


def test_prach_finds_concurrent_roots():
    """Several handsets accessing at once, each on a different root, are all
    seen in one FFT."""
    pr = atkdsp.Prach()
    for s in range(10):
        rs = list(dict.fromkeys([1 + ((s * 13 + j * 211) % 837) for j in range(3)]))
        seq = ref.prach_build_capture([(u, 0) for u in rs], snr_db=10.0, seed=s)
        found = _roots(pr.detect(seq))
        assert all(u in found for u in rs), f"missed one of {rs}, got {found}"


def test_prach_counts_concurrent_accesses_on_one_root():
    """Two handsets on the same root at different cyclic shifts show as two
    peaks in the power-delay profile."""
    pr = atkdsp.Prach()
    for s in range(10):
        u = 1 + (s * 7) % 837
        shifts = [(j * 137) % 839 for j in range(2)]
        seq = ref.prach_build_capture([(u, sh) for sh in shifts],
                                      snr_db=15.0, seed=s)
        cnt = sum(h["count"] for h in pr.detect(seq) if h["root"] == u)
        assert cnt == 2, f"counted {cnt}, expected 2"


def test_prach_is_robust_to_window_misalignment():
    """A cyclic shift of one root is only a constant phase in the tone, so an
    imperfectly aligned window still identifies the root."""
    pr = atkdsp.Prach()
    for timing in (0, 1, 20, 200):
        seq = ref.prach_build_capture([(200, 0)], snr_db=10.0, seed=timing,
                                      timing=timing)
        assert any(h["root"] == 200 for h in pr.detect(seq)), f"roll {timing}"


def test_prach_is_silent_on_noise():
    pr = atkdsp.Prach()
    rng = np.random.default_rng(3)
    fa = 0
    for _ in range(2000):
        seq = ((rng.standard_normal(839) + 1j * rng.standard_normal(839))
               / np.sqrt(2)).astype(np.complex64)
        if pr.detect(seq):
            fa += 1
    assert fa == 0, f"{fa} false detections on noise"


def test_prach_agrees_with_the_twin_on_a_busy_window():
    pr = atkdsp.Prach()
    seq = ref.prach_build_capture([(25, 0), (25, 137), (400, 0), (700, 50)],
                                  snr_db=12.0, seed=1)
    c = pr.detect(seq)
    t = ref.prach_detect(seq)
    assert _roots(c) == _roots(t)
    cc = {h["root"]: h["count"] for h in c}
    tc = {h["root"]: h["count"] for h in t}
    assert cc == tc


def _embed_prach(u, shift, at, total, snr_db, seed, ncp=108):
    rng = np.random.default_rng(seed)
    seq = ref.prach_preamble_time(u, shift)
    occ = np.concatenate([seq[-ncp:], seq])
    x = np.zeros(total, dtype=complex)
    x[at:at + len(occ)] = occ
    p = np.mean(np.abs(occ) ** 2)
    x = x + np.sqrt((p / 10 ** (snr_db / 10)) / 2) * (
        rng.standard_normal(total) + 1j * rng.standard_normal(total))
    return x.astype(np.complex64)


@pytest.mark.parametrize("snr", [10.0, 0.0, -3.0])
def test_prach_scan_finds_an_embedded_preamble(snr):
    """A preamble anywhere in a longer, un-aligned capture is found by the
    sliding scan — the call a survey actually uses."""
    pr = atkdsp.Prach()
    for s in range(8):
        x = _embed_prach(200, (s * 11) % 839, (s * 53) % 2000, 3000, snr, s)
        got = [h["root"] for h in pr.scan(x, win_step=64)]
        assert 200 in got, f"missed at {snr} dB (seed {s})"


def test_prach_scan_agrees_with_its_twin():
    x = _embed_prach(129, 40, 700, 3000, 8.0, 3)
    c = sorted(h["root"] for h in atkdsp.Prach().scan(x, win_step=64))
    t = sorted(h["root"] for h in ref.prach_scan(x, win_step=64))
    assert c == t and 129 in c


def test_prach_scan_is_silent_on_noise():
    pr = atkdsp.Prach()
    rng = np.random.default_rng(2)
    fa = 0
    for _ in range(200):
        x = ((rng.standard_normal(3000) + 1j * rng.standard_normal(3000))
             / np.sqrt(2)).astype(np.complex64)
        if pr.scan(x, win_step=64):
            fa += 1
    assert fa == 0
