"""§32 analog video — known answers for the twin, and the C against the twin.

MERGE NOTE (2026-10-30, staged in D:\\Analyst_Toolkit\\file holding area):
goes in tests/test_video.py. The twin tests run anywhere (numpy only); the C
tests skip until the library carries atkdsp_video_*.

Run from the repo root:  python -m pytest tests/test_video.py -q

Every acceptance test has planted truth (the generator draws a known picture
with a frame counter in it), every detector a negative control, and every
streaming test a chunk-invariance check (docs/ATKDSP_PLAN.md §1, rules 1-3).
Thresholds stated here are the measured ones from 2026-10-30 with margin.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import atkdsp                                  # noqa: E402
from atkdsp import reference as ref            # noqa: E402

FS = 13.5e6


def _drain(v):
    out = []
    while True:
        f = v.frame()
        if f is None:
            return out
        out.append(f)


def _twin(x, **cfg):
    cfg.setdefault("fs", FS)
    v = ref.Video(**cfg)
    v.process(x)
    return v, _drain(v)


def _counters(out):
    return [ref.video_read_counter(p) for p, _m in out]


def _consecutive(ns):
    return len(ns) >= 2 and all(n is not None for n in ns) and \
        all(b == a + 1 for a, b in zip(ns, ns[1:]))


def _fm(x, cnr, fs=FS, seed=0, cfo=20e3):
    return ref.video_fm_demod(ref.video_fm(x, fs, cnr_db=cnr, seed=seed, cfo_hz=cfo), fs)


@pytest.fixture(scope="module")
def ntsc():
    return ref.video_synth(ref.VID_STD_525, fs=FS, seconds=0.25)


@pytest.fixture(scope="module")
def pal():
    return ref.video_synth(ref.VID_STD_625, fs=FS, seconds=0.25)


# ---- configuration ------------------------------------------------------------
def test_defaults_pass_their_own_check():
    assert ref._video_check(ref._vid_norm(ref.video_defaults())) == ""


@pytest.mark.parametrize("bad,words", [
    (dict(fs=3e6), "at least 4 MS/s"),
    (dict(standard=3), "custom_line_hz"),
    (dict(standard=3, custom_line_hz=15500.0, custom_lines=100), "custom_lines"),
    (dict(width=10), "width"),
    (dict(stack=65), "stack"),
    (dict(polarity=5), "polarity"),
])
def test_bad_config_is_refused_in_words(bad, words):
    with pytest.raises(ValueError, match=words):
        ref.Video(**dict(dict(fs=FS), **bad))


# ---- known answers ----------------------------------------------------------------
def _geometry(pix, truth):
    """Column shift of the crosshatch verticals (px) and the error away from
    every edge."""
    def cols(img):
        prof = img[100:140].mean(0)
        out = []
        for c0 in range(32, img.shape[1] - 8, 64):
            w = np.clip(prof[c0 - 8:c0 + 9] - 0.3, 0, None)
            out.append((np.arange(c0 - 8, c0 + 9) * w).sum() / w.sum())
        return np.array(out)
    shift = cols(pix) - cols(truth)
    edge = truth != 0.3
    near = edge.copy()
    for _ in range(4):
        near[1:] |= near[:-1]
        near[:-1] |= near[1:]
        near[:, 1:] |= near[:, :-1]
        near[:, :-1] |= near[:, 1:]
    far = ~near
    far[-1] = False                          # the generator's truth ignores the last half line
    far[:, :2] = far[:, -2:] = False         # the active line's own edges
    return float(np.abs(shift).max()), float((pix - truth)[far].std()), float(np.abs(pix - truth)[far].max())


@pytest.mark.parametrize("std,lpf,line_hz", [(ref.VID_STD_525, 262.5, 4.5e6 / 286),
                                             (ref.VID_STD_625, 312.5, 15625.0)])
def test_clean_picture_is_the_planted_one(std, lpf, line_hz):
    x, truth = ref.video_synth(std, fs=FS, seconds=0.25)
    v, out = _twin(x, yc=ref.VID_YC_OFF)
    assert _consecutive(_counters(out))
    pix, meta = out[-1]
    t = truth["frames"][ref.video_read_counter(pix) - truth["frame0"]]
    shift, rms, worst = _geometry(pix, t)
    assert shift < 0.25                     # columns land within a quarter pixel
    assert rms < 0.01 and worst < 0.05      # levels: black, grey, white where they belong
    assert meta["standard"] == std and meta["v_lock"] == 1 and meta["progressive"] == 0
    assert meta["lines_per_field"] == pytest.approx(lpf, abs=0.01)
    assert meta["line_hz"] == pytest.approx(line_hz, rel=1e-4)
    assert meta["lines"] == 2 * truth["visible"] and meta["width"] == 720


@pytest.mark.parametrize("fs", [8e6, 20e6])
def test_other_rates_lock_and_read(fs):
    x, _t = ref.video_synth(ref.VID_STD_525, fs=fs, seconds=0.25)
    _v, out = _twin(x, fs=fs)
    assert _consecutive(_counters(out))


def test_lock_times():
    """H-lock within 20 lines of the first line pulse; V-lock within two
    fields of the first vertical interval (VIDEO_PLAN §6.2)."""
    T = 286 / 4.5e6
    x, _t = ref.video_synth(ref.VID_STD_525, fs=FS, seconds=0.2, start_lines=40.25)
    v = ref.Video(fs=FS, polarity=ref.VID_POL_NORMAL)
    core = v.cores[0]
    t_h = t_v = None
    step = v.S
    for i in range(0, x.size, step):
        v.process(x[i:i + step])
        if t_h is None and core.track:
            t_h = (i + step) / FS
        if t_v is None and core.v_lock:
            t_v = (i + step) / FS
    assert t_h is not None and t_h <= 20 * T
    # the stream starts at line 40 of field A: the next vertical interval is
    # field B's, at 262.5 - 40 + 3 lines; field lock comes two fields later
    first_vi = (262.5 - 40.25 + 3) * T
    assert t_v is not None and t_v <= first_vi + 2 * 262.5 * T + 4 * T


@pytest.mark.parametrize("cnr", [20, 10])
def test_fm_noise_down_to_10_db(ntsc, cnr):
    x, _t = ntsc
    _v, out = _twin(_fm(x, cnr, seed=cnr))
    assert _consecutive(_counters(out))
    assert abs(out[-1][1]["snr_db"] - cnr) < 3.0          # the sync-to-noise readout tracks it


def test_inverted_input_is_found_by_auto(ntsc):
    x, _t = ntsc
    v, out = _twin(-x)
    assert _consecutive(_counters(out))
    assert {m["polarity"] for _p, m in out} == {2}
    _v2, out2 = _twin(x)
    assert np.array_equal(out[-1][0], out2[-1][0])        # the same picture either way


def test_forced_wrong_polarity_shows_nothing(ntsc):
    x, _t = ntsc
    _v, out = _twin(-x, polarity=ref.VID_POL_NORMAL)
    assert out == []


def test_custom_line_rate_auto_and_forced():
    x, truth = ref.video_synth(ref.VID_STD_CUSTOM, fs=FS, seconds=0.3, line_hz=15500.0, lines=601)
    v, out = _twin(x)
    assert _consecutive(_counters(out))
    m = out[-1][1]
    assert m["standard"] == ref.VID_STD_CUSTOM
    assert m["line_hz"] == pytest.approx(15500.0, rel=1e-3)
    assert m["lines_per_field"] == pytest.approx(300.5, abs=0.3)
    _v, out = _twin(x, standard=ref.VID_STD_CUSTOM, custom_line_hz=15500.0, custom_lines=601)
    assert _consecutive(_counters(out))
    _v, out = _twin(x, standard=ref.VID_STD_525)      # forced to the wrong standard: nothing
    assert out == []


def test_progressive_source():
    x, _t = ref.video_synth(ref.VID_STD_525, fs=FS, seconds=0.15, progressive=True)
    _v, out = _twin(x)
    assert _consecutive(_counters(out))
    assert out[-1][1]["progressive"] == 1 and out[-1][1]["lines"] == 240


def test_one_field_doubles_lines(ntsc):
    x, _t = ntsc
    _v, out = _twin(x, fields=ref.VID_ONE_FIELD)
    p, m = out[-1]
    assert m["lines"] == 480 and np.array_equal(p[0::2], p[1::2])
    assert len(out) == 4                                   # a frame a field: the queue is full


def test_stacking_gains_what_an_average_should():
    """A 1/N exponential average settles at a noise power (1/N) / (2 - 1/N)
    of one frame's: 14.9 dB at N = 16. Measured 14.2 dB on this stream."""
    x, _t = ref.video_synth(ref.VID_STD_525, fs=FS, seconds=1.0, pattern="flat")
    y = _fm(x, 9, seed=5)
    sd = {}
    for n in (1, 16):
        _v, out = _twin(y, stack=n, yc=ref.VID_YC_OFF)
        sd[n] = float((out[-1][0][60:460, 40:680] - 0.5).std())
    assert 20 * np.log10(sd[1] / sd[16]) > 12.0


def test_motion_aware_stacking_does_not_smear_a_moving_bar():
    x, _t = ref.video_synth(ref.VID_STD_525, fs=FS, seconds=0.6, pattern="movingbar")
    _v, out = _twin(_fm(x, 15, seed=7), stack=16, yc=ref.VID_YC_OFF)
    pix, meta = out[-1]
    n = ref.video_read_counter(pix)
    prof = pix[100:400].mean(0)
    c0 = 60 + (16 * n) % (720 - 140)
    assert prof[c0 + 2:c0 + 14].min() > 0.85               # the bar is where it is now ...
    assert prof[c0 - 14:c0 - 2].max() < 0.4                # ... and not where it was
    assert meta["stacked"] > 4                             # the still background did stack


@pytest.mark.parametrize("std", [ref.VID_STD_525, ref.VID_STD_625])
def test_chroma_trap_and_comb_remove_the_subcarrier(std):
    x, _t = ref.video_synth(std, fs=FS, seconds=0.2, pattern="flat", chroma=0.4)
    ripple = {}
    for yc in (ref.VID_YC_OFF, ref.VID_YC_TRAP, ref.VID_YC_COMB):
        _v, out = _twin(x, yc=yc)
        ripple[yc] = float((out[-1][0][60:-20, 40:680] - 0.5).std())
    assert ripple[ref.VID_YC_OFF] > 0.05
    assert ripple[ref.VID_YC_TRAP] < 0.01 and ripple[ref.VID_YC_COMB] < 0.01


@pytest.mark.parametrize("sizes", [[7], [239, 240, 241], [999], [4096], [8192], [1, 5000, 3, 70000]])
def test_chunk_invariance(pal, sizes):
    x, _t = pal
    y = _fm(x, 14, seed=2)
    ref_v, want = _twin(y, stack=4)
    v = ref.Video(fs=FS, stack=4)
    i = k = 0
    while i < y.size:
        n = sizes[k % len(sizes)]
        v.process(y[i:i + n])
        i += n
        k += 1
    got = _drain(v)
    assert len(got) == len(want) and v.stats() == ref_v.stats()
    for (a, ma), (b, mb) in zip(got, want):
        assert np.array_equal(a, b) and ma == mb


def test_a_short_dropout_loses_no_frame():
    x, _t = ref.video_synth(ref.VID_STD_525, fs=FS, seconds=0.5)
    z = ref.video_fm(x, FS, cnr_db=20, seed=9)
    i0, n = int(0.25 * FS), int(0.002 * FS)
    rng = np.random.default_rng(2)
    z[i0:i0 + n] = (0.1 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)
    v = ref.Video(fs=FS)
    got = []
    y = ref.video_fm_demod(z, FS)
    for i in range(0, y.size, int(0.01 * FS)):
        v.process(y[i:i + int(0.01 * FS)])
        got += _drain(v)
    assert _consecutive(_counters(got))
    assert v.stats()["h_relocks"] == 1


# ---- negative controls ------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _negatives():
    rng = np.random.default_rng(4)
    n = int(0.4 * FS)
    t = np.arange(n) / FS
    out = {"white noise": rng.standard_normal(n).astype(np.float32)}
    voice = np.sin(2 * np.pi * 400 * t) + 0.5 * np.sin(2 * np.pi * 1300 * t)
    z = np.exp(1j * 2 * np.pi * np.cumsum(3e3 * voice) / FS) + \
        0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    out["FM voice carrier"] = ref.video_fm_demod(z.astype(np.complex64), FS)
    sym = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * (np.mod(t, 2e-3) < 1e-3)
    out["OFDM-like bursts"] = ref.video_fm_demod(sym.astype(np.complex64), FS)
    T = 286 / 4.5e6
    out["radar-like pulse train at the line rate"] = (
        np.where(np.mod(t, T) < 4.7e-6, -1.0, 0.0) + 0.05 * rng.standard_normal(n)).astype(np.float32)
    return out


@pytest.mark.parametrize("name", list(_negatives()))
def test_negative_controls_make_no_frames(name):
    x = _negatives()[name]
    v, out = _twin(x)
    assert out == []
    assert v.stats()["frames_out"] == 0
    if name.startswith("radar"):
        assert "no fields" in v.describe()               # H-lock alone is never a picture


# ---- live settings ------------------------------------------------------------------
def test_display_settings_change_without_losing_lock():
    x, _t = ref.video_synth(ref.VID_STD_525, fs=FS, seconds=0.4)
    v = ref.Video(fs=FS)
    half = x.size // 2
    v.process(x[:half])
    first = _drain(v)
    v.set(stack=8, width=640, yc=ref.VID_YC_COMB)
    v.process(x[half:])
    after = _drain(v)
    assert first and after
    assert {m["width"] for _p, m in after} == {640}
    assert v.stats()["v_relocks"] == 1 and v.stats()["h_relocks"] == 1


def test_a_new_standard_reacquires():
    x, _t = ref.video_synth(ref.VID_STD_525, fs=FS, seconds=0.4)
    v = ref.Video(fs=FS)
    v.process(x[: x.size // 2])
    v.set(standard=ref.VID_STD_625)
    v.process(x[x.size // 2:])
    _drain(v)
    assert not v.cores[0].v_lock                       # a 525 source is not a 625 one


def test_fs_cannot_change():
    v = ref.Video(fs=FS)
    with pytest.raises(ValueError, match="fs cannot change"):
        v.set(fs=20e6)


# ======================================================================================
# The C against the twin
# ======================================================================================
needs_c = pytest.mark.skipif(not (atkdsp.available() and hasattr(atkdsp, "Video")),
                             reason="atkdsp library without §32 (atkdsp_video_*)")


def _c(x, chunks=None, sets=(), **cfg):
    cfg.setdefault("fs", FS)
    v = atkdsp.Video(**cfg)
    return _feed(v, x, chunks, sets)


def _t(x, chunks=None, sets=(), **cfg):
    cfg.setdefault("fs", FS)
    v = ref.Video(**cfg)
    return _feed(v, x, chunks, sets)


def _feed(v, x, chunks, sets):
    out = []
    step = chunks or x.size
    sets = dict(sets)
    for i in range(0, x.size, step):
        if i in sets:
            v.set(**sets[i])
        v.process(x[i:i + step])
        out += _drain(v)
    return v, out


def _same(c, t):
    vc, oc = c
    vt, ot = t
    assert vc.stats() == vt.stats()
    assert len(oc) == len(ot)
    for (pc, mc), (pt, mt) in zip(oc, ot):
        assert pc.shape == pt.shape
        assert np.abs(pc - pt).max() <= 1e-5            # float rounding only
        for k in mc:
            if isinstance(mc[k], float):
                assert mc[k] == pytest.approx(mt[k], rel=1e-6, abs=1e-6), k
            else:
                assert mc[k] == mt[k], k


@functools.lru_cache(maxsize=1)
def _scenarios():
    x525, _ = ref.video_synth(ref.VID_STD_525, fs=FS, seconds=0.25)
    x625, _ = ref.video_synth(ref.VID_STD_625, fs=FS, seconds=0.25, chroma=0.3)
    xc, _ = ref.video_synth(ref.VID_STD_CUSTOM, fs=FS, seconds=0.3, line_hz=15500.0, lines=601)
    xp, _ = ref.video_synth(ref.VID_STD_525, fs=FS, seconds=0.15, progressive=True)
    x20, _ = ref.video_synth(ref.VID_STD_525, fs=20e6, seconds=0.2)
    return {
        "525 clean": (x525, {}),
        "525 FM 10 dB, stack 8": (_fm(x525, 10, seed=3), dict(stack=8)),
        "625 FM 12 dB, comb, chroma": (_fm(x625, 12, seed=4), dict(yc=ref.VID_YC_COMB)),
        "625 trap, swapped fields": (x625, dict(swap_fields=1)),
        "custom auto": (xc, {}),
        "progressive": (xp, {}),
        "inverted auto": (-x525, {}),
        "one field, width 512": (x525, dict(fields=ref.VID_ONE_FIELD, width=512)),
        "20 MS/s": (x20, dict(fs=20e6)),
        "noise": (np.random.default_rng(1).standard_normal(int(0.2 * FS)).astype(np.float32), {}),
    }


@needs_c
@pytest.mark.parametrize("name", list(_scenarios()))
def test_c_equals_twin(name):
    x, cfg = _scenarios()[name]
    _same(_c(x, chunks=4096, **cfg), _t(x, chunks=4096, **cfg))


@needs_c
def test_c_one_sample_blocks():
    x, _t0 = ref.video_synth(ref.VID_STD_525, fs=FS, seconds=0.1)
    _same(_c(x, chunks=1), _t(x))


@needs_c
def test_c_live_settings_equal_twin():
    x, _t0 = ref.video_synth(ref.VID_STD_525, fs=FS, seconds=0.4)
    sets = ((int(0.15 * FS) // 4096 * 4096, dict(stack=8, width=640)),
            (int(0.25 * FS) // 4096 * 4096, dict(standard=ref.VID_STD_625)),
            (int(0.3 * FS) // 4096 * 4096, dict(standard=ref.VID_STD_AUTO, yc=ref.VID_YC_COMB)))
    _same(_c(x, chunks=4096, sets=sets), _t(x, chunks=4096, sets=sets))


@needs_c
def test_c_frame_peek_and_describe():
    x, _t0 = ref.video_synth(ref.VID_STD_525, fs=FS, seconds=0.2)
    v = atkdsp.Video(fs=FS)
    assert v.process(x) > 0
    assert v.describe().startswith("video: 525-line, interlaced, locked")
    pix, meta = v.frame()
    assert pix.shape == (meta["lines"], meta["width"]) == (480, 720)


@needs_c
def test_c_refuses_a_bad_config():
    with pytest.raises(atkdsp.AtkDspError):
        atkdsp.Video(fs=1e6)
