"""The channelizer (atkdsp.h §28; docs/ATKDSP_PLAN.md §A4,
docs/CHANNELIZER_DF_REQUIREMENTS.md): the oversampled polyphase filterbank, its
multi-element coherent form, and the W1/W2 accumulators direction finding runs
on the output. Twin first (the numpy spec), then C against the twin in uneven
blocks, then the DF-critical R7 guarantees: inter-element phase coherence on a
planted plane wave, bit-identical elements on identical input, and exact
chunk invariance. `needs_lib` tests skip when the library is not built.

Run from the repo root:  python -m pytest tests/test_channelizer.py -q
"""

from __future__ import annotations

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

FS = 2.4e6
NCHAN = 192
SP = FS / NCHAN
BLOCKS = [1, 97, 960, 5000, 65537]


def _tone(fhz, n, seed=0, noise=0.0):
    k = np.arange(n)
    x = np.exp(2j * np.pi * fhz / FS * k)
    if noise:
        rng = np.random.default_rng(seed)
        x = x + noise * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return x.astype(complex)


# ---- the twin, the spec ----------------------------------------------------
def test_twin_centre_tone_lands_in_its_channel():
    y = ref.Pfb(FS, NCHAN).process(_tone(40 * SP, NCHAN * 120))[:, 40:-5]
    pdb = 10 * np.log10((np.abs(y) ** 2).mean(1) + 1e-20)
    assert np.argmax(pdb) == 40
    rej = pdb[40] - np.sort(pdb)[-2]
    assert rej > 80.0, f"centre-channel rejection only {rej:.1f} dB"


def test_twin_edge_tone_is_not_lost():
    # a tone on the shared edge of ch40/ch41 appears in BOTH (R4: 2x oversample)
    y = ref.Pfb(FS, NCHAN).process(_tone(40.5 * SP, NCHAN * 120))[:, 40:-5]
    pdb = 10 * np.log10((np.abs(y) ** 2).mean(1) + 1e-20)
    peak = pdb.max()
    assert pdb[40] - peak > -7 and pdb[41] - peak > -7
    assert pdb[45] - peak < -60          # far channels well down


def test_twin_plane_wave_phase_is_preserved():
    phi = np.deg2rad(37.0)
    n = NCHAN * 200
    base = _tone(40 * SP, n)
    ins = [base * np.exp(1j * e * phi) for e in range(5)]
    o = ref.Pfb(FS, NCHAN, max_nel=5).process_n(ins, 5)[:, 40, 60:-5]
    rel = np.angle(o.mean(1)); rel = rel - rel[0]
    planted = np.array([e * phi for e in range(5)])
    err = np.rad2deg(np.max(np.abs(np.angle(np.exp(1j * (rel - planted))))))
    assert err < 0.1, f"inter-element phase error {err:.3e} deg"


# ---- C against the twin ----------------------------------------------------
@needs_lib
@pytest.mark.parametrize("blk", BLOCKS)
def test_c_matches_twin_in_uneven_blocks(blk):
    x = _tone(40 * SP, NCHAN * 80, noise=0.02, seed=3)
    yt = ref.Pfb(FS, NCHAN).process(x)
    p = atkdsp.Pfb(FS, NCHAN)
    outs = []
    for i in range(0, x.size, blk):
        o = p.process(x[i:i + blk])
        if o.shape[1]:
            outs.append(o)
    yc = np.concatenate(outs, axis=1) if outs else np.zeros((NCHAN, 0), np.complex64)
    m = min(yc.shape[1], yt.shape[1])
    assert m > 0
    rel = np.max(np.abs(yc[:, :m] - yt[:, :m])) / np.max(np.abs(yt[:, :m]))
    assert rel < 1e-4, f"C!=twin rel {rel:.2e}"


@needs_lib
def test_c_chunk_invariance_is_exact():
    x = _tone(40 * SP, NCHAN * 80, noise=0.02, seed=4)
    y1 = atkdsp.Pfb(FS, NCHAN).process(x)
    p = atkdsp.Pfb(FS, NCHAN)
    outs = []
    for seg in (x[:5000], x[5000:5003], x[5003:30000], x[30000:]):
        o = p.process(seg)
        if o.shape[1]:
            outs.append(o)
    y2 = np.concatenate(outs, axis=1)
    m = min(y1.shape[1], y2.shape[1])
    assert np.array_equal(y1[:, :m], y2[:, :m])


@needs_lib
def test_c_elements_are_bit_identical_on_identical_input():
    base = _tone(40 * SP, NCHAN * 80, noise=0.03, seed=5)
    p = atkdsp.Pfb(FS, NCHAN, max_nel=5)
    o = p.process_n([base] * 5, 5)
    assert all(np.array_equal(o[0], o[e]) for e in range(5))


@needs_lib
def test_c_plane_wave_phase_within_tenth_degree():
    phi = np.deg2rad(37.0)
    base = _tone(40 * SP, NCHAN * 200)
    ins = [base * np.exp(1j * e * phi) for e in range(5)]
    o = atkdsp.Pfb(FS, NCHAN, max_nel=5).process_n(ins, 5)[:, 40, 60:-5]
    rel = np.angle(o.mean(1)); rel = rel - rel[0]
    planted = np.array([e * phi for e in range(5)])
    err = np.rad2deg(np.max(np.abs(np.angle(np.exp(1j * (rel - planted))))))
    assert err < 0.1


# ---- W1 / W2 ---------------------------------------------------------------
@needs_lib
def test_w1_covariance_matches_twin():
    phi = np.deg2rad(37.0)
    base = _tone(40 * SP, NCHAN * 80, noise=0.05, seed=6)
    ins = [base * np.exp(1j * e * phi) for e in range(5)]
    oc = atkdsp.Pfb(FS, NCHAN, max_nel=5).process_n(ins, 5)
    nchan, N = NCHAN, 5
    Rc = np.zeros((nchan, N, N), np.complex64); cc = np.zeros(nchan, np.uint32)
    atkdsp.pfb_cov_accumulate(oc, N, nchan, None, Rc, cc)
    Rt = np.zeros((nchan, N, N), np.complex128); ct = np.zeros(nchan, np.uint32)
    ref.pfb_cov_accumulate(oc, N, nchan, None, Rt, ct)
    rel = np.max(np.abs(Rc[40] - Rt[40])) / (np.max(np.abs(Rt[40])) + 1e-20)
    assert rel < 1e-4 and cc[40] == ct[40]
    # Hermitian
    assert np.allclose(Rc[40], Rc[40].conj().T, atol=1e-3)


@needs_lib
def test_w1_gate_and_reset():
    base = _tone(40 * SP, NCHAN * 40, noise=0.05, seed=7)
    oc = atkdsp.Pfb(FS, NCHAN, max_nel=3).process_n([base] * 3, 3)
    N = 3
    R = np.zeros((NCHAN, N, N), np.complex64); counts = np.zeros(NCHAN, np.uint32)
    gate = np.zeros(NCHAN, np.uint8); gate[40] = 1
    atkdsp.pfb_cov_accumulate(oc, N, NCHAN, gate, R, counts)
    assert counts[40] > 0 and counts[41] == 0           # only gated channel touched
    assert np.all(R[41] == 0)
    atkdsp.pfb_cov_reset(R, counts, N, NCHAN, gate)
    assert counts[40] == 0 and np.all(R[40] == 0)


@needs_lib
def test_w2_power_matches_twin():
    base = _tone(40 * SP, NCHAN * 60, noise=0.05, seed=8)
    ins = [base * np.exp(1j * e) for e in range(5)]
    oc = atkdsp.Pfb(FS, NCHAN, max_nel=5).process_n(ins, 5)
    pc, mc = atkdsp.pfb_channel_power(oc, 5, NCHAN)
    pt, mt = ref.pfb_channel_power(oc, 5, NCHAN)
    assert np.max(np.abs(pc - pt)) / (np.max(pt) + 1e-20) < 1e-4
    assert np.max(np.abs(mc - mt)) / (np.max(mt) + 1e-20) < 1e-4
    # the across-element mean is the mean of the per-element powers
    assert np.allclose(mc, pc.mean(0), rtol=1e-4)


# ---- bookkeeping + cal -----------------------------------------------------
@needs_lib
def test_out_rate_and_bookkeeping():
    p = atkdsp.Pfb(FS, NCHAN, oversample=2)
    assert abs(p.out_rate() - 2 * SP) < 1e-6
    assert abs(p.group_delay() - (NCHAN * 12 - 1) / 2.0) < 1e-6
    assert p.block_origin(0) == -(NCHAN * 12 - NCHAN // 2)   # -ov


@needs_lib
def test_set_cal_applies_once():
    base = _tone(40 * SP, NCHAN * 40, noise=0.02, seed=9)
    p0 = atkdsp.Pfb(FS, NCHAN, max_nel=2)
    y0 = p0.process_n([base, base], 2)
    p1 = atkdsp.Pfb(FS, NCHAN, max_nel=2)
    cal = np.ones((2, NCHAN), np.complex64)
    cal[1, :] = np.exp(1j * np.deg2rad(90.0))        # rotate element 1 by 90 deg
    p1.set_cal(cal, 2)
    y1 = p1.process_n([base, base], 2)
    # element 0 unchanged, element 1 rotated by the cal
    assert np.allclose(y1[0], y0[0], atol=1e-5)
    assert np.allclose(y1[1], y0[1] * np.exp(1j * np.deg2rad(90.0)), atol=1e-4)
