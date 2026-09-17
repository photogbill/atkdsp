"""atkdsp vs numpy on ATK's RF hot path, in CPU-seconds per second of signal.

Run from the repo root after building:  python bench/bench_hotpath.py

One 50 ms block at the given rate goes through the same stages ATK's
RfDspWorker runs today — unpack + DC block, one display line, one decoder
channel (mix + decimate to 48 kHz + FM), one analog channel — first with the
numpy twins (which ARE the current ATK algorithms, made stateful), then with
the C kernels. Below 1.0 is real time on one core.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
import atkdsp                                  # noqa: E402
from atkdsp import reference as ref            # noqa: E402


def lowpass(cutoff, fs, ntaps):
    fc = max(1e-4, min(0.499, cutoff / fs))
    n = np.arange(ntaps) - (ntaps - 1) / 2.0
    h = np.sinc(2 * fc * n) * np.hamming(ntaps)
    return (h / np.sum(h)).astype(np.float32)


def timeit(fn, reps):
    fn()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps


def run(fs, block_s=0.05, fft=8192, reps=5):
    n = int(fs * block_s)
    rng = np.random.default_rng(1)
    raw = rng.integers(-2048, 2048, 2 * n, dtype="<i2").tobytes()
    decim = int(round(fs / 48_000))
    h = lowpass(7_500, fs, 65)
    win_c = atkdsp.window("hann", fft)
    win_np = np.hanning(fft).astype(np.float32)

    # ---- numpy twins (= today's ATK algorithms, stateful) ----
    st = np.zeros(1, np.complex64)
    nco_a, nco_b = ref.Nco(250e3, fs), ref.Nco(0.0, fs)
    fir_a, fir_b = ref.Fir(h, decim), ref.Fir(h, decim)
    prev_a, prev_b = np.zeros(1, np.complex64), np.zeros(1, np.complex64)

    def np_block():
        iq = ref.unpack(raw, "ci16q11")                       # no DC (ATK does a mean)
        iq = (iq - iq.mean()).astype(np.complex64)
        ref.power_db(iq[-fft:], win_np)                       # ATK: one line per chunk
        ref.fm_demod(fir_a.process(nco_a.mix(iq)), prev_a)   # decoder channel
        ref.fm_demod(fir_b.process(nco_b.mix(iq)), prev_b)   # analog channel

    # ---- C kernels ----
    cst = np.zeros(1, np.complex64)
    out = np.empty(n, np.complex64)
    mixed = np.empty(n, np.complex64)
    cn_a, cn_b = atkdsp.Nco(250e3, fs), atkdsp.Nco(0.0, fs)
    cf_a, cf_b = atkdsp.Fir(h, decim), atkdsp.Fir(h, decim)
    plan = atkdsp.Fft(fft)
    line = np.empty(fft, np.float32)
    cp_a, cp_b = np.zeros(1, np.complex64), np.zeros(1, np.complex64)
    ch = np.empty(cf_a.out_max(n) + 1, np.complex64)
    audio = np.empty(ch.size, np.float32)

    def c_block_one_line():
        iq = atkdsp.unpack(raw, "ci16q11", 0.002, cst, out)
        plan.power_db(iq[-fft:], win_c, line)
        k = cf_a.process(cn_a.mix(iq, mixed), ch); atkdsp.fm_demod(k, cp_a, out=audio[:k.size])
        k = cf_b.process(cn_b.mix(iq, mixed), ch); atkdsp.fm_demod(k, cp_b, out=audio[:k.size])

    def c_block_poi():
        iq = atkdsp.unpack(raw, "ci16q11", 0.002, cst, out)
        plan.spectrum_reduce(iq, None, win_c, "max", line)     # EVERY frame, 100 % POI
        k = cf_a.process(cn_a.mix(iq, mixed), ch); atkdsp.fm_demod(k, cp_a, out=audio[:k.size])
        k = cf_b.process(cn_b.mix(iq, mixed), ch); atkdsp.fm_demod(k, cp_b, out=audio[:k.size])

    # the REAL channelizer: staged, 60 dB alias rejection, exactly 48 000 Hz out
    ddc_a = atkdsp.Ddc(fs, 250e3, 15_000, 48_000, 60, max_block=n)
    ddc_b = atkdsp.Ddc(fs, 0.0, 12_000, 48_000, 60, max_block=n)
    chd = np.empty(ddc_a.out_max(n) + 8, np.complex64)

    def c_block_ddc():
        iq = atkdsp.unpack(raw, "ci16q11", 0.002, cst, out)
        plan.spectrum_reduce(iq, None, win_c, "max", line)
        k = ddc_a.process(iq, chd); atkdsp.fm_demod(k, cp_a, out=audio[:k.size])
        k = ddc_b.process(iq, chd); atkdsp.fm_demod(k, cp_b, out=audio[:k.size])

    t_np = timeit(np_block, reps) / block_s
    t_c1 = timeit(c_block_one_line, reps) / block_s
    t_cp = timeit(c_block_poi, reps) / block_s
    t_cd = timeit(c_block_ddc, reps) / block_s
    frames = n // fft
    print(f"{fs / 1e6:5.1f} MSPS, {block_s * 1e3:.0f} ms blocks, fft {fft}: "
          f"numpy {t_np:5.2f}  |  C (one line) {t_c1:5.2f}  |  C 100%-POI ({frames} FFTs/block) {t_cp:5.2f}"
          f"  |  C POI + real DDCs {t_cd:5.2f}   -> C is {t_np / t_c1:4.1f}x  CPU-s per real-s")


if __name__ == "__main__":
    print(f"atkdsp {atkdsp.version()} — {atkdsp.build_info()} — threads {atkdsp.get_threads()}")
    for fs in (2.4e6, 10e6, 40e6):
        run(fs)
    print("\n(the numpy column is today's ATK algorithm; the C columns are the same work in atkdsp)")
