# atkdsp

The Analyst Toolkit's native DSP kernels: one shared library, one header that
is the ABI, a `ctypes` binding with a numpy twin for every kernel, and the
tests that hold the two to each other.

It exists because of one measurement. On 2026-09-15 ATK's RF path, running
in numpy, fell one 1.6 ms chunk behind at 10 MSPS and never recovered: with a
backlog the worker thread was never idle, every Python bytecode boundary
became a GIL hand-off, the same chunk cost 25× more than a minute earlier,
and the GUI's timers starved (ATK `FUTURE_PLANS.md` §RF-F, §0). A kernel
called through `ctypes` holds the interpreter lock for none of its running
time, allocates nothing, and costs the same on the ten-thousandth block as on
the first. That is the property this library is for; speed is the bonus.

Measured on a two-core SSE2 container, the same 50 ms block through the same
stages ATK runs today (unpack, DC block, display line, decoder channel,
analog channel), CPU-seconds per second of signal:

| rate | numpy (today's ATK) | atkdsp, one display line | atkdsp, 100 % POI (every FFT) | + two real DDCs (60 dB, exact 48 kHz) |
|---|---|---|---|---|
| 2.4 MSPS | 0.43 | 0.04 | 0.05 | 0.07 |
| 10 MSPS | 1.80 | 0.10 | 0.18 | 0.31 |
| 40 MSPS | 6.68 | 0.39 | **0.64** | 1.04 |

`bench/bench_hotpath.py` reproduces it; run it on the machine that matters.

## Layout

```
include/atkdsp.h        THE ABI. Read its header comment first.
src/                    C99 kernels, one file per group (convert, nco, fir,
                        resample, fft, demod, display, detect)
vendor/pocketfft/       Martin Reinecke's pocketfft (C, BSD-3) — the only
                        third-party code, used through src/fft.c only
python/atkdsp/          __init__.py  the ctypes binding
                        reference.py the numpy twins (the specification)
tests/test_smoke.c      C-only proof the DLL is sane (no Python needed)
tests/test_cross_check.py  every kernel vs its twin, in uneven blocks
bench/                  the measurement above
bin/                    build output: atkdsp.dll / atkdsp.so, atkdsp_smoke
build.bat, build.sh, CMakeLists.txt
```

## Build

Windows: `build.bat` — finds MSVC (the tools ATK's `install.bat` already
requires for llama-cpp-python) and CMake (on PATH, or ATK's
`envs\atk_core`), builds `bin\atkdsp.dll`, runs the smoke test. With no
CMake it drives `cl.exe` directly. Linux/macOS: `./build.sh`.

Then, from the repo root: `python -m pytest tests -q`.

Options (CMake): `ATKDSP_AVX2` (on), `ATKDSP_OPENMP` (on — the batched FFT
runs frames in parallel), `ATKDSP_TESTS` (on).

## The rules, in one place

They are in `include/atkdsp.h` and enforced by the tests; in short:

1. C99, C ABI, no globals, no callbacks. Loadable from anything.
2. **No allocation in a processing call.** `*_create` allocates once;
   `*_process` / `*_exec` / `*_mix` / `*_demod` never do.
3. **Streaming state is explicit and continuous.** NCO phase, FIR history,
   resampler position, FM previous sample — carried by the caller's handle.
   `test_cross_check.py` feeds every streaming kernel in blocks of
   1, 999, 4096, 7 … samples and asserts the result equals one unbroken call.
4. Numbers in, numbers out. No I/O, no printing, no threads the caller
   cannot see (OpenMP inside one call, under `atkdsp_set_threads`).
5. `ATKDSP_ABI_VERSION` bumps on any change to an existing function. The
   binding refuses a mismatched library with a sentence.
6. Every kernel has a numpy twin with the same semantics. A change to a
   kernel is a change to its twin, in the same commit, or the tests say so.
7. Not `-ffast-math` / `/fp:fast`: NaN means "unmeasured" in the stitch and
   display kernels, and fast-math deletes the tests for it. Found the hard
   way on day one.

## What is in v0.2.0 (ABI 1)

| group | kernels | replaces in ATK |
|---|---|---|
| unpack | `atkdsp_unpack` (cu8 / ci8 / ci16 / ci16q11 / cf32, running DC block) | `dsp.iq_to_complex`, `dsp.dc_block` |
| NCO | `atkdsp_nco_*` phase-continuous mixer | `dsp.frequency_shift` (the per-sample `np.exp`) |
| FIR | `atkdsp_fir_*` decimating, carried history | `dsp._decimating_fir` |
| resampler | `atkdsp_resampler_*` polyphase L/M, exact counts | (nothing — 48 077 Hz was fed to dsd-neo as 48 000) |
| FFT | `atkdsp_fft_*`, `atkdsp_window`, `atkdsp_power_db`, **`atkdsp_spectrum_reduce`** (100 % POI: every frame, max/min/avg) | `dsp.spectrum_db` (one frame per chunk, the rest discarded) |
| demod | `atkdsp_fm_demod`, `atkdsp_am_demod`, state carried | `dsp.fm_demodulate`, `dsp.am_demodulate` |
| display | `atkdsp_db_to_pixels`, `atkdsp_decimate_max` | pyqtgraph's per-frame LUT pass, `spectrum_view.decimate_max` |
| detect | `atkdsp_median`, `atkdsp_detect_channels`, `atkdsp_stitch_max` | the per-bin Python loops in `signal_id.detect_channels` and `sweep.stitch` |
| design | `atkdsp_design_lowpass` — Kaiser windowed sinc to a stated stopband | `dsp.design_lowpass` (65 taps at the input rate, whatever the rate) |
| DDC | `atkdsp_ddc_*` — NCO → staged decimation (each stage designed for `atten_db` of alias rejection at ITS rate) → exact-rate polyphase resampler. `describe()` prints the plan, e.g. `10 MSPS: NCO -> /8 (31 taps) -> /2 (9) -> /13 (303) -> 48076.9 Hz -> x624/625 -> 48000 Hz` | `dsp.channelize` — which passed a tone 100 kHz off the channel at −2.9 dB (10 MSPS) / −0.2 dB (40 MSPS); the DDC puts it 80 dB down, and the test asserts 60 |

## How ATK uses it

ATK loads it through `atk/core/dsp_native.py`, which looks in
`vendor\atkdsp\bin\`, then a sibling `..\atkdsp\bin\` checkout, then PATH —
one discovery order, so there is never a second copy loaded. If the library
is absent or the ABI is wrong, ATK runs on the numpy twins and says so on the
Setup page; it never silently computes something different. The twins live
here, in `python/atkdsp/reference.py`, so ATK's fallback and this repo's
specification are one file.

## Roadmap (ABI 2 and later)

Ordered by what ATK's plan needs next. Items below the line are from the
wider capability list (Bill's 2026-09-17 design notes) and are real, but
each is its own project with its own verification.

1. **Overlap-save channel extraction from the block FFT** — every VFO and
   every watchlist channel from the one FFT the display already needs;
   a polyphase filterbank for uniformly spaced channels.
2. **A float32 FFT** behind the same `atkdsp_fft_*` ABI, replacing the
   double-precision pocketfft path when the size is a power of two.
3. **Ring-buffer primitives** for the block ring / DVR (`§RF-F2`, `§RF-F9`):
   sequence numbers, multi-consumer cursors, overflow accounting.
4. A float32 SIMD NCO path: at 40 MSPS the double-precision mixer is now
   the largest single cost in a DDC (~4 ns/sample on SSE2); a float phasor
   with a shorter renormalisation interval would halve it.

---

5. Instantaneous attributes (envelope / phase / frequency in one pass),
   m-th power, delay-and-multiply — the Signals Analysis bench's inner loops.
6. Higher-order cumulants (C20 … C63) for modulation classification.
7. Cyclostationary engines (FAM / SSCA) and pseudo Wigner-Ville.
8. Correlators: sync-word search, LFSR descrambler search, Gardner /
   Mueller-Müller timing.
9. Multi-channel: spatial covariance, cross-ambiguity (TDOA/FDOA), MUSIC —
   for a two-channel bladeRF 2.0 and the KrakenSDR.
10. Cellular survey: PSS/SSS/SSB **correlators** belong here; the broadcast
    channel decoding (MIB/SIB) does not — see the note in ATK's plan.

## Licence

Same terms as ATK. `vendor/pocketfft` is BSD-3 (its `LICENSE.md` is kept
beside it).
"# atkdsp" 
