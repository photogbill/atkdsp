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

Measured on a two-core AVX2 VM, the same 50 ms block through the same stages
ATK runs today (unpack, DC block, display line, decoder channel, analog
channel), CPU-seconds per second of signal — where 1.00 means the worker is
exactly saturated and the first hiccup starts a backlog it never clears:

| rate | numpy (ATK without this) | atkdsp, everything, 100 % POI |
|---|---|---|
| 2.4 MSPS | 0.24 | **0.04** |
| 10 MSPS | 0.68 | **0.15** |
| 40 MSPS | 2.62 | **0.68** |

The atkdsp column transforms EVERY FFT frame in the block (977 of them at
40 MSPS) where the numpy one transforms a single frame per chunk and discards
the rest — so it is doing about a thousand times the spectral work at a
quarter of the cost. `bench/bench_hotpath.py` and ATK's
`tests/bench/bench_chain.py` reproduce it; run them on the machine that
matters, because the display reduction uses every core it is given.

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

## What is in v0.9.0 (ABI 10)

Since v0.8.0: the FFT plan is genuinely safe to share across caller threads
(a per-slot claim replaced `omp_get_thread_num`, which returned 0 for every
thread outside a parallel region, so two callers wrote one scratch buffer);
`atkdsp_spectrum_stats` computes per-bin max/avg/min **and spectral kurtosis**
in one pass over the display's own frames (kurtosis ~2 for noise, ~1 for a
carrier, >2 for a bursty emitter — a per-bin CW/noise/keying call from one
block); and `atkdsp_window_stats` returns a window's coherent gain and ENBW so
a reading can be turned into true dBFS and a bin width in Hz.

| group | kernels | replaces in ATK |
|---|---|---|
| unpack | `atkdsp_unpack` (cu8 / ci8 / ci16 / ci16q11 / cf32, running DC block), **`atkdsp_unpack_dc`** — convert, remove a constant offset and return the block's mean, all in one pass; **`atkdsp_iq_health`** — DC, I/Q gain/phase imbalance, image rejection and clip fraction of a block, one pass | `dsp.iq_to_complex`, `dsp.dc_block` |
| NCO | `atkdsp_nco_*` phase-continuous mixer | `dsp.frequency_shift` (the per-sample `np.exp`) |
| FIR | `atkdsp_fir_*` decimating, carried history | `dsp._decimating_fir` |
| resampler | `atkdsp_resampler_*` polyphase L/M, exact counts | (nothing — 48 077 Hz was fed to dsd-neo as 48 000) |
| FFT | `atkdsp_fft_*` (thread-safe plans), `atkdsp_window`, **`atkdsp_window_stats`**, `atkdsp_power_db`, `atkdsp_spectrum_reduce` (100 % POI: every frame, max/min/avg), **`atkdsp_spectrum_stats`** (max/avg/min + spectral kurtosis, one pass) | `dsp.spectrum_db` (one frame per chunk, the rest discarded) |
| LTE | `atkdsp_lte_*` cell search (PSS/SSS/PCI), PBCH→MIB, PRACH presence, turbo+CRC, SIB1/SI decode | (new; see the header's §11 and the roadmap caveats) |
| demod | `atkdsp_fm_demod`, `atkdsp_am_demod`, state carried | `dsp.fm_demodulate`, `dsp.am_demodulate` |
| display | `atkdsp_db_to_pixels`, `atkdsp_decimate_max` | pyqtgraph's per-frame LUT pass, `spectrum_view.decimate_max` |
| detect | `atkdsp_median`, `atkdsp_detect_channels`, `atkdsp_stitch_max` | the per-bin Python loops in `signal_id.detect_channels` and `sweep.stitch` |
| design | `atkdsp_design_lowpass` — Kaiser windowed sinc to a stated stopband | `dsp.design_lowpass` (65 taps at the input rate, whatever the rate) |
| DDC | `atkdsp_ddc_*` — NCO → staged decimation (each stage designed for `atten_db` of alias rejection at ITS rate) → exact-rate polyphase resampler. `describe()` prints the plan, e.g. `10 MSPS: NCO -> /8 (31 taps) -> /2 (9) -> /13 (303) -> 48076.9 Hz -> x624/625 -> 48000 Hz` | `dsp.channelize` — which passed a tone 100 kHz off the channel at −2.9 dB (10 MSPS) / −0.2 dB (40 MSPS); the DDC puts it 80 dB down, and the test asserts 60 |

## How ATK uses it

ATK's `install.bat` runs `get_atkdsp.bat` (section 7c): it clones or
fast-forwards this repo into `vendor\atkdsp`, runs `build.bat` with the
MSVC tools and the cmake/ninja install.bat already put into ATK's env,
checks the DLL loads through ATK's own Python, and reports `[OK] atkdsp` or
`[--] atkdsp` with the reason. A failure never fails the install. Run
`get_atkdsp.bat` on its own to update, `get_atkdsp.bat /rebuild` to build
again.

Since 2026-09-18 ATK's RF worker goes through `atk/core/rf_chain.py` for
every stage of the hot path. Measured there (`tests/bench/bench_chain.py` in
ATK), on a **two-core AVX2 VM**, CPU-seconds per second of signal for the
whole chain — unpack, DC notch, display line, decoder channel, analog
channel, discriminator:

| rate | ATK on numpy | ATK on atkdsp | FFT frames per display line |
|---|---|---|---|
| 2.4 MSPS | 0.24 | **0.04** | 59 |
| 10 MSPS | 0.68 | **0.15** | 244 |
| 40 MSPS | 2.62 | **0.68** | 977 |

**One kernel ATK does not use: `atkdsp_unpack`'s `dc_alpha`.** The per-sample
DC pole costs 0.138 CPU-seconds per second of signal at 40 MSPS against 0.029
for the conversion it rides on — five times the work, to remove one spike —
and being a serial recurrence it will not vectorise or use a second core.
`atkdsp_unpack_dc` exists because of that measurement: the caller estimates
the offset once per block, and the subtraction rides free in the constant
each format already subtracts while the mean accumulates in the loop that is
already reading. Three passes became one, and ATK's unpack stage went from
0.093 to 0.041 CPU-seconds per second at 40 MSPS. The pole stays here — it
is the right tool for a caller with no blocks — and the cross-check tests it.

ATK loads it through `atk/core/dsp_native.py`, which looks in
`vendor\atkdsp\bin\`, then a sibling `..\atkdsp\bin\` checkout, then PATH —
one discovery order, so there is never a second copy loaded. If the library
is absent or the ABI is wrong, ATK runs on the numpy twins and says so on the
Setup page; it never silently computes something different. The twins live
here, in `python/atkdsp/reference.py`, so ATK's fallback and this repo's
specification are one file.

## Roadmap (ABI 3 and later)

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
4. ~~A float32 SIMD NCO path~~ — **measured, and it is not the answer.**
   Three formulations were benchmarked at 40 MSPS: the double phasor with
   8 lanes, a float phasor re-derived exactly from the double phase every
   256 samples, and a flat unit-stride version shaped like the FIR's dot.
   They came out at 1.54, 1.57 and 1.76 ns/sample — indistinguishable. At
   2 M samples a block the mixer is reading 16 MB and writing 16 MB, so it
   is **bandwidth-bound, not compute-bound**, and no arithmetic is going to
   move it. What WILL move it is not doing the pass: folding the mixer into
   the first decimating stage with pre-rotated complex taps removes a 16 MB
   write and a 16 MB read per channel per block. The taps double the stage's
   multiplies (+0.04) and the NCO disappears (−0.093), so it is worth about
   0.05 CPU-seconds per second per channel — not the 2x a faster multiply
   seemed to promise, and the only version of it that is real.

Item 1 is what ATK's measurement points at hardest: at 40 MSPS its two
channels cost 0.160 and 0.150 CPU-seconds per second of signal against 0.183
for the entire 977-frame display reduction. Channels taken from an analysis
FFT (item 1) would make the fifty-entry watchlist ATK wants cost roughly what
one channel costs now — but see ATK's plan for the constraint the original
design note missed: at 40 MSPS with a 2048-point FFT the bins are 19.5 kHz
apart and a 15 kHz channel does not span one, so the extractor needs its own
FFT sized from the channel, not the display's.

### Two things the 2026-09-18 pass found, worth keeping written down

**The FIR dot product was 4.3x slower than it needed to be, for readability.**
`sum_j h[j] * w[T-1-j]` over interleaved complex walks the window BACKWARDS
at stride two while the taps go forwards. Reversing the taps once at create,
and duplicating each one so the whole thing is a flat unit-stride
multiply-accumulate over 2T floats with eight accumulators (even lanes sum to
the real part, odd lanes to the imaginary), took a 485-tap decimate-by-8 from
1.76 to 0.41 CPU-seconds per second of signal. Same answers — all 91
cross-check tests, unchanged tolerances.

**`build.sh` was quietly building a different library than CMake.** Its
no-cmake fallback had no `-mavx2 -mfma` while `CMakeLists.txt` did, so the
same source on the same machine produced an SSE2 build or an AVX2 one
depending on whether cmake happened to be installed, with only the
`build_info` string to tell them apart. It probes for the flags now.

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
