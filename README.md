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
                        resample, fft, demod, display, detect, ddc, lte*,
                        fsk, dmr_sync, the bitstream files, cfar, burst,
                        ring, pulses, cyclic, tone_id, dsss_*, css, xspec,
                        channelizer, undertow, psync)
vendor/pocketfft/       Martin Reinecke's pocketfft (C, BSD-3) — the only
                        third-party code, used through src/fft.c and, for
                        the burst word's spectra, src/burst.c
python/atkdsp/          __init__.py  the ctypes binding
                        reference.py the numpy twins (the specification)
tests/test_smoke.c      C-only proof the DLL is sane (no Python needed)
tests/test_cross_check.py  every kernel vs its twin, in uneven blocks
tests/test_detection.py    CFAR, the burst word, the ring (v0.16.0)
tests/test_phase3.py       pulse runs, the cyclic probes, the analog IDs (v0.17.0)
tests/test_phase6.py       spread spectrum: the blind + known-code paths, LoRa,
                           the receiver pair and the lattice (v0.18.0)
tests/test_channelizer.py  the polyphase filterbank, its coherent N-element
                           form, the W1/W2 accumulators (v0.19.0), the
                           extractor (v0.20.0), and the phase fixes (v0.21.0)
tests/test_undertow.py     the scale-space survey (v0.21.0)
tests/test_psync.py        candidates, the period search, the bootstrap and
                           the hit blob (v0.22.0)
tests/test_portability.py  what Linux can't compile-check: MSVC's OpenMP is 2.0
tests/test_ring_stress.c   the ring under threads; ring_bench.bat runs it
                        as the 40 MSPS bench
bench/                  the measurement above
bin/                    build output: atkdsp.dll / atkdsp.so, atkdsp_smoke,
                        atkdsp_ring_stress
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

## What it provides

`v0.25.1`, ABI 26. One header, `include/atkdsp.h`, is the whole surface;
every kernel below has a numpy twin in `python/atkdsp/reference.py` and a
cross-check test that holds the two to each other. Grouped by what they are
for:

| Group | Kernels |
|---|---|
| **Sample path** | unpacking of the receiver formats (cu8, ci8, ci16, 12-bit ci16, cf32) with block DC removal, the NCO/mixer, decimating FIR filters, rational and arbitrary-ratio resamplers, a batched FFT (pocketfft, OpenMP across frames), the display reduction (spectrum lines, max-hold, waterfall stitch), digital down-conversion |
| **Demodulation** | FM and AM demodulators, FSK with a front end for sub-optimal RF, the DMR / P25 sync correlator with sample-exact positions for decoder-clocked DF |
| **Bitstream engine** | bit packing and framing, CRC search and polynomial recovery, LFSR descrambler search, block, convolutional and Reed–Solomon FEC, code recognition — naming the code a stream is using, with the evidence |
| **Detection** | CA-CFAR over spectrum frames, the burst tracker and burst descriptor word, the multi-consumer sample ring (the DVR), pulse descriptor words and pulse runs, the cyclostationary probes, analog tone identification, cross-spectra and the receiver-pair covariance |
| **Spread spectrum** | DSSS acquisition and detection, blind code recovery and the known-code library, chirp spread spectrum (LoRa), the lattice detector |
| **Channelizer** | the polyphase filterbank with per-channel power and covariance kept in place, the overlap-save extractor for arbitrary channels, per-channel calibration — the grid behind ATK's trunking taps and "DF everything" |
| **Below the floor** | Undertow, the scale-space survey over (scale, channel) cells; psync, the period-synchronous watch with its bootstrap and hit blob; the full lag scan |
| **Cellular** | LTE cell detection (PSS/SSS), MIB, SIB1 and the system-information blocks, neighbour and inter-frequency lists, PRACH, and turbo decoding |
| **Analog video** | composite video to luma frames with lock metadata: 525 / 625 / measured standards, auto polarity, interlaced or progressive, chroma trap or line comb, motion-aware stacking |

Every result that is a judgement carries its provenance — which method
produced it and from what — and the detection statistics carry a false-alarm
probability rather than a tuned threshold. The release-by-release record of
what each ABI added, what it measured and what it found is
[`CHANGELOG.md`](CHANGELOG.md).

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

Since v0.16.0 the detection track has three ATK callers: `atk/core/cue_detector.py`
(the CFAR and burst tracker over a receiver's spectrum frames, the records
cued DF waits on, and `measure()` with the burst descriptor word),
`atk/core/tdma.py` (frame timing folded from those records' start times —
Python, as the plan says), and `atk/core/iq_cut.py`, whose DVR ring can sit
on the C ring (`settings["sdr"]["dvr_native"]`, off until `ring_bench.bat`
has passed on the PC). Bug Sniffer's engine is lent the CFAR through ATK's
hooks (`energy_cfar`).

## Licence

Same terms as ATK. `vendor/pocketfft` is BSD-3 (its `LICENSE.md` is kept
beside it).

[`CHANGELOG.md`](CHANGELOG.md) — every version, with its measurements.
