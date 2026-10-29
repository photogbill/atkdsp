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
                        ring, pulses, cyclic, tone_id)
vendor/pocketfft/       Martin Reinecke's pocketfft (C, BSD-3) — the only
                        third-party code, used through src/fft.c and, for
                        the burst word's spectra, src/burst.c
python/atkdsp/          __init__.py  the ctypes binding
                        reference.py the numpy twins (the specification)
tests/test_smoke.c      C-only proof the DLL is sane (no Python needed)
tests/test_cross_check.py  every kernel vs its twin, in uneven blocks
tests/test_detection.py    CFAR, the burst word, the ring (v0.16.0)
tests/test_phase3.py       pulse runs, the cyclic probes, the analog IDs (v0.17.0)
tests/test_ring_stress.c   the ring under threads; ring_bench.bat runs it
                        as Bill's 40 MSPS bench
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

## What is in v0.17.0 (ABI 18)

**Phase 3 — "small, high value"** of `docs/ATKDSP_PLAN.md` (§B3, §B4, §C3),
and two corrections phase 2 left owed. Header §22–24, twins in
`reference.py` §22–24, 73 tests in `tests/test_phase3.py` plus 22 in
`test_detection.py` and 11 in `test_coderec.py` for the corrections (954 in
all), smoke lines for every kernel. Built and checked here on Linux; the Windows run is
`build.bat /only atkdsp /test` from ATK.

| | what | how |
|---|---|---|
| **pulse runs** | `atkdsp_pulses_*` | envelope → threshold with hysteresis → runs carried across block edges, one PDW per run: sub-sample TOA and PW (the crossings interpolated), PA, peak, energy, the mean instantaneous frequency and 2.77 σ chirp span over the run's core (flanks trimmed 10 %), a least-squares sweep slope. **Word for word `atk.core.pulse.detect_pulses`** for the same floor and `hyst_db = 0` (the oracle is copied into the test); the floor is the caller's (`atkdsp_mag_median`, the same median), because a streaming median would make the answer depend on the chunking. A run beyond `max_samples` is measured on its first `max_samples` and timed from its real edges (TRUNCATED); an open run is closed by `flush` (FLUSHED); an overflowing output buffer drops and counts (`dropped`), never loses state |
| **cyclic line** | `atkdsp_cyclic_*` | a spectral line at each candidate symbol rate after a nonlinearity — \|x\|² (shaped linear modulations), x², x⁴, x⁸ (hard-limited or not; centre the carrier), \|x[n] − x[n−1]\|² (phase steps, shaped FSK), (f[n] − f[n−1])² (frequency transitions: rectangular FSK, MSK). Segments cut on the global index, mean removed, Hann, a Goertzel at the rate; over the last `nseg` segments: **level** (unit mean for white input), **coherence** \|Σp\| / Σ\|p\| (siga's line-versus-hump test: a hump's phasors have random phases), the line's **offset** from the candidate (the segment-to-segment phase) and the aligned statistics. Thresholds derived — level −ln Pfa, coherence √(−ln Pfa / 0.785 K) — then **measured** (below): conservative, and create refuses a K at which the coherence test cannot be met |
| **cyclic prefix** | `atkdsp_cp_probe_*` | per candidate FFT length N: the lag-N correlation over every pair since reset, \|ΣR\| / ΣE, with the Rayleigh threshold √(−ln Pfa / M); per prefix length W: the symbol-rate line of \|z\|, z the lag product summed over the last W samples (it peaks once a symbol when the window sits on the prefix) — the coherence confirms W and the period (LTE's mixed 160/144 prefix: give the slot period) |
| **analog IDs** | `atkdsp_tone_id_*` | one handle per audio channel: **CTCSS** (a 270/330 Hz Kaiser low-pass decimated to ~1 kHz, 100 ms Hann segments, a DFT bank at the 50 tones; the measured frequency from the segment-to-segment phase, to 0.1 Hz; a line-share test and a 3-segment confirmation that voice pitch does not pass; the squelch-tail **reverse burst** flagged from the phase step; the DCS turn-off tone), **DCS** (134.4 Bd, eight timing phases over the last second, every 23-bit alignment against the Golay(23,12) code of generator 0xC75 with the marker bits and the 104-code table; `dcs_max_errors` corrections; the **inverted alias** reported — 023 normal IS 047 inverted, bit for bit, and no decoder can tell them apart), **DTMF** (20 ms frames every 10 ms, DFTs at the 8 tones and their second harmonics; Q.24: 1.5 % frequency from the frame-to-frame phase, 8 dB twist, 60 % energy share, harmonic talk-off, 40 ms on, a 10 ms dropout forgiven), a **tone tracker** (20 ms FFT frames, parabolic peak, ~1 Hz) with the ZVEI / CCIR / EEA / EIA tables and a matcher in `reference.py` (Quick Call II is matched by frequency against an agency's plan). CTCSS and DCS suppress each other's starts: a radio sends one or the other, and a tone's sign pattern at 134.4 Bd is periodic enough to pass Golay |
| **CFAR `rho2`** | `atkdsp_cfar_config.rho2`, `atkdsp_cfar_n_eff`, `atkdsp_cfar_window_rho2` | the window's bin-to-bin power correlation (Hann: 4/9, 1/36), so the solver uses the equivalent independent cell count — CA and GO/SO as gamma shapes, OS through the rate the independent-cell CA formula maps to (ATK `cue_live.py`'s correction, now in the kernel); the band test too. Zeros reproduce v0.16.0 exactly |
| **fixed bits** | `atkdsp_blockid.fixed_bits` | a payload position that never varies (ASCII's high bit) put single-position checks into the null space and refused true codes on short captures — and the exact search's stretch test read a bit constant *within* a stretch as noise (the 1-in-20). Extras are now judged in the affine space (a constant position is allowed, a broken code check is noise), the single-position checks are set aside and reported, and the checks are put back into a named code where that names one (else the largest dmin for a small code) |

Measured (Linux, gcc; `tests/test_phase3.py`, `test_detection.py`, `test_coderec.py`):

| check | result |
|---|---|
| pulse words against `pulse.detect_pulses` on a 400 k-sample radar scene (CW and 2 MHz chirps at 20 dB) | **every word equal**: start / end exact, TOA within a sample, PA within 1e-3 dB, frequency and chirp within 1e-6 fs — on C and on the twin |
| a pulse straddling a block boundary | reported once, identical to 1e-9, at chunk sizes 1, 7, 239–241, 999, 4096, 8192 |
| noise alone, 2 M samples at the default 10 dB / 4 samples | nothing; with `min_samples = 1` the crossings appear at 9.8e-4 per sample, the Rayleigh rate the header derives |
| QPSK / RRC at 4800 Bd, 10 dB | the clock line at 4800 through \|x\|², the DM energy and x⁴; nothing at 2400, 1200, 3600; the rate within 2.5 Hz |
| 4FSK at 4800 Bd | rectangular pulses: FDIFF finds it from 15 dB (and the 9600 harmonic, as any clock line has); shaped pulses: DM finds it from 15 dB — the two transition nonlinearities cover each other |
| noise, and FM with wandering tones | no line; the LINE rate on 60 candidates × 300 trials of noise at Pfa 0.005 each sits inside the binomial band and above a quarter of the mean (the bounds are conservative, as derived) |
| OFDM 64/16 (Wi-Fi-like) at 20 MSPS, 10 dB and **0 dB** | DETECT at lag 64 with score ≈ 0.2 g/(1+g) within 15 %, the prefix and period confirmed (coherence > 0.95); the wrong prefix or period not; lags 32, 128, 256, 2048 nothing; noise nothing; the score's rate on 64 lags of noise is the Rayleigh rate |
| every CTCSS tone under 300–3000 Hz voice, 3 s | **50 of 50** read as itself, 0.2 Hz, held for the transmission; a 105.0 Hz tone reported non-standard at 105.0 |
| the reverse burst (120°, 180°, 240°) | flagged on the END; a plain end not; an 8 s clean tone not |
| every DCS code under voice, at every bit phase | **104 of 104** as itself with its alias; 023 inverted reads 047 / alias 023; a flipped bit per word corrected and flagged at `max_errors` 1, never at 0; the turn-off tone flagged |
| every DTMF key at 50 / 50 ms | all 16, frequencies within 2 Hz; a 6 dB twist and 1 % detune accepted; 12 dB twist, 3.5 % detune, a 25 ms digit and voice at full level refused |
| ZVEI-1 (70 ms), CCIR (100 ms), EEA (40 ms) sequences | every tone within 1.5 Hz and 20 ms, matched to its table and digits |
| voice, noise, silence alone | no ID of any kind |
| C against the twin, uneven blocks | pulses to 1e-9; probes to 1e-7; tone_id event for event at 8, 22.05 and 48 kHz (times to 1e-9, frequencies to 1e-6) |
| CFAR with the Hann `rho2` on Hann-windowed noise, CA / OS / GO / SO at 1e-3 | inside the binomial 99 % interval; the uncorrected threshold fires > 1.25× too often |
| text under Hamming(7,4) at 28 kbit (two codewords per character) | 20 seeds of 20 named "linear (14,8), 1 fixed bit", dmin 3; a 3-bit payload under cyclic Hamming names "Hamming(7,4), 1 fixed bit"; the interleaved harness case decodes back to the text |
| sanitizers | the smoke test and the whole suite clean under ASan + UBSan |

**Honest limits.** CTCSS under a voice whose fundamental is below 300 Hz at
full level (a radio's audio high-pass removes it, a sound-card feed may
not) breaks up and restarts. DCS polarity is not decidable from the air.
The tone tracker reports what it measures; letter tones in the selcall
tables vary by vendor. MDC-1200 and FleetSync are phase 3b: their framing
is reverse-engineered, and real recordings come before trust. The cyclic
line's thresholds are the Rayleigh-walk bound, conservative by a few tens
of percent at K = 16–50 (measured in `src/cyclic.c`).

## What was in v0.16.0 (ABI 17)

**The detection track** — phase 2 of `docs/ATKDSP_PLAN.md` (§B1, §B2, §A3):
the detector cued DF, Bug Sniffer and Drone Discovery wait on, the record it
hands them, and the ring the DVR cuts from. Header §19–21, twins in
`reference.py` §19–21 (the same sums in the same order, so C == twin hit for
hit), 88 tests in `tests/test_detection.py`, smoke lines for all three, and
`tests/test_ring_stress.c` (threads; also Bill's bench, `ring_bench.bat`).
`ATKDSP_E_OVERRUN` (−6) and `ATKDSP_E_EMPTY` (−7) are new. Also in this
release: **CRC-64/NVME** joins the CRC catalogue (it reached the reveng
catalogue after the table was typed; its check value is asserted like the
rest — 112 of reveng's 113; CRC-82/DARC is wider than the 64-bit engine).

| | what | how |
|---|---|---|
| **CFAR** | `atkdsp_cfar_*` | a 1-D CFAR across frequency on each spectrum frame: CA (default), OS (crowded bands), GO / SO (clutter edges); `n_ref` reference cells each side beyond `n_guard` guards (siga's `train` / `guard`); an optional k-frame average; a **mask** (known emitters, operator mutes) that is never tested and never used as reference; at a band edge or past a mask the window **borrows** from the far side so N, and the false-alarm rate, hold there too. **The threshold is solved**, not looked up: Pfa(α) = ∫ Q(K, Kαt) f_T(t) dt for K-look gamma cells (K = looks × k_frames), Simpson in ln t over the range outside which each tail holds < 1e-9·Pfa, Illinois regula falsi in ln α — one solver for all four statistics; `atkdsp_cfar_alpha` exposes it and `describe()` prints α |
| **the tracker** | same handle | hit bins within `merge_gap` form one blob; a blob joins the track whose span it overlaps most; OPEN after m hits in n frames, CLOSE after k misses, UPDATE when the span grows; one event per track per frame (so `events_max` = frames × max_tracks, and a short buffer is refused before anything is consumed); t0 / t1 inside the k-frame average from a single-frame band test (or flagged COARSE); fc as the excess-weighted centroid; level, noise (T / E[T], unbiased), SNR, band SNR; flags EDGE, MASK_ADJ, FLUSHED |
| **burst descriptor word** | `atkdsp_burst_measure` | the comms PDW from channelised I/Q and a coarse [i0, i1): edges by a least-squares step around the burst's core — at the half-power level for a constant envelope, at the likelihood boundary for a Gaussian-like one (flag GAUSSIAN: such a signal fades through half power on its own); 99 % OBW from the noise-subtracted Welch spectrum with the residual left **signed** (clipped residuals add up to more than the 0.5 % tails at 10 dB); −26 dB width (or NaN below the floor); centroid fc; drift from segment centroids; rise / fall 10–90 %; peak / mean / noise dBFS; SNR; flags for whatever could not be measured |
| **the ring** | `atkdsp_ring_*` | in CALLER memory — private, or a named shared mapping another process attaches to; one producer that never waits, up to 16 consumers with their own cursors; a seqlock per block (2 seq + 1 writing, 2 seq + 2 committed), so a torn block is never returned — it is counted and reported lost, exact to the block; claim / commit zero-copy for the producer, read or peek / release for consumers, `copy_range` for the DVR; per-block meta with RETUNE / OVERFLOW / CAL flags; atomics through a shim (gcc/clang `__atomic`, MSVC Interlocked + barriers) — C99, no threads started |

Measured (Linux, gcc; `tests/test_detection.py`):

| check | result |
|---|---|
| α against the K = 1 closed forms (CA, OS, GO, SO) at Pfa 1e-2 … 1e-12 | within 1e-9; CA with K = 2, 4, 8 looks against the incomplete beta within 1e-9 |
| false-alarm rate on noise, CA / OS / GO / SO at **1e-2, 1e-3, 1e-4** | inside the exact binomial 99 % interval, counted at cells far enough apart to be independent; also with 4- and 2.5-look input, with the 4-frame average, and at the band edges (borrowed windows) |
| planted bursts in 50 ms of a 20 MSPS capture, 1024 bins (a 2 ms carrier, a 5 ms FSK burst, a 10 ms 200 kHz noise-like burst, a 1 ms carrier) and a masked repeater | **one detection each**, fc within a bin (19.5 kHz), t0 within a frame (51.2 µs), t1 within two; the repeater none; nothing else — on the twin and on C |
| OS beside a strong neighbour | a 16 dB signal 10 bins from a 40 dB one: CA misses it, OS finds it |
| burst edges, keyed constant envelope (carrier, FSK) at 100 kS/s | **within 2 samples at 10 dB**, 0 at 20–30 dB (§B2's acceptance) |
| burst edges, Gaussian-like 10 kHz burst | within 30 / 12 / 4 samples at 10 / 20 / 30 dB — its own fades set that |
| 99 % OBW of a 10 kHz flat burst | within a bin at 10–30 dB |
| drift of a carrier (50 kHz/s) | within 5 % at 10–30 dB (FSK: ~10 %, the symbol mix moves each segment's centroid) |
| rise time of a 400-sample raised-cosine ramp at 30 dB | within 10 %; a ramp shorter than 2× the smoothing is flagged unresolved |
| ring, C stress (producer + fast / 3×-work / stalled consumers) | no torn block returned in 3 million blocks at full speed (≈ 6 GB/s, 250 k torn reads caught and counted); read + lost = committed for every consumer, exactly; at 160 MB/s (40 MSPS ci16) the fast and slow consumers lose nothing |
| ring across processes | a second Python process attaches to a named mapping and reads the same seqs, bytes and headers |
| sanitizers | the smoke test, the stress program and the whole Python suite (847 tests, after the review's fixes) clean under ASan + UBSan |

**Checked twice.** An independent review of the first build found eleven
defects, each fixed and held by a test (the last section of
`tests/test_detection.py`, and ATK's `test_cue_detector.py` /
`test_iq_ring_native.py`):

- `burst_measure` wrote past its FFT buffer on a burst over ~1.57 M
  samples and silently truncated its drift segments past 524 288; the
  segments are now at most 32768 samples, spread over the burst (twin too).
- The ring binding's `claim()` / `peek()` views pointed at memory that
  `close()` or the garbage collector could free (a segfault); the views now
  hold the buffer or mapping.
- A NaN reference cell: GO / SO dropped it when it was the left side and
  OS's quickselect had no order for it, so C and twin disagreed; any NaN
  now makes the statistic NaN in every mode.
- The incomplete gamma stopped at 1000 iterations, wrong past a shape of
  ~5e4 (α for K = 1e4 looks off by up to a third); the bound now grows with
  √a, and `looks` is capped at 1e4.
- `atkdsp_ring_attach` accepted a header whose sizes were zero and then read
  past the mapping; every size is checked first.
- The MSVC atomics: `_ARM_BARRIER_ISH` for 32-bit ARM, single-access 64-bit
  loads and stores on 32-bit x86, a fence in the ARM64 32-bit load; and
  `burst.c`'s indices are `ptrdiff_t` (Windows `long` is 32 bits). Built here
  with clang for the x64, x86, ARM64 and ARM MSVC targets.
- `burst_measure` with less noise than its L/8 guard set N to 0 and called a
  whole burst truncated; it now falls back to whatever lies outside, and
  truncated means the edge step is at the buffer's end.
- ATK side: TDMA folding of CFAR times found the frame grid itself as a
  "period" (it now takes the grid's resolution and folds no finer than
  twice it); the native DVR ring held a fourteenth of its budget with small
  pushes (it now packs them) and copied cuts under the producer's lock (a
  3 s stall; now outside it); `measure()` overwrote a t0 the I/Q did not
  hold.

What held up, as the reviewer checked it: the α closed forms over the whole
range, C == twin on 1500 random CFAR streams and 3000 fuzzed burst inputs,
no torn block and exact lost counts in its own peek / release and
copy_range stress, no leaks over 900 create / destroy cycles.

Cost: CA / GO / SO 0.10–0.13 ms per 4096-bin frame, **OS 2.2 ms** (a
quickselect per bin — fine for display frames at tens per second, the use
the plan names; a sliding order statistic would be the fix if OS is wanted
on every FFT frame at 40 MSPS); creating a CFAR solves α in milliseconds;
`burst_measure` 0.85 ms for a 20 000-sample buffer (twin 15 ms).

**Not yet:** Bill's 40 MSPS ten-minute ring bench on his PC
(`ring_bench.bat`, after `build.bat`); a real bladeRF capture for the CFAR
acceptance (the synthetic one stands in); the modulation hint in the burst
word (§B4's probes); the toolkit's `dsp/cfar.py` cross-check (that folder is
not connected — `siga.detect.cfar` is checked on the ATK side); the MSVC
build of the ring's atomics is written and reviewed but compiled here only
by gcc.

## What was in v0.15.0 (ABI 16)

**Naming the code** — phase 1b of `docs/ATKDSP_PLAN.md` §D4: given only the
bits, say which code made them. Header §18, twins in `reference.py` §18 (the
same searches in the same order, the same random draws, so C == twin field for
field), 180 tests in `tests/test_coderec.py`, a smoke line per recogniser.
Every answer carries its evidence (Bill, 2026-10-04): the structure found, a
`log10_pfa` with every test counted, and decode → re-encode agreement through
the phase 1a decoders. Beyond the method's reach the answer is "no code
found", with the estimated BER when there is structure to estimate it from.

| | what | how |
|---|---|---|
| **rank scan** | `gf2_rank_scan` | the deficiency heat map, width × offset, exact elimination on packed words, OpenMP across widths |
| **block codes** | `block_code_id` | exact elimination at widths up to 2n (a whole codeword sits at the same place in every 2n-bit row) then the divisors' offsets; in noise the **cyclic-root count** (each irreducible factor of g divides every clean codeword; one comparison per window and factor via x^−t·P_t mod q), then the **robust null space** (random row subsets, each candidate check verified on the other rows, refused if it holds in a row-index pattern — the signature of a wrong width). Named by generator polynomial: Hamming, Golay, BCH (every primitive polynomial), extended, shortened ("shortened BCH(255,239) t=2 as (200,184)"), an RS code's binary image, cyclic / linear (n,k); canonical G, gpoly, dmin; evidence from the §16d table, §16c BCH or §16b RS decoder |
| | — not a block code | a deficiency that repeats under a shift smaller than n: "no block code (linear recurrence)" for an LFSR, a constant or periodic stream, "(sliding, period 2)" for a convolutional stream (or a block code of length P below `n_lo`, which the bits cannot tell apart) |
| | — not a code at all | before anything is named: checks found at m copies of a shorter length are reduced to it; checks that hold as well a few bits along are a recurrence or a convolutional stream; and the failing codewords must fail the way noise makes them — at random rows (no pattern by row index, G-test over periods 2..64) and mostly with single-error syndromes — else "no code found (n=N fails unlike noise)". The cyclic count must imply a clean share of at least (1 − 0.08)^n and 10 % |
| **every recogniser** | stream check | the statistics assume uniform, memoryless bits, so a bias (\|P(1) − ½\| > 0.1) or a short linear recurrence (Berlekamp–Massey over the first 8192 bits) is refused first: "no code found (biased: P(1) = 0.10)", "… (linear recurrence)" |
| **convolutional** | `conv_code_id` | rate from the redundancy density (deficiency growth per window step); generators from c₁·gᵢ = cᵢ·g₁, smallest K over the n alignments; both over **short contiguous stretches** (sliding windows share bits, so a short stretch is clean far more often than random rows, and noise only lowers an exact deficiency); every generator nonzero and each output's equations verified on the whole stream; a multiple of n wins only with a K no larger (a repeated output is not a longer code); catalogue names (NASA/CCSDS, LTE, IS-95, GSM, DAB); hard-decision Viterbi re-encode as evidence; punctured codes reported as their rate |
| **Reed–Solomon** | `rs_code_id` | on aligned codewords: every primitive field polynomial, both bit orders, the CCSDS dual basis; the representation by summed root significance over 64 codewords, then the roots over 1024; consecutive under some α^prim → fcr, prim, k; half a codeword along the roots must vanish ("not codeword-aligned" otherwise: periodic structure); the §16b decoder as evidence; a reversed-bit / reciprocal twin is the same code: more roots, then the smaller fcr, then MSB first |
| **interleavers** | `interleaver_id` | the interleaved block is itself a linear code, so its period comes from the block search; each factorisation of the period, its multiples and its divisors (a block that is not a multiple of the codeword) is undone at each candidate block start, and the one whose inner code shows at the smallest width gives rows × cols; a plain code is not an interleaver (the inner width must be at most half the raw stream's); the smallest equivalent interleaver is reported (3×14 over Hamming(7,4) is 3×7) |

Measured (Linux, gcc, `tests/test_coderec.py` and the header):

| code | named at BER | notes |
|---|---|---|
| Hamming (7,4), extended (8,4), Golay (23,12) (24,12) | 0 … **5e-2** | 5–6 of 6 trials at 5e-2 from 3000 codewords; BER estimate within 25 % |
| Hamming (15,11), extended (16,11) / Hamming (31,26) | 0 … **2e-2** | at 3e-2: 5 of 6 / 1 of 6 |
| BCH(63,45) / BCH(127,99) / BCH(255,239) | 0 … **1e-2 / 5e-3 / 3e-3** | 6 of 6 trials each; the limit is the count of clean codewords among 24 |
| shortened BCH(63,51)→(50,38), (255,239)→(200,184), Hamming(15,11)→(13,9) | 0 … 1e-3 (tested) | §16c decoder with shortening as evidence |
| random linear (12,6) (20,10) (40,20) | 0 … 1e-2 | robust null space; (24,16) at 1.5e-2 says "too noisy" with BER 0.015 |
| K=3…15 rate 1/2, 1/3, 1/4 (catalogue + K=13, 15) | 0 … **1e-2** from 3000 data bits | K=7 r=1/2 at 2e-2 / 3e-2 |
| RS(15,11), RS(63,55) | 0 … 3e-3 | |
| CCSDS RS(255,223) dual basis, DVB RS(204,188), RS(255,239) | 0 … 1e-3 | needs a few hundred codewords at 1e-3 (13 % of 2040-bit words are clean) |
| interleaved Hamming(7,4) 16×7 and 7×16, Golay(24,12) 8×24, Hamming(15,11) 12×15, Golay(23,12) 10×23 | 0 … 1e-3 | ≈ (2N + 34)·2N bits (a codeword is spread over its block, so from an unknown start it shows at 2N); 2–4 s at N ≈ 200 |
| blocks not a multiple of the codeword: Hamming(7,4) 4×5 and 6×10, Golay(24,12) 6×10, Hamming(8,4) 5×7, BCH(63,51) 5×7 | 0 … 1e-3 (BCH: 0) | period = their common multiple; 10–30 s in noise at periods 280–420. Out of reach: 63-bit codewords at periods ≥ 252 in noise |
| random bits | — | nothing named, 2 k – 2 M bits; 40 short streams, no false alarm. Cost: 200 k bits 0.6 s; 2 M bits 0.9 s at n_hi = 256, 5 s at 512 (the robust search's N^4) |

**What the bits cannot say, and how it is said.** A rate-1/4 code also fits
a longer rate-1/2 description (its streams paired, K about doubled); so a
multiple of n wins when its generators are found with a K no larger — and a
code with a repeated output, (23,23), is the same set of sequences as (1,1),
its shortest form. A block code shorter than `n_lo` seen at a multiple of its
length repeats its checks a codeword along, exactly as a convolutional stream
does: both are "no block code (sliding, period P)". A
bit-reversed RS symbol under the reciprocal polynomial is the same code. A
non-cyclic subcode of a cyclic code shows the cyclic structure first; the
robust search on the aligned codewords then adds the extra checks it can
verify. Partial checks of a wrong width hold only on the rows of one phase of
the alignment cycle — the periodicity test exists for exactly that, and
`tests/test_coderec.py` plants the cases (the shortened Golay (20,8), whose
10-bit rows alternate between a codeword's two halves; a random (40,20)
searched from width 30).

**Checked twice.** An independent review of the first build found defects
in all four recognisers — among them a buffer overrun naming an 8-output code, an integer
overflow in `gf2_rank_scan`, multiples of a length named as codes, BCH named by
too small a t, parity-first extended codes called "linear", plain codes taken
for interleavers, structured and biased streams "recognised", and an RS tie
broken by rounding noise; each is fixed and held by a test (the last section
of `tests/test_coderec.py`). Stress sweeps with a truth oracle followed — a
named code must hold on the noise-free stream, a named interleaver must give
whole codewords back: 2880 block runs, 396 convolutional, 172 interleaver. They
found three more: Golay(24,12) searched outside its length named as (46,35) or
(47,36) — 27 false descriptions in 2880 runs, now none (the 10 % clean floor
and the row-pattern test); a noisy interleaver block that is not a multiple of
the codeword started one bit early (every tied alignment is now a start, and
at equal width the larger deficiency wins); and a codeword longer than the
interleaver block never scanned. 0 false descriptions remain in the sweeps;
the misses are the stated data and noise limits.

**Not yet:** soft-decision recognition (everything here is hard bits); RS
alignment search (`rs_code_id` wants the codeword start — from a sync word, or
from `block_code_id` for codes up to 512 bits); generators for punctured
(k > 1) codes; the bench in ATK (plan §D6) is 1c.

## What was in v0.14.0 (ABI 15)

**The bitstream engine** — phase 1a of `docs/ATKDSP_PLAN.md` §D, the part Bill
called "extremely important to me as a signals analyst". Header §14–17, twins
in `reference.py` §14–17, 218 tests in `tests/test_bitstream.py` (C == twin,
planted truth, negative controls), and a smoke line per kernel. Bits go in one
per byte (siga's form) and are packed to 64-bit words inside.

| | what | notes |
|---|---|---|
| **§14 bits** | `bits_autocorr`, `bits_xcorr`, `bits_search` (errors allowed, inverted / reversed), `bits_columns` (raster sums and parities) | popcount on packed words; 1 M bits × 4096 lags in 0.10 s |
| | `bits_width_scan` | the raster's right width, numerically: a z-score per width that is N(0,1) for random bits at EVERY width, so widths compare and the threshold is a normal quantile; the harmonic rule picks the fundamental |
| | `bits_sync_discover` | recurring words nobody listed: period by the commonest gap, phase by the commonest residue, overlapping windows chained into one pattern (up to 128 bits) and trimmed while that makes it more significant; Bonferroni over words × periods × phases; idle and preambles flagged |
| | `bits_stats` | runs, n-grams 1–8, χ² of words, Maurer's universal test and ApEn (NIST SP 800-22; ApEn checked against NIST's worked example), block entropy by width |
| | `bits_wht`, `bits_walsh`, `bits_cyclic` | parity relations between columns; the analyzer-strip transforms (sequency power, dyadic autocorrelation, cyclic time / frequency) |
| **§15 scramblers** | `bm`, `bm_profile` | Berlekamp–Massey on packed words: an L-stage register from 2L bits; `bm_mask` gives siga's mask |
| | `descramble_add`, `descramble_mult`, `scramble_mult` | siga.bits' mask convention, held to ATK's own `lfsr_sequence` by test; state carried, so any block sizes |
| | `scrambler_search` | every trinomial and pentanomial to degree 32 (36,456 candidates, 0.5 s on 80 kbit) as a feed-forward filter — the multiplicative descrambler, and an annihilator of any additive scrambler whatever its seed; ranked by the G statistic, which is χ² for random bits, so the threshold is derived; must beat doing nothing; additive seeds recovered from a constant run and stepped back to bit 0 |
| **§16 FEC** | `Viterbi` / `conv_encode` | K ≤ 15, rate 1/n ≤ 8, any generators, puncturing, terminated / tail-biting / streaming, soft or hard; float32 metrics the twin reproduces bit for bit; tail-biting identical to the PBCH decoder's (test); streaming output identical however it is chunked |
| | `RS` | GF(2^m), m 3–16, any fcr / prim / shortening, erasures; CCSDS dual basis (the matrix's inverse is NASA's published one, row for row) |
| | `BCH` | binary, narrow-sense, any m ≤ 16 and t, shortened; BCH(31,21) is POCSAG's code, held to ATK's pager encoder |
| | `BlockCode` | any small code from its generator (n ≤ 32, n−k ≤ 20) by syndrome table, dmin found; says when a word is equidistant from two codewords instead of guessing; Hamming and Golay generators derived from their polynomials |
| **§17 CRC** | `crc` + catalogue | width 1–64, Rocksoft parameters, bit-granular messages, table path for long byte runs; 111 named sets, every check value tested, checked against the reveng catalogue |
| | `crc_recover` | the unknown CRC from frames: GCD of frame combinations in which init and xorout cancel (equal lengths, three lengths, or an assumed init of 0 / all-ones), a 2-unknown solve for init, and every candidate must reproduce EVERY frame; width, reflection, field byte order, an excluded header and bits after the field are searched |

Measured when it was written (Linux, gcc AVX2, one core unless OpenMP):

| what | result |
|---|---|
| Viterbi K=7 r=1/2 (171, 133), soft, BPSK/AWGN | BER 2.2e-5 at 3.9 dB, **8.3e-6 at 4.2 dB** (union bound from the code's textbook spectrum: 8.6e-6), 3.3e-6 at 4.7 dB — the 1e-5 crossing at ≈ 4.05 dB against the published 4.2–4.4 |
| Viterbi K=9 r=1/3 (557, 663, 711) | 1.9e-5 at 3.0 dB, 3.5e-6 at 3.3 dB; no published table was checked, the test asserts it beats K=7 r=1/2 by 5× at 3.6 dB |
| Viterbi speed | K=7: 5.1 Mbit/s; K=9: 0.76 Mbit/s; K=15: 10 kbit/s (scalar ACS — the AVX2 butterfly is a later increment; K=15 codes are rare) |
| CCSDS RS(255,223), 16 errors | 109 µs per codeword; 17 errors: 200 of 200 refused, none miscorrected |
| BCH(31,21), 3 errors (beyond t) | measured: 20–60 % land on a wrong codeword, the rest are refused — stated, not assumed |
| `crc_recover` | 32-bit CRC with a 64-bit header scan: 0.22 s; width unknown over 10 frames of ≤ 1500 bits: 0.10 s |
| width scan / sync discovery / stats, 1 M bits | 0.12 s (widths 2–2048) / 0.34 s / 0.03 s |

**What frames cannot decide, and how crc_recover says so.** Three things are
algebra, not bugs, and the tests pin each one down:
- *A constant bit is absorbable.* A header that is the same in every frame,
  or a closing flag after the FCS, is folded into init or xorout by an
  equally valid set at another offset.
- *A polynomial that factors is also a CRC of its factors.* XMODEM's
  x^16 + x^12 + x^5 + 1 = (x+1)·q(x), and its frames are exactly CRC-15s of
  q. With few frames a spurious factor can stand in for one of g's own.
- *An (x+1) factor makes init and xorout trade.* With h = g/(x+1),
  h·x^L ≡ h (mod g) for every L, so (init ⊕ h, xorout ⊕ h) is the same CRC
  on every message — `init_known = 0`.

So every offset is scanned and every set that reproduces every frame is
returned, with `alternatives` counting the others of its width, ranked: a
catalogue set first, then simple sets (init and xorout each 0 or all ones),
then `log10_pfa`, then a byte-aligned offset (a prior from real protocols,
said as one), then the smaller offset. `log10_pfa` itself is derived: each
frame is w bits of equations, g, init and xorout cost w each, and the search's
trials are counted — which is why four frames suffice from 16 bits up and
narrower CRCs need more, as many as that arithmetic says.

**Reviewed before release.** An independent review (a second Claude, not the
author, with an AddressSanitizer + UBSan build and a fuzz sweep) found five
defects, all fixed, each with a regression test: the streaming Viterbi's
`out_max` was nllr + n where a call can release nllr + depth − 1 (a short call
after a long one overflowed the caller's buffer and corrupted the ring); an
unchecked size at `viterbi_create`; a missing terminator in `viterbi_describe`
with eight outputs and a 64-column puncture; a shift by 64 (and a hang) for a
degree-63 scrambler mask; and the Viterbi decodes allocating, against rule 2 —
all scratch now comes from `_create`. Also from the review: MSVC builds
without AVX2 count bits in software (`__popcnt64` needs a POPCNT CPU), the
block-code table build is bounded (2^25 patterns; a degenerate `[I | 0]` code
took 74 s), and inputs past the 31-bit limits are refused instead of
truncated. The whole bitstream suite passes under the sanitizers.

**Housekeeping:** `atkdsp_lte_sib1_decode` and `atkdsp_lte_si_decode` are
marked DO NOT USE in the header (six known physical-layer faults; ATK uses its
own `lte_sib1_rx.py`). A comment only — no ABI change.

**Not yet (then):** naming the code (plan §D4) was phase 1b — v0.15.0 above;
the bench in ATK (plan §D6, `BIT_BENCH_PLAN.md`) is 1c. The Windows build of
this release has not been run from here — `build.bat` picks the new files up by its `src\*.c` glob, and
the OpenMP loops follow the MSVC 2.0 rule the FFT already learned.

## What was in v0.13.0 (ABI 14)

**`atkdsp_dmr_sync` — the DMR / P25 sync correlator** (`src/dmr_sync.c`,
header §13, twin `reference.py` §13), the first thing the KrakenSDR DF engine
was waiting on (ATK `docs/KRAKEN_DF_PLAN.md` §6.5, §7.1; `CUED_DF_PLAN.md`).
A complex-baseband matched filter for the 24-symbol sync words — DMR's nine
(BS/MS voice and data, MS reverse channel, the four direct-mode slot words,
ETSI TS 102 361-1 Table 9.2) and the P25 Phase 1 frame sync — with a bank of
carrier-offset hypotheses, run across 1..64 phase-coherent channels at once.
Each hit carries:

- **the per-channel complex amplitude** of that emitter's sync, phase at the
  window centre. `amps / amps[0]` is the array's steering vector for THAT
  emitter: 24 coherent symbols of processing gain, and blind to a co-channel
  emitter that is not sending a sync at that moment. Because the reference is
  the same for every element, any mismatch between it and a real radio is
  common-mode: it costs SNR, never bearing bias.
- **sample-exact timing** (integer window start + a parabolic sub-sample
  refinement) and **the carrier offset** (between bins, ±1 kHz by default),
- **the sync type and its path tag** — BS (a repeater), MS (a handset),
  direct-mode slot 1/2, voice/data — which is what stops a repeater's bearing
  being fused as a handset's (§7.1),
- **`errors`**: how many sync symbols read the wrong frequency sign, DSD's
  own test. `<= 3` is "confirmed": the label can be trusted.

Measured when it was written (synthetic DMR through a 12.5 kHz channel filter;
SNR in the channel):

| what | result |
|---|---|
| 5-element UCA, target −3 dB per element, a 4FSK emitter 3 dB **stronger** on the same channel 40–140° away | pilot bearing median **2.2°** (p90 6.2°); a covariance bearing over the same burst median **~80°** — it finds the stronger one |
| same, no interferer | pilot median 1.2°, p90 3.0° (covariance 0.5°: it integrates the whole burst) |
| found / confirmed at the default thresholds, 150 bursts per row | **one channel**: −3 dB 42 % / 1 %, 0 dB 99 % / 21 %, +3 dB 100 % / 84 %. **Five channels**: −9 dB 75 % / 1 %, −6 dB 99 % / 3 %, −3 dB 100 % / 42 %, 0 dB 100 % / 99 % |
| what "confirmed" protects | the partial matches a voice burst's payload makes read 5+ errors and are never confirmed (test). A FOUND hit's amplitudes are its emitter's steering vector whatever the label; only the label needs confirming |
| noise, 12.5 kHz channel | default thresholds 0.33 / 0.20 / 0.11 for 1 / 2 / 5 channels — about one hit in five minutes, none confirmed (`sync_default_threshold`). Score is averaged over channels, so noise tightens with more of them: that is the five-element gain above |
| cost, 2.1 GHz AVX2 VM, 9 patterns | 0.65 µs/sample/channel at 48 kHz (3 % of a core); 5 channels at 24 kHz, 5 % |

How it is built, and why each piece is the way it is:

- **Offset bank by per-symbol partials.** One partial correlation per symbol,
  then a phasor per symbol per offset: 24 MACs per hypothesis instead of L.
  The loss inside a symbol is 0.15 dB at 500 Hz, 0.6 dB at 1 kHz.
- **Pruning that is exact.** `(Σ_m |p_m|)²` bounds every offset's `|C|²`
  (triangle inequality), so a pattern that cannot reach the threshold at any
  offset skips its bank — in noise, nearly every sample.
- **All patterns at once, one SIMD lane each.** The references are stored
  transposed and the hot loop is written in AVX2 with eight accumulators in
  flight. Left to themselves gcc's vectoriser produced scalar code here
  (measured 4–8x slower); a plain-C fallback is the same arithmetic.
- **Offset first, then timing.** For FSK a residual carrier offset looks like
  a timing error, so the timing parabola is fitted to EXACT scores at the
  refined offset. The other order biased the sub-sample timing by up to 0.45
  of a sample.
- **One refinement-only bin beyond each edge** of the bank, so a burst at
  −950 Hz is not reported at −1000 for lack of a neighbour (a test caught it).
- **Streaming, bit-identical however it is chunked** — tested with blocks of
  1, 7, 239..241, 999, 4096, 8192 against one call.
- **`flush()`** reports a peak whose guard has not passed: the end of a stream,
  or a one-shot burst span cut from the ring (the §6.5 call).

Not yet proved: **a real DMR radio.** The sync words were checked symbol for
symbol against dsd-neo's table (an independent source; a test holds them), and
the pulse model is DMR's root-raised-cosine, but no recording of a real DMR
channel has been through it. ATK's `dmr_sync_check.bat` runs it over a
SigMF recording and reports what it finds — that is the test that settles it.
P25's C4FM is approximated by the same RRC pulse.

## What was in v0.12.0 (ABI 13) and v0.11.0 (ABI 12)

**The FSK front end** (`src/fsk.c`, header §12) for ATK's pager decoders:
`atkdsp_symdump` (integrate-and-dump with a fractional samples-per-symbol),
`atkdsp_sync_search` (a bit pattern over soft symbols, either polarity) in
0.11.0; `atkdsp_dc_track` and `atkdsp_agc` (adaptive baseline removal and
amplitude normalisation for an off-centre or weak channel) in 0.12.0.

## What was in v0.10.3 (ABI 11)

**The DDC keeps a wide channel's edges.** The final rational resampler's
passband was 0.8 of the output Nyquist whatever the channel: a 10 MHz LTE
carrier (±4.5 MHz occupied) brought to 9.6 MSPS kept only ±3.84 MHz. The
passband now widens to the channel when the channel is wider than that,
capped at 0.97 of Nyquist so the filter stays finite; every channel that
already fitted is unchanged. Twin changed with it. This is what lets ATK
read SIB1 of a 10 MHz cell from a 10 MSPS capture and of a 20 MHz cell from
20 MSPS (ATK's `lte_sib1_rx.nfft_for`). No ABI change.

## What was in v0.10.2 (ABI 11)

**SIB1 on air — two fixes here, the receiver in ATK.** The same recording's
SIB1 decodes in ATK's `atk/core/lte_sib1_rx.py`, a downlink receiver written
again from 36.211/212/213/331 (transmit diversity, PHICH, control REGs in
(subcarrier, symbol) order, the right DCI 1A length, period-3 CFI codewords,
PSS/SSS skipped). Two of the faults it found are in this library:

1. **`lte_turbo_decode` ignored the redundancy version** (`rate_dematch` read
   from k0 = 2R always). SIB1 is sent with RV 0, 2, 3, 1 in turn, so three
   of every four transmissions could never pass CRC. k0 = R·(2⌈Ncb/8R⌉·rv+2)
   now (36.212 5.1.4.1.2), as the twin already had.
2. **`lte_sib1_parse` read a SIB1 extension bit** the ASN.1 does not have
   (SystemInformationBlockType1 grows through nonCriticalExtension), so every
   field after the first two bits came out one bit late — garbage PLMN on
   every real cell. The twin had the same bit, in its encoder too, so the
   round trip passed. Also: q-RxLevMinOffset is now skipped when present, and
   SIB-Type values past its extension marker (SIB19+) are read.

On the capture both parsers and ATK's own now read 311-480, TAC 0x7001,
ECI 0x1B6AA03, band 13. `lte_sib1_decode` (the C physical layer) still has
the other faults and is not used by ATK; it will be ported from the Python
receiver once that has met more cells. `lte_si_decode` shares that physical
layer — SIB2+ does not decode on air yet. No ABI change.

## What was in v0.10.1 (ABI 11)

**The LTE MIB decodes on air.** Replaying a bladeRF recording of a real B13
cell (Verizon, 751 MHz) found three conventions wrong in BOTH `lte_pbch.c`
and its twin `reference.py` — so every synthetic test agreed with the decoder
and neither agreed with 36.211:

1. **DC.** The 72 centre subcarriers are −36..−1 and +1..+36; they were read
   as −36..+35 with DC included, putting the upper half one bin off.
2. **CRS ports 2/3 in the PBCH's slot.** 36.211 6.10.1.2 gives v = 3(n_s mod
   2) for port 2 and 3 + 3(n_s mod 2) for port 3; the PBCH is in slot 1, where
   they are the reverse of the even-slot values that were used. Every
   4-antenna cell's two estimates were swapped.
3. **"Spare" MIB bits.** Release 13 gave five of the ten to LTE-M
   (schedulingInfoSIB1-BR-r13) and Release 15 one more; only the last four are
   still spare. Requiring all ten to be zero rejected every LTE-M cell.

And the SSS score (`lte.c`) was the real part only: a carrier 3 kHz off turns
the SSS against its PSS reference by 1.3 rad and a confident, wrong PCI came
out. It is now |Σ| — phase-invariant, 3 dB at zero offset.
No ABI change. ATK's `tests/test_lte_air.py` carries 60 ms of that recording
and fails with each old convention put back.

## What was in v0.10.0 (ABI 11)

Since v0.9.0: `atkdsp_arb_resampler` — an **arbitrary-ratio (fractional)
resampler**. The rational one (§4) needs the ratio to be a ratio of integers
(it does a nominal 2.4576 MHz → 48 kHz fine, as 255/256); this one resamples by
any real ratio, for a clock measured to a fractional hertz (2457603.1 Hz after
calibration), a non-integer output rate, or a live few-ppm sample-clock
correction via `set_ratio`. It is a polyphase bank with first-order Farrow
interpolation between phases, and its output position is tracked from a global
index rather than accumulated per call, so the same sample lands **bit-for-bit
identically however the stream is chunked** — a coalesced 40 MSPS block and a
run of tiny ones give the same line. The **DDC** now routes to it for exactly
the non-integer rates it used to refuse (an integer ratio still takes the exact
rational path), so a channel off an uncalibrated or fractional clock is native
instead of returning an error and dropping ATK to numpy.

## What was in v0.9.0 (ABI 10)

Since v0.8.0: the FFT plan is genuinely safe to share across caller threads
(a per-slot claim replaced `omp_get_thread_num`, which returned 0 for every
thread outside a parallel region, so two callers wrote one scratch buffer);
`atkdsp_spectrum_stats` computes per-bin max/avg/min **and spectral kurtosis**
in one pass over the display's own frames (kurtosis ~2 for noise, ~1 for a
carrier, >2 for a bursty emitter — a per-bin CW/noise/keying call from one
block); and `atkdsp_window_stats` returns a window's coherent gain and ENBW so
a reading can be turned into true dBFS and a bin width in Hz.

`atkdsp_iq_health` reports a front end's DC offset, I/Q gain/phase imbalance,
image-rejection ratio and clip fraction in one pass over a block.

`atkdsp_lte_detect` now finds MULTIPLE co-channel cells by successive
interference cancellation: it detects the strongest cell, subtracts its PSS/SSS,
and re-detects — so a weaker cell sharing the carrier (the fake-tower case) is
found with an uncorrupted SSS instead of being masked. Verified against srsRAN's
downlink (an independent implementation): a strong cell plus one 6 dB down
returns both PCIs, while a lone cell still returns exactly one (no invented
cells). result[0] is unchanged from the old single-cell path.

| group | kernels | replaces in ATK |
|---|---|---|
| unpack | `atkdsp_unpack` (cu8 / ci8 / ci16 / ci16q11 / cf32, running DC block), **`atkdsp_unpack_dc`** — convert, remove a constant offset and return the block's mean, all in one pass | `dsp.iq_to_complex`, `dsp.dc_block` |
| NCO | `atkdsp_nco_*` phase-continuous mixer | `dsp.frequency_shift` (the per-sample `np.exp`) |
| FIR | `atkdsp_fir_*` decimating, carried history | `dsp._decimating_fir` |
| resampler | `atkdsp_resampler_*` polyphase L/M, exact counts; **`atkdsp_arb_resampler_*`** any real ratio (Farrow), chunking-invariant, live `set_ratio` for ppm correction | (nothing — 48 077 Hz was fed to dsd-neo as 48 000) |
| FFT | `atkdsp_fft_*` (thread-safe plans), `atkdsp_window`, **`atkdsp_window_stats`**, `atkdsp_power_db`, `atkdsp_spectrum_reduce` (100 % POI: every frame, max/min/avg), **`atkdsp_spectrum_stats`** (max/avg/min + spectral kurtosis, one pass) | `dsp.spectrum_db` (one frame per chunk, the rest discarded) |
| LTE | `atkdsp_lte_*` cell search (PSS/SSS/PCI), PBCH→MIB, PRACH presence, turbo+CRC, SIB1/SI decode | (new; see the header's §11 and the roadmap caveats) |
| demod | `atkdsp_fm_demod`, `atkdsp_am_demod`, state carried | `dsp.fm_demodulate`, `dsp.am_demodulate` |
| display | `atkdsp_db_to_pixels`, `atkdsp_decimate_max` | pyqtgraph's per-frame LUT pass, `spectrum_view.decimate_max` |
| detect | `atkdsp_median`, `atkdsp_detect_channels`, `atkdsp_stitch_max` | the per-bin Python loops in `signal_id.detect_channels` and `sweep.stitch` |
| design | `atkdsp_design_lowpass` — Kaiser windowed sinc to a stated stopband | `dsp.design_lowpass` (65 taps at the input rate, whatever the rate) |
| DDC | `atkdsp_ddc_*` — NCO → staged decimation (each stage designed for `atten_db` of alias rejection at ITS rate) → exact-rate polyphase resampler, or the **arbitrary** resampler for a non-integer clock. `describe()` prints the plan, e.g. `10 MSPS: NCO -> /8 (31 taps) -> /2 (9) -> /13 (303) -> 48076.9 Hz -> x624/625 -> 48000 Hz`, or `... -> arb -> 48000 Hz` for a fractional rate | `dsp.channelize` — which passed a tone 100 kHz off the channel at −2.9 dB (10 MSPS) / −0.2 dB (40 MSPS); the DDC puts it 80 dB down, and the test asserts 60 |

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
8. Correlators: sync-word search (**DMR / P25 done, 0.13.0**), LFSR
   descrambler search (**done, 0.14.0** — and the rest of the bitstream
   engine, docs/ATKDSP_PLAN.md §D phase 1a), Gardner / Mueller-Müller timing.
9. Multi-channel: spatial covariance, cross-ambiguity (TDOA/FDOA), MUSIC —
   for a two-channel bladeRF 2.0 and the KrakenSDR.
10. Cellular survey: PSS/SSS/SSB **correlators** belong here; the broadcast
    channel decoding (MIB/SIB) does not — see the note in ATK's plan.

## Licence

Same terms as ATK. `vendor/pocketfft` is BSD-3 (its `LICENSE.md` is kept
beside it).
