# atkdsp — the build plan (2026-10-27)

> Bill, 2026-10-27: *"Go ahead and build 2, 3, 4, 5, 6, 7 … and build 9.
> Keep the C ring buffers, because more will be in C in the future. And the
> rest of your lists …"* — and on the bitstream section: *"Please, this
> section is extremely important to me as a signals analyst."*

**Status (2026-10-28): phases 1a and 1b are BUILT.** 1a — atkdsp v0.14.0 /
ABI 15 (§D1, §D2, §D3, §D5 and the SIB1 housekeeping; README "What was in
v0.14.0"). 1b — atkdsp v0.15.0 / ABI 16 (§D4, naming the code; README "What
is in v0.15.0"). Linux: smoke test and the whole suite pass (361 cross-check +
218 bitstream + 180 code-recognition tests), also under ASan + UBSan. The
Windows build was first run 2026-10-29, from ATK's new top-level `build.bat`
(which builds every native module beside ATK in place): v0.16.0 / ABI 17 builds
with MSVC 19.44, loads through the binding, and `atk.core.dsp_native` takes it.
Bill's 40 MSPS ring bench (`ring_bench.bat`, 600 s at 160 MB/s, 1 MiB x 512) passed
the same day: 91,553 blocks committed, 0 torn reads, the fast and slow (3x work)
consumers lost nothing, the stalled one kept the ring's 512 and lost the rest with
exact counts. The Windows pytest run passed the same evening: 848 passed in 131 s
against the MSVC 19.44 / AVX2 / OpenMP build, through ATK's `build.bat /only atkdsp /test`. Everything else below is still plan; next
in §G is 1c (the bench in ATK, §D6).

**1c is BUILT (2026-10-28)** — in ATK, not here: the Bit Bench workspace
(`ATK\atk\core\bitbench\`, `ATK\atk\ui\bitbench_view.py`), specified by
`ATK\docs\BIT_BENCH_PLAN.md` (which replaces §D6) and recorded in its §10. No
atkdsp source changed for it; ATK now expects ABI 16 (v0.15.0). Its
acceptance — the Scoped B harness, raw bits to payload by buttons alone —
passes on the native library and on the numpy twins. One weakness it found
here (fixed in phase 3, ABI 18): **a structured payload inside a block code** (text under
Hamming(7,4) behind an interleaver) gives `block_code_id` / `interleaver_id`
extra single-position checks — the payload's fixed bits — and they then
refuse the stream as "sliding" or "fails unlike noise" on short captures
(often at 28 kbit; 1 seed in 20 at 42 kbit). The proposed fix is in §D4:
take weight-1 checks out of the null space (report them as fixed bits)
before the sliding and noise-likeness tests.

**Phase 2 is BUILT (2026-10-28)** — atkdsp v0.16.0 / ABI 17 (§B1 CFAR and
the burst tracker, §B2's burst descriptor word, §A3 the ring; README "What is
in v0.16.0"). Linux: smoke test, the ring stress program and the whole suite
(848 tests) pass, also under ASan + UBSan; an independent review's eleven
defects are fixed and held by tests (README, "Checked twice"). Choices made in
the build that differ from the text below: `atkdsp_cfar_create` takes a
config struct (thirteen parameters, not nine — `looks`, the OS rank,
merge gap, min width and the track cap had to be somewhere);
`atkdsp_burst_measure` takes an `nfft` (0 = automatic) and its `thresh_db` 0
picks the edge rule by envelope (half power for a constant envelope, the
likelihood boundary for a Gaussian-like one — measured, see the header);
the ring adds `close`, `peek` / `release` and `describe`, rejects an empty
block (so `read` returning 0 always means "nothing new"), and its counters
struct is `atkdsp_ring_counters` (the name `atkdsp_ring_stats` is the
function). The TDMA timing analyzer is ATK-side Python, as the plan says
(`ATK\atk\core\tdma.py`). Still open: a real bladeRF capture for the CFAR acceptance, and the
toolkit's `dsp/cfar.py` cross-check (that folder is not connected).

**Cross-check done 2026-10-28 (late), and what it found.** The ATK
Diffusion Toolkit's CFAR replaces N with the window's EQUIVALENT number of
independent reference cells (a Hann window's power correlation is 4/9 one
bin apart, 1/36 two apart, so 2 × 16 cells carry ~17 cells' evidence);
§19's CFAR assumes N independent cells, so on Hann-windowed frames its
stated Pfa is exceeded — measured 1.6× at 1e-5 and 1.7× at 1e-6 (6.1×10⁷
noise cells through ATK's live cue frames). ATK corrects for it without an
ABI change (`atk/core/cue_live.py`: the threshold the requested rate needs
with N_eff cells, from the F distribution for K-look cells, is mapped to
the rate that gives the same threshold with N; GO/SO use one side's
N_eff); measured within ±8 % in all four modes, and the two CFARs set the
same threshold bin for bin (ATK `tests/test_cfar_cross_check.py`).
**For the next ABI bump** (LMR_DECODE_PLAN will bump it too — one bump,
coordinated): put the window's correlation in `atkdsp_cfar_config` (a ρ²
table or `n_eff`), so every caller gets the right rate, and the edge-
refinement test (CFAR_EDGE_PFA) likewise — **done in ABI 18 (phase 3)**. Also measured: the per-cell cost
is ~25 ns at k_frames = 1 (dominated by the dB → linear conversion) —
taking linear power would roughly halve it.

**Phase 3 is BUILT (2026-10-29)** — atkdsp v0.17.0 / ABI 18 (§B3 pulse
runs, §B4 both cyclic probes, §C3 the analog IDs except MDC-1200 and
FleetSync; README "What is in v0.17.0"). The two things phase 2 left owed
ride with it, as the cross-check note asked, in the same bump: the window's
ρ² table is in `atkdsp_cfar_config` (`rho2[4]`, with `atkdsp_cfar_n_eff`
and `atkdsp_cfar_window_rho2` to fill it; the band test uses it too), and
`block_code_id` sets aside a payload's fixed bits (`atkdsp_blockid.fixed_bits`),
so the 1c weakness — text under Hamming(7,4) behind an interleaver refused
at 28 kbit — is fixed: 20 seeds of 20 name "linear (14,8), 1 fixed bit",
dmin 3, and the interleaved harness case decodes back to the text. Linux:
smoke test and the whole suite (954 tests: 848 + 73 + 22 + 11) pass, also
under ASan + UBSan. The Windows run is owed (`build.bat /only atkdsp /test`
from ATK), and ATK's `dsp_native.py` now expects ABI 18. Phase 3b (MDC-1200
and FleetSync) waits for real recordings, as §C3 says. Next in §G is phase 4
(the engine) or LMR_DECODE_PLAN's L1/L3, whichever Bill opens first — L1/L3
need the TIA/ETSI documents, which are not connected here.

**What building 3 changed in this plan** (the README has the detail):
- §B3's `atkdsp_pulse_runs` is `atkdsp_pulses_*`, a handle with the
  streaming state (rule 3), and it is word for word
  `atk.core.pulse.detect_pulses` — the same median floor, 10 dB threshold,
  10 % flank trim, 2.77 σ chirp span — so ATK's PDW ring takes it without a
  change of meaning. The floor is the caller's (`atkdsp_mag_median`): a
  running median inside the kernel would make the words depend on the
  chunking. `hyst_db` is extra (0 = detect_pulses' behaviour). The 30 MSPS
  cost is still to be recorded on Bill's PC.
- §B4's `cyclic_line` is `atkdsp_cyclic_*` with five nonlinearities, not
  three: |x|², x^m (m = 2, 4, 8, hard-limited or not), and two transition
  measures the plan's "delay-multiply" turned out to need — DM is
  |x[n] − x[n−1]|² (the plan's x[n]·x*[n−1] has no line for FSK, as the
  tests showed), and FDIFF is (f[n] − f[n−1])² for rectangular FSK and MSK,
  whose DM line is weak. Any clock line has harmonics; a rectangular OOK
  envelope has no |x|² line at all (constant between transitions), DM finds
  it. The noise-only thresholds are derived (level −ln Pfa, coherence
  √(−ln Pfa / 0.785 K)) and measured conservative; the coherence test needs
  K ≥ 16 segments at 1e-3, and create refuses less.
- §B4's `cp_probe` finds the prefix by the symbol-rate line of the
  windowed lag product (the coherence test again), not from peak spacing;
  LTE's mixed prefix is given as the slot period. A CW tone present in the
  band correlates at every lag, so the lag score alone is not specific —
  the prefix stage is the confirmation.
- §C3: CTCSS runs on a 270/330 Hz Kaiser low-pass (not 300 — voice at 250
  Hz was a false tone), 100 ms Hann segments with the frequency from the
  segment-to-segment phase (0.1 Hz), a line-share test and a 3-segment
  confirmation; the reverse burst is a phase step that reads as a frequency
  jump within the tone's lobe. DCS polarity is undecidable from the air (023
  normal is bit for bit 047 inverted), so every event carries its alias.
  CTCSS and DCS suppress each other's starts. DTMF is 20 ms frames every
  10 ms with a 3-frame confirmation (25 ms frames lost 50 ms digits). The
  selcall tables and matcher live in `reference.py` (Python-side matching,
  as the plan said: tones and durations first, tables second). MDC-1200 /
  FleetSync are phase 3b.
- The CFAR cross-check's request is honoured in the kernel (`rho2`), so
  `cue_live.py`'s Python-side correction can now be a config field (ATK
  side, with `git status` checked first).

Written against atkdsp v0.13.0 / ABI 14. The recommended order is §G; the
order is Bill's call.

**What building 1b changed in this plan** (the README has the detail):
- The robust (noisy-stream) search is three methods, not one: a cyclic-root
  count for cyclic codes (one comparison per window and irreducible factor —
  BCH(255,239) at 1e-3 from 24 codewords, where random row subsets cannot
  work), random row subsets for other block codes (each candidate check
  verified on the rows not drawn, and refused if it holds in a row-index
  pattern), and short contiguous stretches for convolutional codes.
- `block_code_id` also names shortened cyclic codes and an RS code's binary
  image, and says "no block code (linear recurrence)" / "(sliding, period
  P)" when the structure repeats under a shift smaller than the length.
- `rs_code_id` needs codeword alignment (a sync word, or `block_code_id` for
  short RS codes); punctured convolutional codes are reported as their rate,
  without generators.
- The ABI signatures take `n_lo/n_hi`, `pfa` and `seed` (the robust search's
  generator), and the result structs carry `robust` (how it was found) and
  the evidence fields.
- Naming a code needs refusals as much as detections: every recogniser first
  refuses streams its statistics do not fit (biased bits, a short linear
  recurrence), and `block_code_id` refuses structure whose failures do not
  look like channel noise (patterned by row, or with arbitrary syndromes).
  An independent review and truth-oracle stress sweeps (a named code must
  hold on the noise-free stream) drove these; the last section of
  `tests/test_coderec.py` holds each finding.

**What building 1a changed in this plan** (the README has the detail):
- `crc_recover` returns EVERY parameter set that reproduces the frames, not
  the first. Constant header/trailer bits, a factoring polynomial and an
  (x+1) factor make some ties undecidable from frames alone (algebra, pinned
  by tests); the struct gained `alternatives` and takes a `pfa`.
- The Viterbi create takes `max_bits` (decision memory allocated once, rule 2)
  and the decoders gained `*_describe` / `*_stats` (rule 5).
- Scrambler masks use siga.bits' convention for both kinds, held to ATK's own
  `lfsr_sequence` by test; the search tries odd-weight polynomials only (3 or
  5 terms), as every primitive polynomial has.

**There are two plans in this folder.** `docs/DSP_EXPANSION_PLAN.md` was
written minutes earlier by another Claude instance, from the same request.
§H lists where the two differ, so Bill can make one of them the plan of
record. This file does not edit or replace that one.

**How to read it.** Each item says *what* it is (with a short theory note
where the theory is the point), a sketch of its ABI, *who in ATK uses it*,
its *acceptance* test, and its honest limits. "ATK side" lines are the
follow-on wiring in `D:\Analyst_Toolkit\ATK`, done as a separate step after
the kernel lands (several Claude instances work on ATK; check `git status`
before touching it).

---

## Contents

- §0 Scope — what is in, what is parked
- §1 Rules every item follows
- §A The engine — float32 FFT, the fused mixer, C ring buffers, the channelizer
- §B Detection and measurement — CFAR, burst descriptors and TDMA timing,
  live pulse runs, cyclic probes, occupancy and baseline, DSSS
- §C Demodulation and signal quality — the linear demod chain and
  equalizer, EVM/MER, analog radio IDs, strong-signal removal, transmitter
  fingerprint features
- §D The bitstream toolbox — the checklist, the C engine, the ATK bench
- §E Longley-Rice (ITM)
- §F Intercept audio — re-demodulation, the classical cleanup chain, how
  the Forensics Workshop and the diffusion toolkit sit on top
- §G Order, sizes and releases
- §H Where this plan differs from `DSP_EXPANSION_PLAN.md`

---

## §0 Scope

**In this plan** (Bill's list of 2026-10-27, mapped to the README roadmap
and the earlier review):

| Part | Items | Asked as |
|---|---|---|
| A | float32 FFT, mixer folded into the first decimating stage, C ring buffers, channelizer | roadmap #2, #4, #3 ("keep"), #1 |
| B | CFAR, burst descriptor words + TDMA timing, `pulse_runs`, `cp_probe` + `cyclic_line`, occupancy + baseline, DSSS detection + chip rate | review #3, new list, review #6, review #5, new list, new list |
| C | linear demod chain + adaptive equalizer, EVM/MER, analog radio IDs, strong-signal removal, transmitter fingerprint features | new list |
| D | the bitstream toolbox: Viterbi, RS, BCH and the rest of the traditional set | review #7 + "anything traditional in Martes, xbit, X-Midas" |
| E | NTIA Longley-Rice (ITM) | review #4 (KRAKEN_DF_PLAN §9.5) |
| F | intercept audio cleanup (diffusion tools may be used; **SAM Audio is not**) | new list |

**Parked — still on the roadmap, not in this round:**

- **Testing `dmr_sync` against a real DMR radio.** This is a bench step,
  not a build: `dmr_sync_check.bat` over a 400–450 MHz SigMF recording. It
  stays first on Bill's bench list, because atkdf's pilot-aided bearings rest
  on it.
- **The NR SSB C port**, and the polar and LDPC decoders NR's MIB and SIB1
  need.
- **TDOA/FDOA cross-ambiguity and MUSIC for DF.** `siga.geoloc` has them in
  numpy, and atkpr keeps its own processor for distribution control. (MVDR
  for *listening* is §F4.)
- **Audio forensics beyond intercept cleanup** — ENF timestamping, splice
  detection and the like. These belong in the Forensics Workshop's own plan,
  not in atkdsp.

**Code that already exists and that this plan builds on rather than
duplicates.** The numpy versions below become the specification the C
kernel is cross-checked against, and their callers switch to the DLL when
it is present:
- the diffusion toolkit's `dsp/cfar.py`, and `siga/detect.cfar` (§B1);
- the toolkit's `fingerprint/features.py` (§C5);
- the toolkit's `repair/speech.py`, `repair/audio_inpaint.py` and
  `repair/wer.py` (§F2);
- `siga/baud.py` (§C1);
- `siga/bits.py` and `siga/bitchain.py` (§D).

**Housekeeping on the first release:** mark `atkdsp_lte_sib1_decode` and
`atkdsp_lte_si_decode` "DO NOT USE" in the header. They still carry six
known physical-layer faults, and ATK uses its own `lte_sib1_rx.py` instead.
The change is a comment only, with no ABI change.

---

## §1 Rules every item follows

The seven header rules (README, "The rules, in one place") hold. On top of
them:

1. **The twin comes first.** The numpy twin in `python/atkdsp/reference.py`
   is written before the C. The cross-check asserts C == twin, fed in blocks
   of 1, 7, 239–241, 999, 4096 and 8192. There are two declared exceptions,
   each with its reason where it occurs: ITM (§E1) and the ring's
   cross-process mode (§A3).
2. **Known answers before real signals.** Every acceptance test has a
   synthetic case with planted truth (the `siga/synth.py` rule) and, where
   the hardware exists, a real-signal case.
3. **Negative controls.** Every detector and search has a test where noise
   or random bits must return "nothing" at the design false-alarm rate. A
   search over N candidates divides its Pfa by N, because a search over
   thousands of candidates always finds one that looks good.
4. **Thresholds are derived, not tuned.** The formula behind each threshold
   sits in the header comment beside its parameter. If a number was
   calibrated, the comment carries the measurement.
5. **Logging (Bill, 2026-10-10).** C may not print (rule 4), so every
   stateful kernel gets two functions. `*_describe()` writes its plan as a
   sentence, the way `atkdsp_ddc_describe` already does. `*_stats()` fills
   a counters struct: samples in, outputs, detections, overruns, resets. The
   binding logs `describe()` at INFO on create, `stats()` at DEBUG every N
   calls, and every non-zero return code at WARNING with the call's
   arguments. ATK's `dsp_native` shows `describe()` on Setup.
6. **Passive, receive-only, no decryption.** Nothing here transmits. The
   bitstream tools *recognise* encryption or compression (flat statistics,
   full linear complexity, no rank deficiency) and say so in those words.
   They never attempt key recovery — the same line as go2key and DroneID.
7. **Licences.**
   - Own code, written from standards and papers.
   - Permissive code may be vendored, pinned, with its licence kept beside
     it (the pocketfft precedent).
   - LGPL/MPL code may serve as a test oracle only.
   - GPL/AGPL code is never read into ours. Ideas may come from published
     descriptions only.
   - SAM Audio appears nowhere in this plan (§F).
8. **Build scripts.** A new source file goes into `CMakeLists.txt`, the
   `cl.exe` fallback list in `build.bat`, and `build.sh`, all in the same
   commit. The smoke test gains a line per kernel.
9. **ABI.** Each release bumps `ATKDSP_ABI_VERSION` once (adding a function
   counts, header rule 5) and records the bump in the header comment block.
   The README gets a "What is in vX" section in the same commit.

---

## §A The engine

### A1. A float32 FFT (roadmap #2)

**What.** Power-of-two transforms move to a float32 SIMD FFT behind the
existing `atkdsp_fft_*` ABI. pocketfft (double) stays for other sizes and
for any caller that asks for double. The gain comes from two places: half
the memory traffic, and twice the lanes per SIMD instruction.

**The number to beat** is the 977-frame display reduction at 40 MSPS. It
costs 0.183 CPU-s/s today (README), where 1.00 means one core exactly
saturated.

**How.** Vendor PFFFT (C, SIMD, a BSD-style FFTPACK licence; the maintained
fork with AVX), pinned to a commit with its licence beside it, rather than
writing a radix-4 kernel from scratch. The licence text is read before the
commit. `atkdsp_fft_create` picks the backend.

```c
atkdsp_fft *atkdsp_fft_create_ex(size_t n, unsigned flags);  /* ATKDSP_FFT_DOUBLE forces pocketfft */
int         atkdsp_fft_backend(const atkdsp_fft *p);         /* 0 pocketfft f64, 1 pffft f32 */
```

**Acceptance.**
- At every size from 64 to 65536, the RMS error against pocketfft is at
  most 2e-6 of the signal RMS (float32 rounding over log2 N stages).
- The NaN semantics of the stitch and display kernels are unchanged (the
  existing tests).
- `spectrum_reduce` at 40 MSPS is measured before and after on Bill's PC,
  and the gain is recorded whatever it turns out to be.

**Uses.** Everything that transforms: the display, `spectrum_stats`, the
channelizer (A4), CFAR (B1), excision (C4), audio noise reduction (F2).

### A2. The mixer folded into the first decimating stage (roadmap #4)

**Theory.** Mixing and then filtering gives the same result as filtering
with taps rotated to the channel frequency, followed by one rotation at the
*low* rate:

    y[m] = e^(−jω·mD) · Σ_k (h_k·e^(jωk)) · x[mD − k]

That removes the per-input-sample complex multiply: the NCO pass that reads
16 MB and writes 16 MB per block at 40 MSPS. The price is that the stage's
taps become complex.

**Expected cost:** +0.04 for the complex taps, −0.093 for the NCO that
disappears, so about −0.05 CPU-s/s per channel at 40 MSPS. This is the
README's estimate from the 2026-09 measurements.

**How.** Inside `atkdsp_ddc_*`; callers see nothing new.
- `describe()` says "fused NCO".
- `atkdsp_ddc_create_ex(…, ATKDSP_DDC_NO_FUSE)` keeps the old path for the
  A/B test.
- The low-rate rotation phase is computed from the global output index (the
  arbitrary resampler's technique), so the output is bit-identical however
  the stream is chunked.
- A retune recomputes the taps.

**Acceptance.**
- Output matches the unfused DDC within 1e-5 relative at 2.4, 10 and
  40 MSPS.
- The 80 dB and 60 dB interferer tests still pass.
- Per-channel cost is measured and recorded.

### A3. C ring buffers (roadmap #3 — kept: "more will be in C in the future")

**What.** The §RF-F2 block ring and the §RF-F9 DVR, as a C structure that
lives in *caller* memory. That one choice is what makes it future-proof:
- The same ring can be a private buffer inside ATK, or a named
  shared-memory mapping that a C program in another process reads (atkpr's
  processor, the next native consumer).
- ATK's Python ring (`iq_cut.IqRing`) keeps working, and the C ring takes
  its place behind the same interface when the DLL is present.

**Design.**
- **One producer, up to 16 consumers**, each with its own cursor.
- **The producer never waits** — an SDR cannot. A slow consumer gets
  overrun and is told exactly how many blocks it lost.
- **Fixed-size blocks, each with a header:** `seq` (u64), first-sample
  index (u64, global), `t_utc_ns`, `fc_hz`, `fs_hz`, `gain_db`, `format`,
  and `flags` (device overflow, **retune boundary**, cal state). A DVR cut
  never spans a retune without saying so.
- **A seqlock per block.** `seq` is odd while the block is being written; a
  reader copies the block and re-checks `seq`. A torn read is retried or
  counted, never returned.
- **Zero-copy both ways.** The producer uses `claim`/`commit`, so the SDR
  reader writes straight into the ring. A consumer can `peek`/`release` with
  validation, or use a copying `read`.
- **`copy_range(seq0, seq1)`** cuts a past span for the DVR. This is what
  sits behind the waterfall's *Export pre-D I/Q* and *DF the selection*.
- **64-bit indices; size bounded only by RAM.** 40 MSPS of ci16 is
  160 MB/s, so a 30 s DVR takes 4.8 GB. The Setup page states the RAM before
  it allocates.
- **Atomics through a small internal shim** (Interlocked* on MSVC,
  `__atomic` on gcc/clang). The code stays C99, and the library still starts
  no threads (rule 4).

```c
typedef struct { uint64_t seq, first_sample; int64_t t_utc_ns; double fc_hz, fs_hz;
                 float gain_db; int32_t format; uint32_t flags, nbytes; } atkdsp_ring_meta;
size_t       atkdsp_ring_bytes(size_t block_bytes, size_t nblocks, int max_consumers);
atkdsp_ring *atkdsp_ring_init(void *mem, size_t mem_bytes, size_t block_bytes,
                              size_t nblocks, int max_consumers);
atkdsp_ring *atkdsp_ring_attach(void *mem, size_t mem_bytes);        /* another process */
void        *atkdsp_ring_claim(atkdsp_ring *r);                        /* producer, zero-copy */
int          atkdsp_ring_commit(atkdsp_ring *r, const atkdsp_ring_meta *m);
int          atkdsp_ring_open(atkdsp_ring *r, int from_oldest);        /* -> consumer id */
ptrdiff_t    atkdsp_ring_read(atkdsp_ring *r, int id, void *out, size_t cap,
                              atkdsp_ring_meta *m, uint64_t *lost);
ptrdiff_t    atkdsp_ring_copy_range(const atkdsp_ring *r, uint64_t seq0, uint64_t seq1,
                                    void *out, size_t cap, atkdsp_ring_meta *metas, size_t mcap);
int          atkdsp_ring_stats(const atkdsp_ring *r, atkdsp_ring_stats *s);
```

**Twin.** A pure-Python ring with the same semantics, single-process.
Cross-process mode is native-only — a declared exception, because shared
memory between processes has no numpy meaning.

**Acceptance.**
- A C stress test: a producer at full speed and three consumers (one at
  100 %, one at 50 %, one stalled). No torn block is ever returned, and the
  lost counts are exact to the block.
- From Python, a second process attaches to a named mapping and reads the
  same `seq`s.
- On Bill's PC: 40 MSPS ci16 sustained for ten minutes with three
  consumers, with the stats recorded.

**ATK side.** `iq_cut.IqRing` gains a native backend; the DVR settings page
shows RAM.

### A4. The channelizer (roadmap #1)

Two tools, because there are two jobs.

**(a) Overlap-save extractor — arbitrary channels from ONE forward FFT per
block.**
- Each channel takes the bins around its centre, multiplies them by its
  filter's frequency response, and inverse-transforms a small FFT. The small
  inverse transform *is* the decimation.
- A final exact-rate stage (the existing rational or arbitrary resampler)
  lands each channel on 48 kHz, or whatever its decoder wants.
- The FFT is sized from the narrowest channel, not from the display. At
  40 MSPS a 2048-point display has 19.5 kHz bins, and a 12.5 kHz channel does
  not span one (README).
- Channels are added, removed or retuned between blocks without disturbing
  the others.
- Phase continuity across blocks comes from a per-block bin-rotation
  correction.

**(b) Polyphase filterbank (PFB) — a uniform grid, all at once.**
- Example: every 12.5 kHz across 20 MHz is 1,600 channels.
- 2× oversampled, so a signal on a channel edge is not lost.
- For "DF everything", Bug Sniffer's sweep-and-cue, per-channel energy and
  occupancy (B5), and survey.

```c
atkdsp_osx *atkdsp_osx_create(double fs, size_t nfft_hint, double atten_db);
int         atkdsp_osx_add(atkdsp_osx *x, double offset_hz, double bw_hz, double out_rate);
int         atkdsp_osx_retune(atkdsp_osx *x, int ch, double offset_hz);
int         atkdsp_osx_remove(atkdsp_osx *x, int ch);
ptrdiff_t   atkdsp_osx_process(atkdsp_osx *x, const atkdsp_cf32 *in, size_t n,
                               atkdsp_cf32 *const *outs, size_t *out_counts, size_t cap_each);
atkdsp_pfb *atkdsp_pfb_create(double fs, unsigned nchan, unsigned oversample, double atten_db);
ptrdiff_t   atkdsp_pfb_process(atkdsp_pfb *p, const atkdsp_cf32 *in, size_t n,
                               atkdsp_cf32 *out /* [nchan][m] */, size_t cap);
```

**Acceptance.**
- Each extracted channel matches a DDC of the same channel within 1e-4
  relative (after aligning group delay), with at least 80 dB of stopband.
- **Cost target:** 50 channels at 40 MSPS cost at most 2× what one DDC
  channel costs today. Measured and recorded.
- Chunk invariance holds.
- Adding or removing a channel mid-stream leaves the other channels'
  output bit-identical.
- PFB: a tone on a channel edge appears in both neighbours at the designed
  level, and nothing leaks beyond `atten_db`.

**Uses.** The fifty-entry watchlist, Bug Sniffer, cued DF's parking, the
analog-ID decoders on many channels at once (C3), and the DSD feeds.

---

## §B Detection and measurement

### B1. CFAR — the detector cued DF and Bug Sniffer wait on

**Theory.** A fixed threshold is wrong the moment the noise floor moves.
CFAR (constant false-alarm rate) sets each cell's threshold from its
neighbours instead. For a square-law cell (one periodogram frame, which is
exponentially distributed) averaged over N reference cells,
`α = N·(Pfa^(−1/N) − 1)` gives exactly the false-alarm rate asked for.
Averaging K frames makes the noise gamma-distributed. The multiplier is then
solved numerically at create time, and `describe()` prints it.

Three variants, for three situations:
- **CA-CFAR** (cell averaging) is the default.
- **OS-CFAR** (k-th order statistic) is for crowded bands, where a
  neighbouring signal would raise CA's threshold and mask a weak one.
- **GO/SO** (greatest-of / smallest-of) is for clutter edges.

**What.**
- A 1-D CFAR across frequency on each frame, using the display's own frames
  or the PFB's channel energies.
- Guard cells, a false-alarm rate set per profile, and an input mask
  (known emitters, operator mutes).
- **A burst tracker across frames.** Adjacent hit bins merge into one
  detection with `fc`, `bw` and level. A detection opens after m hits in n
  frames and closes after k misses. So every burst gets a `t0` and a `t1`,
  and one burst is one record however the blocks fall.

```c
atkdsp_cfar *atkdsp_cfar_create(size_t nbins, int mode /* CA|OS|GO|SO */, int n_ref,
                                int n_guard, double pfa, int k_frames,
                                int open_m, int open_n, int close_k);
int       atkdsp_cfar_set_mask(atkdsp_cfar *c, const uint8_t *mask);
ptrdiff_t atkdsp_cfar_process(atkdsp_cfar *c, const float *frames_db, size_t nframes,
                              double frame_t0, double frame_dt,
                              atkdsp_cfar_event *ev, size_t cap); /* OPEN/UPDATE/CLOSE */
```

**Acceptance.**
- CUED_DF_PLAN §5.1's test: planted bursts in a real 20 MHz bladeRF
  capture. Each burst gives one detection, with `fc` within a bin and `t0`
  within a frame. A masked repeater gives none.
- On noise only, the measured false-alarm rate matches the design rate
  within its binomial 99 % interval, at three different rates. This is the
  test that proves the threshold is derived.

**One CFAR, not three.** Two numpy CFARs already exist, and both use the
same derived α: `siga/detect.cfar` on the bench, and the diffusion toolkit's
`dsp/cfar.py` (its energy proposer, the baseline its learned detectors are
scored against). The C kernel's twin must agree with both, and the toolkit's
energy proposer calls atkdsp when the DLL is present.

**ATK side.** `atk/core/cue_detector.py` (CUED_DF_PLAN). The twin is that
module's fallback engine. Also Bug Sniffer's detector and Drone Discovery.

### B2. Burst descriptor words (the comms PDW) and TDMA timing

**What.** Radar has the PDW; comms needs the same thing — one record per
burst:
- `t0` and `t1`, refined in the time domain on the channelized I/Q (a
  threshold crossing with sub-sample interpolation), so well inside a frame.
- Duration.
- `fc`.
- Occupied bandwidth (99 % and −26 dB).
- Peak and mean power in true dBFS (via `window_stats`).
- SNR.
- Rise and fall times.
- Frequency drift across the burst.
- A modulation hint from B4's probes.
- An analog ID, when C3 decodes one.

The C part is the tracker (B1), the time-domain refine and the per-burst
measurement. The records belong to ATK.

**TDMA timing (ATK, Python — a few records a second).**
- Per channel, burst start times are folded against candidate frame
  periods. The comb scoring takes the fundamental, not a harmonic (siga's
  rule).
- The result is the frame period, slots, slot occupancy and guard time.
- It is matched against known timings: DMR's 60 ms frame of two 30 ms
  slots, TETRA's four slots of ~14.17 ms, P25 Phase 2's two slots. The match
  is a *hint*, never an identification; dsd-neo still decides.
- Pattern of life: bursts per channel per hour, kept in the same store as
  B5.

```c
int atkdsp_burst_measure(const atkdsp_cf32 *iq, size_t n, double fs,
                         size_t i0_coarse, size_t i1_coarse, float thresh_db,
                         atkdsp_bdw *out);
```

**Acceptance.**
- Synthetic bursts with planted `t0`, duration, `fc` and drift at 10–30 dB
  SNR: `t0` within 2 samples at the channel rate, bandwidth within a bin,
  drift within 5 %.
- On Bill's DMR test band, the TDMA analyzer recovers the 30 ms slot
  structure of a real repeater.

**Uses.**
- Bug Sniffer: store-and-forward bugs sleep and then burst.
- Drone Discovery: telemetry bursts.
- Cued DF: a burst is the unit DF works on.
- C5: the fingerprint features take the cut.

### B3. `atkdsp_pulse_runs` — live radar PDWs

**What.** Envelope → threshold with hysteresis → run lengths plus per-run
sums. The run is carried across block edges, so a pulse split by a boundary
is one pulse. Per pulse:
- TOA (sub-sample)
- PW and PA
- Intra-pulse frequency, from the phase slope Σ x[n]·x*[n−1]
- A chirp-slope fit when the pulse is long enough

It feeds `pulse.py`'s PDW ring and deinterleaver (FUTURE_PLANS, live-pulse
section A).

**Acceptance.**
- The PDWs equal `pulse.detect_pulses`' on the same capture, with TOA
  within a sample.
- A pulse straddling a block boundary is reported once at every chunk size.
- The cost at 30 MSPS is recorded.

### B4. Cyclic probes — `atkdsp_cp_probe` and `atkdsp_cyclic_line`

**Theory.**
- **OFDM** copies the end of each symbol into its cyclic prefix. So x[n]
  and x[n + N] are correlated at N = the FFT length — a lag correlation that
  noise and single-carrier signals do not have.
- **A digital signal**, squared or delay-multiplied, has a spectral line at
  its symbol rate. Analog FM and noise do not.

**`cp_probe`.**
- Lag correlation at a list of candidate lags, one per numerology: LTE/NR
  at 15 and 30 kHz subcarrier spacing, Wi-Fi at 312.5 kHz, DVB-T 2k and 8k.
- Integrated with chunk-invariant state. The output is |R|/energy (so gain
  does not move it) plus the integration time.
- A second stage finds the CP length from the spacing of the correlation
  peaks, N + Ncp.

**`cyclic_line`.**
- m-th power (m = 2, 4, 8) and delay-multiply, then a Goertzel at each
  candidate rate: 4800 (DMR, P25, NXDN96), 2400 (NXDN48), 512/1200/2400
  (POCSAG), 18 000 (TETRA), and so on.
- The cost is per candidate, not per span.
- siga's line-versus-hump test, done natively: one Goertzel phasor per
  chunk, and the coherence |Σ p| / Σ|p| (1 for a line, about 1/√K for noise
  or a hump).

**Acceptance.** FUTURE_PLANS' bench-first plan:
- Bill's B13 recordings with noise added until each probe stops working.
- The threshold comes from the noise-only distribution at that integration
  time.
- Noise and analog FM produce nothing.

**Uses.** `channel_gate`'s symbol-rate hint; `spectrum_view` captions
("OFDM · 15 kHz", "digital · 4800 sym/s"); the BDW modulation hint (B2); the
DSSS chip-rate search (B6).

### B5. Long-term occupancy and baseline comparison

**What.** Learn what a place's spectrum normally looks like, then flag what
is new — or what has gone.

- **Per bin, per time-of-day slot** (24 slots by default, so a normal daily
  rhythm does not raise alarms), the kernel keeps:
  - a count;
  - a Welford mean and variance;
  - the maximum;
  - the duty cycle above a threshold;
  - a P² quantile sketch (p10, p50, p90, p99).

  That is tiny per bin; no big histograms.
- **NEW:** a bin exceeds its slot's p99 plus a margin for k frames.
- **GONE:** a bin that is normally busy more than 90 % of the time drops
  below its p10 for k frames.
- Flagged bins merge into regions, and each region becomes a change event.
- **State export/import** as a versioned byte blob. ATK writes it under
  `rf_data`, never to C:\ or AppData.

```c
atkdsp_occ *atkdsp_occ_create(size_t nbins, int nslots, float busy_db, float margin_db, int k);
int       atkdsp_occ_learn(atkdsp_occ *o, const float *line_db, int slot);
ptrdiff_t atkdsp_occ_compare(atkdsp_occ *o, const float *line_db, int slot,
                             atkdsp_occ_event *ev, size_t cap);   /* NEW / GONE regions */
ptrdiff_t atkdsp_occ_export(const atkdsp_occ *o, void *buf, size_t cap);
int       atkdsp_occ_import(atkdsp_occ *o, const void *buf, size_t n);
int       atkdsp_occ_duty(const atkdsp_occ *o, int slot, float *out);
```

**Acceptance.**
- A synthetic 24-hour run, with a daily-varying floor and planted emitters:
  - a planted new emitter is flagged within k frames;
  - a removed carrier is flagged GONE;
  - the daily change alone flags nothing beyond the design rate.
- Export → import → compare is bit-identical.
- `describe()` states the memory used.

**Uses.** Bug Sniffer's park-for-weeks mode (it is the report-back
trigger); survey pattern of life; an occupancy view (duty cycle per
channel); site selection.

### B6. DSSS — detection, chip rate, and the code itself

**Theory — the part that makes it interesting.**
- Spreading multiplies each data symbol by a fast pseudo-random chip
  sequence. The total power stays the same, but it is spread over the
  chip-rate bandwidth instead of the symbol-rate bandwidth. So the power
  spectral density drops by the **processing gain**, G = Rc / Rs.
  - Example: 1 kbit/s spread at 1.023 Mchip/s gives G ≈ 1023, or 30 dB.
- **GPS L1 C/A** arrives at about −128.5 dBm, against about −111 dBm of
  thermal noise in its 2 MHz — roughly **18 dB under the floor**. A receiver
  that knows the code gets the 30 dB back by correlating over each 1 ms code
  period.
- **A blind receiver does not know the code.** But noise has no memory, and
  a short-code DSSS signal repeats its code every period. So the signal's
  autocorrelation has structure at the code period even while the signal
  sits below the floor. That is the handle everything below uses.

**What, in four steps.**
1. **Detect.** Burel's fluctuation-of-correlation detector: cut the signal
   into M windows, estimate the autocorrelation in each, and average
   |R_m(τ)|². Noise gives a flat floor at τ ≠ 0; a DSSS signal gives peaks
   at multiples of the code period. The threshold is derived from the noise
   distribution of that statistic. With enough windows this works well below
   0 dB SNR.
2. **Chip rate.** Chip transitions leave a cyclostationary line at the chip
   rate. Delay-multiply at about half a chip (or square the envelope after
   band-limiting), then run a fine scan using B4's machinery. Squaring also
   gives the carrier offset — a line at 2× the offset for BPSK spreading, 4×
   for QPSK — and that tells the two apart.
3. **Code period and length.** The period comes from step 1's peak spacing,
   and L = period × Rc. An m-sequence gives 2^n − 1: 31, 63, 127, 511,
   1023, and so on.
4. **The code (short codes).**
   - Take the principal eigenvector(s) of the covariance of code-period-long
     windows (Bouder, Azou and Burel). Misaligned windows give two
     eigenvectors, and those locate where the code starts.
   - The sign of the eigenvector at the chip instants *is* the chip
     sequence.
   - Then run Berlekamp–Massey (D2) on the chips. An m-sequence returns its
     LFSR; a Gold code returns linear complexity 2n. Either way the
     generator(s) are named.
   - Despread, and hand the narrowband symbols to the linear demod chain
     (C1).

**Honest limits.**
- Long codes (GPS P(Y), CDMA long codes, anything whose period spans many
  symbols) are *detected* — steps 1 and 2 still work — but not recovered,
  and the result says so.
- Frequency hopping is not DSSS. Hops show up in B1/B2 instead.

```c
int       atkdsp_dsss_fluct(const atkdsp_cf32 *iq, size_t n, size_t win, size_t max_lag, float *rho);
ptrdiff_t atkdsp_dsss_chiprate(const atkdsp_cf32 *iq, size_t n, double fs,
                               double rc_lo, double rc_hi, atkdsp_line *peaks, size_t cap);
int       atkdsp_dsss_code(const atkdsp_cf32 *iq, size_t n, double fs, double rc,
                           double period_samples, int8_t *chips, size_t cap, float *quality);
```

**Acceptance.**
- Synthetic BPSK and QPSK DSSS (L = 31, 127, 1023) at −5, −10 and −15 dB
  in-band SNR:
  - detection at the design Pfa;
  - chip rate within 0.1 %;
  - the period exact;
  - the code recovered (up to sign) for short codes;
  - Berlekamp–Massey names the m-sequence's polynomial.
- Noise, and a strong narrowband carrier, produce nothing.
- **Real signal: GPS L1 C/A**, off Bill's HackRF with antenna power on and
  an active patch antenna:
  - the 1.023 Mchip/s chip rate and the 1 ms period found blind (expected);
  - recovering one satellite's PRN from the eigenvector is a stretch goal,
    because several satellites overlap at different Dopplers. The codes are
    published (IS-GPS-200), so the answer can be checked.

**Uses.** A DSSS instrument on the Signals bench; Bug Sniffer
(spread-spectrum bugs hide under the floor); Drone Discovery; the
fingerprint features (C5 — chip-rate error is a clock fingerprint).

---

## §C Demodulation and signal quality

### C1. The linear demodulation chain, in C

`siga/baud.py` already has Gardner, Mueller–Müller, Costas and a PLL on the
bench. This is the live, streaming version — with the equalizer the bench
lacks. Each stage works alone, and `atkdsp_lindemod` chains them:

- **Matched filter.** RRC, RC, rectangular or Gaussian taps, on the
  existing FIR.
- **Timing recovery.**
  - Gardner: 2 samples per symbol, needs no data decisions.
  - Mueller–Müller: 1 sample per symbol, decision-directed.
  - Oerder–Meyr: feedforward, per block.

  All use a cubic Farrow interpolator and a second-order PI loop, with BnT
  and ζ as parameters. The standard loop-filter design formulas go in the
  header.
- **Carrier recovery.** A frequency-lock loop pulls in large offsets. Then
  either a Costas loop (BPSK/QPSK/8PSK) or a decision-directed PLL (QAM), or
  feedforward M-th power per block (Viterbi–Viterbi).
- **Equalizer.**
  - A fractionally spaced (T/2) FIR.
  - CMA (Godard, p = 2) runs blind until the error settles, then switches to
    decision-directed LMS/NLMS.
  - Optional decision-feedback taps (DFE) for deep notches; an MCMA option
    for QAM, which also corrects phase.
  - **The taps are readable.** A converged equalizer is the inverse of the
    channel, so ATK can draw the multipath delay profile — an analyst's view
    of the channel itself.
- **Soft demapper.** BPSK, QPSK, 8PSK and 16/32/64-QAM (Gray), producing
  max-log LLRs scaled by the measured noise variance. That is the input D3's
  Viterbi wants, so the chain stays soft from the antenna to the decoded
  bits. Soft decisions are worth about 2 dB over hard.

**Theory — CMA.** Godard's insight: a PSK constellation's ring radius is
known even when the data is not (and QAM's rings are). So penalising
|y|² − R² trains the equalizer with no training sequence at all. Once the
eye is open, decisions are reliable enough to switch to LMS, which converges
tighter.

```c
atkdsp_lindemod *atkdsp_lindemod_create(const atkdsp_lindemod_cfg *cfg);
ptrdiff_t atkdsp_lindemod_process(atkdsp_lindemod *d, const atkdsp_cf32 *in, size_t n,
                                  atkdsp_cf32 *sym, float *llr, size_t cap);
int       atkdsp_lindemod_taps(const atkdsp_lindemod *d, atkdsp_cf32 *taps, size_t cap);
/* each stage also stands alone: atkdsp_timing_*, atkdsp_carrier_*, atkdsp_eq_*, atkdsp_demap */
```

**Acceptance.**
- Synthetic BPSK, QPSK, 8PSK and 16QAM through RRC, a three-path channel,
  a 100 ppm clock offset and a 2 kHz CFO:
  - lock within a recorded number of symbols;
  - after equalization, MER within 1 dB of the AWGN-only MER.
- Loop noise bandwidth matches the design BnT within 10 %.
- Chunk invariance holds.
- Results agree with `siga/baud.py`'s estimators on the same synthetic set.

### C2. EVM / MER and modulation quality

**What.** The vector-signal-analyzer metric set, computed on the equalized
symbols:
- RMS and peak EVM, in %, normalized to the reference constellation's
  average power (stated).
- MER, in dB.
- Magnitude error and phase error.
- I/Q origin offset (carrier leakage).
- The *transmitter's* quadrature skew and gain imbalance. These are
  identifiable here only because carrier recovery has locked onto the
  constellation, whose axes are the transmitter's. Without that lock (FM, for
  instance) only the rotation-invariant part is measurable — the image
  rejection, from circularity — which is what the toolkit's
  `fingerprint/features.py` reports.
- Frequency error (Hz and ppm) and symbol-clock error (ppm).
- An EVM-versus-time trace, which shows bursts, fading and PA droop.
- A constellation density map for the display.

**Theory.** MER = 10·log10(Σ|ideal|² / Σ|error|²). For a clean transmitter
on a good channel, MER ≈ SNR. So the gap between MER and the measured SNR is
the impairment the transmitter or the channel added.

**Acceptance.**
- Synthetic constellations with planted I/Q gain and phase imbalance, origin
  offset and clock offset: each impairment recovered within its stated
  tolerance.
- MER within 0.3 dB of the set SNR from 10 to 30 dB.
- Results agree with `siga/modrec`'s constellation statistics.

**Uses.** A constellation + EVM panel on the Signals bench; C5; and a plain
health readout for operators ("this transmitter is sick").

### C3. Analog radio IDs

**Why.** The relief requirement tags DMR and P25 bearings by radio ID.
Analog FM handsets carry identifiers too, in the clear: sub-audible tones
and codes, and data bursts at key-up. So analog bearings can be tagged the
same way.

All of these decoders run on FM-demodulated audio, one instance per
channel, fed by the channelizer (A4):

- **CTCSS** — the 50 standard tones from 67.0 to 254.1 Hz.
  - Low-pass below 300 Hz, decimate, then a fine Goertzel bank.
  - Integration of about 0.3–1 s: the closest pair is 2.3 Hz apart, at the
    bottom of the table.
  - It reports the *measured* tone frequency and level, not just the
    nearest standard tone. A radio's tone error is a fingerprint feature
    (C5), and the tone is a constant-level pilot for multi-receiver
    combining (FORENSICS_PLAN §12.4c).
  - The reverse burst (the phase flip in the squelch tail) marks the end of
    a transmission.
- **DCS** — 134.4 bit/s NRZ, carrying a repeated 23-bit Golay codeword
  (9 code bits + the fixed "100" + 11 parity bits).
  - Integrate-and-dump at 134.4 Bd with a timing search, slide a 23-bit
    window, and check the Golay syndrome.
  - Golay corrects up to 3 errors; the decoder accepts at most 1 by default
    to keep false IDs down (a parameter).
  - Match against the standard code table, normal and inverted.
  - **The trap, handled explicitly:** Golay is a cyclic code, so a shift of
    one DCS code can be a valid codeword of another. The fixed marker and
    the table resolve it, and the tests plant every standard code and require
    each to decode as itself.
- **DTMF** — 8 Goertzels plus their second harmonics (talk-off rejection),
  with ITU-T Q.24 timing, twist and frequency tolerance (operate within
  ±1.5 %, reject beyond ±3.5 %).
- **5-tone selcall** (ZVEI 1/2/3, CCIR, EEA, EIA tables) and **two-tone
  sequential paging** (Motorola Quick Call II, common for US fire and EMS).
  The decoder measures tones and durations first and matches tables second,
  so an unlisted tone plan still reports its frequencies.
- **MDC-1200** (Motorola) and **FleetSync I/II** (Kenwood) — 1200/2400 bit/s
  FFSK data bursts at key-up and de-key, carrying unit IDs. Built on
  `fsk.c`'s symdump and sync search.
  - **Honest:** their framing is mostly known from reverse engineering, not
    from a published standard. Acceptance requires real recordings
    (sigidwiki's audio samples, plus whatever Bill receives) before the
    output is trusted.
  - An open decoder may be used as a test oracle only where its licence
    permits.

**Output:** one event type —
`{kind, value ("CTCSS 127.3", "DCS 023N", "MDC 1A2B", "FS 100-2001", "DTMF 4*7"),
t0, t1, confidence, measured}`.

**Acceptance.**
- Synthetic tones and codes at every standard value, at 6–20 dB audio SNR
  with voice on top: the correct ID every time, and no cross-ID.
- Noise alone, and voice alone, produce nothing at the design rate.
- Real: local repeaters' published tones and codes (RepeaterBook) decoded
  off the air.

**Uses.**
- DF roster tagging in atkdf: analog handsets become per-ID tracks, just
  like DMR ones.
- "This radio stopped checking in" alerts.
- The multi-receiver combine pilot.
- Bug Sniffer: a bug on a business-band channel carrying a tone.

### C4. Strong-signal removal

Four tools, in order from least to most invasive:

1. **Adaptive line enhancer / notch** (Widrow). An LMS predictor learns the
   predictable narrowband part — carriers, whistles, CW — and subtracts it;
   what is left is everything broadband. It follows a drifting carrier, and
   several can be cascaded for several tones.
2. **FFT excision.** A windowed overlap-add (√Hann WOLA): bins above a CFAR
   threshold are attenuated to the local floor. This is the classic fix for
   narrowband interference ahead of a DSSS despreader. It reports the
   fraction of the band it removed.
3. **Pulse blanker.** In the time domain, samples where |x| exceeds k × the
   running median are blanked, with guard samples on either side (ignition,
   radar, switching noise). It reports the fraction blanked.
4. **Reference canceller.** NLMS/RLS subtraction of an adaptively filtered
   reference. The reference can be a second antenna or receiver — or a
   *regenerated* copy of a strong constant-envelope signal: x/|x| captures
   the dominant FM carrier, and the canceller fits its amplitude, phase and
   delay and subtracts it, revealing a weaker co-channel emitter underneath.
   It works best when the interferer is at least 10 dB above the target.

**Rule** (the Forensics Workshop's removed-sound stem): every tool returns
*both* outputs — what is left, and what was removed — so an analyst can
check that the weak signal did not go out with the strong one.

**Acceptance.**
- A weak FSK signal 25 dB under an FM carrier 10 kHz away decodes after the
  canceller, and does not before.
- A CW tone 30 dB over a DSSS signal: despread SNR after excision is within
  1 dB of the clean case.
- Pulse interference: BER restored within a stated margin.
- For every tool, the two outputs sum back to the input within float
  tolerance.

**Uses.** Bug Sniffer (a bug hiding beside a broadcast carrier); the Signals
bench; DSSS (B6); cued DF (a repeater swamping a handset).

### C5. Transmitter fingerprint features

**What.** A feature vector per burst, each field with its own validity bit.

**It starts from what exists.** The diffusion toolkit's
`fingerprint/features.py` already measures the classical set in numpy:
`cfo_ppm`, transmitter image rejection (`tx_irr_db`), LO leakage
(`tx_dc_db`), rise time, overshoot, key-up offset, phase noise, ACPR and
symbol-clock ppm. It also holds the receiver's own fingerprint constant per
receiver profile. On those shared fields, that module is the twin and the
tests assert agreement; the C kernel is its fast path, so every burst can be
measured live. C5 adds the fields marked *(new)* below.

- **Turn-on transient:**
  - the start point, found by change-point detection on amplitude and
    phase;
  - the transient's duration;
  - the amplitude rise profile;
  - the instantaneous-frequency profile *(new)* — PLL settling, highly
    characteristic for analog FM radios.

  Profiles are resampled to a fixed length. Turn-off is measured the same
  way *(new)*.
- **Steady state:**
  - carrier offset (Hz and ppm), and drift across the burst in Hz/s
    *(drift new)*;
  - FM deviation (peak and RMS) and its ± asymmetry *(new)*;
  - symbol-clock error in ppm (digital);
  - transmitter image rejection and origin offset; the full gain/phase split
    for linear modulations, from C2 *(new)*;
  - occupied bandwidth and the spectral regrowth shoulders;
  - phase noise, from an unmodulated stretch;
  - CTCSS tone error, from C3 *(new)*;
  - power ramp-up and ramp-down times.

**ATK side.**
- Per-unit profiles and the comparison (Mahalanobis distance). A probability
  is shown only once it has been calibrated — the Forensics Workshop's
  voiceprint rule.
- Results are filed as *leads*, never identifications.
- The features and the transient cuts also feed the RFMLS training tab.
- The diffusion toolkit's C2 (denoise-before-fingerprint) gets measured
  against this classical set.

**Honest: this is research-grade.**
- It is good at telling a handful of known radios apart at high SNR (about
  20 dB or more), received through the same receiver.
- The receiver adds its own imprint; temperature and battery move CFO; and
  identical-model radios are the hard case.
- Every comparison states how many bursts, and what SNR, it rests on.

**Acceptance.**
- Synthetic radios with planted CFO, drift, imbalance and transient shapes:
  every feature recovered within its tolerance.
- Real: two of Bill's same-model radios separated, or not — the result is
  recorded either way.

---

## §D The bitstream toolbox

Bill: *"extremely important to me as a signals analyst."*

**What already exists in ATK:**
- `siga/bits.py`: slicing, clock recovery, Manchester, differential
  decoding, frame sync with derived thresholds, LFSR search, bit
  statistics.
- `siga/bitchain.py`: xbit's t/s/r/i pattern language, the raster, and the
  take/skip pattern search.

The scope so far is FUTURE_PLANS "Scoped A/B" plus the go2signals notes.
**Bill decided on 2026-10-04 that recognition NAMES the code.** This part is
the whole toolbox: the C engine for everything heavy, and the analyst bench
in ATK on top of it.

### D0. The checklist — what the traditional tools do, and where it lands

**Sources:**
- xbit, as Bill described it.
- BitView's function list (FUTURE_PLANS).
- PROCITEC go2ANALYSE's specs (FUTURE_PLANS).
- R&S CA250's public feature list: entropy tests, column/line sums and
  parity, Boolean operations, mux/demux, file-type identification, scripting.

No public feature list for **Martes** was found; whatever it had that is
missing here, Bill adds. **X-Midas:** ATK already reads and writes its BLUE
files; this toolbox works on the bits that come out of them.

| Capability | ATK today | Lands in |
|---|---|---|
| **Data** | | |
| Import packed / one-bit-per-byte / ASCII 0-1 / hex / BLUE / soft symbols | partial | ATK |
| M-ary symbol streams; symbol→bit **mapping search** (all 24 dibit maps for 4FSK, scored by sync hits) | — | ATK + C scoring |
| Soft bits (LLRs) carried through the chain | — | C1 → D3 |
| **Display** | | |
| Bit raster at an adjustable width | ✅ `bitchain.raster` | ATK: colour tags, column strip |
| Column and line sums / parity under the raster | — | D1 + ATK |
| Width scan (the right raster width found numerically) | — | D1 |
| Frame stack aligned on sync; hex / ASCII / octal panes; x/- L/H ./1 styles; difference of two streams | — | ATK |
| Linear-complexity plot; rank-deficiency heat map (width × offset) | — | D2/D4 + ATK |
| **Manipulation** | | |
| Take / skip / reverse / invert patterns | ✅ | ATK |
| Insert / delete, shift, rotate, mirror, polarity | partial | ATK |
| Boolean ops between streams; XOR with a known sequence; mux / demux n-way | partial | ATK |
| **Line codes and framing** | | |
| NRZ-L/M/S, NRZI, Bi-Phase-L/M/S, differential Manchester | Manchester, differential | ATK |
| Async deframing (start/stop, 5–8 data bits, parity, 1/1.5/2 stop) | — | ATK |
| HDLC flag search + bit de-stuffing, AX.25 | — | ATK (C if measured slow) |
| Known-sync table hits | partial | ATK + D1 search |
| **Sync discovery** (a recurring word nobody listed) | — | D1 |
| **Structure and statistics** | | |
| Autocorrelation and cross-correlation | numpy | D1 (popcount) |
| Pattern search with allowed errors, both polarities | partial | D1 |
| Run lengths, n-grams, chi-square, Maurer universal, approximate / Tsallis entropy, entropy by width | partial | D1 |
| Linear complexity profile (Berlekamp–Massey) | — | D2 |
| Walsh–Hadamard: parity relations between columns | — | D1 |
| **Field discovery**: constant fields, counters, parity relations, length fields, checksum candidates | — | D1 + D5 + ATK |
| **Scrambling** | | |
| Additive descramble and search | ✅ (brute force over degrees) | D2: faster, more degrees |
| Multiplicative (self-synchronizing) descramble and search | — | D2 |
| LFSR from known plaintext (Berlekamp–Massey) | — | D2 |
| Scrambler catalogue (V.35, IESS-308, DVB, 802.11, CCSDS …) | partial | ATK |
| **Decode** | | |
| Viterbi: K ≤ 15, rate 1/n (n ≤ 8), any polynomials, puncturing, soft/hard, tail-biting / terminated / streaming | LTE-only, Python | D3 |
| Reed–Solomon over GF(2^m), m ≤ 16: erasures, shortened, CCSDS dual basis | — | D3 |
| BCH, binary, any n = 2^m − 1 and t, shortened | POCSAG only | D3 |
| Golay, Hamming, QR(16,7) and any small linear code by syndrome table | — | D3 |
| CRC engine: width ≤ 64, any Rocksoft parameter set, bit-granular | per protocol | D3 |
| De-interleave: block, convolutional, helical | — | ATK (index maps) |
| Turbo (LTE) | ✅ `lte_turbo` | — |
| **Recognize — name the code** | | |
| Block code (n, k) by GF(2) rank deficiency over widths and offsets | — | D4 |
| Convolutional code: rate, K, generator polynomials | — | D4 |
| RS: field, n, k, first root | — | D4 |
| BCH / cyclic codes: generator via the GCD of codewords | — | D4 |
| Interleaver dimensions | — | D4 |
| CRC polynomial, init, xorout, reflection, coverage | — | D5 |
| Decode → re-encode agreement as the evidence | — | D3/D4 + ATK |
| **Payload** | | |
| Alphabets: ITA-2 (LTRS/FIGS), ITA-3, ITA-5 / ASCII-7/8, CCIR-476 / SITOR, Varicode, EBCDIC, Cyrillic third shift, Morse, user tables; MSB/LSB, normal/inverse | — | ATK |
| File type from magic numbers at every bit alignment × bit order × polarity | — | ATK (the Forensics Workshop's signature table, if importable) |
| Verdict "encrypted or compressed — no structure to recover" | — | ATK (stated, never attacked) |
| **Workflow** | | |
| Saved, replayable chain of (operation, parameters); editing a step re-runs everything after it | started (`_bit_steps`) | ATK |
| Python scripting hook | — | ATK |
| Bit dossier + LLM adviser (FUTURE_PLANS Scoped B) | scoped | ATK |

### D1. Bit primitives (C)

**Representation.** Callers pass one bit per byte (uint8 0/1 — the numpy
form siga already uses). Kernels pack to 64-bit words internally, so every
correlation is a popcount: 64 bit-comparisons per instruction.

```c
int       atkdsp_bits_autocorr(const uint8_t *b, size_t n, size_t max_lag, float *out);
int       atkdsp_bits_xcorr(const uint8_t *a, size_t na, const uint8_t *b, size_t nb, float *out);
ptrdiff_t atkdsp_bits_search(const uint8_t *b, size_t n, const uint8_t *pat, size_t plen,
                             int max_err, unsigned flags /* BOTH_POLARITY|REVERSED */,
                             atkdsp_bithit *hits, size_t cap);
int       atkdsp_bits_columns(const uint8_t *b, size_t n, size_t width, size_t offset,
                              uint32_t *col_ones, uint32_t *row_ones,
                              uint8_t *col_parity, uint8_t *row_parity);
int       atkdsp_bits_width_scan(const uint8_t *b, size_t n, size_t w_lo, size_t w_hi, float *score);
ptrdiff_t atkdsp_bits_sync_discover(const uint8_t *b, size_t n, int word_len, int min_count,
                                    atkdsp_synccand *out, size_t cap);
int       atkdsp_bits_stats(const uint8_t *b, size_t n, atkdsp_bitstats *out);
int       atkdsp_bits_wht(const uint8_t *b, size_t n, size_t width, const uint32_t *cols,
                          int ncols, int32_t *spectrum);
int       atkdsp_bits_walsh(const uint8_t *b, size_t n, size_t size, int navg,
                            float *sequency_power, float *dyadic_acf);
int       atkdsp_bits_cyclic(const uint8_t *b, size_t n, size_t size, int navg,
                             float *cyclic_time, float *cyclic_freq);
```

What each one does:
- **`width_scan`** scores every raster width by column consistency, the
  mean of |2p − 1| over the columns. A frame length shows as a peak; the
  harmonic rule applies. This finds the raster's "right width" numerically
  instead of by squinting.
- **`sync_discover`** hashes every `word_len`-bit window and keeps the words
  that recur at a fixed period. It finds a sync word nobody put in a table.
- **`bits_stats`** returns:
  - ones density;
  - run-length histograms, for 0s and for 1s;
  - n-gram counts (1 to 8 bits);
  - chi-square;
  - Maurer's universal statistic (NIST SP 800-22 parameters);
  - approximate entropy;
  - block entropy by width.
- **`wht`** — a fast Walsh–Hadamard transform over chosen columns. A linear
  relation between columns (a parity bit, a checksum bit) shows up as a
  large coefficient.
- **`walsh`** and **`cyclic`** — the analyzer-strip types Bill knows from
  the bitstream tools he used (*sequency power*, *dyadic Walsh*, *cyclic
  time*, *cyclic frequency*), over blocks of 256 to 32768 bits with
  averaging:
  - `walsh` gives the Walsh spectrum of the ±1 stream (power by sequency)
    and the dyadic (XOR-shift) autocorrelation. Both are natural for binary
    data, where XOR is the native operation.
  - `cyclic` gives the ordinary bit autocorrelation over time, and its
    spectrum.

  These feed the Bit Bench's analyzer strip (ATK `docs\BIT_BENCH_PLAN.md`).

**Acceptance.**
- Every function against its twin.
- Synthetic frames (sync + counter + payload + a parity column):
  `width_scan` finds the frame length, `sync_discover` finds the sync, `wht`
  finds the parity relation, and the counter column's toggle pattern is
  detected.
- Random bits give nothing at the derived rate.

### D2. Scrambling and linear complexity (C)

**Theory.**
- **An additive scrambler** XORs the data with an LFSR sequence.
  Berlekamp–Massey finds the shortest LFSR that produces a given sequence,
  and it needs only 2L bits for an L-stage register. Idle fill is usually
  all zeros or all ones, so a stretch of scrambled idle *is* the LFSR
  sequence, in the clear.
- **A multiplicative (self-synchronizing) scrambler** feeds back the
  scrambled output. It needs no sync to descramble, and every channel error
  multiplies by the number of taps — which is itself a recognisable sign.

```c
int       atkdsp_bm(const uint8_t *b, size_t n, uint8_t *poly, size_t cap, int *L);
int       atkdsp_bm_profile(const uint8_t *b, size_t n, int32_t *L_of_n);
int       atkdsp_descramble_add(const uint8_t *b, size_t n, uint64_t poly, uint64_t seed, uint8_t *out);
int       atkdsp_descramble_mult(const uint8_t *b, size_t n, uint64_t poly, uint8_t *out);
ptrdiff_t atkdsp_scrambler_search(const uint8_t *b, size_t n, int kind /* ADD|MULT */,
                                  int max_degree /* <= 32 */, int max_taps /* <= 5 */,
                                  atkdsp_scramcand *out, size_t cap);
```

What each one does:
- **`bm_profile`** is linear complexity as a function of length. An LFSR
  plateaus at its length L; random bits track n/2.
- **`scrambler_search`** ranks candidates by the structure of their output:
  entropy drop, sync hits, run-length shape. For multiplicative scramblers
  the search is over sparse polynomials — pairs and triples up to degree 32
  are thousands of candidates, a brute force C can afford. Pfa is divided by
  the number tried.

**Acceptance.**
- Every catalogue scrambler recovered from synthetic scrambled idle (by
  Berlekamp–Massey) and from scrambled framed data (by the search).
- Random bits return "no scrambler" at the derived rate.
- Results agree with `bits.descramble_search` on the degrees it covers.

### D3. FEC decoders and the CRC engine (C)

**Theory — Viterbi.** A convolutional encoder is a small state machine.
Viterbi finds the most likely path through its trellis by keeping one
survivor per state. Soft decisions — how confident each received bit was —
are worth about 2 dB over hard ones. Example: for a BER of 1e-5, uncoded
BPSK needs 9.6 dB Eb/N0; K = 7, rate 1/2 with soft Viterbi needs about
4.4 dB.

- **Viterbi.**
  - K ≤ 15, rate 1/n (n ≤ 8), arbitrary generators (octal).
  - A puncturing matrix (2/3, 3/4, 5/6 and 7/8 from a mother code).
  - Input: soft int8 or float LLRs, or hard bits.
  - Modes: zero-terminated; tail-biting (wrap-around with configurable laps
    — this retires `lte_sib1_rx.viterbi_tb`); streaming (traceback depth
    ≥ 5K).
  - Reports the path metric as a confidence.
  - SIMD add-compare-select (AVX2).
  - Encoders for all of the above, for re-encode agreement (D4) and tests.
- **Reed–Solomon.**
  - GF(2^m) for m = 3 to 16; any field polynomial, first consecutive root,
    primitive element and number of roots.
  - Shortened codes and erasures.
  - Berlekamp–Massey + Chien + Forney; a CCSDS dual-basis option; an
    encoder.
- **BCH (binary).**
  - n = 2^m − 1 (m ≤ 16) with a designed t; the generator is built from
    minimal polynomials; shortened codes.
  - Berlekamp–Massey + Chien.
  - POCSAG's BCH(31,21) is cross-checked against ATK's pager decoder.
- **Small linear block codes, by syndrome table** from a generator matrix
  (n ≤ 32, n − k ≤ 20): Hamming (7,4), (15,11) and (16,11,4); Golay (23,12)
  and (24,12); QR(16,7,6); (13,9); the DMR/P25 set. One decoder, one table
  per code.
- **CRC engine.**
  - Width 1–64, with the Rocksoft parameters (poly, init, refin, refout,
    xorout).
  - **Bit-granular messages**, because a bitstream is not byte-aligned.
    Table-driven when the message is byte-aligned.
  - A catalogue of named CRC parameter sets. Parameter sets are facts; they
    are re-typed, with no code taken.

```c
atkdsp_viterbi *atkdsp_viterbi_create(int K, int n, const uint32_t *polys,
                                      const uint8_t *puncture, int punc_len,
                                      int mode /* TERM|TAILBITE|STREAM */, int tb_depth);
ptrdiff_t atkdsp_viterbi_decode(atkdsp_viterbi *v, const float *llr, size_t nllr,
                                uint8_t *bits, size_t cap, float *metric);
ptrdiff_t atkdsp_conv_encode(int K, int n, const uint32_t *polys, const uint8_t *bits,
                             size_t nbits, int tailbite, uint8_t *out, size_t cap);
atkdsp_rs *atkdsp_rs_create(int m, uint32_t gfpoly, int fcr, int prim, int nroots, int pad);
int        atkdsp_rs_decode(atkdsp_rs *r, uint16_t *sym, const int *erasures, int nerase, int *ncorr);
atkdsp_bch *atkdsp_bch_create(int m, int t, int shorten);
int        atkdsp_bch_decode(atkdsp_bch *b, uint8_t *bits, int *ncorr);
atkdsp_blk *atkdsp_blk_create(int n, int k, const uint8_t *G);      /* syndrome-table decoder */
uint64_t   atkdsp_crc(const uint8_t *bits, size_t nbits, const atkdsp_crcspec *s);
```

**Acceptance.**
- Round trips at every supported parameter.
- Viterbi BER versus Eb/N0 within 0.3 dB of the published soft-decision
  curves for K=7 r=1/2 and K=9 r=1/3 (the test asserts the table).
- RS against the CCSDS 131.0-B parameters and published vectors.
- BCH and RS correct exactly t errors. Beyond t, the miscorrection rate is
  measured and stated rather than assumed.
- Every catalogue CRC reproduces its check value over "123456789".

### D4. Code recognition — name the code (C)

**Theory.** A linear block code (n, k) has n − k parity bits, each a fixed
linear combination of the data bits.
- Write the stream into a matrix n columns wide, aligned on a codeword. The
  rows then live in a k-dimensional subspace, so the GF(2) rank is k, not n
  — a **rank deficiency of n − k**.
- Scan widths and offsets: the deficiency appears at width n and its
  multiples, at the right alignment.
- **Convolutional codes** show the same effect, growing with width. The
  slope gives the rate, the onset gives the constraint length, and the null
  space of the matrix gives the parity-check (dual) code, from which the
  generator polynomials follow.

```c
int atkdsp_gf2_rank_scan(const uint8_t *b, size_t n, size_t w_lo, size_t w_hi, size_t rows,
                         int16_t *deficiency /* [w][offset] */);
int atkdsp_block_code_id(const uint8_t *b, size_t n, atkdsp_blockid *out);   /* n, k, offset, G */
int atkdsp_conv_code_id(const uint8_t *b, size_t n, atkdsp_convid *out);     /* rate, K, polys */
int atkdsp_rs_code_id(const uint8_t *b, size_t n, atkdsp_rsid *out);         /* m, n, k, fcr, gfpoly */
int atkdsp_interleaver_id(const uint8_t *b, size_t n, atkdsp_ilvid *out);
```

How each piece works:
- **Rank scan:** bit-packed Gaussian elimination, OpenMP across widths.
- **Block codes:** (n, k), the alignment, and a systematic generator matrix.
  Then the test for a known code: Hamming, Golay or BCH by generator
  polynomial, using the cyclic test (the GCD of codeword polynomials).
- **Convolutional codes:** rank-by-width plus the dual-code basis. Verified
  by Viterbi decode → re-encode agreement.
- **RS:** symbol-level GF(2^m) rank for m = 4–8, at n = 2^m − 1 and at
  shortened lengths. The generator's roots come from testing which field
  elements are roots of every codeword, which gives fcr and the field
  polynomial.
- **Interleaver:** a block interleaver over coded data shows rank deficiency
  at periods tied to its rows × columns, and its dimensions come out of the
  deficiency pattern.
- **Noise tolerance.** Plain Gaussian elimination breaks on a few bit
  errors. The robust variant runs many small random row subsets and votes on
  which column combinations are (almost) dependent (Valembois;
  Cluzeau–Finiasz). It is practical to roughly 1e-3 to 1e-2 BER, depending
  on code length. Beyond that the answer is "no code found", stated together
  with the estimated BER.

**The evidence rule (Bill, 2026-10-04).** Every named code carries its
evidence: the deficiency table and the decode → re-encode agreement %.
Random bits come back "no code found" — a negative control in every test.

**Acceptance.** Synthetic streams from every code in D3, at BER 0, 1e-4 and
1e-3: (n, k), or rate, K and generators, recovered; interleaved variants
recovered; random bits give none.

### D5. CRC recovery (C)

**Theory.** A CRC is linear (affine, once init and xorout are counted).
- XOR two frames of the same length: their CRC fields XOR too, and init and
  xorout cancel. What is left is a codeword of the bare polynomial.
- Every such XOR is divisible by the generator, so the GCD over GF(2)[x] of
  several XORs is the polynomial (or a multiple of it, to factor).
- Init and xorout then come from one frame by solving a small linear
  system, for each refin/refout combination.

```c
ptrdiff_t atkdsp_crc_recover(const uint8_t *const *frames, const size_t *nbits, int nframes,
                             int width /* 0 = try 3..64 */, int pos /* END | SEARCH */,
                             atkdsp_crccand *out, size_t cap);
```

It handles:
- an unknown width;
- an unknown coverage start (a header the CRC excludes);
- a CRC that is not at the end;
- mixed frame lengths (equal-length pairs are used);
- bit-granular frames.

**False alarms, derived.** With m frames, a random polynomial fits by chance
with probability about 2^(−w(m−1)), and the result reports that number.

**Acceptance.**
- Every catalogue CRC recovered from four frames of random lengths.
- The same with a 7-byte header excluded from the CRC.
- Random frames give nothing.

### D6. The analyst bench in ATK (siga + siga_view)

**The detailed spec is now ATK's `docs\BIT_BENCH_PLAN.md`** (2026-10-27):
the Bit Bench workspace, modelled on the bitstream screens Bill used,
PROCITEC's periodicity and circulation displays, the 89600 VSA panes, and
Hobbits' linked views. The summary below stays as the engine's view of it.

**One bit dossier per stream** (FUTURE_PLANS Scoped B, step 1). Every
result from D1–D5 lands there with its units and its evidence. The LLM
adviser reads the dossier and never writes to it.

**Views:**
- The raster, coloured by provenance — measured, decoded or inferred
  (Bill's colour-by-how-it-was-produced rule) — with a column strip
  underneath (ones density, parity).
- A frame stack aligned on sync.
- Hex / ASCII / octal panes.
- A two-stream diff.
- The linear-complexity plot.
- The rank-deficiency heat map: width × offset, where the code shows as a
  bright line.
- Autocorrelation.
- Constellation and EVM, when the bits came from C1.

**The chain.**
- A saved list of (operation, parameters) steps. Editing a step re-runs
  everything after it.
- Replay on a new stream.
- Export the chain as a recipe file (the Forensics Workshop's idea).

**The soft path.** When the bits came from C1, the chain carries LLRs until
a hard decision is genuinely needed (Viterbi, BCH), and only then goes hard.

**New modules.** `siga/fec.py` (D3/D4), `siga/crc.py` (D3 engine + D5),
`siga/scramble.py` (D2), `siga/fields.py` (field discovery),
`siga/alphabet.py` (codes → text), `siga/linecode.py` (NRZ-M/S, biphase,
async, HDLC). `bits.py` and `bitchain.py` are extended, not replaced.

**Acceptance.** The Scoped B synthetic harness — HDLC framing, NRZI,
Manchester, a K=7 convolutional code, a block interleaver, a scrambler, and
plain random as the control — taken end to end by the deterministic toolbox
alone. Every stream's chain is found and its payload read; the random stream
returns "no structure".

---

## §E Longley-Rice (ITM)

### E1. NTIA ITM, ported to C99

**What.** Point-to-point path loss over a terrain profile, plus area mode.
This covers KRAKEN_DF_PLAN §9.5's three needs: the range likelihood of
§8.7; your own coverage footprint from where you are parked ("if 473 is in
that valley you can't hear him from here"); and repeater footprints.

**How.** NTIA's ITM is public domain and written in C++ (github.com/NTIA/itm).
It is **ported to C99**, not built as a C++ target:
- header rule 1 says no C++;
- the code is procedural;
- public domain permits the port.

NTIA's own C++ is compiled in the *test* build only, as the oracle; it is
never shipped.

```c
int atkdsp_itm_p2p(const double *pfl, size_t npts, double step_m, double h_tx_m, double h_rx_m,
                   double f_mhz, int pol, double eps_r, double sigma, int climate, double N0,
                   int mdvar, double time_pct, double loc_pct, double sit_pct,
                   double *loss_db, long *warnings, atkdsp_itm_inter *inter);
int atkdsp_itm_area(/* area-mode inputs */ double *loss_db, long *warnings);
int atkdsp_itm_footprint(const float *radials /* [nrad][npts] */, size_t nrad, size_t npts,
                         double step_m, const atkdsp_itm_cfg *cfg, float *loss_db /* [nrad][npts] */);
```

ATK cuts the radials from the DEM. C computes each point's loss along its
radial, with OpenMP across radials and no callbacks (rule 1).

**Twin — declared exception.** The fallback is "ITM unavailable", stated on
Setup. The oracle is NTIA's code and its test vectors.

**Acceptance.**
- KRAKEN_DF_PLAN §9.5: NTIA's test vectors reproduced; DTED0 and GLO-30
  agree at their common posts; the Virginia tile list is 36 (that last is
  ATK's — recorded here so it is not lost).
- Agreement with NTIA's C++ within 1e-6 dB on a corpus of random profiles.
- A 400×400 footprint timed on Bill's PC.

---

## §F Intercept audio

FORENSICS_PLAN §12.4 ("the radio door") already assigns this work to
atkdsp; this part builds that list. Two places use the kernels:
- the Forensics Workshop's Restore Audio (analyse → recommend → run, with
  trust tiers) offers them offline;
- ATK's live audio path uses the same kernels in streaming mode.

**No SAM Audio on this path.** SAM's licence bans military and espionage
uses, and intercept audio is exactly that. So the radio door never offers
it, and a Workshop test asserts that the option list for an RF-sourced
recording contains no SAM adapter. SAM stays where Bill approved it — phone
and video recovery, under the owner exception — and nowhere near RF.

### F1. Re-demodulation from I/Q (Recovered tier)

- **A channel filter fitted to the measured occupied bandwidth** (B2), not
  the channel plan. Zero risk; always the first step.
- **PLL and FM-feedback demodulators** — threshold extension of about
  2–3 dB and 5–7 dB respectively (published figures).
- **An extended-Kalman FM demodulator with a fixed-lag / RTS smoother**
  (Einicke). Offline only, because it is non-causal. In simulation it shows
  about 12 dB lower MSE than the discriminator at 7.5 dB SNR. It is
  *unmeasured on real voice*; measuring it on Bill's captures is part of the
  work.
- **Click suppression by IF-amplitude weighting.** The idea comes from
  ka9q-radio's description; its GPL code is never read.
- **Synchronous AM with sideband selection, plus SSB and CW**, moving
  `dsp.py`'s numpy demodulators into C.

**Acceptance.** FORENSICS_PLAN §12.12's harness: Bill's own off-air
recordings, with SNR and intelligibility measured with and without each
step. The discriminator is the baseline.

### F2. The classical cleanup chain (Cleaned tier)

**It starts from what exists.** The diffusion toolkit already has the
classical speech enhancers in numpy:
- `repair/speech.py`: Boll spectral subtraction with Berouti's
  over-subtraction; a decision-directed Wiener filter; Ephraim–Malah
  MMSE-LSA; and MCRA noise tracking, which needs no noise-only lead-in;
- `repair/audio_inpaint.py`: Janssen autoregressive gap filling;
- `repair/wer.py`: the word-error-rate scorer.

Those modules are the specification. F2 is their streaming C version, for
the live headphone path and for long files, and it is cross-checked against
them. What F2 adds is the radio-specific part below: the carrier-present
mask, the hum comb, heterodyne notches, and level normalization.

- **Band limits and notches.** A voice band-limit (300–3400 Hz,
  adjustable); a hum comb (50/60 Hz and harmonics, auto-detected); adaptive
  notches for heterodyne whistles (C4's line enhancer).
- **Click repair.** Impulse detection, then autoregressive interpolation
  across the gap (Janssen).
- **Noise reduction.** MMSE log-spectral amplitude (Ephraim–Malah), a
  decision-directed Wiener mode and spectral subtraction, with MCRA noise
  tracking — the same set and parameters as `repair/speech.py`.
- **The noise profile comes from pauses inside the transmission** (carrier
  up, mouth closed). It never comes from before key-up or from the squelch
  tail. Discriminator noise with no carrier is louder and shaped
  differently, so a profile taken there over-subtracts and eats speech
  (§12.4d). The kernel takes a carrier-present mask, and a test feeds it a
  squelch tail and asserts the tail is excluded.
- **Speech-level normalization** (a slow AGC plus a limiter) — often the
  biggest single win for a human ear.
- **Two modes:** streaming (headphones) and offline (whole file, with
  look-ahead). Every stage also returns the part it removed.

**Acceptance.**
- Whisper word error rate on Bill's transcripts, with and without the chain
  — the same measurement as the diffusion plan's first D2 experiment.
- SNR improvement on synthetic speech mixed with off-air noise.
- The squelch-tail exclusion test.

### F3. How the models sit on top

- **The Forensics Workshop's Restore Audio** offers F1 and F2 as Measured /
  Recovered / Cleaned options, beside DeepFilterNet3, RNNoise and
  SpeechBrain enhancement (Inferred). For an RF-sourced recording the
  classical chain runs first, and it is the baseline every model must beat
  on Bill's off-air noise (§12.12) — none of those models were trained on
  NBFM.
- **The diffusion toolkit's D2** (SGMSE through `repair/speech_learned.py`,
  in its own process; Invented tier) is scored against F2 under the
  toolkit's own rule: "a learned tool that does not beat the classical one is
  not shipped." F2 is that classical baseline, made fast enough to run live.
- **The trust tier travels with the output** (`recipe.py`), so an invented
  syllable never looks recovered.

### F4. Listen along the bearing (MVDR) — after atkdf's calibration path

FORENSICS_PLAN §12.4b: an MVDR/LCMV beamformer on a calibrated, coherent
KrakenSDR clip, steered at the bearing *DF the selection* produced, with
nulls on co-channel interferers. Up to about 7 dB of array gain
(10·log10 5 — arithmetic, not yet measured). The kernel lives here; steering
and calibration come from atkdf. It is listed so it is not lost, and it
builds after A3 and C1.

---

## §G Order, sizes and releases

Each phase is one release with one ABI bump, its twins and cross-checks,
and a README section. Its ATK side follows straight after (with `git status`
checked first).

| Phase | Contents | Size | Why here |
|---|---|---|---|
| **1a** ✅ v0.14.0 | **Bitstream engine:** D1 primitives, D2 scrambling + Berlekamp–Massey, D3 Viterbi / RS / BCH / small codes / CRC engine, D5 CRC recovery (+ the SIB1 housekeeping) | L | Bill's top priority. Pure mathematics, fully verifiable against known answers with no hardware, and it touches nothing the other instances are editing in the RF path |
| **1b** ✅ v0.15.0 | **Name the code:** D4 — rank scan, block / convolutional / RS / interleaver recognition, the robust (noisy-stream) variant | M–L | Needs 1a's encoders and decoders for its decode → re-encode evidence |
| **1c** | **Bitstream bench in ATK:** D6 — dossier, raster colours, chain, the new siga modules **(BUILT 2026-10-28 as ATK's Bit Bench — see its plan's §10)** | L | The engine is only useful to an analyst once the bench drives it |
| 2 | **Detection track:** B1 CFAR, B2 burst descriptors + TDMA, A3 C ring **(BUILT 2026-10-28 — v0.16.0 / ABI 17)** | M–L | Cued DF, Bug Sniffer and Drone Discovery all wait on it |
| 3 | **Small, high value:** C3 analog IDs, B4 cyclic probes, B3 pulse runs **(BUILT 2026-10-29 — v0.17.0 / ABI 18, with the CFAR ρ² table and the fixed-bits fix; MDC-1200 / FleetSync are 3b)** | M | ID-tagged analog bearings, waterfall captions, live PDWs |
| 3b | **C3's data bursts:** MDC-1200, FleetSync I/II | S–M | Reverse-engineered framing: real recordings before trust |
| 4 | **Engine:** A1 float32 FFT, A2 fused mixer, A4 channelizer | L | Before the wider SDRs (B210, Signal Hound); the channelizer feeds C3 on many channels at once |
| 5 | **Demod and quality:** C1 linear chain + equalizer, C2 EVM/MER | L | Completes the soft path into phase 1's Viterbi |
| 6 | **Below the floor:** B6 DSSS, C4 strong-signal removal | M–L | Uses B4, D2 and C1 |
| 7 | **The long watch and identity:** B5 occupancy + baseline, C5 fingerprint features | M | Uses B2, C2 and C3 |
| 8 | **Intercept audio:** F1, F2, and the F3 wiring into the Workshop | M | Independent; the Workshop's radio door is waiting for it |
| 9 | **Propagation:** E1 ITM | M | Independent; moves up if DF footprints are wanted sooner |
| — | F4 MVDR listening | M | After atkdf's calibration path |

Sizes: S = one session; M = two or three; L = more. These are rough, and
they are not promises. Bill's bench tests (real radios, his PC's timings) go
at the end of each phase — they are not run from the cloud.

---

## §H Where this plan differs from `DSP_EXPANSION_PLAN.md`

The two plans cover the same request. They share the header rules, phase
the work by ABI bump, and put twins and real-signal acceptance first. They
differ here:

| Topic | `DSP_EXPANSION_PLAN.md` | This plan | Why it matters |
|---|---|---|---|
| Order | Foundation first: FFT / mixer / ring → CFAR / channelizer → probes → bitstream fourth (offers to swap) | Bitstream first (1a–1c), then the detection track | Bill named the bitstream work as his priority. It needs nothing from the other phases, and it can be fully tested without hardware |
| Naming codes | "Report candidates and periodicities, never name a code it has not decoded" | Rank-deficiency recognition names (n, k), rate / K / generators, and RS parameters. The evidence is decode → re-encode agreement, with random bits as the control | Bill decided on 2026-10-04 to name the codes (FUTURE_PLANS "Scoped A") |
| Bitstream depth | Decoders plus the traditional toolbox | Adds Berlekamp–Massey and linear complexity, sync discovery, width scan, Walsh–Hadamard parity relations, field discovery, multiplicative scrambler search, the soft-decision path, and CRC recovery inside the bitstream phase | The R&S CA250 and go2ANALYSE feature lists treat these as standard |
| SAM | "Segment Anything is for imagery elsewhere in ATK; wrong tool for audio" | SAM Audio *is* in the Forensics Workshop's audio recovery (`media/sam_audio.py`). It is kept out of intercept audio because its licence bans military and espionage use | The reason decides where else it must not appear |
| Intercept audio | atkdsp supplies cheap filters; the learned denoising is a toolkit call | The C version of the toolkit's classical chain, plus the atkdsp list in FORENSICS_PLAN §12.4: EKF / PLL / FM-feedback demodulators, click suppression, synchronous AM, MVDR listening | §12.4 already assigns those items to atkdsp |
| Audio forensics | A separate Workshop track: ENF, splice detection, codec history | Parked as Workshop work, not planned here | The same conclusion; the ideas live in that plan |
| ITM | A C++ target or C99, to be decided at build time | Ported to C99 | Header rule 1: no C++ |
| float32 FFT | An in-house radix-2 | Vendored PFFFT, pinned | Radix-2 alone is slower than a radix-4 SIMD library, and a pinned vendored copy brings no surprise upgrades either |
| Ring | The library allocates it (`create`) | It lives in caller memory, so a named shared mapping lets another process (atkpr, atkdaq) read it | Bill kept the ring because more C consumers are coming, and those consumers are separate programs |
| Existing numpy code | Notes the toolkit's denoisers | Uses the toolkit's CFAR, fingerprint features and speech enhancers as the twins and specs | Avoids a third copy of each |

**Either way:** one plan of record, and one instance building atkdsp at a
time. Two instances writing `src/` and bumping the ABI number at once will
collide.
