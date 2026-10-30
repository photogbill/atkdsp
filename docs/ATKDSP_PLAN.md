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
under ASan + UBSan. Windows the same evening (commit ac9e395): MSVC 19.44 /
AVX2 / OpenMP builds, ATK takes ABI 18, 954 passed in 244 s through ATK's
`build.bat /only atkdsp /test`. ATK's `dsp_native.py` expects ABI 18. Phase 3b (MDC-1200
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

**§B6 revised (2026-10-29, evening).** Bill asked for the DSSS section to
be improved comprehensively and for the literature (arXiv, IEEE, the
Chinese signal-processing journals) to be read for ideas. It was —
`docs/DSSS_LITERATURE.md` has every paper, what it adds, its evidence
grade and what could not be read — and §B6 is rewritten around what an
analyst actually meets (Wi-Fi Barker, Zigbee, GPS, DSM's GFSK chips, 406
SGB, CDMA pilots, bugs), a fused three-handle detector of ours with stated
false-alarm rates, the eigen step as the literature does it now, the
generator polynomial through chip errors, a first-class known-code path
(acquire, track, the power-distortion GNSS monitor), the Kraken's channels
as a detector, acceptance in tiers Bill can run with a HackRF, and **a
provenance field on every method** (Bill's rule: Chinese-journal methods
identifiable in ATK's menu). B6b adds LoRa. **B6c, Undertow**, is the
answer to Bill's next question the same evening — a new technique for
unknown DSSS at unknown chip rate and bandwidth, as far below the floor as
physics allows, that records and catalogues: a second-order correlation
radiometer across receivers at its centre (the DSSS literature is
single-receiver and fourth-order), a scale-space survey, a long watch with
drift compensation, bootstrap despreading over the ring, a consistency
lattice, and the catalogue. The §G rows for phase 6 grow accordingly.
Nothing of it is built.

**Phase 6a is BUILT (2026-10-29)** — atkdsp v0.18.0 / ABI 19 (§B6 the blind
path and the known-code path, §B6b LoRa, and the part of §B6c Undertow that
does not need the channelizer or the baseline: the receiver-pair cross-
spectrum, the alignment, and the consistency lattice). The fused three-handle
detector, the chip-rate scan, blind code recovery (wide-window eigenvector,
Rao–Edelman, carrier offset, vote, the polynomial through chip errors),
acquisition and tracking with C/N₀ and the power-distortion monitor, the LoRa
probe, and the provenance table are all in. Header §25–27, twins in
`reference.py` §25–27, 44 tests in `tests/test_phase6.py`, smoke lines; the
whole suite and the smoke test pass on Linux, also under ASan + UBSan. Every
method carries its provenance (country included), so ATK groups the method
menu by origin per Bill's rule. ATK's `dsp_native.py` now expects ABI 19.
Not in this release: the scale-space survey and the catalogue (wait for A4
and B5), MDC-1200 / FleetSync (phase 3b), C4 strong-signal removal, and the
learned fuser (`docs/DSSS_LITERATURE.md` §K). The Windows run is owed
(`build.bat /only atkdsp /test`).

**Phase 6b, Undertow piece 1, is BUILT (2026-10-30)** — atkdsp v0.21.0 /
ABI 22: §B6c's scale-space survey (`atkdsp_undertow`, header §29, twin
`reference.py` §29, 13 tests in `tests/test_undertow.py`, smoke lines), with
two channelizer corrections and two bit-identical PFB speedups riding along
(§A4's note). One receiver; no false alarms on empty coloured noise at a
survey-wide Pfa of 1e-3; a 31-chip code at −6 dB in-band found in 2 s with
its exact period. What changed from the design is in §B6c piece 1's note.
Next: piece 2 as measured (its note) — the period-synchronous long watch
with (T, Ṫ) refinement, fused with piece 4's bootstrap; then piece 6's
kernel half (the hit blob).

**Phase 6b, Undertow pieces 2, 4 and 6 (kernel half), are BUILT
(2026-10-30)** — atkdsp v0.22.0 / ABI 23: the survey's candidate periods, the
period-synchronous search `atkdsp_psync` (period and period rate), the
bootstrap `atkdsp_bootstrap` (psync -> the code), and the versioned hit blob
(header §30, twin `reference.py` §30, 18 tests in `tests/test_psync.py`,
smoke lines), with a correction to `atkdsp_dsss_code`'s carrier offset. At
the real survey rate, 2 s: the exact code, named, at −16 and −18 dB in-band,
where the fluctuation handle is silent. What building them changed is in
piece 2's and piece 4's notes. Not here: the catalogue (ATK, `rf_data`), the
radiometer in the cells and a full blind scan of every lag (GPU).
`dsss_code`'s 2-adic decimation and Frobenius parity stack followed in
v0.23.0 (BUILT 2026-10-30), with a fix for its window-boundary tie and chip
regeneration from a named generator: at 127 chips, −16 dB, 150 periods, 17/40
exact codes in v0.22 → 40/40 (README).

**Phase 6b, Undertow piece 5 (the full lag scan), is BUILT (2026-10-30)** —
atkdsp v0.24.0 / ABI 25, on the CPU (the plan said GPU; the FFA made it
cheap enough without): `atkdsp_lagscan` (header §31, twin `reference.py`
§31, 17 tests in `tests/test_lagscan.py`, a smoke line). Lag products at
every period of a cell by the Fast Folding Algorithm; a null built from the
cell's own correlation with a saddlepoint tail (the moment-matched Gamma it
replaced was 7× anti-conservative, measured); detections resolved into
families by nested direct folds and CLEAN. Three things building it changed,
each in the README: the null, the families (a strong code detects in 83
rows), and the FFA's trial period (P/(1 − x/P), not P + x). Measured: the
period at −15 dB in-band over 4 s of a 60 kS/s cell (6/6), where the
survey's candidates lose it near −12 dB. ATK's LPI / LPD page runs it per
cell and hands each family to Recover. Not here: the m-test for cyclic-line
families on unfiltered chips (only PFB cells were seen to need none), and
the GPU form.

**What building phase 6a changed in this plan** (the README and
`docs/DSSS_LITERATURE.md` have the detail):
- The detector's thresholds are the chi-square tail of the Gamma-distributed
  fluctuation statistic, not a Gaussian: at a few tens of windows a Gaussian
  threshold false-alarms tens of percent. Measured to its Pfa.
- Code recovery is sign-invariant (the eigenvector's sign is arbitrary; the
  bits fed to Berlekamp–Massey from −code are the complement, of linear
  complexity r+1), and names by parity-check counting whenever BM does not
  cleanly name — so one chip error does not defeat the naming.
- Acquisition is per-period PCPS (one period correlated at a time, coherent-
  summed), so the peak-to-second-peak is real; the earlier whole-block form
  gave equal peaks at every period offset.
- Tracking's carrier loop is a Costas two-quadrant arctan (data-bit
  insensitive) with a velocity-integrator 2nd-order filter; it despreads at
  zero bit errors to −10 dB.
- The receiver-pair align finds the lag by a block-non-coherent cross-
  correlation (robust to a large LO offset that decorrelates one long sum)
  and reads the LO from the block-summed product (which kills the band
  carrier the signal×noise cross terms ride).
- The receiver-pair covariance/eigh of §B6c and the channelizer-DF's W1
  (`docs/CHANNELIZER_DF_REQUIREMENTS.md`) are the same primitive — one
  Hermitian rank-1 accumulate plus one `eigh`, not one shared API. xspec
  keeps the frequency axis; W1 collapses it. Unify at that primitive when
  the channelizer is built.

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
  live pulse runs, cyclic probes, occupancy and baseline, spread spectrum
  (DSSS blind and known-code, the GNSS monitor; LoRa in B6b; Undertow,
  the below-the-floor watch, in B6c)
- §C Demodulation and signal quality — the linear demod chain and
  equalizer, EVM/MER, analog radio IDs, strong-signal removal, transmitter
  fingerprint features, analog video (C6, `docs/VIDEO_PLAN.md`)
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

> **DF requirements (2026-10-29):** before building, read `docs/CHANNELIZER_DF_REQUIREMENTS.md` — the multi-element (N coherent streams, one block clock) form, sample-index bookkeeping, the per-channel covariance accumulator and channel-grid powers that make this kernel the KrakenSDR "DF everything" engine. Nothing there changes the ABI below; it adds `_process_n` forms and three small outputs. The receiver-pair covariance/eigh of §B6c and the channelizer's W1 are the **same primitive** — one Hermitian rank-1 accumulate plus one `eigh` — not one API: xspec keeps the frequency axis, W1 collapses it.

> **Part 1 BUILT (2026-10-29)** — atkdsp v0.19.0 / ABI 20, tool (b) the PFB.
> The uniform 2× oversampled filterbank, its multi-element coherent form
> (`atkdsp_pfb_process_n`, R1–R3), the per-window cal hook (R5), and the W1
> covariance + W2 power accumulators DF runs on the output. Header §28, twin in
> `reference.py` §28, `src/channelizer.c`, 16 tests in `tests/test_channelizer.py`,
> a smoke line; whole suite (1014) + smoke pass on Linux, also under ASan + UBSan.
> Verified against the DF contract: C == twin 4.7e-8; centre-channel rejection
> 93 dB; **inter-element phase 8.7e-6°** (R7 ≤ 0.1°); elements bit-identical on
> identical input; chunk invariance exact. ATK's `dsp_native.py` needs ABI 20.
> **Part 2 BUILT (2026-10-30)** — atkdsp v0.20.0 / ABI 21, tool (a) the
> overlap-save extractor (`atkdsp_osx`). Arbitrary channels at arbitrary
> centres: one forward FFT per block, per-channel fold-to-`nfft/decim` + small
> inverse (the small transform is the integer decimation), then the arbitrary
> resampler to each channel's `out_rate`; add/retune/remove on stable slots
> without disturbing the others; the N-element form (`atkdsp_osx_process_n`,
> W4) keeps them phase-coherent for DF. Header §28a, twin in `reference.py`
> §28a, 6 tests (22 total in `tests/test_channelizer.py`), a smoke line.
> Verified: in-band unit gain (matches a DDC), −111 dB stopband, C == twin
> 5.8e-7, add/retune/remove leaves others bit-identical, chunk invariance
> exact, W4 inter-element phase 0.008°. Both channelizer tools are now in.
>
> **Speed (2026-10-30, v0.23.1, no ABI change).** Bill's three findings, all
> confirmed: one radio ran the PFB on one thread (the parallel loop was over
> elements); the fold's double-product-then-float mix did not vectorise; the
> stride-m output store missed the cache. Now threads over hops, a float fold
> over duplicated taps, a 32-hop tile written channel by channel, the FFT in
> place on the fold's buffer, and pocketfft's work buffer on the stack (its
> per-transform malloc was a shared lock on several threads). Measured on 2
> cores: one radio 2.3–2.8× faster on one thread, 3.7–4.4× on two (20 MS/s at
> ~100 kHz channels: 1.1× → 4.7× real time); identical at any thread count;
> 1.1e-7 from v0.23.0, 8.8e-8 from the twin. `pfb_bench.bat` measures it on the
> machine that matters. The OSX extractor is single-threaded still (ATK does
> not use it yet; parallel over channels is the obvious form when it does).
>
> **Corrections (2026-10-30, v0.21.0 / ABI 22).** (1) The PFB's per-hop bin
> rotation had the wrong sign and referenced each hop to its own start: right
> at 2× only up to a constant (−1)^k per bin, wrong at 3× and 4× (a tone 3 kHz
> above bin 1 came out at −197 kHz). Found by ATK's trunking work
> (`TRUNKED_RADIO_PLAN.md` §4A); fixed as written there — every bin is now the
> true baseband referenced to input sample 0, at any oversampling. At 2× the
> output changes by exactly (−1)^k, so W1/W2 and every DF covariance are
> unchanged bit for bit. (2) The extractor mixed relative to each block's
> start: an odd bin shift flipped the sign every block and up to half a bin of
> residual offset stayed in. Now one absolute-time phase per output sample —
> on centre and phase-continuous across blocks and calls, like a DDC. The old
> tests checked magnitude at 2× and an even bin; 14 new ones check frequency
> and phase at 2×/3×/4× and odd and fractional offsets (verified to fail
> against the old code). **Speed:** `_process_n` runs elements in parallel
> and the fold reads windows in place (both bit-identical): the Kraken DF
> case went from 1.0× to 1.5× real time on 2 cores (kernel alone; unchanged
> single-threaded — the fold's arithmetic is the cost, so its vectorisation
> is the follow-on for 40 MSPS).
>
> **Still not here:** the band-edge guard marking (caller-side, R4), and W3
> (the channel-grid CFAR gate, wiring on the `atkdf` side over W2 + the
> existing CFAR). Each release's Windows run: `build.bat /only atkdsp /test`.

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

### B6. Spread spectrum — detection, chip rate, the code, and despreading

*Revised 2026-10-29, after phase 3, at Bill's request ("improve it
comprehensively"), against the literature read the same day —
`docs/DSSS_LITERATURE.md` holds every paper, what it adds, and what could
not be read. The first draft's four blind steps are kept; the box at the
end lists what changed. Nothing here is built yet.*

**Provenance rule (Bill, 2026-10-29).** Every method here carries where it
came from — authors, year, venue, country — as a fixed string in the
header (`ATKDSP_DSSS_METHOD_*` with an `atkdsp_method_info` table) and in
every result struct and `describe()`, so ATK can group the instrument's
method menu by origin: **Chinese-journal methods are identifiable** (their
own drop-down, or a flag on each entry), and so are ours. The tags used
below: **CN** China, **FR** France, **SG** Singapore, **FI** Finland, **US**,
**ES** Spain, **DE** Germany, **UK**, **BE** Belgium, **ATK** ours. Where a
CN method is known to us from its abstract only (the literature file says
which), the kernel is rebuilt from the idea and the tag still names the
source.

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
  sits below the floor. That is the handle the blind path uses — and its
  price is known: Burel's detector pays **four dB of output for every dB
  of input**, and buys it back at 10·log10(windows) (literature A1).
- **Chip transitions are periodic whatever the code.** Squaring or
  delay-multiplying the signal leaves a spectral line at the chip rate, long
  code or short, one user or many. That is the handle that does not need a
  period, and the one that names the chip rate.
- **Correlation is the third handle.** Across the Kraken's five antennas a
  signal is correlated and noise is not; across oversampled or shaped
  chips the same holds within one channel. Tests built on that need no
  noise-power estimate, which is what a radiometer lacks (A2, A3, F1).
- **Many codes an analyst meets are published.** Then the problem is not
  blind at all: it is acquisition (which code, what code phase, what
  Doppler) and tracking — the GPS receiver's problem, cheap and exact. The
  kernel does both, blind and known, and the known path is also the second
  half of the blind one (once a code is recovered it is a known code).

**What the analyst will actually meet.** The design is driven by this table,
not by the textbook BPSK case alone. "Blind?" is whether the chip sequence
can be recovered without the code: it needs a short code (one period
spanning few symbols), chips that carry the data as a sign, and one user.

| Signal | Chip rate | Code | Chips carry | Known? | Blind? |
|---|---|---|---|---|---|
| GPS L1 C/A | 1.023 Mc/s | 1023 chips, 1 ms; 20 periods per 50 bit/s bit | BPSK | IS-GPS-200, every PRN | detect and period yes; one SV's code no — 8–12 SVs overlap at different Dopplers |
| GLONASS L1OF | 511 kc/s | 511 chips, 1 ms (one code, FDMA) | BPSK | ICD | as GPS |
| Galileo E1, GPS L1C, M-code | 1.023 (BOC) | 4092 / 10230 chips + secondary codes | BOC: a square subcarrier on the chips | ICD (M-code no) | detect: lines at the subcarrier AND the chip rate, a split spectrum; code no |
| 802.11b beacons, 1 and 2 Mbit/s | 11 Mc/s | Barker-11, one symbol per period | DBPSK / DQPSK | standard | **yes — the cleanest real case, any Wi-Fi AP sends them** |
| 802.11b 5.5 / 11 Mbit/s | 11 Mc/s | CCK: 8-chip codes chosen by the data | complex | standard | detect and chip rate; code set, not a code |
| IEEE 802.15.4 (Zigbee, Thread) | 2 Mc/s | 16 sequences of 32 chips, one per 4-bit symbol | O-QPSK, half-sine (MSK-like) | standard | detect, chip rate, period 16 µs; the sequence SET by clustering, named against the standard |
| COSPAS-SARSAT SGB (406 MHz) | 38.4 kc/s | spreading factor 256, I and Q codes from x²³ + x¹⁸ + 1, one 1 s burst | OQPSK | T.018 | known-code only (INTERCEPT_IDEAS_PLAN §I2 already says so) |
| IS-95 / CDMA2000 pilot | 1.2288 Mc/s | short PN 2¹⁵ chips, 26.67 ms; traffic under the 2⁴² long code | QPSK | IS-95 / C.S0002 | pilot: detect, chip rate, period; the long code no |
| UMTS CPICH | 3.84 Mc/s | scrambling code 38400 chips, 10 ms | QPSK | 25.213 (512 primary codes) | detect, chip rate, period; the code by known-code search |
| Spektrum DSM2 / DSMX, CYRF6936 toys and RC links | 1 Mc/s | 32- or 64-chip PN codes; SDR 1 bit, DDR 2, 8DR 8 bits per code (derived codes); DSMX also hops 23 channels | **GFSK, ±270–323 kHz** — the chips are frequency, not phase | datasheet modes; the PN tables are in open-source RC firmware (licence check before use) | detect and chip rate yes (FDIFF / DM lines); the code from the FM-demodulated chips, not from the I/Q eigenvector |
| Spread-spectrum "bugs" (commercial 900 MHz / 2.4 GHz audio) | 0.5–10 Mc/s typical | short codes, proprietary | BPSK/QPSK usually | no | the design case for the blind path |
| LoRa / ELRS / Crossfire / Meshtastic / Cricket | — | chirp spread spectrum, SF 5–12, BW 125 kHz–1.6 MHz | chirps | Semtech, open | **not DSSS** — see B6b below |
| Frequency hopping (Bluetooth, FHSS RC, SINCGARS) | — | — | — | — | not DSSS; hops are B1/B2 bursts |

Three consequences for the design. (1) Chips are not always a sign on a
carrier: O-QPSK/MSK and GFSK chips need the transition nonlinearities phase
3 built (DM and FDIFF in `atkdsp_cyclic`) for the chip rate, and the GFSK
case needs the FM-demodulated chip stream for the code. (2) Several
important signals use a *set* of codes selected by the data (CCK, 802.15.4,
8DR), so "the code" is sometimes "the codebook", found by clustering
period-long windows, not by one eigenvector. (3) The known-code path is
where most of the operational value is — GPS, 406 SGB, Wi-Fi, Zigbee — so
it is a first-class deliverable, not a footnote to the blind path.

**The chain, in six steps.** Each step reports what it found, its
evidence and its provenance, and the next step takes the previous step's
output, so an analyst can stop anywhere and still have an answer ("spread,
2.000 Mc/s, period 16 µs, code set of 16 — 802.15.4") rather than nothing.
Every threshold states its Pfa and is measured in the harness the way
phase 3 measured its own (binomial bands on noise-only runs).

0. **Precondition.** Run after C4's excision when a narrowband interferer
   is present (a CW tone 30 dB up swamps every statistic below), and run
   B4's `cp_probe` first on any candidate: an OFDM cyclic prefix also
   correlates at a fixed lag. The two are told apart by the shape of the
   lag peak — OFDM's is a triangle Ncp wide at lag N with nothing at 2N;
   a short code's is a full replica at every multiple of the period.
1. **Detect — three handles, fused (ATK).** Three statistics on the same
   windows, each with its own noise-only distribution, each honest about
   the class it cannot see, combined as a sum of log-likelihood ratios so
   that a short-code BPSK bug, a long-code CDMA pilot and a GFSK-chip DSM
   link are all caught by whichever handle fits, and the evidence says
   which one did. No paper read does this fusion; it is ours.
   - **Repetition — Burel's fluctuation-of-correlation detector (FR
     2000).** Cut the signal into M windows (window ≥ the longest period
     sought), estimate the autocorrelation in each, and average |R_m(τ)|².
     Noise gives a flat floor for τ ≠ 0 at the estimator's own variance
     σ⁴/T (the analytic threshold); a short-code signal gives peaks at
     every multiple of the period, whatever the carrier offset (Doppler
     rotates R's phase, not its magnitude) and without chip timing.
     Reported −13 dB at 50 windows, −16.6 dB at best (literature A1); the
     periodicity test (≥ 6 equally spaced peaks) takes the false alarms
     out. Streaming: windows cut on the global index, a running mean,
     `read` gives ρ(τ), its floor and the peaks — chunk-invariant by
     construction. The UI shows the sizing law: 4·SNR_in + 10·log10(M).
     Blind to: long codes.
   - **Transitions — the chip-rate line with a stated Pfa (FI/US 2007).**
     The cyclic autocorrelation at α = k·f_chip after the nonlinearity
     (step 2 computes it anyway), tested as Lundén's χ² GLRT; when the chip
     rate is unknown the *maximum over the scanned rates* uses the
     max-statistic CDF, so a scan over a thousand candidate rates still
     reports a true Pfa. Long code or short, one user or many. Blind to:
     nothing periodic in its chips — but a CW tone lights it (step 0).
   - **Correlation — no noise power needed (SG 2008; US 2012 for the
     Kraken).** One channel: Zeng & Liang's CAV test on L lags (threshold
     from L, Ns, Pfa alone; holds Pfa under 0.5–2 dB of noise uncertainty
     where a radiometer collapses; needs ≥ 2 samples per chip or shaped
     chips — at one rectangular sample per chip it is blind, and says so).
     Five channels: Urriza–Cabric's canonical-correlation cyclic test
     (threshold from the channel count alone, immune to spatially coloured
     noise, ≈ 20× cheaper than SCF combining) and the MME eigenvalue test
     on the spatial covariance (Tracy–Widom threshold). The conjugate
     cycle at 2·f_c is present for BPSK chips and absent for QPSK chips,
     which classifies the chip modulation for free. ATK-side wiring in
     atkdf; the kernel supplies the covariance accumulation and the tests.
   - **Secondary indicators, display only:** a Jarque–Bera fourth-order
     score (ES 2025) — PSK chips are non-Gaussian, Gaussian-looking chips
     defeat it, which is itself a hint; and the cyclic-energy fraction
     (US 2026) as a meter, never a decision.
   - **Symbol period under a long code** — the variance of the windowed
     correlation as a function of the window start is periodic in the
     symbol period (FR 2002; a 2020 Springer WPC paper does it from
     covariance norms over trial window lengths). That gives Rs, hence
     G = Rc/Rs, which classifies the link (a 20 dB-gain telemetry link is
     not a 43 dB CDMA pilot). Report "spread: chip rate found, no period"
     for long codes and stop at step 2 — honestly, the long-code case is
     detected and measured, never recovered.
2. **Chip rate.**
   - **The line scan.** Phase 3's `atkdsp_cyclic` tests *candidate* rates
     with one Goertzel each; chip rates are a continuum (0.0384, 0.511, 1,
     1.023, 1.2288, 2, 3.84, 11 Mc/s and every proprietary one), so B6 adds
     `atkdsp_cyclic_scan`: the same segmentation, mean removal and Hann,
     but an FFT of the nonlinearity output over the segment, accumulated
     over segments as level and coherence per bin, the χ² test of step 1
     per bin with the max-statistic Pfa for the scan, then the line's
     exact frequency from the bin's segment-to-segment phase (the phase-3
     method; 0.1 Hz at 100 ms segments). Nonlinearities as in phase 3: |x|²
     and DM for shaped chips, x² and x⁴ for BPSK/QPSK chips (a line at 2×
     or 4× the carrier offset names the chip modulation and gives the
     offset), FDIFF for GFSK/MSK chips. Harmonics confirm (any clock line
     has them); a BOC signal shows the subcarrier line and the chip line
     both.
   - **The spectrum's own evidence.** Rectangular chips put sinc² nulls at
     ±Rc in the PSD and the autocorrelation's τ = 0 peak is a triangle
     2Tc wide; RRC-shaped chips show the roll-off edge at Rc(1 + α)/2.
     Cheap, coarse, and the cross-check the line scan is held to. Once a
     code is in hand, the chip rate read off the denoised eigenvector (CN
     2017, IET) is a third cross-check.
3. **Period, length and symbols per period.** The period from step 1's
   peak spacing, refined by the parabolic peak; L = period × Rc (a
   maximal-length code gives 2ⁿ − 1: 31, 63, 127, 511, 1023; a Gold code
   the same lengths; Barker 11 or 13; 802.15.4 gives 32 at 16 µs). The
   sign pattern of R at successive multiples of the period says whether a
   data symbol spans one period (signs random) or many (signs equal in runs
   of 20 for GPS). Chip timing comes from the chip-rate line's phase; the
   window offset from the CFO-robust covariance-norm search
   `argmax_θ ‖R_y(θ)‖_F` (FR 2002).
4. **The code (short codes, one user).**
   - **Before trusting an eigenvector, the identifiability rule (US
     2007, Rao–Edelman):** with m samples per period and n periods, the
     signal eigenvalue must exceed σ²(1 + √(m/n)); below that no eigen
     method can see it. The kernel computes the minimum n for the measured
     SNR and refuses with "need N more periods" instead of returning a
     garbage vector; the MME ratio with its Tracy–Widom threshold (SG
     2009) is the acceptance test.
   - **Wide-window eigenvector (FR 2000; CN 2012 as an SVD).** Windows of
     2N overlapping by N; the principal eigenvector v (length 2N) holds one
     contiguous code; k_max = argmax_k Σ_{i=k}^{k+N−1} |v(i)|²; the chips
     are sign(v(k_max..k_max+N−1)). No two-vector sign ambiguity, and the
     timing comes free. Power iteration for the top two vectors (C99
     needs no LAPACK for two); **PAST subspace tracking** (CN 2012;
     Atlantis 2015) at O(N) per window when N is in the thousands.
   - **The carrier offset, estimated on the eigenvector (BE 2015).** An
     offset α cycles/sample turns v into diag(e^{j2παi})·v_clean; a
     single-tone estimate on v² (BPSK) attains the Cramér–Rao bound at
     −8 dB on the eigenvector where the raw signal's square needs +12 dB.
     Derotate, re-sign. |α| < 0.25 for BPSK; outside that, step 2's x²
     line gives the coarse offset first.
   - **Decision-directed refinement — alternating sign-despread (CN
     2021, the k-means paper), with our vote margin (ATK):** despread
     every period with the estimate, decide the symbol, multiply back,
     take a majority vote per chip across all periods, iterate; O(M·L) per
     pass. The vote margin per chip is reported, because the next step
     needs to know which chips it may trust. A code verified across
     data-modulated periods uses the MBAPDI / GLRT combiners (ES 2018),
     not Σ|y|² (the worst choice) nor the differential form (breaks with
     bits).
   - **The generator polynomial, through chip errors (FR 2004, Mitton /
     Cluzeau).** Berlekamp–Massey (D2) names an m-sequence's LFSR from 2n
     error-free chips and lies on one wrong chip (the linear-complexity
     profile's jump is the tell, and `atkdsp_bm_profile` reports it). So
     BM runs only on chips the vote margin certifies; otherwise the
     parity-check counting statistic Z_Q = Σ(−1)^{parity} over the
     candidate primitive polynomials of degree r (60 candidates at r = 10),
     Z > k√N with N ≈ k²/ε^{2w} chips, ε = 1 − 2·(chip error rate) — ≈ 95
     chips at 10 % errors for a trinomial, a Walsh–Hadamard batch over the
     low-degree candidates — and a few fast-correlation-attack flip passes
     when the error rate is above ≈ 5 %. A Gold code shows linear
     complexity 2n, a Kasami ≈ 2.5n; either way the generator(s) are
     named, or the result says "L chips recovered, generator not
     confirmed, vote margin X" and hands the chips over anyway.
   - **Code sets** (802.15.4, CCK, 8DR): k-means on the aligned windows
     (k = 16 for 802.15.4 is known; otherwise the eigenvalue count says how
     many), the cluster centres' signs are the codebook, matched against
     the published sets in `reference.py`.
   - **QPSK chips and two users on one period (CN 2023, Fast-ICA):** the
     symbol × chip matrix is an instantaneous two-source mixture; a
     kurtosis fixed point with deflation separates the I and Q codes (or
     two synchronous users), cheaper than the EVD when symbols < chips;
     needs symbol-aligned windows (step 3's θ search) and the CFO removed
     (≤ 10⁻⁴ cycles per chip). Reported to −10 dB at 750 symbols.
   - **Several users on one code family** (GPS): the eigen-decomposition
     gives a subspace, not a code; ICA separates users with different data
     in principle (the 2023 result; a 2019 Lancaster method does whole
     multi-rate CDMA systems with triple-correlation tables — reference
     only), but at GPS's SNR and Doppler spread it is a research item. The
     plan says so and the GPS code comes from step 5 instead.
   - **GFSK chips:** FM-demodulate first (phase 3's FDIFF path), then the
     eigenvector of the demodulated chip waveform. The data bit inverts the
     chips, which negates the frequency, so the covariance has the same
     ±structure as BPSK.
5. **Acquire, track, despread — known codes and recovered ones alike.**
   - `atkdsp_dsss_acquire`: parallel code-phase search — the FFT of one
     code period of signal times the conjugate FFT of the code, at each
     Doppler bin; the GLRT statistic Γ = 2K·S_max/P_in with its CFAR
     threshold γ = −ln(1 − (1 − Pfa)^{1/N_cells}) (each normalised cell is
     Exp(1) under H0; χ² with 2d dof for d non-coherent dwells — GNSS-SDR's
     form), the peak-to-second-peak ratio reported beside it; Doppler bin
     width 0.5–0.7 of 1/T_coh (DE 2011: narrower bins raise the global
     false-alarm rate), a second pass at an eighth of the step around the
     peak; coherent over one period, non-coherent over k periods for the
     weak case with the MBAPDI combiner (ES 2018) across unknown data
     bits; delay-multiply bit stripping (CN 2020) before a long coherent
     accumulation as an option. Returns code phase (sub-chip by parabola),
     Doppler, C/N₀.
   - `atkdsp_dsss_track`: an early–prompt–late DLL for code phase, a
     Costas / arctan PLL for carrier (or FLL-assisted), one handle per
     code, chunk-invariant; outputs the despread symbols at Rs plus C/N₀
     (M2M4 as the estimator, NWPR beside it, with a validity flag below
     25 dB-Hz where both bias low by ≈ 8 dB — literature E4), lock state
     and the code-phase and Doppler histories. The despread symbols go to
     C1's chain for the data (or to the 406 SGB decoder, or to GPS's 50
     bit/s navigation message — ATK side).
   - `atkdsp_dsss_pd_monitor` — the power-distortion monitor (US 2017/
     2018, Humphreys's group; real-data validated on TEXBAT): received
     power over 200 ms and the symmetric-difference distortion of the
     correlation peak at ±0.15 chip (Rayleigh under H0, σ² = 8τ_d),
     classified clean / multipath / spoof / jam; the 11-tap maximum-
     likelihood residual as the upgrade. This is the RF detector
     `gnss_status.py` has been waiting for.
   - **The code catalogue** in `reference.py`: GPS C/A (G1/G2 taps per
     PRN, IS-GPS-200), GLONASS, Barker-11/13, the 802.15.4 sixteen, the
     IS-95 short PN and UMTS primary scrambling generators, T.018's SGB
     generator and start states (read T.018 first, as INTERCEPT_IDEAS_PLAN
     says). DSM PN tables only where the source licence allows. A
     recovered code is added to the session's catalogue with its evidence
     and provenance, so it is acquired and tracked the same way from then
     on.

**Honest limits.**
- Long codes (GPS P(Y), CDMA traffic, anything whose period spans many
  symbols) are *detected and measured* (chip rate, symbol rate, gain) but
  not recovered, and the result says so.
- Blind code recovery needs one user, a short code and a clean band: it is
  the spread-spectrum bug case. GPS is many users under the floor; its
  value is in step 5, and in step 1's "there is spread energy at 1.023
  Mc/s with a 1 ms period" as the blind cross-check.
- Every operating point quoted from the literature is simulation or a
  wired SDR bench (the power-distortion monitor is the one exception);
  ours are measured in the harness against a planted known-good emitter,
  and the kernel reports its own measured rates, not the papers'.
- The polynomial step is exact or it says why not: the vote margin and the
  linear-complexity profile are the honesty mechanisms; the
  parity-check statistic needs a chip error rate under ≈ 30 % to be
  practical at all.
- GFSK chips with 8DR derived codes: the chip rate and the hop pattern
  are the identification; the data path is a known-code one.
- A CW tone correlates at every lag and lights every nonlinearity: step 0
  is not optional when one is present.
- C/N₀-only spoofing detection is weak (Jammertest 2024: meaconing caught
  0–3 % on GPS C/N₀ alone); the distortion monitor is the detector, C/N₀
  the supporting evidence.
- Frequency hopping is not DSSS. Hops show up in B1/B2 instead.

```c
/* provenance — a fixed table in the header; every result struct and describe() carries an index into it */
typedef struct { const char *name, *authors, *year, *venue, *country; } atkdsp_method_info;   /* country: "CN", "FR", "ATK", ... */
ATKDSP_API const atkdsp_method_info *atkdsp_method(int method_id);

/* blind path — streaming handles, chunk-invariant (rule 3) */
atkdsp_dsss_det *atkdsp_dsss_det_create(double fs, size_t win, size_t max_lag, int nwin_min, double pfa, int handles);
                       /* handles: bitmask REPETITION | TRANSITION | CORRELATION; fused LLR, per-handle verdicts */
int   atkdsp_dsss_det_process(atkdsp_dsss_det *h, const atkdsp_cf32 *x, size_t n);
int   atkdsp_dsss_det_read(const atkdsp_dsss_det *h, atkdsp_dsss_det_result *out);
                       /* rho(tau), floor, peaks, period, symbol period, per-handle statistics + Pfa, which handle fired, method ids */
atkdsp_cyclic_scan *atkdsp_cyclic_scan_create(double fs, int nonlin, size_t seg_len, int nseg, double f_lo, double f_hi, double pfa);
                       /* ... process / read: every line in [f_lo, f_hi) with level, coherence, chi2 statistic, exact frequency, harmonics */
int   atkdsp_dsss_code(const atkdsp_cf32 *iq, size_t n, double fs, double rc, double period_s,
                       int chip_mod, int method, int8_t *chips, size_t cap, atkdsp_dsss_code_result *out);
                       /* method: EIGEN_WIDE | PAST | KMEANS_REFINE | FASTICA (each with its provenance);
                          out: L, start, cfo, identifiability margin, quality, vote margin per chip (min/mean),
                               generator polynomial + how it was found (BM | PARITY_COUNT | none), LC profile verdict */
/* known-code path — serves GPS, SGB, Wi-Fi, Zigbee and every recovered code */
int   atkdsp_dsss_acquire(const atkdsp_cf32 *iq, size_t n, double fs, const int8_t *code, size_t L, double rc,
                          double dop_lo, double dop_hi, double dop_step, int ncoh, int nnoncoh, int combiner, double pfa,
                          atkdsp_dsss_acq *out);       /* code phase, doppler, Gamma + its Pfa, peak ratio, C/N0 + validity */
atkdsp_dsss_trk *atkdsp_dsss_trk_create(double fs, const int8_t *code, size_t L, double rc, double code_phase, double doppler);
ptrdiff_t atkdsp_dsss_trk_process(atkdsp_dsss_trk *h, const atkdsp_cf32 *x, size_t n, atkdsp_cf32 *symbols, size_t cap);
int   atkdsp_dsss_pd_monitor(const atkdsp_dsss_trk *h, atkdsp_dsss_pd *out);   /* power, distortion, verdict: clean / multipath / spoof / jam */
/* multi-channel: covariance accumulation, MME / CCA cyclic tests under atkdsp_eig_* (shared with atkdf) */
/* + describe / stats on every handle (rule 5) */
```

**Acceptance.**
- Synthetic BPSK, QPSK, O-QPSK (half-sine) and GFSK-chip DSSS, L = 11, 31,
  127, 1023, at −5, −10, −15 and −20 dB in-band SNR, with and without a
  CW tone 30 dB up:
  - each handle's false-alarm rate on noise inside the binomial band at
    its stated Pfa, including the chip-rate *scan* (a thousand candidate
    rates) and the fused verdict;
  - detection at each SNR recorded per handle (the table the UI shows),
    against Burel's sizing law;
  - chip rate within 0.1 % and the chip modulation named;
  - the period exact; symbols per period right;
  - the code recovered (up to sign/phase), the identifiability refusal
    firing exactly when Rao–Edelman says it must, the vote margin
    reported;
  - the generator named by BM on certified chips and by parity-check
    counting at 10 % and 25 % planted chip errors, refused honestly at
    40 %;
  - the 802.15.4 codebook recovered as 16 clusters and named;
  - the long-code case reports chip rate, symbol rate and gain and says
    "not recovered".
- Noise, a strong narrowband carrier, analog FM, OFDM (the cp_probe case)
  produce nothing from the DSSS detector; the OFDM case is named as OFDM.
- Known-code: acquisition finds a planted code at −25 dB with k non-
  coherent periods at its stated Pfa, tracking holds it through a 1 Hz/s
  Doppler ramp, the despread symbols decode, and the C/N₀ estimate is
  within 1 dB above 25 dB-Hz and flagged below.
- **Real, tier 1 — 802.11b beacons** off the HackRF at 2.4 GHz (any AP, 20
  MSPS): blind detection, 11.000 Mc/s, the 1 µs period, Barker-11 recovered
  up to sign, DBPSK symbols despread. The honest first real test: one
  user, short code, strong signal.
- **Real, tier 2 — an 802.15.4 device** (a Zigbee bulb or Thread sensor):
  2 Mc/s, 16 µs, the sixteen sequences.
- **Real, tier 3 — GPS L1 C/A** off the HackRF with antenna power and an
  active patch antenna: blind — 1.023 Mc/s and the 1 ms period from under
  the floor (the fluctuation detector with thousands of windows); known —
  every visible PRN acquired with Doppler and C/N₀ (the published codes
  check the answer), and tracking's 50 bit/s stream (the nav message is
  ATK-side); the power-distortion monitor reads "clean" on the open-sky
  capture and "multipath" beside the house. The 1 ms period is also the
  BOC cross-check: Galileo E1 shows its subcarrier line beside it. (The
  TEXBAT recordings, if obtainable, are the spoofing acceptance.)
- **Real, if a transmitter is at hand — a DSM2/DSMX radio**: chip rate 1
  Mc/s from FDIFF, the GFSK chip class named, DSMX's 23-channel hop seen as
  bursts in B1/B2, DSM2's two channels static.
- **Multi-channel**: on the Kraken, a DSSS emitter planted 6 dB under the
  single-channel floor detected by the CCA and MME tests at their stated
  Pfa, its chips classified BPSK/QPSK by the conjugate cycle, and a
  bearing from despread-then-MUSIC (atkdf) within atkdf's usual error.
- The cost at 20 MSPS is recorded on Bill's PC (the fluctuation detector
  is one FFT per window; the line scan one FFT per segment; the
  eigenvector two power iterations per update).

**Uses.**
- A spread-spectrum instrument on the Signals bench: detect → chip rate →
  period → code → despread, each stage with its evidence and provenance,
  and a known-code acquisition panel with the catalogue. The method menu
  is grouped by provenance (Bill's rule).
- Bug Sniffer: spread-spectrum bugs hide under the floor; the blind path
  is for them, and the Kraken's correlation tests find them before a chip
  is known.
- GNSS status (`gnss_status.py` has DENIED and UNTRUSTED states that "a
  detector said so", and today no RF detector): acquisition across all
  PRNs plus the power-distortion monitor give the RF evidence — no SV
  acquired with a clear sky and a raised floor is DENIED; a distortion
  verdict of spoof, or too many SVs at the same C/N₀ with one common
  Doppler relationship, is UNTRUSTED. The ADS-B NACp layer stays the area
  picture; this is the local one.
- Drone Discovery: link identification by chip rate, chip modulation and
  hop pattern (1 Mc/s GFSK chips over 23 channels is a DSMX; 2 Mc/s O-QPSK
  is 802.15.4; a 250 kHz–1.6 MHz chirp is LoRa / ELRS).
- atkdf: despread-then-DF — correlate each Kraken channel with the known
  or recovered code before MUSIC, and the bearing gets the processing
  gain. A DSSS emitter under the floor then gets a bearing.
- The fingerprint features (C5): chip-rate error is a clock fingerprint;
  the code itself (which PN, which start state) is an identity.
- 406 SGB (INTERCEPT_IDEAS_PLAN §I2) can use step 5 if numpy is too slow,
  as that plan allows.

### B6b. Chirp spread spectrum — LoRa and its relatives

Not DSSS, but the spread-spectrum signal Bill will meet most (Cricket is
LoRa; ELRS and Crossfire fly drones; Meshtastic is everywhere), so it
belongs beside B6 and shares its bench. A LoRa symbol is one linear chirp
across the bandwidth, cyclically shifted by the symbol value; 2^SF values
per symbol; a preamble of 8 unshifted up-chirps, two sync chirps, two and a
quarter down-chirps.

- **Detect and name (SF, BW):** multiply the signal by the conjugate base
  up-chirp for each candidate (SF 5–12 × BW 7.8 kHz–1.6 MHz, a small grid)
  and FFT over one symbol: a LoRa signal collapses to a single bin per
  symbol; the preamble gives the same bin eight times — the coherence test
  again. Noise collapses to nothing. Report SF, BW, the carrier offset
  (from the up/down-chirp pair) and the symbol timing. (An STFT
  instantaneous-frequency classifier in the literature works only above
  −5 dB; the dechirp is the detector.)
- **Demodulate:** the bin index per symbol is the symbol; gray mapping,
  de-interleave, Hamming decode and de-whiten per the open LoRa
  descriptions (gr-lora, LoRa-SDR; read before writing — the exact
  interleaver and whitening were reverse-engineered and differ by SF and
  header mode). Cricket's own packets are the acceptance oracle, with
  the LoRa header's CRC as the proof.
- **Honest:** ELRS runs a modified LoRa at 2.4 GHz and FLRC (not chirp)
  at its highest rates; the detector names chirp parameters, and a
  protocol decode is per-protocol, later.

```c
atkdsp_css_probe *atkdsp_css_probe_create(double fs, const atkdsp_css_cand *cands, int ncand, double pfa);
/* process / read: per candidate (SF, BW): level, coherence, offset, timing; flags DETECT / PREAMBLE */
ptrdiff_t atkdsp_css_demod(const atkdsp_cf32 *iq, size_t n, double fs, int sf, double bw, double offset,
                           double t_sym0, uint16_t *symbols, size_t cap);   /* raw symbol values */
```

**Acceptance.** Synthetic LoRa at every SF/BW at −10 dB (LoRa's own
operating region — SF 12 demodulates at −20 dB), the right (SF, BW) and
nothing else; noise and DSSS produce nothing; real: Cricket's own frames
off the HackRF, symbols matching what Cricket sent, header CRC good.

**Uses.** Drone Discovery (ELRS / Crossfire control links named by their
chirp parameters); Cricket self-test and range work; Bug Sniffer (LoRa
trackers); the Signals bench.

> **What this revision changed, and why.** (1) A table of the real
> signals, because the first draft designed for the textbook BPSK short
> code and most of what Bill will meet is not that: GFSK chips (DSM), MSK
> chips (802.15.4, SGB), codebooks (802.15.4, CCK), many users (GPS),
> long codes (CDMA). (2) The known-code path — acquire, track, despread,
> a published-code catalogue and the power-distortion monitor — is now a
> first-class deliverable; it is where GPS, 406 SGB, Wi-Fi and Zigbee get
> their value, and it is the second half of the blind path anyway. (3)
> Detection is three handles fused (ours), each with a threshold that
> states its Pfa and needs no noise-power estimate where the literature
> offers one — the first draft had one handle and "a line above the
> floor". (4) A line *scan* (FFT) for the chip rate with a true Pfa over
> the scan, since chip rates are a continuum and phase 3's per-candidate
> Goertzel cannot search one. (5) Long codes get a symbol-period
> estimator and a processing-gain figure instead of only "detected".
> (6) The eigen step done as the literature does it now: an
> identifiability rule before trusting it, the wide-window cut instead of
> two-vector alignment, the carrier offset estimated on the eigenvector,
> PAST for long periods, alternating-sign refinement with a vote margin,
> Fast-ICA for QPSK chips. (7) The generator polynomial through chip
> errors by parity-check counting, with Berlekamp–Massey only on certified
> chips — the first draft would have named wrong polynomials. (8) The
> Kraken's five channels as a detector (CCA, MME) and despread-then-DF,
> which is the piece that turns "there is a spread bug" into a bearing.
> (9) Real acceptance in tiers that can actually be run from Nokesville
> with a HackRF — 802.11b first, Zigbee second, GPS third — rather than
> GPS alone, which is the hardest case (many users, 18 dB under) and the
> one blind recovery cannot do. (10) The GNSS DENIED / UNTRUSTED states
> get their RF detector, the one method in the review with real-data
> validation. (11) Provenance as a field of every method, per Bill's
> rule. (12) B6b for LoRa, which is the spread spectrum Bill owns an
> emitter of.

### B6c. Undertow — the below-the-floor spread-spectrum watch (new, 2026-10-29)

*Bill, the same evening: "a new, innovative technique that can look for
unknown DSSS with unknown chip rates and unknown bandwidths … what hasn't
anyone done … which techniques have never been put together … the best,
below the noise floor DSSS auto-detection possible, that can record and
catalog DSSS signals." This is the answer. Working name Undertow — what
moves under the surface. The claims of novelty are made against the
literature read today (`docs/DSSS_LITERATURE.md`), not against everything
ever written; where a piece is known, the source is named, and the
novelty is the assembly. Provenance tag for the whole: **ATK**, with each
borrowed part tagged where it is used.*

**Why the literature stops where it does.** Every blind DSSS detector in
the review is a single-receiver, single-band method: it is handed a
channel already centred on the signal at roughly the chip-rate bandwidth,
and it decides. None searches frequency and bandwidth; none integrates
longer than a capture; none uses a second receiver; none feeds what it
found back into a matched filter on the data it just looked at; none
keeps what it found. And they share a physical ceiling. Burel's statistic,
the cyclic line and the eigen methods are all **fourth-order** — they
square the signal (or a correlation of it) before averaging — so their
deflection grows as SNR² · √(N·L): each dB of input costs four dB of
output, as the thesis's law says. A **second-order** detector — one that
correlates the signal with an independent copy of itself — grows as
SNR · √N. At −20 dB in-band, SNR² is −40 dB; for a 1023-chip code the
fourth-order path gives back √L ≈ 32 of that and still loses by ≈ 10 dB
in deflection; for a 31-chip bug it loses by ≈ 25 dB. The only
second-order blind detector is the radiometer, and it is useless because
it needs the noise floor to a tenth of a dB (the SNR wall; literature A8).
**Undertow's centre is a second-order detector with no SNR wall: the
correlation radiometer** — two receivers whose noise is independent, so
the averaged cross-spectrum of the two has a floor that falls as 1/√N
instead of a fixed floor with ripple. Radio astronomy has run on this for
seventy years (an interferometer sees sources a total-power radiometer
cannot); `DSP_EXPANSION_PLAN.md` Part C lists the correlation radiometer
for exactly that reason; **nothing in the DSSS literature uses it**. ATK
has the receivers: any two whose samples can be put on one time axis —
a B210's pair, an SM200 beside a BB60C on one 10 MHz, two remote nodes on
Cricket's GPS timing, the Kraken's five channels below 1.7 GHz (the
receivers subsection below says what each tier gives).
Around that centre, four more pieces, each a combination that has not
been made.

**The six pieces.**

1. **The scale-space survey — where, and how wide.** A dyadic polyphase
   bank (A4's channelizer at bandwidths fs/2, fs/4, … down to ≈ 20 kHz)
   turns the unknown (f_c, Rc) into a search over cells (channel, scale).
   In each cell, three cheap statistics are accumulated:
   - the small-lag autocorrelation |R(τ)| for τ up to a few chips at that
     scale — for white noise its mean is zero off τ = 0, for a DSSS signal
     it is the chip waveform's triangle (rectangular chips) or its RRC
     curve, whose width *is* Tc and whose height is P_s: a blind estimate
     of chip rate and signal power with no code, at any carrier offset
     (the offset rotates R's phase, not its magnitude);
   - the α-spectrum of the cell after |x|², x² and x⁴: lines at the chip
     rate and its harmonics, at 2× and 4× the carrier offset;
   - the baseline-relative PSD shape, with B5's learned floor removed, fit
     against the chip-shape family (sinc², RRC, GFSK, BOC split).
   **The scale constraint is the new part:** at the cell whose bandwidth
   matches the signal, the chip-rate line must fall in a known window of
   relative frequency (α ∈ [B/2, B] for the shapes above), the ACF triangle
   must be one to two samples wide, and the PSD bump must fill the cell. A
   line anywhere else in that cell is some other emitter's symbol clock.
   The feature is a scale-space extremum — the way a blob detector finds a
   spot at its own size — and it cuts the false-alarm space from "every
   bin of every α-spectrum" to one window per cell. The three statistics
   are then one χ²-tested verdict per cell (literature A4's max-statistic
   form gives the true Pfa over the whole survey).

   > **BUILT (2026-10-30), atkdsp v0.21.0 — and what building it changed.**
   > One PFB per dyadic scale (`nchan0 << s`), cells = (scale, channel). Three
   > handles per cell, each a calibrated tail probability, then one survey-wide
   > correction (cells × handles), so `pfa` is the survey's, not the cell's:
   > **FLUCT** — Burel's fluctuation (§25) on the cell, the period-giver;
   > **ACF** — the chip-shape |R(τ)|/R(0) for τ ≤ a few lags, scored by a
   > predictive Student-t against the cells either side (not against zero: a
   > channelized cell's noise is not white); **CYC** — the |y|² α-spectrum
   > line, searched only in the scale's window [B/2, B]. A hit merges every
   > cell and scale one emitter lights, keeps the strongest cell's fc, bw
   > and period and each handle's best evidence, and with Rc and the period
   > gives L and the CONSISTENT bit. Measured against that design:
   > - **ACF has an SNR wall.** It localises to about −10 dB in coloured
   >   noise within seconds and *never deeper*: its reference is the
   >   neighbouring cells, whose noise is alike but not identical, and the
   >   worst empty cell's score grew with the watch. Kept for where and how
   >   wide above −10 dB; not a depth handle.
   > - **The fluctuation statistic is the wall-free one** (noise has no
   >   period whatever its colour): −6 dB in 0.5 s, about −10 dB in 2 s. It
   >   is the survey's depth, and it is fourth order — each 10× of watch
   >   buys ~2.5 dB (−18 dB ≈ an hour on Burel's law). It is blind to code
   >   phase, so drift does not smear it (1 ppm moves a 1023-chip period
   >   peak by 0.002 samples); piece 2's note says what does go deeper.
   > - **The cyclic line is above-the-floor only** (strong at 0 dB, gone by
   >   −6 dB, SNR² as theory says) and aliases into the window at scales
   >   narrower than the signal; it gives Rc where it is seen and is
   >   cross-checked by the period, never trusted alone.
   > - **`period_min` must clear the channelized noise's correlation
   >   length.** Burel's statistic with `period_min = 4` fired on every
   >   empty cell (log10 Pfa −145): a 2× cell's noise is correlated to
   >   lag ~10, and with hundreds of windows the Gamma model reads that bump
   >   as a period. The survey uses 16. A test pins both halves.
   > - **Not built from this paragraph:** the PSD-shape fit against B5's
   >   learned floor (B5 is not built), the x²/x⁴ carrier lines (B6's chain
   >   does them on a hit), and one χ² verdict over the three statistics (a
   >   min-p with the Bonferroni factor is what is honest when the three
   >   are this unlike in depth). Single receiver only; the radiometer
   >   (piece 3, v0.18.0) is not yet in the cells. The built API is header
   >   §29 (`atkdsp_undertow_cfg`, one stream), not the sketch below.
   > - Throughput: 0.5× real time on one core at 2.4 MSPS, 4 scales, 120
   >   cells; 0.9–1.07× on two; deterministic at any thread count. 40 MSPS
   >   live needs the PFB fold vectorised — done in v0.23.1 (§A4's speed
   >   note); the survey inherits it.
2. **The long watch with drift compensation — how long.** Below the floor,
   time is the only free gain, and the first draft threw it away by
   working per capture. Undertow accumulates each cell's statistics for
   minutes to hours, and the sizing is Burel's law in reverse: the UI says
   "at this rate, −22 dB shows in six minutes". The catch, and the new
   piece: over minutes the code period in samples *drifts* (1 ppm of chip
   clock over 1000 s is a whole GPS period; Doppler does the same to a
   satellite), so the period peak smears across lags and the gain is lost.
   Radar solved this for migrating targets — the Keystone transform, or
   simply accumulating in a (lag, lag-rate) plane and taking the maximum
   along the drift lines. Nobody applies it to blind period detection.
   With it, the fluctuation statistic integrates coherently for as long as
   the emitter is on; a million 1 ms windows on a 1023-chip code puts the
   Burel deflection at −25 dB in-band well above threshold on paper, and
   the measured number is what the acceptance records.

   > **Measured while building piece 1 (2026-10-30): the drift argument is
   > right, the statistic it was aimed at is not.** Burel's fluctuation
   > squares each window's correlation, so it is blind to code phase: a
   > clock offset only moves its peak by ε·T samples (1 ppm on a 1023-chip
   > period is 0.002 samples), and it does not need drift compensation. It
   > is also fourth order with data on the code, so it buys ~2.5 dB per 10×
   > of watch however long it runs — the −25 dB above was optimistic for it.
   > What *is* second order is the period-synchronous eigenvector (the wide
   > window of B6 step 4): learn the code from every period-aligned window
   > at once. Measured on the survey's own cells (31-chip code, one bit per
   > period, 2 s at 600 kS/s): at the known period, λ₁ against the same
   > statistic at a mistuned period gives **−18 dB at z ≈ 44 and −22 dB at
   > z ≈ 8**, where the fluctuation handle loses the period by −14 dB; from
   > 0.5 s to 2 s its evidence grows faster than √N (BBP: the spike clears
   > the noise bulk when T·SNR > √(2T/n), so depth ∝ 1/√N, ~5 dB per 10×,
   > about √T short of knowing the code). Its price is the drift problem
   > exactly as stated above: the windows must stay aligned to a fraction
   > of a chip over n periods, so T must be known to ~chip/n — about one
   > part in a million for those 2 s — and its rate Ṫ held over minutes. So
   > piece 2 is rebuilt as the **period-synchronous long watch**: the
   > survey's hits and near-threshold period candidates seed accumulators,
   > each refined in (T, Ṫ) by maximising λ₁ (the lag-rate search, on the
   > statistic that needs it), with a stated Pfa against the off-period
   > reference (which also absorbs the cell's coloured-noise eigen-spread).
   > The eigenvector it ends with is piece 4's starting code, so 2 and 4
   > are one stage. The coherent cyclic line has the same need, in (α, α̇).
   >
   > **BUILT (2026-10-30), atkdsp v0.22.0 — `atkdsp_psync`, and what building
   > it changed.** The survey supplies candidates (`atkdsp_undertow_candidates`:
   > each cell's strongest fluctuation peaks below its own threshold — the true
   > period is usually in a cell's top 8 to ~2 dB under where the handle fires,
   > often behind its multiples). psync searches T₀ ± 1 sample, escalating the
   > span until a look-elsewhere-corrected gate passes, then refines period and
   > rate to all the data. Measured, and built in:
   > - **The references must be fitted as a line in the period.** λ₁ of pure
   >   noise trends with how much the wide windows overlap; references all on
   >   one side read that trend as signal and stopped the escalation early on
   >   noise. Symmetric references, a least-squares line, a predictive
   >   Student t against it — the grid is scored against the line too.
   > - **λ₂ separates a repeating code from symbol-clocked data.** Plain QPSK
   >   bursts fire the fluctuation handle at their symbol clock (log10 Pfa
   >   −120); psync at those lags finds several comparable components or
   >   nothing (MULTI / no detection). Survey false hits are refuted here.
   > - **Candidates are often 2T or 3T**, where psync also says MULTI
   >   (dominance 0.9, 0.8) — the bootstrap tries T/2, T/3, T/4.
   > - **The rate search** (the lag-rate idea): on a 20 ppm/s ramp λ₁ +9 %
   >   and the rate to about ±30 % — λ₁ is bumpy at 1–2 % over the rate
   >   because window starts are nearest-sample, which bounds the precision.
   >   Its first active level could choose a rate by noise and, with the step
   >   count capped at rate_max/step, never walk back (measured, fixed: ±6
   >   steps at every level).
   > - Depth at the real rate, 2 s: −16 and −18 dB detected (z 35–79), period
   >   to 2.4·10⁻⁶ samples. Cost ~25 s per candidate on 2 cores — a
   >   per-candidate job; the first stage's grid is the GPU's best target.
3. **The correlation radiometer — the second-order centre.** For every
   receiver pair, the averaged cross-spectrum ⟨X₁(f)·X₂*(f)⟩ over the
   survey band. Independent noise averages to zero; a common signal
   leaves its spectral shape — the sinc² bump, the RRC edge, the BOC
   split — on a floor that keeps falling. Three things come out at once:
   detection with a stated Pfa (literature F1's canonical-correlation
   test for the calibrated verdict; it does not care that the two front
   ends' noise is coloured differently), the **bandwidth and shape** read
   off the cross-spectrum directly, and the **phase slope across the bump
   = the time difference of arrival** — a hyperbola on the map before a
   chip of the code is known, and with three nodes a fix with its CEP.
   What the DSSS papers never had is the second receiver; what the
   cooperative-sensing papers never did is read the shape and the TDOA
   from the same cross-spectrum and hand them to a code-recovery chain.
   Honest conditions: the two antennas must see the same signal (a bug
   sweep does; two nodes a kilometre apart see the same satellite but not
   the same bug), common-mode spurs (LO leakage, DC, a shared supply) cross-
   correlate and are masked by a learned noise-only cross-spectrum
   baseline (F2's whitening generalised to pairs), and antenna coupling
   adds a small correlated noise term that the same baseline removes.
4. **Bootstrap despreading — the loop that learns.** The ring (A3) holds
   the last minutes of the survey. The moment a cell's verdict fires, the
   accumulated windows already hold the code's eigenvector (B6 step 4, the
   literature's wide-window form with the identifiability check), and
   that estimate becomes a **matched filter run backwards over the ring**:
   acquisition and tracking (B6 step 5) on the recorded history, which
   returns the full processing gain G on a signal that was only
   marginally detected — a −20 dB blind detection despread with a
   recovered 1023-chip code is a +10 dB symbol stream. The despread
   symbols certify periods, certified periods refine the chips (the vote
   margin), refined chips sharpen the matched filter: the turbo principle
   applied to blind DSSS, which the iterative-estimation papers do within
   one capture and never with a recorder and a tracker in the loop. The
   loop ends with the generator polynomial (B6's parity-check counting)
   and the code entering the catalogue as *known* — from then on this
   emitter is found by acquisition at known-code sensitivity, 10–15 dB
   below where blind detection first saw it. The system gets better at
   finding what it has seen.

   > **BUILT (2026-10-30), atkdsp v0.22.0 — `atkdsp_bootstrap` (the kernel
   > half; the loop over the ring and the catalogue entry are ATK's).** psync
   > (with the harmonic retry), then the carrier from the eigenvector's P-th
   > power, the stream derotated and resampled so a period is exactly ⌊T⌉
   > samples (following the rate), the chip count, and `atkdsp_dsss_code`.
   > The chip rate below the floor falls out: L ÷ period. What it took:
   > - **The chip count is the hard part.** In a survey cell the filter cuts
   >   the code's spectrum at ~2/3 of the chip rate, so the chip-rate line is
   >   weak; vote margins cannot choose (a consistent waveform votes
   >   consistently at any sampling). Two scores of the eigenvector — the
   >   chip-centre envelope (band-limited chips) and the transition line
   >   (sharp chips), each failing where the other works — are shortlisted,
   >   decoded on a slice, and a NAMED code wins. The margin is thin at depth
   >   (envelope ratio ≈ 1.0); for a code with no LFSR generator the count is
   >   weakly determined and `line_ratio` says so.
   > - **`dsss_code`'s carrier offset was broken** for offsets from about
   >   1/(capture) to the in-period limit (the period-summed P-th power
   >   cancels itself); now read on the eigenvector in two steps. Found here;
   >   no earlier test had an offset.
   > - Measured at the real rate, 2 s, +40 kHz: the exact code named
   >   (x⁵+x²+1) at −16 and −18 dB, the offset read to the hertz.
   > BUILT in v0.23.0 (2026-10-30): `dsss_code` averaging an m-sequence's
   > 2-adic decimations (CODE_DECIM) and the Frobenius parity stack
   > (`atkdsp_parity_z_frob`, each image a hypothesis in the threshold —
   > counting only the polynomials false-named 5× its design), plus two
   > things measuring them found: the window-boundary eigenvalue tie (2–8
   > wrong decodes in 40 → 0) and regeneration of the chips from a named
   > generator (CODE_REGEN). End to end the gain is the tie fix, the stack and
   > regeneration; decimation changed no outcome measured (the identifiability
   > gate refuses first) — its value is the margins and codes past
   > `poly_max_degree`.
5. **The consistency lattice — the false-alarm killer and the fingerprint.**
   Five quantities are estimated by independent statistics: Tc from the
   ACF triangle, Rc from the PSD nulls, Rc from the cyclic line, T from
   the period peak, and the carrier offset from the x² / x⁴ lines. They
   must agree: T·Rc must be an integer L, preferably one of the code-
   length families (2ⁿ − 1 for m-sequences and Gold, Barker 11/13,
   802.15.4's 32, GPS's 1023, GLONASS's 511, Galileo's 4·1023, CDMA's
   2¹⁵, UMTS's 38400 — a boost, not a requirement, since a proprietary
   bug may use any length); and the **common-oscillator test**: a
   transmitter derives carrier and chip clock from one reference, so the
   carrier's fractional error (offset / f_c) equals the chip rate's
   (ΔRc / Rc) to within the receiver's own reference error. A detection
   whose five numbers agree is a detection; one whose numbers disagree is
   two emitters, an OFDM prefix (B4), or noise. The same ppm pair,
   stable over sessions, is the emitter's **fingerprint** (C5) — the
   crystal does not change when the channel does.
6. **The catalogue and the recorder — "record and catalog".** Each emitter
   gets a dossier under `rf_data` (never C:\, never AppData): f_c, Rc,
   chip shape and modulation, T, L, the chips with their vote margins,
   the generator polynomial if named, symbol rate and G, the ppm pair,
   first and last seen, the occupancy schedule (B5 keeps it — the Bug
   Sniffer park-mode trigger is "a catalogued or new spread emitter came
   up"), bearings (atkdf, despread-then-DF) and TDOA hyperbolas, C/N₀
   history, the provenance of every method that contributed, and I/Q
   snippets at detection and at best SNR. The catalogue is the known-code
   list the next session starts from, and the thing a report is written
   from.

   > **Kernel half BUILT (2026-10-30), atkdsp v0.22.0:** the versioned hit
   > blob (`atkdsp_undertow_hit_export` / `_import`; little-endian, fixed field
   > order, CRC-32; context, hit, bootstrap, chips). The dossier, the occupancy
   > schedule, the snippets and the catalogue view are ATK's.

**What is new, stated carefully.** Against the literature read today:
(a) the scale-constrained cyclic line in a dyadic bank — wideband
cyclostationary sensing exists (the strip spectral correlation
analyser), but not the constraint that makes the line's position a
function of the cell, which is what turns a survey-wide scan into a
Pfa-stated detector; (b) Keystone / lag-rate accumulation for a blind
period peak — radar's method, not applied to DSSS; (c) the correlation
radiometer with shape and TDOA read from the cross-spectrum, feeding a
code-recovery chain — interferometry's method, not in the DSSS literature;
(d) the bootstrap loop over a recorder with a tracker in it; (e) the
consistency lattice with the common-oscillator test as the decision rule
(the ppm pair is known in transmitter fingerprinting; as a *detection*
consistency test it is not). Each piece alone is a modest step; together
they change what the floor means: with two receivers and an hour,
Undertow's design point is **−25 dB in-band for a 31-chip bug** (the
fourth-order path alone cannot get there), and anything it has seen once
is found again at known-code sensitivity.

**The receivers Undertow runs on — common timing across radios (Bill,
20:23: "the KrakenSDR is less than ideal, so make sure it supports multiple
receivers on a common timing sync — USRPs, Signal Hounds, even Airspys").**
Agreed, and the Kraken's weakness is not coherence but reach: 2.4 MSPS,
8 bits, nothing above 1.766 GHz, so no 2.4 GHz bugs, no 5.8 GHz drone
links, no 11 Mc/s Wi-Fi. The correlation radiometer needs two receivers
whose noise is independent and whose samples can be put on one time
axis; it does not need them to be the same radio, or to share an LO.
Three sync tiers, each honest about what it gives (ATK's
`SDR_SUPPORT_PLAN.md` already grades radios by driver family and proves
coherence in Radio Check step 7; this adds the across-units case):

| Tier | What is shared | Radios | What Undertow gets |
|---|---|---|---|
| **1 — sample-coherent** | one sample clock and one LO across channels | KrakenSDR (5); USRP B210 / X310 two-channel (atkuhd already streams the B210 pair); bladeRF 2.0 micro RX1+RX2; Pluto 2r2t; LimeSDR | everything: the cross-spectrum integrates coherently without limit, and DF (MUSIC) works — once Radio Check step 7 has proven it |
| **2 — reference-locked** | a 10 MHz reference between separate units, plus a PPS or trigger for the sample start | USRP N210 / X310 / N3xx with a GPSDO or external REF + PPS (UHD's clock and time sources: the textbook multi-unit case); B210 with REF IN + PPS IN; **Signal Hound SM200** (10 MHz in; GPS-disciplined, timestamped I/Q on the B/C models — the best wideband option, 40–160 MHz spans to 20 GHz) and BB60C (10 MHz ref in); bladeRF 2.0 (REF IN + trigger); HackRF One (CLKIN 10 MHz); Airspy R2 (an external 10 MHz clock input, per Airspy's documentation — confirm on the unit; the Mini has none) | the sample *rates* agree exactly, so no rate drift; the sample *start* is aligned to about a sample by the PPS, the rest by a reference emitter; each unit's LO phase is unknown and wanders slowly (two synthesizers on one 10 MHz are frequency-locked, not phase-locked) — the cross-spectrum's phase slope (TDOA) is unaffected, coherent integration is limited by the wander unless phase-referenced (below); DF across units needs a per-retune phase calibration |
| **3 — free-running** | nothing but what the signal and GPS give | RTL dongles, HackRF without CLKIN, Airspy Mini, SDRplay, and every `atk_client` remote node | sample-rate offset (tens of ppm, drifting), time offset, LO offset (1 ppm at 2.4 GHz is 2.4 kHz) and LO wander, all to be measured — three ways to do it, below; with them, the correlation radiometer still works, in phase-stable blocks |

The alignment is a kernel job (`atkdsp_align_*`), because every pair needs
it and Tier 1 is only the case where the numbers are zero:

- **GPS-stamped starts and PPS-measured clocks.** A node with a PPS
  (Cricket's T-Beam already gives one to every `atk_client`; a GPSDO gives
  one to a fixed site) timestamps the capture start to the PPS edge and
  measures its own sample clock against PPS over minutes (the ppm the
  GNSS-denied plan already learns for Cricket's holdover). That fixes
  time offset to a few samples and sample-rate offset to a fraction of a
  ppm; the kernel resamples one stream by (1 + ε) (A2's exact-rate
  resampler) and shifts it in time. LO offset and wander remain.
- **The reference emitter — differential alignment.** A strong emitter
  both receivers hear (a broadcast carrier, a pager, a repeater's pilot,
  or Bill's own known-good emitter) is cross-correlated between the two
  streams: its lag gives the residual time offset, the lag's drift gives
  the sample-rate offset, its cross-spectrum frequency gives the LO
  offset, and — the piece radio astronomy adds — **its cross-spectrum
  phase, tracked over time, is the LO wander between the units, so
  applying its conjugate to the whole band cancels that wander:** VLBI's
  phase referencing on a calibrator source, which turns two free-running
  receivers into a coherently integrating pair for as long as the
  reference is heard. `CUED_DF_PLAN.md` already noted the same idea from
  Leffke et al. (AMOS 2024, after Kawase) for a two-site interferometer;
  this is where it is used. It needs no GPS at all, which is the case the
  GNSS-denied plan worried about.
- **GPS as the common clock.** Where a node can capture L1 (an RTL with a
  bias-tee and a patch antenna can), B6 step 5's acquisition of any
  visible PRN gives that node's absolute time and its sample-clock error
  from the code phase and the code Doppler — every node disciplined to
  the same satellites, with no PPS wiring and no shared reference. The
  known-code path pays for itself here.

**What each receiver count buys (Bill, 20:27: "two- and four-channel
SDRs, and what is the most we can do with just a bladeRF by itself?").**
Design points, to be measured, for a 31-chip short code (the bug case)
and a 1023-chip one (the GPS case), ten minutes of watch, a stationary
emitter, drift compensated:

| Receivers | Second-order centre? | Blind depth, design point | Also |
|---|---|---|---|
| **one** (Bill's bladeRF 1.0: one channel, 300 MHz–3.8 GHz, 28 MHz analog, 40 MSPS 12-bit) | no — fourth-order only | ≈ −18 dB in-band for 31 chips, ≈ −23 dB for 1023, from Burel's law with 6·10⁵ windows (4·SNR + 10·log M + 20·log L − 3 against a 4σ floor); the scale-space survey over one 28 MHz slice at a time | **the known-code path is single-receiver and loses nothing**: GPS at −18 dB in 2 MHz is routine, Wi-Fi Barker, Zigbee, 406 SGB; bootstrap despreading gives the full G once a code is in hand; the catalogue, the consistency lattice and B5's floor all work. Not reachable: 5.8 GHz, HF, and anything the fourth-order wall hides |
| **two channels, one box** (B210, bladeRF 2.0 micro RX1+RX2, Pluto 2r2t, Lime) | yes — one pair, sample-coherent | ≈ −25 dB for 31 chips (the correlation radiometer's SNR·√N law with independent noise), deeper for long codes | one interferometric baseline: a bearing with a left/right ambiguity (not a hyperbola — the baseline is the antenna spacing), 3 dB of array gain, **one spatial null** (C4 in space: a CW tone 30 dB up nulled before any statistic); TDOA for geolocation needs the two antennas apart, which one box can do with cable runs of tens of metres |
| **four channels** (X310 with two TwinRX, N310, two B210s on one 10 MHz, or a 2.0 micro pair) | yes — six pairs | the same depth as two channels plus 6 dB of array gain and √6 more independent pairs in the eigenvalue test; the per-pair MME / CCA tests become one 4×4 cross-spectral-matrix test | unambiguous azimuth DF with a planar layout, MUSIC resolving up to three emitters, three spatial nulls, and **despread-then-DF** — a bearing on a bug under the floor |
| **five** (Kraken) | yes — ten pairs, below 1.766 GHz only | as four, at 2.4 MSPS and 8 bits | the DF array it already is |
| **two sites** (any two of the above, tiers 2–3, aligned as described) | yes | as the pair | **TDOA hyperbolas** and, with three sites, a fix with its CEP; the reference emitter makes it work without GPS |

The bladeRF 1.0 by itself is therefore a complete single-receiver Undertow
— survey, long watch, bootstrap, catalogue, and every known-code
instrument — and what it cannot do is exactly the second-order centre and
spatial work: that needs a second channel, and the cheapest step up is a
bladeRF 2.0 micro (two coherent channels to 6 GHz, 56 MHz) or a B210 on
the atkuhd path that already streams both channels.

ATK side, for `SDR_SUPPORT_PLAN.md`'s radio record: four fields —
`ref_in_10mhz`, `pps_or_trigger_in`, `timestamped_stream`,
`multi_unit_sync_tier` — and a **Radio Check step 8**: two units on one
reference (or two free-running units with a reference emitter), the
measured time offset, rate offset, LO offset and LO wander over ten
minutes, and the tier they earn. The multi-unit DAQ (open N receivers of
any family, emit one timestamped multi-stream into the ring) is the
generalisation of atkdaq, on the ATK side; the kernel supplies the
alignment, the pair cross-spectra and the tests. Undertow's own
acceptance then runs on three pairs — a B210's two channels (tier 1), a
B210 with an SM200 or BB60C on one 10 MHz (tier 2), and two HackRFs on
nothing (tier 3) — and records the depth each pair reaches on the planted
emitter. The order of preference for a two-receiver bug sweep is what the
table implies: SM200 pair or B210 pair, then any two reference-locked
units, then anything with a reference emitter in the band; the Kraken
stays the DF array below 1.7 GHz.

**Where a neural network belongs.** Bill remembered the CNN papers; they
are read in `docs/DSSS_LITERATURE.md` §K. On raw I/Q a network learns the
same fourth-order features the classical statistics compute, beats only
the one-capture autocorrelation test (by 3–4 dB, in simulation, mostly
without a stated false-alarm rate), and cannot cross the second-order line
that needs a second receiver. So in Undertow a network sits *above* the
kernel: an optional learned fuser over the per-cell statistics, trained
on the harness's synthetic signals in ATK's Python (TorchSig-style),
calibrated by a noise score pool (the one good idea in that line, **CN
2025**, NUDT), tagged **ATK** for the fuser itself, and judged by the same
three numbers as every other handle. Never inside the C library.

**Honest limits.** One receiver loses the second-order centre: single-
receiver Undertow is the scale-space survey plus the long watch — still
new, still fourth-order, so −15 to −20 dB with patience, not −25. A moving
emitter or a hopping one breaks the long watch (hops are B1/B2's). Two
receivers must see the same signal, and a learned noise-only cross-
spectrum is a precondition, not an option. Bootstrap despreading needs a
short code and one user; for long codes the catalogue entry is "chip rate,
symbol rate, gain, shape, TDOA" and stays honest about it. Everything
above the floor that Undertow finds, B6's plain chain finds faster;
Undertow is for what the chain cannot see.

```c
/* the survey — one handle per receiver (or receiver pair), streaming over the ring */
atkdsp_undertow *atkdsp_undertow_create(double fs, int nscales, double f_lo, double f_hi, int nrx, double pfa,
                                        const atkdsp_undertow_cfg *cfg);     /* cfg: window, max lag-rate, baseline source, handles */
int   atkdsp_undertow_process(atkdsp_undertow *u, const atkdsp_cf32 *const *x, size_t n);   /* nrx streams, same clock */
int   atkdsp_undertow_baseline(atkdsp_undertow *u, const atkdsp_cf32 *const *x, size_t n);  /* noise-only learning (floor, cross-spectrum) */
ptrdiff_t atkdsp_undertow_read(const atkdsp_undertow *u, atkdsp_undertow_hit *hits, size_t cap);
      /* per hit: f_c, Rc (three estimates + agreement), Tc, shape, T, L, carrier offset, ppm pair, per-statistic
         values with their Pfa, which handles fired, TDOA per pair with its error, time-to-detect, method provenance */
int   atkdsp_undertow_bootstrap(atkdsp_undertow *u, const atkdsp_ring *ring, int hit, atkdsp_dsss_code_result *code,
                                atkdsp_dsss_trk **trk);  /* eigenvector -> acquire/track over the ring -> refined code */
/* catalogue I/O is ATK-side Python (rf_data); the kernel exports a hit as a versioned byte blob, like B5's state */
```

**Acceptance — the honest way to measure depth.** Bill's rule: always
against a known-good emitter. A planted emitter he controls — a CYRF6936
module or an 802.11b card into a step attenuator, and Cricket for LoRa —
walked down in 2 dB steps until Undertow stops; the recorded depth, the
time-to-detect at each step, and the false-alarm rate on the empty band
over a night are the three numbers the section is judged by, per handle
and fused, with one receiver and with two. Then: the planted emitter at a
depth the blind chain cannot see, found, bootstrapped, catalogued, powered
off, powered on an hour later at 10 dB less — found by acquisition from
the catalogue. GPS as the free satellite case: every visible PRN's TDOA
between two nodes against the ephemeris. And the lattice's refusals:
an OFDM signal, two overlapping emitters and a CW tone each produce a
disagreement, not a detection.

**Uses.** Bug Sniffer's park mode and sweep (the design case: a spread bug
under the floor, found, fingerprinted, bearing and hyperbola on the map);
Drone Discovery (an unknown control link catalogued by chip rate and ppm
pair before anyone knows the vendor); the Signals bench (the survey view
— cells lit by their verdict, colour by which handle fired, provenance in
the tooltip); atkdf (despread-then-DF and the TDOA layer); the fingerprint
features (C5); the long-term baseline (B5) as its memory.

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

**C1.1 Asks from ATK's live inspector (2026-10-30;
`ATK\docs\SIGDIGGER_PARITY_PLAN.md` §L).** The inspector is the first
streaming consumer of `atkdsp_lindemod`. It runs live with the operator
turning knobs, so it needs:

1. **`atkdsp_lindemod_set(d, cfg)`** — every parameter (matched filter, TED,
   loop BnT and ζ, Costas order, equalizer mode and μ) changes at the next
   symbol boundary **without a reset**. A reset is a separate call. Test:
   change BnT mid-stream on a locked QPSK; no unlock, and the MER dip is
   recorded.
2. **Per-block lock metrics:**
   - timing-error variance;
   - a carrier lock indicator (the mean of cos(M·φ) for Costas);
   - MER since the last call;
   - the current frequency estimate (FLL/PLL), in Hz;
   - the samples-per-symbol estimate.

   These come back in a struct filled by `atkdsp_lindemod_metrics`. They
   drive the inspector's lock lamps.
3. **Each stage callable alone,** as the section already says
   (`atkdsp_timing_*`, `atkdsp_carrier_*`, `atkdsp_eq_*`), so the inspector
   can show the signal after each stage (constellation before and after the
   equalizer).
4. **The equalizer taps readable** (`atkdsp_lindemod_taps`, already planned).
   The inspector draws them as the channel's delay profile.
5. **Cost known.** The README records µs per symbol for each configuration
   on Bill's PC (14 cores, AVX2), so ATK can budget how many inspectors run
   at once.
6. **Chunk invariance** with blocks of 1 symbol's worth of samples, because
   the inspector feeds whatever the queue holds.

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

### C6. Analog video — composite to pictures (2026-10-30)

**BUILT in v0.25.0 / ABI 26** (header §32, twin `reference.py` §32, 51 tests
in `tests/test_video.py`, a smoke block) — staged by the instance that
reviewed SigDigger, merged here.

Specified in full in `docs/VIDEO_PLAN.md`. In short:
- **Input:** real composite samples after the FM or AM detector.
- **Output:** luma frames with lock metadata.
- **Inside:** a sync slicer at the standard's 50 % point, with sub-sample
  edges; a second-order line-locked loop that coasts through the vertical
  interval; field detection and order; 525 / 625 / measured-custom
  standards; a chroma trap or comb; motion-aware frame stacking.

**Written from the standards, not from sigutils' `tvproc.c`.** Rule 7:
LGPL is a test oracle only, and the sigutils repo's top-level licence says
GPL-3. A stock SigDigger may be run as a black-box oracle on our synthetic
files.

**Consumers:** ATK's viewer (`atk/core/video_rx.py`), Bug Sniffer's
"Look", and Drone Discovery's FPV picture.

**Negative control that matters:** a 15,734 Hz pulse train (radar-like) must
give H-lock at most, and never a frame.

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
| A4a ✅ v0.19.0 | **Channelizer part 1:** the PFB (uniform grid, coherent N-element `_process_n`), W1 covariance, W2 power — the DF-everything core (BUILT 2026-10-29) | M | `atkdf` is blocked on it; overlap-save extractor (A4b) is a later ABI |
| A4b ✅ v0.20.0 | **Channelizer part 2:** the overlap-save extractor (`atkdsp_osx`, arbitrary channels, coherent N-element, per-channel resample) (BUILT 2026-10-30) | M | Operator/decoder-clocked/off-grid targets; completes A4 |
| 5 | **Demod and quality:** C1 linear chain + equalizer, C2 EVM/MER | L | Completes the soft path into phase 1's Viterbi |
| V ✅ v0.25.0 | **Analog video:** C6 (`docs/VIDEO_PLAN.md`) — twin + generator, then C (BUILT 2026-10-30) | M | Independent of every other phase; Bug Sniffer's "Look" and Drone Discovery's FPV picture wait on ATK's viewer |
| 6a ✅ v0.18.0 | **Spread spectrum:** B6 blind + known-code paths, B6b LoRa, B6c's receiver pair / align / lattice (BUILT 2026-10-29) | L | Uses B4, D2; the survey + catalogue of B6c wait for A4 and B5 |
| 6 | **Below the floor (rest):** B6c's scale-space survey and catalogue, C4 strong-signal removal, MDC/FleetSync (3b), the learned fuser | M–L | After A4 (the channelizer) and B5 (the baseline) |
| 6b-1 ✅ v0.21.0 | **Undertow piece 1:** B6c's scale-space survey over the dyadic PFB — fluctuation, ACF and scale-window cyclic handles per cell, survey-wide Pfa, merged hits (BUILT 2026-10-30) | M | Single receiver; ACF is −10 dB-walled, the fluctuation handle carries the depth |
| 6b-2 ✅ v0.22.0 | **Undertow pieces 2, 4, 6 (kernel):** survey candidates, the period-synchronous search with period-rate refinement, the bootstrap to the code, the hit blob; `dsss_code`'s offset fix (BUILT 2026-10-30) | M | −18 dB to the named code in 2 s at the real rate; ~25 s a candidate on 2 cores |
| 6b-3 ✅ v0.23.0 | **`dsss_code` deeper:** 2-adic decimation averaging, the Frobenius parity stack, the window-boundary tie fix, regeneration from a named generator (BUILT 2026-10-30) | S | 127 chips, −16 dB, 150 periods: 17/40 exact → 40/40 |
| 6b-5 ✅ v0.24.0 | **Undertow piece 5, the full lag scan:** every period of a cell by the FFA over lag products, a saddlepoint null from the cell's own correlation, families by nested folds and CLEAN (BUILT 2026-10-30) | M | −15 dB over 4 s at 60 kS/s; −18 dB over 2 s at 600 kS/s; 14 s a 600 kS/s cell on 2 cores |
| 6b | **Undertow (rest):** B6c — the radiometer in the cells, the full blind lag scan (BUILT v0.24.0, CPU), the catalogue and survey view (ATK's LPI/LPD page, BUILT) | L | After B6, A4 (the dyadic bank), B5 (the baseline); two receivers for the centre piece; Bill's planted emitter and a step attenuator for the acceptance |
| 7 | **The long watch and identity:** B5 occupancy + baseline, C5 fingerprint features | M | Uses B2, C2 and C3 |
| 8 | **Intercept audio:** F1, F2, and the F3 wiring into the Workshop | M | Independent; the Workshop's radio door is waiting for it |
| 9 | **Propagation:** E1 ITM | M | Independent; moves up if DF footprints are wanted sooner |
| — | F4 MVDR listening | M | After atkdf's calibration path |

Phase 5 (C1) also unblocks ATK's live inspector (`ATK\docs\SIGDIGGER_PARITY_PLAN.md` §L); its asks are C1.1.

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
