# atkdsp — Signals-Analysis Expansion Plan

Status: PLAN, 2026-10-27. Owner: Bill. Drafted with Claude.
Current library: v0.13.0 / ABI 14 (built clean, 361 cross-check tests pass).

This plan covers the kernels Bill asked for on 2026-10-27:

- **Build now:** README roadmap items 2, 3, 4, 5, 6, 7, 9, and keep the C ring
  buffers (roadmap item 3 of the upper list). Plus the bitstream workbench
  traditions from martes / xbit / X-Midas.
- **Plan then build:** DSSS detect + chip-rate, analog radio IDs, comms burst
  descriptors, adaptive equalizer + EVM/MER, general CRC + polynomial recovery,
  long-term occupancy + baseline, strong-signal removal, TX fingerprinting,
  intercept audio cleanup.
- **Plan then build (added 2026-10-28):** Part C — search kernels from radio
  astronomy and SETI: spectral kurtosis, drift-rate search, periodic search,
  single-pulse search, faint-streak search, correlation radiometer.
- **Separate track:** audio forensics → the Digital Forensics Workshop, NOT
  atkdsp, and NOT SAM. (Intercept cleanup reuses the Diffusion Toolkit's
  existing `test_classical_denoise` / `test_diffusion_denoiser` /
  `repair_audio_inpaint` paths.)

## The rules every kernel below inherits (atkdsp.h, unchanged)

1. C99, C ABI, no globals, no callbacks.
2. **No allocation in a processing call.** `*_create` allocates once.
3. **Streaming state is explicit and continuous** — carried by the caller's
   handle; `test_cross_check.py` feeds blocks of 1, 7, 999, 4096 … and asserts
   the result equals one unbroken call.
4. Numbers in, numbers out. No I/O, no printing.
5. `ATKDSP_ABI_VERSION` bumps on any change to an existing function.
6. **Every kernel has a numpy twin with the same semantics** (`reference.py`),
   and ATK falls back to the twin when the DLL is absent.
7. Not `-ffast-math`: NaN means "unmeasured".

Plus Bill's standing rules: comprehensive debug logging; no writes to `C:\` or
AppData; results surfaced striking and **color-coded by how they were
produced**; built to the 18-year-old-operator level ("Bill proof").

## Honest scope note

This is ~15 kernels, several of them projects in their own right (ITM, a
general Viterbi, LDPC later). The library's whole value is that a kernel is
**validated against a real signal**, not just a round-trip — a confident wrong
answer is the one failure mode we refuse. So this is phased, each phase is its
own ABI bump with twins + cross-check + a real-capture acceptance, and nothing
goes live in ATK until its acceptance passes. Build order is by dependency and
by what the DF / Bug-Sniffer / Drone work is waiting on, not by Bill's list
order.

---

# PART A — build now

## Phase A1 — foundation (low risk, everything else leans on it)

### A1.1 `atkdsp_fft` float32 path  (roadmap 9a)
- **What:** a radix-2 float32 FFT behind the existing `atkdsp_fft_*` ABI for
  power-of-two sizes; keep the pocketfft double path for everything else.
- **ABI:** no new functions; a flag on `atkdsp_fft_create` or auto by size.
- **Twin:** `np.fft` at float32. **Cross-check:** agreement to float tol vs the
  double path; NaN-handling rule preserved.
- **Accept:** display reduction unchanged to the eye on a real capture; cost
  drop measured on `bench_hotpath.py`.

### A1.2 Mixer folded into the first decimating stage  (roadmap 9b)
- **What:** pre-rotated complex taps on the DDC's first FIR stage so the NCO
  pass disappears (README measured ~0.05 CPU-s/s/channel saved; a 16 MB read +
  16 MB write per channel per block removed).
- **ABI:** internal to `atkdsp_ddc`; `describe()` prints the folded plan.
- **Accept:** DDC output bit-close to the current path; interferer still 80 dB
  down; cost drop measured.

### A1.3 Ring-buffer primitives  (KEEP — roadmap 3)
- **What:** C block-ring: sequence numbers, multi-consumer cursors, overflow
  accounting, for the §RF-F2 ring / DVR. Built now because atkpr and atkdaq
  (both C) will consume the same ring.
- **ABI:** `atkdsp_ring_create/push/claim/release/stats`.
- **Twin + cross-check:** a Python producer/consumer mirror; overflow counts
  match; no torn blocks under uneven push/claim sizes.

**Phase A1 → v0.14.0 / ABI 15.**

## Phase A2 — detection + channelization (the DF / Bug-Sniffer blockers)

### A2.1 `atkdsp_cfar`  (roadmap 3)
- **What:** cell-averaging CFAR per bin with guard cells and a per-profile
  false-alarm rate, over the §RF-F3 spectrum frames; merges adjacent bins into
  one detection `{fc, bw, level, t0, t1}`; a mask removes known-emitter
  frequencies and operator-muted spans. Add **OS-CFAR** (ordered-statistic) for
  clutter edges. Streaming: t0/t1 carried across block edges.
- **Twin first** (CUED_DF_PLAN §5.1 says numpy twin before C).
- **Accept:** on a 20 MHz bladeRF capture with planted bursts, every burst
  above the set SNR → one detection, fc within a bin, t0 within one frame; a
  masked repeater → none; measured Pfa matches the configured rate.
- **Feeds:** `cue_detector.py` (cued DF) and the Bug-Sniffer trigger.

### A2.2 `atkdsp_pfb` + `atkdsp_osc_extract`  (roadmap 1/2 — the channelizer)
- **DF requirements (2026-10-29): read `docs/CHANNELIZER_DF_REQUIREMENTS.md` first** — N-element processing on one block clock, sample-index bookkeeping, per-channel covariance accumulator, channel-grid powers for the CFAR. It is what makes the PFB the Kraken's "DF everything" engine.
- **What, two forms:**
  - **Uniform polyphase filterbank** — N equally-spaced channels from one input
    FFT + polyphase prototype, so the 50-entry watchlist costs ~one channel.
  - **Overlap-save per-channel extractor** sized from the channel, for when the
    display FFT's bins are too coarse (README: 19.5 kHz bins vs a 15 kHz
    channel at 40 MSPS / 2048-pt).
- **ABI:** `atkdsp_pfb_create(fs, nchan, taps_per_chan, atten_db)` →
  `_process` giving nchan decimated streams; `atkdsp_osc_extract(fc, bw)`.
- **Twin + cross-check:** each output channel bit-identical across chunking; a
  tone in channel k appears only in k at ≥ X dB isolation.
- **Accept:** real wideband capture, known emitters land in the right channels
  at the modeled isolation.

**Phase A2 → v0.15.0 / ABI 16.**

## Phase A3 — quick identifiers + pulse descriptors (small, high-value)

### A3.1 `atkdsp_cp_probe`  (roadmap 5a)
- OFDM cyclic-prefix autocorrelation at lag = FFT length, integrated,
  chunk-invariant → "OFDM · SCS x kHz" caption with a ratio-to-window-energy
  strength (gain-independent).

### A3.2 `atkdsp_cyclic_line`  (roadmap 5b)
- m-th-power + Goertzel at a fixed candidate-rate list (4800 / 2400 / 1200 /
  512 / 18000 …) → "digital at 4800 sym/s" **hint for routing, never an ID**
  (dsd-neo still says what it is).

### A3.3 `atkdsp_pulse_runs`  (roadmap 6)
- Envelope → threshold (block median) → run-length + per-segment sums → PDWs,
  run carried across the block edge (a pulse split by a boundary stays one).
- **Accept:** the METEOR-500C / known-radar cases in `pulse.py` reproduce; live
  worker shows the same PDWs as the one-shot.

All three: twins + cross-check; validated by adding noise to Bill's B13
recordings and finding where each probe stops.

**Phase A3 → v0.16.0 / ABI 17.**

## Phase A4 — the bitstream / FEC workbench  (roadmap 7 — Bill's priority)

> Bill, flagged extremely important. The design target is **BitView / xbit /
> martes-class, and better**: the analyst MEASURES a stream, then CHANGES it
> through a reversible, re-runnable chain, in the notation they already think
> in. `bits.py`, `bitchain.py` already hold the start (the `t/s/r/i/@` pattern
> language, Manchester/NRZI/descramble, LFSR search). This phase adds the hot
> loops in C and the traditional toolbox around them.

### C hot loops (atkdsp)
- **`atkdsp_viterbi`** — general: any constraint length K, any generator
  polynomials, puncturing / de-puncturing, soft input, **tail-biting**. Retires
  the LTE-only Python `viterbi_tb` in `lte_sib1_rx.py`.
- **`atkdsp_rs`** — Reed-Solomon over GF(2^m): any (n, k), configurable
  primitive polynomial, generator root (fcr) and symbol size. Berlekamp-Massey.
- **`atkdsp_bch`** — BCH over GF(2^m): any (n, k, t).
- **`atkdsp_conv_encode`** — the matching encoder, for test vectors.

### Traditional toolbox (ATK numpy / twins) — the xbit / martes / X-Midas set
- **Line codes:** NRZ-L/M/S, NRZI, RZ, bi-phase-L/M/S (Manchester + variants),
  Miller / delay modulation, CMI — encode and decode, and **detect** which.
- **Framing / destuff:** HDLC bit de-stuffing, flag search, frame-length
  autocorrelation ("something repeats every 255 bits").
- **De-interleavers:** block (reshape an r×c matrix you can resize live),
  convolutional, helical; plus interleaver-depth detection.
- **Scramblers:** additive and multiplicative LFSR, descrambler/polynomial
  search (extend the existing `descramble_search`).
- **CRC:** general CRC 1–32 (poly, init, reflect-in/out, final XOR) — moves to
  Phase B6's recovery engine for the brute/solve search.
- **Alphabets:** Baudot / ITA-2, ITA-3 (ARQ), ITA-5 / ASCII, EBCDIC, Varicode,
  CCIR/numeric paging, ASCII-with-parity — with auto bit-order / polarity /
  parity trials.
- **The X-Midas / martes habits that make it a workbench, not a function list:**
  - the **editable bit raster** — selectable word width, display base
    (bin/oct/hex), MSB/LSB first, scroll, column-align to a found period;
  - the **reversible analysis chain / "analysis set"** — ordered steps where
    changing one re-runs everything after it, **saved and reloadable** (this is
    BitView's signature and what Bill asked for);
  - one-click **polarity / endianness / differential / inversion** toggles that
    re-run the chain;
  - **symbol statistics**, run-length histograms, per-position bit-transition
    density, entropy (FEC-presence estimate), and a **negative control** on
    every search (the `bits.py` rule — report a found period only when the
    shuffled stream does not show it).
  - X-Midas BLUE interop is already covered by `bluefile.py`; keep the
    raster's import/export compatible with it.
- **FEC recognition stays honest:** report CANDIDATES and PERIODICITIES, never
  name a code it has not decoded (the `bits.py`/`emitters.py` rule).

- **Twins + cross-check:** every C decoder validated by encode→decode round
  trip AND against an independent reference stream (a known RS-coded frame, a
  known POCSAG/FLEX alphabet capture).
- **Accept:** Bill drives a real unknown-ish capture end to end — raster →
  find period → deinterleave → descramble → alphabet → readable — and the chain
  reloads and reproduces it.

**Phase A4 → v0.17.0 / ABI 18.** (Largest phase; may sub-split Viterbi / RS /
BCH into their own ABI bumps.)

---

# PART B — plan, then build

Each is twin-first, then C where a hot loop demands it, same acceptance
discipline (validated on a real or realistically-degraded signal).

### B1 — DSSS detect + chip-rate
Despreading / PN search, delay-multiply + cyclostationary chip-rate estimate,
PN-length estimate; below-noise detection. Extends `pulse.py`'s chip measurement
from inside-a-pulse to a free-running link. **Useful to Drone Discovery** (many
control links are spread/hopped).

### B2 — analog radio IDs
Label analog FM bearings by unit, the way DMR bearings already carry radio ID —
so the relief use-case covers analog handsets too:
- **CTCSS** (Goertzel bank over the 50 standard sub-audible tones),
- **DCS** (23-bit Golay-coded sub-audible digital),
- **DTMF** (dual-Goertzel), **two-tone / Select-5 / ZVEI / CCIR** paging,
- **MDC-1200** and **FleetSync** (1200-bps FFSK — reuses the FSK front end).
Twin-first; C only if a live bank is too slow.

### B3 — comms burst descriptors
The comms analogue of a radar PDW: `{t0, dur, fc, bw, power, rise, fall, slot}`
+ TDMA slot-structure estimate. Builds on A2.1 (CFAR) + A2.2 (channelizer).
Feeds pattern-of-life and TDMA analysis.

### B4 — adaptive equalizer + EVM/MER
Blind **CMA** (constant-modulus) and decision-directed **LMS/RLS** equalizers,
then **EVM / MER** on the equalized constellation. Without this, real-channel
PSK/QAM constellations stay smeared after timing/carrier recovery. C for the
tap-update loop.

### B5 — (folded into A4's CRC + here) general CRC + **polynomial recovery**
Brute search over poly/init/reflect/xorout (C for speed) **and** the
linear-algebra recovery of an unknown CRC from a handful of known-good frames
(CRC RevEng-style). One of the most-used unknown-protocol tools.

### B6 — long-term occupancy + baseline
Per-bin busy-fraction accumulation over hours → a learned baseline → deviation
alerts ("this wasn't here yesterday"). This is **Bug-Sniffer's leave-it-for-
weeks job** and survey pattern-of-life. Check whether the bugsniffer repo
already has a Python version before building in C.

### B7 — strong-signal removal / excision
Estimate and subtract a strong carrier (notch / projection / successive
interference cancellation) so a weak emitter beside it appears — a classic
place to hide a bug, and a general co-channel helper.

### B8 — TX fingerprinting (SEI / RF-DNA)
Per-burst features: turn-on transient, frequency drift, power ramp, I/Q
offset/imbalance signature → features for the **RFMLS training tab**.
**Honest framing:** good at telling a few known radios apart, not at
identifying unknown ones. Research-grade, labeled as such.

### B9 — intercept audio cleanup
Hum/notch (50/60 Hz + harmonics), bandpass, de-emphasis, AGC, and
spectral-subtraction / Wiener denoise before Whisper and the headphones.
**Reuse the Diffusion Toolkit** — it already has `classical_denoise`,
`diffusion_denoiser` and `repair_audio_inpaint`. atkdsp supplies only the
cheap signal-level filters; the learned denoise is a toolkit call.

---

# PART C — radio-astronomy search kernels (plan, then build; added 2026-10-28)

Bill, 2026-10-28, on fishing radio astronomy for ways to find weak,
hard-to-see signals: *"I think the ones that you recommend, we should add.
Even if they are only for specific scenarios, just make sure that when they
hover over that option, that text appears explaining when and how it should
be used."*

Radio astronomy and SETI have spent decades pulling signals out from under
the noise. What transfers to ATK is their **search over unknown parameters**
(drift, period, width) and their **noise statistics**; imaging does not (it
assumes a sky that holds still for hours and many antennas). These kernels
are the hot loops; how the detector, Sift, DF and passive radar use them —
and the **hover text** each operator option carries — is in
`ATK_Diffusion_Toolkit/DETECTION_DESIGN.md` §4.6, `atkdf/docs/ATKDF_PLAN.md`
§13 and `atkpr/docs/ATKPR_PLAN.md` §6A.18. Same discipline as Part B:
twin first, C where the loop demands it, every threshold derived or measured
(never tuned), every search with a negative control in its test, and a real
or realistically degraded acceptance before anything goes live.

### C1 — `atkdsp_sk` spectral kurtosis
- **What:** per FFT bin, accumulate S1 = Σ P and S2 = Σ P² over M frames;
  the generalized estimator (Nita & Gary 2010, N = 1, d = 1)
  SK = (M+1)/(M−1) · (M·S2/S1² − 1). Gaussian noise → 1; impulsive or
  intermittent energy → above 1; constant envelope → below 1 (a pure tone
  → 0). Thresholds for a two-sided flag come from the estimator's own
  distribution at the configured false-flag rate, not by hand.
- **Out:** per-bin SK and a flag mask; streaming — the accumulators carry
  across block edges.
- **Feeds:** the toolkit's noise floor (`dsp/floor.py`: flagged bins are
  left out of the estimate), an `SK↑` flag on boxes, and excision of
  impulsive cells before long integrations (Sift dwells, the escalation
  buffer, atkdf's covariance, atkpr's CPI).
- **Blind spot, said in the docstring:** OFDM, LTE and anything noise-like
  look Gaussian to it.
- **Accept:** on terminated captures the flag rate matches the configured
  rate; a planted low-duty pulse train is flagged where the time-averaged
  power shows nothing; a tone reads below 1; a real capture with a known
  impulsive source (a switching supply, ignition noise) is flagged.
- First of Part C: the cheapest, and three consumers wait on it.

### C2 — `atkdsp_drift_tree` drift-rate search (SETI's Taylor de-Doppler tree)
- **What:** given N_t floor-normalised power spectra, sum power along every
  straight track drifting up to one bin per frame (0 … N_t bins across the
  block; negative drifts by mirroring), for every start bin, in
  N_f·N_t·log2(N_t) operations instead of N_f·N_t²; faster drifts by
  shifting or scrunching the block. A carrier that drifts keeps its full
  integration gain instead of smearing across bins. The same tree runs
  across Sift's per-segment statistics, summing along drifting α tracks
  instead of one α bin.
- **Out:** per (start bin, drift rate) summed power; threshold from the
  Gamma(N_t) distribution of summed exponential cells at Pfa / trials.
- **Feeds:** Sift (the drift loss its review measured — 13.9 → 4.8 for a
  carrier drifting 4 Hz over 4 s, `DETECTION_DESIGN.md` §4.4), the Bug
  Sniffer (cheap oscillators warming up), drift rate as a fingerprint
  feature (B8); a large drift rate is a linear sweep, so sweep jammers and
  LFM chirps fall out of the same search.
- **Accept:** Sift's own drift case, with the margin recovered measured
  against the 13.9 / 4.8 pair and reported (not claimed); noise-only
  false-hit rate matches the configured rate; a linear sweep found at its
  rate.

### C3 — `atkdsp_periodic` periodic search (pulsar methods)
- **What, four tools on the floor-normalised power or envelope series:**
  harmonic summing on its FFT (1, 2, 4, 8, 16 harmonics) for narrow
  repeating pulses; epoch folding at a candidate period with a χ²
  significance against a flat profile; the fast folding algorithm (FFA)
  for long periods (~0.5 s and up) where harmonic summing is weak; a small
  acceleration (period-derivative) grid for a moving emitter.
- **The trap from the bench, inherited:** the tallest peak is a harmonic as
  often as the fundamental — score the comb, report the fundamental.
- **Overlap check before building:** ATK's `siga` pulse/PRI code and A3.3
  `pulse_runs` need each pulse visible; this works *below* single-pulse
  detection. Read `pulse.py` first for anything that already folds (not
  checked 2026-10-28).
- **Feeds:** beacons, radar PRIs, TDMA frames, switching supplies,
  telemetry framing — Sift, the Cuts viewer, Drone Discovery.
- **Accept:** a planted low-duty pulse train found at a per-pulse SNR where
  no single pulse is detectable; a noise-only negative control at the
  configured rate; a long-period beacon found by the FFA where harmonic
  summing misses it.

### C4 — `atkdsp_boxcar_ladder` single-pulse search (fast-radio-burst method)
- **What:** boxcar filters at widths 1, 2, 4 … 2^k frames on each channel's
  floor-normalised power; S/N normalised per width; best width wins;
  detections deduplicated across widths; threshold from Pfa / trials. A
  burst of any length lands within ~1.5 dB of a matched filter, where a
  fixed tile dilutes a short burst and splits a long one.
- **Feeds:** a proposer beside CFAR (A2.1); boxes carry the width.
- **Accept:** equal-energy bursts of 1 ms, 10 ms, 100 ms and 1 s all found
  within the ladder's predicted loss; noise-only rate matches.

### C5 — `atkdsp_sumthreshold` faint-streak search (AOFlagger's SumThreshold, turned around)
- **What:** flag runs of M = 1, 2, 4, 8 … consecutive cells along time or
  frequency whose mean exceeds χ_M = χ_1 / ρ^(log2 M), iterating with
  already-flagged cells clamped (Offringa et al. 2010). Astronomers use the
  flags to delete faint streaks; here the streaks are the find — long weak
  carriers and short broadband clicks that cell-by-cell CFAR misses.
- **Thresholds measured, not borrowed:** the published χ_1 and ρ are
  empirical; ours are set on terminated and quiet-band captures to the
  profile's false-alarm rate, with the measurement in the docstring.
- **Feeds:** a proposer beside CFAR; the floor mask; the Bug Sniffer.
- **Accept:** a continuous carrier below CFAR's per-cell threshold found;
  noise-only rate matches; compared with CFAR on the same capture.

### C6 — `atkdsp_xcorr_radiometer` correlation radiometer (radio interferometry)
- **What:** per bin, accumulate the cross-spectra X_i·X_j* over every pair
  of coherent channels (ten pairs on the Kraken) over long integrations.
  Each channel's own receiver noise does not correlate and averages toward
  zero as 1/√(B·T); only energy reaching both antennas survives. That
  sidesteps the noise-level uncertainty (Tandra–Sahai) that caps every
  single-receiver energy detector.
- **Out:** per-bin correlated power and its significance (from the
  measured spread of the pair products), and the pair phases (a rough
  bearing for atkdf).
- **What still correlates, said in the result:** noise that reaches every
  antenna (city noise, nearby electronics), and the Kraken's shared-clock
  spurs and board crosstalk — mapped first with terminated inputs, the way
  Sift maps the radio's own features, and masked.
- **Investigate, not planned:** the Kraken's calibration noise source as a
  Dicke-style reference for a gain-drift-free total-power mode — check on
  the hardware whether its switch isolates the antennas.
- **Needs:** coherent multi-channel frames from atkdaq.
- **Accept:** a band-limited Gaussian emitter far below the single-channel
  floor detected at the integration time the radiometer equation predicts;
  terminated inputs read zero after the spur mask; compared with
  single-channel total power at the same integration time.

**Build order inside Part C:** C1 → C2 → C4 → C5 → C3 → C6 (C6 waits on the
Kraken's coherent frames).

---

# Separate track — audio forensics (NOT atkdsp, NOT SAM)

Lands in **Digital Forensics Workshop** (`forensics_workshop`), which already
has `FORENSICS_PLAN.md` and a package. Candidate capabilities: **ENF**
(electric-network-frequency) authentication and timestamping, splice / edit /
insertion detection, codec / compression-history detection, channel &
noise-floor consistency, spectrographic analysis. Explicitly **no SAM**
(Segment Anything is for imagery elsewhere in ATK; wrong tool for audio).
To be planned in that repo's plan, not here.

---

# Dependency order & ABI map

```
A1 (fft32, mixer-fold, ring)        v0.14 / ABI15   ── foundation
A2 (cfar, channelizer)              v0.15 / ABI16   ── unblocks cued-DF, Bug-Sniffer
A3 (cp_probe, cyclic_line, pulse)   v0.16 / ABI17   ── small, high-value captions/PDWs
A4 (viterbi, rs, bch + workbench)   v0.17 / ABI18   ── Bill's priority; may sub-split
────────────────────────────────────────────────────
B1..B9                              v0.18+          ── twin-first each, as scheduled
C1..C6                              v0.19+          ── twin-first each; C1 SK first, C6 waits on the Kraken
ITM propagation (roadmap 4)         own target/ABI  ── C++ or C99, vendor NTIA, see §9.5
```

**ITM (roadmap 4)** is tracked as its own target under `src/itm/` with its own
acceptance (reproduce NTIA published test vectors); scheduled alongside A2
since DF range/coverage wants it, but it does not block the A-phase ABI chain.

# Open decisions for Bill
1. Build order: this plan runs **A1 → A2 → A3 → A4** (foundation first). If you
   want the **bitstream workbench (A4) first** because it's the priority, say
   so — it only needs A1, not A2/A3.
2. ITM in C++ (one CXX target) or ported to C99 — decide at build time.
3. Audio forensics: confirm it lives in Digital Forensics Workshop, and I'll
   plan it in that repo's `FORENSICS_PLAN.md`.
