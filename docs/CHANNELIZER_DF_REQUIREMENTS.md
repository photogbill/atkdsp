# Channelizer (A4 / A2.2): what direction finding needs from it

> **Written 2026-10-29 for whoever builds the channelizer tonight.** Companion
> to `ATKDSP_PLAN.md` §A4 and `DSP_EXPANSION_PLAN.md` §A2.2 — those say what
> the channelizer is for the decoders and the watchlist; this says what it
> must ALSO do so the same kernel serves `atkdf` (ATK's DF engine) on the
> KrakenSDR's five coherent streams and, later, on larger phased arrays.
> Nothing here changes the ABI already sketched in §A4; it adds a
> multi-element form and three small outputs. Bill, 2026-10-29: *"I'm
> building the channelizer later tonight"* — and *"put it in the DSP folder
> if it needs to be built in there."* The DF plan that asks for this is
> `ATK\docs\DF_P25_IMPROVEMENT_PLAN.md`.

## 1. Why DF cares

Today `atk/core/df_live.py` direction-finds ONE target band per 50 ms Kraken
frame. The Kraken's 2.4 MHz window holds **192 channels at 12.5 kHz** (P25
Phase 1/2, DMR) or 384 at 6.25 kHz (NXDN, P25 narrow). Every retune costs a
noise-source calibration — first light (`atkdaq\bench\first_light.log`)
measured ~0.2–0.6 s blind per hop because the tuner PLL phase is arbitrary on
each relock — so **the cheapest DF is the one that never retunes**: sit in a
window and DF every active channel in it at once ("DF everything",
`CUED_DF_PLAN.md` §5.5). That needs the five element streams split into
channels *coherently*, a 5×5 covariance accumulated per channel over each
burst, and an eigendecomposition per channel. Channelising before the
covariance is also the ≈23 dB in-channel processing gain the sensitivity
budget counts on (`CUED_DF_PLAN.md` §7A).

The DF engine does not care whether the channelizer is the uniform PFB (form
b) or the overlap-save extractor (form a): "DF everything" uses the PFB; a
decoder-clocked or operator-chosen target uses the extractor. Both forms need
the multi-element guarantees below.

## 2. Hard requirements (coherence — a bearing error if violated)

**R1. Multi-element processing with ONE block alignment.** Either an explicit
API —

```c
/* N element streams, sample-aligned, identical length; one shared block
   clock; out[el][chan][m]. nel = 5 for the Kraken; up to 32 for arrays. */
ptrdiff_t atkdsp_pfb_process_n(atkdsp_pfb *p, const atkdsp_cf32 *const *ins,
                               unsigned nel, size_t n,
                               atkdsp_cf32 *out /* [nel][nchan][m] */, size_t cap);
ptrdiff_t atkdsp_osx_process_n(atkdsp_osx *x, const atkdsp_cf32 *const *ins,
                               unsigned nel, size_t n,
                               atkdsp_cf32 *const *outs /* [nel][ch] */,
                               size_t *out_counts, size_t cap_each);
```

— or a documented, tested guarantee that N separate instances created with
identical parameters and fed sample-aligned input produce outputs with
identical block boundaries and an identical phase reference (no per-instance
state that depends on anything but the parameters and the sample index). The
explicit API is preferred: it is one FFT plan, one prototype, one block
counter, and it makes R2 true by construction.

**R2. Same prototype filter, same per-block bin rotation, on every element.**
The §A4 phase-continuity correction ("a per-block bin-rotation correction")
is a function of block index and channel, never of the data, and must be
applied identically to all elements. Any per-element difference — a different
tap set, a different start block, a rotation computed from the element's own
samples — is a per-element phase error that lands directly on the bearing.

**R3. Sample-index bookkeeping, stated.** For output block *b* of channel *k*,
the input sample it is centred on (`s0 + b·hop − group_delay`), and the
prototype's group delay in input samples, exposed by a function (e.g.
`atkdsp_pfb_block_origin(p, b)` / `atkdsp_pfb_group_delay(p)`). atkdf
addresses bursts by stream sample index (`atkdf.identity.spans`, the atkdaq
header's `stream_sample0`), and a decoder-clocked span must map to the same
channel-domain samples on every element.

**R4. Oversampled uniform grid, configurable spacing.** PFB 2× oversampled
(output rate = 2 × channel spacing) as §A4 already says — a signal on a
channel edge or slightly off-centre (P25 VHF/UHF plans use a 7.5 kHz / 6.25
kHz raster; rebanded 800 MHz uses 6.25 kHz offsets) must not be split or
aliased. Spacing selectable: 12.5 kHz, 6.25 kHz, 25 kHz, and generic. At
2.4 MSPS / 12.5 kHz: 192 channels, 25 kHz out (≥ 2 × 6000 sym/s P25 Phase 2,
≥ 4 × 4800 Phase 1 — enough for `dmr_sync.c` and the P25 sync correlators to
run on the channel output directly).

**R5. Per-channel calibration hook (or an explicit statement that DF does it
after).** atkdaq's receiver calibration is a complex weight plus a fractional-
delay phase SLOPE per element (`atkdf.calibration.band_factors`). On a
channelised output it collapses to one complex factor per element per channel
at the channel's centre frequency — an `(nel × nchan)` table, one complex
multiply per output sample. Either accept the table (`atkdsp_pfb_set_cal(p,
const atkdsp_cf32 *table)`) or leave it to Python; both are fine, but say
which, so it is applied exactly once.

**R6. No data-dependent state in the channel path.** No AGC, no DC tracking,
no per-channel normalisation inside the PFB/extractor — `dc_track` and `agc`
(ABI 13) break inter-element phase and amplitude and belong downstream of the
covariance, never upstream. Chunk invariance (§A4) must hold per element.

**R7. Numerical statement.** float32 is fine. Cross-check: identical input on
all N elements → outputs bit-identical across elements; a planted plane wave
across 5 synthetic elements → the inter-element phases at the channel output
equal the planted phases within 0.1° after the group delay is accounted for.

## 3. Strongly wanted (each is small; together they are the DF-everything engine)

**W1. Per-channel covariance accumulator in C.**

```c
/* R[k] += sum over m of y_k(m) y_k(m)^H for every channel k whose gate bit
   is set (all channels when gate == NULL); counts[k] += m. R is
   [nchan][nel][nel] complex, Hermitian; only the upper triangle need be
   computed. Reset per channel on burst start (see W3). */
void atkdsp_pfb_cov_accumulate(const atkdsp_cf32 *out /* [nel][nchan][m] */,
                               unsigned nel, unsigned nchan, size_t m,
                               const uint8_t *gate, atkdsp_cf32 *R,
                               uint32_t *counts);
```

Cost: nel² complex MACs per output sample per channel — 25 × 2.4 M × 2
(oversample) ≈ 120 M cMAC/s for the Kraken at full occupancy; 16 elements
≈ 1.2 G cMAC/s, still one core. atkdf then does one 5×5 (or 16×16) `eigh` per
active channel per report interval, which is nothing. Without W1 the Python
side forms R from the channel streams itself (numpy, fine for the Kraken,
marginal for a 16-element array at 20 MHz).

**W2. Per-channel, per-element power per block** (mean |y|²): feeds the
channel-grid CFAR (`atk/core/cue_detector.py` runs today on display bins),
the DF-everything gate, and the per-channel level display that shows a dead
or swapped antenna on one channel before anything else does
(`doa.band_snapshots` returns `channel_db` for this reason).

**W3. Burst edges on the channel grid, with sample indices.** If the CFAR /
burst tracker (ABI 17) can run on W2's channel powers and report OPEN /
CLOSE per channel with the input sample index, then W1's accumulator resets
at OPEN, reports at CLOSE and every *T* ms in between, and atkdf gets one
bearing per burst with the whole burst integrated — the "integrate the whole
burst" row of the sensitivity budget — instead of one bearing per 50 ms
frame as now. This is the piece that turns a 3 s P25 PTT into tens of
thousands of snapshots.

**W4. The extractor (form a) multi-element too.** The operator's target, a
decoder-clocked burst, or a channel on an off-grid raster goes through the
overlap-save extractor at an arbitrary centre; it needs the same R1–R3
guarantees. Retuning or adding one extractor channel must leave the others
bit-identical on every element (§A4's acceptance, extended to N).

**W5. Nothing for TDMA.** P25 Phase 2 slots are separated downstream from the
sync correlator's timing; the channelizer only has to keep the output rate
≥ 2 × symbol rate (R4 does).

## 4. Sizing, so nothing is over-built

| | Kraken | phased array (planning number) |
|---|---|---|
| elements | 5 | 8–32 |
| input | 5 × 2.4 MSPS, int8 → cf32 | N × 10–50 MSPS |
| channels @12.5 kHz | 192 | 800–4000 |
| PFB out | 192 × 25 kHz × 5 | N × nchan × 25 kHz |
| W1 cost, full occupancy | ≈120 M cMAC/s | ≈1–20 G cMAC/s (needs the gate; typical occupancy is < 10 %) |
| eigh per channel, 10 Hz | 1920 × 5×5 /s — negligible | nchan × N×N /s — numpy fine to N=16; torch for more |

## 5. Acceptance DF adds (sim, no hardware — the planted-truth rule)

1. **Planted plane wave through the PFB.** Five synthetic elements (atkdf's
   `sim.py` makes them, UCA-5 r = 0.18 m, 460 MHz), one 12.5 kHz emitter at
   bearing θ in channel k: MUSIC on R_k from W1 (or from the channel streams)
   returns θ within 0.1° of MUSIC on the un-channelised band-limited
   snapshots (`atkdf.doa.band_snapshots`), and the spectrum peak is in
   channel k only.
2. **Two emitters, two channels, two bearings.** 60 dB level difference,
   adjacent channels: the weak one's bearing moves by < 0.2° with the strong
   one present (stopband is doing its job across elements, not just on one).
3. **Chunk invariance across elements.** The same stream chunked 3 ways on
   element 2 only → element 2's outputs bit-identical to the other chunkings.
4. **Extractor retune.** Add/retune/remove one extractor channel → every other
   channel's output on every element bit-identical.
5. **Numpy twin.** `python/atkdsp/reference.py` carries the N-element form
   and the accumulator; `atkdf` tests run against the twin when the DLL is
   absent, as every other kernel does.

## 6. How atkdf will use it (so the shape of the API fits)

```
atkdaq frame (5 ch, s0, header) ──▶ cal table from header (R5) ──▶ PFB_n (R1)
      ──▶ W2 powers ──▶ CFAR on the channel grid ──▶ gate / OPEN / CLOSE (W3)
      ──▶ W1 accumulate R_k over the burst ──▶ atkdf.doa.estimate(Covariance(R_k, counts, f_k))
      ──▶ one BearingRecord per burst per channel (keyed by channel until the
          decoder names it; sync-aided steering vector from dmr_sync.c on the
          channel output when it fires) ──▶ registry / posterior / map
```

The scheduler (`atkdf/scheduler.py`, built, not yet live) keeps choosing
*which window* the Kraken sits in; inside the window the channelizer makes
every channel free. That is the whole point.
