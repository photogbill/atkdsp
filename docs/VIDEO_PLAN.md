# atkdsp §C6 — analog video: composite samples to pictures (plan, 2026-10-30)

> **For the instance working on atkdsp:** this file was staged in
> `D:\Analyst_Toolkit\file holding area` because atkdsp had uncommitted work
> in progress when it was written (Bill, 2026-10-30: *"put it into here, and
> I'll have the claude working on DSP incorporate it when it's finished"*).
> It belongs at `atkdsp\docs\VIDEO_PLAN.md`. The lines to add to
> `ATKDSP_PLAN.md` (Contents, §C, §G) are in `ATKDSP_PLAN_additions.md`
> beside this file.

**Status (2026-10-30, evening): BUILT in the holding area, waiting to be
merged.** The twin, the C kernel, the binding, the tests and the smoke block
are in `D:\Analyst_Toolkit\file holding area\video\`; `README_VIDEO_MERGE.md`
says where each goes. 51 tests pass in the cloud container (numpy twin; the C
built with atkdsp's own flags, -O3 AVX2 FMA, and again under ASan + UBSan);
the C equals the twin frame for frame. Not yet run on Windows or on a real
transmitter.

### As built — where the code differs from the plan below

The plan text below is kept as written; where they differ, **the code and
this list win**:

| Plan said | Built | Why |
|---|---|---|
| Slicer on the raw composite | Slicer on a **0.5 us box-smoothed** copy; edge times corrected by the box delay; the picture still rendered from the raw samples | FM noise is strongest at the band edge; without the box a noisy sync tip fragmented into short pulses and acquisition failed below about 20 dB CNR. With it, lock holds to 8 dB (13.5 MS/s) and 6 dB (20 MS/s) |
| Acquisition slice "5 % beyond on the sync side" | `q02 + 0.15 d`, hysteresis `0.05 d` (d = median - q02 of the last two smoothed segments) | Derived so the release level stays below blank for any picture (comment in `video.c`) |
| Line lock lost after 16 missing lines | **64** (4 ms) | A short RF dropout now coasts through without losing a frame |
| V-lock needs two consistent fields after every loss | After a loss, **one** field interval matching the field length seen before is enough | Same source coming back: relocks a field sooner |
| Chroma trap at the sample rate (fs >= 3 fsc) | A 31-tap Hamming band-stop on the **rendered pixel row** (13.5 MHz at 720 px) | Works at any input rate that holds the picture, and costs 31 taps a pixel rather than a long filter a sample |
| De-emphasis (F.405) in the kernel config | **Left out** of v1 | The F.405 network's values must come from the Recommendation text, which was not at hand; it belongs in ATK's front end (channel, FM detector, de-emphasis, low-pass) when it is added |
| Stacking gain "10 log10 N" | A **1/N exponential average** after the first N frames: about 10 log10(2N - 1) dB at steady state (measured 14.2 dB at N = 16; theory 14.9) | Keeps following a slow change instead of freezing |
| `stacked` = the minimum block depth | The **median** block depth | A frame counter or any moving thing pins the minimum at 1 |
| Config and meta as sketched in §4 | As in `video\atkdsp_h_video.h` (`atkdsp_video_config` 64 B, `atkdsp_video_meta` 64 B, `atkdsp_video_counters` 88 B, `atkdsp_video_frame(v, NULL, 0, &meta)` peeks the size) | The header is the ABI |
| Segment length unspecified | **S = round(fs x 64 us)**: every threshold and render happens at segment boundaries on the global index | Chunk invariance by construction, and C = twin decision for decision |

**Measured (cloud container, 2 slow cores):**
- 20 MS/s: 4-5x real time with the trap, 6x without.
- 13.5 MS/s: 5-6x.
- Geometry within 0.1 px of the planted picture; levels within 0.01 rms.
- Every negative control makes no frame: noise, an FM voice carrier,
  OFDM-like bursts, and a radar-like pulse train at the line rate (line lock,
  "no fields yet").

The umbrella plan is
`ATK\docs\SIGDIGGER_PARITY_PLAN.md` §V; the consumers are ATK's viewer
(`atk/core/video_rx.py`, `atk/ui/video_view.py`), Bug Sniffer's Look step
and Drone Discovery's FPV picture.

---

## 0. Why it is ours, and the clean-room line

Bill asked whether sigutils' `tvproc.c` could simply be put into atkdsp.
It cannot:

- **§1 rule 7.** LGPL code may be a test oracle only.
- **The licence is contradictory.** sigutils' repo `LICENSE` says GPL-3,
  while the file header says LGPL-3.
- **It isn't one file.** It needs a slice of sigutils' filters, types and
  logging.

So this kernel is written from the public standards.

- **Inputs allowed:**
  - SMPTE ST 170 (525-line composite);
  - ITU-R BT.1700 and BT.470 (the 525- and 625-line systems);
  - ITU-R BT.601 (active-line sampling);
  - ITU-R F.405 (FM television pre-emphasis);
  - Jack, *Video Demystified*;
  - Rice, *Digital Communications: A Discrete-Time Approach* (loop design);
  - `ATK\SigDigger_User_Manual.pdf`, as a description of behaviour only.
- **Not allowed:** reading the source of sigutils, suscan, SigDigger or
  SuWidgets — not even "for reference".
- **Oracle:** a stock SigDigger may be *run*, as a black box, on our
  synthetic files (§6.6). It is never vendored, linked or shipped.
- **Record your inputs:** the module docstring and the commit message name
  the inputs used.

**Where the exact numbers come from.** The figures in §2 are nominal and
approximate (≈). The builder copies the exact values and tolerances from
the standard text into the header comment, with clause numbers. No timing
constant enters the code from memory.

---

## 1. What it does, and what it does not

**Input:** real composite-video samples (`float32`) at `fs`. They come from
blocks atkdsp already has:
- FM video: OSX / DDC channel → `atkdsp_fm_demod` → low-pass;
- AM video: OSX / DDC channel → `atkdsp_am_demod` → low-pass.

The kernel does not channelize or demodulate RF. It starts at composite.

**Output:** frames of luma as `float32`, 0 = black and 1 = white,
`width × visible_lines`, each with a metadata struct (§4). The kernel applies
no gamma and no brightness or contrast; the viewer does.

**Rates:**
- `fs` ≥ 8 MS/s for full horizontal resolution.
- 4–8 MS/s works with less detail, and `describe()` says how many pixels per
  line are real.
- Below 4 MS/s, `create` refuses with `ATKDSP_EINVAL`, and the binding gives
  the sentence: *"composite video needs at least 4 MS/s; this is X"*.

**Out of scope for v1:**
- colour decoding (luma only, with a chroma trap or comb, §3.6);
- teletext and closed captions;
- digital video of any kind.

Colour is §8, a later ABI.

---

## 2. The signal, in numbers (nominal — exact values from the standards)

| Quantity | 525-line (NTSC-M family) | 625-line (PAL-B/G/D/I family) |
|---|---|---|
| Line frequency | 15,734.26 Hz (4.5 MHz / 286) | 15,625 Hz |
| Line period | ≈ 63.556 µs | 64.000 µs |
| Field rate | 59.94 Hz | 50 Hz |
| Lines per frame / per field | 525 / 262.5 | 625 / 312.5 |
| Horizontal sync width | ≈ 4.7 µs | ≈ 4.7 µs |
| Equalizing pulse width | ≈ 2.3 µs (half-line spacing) | ≈ 2.35 µs |
| Broad (vertical) pulse width | ≈ 27.1 µs | ≈ 27.3 µs |
| Equalizing / broad pulses per field | 6 / 6 / 6 | 5 / 5 / 5 |
| Horizontal blanking | ≈ 10.9 µs | ≈ 12.0 µs |
| Active line | ≈ 52.6 µs | ≈ 52.0 µs |
| Levels | sync −40 IRE, blank 0, black +7.5 (setup), white 100 | sync −300 mV, blank 0, black 0, white +700 mV |
| Colour subcarrier (trap/comb only in v1) | 3,579,545 Hz | 4,433,618.75 Hz |
| Reference point **0H** | 50 % point of the sync leading edge | the same |

**Custom** sources (cheap cameras, odd frame rates) are described by the
**measured** line rate and lines per field, with pulse widths scaled by the
line period.

---

## 3. The algorithm

Each stage below is a function of its own in the twin and a static function
in the C. Thresholds are **derived** (§1 rule 4), and each derivation is
written in the header comment beside its parameter.

### 3.1 Levels, AGC and polarity

- **Before lock: the sync-side tail.** Sync occupies about 4.7/63.6 ≈ 7.4 %
  of every line, so the sync tip is the extreme 7 % tail of the amplitude
  distribution.
  - The acquisition slice level is the value with 5 % of a two-line window
    beyond it on the sync side. This is derived from the sync duty cycle
    with margin for noise.
  - Blank is the median of the remaining samples nearest the slice.
- **Polarity.** Run the slicer on both tails. The true sync tail is the one
  whose crossings repeat at a line period between 15.0 and 16.5 kHz:
  measure the periodicity of the sliced signal at that lag.
  - `AUTO` picks the tail with the stronger periodicity.
  - It changes only after 2 consecutive fields of contrary evidence.
  - Ambiguous evidence is `UNKNOWN`, not a guess.
- **After lock: measured per line.**
  - The sync tip is the median of the samples inside the sync pulse (from
    0H + 0.5 µs to the trailing edge − 0.5 µs).
  - Blank is the median over the back porch. A colour burst is symmetric
    about blank, so the median is not moved by it — this is why the median
    is used.
  - Both are smoothed over 8 lines. 8 lines is the derivation: it averages
    noise by √8 and still follows a fade in under 1 ms.
- **AGC** scales so that tip→blank is the standard's sync amplitude. Per
  line, the **DC restore** subtracts that line's blank.
- **Noise** σ is the standard deviation over the flat part of the back porch
  (after the burst), smoothed over 64 lines. It drives §3.2's hysteresis and
  §3.7's motion test, and gives the SNR estimate in the metadata.

### 3.2 Sync slicer and pulse classes

- **The slice level** is 50 % between tip and blank. That level is the
  standard's own definition of 0H, which is why it is used.
- **Hysteresis** is ±3σ of the measured noise, clamped to 5–20 % of the sync
  amplitude.
- **Edge times** come from linear interpolation between the two samples that
  straddle the slice. This gives sub-sample precision; at 20 MS/s one sample
  is 50 ns, and §3.3's target is under 25 ns.
- **Width = trailing edge − leading edge.** Each pulse is classified by the
  nearest nominal width, with boundaries at the midpoints:
  - equalizing ≈ 2.3 µs;
  - H ≈ 4.7 µs;
  - broad ≈ 27 µs.
  
  For custom sources, the widths scale with the measured line period.
- **Anything outside every class is noise** and is counted in
  `stats.pulses_rejected`.

### 3.3 Horizontal: the line-locked loop

- **Prediction:** the next 0H is expected at `t_pred = t_last + T_line`.
- **Acceptance window:**
  - **Acquire:** an H leading edge within ±T_line/2.
  - **Track:** within ±1.5 µs. A noise spike inside the window is rarely
    H-shaped (§3.2), so it is rejected by width first.
- **The loop:**
  - phase error `e = t_edge − t_pred`;
  - a second-order loop: `T_line += β·e`, `t_last = t_pred + α·e`;
  - α and β come from the noise bandwidth B_L and damping ζ = 0.707 by the
    standard formulas, written in the header.
- **Bandwidths, derived from the jitter target:**
  - A 720-pixel line of 52.6 µs has 73 ns pixels.
  - Track B_L is chosen so that rms timing jitter stays under ⅓ pixel
    (≈ 25 ns) at the edge-noise σ measured in §3.1. The header shows the
    arithmetic.
  - Acquire B_L is 10× track.
  - The loop switches to track after 32 consecutive accepted lines.
- **Coasting:**
  - A missing or rejected pulse coasts: `t_last = t_pred`, and the miss is
    counted.
  - After 16 consecutive misses (≈ 1 ms), the loop drops back to acquire.
- **H-lock quality (0–1)** is the fraction of the last 64 lines with an
  accepted pulse, times a term that falls as the rms of `e` approaches the
  window.
- **The vertical interval:** equalizing and broad pulses arrive at half-line
  spacing. **The loop coasts through it** (§3.4 says when it starts and
  ends) and is not updated by those pulses.

### 3.4 Vertical: fields and their order

- **The vertical interval** begins at the first broad pulse after at least
  (lines per field − 10) lines since the previous field.
- **The field start** is the 0H of the line holding that first broad pulse.
- **Field order** comes from the first broad pulse's leading edge measured
  against the H grid: `φ = (t_broad − t_0H_nearest) / T_line`.
  - φ ≈ 0 is one field and φ ≈ 0.5 the other. **Take the mapping to
    "field 1 / field 2" from the standard text**, not from memory.
  - The same φ in every field means a **progressive** source.
- **Standard (AUTO):**
  - measured line rate within 0.5 % of 15,734.26 Hz and about 262.5 lines
    per field → 525;
  - within 0.5 % of 15,625 Hz and about 312.5 → 625;
  - anything else → **CUSTOM**, with the measured numbers.
- **V-lock** needs **2 consecutive fields** with line counts within ±1 of
  each other, and a broad-pulse group of the expected count in each.
- **Flywheel:** the next field start is predicted. An interval within
  ±3 lines of the prediction is accepted. The flywheel coasts up to 4 fields
  before V-lock is lost.
- **No V-lock, no frames.** A train of line-rate pulses with no field
  structure is **not video** (§6.5).

### 3.5 Rendering a line

- **Where the active line is:** the active region starts and ends at fixed
  offsets from 0H. Take them from BT.601's sample counts for the standard;
  for custom sources, scale by the line period.
- **Resampling:** that region is resampled to `width` pixels (default 720)
  with a Keys cubic interpolator at the sub-sample 0H from §3.3.
- **Levels to 0..1:** `(v − black) / (white − black)`, where black includes
  the 525-line setup, clipped to [0, 1].
- **Visible lines:** the vertical blanking lines of each field are dropped
  (counts from the standard; custom: the measured blanking, with 7 % of the
  field as the fallback).

### 3.6 Chroma (v1: luma only)

- **`YC_TRAP`** — a notch at the subcarrier, when `fs` ≥ 3× the subcarrier.
- **`YC_COMB`** — a line comb:
  - 525-line: the average of adjacent lines (the chroma phase alternates);
  - 625-line: two-line spacing.
  
  The comb is sharper than the trap but smears vertical detail.
- **`YC_OFF`.**

The default is TRAP when the rate allows it, and OFF otherwise.
`describe()` says which is in force.

### 3.7 Frames, fields and stacking

- **`INTERLACED`** weaves two fields into one frame.
- **`ONE_FIELD`** doubles the lines of each field. Use it for moving
  scenes, where weaving combs.
- **`PROGRESSIVE`** makes every field a frame. It is chosen automatically
  when §3.4 finds a progressive source.
- **Stacking (N = 1–64):**
  - A running average per pixel over N frames.
  - **Motion-aware** (default on): per 8×8 block, if the new frame differs
    from the average by more than 4σ_pixel, the block restarts from the new
    frame. σ_pixel comes from §3.1's noise σ scaled to the pixel bandwidth.
  - The gain on static areas is ≈ 10·log10(N) dB.
  - The frame's metadata carries `stacked` = the minimum N over its blocks,
    so the viewer can say how much of the picture is truly stacked.

---

## 4. API (sketch — names follow the header's conventions)

```c
enum { ATKDSP_VID_AUTO = 0, ATKDSP_VID_525, ATKDSP_VID_625, ATKDSP_VID_CUSTOM };
enum { ATKDSP_VID_POL_AUTO = 0, ATKDSP_VID_POL_NORMAL, ATKDSP_VID_POL_INVERTED };
enum { ATKDSP_VID_YC_OFF = 0, ATKDSP_VID_YC_TRAP, ATKDSP_VID_YC_COMB };
enum { ATKDSP_VID_INTERLACED = 0, ATKDSP_VID_ONE_FIELD, ATKDSP_VID_PROGRESSIVE };
enum { ATKDSP_VID_DEEMPH_NONE = 0, ATKDSP_VID_DEEMPH_F405_525, ATKDSP_VID_DEEMPH_F405_625 };

typedef struct {
    double fs;                    /* composite sample rate, Hz (>= 4e6) */
    int    standard;              /* ATKDSP_VID_AUTO ... */
    double custom_line_hz;        /* CUSTOM only */
    int    custom_lines;          /* CUSTOM only: lines per frame */
    int    polarity, yc, fields, deemph;
    int    width;                 /* pixels per line, 64..1920; default 720 */
    int    stack;                 /* 1..64 */
    int    motion_aware;          /* 0/1 */
} atkdsp_video_cfg;

typedef struct {
    int64_t sample0;              /* input sample index of the first line's 0H (caller -> UTC) */
    int     standard, polarity, progressive, field_order;
    int     width, lines;         /* this frame's geometry */
    double  line_hz;              /* measured */
    double  lines_per_field;      /* measured */
    float   h_lock;               /* 0..1 */
    int     v_lock;               /* 0/1 */
    float   snr_db;               /* sync-amplitude to noise, from the back porch */
    int     stacked;              /* min frames averaged over the picture */
} atkdsp_video_meta;

typedef struct {
    uint64_t samples_in, lines_locked, lines_coasted, pulses_rejected;
    uint64_t fields, frames_out, frames_dropped, h_relocks, v_relocks, polarity_flips;
} atkdsp_video_stats;

ATKDSP_API atkdsp_video *atkdsp_video_create(const atkdsp_video_cfg *cfg);
ATKDSP_API void          atkdsp_video_destroy(atkdsp_video *v);
ATKDSP_API void          atkdsp_video_reset(atkdsp_video *v);
/* new settings take effect at the next line boundary — never a reset */
ATKDSP_API int           atkdsp_video_set(atkdsp_video *v, const atkdsp_video_cfg *cfg);
/* returns the number of frames now waiting (>= 0) or a negative ATKDSP_E* */
ATKDSP_API ptrdiff_t     atkdsp_video_process(atkdsp_video *v, const float *in, size_t n);
/* pops the oldest waiting frame into pix (cap floats); 1 = a frame, 0 = none */
ATKDSP_API int           atkdsp_video_frame(atkdsp_video *v, float *pix, size_t cap,
                                            atkdsp_video_meta *meta);
ATKDSP_API int           atkdsp_video_describe(const atkdsp_video *v, char *buf, size_t cap);
ATKDSP_API int           atkdsp_video_stats(const atkdsp_video *v, atkdsp_video_stats *s);
```

- **Frame queue:** a bounded queue of 4 frames. When it is full, the oldest
  frame is dropped and counted in `frames_dropped`. The kernel never blocks.
- **`describe()` example:** *"525-line, interlaced, locked (H 0.98, V yes),
  15,734.3 Hz measured, 720 px of which 640 are real at 12 MS/s, chroma trap
  at 3.58 MHz, stack 16 motion-aware, normal polarity (auto)."*

---

## 5. The twin, the build, the ABI

- **The twin comes first:** `python/atkdsp/reference.py`, at the next free §
  number — §32 (§31 when this was written; the lag scan took it). The numpy twin runs at file
  speed; that is expected and stated in its docstring.
- **Build:** `src/video.c`, added to:
  - `CMakeLists.txt`;
  - `build.bat`'s `cl.exe` list;
  - `build.sh` (§1 rule 8);
  - one smoke-test line.
- **One ABI bump** at merge time. The number is whatever is next then — the
  DSSS work in progress may have taken some.
- **README:** a "What is in vX" section.
- **The binding** (`python/atkdsp/__init__.py`) wraps the kernel. ATK's
  `atk/core/dsp_native.py` reports it on Setup.
- **Logging** (§1 rule 5): `describe()` at INFO on create and on every lock
  change; `stats()` at DEBUG every 10 s; every non-zero return at WARNING with
  its arguments.

---

## 6. Tests and acceptance

### 6.1 The generator (twin side, ships with the tests)

`video_synth(pattern, standard, fs, seconds, ...)` produces composite video
with the standard's exact sync structure:
- equalizing and broad pulses at half-line spacing;
- the interlace offset;
- an optional colour burst.

**Patterns with known answers:**
- an 8-step grey ramp — levels;
- a crosshatch with lines at known pixel columns and rows — geometry;
- a **frame-counter block**: the frame number in binary as black and white
  blocks, so dropped, duplicated or reordered frames are detected exactly;
- an optional colour-bar field, to test the trap and the comb.

**Impairments:**
- an FM modulator (deviation as a parameter) → AWGN at a stated CNR → CFO;
- sample-rate offset ±100 ppm;
- line rate off-nominal ±1 %;
- sinusoidal line-period jitter (20 ns rms);
- a 2 ms dropout;
- inversion.

The Diffusion Toolkit's `synth/native.py` `_syn_fpv` is a second, independent
source of realistic FPV-like composite for smoke tests. It has random luma,
so it has no picture truth.

### 6.2 Known answers (twin and C, the same thresholds)

| Case | Pass |
|---|---|
| Clean 525 and 625, crosshatch | grid positions within 0.25 px rms; ramp steps within 2 % |
| Lock time from cold | H-lock within 20 lines of the first H pulse; V-lock within 2 fields after the first vertical interval |
| Frame counter, CNR ≥ the documented threshold | exact sequence — no drops, no duplicates |
| Dropout of 2 ms mid-field | relock within 2 fields; the counter resumes with a counted gap, never a duplicate |
| Inverted input, polarity AUTO | correct polarity within 2 fields; picture identical to the normal case |
| Custom: 15,500 Hz, 600 lines | locks as CUSTOM with the measured numbers within 0.1 % |
| Progressive (240p-style) | detected as progressive; no combing |
| Stack N = 16, static crosshatch at low CNR | PSNR improves 12 dB ± 1.5 dB |
| Motion-aware on a moving bar | the bar is not smeared beyond one 8×8 block |

### 6.3 Chunk invariance (§1 rule 1)

Blocks of 1, 7, 239–241, 999, 4096 and 8192 give identical frames and
metadata. The C equals the twin to float tolerance.

### 6.4 SNR curve

Record PSNR against input CNR for 525 and 625 at 8, 12 and 20 MS/s. The
table goes into the README with the threshold where the frame counter
starts to fail. That threshold is the honest "works down to" figure the
viewer quotes.

### 6.5 Negative controls (§1 rule 3)

Each must produce **zero frames**, and `describe()` must say why:
- white noise (10 minutes): *"no line-rate pulses"*;
- an FM voice carrier (3 kHz deviation, tones and speech);
- an OFDM burst train;
- **a pulse train at exactly 15,734 Hz** (radar-like): *"line-rate pulses
  but no fields — not video"*. H-lock alone must never output frames.

### 6.6 Oracle (optional, Linux, black box)

Run a stock SigDigger's TV processor on the §6.1 files and compare
lock / no-lock and the standard detected. Disagreements are investigated
from the standards, never from its source.

### 6.7 Real signals (Bill's bench, after merge)

- An FCC-certified low-power analog video transmitter (bugsniffer's
  test-fixture list) at 1.2 or 2.4 GHz on the bladeRF 1.0, through ATK's
  front end.
- A 5.8 GHz FPV camera on the HackRF at 20 MS/s.
- **Pass:** lock within 1 s; a readable snapshot; the lock lamps steady for
  a minute.

---

## 7. Size and order

**M** — the twin and the generator in one session; the C and the
cross-checks in another. Independent of every other phase. Bug Sniffer's
Look step and Drone Discovery's FPV picture wait on it.

## 8. Later (separate ABI)

- **Colour.**
  - 525-line: a burst-locked subcarrier loop, I/Q demodulation and the
    hue reference;
  - 625-line: the same, plus the V-switch and its line identification.
  
  Output YCbCr. The same tests, with the colour-bar field.
- **The FM-video front end as one call:** `atkdsp_video_front` (channel →
  FM → de-emphasis → low-pass), once ATK has used the separate pieces long
  enough to know the right defaults.
- **The audio subcarrier** (6.0 / 6.5 MHz) stays ATK-side: an OSX channel on
  the composite, then `atkdsp_fm_demod`. No kernel is needed.
