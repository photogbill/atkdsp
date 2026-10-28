# Native LMR decoding in atkdsp — P25, DMR and kin, without DSD

**Repo:** atkdsp (kernels and the `lmr` Python layer), plus a separate private component
for AMBE+2 (§5)
**Written:** 2026-10-28
**Status:** plan; build to start on Bill's go

Bill, 2026-10-28: *"Would this let us break free of dsd completely by putting decoding into
our own dsp library?"*, then *"build everything so we can test it. But we will not release
the ambe+2 part until May of 2028 when the patent expires… leave dsd as the primary for
that aspect and put the native ambe+2 behind a password that we can release once we
acquire licensing or the patent expires."*

---

## 0. Scope in one paragraph

ATK decodes P25 (Phase 1 and Phase 2), DMR, NXDN, dPMR and EDACS/ProVoice **natively**:

- **What it decodes:** demodulation, framing, error correction, trunking and link-control
  messages, radio IDs, and IMBE voice.
- **What it gives that DSD's console never could:**
  - every frame stamped with its exact sample index, for DF;
  - a per-frame bit-error measurement, for propagation acceptance testing and DAQ;
  - every channel of a wideband capture decoded in one process.
- **AMBE+2** (DMR, NXDN, dPMR and P25 Phase 2 voice) is built and tested on Bill's machine
  only. It sits in a separate component that never ships, behind a release password.
  **DSD stays the primary AMBE+2 voice decoder** until licensing or patent expiry.
- **DSD stays as the fallback** for anything not yet native. With the original DSD's ISC
  code and the GPL forks' protocol knowledge (§1), X2-TDMA and Yaesu System Fusion can
  eventually go native too. D-STAR, including its original-AMBE voice, goes native (§4).

---

## 1. Rules

1. **Specifications first; licences decide what DSD-family code may be reused.** Checked
   2026-10-28:

   | Source | Licence | May ATK reuse it? |
   |---|---|---|
   | **Original DSD** (szechyjs/dsd) | **ISC** (permissive), except `dstar_header.c/h`, `descramble.h`, `fcs.h`, which are **GPL-2** | **Yes**, adapt freely, keeping its copyright and permission notice in `THIRD_PARTY_NOTICES`. Except the three GPL-2 files. |
   | **mbelib** (szechyjs/mbelib) | **ISC** | **Yes**, same terms. Copyright is not patent: AMBE+2 stays gated (§5) whatever its source. |
   | **dsd-neo** (what ATK runs today; GPL-3.0 per its shipped `LICENSE`) and dsd-fme | **GPL** | **Code, no. Knowledge, yes.** Protocol facts (sync words, frame layouts, field positions, what a value means) are read from it and written up in our own words in `docs\protocol_notes\`. The implementation is then written from those notes and from the specifications, never translated line by line. |

   - Where a specification exists (P25, DMR, NXDN, dPMR), it is the authority. DSD code
     is a cross-reference.
   - Where none exists (X2-TDMA, Yaesu Fusion, parts of ProVoice), the old ISC DSD is
     the starting point, and the GPL forks contribute knowledge only.
   - dsd-neo, run as an external program, remains the **oracle** for Compare mode.
2. **TIA documents stay out of git**, the same rule as the propagation repo (TIA
   copyright). Each function cites its document and section. No spec text or tables are
   pasted into source comments beyond what is needed to name a field.
3. **Passive, and no decryption.** For encrypted traffic, ATK reports what the protocol
   sends in the clear (algorithm ID, key ID, the fact that it is encrypted) and nothing
   more. That is ATK's existing rule.
4. **atkdsp's own rules apply:** C kernels, a numpy twin for each, ABI bumps per phase, and
   acceptance against a **real signal**, not just a round trip.
5. **A native decoder becomes primary for a protocol only after it earns it.** In Compare
   mode on Bill's recorded captures it must match or beat dsd-neo (§7).
6. **Comprehensive logging:** per frame, protocol, sync quality, corrected bits, FEC
   failures and decision path, all to the decoder log.

---

## 2. Specifications in hand, and what is still needed

**In hand** (`atk_propagation_analysis\TIA-102_Series_Documents\`, Bill's copies):

| Document | Gives |
|---|---|
| TIA-102.BAAA-A FDMA Common Air Interface | C4FM/CQPSK Phase 1: frame sync, NID (NAC + DUID, BCH), HDU, LDU1/LDU2, TDU, TSBK, PDU, status symbols, interleaving, Golay/Hamming/RS/trellis codes |
| TIA-102.BAAC-A Reserved Values | NAC, MFID, ALGID and other code points |
| TIA-102.AABB-B / AABC-B Trunking Control Channel formats and messages | TSBK/MBT opcodes: grants, affiliations, registrations, network/site status, adjacent sites. **Trunk following** comes from this. |
| TIA-102.AABF-A Link Control Word formats | LC formats: talkgroup, source ID, emergency, encryption, GPS where present |
| TIA-102.BBAB / BBAC (+BBAC-1) TDMA physical and MAC | Phase 2 H-CPM (inbound) / H-DQPSK (outbound), superframes, MAC PDUs (MAC_PTT, MAC_ACTIVE, grants, IDs) |
| TIA-102.BABA IMBE Vocoder | IMBE 7200/4400 full rate: frame FEC, PN, parameter decode, synthesis |
| TIA-102.BABA-1 Half-Rate Vocoder Addendum | AMBE+2 3600/2450 half rate (Phase 2). The same vocoder family DMR/NXDN/dPMR use. **Gated, §5.** |
| TIA-102.BABC Vocoder Reference Test | **Conformance test vectors**: the oracle for both vocoders |
| TIA-EIA IS-69.5 EDACS IMBE Implementation | EDACS digital voice (ProVoice) with IMBE, so ProVoice can go native too |
| TIA-102.CAAA-E / CAAB-D | Measurement methods and receiver performance: the yardsticks for §7 |
| TSB-102.BAAD, TSB-102.BBAA, TSB-102-A | Operational descriptions and overviews |

**Also in hand (added by Bill 2026-10-28, same folder; git-ignored):**

| Document | Gives |
|---|---|
| **ETSI TS 102 361-1 V2.1.1** DMR Air Interface | 4FSK, bursts and sync, CACH, slot type, BPTC/Golay/Hamming/QR/RS(12,9), embedded signalling, LC and CSBK structure |
| **ETSI TS 102 361-2 V2.5.1** voice and generic services | Call set-up and services, voice superframe structure, LC opcodes, talker alias |
| **ETSI TS 102 361-3 V1.3.1** data protocol | Data headers, confirmed/unconfirmed data, rate ½ / ¾ / 1 trellis |
| **ETSI TS 102 361-4 V1.9.1** trunking (Tier III) | Control channel CSBKs, grants, registration, site and system identity, so DMR trunk following |
| **ETSI TS 102 658 V2.6.1** dPMR | 6.25 kHz FDMA framing, FEC, IDs |
| NXDN TS 2-A v1.1 | NXDN **conformance / transceiver performance tests**. Useful for §7 receiver yardsticks, but **not the air interface** |

**Still needed:** the NXDN **Technical Specifications Part 1** (Common Air Interface, plus
its trunking sub-parts if available) for L5's NXDN half. dPMR in L5 can proceed now.

---

## 3. Architecture

```
I/Q (any SDR profile)
  └─ atkdsp channelizer (A2.2) ─┬─ per channel:
                                │   demod   C4FM/4FSK (fsk.c) · CQPSK/LSM · H-DQPSK · H-CPM
                                │   timing  symbol sync + freq tracking (soft symbols kept)
                                │   sync    dmr_sync.c (exists: DMR + P25 Ph1) + Ph2/NXDN/dPMR words
                                │   frame   status-symbol strip, deinterleave, descramble
                                │   FEC     BCH(63,16) · Golay(23,12)/(24,12) · Hamming family ·
                                │           cyclic(16,8) · RS over GF(64) (24,12)/(24,16)/(36,20) ·
                                │           RS(12,9) · BPTC(196,96) · QR(16,7) · trellis ½ and ¾ ·
                                │           CRC-CCITT/-9/-32 and DMR CRCs  (on atkdsp ABI 16's bitstream engine:
                                │           atkdsp_bch / _rs / _viterbi / _crc already built, plus
                                │           the small block codes; nothing duplicated)
                                │   out     frame{t_sample, proto, chan, corrected_bits, failed, bits}
                                └─ atkdsp.lmr (Python): message semantics per spec table →
                                    events {IDs, talkgroup, NAC/CC, site, grant, emergency, enc flags}
                                    voice frames → vocoder → PCM → ATK's TransmissionGate → Whisper
```

- **C (atkdsp):** everything per-sample and per-bit. New sources: `p25_fdma.c`,
  `p25_tdma.c`, `dmr.c`, `nxdn.c`, `dpmr.c`, `demod_psk.c` (CQPSK, H-DQPSK), `demod_cpm.c`
  (H-CPM), `imbe.c`. Each comes with its numpy twin.
- **Python (`atkdsp/python/atkdsp/lmr/`):** the message tables (opcodes, LC formats, MAC
  PDUs) as data with spec citations, frame assembly, superframe state and trunk-following
  state. These are tables, not DSP, and change by table edit.
- **ATK (`atk/core/lmr_native.py`):** the host adapter beside `dsd_controller.py`, with
  the same outputs. **Decoder choice per protocol:** Native / DSD / **Compare** (both run;
  agreement scored and logged).

**What the outputs feed elsewhere in ATK:**
- **atkdf:** decoder-clocked DoA. Each frame's exact sample index sets the DoA window, as
  KRAKEN_DF_PLAN §7.1 wanted from dsd-neo timing. Radio-ID-tagged bearings and the roster
  come from the native events.
- **Propagation** (atk_propagation_analysis §4.1a): BER per frame gives the acceptance
  test's **BER-based tile score** (the measured-BER method in TSB-88), and an estimate of
  delivered audio quality.
- **RF tab:** in-span trunk following and colour-coded waterfall boxes per protocol and
  talkgroup (the processing mark already exists).
- **Bit Bench:** raw and corrected bits of any frame, one click away.

---

## 4. Vocoders

| Vocoder | Used by | Status | Default |
|---|---|---|---|
| **IMBE** (full rate) | P25 Phase 1, EDACS ProVoice | Core US patents filed early–mid 1990s, widely reported **expired** | **Native, our own code** from TIA-102.BABA (Bill, 2026-10-28). mbelib is only a cross-check. Becomes the default once it passes BABC conformance and Compare. |
| **AMBE+2** (half rate, 3600/2450) | P25 Phase 2, DMR, NXDN, dPMR | **Patent-encumbered** (DVSI). Bill's stated release date is **May 2028**, or earlier with a licence | **DSD primary.** Native only in the gated component (§5). |
| AMBE (original) | D-STAR | Bill, 2026-10-28: the original AMBE patents expired long ago. | **Native, our own code** (Bill, 2026-10-28: "for the others, we do natively"). D-STAR framing comes from the published D-STAR specification. Original AMBE has no public codec specification, so mbelib (ISC) is the **reference and cross-check**, not the code. Before release, confirm that the codec in D-STAR's AMBE-2020 chip is plain AMBE and not a later AMBE+ variant. |

Not legal advice. Before release, confirm the controlling AMBE+2 patent numbers and their
actual expiry dates. One patent family can hold several patents with different dates.

**How the AMBE+2 decoder is derived does not change the patent.** A patent covers the
method itself, whether it was learned from TIA-102.BABA-1, from mbelib (ISC) or from
analysing DSD. Working from DSD/mbelib instead of the TIA document would only change the
*copyright* side, and the TIA document was never the problem there: it is a standard
written to be implemented. **Decided (Bill, 2026-10-28): AMBE+2 alone is built on mbelib; every other vocoder and protocol is our own native code.** The AMBE+2 decoder is built on mbelib
(szechyjs/mbelib, ISC):
- Its `ambe3600x2450` path is vendored into the gated `atk_ambe2` component, with mbelib's
  copyright and ISC notice kept in `THIRD_PARTY_NOTICES`.
- It is **not** taken from `mbe-neo`, the fork bundled with dsd-neo, whose licence is
  unverified.
- BABA-1's normative tables and the **BABC conformance vectors measure it.** mbelib was
  never conformance-tested, so differences are expected. They are logged, and any fixes
  stay in our copy.
- **The gate (§5) applies regardless.**

---

## 5. The AMBE+2 gate

There are three layers. The first is the one that matters.

1. **Not in any shipped install.** The native half-rate vocoder lives in its own private
   component, `D:\Analyst_Toolkit\atk_ambe2` (C source → `atk_ambe2.dll`). It is **never in
   atkdsp's repo, never cloned by `install.bat`, never in a customer build.** This is the
   same model Bill set for DJI DroneID (2026-10-27): *"an optional component that doesn't
   come with atk, that the user, and only with proper authorization, can install."* On
   Bill's machine it is built by its own `get_ambe2.bat`.
2. **A release password, even where it is installed.** The component loads only after the
   release password is entered in Setup → Optional Components.
   - The password is checked against a salted, slow hash. Only the hash is stored.
   - This is a **release** gate, separate from the operator admin unlock. The admin role
     cannot open it.
   - Every unlock and every decode session is written to ATK's existing hash-chained audit
     log (`data\audit\gated_audit.jsonl`), with who and why.
3. **DSD stays primary for AMBE+2 voice even when unlocked.** Native AMBE+2 runs only in
   **Compare** mode, or when chosen explicitly for a test, so testing never changes what an
   operator hears by default.

**Release** happens in one step, at Bill's decision, once licensing is in hand or the
patent date is confirmed passed:
1. The component moves into the installer.
2. The gate is removed.
3. Native becomes the AMBE+2 default after its Compare record.

**Honest notes:**
- A password in a local application is an accountability gate, not copy protection.
  Keeping the code out of shipped installs (layer 1) is the real control.
- US patent law has no general personal-use exception. Building and testing privately
  before expiry is Bill's call to make, and has been made. The plan keeps it private,
  logged, and off by default.

---

## 6. Phases

Each phase is its own ABI bump, with twins, cross-checks and real-signal acceptance.

| Phase | Builds | Needs | Done when |
|---|---|---|---|
| **L1 P25 Phase 1 control and data** | C4FM demod and timing; NID BCH; HDU/LDU1/LDU2/TDU/TSBK/PDU; all Phase 1 FEC; LC and TSBK message tables; trunk-following state; frame timestamps; BER per frame | specs in hand | On Bill's recorded P25 captures: every frame dsd-neo decodes, native decodes, with **identical** NAC, talkgroups and source IDs; on synthesized C4FM with noise, decode success vs C/N plotted against CAAB-D reference sensitivity |
| **L2 IMBE voice** | `imbe.c`: frame FEC, PN, parameter decode, synthesis; PCM into the existing Whisper path | BABA, BABC | BABC conformance vectors pass; A/B listening against dsd-neo/mbelib on the same captures; then native IMBE becomes default |
| **L3 DMR control, data and IDs** | 4FSK (sync exists), CACH, slot type, BPTC, CSBK, voice and data LC, embedded LC, colour code, talker alias; Tier II/III trunking events | ETSI TS 102 361-1..4 (in hand) | Identical IDs and colour codes vs dsd-neo on Bill's DMR repeater captures (his DF reference emitters) |
| **L4 P25 Phase 2 TDMA control and IDs** | H-DQPSK and H-CPM demod; superframe and slot sync; MAC PDUs; IDs and grants | BBAB/BBAC in hand | Identical IDs vs dsd-neo on Phase 2 captures (where available locally) |
| **L5 dPMR, NXDN, D-STAR** | 4FSK framing and FEC, control and IDs; D-STAR framing and original-AMBE voice | ETSI TS 102 658 (in hand); NXDN TS Part 1 (still needed); D-STAR specification (public) | Same agreement test |
| **L6 CQPSK/LSM simulcast demod; EDACS ProVoice** | Simulcast-tolerant PSK receiver; ProVoice framing to IMBE | in hand | LSM captures decode at least as well as C4FM-mode DSD |
| **L7 AMBE+2 (gated component)** | `atk_ambe2`: **mbelib's** `ambe3600x2450` decode (ISC, vendored, notice kept) wrapped in atkdsp's ABI conventions with a numpy-free C path; the gate (§5) | mbelib (ISC); BABA-1 + BABC to measure it | BABC half-rate vectors pass; Compare mode against DSD on DMR captures; **stays gated** |

L1 and L3 are what DF and propagation need first. Vocoders come after the metadata,
because IDs and timing are what the analysis uses; voice is for listening.

---

## 7. How native earns "primary"

**Compare mode** runs native and dsd-neo on the same channel and logs agreement per frame:
sync, IDs, corrected bits and decode failures. A protocol's default flips to Native only
when, over Bill's capture library:

- native decodes **at least** every frame dsd-neo does;
- every ID native reports is **identical** where both decode;
- on synthesized signals with noise, native's success-vs-C/N curve is at or better than
  dsd-neo's.

The scorecard is shown in Setup, so the switch is visible and reversible.

---

## 8. Open items for Bill

1. ~~ETSI DMR and dPMR standards~~: **in hand** (2026-10-28). Still wanted: **NXDN TS
   Part 1** (Common Air Interface; the TS 2-A in hand is the conformance test).
2. Create the private `atk_ambe2` folder or repo when L7 starts, and pick the release
   password. Only its salted hash is stored.
3. Before release: confirm the AMBE+2 patent numbers and expiry dates (Bill's stated date:
   May 2028), or the licence.
