"""NR (5G) SS/PBCH block cell search — the validated Python specification.

PHASE 1 of the NR survey: PSS -> N_ID2, SSS -> N_ID1, PCI = 3*N_ID1 + N_ID2,
timing and carrier frequency offset. Downlink broadcast only: the SSB is the
cell announcing itself, nothing about any subscriber. 3GPP TS 38.211 7.4.2
(PSS/SSS sequences), 7.4.3 (SSB resource map), 5.3 (OFDM), 4.3.2 (numerology).

WHAT IS DIFFERENT FROM LTE, because each one changed the design:

1. THE SSB IS NOT AT THE CARRIER CENTRE. LTE put its sync signals in the
   middle 1.08 MHz of every carrier; NR puts the SSB (240 subcarriers, 4
   symbols) at a point of the SYNCHRONISATION RASTER (GSCN, 38.104 5.4.3),
   which is sparse — 1.44 MHz steps above 3 GHz, ~1.2 MHz below — and
   band-specific. So the search is over GSCN points, not over the channel
   raster, and it is cheaper than LTE's: n71 has 78 points in 35 MHz.
   `atk/core/nr_bands.py` owns the raster; this module owns one look.
2. THE SUBCARRIER SPACING IS PER BAND. 15 kHz (Case A: n71, n5, n2, n25 ...)
   or 30 kHz (Case B/C: n41, n77, n78, n48 ...). Everything here is written
   in units of the SCS: the stream is at 256*SCS (3.84 or 7.68 MSPS), the
   FFT is 256, a symbol is 256 + 18 samples, and the same code serves both.
3. THE SSS IS COHERENT AGAINST THE PSS. Both occupy the same 127
   subcarriers two symbols apart, so the PSS is a channel estimate for the
   SSS for free: correlate Y_sss * conj(Y_pss) * d_pss against d_sss and the
   channel cancels. LTE's SSS needed its own handling of the two half-frames.
4. THE PERIOD IS 20 ms, NOT 5. A cell may send its SSB burst every 5, 10, 20,
   40, 80 or 160 ms; a UE must assume 20 ms for initial search (38.213 4.1),
   so a look needs >= 20 ms of samples, and confirmation ("the same PCI came
   back where the period says") needs 40.

WHAT IT DOES NOT DO, on purpose: decode the PBCH (Phase 2: polar decoder ->
MIB) or SIB1 (Phase 3: CORESET0/PDCCH/PDSCH/LDPC). It answers "is there a
cell at this GSCN point, which one, how strong, how far off frequency".

Every number below is checked against something outside this code in
test_nr_ssb.py: the sequence generators against the spec's own recurrences
and known cross-correlation properties, the detector against a synthetic
SSB it did not build (the modulator is written from the spec's resource map,
the detector from its own), across SNR, CFO and timing, with a noise-only
negative control. The m-sequence initial states and the resource map come
from 38.211 and are the two places a transcription error would hide — the
test that the three PSS sequences are cyclic shifts of one m-sequence by
exactly 43 catches the first; a real capture is the only proof of the second.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Numerology
# ---------------------------------------------------------------------------
NFFT = 256            # smallest power of two holding 240 subcarriers
NSC = 240             # SSB width in subcarriers
CP = 18               # normal cyclic prefix at N=256 (144 at N=2048)
SYM = NFFT + CP       # samples per SSB symbol
NSYM = 4              # symbols in an SS/PBCH block
K_SYNC0 = 56          # PSS and SSS occupy subcarriers 56..182 (127 of them)
NSYNC = 127
N_ID1_MAX = 336
N_ID2_MAX = 3
PCI_MAX = 1008

SSB_PERIOD_MS = 20    # what a UE assumes for initial cell search


def sample_rate(scs_hz: float) -> float:
    """The rate this module wants its input at: 256 * SCS."""
    return NFFT * float(scs_hz)


def long_cp(mu: int) -> int:
    """CP of symbols 0 and 7*2^mu of each 0.5 ms, at N=256: 18 + 2*2^mu.
    (16*kappa Tc of extra prefix; one sample at 256*SCS is 512/2^mu Tc.)"""
    return CP + 2 * (1 << mu)


def symbol_start(l: int, mu: int) -> int:
    """Sample index of the USEFUL part of OFDM symbol l (l counts from the
    start of a 5 ms half-frame) at 256*SCS. Symbols 0 and 7*2^mu of every
    0.5 ms carry the long CP. Sanity: symbol 14*2^mu starts at 1 ms."""
    per_half_ms = 7 << mu
    n_half = l // per_half_ms
    rem = l % per_half_ms
    base = n_half * (per_half_ms * SYM + 2 * (1 << mu))   # 0.5 ms in samples
    return base + rem * SYM + long_cp(mu)


def ssb_symbols(case: str, mu: int, above_3ghz: bool = False) -> list[int]:
    """First symbol index of every candidate SSB in a 5 ms half-frame
    (38.213 4.1).
      Case A, 15 kHz: {2, 8} + 14n         n = 0,1 (<= 3 GHz), 0..3 above
      Case B, 30 kHz: {4, 8, 16, 20} + 28n  n = 0   (<= 3 GHz), 0,1 above
      Case C, 30 kHz: {2, 8} + 14n         n = 0,1 (<= 3 GHz), 0..3 above
    L = 4 candidates at or below 3 GHz, 8 above (FR1 up to 6 GHz)."""
    case = case.upper()
    if case == "A":
        if mu != 0:
            raise ValueError("Case A is 15 kHz")
        base, stride, ns = (2, 8), 14, (4 if above_3ghz else 2)
    elif case == "B":
        if mu != 1:
            raise ValueError("Case B is 30 kHz")
        base, stride, ns = (4, 8, 16, 20), 28, (2 if above_3ghz else 1)
    elif case == "C":
        if mu != 1:
            raise ValueError("Case C is 30 kHz")
        base, stride, ns = (2, 8), 14, (4 if above_3ghz else 2)
    else:
        raise ValueError(f"unknown SSB case {case!r}")
    return [b + stride * n for n in range(ns) for b in base]


# ---------------------------------------------------------------------------
# Sequences (38.211 7.4.2)
# ---------------------------------------------------------------------------
def _mseq(taps: tuple[int, int], init: list[int]) -> np.ndarray:
    """x(i+7) = (x(i+a) + x(i+b)) mod 2 for (a, b) = taps, seeded with
    x(0..6) = init. Returns 127 values."""
    x = np.zeros(NSYNC, dtype=np.int8)
    x[:7] = init
    a, b = taps
    for i in range(NSYNC - 7):
        x[i + 7] = (x[i + a] + x[i + b]) & 1
    return x


# PSS: x(i+7) = (x(i+4) + x(i)) mod 2, [x(6) .. x(0)] = [1 1 1 0 1 1 0]
_X_PSS = _mseq((4, 0), [0, 1, 1, 0, 1, 1, 1])
# SSS: x0(i+7) = (x0(i+4) + x0(i)),  x1(i+7) = (x1(i+1) + x1(i)),
#      [x(6) .. x(0)] = [0 0 0 0 0 0 1] for both
_X0_SSS = _mseq((4, 0), [1, 0, 0, 0, 0, 0, 0])
_X1_SSS = _mseq((1, 0), [1, 0, 0, 0, 0, 0, 0])


def pss_d(nid2: int) -> np.ndarray:
    """d_PSS(n) = 1 - 2 x((n + 43 N_ID2) mod 127), n = 0..126. Real +-1."""
    if not 0 <= nid2 < N_ID2_MAX:
        raise ValueError("N_ID2 is 0..2")
    n = np.arange(NSYNC)
    return (1 - 2 * _X_PSS[(n + 43 * nid2) % NSYNC]).astype(np.float32)


def sss_d(nid1: int, nid2: int) -> np.ndarray:
    """d_SSS(n) = [1 - 2 x0((n + m0) mod 127)] [1 - 2 x1((n + m1) mod 127)],
    m0 = 15 floor(N_ID1 / 112) + 5 N_ID2, m1 = N_ID1 mod 112."""
    if not 0 <= nid1 < N_ID1_MAX or not 0 <= nid2 < N_ID2_MAX:
        raise ValueError("N_ID1 is 0..335, N_ID2 is 0..2")
    n = np.arange(NSYNC)
    m0 = 15 * (nid1 // 112) + 5 * nid2
    m1 = nid1 % 112
    a = 1 - 2 * _X0_SSS[(n + m0) % NSYNC]
    b = 1 - 2 * _X1_SSS[(n + m1) % NSYNC]
    return (a * b).astype(np.float32)


def sss_table(nid2: int) -> np.ndarray:
    """All 336 SSS sequences for one N_ID2, shape (336, 127) — the
    detector's correlation bank."""
    return np.stack([sss_d(n1, nid2) for n1 in range(N_ID1_MAX)])


# ---------------------------------------------------------------------------
# Resource map (38.211 7.4.3.1)
# ---------------------------------------------------------------------------
def sync_subcarriers() -> np.ndarray:
    """k = 56..182: where PSS (symbol 0) and SSS (symbol 2) sit."""
    return np.arange(K_SYNC0, K_SYNC0 + NSYNC)


def pbch_subcarriers(l: int) -> np.ndarray:
    """PBCH REs in SSB symbol l: symbols 1 and 3 use all 240 subcarriers;
    symbol 2 uses 0..47 and 192..239 either side of the SSS."""
    if l in (1, 3):
        return np.arange(NSC)
    if l == 2:
        return np.concatenate([np.arange(0, 48), np.arange(192, NSC)])
    raise ValueError("PBCH is in symbols 1, 2, 3")


def dmrs_subcarriers(l: int, pci: int) -> np.ndarray:
    """PBCH DM-RS: every fourth PBCH subcarrier, k = 4m + v, v = PCI mod 4."""
    k = pbch_subcarriers(l)
    return k[(k % 4) == (pci % 4)]


def k_to_bin(k) -> np.ndarray:
    """SSB subcarrier k (0..239) -> FFT bin, with the SSB centred on DC:
    k = 120 is DC. (NR has no unused DC subcarrier; k=120 carries data.)"""
    return (np.asarray(k) - NSC // 2) % NFFT


# ---------------------------------------------------------------------------
# Synthetic SSB (the modulator, from the resource map)
# ---------------------------------------------------------------------------
def ssb_grid(pci: int, rng=None, pbch_symbols=None) -> np.ndarray:
    """The 4 x 240 SS/PBCH block for one PCI. PBCH REs carry random QPSK
    (Phase 2 will carry the real PBCH); DM-RS positions are marked so the
    same grid serves Phase 2's tests. Returns complex64 (NSYM, NSC)."""
    nid2 = pci % 3
    nid1 = pci // 3
    g = np.zeros((NSYM, NSC), np.complex64)
    ks = sync_subcarriers()
    g[0, ks] = pss_d(nid2)
    g[2, ks] = sss_d(nid1, nid2)
    rng = np.random.default_rng(0) if rng is None else rng
    for l in (1, 2, 3):
        k = pbch_subcarriers(l)
        if pbch_symbols is not None:
            g[l, k] = pbch_symbols[l][: k.size]
        else:
            bits = rng.integers(0, 2, (k.size, 2))
            g[l, k] = ((1 - 2 * bits[:, 0]) + 1j * (1 - 2 * bits[:, 1])) / np.sqrt(2)
    return g


def ofdm_symbol(freq_row: np.ndarray, cp: int = CP) -> np.ndarray:
    """One SSB symbol in time: IFFT of the 240 subcarriers on bins
    (k - 120) mod 256, scaled so a unit subcarrier has unit power per
    sample, with `cp` samples of cyclic prefix in front."""
    X = np.zeros(NFFT, np.complex64)
    X[k_to_bin(np.arange(NSC))] = freq_row
    x = np.fft.ifft(X) * np.sqrt(NFFT)
    if cp <= 0:
        return x.astype(np.complex64)
    return np.concatenate([x[-cp:], x]).astype(np.complex64)


def ssb_samples(grid: np.ndarray) -> np.ndarray:
    """4 symbols with normal CP: 4 * 274 = 1096 samples."""
    return np.concatenate([ofdm_symbol(grid[l]) for l in range(NSYM)])


def half_frame_samples(mu: int) -> int:
    """Samples in 5 ms at 256*SCS: 10 * (7*2^mu * 274 + 2*2^mu)."""
    return 10 * ((7 << mu) * SYM + 2 * (1 << mu))


def build_stream(pci: int, mu: int, case: str, n_ms: int = 40,
                 snr_db: float = 30.0, cfo_hz: float = 0.0,
                 ssb_period_ms: int = 20, start_offset: int = 0,
                 above_3ghz: bool = False, seed: int = 0,
                 amplitude: float = 1.0, ssb_mask=None):
    """A stream at 256*SCS carrying one cell's SSB bursts on the Case A/B/C
    positions every `ssb_period_ms`, with AWGN at `snr_db` (SSB symbol power
    over noise power in the 240 SSB subcarriers) and a carrier offset.
    `ssb_mask` selects which candidate SSB indices are transmitted (a cell
    with fewer beams sends fewer). Returns (stream, truth) where truth lists
    the absolute sample index of every transmitted SSB's PSS useful part."""
    rng = np.random.default_rng(seed)
    fs = sample_rate(15e3 * (1 << mu))
    n = int(round(n_ms * fs / 1e3))
    x = np.zeros(n + start_offset, np.complex64)
    grid = ssb_grid(pci, rng)
    blk = ssb_samples(grid) * amplitude
    hf = half_frame_samples(mu)
    truth = []
    syms = ssb_symbols(case, mu, above_3ghz)
    if ssb_mask is not None:
        syms = [s for i, s in enumerate(syms) if ssb_mask[i]]
    period = int(round(ssb_period_ms * fs / 1e3))
    t0 = start_offset
    while t0 < x.size:
        for l in syms:
            s = t0 + symbol_start(l, mu) - CP          # start of the PSS's CP
            if s + blk.size <= x.size:
                x[s:s + blk.size] += blk
                truth.append(s + CP)
        t0 += period
    # noise: per-sample noise power so that SSB SNR (in its 240 subcarriers)
    # is snr_db. The SSB occupies 240/256 of the band, power ~amplitude^2.
    sig_pow = amplitude ** 2 * NSC / NFFT
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    x += (np.sqrt(noise_pow / 2) *
          (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
          ).astype(np.complex64)
    if cfo_hz:
        t = np.arange(x.size) / fs
        x *= np.exp(2j * np.pi * cfo_hz * t).astype(np.complex64)
    return x, truth


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
@dataclass
class SsbHit:
    nid2: int
    nid1: int
    pci: int
    offset: int          # absolute index of the PSS useful part
    metric: float        # normalised PSS correlation, 0..1
    sss_score: float     # normalised coherent SSS correlation, 0..1
    sss_margin: float    # best / second-best SSS score (1 = ambiguous)
    cfo_hz: float
    confirmed: bool = False   # seen again where the period said


#: Fractional-CFO template bank, in subcarriers. A time-domain PSS
#: correlation loses ~sinc(cfo / SCS): 36 % at half a subcarrier, which at
#: 15 kHz is 7.5 kHz — an uncalibrated SDR's error at 600 MHz is a few kHz,
#: at 3.5 GHz it can be more. Three templates per N_ID2 keep the worst case
#: inside +-SCS/2 at 5 % and the integer search (below) takes the rest.
CFO_BANK = (-1.0 / 3.0, 0.0, 1.0 / 3.0)


def pss_time_template(nid2: int, cfo_sc: float = 0.0) -> np.ndarray:
    """The PSS symbol's useful part (256 samples, no CP), unit energy,
    pre-rotated by `cfo_sc` subcarriers of carrier offset."""
    row = np.zeros(NSC, np.complex64)
    row[sync_subcarriers()] = pss_d(nid2)
    t = ofdm_symbol(row, cp=0)
    if cfo_sc:
        t = t * np.exp(2j * np.pi * cfo_sc * np.arange(NFFT) / NFFT)
    return (t / np.linalg.norm(t)).astype(np.complex64)


def _xcorr_norm(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """|sum_i x[n+i] conj(t[i])| / (||x[n:n+T]|| ||t||) for every n where
    the window fits. FFT correlation; window energy by cumulative sum."""
    T = t.size
    N = x.size
    if N < T:
        return np.zeros(0, np.float32)
    L = 1 << int(math.ceil(math.log2(N + T)))
    X = np.fft.fft(x, L)
    H = np.fft.fft(np.conj(t[::-1]), L)
    c = np.fft.ifft(X * H)[T - 1:N]
    e = np.concatenate([[0.0], np.cumsum(np.abs(x) ** 2)])
    win = e[T:N + 1] - e[:N - T + 1]
    win = np.maximum(win, 1e-20)
    return (np.abs(c) / np.sqrt(win)).astype(np.float32)   # t is unit energy


def frac_cfo(x_sym: np.ndarray, t: np.ndarray, fs: float) -> float:
    """Fractional CFO from the phase advance between the two halves of the
    PSS symbol correlated against the template. Unambiguous for
    |cfo| < fs / (2 * 128) = SCS."""
    h = NFFT // 2
    a = np.vdot(t[:h], x_sym[:h])
    b = np.vdot(t[h:], x_sym[h:])
    ph = np.angle(b * np.conj(a))
    return float(ph / (2 * np.pi) * fs / h)


def _derotate(x: np.ndarray, cfo_hz: float, fs: float, n0: int = 0):
    n = np.arange(n0, n0 + x.size)
    return x * np.exp(-2j * np.pi * cfo_hz * n / fs)


def detect(x: np.ndarray, mu: int, threshold: float = 0.30,
           int_cfo_max: int = 3, max_hits: int = 32,
           start_index: int = 0) -> list[SsbHit]:
    """Find SS/PBCH blocks in `x` (complex, at 256*SCS, >= 20 ms).

    1. PSS: normalised time-domain correlation with nine templates (three
       N_ID2 x CFO_BANK); local maxima above `threshold` (a symbol apart)
       are candidates.
    2. CFO in three steps: the bank's coarse offset; the phase advance
       between the PSS symbol's halves (fractional, +-1 SCS); the PSS
       spectrum tried at -int_cfo_max..int_cfo_max subcarrier shifts
       (integer); then, after the SSS decision, the phase the channel-
       cancelled SSS has advanced over two symbols (fine: ~25 Hz median at
       10 dB, ~90 Hz at 0 dB, 15 kHz SCS — measured in test_nr_ssb).
    3. SSS: coherent against the PSS (same subcarriers), all 336 N_ID1.
    4. Confirmation: the same PCI at the same phase 20 ms later.

    Reach: |CFO| up to about one subcarrier (15 or 30 kHz) is found; beyond
    that the time-domain PSS correlation has decayed before the integer
    search can run. A survey tunes to an exact GSCN point, so the CFO seen is
    the SDR's own error — a few kHz at most on a calibrated bladeRF.

    `metric` does not move with gain (it is a normalised correlation), so a
    threshold is a statement about noise, not about signal level:
    test_nr_ssb measures the noise-only maximum over 40 ms (0.26 on twelve
    trials) and the default 0.30 sits above it. Below about -3 dB the
    detector stops finding the cell rather than inventing one."""
    x = np.asarray(x, np.complex64)
    fs = sample_rate(15e3 * (1 << mu))
    scs = fs / NFFT
    tpl0 = [pss_time_template(u) for u in range(N_ID2_MAX)]
    bank = [(u, c, pss_time_template(u, c))
            for u in range(N_ID2_MAX) for c in CFO_BANK]
    tabs = [sss_table(u) for u in range(N_ID2_MAX)]
    corr = np.stack([_xcorr_norm(x, t) for _, _, t in bank])   # (9, N-T+1)
    if corr.shape[1] == 0:
        return []
    best_i = corr.argmax(axis=0)
    best = corr.max(axis=0)
    # local maxima, at least a symbol apart, above threshold
    cand = np.flatnonzero(best > threshold)
    hits: list[SsbHit] = []
    order = cand[np.argsort(-best[cand])]
    taken: list[int] = []
    ks = sync_subcarriers()
    bins = k_to_bin(ks)
    for n0 in order:
        if len(hits) >= max_hits:
            break
        if any(abs(n0 - t0) < SYM for t0 in taken):
            continue
        taken.append(int(n0))
        u, c_sc, _ = bank[int(best_i[n0])]
        sss_start = n0 + 2 * SYM
        if sss_start + NFFT > x.size:
            continue
        # -- CFO ------------------------------------------------------------
        seg = x[n0:n0 + 3 * SYM]                     # PSS .. SSS symbols
        f_cfo = c_sc * scs                           # the bank's coarse guess
        seg = _derotate(seg, f_cfo, fs)
        f_frac = frac_cfo(seg[:NFFT], tpl0[u], fs)
        seg = _derotate(seg, f_frac, fs)
        f_cfo += f_frac
        Y_pss = np.fft.fft(seg[:NFFT])
        d = pss_d(u)
        best_int, best_val = 0, -1.0
        for di in range(-int_cfo_max, int_cfo_max + 1):
            v = abs(np.vdot(d, Y_pss[(bins + di) % NFFT]))
            if v > best_val:
                best_val, best_int = v, di
        if best_int:
            seg = _derotate(seg, best_int * scs, fs)
            f_cfo += best_int * scs
            # re-estimate the fraction now the integer part is gone
            f2 = frac_cfo(seg[:NFFT], tpl0[u], fs)
            if abs(f2) < scs:
                seg = _derotate(seg, f2, fs)
                f_cfo += f2
            Y_pss = np.fft.fft(seg[:NFFT])
        Y_sss = np.fft.fft(seg[2 * SYM:2 * SYM + NFFT])
        P = Y_pss[bins]
        S = Y_sss[bins]
        # -- SSS, coherent: channel cancels in S * conj(P) * d_pss ------------
        z = S * np.conj(P) * d
        scores = np.abs(tabs[u] @ z)
        norm = float(np.sum(np.abs(S) * np.abs(P))) + 1e-20
        i1 = int(scores.argmax())
        s1 = float(scores[i1] / norm)
        s2 = float(np.partition(scores, -2)[-2] / norm)
        # -- fine CFO: the phase the channel-cancelled SSS has advanced over
        # the two symbols since the PSS. Baseline 2*SYM samples, four times
        # the half-symbol estimate's, unambiguous to +-fs/(4*SYM) = +-0.23 SCS,
        # which the corrections above leave well inside.
        ph = float(np.angle(np.vdot(tabs[u][i1], z)))
        f_fine = ph / (2 * np.pi) * fs / (2 * SYM)
        f_cfo += f_fine
        hits.append(SsbHit(nid2=u, nid1=i1, pci=3 * i1 + u,
                           offset=int(n0) + int(start_index),
                           metric=float(best[n0]), sss_score=s1,
                           sss_margin=(s1 / s2 if s2 > 0 else float("inf")),
                           cfo_hz=f_cfo))
    _confirm(hits, fs)
    hits.sort(key=lambda h: (-h.confirmed, -h.metric))
    return hits


def _confirm(hits: list[SsbHit], fs: float, tol: int = 2) -> None:
    """Mark a hit confirmed when another hit with the same PCI sits one SSB
    period (20 ms, or a multiple) away in time, within `tol` samples. A
    cell sending 4 beams also puts hits 6 or 8 symbols apart inside one
    burst; those corroborate but do not confirm — a burst is one event."""
    period = int(round(SSB_PERIOD_MS * fs / 1e3))
    by_pci: dict[int, list[SsbHit]] = {}
    for h in hits:
        by_pci.setdefault(h.pci, []).append(h)
    for group in by_pci.values():
        for a in group:
            for b in group:
                if a is b:
                    continue
                dt = abs(a.offset - b.offset)
                if dt < period // 2:
                    continue
                r = dt % period
                if min(r, period - r) <= tol:
                    a.confirmed = True
                    b.confirmed = True


def summarise(hits: list[SsbHit]) -> dict[int, dict]:
    """Per-PCI summary: strongest metric, hit count, confirmed, mean CFO."""
    out: dict[int, dict] = {}
    for h in hits:
        s = out.setdefault(h.pci, {"pci": h.pci, "metric": 0.0, "hits": 0,
                                   "confirmed": False, "cfo_hz": 0.0,
                                   "sss_score": 0.0})
        s["hits"] += 1
        s["metric"] = max(s["metric"], h.metric)
        s["sss_score"] = max(s["sss_score"], h.sss_score)
        s["confirmed"] |= h.confirmed
        s["cfo_hz"] += (h.cfo_hz - s["cfo_hz"]) / s["hits"]
    return out
