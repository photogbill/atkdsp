"""numpy twins of every atkdsp kernel — the specification, in the language the
tests are written in.

Each function here has the SAME semantics as the C kernel of the same name,
including streaming state (carried explicitly, as the C does), so
``tests/test_cross_check.py`` can feed both the same blocks and assert they
agree to float precision. They are also the fallback ATK uses when the
library is not present: slower, never different.

Only numpy is required. scipy is used for nothing here on purpose — the
reference must be runnable anywhere ATK is.
"""

from __future__ import annotations

import numpy as np

# ---- 1. unpack ----------------------------------------------------------------
_SCALE = {"cu8": 1 / 127.5, "ci8": 1 / 128.0, "ci16": 1 / 32768.0, "ci16_le": 1 / 32768.0,
          "cs16": 1 / 32768.0, "ci16q11": 1 / 2048.0, "cf32": 1.0, "cf32_le": 1.0}
_DTYPE = {"cu8": np.uint8, "ci8": np.int8, "ci16": "<i2", "ci16_le": "<i2", "cs16": "<i2",
          "ci16q11": "<i2", "cf32": "<f4", "cf32_le": "<f4"}


def bytes_per_sample(fmt: str) -> int:
    return 2 * np.dtype(_DTYPE[fmt]).itemsize


def unpack(raw, fmt: str, dc_alpha: float = 0.0, dc_state: np.ndarray | None = None) -> np.ndarray:
    item = np.dtype(_DTYPE[fmt]).itemsize
    buf = np.frombuffer(raw, dtype=np.uint8)
    usable = (buf.size // (2 * item)) * (2 * item)      # whole samples only
    b = buf[:usable].view(_DTYPE[fmt])
    f = b.astype(np.float32)
    if fmt == "cu8":
        f = f - np.float32(127.5)
    f = f * np.float32(_SCALE[fmt])
    out = np.empty(f.size // 2, dtype=np.complex64)
    out.real = f[0::2]
    out.imag = f[1::2]
    if dc_alpha > 0:
        s = complex(dc_state[0])
        a = float(dc_alpha)
        y = np.empty_like(out)
        for i in range(out.size):          # the C is a one-pole IIR: inherently serial
            s = s + a * (complex(out[i]) - s)
            y[i] = out[i] - s
        dc_state[0] = s
        return y
    return out


def unpack_dc(raw, fmt: str, offset: complex = 0j, out=None, want_mean: bool = True):
    """Twin of atkdsp.unpack_dc: (samples - offset, mean_before_subtraction)."""
    x = unpack(raw, fmt)
    m = complex(np.mean(x, dtype=np.complex128)) if x.size else 0j
    y = (x - np.complex64(offset)).astype(np.complex64) if offset else x
    return y, m


# ---- 2. NCO -------------------------------------------------------------------
class Nco:
    def __init__(self, freq_hz: float, sample_rate: float):
        self.phase = 0.0
        self.set_freq(freq_hz, sample_rate)

    def set_freq(self, freq_hz: float, sample_rate: float) -> None:
        self.step = -2.0 * np.pi * freq_hz / sample_rate if sample_rate > 0 else 0.0

    def mix(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.complex64)
        n = np.arange(x.size, dtype=np.float64)
        lo = np.exp(1j * (self.phase + self.step * n)).astype(np.complex64)
        self.phase = float(np.angle(np.exp(1j * (self.phase + self.step * x.size))))
        return (x * lo).astype(np.complex64)


# ---- 3. decimating FIR -------------------------------------------------------
class Fir:
    """y = convolve(x_all, h)[:len(x_all)][::decim], with the stream carried."""

    def __init__(self, taps, decim: int = 1):
        self.h = np.asarray(taps, dtype=np.float32)
        self.decim = int(decim)
        self.reset()

    def reset(self) -> None:
        self.hist = np.zeros(self.h.size - 1, dtype=np.complex64)
        self.pos = 0

    def process(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.complex64)
        full = np.concatenate([self.hist, x])
        # causal filter evaluated at every input index of x
        y = np.convolve(full.astype(np.complex128), self.h.astype(np.float64))
        y = y[self.hist.size: self.hist.size + x.size]
        first = (self.decim - self.pos) % self.decim
        out = y[first::self.decim].astype(np.complex64)
        H = self.hist.size
        if H:
            self.hist = full[-H:].copy()
        self.pos = (self.pos + x.size) % self.decim
        return out


# ---- 4. rational resampler ----------------------------------------------------
class Resampler:
    """upfirdn(h*up, x_all, up, down) restricted to outputs whose time has arrived."""

    def __init__(self, up: int, down: int, taps):
        self.up, self.down = int(up), int(down)
        self.h = np.asarray(taps, dtype=np.float32) * np.float32(self.up)
        self.Q = (self.h.size + self.up - 1) // self.up
        self.reset()

    def reset(self) -> None:
        self.hist = np.zeros(self.Q, dtype=np.complex64)
        self.t_rel = 0

    def process(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.complex64)
        L, M, T = self.up, self.down, self.h.size
        full = np.concatenate([self.hist, x])
        Q = self.Q
        outs = []
        t = self.t_rel
        span = x.size * L
        while t < span:
            n0 = t // L + Q           # index into `full`
            p = t % L
            acc = 0.0 + 0.0j
            j = p
            q = 0
            while j < T:
                k = n0 - q
                if k < 0:
                    break
                acc += float(self.h[j]) * complex(full[k])
                j += L
                q += 1
            outs.append(acc)
            t += M
        self.t_rel = t - span
        self.hist = full[-Q:].copy()
        return np.asarray(outs, dtype=np.complex64)


# ---- 5. FFT -------------------------------------------------------------------
def window(kind: str, n: int) -> np.ndarray:
    if n == 1:
        return np.ones(1, dtype=np.float32)
    k = np.arange(n, dtype=np.float64)
    x = 2 * np.pi * k / (n - 1)
    if kind == "rectangular":
        w = np.ones(n)
    elif kind == "hann":
        w = 0.5 - 0.5 * np.cos(x)
    elif kind == "hamming":
        w = 0.54 - 0.46 * np.cos(x)
    elif kind == "blackmanharris":
        w = 0.35875 - 0.48829 * np.cos(x) + 0.14128 * np.cos(2 * x) - 0.01168 * np.cos(3 * x)
    else:
        raise ValueError(kind)
    return w.astype(np.float32)


def fft(x, inverse: bool = False) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex64)
    return (np.fft.ifft(x) if inverse else np.fft.fft(x)).astype(np.complex64)


def power_db(x, win=None) -> np.ndarray:
    """Identical to atk.core.dsp.spectrum_db for a full-length frame."""
    x = np.asarray(x, dtype=np.complex64)
    n = x.size
    seg = x * np.asarray(win, dtype=np.float32) if win is not None else x
    spec = np.fft.fftshift(np.fft.fft(seg))
    mag = np.abs(spec) / n
    return (20.0 * np.log10(mag + 1e-12)).astype(np.float32)


def spectrum_reduce(x, n: int, hop: int | None = None, win=None, detector: str = "max"):
    x = np.asarray(x, dtype=np.complex64)
    hop = int(hop or n)
    if x.size < n:
        return np.full(n, -240.0, dtype=np.float32), 0
    frames = (x.size - n) // hop + 1
    w = np.asarray(win, dtype=np.float32) if win is not None else None
    acc = None
    for f in range(frames):
        seg = x[f * hop: f * hop + n]
        if w is not None:
            seg = seg * w
        spec = np.fft.fftshift(np.fft.fft(seg))
        pw = (np.abs(spec) ** 2) / (n * n)
        if acc is None:
            acc = pw.astype(np.float64)
        elif detector == "max":
            acc = np.maximum(acc, pw)
        elif detector == "min":
            acc = np.minimum(acc, pw)
        else:
            acc = acc + pw
    if detector == "avg":
        acc = acc / frames
    mag = np.sqrt(acc)
    return (20.0 * np.log10(mag + 1e-12)).astype(np.float32), frames


# ---- 6. demodulators ----------------------------------------------------------
def fm_demod(x, prev: np.ndarray | None = None, gain: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex64)
    p = np.complex64(0) if prev is None else prev[0]
    full = np.concatenate([[p], x]).astype(np.complex64)
    out = (np.angle(full[1:] * np.conj(full[:-1])) * gain).astype(np.float32)
    if prev is not None and x.size:
        prev[0] = x[-1]
    return out


def am_demod(x, dc_state: np.ndarray | None = None, alpha: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex64)
    env = np.abs(x).astype(np.float32)
    if alpha > 0 and dc_state is not None:
        s = float(dc_state[0])
        out = np.empty_like(env)
        for i in range(env.size):
            s += alpha * (float(env[i]) - s)
            out[i] = env[i] - s
        dc_state[0] = s
        return out
    return env


# ---- 7. display ---------------------------------------------------------------
def db_to_pixels(db, lo: float, hi: float, lut256) -> np.ndarray:
    db = np.asarray(db, dtype=np.float32)
    lut = np.asarray(lut256, dtype=np.uint32)
    span = hi - lo
    k = 255.0 / span if span > 0 else 0.0
    t = np.clip((db - lo) * k, 0.0, 255.0)
    idx = np.where(np.isnan(db), 0, (np.nan_to_num(t) + 0.5).astype(np.int64))
    idx = np.clip(idx, 0, 255)
    return lut[idx]


def decimate_max(x, m: int) -> np.ndarray:
    """Same partition as atk.ui.spectrum_view.decimate_max."""
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    if m >= n:
        return np.concatenate([x, np.full(m - n, x[-1], dtype=np.float32)])
    per = n // m
    used = per * m
    out = x[:used].reshape(m, per).max(axis=1).astype(np.float32)
    if used < n:
        out[-1] = max(float(out[-1]), float(x[used:].max()))
    return out


# ---- 8. detection -------------------------------------------------------------
def median(x) -> float:
    return float(np.median(np.asarray(x, dtype=np.float32)))


class Channel:
    def __init__(self, start_bin, end_bin, centroid_bin, peak_db, snr_db):
        self.start_bin, self.end_bin = int(start_bin), int(end_bin)
        self.centroid_bin, self.peak_db, self.snr_db = float(centroid_bin), float(peak_db), float(snr_db)

    def as_tuple(self):
        return (self.start_bin, self.end_bin, self.centroid_bin, self.peak_db, self.snr_db)


def smooth5(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size < 5:
        return x.copy()
    pad = np.pad(x, 2, mode="edge")
    return np.convolve(pad, np.ones(5, dtype=np.float32) / np.float32(5), mode="valid").astype(np.float32)


def detect_channels(line, floor_db: float, threshold_db: float, gap_bins: int, min_run: int) -> list:
    """atk.core.signal_id.detect_channels with the policy parameters made explicit."""
    line = np.asarray(line, dtype=np.float32)
    n = line.size
    sm = smooth5(line)
    thr = floor_db + threshold_db
    above = sm > thr
    gap = max(2, int(gap_bins))
    min_run = max(2, int(min_run))
    i = 0
    while i < n:
        if not above[i]:
            j = i
            while j < n and not above[j]:
                j += 1
            if i > 0 and j < n and (j - i) < gap:
                above[i:j] = True
            i = j
        else:
            i += 1
    out = []
    i = 0
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if j - i >= min_run:
                seg = line[i:j].astype(np.float64)
                w = np.power(10.0, seg / 10.0)
                centroid = i + float(np.sum(np.arange(j - i) * w) / (w.sum() + 1e-12))
                peak = float(seg.max())
                out.append(Channel(i, j, centroid, peak, peak - floor_db))
            i = j
        else:
            i += 1
    out.sort(key=lambda c: -c.snr_db)
    return out


def stitch_max(seg, seg_lo_hz, seg_hz_per_bin, keep_lo_hz, keep_hi_hz,
               out_lo_hz, out_hz_per_bin, out: np.ndarray) -> np.ndarray:
    """atk.core.sweep.stitch's inner loop, vectorised, in place on `out`."""
    seg = np.asarray(seg, dtype=np.float32)
    nout = out.size
    f = seg_lo_hz + (np.arange(seg.size) + 0.5) * seg_hz_per_bin
    out_hi = out_lo_hz + out_hz_per_bin * nout
    mask = (f >= keep_lo_hz) & (f <= keep_hi_hz) & (f >= out_lo_hz) & (f <= out_hi)
    idx = ((f[mask] - out_lo_hz) / out_hz_per_bin).astype(np.int64)
    np.clip(idx, 0, nout - 1, out=idx)
    vals = seg[mask]
    for k, v in zip(idx, vals):
        cur = out[k]
        if np.isnan(cur) or v > cur:
            out[k] = v
    return out


# ---- 9. filter design ----------------------------------------------------------
def _bessel_i0(x: float) -> float:
    s = 1.0; term = 1.0; y = x * 0.5
    for k in range(1, 500):
        term *= y / k
        t2 = term * term
        s += t2
        if t2 < s * 1e-17:
            break
    return s


def kaiser_beta(atten_db: float) -> float:
    if atten_db > 50.0:
        return 0.1102 * (atten_db - 8.7)
    if atten_db >= 21.0:
        return 0.5842 * (atten_db - 21.0) ** 0.4 + 0.07886 * (atten_db - 21.0)
    return 0.0


def kaiser_length(atten_db: float, transition_hz: float, rate: float) -> int:
    import math
    dw = 2.0 * math.pi * transition_hz / rate
    n = ((atten_db - 8.0) if atten_db > 8.0 else 1.0) / (2.285 * dw)
    N = int(math.ceil(n)) + 1
    if N < 3:
        N = 3
    if N % 2 == 0:
        N += 1
    return N


def design_lowpass(fp_hz: float, fs_hz: float, atten_db: float, rate: float) -> np.ndarray:
    """Line-for-line the C designer: double arithmetic, one cast per tap."""
    import math
    if rate <= 0 or fp_hz < 0 or fs_hz <= fp_hz or fs_hz > rate * 0.5 + 1e-9:
        raise ValueError("bad band edges")
    N = kaiser_length(atten_db, fs_hz - fp_hz, rate)
    beta = kaiser_beta(atten_db)
    i0b = _bessel_i0(beta)
    fc = 0.5 * (fp_hz + fs_hz) / rate
    M = (N - 1) * 0.5
    h = []
    for n in range(N):
        t = n - M
        x = 2.0 * fc * t
        sinc = 1.0 if t == 0.0 else math.sin(math.pi * x) / (math.pi * x)
        r = t / M
        w = _bessel_i0(beta * math.sqrt(1.0 - r * r if r * r < 1.0 else 0.0)) / i0b
        h.append(2.0 * fc * sinc * w)
    s = sum(h) or 1.0
    return np.array([np.float32(v / s) for v in h], dtype=np.float32)


# ---- 10. DDC --------------------------------------------------------------------
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def plan_factors(D: int) -> list:
    out = []
    while D > 1 and len(out) < 12:
        f = 0
        for c in range(8, 1, -1):
            if D % c == 0:
                f = c
                break
        if not f:
            for c in range(9, D + 1):
                if D % c == 0:
                    f = c
                    break
        out.append(f)
        D //= f
    return out


class Ddc:
    """NCO -> staged decimation -> exact-rate resampler, planned as the C does."""

    def __init__(self, sample_rate, offset_hz, channel_bw_hz, out_rate, atten_db=60.0):
        import math
        fs, bw = float(sample_rate), float(channel_bw_hz)
        atten_db = max(20.0, float(atten_db))
        self.fs, self.bw, self.out_rate_req, self.atten = fs, bw, float(out_rate), atten_db
        self.nco = Nco(offset_hz, fs)
        floor_rate = max(float(out_rate), 1.25 * bw)
        D = max(1, int(math.floor(fs / floor_rate)))
        self.r = fs / D
        self.factors = plan_factors(D)
        self.stages = []
        self.ntaps = []
        R = fs
        for i, f in enumerate(self.factors):
            Rn = R / f
            fp = 0.5 * bw
            fstop = Rn - 0.5 * bw
            if i == len(self.factors) - 1 and fstop > bw:
                fstop = bw
            if fstop > 0.5 * R:
                fstop = 0.5 * R
            h = design_lowpass(fp, fstop, atten_db, R)
            self.stages.append(Fir(h, f)); self.ntaps.append(h.size)
            R = Rn
        if not self.factors:
            fp = 0.5 * bw
            fstop = bw if bw < 0.5 * fs else 0.5 * fs
            h = design_lowpass(fp, fstop, atten_db, fs)
            self.stages.append(Fir(h, 1)); self.factors = [1]; self.ntaps = [h.size]
        self.up = self.down = 1
        self.rs = None
        if abs(self.r - float(out_rate)) > 1e-9 * float(out_rate):
            num, den = float(out_rate) * D, fs
            g = _gcd(int(num), int(den))
            self.up, self.down = int(num) // g, int(den) // g
            hi = self.r * self.up
            nyq = 0.5 * min(self.r, float(out_rate))
            h = design_lowpass(nyq * 0.8, nyq, atten_db, hi)
            self.rs = Resampler(self.up, self.down, h)

    @property
    def out_rate(self) -> float:
        return self.out_rate_req if self.rs is not None else self.r

    def plan(self):
        return list(self.factors), list(self.ntaps), self.up, self.down

    def set_offset(self, offset_hz: float) -> None:
        self.nco.set_freq(offset_hz, self.fs)

    def reset(self) -> None:
        self.nco.phase = 0.0
        for s in self.stages:
            s.reset()
        if self.rs is not None:
            self.rs.reset()

    def process(self, x) -> np.ndarray:
        y = self.nco.mix(x)
        for s in self.stages:
            y = s.process(y)
        if self.rs is not None:
            y = self.rs.process(y)
        return y


# ---- 11. LTE cell search ------------------------------------------------------
#
# 3GPP TS 36.211 §6.11.1 (PSS) and §6.11.2 (SSS), written out plainly. This is
# the specification the C in src/lte.c is held to.

LTE_RATE = 1_920_000        # 128-point FFT at 15 kHz subcarrier spacing
LTE_SYM = 128
LTE_PSS_PERIOD = 9600       # 5 ms: the PSS is in slots 0 and 10
LTE_SSS_BACK = 137          # the SSS is the symbol before, 9 CP + 128
_LTE_ROOTS = (25, 29, 34)


def lte_pss_values(nid2: int) -> np.ndarray:
    """d_u(n), the 62 Zadoff-Chu values. The exponent jumps at n=31 because
    the centre subcarrier (DC) is not used."""
    u = _LTE_ROOTS[int(nid2)]
    n = np.arange(62)
    k = np.where(n < 31, n * (n + 1), (n + 1) * (n + 2))
    return np.exp(-1j * np.pi * u * k / 63.0).astype(np.complex128)


def _lte_to_symbol(carr: np.ndarray) -> np.ndarray:
    """62 values -> the 128-sample time-domain symbol. -31..-1, DC skipped,
    +1..+31."""
    X = np.zeros(LTE_SYM, dtype=np.complex128)
    X[LTE_SYM - 31:] = carr[:31]
    X[1:32] = carr[31:]
    return (np.fft.ifft(X) * LTE_SYM / np.sqrt(62.0)).astype(np.complex64)


def lte_pss_symbol(nid2: int) -> np.ndarray:
    return _lte_to_symbol(lte_pss_values(nid2))


def _lte_mseq(taps) -> np.ndarray:
    x = [0, 0, 0, 0, 1]
    for i in range(31 - 5):
        x.append(int(sum(x[i + t] for t in taps) % 2))
    return 1 - 2 * np.array(x[:31], dtype=np.int8)


def lte_sss_symbol(nid1: int, nid2: int, subframe: int) -> np.ndarray:
    """The 62 +1/-1 values. Subframe 0 and 5 swap two of the sequences, which
    is how the SSS also tells you where you are in the frame."""
    S, C, Z = _lte_mseq((2, 0)), _lte_mseq((3, 0)), _lte_mseq((4, 2, 1, 0))
    qp = int(nid1) // 30
    q = (int(nid1) + qp * (qp + 1) // 2) // 30
    mp = int(nid1) + q * (q + 1) // 2
    m0, m1 = mp % 31, (mp % 31 + mp // 31 + 1) % 31
    n = np.arange(31)
    s0, s1 = S[(n + m0) % 31], S[(n + m1) % 31]
    c0, c1 = C[(n + nid2) % 31], C[(n + nid2 + 3) % 31]
    z0, z1 = Z[(n + (m0 % 8)) % 31], Z[(n + (m1 % 8)) % 31]
    d = np.zeros(62, dtype=np.float32)
    if int(subframe) == 0:
        d[0::2], d[1::2] = s0 * c0, s1 * c1 * z0
    else:
        d[0::2], d[1::2] = s1 * c0, s0 * c1 * z1
    return d


def lte_detect(x, min_metric: float = 0.06):
    """The strongest cell in `x` (which must be at LTE_RATE), or []."""
    x = np.asarray(x, dtype=np.complex128)
    n = x.size
    if n < 2 * LTE_PSS_PERIOD:
        return []
    valid = n - LTE_SYM
    e = np.convolve(np.abs(x) ** 2, np.ones(LTE_SYM), mode="valid")[:valid + 1]
    best = None
    for nid2 in range(3):
        p = np.asarray(lte_pss_symbol(nid2), dtype=np.complex128)
        pe = float(np.vdot(p, p).real)
        c = np.correlate(x, p, mode="valid")[:valid + 1]
        m = (np.abs(c) ** 2) / np.maximum(e * pe, 1e-20)
        k = int(np.argmax(m))
        if m[k] < min_metric:
            continue
        # the mate must reach half of THIS peak, not half of the threshold:
        # a cell's two occurrences are within a fade of each other, noise
        # clearing an absolute bar twice is not
        mate = any(0 <= k + d <= valid and m[k + d] > 0.5 * m[k]
                   for d in (-LTE_PSS_PERIOD, LTE_PSS_PERIOD))
        if not mate:
            continue
        if best is None or m[k] > best["metric"]:
            best = {"nid2": nid2, "offset": k, "metric": float(m[k])}
    if best is None:
        return []
    k, nid2 = best["offset"], best["nid2"]
    p = np.asarray(lte_pss_symbol(nid2), dtype=np.complex128)
    h1 = np.vdot(p[:64], x[k:k + 64])
    h2 = np.vdot(p[64:], x[k + 64:k + LTE_SYM])
    best["cfo_hz"] = float(np.angle(h2 * np.conj(h1)) * LTE_RATE / (2 * np.pi * 64))
    best.update(nid1=-1, pci=-1, subframe=-1, sss_score=0.0)
    at = k - LTE_SSS_BACK
    if at < 0:
        at += LTE_PSS_PERIOD
    if 0 <= at and at + LTE_SYM <= n:
        idx = np.r_[LTE_SYM - 31:LTE_SYM, 1:32]
        Yp, Ys = np.fft.fft(x[k:k + LTE_SYM]), np.fft.fft(x[at:at + LTE_SYM])
        H = Yp[idx] * np.conj(lte_pss_values(nid2))
        r = np.real(Ys[idx] * np.conj(H)) / (np.abs(H) + 1e-12)
        scores = [(float(np.dot(lte_sss_symbol(n1, nid2, sf), r)), n1, sf)
                  for n1 in range(168) for sf in (0, 5)]
        sc, n1, sf = max(scores)
        best.update(nid1=n1, subframe=sf, pci=3 * n1 + nid2, sss_score=sc / 62.0)
    return [best]


# ---- 11b. LTE PBCH -> MIB (the spec src/lte_pbch.c is held to) -----------
_PB_NFFT, _PB_NSC, _PB_DC = 128, 72, 36
_PB_CP0, _PB_CP1 = 10, 9
_PB_NRB_MAX = 110
_PB_SQ = 1.0 / np.sqrt(2.0)
_PB_PERM = np.array([1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31,
                     0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30])
_PB_GEN = (0o133, 0o171, 0o165)
LTE_PBCH_OFFSET = 128            # PSS data-start -> PBCH block start
LTE_PBCH_BLOCK = 549             # samples in the 4-symbol block


def lte_gold(c_init: int, length: int) -> np.ndarray:
    """36.211 7.2 length-31 Gold sequence, Nc=1600."""
    nc = 1600
    n = length + nc
    x1 = np.zeros(n + 31, dtype=np.int8); x1[0] = 1
    x2 = np.zeros(n + 31, dtype=np.int8)
    for i in range(31):
        x2[i] = (int(c_init) >> i) & 1
    for i in range(n):
        x1[i + 31] = (x1[i + 3] ^ x1[i]) & 1
        x2[i + 31] = (x2[i + 3] ^ x2[i + 2] ^ x2[i + 1] ^ x2[i]) & 1
    return np.array([(x1[i + nc] ^ x2[i + nc]) & 1 for i in range(length)],
                    dtype=np.int8)


def lte_crc16(bits) -> np.ndarray:
    """36.212 5.1.1 gCRC16 = D^16+D^12+D^5+1, MSB-first."""
    reg = 0
    for b in bits:
        reg = (reg << 1) | (int(b) & 1)
        if reg & (1 << 16):
            reg ^= 0x11021
    for _ in range(16):
        reg <<= 1
        if reg & (1 << 16):
            reg ^= 0x11021
    return np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=np.int8)


_PB_MASK = {1: np.zeros(16, np.int8), 2: np.ones(16, np.int8),
            4: np.array([0, 1] * 8, np.int8)}


def lte_conv_encode(bits) -> np.ndarray:
    """Tail-biting rate-1/3 conv code, 36.212 5.1.3.1."""
    bits = np.asarray(bits, np.int8)
    K = len(bits)
    reg = list(bits[-6:][::-1])
    taps = [[(g >> (6 - j)) & 1 for j in range(7)] for g in _PB_GEN]
    out = np.empty(3 * K, np.int8)
    for k in range(K):
        cur = int(bits[k]); full = [cur] + reg[:6]
        for i, t in enumerate(taps):
            out[3 * k + i] = sum(full[j] * t[j] for j in range(7)) & 1
        reg = [cur] + reg[:5]
    return out


def _pb_trellis():
    ob = np.zeros((64, 2, 3), np.int8)
    taps = [[(g >> (6 - j)) & 1 for j in range(7)] for g in _PB_GEN]
    for s in range(64):
        reg = [(s >> i) & 1 for i in range(6)]
        for cur in (0, 1):
            full = [cur] + reg
            for i, t in enumerate(taps):
                ob[s, cur, i] = sum(full[j] * t[j] for j in range(7)) & 1
    return ob
_PB_BM = (1 - 2 * _pb_trellis().reshape(128, 3)).astype(np.float64)
_PB_PA = np.arange(64) // 2
_PB_PB = np.arange(64) // 2 + 32


def lte_viterbi_tb(llr, laps: int = 3) -> np.ndarray:
    """Wrap-around Viterbi for the tail-biting code."""
    llr = np.asarray(llr, float)
    K = len(llr) // 3
    pm = np.zeros(64)
    steps = laps * K
    bp = np.empty((steps, 64), np.int32)
    L = llr.reshape(K, 3)
    step = 0
    for _ in range(laps):
        for k in range(K):
            cand = np.repeat(pm, 2) + (_PB_BM @ L[k])
            A, B = cand[:64], cand[64:]
            takeA = A >= B
            pm = np.where(takeA, A, B)
            bp[step] = np.where(takeA, _PB_PA, _PB_PB)
            step += 1
    s = int(np.argmax(pm))
    dec = np.zeros(steps, np.int8)
    for step in range(steps - 1, -1, -1):
        dec[step] = s & 1
        s = int(bp[step, s])
    mid = (laps // 2) * K
    return dec[mid:mid + K].copy()


def _pb_subblock_idx(D=40):
    C = 32; R = -(-D // C); nd = R * C - D
    seq = [-1] * nd + list(range(D))
    return np.array(seq).reshape(R, C)[:, _PB_PERM].reshape(-1, order='F')


def lte_rate_match(streams, E: int) -> np.ndarray:
    idx = _pb_subblock_idx(len(streams[0]))
    w = np.concatenate([np.array([-1 if i < 0 else int(s[i]) for i in idx], np.int8)
                        for s in streams])
    out = np.empty(E, np.int8); Kw = len(w); j = k = 0
    while k < E:
        v = w[j % Kw]; j += 1
        if v >= 0:
            out[k] = v; k += 1
    return out


def lte_rate_dematch(e_llr, D, seg):
    idx = _pb_subblock_idx(D); Rc = len(idx); Kw = 3 * Rc
    isnull = np.zeros(Kw, bool)
    for si in range(3):
        for pos, i in enumerate(idx):
            if i < 0:
                isnull[si * Rc + pos] = True
    w = np.zeros(Kw); start = seg[0]; j = produced = ei = 0
    target = seg[1] - seg[0]
    while produced < start:
        if not isnull[j % Kw]:
            produced += 1
        j += 1
    while ei < target:
        pos = j % Kw
        if not isnull[pos]:
            w[pos] += e_llr[ei]; ei += 1
        j += 1
    out = np.empty(3 * D)
    for si in range(3):
        col = w[si * Rc:(si + 1) * Rc]; dd = np.zeros(D)
        for pos, i in enumerate(idx):
            if i >= 0:
                dd[i] += col[pos]
        out[si::3] = dd
    return out


def _pb_qpsk(bits):
    b = np.asarray(bits, np.int8).reshape(-1, 2)
    return _PB_SQ * ((1 - 2 * b[:, 0]) + 1j * (1 - 2 * b[:, 1]))


def lte_crs_central(n_id, l):
    ci = (1 << 10) * (7 * 2 + l + 1) * (2 * n_id + 1) + 2 * n_id + 1
    c = lte_gold(ci, 4 * _PB_NRB_MAX)
    r = _PB_SQ * (1 - 2 * c[0::2]) + 1j * _PB_SQ * (1 - 2 * c[1::2])
    return r[_PB_NRB_MAX - 6:_PB_NRB_MAX + 6]


def lte_crs_pos(n_id, port, l):
    vs = n_id % 6
    if port == 0:   v = 0 if l == 0 else 3
    elif port == 1: v = 3 if l == 0 else 0
    elif port == 2: v = 0
    else:           v = 3
    return [6 * m + (v + vs) % 6 for m in range(12)]
def _pb_crs_sym(port):
    return 0 if port <= 1 else 1


def lte_pbch_re_map(n_id):
    used = {l: set() for l in range(4)}
    for port in range(4):
        l = _pb_crs_sym(port)
        for k in lte_crs_pos(n_id, port, l):
            used[l].add(k)
    return [(k, l) for l in range(4) for k in range(_PB_NSC) if k not in used[l]]


def lte_precode(y, n_ports):
    n = len(y); x = np.zeros((n_ports, n), complex); S = _PB_SQ
    if n_ports == 1:
        x[0] = y
    elif n_ports == 2:
        for i in range(0, n, 2):
            x[0, i], x[1, i] = S*y[i], -S*np.conj(y[i+1])
            x[0, i+1], x[1, i+1] = S*y[i+1], S*np.conj(y[i])
    else:
        for i in range(0, n, 4):
            x[0, i], x[2, i] = S*y[i], -S*np.conj(y[i+1])
            x[0, i+1], x[2, i+1] = S*y[i+1], S*np.conj(y[i])
            x[1, i+2], x[3, i+2] = S*y[i+2], -S*np.conj(y[i+3])
            x[1, i+3], x[3, i+3] = S*y[i+3], S*np.conj(y[i+2])
    return x


def _pb_alamouti(r0, r1, h0, h1):
    y0 = np.conj(h0)*r0 + h1*np.conj(r1)
    y1 = np.conj(h0)*r1 - h1*np.conj(r0)
    return y0, y1


def _pb_kbin(k):
    return (k - _PB_DC) % _PB_NFFT
def _pb_ofdm_mod(grid, first):
    X = np.zeros(_PB_NFFT, complex)
    for k in range(_PB_NSC):
        X[_pb_kbin(k)] = grid[k]
    xt = np.fft.ifft(X) * _PB_NFFT / np.sqrt(_PB_NSC)
    cp = _PB_CP0 if first else _PB_CP1
    return np.concatenate([xt[-cp:], xt])
def _pb_ofdm_demod(samples, first):
    cp = _PB_CP0 if first else _PB_CP1
    X = np.fft.fft(samples[cp:cp + _PB_NFFT])
    return np.array([X[_pb_kbin(k)] for k in range(_PB_NSC)])


def lte_mib_pack(dl_bw, phich_dur, phich_res, sfn):
    bw = {6: 0, 15: 1, 25: 2, 50: 3, 75: 4, 100: 5}[dl_bw]
    bits = [(bw >> (2-i)) & 1 for i in range(3)]
    bits += [phich_dur & 1] + [(phich_res >> (1-i)) & 1 for i in range(2)]
    bits += [((sfn >> 2) >> (7-i)) & 1 for i in range(8)] + [0]*10
    return np.array(bits, np.int8)
def _pb_unpack(b, i0):
    bw = {0:6,1:15,2:25,3:50,4:75,5:100}
    code = int(b[0])*4 + int(b[1])*2 + int(b[2])
    if code > 5 or np.any(b[14:24] != 0):
        return None
    sfn_msb = 0
    for x in b[6:14]:
        sfn_msb = (sfn_msb << 1) | int(x)
    return dict(dl_bw_rb=bw[code], phich_dur=int(b[3]),
                phich_res=int(b[4])*2+int(b[5]), sfn=(sfn_msb << 2) | i0)


def lte_build_pbch_samples(mib24, n_id, n_ports, sfn, H_ports,
                           snr_db=100.0, seed=0):
    """Synthesize the 4 PBCH OFDM symbols (slot 1, subframe 0) as samples."""
    i = sfn & 3
    frame = np.concatenate([np.asarray(mib24, np.int8),
                            lte_crc16(mib24) ^ _PB_MASK[n_ports]]).astype(np.int8)
    coded = lte_conv_encode(frame)
    b = lte_rate_match([coded[0::3], coded[1::3], coded[2::3]], 1920)
    bscr = (b ^ lte_gold(n_id, 1920)).astype(np.int8)
    y = _pb_qpsk(bscr[i*480:(i+1)*480])
    x = lte_precode(y, n_ports)
    remap = lte_pbch_re_map(n_id)
    grids = [np.zeros(_PB_NSC, complex) for _ in range(4)]
    for idx, (k, l) in enumerate(remap):
        grids[l][k] = sum(H_ports[p][k] * x[p, idx] for p in range(n_ports))
    for p in range(n_ports):
        l = _pb_crs_sym(p)
        r = lte_crs_central(n_id, l); pos = lte_crs_pos(n_id, p, l)
        for m, k in enumerate(pos):
            grids[l][k] += H_ports[p][k] * r[m]
    sig = np.concatenate([_pb_ofdm_mod(grids[l], l == 0) for l in range(4)])
    if snr_db < 90:
        rng = np.random.default_rng(seed)
        npow = np.mean(np.abs(sig)**2) / (10**(snr_db/10))
        sig = sig + np.sqrt(npow/2)*(rng.standard_normal(len(sig))
                                     + 1j*rng.standard_normal(len(sig)))
    return sig.astype(np.complex64)


def _pb_channels(Y, n_id):
    H = {}
    for p in range(4):
        l = _pb_crs_sym(p)
        r = lte_crs_central(n_id, l); pos = lte_crs_pos(n_id, p, l)
        hk = np.array([Y[l][k] / r[m] for m, k in enumerate(pos)])
        pos = np.array(pos)
        H[p] = (np.interp(np.arange(_PB_NSC), pos, hk.real)
                + 1j*np.interp(np.arange(_PB_NSC), pos, hk.imag))
    return H
def _pb_soft(Y, H, n_ports, remap):
    yhat = np.zeros(240, complex)
    if n_ports == 1:
        for idx, (k, l) in enumerate(remap):
            yhat[idx] = np.conj(H[0][k]) * Y[l][k]
    elif n_ports == 2:
        for idx in range(0, 240, 2):
            k0, l0 = remap[idx]; k1, l1 = remap[idx+1]
            yhat[idx], yhat[idx+1] = _pb_alamouti(Y[l0][k0], Y[l1][k1],
                                                  H[0][k0], H[1][k0])
    else:
        for idx in range(0, 240, 4):
            for g in range(2):
                a, b = (0, 2) if g == 0 else (1, 3)
                i0 = idx+2*g; i1 = idx+2*g+1
                k0, l0 = remap[i0]; k1, l1 = remap[i1]
                yhat[i0], yhat[i1] = _pb_alamouti(Y[l0][k0], Y[l1][k1],
                                                  H[a][k0], H[b][k0])
    llr = np.empty(480); llr[0::2] = yhat.real; llr[1::2] = yhat.imag
    return llr
def _pb_derotate(block, cfo_hz):
    if not cfo_hz:
        return block
    n = np.arange(len(block))
    return block * np.exp(-2j*np.pi*cfo_hz*n/LTE_RATE)
def _pb_demod(block, cfo_hz):
    block = _pb_derotate(np.asarray(block, complex), cfo_hz)
    offs = [0, _PB_CP0+_PB_NFFT, _PB_CP0+_PB_NFFT+_PB_CP1+_PB_NFFT,
            _PB_CP0+_PB_NFFT+2*(_PB_CP1+_PB_NFFT)]
    return [_pb_ofdm_demod(block[offs[l]:], l == 0) for l in range(4)]


def lte_mib_decode_frames(blocks, n_id, cfo_hz=0.0):
    """Soft-combine 1..N consecutive PBCH blocks; single-frame first, then
    every aligned 4-frame window. Returns a MIB dict or None."""
    remap = lte_pbch_re_map(n_id); c = lte_gold(n_id, 1920)
    frames = [(_pb_demod(b, cfo_hz),) for b in blocks]
    frames = [(Y, _pb_channels(Y, n_id)) for (Y,) in frames]
    for wlen in (1, 4):
        for fs in range(0, len(frames) - wlen + 1):
            for n_ports in (1, 2, 4):
                raw = [_pb_soft(frames[fs+f][0], frames[fs+f][1], n_ports, remap)
                       for f in range(wlen)]
                for i0 in range(4):
                    acc = np.zeros(120)
                    for f in range(wlen):
                        seg = (i0 + f) % 4
                        cc = c[seg*480:(seg+1)*480]
                        acc += lte_rate_dematch((1 - 2*cc) * raw[f], 40,
                                                (seg*480, (seg+1)*480))
                    bits = lte_viterbi_tb(acc)
                    calc = lte_crc16(bits[:24]) ^ _PB_MASK[n_ports]
                    if not np.array_equal(calc & 1, bits[24:] & 1):
                        continue
                    info = _pb_unpack(bits[:24], i0)
                    if info:
                        info['n_ports'] = n_ports
                        return info
    return None


def lte_mib_decode(block, n_id, cfo_hz=0.0):
    """Decode the MIB from a single PBCH block. Returns a dict or None."""
    return lte_mib_decode_frames([block], n_id, cfo_hz)


# ---- 11c. Passive PRACH detector (the spec src/lte_prach.c is held to) ----
PRACH_NZC = 839
PRACH_MIN_METRIC = 40.0
_PRACH_COUNT_FRAC = 0.30


def prach_zc(u: int, N: int = PRACH_NZC) -> np.ndarray:
    """Frequency-domain Zadoff-Chu root x_u(k), 36.211 5.7.2."""
    n = np.arange(N)
    return np.exp(-1j * np.pi * u * n * (n + 1) / N)


def prach_preamble_time(u: int, shift: int = 0, N: int = PRACH_NZC) -> np.ndarray:
    """One preamble's sequence window in time (N samples at the PRACH rate)."""
    xu = prach_zc(u, N) * np.exp(-2j * np.pi * np.arange(N) * shift / N)
    return np.fft.ifft(xu)


def prach_build_capture(accesses, N=PRACH_NZC, snr_db=10.0, channel=None,
                        seed=0, timing=0) -> np.ndarray:
    """A window of PRACH-band samples with the given (root, shift) accesses."""
    y = np.zeros(N, dtype=complex)
    for (u, sh) in accesses:
        y += (1.0 if channel is None else channel) * prach_preamble_time(u, sh, N)
    if timing:
        y = np.roll(y, timing)
    if snr_db < 90:
        rng = np.random.default_rng(seed)
        p = np.mean(np.abs(y) ** 2) if np.any(y) else 1.0
        npow = p / (10 ** (snr_db / 10.0))
        y = y + np.sqrt(npow / 2) * (rng.standard_normal(N)
                                     + 1j * rng.standard_normal(N))
    return y.astype(np.complex64)


def prach_detect(seq, min_metric=PRACH_MIN_METRIC, N=PRACH_NZC,
                 count_thresh=_PRACH_COUNT_FRAC):
    """Detect PRACH preambles. Returns [{root, count, metric, delay}], the
    strongest tone first."""
    seq = np.asarray(seq, complex)
    Y = np.fft.fft(seq)
    D = Y[1:] * np.conj(Y[:-1])
    S = np.abs(np.fft.fft(D, N)) ** 2
    med = float(np.median(S)) + 1e-30
    out = []
    claimed = np.zeros(N, dtype=bool)
    while True:
        b = int(np.argmax(np.where(claimed, -1.0, S)))
        if claimed[b] or S[b] / med < min_metric:
            break
        for d in range(-2, 3):
            claimed[(b + d) % N] = True
        u = (N - b) % N
        if u == 0:
            continue
        pdp = np.abs(np.fft.ifft(Y * np.conj(prach_zc(u, N)))) ** 2
        top = pdp.max()
        thr = count_thresh * top
        count, delay, dtop = 0, 0, -1.0
        for k in range(N):
            if (pdp[k] >= thr and pdp[k] >= pdp[(k - 1) % N]
                    and pdp[k] > pdp[(k + 1) % N]):
                count += 1
                if pdp[k] > dtop:
                    dtop, delay = pdp[k], k
        out.append({"root": u, "count": count,
                    "metric": float(S[b] / med), "delay": delay})
    return out


def prach_scan(stream, win_step=64, min_metric=PRACH_MIN_METRIC, N=PRACH_NZC):
    """Slide an N-sample window across a longer capture; merge hits by root."""
    stream = np.asarray(stream, complex)
    merged = {}
    start = 0
    while start + N <= len(stream):
        for h in prach_detect(stream[start:start + N], min_metric, N):
            r = h["root"]
            if r not in merged:
                merged[r] = dict(h)
            else:
                if h["metric"] > merged[r]["metric"]:
                    merged[r]["metric"] = h["metric"]
                    merged[r]["delay"] = h["delay"]
                merged[r]["count"] = max(merged[r]["count"], h["count"])
        start += win_step
    return list(merged.values())
