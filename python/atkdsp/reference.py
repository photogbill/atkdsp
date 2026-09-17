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
