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


def iq_health(x, clip_level: float = 0.99) -> dict:
    """Twin of atkdsp_iq_health: DC, gain/phase imbalance, image rejection and
    clip fraction of a converted block."""
    z = np.asarray(x, dtype=np.complex64)
    n = z.size
    out = {"dc_re": 0.0, "dc_im": 0.0, "rms": 0.0, "gain_imbalance_db": 0.0,
           "phase_error_deg": 0.0, "image_rejection_db": 0.0,
           "clip_fraction": 0.0, "n": n}
    if n == 0:
        return out
    thr = clip_level if clip_level > 0 else 0.99
    I = z.real.astype(np.float64)
    Q = z.imag.astype(np.float64)
    mI, mQ = float(I.mean()), float(Q.mean())
    out["dc_re"], out["dc_im"] = mI, mQ
    out["rms"] = float(np.sqrt(np.mean(I * I + Q * Q)))
    vI = max(0.0, float(I.var()))
    vQ = max(0.0, float(Q.var()))
    cIQ = float(((I - mI) * (Q - mQ)).mean())
    if vI > 1e-30 and vQ > 1e-30:
        out["gain_imbalance_db"] = 10.0 * np.log10(vI / vQ)
        s = max(-1.0, min(1.0, cIQ / np.sqrt(vI * vQ)))
        phi = float(np.arcsin(s))
        out["phase_error_deg"] = np.degrees(phi)
        a = np.sqrt(vI / vQ)
        c = np.cos(phi)
        num, den = a * a + 1 + 2 * a * c, a * a + 1 - 2 * a * c
        out["image_rejection_db"] = 10.0 * np.log10(num / den) if den > 1e-30 else 1000.0
    out["clip_fraction"] = float(np.mean(
        (np.abs(I) >= thr) | (np.abs(Q) >= thr)))
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


# ---- 4b. arbitrary (fractional) resampler -------------------------------------
class ArbResampler:
    """Resample by ANY positive out/in ratio — the twin of atkdsp_arb_resampler.

    A prototype low-pass designed at in_rate*P is split into P polyphase
    branches (interpolate-by-P, gain P folded in). An output at continuous input
    position ``pos`` takes i=floor(pos) as the newest input and frac=pos-i as the
    sub-sample delay; the branch b=floor(frac*P) is applied and LINEARLY
    interpolated toward the next branch (a first-order Farrow) by mu=frac*P-b,
    using the forward-difference bank d[k]=p[k+1]-p[k] (p[PQ]=0). ``pos`` steps
    by in/out per output and is carried across blocks, so a block boundary is
    invisible. Same newest-first FIR convention as the rational Resampler.
    """

    def __init__(self, in_rate: float, out_rate: float, atten_db: float = 60.0,
                 nphase: int = 64):
        if in_rate <= 0.0 or out_rate <= 0.0:
            raise ValueError("in_rate and out_rate must be positive")
        if atten_db <= 0.0:
            atten_db = 60.0
        P = int(nphase) if nphase else 64
        nyq = 0.5 * min(in_rate, out_rate)     # anti-alias down, don't widen up
        proto = design_lowpass(0.8 * nyq, nyq, atten_db, in_rate * P)
        nt = proto.size
        Q = (nt + P - 1) // P
        PQ = P * Q
        p = np.zeros(PQ, dtype=np.float64)
        p[:nt] = proto.astype(np.float64)
        p *= float(P)                          # interpolate-by-P gain
        d = np.empty(PQ, dtype=np.float64)
        d[:PQ - 1] = p[1:] - p[:-1]
        d[PQ - 1] = -p[PQ - 1]                 # p[PQ] == 0
        self.P, self.Q = P, Q
        self.p, self.d = p, d
        self.in_rate = float(in_rate)
        self.out_rate = float(out_rate)
        self.step = in_rate / out_rate         # input samples per output
        self.reset()

    def reset(self) -> None:
        self.hist = np.zeros(self.Q, dtype=np.complex64)   # hist[Q-1] = x[-1]
        # Blocking-invariant position: output K sits at base + K*step (since the
        # last anchor), in the block that starts at the exact integer input
        # offset in_off. Because in_off is an integer and K*step is a pure
        # function of K, the same output lands identically however the stream is
        # cut into blocks — matching the C, which no longer accumulates `pos`.
        self.base = 0.0
        self.K = 0
        self.in_off = 0

    def set_ratio(self, ratio: float) -> None:
        """Nudge the ratio (out/in) live, WITHOUT rebuilding the prototype —
        for a few-ppm sample-clock correction. A large change wants a rebuild,
        since the prototype's cutoff was fixed at create."""
        if ratio <= 0.0:
            return
        self.base = self.base + self.K * self.step     # re-anchor: keep next output
        self.K = 0
        self.out_rate = self.in_rate * ratio
        self.step = 1.0 / ratio

    def ratio(self) -> float:
        return 1.0 / self.step if self.step > 0.0 else 0.0

    def out_max(self, n_in: int) -> int:
        if self.step <= 0.0:
            return 0
        local = self.base + self.K * self.step - self.in_off
        if local >= n_in:
            return 0
        return int((n_in - local) / self.step) + 2

    def process(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.complex64)
        n = x.size
        P, Q = self.P, self.Q
        p, d = self.p, self.d
        step, base, inoff = self.step, self.base, self.in_off
        # hist (Q) then x, so full[Q + m] addresses x[m] and full[Q-1] is x[-1] —
        # the same newest-first indexing the C does with in[] and its history.
        full = np.concatenate([self.hist, x])
        qidx = P * np.arange(Q)
        outs = []
        K = self.K
        local = base + K * step - inoff
        while local < n:
            i = int(np.floor(local))
            frac = local - i
            ph = frac * P
            b = int(ph)
            if b >= P:
                b = P - 1
            mu = ph - b
            c = p[b + qidx] + mu * d[b + qidx]        # Farrow coefficients, q=0..Q-1
            bi = Q + i
            seg = full[bi - Q + 1: bi + 1][::-1]       # x[i], x[i-1], ..., x[i-(Q-1)]
            outs.append(np.dot(c, seg))
            K += 1
            local = base + K * step - inoff
        self.K = K
        self.in_off += n
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


def spectrum_stats(x, n: int, hop: int | None = None, win=None):
    """Per-bin max/avg/min (dB) and spectral kurtosis in one pass. The readable
    statement of atkdsp_spectrum_stats. Returns (max_db, avg_db, min_db, sk,
    frames); sk is dimensionless, ~2 for Gaussian noise and ~1 for a tone."""
    x = np.asarray(x, dtype=np.complex64)
    hop = int(hop or n)
    if x.size < n:
        z = np.full(n, -240.0, dtype=np.float32)
        return z, z.copy(), z.copy(), np.ones(n, np.float32), 0
    frames = (x.size - n) // hop + 1
    w = np.asarray(win, dtype=np.float32) if win is not None else None
    powers = np.empty((frames, n), dtype=np.float64)
    for f in range(frames):
        seg = x[f * hop: f * hop + n]
        if w is not None:
            seg = seg * w
        spec = np.fft.fftshift(np.fft.fft(seg))
        powers[f] = (np.abs(spec) ** 2) / (n * n)
    su = powers.sum(axis=0)
    s2 = (powers ** 2).sum(axis=0)
    mx = powers.max(axis=0)
    mn = powers.min(axis=0)
    max_db = (10.0 * np.log10(mx + 1e-12)).astype(np.float32)
    min_db = (10.0 * np.log10(mn + 1e-12)).astype(np.float32)
    avg_db = (10.0 * np.log10(su / frames + 1e-12)).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        sk = np.where((frames >= 2) & (su > 0.0),
                      frames * s2 / (su * su), 1.0).astype(np.float32)
    return max_db, avg_db, min_db, sk, frames


def window_stats(win) -> tuple:
    """(coherent_gain, enbw_bins) of a window — the twin of atkdsp_window_stats.
    coherent_gain = mean(w); enbw_bins = n*sum(w^2)/sum(w)^2."""
    w = np.asarray(win, dtype=np.float64)
    n = w.size
    s = float(w.sum())
    s2 = float((w * w).sum())
    cg = s / n if n else 0.0
    enbw = (n * s2 / (s * s)) if s > 0.0 else 0.0
    return cg, enbw


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
        self.arb = None
        if abs(self.r - float(out_rate)) > 1e-9 * float(out_rate):
            num, den = float(out_rate) * D, fs
            if num > 4e9 or den > 4e9 or math.floor(num) != num or math.floor(den) != den:
                # a real-number ratio (an odd sample clock, a huge denominator):
                # the arbitrary resampler, as the C does.
                self.arb = ArbResampler(self.r, float(out_rate), atten_db)
            else:
                g = _gcd(int(num), int(den))
                self.up, self.down = int(num) // g, int(den) // g
                hi = self.r * self.up
                nyq = 0.5 * min(self.r, float(out_rate))
                # widened to a channel wider than 0.8 of Nyquist, as the C
                fp = nyq * 0.8
                if 0.5 * bw > fp:
                    fp = min(0.5 * bw, 0.97 * nyq)
                h = design_lowpass(fp, nyq, atten_db, hi)
                self.rs = Resampler(self.up, self.down, h)

    @property
    def out_rate(self) -> float:
        return (self.out_rate_req if (self.rs is not None or self.arb is not None)
                else self.r)

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
        if self.arb is not None:
            self.arb.reset()

    def process(self, x) -> np.ndarray:
        y = self.nco.mix(x)
        for s in self.stages:
            y = s.process(y)
        if self.rs is not None:
            y = self.rs.process(y)
        elif self.arb is not None:
            y = self.arb.process(y)
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


#: CFAR margin (sigmas) for the non-coherent PSS fold; matches PSS_NI_K in
#: src/lte.c.
LTE_PSS_NI_K = 9.0


#: Secondary-cell gates after cancellation — must match src/lte.c. The metric
#: is decisive: a real cell dominates the residual (~0.8-1.0), a stub is ~0.07.
LTE_SEC_METRIC_MIN = 0.30
LTE_SEC_SSS_MIN = 0.5


def _lte_detect_one(x, valid, min_metric):
    """The proven single-cell detector on one buffer: strongest PSS peak with a
    5 ms mate, else the non-coherent fold + CFAR. Returns (nid2, k, metric) or
    None. Mirrors detect_one in src/lte.c."""
    P = LTE_PSS_PERIOD
    e = np.convolve(np.abs(x) ** 2, np.ones(LTE_SYM), mode="valid")[:valid + 1]
    best = None
    ni = None
    for nid2 in range(3):
        p = np.asarray(lte_pss_symbol(nid2), dtype=np.complex128)
        pe = float(np.vdot(p, p).real)
        c = np.correlate(x, p, mode="valid")[:valid + 1]
        m = (np.abs(c) ** 2) / np.maximum(e * pe, 1e-20)
        k = int(np.argmax(m))
        if m[k] >= min_metric:
            mate = any(0 <= k + d <= valid and m[k + d] > 0.5 * m[k]
                       for d in (-P, P))
            if mate and (best is None or m[k] > best[2]):
                best = (nid2, k, float(m[k]))
        nb = (valid + 1) // P
        if nb >= 1:
            mf = m[:nb * P].reshape(nb, P).mean(axis=0)
            rpk = int(np.argmax(mf))
            other = np.delete(mf, rpk)
            mean = float(other.mean()); sd = float(other.std())
            margin = (mf[rpk] - mean) / sd if sd > 1e-30 else 0.0
            if margin >= LTE_PSS_NI_K and (ni is None or margin > ni[0]):
                occ = np.arange(rpk, valid + 1, P)
                bk = int(occ[np.argmax(m[occ])])
                ni = (margin, nid2, bk, float(m[bk]))
    if best is not None:
        return best
    if ni is not None:
        return (ni[1], ni[2], ni[3])
    return None


def _lte_project_out(buf, ref, energy, at):
    if energy <= 1e-20:
        return
    seg = buf[at:at + LTE_SYM]
    g = np.vdot(ref, seg) / energy      # <ref, seg> / |ref|^2
    buf[at:at + LTE_SYM] = seg - g * ref


def _lte_subtract_cell(buf, n, cell):
    """Remove a decoded cell's PSS (every 5 ms) and SSS (137 before each PSS,
    subframe alternating). Mirrors subtract_cell in src/lte.c."""
    P = LTE_PSS_PERIOD
    nid2 = cell["nid2"]
    pss = np.asarray(lte_pss_symbol(nid2), dtype=np.complex128)
    pe = float(np.vdot(pss, pss).real)
    have_sss = cell["nid1"] >= 0
    if have_sss:
        def sss_time(sf):
            v = np.asarray(lte_sss_symbol(cell["nid1"], nid2, sf), float)
            g = np.zeros(LTE_SYM, complex)
            g[LTE_SYM - 31:] = v[:31]; g[1:32] = v[31:]
            return np.fft.ifft(g) * LTE_SYM / np.sqrt(62)
        sss = {0: sss_time(0), 5: sss_time(5)}
        se = {sf: float(np.vdot(s, s).real) for sf, s in sss.items()}
    base = cell["offset"] % P
    j0 = (cell["offset"] - base) // P
    kk = base
    while kk + LTE_SYM <= n:
        j = (kk - base) // P
        sf = cell["subframe"]
        if (j - j0) & 1:
            sf = 5 if sf == 0 else 0
        _lte_project_out(buf, pss, pe, kk)
        if have_sss:
            a = kk - LTE_SSS_BACK
            if 0 <= a and a + LTE_SYM <= n:
                _lte_project_out(buf, sss[sf], se[sf], a)
        kk += P


def _lte_decode_cell(x, n, nid2, k, metric):
    """CFO + SSS for one (nid2, offset). Mirrors decode_cell in src/lte.c."""
    p = np.asarray(lte_pss_symbol(nid2), dtype=np.complex128)
    cell = {"nid2": nid2, "offset": int(k), "metric": float(metric),
            "nid1": -1, "pci": -1, "subframe": -1, "sss_score": 0.0}
    h1 = np.vdot(p[:64], x[k:k + 64])
    h2 = np.vdot(p[64:], x[k + 64:k + LTE_SYM])
    cell["cfo_hz"] = float(np.angle(h2 * np.conj(h1)) * LTE_RATE / (2 * np.pi * 64))
    at = k - LTE_SSS_BACK
    if at < 0:
        at += LTE_PSS_PERIOD
    if 0 <= at and at + LTE_SYM <= n:
        idx = np.r_[LTE_SYM - 31:LTE_SYM, 1:32]
        Yp, Ys = np.fft.fft(x[k:k + LTE_SYM]), np.fft.fft(x[at:at + LTE_SYM])
        H = Yp[idx] * np.conj(lte_pss_values(nid2))
        # the SSS equalised by the PSS, kept COMPLEX, and scored by |sum|:
        # phase-invariant, as src/lte.c decode_cell (2026-09-26) — a carrier
        # offset turns the SSS against its PSS reference by 2*pi*f*71us
        r = Ys[idx] * np.conj(H) / (np.abs(H) + 1e-12)
        scores = [(float(abs(np.dot(lte_sss_symbol(n1, nid2, sf), r))), n1, sf)
                  for n1 in range(168) for sf in (0, 5)]
        sc, n1, sf = max(scores)
        cell.update(nid1=n1, subframe=sf, pci=3 * n1 + nid2, sss_score=sc / 62.0)
    return cell


def lte_detect(x, min_metric: float = 0.06, cap: int = 8):
    """Every co-channel cell in `x` (at LTE_RATE), strongest first, or []. Mirrors
    src/lte.c: successive interference cancellation. Detect the strongest cell
    (pass 0 is the old single-cell result), decode it, SUBTRACT its PSS/SSS from
    a residual copy, and detect again — only after the strong cell is removed
    does a weaker co-channel cell rise above the interference floor with an
    uncorrupted SSS. Stops on nothing found, a repeated PCI, or `cap`. See
    atkdsp_lte_detect."""
    x = np.asarray(x, dtype=np.complex128)
    n = x.size
    if n < 2 * LTE_PSS_PERIOD:
        return []
    valid = n - LTE_SYM
    work = x.copy()
    out = []
    while len(out) < cap:
        buf = x if not out else work
        got = _lte_detect_one(buf, valid, min_metric)
        if got is None:
            break
        nid2, k, metric = got
        cell = _lte_decode_cell(buf, n, nid2, k, metric)
        if out:
            # a residual pass is only believed with a strong post-cancellation
            # metric and a real SSS; below either it is a stub. A repeated PCI
            # is the same cell resurfacing.
            if (cell["pci"] < 0 or metric < LTE_SEC_METRIC_MIN
                    or cell["sss_score"] < LTE_SEC_SSS_MIN
                    or any(o["pci"] == cell["pci"] for o in out)):
                break
        out.append(cell)
        if len(out) >= cap:
            break
        if len(out) == 1:
            work = x.copy()
        _lte_subtract_cell(work, n, cell)
    return out


# ---- 11b. LTE PBCH -> MIB (the spec src/lte_pbch.c is held to) -----------
_PB_NFFT, _PB_NSC, _PB_DC = 128, 72, 36
_PB_CP0, _PB_CP1 = 10, 9
_PB_NRB_MAX = 110
_PB_SQ = 1.0 / np.sqrt(2.0)
#: Transform-domain CRS channel-estimate denoising: keep this many of the 12
#: IDFT "delay" taps (the channel lives in the first few; noise fills all 12).
#: Must match PBCH_CHEST_NTAP in src/lte_pbch.c. 0 disables it.
_PB_CHEST_NTAP = 4
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
    elif port == 2: v = 3          # slot 1 (odd): v = 3(n_s mod 2)
    else:           v = 0          # 3 + 3(n_s mod 2) = 6 -> 0
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
    """Central-72 subcarrier k -> FFT bin, DC skipped (36.211 6.6.4). See
    src/lte_pbch.c demod_block for the 2026-09-26 correction."""
    off = k - _PB_DC
    return off + 1 if off >= 0 else off + _PB_NFFT
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


def lte_mib_pack(dl_bw, phich_dur, phich_res, sfn, sib1_br=0,
                 si_unchanged_br=0):
    """36.331 MIB, 24 bits. `sib1_br` is schedulingInfoSIB1-BR-r13 (0..31,
    non-zero on a cell running LTE-M) and `si_unchanged_br` is
    systemInfoUnchanged-BR-r15; the last four bits are spare."""
    bw = {6: 0, 15: 1, 25: 2, 50: 3, 75: 4, 100: 5}[dl_bw]
    bits = [(bw >> (2-i)) & 1 for i in range(3)]
    bits += [phich_dur & 1] + [(phich_res >> (1-i)) & 1 for i in range(2)]
    bits += [((sfn >> 2) >> (7-i)) & 1 for i in range(8)]
    bits += [(int(sib1_br) >> (4-i)) & 1 for i in range(5)]
    bits += [int(si_unchanged_br) & 1] + [0]*4
    return np.array(bits, np.int8)
def _pb_unpack(b, i0):
    bw = {0:6,1:15,2:25,3:50,4:75,5:100}
    code = int(b[0])*4 + int(b[1])*2 + int(b[2])
    # bits 14..19 carry the Rel-13/15 LTE-M fields; only 20..23 are spare
    if code > 5 or np.any(b[20:24] != 0):
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


def _pb_chest_denoise(hk, ntap=_PB_CHEST_NTAP):
    """12-point transform-domain denoise of the CRS knots (see src/lte_pbch.c
    chest_denoise12). g = ifft(hk); keep first ntap taps; hk = fft(g)."""
    if ntap <= 0 or ntap >= len(hk):
        return hk
    g = np.fft.ifft(hk)
    g[ntap:] = 0.0
    return np.fft.fft(g)


def _pb_channels(Y, n_id):
    H = {}
    for p in range(4):
        l = _pb_crs_sym(p)
        r = lte_crs_central(n_id, l); pos = lte_crs_pos(n_id, p, l)
        hk = np.array([Y[l][k] / r[m] for m, k in enumerate(pos)])
        hk = _pb_chest_denoise(hk)
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
PRACH_MIN_METRIC = 40.0          # legacy strong-only gate (explicit callers)
PRACH_CAND_GATE = 8.0            # loose D-tone gate to surface candidates
PRACH_COH_THR = 30.0             # coherent PDP peak/median confirm threshold
PRACH_MAX_CAND = 8               # candidates examined per window (cost bound)
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


def prach_detect(seq, min_metric=PRACH_CAND_GATE, N=PRACH_NZC,
                 count_thresh=_PRACH_COUNT_FRAC):
    """Detect PRACH preambles. Returns [{root, count, metric, delay}], the
    strongest tone first. Mirrors src/lte_prach.c: the D-tone FFT surfaces
    candidate roots at a loose gate, each confirmed on its coherent power-delay
    profile (peak/median >= PRACH_COH_THR); at most PRACH_MAX_CAND examined."""
    seq = np.asarray(seq, complex)
    Y = np.fft.fft(seq)
    D = Y[1:] * np.conj(Y[:-1])
    S = np.abs(np.fft.fft(D, N)) ** 2
    med = float(np.median(S)) + 1e-30
    out = []
    claimed = np.zeros(N, dtype=bool)
    ncand = 0
    while ncand < PRACH_MAX_CAND:
        b = int(np.argmax(np.where(claimed, -1.0, S)))
        if claimed[b] or S[b] / med < min_metric:
            break
        for d in range(-2, 3):
            claimed[(b + d) % N] = True
        ncand += 1
        u = (N - b) % N
        if u == 0:
            continue
        pdp = np.abs(np.fft.ifft(Y * np.conj(prach_zc(u, N)))) ** 2
        top = float(pdp.max())
        # coherent confirm: matched-filter peak over the PDP's own median
        if top / (float(np.median(pdp)) + 1e-30) < PRACH_COH_THR:
            continue
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


def prach_scan(stream, win_step=64, min_metric=PRACH_CAND_GATE, N=PRACH_NZC):
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

# ============================================================================
# 11d. LTE turbo decoder + CRC-24A (twin of src/lte_turbo.c)
# ----------------------------------------------------------------------------
# The rate-1/3 turbo code (36.212 5.1.3.2) protecting the PDSCH transport block
# a SIB rides on: two 8-state RSC constituents joined by a QPP interleaver,
# rate-matched (5.1.4.1), decoded by iterative max-log-MAP. Downlink broadcast
# data only. The C reads doubles; this reads float64 — the same arithmetic.
# ============================================================================

# QPP interleaver f1/f2 (36.212 Table 5.1.3-3), the full table, K -> (f1, f2).
_QPP_TABLE = {
    40:(3,10), 48:(7,12), 56:(19,42), 64:(7,16), 72:(7,18), 80:(11,20),
    88:(5,22), 96:(11,24), 104:(7,26), 112:(41,84), 120:(103,90),
    128:(15,32), 136:(9,34), 144:(17,108), 152:(9,38), 160:(21,120),
    168:(101,84), 176:(21,44), 184:(57,46), 192:(23,48), 200:(13,50),
    208:(27,52), 216:(11,36), 224:(27,56), 232:(85,58), 240:(29,60),
    248:(33,62), 256:(15,32), 264:(17,198), 272:(33,68), 280:(103,210),
    288:(19,36), 296:(19,74), 304:(37,76), 312:(19,78), 320:(21,120),
    328:(21,82), 336:(115,84), 344:(193,86), 352:(21,44), 360:(133,90),
    368:(81,46), 376:(45,94), 384:(23,48), 392:(243,98), 400:(151,40),
    408:(155,102), 416:(25,52), 424:(51,106), 432:(47,72), 440:(91,110),
    448:(29,168), 456:(29,114), 464:(247,58), 472:(29,118), 480:(89,180),
    488:(91,122), 496:(157,62), 504:(55,84), 512:(31,64), 528:(17,66),
    544:(35,68), 560:(227,420), 576:(65,96), 592:(19,74), 608:(37,76),
    624:(41,234), 640:(39,80), 656:(185,82), 672:(43,252), 688:(21,86),
    704:(155,44), 720:(79,120), 736:(139,92), 752:(23,94), 768:(217,48),
    784:(25,98), 800:(17,80), 816:(127,102), 832:(25,52), 848:(239,106),
    864:(17,48), 880:(137,110), 896:(215,112), 912:(29,114), 928:(15,58),
    944:(147,118), 960:(29,60), 976:(59,122), 992:(65,124), 1008:(55,84),
    1024:(31,64), 1056:(17,66), 1088:(171,204), 1120:(67,140), 1152:(35,72),
    1184:(19,74), 1216:(39,76), 1248:(19,78), 1280:(199,240), 1312:(21,82),
    1344:(211,252), 1376:(21,86), 1408:(43,88), 1440:(149,60), 1472:(45,92),
    1504:(49,846), 1536:(71,48), 1568:(13,28), 1600:(17,80), 1632:(25,102),
    1664:(183,104), 1696:(55,954), 1728:(127,96), 1760:(27,110),
    1792:(29,112), 1824:(29,114), 1856:(57,116), 1888:(45,354),
    1920:(31,120), 1952:(59,610), 1984:(185,124), 2016:(113,420),
    2048:(31,64), 2112:(17,66), 2176:(171,136), 2240:(209,420),
    2304:(253,216), 2368:(367,444), 2432:(265,456), 2496:(181,468),
    2560:(39,80), 2624:(27,164), 2688:(127,504), 2752:(143,172),
    2816:(43,88), 2880:(29,300), 2944:(45,92), 3008:(157,188), 3072:(47,96),
    3136:(13,28), 3200:(111,240), 3264:(443,204), 3328:(51,104),
    3392:(51,212), 3456:(451,192), 3520:(257,220), 3584:(57,336),
    3648:(313,228), 3712:(271,232), 3776:(179,236), 3840:(331,120),
    3904:(363,244), 3968:(375,248), 4032:(127,168), 4096:(31,64),
    4160:(33,130), 4224:(43,264), 4288:(33,134), 4352:(477,408),
    4416:(35,138), 4480:(233,280), 4544:(357,142), 4608:(337,480),
    4672:(37,146), 4736:(71,444), 4800:(71,120), 4864:(37,152),
    4928:(39,462), 4992:(127,234), 5056:(39,158), 5120:(39,80),
    5184:(31,96), 5248:(113,902), 5312:(41,166), 5376:(251,336),
    5440:(43,170), 5504:(21,86), 5568:(43,174), 5632:(45,176),
    5696:(45,178), 5760:(161,120), 5824:(89,182), 5888:(323,184),
    5952:(47,186), 6016:(23,94), 6080:(47,190), 6144:(263,480)
}


def lte_qpp(K):
    f1, f2 = _QPP_TABLE[K]
    i = np.arange(K, dtype=np.int64)
    return ((f1 * i + f2 * i * i) % K).astype(np.int64)


# constituent RSC trellis: state=(r1<<2)|(r2<<1)|r3, r1 newest.
def _t_rsc_step(state, u):
    r1 = (state >> 2) & 1; r2 = (state >> 1) & 1; r3 = state & 1
    a = u ^ r2 ^ r3                 # feedback g0 = 1 + D^2 + D^3
    z = a ^ r1 ^ r3                 # parity   g1 = 1 + D + D^3
    return (a << 2) | (r1 << 1) | r2, z


_T_NS = np.zeros((8, 2), np.int32)
_T_PAR = np.zeros((8, 2), np.int32)
_T_TAIL_U = np.zeros(8, np.int32)
for _s in range(8):
    for _u in (0, 1):
        _T_NS[_s, _u], _T_PAR[_s, _u] = _t_rsc_step(_s, _u)
    _T_TAIL_U[_s] = ((_s >> 1) & 1) ^ (_s & 1)


def _t_rsc_encode(u):
    K = len(u); s = 0
    sysb = np.empty(K + 3, np.int8); par = np.empty(K + 3, np.int8)
    for k in range(K):
        sysb[k] = u[k]; par[k] = _T_PAR[s, u[k]]; s = _T_NS[s, u[k]]
    for j in range(3):
        ut = _T_TAIL_U[s]; sysb[K + j] = ut; par[K + j] = _T_PAR[s, ut]; s = _T_NS[s, ut]
    return sysb, par


def lte_turbo_encode_d(u):
    """u (K bits) -> the three rate-matcher input streams d0,d1,d2 (K+4 each)."""
    K = len(u)
    sys1, par1 = _t_rsc_encode(u)
    pi = lte_qpp(K)
    sys2, par2 = _t_rsc_encode(np.asarray(u)[pi])
    xt = sys1[K:K + 3]; zt = par1[K:K + 3]
    xpt = sys2[K:K + 3]; zpt = par2[K:K + 3]
    d0 = np.empty(K + 4, np.int8); d1 = np.empty(K + 4, np.int8); d2 = np.empty(K + 4, np.int8)
    d0[:K] = u; d1[:K] = par1[:K]; d2[:K] = par2[:K]
    d0[K], d1[K], d2[K] = xt[0], zt[0], xt[1]
    d0[K + 1], d1[K + 1], d2[K + 1] = zt[1], xt[2], zt[2]
    d0[K + 2], d1[K + 2], d2[K + 2] = xpt[0], zpt[0], xpt[1]
    d0[K + 3], d1[K + 3], d2[K + 3] = zpt[1], xpt[2], zpt[2]
    return d0, d1, d2


_TPERM = np.array([0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30,
                   1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31])


def _lte_turbo_subblock_maps(D):
    C = 32
    R = -(-D // C)
    KPi = R * C
    ND = KPi - D
    padded = np.array([-1] * ND + list(range(D)))
    M = padded.reshape(R, C)
    idx01 = M[:, _TPERM].reshape(-1, order='F')
    k = np.arange(KPi)
    pi2 = (_TPERM[k // R] + C * (k % R) + 1) % KPi
    idx2 = padded[pi2]
    return idx01, idx2, R, KPi


def _lte_k0(rv, R, Ncb):
    return R * (2 * (-(-Ncb // (8 * R))) * rv + 2)


def lte_rate_match_turbo(d0, d1, d2, E, rv=0, Ncb=None):
    D = len(d0)
    idx01, idx2, R, KPi = _lte_turbo_subblock_maps(D)
    v0 = np.array([-1 if i < 0 else int(d0[i]) for i in idx01], np.int8)
    v1 = np.array([-1 if i < 0 else int(d1[i]) for i in idx01], np.int8)
    v2 = np.array([-1 if i < 0 else int(d2[i]) for i in idx2], np.int8)
    w = np.empty(3 * KPi, np.int8)
    w[:KPi] = v0; w[KPi::2] = v1; w[KPi + 1::2] = v2
    Kw = 3 * KPi
    if Ncb is None:
        Ncb = Kw
    j = _lte_k0(rv, R, Ncb)
    out = np.empty(E, np.int8); k = 0
    while k < E:
        val = w[j % Ncb]; j += 1
        if val >= 0:
            out[k] = val; k += 1
    return out


def lte_rate_dematch_turbo(e_llr, D, E, rv=0, Ncb=None):
    idx01, idx2, R, KPi = _lte_turbo_subblock_maps(D)
    Kw = 3 * KPi
    if Ncb is None:
        Ncb = Kw
    isnull = np.zeros(Ncb, bool)
    isnull[:KPi] = idx01 < 0
    isnull[KPi::2] = idx01 < 0
    isnull[KPi + 1::2] = idx2 < 0
    w = np.zeros(Ncb)
    j = _lte_k0(rv, R, Ncb); k = 0
    while k < E:
        pos = j % Ncb
        if not isnull[pos]:
            w[pos] += e_llr[k]; k += 1
        j += 1
    v0 = w[:KPi]; v1 = w[KPi::2][:KPi]; v2 = w[KPi + 1::2][:KPi]
    d0 = np.zeros(D); d1 = np.zeros(D); d2 = np.zeros(D)
    for kk in range(KPi):
        if idx01[kk] >= 0:
            d0[idx01[kk]] += v0[kk]; d1[idx01[kk]] += v1[kk]
        if idx2[kk] >= 0:
            d2[idx2[kk]] += v2[kk]
    return d0, d1, d2


_T_NEG = -1e18
#: Extrinsic scaling for max-log-MAP (see TURBO_EXT_SCALE in src/lte_turbo.c).
_TURBO_EXT_SCALE = 0.75


def _t_bcjr(sys_llr, par_llr, apri):
    n = len(sys_llr); K = n - 3
    alpha = np.full((n + 1, 8), _T_NEG); alpha[0, 0] = 0.0
    for k in range(n):
        ap = apri[k] if k < K else 0.0
        for s in range(8):
            if alpha[k, s] <= _T_NEG:
                continue
            for u in (0, 1):
                ns = _T_NS[s, u]; z = _T_PAR[s, u]
                g = 0.5 * (sys_llr[k] * (1 - 2 * u) + par_llr[k] * (1 - 2 * z)
                           + (ap * (1 - 2 * u) if k < K else 0.0))
                v = alpha[k, s] + g
                if v > alpha[k + 1, ns]:
                    alpha[k + 1, ns] = v
    beta = np.full((n + 1, 8), _T_NEG); beta[n, 0] = 0.0
    for k in range(n - 1, -1, -1):
        ap = apri[k] if k < K else 0.0
        for s in range(8):
            best = _T_NEG
            for u in (0, 1):
                ns = _T_NS[s, u]; z = _T_PAR[s, u]
                g = 0.5 * (sys_llr[k] * (1 - 2 * u) + par_llr[k] * (1 - 2 * z)
                           + (ap * (1 - 2 * u) if k < K else 0.0))
                v = beta[k + 1, ns] + g
                if v > best:
                    best = v
            beta[k, s] = best
    ext = np.zeros(K)
    for k in range(K):
        ap = apri[k]; m1 = _T_NEG; m0 = _T_NEG
        for s in range(8):
            if alpha[k, s] <= _T_NEG:
                continue
            for u in (0, 1):
                ns = _T_NS[s, u]; z = _T_PAR[s, u]
                g = 0.5 * (sys_llr[k] * (1 - 2 * u) + par_llr[k] * (1 - 2 * z)
                           + ap * (1 - 2 * u))
                v = alpha[k, s] + g + beta[k + 1, ns]
                if u == 1:
                    m1 = max(m1, v)
                else:
                    m0 = max(m0, v)
        ext[k] = (m0 - m1) - sys_llr[k] - ap
    return ext


def lte_turbo_decode_from_d(d0, d1, d2, K, iters=8):
    sys = d0[:K]
    par1 = np.concatenate([d1[:K], [d1[K], d0[K + 1], d2[K + 1]]])
    par2 = np.concatenate([d2[:K], [d1[K + 2], d0[K + 3], d2[K + 3]]])
    tail1_sys = np.array([d0[K], d2[K], d1[K + 1]])
    tail2_sys = np.array([d0[K + 2], d2[K + 2], d1[K + 3]])
    pi = lte_qpp(K); inv = np.argsort(pi)
    sys1 = np.concatenate([sys, tail1_sys])
    sys2 = np.concatenate([sys[pi], tail2_sys])
    apri = np.zeros(K)
    for _ in range(iters):
        e1 = _t_bcjr(sys1, par1, apri)
        e2 = _t_bcjr(sys2, par2, _TURBO_EXT_SCALE * e1[pi])
        apri = _TURBO_EXT_SCALE * e2[inv]
    e1 = _t_bcjr(sys1, par1, apri)
    llr = sys + apri + e1
    return (llr < 0).astype(np.int8)


# CRC-24A (36.212 5.1.1)
_G24A = 0x1864CFB


def lte_crc24a(bits):
    reg = 0
    for b in bits:
        reg = (reg << 1) | (int(b) & 1)
        if reg & (1 << 24):
            reg ^= _G24A
    for _ in range(24):
        reg <<= 1
        if reg & (1 << 24):
            reg ^= _G24A
    return np.array([(reg >> (23 - i)) & 1 for i in range(24)], np.int8)


def lte_crc24a_check(bits):
    return np.array_equal(lte_crc24a(bits[:-24]), np.asarray(bits[-24:]) & 1)


def lte_turbo_decode(llr, K, rv=0, iters=8):
    """E descrambled LLRs (>0 favours bit 0), code-block size K -> the K-24
    payload bits on a CRC-24A pass, else None. Twin of atkdsp_lte_turbo_decode."""
    E = len(llr)
    d0, d1, d2 = lte_rate_dematch_turbo(np.asarray(llr, float), K + 4, E, rv)
    bits = lte_turbo_decode_from_d(d0, d1, d2, K, iters)
    if lte_crc24a_check(bits):
        return bits[:-24]
    return None


# ============================================================================
# 11e. SIB1 -> tower identity (ASN.1 UPER; twin of src/lte_sib1.c)
# ----------------------------------------------------------------------------
# The identity-bearing part of SystemInformationBlockType1: PLMN list (operator
# MCC/MNC), trackingAreaCode (16-bit) and the 28-bit E-UTRAN cell identity
# (ECI). Works on the decoded transport-block bit array (one bit per element,
# MSB first). A matching encoder validates the decoder by a round trip.
# 36.331 ASN.1 (R8 baseline structure). Downlink broadcast only.
# ============================================================================

def _sib1_nbits(rng):
    if rng <= 1:
        return 0
    return int(rng - 1).bit_length()


def _sib1_write(bits, value, n):
    for i in range(n - 1, -1, -1):
        bits.append((value >> i) & 1)


class _Sib1Reader:
    def __init__(self, bits):
        self.b = list(np.asarray(bits, np.int8) & 1)
        self.pos = 0
        self.bad = False

    def u(self, n):
        v = 0
        for _ in range(n):
            if self.pos >= len(self.b):
                self.bad = True
                return 0
            v = (v << 1) | int(self.b[self.pos]); self.pos += 1
        return v

    def bit(self):
        return self.u(1)


# si-Periodicity index -> radio frames; si-WindowLength index -> ms.
_SI_PERIODICITY_RF = [8, 16, 32, 64, 128, 256, 512]
_SI_WINDOW_MS = [1, 2, 5, 10, 15, 20, 40]


def lte_sib1_encode(plmns, tac, cellid, csg_identity=None, sched=None,
                    si_window_idx=5, freq_band=1):
    """Build BCCH-DL-SCH-Message bits carrying a (full) SIB1. plmns = list of
    {mcc:[3] or None, mnc:[2 or 3], reserved?}. `sched` is the schedulingInfoList
    as [{periodicity_idx, sibs:[SIB-Type ints]}]; a default (one SI message
    carrying SIB3) is used when omitted. Returns an int8 bit array."""
    if sched is None:
        sched = [{"periodicity_idx": 1, "sibs": [3]}]
    b = []
    _sib1_write(b, 0, 1)                 # BCCH-DL-SCH type CHOICE -> c1
    _sib1_write(b, 1, 1)                 # c1 CHOICE -> systemInformationBlockType1
    # (SIB1 has no extension marker — see lte_sib1_decode)
    _sib1_write(b, 0, 1)                 # p-Max absent
    _sib1_write(b, 0, 1)                 # tdd-Config absent
    _sib1_write(b, 0, 1)                 # nonCriticalExtension absent
    _sib1_write(b, 1 if csg_identity is not None else 0, 1)   # csg-Identity opt
    _sib1_write(b, len(plmns) - 1, _sib1_nbits(6))            # SIZE(1..6)
    for p in plmns:
        mcc = p.get("mcc")
        _sib1_write(b, 1 if mcc is not None else 0, 1)
        if mcc is not None:
            for d in mcc:
                _sib1_write(b, int(d), 4)
        mnc = p["mnc"]
        _sib1_write(b, len(mnc) - 2, 1)
        for d in mnc:
            _sib1_write(b, int(d), 4)
        _sib1_write(b, 0 if p.get("reserved", False) else 1, 1)
    _sib1_write(b, tac, 16)
    _sib1_write(b, cellid, 28)
    _sib1_write(b, 0, 1)                 # cellBarred
    _sib1_write(b, 0, 1)                 # intraFreqReselection
    _sib1_write(b, 0, 1)                 # csg-Indication
    if csg_identity is not None:
        _sib1_write(b, csg_identity, 27)
    # cellSelectionInfo
    _sib1_write(b, 0, 1)                 # q-RxLevMinOffset absent
    _sib1_write(b, -60 - (-70), _sib1_nbits(49))     # q-RxLevMin (-70..-22)
    # (p-Max absent) ; freqBandIndicator (1..64)
    _sib1_write(b, freq_band - 1, _sib1_nbits(64))
    # schedulingInfoList SIZE(1..32)
    _sib1_write(b, len(sched) - 1, _sib1_nbits(32))
    for si in sched:
        _sib1_write(b, si.get("periodicity_idx", 1), _sib1_nbits(7))
        mapping = si.get("sibs", [])
        _sib1_write(b, len(mapping), _sib1_nbits(32))    # SIB-MappingInfo SIZE(0..31)
        for sib_type in mapping:
            _sib1_write(b, 0, 1)                          # SIB-Type ext bit
            _sib1_write(b, sib_type - 3, 4)               # sibType3->0, root 16
    # (tdd-Config absent) ; si-WindowLength (7) ; systemInfoValueTag (0..31)
    _sib1_write(b, si_window_idx, _sib1_nbits(7))
    _sib1_write(b, 0, 5)
    return np.array(b, np.int8)


def lte_sib1_decode(bits):
    """Decode SIB1 transport-block bits: the tower identity, and (best-effort)
    the scheduling that says where the SIB2+ messages sit. Returns a dict or
    None on a structural mismatch. `sched`/`si_window_ms` are empty/0 if the
    bits stop after the identity (an identity-only stream)."""
    r = _Sib1Reader(bits)
    if r.u(1) != 0:
        return None                     # not c1
    if r.u(1) != 1:
        return None                     # not SIB1
    # SystemInformationBlockType1 has NO extension marker; reading one put
    # every later field one bit late and no real SIB1 ever parsed (Bill's B13
    # capture, 2026-09-26 — this twin and the C agreed with each other).
    has_pmax = r.u(1); has_tdd = r.u(1); r.u(1)   # p-Max / tdd / nonCrit present
    csg_present = r.u(1)
    n_plmn = r.u(_sib1_nbits(6)) + 1
    if n_plmn < 1 or n_plmn > 6:
        return None
    plmns = []
    last_mcc = None
    for _ in range(n_plmn):
        mcc_present = r.u(1)
        mcc = [r.u(4) for _ in range(3)] if mcc_present else None
        if mcc is not None:
            last_mcc = mcc
        else:
            mcc = last_mcc
        mnc_len = r.u(1) + 2
        mnc = [r.u(4) for _ in range(mnc_len)]
        reserved = (r.u(1) == 0)
        plmns.append({"mcc": mcc, "mnc": mnc, "reserved": reserved})
    tac = r.u(16)
    cellid = r.u(28)
    cell_barred = r.u(1)
    r.u(1)                              # intraFreqReselection
    r.u(1)                              # csg-Indication
    csg_id = r.u(27) if csg_present else None
    if r.bad:
        return None
    out = {"plmns": plmns, "tac": tac, "cellid": cellid,
           "cell_barred": cell_barred, "csg_id": csg_id,
           "freq_band": 0, "sched": [], "si_window_ms": 0, "si_window_idx": -1}
    # -- scheduling (best-effort; identity above is what gates the return) --
    save = r.pos
    has_qoff = r.u(1)                   # q-RxLevMinOffset present
    r.u(_sib1_nbits(49))               # q-RxLevMin
    if has_qoff:
        r.u(3)                         # q-RxLevMinOffset (1..8)
    if has_pmax:
        r.u(_sib1_nbits(64))           # p-Max
    freq_band = r.u(_sib1_nbits(64)) + 1
    nsi = r.u(_sib1_nbits(32)) + 1
    sched = []
    for _ in range(nsi):
        per_idx = r.u(_sib1_nbits(7))
        nmap = r.u(_sib1_nbits(32))
        sibs = []
        for _ in range(nmap):
            if r.u(1):                 # SIB-Type past its extension marker
                big, e = r.u(1), r.u(6)
                sibs.append((19, 20, 21, 24, 25, 26)[e]
                            if not big and e < 6 else 0)
            else:                      # 16 root values: sibType3..18
                sibs.append(r.u(4) + 3)
        sched.append({"periodicity_idx": per_idx,
                      "periodicity_rf": (_SI_PERIODICITY_RF[per_idx]
                                         if per_idx < 7 else 0),
                      "sibs": sibs})
    if has_tdd:
        r.u(_sib1_nbits(7)); r.u(_sib1_nbits(9))     # tdd-Config
    si_window_idx = r.u(_sib1_nbits(7))
    if not r.bad:
        out["freq_band"] = freq_band
        out["sched"] = sched
        out["si_window_idx"] = si_window_idx
        out["si_window_ms"] = (_SI_WINDOW_MS[si_window_idx]
                               if si_window_idx < 7 else 0)
    else:
        r.pos = save                   # identity-only stream; leave sched empty
    return out


def lte_plmn_str(p):
    """'MCC-MNC', e.g. '310-410'. MNC keeps its 2- or 3-digit count."""
    mcc = "".join(str(d) for d in p["mcc"]) if p["mcc"] else "???"
    mnc = "".join(str(d) for d in p["mnc"])
    return f"{mcc}-{mnc}"


# ============================================================================
# 11f. SIB1 physical layer (twin of src/lte_sib1_phy.c)
# ----------------------------------------------------------------------------
# Full-bandwidth downlink subframe -> tower identity. OFDM (de)modulation at the
# cell's numerology, a port-0 CRS channel estimate, PCFICH (CFI), a blind
# SI-RNTI PDCCH search (DCI 1A -> RB allocation), then PDSCH descramble + turbo
# + ASN.1. Downlink broadcast only.
#
# NOTE ON GEOMETRY: the REG/CCE numbering and the PCFICH/PDCCH RE choice below
# are a self-consistent statement of 36.211 6.2.4/6.7/6.8 — a transmit/receive
# round trip proves the coding, scrambling, interleaving and RNTI-masked CRC,
# NOT the spec's exact RE positions on a real air capture. Confirm against a
# live signal before trusting a live decode. 36.211 / 36.212 / 36.331.
# ============================================================================

_SIB1_NFFT = {6: 128, 15: 256, 25: 512, 50: 1024, 75: 1536, 100: 2048}
_SC_PER_RB = 12
_SYMS_SF = 14
_SYMS_SLOT = 7
LTE_SI_RNTI = 0xFFFF


def lte_grid_params(n_rb):
    nfft = _SIB1_NFFT[n_rb]
    return {"n_rb": n_rb, "nfft": nfft, "n_sc": n_rb * _SC_PER_RB,
            "cp_long": nfft * 160 // 2048, "cp_short": nfft * 144 // 2048}


def _sib1_sc_to_bin(k, n_sc, nfft):
    off = k - n_sc // 2
    if off >= 0:
        off += 1                       # skip DC
    return off % nfft


def _sib1_cp(sym_in_slot, gp):
    return gp["cp_long"] if sym_in_slot == 0 else gp["cp_short"]


def lte_ofdm_modulate(grid, gp):
    n_sc, nfft = gp["n_sc"], gp["nfft"]
    out = []
    for l in range(_SYMS_SF):
        X = np.zeros(nfft, complex)
        for k in range(n_sc):
            X[_sib1_sc_to_bin(k, n_sc, nfft)] = grid[k, l]
        xt = np.fft.ifft(X) * nfft / np.sqrt(n_sc)
        cp = _sib1_cp(l % _SYMS_SLOT, gp)
        out.append(np.concatenate([xt[-cp:], xt]))
    return np.concatenate(out)


def lte_ofdm_demodulate(samples, gp):
    n_sc, nfft = gp["n_sc"], gp["nfft"]
    grid = np.zeros((n_sc, _SYMS_SF), complex)
    pos = 0
    for l in range(_SYMS_SF):
        pos += _sib1_cp(l % _SYMS_SLOT, gp)
        xt = samples[pos:pos + nfft]; pos += nfft
        X = np.fft.fft(xt) * np.sqrt(n_sc) / nfft
        for k in range(n_sc):
            grid[k, l] = X[_sib1_sc_to_bin(k, n_sc, nfft)]
    return grid


# ---- CRS (port 0), whole band -------------------------------------------
def lte_crs_seq(n_id, n_s, l):
    ci = (1 << 10) * (7 * (n_s + 1) + l + 1) * (2 * n_id + 1) + 2 * n_id + 1
    c = lte_gold(ci, 4 * _PB_NRB_MAX)
    return _PB_SQ * (1 - 2 * c[0::2]) + 1j * _PB_SQ * (1 - 2 * c[1::2])


def lte_crs_re(n_id, n_rb, n_s, l, v):
    vshift = n_id % 6
    r = lte_crs_seq(n_id, n_s, l)
    m0 = _PB_NRB_MAX - n_rb
    return [(6 * m + (v + vshift) % 6, r[m0 + m]) for m in range(2 * n_rb)]


def _crs_syms_in_sf():
    out = []
    for n_s in range(2):
        for l in (0, 4):
            out.append((n_s * _SYMS_SLOT + l, n_s, l, 0 if l == 0 else 3))
    return out


def lte_crs_positions(n_id, n_rb):
    occ = {}
    for sym, n_s, l, v in _crs_syms_in_sf():
        for k, val in lte_crs_re(n_id, n_rb, n_s, l, v):
            occ[(k, sym)] = val
    return occ


def lte_crs_re_set(n_id, n_rb, n_ports):
    vshift = n_id % 6
    occ = set()
    for n_s in range(2):
        base = n_s * _SYMS_SLOT
        for port in range(n_ports):
            if port <= 1:
                syms = ((0, 0 if port == 0 else 3), (4, 3 if port == 0 else 0))
            else:
                syms = ((1, 0 if port == 2 else 3),)
            for l, v in syms:
                for m in range(2 * n_rb):
                    occ.add((6 * m + (v + vshift) % 6, base + l))
    return occ


#: Keep cnt/_SIB_CHEST_NTAP_DEN transform-domain taps (match SIB_CHEST_NTAP_DEN
#: in src/lte_sib1_phy.c).
_SIB_CHEST_NTAP_DEN = 3


def _sib_chest_denoise(hval, den=_SIB_CHEST_NTAP_DEN):
    """Transform-domain denoise of the uniformly spaced CRS channel knots (see
    chest_denoise in src/lte_sib1_phy.c). g = ifft(hk); keep cnt/den taps."""
    cnt = len(hval)
    ntap = cnt // den
    if ntap <= 0 or ntap >= cnt:
        return hval
    g = np.fft.ifft(hval)
    g[ntap:] = 0.0
    return np.fft.fft(g)


def _sib1_equalize(grid, n_id, n_rb, gp):
    """Port-0 single-tap zero-forcing: estimate H at CRS REs per CRS symbol,
    linear-interpolate across the band, apply the nearest CRS symbol's H."""
    n_sc = gp["n_sc"]
    crs_syms = _crs_syms_in_sf()
    Hsym = {}
    for sym, n_s, l, v in crs_syms:
        pos = []; hval = []
        for k, ref_val in lte_crs_re(n_id, n_rb, n_s, l, v):
            y = grid[k, sym]
            pos.append(k); hval.append(y * np.conj(ref_val))    # |crs|=1
        pos = np.array(pos); hval = np.array(hval)
        hval = _sib_chest_denoise(hval)                         # transform-domain denoise
        H = (np.interp(np.arange(n_sc), pos, hval.real)
             + 1j * np.interp(np.arange(n_sc), pos, hval.imag))
        Hsym[sym] = H
    crs_sym_list = sorted(Hsym)
    eq = np.zeros_like(grid)
    for l in range(_SYMS_SF):
        nearest = min(crs_sym_list, key=lambda cs: (abs(cs - l), cs))
        H = Hsym[nearest]
        denom = np.abs(H) ** 2 + 1e-9
        eq[:, l] = grid[:, l] * np.conj(H) / denom
    return eq


# ---- PDSCH ---------------------------------------------------------------
def lte_pdsch_re_list(n_id, n_rb, alloc_rbs, cfi, n_ports):
    crs = lte_crs_re_set(n_id, n_rb, n_ports)
    res = []
    for sym in range(cfi, _SYMS_SF):
        for rb in alloc_rbs:
            for sub in range(_SC_PER_RB):
                k = rb * _SC_PER_RB + sub
                if (k, sym) not in crs:
                    res.append((k, sym))
    return res


def lte_pdsch_scramble(n_id, n_rnti, subframe, length):
    n_s = subframe * 2
    c_init = (n_rnti << 14) | (0 << 13) | ((n_s // 2) << 9) | n_id
    return lte_gold(c_init, length)


def lte_pdsch_encode(tb_bits, n_id, n_rnti, subframe, E, K):
    frame = np.concatenate([tb_bits, lte_crc24a(tb_bits)]).astype(np.int8)
    assert len(frame) == K
    d0, d1, d2 = lte_turbo_encode_d(frame)
    e = lte_rate_match_turbo(d0, d1, d2, E)
    c = lte_pdsch_scramble(n_id, n_rnti, subframe, E)
    return (e ^ c).astype(np.int8)


def lte_pdsch_decode(llr_scr, n_id, n_rnti, subframe, K):
    E = len(llr_scr)
    c = lte_pdsch_scramble(n_id, n_rnti, subframe, E)
    llr = (1 - 2 * c) * llr_scr
    return lte_turbo_decode(llr, K)              # payload (K-24) or None


# ---- REGs, PCFICH, PDCCH -------------------------------------------------
def _sib1_sym_regs(n_id, n_rb, l, n_ports):
    vshift = n_id % 6
    has_crs = (l == 0) or (l == 1 and n_ports == 4)
    regs = []
    if has_crs:
        crs_res = {(6 * m + (vshift + o) % 6) for m in range(2 * n_rb) for o in (0, 3)}
        for g in range(2 * n_rb):
            group = [g * 6 + i for i in range(6)]
            data = [k for k in group if k not in crs_res]
            regs.append([(k, l) for k in data[:4]])
    else:
        for g in range(3 * n_rb):
            regs.append([(g * 4 + i, l) for i in range(4)])
    return regs


def lte_pcfich_reg_idx(n_id, n_rb):
    n_regs0 = 2 * n_rb
    k_bar = n_id % (2 * n_rb)
    return [(k_bar + i * (n_regs0 // 4)) % n_regs0 for i in range(4)]


def _sib1_perm_regs(seq, n_id):
    perm = _PB_PERM
    D = len(seq); C = 32; R = -(-D // C); nd = R * C - D
    padded = [None] * nd + list(seq)
    M = [padded[r * C:(r + 1) * C] for r in range(R)]
    out = []
    for c in range(C):
        for r in range(R):
            out.append(M[r][perm[c]])
    out = [x for x in out if x is not None]
    sh = n_id % len(out)
    return out[sh:] + out[:sh]


def lte_control_regs(n_id, n_rb, cfi, n_ports):
    all_regs = [_sib1_sym_regs(n_id, n_rb, l, n_ports) for l in range(cfi)]
    pc = set(lte_pcfich_reg_idx(n_id, n_rb))
    seq = []
    maxregs = max(len(r) for r in all_regs)
    for i in range(maxregs):
        for l in range(cfi):
            if i < len(all_regs[l]):
                if l == 0 and i in pc:
                    continue
                seq.append(all_regs[l][i])
    return _sib1_perm_regs(seq, n_id)


def lte_pcfich_res(n_id, n_rb):
    regs = _sib1_sym_regs(n_id, n_rb, 0, 2)
    out = []
    for idx in lte_pcfich_reg_idx(n_id, n_rb):
        out.extend(regs[idx])
    return out


# PCFICH codewords (36.212 5.3.4)
_CFI_WORD = {1: np.array([0, 1, 1, 0] * 8, np.int8),
             2: np.array([1, 0, 1, 1] * 8, np.int8),
             3: np.array([1, 1, 0, 1] * 8, np.int8)}


def lte_pcfich_cinit(n_id, n_s):
    return ((n_s // 2 + 1) * (2 * n_id + 1) << 9) + n_id


def lte_pcfich_encode(cfi, n_id, n_s):
    b = _CFI_WORD[cfi]
    c = lte_gold(lte_pcfich_cinit(n_id, n_s), 32)
    return _pb_qpsk((b ^ c).astype(np.int8))


def lte_pcfich_decode(sym16, n_id, n_s):
    c = lte_gold(lte_pcfich_cinit(n_id, n_s), 32)
    llr = np.empty(32)
    a = np.asarray(sym16)
    llr[0::2] = a.real; llr[1::2] = a.imag
    llr = (1 - 2 * c) * llr
    best, best_cfi = -1e30, 1
    for cfi in (1, 2, 3):
        score = float(np.dot(1 - 2 * _CFI_WORD[cfi], llr))
        if score > best:
            best, best_cfi = score, cfi
    return best_cfi


def lte_pdcch_cinit(n_id, n_s):
    return (n_s // 2 << 9) + n_id


def lte_dci_crc_attach(dci_bits, rnti):
    p = lte_crc16(dci_bits)
    mask = np.array([(rnti >> (15 - i)) & 1 for i in range(16)], np.int8)
    return np.concatenate([dci_bits, p ^ mask]).astype(np.int8)


def lte_dci_crc_check(bits, rnti):
    payload, rx = bits[:-16], bits[-16:]
    calc = lte_crc16(payload)
    mask = np.array([(rnti >> (15 - i)) & 1 for i in range(16)], np.int8)
    return np.array_equal((calc ^ mask) & 1, rx & 1)


def lte_pdcch_encode(dci_bits, rnti, E, n_id, n_s, cce_offset=0):
    frame = lte_dci_crc_attach(dci_bits, rnti)
    coded = lte_conv_encode(frame)
    e = lte_rate_match([coded[0::3], coded[1::3], coded[2::3]], E)
    c = lte_gold(lte_pdcch_cinit(n_id, n_s), cce_offset * 72 + E)[cce_offset * 72:]
    return (e ^ c).astype(np.int8)


def lte_pdcch_decode_candidate(scr_bits, D, rnti, n_id, n_s, cce_offset=0):
    E = len(scr_bits)
    c = lte_gold(lte_pdcch_cinit(n_id, n_s), cce_offset * 72 + E)[cce_offset * 72:]
    llr = (1 - 2 * c) * scr_bits
    d = lte_rate_dematch(llr, D, seg=(0, E))
    bits = lte_viterbi_tb(d)
    if not lte_dci_crc_check(bits, rnti):
        return None
    return bits[:-16]


# ---- DCI 1A: RIV <-> (rb_start, L) --------------------------------------
def lte_riv(rb_start, L, n_rb):
    if (L - 1) <= n_rb // 2:
        return n_rb * (L - 1) + rb_start
    return n_rb * (n_rb - L + 1) + (n_rb - 1 - rb_start)


def lte_riv_inv(v, n_rb):
    for L in range(1, n_rb + 1):
        for st in range(0, n_rb - L + 1):
            if lte_riv(st, L, n_rb) == v:
                return st, L
    return None


def _riv_bits(n_rb):
    return int(n_rb * (n_rb + 1) // 2 - 1).bit_length()


def lte_dci_1a_encode(rb_start, L, n_rb, dci_len=None):
    """A minimal DCI format 1A for SI-RNTI: format flag + localized + RIV, the
    rest zero-padded to dci_len (default: the 25-RB layout's 24 bits)."""
    if dci_len is None:
        dci_len = 24
    rb = _riv_bits(n_rb)
    b = [1, 0]
    v = lte_riv(rb_start, L, n_rb)
    b += [(v >> (rb - 1 - i)) & 1 for i in range(rb)]
    while len(b) < dci_len:
        b.append(0)
    return np.array(b[:dci_len], np.int8), v


def lte_dci_1a_riv(bits, n_rb):
    rb = _riv_bits(n_rb)
    v = 0
    for i in range(rb):
        v = (v << 1) | int(bits[2 + i])
    return v


# candidate turbo K sizes (QPP table), ascending
_QPP_KS = sorted(_QPP_TABLE)


def _sib1_candidate_Ks(E):
    """Valid turbo K to try for E coded bits: K-24 = TBS >= 1, K <= 3E (rate
    >= 1/3-ish), capped. CRC-24 gates the accept, so trying a handful is safe."""
    hi = min(3 * E, 6144)
    return [K for K in _QPP_KS if 40 <= K <= hi]


# common search space: (aggregation level, #candidates)
_COMMON_SS = [(4, 4), (8, 2)]
_DCI_D = 40                                   # DCI 1A (24) + CRC-16 = 40 bits


def lte_sib1_decode_iq(samples, n_rb, n_id, n_ports, subframe,
                       cfo_hz=0.0, equalize=True):
    """Full receive chain: subframe I/Q at the cell numerology -> tower
    identity dict, or None. `samples` covers one subframe (14 OFDM symbols)."""
    gp = lte_grid_params(n_rb)
    s = np.asarray(samples, complex)
    if cfo_hz:
        fs = gp["nfft"] * 15000.0                 # full-BW sample rate
        n = np.arange(len(s))
        s = s * np.exp(-2j * np.pi * cfo_hz * n / fs)
    grid = lte_ofdm_demodulate(s, gp)
    if equalize:
        grid = _sib1_equalize(grid, n_id, n_rb, gp)
    n_s = subframe * 2

    # 1) PCFICH -> CFI
    pc = [grid[k, l] for (k, l) in lte_pcfich_res(n_id, n_rb)]
    cfi = lte_pcfich_decode(pc, n_id, n_s)

    # 2) PDCCH blind SI-RNTI search over the common search space
    regs = lte_control_regs(n_id, n_rb, cfi, n_ports)
    cce_res = [re for reg in regs for re in reg]      # 36 REs per CCE
    n_cce = len(cce_res) // 36
    dci = None
    for al, ncand in _COMMON_SS:
        for m in range(ncand):
            cce = m * al
            if cce + al > n_cce:
                continue
            seg = cce_res[cce * 36:(cce + al) * 36]
            a = np.array([grid[k, l] for (k, l) in seg])
            llr = np.empty(len(a) * 2)
            llr[0::2] = a.real; llr[1::2] = a.imag
            cand = lte_pdcch_decode_candidate(llr, _DCI_D, LTE_SI_RNTI,
                                              n_id, n_s, cce_offset=cce)
            if cand is not None:
                dci = cand
                break
        if dci is not None:
            break
    if dci is None:
        return None

    # 3) DCI 1A -> allocation -> PDSCH
    v = lte_dci_1a_riv(dci, n_rb)
    ri = lte_riv_inv(v, n_rb)
    if ri is None:
        return None
    st, L = ri
    alloc = list(range(st, st + L))
    res = lte_pdsch_re_list(n_id, n_rb, alloc, cfi, n_ports)
    a = np.array([grid[k, l] for (k, l) in res])
    llr = np.empty(len(a) * 2)
    llr[0::2] = a.real; llr[1::2] = a.imag
    E = len(llr)
    c = lte_pdsch_scramble(n_id, LTE_SI_RNTI, subframe, E)
    dllr = (1 - 2 * c) * llr

    # 4) turbo (CRC-24 gates the transport-block size)
    for K in _sib1_candidate_Ks(E):
        payload = lte_turbo_decode(dllr, K)
        if payload is not None:
            info = lte_sib1_decode(payload)
            if info is not None:
                info["cfi"] = cfi
                info["alloc"] = (st, L)
                info["K"] = K
                return info
    return None


# ============================================================================
# 11g. SystemInformation: SIB2-5 network fingerprint (twin of src/lte_si.c)
# ----------------------------------------------------------------------------
# BCCH-DL-SCH -> c1 -> systemInformation -> SystemInformation-r8-IEs ->
# sib-TypeAndInfo (a CHOICE per SIB). Extracts the fields that make a network
# fingerprint: SIB2 (access barring, PRACH config, UL carrier freq/bandwidth,
# reference-signal power), SIB3 (reselection), SIB4 (intra-freq neighbour PCIs),
# SIB5 (inter-freq carriers + neighbour PCIs). Encoder + decoder, so a round
# trip proves the codec self-consistent. 36.331 R8 ASN.1; the neighbour lists
# and barring are high-confidence, the deep radioResourceConfig fields (PRACH,
# UL freq) most need a live-capture check. Downlink broadcast only.
# ============================================================================

class _SiW:
    def __init__(self):
        self.bits = []

    def u(self, value, n):
        for i in range(n - 1, -1, -1):
            self.bits.append((int(value) >> i) & 1)

    def b(self, flag):
        self.bits.append(1 if flag else 0)

    def arr(self):
        return np.array(self.bits, np.int8)


def _si_ei(w, value, lb, ub):
    w.u(int(value) - lb, _sib1_nbits(ub - lb + 1))


def _si_di(r, lb, ub):
    return r.u(_sib1_nbits(ub - lb + 1)) + lb


def _si_ext_enc(w, idx, root):
    w.b(0); w.u(idx, _sib1_nbits(root))


def _si_ext_dec(r, root):
    if r.bit():
        return -1
    return r.u(_sib1_nbits(root))


# ---- SIB4 (intra-freq neighbours) ----------------------------------------
def _lte_enc_sib4(w, s):
    neigh = s.get("neighbors", []); black = s.get("blacklist", [])
    csg = s.get("csg_range")
    w.b(0); w.b(bool(neigh)); w.b(bool(black)); w.b(csg is not None)
    if neigh:
        _si_ei(w, len(neigh), 1, 16)
        for nc in neigh:
            w.b(0); _si_ei(w, nc["pci"], 0, 503); w.u(nc.get("q_off", 15), _sib1_nbits(31))
    if black:
        _si_ei(w, len(black), 1, 16)
        for bc in black:
            _lte_enc_pcid(w, bc)
    if csg is not None:
        _lte_enc_pcid(w, csg)


def _lte_dec_sib4(r):
    r.bit(); has_n = r.bit(); has_b = r.bit(); has_csg = r.bit()
    out = {"neighbors": [], "blacklist": [], "csg_range": None}
    if has_n:
        for _ in range(_si_di(r, 1, 16)):
            r.bit(); pci = _si_di(r, 0, 503); q = r.u(_sib1_nbits(31))
            out["neighbors"].append({"pci": pci, "q_off": q})
    if has_b:
        for _ in range(_si_di(r, 1, 16)):
            out["blacklist"].append(_lte_dec_pcid(r))
    if has_csg:
        out["csg_range"] = _lte_dec_pcid(r)
    return out


def _lte_enc_pcid(w, pr):
    rng = pr.get("range")
    w.b(rng is not None); _si_ei(w, pr["start"], 0, 503)
    if rng is not None:
        w.u(rng, _sib1_nbits(16))


def _lte_dec_pcid(r):
    has = r.bit(); start = _si_di(r, 0, 503)
    return {"start": start, "range": r.u(_sib1_nbits(16)) if has else None}


# ---- SIB5 (inter-freq carriers) ------------------------------------------
def _lte_enc_sib5(w, s):
    carriers = s["carriers"]
    w.b(0); _si_ei(w, len(carriers), 1, 8)
    for c in carriers:
        p_max = c.get("p_max"); crp = c.get("resel_priority")
        neigh = c.get("neighbors", []); black = c.get("blacklist", [])
        w.b(0); w.b(p_max is not None); w.b(False); w.b(crp is not None)
        w.b(False); w.b(bool(neigh)); w.b(bool(black))
        _si_ei(w, c["dl_earfcn"], 0, 65535); _si_ei(w, c.get("q_rxlevmin", -60), -70, -22)
        if p_max is not None:
            _si_ei(w, p_max, -30, 33)
        _si_ei(w, c.get("t_resel", 0), 0, 7)
        _si_ei(w, c.get("thresh_high", 0), 0, 31); _si_ei(w, c.get("thresh_low", 0), 0, 31)
        w.u(c.get("allowed_meas_bw", 5), _sib1_nbits(6)); w.b(c.get("presence_ant1", False))
        if crp is not None:
            _si_ei(w, crp, 0, 7)
        w.u(c.get("neigh_cell_cfg", 0), 2)
        if neigh:
            _si_ei(w, len(neigh), 1, 16)
            for nc in neigh:
                _si_ei(w, nc["pci"], 0, 503); w.u(nc.get("q_off", 15), _sib1_nbits(31))
        if black:
            _si_ei(w, len(black), 1, 16)
            for bc in black:
                _lte_enc_pcid(w, bc)


def _lte_dec_sib5(r):
    r.bit(); nfreq = _si_di(r, 1, 8); carriers = []
    for _ in range(nfreq):
        r.bit()
        has_pmax = r.bit(); has_sf = r.bit(); has_crp = r.bit()
        has_qoff = r.bit(); has_neigh = r.bit(); has_black = r.bit()
        dl = _si_di(r, 0, 65535); qmin = _si_di(r, -70, -22)
        pmax = _si_di(r, -30, 33) if has_pmax else None
        _si_di(r, 0, 7)                          # t-ReselectionEUTRA
        if has_sf:
            r.u(2); r.u(2)
        _si_di(r, 0, 31); _si_di(r, 0, 31)       # threshX-High/Low
        r.u(_sib1_nbits(6)); r.bit()             # allowedMeasBW, presenceAntennaPort1
        crp = _si_di(r, 0, 7) if has_crp else None
        r.u(2)                                   # neighCellConfig
        if has_qoff:
            r.u(_sib1_nbits(31))
        neigh = []
        if has_neigh:
            for _ in range(_si_di(r, 1, 16)):
                pci = _si_di(r, 0, 503); q = r.u(_sib1_nbits(31))
                neigh.append({"pci": pci, "q_off": q})
        black = []
        if has_black:
            for _ in range(_si_di(r, 1, 16)):
                black.append(_lte_dec_pcid(r))
        carriers.append({"dl_earfcn": dl, "q_rxlevmin": qmin, "p_max": pmax,
                         "resel_priority": crp, "neighbors": neigh, "blacklist": black})
    return {"carriers": carriers}


# ---- SIB3 (cell reselection) ---------------------------------------------
def _lte_enc_sib3(w, s):
    w.b(0)
    w.b(False); w.u(s.get("q_hyst", 0), _sib1_nbits(16))       # cellReselectionInfoCommon
    s_non = s.get("s_non_intra_search")
    w.b(s_non is not None)
    if s_non is not None:
        _si_ei(w, s_non, 0, 31)
    _si_ei(w, s.get("thresh_serving_low", 0), 0, 31); _si_ei(w, s.get("resel_priority", 0), 0, 7)
    p_max = s.get("p_max"); s_intra = s.get("s_intra_search"); amb = s.get("allowed_meas_bw")
    w.b(p_max is not None); w.b(s_intra is not None); w.b(amb is not None); w.b(False)
    _si_ei(w, s.get("q_rxlevmin", -60), -70, -22)
    if p_max is not None:
        _si_ei(w, p_max, -30, 33)
    if s_intra is not None:
        _si_ei(w, s_intra, 0, 31)
    if amb is not None:
        w.u(amb, _sib1_nbits(6))
    w.b(s.get("presence_ant1", False)); w.u(s.get("neigh_cell_cfg", 0), 2)
    _si_ei(w, s.get("t_resel", 0), 0, 7)


def _lte_dec_sib3(r):
    r.bit(); has_speed = r.bit(); q_hyst = r.u(_sib1_nbits(16))
    if has_speed:
        r.u(3); r.u(3); _si_di(r, 1, 16); _si_di(r, 1, 16); r.u(2); r.u(2)
    has_snon = r.bit()
    s_non = _si_di(r, 0, 31) if has_snon else None
    _si_di(r, 0, 31)                             # threshServingLow
    resel_prio = _si_di(r, 0, 7)
    has_pmax = r.bit(); has_sintra = r.bit(); has_amb = r.bit(); has_sf = r.bit()
    q_rxlevmin = _si_di(r, -70, -22)
    p_max = _si_di(r, -30, 33) if has_pmax else None
    s_intra = _si_di(r, 0, 31) if has_sintra else None
    if has_amb:
        r.u(_sib1_nbits(6))
    r.bit(); r.u(2); _si_di(r, 0, 7)             # presenceAnt1, neighCellConfig, t-Resel
    if has_sf:
        r.u(2); r.u(2)
    return {"q_hyst": q_hyst, "s_non_intra_search": s_non, "resel_priority": resel_prio,
            "q_rxlevmin": q_rxlevmin, "p_max": p_max, "s_intra_search": s_intra}


# ---- SIB2 (access barring + radioResourceConfigCommon + freqInfo) ---------
def _lte_enc_acbc(w, cfg):
    w.u(cfg.get("factor", 0), _sib1_nbits(16)); w.u(cfg.get("time", 0), _sib1_nbits(8))
    w.u(cfg.get("special", 0), 5)


def _lte_dec_acbc(r):
    return {"factor": r.u(_sib1_nbits(16)), "time": r.u(_sib1_nbits(8)), "special": r.u(5)}


def _lte_enc_rrc(w, rrc):
    w.b(0)                                       # RRC-CommonSIB ext
    w.b(0)                                       # RACH-ConfigCommon ext
    w.b(False); w.u(rrc.get("num_ra_preambles", 0), _sib1_nbits(16))
    w.u(0, _sib1_nbits(4)); w.u(0, _sib1_nbits(16))
    w.u(0, _sib1_nbits(11)); w.u(0, _sib1_nbits(8)); w.u(0, _sib1_nbits(8))
    _si_ei(w, 4, 1, 8)
    w.b(0); w.u(0, _sib1_nbits(4))               # bcch-Config
    w.b(0); w.u(0, _sib1_nbits(4)); w.u(0, _sib1_nbits(8))   # pcch-Config
    _si_ei(w, rrc.get("prach_root", 0), 0, 837)
    _si_ei(w, rrc.get("prach_config_index", 0), 0, 63)
    w.b(rrc.get("prach_high_speed", False))
    _si_ei(w, rrc.get("prach_zcc", 0), 0, 15); _si_ei(w, rrc.get("prach_freq_offset", 0), 0, 94)
    _si_ei(w, rrc.get("ref_sig_power", 0), -60, 50); _si_ei(w, 0, 0, 3)
    _si_ei(w, 1, 1, 4); w.u(0, 1); _si_ei(w, 0, 0, 98); w.b(False)
    w.b(False); _si_ei(w, 0, 0, 29); w.b(False); _si_ei(w, 0, 0, 7)
    w.u(0, _sib1_nbits(3)); _si_ei(w, 0, 0, 98); _si_ei(w, 0, 0, 7); _si_ei(w, 0, 0, 2047)
    w.b(0)                                       # soundingRS release
    _si_ei(w, 0, -126, 24); w.u(0, _sib1_nbits(8)); _si_ei(w, -100, -127, -96)
    w.u(0, _sib1_nbits(3)); w.u(0, _sib1_nbits(3)); w.u(0, _sib1_nbits(4))
    w.u(0, _sib1_nbits(3)); w.u(0, _sib1_nbits(3)); _si_ei(w, 0, -1, 6)
    w.u(rrc.get("ul_cp", 0), _sib1_nbits(2))


def _lte_dec_rrc(r):
    out = {}
    r.bit(); r.bit(); has_grpA = r.bit(); r.u(_sib1_nbits(16))
    if has_grpA:
        r.u(_sib1_nbits(15)); r.u(_sib1_nbits(4)); r.u(_sib1_nbits(8))
    r.u(_sib1_nbits(4)); r.u(_sib1_nbits(16))
    r.u(_sib1_nbits(11)); r.u(_sib1_nbits(8)); r.u(_sib1_nbits(8)); _si_di(r, 1, 8)
    r.bit(); r.u(_sib1_nbits(4)); r.bit(); r.u(_sib1_nbits(4)); r.u(_sib1_nbits(8))
    out["prach_root"] = _si_di(r, 0, 837)
    out["prach_config_index"] = _si_di(r, 0, 63)
    out["prach_high_speed"] = bool(r.bit())
    out["prach_zcc"] = _si_di(r, 0, 15)
    out["prach_freq_offset"] = _si_di(r, 0, 94)
    out["ref_sig_power"] = _si_di(r, -60, 50); _si_di(r, 0, 3)
    _si_di(r, 1, 4); r.u(1); _si_di(r, 0, 98); r.bit()
    r.bit(); _si_di(r, 0, 29); r.bit(); _si_di(r, 0, 7)
    r.u(_sib1_nbits(3)); _si_di(r, 0, 98); _si_di(r, 0, 7); _si_di(r, 0, 2047)
    if r.bit():
        r.bit(); r.u(_sib1_nbits(8)); r.u(_sib1_nbits(16)); r.bit()
    _si_di(r, -126, 24); r.u(_sib1_nbits(8)); _si_di(r, -127, -96)
    r.u(_sib1_nbits(3)); r.u(_sib1_nbits(3)); r.u(_sib1_nbits(4))
    r.u(_sib1_nbits(3)); r.u(_sib1_nbits(3)); _si_di(r, -1, 6)
    out["ul_cp"] = r.u(_sib1_nbits(2))
    return out


def _lte_enc_uetimers(w):
    w.b(0)
    for cnt in (8, 8, 7, 8, 7, 8):
        w.u(0, _sib1_nbits(cnt))


def _lte_dec_uetimers(r):
    r.bit()
    for cnt in (8, 8, 7, 8, 7, 8):
        r.u(_sib1_nbits(cnt))


def _lte_enc_mbsfn(w, lst):
    _si_ei(w, len(lst), 1, 8)
    for _ in lst:
        w.u(0, _sib1_nbits(6)); _si_ei(w, 0, 0, 7); w.b(0); w.u(0, 6)


def _lte_dec_mbsfn(r):
    for _ in range(_si_di(r, 1, 8)):
        r.u(_sib1_nbits(6)); _si_di(r, 0, 7)
        if r.bit() == 0:
            r.u(6)
        else:
            r.u(24)


_UL_BW_RB = [6, 15, 25, 50, 75, 100]


def _lte_enc_sib2(w, s):
    ac = s.get("ac_barring"); mbsfn = s.get("mbsfn", [])
    w.b(0); w.b(ac is not None); w.b(bool(mbsfn))
    if ac is not None:
        mo_sig = ac.get("mo_signalling"); mo_dat = ac.get("mo_data")
        w.b(mo_sig is not None); w.b(mo_dat is not None); w.b(ac.get("emergency", False))
        if mo_sig is not None:
            _lte_enc_acbc(w, mo_sig)
        if mo_dat is not None:
            _lte_enc_acbc(w, mo_dat)
    _lte_enc_rrc(w, s.get("rrc", {}))
    _lte_enc_uetimers(w)
    ul_e = s.get("ul_earfcn"); ul_bw = s.get("ul_bandwidth")
    w.b(ul_e is not None); w.b(ul_bw is not None)
    if ul_e is not None:
        _si_ei(w, ul_e, 0, 65535)
    if ul_bw is not None:
        w.u(ul_bw, _sib1_nbits(6))
    _si_ei(w, s.get("add_spectrum_emission", 1), 1, 32)
    if mbsfn:
        _lte_enc_mbsfn(w, mbsfn)
    w.u(s.get("time_align_timer", 7), _sib1_nbits(8))


def _lte_dec_sib2(r):
    out = {}
    r.bit(); has_ac = r.bit(); has_mbsfn = r.bit()
    if has_ac:
        has_sig = r.bit(); has_dat = r.bit()
        out["ac_barring"] = True
        out["ac_barring_emergency"] = bool(r.bit())
        out["ac_barring_mo_signalling"] = _lte_dec_acbc(r) if has_sig else None
        out["ac_barring_mo_data"] = _lte_dec_acbc(r) if has_dat else None
    else:
        out["ac_barring"] = False
    out["rrc"] = _lte_dec_rrc(r)
    _lte_dec_uetimers(r)
    has_ulf = r.bit(); has_ulbw = r.bit()
    out["ul_earfcn"] = _si_di(r, 0, 65535) if has_ulf else None
    bw_idx = r.u(_sib1_nbits(6)) if has_ulbw else None
    out["ul_bandwidth_rb"] = (_UL_BW_RB[bw_idx] if (bw_idx is not None and bw_idx < 6) else 0)
    _si_di(r, 1, 32)                             # additionalSpectrumEmission
    if has_mbsfn:
        _lte_dec_mbsfn(r)
    out["time_align_timer"] = r.u(_sib1_nbits(8))
    return out


# ---- SystemInformation container -----------------------------------------
_SIB_CHOICE_ROOT = 10
_LTE_SI_ENC = {2: _lte_enc_sib2, 3: _lte_enc_sib3, 4: _lte_enc_sib4, 5: _lte_enc_sib5}
_LTE_SI_DEC = {2: _lte_dec_sib2, 3: _lte_dec_sib3, 4: _lte_dec_sib4, 5: _lte_dec_sib5}


def lte_si_encode(sibs):
    """sibs: [(sib_type_int, sib_dict), ...] in one SI message. int8 bit array."""
    w = _SiW()
    w.b(0); w.b(0); w.b(0); w.b(0)               # c1/systemInformation/r8/nonCrit
    _si_ei(w, len(sibs), 1, 32)
    for st, sd in sibs:
        _si_ext_enc(w, st - 2, _SIB_CHOICE_ROOT)
        _LTE_SI_ENC[st](w, sd)
    return w.arr()


def lte_si_decode(bits):
    """Decode a SystemInformation message's transport-block bits to
    {"sibs": [(type, dict), ...]}, or None on a structural mismatch."""
    r = _Sib1Reader(bits)
    if r.bit() != 0 or r.bit() != 0 or r.bit() != 0:
        return None                              # not c1/systemInformation/r8
    r.bit()                                       # nonCriticalExtension present
    n = _si_di(r, 1, 32)
    sibs = []
    for _ in range(n):
        idx = _si_ext_dec(r, _SIB_CHOICE_ROOT)
        if idx < 0:
            break
        st = idx + 2
        if st not in _LTE_SI_DEC:
            break                                 # sib6..sib11 not modelled
        sibs.append((st, _LTE_SI_DEC[st](r)))
    if r.bad:
        return None
    return {"sibs": sibs}

# ---- 12. FSK front end: symbol integrate-and-dump + sync-word search ----------
class SymDump:
    """Integrate-and-dump symbol extractor; twin of atkdsp.SymDump (atkdsp.h §12)."""

    def __init__(self, sps, phase=0.0):
        if not (sps >= 1.0):
            raise ValueError("sps >= 1")
        self.sps = float(sps)
        if phase < 0.0 or phase >= self.sps:
            phase = 0.0
        self.acc = float(phase)
        self.sum = 0.0
        self.cnt = 0

    def reset(self):
        self.acc = 0.0; self.sum = 0.0; self.cnt = 0

    def out_max(self, n_in):
        return int(n_in / self.sps) + 2

    def process(self, x):
        x = np.asarray(x, dtype=np.float32)
        out = []
        for i in range(x.size):
            self.sum += float(x[i])
            self.cnt += 1
            self.acc += 1.0
            if self.acc >= self.sps:
                self.acc -= self.sps
                out.append(np.float32(self.sum / self.cnt))
                self.sum = 0.0
                self.cnt = 0
        return np.asarray(out, dtype=np.float32)


def sync_search(sym, pattern, max_err=0):
    """Twin of atkdsp.sync_search. Returns a list of (pos, polarity, errors)."""
    sym = np.asarray(sym, dtype=np.float32)
    pat = np.asarray(pattern, dtype=np.int8)
    P = pat.size
    if sym.size < P:
        return []
    bits = (sym < 0).astype(np.int8)
    hits = []
    for i in range(0, sym.size - P + 1):
        err = int(np.sum(bits[i:i + P] != pat))
        if err <= max_err:
            hits.append((i + P, 0, err))
        elif P - err <= max_err:
            hits.append((i + P, 1, P - err))
    return hits

def dc_track(x, state=None, alpha=0.01):
    """Twin of atkdsp.dc_track. Leaky-integrator baseline removal."""
    x = np.asarray(x, dtype=np.float32)
    if not (alpha > 0.0):
        return x.copy()
    s = 0.0 if state is None else float(state[0])
    out = np.empty_like(x)
    for i in range(x.size):
        xi = float(x[i])
        s += alpha * (xi - s)
        out[i] = np.float32(xi - s)
    if state is not None and x.size:
        state[0] = np.float32(s)
    return out


def agc(x, state=None, alpha=0.01, ref=1.0):
    """Twin of atkdsp.agc. Leaky-integrator amplitude normalisation."""
    x = np.asarray(x, dtype=np.float32)
    if not (ref > 0.0):
        ref = 1.0
    if not (alpha > 0.0):
        return x.copy()
    a = 0.0 if state is None else float(state[0])
    out = np.empty_like(x)
    for i in range(x.size):
        xi = float(x[i])
        a += alpha * (abs(xi) - a)
        out[i] = np.float32(xi * (ref / (a if a > 1e-9 else 1e-9)))
    if state is not None and x.size:
        state[0] = np.float32(a)
    return out


# ---- 13. 4FSK sync correlator: DMR / P25 pilots (the spec src/dmr_sync.c is held to)
#
# A COMPLEX-BASEBAND matched filter for the 24-symbol sync words of DMR (ETSI
# TS 102 361-1 Table 9.2) and the P25 Phase 1 frame sync (TIA-102.BAAA), with a
# bank of carrier-offset hypotheses, run across one or more phase-coherent
# channels at once (the KrakenSDR's five elements). ATK's KRAKEN_DF_PLAN has two
# consumers for it:
#
#   §6.5 pilot-aided steering vectors: the per-channel complex amplitude of THAT
#        emitter's sync at its peak. Its ratios are the array's steering vector,
#        with the processing gain of 24 coherently-integrated symbols (~14 dB)
#        over an energy estimate. Because the reference waveform is the same for
#        every element, any mismatch between it and the real transmitter is
#        common-mode: it costs SNR, never bearing bias.
#   §7.1 decoder-clocked DoA: sample-exact burst timing and the sync TYPE —
#        BS- or MS-sourced, direct-mode slot, voice or data — which is the path
#        tag that keeps a repeater's bearing from being fused as a handset's.
#
# The model. A sync symbol a in {+3, +1, -1, -3} is a frequency of (a/3)*dev
# (DMR: +3 -> +1944 Hz; P25: +1800 Hz). Each symbol's frequency pulse is a
# root-raised-cosine (DMR's transmit filter, alpha 0.2; used for P25's C4FM as
# an approximation) truncated to +-4 symbols, or a rectangle (plain CPFSK) when
# alpha < 0. The phase is 2*pi*(dev/R)*sum_m (a_m/3)*G(u - m - 1/2), with G the
# pulse's running integral (unit total area) and u the time in symbols from the
# sync's leading edge. Reference sample i sits at u = (i + 1/2)/sps, the window
# is L = round(24*sps) samples, and the phase is zero at the window centre, so
# an amplitude's phase is the carrier's phase at the middle of the sync.
#
# The correlator. Per window (ending at stream sample n), per pattern, per
# channel: one partial correlation per SYMBOL, p_m = sum_{i in symbol m}
# x[i]*conj(r[i]). A carrier offset df is then a phasor per symbol,
# C = sum_m p_m * exp(-j*2*pi*df*tau_m/fs) (tau_m the symbol's centre relative
# to the window's), so the bank costs 24 complex MACs per hypothesis instead of
# L. Within one symbol the offset is not compensated: at 1 kHz that loses 0.6 dB,
# at 500 Hz 0.15 dB. The score is the normalised correlation summed over
# channels — sum_c |C_c|^2 / (L * sum_c E_c), E_c the window's energy — in
# [0, 1], independent of gain, and non-coherent across channels so detection
# needs no bearing.
#
# Peak picking, streamed. A window scoring >= threshold becomes the candidate
# if none is pending or it beats the pending one; a candidate is emitted once
# `guard` samples have passed it with nothing better. Each emitted hit is then
# refined: a parabola through the scores at n-1, n, n+1 gives the sub-sample
# timing, one through the neighbouring offset bins the frequency, and the
# per-channel amplitudes are taken at that refined offset by exact per-sample
# derotation (no per-symbol approximation), with the score recomputed the same
# way. Everything depends only on the samples, so a block boundary is
# invisible: the C is bit-identical however the stream is chunked.

SYNC_DMR = 0
SYNC_P25 = 1
SYNC_INVERT = 1               # flags: the stream is spectrally inverted
SYNC_SYMBOLS = 24
SYNC_RATE = 4800.0            # symbols/s, both sets
SYNC_PULSE_SPAN = 4           # the RRC pulse is truncated to +-4 symbols
SYNC_PULSE_RES = 64           # running-integral table points per symbol
SYNC_FS_MIN = 9600.0          # 2 samples/symbol
SYNC_FS_MAX = 384000.0        # 80 samples/symbol — channelise wider streams first
SYNC_MAX_CHANNELS = 64
SYNC_SPAN_MAX = 5000.0
SYNC_MAX_BINS = 201
SYNC_GUARD_SYMBOLS = 66       # default guard: the rest of a DMR burst either side
SYNC_CONFIRM_ERRORS = 3       # a hit with <= this many symbol errors is "confirmed"

# (name, 48-bit pattern, source, kind, slot). ETSI TS 102 361-1 Table 9.2;
# checked symbol for symbol against dsd-neo's sync_patterns.h (an independent
# source) on 2026-10-25. "source" is the path tag: BS = a repeater/base station
# transmitted it, MS = a handset did, direct = handset-to-handset (DMO).
_DMR_SYNC = (
    ("BS voice", 0x755FD7DF75F7, "BS", "voice", 0),
    ("BS data", 0xDFF57D75DF5D, "BS", "data", 0),
    ("MS voice", 0x7F7D5DD57DFD, "MS", "voice", 0),
    ("MS data", 0xD5D7F77FD757, "MS", "data", 0),
    ("MS RC", 0x77D55F7DFD77, "MS", "rc", 0),
    ("DM TS1 voice", 0x5D577F7757FF, "direct", "voice", 1),
    ("DM TS1 data", 0xF7FDD5DDFD55, "direct", "data", 1),
    ("DM TS2 voice", 0x7DFFD5F55D5F, "direct", "voice", 2),
    ("DM TS2 data", 0xD7557F5FF7F5, "direct", "data", 2),
)
# TIA-102.BAAA frame sync. Inbound and outbound share it, so there is no path tag.
_P25_SYNC = (
    ("P25 frame sync", 0x5575F5FF77FF, "", "sync", 0),
)
SYNC_SETS = {
    SYNC_DMR: {"name": "DMR", "dev_hz": 1944.0, "alpha": 0.2, "patterns": _DMR_SYNC},
    SYNC_P25: {"name": "P25", "dev_hz": 1800.0, "alpha": 0.2, "patterns": _P25_SYNC},
}
# Dibit -> symbol, ETSI TS 102 361-1 Table 10.3 (and TIA-102.BAAA): 01 +3,
# 00 +1, 10 -1, 11 -3. Every sync word uses only the outer two.
_SYNC_DIBIT = {1: 3, 0: 1, 2: -1, 3: -3}


def sync_set_id(s) -> int:
    """'dmr' / 'p25' / SYNC_DMR / SYNC_P25 -> the set number."""
    if isinstance(s, str):
        key = s.strip().lower()
        for k, v in SYNC_SETS.items():
            if v["name"].lower() == key:
                return k
        raise ValueError(f"unknown sync set {s!r} (dmr, p25)")
    s = int(s)
    if s not in SYNC_SETS:
        raise ValueError(f"unknown sync set {s}")
    return s


def dmr_sync_patterns(sync_set=SYNC_DMR) -> int:
    """How many sync words the set has."""
    return len(SYNC_SETS[sync_set_id(sync_set)]["patterns"])


def dmr_sync_pattern(sync_set, idx) -> np.ndarray:
    """The 24 symbols (+-3, +-1) of one sync word, first transmitted first."""
    pats = SYNC_SETS[sync_set_id(sync_set)]["patterns"]
    if not 0 <= int(idx) < len(pats):
        raise ValueError(f"pattern {idx} out of range")
    word = pats[int(idx)][1]
    return np.array([_SYNC_DIBIT[(word >> (46 - 2 * i)) & 3] for i in range(SYNC_SYMBOLS)],
                    dtype=np.int8)


def dmr_sync_pattern_name(sync_set, idx) -> str:
    pats = SYNC_SETS[sync_set_id(sync_set)]["patterns"]
    if not 0 <= int(idx) < len(pats):
        raise ValueError(f"pattern {idx} out of range")
    return pats[int(idx)][0]


def dmr_sync_pattern_info(sync_set, idx) -> dict:
    """name / source / kind / slot for one pattern (the tags a hit carries)."""
    name, _, source, kind, slot = SYNC_SETS[sync_set_id(sync_set)]["patterns"][int(idx)]
    return {"name": name, "source": source, "kind": kind, "slot": slot}


def _sync_mask(sync_set, patterns) -> int:
    """None -> every pattern; an int -> a bitmask as given; else an iterable
    of indices or names."""
    n = dmr_sync_patterns(sync_set)
    full = (1 << n) - 1
    if patterns is None:
        return full
    if isinstance(patterns, (int, np.integer)):
        return int(patterns) & full
    names = [p[0].lower() for p in SYNC_SETS[sync_set_id(sync_set)]["patterns"]]
    mask = 0
    for p in patterns:
        if isinstance(p, str):
            try:
                mask |= 1 << names.index(p.strip().lower())
            except ValueError:
                raise ValueError(f"unknown pattern {p!r}; the set has {names}") from None
        else:
            if not 0 <= int(p) < n:
                raise ValueError(f"pattern {p} out of range")
            mask |= 1 << int(p)
    return mask


def _sync_rrc(u: float, alpha: float) -> float:
    """Root-raised-cosine impulse response at u symbols (peak 1 - a + 4a/pi)."""
    if u == 0.0:
        return 1.0 - alpha + 4.0 * alpha / np.pi
    if alpha > 0.0 and abs(1.0 - 16.0 * alpha * alpha * u * u) < 1e-9:
        q = np.pi / (4.0 * alpha)
        return (alpha / np.sqrt(2.0)) * ((1.0 + 2.0 / np.pi) * np.sin(q)
                                         + (1.0 - 2.0 / np.pi) * np.cos(q))
    num = np.sin(np.pi * u * (1.0 - alpha)) + 4.0 * alpha * u * np.cos(np.pi * u * (1.0 + alpha))
    den = np.pi * u * (1.0 - 16.0 * alpha * alpha * u * u)
    return num / den


def _sync_pulse_table(alpha: float):
    """Running integral of the frequency pulse on u = -4 .. +4 in 1/64-symbol
    steps (trapezoid), normalised to a total of exactly 1. None = rectangle."""
    if alpha < 0.0:
        return None
    n = 2 * SYNC_PULSE_SPAN * SYNC_PULSE_RES
    du = 1.0 / SYNC_PULSE_RES
    h = [_sync_rrc(-SYNC_PULSE_SPAN + k * du, alpha) for k in range(n + 1)]
    g = np.zeros(n + 1)
    acc = 0.0
    for k in range(1, n + 1):
        acc += 0.5 * (h[k - 1] + h[k]) * du
        g[k] = acc
    return g / g[n]


def _sync_cum(tab, u: np.ndarray) -> np.ndarray:
    """G(u): the pulse's running integral, 0 before it and 1 after it."""
    u = np.asarray(u, dtype=np.float64)
    if tab is None:
        return np.clip(u + 0.5, 0.0, 1.0)
    span = float(SYNC_PULSE_SPAN)
    out = np.where(u >= span, 1.0, 0.0)
    inside = (u > -span) & (u < span)
    x = (u[inside] + span) * SYNC_PULSE_RES
    k = np.floor(x).astype(np.int64)
    k = np.minimum(k, tab.size - 2)
    f = x - k
    out[inside] = tab[k] + f * (tab[k + 1] - tab[k])
    return out


def _sync_alpha(sync_set, alpha) -> float:
    if alpha is None or (isinstance(alpha, float) and np.isnan(alpha)):
        return float(SYNC_SETS[sync_set_id(sync_set)]["alpha"])
    return float(alpha)


def _sync_window(sample_rate: float) -> int:
    return int(np.floor(SYNC_SYMBOLS * sample_rate / SYNC_RATE + 0.5))


def _sync_phase(sym, dev_hz, tab, u) -> np.ndarray:
    """2*pi*(dev/R)*sum_m (a_m/3)*G(u - m - 1/2), at every u."""
    u = np.asarray(u, dtype=np.float64)
    acc = np.zeros(u.shape)
    for m in range(len(sym)):                       # ascending m, as the C sums
        acc += (float(sym[m]) / 3.0) * _sync_cum(tab, u - m - 0.5)
    return 2.0 * np.pi * (dev_hz / SYNC_RATE) * acc


def sync_reference(sync_set, idx, sample_rate: float, alpha=None, invert: bool = False) -> np.ndarray:
    """The reference waveform of one sync word at `sample_rate`: L unit-modulus
    complex samples, phase zero at the window centre. Twin of
    atkdsp_dmr_sync_reference."""
    sid = sync_set_id(sync_set)
    a = _sync_alpha(sid, alpha)
    sym = dmr_sync_pattern(sid, idx)
    sps = float(sample_rate) / SYNC_RATE
    L = _sync_window(sample_rate)
    tab = _sync_pulse_table(a)
    dev = SYNC_SETS[sid]["dev_hz"]
    u = (np.arange(L) + 0.5) / sps
    ph = _sync_phase(sym, dev, tab, u) - _sync_phase(sym, dev, tab, np.array([L / (2.0 * sps)]))[0]
    r = np.exp(1j * ph)
    return np.conj(r) if invert else r


def sync_segments(sample_rate: float) -> np.ndarray:
    """Symbol boundaries in the window: 25 indices, segment m = [b_m, b_m+1)."""
    sps = float(sample_rate) / SYNC_RATE
    L = _sync_window(sample_rate)
    b = [min(L, max(0, int(np.ceil(m * sps - 0.5)))) for m in range(SYNC_SYMBOLS)]
    b.append(L)
    return np.array(b, dtype=np.int64)


def sync_build_burst(sync_set, idx, sample_rate: float, payload=None, rng=None,
                     alpha=None, cfo_hz: float = 0.0, phase: float = 0.0,
                     invert: bool = False):
    """A synthetic burst for tests and simulation: 54 payload symbols, the sync
    word, 54 more (DMR's 132-symbol burst, sync in the middle), shaped and
    frequency-modulated exactly as the reference is. Returns (iq complex64,
    window_start) — the index the correlator should report as `start` (exact
    when 54*sps is an integer)."""
    sid = sync_set_id(sync_set)
    a = _sync_alpha(sid, alpha)
    sync = dmr_sync_pattern(sid, idx).astype(np.float64)
    if payload is None:
        rng = np.random.default_rng(0) if rng is None else rng
        payload = rng.choice(np.array([3, 1, -1, -3]), size=108)
    payload = np.asarray(payload, dtype=np.float64)
    half = payload.size // 2
    sym = np.concatenate([payload[:half], sync, payload[half:]])
    sps = float(sample_rate) / SYNC_RATE
    n = int(np.floor(sym.size * sps + 0.5))
    tab = _sync_pulse_table(a)
    dev = SYNC_SETS[sid]["dev_hz"]
    u = (np.arange(n) + 0.5) / sps
    ph = _sync_phase(sym, dev, tab, u)
    if invert:
        ph = -ph
    t = np.arange(n) / float(sample_rate)
    iq = np.exp(1j * (ph + 2.0 * np.pi * cfo_hz * t + phase))
    return iq.astype(np.complex64), int(np.floor(half * sps + 0.5))


def _sync_symbol_errors(w, seg, sym, cfo_hz, fs, invert) -> int:
    """How many of the 24 sync symbols have the wrong frequency SIGN, measured
    DSD-style from the phase advance inside each symbol (sum over channels of
    x[i] * conj(x[i-1]), carrier offset removed). The channels' phases cancel in
    the product, so this needs no bearing. A true sync reads 0 down to a few dB
    SNR; a partial match in payload reads several."""
    rot = np.exp(-2j * np.pi * cfo_hz / fs)
    d = w[:, 1:] * np.conj(w[:, :-1])                  # d[:, i-1] is the step into sample i
    errors = 0
    for m in range(SYNC_SYMBOLS):
        acc = np.sum(d[:, seg[m]:seg[m + 1] - 1]) * rot if seg[m + 1] - seg[m] > 1 else 0j
        # the pairs (i-1, i) for i in (b_m, b_m+1): both samples inside the symbol
        want = 1 if sym[m] > 0 else -1
        if invert:
            want = -want
        got = 1 if acc.imag > 0 else (-1 if acc.imag < 0 else 0)
        errors += got != want
    return int(errors)


def sync_default_threshold(n_channels: int) -> float:
    """The default score threshold for n channels: the level that pure noise in
    a 12.5 kHz channel crosses about once in five minutes (and never with a
    confirmed hit). Measured 2026-10-25 on 4-5 minutes of filtered noise each:
    one channel 0.33, two 0.20, five 0.11 — the score is averaged across
    channels, so noise tightens and the threshold can fall, which is what lets
    the five-element Kraken detect syncs well below 0 dB per element. Between
    and beyond the measured points: 0.33 * n^-0.68, then 0.08 * sqrt(8/n)
    past eight channels (above the Gamma-tail model, on the safe side)."""
    n = max(1, int(n_channels))
    if n <= 8:
        return round(float(0.33 * n ** -0.68), 3)
    return round(float(0.08 * np.sqrt(8.0 / n)), 3)


def _sync_parab(a, b, c) -> float:
    """Vertex offset (-0.5..0.5) of the parabola through (-1,a), (0,b), (1,c);
    0 when a side is missing or it is not a maximum."""
    if a is None or c is None:
        return 0.0
    den = a - 2.0 * b + c
    if not den < 0.0:
        return 0.0
    return float(min(0.5, max(-0.5, 0.5 * (a - c) / den)))


class DmrSync:
    """Streaming 4FSK sync correlator; twin of atkdsp.DmrSync (atkdsp.h §13).

    Feed it phase-coherent channels — shape (N,) for one, (n_channels, N) for
    several — at `sample_rate`, in blocks of any size. `process` returns the
    syncs whose peaks are final (a peak is final once `guard` samples have
    passed it; `flush` emits a pending one at the end of a stream or a one-shot
    buffer), each a dict:

      start    stream index of the window's first sample (the sync's leading
               edge is at start + frac - 1/2)
      frac     sub-sample timing refinement, -0.5 .. 0.5
      cfo_hz   carrier offset, refined between bins
      score    normalised correlation, 0..1, summed over channels
      errors   sync symbols whose frequency SIGN reads wrong (0..24)
      confirmed  errors <= SYNC_CONFIRM_ERRORS: the type label can be trusted
      pattern  index in the set; name / source (BS, MS, direct) / kind / slot
      amps     complex64 per channel: the sync's complex amplitude, phase at
               the window centre. amps / amps[0] is the steering vector.

    An unconfirmed hit inside an emitter's burst is a partial match in its
    payload: the label is wrong, but it is still THAT emitter's signal, so its
    amps are still its steering vector (at less SNR)."""

    def __init__(self, sample_rate: float, n_channels: int = 1, sync_set=SYNC_DMR,
                 patterns=None, cfo_span_hz: float = 1000.0, cfo_step_hz: float = 0.0,
                 threshold=None, guard: int = 0, alpha=None, invert: bool = False):
        sid = sync_set_id(sync_set)
        fs = float(sample_rate)
        if threshold is None:
            threshold = sync_default_threshold(n_channels)
        if not SYNC_FS_MIN <= fs <= SYNC_FS_MAX:
            raise ValueError(f"sample_rate {fs} outside {SYNC_FS_MIN:.0f}..{SYNC_FS_MAX:.0f} Hz "
                             f"(channelise to the 12.5 kHz channel first)")
        if not 1 <= int(n_channels) <= SYNC_MAX_CHANNELS:
            raise ValueError("n_channels 1..64")
        mask = _sync_mask(sid, patterns)
        if mask == 0:
            raise ValueError("no patterns selected")
        a = _sync_alpha(sid, alpha)
        if a > 1.0:
            raise ValueError("alpha <= 1 (negative = rectangular pulses)")
        span = float(cfo_span_hz)
        if not 0.0 <= span <= SYNC_SPAN_MAX:
            raise ValueError("cfo_span_hz 0..5000")
        step = float(cfo_step_hz)
        if step < 0.0:
            raise ValueError("cfo_step_hz >= 0 (0 = automatic)")
        if step == 0.0:
            step = 0.5 * SYNC_RATE / SYNC_SYMBOLS
        kmax = int(np.ceil(span / step - 1e-9)) if span > 0.0 else 0
        if 2 * kmax + 1 > SYNC_MAX_BINS:
            raise ValueError("too many offset bins (cfo_span_hz / cfo_step_hz > 100)")
        thr = float(threshold)
        if not 0.0 < thr <= 1.0:
            raise ValueError("threshold in (0, 1]")
        L = _sync_window(fs)
        g = int(guard) if int(guard) > 0 else int(np.floor(SYNC_GUARD_SYMBOLS * fs / SYNC_RATE + 0.5))
        if g > 16 * L:
            raise ValueError("guard <= 16 windows")
        self.sync_set = sid
        self.sample_rate = fs
        self.n_channels = int(n_channels)
        self.mask = mask
        self.active = [p for p in range(dmr_sync_patterns(sid)) if mask >> p & 1]
        self.alpha = a
        self.invert = bool(invert)
        self.L = L
        self.cfo_step = step
        self.kmax = kmax
        self.n_cfo = 2 * kmax + 1
        self.cfo_span = span
        self.threshold = thr
        self.guard = g
        self.history = L + g + 2
        self._ref = [sync_reference(sid, p, fs, a, invert) for p in range(dmr_sync_patterns(sid))]
        self._seg = sync_segments(fs)
        M = SYNC_SYMBOLS
        tau = (self._seg[:-1] + self._seg[1:] - 1) / 2.0 - (L - 1) / 2.0
        # bins -1 .. K: the outer two exist only so an edge hit can be refined
        df = (np.arange(-1, self.n_cfo + 1) - kmax) * step
        self._rot = np.exp(-2j * np.pi * df[:, None] * tau[None, :] / fs)       # (K+2, M)
        P = len(self.active)
        R = np.zeros((L, P * M), dtype=np.complex128)
        for pa, p in enumerate(self.active):
            cr = np.conj(self._ref[p])
            for m in range(M):
                R[self._seg[m]:self._seg[m + 1], pa * M + m] = cr[self._seg[m]:self._seg[m + 1]]
        self._R = R
        self.reset()

    # -- state ---------------------------------------------------------------
    def reset(self):
        self._hist = np.zeros((self.n_channels, 0), dtype=np.complex64)
        self._total = 0
        self._cand = None          # (n_abs, score, active index, cfo bin)

    def hits_max(self, n_in: int) -> int:
        return int(n_in) // self.guard + 2

    def reference(self, idx) -> np.ndarray:
        return self._ref[int(idx)].astype(np.complex64)

    def describe(self) -> str:
        return _sync_describe(self)

    # -- the arithmetic --------------------------------------------------------
    def _scores(self, buf, e_lo, e_hi):
        """Best (score, active*K + bin) for windows ending at e_lo .. e_hi-1."""
        L, K, M = self.L, self.n_cfo, SYNC_SYMBOLS
        P = len(self.active)
        n = e_hi - e_lo
        best = np.full(n, -1.0)
        arg = np.zeros(n, dtype=np.int64)
        step = 4096
        for c0 in range(0, n, step):
            c1 = min(n, c0 + step)
            acc = np.zeros((c1 - c0, P, K))
            E = np.zeros(c1 - c0)
            for c in range(self.n_channels):
                xc = buf[c].astype(np.complex128)
                W = np.lib.stride_tricks.sliding_window_view(xc, L)[e_lo - L + 1 + c0:e_lo - L + 1 + c1]
                parts = (W @ self._R).reshape(c1 - c0, P, M)
                C = parts @ self._rot[1:-1].T                                   # (n, P, K)
                acc += C.real ** 2 + C.imag ** 2
                E += np.sum(W.real ** 2 + W.imag ** 2, axis=1)
            ok = E > 1e-30
            s = np.zeros_like(acc)
            s[ok] = acc[ok] / (L * E[ok])[:, None, None]
            flat = s.reshape(c1 - c0, P * K)
            a = np.argmax(flat, axis=1)
            b = flat[np.arange(c1 - c0), a]
            b[~ok] = -1.0
            best[c0:c1] = b
            arg[c0:c1] = a
        return best, arg

    def _score_at(self, buf, e, pa, k) -> float:
        L = self.L
        w = buf[:, e - L + 1:e + 1].astype(np.complex128)
        prod = w * np.conj(self._ref[self.active[pa]])[None, :]
        parts = np.add.reduceat(prod, self._seg[:-1], axis=1)                   # (ch, M)
        C = parts @ self._rot[k + 1]
        E = float(np.sum(w.real ** 2 + w.imag ** 2))
        if E <= 1e-30:
            return 0.0
        return float(np.sum(C.real ** 2 + C.imag ** 2) / (L * E))

    def _exact(self, buf, e, ref):
        """(amps, score) of the window ending at e against `ref` (conj(r) times
        the refined derotation) — exact per-sample, no per-symbol phasors."""
        L = self.L
        w = buf[:, e - L + 1:e + 1].astype(np.complex128)
        amps = (w * ref[None, :]).sum(axis=1) / L
        E = float(np.sum(w.real ** 2 + w.imag ** 2))
        score = float(L * np.sum(np.abs(amps) ** 2) / E) if E > 1e-30 else 0.0
        return amps, score

    def _emit(self, buf, base, cand) -> dict:
        # The offset first (a parabola through the neighbouring bins), THEN the
        # timing, from exact scores at that offset: for FSK a residual offset
        # looks like a timing error, so the other order biases the timing.
        n_abs, _, pa, k = cand
        L, fs = self.L, self.sample_rate
        e = n_abs - base
        g0 = self._score_at(buf, e, pa, k)
        gm = self._score_at(buf, e, pa, k - 1)          # bins -1 and K exist for this
        gp = self._score_at(buf, e, pa, k + 1)
        cfo = (k - self.kmax + _sync_parab(gm, g0, gp)) * self.cfo_step
        p = self.active[pa]
        i = np.arange(L)
        ref = np.conj(self._ref[p]) * np.exp(-2j * np.pi * cfo * (i - (L - 1) / 2.0) / fs)
        amps, score = self._exact(buf, e, ref)
        sm = self._exact(buf, e - 1, ref)[1] if n_abs - 1 >= L - 1 else None
        sp = self._exact(buf, e + 1, ref)[1] if e + 1 < buf.shape[1] else None
        dt = _sync_parab(sm, score, sp)
        w = buf[:, e - L + 1:e + 1].astype(np.complex128)
        errors = _sync_symbol_errors(w, self._seg, dmr_sync_pattern(self.sync_set, p),
                                     cfo, fs, self.invert)
        hit = {"start": int(n_abs - L + 1), "frac": dt, "cfo_hz": float(cfo),
               "score": score, "errors": errors,
               "confirmed": errors <= SYNC_CONFIRM_ERRORS, "pattern": int(p)}
        hit.update(dmr_sync_pattern_info(self.sync_set, p))
        hit["amps"] = amps.astype(np.complex64)
        return hit

    def process(self, x) -> list:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[0] != self.n_channels:
            raise ValueError(f"expected shape ({self.n_channels}, N)")
        x = x.astype(np.complex64)
        N = x.shape[1]
        hist_len = self._hist.shape[1]
        buf = np.concatenate([self._hist, x], axis=1)
        base = self._total - hist_len
        best = np.full(N, -1.0)
        arg = np.zeros(N, dtype=np.int64)
        e_lo = max(self.L - 1, hist_len)
        if e_lo < hist_len + N:
            b, a = self._scores(buf, e_lo, hist_len + N)
            best[e_lo - hist_len:] = b
            arg[e_lo - hist_len:] = a
        K, G = self.n_cfo, self.guard
        hits = []
        for j in np.flatnonzero(best >= self.threshold):
            n = self._total + int(j)
            if self._cand is not None and n - self._cand[0] >= G:
                hits.append(self._emit(buf, base, self._cand))
                self._cand = None
            if self._cand is None or best[j] > self._cand[1]:
                self._cand = (n, float(best[j]), int(arg[j]) // K, int(arg[j]) % K)
        if self._cand is not None and N > 0 and (self._total + N - 1) - self._cand[0] >= G:
            hits.append(self._emit(buf, base, self._cand))
            self._cand = None
        self._total += N
        keep = min(self.history, buf.shape[1])
        self._hist = buf[:, buf.shape[1] - keep:].copy()
        return hits

    def flush(self) -> list:
        """Emit the pending candidate now, without waiting out its guard — for
        the end of a stream or a one-shot buffer (a burst span from the ring).
        The stream may continue afterwards; a later, better peak within the
        guard is then reported as well."""
        if self._cand is None:
            return []
        base = self._total - self._hist.shape[1]
        hit = self._emit(self._hist, base, self._cand)
        self._cand = None
        return [hit]


def _sync_describe(s) -> str:
    """The same sentence atkdsp_dmr_sync_describe writes."""
    pulse = "rect" if s.alpha < 0 else f"RRC a={s.alpha:.2f}"
    return (f"{SYNC_SETS[s.sync_set]['name']} sync: {s.sample_rate:.0f} Hz "
            f"({s.sample_rate / SYNC_RATE:.2f} sps), {s.L}-sample window, "
            f"{len(s.active)} pattern{'s' if len(s.active) != 1 else ''}, "
            f"CFO +-{s.kmax * s.cfo_step:.0f} Hz in {s.n_cfo} bin{'s' if s.n_cfo != 1 else ''} "
            f"of {s.cfo_step:.0f} Hz, threshold {s.threshold:.2f}, guard {s.guard}, {pulse}, "
            f"{s.n_channels} channel{'s' if s.n_channels != 1 else ''}"
            f"{', inverted' if s.invert else ''}")


# ============================================================================
# THE BITSTREAM ENGINE (§14-17) — the twins of src/bits.c, src/scramble.c,
# src/fec_conv.c, src/fec_rs.c, src/fec_blk.c, src/crc.c, src/crc_recover.c.
# ----------------------------------------------------------------------------
# Bits are one per uint8 (only bit 0 read), first bit first; integer words are
# MSB-first. Every function here computes what the C computes, in the same
# order where order matters (the scrambler search's G statistic, the Viterbi
# metrics in float32, the CRC recovery's search), so the cross-check can ask
# for equality — and so ATK's fallback, when the DLL is absent, is slower but
# never different.
# ============================================================================
import math as _m

_LN2 = _m.log(2.0)
_LN10 = _m.log(10.0)


def _bits(b) -> np.ndarray:
    return (np.ascontiguousarray(b, dtype=np.uint8).ravel() & 1).astype(np.uint8)


# ---- §14 helpers: the statistics src/stat.c shares ---------------------------
def _igam_iters(a) -> int:
    """The iteration bound src/stat.c uses: both expansions need O(sqrt(a))."""
    return 1000 + int(10.0 * _m.sqrt(a if a > 0.0 else 0.0))


def _gamma_p_series(a, x):
    ap, dl = a, 1.0 / a
    s = dl
    for _ in range(_igam_iters(a)):
        ap += 1.0
        dl *= x / ap
        s += dl
        if abs(dl) < abs(s) * 1e-16:
            break
    return s * _m.exp(-x + a * _m.log(x) - _m.lgamma(a))


def _gamma_q_cf_log(a, x):
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, _igam_iters(a)):
        an = -float(i) * (float(i) - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        dl = d * c
        h *= dl
        if abs(dl - 1.0) < 1e-16:
            break
    return _m.log(h)


def log_igamc(a, x) -> float:
    """ln Q(a, x), the regularised upper incomplete gamma, finite in the tail."""
    if not a > 0.0 or not x > 0.0:
        return 0.0
    if x < a + 1.0:
        p = _gamma_p_series(a, x)
        return -_m.inf if p >= 1.0 else _m.log1p(-p)
    return -x + a * _m.log(x) - _m.lgamma(a) + _gamma_q_cf_log(a, x)


def igamc(a, x) -> float:
    return _m.exp(log_igamc(a, x))


def log10_binom_tail(n, c, ln_q) -> float:
    """log10 P(Binomial(n, q) >= c), 0 at or below the mean."""
    if c <= 0.0:
        return 0.0
    if c > n:
        return -_m.inf
    q = _m.exp(ln_q)
    if c <= n * q:
        return 0.0
    ln_1mq = _m.log1p(-q)

    def term(k):
        return (_m.lgamma(n + 1.0) - _m.lgamma(k + 1.0) - _m.lgamma(n - k + 1.0)
                + k * ln_q + (n - k) * ln_1mq)
    t0 = term(c)
    s = 1.0
    k = c + 1.0
    while k <= n:
        d = term(k) - t0
        if d < -40.0:
            break
        s += _m.exp(d)
        k += 1.0
    r = (t0 + _m.log(s)) / _LN10
    return 0.0 if r > 0.0 else r


_ACK_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_ACK_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
_ACK_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_ACK_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00)


def _norm_ppf(q):
    a, b, c, d = _ACK_A, _ACK_B, _ACK_C, _ACK_D
    plow = 0.02425
    if q <= 0.0:
        return -_m.inf
    if q >= 1.0:
        return _m.inf
    if q < plow:
        t = _m.sqrt(-2.0 * _m.log(q))
        x = ((((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5])
             / ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1.0))
    elif q <= 1.0 - plow:
        r = q - 0.5
        s = r * r
        x = ((((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r
             / (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1.0))
    else:
        t = _m.sqrt(-2.0 * _m.log(1.0 - q))
        x = -((((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5])
              / ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1.0))
    e = 0.5 * _m.erfc(-x / _m.sqrt(2.0)) - q
    u = e * _m.sqrt(2.0 * _m.pi) * _m.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


def norm_isf(p) -> float:
    """z with P(N(0,1) > z) = p (Acklam + one Halley step)."""
    return -_norm_ppf(p)


def chi_width(nbits) -> int:
    for w in range(8, 0, -1):
        if nbits // w >= 5 << w:
            return w
    return 0


def _words(b, w):
    """Non-overlapping w-bit words from bit 0, MSB-first."""
    nw = b.size // w
    if nw == 0:
        return np.zeros(0, np.int64)
    blk = b[:nw * w].reshape(nw, w).astype(np.int64)
    return blk @ (1 << np.arange(w - 1, -1, -1, dtype=np.int64))


def gstat(b, w) -> float:
    """G = 2 sum O ln(O/E) of w-bit words against uniform (cells ascending)."""
    words = _words(_bits(b), w)
    cells = 1 << w
    if words.size == 0:
        return float("nan")
    cnt = np.bincount(words, minlength=cells)
    e = words.size / cells
    g = 0.0
    for v in range(cells):
        o = int(cnt[v])
        if o:
            g += o * _m.log(o / e)
    return 2.0 * g


# ---- §14 bit primitives --------------------------------------------------------
BITS_INVERTED = 1
BITS_REVERSED = 2
SYNC_MAXBITS = 128
SYNC_REPETITIVE = 1


def bits_autocorr(b, max_lag: int) -> np.ndarray:
    b = _bits(b)
    n = b.size
    out = np.full(max_lag + 1, np.nan, np.float32)
    for k in range(min(max_lag, n - 1) + 1):
        m = n - k
        d = int(np.count_nonzero(b[:m] != b[k:k + m]))
        out[k] = np.float32((m - 2.0 * d) / m)
    return out


def bits_xcorr(a, p) -> np.ndarray:
    a = _bits(a)
    p = _bits(p)
    nb = p.size
    if nb < 1 or nb > a.size:
        raise ValueError("bits_xcorr needs 1 <= len(p) <= len(a)")
    win = np.lib.stride_tricks.sliding_window_view(a, nb)
    d = np.count_nonzero(win != p[None, :], axis=1)
    return ((nb - 2.0 * d) / nb).astype(np.float32)


def bits_search(b, pat, max_err: int = 0, flags: int = 0) -> list:
    """[(pos, errors, variant)] — see atkdsp.h §14."""
    b = _bits(b)
    pat = _bits(pat)
    plen = pat.size
    if plen < 1 or plen > 4096 or max_err < 0:
        raise ValueError("bits_search: plen 1..4096, max_err >= 0")
    if plen > b.size:
        return []
    win = np.lib.stride_tricks.sliding_window_view(b, plen)
    d0 = np.count_nonzero(win != pat[None, :], axis=1).astype(np.int64)
    cand = [d0]
    names = [0]
    if flags & BITS_INVERTED:
        cand.append(plen - d0); names.append(1)
    if flags & BITS_REVERSED:
        d2 = np.count_nonzero(win != pat[::-1][None, :], axis=1).astype(np.int64)
        cand.append(d2); names.append(2)
        if flags & BITS_INVERTED:
            cand.append(plen - d2); names.append(3)
    stack = np.stack(cand)
    best_i = np.argmin(stack, axis=0)                 # first minimum = lowest variant
    best = stack[best_i, np.arange(stack.shape[1])]
    idx = np.nonzero(best <= max_err)[0]
    return [(int(p), int(best[p]), int(names[best_i[p]])) for p in idx]


def bits_columns(b, width: int, offset: int = 0):
    """(col_ones uint32[width], row_ones uint32[rows], col_parity, row_parity)."""
    b = _bits(b)
    rows = (b.size - offset) // width if offset < b.size else 0
    r = b[offset:offset + rows * width].reshape(rows, width).astype(np.uint32)
    col = r.sum(axis=0, dtype=np.uint32) if rows else np.zeros(width, np.uint32)
    row = r.sum(axis=1, dtype=np.uint32)
    return col, row, (col & 1).astype(np.uint8), (row & 1).astype(np.uint8)


def _width_mu(R):
    h = float(R // 2)
    if h < 1.0:
        return 1.0
    return _m.exp(_m.lgamma(2.0 * h + 1.0) - 2.0 * _m.lgamma(h + 1.0) - 2.0 * h * _LN2)


def bits_width_scan(b, w_lo: int, w_hi: int):
    """(z, raw) float32 for widths w_lo..w_hi — see atkdsp.h §14."""
    b = _bits(b)
    n = b.size
    z = np.full(w_hi - w_lo + 1, np.nan, np.float32)
    raw = np.full(w_hi - w_lo + 1, np.nan, np.float32)
    for i, w in enumerate(range(w_lo, w_hi + 1)):
        R = n // w
        if R < 2:
            continue
        cnt = b[:R * w].reshape(R, w).sum(axis=0, dtype=np.int64)
        S = int(np.abs(2 * cnt - R).sum())
        rv = S / (float(w) * float(R))
        mu = _width_mu(R)
        var = 1.0 / R - mu * mu
        raw[i] = np.float32(rv)
        if var > 0.0:
            z[i] = np.float32((rv - mu) * _m.sqrt(float(w)) / _m.sqrt(var))
    return z, raw


def bits_sync_discover(b, word_len: int = 16, min_count: int = 3, pfa: float = 1e-3,
                       cap: int = 64) -> list:
    """Recurring words nobody listed — see atkdsp.h §14. Returns dicts, best first."""
    b = _bits(b)
    n = b.size
    L = int(word_len)
    if L < 8 or L > 32:
        raise ValueError("word_len 8..32")
    if min_count < 2:
        min_count = 3
    if not pfa > 0.0:
        pfa = 1e-3
    if n < 2 * L:
        return []
    log10_thr = _m.log10(pfa)
    keys = _words_sliding(b, L)
    pos = np.arange(keys.size, dtype=np.uint64)
    comp = np.sort((keys.astype(np.uint64) << np.uint64(32)) | pos)
    ks = (comp >> np.uint64(32)).astype(np.int64)
    ps = (comp & np.uint64(0xffffffff)).astype(np.int64)
    bounds = np.flatnonzero(np.diff(ks)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [ks.size]])
    nodes = []
    for g0, g1 in zip(starts, ends):
        if g1 - g0 < min_count:
            continue
        kept = []
        for p in ps[g0:g1]:
            if not kept or p - kept[-1] >= L:
                kept.append(int(p))
        if len(kept) < min_count:
            continue
        diffs = sorted(kept[i + 1] - kept[i] for i in range(len(kept) - 1))
        P, best = diffs[0], 0
        i = 0
        while i < len(diffs):
            j = i
            while j < len(diffs) and diffs[j] == diffs[i]:
                j += 1
            if j - i > best:
                best, P = j - i, diffs[i]
            i = j
        res = sorted(k % P for k in kept)
        bestph, bestc = 0, 0
        i = 0
        while i < len(res):
            j = i
            while j < len(res) and res[j] == res[i]:
                j += 1
            if j - i > bestc:
                bestc, bestph = j - i, res[i]
            i = j
        if bestc < min_count:
            continue
        first = next(k for k in kept if k % P == bestph)
        nodes.append({"key": int(ks[g0]), "period": P, "phase": bestph, "count": bestc,
                      "first": first, "succ": -1, "pred": -1, "used": False})
    if not nodes:
        return []
    nodes.sort(key=lambda d: (d["period"], d["phase"], d["key"]))
    lowmask = (1 << (L - 1)) - 1
    index = {}
    for i, d in enumerate(nodes):
        index.setdefault((d["period"], d["phase"]), []).append(i)
    for i, d in enumerate(nodes):
        P = d["period"]
        ph = (d["phase"] + 1) % P
        want = d["key"] & lowmask
        best = -1
        for j in index.get((P, ph), []):
            if j == i or (nodes[j]["key"] >> 1) != want:
                continue
            a, c = d["count"], nodes[j]["count"]
            mn, mx = min(a, c), max(a, c)
            if mn != mx and not (float(mn) > float(mx) / 2.0 + 1.5 * _m.sqrt(float(mx))):
                continue
            if best < 0 or nodes[j]["count"] > nodes[best]["count"]:
                best = j
        d["succ"] = best
    for i, d in enumerate(nodes):
        s = d["succ"]
        if s < 0:
            continue
        p = nodes[s]["pred"]
        if p < 0:
            nodes[s]["pred"] = i
        elif d["count"] > nodes[p]["count"]:
            nodes[p]["succ"] = -1
            nodes[s]["pred"] = i
        else:
            d["succ"] = -1
    log10_T = L * _m.log10(2.0) + _m.log10(n / L)
    out = []
    for pass_ in range(2):
        for s, d in enumerate(nodes):
            if d["used"] or (pass_ == 0 and d["pred"] >= 0):
                continue
            chain = [(d["key"] >> (L - 1 - j)) & 1 for j in range(L)]
            d["used"] = True
            cur = d["succ"]
            while cur >= 0 and not nodes[cur]["used"] and len(chain) < SYNC_MAXBITS:
                chain.append(nodes[cur]["key"] & 1)
                nodes[cur]["used"] = True
                cur = nodes[cur]["succ"]
            P, ph = d["period"], d["phase"]
            chain = np.array(chain, np.uint8)
            lo, hi = 0, chain.size
            sc = _sync_score(b, chain[lo:hi], (ph + lo) % P, P, log10_T)
            while hi - lo > L:
                a = _sync_score(b, chain[lo + 1:hi], (ph + lo + 1) % P, P, log10_T)
                z = _sync_score(b, chain[lo:hi - 1], (ph + lo) % P, P, log10_T)
                if z[3] <= a[3] and z[3] < sc[3]:
                    hi -= 1
                    sc = z
                elif a[3] < sc[3]:
                    lo += 1
                    sc = a
                else:
                    break
            first, count, expected, l10 = sc
            if count < min_count or l10 > log10_thr:
                continue
            pa = chain[lo:hi].copy()
            ln = pa.size
            flags = 0
            for p0 in range(1, ln // 2 + 1):
                if np.array_equal(pa[:ln - p0], pa[p0:]):
                    flags |= SYNC_REPETITIVE
                    break
            out.append({"first": first, "period": P, "length": ln, "count": count,
                        "flags": flags, "expected": expected,
                        "log10_pfa": l10, "pattern": pa})
    out.sort(key=lambda c: (c["log10_pfa"], -c["count"], c["first"], c["period"]))
    return out[:cap]


def _sync_score(b, pat, ph, P, log10_T):
    """(first, count, expected, log10_pfa) of pat's exact matches at ph + kP."""
    ln = pat.size
    n = b.size
    first, count, slots = -1, 0, 0.0
    pos = ph
    while pos + ln <= n:
        slots += 1.0
        if np.array_equal(b[pos:pos + ln], pat):
            if first < 0:
                first = pos
            count += 1
        pos += P
    ln_q = -ln * _LN2
    return (first, count, slots * _m.exp(ln_q),
            log10_binom_tail(slots, float(count), ln_q) + log10_T + _m.log10(float(P)))


def _words_sliding(b, L):
    """Every L-bit window's value, MSB-first."""
    win = np.lib.stride_tricks.sliding_window_view(b.astype(np.int64), L)
    return win @ (1 << np.arange(L - 1, -1, -1, dtype=np.int64))


_MAURER_E = (0, 0.7326495, 1.5374383, 2.4016068, 3.3112247, 4.2534266, 5.2177052,
             6.1962507, 7.1836656, 8.1764248, 9.1723243, 10.170032, 11.168765,
             12.168070, 13.167693, 14.167488, 15.167379)
_MAURER_V = (0, 0.690, 1.338, 1.901, 2.358, 2.705, 2.954, 3.125, 3.238, 3.311, 3.356,
             3.384, 3.401, 3.410, 3.416, 3.419, 3.421)


def _apen_phi(b, m):
    n = b.size
    ext = np.concatenate([b, b[:m - 1]]).astype(np.int64)
    v = np.lib.stride_tricks.sliding_window_view(ext, m)[:n] @ (1 << np.arange(m - 1, -1, -1))
    cnt = np.bincount(v, minlength=1 << m)
    phi = 0.0
    for c in cnt:
        if c:
            p = c / n
            phi += p * _m.log(p)
    return phi


def bits_stats(b) -> dict:
    """The twin of atkdsp_bits_stats (atkdsp.h §14)."""
    b = _bits(b)
    n = b.size
    o = {"n": n, "ones": int(b.sum())}
    o["density"] = o["ones"] / n if n else float("nan")
    run0 = np.zeros(33, np.uint64)
    run1 = np.zeros(33, np.uint64)
    if n:
        edges = np.flatnonzero(np.diff(b)) + 1
        st = np.concatenate([[0], edges])
        en = np.concatenate([edges, [n]])
        for s, e in zip(st, en):
            k = min(32, int(e - s))
            (run1 if b[s] else run0)[k] += 1
    o["run0"], o["run1"] = run0, run1
    ng = np.zeros((8, 256), np.uint64)
    for k in range(1, 9):
        if n < k:
            break
        v = _words_sliding(b, k)
        ng[k - 1, :1 << k] = np.bincount(v, minlength=1 << k)
    o["ngram"] = ng
    w = chi_width(n)
    o["chi2_bits"] = w
    if w:
        words = _words(b, w)
        cells = 1 << w
        cnt = np.bincount(words, minlength=cells)
        e = words.size / cells
        x2 = 0.0
        for c in cnt:
            d = float(c) - e
            x2 += d * d / e
        o["chi2"] = x2
        o["chi2_p"] = igamc((cells - 1) / 2.0, x2 / 2.0)
    else:
        o["chi2"] = o["chi2_p"] = float("nan")
    L = 0
    for l in range(16, 0, -1):
        if n >= 1010.0 * l * (1 << l):
            L = l
            break
    o["maurer_L"] = L
    if L:
        Q = 10 << L
        K = n // L - Q
        vals = _words(b, L)[:Q + K]
        T = np.zeros(1 << L, np.int64)
        s = 0.0
        for i in range(1, Q + K + 1):
            v = int(vals[i - 1])
            if i > Q:
                s += _m.log(float(i - T[v])) / _LN2
            T[v] = i
        fn = s / K
        c = 0.7 - 0.8 / L + (4.0 + 32.0 / L) * _m.pow(float(K), -3.0 / L) / 15.0
        sigma = c * _m.sqrt(_MAURER_V[L] / K)
        o["maurer"] = fn
        o["maurer_p"] = _m.erfc(abs(fn - _MAURER_E[L]) / (_m.sqrt(2.0) * sigma))
    else:
        o["maurer"] = o["maurer_p"] = float("nan")
    if n >= 128:
        lg = n.bit_length() - 1
        mm = min(10, lg - 6)
        o["apen_m"] = mm
        ap = _apen_phi(b, mm) - _apen_phi(b, mm + 1)
        o["apen"] = ap
        o["apen_p"] = igamc(float(1 << (mm - 1)), 2.0 * n * (_LN2 - ap) / 2.0)
    else:
        o["apen_m"] = 0
        o["apen"] = o["apen_p"] = float("nan")
    be = np.full(16, np.nan)
    for w in range(1, 17):
        cells = 1 << w
        nw = n // w
        if nw < cells:
            continue
        cnt = np.bincount(_words(b, w), minlength=cells)
        h = 0.0
        for c in cnt:
            if c:
                p = c / nw
                h -= p * _m.log(p)
        be[w - 1] = h / _LN2 / w
    o["block_entropy"] = be
    return o


def _fwht_int(x):
    x = np.array(x, dtype=np.int64)
    h = 1
    while h < x.size:
        y = x.reshape(-1, 2 * h)
        a = y[:, :h].copy()
        bb = y[:, h:].copy()
        y[:, :h] = a + bb
        y[:, h:] = a - bb
        x = y.reshape(-1)
        h *= 2
    return x


def bits_wht(b, width: int, cols) -> tuple:
    """(spectrum int32[2^ncols], rows) — see atkdsp.h §14."""
    b = _bits(b)
    cols = list(cols)
    rows = b.size // width
    r = b[:rows * width].reshape(rows, width).astype(np.int64)
    v = np.zeros(rows, np.int64)
    for j, c in enumerate(cols):
        v |= r[:, c] << j
    h = np.bincount(v, minlength=1 << len(cols))
    return _fwht_int(h).astype(np.int32), rows


def _sequency_index(size):
    lg = size.bit_length() - 1
    out = np.empty(size, np.int64)
    for s in range(size):
        g = s ^ (s >> 1)
        h = 0
        for k in range(lg):
            h |= ((g >> k) & 1) << (lg - 1 - k)
        out[s] = h
    return out


def bits_walsh(b, size: int, navg: int = 0):
    """(sequency_power, dyadic_acf, blocks) — see atkdsp.h §14."""
    b = _bits(b)
    blocks = b.size // size
    if navg > 0:
        blocks = min(blocks, navg)
    lim = 4.0e18 / (float(size) ** 3)
    if blocks > lim:
        blocks = int(lim)
    if blocks == 0:
        nanv = np.full(size, np.nan, np.float32)
        return nanv, nanv.copy(), 0
    P = np.zeros(size, np.int64)
    for k in range(blocks):
        x = 1 - 2 * b[k * size:(k + 1) * size].astype(np.int64)
        X = _fwht_int(x)
        P += X * X
    norm = float(size) * blocks
    seq = (P[_sequency_index(size)] / norm).astype(np.float32)
    R = _fwht_int(P)
    dy = (R / (float(size) * float(size) * blocks)).astype(np.float32)
    return seq, dy, blocks


def bits_cyclic(b, size: int, navg: int = 0):
    """(cyclic_time, cyclic_freq, blocks) — see atkdsp.h §14."""
    b = _bits(b)
    blocks = b.size // size
    if navg > 0:
        blocks = min(blocks, navg)
    nf = size // 2 + 1
    if blocks == 0:
        return np.full(size, np.nan, np.float32), np.full(nf, np.nan, np.float32), 0
    psum = np.zeros(nf)
    rsum = np.zeros(size, np.int64)
    for k in range(blocks):
        x = 1.0 - 2.0 * b[k * size:(k + 1) * size]
        X = np.fft.rfft(x)
        pw = X.real ** 2 + X.imag ** 2
        psum += pw
        rsum += np.rint(np.fft.irfft(pw, n=size)).astype(np.int64)
    norm = float(size) * blocks
    return (rsum / norm).astype(np.float32), (psum / norm).astype(np.float32), blocks


# ---- §15 scramblers and linear complexity -------------------------------------
SCRAM_ADD = 0
SCRAM_MULT = 1


def bm(b) -> tuple:
    """(L, poly uint8[L+1]) — the shortest LFSR (Berlekamp-Massey over GF(2))."""
    L, C, _ = _bm_run(_bits(b))
    return L, np.array([(C >> i) & 1 for i in range(L + 1)], np.uint8)


def bm_profile(b) -> np.ndarray:
    return _bm_run(_bits(b))[2]


def _bm_run(s):
    n = s.size
    C, B, L, m = 1, 1, 0, 1
    prof = np.zeros(n, np.int32)
    # the stream as an int, bit i = s[i]
    S = int.from_bytes(np.packbits(s[::-1]).tobytes(), "big") >> ((-n) % 8) if n else 0
    for N in range(n):
        # d = XOR_{i=0..L} c_i s[N-i]
        lo = N - L
        win = (S >> lo) & ((1 << (L + 1)) - 1)          # bits s[N-L..N]
        rev = int(format(win, f"0{L + 1}b")[::-1], 2)   # bit i = s[N-i]
        d = bin(C & rev).count("1") & 1
        if d:
            if 2 * L <= N:
                T = C
                C ^= B << m
                L = N + 1 - L
                B = T
                m = 1
            else:
                C ^= B << m
                m += 1
        else:
            m += 1
        prof[N] = L
    return L, C, prof


def bm_mask(poly) -> int:
    """A Berlekamp-Massey connection polynomial c_0..c_L -> the siga.bits mask
    (bit i = c_{L-i}): the register r[t+L] = XOR_{i<L, bit i} r[t+i]."""
    poly = list(np.asarray(poly, np.uint8))
    L = len(poly) - 1
    return sum(int(poly[j]) << (L - j) for j in range(L + 1))


def _mask_parts(poly):
    poly = int(poly)
    D = poly.bit_length() - 1
    if D < 2 or D > 63 or not poly & 1:
        raise ValueError("scrambler mask: degree 2..63 with bit 0 set")
    return D, poly & ((1 << D) - 1)


def _par(x):
    return bin(x).count("1") & 1


def descramble_add(b, poly, state) -> np.ndarray:
    """Additive descramble; `state` (a 1-element uint64 array) is the register,
    updated in place. See atkdsp.h §15."""
    b = _bits(b)
    D, low = _mask_parts(poly)
    s = int(state[0]) & ((1 << D) - 1)
    if not s:
        raise ValueError("descramble_add: zero register")
    out = np.empty_like(b)
    for t in range(b.size):
        out[t] = b[t] ^ (s & 1)
        s = (s >> 1) | (_par(s & low) << (D - 1))
    state[0] = s
    return out


def descramble_mult(b, poly, state) -> np.ndarray:
    b = _bits(b)
    D, low = _mask_parts(poly)
    s = int(state[0]) & ((1 << D) - 1)
    out = np.empty_like(b)
    for t in range(b.size):
        x = int(b[t])
        out[t] = x ^ _par(s & low)
        s = (s >> 1) | (x << (D - 1))
    state[0] = s
    return out


def scramble_mult(b, poly, state) -> np.ndarray:
    b = _bits(b)
    D, low = _mask_parts(poly)
    s = int(state[0]) & ((1 << D) - 1)
    out = np.empty_like(b)
    for t in range(b.size):
        x = int(b[t]) ^ _par(s & low)
        out[t] = x
        s = (s >> 1) | (x << (D - 1))
    state[0] = s
    return out


def _enum_masks(maxD, max_terms):
    out = []
    for D in range(2, maxD + 1):
        ends = (1 << D) | 1
        for a in range(1, D):
            out.append((ends | (1 << a), D, 3))
        if max_terms >= 5:
            for a in range(1, D):
                for bb in range(a + 1, D):
                    for cc in range(bb + 1, D):
                        out.append((ends | (1 << a) | (1 << bb) | (1 << cc), D, 5))
    return out


def scrambler_search(b, kind: int = SCRAM_MULT, max_degree: int = 24, max_terms: int = 5,
                     pfa: float = 1e-3, cap: int = 16):
    """(candidates, g_identity) — see atkdsp.h §15."""
    b = _bits(b)
    n = b.size
    if kind not in (SCRAM_ADD, SCRAM_MULT) or not 2 <= max_degree <= 32 or max_terms not in (3, 5):
        raise ValueError("scrambler_search: kind ADD/MULT, max_degree 2..32, max_terms 3 or 5")
    if not pfa > 0.0:
        pfa = 1e-3
    t0 = max_degree
    if n <= t0:
        return [], float("nan")
    m = n - t0
    w = chi_width(m)
    if not w:
        return [], float("nan")
    g_id = gstat(b[t0:], w)
    cands = []
    for poly, D, terms in _enum_masks(max_degree, max_terms):
        low = poly & ((1 << D) - 1)
        y = b[t0:].copy()
        for i in range(D):
            if (low >> i) & 1:
                dly = D - i
                y ^= b[t0 - dly:n - dly]
        cands.append((gstat(y, w), D, terms, poly))
    cands.sort(key=lambda c: (-c[0], c[1], c[2], c[3]))
    dof = float((1 << w) - 1)
    trials = len(cands) + 1.0
    log10_thr = _m.log10(pfa)
    margin = norm_isf(pfa / trials) * _m.sqrt(4.0 * dof)
    out = []
    for g, D, terms, poly in cands:
        if len(out) >= cap:
            break
        l10 = log_igamc(dof / 2.0, g / 2.0) / _LN10 + _m.log10(trials)
        if l10 > log10_thr or not (g - g_id > margin):
            break
        c = {"poly": poly, "seed": 0, "g": g, "log10_pfa": l10, "g_data": float("nan"),
             "degree": D, "terms": terms, "seeded": 0, "block_bits": w}
        out.append(c)
        if kind != SCRAM_ADD:
            continue
        low = poly & ((1 << D) - 1)
        y = b[D:].copy()
        for i in range(D):
            if (low >> i) & 1:
                y ^= b[i:n - D + i]
        # longest constant run of y (first longest), as positions in b
        best_start, best_len, best_val = 0, 0, 0
        run_val, run_start, run_len = -1, D, 0
        for idx in range(y.size):
            yv = int(y[idx])
            if yv == run_val:
                run_len += 1
            else:
                run_val, run_start, run_len = yv, D + idx, 1
            if run_len > best_len:
                best_len, best_start, best_val = run_len, run_start, yv
        if best_len < 2 * D:
            continue
        ts = best_start - D
        s = 0
        for k in range(D):
            s |= (int(b[ts + k]) ^ best_val) << k
        smask = (1 << D) - 1
        lowsh = low >> 1
        for _t in range(ts, 0, -1):
            prev = ((s >> (D - 1)) & 1) ^ _par(s & lowsh)
            s = ((s << 1) | prev) & smask
        if not s:
            continue
        st = np.array([s], np.uint64)
        d = descramble_add(b, poly, st)
        c["seed"], c["seeded"], c["g_data"] = s, 1, gstat(d[t0:], w)
    return out, g_id


# ---- §16a convolutional codes ---------------------------------------------------
CONV_TERM = 0
CONV_TAILBITE = 1
CONV_STREAM = 2


def _punc(n, puncture):
    if puncture is None:
        return np.ones((n, 1), np.uint8)
    p = np.asarray(puncture, np.uint8)
    return (p.reshape(n, -1) != 0).astype(np.uint8)


def conv_encode(K: int, polys, bits, mode: int = CONV_TERM, puncture=None) -> np.ndarray:
    """The encoder of atkdsp.h §16a (MSB of each generator = the current bit)."""
    polys = [int(p) for p in polys]
    n = len(polys)
    bits = _bits(bits)
    N = bits.size
    keep = _punc(n, puncture)
    P = keep.shape[1]
    s = 0
    if mode == CONV_TAILBITE:
        for j in range(K - 1):
            idx = (N - (K - 1) + j) % N
            s = (int(bits[idx]) << (K - 2)) | (s >> 1)
    steps = N + (K - 1 if mode == CONV_TERM else 0)
    out = []
    for t in range(steps):
        u = int(bits[t]) if t < N else 0
        reg = (u << (K - 1)) | s
        for i in range(n):
            if keep[i, t % P]:
                out.append(_par(reg & polys[i]))
        s = reg >> 1
    return np.array(out, np.uint8)


class Viterbi:
    """The twin of atkdsp_viterbi (atkdsp.h §16a), float32 metrics, the same
    decisions bit for bit."""

    def __init__(self, K: int, polys, puncture=None, mode: int = CONV_TERM, depth: int = 0,
                 max_bits: int = 4096):
        self.K = int(K)
        self.polys = [int(p) for p in polys]
        self.n = len(self.polys)
        self.S = 1 << (self.K - 1)
        self.mode = int(mode)
        self.keep = _punc(self.n, puncture)
        self.P = self.keep.shape[1]
        self.kp = self.keep.sum(axis=0).astype(int)
        if (self.kp == 0).any():
            raise ValueError("every input bit must send something")
        self.punctured = puncture is not None and not self.keep.all()
        if depth == 0 and self.mode == CONV_STREAM:
            depth = (12 if puncture is not None else 6) * self.K
        self.depth = int(depth)
        self.max_bits = int(max_bits)
        ow = np.zeros(2 * self.S, np.int64)
        for p in range(self.S):
            for u in range(2):
                reg = (u << (self.K - 1)) | p
                ow[2 * p + u] = sum(_par(reg & self.polys[i]) << i for i in range(self.n))
        self.ow = ow
        half = self.S // 2
        j = np.arange(half)
        self._p0 = 2 * j
        self._p1 = 2 * j + 1
        self.reset()

    def reset(self):
        self.pm = np.zeros(self.S, np.float32)
        self.steps_done = 0
        self.released = 0
        self.pend = []
        self.dec = []

    def _acs(self, x):
        ncw = 1 << self.n
        met = np.zeros(ncw, np.float32)
        for cw in range(ncw):
            mm = np.float32(0.0)
            for i in range(self.n):
                mm = (mm - x[i]) if (cw >> i) & 1 else (mm + x[i])
            met[cw] = mm
        half = self.S // 2
        pm = self.pm
        nm = np.empty(self.S, np.float32)
        d = np.zeros(self.S, np.uint8)
        for u in (0, 1):
            m0 = pm[self._p0] + met[self.ow[2 * self._p0 + u]]
            m1 = pm[self._p1] + met[self.ow[2 * self._p1 + u]]
            take0 = m0 >= m1
            nm[u * half:(u + 1) * half] = np.where(take0, m0, m1)
            d[u * half:(u + 1) * half] = (~take0).astype(np.uint8)
        nm = (nm - nm.max()).astype(np.float32)
        self.pm = nm
        return d

    def _trace(self, decs, s):
        top = self.K - 2
        out = np.empty(len(decs), np.uint8)
        for t in range(len(decs) - 1, -1, -1):
            out[t] = (s >> top) & 1
            s = ((s << 1) & (self.S - 1)) | int(decs[t][s])
        return out

    def _steps(self, nllr):
        full = nllr // int(self.kp.sum())
        steps = full * self.P
        rem = nllr - full * int(self.kp.sum())
        while rem > 0:
            k = int(self.kp[steps % self.P])
            if k > rem:
                return -1
            rem -= k
            steps += 1
        return steps

    def decode(self, llr):
        llr = np.ascontiguousarray(llr, np.float32)
        if self.mode == CONV_STREAM:
            self.metric = float("nan")
            return self._stream(llr)
        steps = self._steps(llr.size)
        if steps < 0:
            raise ValueError("no step count produces that many LLRs")
        term = self.mode == CONV_TERM
        nbits = steps - (self.K - 1) if term else steps
        if nbits < 1 or nbits > self.max_bits:
            raise ValueError("block size out of range")
        off = np.concatenate([[0], np.cumsum([self.kp[t % self.P] for t in range(steps)])])

        def gather(t):
            j = t % self.P
            o = int(off[t])
            x = np.zeros(self.n, np.float32)
            for i in range(self.n):
                if self.keep[i, j]:
                    x[i] = llr[o]
                    o += 1
            return x
        if term:
            self.pm = np.full(self.S, -np.inf, np.float32)
            self.pm[0] = 0.0
            decs = [self._acs(gather(t)) for t in range(steps)]
            bits = self._trace(decs, 0)[:nbits]
        else:
            W = self.depth if (self.depth and self.depth < steps) else steps
            self.pm = np.zeros(self.S, np.float32)
            decs = [self._acs(gather((e + steps - W) % steps)) for e in range(W + steps + W)]
            bits = self._trace(decs, int(np.argmax(self.pm)))[W:W + nbits]
        code = conv_encode(self.K, self.polys, bits, self.mode,
                           self.keep if self.punctured else None)
        num = float(np.sum(np.where(code == 1, -llr.astype(np.float64), llr.astype(np.float64))))
        den = float(np.sum(np.abs(llr.astype(np.float64))))
        self.metric = float(np.float32(num / den)) if den > 0 else float("nan")
        self.pm = np.zeros(self.S, np.float32)
        return bits

    def _stream(self, llr):
        out = []
        k = 0
        D = self.depth
        while True:
            j = self.steps_done % self.P
            while len(self.pend) < self.n:
                i = len(self.pend)
                if self.keep[i, j]:
                    if k >= llr.size:
                        break
                    self.pend.append(np.float32(llr[k]))
                    k += 1
                else:
                    self.pend.append(np.float32(0.0))
            if len(self.pend) < self.n:
                break
            self.dec.append(self._acs(np.array(self.pend, np.float32)))
            self.steps_done += 1
            self.pend = []
            if self.steps_done >= self.released + 2 * D:
                bits = self._trace(self.dec[self.released:self.steps_done], int(np.argmax(self.pm)))
                out.extend(bits[:D])
                self.released += D
        return np.array(out, np.uint8)

    def flush(self):
        if self.mode != CONV_STREAM:
            return np.zeros(0, np.uint8)
        left = self.steps_done - self.released
        self.pend = []
        if not left:
            return np.zeros(0, np.uint8)
        bits = self._trace(self.dec[self.released:self.steps_done], int(np.argmax(self.pm)))
        self.released = self.steps_done
        return bits


# ---- §16b/c Reed-Solomon and BCH over GF(2^m) -------------------------------------
GF_DEFAULT = {3: 0xB, 4: 0x13, 5: 0x25, 6: 0x43, 7: 0x89, 8: 0x11D, 9: 0x211, 10: 0x409,
              11: 0x805, 12: 0x1053, 13: 0x201B, 14: 0x4443, 15: 0x8003, 16: 0x1100B}
CCSDS_TAL = (0x8d, 0xef, 0xec, 0x86, 0xfa, 0x99, 0xaf, 0x7b)
E_DECODE = -5


class _GF:
    def __init__(self, m, poly):
        if not 3 <= m <= 16 or (poly >> m) != 1:
            raise ValueError("field: m 3..16, polynomial of degree m")
        self.m, self.nn, self.poly = m, (1 << m) - 1, poly
        exp = [0] * (2 * self.nn)
        log = [-1] * (self.nn + 1)
        x = 1
        for i in range(self.nn):
            if log[x] != -1:
                raise ValueError("polynomial not primitive")
            exp[i] = x
            log[x] = i
            x <<= 1
            if x >> m:
                x ^= poly
        if x != 1:
            raise ValueError("polynomial not primitive")
        for i in range(self.nn, 2 * self.nn):
            exp[i] = exp[i - self.nn]
        self.exp, self.log = exp, log

    def mul(self, a, b):
        return 0 if not a or not b else self.exp[self.log[a] + self.log[b]]

    def inv(self, a):
        return self.exp[(self.nn - self.log[a]) % self.nn]

    def pw(self, e):
        return self.exp[e % self.nn]

    def eval_pos(self, sym, beta):
        acc = 0
        for v in sym:
            acc = self.mul(acc, beta) ^ int(v)
        return acc

    def bm(self, S, lam, L0):
        nsyn = len(S)
        cap = nsyn + 1
        B = list(lam)
        L, msh, bd = L0, 1, 1
        for r in range(L0, nsyn):
            d = 0
            for i in range(0, min(L, r) + 1):
                d ^= self.mul(lam[i], S[r - i])
            if not d:
                msh += 1
                continue
            coef = self.mul(d, self.inv(bd))
            if 2 * L <= r + L0:
                T = list(lam)
                for i in range(cap - msh):
                    lam[i + msh] ^= self.mul(coef, B[i])
                L = r + 1 + L0 - L
                B = T
                bd, msh = d, 1
            else:
                for i in range(cap - msh):
                    lam[i + msh] ^= self.mul(coef, B[i])
                msh += 1
        deg = max([i for i in range(cap) if lam[i]] or [0])
        return L if deg == L else -1


class RS:
    """The twin of atkdsp_rs (atkdsp.h §16b)."""

    def __init__(self, m, gfpoly, fcr, prim, nroots, pad=0, dual_basis=False):
        self.gf = _GF(m, gfpoly)
        nn = self.gf.nn
        if nroots < 1 or pad < 0 or nroots >= nn - pad or not 0 <= fcr < nn or not 1 <= prim < nn:
            raise ValueError("RS parameters out of range")
        if _m.gcd(prim, nn) != 1:
            raise ValueError("prim must be coprime to 2^m - 1")
        if dual_basis and not (m == 8 and gfpoly == 0x187):
            raise ValueError("dual basis is CCSDS: m = 8, gfpoly 0x187")
        self.m, self.fcr, self.prim, self.nroots, self.pad = m, fcr, prim, nroots, pad
        self.n = nn - pad
        self.k = self.n - nroots
        self.dual = bool(dual_basis)
        g = [1] + [0] * nroots
        for i in range(nroots):
            root = self.gf.pw(prim * (fcr + i))
            for d in range(i + 1, 0, -1):
                g[d] = g[d - 1] ^ self.gf.mul(g[d], root)
            g[0] = self.gf.mul(g[0], root)
        self.gen = g
        if self.dual:
            tal = [0] * 256
            for x in range(256):
                v = 0
                for j in range(8):
                    if (x >> j) & 1:
                        v ^= CCSDS_TAL[7 - j]
                tal[x] = v
            tal1 = [0] * 256
            for x in range(256):
                tal1[tal[x]] = x
            self.tal, self.tal1 = tal, tal1

    def _enc(self, data):
        nr = self.nroots
        reg = [0] * nr
        for d in data:
            fb = int(d) ^ reg[0]
            for i in range(nr - 1):
                reg[i] = reg[i + 1] ^ self.gf.mul(fb, self.gen[nr - 1 - i])
            reg[nr - 1] = self.gf.mul(fb, self.gen[0])
        return reg

    def encode(self, data) -> np.ndarray:
        data = [int(v) for v in data]
        if self.dual:
            par = self._enc([self.tal1[v] for v in data])
            par = [self.tal[v] for v in par]
        else:
            par = self._enc(data)
        return np.array(par, np.uint16)

    def decode(self, sym, erasures=()):
        """(corrected uint16 array, count, positions); count = E_DECODE on failure."""
        gf = self.gf
        n, nr = self.n, self.nroots
        sym = [int(v) for v in sym]
        erasures = [int(e) for e in erasures]
        if len(erasures) > nr:
            return np.array(sym, np.uint16), E_DECODE, []
        w = [self.tal1[v] for v in sym] if self.dual else list(sym)
        S = [gf.eval_pos(w, gf.pw(self.prim * (self.fcr + i))) for i in range(nr)]
        if not any(S) and not erasures:
            return np.array(sym, np.uint16), 0, []
        lam = [1] + [0] * nr
        for e_i, e in enumerate(erasures):
            X = gf.pw(self.prim * (n - 1 - e))
            for d in range(e_i + 1, 0, -1):
                lam[d] ^= gf.mul(lam[d - 1], X)
        L = gf.bm(S, lam, len(erasures))
        fail = (np.array(sym, np.uint16), E_DECODE, [])
        if L < 0 or 2 * L - len(erasures) > nr:
            return fail
        loc = []
        for j in range(n):
            deg = n - 1 - j
            v = 0
            for i in range(L + 1):
                if lam[i]:
                    v ^= gf.mul(lam[i], gf.pw(-self.prim * deg * i))
            if not v:
                if len(loc) >= nr:
                    return fail
                loc.append(j)
        if len(loc) != L:
            return fail
        om = []
        for i in range(nr):
            a = 0
            for k in range(0, min(i, L) + 1):
                a ^= gf.mul(lam[k], S[i - k])
            om.append(a)
        for j in loc:
            xi = self.prim * (n - 1 - j)
            num = 0
            for i in range(nr):
                if om[i]:
                    num ^= gf.mul(om[i], gf.pw(-xi * i))
            den = 0
            for i in range(1, L + 1, 2):
                if lam[i]:
                    den ^= gf.mul(lam[i], gf.pw(-xi * (i - 1)))
            if not den:
                return fail
            y = gf.mul(gf.mul(num, gf.inv(den)), gf.pw(xi * (1 - self.fcr)))
            w[j] ^= y
        for i in range(nr):
            if gf.eval_pos(w, gf.pw(self.prim * (self.fcr + i))):
                return fail
        outv = list(sym)
        for j in loc:
            outv[j] = self.tal[w[j]] if self.dual else w[j]
        return np.array(outv, np.uint16), len(loc), loc


class BCH:
    """The twin of atkdsp_bch (atkdsp.h §16c)."""

    def __init__(self, m, t, shorten=0, gfpoly=0):
        if not gfpoly:
            gfpoly = GF_DEFAULT[m]
        self.gf = _GF(m, gfpoly)
        nn = self.gf.nn
        if t < 1 or 2 * t >= nn or not 0 <= shorten < nn:
            raise ValueError("BCH parameters out of range")
        used = [False] * nn
        g = [1]
        for i in range(1, 2 * t + 1):
            if used[i % nn]:
                continue
            mp = [1]
            e = i % nn
            while True:
                used[e] = True
                root = self.gf.exp[e]
                mp = [0] + mp
                for d in range(len(mp) - 1):
                    mp[d] ^= self.gf.mul(mp[d + 1], root)
                e = (e * 2) % nn
                if e == i % nn or len(mp) - 1 > m:
                    break
            if any(c > 1 for c in mp):
                raise ValueError("minimal polynomial not binary")
            prod = [0] * (len(g) + len(mp) - 1)
            for a, ca in enumerate(mp):
                if ca:
                    for bi, cb in enumerate(g):
                        prod[a + bi] ^= cb
            g = prod
        self.m, self.t, self.shorten = m, t, shorten
        self.n = nn - shorten
        self.k = self.n - (len(g) - 1)
        if self.k < 1:
            raise ValueError("BCH: no data bits left")
        self.gen = g                        # gen[d] = coefficient of x^d

    def generator(self) -> np.ndarray:
        return np.array(self.gen[::-1], np.uint8)

    def encode(self, data) -> np.ndarray:
        r = self.n - self.k
        reg = [0] * r
        data = _bits(data)
        for d in data:
            fb = int(d) ^ reg[0]
            for i in range(r - 1):
                reg[i] = reg[i + 1] ^ (fb & self.gen[r - 1 - i])
            reg[r - 1] = fb & self.gen[0]
        return np.concatenate([data, np.array(reg, np.uint8)])

    def decode(self, code):
        """(corrected uint8 array, count, positions); count = E_DECODE on failure."""
        gf = self.gf
        code = list(_bits(code))
        n, ns = self.n, 2 * self.t

        def syn():
            S = []
            for i in range(ns):
                beta = gf.exp[(i + 1) % gf.nn]
                acc = 0
                for v in code:
                    acc = gf.mul(acc, beta) ^ int(v)
                S.append(acc)
            return S
        S = syn()
        if not any(S):
            return np.array(code, np.uint8), 0, []
        lam = [1] + [0] * ns
        L = gf.bm(S, lam, 0)
        orig = np.array(code, np.uint8)
        if L < 1 or L > self.t:
            return orig, E_DECODE, []
        loc = []
        for j in range(n):
            deg = n - 1 - j
            v = 0
            for i in range(L + 1):
                if lam[i]:
                    v ^= gf.mul(lam[i], gf.pw(-deg * i))
            if not v:
                if len(loc) >= L:
                    return orig, E_DECODE, []
                loc.append(j)
        if len(loc) != L:
            return orig, E_DECODE, []
        for j in loc:
            code[j] ^= 1
        if any(syn()):
            return orig, E_DECODE, []
        return np.array(code, np.uint8), L, loc


# ---- §16d small block codes ----------------------------------------------------------
class BlockCode:
    """The twin of atkdsp_blk (atkdsp.h §16d): syndrome-table decoding."""

    def __init__(self, G):
        G = (np.asarray(G, np.uint8) & 1)
        k, n = G.shape
        if not (2 <= n <= 32 and 1 <= k < n and n - k <= 20):
            raise ValueError("block code: n <= 32, n - k <= 20")
        self.n, self.k, self.r = n, k, n - k
        rows = [sum(int(G[i, j]) << j for j in range(n)) for i in range(k)]
        self.G = list(rows)
        R = list(rows)
        M = [1 << i for i in range(k)]
        piv = []
        rank = 0
        for col in range(n):
            if rank >= k:
                break
            p = next((i for i in range(rank, k) if (R[i] >> col) & 1), -1)
            if p < 0:
                continue
            R[p], R[rank] = R[rank], R[p]
            M[p], M[rank] = M[rank], M[p]
            for i in range(k):
                if i != rank and (R[i] >> col) & 1:
                    R[i] ^= R[rank]
                    M[i] ^= M[rank]
            piv.append(col)
            rank += 1
        if rank < k:
            raise ValueError("generator rank < k")
        self.R, self.M, self.piv = R, M, piv
        H = []
        for col in range(n):
            if col in piv:
                continue
            h = 1 << col
            for i in range(k):
                if (R[i] >> col) & 1:
                    h |= 1 << piv[i]
            H.append(h)
        self.H = H
        ns = 1 << self.r
        leader = [0] * ns
        wt = [255] * ns
        wt[0] = 0
        filled = 1
        dmin = 0
        if k <= 20:
            cw, best = 0, n + 1
            for i in range(1, 1 << k):
                bit = (i & -i).bit_length() - 1
                cw ^= rows[bit]
                best = min(best, bin(cw).count("1"))
            dmin = best
        from itertools import combinations
        w = 1
        done, ncomb = 0.0, 1.0
        while w <= n and (filled < ns or dmin == 0):
            ncomb = ncomb * float(n - w + 1) / float(w)
            if done + ncomb > float(1 << 25):           # the C's work bound
                break
            done += ncomb
            for comb in combinations(range(n), w):
                x = 0
                for j in comb:
                    x |= 1 << j
                s = self._syn(x)
                if s == 0:
                    if not dmin:
                        dmin = w
                elif wt[s] == 255:
                    wt[s], leader[s] = w, x
                    filled += 1
                elif wt[s] == w and leader[s] != x:
                    leader[s] = -1
            w += 1
        self.leader, self.wt, self.dmin = leader, wt, dmin

    def _syn(self, v):
        s = 0
        for i, h in enumerate(self.H):
            s |= (bin(v & h).count("1") & 1) << i
        return s

    def encode(self, data) -> np.ndarray:
        v = 0
        for i, d in enumerate(_bits(data)):
            if d:
                v ^= self.G[i]
        return np.array([(v >> j) & 1 for j in range(self.n)], np.uint8)

    def decode(self, code):
        """(corrected code, data, count); count = E_DECODE when ambiguous."""
        code = _bits(code)
        v = sum(int(code[j]) << j for j in range(self.n))
        s = self._syn(v)
        if self.wt[s] == 255 or self.leader[s] == -1:
            return code.copy(), None, E_DECODE
        e = self.leader[s]
        v ^= e
        d = 0
        for i in range(self.k):
            if (v >> self.piv[i]) & 1:
                d ^= self.M[i]
        return (np.array([(v >> j) & 1 for j in range(self.n)], np.uint8),
                np.array([(d >> i) & 1 for i in range(self.k)], np.uint8), bin(e).count("1"))


def cyclic_generator_matrix(n: int, gpoly: int) -> np.ndarray:
    """k x n generator of the cyclic code with generator polynomial gpoly (bit d =
    coefficient of x^d): row i = x^i g(x), position j = coefficient of x^(n-1-j)."""
    r = gpoly.bit_length() - 1
    k = n - r
    G = np.zeros((k, n), np.uint8)
    for i in range(k):
        poly = gpoly << i
        for d in range(n):
            G[i, n - 1 - d] = (poly >> d) & 1
    return G


def extend_with_parity(G) -> np.ndarray:
    """Append an overall even-parity column (Hamming(8,4), (16,11), Golay(24,12))."""
    G = np.asarray(G, np.uint8)
    return np.concatenate([G, (G.sum(axis=1) & 1)[:, None].astype(np.uint8)], axis=1)


#: Small codes derived from their cyclic generator polynomials (facts that
#: rederive: Hamming from a primitive polynomial, Golay(23,12) from x^11 + x^10 +
#: x^6 + x^5 + x^4 + x^2 + 1). Protocol-specific matrices (DMR's (13,9), (16,11)
#: as ETSI writes them) belong with their decoders in ATK, checked against ETSI.
SMALL_CODES = {
    "hamming(7,4)": lambda: cyclic_generator_matrix(7, 0b1011),
    "hamming(15,11)": lambda: cyclic_generator_matrix(15, 0b10011),
    "hamming(31,26)": lambda: cyclic_generator_matrix(31, 0b100101),
    "hamming(8,4)": lambda: extend_with_parity(cyclic_generator_matrix(7, 0b1011)),
    "hamming(16,11)": lambda: extend_with_parity(cyclic_generator_matrix(15, 0b10011)),
    "golay(23,12)": lambda: cyclic_generator_matrix(23, 0xC75),
    "golay(24,12)": lambda: extend_with_parity(cyclic_generator_matrix(23, 0xC75)),
}


# ---- §17 the CRC engine and CRC recovery -------------------------------------------------
#: The catalogue, re-typed independently of src/crc.c from the reveng catalogue:
#: (name, width, poly, init, refin, refout, xorout, check). The tests hold the
#: two copies and every check value to each other.
CRC_CATALOG = [
    ("CRC-3/GSM", 3, 0x3, 0x0, 0, 0, 0x7, 0x4),
    ("CRC-3/ROHC", 3, 0x3, 0x7, 1, 1, 0x0, 0x6),
    ("CRC-4/G-704", 4, 0x3, 0x0, 1, 1, 0x0, 0x7),
    ("CRC-4/INTERLAKEN", 4, 0x3, 0xf, 0, 0, 0xf, 0xb),
    ("CRC-5/EPC-C1G2", 5, 0x09, 0x09, 0, 0, 0x00, 0x00),
    ("CRC-5/G-704", 5, 0x15, 0x00, 1, 1, 0x00, 0x07),
    ("CRC-5/USB", 5, 0x05, 0x1f, 1, 1, 0x1f, 0x19),
    ("CRC-6/CDMA2000-A", 6, 0x27, 0x3f, 0, 0, 0x00, 0x0d),
    ("CRC-6/CDMA2000-B", 6, 0x07, 0x3f, 0, 0, 0x00, 0x3b),
    ("CRC-6/DARC", 6, 0x19, 0x00, 1, 1, 0x00, 0x26),
    ("CRC-6/G-704", 6, 0x03, 0x00, 1, 1, 0x00, 0x06),
    ("CRC-6/GSM", 6, 0x2f, 0x00, 0, 0, 0x3f, 0x13),
    ("CRC-7/MMC", 7, 0x09, 0x00, 0, 0, 0x00, 0x75),
    ("CRC-7/ROHC", 7, 0x4f, 0x7f, 1, 1, 0x00, 0x53),
    ("CRC-7/UMTS", 7, 0x45, 0x00, 0, 0, 0x00, 0x61),
    ("CRC-8/AUTOSAR", 8, 0x2f, 0xff, 0, 0, 0xff, 0xdf),
    ("CRC-8/BLUETOOTH", 8, 0xa7, 0x00, 1, 1, 0x00, 0x26),
    ("CRC-8/CDMA2000", 8, 0x9b, 0xff, 0, 0, 0x00, 0xda),
    ("CRC-8/DARC", 8, 0x39, 0x00, 1, 1, 0x00, 0x15),
    ("CRC-8/DVB-S2", 8, 0xd5, 0x00, 0, 0, 0x00, 0xbc),
    ("CRC-8/GSM-A", 8, 0x1d, 0x00, 0, 0, 0x00, 0x37),
    ("CRC-8/GSM-B", 8, 0x49, 0x00, 0, 0, 0xff, 0x94),
    ("CRC-8/HITAG", 8, 0x1d, 0xff, 0, 0, 0x00, 0xb4),
    ("CRC-8/I-432-1", 8, 0x07, 0x00, 0, 0, 0x55, 0xa1),
    ("CRC-8/I-CODE", 8, 0x1d, 0xfd, 0, 0, 0x00, 0x7e),
    ("CRC-8/LTE", 8, 0x9b, 0x00, 0, 0, 0x00, 0xea),
    ("CRC-8/MAXIM-DOW", 8, 0x31, 0x00, 1, 1, 0x00, 0xa1),
    ("CRC-8/MIFARE-MAD", 8, 0x1d, 0xc7, 0, 0, 0x00, 0x99),
    ("CRC-8/NRSC-5", 8, 0x31, 0xff, 0, 0, 0x00, 0xf7),
    ("CRC-8/OPENSAFETY", 8, 0x2f, 0x00, 0, 0, 0x00, 0x3e),
    ("CRC-8/ROHC", 8, 0x07, 0xff, 1, 1, 0x00, 0xd0),
    ("CRC-8/SAE-J1850", 8, 0x1d, 0xff, 0, 0, 0xff, 0x4b),
    ("CRC-8/SMBUS", 8, 0x07, 0x00, 0, 0, 0x00, 0xf4),
    ("CRC-8/TECH-3250", 8, 0x1d, 0xff, 1, 1, 0x00, 0x97),
    ("CRC-8/WCDMA", 8, 0x9b, 0x00, 1, 1, 0x00, 0x25),
    ("CRC-10/ATM", 10, 0x233, 0x000, 0, 0, 0x000, 0x199),
    ("CRC-10/CDMA2000", 10, 0x3d9, 0x3ff, 0, 0, 0x000, 0x233),
    ("CRC-10/GSM", 10, 0x175, 0x000, 0, 0, 0x3ff, 0x12a),
    ("CRC-11/FLEXRAY", 11, 0x385, 0x01a, 0, 0, 0x000, 0x5a3),
    ("CRC-11/UMTS", 11, 0x307, 0x000, 0, 0, 0x000, 0x061),
    ("CRC-12/CDMA2000", 12, 0xf13, 0xfff, 0, 0, 0x000, 0xd4d),
    ("CRC-12/DECT", 12, 0x80f, 0x000, 0, 0, 0x000, 0xf5b),
    ("CRC-12/GSM", 12, 0xd31, 0x000, 0, 0, 0xfff, 0xb34),
    ("CRC-12/UMTS", 12, 0x80f, 0x000, 0, 1, 0x000, 0xdaf),
    ("CRC-13/BBC", 13, 0x1cf5, 0x0000, 0, 0, 0x0000, 0x04fa),
    ("CRC-14/DARC", 14, 0x0805, 0x0000, 1, 1, 0x0000, 0x082d),
    ("CRC-14/GSM", 14, 0x202d, 0x0000, 0, 0, 0x3fff, 0x30ae),
    ("CRC-15/CAN", 15, 0x4599, 0x0000, 0, 0, 0x0000, 0x059e),
    ("CRC-15/MPT1327", 15, 0x6815, 0x0000, 0, 0, 0x0001, 0x2566),
    ("CRC-16/ARC", 16, 0x8005, 0x0000, 1, 1, 0x0000, 0xbb3d),
    ("CRC-16/CDMA2000", 16, 0xc867, 0xffff, 0, 0, 0x0000, 0x4c06),
    ("CRC-16/CMS", 16, 0x8005, 0xffff, 0, 0, 0x0000, 0xaee7),
    ("CRC-16/DDS-110", 16, 0x8005, 0x800d, 0, 0, 0x0000, 0x9ecf),
    ("CRC-16/DECT-R", 16, 0x0589, 0x0000, 0, 0, 0x0001, 0x007e),
    ("CRC-16/DECT-X", 16, 0x0589, 0x0000, 0, 0, 0x0000, 0x007f),
    ("CRC-16/DNP", 16, 0x3d65, 0x0000, 1, 1, 0xffff, 0xea82),
    ("CRC-16/EN-13757", 16, 0x3d65, 0x0000, 0, 0, 0xffff, 0xc2b7),
    ("CRC-16/GENIBUS", 16, 0x1021, 0xffff, 0, 0, 0xffff, 0xd64e),
    ("CRC-16/GSM", 16, 0x1021, 0x0000, 0, 0, 0xffff, 0xce3c),
    ("CRC-16/IBM-3740", 16, 0x1021, 0xffff, 0, 0, 0x0000, 0x29b1),
    ("CRC-16/IBM-SDLC", 16, 0x1021, 0xffff, 1, 1, 0xffff, 0x906e),
    ("CRC-16/ISO-IEC-14443-3-A", 16, 0x1021, 0xc6c6, 1, 1, 0x0000, 0xbf05),
    ("CRC-16/KERMIT", 16, 0x1021, 0x0000, 1, 1, 0x0000, 0x2189),
    ("CRC-16/LJ1200", 16, 0x6f63, 0x0000, 0, 0, 0x0000, 0xbdf4),
    ("CRC-16/M17", 16, 0x5935, 0xffff, 0, 0, 0x0000, 0x772b),
    ("CRC-16/MAXIM-DOW", 16, 0x8005, 0x0000, 1, 1, 0xffff, 0x44c2),
    ("CRC-16/MCRF4XX", 16, 0x1021, 0xffff, 1, 1, 0x0000, 0x6f91),
    ("CRC-16/MODBUS", 16, 0x8005, 0xffff, 1, 1, 0x0000, 0x4b37),
    ("CRC-16/NRSC-5", 16, 0x080b, 0xffff, 1, 1, 0x0000, 0xa066),
    ("CRC-16/OPENSAFETY-A", 16, 0x5935, 0x0000, 0, 0, 0x0000, 0x5d38),
    ("CRC-16/OPENSAFETY-B", 16, 0x755b, 0x0000, 0, 0, 0x0000, 0x20fe),
    ("CRC-16/PROFIBUS", 16, 0x1dcf, 0xffff, 0, 0, 0xffff, 0xa819),
    ("CRC-16/RIELLO", 16, 0x1021, 0xb2aa, 1, 1, 0x0000, 0x63d0),
    ("CRC-16/SPI-FUJITSU", 16, 0x1021, 0x1d0f, 0, 0, 0x0000, 0xe5cc),
    ("CRC-16/T10-DIF", 16, 0x8bb7, 0x0000, 0, 0, 0x0000, 0xd0db),
    ("CRC-16/TELEDISK", 16, 0xa097, 0x0000, 0, 0, 0x0000, 0x0fb3),
    ("CRC-16/TMS37157", 16, 0x1021, 0x89ec, 1, 1, 0x0000, 0x26b1),
    ("CRC-16/UMTS", 16, 0x8005, 0x0000, 0, 0, 0x0000, 0xfee8),
    ("CRC-16/USB", 16, 0x8005, 0xffff, 1, 1, 0xffff, 0xb4c8),
    ("CRC-16/XMODEM", 16, 0x1021, 0x0000, 0, 0, 0x0000, 0x31c3),
    ("CRC-17/CAN-FD", 17, 0x1685b, 0x00000, 0, 0, 0x00000, 0x04f03),
    ("CRC-21/CAN-FD", 21, 0x102899, 0x000000, 0, 0, 0x000000, 0x0ed841),
    ("CRC-24/BLE", 24, 0x00065b, 0x555555, 1, 1, 0x000000, 0xc25a56),
    ("CRC-24/FLEXRAY-A", 24, 0x5d6dcb, 0xfedcba, 0, 0, 0x000000, 0x7979bd),
    ("CRC-24/FLEXRAY-B", 24, 0x5d6dcb, 0xabcdef, 0, 0, 0x000000, 0x1f23b8),
    ("CRC-24/INTERLAKEN", 24, 0x328b63, 0xffffff, 0, 0, 0xffffff, 0xb4f3e6),
    ("CRC-24/LTE-A", 24, 0x864cfb, 0x000000, 0, 0, 0x000000, 0xcde703),
    ("CRC-24/LTE-B", 24, 0x800063, 0x000000, 0, 0, 0x000000, 0x23ef52),
    ("CRC-24/OPENPGP", 24, 0x864cfb, 0xb704ce, 0, 0, 0x000000, 0x21cf02),
    ("CRC-24/OS-9", 24, 0x800063, 0xffffff, 0, 0, 0xffffff, 0x200fa5),
    ("CRC-30/CDMA", 30, 0x2030b9c7, 0x3fffffff, 0, 0, 0x3fffffff, 0x04c34abf),
    ("CRC-31/PHILIPS", 31, 0x04c11db7, 0x7fffffff, 0, 0, 0x7fffffff, 0x0ce9e46c),
    ("CRC-32/AIXM", 32, 0x814141ab, 0x00000000, 0, 0, 0x00000000, 0x3010bf7f),
    ("CRC-32/AUTOSAR", 32, 0xf4acfb13, 0xffffffff, 1, 1, 0xffffffff, 0x1697d06a),
    ("CRC-32/BASE91-D", 32, 0xa833982b, 0xffffffff, 1, 1, 0xffffffff, 0x87315576),
    ("CRC-32/BZIP2", 32, 0x04c11db7, 0xffffffff, 0, 0, 0xffffffff, 0xfc891918),
    ("CRC-32/CD-ROM-EDC", 32, 0x8001801b, 0x00000000, 1, 1, 0x00000000, 0x6ec2edc4),
    ("CRC-32/CKSUM", 32, 0x04c11db7, 0x00000000, 0, 0, 0xffffffff, 0x765e7680),
    ("CRC-32/ISCSI", 32, 0x1edc6f41, 0xffffffff, 1, 1, 0xffffffff, 0xe3069283),
    ("CRC-32/ISO-HDLC", 32, 0x04c11db7, 0xffffffff, 1, 1, 0xffffffff, 0xcbf43926),
    ("CRC-32/JAMCRC", 32, 0x04c11db7, 0xffffffff, 1, 1, 0x00000000, 0x340bc6d9),
    ("CRC-32/MEF", 32, 0x741b8cd7, 0xffffffff, 1, 1, 0x00000000, 0xd2c22f51),
    ("CRC-32/MPEG-2", 32, 0x04c11db7, 0xffffffff, 0, 0, 0x00000000, 0x0376e6e7),
    ("CRC-32/XFER", 32, 0x000000af, 0x00000000, 0, 0, 0x00000000, 0xbd0be338),
    ("CRC-40/GSM", 40, 0x0004820009, 0x0, 0, 0, 0xffffffffff, 0xd4164fc646),
    ("CRC-64/ECMA-182", 64, 0x42f0e1eba9ea3693, 0x0, 0, 0, 0x0, 0x6c40df5f0b497347),
    ("CRC-64/GO-ISO", 64, 0x000000000000001b, 0xffffffffffffffff, 1, 1,
     0xffffffffffffffff, 0xb90956c775a41001),
    ("CRC-64/MS", 64, 0x259c84cba6426349, 0xffffffffffffffff, 1, 1, 0x0, 0x75d4b74f024eceea),
    ("CRC-64/NVME", 64, 0xad93d23594c93659, 0xffffffffffffffff, 1, 1, 0xffffffffffffffff,
     0xae8b14860a799888),
    ("CRC-64/REDIS", 64, 0xad93d23594c935a9, 0x0, 1, 1, 0x0, 0xe9c6d914c4b8d9ca),
    ("CRC-64/WE", 64, 0x42f0e1eba9ea3693, 0xffffffffffffffff, 0, 0,
     0xffffffffffffffff, 0x62ec59e3f1a4f00a),
    ("CRC-64/XZ", 64, 0x42f0e1eba9ea3693, 0xffffffffffffffff, 1, 1,
     0xffffffffffffffff, 0x995dc9bbdf1939fa),
]


def crc_spec(name: str) -> dict:
    """The catalogue set called `name`, as a spec dict."""
    for e in CRC_CATALOG:
        if e[0] == name:
            return {"width": e[1], "poly": e[2], "init": e[3], "refin": e[4], "refout": e[5],
                    "xorout": e[6], "check": e[7]}
    raise KeyError(name)


def _reflect(v, w):
    r = 0
    for _ in range(w):
        r = (r << 1) | (v & 1)
        v >>= 1
    return r


def crc_register(bits, width, poly, init, refin) -> int:
    bits = _bits(bits)
    mask = (1 << width) - 1
    top = 1 << (width - 1)
    reg = init & mask
    seq = bits
    if refin:
        whole = (bits.size // 8) * 8
        seq = np.concatenate([bits[:whole].reshape(-1, 8)[:, ::-1].ravel(), bits[whole:]])
    for v in seq:
        fb = (1 if reg & top else 0) ^ int(v)
        reg = (reg << 1) & mask
        if fb:
            reg ^= poly
    return reg


def crc(bits, spec) -> int:
    """The CRC of a bit message under a Rocksoft spec (dict) — atkdsp.h §17."""
    w = spec["width"]
    reg = crc_register(bits, w, spec["poly"], spec["init"], spec["refin"])
    if spec["refout"]:
        reg = _reflect(reg, w)
    return (reg ^ spec["xorout"]) & ((1 << w) - 1)


def ascii_bits(s) -> np.ndarray:
    """Bytes (or str) -> bits, each byte MSB first."""
    if isinstance(s, str):
        s = s.encode()
    return np.unpackbits(np.frombuffer(bytes(s), np.uint8))


def crc_check_value(spec) -> int:
    return crc(ascii_bits(b"123456789"), spec)


CRC_END = 0
CRC_SEARCH = 1
_COMBOS = ((0, 0, 0), (1, 1, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 1, 1), (0, 1, 1), (1, 0, 1))


def _deg(p):
    return p.bit_length() - 1


def _pmod(a, g):
    dg = _deg(g)
    da = _deg(a)
    while a and da >= dg:
        a ^= g << (da - dg)
        da = _deg(a)
    return a


def _pgcd(a, b):
    while b:
        a, b = b, _pmod(a, b)
    return a


def _pdiv(a, g):
    q = 0
    dg = _deg(g)
    while a and _deg(a) >= dg:
        k = _deg(a) - dg
        q |= 1 << k
        a ^= g << k
    return q, a


def _strip_x(p):
    if not p:
        return p
    return p >> ((p & -p).bit_length() - 1)


def _mulmod(u, v, g, w):
    r = 0
    t = u
    top = 1 << w
    for k in range(w):
        if (v >> k) & 1:
            r ^= t
        t <<= 1
        if t & top:
            t ^= g
    return r


def _xpow(L, g, w):
    r, base = 1, _pmod(2, g)
    while L:
        if L & 1:
            r = _mulmod(r, base, g, w)
        base = _mulmod(base, base, g, w)
        L >>= 1
    return r


def _solve_init(a, rhs, g, w, cap=256):
    col = []
    t = a
    for _k in range(w):
        col.append(t)
        t <<= 1
        if t & (1 << w):
            t ^= g
    rows = [sum(((col[k] >> r) & 1) << k for k in range(w)) for r in range(w)]
    aug = [(rhs >> r) & 1 for r in range(w)]
    piv = []
    r0 = 0
    for c in range(w):
        if r0 >= w:
            break
        p = next((r for r in range(r0, w) if (rows[r] >> c) & 1), -1)
        if p < 0:
            continue
        rows[p], rows[r0] = rows[r0], rows[p]
        aug[p], aug[r0] = aug[r0], aug[p]
        for r in range(w):
            if r != r0 and (rows[r] >> c) & 1:
                rows[r] ^= rows[r0]
                aug[r] ^= aug[r0]
        piv.append(c)
        r0 += 1
    np_ = len(piv)
    if any(aug[r] for r in range(np_, w)):
        return []
    free = [c for c in range(w) if c not in piv][:8]
    out = []
    for mm in range(1 << len(free)):
        if len(out) >= cap:
            break
        x = 0
        for f, c in enumerate(free):
            if (mm >> f) & 1:
                x |= 1 << c
        for i in range(np_):
            v = aug[i] ^ (bin(rows[i] & x & ~(1 << piv[i])).count("1") & 1)
            if v:
                x |= 1 << piv[i]
        out.append(x)
    return out


def _field_value(f, w, layout):
    if not layout:
        v = 0
        for x in f:
            v = (v << 1) | int(x)
        return v
    v = 0
    for k in range(w // 8):
        byte = 0
        for x in f[8 * k:8 * k + 8]:
            byte = (byte << 1) | int(x)
        v |= byte << (8 * k)
    return v


def _bits_int(bits):
    v = 0
    for x in bits:
        v = (v << 1) | int(x)
    return v


def crc_recover(frames, width: int = 0, pos: int = CRC_END, max_header: int = 0,
                pfa: float = 1e-3, cap: int = 8) -> list:
    """CRC recovery — the twin of atkdsp_crc_recover (atkdsp.h §17), the same
    search in the same order. `frames` is a list of bit arrays. Returns dicts."""
    frames = [_bits(f) for f in frames]
    nf = len(frames)
    if width and not 3 <= width <= 64:
        raise ValueError("width 0 or 3..64")
    if not pfa > 0.0:
        pfa = 1e-3
    if nf < 3:
        return []
    lens = [f.size for f in frames]
    distinct = any(x != lens[0] for x in lens)
    minlen = min(lens)
    max_off = max_header if pos == CRC_END else (minlen - 7 if minlen > 7 else 0)
    trials = 0.0
    log10_thr = _m.log10(pfa)
    cands = []
    catalog = [{"width": e[1], "poly": e[2], "init": e[3], "refin": e[4], "refout": e[5],
                "xorout": e[6]} for e in CRC_CATALOG]

    def verify(spec, layout, h, t):
        w = spec["width"]
        for f in frames:
            cov = f[h:f.size - t]
            L = cov.size - w
            if crc(cov[:L], spec) != _field_value(cov[L:], w, layout):
                return False
        return True

    for off in range(max_off + 1):
        h = off if pos == CRC_END else 0
        t = 0 if pos == CRC_END else off
        level = []
        for w in ([width] if width else range(3, 65)):
            for ci, (refin, refout, layout) in enumerate(_COMBOS):
                if layout and w % 8:
                    continue
                if not width and off > 0 and ci > 0:
                    continue
                ps, Ls = [], []
                ok = True
                for f in frames:
                    F = f.size
                    if F < h + t + w + 1:
                        ok = False
                        break
                    cov = f[h:F - t]
                    L = cov.size - w
                    msg = cov[:L]
                    if refin:
                        if L % 8:
                            ok = False
                            break
                        msg = msg.reshape(-1, 8)[:, ::-1].ravel()
                    R = _field_value(cov[L:], w, layout)
                    if refout:
                        R = _reflect(R, w)
                    ps.append((_bits_int(msg) << w) | R)
                    Ls.append(L)
                if not ok:
                    continue
                for hyp in range(3):
                    trials += 1.0
                    G = _recover_G(ps, Ls, w, hyp)
                    if G is None:
                        continue
                    e = _deg(G) - w
                    if e > 8:
                        continue
                    gs = []
                    if e == 0:
                        gs.append(G)
                    else:
                        for hm in range(1 << (e - 1 if e > 1 else 0)):
                            hv = 1 | (1 << e) | (hm << 1)
                            q, r = _pdiv(G, hv)
                            if r:
                                continue
                            gs.append(q)
                    for gq in gs:
                        if _deg(gq) != w or not gq & 1:
                            continue
                        glow = gq & ((1 << w) - 1)
                        ib = next((i for i in range(1, nf) if Ls[i] != Ls[0]), -1)
                        r0 = _pmod(ps[0], gq)
                        xl0 = _xpow(Ls[0], gq, w)
                        init_known = True
                        if hyp == 1:
                            sols = [0]
                        elif hyp == 2:
                            sols = [(1 << w) - 1]
                        elif ib >= 0:
                            r1 = _pmod(ps[ib], gq)
                            xl1 = _xpow(Ls[ib], gq, w)
                            sols = _solve_init(xl0 ^ xl1, r0 ^ r1, gq, w)
                        else:
                            sols = [0]
                            init_known = False
                        nver = 0
                        best = None
                        for I in sols:
                            Xp = r0 ^ _mulmod(I, xl0, gq, w)
                            spec = {"width": w, "poly": glow, "init": I, "refin": refin,
                                    "refout": refout,
                                    "xorout": _reflect(Xp, w) if refout else Xp}
                            if not verify(spec, layout, h, t):
                                continue
                            if not nver:
                                best = spec
                            nver += 1
                        if not nver:
                            continue
                        if nver > 1:
                            init_known = False
                        if hyp != 0:
                            if ib < 0:
                                init_known = False
                            else:
                                r1 = _pmod(ps[ib], gq)
                                xl1 = _xpow(Ls[ib], gq, w)
                                nfit = 0
                                for Ia in _solve_init(xl0 ^ xl1, r0 ^ r1, gq, w):
                                    if nfit >= 2:
                                        break
                                    Xa = r0 ^ _mulmod(Ia, xl0, gq, w)
                                    s2 = dict(best, init=Ia, xorout=_reflect(Xa, w) if refout else Xa)
                                    if verify(s2, layout, h, t):
                                        nfit += 1
                                if nfit != 1:
                                    init_known = False
                        u = 3 if (hyp == 0 and distinct) else 2
                        cat = -1
                        if not init_known:
                            for k, cs in enumerate(catalog):
                                if (cs["width"] != w or cs["poly"] != glow or cs["refin"] != refin
                                        or cs["refout"] != refout):
                                    continue
                                if verify(cs, layout, h, t):
                                    best = dict(best, init=cs["init"], xorout=cs["xorout"])
                                    cat = k
                                    break
                        else:
                            for k, cs in enumerate(catalog):
                                if all(cs[x] == best[x] for x in
                                       ("width", "poly", "init", "refin", "refout", "xorout")):
                                    cat = k
                                    break
                        best = dict(best, check=crc_check_value(best))
                        c = {"spec": best, "layout": layout, "header_bits": h,
                             "trailer_bits": t, "init_known": int(init_known),
                             "frames_ok": nf, "catalog": cat,
                             "eq_bits": float((nf - u) * w)}
                        key = (w, glow, best["init"], best["xorout"], refin, refout, layout, h, t)
                        if any(x.get("_key") == key for x in cands + level):
                            continue
                        c["_key"] = key
                        level.append(c)
        cands.extend(level)
    kept = []
    for c in cands:
        eq = c.pop("eq_bits")
        c["log10_pfa"] = _m.log10(trials) - eq * _m.log10(2.0)
        if eq > 0.0 and c["log10_pfa"] <= log10_thr:
            kept.append(c)
    for c in kept:
        c["alternatives"] = sum(1 for d in kept
                                if d is not c and d["spec"]["width"] == c["spec"]["width"])
        c.pop("_key", None)

    def simple(sp):
        ones = (1 << sp["width"]) - 1
        return sp["init"] in (0, ones) and sp["xorout"] in (0, ones)
    kept.sort(key=lambda c: (c["catalog"] < 0, not simple(c["spec"]), c["log10_pfa"],
                             (c["header_bits"] + c["trailer_bits"]) % 8 != 0,
                             c["header_bits"] + c["trailer_bits"]))
    return kept[:cap]


def _recover_G(ps, Ls, w, hyp):
    nf = len(ps)
    if hyp == 0:
        GD, nd = 0, 0
        for i in range(nf):
            first = next((k for k in range(i) if Ls[k] == Ls[i]), -1)
            if first < 0:
                continue
            d = ps[i] ^ ps[first]
            if not d:
                continue
            GD = d if not nd else _pgcd(GD, d)
            nd += 1
        GE, ne, gd = 0, 0, 0
        for i in range(nf):
            for j in range(i + 1, nf):
                if Ls[j] == Ls[i]:
                    continue
                for k in range(j + 1, nf):
                    if ne >= 16:
                        break
                    if Ls[k] == Ls[i]:
                        continue
                    dij, dik = ps[i] ^ ps[j], ps[i] ^ ps[k]
                    E = (dij << Ls[i]) ^ (dij << Ls[k]) ^ (dik << Ls[i]) ^ (dik << Ls[j])
                    if not E:
                        continue
                    gd = _m.gcd(gd, _m.gcd(abs(Ls[i] - Ls[j]), abs(Ls[i] - Ls[k])))
                    GE = E if not ne else _pgcd(GE, E)
                    ne += 1
                if ne >= 16:
                    break
            if ne >= 16:
                break
        if ne >= 2:
            GE = _strip_x(GE)
            q, r = _pdiv(GE, 1 | (1 << gd))
            if not r:
                GE = q
        if nd and ne >= 2:
            G = _pgcd(GD, GE)
        elif nd:
            G = GD
        elif ne >= 2:
            G = GE
        else:
            return None
    else:
        I0 = (1 << w) - 1 if hyp == 2 else 0
        G, np_ = 0, 0
        for j in range(1, nf):
            d = ps[0] ^ ps[j]
            if I0:
                d ^= (I0 << Ls[0]) ^ (I0 << Ls[j])
            if not d:
                continue
            G = d if not np_ else _pgcd(G, d)
            np_ += 1
        if np_ < 2:
            return None
    if not G:
        return None
    G = _strip_x(G)
    return G if _deg(G) >= w else None


# ============================================================================
# §18 naming the code — the twin of src/gf2.c and src/coderec.c
# ----------------------------------------------------------------------------
# The same searches in the same order, the same random draws (xorshift64*),
# the same canonical forms, so the C answer is reproduced field for field.
# Rows and checks are Python ints: bit j = column j = the j-th bit in time.
# Slow, and meant for small cases: the tests compare it with the C on them.
# ============================================================================

_M64 = (1 << 64) - 1
_LN2 = 0.693147180559945309417
_LOG10_2 = 0.30102999566398119521
ROBUST_TMAX = 64
ROBUST_PMAX = 0.05
ROBUST_VMIN = 32
PERIODIC_L10 = -6.0
ID_ROWS = 4096


def _rng_seed(seed: int) -> int:
    seed &= _M64
    return seed if seed else 0x9E3779B97F4A7C15


def _rng(s: int):
    """xorshift64*: (new state, output)."""
    x = s
    x ^= x >> 12
    x = (x ^ (x << 25)) & _M64
    x ^= x >> 27
    return x, (x * 0x2545F4914F6CDD1D) & _M64


def _pop(x: int) -> int:
    return bin(x).count("1")


def gf2_canon(vs) -> list:
    """Canonical form: reduced row echelon, pivot = each vector's highest set
    bit, highest pivot first, zero vectors dropped (unique for the span)."""
    piv = {}
    for v in vs:
        for p in sorted(piv, reverse=True):
            if (v >> p) & 1:
                v ^= piv[p]
        if v:
            p = v.bit_length() - 1
            for q in list(piv):
                if (piv[q] >> p) & 1:
                    piv[q] ^= v
            piv[p] = v
    return [piv[p] for p in sorted(piv, reverse=True)]


def gf2_rank(rows) -> int:
    return len(gf2_canon(rows))


def gf2_nullspace(rows, w: int) -> list:
    """Every h with parity(row & h) = 0 for all rows, canonical."""
    E = gf2_canon(rows)
    pivs = [v.bit_length() - 1 for v in E]
    ps = set(pivs)
    basis = []
    for f in range(w - 1, -1, -1):
        if f in ps:
            continue
        v = 1 << f
        for p, row in zip(pivs, E):
            if (row >> f) & 1:
                v |= 1 << p
        basis.append(v)
    return gf2_canon(basis)


def _unsat(rows, h) -> int:
    return sum(_pop(r & h) & 1 for r in rows)


def _in_span(basis, h) -> bool:
    for s in basis:
        if (h >> (s.bit_length() - 1)) & 1:
            h ^= s
    return h == 0


def _period_l10(rows, h) -> float:
    return _ok_period_l10([1 - (_pop(r & h) & 1) for r in rows], 32)


def _ok_period_l10(ok, pmax) -> float:
    """The twin of atk_ok_period_l10: periods 2..min(pmax, R/16, 64)."""
    R = len(ok)
    cmax = min(R // 16, pmax, 64)
    if cmax < 2:
        return 0.0
    ok = np.array(ok, np.int64)
    K = int(ok.sum())
    if K == 0 or K == R:
        return 0.0
    pr = K / R
    idx = np.arange(R)
    best = 0.0
    for c in range(2, cmax + 1):
        nn = np.bincount(idx % c, minlength=c)
        kk = np.bincount(idx % c, weights=ok, minlength=c).astype(np.int64)
        G = 0.0
        for j in range(c):
            a, b = float(kk[j]), float(nn[j] - kk[j])
            if a > 0:
                G += a * _m.log(a / (float(nn[j]) * pr))
            if b > 0:
                G += b * _m.log(b / (float(nn[j]) * (1.0 - pr)))
        l10 = log_igamc(0.5 * (c - 1), G) / _m.log(10.0)
        if l10 < best:
            best = l10
    return best + _m.log10(cmax - 1)


def gf2_robust(rows, w: int, seed: int, pfa: float, cap: int, quit: int = 16):
    """The twin of atk_gf2_robust_q: (canonical basis, info dict)."""
    R = len(rows)
    S = w + 2
    info = {"log10_pfa": 0.0, "ber": float("nan"), "trials": 0, "complete": 0}
    if R < S + ROBUST_VMIN or cap < 1:
        return [], info
    V = R - S
    thr = _m.log10(pfa if pfa > 0 else 1e-3)
    ltr = _m.log10(ROBUST_TMAX * w)
    st = _rng_seed(seed)
    basis, pest = [], []
    m = still = best_full = prod = 0
    weakest = -_m.inf
    t = 0
    while t < ROBUST_TMAX:
        if m == 0 and t >= quit:
            break
        if m > 0 and still >= 8:
            break
        idx = list(range(R))
        for i in range(S):
            st, x = _rng(st)
            j = i + x % (R - i)
            idx[i], idx[j] = idx[j], idx[i]
        sub = [rows[idx[i]] for i in range(S)]
        val = [rows[idx[i]] for i in range(S, R)]
        nb = gf2_nullspace(sub, w)
        grew = kept = 0
        for h in nb:
            u = _unsat(val, h)
            l10 = log10_binom_tail(V, V - u, -_LN2) + ltr
            if l10 > thr:
                continue
            wt = _pop(h)
            fmax = 0.5 * (1.0 - (1.0 - 2.0 * ROBUST_PMAX) ** wt)
            if u > V * fmax + 3.0 * _m.sqrt(V * fmax * (1.0 - fmax)):
                continue
            fresh = m < cap and not _in_span(basis, h)
            if fresh and _period_l10(rows, h) <= PERIODIC_L10:
                continue
            kept += 1
            f = u / V
            pest.append(0.5 * (1.0 - (1.0 - 2.0 * f) ** (1.0 / wt)) if f < 0.5 else 0.5)
            if l10 > weakest:
                weakest = l10
            if fresh:
                basis = gf2_canon(basis + [h])
                m = len(basis)
                grew = 1
        d = len(nb)
        if d > 0 and kept == d and d > best_full:
            best_full = d
        still = 0 if grew else still + 1
        prod = 0 if grew else prod + (kept > 0)
        t += 1
    info["trials"] = t
    info["log10_pfa"] = weakest if m > 0 else 0.0
    if pest:
        info["ber"] = sorted(pest)[len(pest) // 2]
    info["complete"] = int(m > 0 and (best_full == m or prod >= 8))
    return basis, info


def _def_l10(d, R, w):
    """log10 of the bound 4 x 2^-d(R-w+d) on P(deficiency >= d) (header §18)."""
    return 0.6020599913279624 - d * (R - w + d) * _LOG10_2


def _rows(b, off: int, w: int, R: int, stride: int) -> list:
    """R rows of w bits from bit off, every `stride` bits, as ints."""
    if R <= 0:
        return []
    win = np.lib.stride_tricks.sliding_window_view(b[off:off + (R - 1) * stride + w], w)[::stride][:R]
    pk = np.packbits(win, axis=1, bitorder="little")
    return [int.from_bytes(r.tobytes(), "little") for r in pk]


def gf2_rank_scan(b, w_lo: int, w_hi: int, n_offsets: int = 1, rows: int = 0) -> np.ndarray:
    b = _bits(b)
    n = b.size
    out = np.full((w_hi - w_lo + 1, n_offsets), -1, np.int16)
    for w in range(w_lo, w_hi + 1):
        for o in range(n_offsets):
            if o >= w or o >= n:
                continue
            R = rows if rows else w + 32
            R = min(R, (n - o) // w)
            if R < w + 16:
                continue
            out[w - w_lo, o] = w - gf2_rank(_rows(b, o, w, R, w))
    return out


# ---- polynomials (Python ints, bit i = x^i) -----------------------------------
def _word_poly(v: int, n: int) -> int:
    """A codeword (bit j = position j) as a polynomial: position j = x^(n-1-j)."""
    p = 0
    for j in range(n):
        if (v >> j) & 1:
            p |= 1 << (n - 1 - j)
    return p


def _pmul(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        b >>= 1
    return r


def _is_primitive(f: int, m: int) -> bool:
    if (f >> m) != 1 or not f & 1:
        return False
    nn = (1 << m) - 1
    x = 1
    for i in range(1, nn + 1):
        x <<= 1
        if x >> m:
            x ^= f
        if x == 1:
            return i == nn
    return False


def _poly_order(g: int, emax: int) -> int:
    dg = _deg(g)
    if dg < 1 or not g & 1:
        return 0
    y = 1
    for e in range(1, emax + 1):
        y <<= 1
        if (y >> dg) & 1:
            y ^= g
        if y == 1:
            return e
    return 0


# ---- the block search ---------------------------------------------------------
def _def_at(b, n, o, w, robust, seed, pfa):
    """(deficiency or -2, log10_pfa, complete)."""
    fit = (n - o) // w if o < n else 0
    if not robust:
        R = min(w + 32, fit)
        if R < w + 16:
            return -2, 0.0, 1
        d = w - gf2_rank(_rows(b, o, w, R, w))
        return d, (_def_l10(d, R, w) if d > 0 else 0.0), 1
    R = min(fit, w + 2 + 1024)
    if R < w + 2 + ROBUST_VMIN:
        return -2, 0.0, 1
    quit = (3 if w > 512 else 6) if robust == 2 else 16
    basis, info = gf2_robust(_rows(b, o, w, R, w), w, seed, pfa, w, quit)
    d = len(basis)
    return d, (info["log10_pfa"] if d > 0 else 0.0), info["complete"]


def _def_period(dv, N):
    for P in range(1, N):
        if N % P:
            continue
        if all(dv[o] == dv[(o + P) % N] for o in range(N)):
            return P
    return N


TIES_MAX = 8


def _block_search(b, n, n_lo, n_hi, pfa, seed, modes):
    f = {"N": 0, "offset": 0, "robust": 0, "w0": 0, "d0": 0, "complete": 0, "slide": 0,
         "log10_pfa": 0.0, "ties": []}
    thr = _m.log10(pfa)
    wlo, whi = n_lo, 2 * n_hi
    lw = _m.log10(whi - wlo + 1)
    found = False
    if modes & 1:
        w0, l0 = 0, 0.0
        for w in range(wlo, whi + 1):
            d, l, _c = _def_at(b, n, 0, w, 0, 0, pfa)
            if d == -2:
                break
            if d > 0 and l + lw <= thr:
                w0, l0 = w, l + lw
                break
        j = w0 // n_lo if w0 else 0
        while j >= 1 and not found:
            if w0 % j == 0 and n_lo <= w0 // j <= n_hi:
                N = w0 // j
                bo, bd, bl = -1, 0, 0.0
                dv = []
                for o in range(N):
                    d, l, _c = _def_at(b, n, o, N, 0, 0, pfa)
                    dv.append(d)
                    if d > bd:
                        bd, bo, bl = d, o, l
                if not (bo < 0 or bl + _m.log10(N) > thr):
                    P = _def_period(dv, N)
                    if P < N:
                        f.update(N=N, offset=bo, d0=bd, slide=P, w0=w0, log10_pfa=bl + _m.log10(N))
                        found = True
                        break
                    fit = (n - bo) // N
                    Rb = min(fit, ID_ROWS)
                    same = N - gf2_rank(_rows(b, bo, N, Rb, N)) == bd
                    R1 = N + 32
                    jj = 1
                    while jj <= 3 and same and fit >= 2 * R1:
                        r0 = jj * (fit - R1) // 3
                        if N - gf2_rank(_rows(b, bo + r0 * N, N, R1, N)) != bd:
                            same = False
                        jj += 1
                    if not same:
                        break
                    f.update(N=N, offset=bo, robust=0, w0=w0, d0=bd, complete=1,
                             log10_pfa=max(l0, bl + _m.log10(N)),
                             ties=[o for o in range(N) if dv[o] == bd][:TIES_MAX])
                    found = True
            j -= 1
    if not found and modes & 2:
        ln = _m.log10(n_hi - n_lo + 1)
        for N in range(n_lo, min(n_hi, 512) + 1):
            d2, l, _c = _def_at(b, n, 0, 2 * N, 2, (seed + N) & _M64, pfa)
            if d2 == -2:
                break
            if d2 <= 0 or l + ln > thr:
                continue
            l0 = l + ln
            bo, bd, bl, bc = -1, 0, 0.0, 0
            ties = []
            for o in range(N):
                d, l, c = _def_at(b, n, o, N, 1, (seed + 7919 * (o + 1)) & _M64, pfa)
                if d > bd:
                    bd, bo, bl, bc = d, o, l, c
                    ties = [o]
                elif d > 0 and d == bd and len(ties) < TIES_MAX:
                    ties.append(o)
            lo = bl + _m.log10(N) + ln
            if bo < 0 or lo > thr:
                continue
            f.update(N=N, offset=bo, robust=1, w0=2 * N, d0=bd, complete=bc, log10_pfa=max(l0, lo),
                     ties=ties)
            found = True
            break
    return found, f


# ---- cyclic roots -------------------------------------------------------------
CYC_DMAX, CYC_R0, CYC_RMIN, CYC_SMIN = 11, 24, 8, 3.0
_IRR = None


def _irr_list():
    global _IRR
    if _IRR is None:
        out = []
        for d in range(1, CYC_DMAX + 1):
            for f in range((1 << d) | 1, 2 << d, 2):
                irr = True
                for q, dq, _e, _p in out:
                    if 2 * dq > d:
                        break
                    if _pmod(f, q) == 0:
                        irr = False
                        break
                if not irr:
                    continue
                y = _pmod(2, f)
                e = 1
                while y != 1 and e < (1 << d):
                    y <<= 1
                    if y >> d:
                        y ^= f
                    e += 1
                out.append((f, d, e, int(e == (1 << d) - 1)))
        _IRR = out
    return _IRR


def _cyc_q(b, T, q, e):
    """Q_t, t = 0..T, over b[0..T)."""
    v = np.zeros(e, np.int64)
    y = (q ^ 1) >> 1
    for i in range(e):
        v[i] = y
        y = (y ^ q) >> 1 if y & 1 else y >> 1
    vals = np.where(b[:T] & 1, v[np.arange(T) % e], 0)
    return np.concatenate([[0], np.bitwise_xor.accumulate(vals)]) if T else np.zeros(1, np.int64)


def _field_m(L):
    m = 1
    while (1 << m) - 1 < L:
        m += 1
    return m


def _cyc_relevant(f, L, var):
    q, d, e, prim = f
    if L % e == 0:
        return True
    if var:
        return False
    m = _field_m(L)
    return bool(prim) and (d == m or d == m + 1) and e > L


def _cyclic_scan(b, n, n_lo, n_hi, pfa):
    cf = {"N": 0, "offset": 0, "var": 0, "score": 0.0, "smax": 0.0, "log10_pfa": 0.0}
    irr = _irr_list()
    n_lo = max(n_lo, 3)
    if n_hi < n_lo:
        return False, cf
    R0, T = {}, 0
    for N in range(n_lo, n_hi + 1):
        if N > n:
            continue
        r = min((n - N + 1) // N, CYC_R0)
        if r < CYC_RMIN:
            continue
        R0[N] = r
        T = max(T, (r + 1) * N)
    T = min(T, n)
    if not R0:
        return False, cf
    sig = {}
    for r in set(R0.values()):
        for d in range(1, CYC_DMAX + 1):
            sig[(r, d)] = [-log10_binom_tail(r, c, -d * _LN2) for c in range(r + 1)]
    score = {N: np.zeros((2, N)) for N in R0}
    smax = {N: np.zeros((2, N)) for N in R0}
    ntests = 0.0
    for f in irr:
        q, d, e, _p = f
        if not any(_cyc_relevant(f, N, 0) or (N > 3 and _cyc_relevant(f, N - 1, 1)) for N in R0):
            continue
        Q = _cyc_q(b, T, q, e)
        for N in sorted(R0):
            r = R0[N]
            for var in (0, 1):
                L = N - var
                if L < 3 or not _cyc_relevant(f, L, var):
                    continue
                base = (np.arange(r)[:, None] * N + np.arange(N)[None, :])
                cnt = (Q[base] == Q[base + L]).sum(axis=0)
                s = np.array(sig[(r, d)])[cnt]
                score[N][var] += np.where(s >= CYC_SMIN, s, 0.0)
                smax[N][var] = np.maximum(smax[N][var], s)
                ntests += N
    best = 0.0
    for N in sorted(R0):
        bo, bv, bs = -1, 0, 0.0
        for var in (0, 1):
            for o in range(N):
                s = float(score[N][var][o])
                if s > bs:
                    bs, bo, bv = s, o, var
        if bo < 0:
            continue
        if best == 0.0 or bs > 1.25 * best:
            best = bs
            cf.update(N=N, offset=bo, var=bv, score=bs, smax=float(smax[N][bv][bo]))
    cf["log10_pfa"] = -cf["smax"] + _m.log10(ntests if ntests > 1.0 else 1.0)
    return best > 0.0 and cf["log10_pfa"] <= _m.log10(pfa), cf


def _cyclic_generator(b, n, cf):
    N, L = cf["N"], cf["N"] - cf["var"]
    R = min((n - cf["offset"]) // N, 1024)
    if R * N > 262144:
        R = 262144 // N
    if R < CYC_RMIN:
        return 0, 0, 0.0
    irr = _irr_list()
    T = R * N
    s = b[cf["offset"]:cf["offset"] + T]
    cnt, cq, sq = [], [], []
    cbest = 0.0
    for q, d, e, _p in irr:
        Q = _cyc_q(s, T, q, e)
        k = np.arange(R) * N
        c = int((Q[k + L] == Q[k]).sum())
        pi = 2.0 ** -d
        fr = c / R
        cnt.append(c)
        cq.append((fr - pi) / (1.0 - pi))
        sq.append(_m.sqrt((fr * (1.0 - fr) + 1.0 / R) / R) / (1.0 - pi))
        if cq[-1] - 2.0 * sq[-1] > cbest:
            cbest = cq[-1] - 2.0 * sq[-1]
    thr = -3.0 - _m.log10(len(irr))
    g, dg = 1, 0
    for i, (q, d, e, _p) in enumerate(irr):
        if cq[i] + 3.0 * sq[i] < cbest:
            continue
        if log10_binom_tail(R, cnt[i], -d * _LN2) > thr:
            continue
        if dg + d >= L:
            continue
        g = _pmul(g, q)
        dg += d
    return dg, g, cbest


def _cyclic_checks(b, n, cf, g, dg):
    N, L = cf["N"], cf["N"] - cf["var"]
    k = L - dg
    if k < 1:
        return []
    ext = 0
    if cf["var"]:
        R = min((n - cf["offset"]) // N, 1024)
        last = first = tot = 0
        for r in range(R):
            p0 = cf["offset"] + r * N
            w = b[p0:p0 + N]
            p, par = 0, 0
            for j in range(L):
                if w[j] & 1:
                    p |= 1 << (L - 1 - j)
                    par ^= 1
            if _pmod(p, g):
                continue
            tot += 1
            last += int((w[L] & 1) == par)
            if p0 >= 1:
                first += int((b[p0 - 1] & 1) == par)
        if tot >= CYC_RMIN and last >= 0.9 * tot:
            ext = 1
        elif tot >= CYC_RMIN and first >= 0.9 * tot:
            ext = 2
    sh = 1 if ext == 2 else 0
    G = []
    for i in range(k):
        row, par = 0, 0
        for e in range(dg + 1):
            if (g >> e) & 1:
                row |= 1 << (L - 1 - (e + i) + sh)
                par ^= 1
        if ext and par:
            row |= 1 << (L if ext == 1 else 0)
        G.append(row)
    if ext == 2:
        cf["offset"] = (cf["offset"] + N - 1) % N
    if not ext and cf["var"]:
        G.append(1 << L)
    return gf2_nullspace(G, N)


# ---- naming -------------------------------------------------------------------
def _is_cyclic(G, H, n):
    for c in G:
        s = ((c >> 1) | ((c & 1) << (n - 1)))
        if any(_pop(s & h) & 1 for h in H):
            return False
    return True


def _cyclic_gen(G, n):
    a = (1 << n) | 1
    for row in G:
        a = _pgcd(a, _word_poly(row, n))
    return a


def _name_cyclic(g, L, k, cap):
    """(name, bch (t, f, m) or None) — the C name_cyclic, the name cut to cap - 1."""
    dg = _deg(g)

    def cut(s):
        return s[:cap - 1]
    if dg == 1:
        return cut(f"single parity check ({L},{k})"), None
    if k == 1:
        return cut(f"repetition ({L},1)"), None
    if L == 23 and dg == 11 and g in (0xC75, 0xAE3):
        return cut("Golay(23,12)"), None
    for m in range(3, 10):
        if L != (1 << m) - 1:
            continue
        for f in range((1 << m) | 1, 2 << m, 2):
            if not _is_primitive(f, m):
                continue
            t, tbest = 1, 0
            while 2 * t < L:
                try:
                    bc = BCH(m, t, 0, f)
                except ValueError:
                    break
                if bc.n - bc.k > dg:
                    break
                if bc.n - bc.k == dg:
                    q = sum(int(c) << dd for dd, c in enumerate(bc.gen))
                    if q == g:
                        tbest = t
                t += 1
            if tbest:
                nm = f"Hamming({L},{k})" if tbest == 1 else f"BCH({L},{k}) t={tbest}"
                return cut(nm), (tbest, f, m)
    return cut(f"cyclic ({L},{k})"), None


BIAS_MAX, LC_BITS, CYC_PMAX, CYC_CMIN = 0.1, 8192, 0.08, 0.1


def _cyc_floor(L):
    return max((1.0 - CYC_PMAX) ** L, CYC_CMIN)


def _stream_unfit(b):
    """(0 fit | 1 biased | 2 recurrence, P(1), linear complexity)."""
    n = b.size
    p1, lc = 0.5, -1
    m = min(n, LC_BITS)
    if m >= 256:
        lc = bm(b[:m])[0]
        if 2 * lc + 64 < m:
            return 2, p1, lc
    if n >= 2000:
        p1 = int(np.count_nonzero(b & 1)) / n
        if abs(p1 - 0.5) > BIAS_MAX:
            return 1, p1, lc
    return 0, p1, lc


def _block_copies(H, N, L):
    """(d0, block checks) when the code is N/L copies of one length-L code, else (0, [])."""
    m = N // L
    d = len(H)
    if m < 2 or N % L or d % m:
        return 0, []
    d0 = d // m
    if d0 < 1:
        return 0, []
    mask = (1 << L) - 1
    blk0 = []
    for i, h in enumerate(H):
        top, low = h.bit_length() - 1, (h & -h).bit_length() - 1
        blk = top // L
        if low // L != blk or blk != m - 1 - i // d0:
            return 0, []
        if blk == 0:
            blk0.append(h & mask)
    for i, h in enumerate(H):
        if (h >> ((m - 1 - i // d0) * L)) & mask != blk0[i % d0]:
            return 0, []
    return d0, blk0


def _frac_valid(b, off, N, R, H):
    rows = _rows(b, off, N, R, N)
    return sum(all(not (_pop(r & h) & 1) for h in H) for r in rows) / R if R else 0.0


def _sliding_period(b, n, off, N, H):
    if off + N - 1 >= n:
        return 0
    R = min((n - off - (N - 1)) // N, 1024)
    if R < 16:
        return 0
    chance = 2.0 ** -min(len(H), 60)
    e0 = _frac_valid(b, off, N, R, H) - chance
    if e0 <= 0.1:
        return 0
    for p in range(1, N):
        if N % p:
            continue
        if _frac_valid(b, off + p, N, R, H) - chance >= 0.8 * e0:
            return p
    return 0


def _noise_like(rows, N, H):
    d = len(H)
    col = []
    for j in range(N):
        s = 0
        for c, h in enumerate(H):
            if (h >> j) & 1:
                s |= 1 << c
        col.append(s)
    cols = set(col)
    F = S = 0
    ok = []
    for r in rows:
        s = 0
        for c, h in enumerate(H):
            if _pop(r & h) & 1:
                s |= 1 << c
        ok.append(0 if s else 1)
        if not s:
            continue
        F += 1
        S += s in cols
    if _ok_period_l10(ok, 64) <= PERIODIC_L10:
        return False
    if F < 8:
        return True
    R = len(rows)
    V = (R - F) / R
    p = 1.0 - V ** (1.0 / N) if V > 0.0 else 1.0
    q = N * p * (1.0 - p) ** (N - 1) / (1.0 - (1.0 - p) ** N) if p < 1.0 else 0.0
    if not q > 0.0 or q >= 1.0:
        return True if q >= 1.0 else S >= F // 2
    l10 = log10_binom_tail(F, F - S, _m.log(1.0 - q))
    return not (l10 <= -4.0 and S < 0.8 * q * F)


def _name_extended(Gm, k, N):
    """(name, g) of an extended code (parity bit last, or first), or (None, 0)."""
    L = N - 1
    if L < 2:
        return None, 0
    for first in (0, 1):
        Gp = gf2_canon([(row >> first) & ((1 << L) - 1) for row in Gm])
        Hp = gf2_nullspace(Gp, L)
        if len(Gp) == k and len(Hp) == L - k and _is_cyclic(Gp, Hp, L):
            g = _cyclic_gen(Gp, L)
            inner, _b = _name_cyclic(g, L, k, 40)
            if "(" in inner and ")" in inner[inner.index("("):]:
                lp = inner.index("(")
                rp = inner.index(")", lp)
                nm = f"extended {inner[:lp]}({N},{k}){inner[rp + 1:]}{' parity first' if first else ''}"
            else:
                nm = f"extended {inner}"
            return nm[:47], g
    return None, 0


def block_code_id(b, n_lo: int = 4, n_hi: int = 256, pfa: float = 1e-3, seed: int = 0) -> dict:
    """The twin of atkdsp_block_code_id (header §18); the dict of the
    binding: the atkdsp_blockid fields, 'found', 'G' and 'gpoly'."""
    b = _bits(b)
    n = b.size
    if not (2 <= n_lo <= n_hi <= 512):
        raise ValueError("block_code_id: 2 <= n_lo <= n_hi <= 512")
    if not pfa > 0:
        pfa = 1e-3
    out = {"n": 0, "k": 0, "offset": 0, "deficiency": 0, "cyclic": 0, "extended": 0, "dmin": 0,
           "rows_used": 0, "robust": 0, "shortened": 0, "log10_pfa": 0.0, "ber_est": float("nan"),
           "agreement": float("nan"), "valid_words": float("nan"), "name": "no code found",
           "found": False, "G": None, "gpoly": None}
    u, p1, _lc = _stream_unfit(b)
    if u == 1:
        out["name"] = f"no code found (biased: P(1) = {p1:.2f})"
        return out
    if u == 2:
        out["name"] = "no block code (linear recurrence)"
        return out
    found, f = _block_search(b, n, n_lo, n_hi, pfa, seed, 1)
    if found and f["slide"]:
        out.update(n=f["N"], offset=f["offset"], deficiency=f["d0"], log10_pfa=f["log10_pfa"])
        out["name"] = ("no block code (linear recurrence)" if f["slide"] == 1
                       else f"no block code (sliding, period {f['slide']})")
        return out
    cyc, cf, cg, cdg = False, None, 0, 0
    if not found:
        ok, cf = _cyclic_scan(b, n, n_lo, n_hi, pfa)
        if ok:
            cdg, cg, clean = _cyclic_generator(b, n, cf)
            if cdg > 0 and clean < _cyc_floor(cf["N"] - cf["var"]):
                cdg = 0
            e = _poly_order(cg, cf["N"]) if cdg > 0 and not cf["var"] else 0
            if 0 < e < cf["N"] and cf["N"] % e == 0:
                for L in range(e, cf["N"], e):
                    if cf["N"] % L or L < n_lo or L < 3:
                        continue
                    c2 = dict(cf, N=L, offset=cf["offset"] % L)
                    d2, g2, cl2 = _cyclic_generator(b, n, c2)
                    if d2 > 0 and cl2 >= _cyc_floor(L):
                        cf, cg, cdg = c2, g2, d2
                        break
            cyc = cdg > 0
        if not cyc:
            found, f = _block_search(b, n, n_lo, n_hi, pfa, seed, 2)
            if not found:
                return out
    N = cf["N"] if cyc else f["N"]
    H, robust, complete, rob_ber = [], 0, 1, float("nan")
    if cyc:
        H = _cyclic_checks(b, n, cf, cg, cdg)
        f.update(N=N, offset=cf["offset"], robust=2, log10_pfa=cf["log10_pfa"], d0=len(H))
        robust = 2
    R = min((n - f["offset"]) // N, ID_ROWS)
    rows = _rows(b, f["offset"], N, R, N)
    if cyc:
        Hr, info = gf2_robust(rows, N, seed ^ 0xC1C, pfa, N)
        if Hr:
            H1 = gf2_canon(H + Hr)
            if len(H1) > len(H) and not info["complete"]:
                complete, rob_ber = 0, info["ber"]
            H = H1
    else:
        H = [] if f["robust"] else gf2_nullspace(rows, N)
        if not f["robust"] and len(H) != f["d0"]:
            H = []
        robust = f["robust"]
        if not H:
            H, info = gf2_robust(rows, N, seed ^ 0xB10C, pfa, N)
            robust = 1
            complete = info["complete"]
            rob_ber = info["ber"]
    if H and complete:
        for L in range(2, N):
            if N % L or L < n_lo:
                continue
            d0, Hb = _block_copies(H, N, L)
            if d0 > 0:
                N, H = L, Hb
                R = min((n - f["offset"]) // N, ID_ROWS)
                rows = _rows(b, f["offset"], N, R, N)
                break
    d = len(H)
    if d > 0 and not complete:
        out.update(n=N, offset=f["offset"], deficiency=d, robust=robust, rows_used=R,
                   log10_pfa=f["log10_pfa"], ber_est=rob_ber, name=f"no code found (n={N} too noisy)")
        return out
    if 0 < d < N:
        P = _sliding_period(b, n, f["offset"], N, H)
        if P:
            out.update(n=N, offset=f["offset"], deficiency=d, log10_pfa=f["log10_pfa"])
            out["name"] = ("no block code (same checks at every offset)" if P == 1
                           else f"no block code (sliding, period {P})")
            return out
        if not _noise_like(rows, N, H):
            out.update(n=N, offset=f["offset"], deficiency=d, log10_pfa=f["log10_pfa"], rows_used=R,
                       name=f"no code found (n={N} fails unlike noise)")
            return out
    if not (0 < d < N):
        return out
    k = N - d
    Gm = gf2_nullspace(H, N)
    if len(Gm) != k:
        return out
    out.update(n=N, k=k, offset=f["offset"], deficiency=d, rows_used=R, robust=robust,
               log10_pfa=f["log10_pfa"])
    pe = []
    for h in H:
        uu = _unsat(rows, h)
        fr = uu / R
        wt = _pop(h)
        pe.append(0.5 * (1.0 - (1.0 - 2.0 * fr) ** (1.0 / wt)) if fr < 0.5 else 0.5)
    pe.sort()
    out["ber_est"] = pe[len(pe) // 2]
    out["valid_words"] = sum(all(not (_pop(r & h) & 1) for h in H) for r in rows) / R
    g, bch, generic = 0, None, False
    out["cyclic"] = int(_is_cyclic(Gm, H, N))
    if out["cyclic"]:
        g = _cyclic_gen(Gm, N)
        out["name"], bch = _name_cyclic(g, N, k, 48)
        generic = out["name"].startswith("cyclic (")
    else:
        gs = 0
        for i, row in enumerate(Gm):
            p = _word_poly(row, N)
            gs = p if i == 0 else _pgcd(gs, p)
        e = _poly_order(gs, 1023) if _deg(gs) == N - k else 0
        if e > N:
            g = gs
            inner, bch = _name_cyclic(gs, e, e - (N - k), 40)
            out["shortened"] = e - N
            out["name"] = f"shortened {inner} as ({N},{k})"[:47]
    if not out["cyclic"] and not out["shortened"]:
        ones = (1 << N) - 1
        if len(gf2_canon(H + [ones])) == d:
            nm, ge = _name_extended(Gm, k, N)
            if nm:
                out["extended"], out["name"], g = 1, nm, ge
        if not out["extended"]:
            out["name"] = f"linear ({N},{k})"
    Gb = np.array([[(row >> j) & 1 for j in range(N)] for row in Gm], np.uint8)
    changed = decoded = 0
    if N <= 32 and N - k <= 20:
        try:
            bc = BlockCode(Gb)
        except ValueError:
            bc = None
        if bc is not None:
            out["dmin"] = bc.dmin
            for r in rows:
                word = np.array([(r >> j) & 1 for j in range(N)], np.uint8)
                _c, _d, cc = bc.decode(word)
                if cc >= 0:
                    changed += cc
                    decoded += N
    elif bch is not None and (out["cyclic"] or out["shortened"]):
        t, fp, m = bch
        bc = BCH(m, t, out["shortened"], fp)
        for r in rows:
            word = np.array([(r >> j) & 1 for j in range(N)], np.uint8)
            _c, cc, _p = bc.decode(word)
            if cc >= 0:
                changed += cc
                decoded += N
    if decoded:
        out["agreement"] = 1.0 - changed / decoded
    if (not out["cyclic"] or generic) and not out["extended"] and not out["shortened"]:
        for mm in range(3, 9):
            if N % mm or (N - k) % mm or N // mm > (1 << mm) - 1 or N // mm < 3:
                continue
            rs = rs_code_id(b, out["offset"], mm, N // mm, pfa)
            if not rs["found"] or rs["nroots"] * mm != N - k:
                continue
            L, NNr = N // mm, (1 << mm) - 1
            rc2 = RS(mm, rs["gfpoly"], rs["fcr"], rs["prim"], rs["nroots"], NNr - L, bool(rs["dual_basis"]))
            same = True
            for row in Gm:
                bits = np.array([(row >> j) & 1 for j in range(N)], np.uint8)
                sy = _rs_syms(bits, 0, 1, L, mm, rs["lsb_first"])[0]
                if rc2.decode(sy)[1] != 0:
                    same = False
                    break
            if not same:
                continue
            out["name"] = f"RS({rs['n']},{rs['k']}) GF(2^{mm}) binary image"
            if rs["agreement"] == rs["agreement"]:
                out["agreement"] = rs["agreement"]
            break
    out["found"] = True
    out["G"] = Gb
    if out["cyclic"] or out["extended"] or out["shortened"]:
        dg = _deg(g)
        out["gpoly"] = np.array([(g >> (dg - i)) & 1 for i in range(dg + 1)], np.uint8)
    return out


# ---- convolutional codes --------------------------------------------------------
CONV_T, CONV_XROWS = 64, 32
CONV_CATALOG = [
    (7, (0o171, 0o133), "NASA/CCSDS, 802.11, DVB-S"),
    (7, (0o133, 0o171, 0o165), "LTE (36.212)"),
    (9, (0o561, 0o753), "IS-95 / cdma2000 / 3GPP r=1/2"),
    (9, (0o557, 0o663, 0o711), "IS-95 / 3GPP r=1/3"),
    (5, (0o23, 0o33), "GSM"),
    (7, (0o133, 0o171, 0o145, 0o133), "DAB / DRM mother code"),
    (3, (0o7, 0o5), "textbook K=3"),
]


def _win_best(b, n, n0, L):
    """(best deficiency or -2, hits, lower, log10_pfa) over the stretches."""
    w = n0 * (L + 1)
    R = w + CONV_XROWS
    span = n0 * (R - 1) + w
    if span > n:
        return -2, 0, 0, 0.0
    room = (n - span) // n0
    T = min(CONV_T, room + 1)
    best = hits = lower = 0
    for j in range(T):
        st = n0 * ((room * j) // (T - 1) if T > 1 else 0)
        d = w - gf2_rank(_rows(b, st, w, R, n0))
        if d > best:
            lower += hits
            best, hits = d, 1
        elif d == best:
            hits += 1
        else:
            lower += 1
    return best, hits, lower, (_def_l10(best, R, w) if best > 0 else 0.0)


def _gen_rows(b, nn, ph, K, t0, steps):
    rows = []
    for t in range(t0 + K - 1, t0 + K - 1 + steps):
        for i in range(1, nn):
            row = 0
            for j in range(K):
                c1 = ph + (t - j) * nn
                if b[c1] & 1:
                    row |= 1 << (i * K + j)
                if b[c1 + i] & 1:
                    row |= 1 << j
            rows.append(row)
    return rows


def _conv_generators(b, n, nn, max_K, pfa):
    thr = _m.log10(pfa)
    ltests = _m.log10(CONV_T * nn * max_K)
    for K in range(2, max_K + 1):
        w = nn * K
        ss = (w + CONV_XROWS + nn - 2) // (nn - 1)
        for ph in range(nn):
            avail = (n - ph) // nn
            if avail < K + ss:
                continue
            room = avail - (K - 1) - ss
            vs = min(avail - (K - 1), 4096 // (nn - 1))
            V = vs * (nn - 1)
            Mv = _gen_rows(b, nn, ph, K, 0, vs)
            T = min(CONV_T, room + 1)
            for j in range(T):
                t0 = (room * j) // (T - 1) if T > 1 else 0
                basis = gf2_nullspace(_gen_rows(b, nn, ph, K, t0, ss), w)
                for h in basis[:4]:
                    wts = [_pop((h >> (s * K)) & ((1 << K) - 1)) for s in range(nn)]
                    if 0 in wts:
                        continue
                    u = _unsat(Mv, h)
                    l10 = log10_binom_tail(V, V - u, -_LN2) + ltests
                    if l10 > thr:
                        continue
                    bad = False
                    for i in range(1, nn):
                        sub = Mv[i - 1::nn - 1]
                        ui, vi = _unsat(sub, h), len(sub)
                        fmax = 0.5 * (1.0 - (1.0 - 2.0 * ROBUST_PMAX) ** (wts[0] + wts[i]))
                        if ui > vi * fmax + 3.0 * _m.sqrt(vi * fmax * (1.0 - fmax)):
                            bad = True
                            break
                    if bad:
                        continue
                    return K, ph, h, (u > 0 or j > 0)
    return 0, 0, 0, False


def conv_code_id(b, max_n: int = 8, max_K: int = 15, pfa: float = 1e-3, seed: int = 0) -> dict:
    """The twin of atkdsp_conv_code_id (header §18)."""
    b = _bits(b)
    n = b.size
    if not (2 <= max_n <= 8 and 2 <= max_K <= 15):
        raise ValueError("conv_code_id: max_n 2..8, max_K 2..15")
    if not pfa > 0:
        pfa = 1e-3
    out = {"n": 0, "k": 0, "K": 0, "phase": 0, "polys": [], "robust": 0, "rows_used": 0,
           "density": float("nan"), "log10_pfa": 0.0, "ber_est": float("nan"),
           "agreement": float("nan"), "name": "no code found", "found": False}
    u, p1, _lc = _stream_unfit(b)
    if u == 1:
        out["name"] = f"no code found (biased: P(1) = {p1:.2f})"
        return out
    if u == 2:
        out["name"] = "no code found (linear recurrence)"
        return out
    thr = _m.log10(pfa)
    Lmax = max_K + 2
    ltests = _m.log10(CONV_T * Lmax * (max_n - 1))
    dens, l10n, rob, Rn, unsure = {}, {}, {}, {}, {}
    best = 0.0
    for n0 in range(2, max_n + 1):
        dens[n0], l10n[n0], rob[n0], Rn[n0], unsure[n0] = 0.0, 0.0, 0, 0, 0
        L0 = d0 = h0 = lo0 = 0
        l0 = 0.0
        for L in range(1, Lmax + 1):
            d, h, lo, l = _win_best(b, n, n0, L)
            if d == -2:
                break
            if d > 0 and l + ltests <= thr:
                L0, d0, h0, lo0, l0 = L, d, h, lo, l
                break
        if not L0:
            continue
        d4, h4, lo4, _l4 = _win_best(b, n, n0, L0 + 4)
        if d4 < d0:
            continue
        dens[n0] = (d4 - d0) / (4.0 * n0)
        l10n[n0] = l0 + ltests
        Rn[n0] = n0 * (L0 + 1) + CONV_XROWS
        rob[n0] = int(lo0 > 0 or lo4 > 0)
        unsure[n0] = int(h0 < 2 or h4 < 2)
        if dens[n0] > best + 1e-9:
            best = dens[n0]
    nn, robust, l10best = 0, 0, 0.0
    if best > 0.0:
        for n0 in range(2, max_n + 1):
            if dens[n0] >= best - 0.05:
                nn, robust, l10best, out["rows_used"] = n0, rob[n0], l10n[n0], Rn[n0]
                break
    if best <= 0.0 or not nn:
        return out
    dens_unsure = unsure[nn]
    out.update(n=nn, density=best, robust=robust, log10_pfa=l10best)
    k = max(1, nn - int(_m.floor(best * nn + 0.5)))
    if k >= nn:
        out.update(n=0, density=float("nan"), robust=0, log10_pfa=0.0, rows_used=0)
        return out
    out["k"] = k
    out["found"] = True
    bestK = bestph = gn = 0
    sol = 0
    for n0 in range(2, nn + 1):
        if nn % n0 or (not dens_unsure and n0 * k != nn and n0 != nn):
            continue
        K, ph, h, noisy = _conv_generators(b, n, n0, max_K, pfa)
        if K:
            bestK, bestph, sol, gn = K, ph, h, n0
            if noisy:
                robust = 1
            break
    m = 2
    while bestK and gn * m <= max_n:
        n0 = gn * m
        if dens[n0] > 0.0:
            K2, ph2, h2, noisy = _conv_generators(b, n, n0, max_K, pfa)
            if K2 and K2 <= bestK:
                gn, bestK, bestph, sol = n0, K2, ph2, h2
                out["density"], l10best = dens[n0], l10n[n0]
                if noisy:
                    robust = 1
        m += 1
    if not bestK:
        if out["k"] > 1:
            rate = f"{out['k'] & 7}/{nn & 15}"
            out["name"] = (f"rate {rate} convolutional? (density unsure)" if dens_unsure
                           else f"rate {rate} convolutional (k > 1: no generators)")
        else:
            out["name"] = f"convolutional, rate 1/{nn} (generators not found)"
        return out
    nn = gn
    out.update(n=nn, k=1, robust=robust, log10_pfa=l10best)
    polys = []
    for s in range(nn):
        p = 0
        for j in range(bestK):
            if (sol >> (s * bestK + j)) & 1:
                p |= 1 << (bestK - 1 - j)
        polys.append(p)
    while True:
        any_lo = any(p & 1 for p in polys)
        any_hi = any((p >> (bestK - 1)) & 1 for p in polys)
        if bestK <= 2 or (any_lo and any_hi):
            break
        if not any_lo:
            polys = [p >> 1 for p in polys]
        bestK -= 1
    out.update(K=bestK, phase=bestph, polys=polys)
    steps = min((n - bestph) // nn, 20000, (1 << 25) >> (bestK - 1))
    llr = np.where(b[bestph:bestph + steps * nn] & 1, -1.0, 1.0).astype(np.float32)
    v = Viterbi(bestK, polys, None, CONV_STREAM, 0)
    dec = np.concatenate([v.decode(llr), v.flush()]).astype(np.uint8)
    re = conv_encode(bestK, polys, dec, CONV_STREAM)
    lo = nn * (bestK - 1)
    if re.size > lo:
        seg = b[bestph + lo:bestph + re.size] & 1
        out["agreement"] = float(np.mean(re[lo:] == seg))
        out["ber_est"] = 1.0 - out["agreement"]
    ps = ",".join(f"{p:o}" for p in polys)
    cat = None
    for K, cp, name in CONV_CATALOG:
        if K != bestK or len(cp) != nn:
            continue
        left = list(cp)
        ok = True
        for p in polys:
            if p in left:
                left.remove(p)
            else:
                ok = False
                break
        if ok:
            cat = name
            break
    out["name"] = (f"K={bestK} r=1/{nn} ({ps}) {cat}" if cat else f"K={bestK} r=1/{nn} ({ps})")[:63]
    return out


# ---- Reed-Solomon -----------------------------------------------------------------
_CCSDS_TAL = (0x8d, 0xef, 0xec, 0x86, 0xfa, 0x99, 0xaf, 0x7b)


def _rs_syms(b, offset, R, L, mm, lsb):
    seg = b[offset:offset + R * L * mm].reshape(R * L, mm).astype(np.int64)
    sh = np.arange(mm) if lsb else np.arange(mm)[::-1]
    return (seg << sh).sum(axis=1).reshape(R, L)


def _rs_counts(sym, NN, expt, logt):
    """cnt[e] = codewords with c(alpha^e) = 0, Horner over the symbols."""
    R, L = sym.shape
    e = np.arange(NN)[None, :]
    acc = np.zeros((R, NN), np.int64)
    for j in range(L):
        nz = acc != 0
        acc = np.where(nz, expt[np.where(nz, logt[acc], 0) + e], 0) ^ sym[:, j][:, None]
    return (acc == 0).sum(axis=0)


def rs_code_id(b, offset: int = 0, m: int = 0, nsym: int = 0, pfa: float = 1e-3) -> dict:
    """The twin of atkdsp_rs_code_id (header §18)."""
    b = _bits(b)
    n = b.size
    if m < 0 or m > 8 or (m and m < 3) or nsym < 0:
        raise ValueError("rs_code_id: m 3..8 or 0")
    if not pfa > 0:
        pfa = 1e-3
    out = {"m": 0, "n": 0, "k": 0, "nroots": 0, "fcr": 0, "prim": 0, "gfpoly": 0, "dual_basis": 0,
           "lsb_first": 0, "rows_used": 0, "log10_pfa": 0.0, "valid_words": float("nan"),
           "agreement": float("nan"), "name": "no code found", "found": False}
    if offset >= n:
        return out
    u, p1, _lc = _stream_unfit(b[offset:])
    if u == 1:
        out["name"] = f"no code found (biased: P(1) = {p1:.2f})"
        return out
    if u == 2:
        out["name"] = "no code found (linear recurrence)"
        return out
    thr = _m.log10(pfa)
    ms = [m] if m else list(range(3, 9))
    nrep = 0.0
    for mm in ms:
        nrep += 2.0 * sum(_is_primitive(f, mm) for f in range((1 << mm) | 1, 2 << mm, 2))
        if mm == 8:
            nrep += 2.0
    tal = [0] * 256
    for x in range(256):
        v = 0
        for j in range(8):
            if (x >> j) & 1:
                v ^= _CCSDS_TAL[7 - j]
        tal[x] = v
    tal1 = np.zeros(256, np.int64)
    for x in range(256):
        tal1[tal[x]] = x
    best_score, bm, bf, bdual, blsb = 0.0, 0, 0, 0, 0
    ntests = 0.0
    stab = {}
    twin_f, twin_score = 0, 0.0
    cand = []
    for ps in range(3):
        if ps == 1:
            twin_f = 0
            if not bdual:
                twin_f = sum(((bf >> i) & 1) << (bm - i) for i in range(bm + 1))
                twin_score = stab.get((1 - blsb, twin_f & 511), 0.0)
        for mm in ms:
            if ps and mm != bm:
                continue
            if ps == 2 and (not twin_f or twin_score < 0.5 * best_score):
                continue
            NN = (1 << mm) - 1
            L = nsym if nsym else NN
            if L > NN or L < 3:
                continue
            fit = (n - offset) // (L * mm)
            R = min(fit, 1024 if ps else 64)
            if R < 4:
                continue
            for f in range((1 << mm) | 1, 2 << mm, 2):
                if (ps == 1 and f != bf) or (ps == 2 and f != twin_f):
                    continue
                if not _is_primitive(f, mm):
                    continue
                expt = np.zeros(2 * NN, np.int64)
                logt = np.zeros(1 << mm, np.int64)
                x = 1
                for i in range(NN):
                    expt[i] = x
                    logt[x] = i
                    x <<= 1
                    if x >> mm:
                        x ^= f
                expt[NN:] = expt[:NN]
                for dual in range(2 if (mm == 8 and f == 0x187) else 1):
                    for lsb in (0, 1):
                        if ps == 1 and (dual != bdual or lsb != blsb):
                            continue
                        if ps == 2 and (dual or lsb == blsb):
                            continue
                        sym = _rs_syms(b, offset, R, L, mm, lsb)
                        if dual:
                            sym = tal1[sym]
                        cnt = _rs_counts(sym, NN, expt, logt)
                        if not ps:
                            sc = 0.0
                            for e in range(NN):
                                if cnt[e] < 2:
                                    continue
                                lv = log10_binom_tail(R, int(cnt[e]), -mm * _LN2)
                                if lv < -2.0:
                                    sc -= lv
                            ntests += NN
                            if not dual:
                                stab[(lsb, f & 511)] = sc
                            if sc > best_score * (1.0 + 1e-9) + 1e-12:
                                best_score, bm, bf, bdual, blsb = sc, mm, f, dual, lsb
                            continue
                        ltr = _m.log10(NN * nrep)
                        roots, weakest = [], -_m.inf
                        for e in range(NN):
                            lv = log10_binom_tail(R, int(cnt[e]), -mm * _LN2) + ltr
                            if lv <= thr:
                                roots.append(e)
                                weakest = max(weakest, lv)
                        nr = len(roots)
                        if nr < 2 or nr >= L:
                            continue
                        prim, fcr = 0, -1
                        for pp in range(1, NN):
                            if prim:
                                break
                            if _m.gcd(pp, NN) != 1:
                                continue
                            pinv = 1
                            while (pinv * pp) % NN != 1:
                                pinv += 1
                            mark = [0] * NN
                            for r_ in roots:
                                mark[(r_ * pinv) % NN] = 1
                            for s0 in range(NN):
                                if not mark[s0] or mark[(s0 + NN - 1) % NN]:
                                    continue
                                ln_ = 0
                                while ln_ < nr and mark[(s0 + ln_) % NN]:
                                    ln_ += 1
                                if ln_ == nr:
                                    prim, fcr = pp, s0
                                break
                        if not prim:
                            continue
                        cand.append({"m": mm, "n": L, "nroots": nr, "k": L - nr, "fcr": fcr,
                                     "prim": prim, "gfpoly": f, "dual_basis": dual,
                                     "lsb_first": lsb, "log10_pfa": weakest})
        if not ps and best_score + thr - _m.log10(ntests if ntests > 1.0 else 1.0) < 0.0:
            return out
    c0, c1 = (cand + [None, None])[:2]
    if c1 is not None and (c1["nroots"] > c0["nroots"]
                           or (c1["nroots"] == c0["nroots"] and c1["fcr"] < c0["fcr"])
                           or (c1["nroots"] == c0["nroots"] and c1["fcr"] == c0["fcr"]
                               and c1["lsb_first"] < c0["lsb_first"])):
        c0 = c1
    if c0 is None:
        return out
    mm, L = c0["m"], c0["n"]
    NN = (1 << mm) - 1
    # half a codeword along, a code's roots vanish; structure that keeps them
    # is not codeword-aligned
    off2 = offset + (L // 2) * mm
    fit2 = (n - off2) // (L * mm) if off2 < n else 0
    R2 = min(fit2, 1024)
    kept = 0
    if R2 >= 4:
        f = c0["gfpoly"]
        expt = np.zeros(2 * NN, np.int64)
        logt = np.zeros(1 << mm, np.int64)
        x = 1
        for i in range(NN):
            expt[i] = x
            logt[x] = i
            x <<= 1
            if x >> mm:
                x ^= f
        expt[NN:] = expt[:NN]
        sym = _rs_syms(b, off2, R2, L, mm, c0["lsb_first"])
        if c0["dual_basis"]:
            sym = tal1[sym]
        cnt = _rs_counts(sym, NN, expt, logt)
        ltr = _m.log10(NN * nrep)
        for i in range(c0["nroots"]):
            e = (c0["prim"] * (c0["fcr"] + i)) % NN
            if log10_binom_tail(R2, int(cnt[e]), -mm * _LN2) + ltr <= thr:
                kept += 1
    if kept:
        out["name"] = "no code found (not codeword-aligned)"
        return out
    out.update(c0)
    rs = RS(mm, out["gfpoly"], out["fcr"], out["prim"], out["nroots"], NN - L, bool(out["dual_basis"]))
    R = min((n - offset) // (L * mm), ID_ROWS)
    sym = _rs_syms(b, offset, R, L, mm, out["lsb_first"])
    ok = corr = 0
    for r in range(R):
        fixed, c, _p = rs.decode(sym[r])
        if c >= 0:
            ok += 1
            corr += int(np.sum(np.asarray(fixed, np.int64) != sym[r]))
    out["rows_used"] = R
    out["valid_words"] = ok / R if R else float("nan")
    if ok:
        out["agreement"] = 1.0 - corr / (ok * L)
    if (mm == 8 and out["gfpoly"] == 0x187 and out["fcr"] == 112 and out["prim"] == 11
            and out["nroots"] == 32 and out["dual_basis"]):
        out["name"] = f"CCSDS RS({L},{out['k']}) dual basis"
    else:
        out["name"] = (f"RS({L},{out['k']}) GF(2^{mm}) 0x{out['gfpoly']:X} fcr {out['fcr']} "
                       f"prim {out['prim']}{' dual basis' if out['dual_basis'] else ''}")[:47]
    out["found"] = True
    return out


# ---- block interleavers -------------------------------------------------------------
def _ilv_blocks(N, wmax, robust):
    need = (wmax + 2 + 256 + 1) * wmax if robust else (wmax + 34) * wmax
    return max(64, (need + N - 1) // N + 1)


def _inner_width(y, wmax, wstop, robust, seed, pfa):
    """(smallest width with a significant deficiency, its deficiency), or (0, 0)."""
    thr = _m.log10(pfa)
    ny = y.size
    w = 2
    while w <= wmax and w < wstop:
        fit = ny // w
        d, l = 0, 0.0
        if not robust:
            R = w + 32
            if R + 1 > fit:
                R = fit - 1
            if R < w + 16:
                break
            for k4 in range(4):
                if d > 0 and l <= thr:
                    break
                o = (k4 * w) // 4
                if k4 and o == ((k4 - 1) * w) // 4:
                    continue
                d = w - gf2_rank(_rows(y, o, w, R, w))
                l = _def_l10(d, R, w) + _m.log10(4.0 * wmax) if d > 0 else 0.0
        else:
            Rr = min(fit, w + 2 + 1024)
            if Rr < w + 2 + ROBUST_VMIN:
                break
            basis, info = gf2_robust(_rows(y, 0, w, Rr, w), w, (seed + w) & _M64, pfa, w, 6)
            d = len(basis)
            l = info["log10_pfa"] + _m.log10(wmax) if d > 0 else 0.0
        if d > 0 and l <= thr:
            return w, d
        w += 1
    return 0, 0


def _deinterleave(b, st, cap, N, r):
    c = N // r
    blk = b[st:st + cap * N].reshape(cap, c, r)      # src index col*r + row
    return (blk.transpose(0, 2, 1).reshape(-1) & 1).astype(np.uint8)   # dst row*c + col


def interleaver_id(b, n_lo: int = 8, n_hi: int = 512, pfa: float = 1e-3, seed: int = 0) -> dict:
    """The twin of atkdsp_interleaver_id (header §18)."""
    b = _bits(b)
    n = b.size
    if not (4 <= n_lo <= n_hi <= 1024):
        raise ValueError("interleaver_id: 4 <= n_lo <= n_hi <= 1024")
    if not pfa > 0:
        pfa = 1e-3
    out = {"period": 0, "offset": 0, "rows": 0, "cols": 0, "inner_width": 0,
           "inner_deficiency": 0, "log10_pfa": 0.0, "found": False}
    if _stream_unfit(b)[0]:
        return out
    found, f = _block_search(b, n, n_lo, n_hi, pfa, seed, 3)
    if not found or f["slide"]:
        return out
    P, rob = f["N"], f["robust"]
    out.update(period=P, offset=f["offset"], log10_pfa=f["log10_pfa"])
    # the raw stream's own inner width: the de-interleaved code must be
    # at least twice as local
    w_raw = 0
    wmax = min(P, 128)
    blocks = (n - f["offset"]) // P
    cap = min(_ilv_blocks(P, wmax, rob), blocks)
    if cap >= 2:
        w_raw, _d = _inner_width(b[f["offset"]:f["offset"] + cap * P], wmax, wmax + 1, rob,
                                 seed ^ 0x1D, pfa)
    wlimit = w_raw // 2 + 1 if w_raw else 1 << 20
    best_w = best_d = 0
    for ps in (1, 0):
        Ns = (range(P, min(n_hi, 4 * P) + 1, P) if ps
              else [N for N in range(4, P) if P % N == 0 and N >= n_lo])
        ties = f["ties"] or [f["offset"]]
        for N in Ns:
            m = N // P
            nst = len(ties) * m if ps else (len(ties) if rob else N)
            for j in range(nst):
                st = ties[j // m] + (j % m) * P if ps else (ties[j] if rob else j)
                wmax = min(max(N, P), 128)
                blocks = (n - st) // N if st < n else 0
                cap = min(_ilv_blocks(N, wmax, rob), blocks)
                if cap < 2:
                    break
                for r in range(2, N // 2 + 1):
                    if N % r or N // r < 2:
                        continue
                    y = _deinterleave(b, st, cap, N, r)
                    stop = min(best_w + 1, wlimit) if best_w else wlimit
                    w, dd = _inner_width(y, wmax, stop, rob, (seed + r * 4096) & _M64, pfa)
                    if w > 0 and (not best_w or w < best_w or dd > best_d):
                        best_w, best_d = w, dd
                        out.update(rows=r, cols=N // r, period=N, offset=st % N)
    out["inner_width"], out["inner_deficiency"] = best_w, best_d
    if not best_w:
        out["rows"] = out["cols"] = 0
        return out
    out["found"] = True
    return out


# ============================================================================
# THE DETECTION TRACK (v0.16.0, ABI 17) — docs/ATKDSP_PLAN.md §B1, §B2, §A3
# ============================================================================
# §19 CFAR and the burst tracker, §20 the burst descriptor word, §21 the ring.
# Each is the specification src/cfar.c, src/burst.c and src/ring.c are held to.

# ---- §19 helpers: the incomplete gamma in log space, both tails ---------------
def log_igam_pq(a, x):
    """(ln P(a, x), ln Q(a, x)) — both regularised incomplete gammas, each
    finite far into its own tail. The arithmetic of src/stat.c's
    atk_log_igam_pq, step for step."""
    if not x > 0.0:
        return -_m.inf, 0.0
    lg = _m.lgamma(a)
    if x < a + 1.0:
        ap, dl = a, 1.0 / a
        s = dl
        for _ in range(_igam_iters(a)):
            ap += 1.0
            dl *= x / ap
            s += dl
            if abs(dl) < abs(s) * 1e-16:
                break
        lnp = _m.log(s) - x + a * _m.log(x) - lg
        p = _m.exp(lnp)
        return lnp, (_m.log1p(-p) if p < 1.0 else -_m.inf)
    lnq = -x + a * _m.log(x) - lg + _gamma_q_cf_log(a, x)
    q = _m.exp(lnq)
    return (_m.log1p(-q) if q < 1.0 else -_m.inf), lnq


def _v_log_igam_pq(a, x):
    """log_igam_pq over an array of x (one shape a): the same series and
    continued fraction, each element stopping where the scalar loop would."""
    x = np.asarray(x, dtype=np.float64)
    lnp = np.full(x.shape, -np.inf)
    lnq = np.zeros(x.shape)
    lg = _m.lgamma(a)
    pos = x > 0.0
    ser = pos & (x < a + 1.0)
    cf = pos & ~ser
    if ser.any():
        xs = x[ser]
        dl = np.full(xs.shape, 1.0 / a)
        s = dl.copy()
        act = np.ones(xs.shape, bool)
        ap = a
        for _ in range(_igam_iters(a)):
            ap += 1.0
            dl[act] *= xs[act] / ap
            s[act] += dl[act]
            act &= ~(np.abs(dl) < np.abs(s) * 1e-16)
            if not act.any():
                break
        lp = np.log(s) - xs + a * np.log(xs) - lg
        p = np.exp(lp)
        lnp[ser] = lp
        with np.errstate(divide="ignore"):
            lnq[ser] = np.where(p < 1.0, np.log1p(-np.minimum(p, 1.0)), -np.inf)
    if cf.any():
        xs = x[cf]
        tiny = 1e-300
        b = xs + 1.0 - a
        c = np.full(xs.shape, 1.0 / tiny)
        d = 1.0 / b
        h = d.copy()
        act = np.ones(xs.shape, bool)
        for i in range(1, _igam_iters(a)):
            an = -float(i) * (float(i) - a)
            b = b + 2.0
            dn = an * d + b
            dn = np.where(np.abs(dn) < tiny, tiny, dn)
            cn = b + an / c
            cn = np.where(np.abs(cn) < tiny, tiny, cn)
            dn = 1.0 / dn
            dl = dn * cn
            d = np.where(act, dn, d)
            c = np.where(act, cn, c)
            h = np.where(act, h * dl, h)
            act &= ~(np.abs(dl - 1.0) < 1e-16)
            if not act.any():
                break
        lq = -xs + a * np.log(xs) - lg + np.log(h)
        q = np.exp(lq)
        lnq[cf] = lq
        with np.errstate(divide="ignore"):
            lnp[cf] = np.where(q < 1.0, np.log1p(-np.minimum(q, 1.0)), -np.inf)
    return lnp, lnq


# ---- §19 CFAR -------------------------------------------------------------------
CFAR_CA, CFAR_OS, CFAR_GO, CFAR_SO = 0, 1, 2, 3
CFAR_MODE_NAMES = ("CA", "OS", "GO", "SO")
CFAR_OPEN, CFAR_UPDATE, CFAR_CLOSE = 1, 2, 3
CFAR_T0_COARSE, CFAR_T1_COARSE, CFAR_EDGE, CFAR_MASK_ADJ, CFAR_FLUSHED = 1, 2, 4, 8, 16
#: The single-frame band test that places t0 / t1 inside the k-frame average
#: runs only inside a confirmed track, over at most k frames, so its rate is
#: a per-track chance of an edge one frame early (§19 in atkdsp.h).
CFAR_EDGE_PFA = 1e-3
_K10 = _m.log(10.0) / 10.0          # dB -> natural log of power


def cfar_defaults() -> dict:
    """The defaults atkdsp_cfar_defaults() writes."""
    return dict(mode=CFAR_CA, n_ref=16, n_guard=4, k_frames=1, pfa=1e-6, looks=1.0,
                os_k=0, open_m=2, open_n=3, close_k=3, merge_gap=1, min_bins=1,
                max_tracks=256)


def _cfar_check(nbins, c) -> str:
    """'' when the configuration is valid, else the reason (the C returns NULL)."""
    N = 2 * c["n_ref"]
    if c["mode"] not in (0, 1, 2, 3):
        return "mode must be CA, OS, GO or SO"
    if not 1 <= c["n_ref"] <= 256 or not 0 <= c["n_guard"] <= 256:
        return "n_ref 1..256 and n_guard 0..256"
    if nbins < N + 2 * c["n_guard"] + 1 or nbins > (1 << 24) or nbins * N > (1 << 26):
        return "nbins must hold one full window (2 n_ref + 2 n_guard + 1) and nbins x 2 n_ref <= 2^26"
    if not 1 <= c["k_frames"] <= 64 or not 0.25 <= c["looks"] <= 1e4:
        return "k_frames 1..64 and looks 0.25..1e4"
    if not 1e-15 <= c["pfa"] <= 0.5:
        return "pfa 1e-15..0.5"
    if c["mode"] == CFAR_OS and not 0 <= c["os_k"] <= N:
        return "os_k 1..2 n_ref (0 = three quarters)"
    if not 1 <= c["open_n"] <= 64 or not 1 <= c["open_m"] <= c["open_n"]:
        return "open_m 1..open_n and open_n 1..64"
    if not 1 <= c["close_k"] <= 1000 or not 0 <= c["merge_gap"] <= nbins:
        return "close_k 1..1000 and merge_gap 0..nbins"
    if not 1 <= c["min_bins"] <= nbins or not 1 <= c["max_tracks"] <= 65536:
        return "min_bins 1..nbins and max_tracks 1..65536"
    return ""


def _ln_gpdf(s, t):
    """ln of the Gamma(s) density with mean 1 (scale 1/s) at t (> 0)."""
    return s * _m.log(s) + (s - 1.0) * np.log(t) - s * t - _m.lgamma(s)


def _cfar_log_density(mode, t, N, K, k):
    """ln density of the reference statistic T (noise mean 1 per cell) at t."""
    n = N // 2
    if mode == CFAR_CA:
        return _ln_gpdf(N * K, t)
    if mode in (CFAR_GO, CFAR_SO):
        s = n * K
        lp, lq = _v_log_igam_pq(s, s * t)
        return _m.log(2.0) + (lp if mode == CFAR_GO else lq) + _ln_gpdf(s, t)
    lp, lq = _v_log_igam_pq(K, K * t)
    out = (_m.log(N) + _m.lgamma(N) - _m.lgamma(k) - _m.lgamma(N - k + 1)
           + _ln_gpdf(K, t))
    if k > 1:
        out = out + (k - 1) * lp
    if N > k:
        out = out + (N - k) * lq
    return out


def _cfar_log_cdf_sf(mode, t, N, K, k):
    """(ln P(T <= t), ln P(T > t)) — the tails that bound the integration."""
    n = N // 2
    if mode == CFAR_CA:
        return log_igam_pq(N * K, N * K * t)
    if mode in (CFAR_GO, CFAR_SO):
        s = n * K
        lp, lq = log_igam_pq(s, s * t)
        if mode == CFAR_GO:
            return 2.0 * lp, lq + _m.log1p(_m.exp(lp))
        return lp + _m.log1p(_m.exp(lq)), 2.0 * lq
    lp, lq = log_igam_pq(K, K * t)
    cdf = -_m.inf if lp == -_m.inf else log10_binom_tail(N, k, lp) * _LN10
    sf = -_m.inf if lq == -_m.inf else log10_binom_tail(N, N - k + 1, lq) * _LN10
    return cdf, sf


def _cfar_grid(mode, N, K, k, pfa):
    """The integration grid: u = ln t over [u_lo, u_hi] and the Simpson-weighted
    density G_j (sum G_j = 1 - the two tails), plus E[T]. The tails cut off
    are each below 1e-9 x pfa, so truncation moves Pfa by < 2e-9 of itself."""
    ln_eps = _m.log(1e-9 * pfa)
    lo, hi = -700.0, 7.0
    if _cfar_log_cdf_sf(mode, _m.exp(lo), N, K, k)[0] >= ln_eps:
        u_lo = lo
    else:
        a, b = lo, hi
        for _ in range(60):
            mid = 0.5 * (a + b)
            if _cfar_log_cdf_sf(mode, _m.exp(mid), N, K, k)[0] < ln_eps:
                a = mid
            else:
                b = mid
        u_lo = a
    if _cfar_log_cdf_sf(mode, _m.exp(hi), N, K, k)[1] > ln_eps:
        u_hi = hi
    else:
        a, b = u_lo, hi
        for _ in range(60):
            mid = 0.5 * (a + b)
            if _cfar_log_cdf_sf(mode, _m.exp(mid), N, K, k)[1] > ln_eps:
                a = mid
            else:
                b = mid
        u_hi = b
    s_ref = (N // 2) * K if mode in (CFAR_GO, CFAR_SO) else N * K
    h0 = min(0.01, 0.2 / _m.sqrt(s_ref))
    span = u_hi - u_lo
    M = 2 * int(_m.ceil(span / (2.0 * h0)))
    M = min(max(M, 512), 131072)
    h = span / M
    u = u_lo + h * np.arange(M + 1, dtype=np.float64)
    t = np.exp(u)
    w = np.full(M + 1, 2.0)
    w[1::2] = 4.0
    w[0] = w[M] = 1.0
    w *= h / 3.0
    with np.errstate(over="ignore", invalid="ignore"):
        lg = _cfar_log_density(mode, t, N, K, k) + u
        G = np.where(np.isfinite(lg), np.exp(np.where(np.isfinite(lg), lg, 0.0)), 0.0) * w
    ET = 0.0
    for j in range(M + 1):            # sequential, as the C sums
        ET += G[j] * t[j]
    return t, G, ET


def _cfar_pfa(t, G, K, alpha):
    """Pfa(alpha) = sum_j G_j Q(K, K alpha t_j), summed in index order."""
    q = np.exp(_v_log_igam_pq(K, K * alpha * t)[1])
    v = G * q
    s = 0.0
    for x in v:
        s += x
    return s


def cfar_alpha(mode, n_ref, pfa, K=1.0, os_k=0):
    """The threshold multiplier for `pfa` per cell: (alpha, E[T]). T is the
    reference statistic in units of the noise mean (CA: the mean of N = 2
    n_ref cells; GO / SO: the greater / smaller of the two side means; OS:
    the os_k-th smallest of the N cells); the cell under test is K-look
    gamma. Pfa(alpha) = integral Q(K, K alpha t) f_T(t) dt, solved for alpha
    by regula falsi (Illinois) in ln alpha. K = 1 reproduces the closed forms:
    CA alpha = N (Pfa^(-1/N) - 1); OS Pfa = prod_{i<k} (N-i)/(N-i+alpha)."""
    N = 2 * int(n_ref)
    k = int(os_k) or (3 * N + 3) // 4
    t, G, ET = _cfar_grid(mode, N, float(K), k, float(pfa))
    lnpfa = _m.log(pfa)

    def f(v):
        return _m.log(max(_cfar_pfa(t, G, float(K), _m.exp(v)), 1e-300)) - lnpfa
    a, b = _m.log(1e-3), _m.log(1e18)
    fa, fb = f(a), f(b)
    if not (fa > 0.0 and fb < 0.0):
        raise ValueError("pfa cannot be reached with alpha in 1e-3..1e18 — "
                         "use more reference cells")
    side, c, prev = 0, a, None
    for _ in range(300):
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        if fc == 0.0:
            break
        if fc > 0.0:
            a, fa = c, fc
            if side == 1:
                fb *= 0.5
            side = 1
        else:
            b, fb = c, fc
            if side == -1:
                fa *= 0.5
            side = -1
        if prev is not None and abs(c - prev) <= 1e-14 * max(1.0, abs(c)):
            break
        prev = c
    return _m.exp(c), ET


def cfar_edge_beta(nb, looks):
    """beta with P(Gamma(nb*looks) / (nb*looks) > beta) = CFAR_EDGE_PFA: the
    single-frame band test for nb bins. Bisection in ln beta, 100 steps."""
    s = float(nb) * float(looks)
    target = _m.log(CFAR_EDGE_PFA)
    a, b = _m.log(1e-6), _m.log(1e6)
    for _ in range(100):
        mid = 0.5 * (a + b)
        if log_igam_pq(s, s * _m.exp(mid))[1] > target:
            a = mid
        else:
            b = mid
    return _m.exp(b)


def _popcount(x: int) -> int:
    return bin(x).count("1")


class Cfar:
    """Twin of atkdsp_cfar (atkdsp.h §19): a 1-D CFAR across frequency on each
    frame and a burst tracker across frames. Frames are power spectra in dB,
    one row per frame, AVERAGE power per bin (a max-hold frame is not gamma
    and breaks the false-alarm rate). Events are dicts with the C struct's
    fields; see atkdsp.h §19 for every definition."""

    def __init__(self, nbins: int, **cfg):
        c = cfar_defaults()
        for key, v in cfg.items():
            if key not in c:
                raise ValueError(f"unknown CFAR option {key!r}")
            c[key] = v
        for key in c:
            c[key] = float(c[key]) if key in ("pfa", "looks") else int(c[key])
        why = _cfar_check(int(nbins), c)
        if why:
            raise ValueError(f"CFAR refused: {why}")
        self.nbins = int(nbins)
        self.cfg = c
        N = 2 * c["n_ref"]
        if c["mode"] == CFAR_OS and c["os_k"] == 0:
            c["os_k"] = (3 * N + 3) // 4
        self.K = c["looks"] * c["k_frames"]
        self.alpha, self.et = cfar_alpha(c["mode"], c["n_ref"], c["pfa"], self.K, c["os_k"])
        if c["mode"] in (CFAR_GO, CFAR_SO):
            self.alpha_ca, _ = cfar_alpha(CFAR_CA, c["n_ref"], c["pfa"], self.K)
        else:
            self.alpha_ca = self.alpha
        self._beta = {}
        self.mask = np.zeros(self.nbins, bool)
        self._tables()
        k = c["k_frames"]
        self._hist = np.zeros((k, self.nbins))
        self._ht = np.zeros(k)
        self._hd = np.zeros(k)
        self._seen = 0
        self._frame = 0                      # global frame index
        self._tracks = []
        self._next_id = 1
        self._last = None
        self._st = dict(frames=0, warmup=0, cells_tested=0, cell_hits=0, untestable=0,
                        opened=0, closed=0, updates=0, dropped=0, resets=0)

    # -- configuration --------------------------------------------------------
    def set_mask(self, mask) -> None:
        m = np.asarray(mask)
        if m.size != self.nbins:
            raise ValueError("mask must have nbins entries")
        self.mask = m.astype(bool).ravel().copy()
        self._tables()

    def _tables(self) -> None:
        """Per bin: the N reference cells (left nearest-first, right
        nearest-first, then any cells borrowed from the far side at an edge
        or past a mask), whether it is testable, and whether both sides were
        whole (GO / SO need that; a shifted bin falls back to CA)."""
        nb, nr, ng = self.nbins, self.cfg["n_ref"], self.cfg["n_guard"]
        N = 2 * nr
        idx = np.zeros((nb, N), np.int64)
        ok = np.zeros(nb, bool)
        sided = np.zeros(nb, bool)
        m = self.mask
        for i in range(nb):
            if m[i]:
                continue
            left, right = [], []
            j = i - ng - 1
            while j >= 0 and len(left) < nr:
                if not m[j]:
                    left.append(j)
                j -= 1
            jl = j
            j = i + ng + 1
            while j < nb and len(right) < nr:
                if not m[j]:
                    right.append(j)
                j += 1
            jr = j
            ext = []
            short_l, short_r = nr - len(left), nr - len(right)
            j = jr
            while short_l > 0 and j < nb:
                if not m[j]:
                    ext.append(j)
                    short_l -= 1
                j += 1
            j = jl
            while short_r > 0 and j >= 0:
                if not m[j]:
                    ext.append(j)
                    short_r -= 1
                j -= 1
            cells = left + right + ext
            if len(cells) < N:
                continue
            idx[i] = cells
            ok[i] = True
            sided[i] = not ext
        self._idx, self._ok, self._sided = idx, ok, sided

    def _beta_for(self, nb: int) -> float:
        if nb not in self._beta:
            self._beta[nb] = cfar_edge_beta(nb, self.cfg["looks"])
        return self._beta[nb]

    # -- the per-frame test ---------------------------------------------------
    def _reference(self, X):
        c = self.cfg
        N, n = 2 * c["n_ref"], c["n_ref"]
        idx = self._idx
        mode = c["mode"]
        alpha = np.full(self.nbins, self.alpha)
        noise_scale = np.full(self.nbins, self.et)
        if mode == CFAR_OS:
            Xr = X[idx]
            Z = np.partition(Xr, c["os_k"] - 1, axis=1)[:, c["os_k"] - 1]
            Z = np.where(np.isnan(Xr).any(axis=1), np.nan, Z)     # a NaN cell: T is NaN
        else:
            Z = np.zeros(self.nbins)
            for col in range(N):
                Z += X[idx[:, col]]
            Z /= N
            if mode in (CFAR_GO, CFAR_SO):
                L = np.zeros(self.nbins)
                R = np.zeros(self.nbins)
                for col in range(n):
                    L += X[idx[:, col]]
                for col in range(n, N):
                    R += X[idx[:, col]]
                L /= n
                R /= n
                side = np.maximum(L, R) if mode == CFAR_GO else np.minimum(L, R)
                s = self._sided
                Z = np.where(s, side, Z)
                alpha = np.where(s, self.alpha, self.alpha_ca)
                noise_scale = np.where(s, self.et, 1.0)
        return Z, alpha, noise_scale

    def process(self, frames_db, frame_t0: float, frame_dt: float) -> list:
        F = np.asarray(frames_db, dtype=np.float32)
        if F.ndim == 1:
            F = F.reshape(1, -1)
        if F.shape[1] != self.nbins:
            raise ValueError("frames must have nbins columns")
        out = []
        for j in range(F.shape[0]):
            out += self._one(F[j], float(frame_t0) + j * float(frame_dt), float(frame_dt))
        return out

    def _one(self, row, tf, dt) -> list:
        c = self.cfg
        k = c["k_frames"]
        st = self._st
        st["frames"] += 1
        raw = np.exp(np.maximum(row.astype(np.float64), -300.0) * _K10)
        slot = self._frame % k
        self._hist[slot] = raw
        self._ht[slot] = tf
        self._hd[slot] = dt
        self._seen += 1
        F = self._frame
        self._frame += 1
        if self._seen < k:
            st["warmup"] += 1
            return []
        X = np.zeros(self.nbins)
        for r in range(k):
            X += self._hist[(F + 1 + r) % k]
        X /= k
        Z, alpha, scale = self._reference(X)
        ok = self._ok
        thr = alpha * Z
        hits = ok & (X > thr)
        noise = Z / scale
        st["cells_tested"] += int(ok.sum())
        st["cell_hits"] += int(hits.sum())
        st["untestable"] += int((~ok & ~self.mask).sum())
        self._last = (X, np.where(ok, thr, np.nan), hits)
        # blobs
        blobs = []
        hi_idx = np.flatnonzero(hits)
        g = c["merge_gap"]
        if hi_idx.size:
            start = prev = int(hi_idx[0])
            members = [start]
            for b in hi_idx[1:]:
                b = int(b)
                if b - prev - 1 <= g:
                    members.append(b)
                else:
                    blobs.append(members)
                    members = [b]
                prev = b
            blobs.append(members)
        obs = []
        for mem in blobs:
            lo, hi = mem[0], mem[-1]
            if hi - lo + 1 < c["min_bins"]:
                continue
            mb = np.asarray(mem)
            e = np.maximum(X[mb] - noise[mb], 0.0)
            sbe = se = sx = sn = 0.0
            for q in range(mb.size):
                sbe += float(mb[q]) * e[q]
                se += e[q]
                sx += X[mb[q]]
                sn += noise[mb[q]]
            p = int(mb[int(np.argmax(X[mb]))])
            obs.append(dict(lo=lo, hi=hi, sbe=sbe, se=se, sx=sx, sn=sn,
                            px=float(X[p]), pn=float(noise[p])))
        # association
        m = g + 1
        assign = {}
        newb = []
        for o in obs:
            best, bov = None, None
            for tr in self._tracks:
                if o["lo"] <= tr["cur_hi"] + m and o["hi"] >= tr["cur_lo"] - m:
                    ov = min(o["hi"], tr["cur_hi"]) - max(o["lo"], tr["cur_lo"]) + 1
                    if bov is None or ov > bov:
                        best, bov = tr, ov
            if best is None:
                newb.append(o)
            else:
                assign.setdefault(best["id"], []).append(o)
        for o in newb:
            if len(self._tracks) >= c["max_tracks"]:
                st["dropped"] += 1
                continue
            tr = dict(id=self._next_id, state=0, hist=0, misses=0, n_hits=0,
                      cur_lo=o["lo"], cur_hi=o["hi"], ulo=o["lo"], uhi=o["hi"],
                      sbe=0.0, se=0.0, px=-1.0, pn=0.0, band=0.0, flags=0,
                      t0=0.0, last_hit=-1, t_hit_end=0.0, dt_hit=dt,
                      last_raw=-1, t_raw_end=0.0, new=True)
            self._next_id = (self._next_id + 1) & 0xFFFFFFFF or 1
            self._tracks.append(tr)
            assign[tr["id"]] = [o]
        # update
        events = []
        keep = []
        maskn = (1 << c["open_n"]) - 1
        for tr in self._tracks:
            ob = assign.get(tr["id"])
            grew = False
            if ob:
                lo = min(o["lo"] for o in ob)
                hi = max(o["hi"] for o in ob)
                tr["hist"] = ((tr["hist"] << 1) | 1) & maskn
                tr["misses"] = 0
                tr["n_hits"] += 1
                tr["cur_lo"], tr["cur_hi"] = lo, hi
                if not tr["new"]:
                    grew = lo < tr["ulo"] or hi > tr["uhi"]
                tr["ulo"], tr["uhi"] = min(tr["ulo"], lo), max(tr["uhi"], hi)
                sx = sn = 0.0
                for o in ob:
                    tr["sbe"] += o["sbe"]
                    tr["se"] += o["se"]
                    sx += o["sx"]
                    sn += o["sn"]
                    if o["px"] > tr["px"]:
                        tr["px"], tr["pn"] = o["px"], o["pn"]
                if sn > 0.0 and sx / sn > tr["band"]:
                    tr["band"] = sx / sn
                tr["last_hit"] = F
                tr["t_hit_end"] = tf + dt
                tr["dt_hit"] = dt
                if tr["new"]:
                    tr["new"] = False
                    if k == 1:
                        tr["t0"] = tf
                    else:
                        tr["t0"] = None
                        for r in range(k):
                            s = (F + 1 + r) % k
                            if self._band_pass(self._hist[s], noise, lo, hi):
                                tr["t0"] = float(self._ht[s])
                                break
                        if tr["t0"] is None:
                            tr["t0"] = tf - 0.5 * (k - 1) * dt
                            tr["flags"] |= CFAR_T0_COARSE
            else:
                tr["hist"] = (tr["hist"] << 1) & maskn
                tr["misses"] += 1
            if k > 1 and self._band_pass(raw, noise, tr["cur_lo"], tr["cur_hi"]):
                tr["last_raw"] = F
                tr["t_raw_end"] = tf + dt
            if tr["state"] == 0:
                if _popcount(tr["hist"]) >= c["open_m"]:
                    tr["state"] = 1
                    st["opened"] += 1
                    events.append(self._event(CFAR_OPEN, tr))
                elif tr["hist"] == 0:
                    continue
            elif tr["misses"] >= c["close_k"]:
                events.append(self._close(tr, 0))
                continue
            elif grew:
                st["updates"] += 1
                events.append(self._event(CFAR_UPDATE, tr))
            keep.append(tr)
        self._tracks = keep
        return events

    def _band_pass(self, raw, noise, lo, hi) -> bool:
        sr = sz = 0.0
        nb = 0
        for b in range(lo, hi + 1):
            if self._ok[b]:
                sr += raw[b]
                sz += noise[b]
                nb += 1
        return nb > 0 and sr > self._beta_for(nb) * sz

    def _flags(self, tr) -> int:
        f = tr["flags"]
        nb = self.nbins
        lo, hi = tr["ulo"], tr["uhi"]
        if not self._sided[lo:hi + 1].all() or not self._ok[lo:hi + 1].all():
            f |= CFAR_EDGE
        m = self.cfg["merge_gap"] + 1
        if self.mask[max(0, lo - m):min(nb, hi + m + 1)].any():
            f |= CFAR_MASK_ADJ
        return f

    def _event(self, typ, tr, t1=float("nan"), extra=0) -> dict:
        fc = float(tr["sbe"] / tr["se"]) if tr["se"] > 0.0 else 0.5 * (tr["ulo"] + tr["uhi"])
        lv = float(np.float32(10.0 * _m.log10(tr["px"]))) if tr["px"] > 0 else float("-inf")
        nz = float(np.float32(10.0 * _m.log10(tr["pn"]))) if tr["pn"] > 0 else float("-inf")
        return dict(type=typ, id=tr["id"], flags=self._flags(tr) | extra,
                    lo=tr["ulo"], hi=tr["uhi"], n_hits=tr["n_hits"], t0=tr["t0"], t1=t1,
                    fc_bin=fc, level_db=lv, noise_db=nz,
                    snr_db=float(np.float32(lv - nz)) if tr["pn"] > 0 else float("inf"),
                    band_snr_db=float(np.float32(10.0 * _m.log10(tr["band"])))
                    if tr["band"] > 0 else float("-inf"))

    def _close(self, tr, extra) -> dict:
        k = self.cfg["k_frames"]
        if k == 1:
            t1 = tr["t_hit_end"]
        elif tr["last_raw"] >= tr["last_hit"] - (k - 1):
            t1 = tr["t_raw_end"]
        else:
            t1 = tr["t_hit_end"] - 0.5 * (k - 1) * tr["dt_hit"]
            tr["flags"] |= CFAR_T1_COARSE
        self._st["closed"] += 1
        return self._event(CFAR_CLOSE, tr, t1, extra)

    def flush(self) -> list:
        """Close every open track now (CLOSE, flag FLUSHED); drop the pending
        ones. The stream may continue."""
        ev = [self._close(tr, CFAR_FLUSHED) for tr in self._tracks if tr["state"] == 1]
        self._tracks = []
        return ev

    def reset(self) -> None:
        self._tracks = []
        self._seen = 0
        self._frame = 0
        self._hist[:] = 0.0
        self._last = None
        self._st["resets"] += 1

    def last(self):
        """(averaged power dB, threshold dB (NaN untested), hits) of the last
        tested frame, or None before one."""
        if self._last is None:
            return None
        X, thr, hits = self._last
        with np.errstate(divide="ignore", invalid="ignore"):
            return ((10.0 * np.log10(X)).astype(np.float32),
                    (10.0 * np.log10(thr)).astype(np.float32), hits.astype(np.uint8))

    def events_max(self, nframes: int) -> int:
        return int(nframes) * self.cfg["max_tracks"]

    def stats(self) -> dict:
        d = dict(self._st)
        d["tracks"] = len(self._tracks)
        return d

    def describe(self) -> str:
        return cfar_describe(self.nbins, self.cfg, self.alpha, int(self.mask.sum()),
                             len(self._tracks))


def cfar_describe(nbins, c, alpha, masked, ntracks) -> str:
    mode = CFAR_MODE_NAMES[c["mode"]]
    os_txt = f" (rank {c['os_k']} of {2 * c['n_ref']})" if c["mode"] == CFAR_OS else ""
    return (f"CFAR {mode}{os_txt}: {c['n_ref']}+{c['n_ref']} reference cells, "
            f"{c['n_guard']} guard each side, Pfa {c['pfa']:.3g} per cell per frame, "
            f"K = {c['looks']:g} look(s) x {c['k_frames']} frame(s), alpha {alpha:.4g} "
            f"({10.0 * _m.log10(alpha):.2f} dB); open {c['open_m']} of {c['open_n']}, "
            f"close after {c['close_k']} misses, merge gap {c['merge_gap']}, "
            f"min {c['min_bins']} bin(s); {nbins} bins, {masked} masked, "
            f"tracks {ntracks}/{c['max_tracks']}")


# ---- §20 the burst descriptor word ---------------------------------------------------
BDW_TRUNC_START, BDW_TRUNC_END, BDW_NO_NOISE, BDW_LOW_SNR = 1, 2, 4, 8
BDW_RISE_UNRESOLVED, BDW_FALL_UNRESOLVED, BDW_BW26_FLOOR, BDW_SHORT = 16, 32, 64, 128
BDW_GAUSSIAN = 256
BDW_DEFAULT_THRESH_DB = 3.0102999566398120     # half power, when thresh_db > 0 is asked for without N


def _bdw_hann(n):
    """The periodic (DFT-even) Hann window both implementations use."""
    j = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * j / n)


def _seq_mean(v) -> float:
    s = 0.0
    for x in v:
        s += x
    return s / len(v)


def _bdw_welch(x, nfft):
    """Hann, hop nfft/2, |FFT|^2 averaged over whole segments, shifted so
    index 0 is -fs/2. Returns (psd, segments, sum w^2)."""
    w = _bdw_hann(nfft)
    hop = nfft // 2
    acc = np.zeros(nfft)
    nseg = 0
    s = 0
    while s + nfft <= x.size:
        X = np.fft.fft(x[s:s + nfft] * w)
        acc += X.real * X.real + X.imag * X.imag
        nseg += 1
        s += hop
    return np.fft.fftshift(acc / max(nseg, 1)), nseg, float(np.sum(w * w))


def _bdw_obw(D, fs, nfft, frac=0.99):
    """The occupied band of a noise-subtracted PSD D (shifted, SIGNED: the
    residual of the noise is left to cancel, because clipping it at zero
    adds a positive bias of every bin's fluctuation, and at 10 dB that alone
    is more than the 0.5 % tails). The lower edge is where the power summed
    from the bottom reaches (1 - frac)/2 of the total, the upper where the
    sum from the top does, each interpolated in its bin: (f_lo, f_hi, k_lo,
    k_hi), or None when there is no power above the noise."""
    df = fs / nfft
    T = 0.0
    for v in D:
        T += v
    if not T > 0.0:
        return None
    tail = 0.5 * (1.0 - frac) * T
    cum = 0.0
    f_lo = None
    k_lo = 0
    for k in range(nfft):
        prev = cum
        cum += D[k]
        if cum >= tail and D[k] > 0.0:
            f_lo = (k - nfft // 2) * df - 0.5 * df + df * (tail - prev) / D[k]
            k_lo = k
            break
    cum = 0.0
    f_hi = None
    k_hi = nfft - 1
    for k in range(nfft - 1, -1, -1):
        prev = cum
        cum += D[k]
        if cum >= tail and D[k] > 0.0:
            f_hi = (k - nfft // 2) * df + 0.5 * df - df * (tail - prev) / D[k]
            k_hi = k
            break
    if f_lo is None or f_hi is None or k_hi < k_lo:
        return None
    return f_lo, f_hi, k_lo, k_hi


def _bdw_auto_nfft(L):
    """The largest power of two <= L / 8, in 16..4096: eight or more
    half-overlapped segments, so the Hann taper at the ends weighs little."""
    m = 16
    while m * 2 <= L // 8 and m * 2 <= 4096:
        m *= 2
    return m


def _bdw_runs(ps, lvl, gap):
    """Runs of ps >= lvl, merged across gaps of <= gap samples: [(a, b)]."""
    runs = []
    n = ps.size
    j = 0
    while j < n:
        if ps[j] >= lvl:
            a = j
            while j < n and ps[j] >= lvl:
                j += 1
            b = j - 1
            if runs and a - runs[-1][1] - 1 <= gap:
                runs[-1] = (runs[-1][0], b)
            else:
                runs.append((a, b))
        else:
            j += 1
    return runs


def _bdw_smooth(p, W):
    """Centred moving average of width W (odd), shrinking at the ends."""
    n = p.size
    cs = np.zeros(n + 1)
    cs[1:] = np.cumsum(p)
    h = W // 2
    j = np.arange(n)
    a = np.maximum(j - h, 0)
    b = np.minimum(j + h + 1, n)
    return (cs[b] - cs[a]) / (b - a)


def _bdw_cusum_start(p, lvl, a, b):
    """argmax over tau in [a, b+1] of sum_{j=tau}^{b} (p_j - lvl) (the
    least-squares step between the two levels); ties keep the later tau."""
    best, tau, acc = 0.0, b + 1, 0.0
    for j in range(b, a - 1, -1):
        acc += p[j] - lvl
        if acc > best:
            best, tau = acc, j
    return tau


def _bdw_cusum_end(p, lvl, a, b):
    best, tau, acc = 0.0, a - 1, 0.0
    for j in range(a, b + 1):
        acc += p[j] - lvl
        if acc > best:
            best, tau = acc, j
    return tau


def burst_measure(iq, fs: float, i0: int, i1: int, thresh_db: float = 0.0,
                  nfft: int = 0) -> dict:
    """Twin of atkdsp_burst_measure (atkdsp.h §20): one burst's descriptor
    word from channelised I/Q and a coarse [i0, i1) (the CFAR's t0 / t1 in
    samples). Raises ValueError for bad arguments and LookupError when there
    is no burst at the coarse position (the C's ATKDSP_E_ARG / E_EMPTY)."""
    x = np.asarray(iq, dtype=np.complex64)
    n = x.size
    i0, i1, fs = int(i0), int(i1), float(fs)
    if not fs > 0.0 or n < 32 or not 0 <= i0 < i1 <= n or i1 - i0 < 16:
        raise ValueError("burst_measure: fs > 0, n >= 32, 0 <= i0 < i1 <= n, i1 - i0 >= 16")
    if thresh_db is None or not float(thresh_db) <= 40.0:      # NaN too
        raise ValueError("burst_measure: thresh_db 0..40")
    thr = float(thresh_db) if thresh_db > 0.0 else 0.0
    xr = x.real.astype(np.float64)
    xi = x.imag.astype(np.float64)
    p = xr * xr + xi * xi
    flags = 0
    L = i1 - i0
    g = L // 8 + 8
    m0, m1 = i0 + L // 4, i1 - L // 4
    P = _seq_mean(p[m0:m1])
    outside = np.concatenate([p[:max(i0 - g, 0)], p[min(i1 + g, n):]])
    guarded = outside.size >= 32
    if not guarded:                # no room for the guard: whatever lies outside
        outside = np.concatenate([p[:i0], p[i1:]])
    if outside.size >= 32:
        N = _seq_mean(outside)
        # unguarded samples that sit within 1 dB of the plateau are the burst
        # itself (it fills the buffer), not noise
        if not guarded and not P > 1.25 * N:
            N = 0.0
            flags |= BDW_NO_NOISE
    else:
        N = 0.0
        flags |= BDW_NO_NOISE
    if not P > N:
        raise LookupError("burst_measure: no burst at the coarse position")
    # the preliminary bandwidth, for the smoothing
    xm = x[m0:m1].astype(np.complex128)
    nf0 = _bdw_auto_nfft(xm.size)
    psd, nseg, u = _bdw_welch(xm, nf0)
    ob = _bdw_obw(psd - N * u, fs, nf0)
    b_est = fs / nf0
    if ob is not None and ob[1] - ob[0] > b_est:
        b_est = ob[1] - ob[0]
    wmax = max(1, L // 4)
    W = int(_m.floor(32.0 * fs / b_est + 0.5))
    W = min(max(W, 3), wmax)
    if W % 2 == 0:
        W = W - 1 if W > 1 else 1
    Wr = int(_m.floor(fs / b_est + 0.5))
    Wr = min(max(Wr, 1), W, 63)
    if Wr % 2 == 0:
        Wr -= 1
    mid = (i0 + i1) // 2

    def edges(N, P, gauss=False):
        if thr > 0.0 or not N > 0.0 or not gauss:
            t_ = thr if thr > 0.0 else BDW_DEFAULT_THRESH_DB
            lvl = N + (P - N) * 10.0 ** (-t_ / 10.0)
        else:
            # the likelihood boundary: noise and burst as exponential power
            # (Gaussian noise, Gaussian-like signal) with means N and P; the
            # per-sample log-likelihood ratio changes sign at this level
            lvl = _m.log(P / N) * N * P / (P - N)
        ps = _bdw_smooth(p, W)
        runs = _bdw_runs(ps, N + 0.25 * (P - N), W)
        sel = None
        for a, b in runs:
            if a <= mid <= b:
                sel = (a, b)
                break
        if sel is None:
            best = 0
            for a, b in runs:
                ov = min(b, m1 - 1) - max(a, m0) + 1
                if ov > best:
                    best, sel = ov, (a, b)
        if sel is None:
            raise LookupError("burst_measure: no burst at the coarse position")
        e0, e1 = sel
        R = W + 8
        half = (e1 - e0) // 2
        f = 0
        # the step is searched for even when the smoothed core reaches the
        # buffer's end (smoothing spreads a burst by half a window, which can
        # exceed a short margin); truncated means the step is AT the end
        a, b = max(0, e0 - R), min(n - 1, e0 + min(R, half))
        tau = _bdw_cusum_start(p, lvl, a, b)
        t0 = tau - 0.5
        if tau == 0:
            f |= BDW_TRUNC_START
        a, b = max(0, e1 - min(R, half)), min(n - 1, e1 + R)
        tau = _bdw_cusum_end(p, lvl, a, b)
        t1 = tau + 0.5
        if tau == n - 1:
            f |= BDW_TRUNC_END
        return lvl, t0, t1, f
    lvl, t0, t1, ef = edges(N, P)
    # second pass: the noise from outside the measured burst, the plateau from inside
    R2 = 2 * W + 8
    lo_cut, hi_cut = int(_m.floor(t0)) - R2, int(_m.ceil(t1)) + R2
    outside = np.concatenate([p[:max(lo_cut, 0)], p[min(max(hi_cut, 0), n):]])
    if outside.size >= 32:
        N = _seq_mean(outside)
        flags &= ~BDW_NO_NOISE
    a, b = int(_m.ceil(t0)) + W, int(_m.floor(t1)) - W
    if b - a + 1 < 16:
        a, b = max(int(_m.ceil(t0)), 0), min(int(_m.floor(t1)), n - 1)
    if b < a:
        raise LookupError("burst_measure: no burst at the coarse position")
    core = p[a:b + 1]
    P = _seq_mean(core)
    if not P > N:
        raise LookupError("burst_measure: no burst at the coarse position")
    # How the burst's own power fluctuates, noise removed: ~0 for a constant
    # envelope (FM, FSK, a carrier), ~1 for a Gaussian-like signal (OFDM,
    # spread spectrum, noise). A constant envelope's edge is its half-power
    # crossing; a Gaussian one fades through that level on its own, so its
    # edge is the likelihood boundary instead (atkdsp.h §20).
    var = 0.0
    for v in core:
        var += (v - P) * (v - P)
    var /= core.size
    S_ = P - N
    kappa = (var - N * N - 2.0 * S_ * N) / (S_ * S_)
    gauss = kappa > 0.5
    if gauss and thr == 0.0 and N > 0.0:
        flags |= BDW_GAUSSIAN
    lvl, t0, t1, ef = edges(N, P, gauss)
    flags |= ef
    s0, s1 = max(int(_m.ceil(t0)), 0), min(int(_m.floor(t1)), n - 1)
    if s1 < s0:
        raise LookupError("burst_measure: no burst at the coarse position")
    seg = p[s0:s1 + 1]
    pk = float(np.max(seg))
    mean_p = _seq_mean(seg)
    snr = 10.0 * _m.log10((P - N) / N) if N > 0.0 else float("inf")
    if N > 0.0 and snr < 6.0:
        flags |= BDW_LOW_SNR
    # rise and fall, 10-90 % amplitude (1 % and 81 % of the power above noise)
    pr = _bdw_smooth(p, Wr)
    l10, l90 = N + 0.01 * (P - N), N + 0.81 * (P - N)

    def cross(j_from, j_to, level):
        """first crossing of `level` walking j_from -> j_to (step +-1),
        interpolated; None if it never crosses."""
        step = 1 if j_to >= j_from else -1
        j = j_from
        while j != j_to:
            k = j + step
            a_, b_ = pr[j], pr[k]
            if (a_ < level) != (b_ < level):
                return j + step * (level - a_) / (b_ - a_)
            j = k
        return None
    noise_sd = N / _m.sqrt(Wr) if N > 0.0 else 0.0
    if gauss:            # its envelope fades on its own: no 10-90 % to measure
        flags |= BDW_RISE_UNRESOLVED | BDW_FALL_UNRESOLVED
    rise = fall = float("nan")
    if not ef & BDW_TRUNC_START:
        c0 = int(_m.floor(t0))
        lim = max(0, c0 - R2)
        t10 = cross(min(c0 + 1, n - 1), lim, l10)
        t90 = cross(max(c0, 0), min(n - 1, s1), l90)
        if t10 is not None and t90 is not None:
            rise = (t90 - t10) / fs
        if 0.01 * (P - N) < 3.0 * noise_sd or not rise > (2.0 * Wr / fs):
            flags |= BDW_RISE_UNRESOLVED
    else:
        flags |= BDW_RISE_UNRESOLVED
    if not ef & BDW_TRUNC_END:
        c1 = int(_m.ceil(t1))
        lim = min(n - 1, c1 + R2)
        t10 = cross(max(c1 - 1, 0), lim, l10)
        t90 = cross(min(c1, n - 1), max(0, s0), l90)
        if t10 is not None and t90 is not None:
            fall = (t10 - t90) / fs
        if 0.01 * (P - N) < 3.0 * noise_sd or not fall > (2.0 * Wr / fs):
            flags |= BDW_FALL_UNRESOLVED
    else:
        flags |= BDW_FALL_UNRESOLVED
    # the spectrum of the burst: [s0, s1] widened by half a segment each side,
    # so the burst's whole length sits under the overlapped Hann windows'
    # flat sum (the noise outside it is subtracted with the rest)
    nb_len = s1 - s0 + 1
    nf = int(nfft) if nfft else _bdw_auto_nfft(nb_len)
    if nf < 16 or nf & (nf - 1) or nf > 65536:
        nf = _bdw_auto_nfft(nb_len)
    A_, B_ = max(0, s0 - nf // 2), min(n, s1 + 1 + nf // 2)
    if B_ - A_ < nf:
        nf = 16
        while nf * 2 <= B_ - A_:
            nf *= 2
        if B_ - A_ < 16:
            raise LookupError("burst_measure: the burst is too short to measure")
    psd, nseg, u = _bdw_welch(x[A_:B_].astype(np.complex128), nf)
    nbin = N * u
    D = psd - nbin
    ob = _bdw_obw(D, fs, nf)
    if ob is None:
        raise LookupError("burst_measure: no power above the noise in the burst")
    f_lo, f_hi, k_lo, k_hi = ob
    df = fs / nf
    sw = sf = 0.0
    for k in range(k_lo, k_hi + 1):
        sw += D[k]
        sf += (k - nf // 2) * df * D[k]
    fc = sf / sw if sw > 0.0 else 0.5 * (f_lo + f_hi)
    dmax = float(np.max(D))
    line = dmax * 10.0 ** (-2.6)
    if nbin * (1.0 + 3.0 / _m.sqrt(nseg)) > line:
        flags |= BDW_BW26_FLOOR
        bw26 = float("nan")
    else:
        ka = next(k for k in range(nf) if D[k] >= line)
        kb = next(k for k in range(nf - 1, -1, -1) if D[k] >= line)
        bw26 = (kb - ka + 1) * df
    # drift: the centroid of each segment's spectrum inside the occupied band
    c0, c1 = s0 + Wr, s1 - Wr + 1
    if c1 - c0 < 16:
        c0, c1 = s0, s1 + 1
    nsegd = min(8, (c1 - c0) // 64)
    drift = float("nan")
    if nsegd >= 3:
        step = (c1 - c0) // nsegd          # segments spread over the whole burst,
        Ls = min(step, 32768)              # each at most 32768 samples long
        nf2 = 16
        while nf2 < 2 * Ls and nf2 < 65536:
            nf2 *= 2
        ws = _bdw_hann(Ls)
        us = float(np.sum(ws * ws))
        dres = fs / Ls
        flo, fhi = f_lo - 2.0 * dres, f_hi + 2.0 * dres
        ts, fcs = [], []
        for sg in range(nsegd):
            a = c0 + sg * step + (step - Ls) // 2
            buf = np.zeros(nf2, np.complex128)
            buf[:Ls] = x[a:a + Ls].astype(np.complex128) * ws
            X = np.fft.fftshift(np.fft.fft(buf))
            Pk = X.real * X.real + X.imag * X.imag
            sw = sf = 0.0
            for k in range(nf2):
                fk = (k - nf2 // 2) * (fs / nf2)
                if flo <= fk <= fhi:
                    d = Pk[k] - N * us
                    sw += d
                    sf += fk * d
            if sw > 0.0:
                ts.append((a + 0.5 * Ls - c0) / fs)
                fcs.append(sf / sw)
        if len(ts) >= 3:
            mt = _seq_mean(ts)
            mf = _seq_mean(fcs)
            sxy = sxx = 0.0
            for t_, f_ in zip(ts, fcs):
                sxy += (t_ - mt) * (f_ - mf)
                sxx += (t_ - mt) * (t_ - mt)
            drift = sxy / sxx if sxx > 0.0 else float("nan")
    if not drift == drift:
        flags |= BDW_SHORT

    def db(v):
        return float(np.float32(10.0 * _m.log10(v))) if v > 0.0 else float("-inf")
    return dict(t0=t0, t1=t1, duration_s=(t1 - t0) / fs, fc_hz=fc, drift_hz_s=drift,
                obw99_hz=f_hi - f_lo, bw26_hz=bw26, rise_s=rise, fall_s=fall,
                peak_dbfs=db(pk), mean_dbfs=db(mean_p),
                noise_dbfs=db(N) if not flags & BDW_NO_NOISE else float("nan"),
                snr_db=float(np.float32(snr)) if not flags & BDW_NO_NOISE else float("nan"),
                mod_hint=0, flags=flags, nfft=nf, window=W)


# ---- §21 the ring ------------------------------------------------------------------
RING_MAX_CONSUMERS = 16
RING_HEADER_BYTES = 256
RING_CONSUMER_BYTES = 64
RING_SLOT_HEADER = 64
RING_META_KEYS = ("seq", "first_sample", "t_utc_ns", "fc_hz", "fs_hz", "gain_db", "format",
                  "flags", "nbytes")
RING_RETUNE = 1          # meta flags: the block follows a retune
RING_OVERFLOW = 2        # the device reported lost samples before this block
RING_CAL = 4             # calibration state changed


def _r64(v):
    return (int(v) + 63) // 64 * 64


def ring_bytes(block_bytes: int, nblocks: int, max_consumers: int) -> int:
    """Bytes of caller memory a ring needs; 0 for an impossible shape."""
    if block_bytes < 1 or nblocks < 2 or not 1 <= max_consumers <= RING_MAX_CONSUMERS:
        return 0
    return (RING_HEADER_BYTES + RING_CONSUMER_BYTES * max_consumers
            + nblocks * (RING_SLOT_HEADER + _r64(block_bytes)))


class Ring:
    """Twin of atkdsp_ring (atkdsp.h §21): the same semantics in one process
    — one producer that never waits, up to 16 consumers each with its own
    cursor, a seqlock per block (lock = 2 seq + 1 while seq is written, 2 seq
    + 2 once committed), lost counts exact to the block. Cross-process
    attach is native only (a declared exception: shared memory between
    processes has no numpy meaning)."""

    def __init__(self, block_bytes: int, nblocks: int, max_consumers: int = 4):
        if not ring_bytes(block_bytes, nblocks, max_consumers):
            raise ValueError("ring: block_bytes >= 1, nblocks >= 2, max_consumers 1..16")
        self.block_bytes = _r64(block_bytes)
        self.nblocks = int(nblocks)
        self.max_consumers = int(max_consumers)
        self.data = np.zeros((self.nblocks, self.block_bytes), np.uint8)
        self.lock = [0] * self.nblocks
        self.meta = [None] * self.nblocks
        self.write_seq = 0
        self.claimed = False
        self.torn = 0
        self.cons = [None] * self.max_consumers

    # -- producer -------------------------------------------------------------
    def claim(self) -> np.ndarray:
        s = self.write_seq
        i = s % self.nblocks
        self.lock[i] = 2 * s + 1
        self.claimed = True
        return self.data[i]

    def commit(self, nbytes: int, **meta) -> int:
        if not self.claimed or not 1 <= int(nbytes) <= self.block_bytes:
            raise ValueError("ring_commit: claim first; 1 <= nbytes <= block_bytes")
        s = self.write_seq
        i = s % self.nblocks
        m = {k: 0 for k in RING_META_KEYS}
        for k, v in meta.items():
            if k not in m or k in ("seq", "nbytes"):
                raise ValueError(f"unknown ring meta {k!r}")
            m[k] = v
        m["seq"], m["nbytes"] = s, int(nbytes)
        for k in ("fc_hz", "fs_hz", "gain_db"):
            m[k] = float(m[k])
        self.meta[i] = m
        self.lock[i] = 2 * s + 2
        self.write_seq = s + 1
        self.claimed = False
        return 0

    def write(self, data, **meta) -> int:
        b = np.frombuffer(bytes(data), np.uint8)
        if b.size > self.block_bytes:
            raise ValueError("ring_write: more than one block")
        blk = self.claim()
        blk[:b.size] = b
        return self.commit(b.size, **meta)

    # -- consumers ------------------------------------------------------------
    def _oldest(self, w):
        return w - self.nblocks if w > self.nblocks else 0

    def open(self, from_oldest: bool = False) -> int:
        for i in range(self.max_consumers):
            if self.cons[i] is None:
                w = self.write_seq
                self.cons[i] = dict(cursor=self._oldest(w) if from_oldest else w, lost=0,
                                    nread=0, peek=-1)
                return i
        raise IndexError("ring_open: no consumer slot free")

    def close(self, cid: int) -> None:
        self.cons[cid] = None

    close_consumer = close          # the binding's name for it

    def _next(self, c):
        """Advance c's cursor past what is gone; (seq, slot, lost) of the next
        readable block, or None (nothing new)."""
        lost = 0
        while True:
            s, w = c["cursor"], self.write_seq
            if s >= w:
                return None, lost
            if w - s > self.nblocks:
                lost += w - self.nblocks - s
                s = c["cursor"] = w - self.nblocks
            i = s % self.nblocks
            v = self.lock[i]
            if v != 2 * s + 2:
                if v > 2 * s + 2:          # lapped (or being rewritten): it is gone
                    lost += 1
                    c["cursor"] = s + 1
                    continue
                return None, lost
            return (s, i), lost

    def read(self, cid: int, cap: int | None = None):
        """(bytes, meta, lost) of the next block, or (None, None, lost) when
        nothing new. A cap below the block's size raises without consuming."""
        c = self.cons[cid]
        nxt, lost = self._next(c)
        c["lost"] += lost
        if nxt is None:
            return None, None, lost
        s, i = nxt
        m = dict(self.meta[i])
        if cap is not None and m["nbytes"] > cap:
            raise OverflowError("ring_read: cap below the block's size")
        out = self.data[i, :m["nbytes"]].copy()
        c["cursor"] = s + 1
        c["nread"] += 1
        return out, m, lost

    def peek(self, cid: int):
        """Zero-copy: (view, meta, lost); release() says whether it held."""
        c = self.cons[cid]
        nxt, lost = self._next(c)
        c["lost"] += lost
        if nxt is None:
            return None, None, lost
        s, i = nxt
        c["peek"] = s
        return self.data[i, :self.meta[i]["nbytes"]], dict(self.meta[i]), lost

    def release(self, cid: int) -> int:
        c = self.cons[cid]
        s = c["peek"]
        if s < 0:
            raise ValueError("ring_release: nothing peeked")
        c["peek"] = -1
        c["cursor"] = s + 1
        if self.lock[s % self.nblocks] == 2 * s + 2:
            c["nread"] += 1
            return 0
        c["lost"] += 1
        self.torn += 1
        return -6

    def copy_range(self, seq0: int, seq1: int, cap: int | None = None, mcap: int | None = None):
        """Blocks seq0 <= seq < seq1 still in the ring, packed: (bytes, metas)."""
        w = self.write_seq
        parts, metas, used = [], [], 0
        for s in range(max(int(seq0), self._oldest(w)), min(int(seq1), w)):
            i = s % self.nblocks
            if self.lock[i] != 2 * s + 2:
                continue
            m = self.meta[i]
            if (cap is not None and used + m["nbytes"] > cap) or \
                    (mcap is not None and len(metas) >= mcap):
                break
            parts.append(self.data[i, :m["nbytes"]].copy())
            metas.append(dict(m))
            used += m["nbytes"]
        return (np.concatenate(parts) if parts else np.zeros(0, np.uint8)), metas

    def stats(self) -> dict:
        w = self.write_seq
        return dict(committed=w, oldest=self._oldest(w), block_bytes=self.block_bytes,
                    nblocks=self.nblocks, max_consumers=self.max_consumers,
                    active=sum(c is not None for c in self.cons), torn=self.torn,
                    lost=[c["lost"] if c else 0 for c in self.cons],
                    read=[c["nread"] if c else 0 for c in self.cons],
                    lag=[w - c["cursor"] if c else 0 for c in self.cons])

    def describe(self) -> str:
        return ring_describe(self.stats())


def ring_describe(st) -> str:
    total = st["nblocks"] * st["block_bytes"]
    return (f"ring: {st['nblocks']} blocks x {st['block_bytes']} bytes "
            f"({total / 1048576.0:.1f} MiB of data), {st['active']} of "
            f"{st['max_consumers']} consumers, {st['committed']} committed, "
            f"{st['torn']} torn reads")
