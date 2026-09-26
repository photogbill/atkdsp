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
                h = design_lowpass(nyq * 0.8, nyq, atten_db, hi)
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
    _sib1_write(b, 0, 1)                 # SIB1 extension bit
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
            _sib1_write(b, sib_type - 3, _sib1_nbits(9))  # sibType3->0, root 9
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
    _ext = r.u(1)
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
    r.u(1)                              # q-RxLevMinOffset present
    r.u(_sib1_nbits(49))               # q-RxLevMin
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
            r.u(1)                     # SIB-Type ext bit
            sibs.append(r.u(_sib1_nbits(9)) + 3)
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
