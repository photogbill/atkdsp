"""Passive PRACH handset-presence detector — prototype / spec.

WHAT AND WHY. A powered LTE handset transmits a PRACH preamble when it
accesses a cell (attach, tracking-area update, re-establishment). The
preamble is a Zadoff-Chu sequence. Detecting it passively says "a handset is
alive and transmitting here" — presence and, with direction finding, location
— which is the actionable intelligence for search and rescue. It carries NO
subscriber identity and this detector extracts none: it reports that an
access happened, the root sequence used, and how many concurrent accesses,
never who.

THE METHOD (root-agnostic, one FFT). After CP removal and windowing, the
DFT of the received sequence is the frequency-domain ZC of the transmitted
root, times the channel:  Y[k] = H . exp(-j pi u k(k+1)/N).  The adjacent
product  D[k] = Y[k+1] conj(Y[k]) = |H|^2 . exp(-j2pi u k/N) . const  is a
PURE TONE whose frequency is the root u. So one FFT of D reveals every root
present at once (a peak per concurrent root), with no bank of 838
correlators. A cyclic shift of a root adds only a constant phase to D, so it
does not move the tone: shifts of one root share a tone. To count concurrent
accesses on a detected root, one correlation with that root gives the power-
delay profile, whose peaks are the individual preambles.

3GPP TS 36.211 sec 5.7. N_ZC = 839 (preamble formats 0-3, FDD).
"""
import numpy as np

N_ZC = 839                       # Zadoff-Chu length (formats 0-3)
F_RA = 839 * 1250.0             # 1.048875 MHz: 839 subcarriers at 1.25 kHz


def zc_root(u: int, N: int = N_ZC) -> np.ndarray:
    """Frequency-domain ZC root sequence x_u(k), 36.211 5.7.2."""
    n = np.arange(N)
    return np.exp(-1j * np.pi * u * n * (n + 1) / N)


def preamble_time(u: int, shift: int = 0, N: int = N_ZC) -> np.ndarray:
    """One preamble's sequence portion in time (N samples at F_RA).

    The transmitted PRACH maps the ZC onto subcarriers and IFFTs; the DFT of
    the received sequence window returns the frequency-domain ZC. Modelling
    the window directly as IFFT(X_u) with a cyclic shift keeps the prototype
    at the detection rate and is exact for what the detector sees."""
    xu = zc_root(u, N)
    xu = xu * np.exp(-2j * np.pi * np.arange(N) * shift / N)   # cyclic shift
    return np.fft.ifft(xu)


def build_capture(accesses, N=N_ZC, snr_db=10.0, channel=None, seed=0,
                  timing=0):
    """A window of PRACH-band time samples with the listed accesses.

    `accesses` = list of (root, shift). `timing` cyclically rotates the whole
    window to model imperfect occasion alignment."""
    y = np.zeros(N, dtype=complex)
    for (u, sh) in accesses:
        h = 1.0 if channel is None else channel
        y += h * preamble_time(u, sh, N)
    if timing:
        y = np.roll(y, timing)
    if snr_db < 90:
        rng = np.random.default_rng(seed)
        p = np.mean(np.abs(y) ** 2) if np.any(y) else 1.0
        npow = p / (10 ** (snr_db / 10.0))
        y = y + np.sqrt(npow / 2) * (rng.standard_normal(N)
                                     + 1j * rng.standard_normal(N))
    return y


#: Peak-to-median the differential tone must clear. Noise peaks at ~19 over
#: hundreds of windows; a clean preamble scores hundreds of thousands and one
#: at -5 dB SNR still scores ~50. 40 sits well clear of noise with detection
#: to about -5 dB.
MIN_METRIC = 40.0


def detect(seq, min_metric=MIN_METRIC, N=N_ZC, count_thresh=0.30):
    """Detect PRACH preambles in a sequence window (N time samples at F_RA).

    Returns a list of {root, tone_metric, count, delays}. `min_metric` is the
    peak-to-median ratio the differential tone must clear; `count_thresh` is
    the PDP peak height (fraction of the strongest) that counts as a separate
    concurrent access on a root."""
    seq = np.asarray(seq, complex)
    Y = np.fft.fft(seq)                       # -> H * X_u (per root present)
    D = Y[1:] * np.conj(Y[:-1])               # pure tone(s), frequency = -u/N
    S = np.abs(np.fft.fft(D, N)) ** 2         # tone spectrum: a peak per root
    med = np.median(S) + 1e-30
    out = []
    order = np.argsort(S)[::-1]
    claimed = np.zeros(N, dtype=bool)
    for b in order:
        if S[b] / med < min_metric:
            break
        if claimed[b]:
            continue
        # a real tone is one sharp bin; guard neighbours against re-picking
        for d in range(-2, 3):
            claimed[(b + d) % N] = True
        u = (N - int(b)) % N                  # tone is at -u/N -> bin N-u
        if u == 0 or u >= N:
            continue
        # per-root PDP: correlate with this root, IFFT -> delay peaks
        pdp = np.abs(np.fft.ifft(Y * np.conj(zc_root(u, N)))) ** 2
        top = pdp.max()
        peaks = _pdp_peaks(pdp, count_thresh * top)
        out.append({"root": u, "tone_metric": float(S[b] / med),
                    "count": len(peaks), "delays": peaks})
    return out


def _pdp_peaks(pdp, thr):
    """Local maxima above `thr`, as (index) list."""
    peaks = []
    n = len(pdp)
    for i in range(n):
        if pdp[i] >= thr and pdp[i] >= pdp[(i - 1) % n] and pdp[i] > pdp[(i + 1) % n]:
            peaks.append(i)
    return peaks
