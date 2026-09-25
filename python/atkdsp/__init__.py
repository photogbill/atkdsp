"""atkdsp — Python binding for the Analyst Toolkit's native DSP kernels.

Pure ``ctypes`` + numpy; nothing to compile on the Python side. Every kernel
here has a numpy twin in :mod:`atkdsp.reference` with the same signature and
the same semantics, and ``tests/test_cross_check.py`` asserts they agree.
That gives three things at once: a readable specification of what each C
kernel does, a fallback when the library is missing (ATK runs slower, not
not-at-all), and a regression net a C change cannot slip through.

Loading: :func:`load` looks for ``atkdsp.dll`` / ``atkdsp.so`` in
``$ATKDSP_LIB`` (a file path), then ``<this package>/../../bin``, then
beside this file, then the process's library search path. It refuses a
library whose ABI version it does not know — a stale copy is an error with a
sentence, never wrong numbers.

Every call into C releases the GIL for its duration (ctypes does this for
foreign calls), which is the property the library exists for: a 50 ms block
processed here holds the interpreter lock for none of those 50 ms.
"""

from __future__ import annotations

import ctypes as C
import os
import sys
from pathlib import Path

import numpy as np

__all__ = [
    "ABI_VERSION", "AtkDspError", "load", "available", "lib_path", "version",
    "build_info", "set_threads", "get_threads",
    "FMT", "DET", "WIN", "bytes_per_sample",
    "unpack", "unpack_dc", "iq_health",
    "Nco", "Fir", "Resampler", "ArbResampler", "Fft", "window", "power_db",
    "spectrum_reduce", "spectrum_stats", "window_stats",
    "fm_demod", "am_demod", "db_to_pixels", "decimate_max",
    "median", "detect_channels", "stitch_max", "Channel",
]

#: The ABI this binding was written against. A library reporting anything
#: else is refused by :func:`load`.
#: 11 adds the arbitrary/fractional resampler (§4b) and the DDC's use of it for
#: non-integer output rates.
ABI_VERSION = 11

FMT = {"cu8": 0, "ci8": 1, "ci16": 2, "ci16_le": 2, "cs16": 2,
       "ci16q11": 3, "cf32": 4, "cf32_le": 4}
DET = {"max": 0, "min": 1, "avg": 2}
WIN = {"rectangular": 0, "hann": 1, "hamming": 2, "blackmanharris": 3}

LIB_NAME = "atkdsp.dll" if sys.platform == "win32" else "atkdsp.so"


class AtkDspError(RuntimeError):
    """A kernel returned an error code, or the library could not be loaded."""


class _Cf32(C.Structure):
    _fields_ = [("re", C.c_float), ("im", C.c_float)]


class _Nco(C.Structure):
    _fields_ = [("phase", C.c_double), ("step", C.c_double)]


class _IqHealth(C.Structure):
    _fields_ = [("dc_re", C.c_double), ("dc_im", C.c_double),
                ("rms", C.c_double), ("gain_imbalance_db", C.c_double),
                ("phase_error_deg", C.c_double), ("image_rejection_db", C.c_double),
                ("clip_fraction", C.c_double), ("n", C.c_size_t)]


class _ChannelC(C.Structure):
    _fields_ = [("start_bin", C.c_int), ("end_bin", C.c_int),
                ("centroid_bin", C.c_float), ("peak_db", C.c_float),
                ("snr_db", C.c_float)]


class Channel:
    """One detected channel, in bins (the caller knows the bin width)."""
    __slots__ = ("start_bin", "end_bin", "centroid_bin", "peak_db", "snr_db")

    def __init__(self, start_bin, end_bin, centroid_bin, peak_db, snr_db):
        self.start_bin = int(start_bin)
        self.end_bin = int(end_bin)
        self.centroid_bin = float(centroid_bin)
        self.peak_db = float(peak_db)
        self.snr_db = float(snr_db)

    def __repr__(self) -> str:
        return (f"Channel(bins {self.start_bin}..{self.end_bin}, centroid "
                f"{self.centroid_bin:.1f}, peak {self.peak_db:.1f} dB, "
                f"snr {self.snr_db:.1f} dB)")

    def __eq__(self, other) -> bool:
        return (isinstance(other, Channel)
                and (self.start_bin, self.end_bin) == (other.start_bin, other.end_bin)
                and abs(self.centroid_bin - other.centroid_bin) < 1e-3
                and abs(self.peak_db - other.peak_db) < 1e-4
                and abs(self.snr_db - other.snr_db) < 1e-4)


_lib = None
_lib_path: Path | None = None
_BIND_EXTRA: list = []          # binders registered further down the file
_load_error: str | None = None

_ERRORS = {-1: "bad argument", -2: "out of memory", -3: "output buffer too small",
           -4: "FFT backend failure"}


def _check(rc: int, what: str) -> int:
    if rc < 0:
        raise AtkDspError(f"{what}: {_ERRORS.get(rc, f'error {rc}')}")
    return rc


def _candidates() -> list[Path]:
    here = Path(__file__).resolve().parent
    out = []
    env = os.environ.get("ATKDSP_LIB")
    if env:
        out.append(Path(env))
    out.append(here.parent.parent / "bin" / LIB_NAME)      # the repo's bin/
    out.append(here / LIB_NAME)                            # beside the package
    out.append(Path(LIB_NAME))                             # search path / cwd
    return out


def _bind(lib) -> None:
    P = C.POINTER
    cf = P(_Cf32)
    fp = P(C.c_float)
    lib.atkdsp_abi_version.restype = C.c_int
    lib.atkdsp_version_string.restype = C.c_char_p
    lib.atkdsp_build_info.restype = C.c_char_p
    lib.atkdsp_bytes_per_sample.restype = C.c_int
    lib.atkdsp_bytes_per_sample.argtypes = [C.c_int]
    lib.atkdsp_set_threads.restype = C.c_int
    lib.atkdsp_set_threads.argtypes = [C.c_int]
    lib.atkdsp_get_threads.restype = C.c_int

    lib.atkdsp_unpack.restype = C.c_ssize_t
    lib.atkdsp_unpack.argtypes = [C.c_void_p, C.c_size_t, C.c_int, cf, C.c_size_t,
                                  C.c_float, P(_Cf32)]

    lib.atkdsp_unpack_dc.restype = C.c_ssize_t
    lib.atkdsp_unpack_dc.argtypes = [C.c_void_p, C.c_size_t, C.c_int, cf, C.c_size_t,
                                     C.c_float, C.c_float,
                                     P(C.c_double), P(C.c_double)]

    lib.atkdsp_iq_health.restype = C.c_int
    lib.atkdsp_iq_health.argtypes = [cf, C.c_size_t, C.c_float, P(_IqHealth)]

    lib.atkdsp_nco_init.argtypes = [P(_Nco), C.c_double, C.c_double]
    lib.atkdsp_nco_set_freq.argtypes = [P(_Nco), C.c_double, C.c_double]
    lib.atkdsp_nco_mix.argtypes = [P(_Nco), cf, cf, C.c_size_t]

    lib.atkdsp_fir_create.restype = C.c_void_p
    lib.atkdsp_fir_create.argtypes = [fp, C.c_size_t, C.c_uint]
    lib.atkdsp_fir_destroy.argtypes = [C.c_void_p]
    lib.atkdsp_fir_reset.argtypes = [C.c_void_p]
    lib.atkdsp_fir_out_max.restype = C.c_size_t
    lib.atkdsp_fir_out_max.argtypes = [C.c_void_p, C.c_size_t]
    lib.atkdsp_fir_process.restype = C.c_ssize_t
    lib.atkdsp_fir_process.argtypes = [C.c_void_p, cf, C.c_size_t, cf, C.c_size_t]

    lib.atkdsp_resampler_create.restype = C.c_void_p
    lib.atkdsp_resampler_create.argtypes = [C.c_uint, C.c_uint, fp, C.c_size_t]
    lib.atkdsp_resampler_destroy.argtypes = [C.c_void_p]
    lib.atkdsp_resampler_reset.argtypes = [C.c_void_p]
    lib.atkdsp_resampler_out_max.restype = C.c_size_t
    lib.atkdsp_resampler_out_max.argtypes = [C.c_void_p, C.c_size_t]
    lib.atkdsp_resampler_process.restype = C.c_ssize_t
    lib.atkdsp_resampler_process.argtypes = [C.c_void_p, cf, C.c_size_t, cf, C.c_size_t]

    lib.atkdsp_arb_resampler_create.restype = C.c_void_p
    lib.atkdsp_arb_resampler_create.argtypes = [C.c_double, C.c_double, C.c_double, C.c_uint]
    lib.atkdsp_arb_resampler_destroy.argtypes = [C.c_void_p]
    lib.atkdsp_arb_resampler_reset.argtypes = [C.c_void_p]
    lib.atkdsp_arb_resampler_set_ratio.argtypes = [C.c_void_p, C.c_double]
    lib.atkdsp_arb_resampler_ratio.restype = C.c_double
    lib.atkdsp_arb_resampler_ratio.argtypes = [C.c_void_p]
    lib.atkdsp_arb_resampler_out_max.restype = C.c_size_t
    lib.atkdsp_arb_resampler_out_max.argtypes = [C.c_void_p, C.c_size_t]
    lib.atkdsp_arb_resampler_process.restype = C.c_ssize_t
    lib.atkdsp_arb_resampler_process.argtypes = [C.c_void_p, cf, C.c_size_t, cf, C.c_size_t]

    lib.atkdsp_fft_create.restype = C.c_void_p
    lib.atkdsp_fft_create.argtypes = [C.c_size_t]
    lib.atkdsp_fft_destroy.argtypes = [C.c_void_p]
    lib.atkdsp_fft_length.restype = C.c_size_t
    lib.atkdsp_fft_length.argtypes = [C.c_void_p]
    lib.atkdsp_fft_exec.restype = C.c_int
    lib.atkdsp_fft_exec.argtypes = [C.c_void_p, cf, cf, C.c_int]
    lib.atkdsp_window.restype = C.c_int
    lib.atkdsp_window.argtypes = [C.c_int, C.c_size_t, fp]
    lib.atkdsp_power_db.restype = C.c_int
    lib.atkdsp_power_db.argtypes = [C.c_void_p, cf, fp, fp]
    lib.atkdsp_spectrum_reduce.restype = C.c_ssize_t
    lib.atkdsp_spectrum_reduce.argtypes = [C.c_void_p, cf, C.c_size_t, C.c_size_t, fp,
                                           C.c_int, fp]
    lib.atkdsp_spectrum_stats.restype = C.c_ssize_t
    lib.atkdsp_spectrum_stats.argtypes = [C.c_void_p, cf, C.c_size_t, C.c_size_t, fp,
                                          fp, fp, fp, fp]
    lib.atkdsp_window_stats.restype = C.c_int
    lib.atkdsp_window_stats.argtypes = [fp, C.c_size_t, P(C.c_double), P(C.c_double)]

    lib.atkdsp_fm_demod.argtypes = [cf, C.c_size_t, fp, P(_Cf32), C.c_float]
    lib.atkdsp_am_demod.argtypes = [cf, C.c_size_t, fp, fp, C.c_float]

    lib.atkdsp_db_to_pixels.argtypes = [fp, C.c_size_t, C.c_float, C.c_float,
                                        P(C.c_uint32), P(C.c_uint32)]
    lib.atkdsp_decimate_max.argtypes = [fp, C.c_size_t, fp, C.c_size_t]

    lib.atkdsp_median.restype = C.c_float
    lib.atkdsp_median.argtypes = [fp, C.c_size_t, fp]
    lib.atkdsp_detect_channels.restype = C.c_ssize_t
    lib.atkdsp_detect_channels.argtypes = [fp, C.c_size_t, C.c_float, C.c_float, C.c_int,
                                           C.c_int, fp, P(_ChannelC), C.c_size_t]
    lib.atkdsp_stitch_max.argtypes = [fp, C.c_size_t, C.c_double, C.c_double, C.c_double,
                                      C.c_double, C.c_double, C.c_double, fp, C.c_size_t]


def load(path: str | os.PathLike | None = None):
    """Load the library (once). Returns the ctypes handle; raises AtkDspError."""
    global _lib, _lib_path, _load_error
    if _lib is not None and path is None:
        return _lib
    cands = [Path(path)] if path else _candidates()
    errors = []
    for cand in cands:
        try:
            if cand.name == LIB_NAME and not cand.exists() and cand.parent == Path("."):
                lib = C.CDLL(LIB_NAME)                 # let the OS search
            else:
                if not cand.exists():
                    errors.append(f"{cand}: not found")
                    continue
                lib = C.CDLL(str(cand))
        except OSError as e:
            errors.append(f"{cand}: {e}")
            continue
        lib.atkdsp_abi_version.restype = C.c_int
        abi = int(lib.atkdsp_abi_version())
        if abi != ABI_VERSION:
            errors.append(f"{cand}: ABI {abi}, this binding needs {ABI_VERSION}")
            continue
        _bind(lib)
        for extra in _BIND_EXTRA:
            extra(lib)
        _lib, _lib_path, _load_error = lib, cand, None
        return lib
    _load_error = "; ".join(errors) or "no candidate paths"
    raise AtkDspError(f"could not load {LIB_NAME}: {_load_error}")


def available() -> bool:
    """True when the library loads. Never raises."""
    try:
        load()
        return True
    except AtkDspError:
        return False


def lib_path() -> Path | None:
    return _lib_path


def version() -> str:
    return load().atkdsp_version_string().decode()


def build_info() -> str:
    return load().atkdsp_build_info().decode()


def set_threads(n: int) -> int:
    return int(load().atkdsp_set_threads(int(n)))


def get_threads() -> int:
    return int(load().atkdsp_get_threads())


def bytes_per_sample(fmt) -> int:
    return int(load().atkdsp_bytes_per_sample(_fmt(fmt)))


# -- helpers ------------------------------------------------------------------
def _fmt(fmt) -> int:
    if isinstance(fmt, str):
        try:
            return FMT[fmt]
        except KeyError:
            raise AtkDspError(f"unknown sample format {fmt!r}") from None
    return int(fmt)


def _cf32(a, name="array") -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.complex64)
    if a.ndim != 1:
        raise AtkDspError(f"{name} must be one-dimensional")
    return a


def _f32(a, name="array") -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float32)
    if a.ndim != 1:
        raise AtkDspError(f"{name} must be one-dimensional")
    return a


def _cfp(a: np.ndarray):
    return a.ctypes.data_as(C.POINTER(_Cf32))


def _fp(a: np.ndarray):
    return a.ctypes.data_as(C.POINTER(C.c_float))


# -- 1. unpack ----------------------------------------------------------------
def unpack(raw, fmt, dc_alpha: float = 0.0, dc_state: np.ndarray | None = None,
           out: np.ndarray | None = None) -> np.ndarray:
    """Raw bytes → complex64. ``dc_state`` is a 1-element complex64 array the
    caller keeps between calls when ``dc_alpha > 0``. ``raw`` may be bytes,
    a bytearray, a memoryview or a numpy array; no copy is made of it."""
    lib = load()
    src = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) \
        else np.ascontiguousarray(raw).view(np.uint8).reshape(-1)
    nbytes = src.size
    n = nbytes // bytes_per_sample(fmt)
    if out is None:
        out = np.empty(n, dtype=np.complex64)
    elif out.dtype != np.complex64 or not out.flags.c_contiguous:
        raise AtkDspError("out must be a contiguous complex64 array")
    st = None
    if dc_alpha > 0:
        if dc_state is None or dc_state.dtype != np.complex64 or dc_state.size != 1:
            raise AtkDspError("dc_state must be a 1-element complex64 array")
        st = dc_state.ctypes.data_as(C.POINTER(_Cf32))
    got = _check(lib.atkdsp_unpack(src.ctypes.data, nbytes, _fmt(fmt), _cfp(out), out.size,
                                   float(dc_alpha), st), "unpack")
    return out[:got]


def unpack_dc(raw, fmt, offset: complex = 0j, out: np.ndarray | None = None,
              want_mean: bool = True):
    """Raw bytes -> complex64 with ``offset`` removed; returns ``(x, mean)``.

    ``mean`` is the mean of the samples BEFORE the offset was taken out, so a
    caller keeping a running estimate gets it without a second pass. Both the
    subtraction and the mean are free: see atkdsp.h §1b.
    """
    lib = load()
    src = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) \
        else np.ascontiguousarray(raw).view(np.uint8).reshape(-1)
    nbytes = src.size
    n = nbytes // bytes_per_sample(fmt)
    if out is None:
        out = np.empty(n, dtype=np.complex64)
    elif out.dtype != np.complex64 or not out.flags.c_contiguous:
        raise AtkDspError("out must be a contiguous complex64 array")
    mr, mi = C.c_double(0.0), C.c_double(0.0)
    pr = C.byref(mr) if want_mean else None
    pi = C.byref(mi) if want_mean else None
    got = _check(lib.atkdsp_unpack_dc(src.ctypes.data, nbytes, _fmt(fmt), _cfp(out),
                                      out.size, float(offset.real), float(offset.imag),
                                      pr, pi), "unpack_dc")
    return out[:got], complex(mr.value, mi.value)


# -- 2. NCO -------------------------------------------------------------------
class Nco:
    """Phase-continuous complex mixer. ``mix`` shifts a channel at +freq to DC."""

    def __init__(self, freq_hz: float, sample_rate: float):
        self._lib = load()
        self._s = _Nco()
        self._lib.atkdsp_nco_init(C.byref(self._s), float(freq_hz), float(sample_rate))

    @property
    def phase(self) -> float:
        return float(self._s.phase)

    def set_freq(self, freq_hz: float, sample_rate: float) -> None:
        self._lib.atkdsp_nco_set_freq(C.byref(self._s), float(freq_hz), float(sample_rate))

    def mix(self, x, out: np.ndarray | None = None) -> np.ndarray:
        x = _cf32(x, "x")
        if out is None:
            out = np.empty_like(x)
        self._lib.atkdsp_nco_mix(C.byref(self._s), _cfp(x), _cfp(out), x.size)
        return out


# -- 3. decimating FIR --------------------------------------------------------
class Fir:
    """Decimating FIR with carried history; see atkdsp.h §3."""

    def __init__(self, taps, decim: int = 1):
        self._lib = load()
        t = _f32(taps, "taps")
        self._taps = t
        self._h = self._lib.atkdsp_fir_create(_fp(t), t.size, int(decim))
        if not self._h:
            raise AtkDspError("fir_create failed")
        self.decim = int(decim)
        self.ntaps = int(t.size)

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.atkdsp_fir_destroy(h)

    def reset(self) -> None:
        self._lib.atkdsp_fir_reset(self._h)

    def out_max(self, n_in: int) -> int:
        return int(self._lib.atkdsp_fir_out_max(self._h, int(n_in)))

    def process(self, x, out: np.ndarray | None = None) -> np.ndarray:
        x = _cf32(x, "x")
        cap = self.out_max(x.size)
        if out is None:
            out = np.empty(cap, dtype=np.complex64)
        got = _check(self._lib.atkdsp_fir_process(self._h, _cfp(x), x.size, _cfp(out), out.size),
                     "fir_process")
        return out[:got]


# -- 4. rational resampler ----------------------------------------------------
class Resampler:
    """Polyphase L/M resampler with carried state; see atkdsp.h §4."""

    def __init__(self, up: int, down: int, taps):
        self._lib = load()
        t = _f32(taps, "taps")
        self._h = self._lib.atkdsp_resampler_create(int(up), int(down), _fp(t), t.size)
        if not self._h:
            raise AtkDspError("resampler_create failed")
        self.up, self.down, self.ntaps = int(up), int(down), int(t.size)

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.atkdsp_resampler_destroy(h)

    def reset(self) -> None:
        self._lib.atkdsp_resampler_reset(self._h)

    def out_max(self, n_in: int) -> int:
        return int(self._lib.atkdsp_resampler_out_max(self._h, int(n_in)))

    def process(self, x, out: np.ndarray | None = None) -> np.ndarray:
        x = _cf32(x, "x")
        cap = self.out_max(x.size)
        if out is None:
            out = np.empty(cap, dtype=np.complex64)
        got = _check(self._lib.atkdsp_resampler_process(self._h, _cfp(x), x.size, _cfp(out),
                                                        out.size), "resampler_process")
        return out[:got]


# -- 4b. arbitrary (fractional) resampler -------------------------------------
class ArbResampler:
    """Resample by ANY positive out/in ratio; see atkdsp.h §4b.

    A polyphase bank with first-order Farrow interpolation between phases, so a
    ratio that is a real number (an odd sample clock, a huge reduced
    denominator, a few-ppm clock correction) is native rather than an integer
    L/M. The fractional position and history are carried, so a block boundary is
    invisible and N seconds in produce ~N*ratio samples out.
    """

    def __init__(self, in_rate: float, out_rate: float, atten_db: float = 60.0,
                 nphase: int = 64):
        self._lib = load()
        self._h = self._lib.atkdsp_arb_resampler_create(
            float(in_rate), float(out_rate), float(atten_db), int(nphase))
        if not self._h:
            raise AtkDspError("arb_resampler_create failed")
        self.in_rate, self.out_rate = float(in_rate), float(out_rate)

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.atkdsp_arb_resampler_destroy(h)

    def reset(self) -> None:
        self._lib.atkdsp_arb_resampler_reset(self._h)

    def set_ratio(self, ratio: float) -> None:
        """Nudge the ratio (out/in) live without rebuilding the prototype — for
        a few-ppm sample-clock correction."""
        self._lib.atkdsp_arb_resampler_set_ratio(self._h, float(ratio))
        self.out_rate = self.in_rate * float(ratio)

    def ratio(self) -> float:
        return float(self._lib.atkdsp_arb_resampler_ratio(self._h))

    def out_max(self, n_in: int) -> int:
        return int(self._lib.atkdsp_arb_resampler_out_max(self._h, int(n_in)))

    def process(self, x, out: np.ndarray | None = None) -> np.ndarray:
        x = _cf32(x, "x")
        cap = self.out_max(x.size)
        if out is None:
            out = np.empty(cap, dtype=np.complex64)
        got = _check(self._lib.atkdsp_arb_resampler_process(self._h, _cfp(x), x.size,
                                                            _cfp(out), out.size),
                     "arb_resampler_process")
        return out[:got]


# -- 5. FFT -------------------------------------------------------------------
class Fft:
    """A plan for one length. Forward unscaled, inverse scaled 1/n."""

    def __init__(self, n: int):
        self._lib = load()
        self._h = self._lib.atkdsp_fft_create(int(n))
        if not self._h:
            raise AtkDspError(f"fft_create({n}) failed")
        self.n = int(n)

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.atkdsp_fft_destroy(h)

    def exec(self, x, inverse: bool = False, out: np.ndarray | None = None) -> np.ndarray:
        x = _cf32(x, "x")
        if x.size != self.n:
            raise AtkDspError(f"fft: expected {self.n} samples, got {x.size}")
        if out is None:
            out = np.empty(self.n, dtype=np.complex64)
        _check(self._lib.atkdsp_fft_exec(self._h, _cfp(x), _cfp(out), 1 if inverse else 0), "fft_exec")
        return out

    def power_db(self, x, win: np.ndarray | None = None, out: np.ndarray | None = None) -> np.ndarray:
        x = _cf32(x, "x")
        if x.size != self.n:
            raise AtkDspError(f"power_db: expected {self.n} samples, got {x.size}")
        w = None if win is None else _f32(win, "win")
        if out is None:
            out = np.empty(self.n, dtype=np.float32)
        _check(self._lib.atkdsp_power_db(self._h, _cfp(x), None if w is None else _fp(w), _fp(out)),
               "power_db")
        return out

    def spectrum_reduce(self, x, hop: int | None = None, win: np.ndarray | None = None,
                        detector="max", out: np.ndarray | None = None):
        """100 % POI: every frame in x reduced to one line. Returns (line, frames)."""
        x = _cf32(x, "x")
        w = None if win is None else _f32(win, "win")
        if out is None:
            out = np.empty(self.n, dtype=np.float32)
        det = DET[detector] if isinstance(detector, str) else int(detector)
        frames = _check(self._lib.atkdsp_spectrum_reduce(
            self._h, _cfp(x), x.size, int(hop or self.n), None if w is None else _fp(w), det,
            _fp(out)), "spectrum_reduce")
        return out, int(frames)

    def spectrum_stats(self, x, hop: int | None = None, win: np.ndarray | None = None,
                       want=("max", "avg", "min", "sk")):
        """Per-bin max/avg/min (dB) and spectral kurtosis in one pass over the
        frames. Returns a dict of the requested arrays plus ``frames``. Skipped
        outputs cost nothing on the C side (their pointer is NULL)."""
        x = _cf32(x, "x")
        w = None if win is None else _f32(win, "win")
        want = set(want)
        outs = {k: (np.empty(self.n, dtype=np.float32) if k in want else None)
                for k in ("max", "avg", "min", "sk")}
        frames = _check(self._lib.atkdsp_spectrum_stats(
            self._h, _cfp(x), x.size, int(hop or self.n),
            None if w is None else _fp(w),
            _fp(outs["max"]) if outs["max"] is not None else None,
            _fp(outs["avg"]) if outs["avg"] is not None else None,
            _fp(outs["min"]) if outs["min"] is not None else None,
            _fp(outs["sk"]) if outs["sk"] is not None else None),
            "spectrum_stats")
        res = {k: v for k, v in outs.items() if v is not None}
        res["frames"] = int(frames)
        return res


def window(kind, n: int) -> np.ndarray:
    k = WIN[kind] if isinstance(kind, str) else int(kind)
    out = np.empty(int(n), dtype=np.float32)
    _check(load().atkdsp_window(k, int(n), _fp(out)), "window")
    return out


def power_db(x, win=None) -> np.ndarray:
    x = _cf32(x, "x")
    return Fft(x.size).power_db(x, win)


def spectrum_reduce(x, n: int, hop: int | None = None, win=None, detector="max"):
    return Fft(n).spectrum_reduce(x, hop, win, detector)


def spectrum_stats(x, n: int, hop: int | None = None, win=None,
                   want=("max", "avg", "min", "sk")):
    return Fft(n).spectrum_stats(x, hop, win, want)


def window_stats(win) -> tuple:
    """(coherent_gain, enbw_bins) of a window array."""
    w = _f32(win, "win")
    cg, enbw = C.c_double(0.0), C.c_double(0.0)
    _check(load().atkdsp_window_stats(_fp(w), w.size, C.byref(cg), C.byref(enbw)),
           "window_stats")
    return cg.value, enbw.value


def iq_health(x, clip_level: float = 0.0) -> dict:
    """Front-end health of a converted block: DC offset, I/Q gain and phase
    imbalance, the derived image rejection, and the clip fraction. Returns a
    dict with the same keys as the numpy twin."""
    x = _cf32(x, "x")
    h = _IqHealth()
    _check(load().atkdsp_iq_health(_cfp(x), x.size, float(clip_level), C.byref(h)),
           "iq_health")
    return {"dc_re": h.dc_re, "dc_im": h.dc_im, "rms": h.rms,
            "gain_imbalance_db": h.gain_imbalance_db,
            "phase_error_deg": h.phase_error_deg,
            "image_rejection_db": h.image_rejection_db,
            "clip_fraction": h.clip_fraction, "n": int(h.n)}


# -- 6. demodulators ----------------------------------------------------------
def fm_demod(x, prev: np.ndarray | None = None, gain: float = 1.0,
             out: np.ndarray | None = None) -> np.ndarray:
    """``prev`` is a 1-element complex64 array carried between calls; None
    means "start from zero" (first sample demodulates to 0)."""
    x = _cf32(x, "x")
    if out is None:
        out = np.empty(x.size, dtype=np.float32)
    if prev is None:
        prev = np.zeros(1, dtype=np.complex64)
    load().atkdsp_fm_demod(_cfp(x), x.size, _fp(out), prev.ctypes.data_as(C.POINTER(_Cf32)),
                           float(gain))
    return out


def am_demod(x, dc_state: np.ndarray | None = None, alpha: float = 0.0,
             out: np.ndarray | None = None) -> np.ndarray:
    x = _cf32(x, "x")
    if out is None:
        out = np.empty(x.size, dtype=np.float32)
    st = None if dc_state is None else _fp(dc_state)
    load().atkdsp_am_demod(_cfp(x), x.size, _fp(out), st, float(alpha))
    return out


# -- 7. display ---------------------------------------------------------------
def db_to_pixels(db, lo: float, hi: float, lut256, out: np.ndarray | None = None) -> np.ndarray:
    db = _f32(db, "db")
    lut = np.ascontiguousarray(lut256, dtype=np.uint32)
    if lut.size != 256:
        raise AtkDspError("lut256 must have 256 entries")
    if out is None:
        out = np.empty(db.size, dtype=np.uint32)
    load().atkdsp_db_to_pixels(_fp(db), db.size, float(lo), float(hi),
                               lut.ctypes.data_as(C.POINTER(C.c_uint32)),
                               out.ctypes.data_as(C.POINTER(C.c_uint32)))
    return out


def decimate_max(x, m: int, out: np.ndarray | None = None) -> np.ndarray:
    """Peak-preserving reduction of x to m columns (m > len(x) pads with the last value)."""
    x = _f32(x, "x")
    m = int(m)
    if out is None:
        out = np.empty(m, dtype=np.float32)
    load().atkdsp_decimate_max(_fp(x), x.size, _fp(out), m)
    return out


# -- 8. detection -------------------------------------------------------------
def median(x, scratch: np.ndarray | None = None) -> float:
    x = _f32(x, "x")
    if scratch is None or scratch.size < x.size:
        scratch = np.empty(x.size, dtype=np.float32)
    return float(load().atkdsp_median(_fp(x), x.size, _fp(scratch)))


def detect_channels(line, floor_db: float, threshold_db: float, gap_bins: int,
                    min_run: int, max_channels: int = 256,
                    scratch: np.ndarray | None = None) -> list[Channel]:
    line = _f32(line, "line")
    if scratch is None or scratch.size < line.size:
        scratch = np.empty(line.size, dtype=np.float32)
    buf = (_ChannelC * int(max_channels))()
    got = _check(load().atkdsp_detect_channels(_fp(line), line.size, float(floor_db),
                                               float(threshold_db), int(gap_bins), int(min_run),
                                               _fp(scratch), buf, int(max_channels)),
                 "detect_channels")
    return [Channel(c.start_bin, c.end_bin, c.centroid_bin, c.peak_db, c.snr_db)
            for c in buf[:got]]


def stitch_max(seg, seg_lo_hz: float, seg_hz_per_bin: float, keep_lo_hz: float,
               keep_hi_hz: float, out_lo_hz: float, out_hz_per_bin: float,
               out: np.ndarray) -> np.ndarray:
    """Max-accumulate one segment into ``out`` (float32, NaN = unmeasured), in place."""
    seg = _f32(seg, "seg")
    if out.dtype != np.float32 or not out.flags.c_contiguous:
        raise AtkDspError("out must be a contiguous float32 array")
    load().atkdsp_stitch_max(_fp(seg), seg.size, float(seg_lo_hz), float(seg_hz_per_bin),
                             float(keep_lo_hz), float(keep_hi_hz), float(out_lo_hz),
                             float(out_hz_per_bin), _fp(out), out.size)
    return out


# -- 9/10. filter design and the DDC -------------------------------------------
def _bind_ddc(lib) -> None:
    P = C.POINTER
    cf = P(_Cf32)
    fp = P(C.c_float)
    lib.atkdsp_design_lowpass.restype = C.c_ssize_t
    lib.atkdsp_design_lowpass.argtypes = [C.c_double, C.c_double, C.c_double, C.c_double,
                                          fp, C.c_size_t]
    lib.atkdsp_ddc_create.restype = C.c_void_p
    lib.atkdsp_ddc_create.argtypes = [C.c_double, C.c_double, C.c_double, C.c_double,
                                      C.c_double, C.c_size_t]
    lib.atkdsp_ddc_destroy.argtypes = [C.c_void_p]
    lib.atkdsp_ddc_reset.argtypes = [C.c_void_p]
    lib.atkdsp_ddc_set_offset.argtypes = [C.c_void_p, C.c_double]
    lib.atkdsp_ddc_out_rate.restype = C.c_double
    lib.atkdsp_ddc_out_rate.argtypes = [C.c_void_p]
    lib.atkdsp_ddc_out_max.restype = C.c_size_t
    lib.atkdsp_ddc_out_max.argtypes = [C.c_void_p, C.c_size_t]
    lib.atkdsp_ddc_process.restype = C.c_ssize_t
    lib.atkdsp_ddc_process.argtypes = [C.c_void_p, cf, C.c_size_t, cf, C.c_size_t]
    lib.atkdsp_ddc_describe.restype = C.c_int
    lib.atkdsp_ddc_describe.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t]
    lib.atkdsp_ddc_plan.restype = C.c_int
    lib.atkdsp_ddc_plan.argtypes = [C.c_void_p, P(C.c_uint), P(C.c_uint), C.c_size_t,
                                    P(C.c_uint), P(C.c_uint)]


_BIND_EXTRA.append(_bind_ddc)


def design_lowpass(fp_hz: float, fs_hz: float, atten_db: float, rate: float) -> np.ndarray:
    """Kaiser-windowed sinc low-pass, unity DC gain, odd length."""
    lib = load()
    n = _check(lib.atkdsp_design_lowpass(float(fp_hz), float(fs_hz), float(atten_db),
                                         float(rate), None, 0), "design_lowpass")
    out = np.empty(int(n), dtype=np.float32)
    _check(lib.atkdsp_design_lowpass(float(fp_hz), float(fs_hz), float(atten_db), float(rate),
                                     _fp(out), out.size), "design_lowpass")
    return out


class Ddc:
    """Wideband I/Q → one channel at an exact rate; see atkdsp.h §10."""

    def __init__(self, sample_rate: float, offset_hz: float, channel_bw_hz: float,
                 out_rate: float, atten_db: float = 60.0, max_block: int = 1 << 20):
        self._lib = load()
        self._h = self._lib.atkdsp_ddc_create(float(sample_rate), float(offset_hz),
                                              float(channel_bw_hz), float(out_rate),
                                              float(atten_db), int(max_block))
        if not self._h:
            raise AtkDspError(f"ddc_create({sample_rate}, {offset_hz}, {channel_bw_hz}, "
                              f"{out_rate}) failed — rates must be integer Hz and "
                              f"out_rate, bw <= sample_rate")
        self.sample_rate = float(sample_rate)
        self.max_block = int(max_block)

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.atkdsp_ddc_destroy(h)

    @property
    def out_rate(self) -> float:
        return float(self._lib.atkdsp_ddc_out_rate(self._h))

    def reset(self) -> None:
        self._lib.atkdsp_ddc_reset(self._h)

    def set_offset(self, offset_hz: float) -> None:
        self._lib.atkdsp_ddc_set_offset(self._h, float(offset_hz))

    def out_max(self, n_in: int) -> int:
        return int(self._lib.atkdsp_ddc_out_max(self._h, int(n_in)))

    def plan(self):
        """(stage factors, stage tap counts, up, down)."""
        f = (C.c_uint * 16)(); t = (C.c_uint * 16)()
        up = C.c_uint(); down = C.c_uint()
        n = _check(self._lib.atkdsp_ddc_plan(self._h, f, t, 16, C.byref(up), C.byref(down)),
                   "ddc_plan")
        return list(f[:n]), list(t[:n]), int(up.value), int(down.value)

    def describe(self) -> str:
        buf = C.create_string_buffer(512)
        self._lib.atkdsp_ddc_describe(self._h, buf, 512)
        return buf.value.decode()

    def process(self, x, out: np.ndarray | None = None) -> np.ndarray:
        x = _cf32(x, "x")
        if x.size > self.max_block:
            raise AtkDspError(f"ddc: block of {x.size} exceeds max_block {self.max_block}")
        cap = self.out_max(x.size)
        if out is None:
            out = np.empty(cap, dtype=np.complex64)
        got = _check(self._lib.atkdsp_ddc_process(self._h, _cfp(x), x.size, _cfp(out), out.size),
                     "ddc_process")
        return out[:got]




# -- 11. LTE cell search --------------------------------------------------------
class _LteCell(C.Structure):
    _fields_ = [("nid2", C.c_int), ("nid1", C.c_int), ("pci", C.c_int),
                ("subframe", C.c_int), ("offset", C.c_longlong),
                ("metric", C.c_float), ("sss_score", C.c_float),
                ("cfo_hz", C.c_float)]


class _LteMib(C.Structure):
    _fields_ = [("dl_bw_rb", C.c_int), ("phich_dur", C.c_int),
                ("phich_res", C.c_int), ("sfn", C.c_int), ("n_ports", C.c_int)]


class _PrachHit(C.Structure):
    _fields_ = [("root", C.c_int), ("count", C.c_int),
                ("metric", C.c_float), ("delay", C.c_int)]


_LTE_MAX_PLMN = 6
_LTE_MAX_SI = 8
_LTE_MAX_SIBMAP = 8
_LTE_MAX_NEIGH = 16
_LTE_MAX_FREQ = 8


class _LtePlmn(C.Structure):
    _fields_ = [("mcc", C.c_int * 3), ("mnc", C.c_int * 3),
                ("mnc_len", C.c_int), ("reserved", C.c_int)]


class _LteSched(C.Structure):
    _fields_ = [("periodicity_rf", C.c_int), ("n_sibs", C.c_int),
                ("sibs", C.c_int * _LTE_MAX_SIBMAP)]


class _LteSib1(C.Structure):
    _fields_ = [("n_plmn", C.c_int), ("plmn", _LtePlmn * _LTE_MAX_PLMN),
                ("tac", C.c_int), ("cell_id", C.c_uint),
                ("cell_barred", C.c_int), ("csg_id", C.c_int),
                ("freq_band", C.c_int), ("si_window_ms", C.c_int),
                ("n_sched", C.c_int), ("sched", _LteSched * _LTE_MAX_SI)]


class _LteNeigh(C.Structure):
    _fields_ = [("pci", C.c_int), ("q_off", C.c_int)]


class _LteSib2(C.Structure):
    _fields_ = [("present", C.c_int), ("barring", C.c_int),
                ("barring_emergency", C.c_int), ("prach_root", C.c_int),
                ("prach_config_index", C.c_int), ("prach_high_speed", C.c_int),
                ("prach_zcc", C.c_int), ("prach_freq_offset", C.c_int),
                ("ref_sig_power", C.c_int), ("ul_earfcn", C.c_int),
                ("ul_bandwidth_rb", C.c_int), ("time_align_timer", C.c_int)]


class _LteSib3(C.Structure):
    _fields_ = [("present", C.c_int), ("q_rxlevmin", C.c_int),
                ("s_intra_search", C.c_int), ("resel_priority", C.c_int)]


class _LteSib4(C.Structure):
    _fields_ = [("present", C.c_int), ("n_neigh", C.c_int),
                ("neigh", _LteNeigh * _LTE_MAX_NEIGH),
                ("n_black", C.c_int), ("black_start", C.c_int * _LTE_MAX_NEIGH)]


class _LteInterFreq(C.Structure):
    _fields_ = [("dl_earfcn", C.c_int), ("n_neigh", C.c_int),
                ("neigh_pci", C.c_int * _LTE_MAX_NEIGH)]


class _LteSib5(C.Structure):
    _fields_ = [("present", C.c_int), ("n_freq", C.c_int),
                ("freq", _LteInterFreq * _LTE_MAX_FREQ)]


class _LteSi(C.Structure):
    _fields_ = [("sib2", _LteSib2), ("sib3", _LteSib3),
                ("sib4", _LteSib4), ("sib5", _LteSib5)]


LTE_RATE = 1_920_000
LTE_SYM = 128
LTE_PSS_PERIOD = 9600
LTE_FRAME = 19200
LTE_PBCH_OFFSET = 128
LTE_PBCH_BLOCK = 549
PRACH_NZC = 839


def _bind_lte(lib) -> None:
    P = C.POINTER
    cf = P(_Cf32)
    lib.atkdsp_lte_create.restype = C.c_void_p
    lib.atkdsp_lte_destroy.argtypes = [C.c_void_p]
    lib.atkdsp_lte_pss_symbol.restype = C.c_int
    lib.atkdsp_lte_pss_symbol.argtypes = [C.c_int, cf]
    lib.atkdsp_lte_sss_symbol.restype = C.c_int
    lib.atkdsp_lte_sss_symbol.argtypes = [C.c_int, C.c_int, C.c_int, P(C.c_float)]
    lib.atkdsp_lte_detect.restype = C.c_ssize_t
    lib.atkdsp_lte_detect.argtypes = [C.c_void_p, cf, C.c_size_t, C.c_float,
                                      P(_LteCell), C.c_size_t]
    lib.atkdsp_lte_gold.restype = C.c_int
    lib.atkdsp_lte_gold.argtypes = [C.c_uint, C.c_int, P(C.c_byte)]
    lib.atkdsp_lte_mib_decode.restype = C.c_int
    lib.atkdsp_lte_mib_decode.argtypes = [C.c_void_p, cf, C.c_int, C.c_double,
                                          P(_LteMib)]
    lib.atkdsp_lte_mib_decode_combined.restype = C.c_int
    lib.atkdsp_lte_mib_decode_combined.argtypes = [C.c_void_p, cf, C.c_int,
                                          C.c_ssize_t, C.c_int, C.c_double,
                                          P(_LteMib)]
    lib.atkdsp_prach_create.restype = C.c_void_p
    lib.atkdsp_prach_destroy.argtypes = [C.c_void_p]
    lib.atkdsp_prach_zc.restype = C.c_int
    lib.atkdsp_prach_zc.argtypes = [C.c_int, cf]
    lib.atkdsp_prach_detect.restype = C.c_ssize_t
    lib.atkdsp_prach_detect.argtypes = [C.c_void_p, cf, C.c_size_t, C.c_float,
                                        P(_PrachHit), C.c_size_t]
    lib.atkdsp_prach_scan.restype = C.c_ssize_t
    lib.atkdsp_prach_scan.argtypes = [C.c_void_p, cf, C.c_size_t, C.c_int,
                                      C.c_float, P(_PrachHit), C.c_size_t]
    lib.atkdsp_lte_crc24a.restype = C.c_int
    lib.atkdsp_lte_crc24a.argtypes = [P(C.c_byte), C.c_int, P(C.c_byte)]
    lib.atkdsp_lte_turbo_decode.restype = C.c_int
    lib.atkdsp_lte_turbo_decode.argtypes = [P(C.c_float), C.c_size_t, C.c_int,
                                            C.c_int, C.c_int, P(C.c_byte)]
    lib.atkdsp_lte_sib1_parse.restype = C.c_int
    lib.atkdsp_lte_sib1_parse.argtypes = [P(C.c_byte), C.c_int, P(_LteSib1)]
    lib.atkdsp_lte_sib1_decode.restype = C.c_int
    lib.atkdsp_lte_sib1_decode.argtypes = [cf, C.c_size_t, C.c_int, C.c_int,
                                           C.c_int, C.c_int, C.c_double, C.c_int,
                                           P(_LteSib1)]
    lib.atkdsp_lte_si_parse.restype = C.c_int
    lib.atkdsp_lte_si_parse.argtypes = [P(C.c_byte), C.c_int, P(_LteSi)]
    lib.atkdsp_lte_si_decode.restype = C.c_int
    lib.atkdsp_lte_si_decode.argtypes = [cf, C.c_size_t, C.c_int, C.c_int,
                                         C.c_int, C.c_int, C.c_double, C.c_int,
                                         P(_LteSi)]


_BIND_EXTRA.append(_bind_lte)


def lte_pss_symbol(nid2: int) -> np.ndarray:
    out = np.empty(LTE_SYM, dtype=np.complex64)
    _check(load().atkdsp_lte_pss_symbol(int(nid2), _cfp(out)), "lte_pss_symbol")
    return out


def lte_sss_symbol(nid1: int, nid2: int, subframe: int) -> np.ndarray:
    out = np.empty(62, dtype=np.float32)
    _check(load().atkdsp_lte_sss_symbol(int(nid1), int(nid2), int(subframe),
                                        _fp(out)), "lte_sss_symbol")
    return out


def lte_gold(c_init: int, length: int) -> np.ndarray:
    """36.211 7.2 Gold sequence, length <= 1920 (0/1 as int8)."""
    out = np.empty(int(length), dtype=np.int8)
    ptr = out.ctypes.data_as(C.POINTER(C.c_byte))
    _check(load().atkdsp_lte_gold(int(c_init) & 0xFFFFFFFF, int(length), ptr),
           "lte_gold")
    return out


class Lte:
    """LTE cell search. Feed it I/Q at LTE_RATE; it finds cells."""

    def __init__(self):
        self._lib = load()
        self._h = self._lib.atkdsp_lte_create()
        if not self._h:
            raise AtkDspError("lte_create failed")

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.atkdsp_lte_destroy(h)

    def detect(self, x, min_metric: float = 0.06, max_cells: int = 4) -> list:
        x = _cf32(x, "x")
        buf = (_LteCell * int(max_cells))()
        got = _check(self._lib.atkdsp_lte_detect(self._h, _cfp(x), x.size,
                                                 float(min_metric), buf,
                                                 int(max_cells)), "lte_detect")
        return [{"nid2": c.nid2, "nid1": c.nid1, "pci": c.pci,
                 "subframe": c.subframe, "offset": int(c.offset),
                 "metric": float(c.metric), "sss_score": float(c.sss_score),
                 "cfo_hz": float(c.cfo_hz)} for c in buf[:got]]

    @staticmethod
    def _mib_dict(m):
        return {"dl_bw_rb": m.dl_bw_rb, "phich_dur": m.phich_dur,
                "phich_res": m.phich_res, "sfn": m.sfn, "n_ports": m.n_ports}

    def mib_decode(self, block, n_id: int, cfo_hz: float = 0.0):
        """Decode the MIB from one PBCH block (>= LTE_PBCH_BLOCK samples).
        Returns a dict or None."""
        block = _cf32(block, "block")
        out = _LteMib()
        rc = _check(self._lib.atkdsp_lte_mib_decode(self._h, _cfp(block),
                    int(n_id), float(cfo_hz), C.byref(out)), "lte_mib_decode")
        return self._mib_dict(out) if rc == 1 else None

    def mib_decode_frames(self, blocks, n_id: int, cfo_hz: float = 0.0):
        """Soft-combine consecutive PBCH blocks. `blocks` is a list of arrays
        (each >= LTE_PBCH_BLOCK samples). Returns a dict or None."""
        buf = np.concatenate([_cf32(b, "block")[:LTE_PBCH_BLOCK] for b in blocks])
        out = _LteMib()
        rc = _check(self._lib.atkdsp_lte_mib_decode_combined(self._h, _cfp(buf),
                    len(blocks), LTE_PBCH_BLOCK, int(n_id), float(cfo_hz),
                    C.byref(out)), "lte_mib_decode_combined")
        return self._mib_dict(out) if rc == 1 else None



def prach_zc(u: int) -> np.ndarray:
    """Frequency-domain Zadoff-Chu root sequence (839 values)."""
    out = np.empty(PRACH_NZC, dtype=np.complex64)
    _check(load().atkdsp_prach_zc(int(u), _cfp(out)), "prach_zc")
    return out


def lte_crc24a(bits) -> np.ndarray:
    """CRC-24A (36.212 5.1.1) of `bits` (0/1), 24 parity bits (MSB first)."""
    b = np.ascontiguousarray(np.asarray(bits, np.int8))
    out = np.empty(24, dtype=np.int8)
    _check(load().atkdsp_lte_crc24a(b.ctypes.data_as(C.POINTER(C.c_byte)),
           b.size, out.ctypes.data_as(C.POINTER(C.c_byte))), "lte_crc24a")
    return out


def lte_turbo_decode(llr, K: int, rv: int = 0, iters: int = 8):
    """Decode E descrambled LLRs (>0 favours bit 0) for turbo code-block size
    K. Returns the K-24 payload bits on a CRC-24A pass, else None."""
    e = np.ascontiguousarray(llr, dtype=np.float32)
    out = np.empty(int(K) - 24, dtype=np.int8)
    rc = _check(load().atkdsp_lte_turbo_decode(
        e.ctypes.data_as(C.POINTER(C.c_float)), e.size, int(K), int(rv),
        int(iters), out.ctypes.data_as(C.POINTER(C.c_byte))), "lte_turbo_decode")
    return out if rc == 1 else None


def lte_sib1_parse(bits):
    """Parse the tower identity (PLMN list, TAC, ECI) from decoded SIB1
    transport-block bits (one bit per element, MSB first). Returns a dict, or
    None on a structural mismatch."""
    b = np.ascontiguousarray(np.asarray(bits, np.int8))
    out = _LteSib1()
    rc = _check(load().atkdsp_lte_sib1_parse(b.ctypes.data_as(C.POINTER(C.c_byte)),
                b.size, C.byref(out)), "lte_sib1_parse")
    if rc != 1:
        return None
    return _sib1_dict(out)


def _sib1_dict(out):
    plmns = []
    for i in range(out.n_plmn):
        p = out.plmn[i]
        mcc = None if p.mcc[0] < 0 else [p.mcc[0], p.mcc[1], p.mcc[2]]
        plmns.append({"mcc": mcc, "mnc": [p.mnc[j] for j in range(p.mnc_len)],
                      "reserved": bool(p.reserved)})
    sched = []
    for i in range(out.n_sched):
        sc = out.sched[i]
        sched.append({"periodicity_rf": int(sc.periodicity_rf),
                      "sibs": [int(sc.sibs[j]) for j in range(sc.n_sibs)]})
    return {"plmns": plmns, "tac": int(out.tac), "cellid": int(out.cell_id),
            "cell_barred": int(out.cell_barred),
            "csg_id": None if out.csg_id < 0 else int(out.csg_id),
            "freq_band": int(out.freq_band), "si_window_ms": int(out.si_window_ms),
            "sched": sched}


def _si_dict(out):
    s2, s3, s4, s5 = out.sib2, out.sib3, out.sib4, out.sib5
    d = {}
    if s2.present:
        d["sib2"] = {
            "barring": bool(s2.barring), "barring_emergency": bool(s2.barring_emergency),
            "prach_root": int(s2.prach_root), "prach_config_index": int(s2.prach_config_index),
            "prach_high_speed": bool(s2.prach_high_speed), "prach_zcc": int(s2.prach_zcc),
            "prach_freq_offset": int(s2.prach_freq_offset), "ref_sig_power": int(s2.ref_sig_power),
            "ul_earfcn": None if s2.ul_earfcn < 0 else int(s2.ul_earfcn),
            "ul_bandwidth_rb": int(s2.ul_bandwidth_rb),
            "time_align_timer": int(s2.time_align_timer)}
    if s3.present:
        d["sib3"] = {"q_rxlevmin": int(s3.q_rxlevmin),
                     "s_intra_search": None if s3.s_intra_search < 0 else int(s3.s_intra_search),
                     "resel_priority": int(s3.resel_priority)}
    if s4.present:
        d["sib4"] = {
            "neighbors": [{"pci": int(s4.neigh[i].pci), "q_off": int(s4.neigh[i].q_off)}
                          for i in range(s4.n_neigh)],
            "blacklist": [int(s4.black_start[i]) for i in range(s4.n_black)]}
    if s5.present:
        d["sib5"] = {"carriers": [
            {"dl_earfcn": int(s5.freq[i].dl_earfcn),
             "neighbors": [int(s5.freq[i].neigh_pci[j]) for j in range(s5.freq[i].n_neigh)]}
            for i in range(s5.n_freq)]}
    return d


def lte_si_parse(bits):
    """Parse a SystemInformation message's transport-block bits (one bit per
    element, MSB first) into a dict of the SIBs it carries (sib2/3/4/5), or None
    on a structural mismatch."""
    b = np.ascontiguousarray(np.asarray(bits, np.int8))
    out = _LteSi()
    rc = _check(load().atkdsp_lte_si_parse(b.ctypes.data_as(C.POINTER(C.c_byte)),
                b.size, C.byref(out)), "lte_si_parse")
    return _si_dict(out) if rc == 1 else None


def lte_si_decode(samples, n_rb, n_id, n_ports, subframe, cfo_hz=0.0,
                  equalize=True):
    """Full receive chain for a SystemInformation message (same PHY as
    lte_sib1_decode). Returns the SIB2-5 fingerprint dict or None."""
    s = _cf32(samples, "samples")
    out = _LteSi()
    rc = _check(load().atkdsp_lte_si_decode(_cfp(s), s.size, int(n_rb), int(n_id),
                int(n_ports), int(subframe), float(cfo_hz), 1 if equalize else 0,
                C.byref(out)), "lte_si_decode")
    return _si_dict(out) if rc == 1 else None


def lte_sib1_decode(samples, n_rb: int, n_id: int, n_ports: int, subframe: int,
                    cfo_hz: float = 0.0, equalize: bool = True):
    """Full SIB1 receive chain over one subframe of I/Q (at nfft*15 kHz):
    OFDM demod, CRS equalisation, PCFICH, blind SI-RNTI PDCCH, PDSCH turbo and
    ASN.1. Returns the tower-identity dict (operator PLMN, TAC, ECI) or None."""
    s = _cf32(samples, "samples")
    out = _LteSib1()
    rc = _check(load().atkdsp_lte_sib1_decode(_cfp(s), s.size, int(n_rb),
                int(n_id), int(n_ports), int(subframe), float(cfo_hz),
                1 if equalize else 0, C.byref(out)), "lte_sib1_decode")
    return _sib1_dict(out) if rc == 1 else None


class Prach:
    """Passive PRACH handset-presence detector. Feed a sequence window
    (>= PRACH_NZC samples at the PRACH rate); get the preambles present."""

    def __init__(self):
        self._lib = load()
        self._h = self._lib.atkdsp_prach_create()
        if not self._h:
            raise AtkDspError("prach_create failed")

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.atkdsp_prach_destroy(h)

    def detect(self, seq, min_metric: float = 0.0, max_hits: int = 16) -> list:
        seq = _cf32(seq, "seq")
        buf = (_PrachHit * int(max_hits))()
        got = _check(self._lib.atkdsp_prach_detect(self._h, _cfp(seq), seq.size,
                     float(min_metric), buf, int(max_hits)), "prach_detect")
        return [{"root": h.root, "count": h.count,
                 "metric": float(h.metric), "delay": h.delay}
                for h in buf[:got]]

    def scan(self, stream, win_step: int = 64, min_metric: float = 0.0,
             max_hits: int = 16) -> list:
        """Slide a window across a longer uplink capture; merge by root."""
        stream = _cf32(stream, "stream")
        buf = (_PrachHit * int(max_hits))()
        got = _check(self._lib.atkdsp_prach_scan(self._h, _cfp(stream),
                     stream.size, int(win_step), float(min_metric), buf,
                     int(max_hits)), "prach_scan")
        return [{"root": h.root, "count": h.count,
                 "metric": float(h.metric), "delay": h.delay}
                for h in buf[:got]]


__all__ += ["design_lowpass", "Ddc", "Lte", "lte_pss_symbol", "lte_sss_symbol",
            "lte_gold", "LTE_RATE", "LTE_SYM", "LTE_PSS_PERIOD", "LTE_FRAME",
            "LTE_PBCH_OFFSET", "LTE_PBCH_BLOCK", "Prach", "prach_zc", "PRACH_NZC",
            "lte_crc24a", "lte_turbo_decode", "lte_sib1_parse", "lte_sib1_decode",
            "lte_si_parse", "lte_si_decode"]
