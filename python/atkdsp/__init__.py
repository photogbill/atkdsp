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
    "SymDump", "sync_search", "SyncHit", "dc_track", "agc",
]

#: The ABI this binding was written against. A library reporting anything
#: else is refused by :func:`load`.
#: 11 adds the arbitrary/fractional resampler (§4b) and the DDC's use of it for
#: non-integer output rates.
#: 12 adds the FSK front end: symdump (integrate-and-dump) + sync_search (§12).
#: 13 adds dc_track + agc (adaptive DC removal + AGC) for off-centre RF (§12).
#: 14 adds the DMR / P25 sync correlator, DmrSync (§13).
#: 15 adds the bitstream engine (§14-17): bit primitives, scramblers, FEC, CRC.
ABI_VERSION = 15

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

# -- 12. FSK front end: symbol integrate-and-dump + sync-word search -----------
class _SyncHit(C.Structure):
    _fields_ = [("pos", C.c_longlong), ("polarity", C.c_int), ("errors", C.c_int)]


class SyncHit:
    """One sync-word match: pos (first symbol after the sync), polarity
    (0 normal, 1 inverted), errors (bit mismatches)."""
    __slots__ = ("pos", "polarity", "errors")

    def __init__(self, pos, polarity, errors):
        self.pos = int(pos); self.polarity = int(polarity); self.errors = int(errors)

    def __repr__(self):
        return f"SyncHit(pos={self.pos}, polarity={self.polarity}, errors={self.errors})"

    def __eq__(self, other):
        return (isinstance(other, SyncHit)
                and (self.pos, self.polarity, self.errors)
                == (other.pos, other.polarity, other.errors))


def _bind_fsk(lib) -> None:
    P = C.POINTER
    fp = P(C.c_float)
    lib.atkdsp_symdump_create.restype = C.c_void_p
    lib.atkdsp_symdump_create.argtypes = [C.c_double, C.c_double]
    lib.atkdsp_symdump_destroy.argtypes = [C.c_void_p]
    lib.atkdsp_symdump_reset.argtypes = [C.c_void_p]
    lib.atkdsp_symdump_out_max.restype = C.c_size_t
    lib.atkdsp_symdump_out_max.argtypes = [C.c_void_p, C.c_size_t]
    lib.atkdsp_symdump_process.restype = C.c_ssize_t
    lib.atkdsp_symdump_process.argtypes = [C.c_void_p, fp, C.c_size_t, fp, C.c_size_t]
    lib.atkdsp_sync_search.restype = C.c_ssize_t
    lib.atkdsp_sync_search.argtypes = [fp, C.c_size_t, P(C.c_byte), C.c_size_t, C.c_int,
                                       P(_SyncHit), C.c_size_t]
    lib.atkdsp_dc_track.argtypes = [fp, C.c_size_t, fp, fp, C.c_float]
    lib.atkdsp_agc.argtypes = [fp, C.c_size_t, fp, fp, C.c_float, C.c_float]


_BIND_EXTRA.append(_bind_fsk)


class SymDump:
    """FSK symbol integrate-and-dump; see atkdsp.h §12."""

    def __init__(self, sps: float, phase: float = 0.0):
        self._lib = load()
        self._h = self._lib.atkdsp_symdump_create(float(sps), float(phase))
        if not self._h:
            raise AtkDspError("symdump_create failed (need sps >= 1)")
        self.sps = float(sps)

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.atkdsp_symdump_destroy(h)

    def reset(self) -> None:
        self._lib.atkdsp_symdump_reset(self._h)

    def out_max(self, n_in: int) -> int:
        return int(self._lib.atkdsp_symdump_out_max(self._h, int(n_in)))

    def process(self, x, out=None) -> np.ndarray:
        x = _f32(x, "x")
        cap = self.out_max(x.size)
        if out is None:
            out = np.empty(cap, dtype=np.float32)
        got = _check(self._lib.atkdsp_symdump_process(self._h, _fp(x), x.size,
                                                      _fp(out), out.size), "symdump_process")
        return out[:got]


def sync_search(sym, pattern, max_err: int = 0, cap: int = 4096) -> list:
    """Find `pattern` (0/1 bits) in the sign of soft symbols `sym`. Returns a
    list of SyncHit. See atkdsp.h §12."""
    lib = load()
    s = _f32(sym, "sym")
    pat = np.ascontiguousarray(pattern, dtype=np.int8)
    out = (_SyncHit * int(cap))()
    n = _check(lib.atkdsp_sync_search(_fp(s), s.size,
                                      pat.ctypes.data_as(C.POINTER(C.c_byte)), pat.size,
                                      int(max_err), out, int(cap)), "sync_search")
    return [SyncHit(out[i].pos, out[i].polarity, out[i].errors) for i in range(n)]

def dc_track(x, state=None, alpha: float = 0.01) -> np.ndarray:
    """Adaptive DC/baseline removal (atkdsp.h §12). `state` is a float32[1] the
    caller carries across calls (or None for a one-shot). Returns the tracked
    signal (input minus the slow mean)."""
    lib = load()
    x = _f32(x, "x")
    out = np.empty_like(x)
    st = np.zeros(1, np.float32) if state is None else state
    lib.atkdsp_dc_track(_fp(x), x.size, _fp(out), _fp(st), C.c_float(alpha))
    return out


def agc(x, state=None, alpha: float = 0.01, ref: float = 1.0) -> np.ndarray:
    """Adaptive amplitude normalisation (atkdsp.h §12). `state` is a float32[1]
    carried across calls (or None). Scales the signal toward `ref`."""
    lib = load()
    x = _f32(x, "x")
    out = np.empty_like(x)
    st = np.zeros(1, np.float32) if state is None else state
    lib.atkdsp_agc(_fp(x), x.size, _fp(out), _fp(st), C.c_float(alpha), C.c_float(ref))
    return out


# -- 13. 4FSK sync correlator: DMR / P25 pilots and burst timing ----------------
# The model, the defaults and what they were measured against are in atkdsp.h
# §13, src/dmr_sync.c and the twin (reference.py §13). The pattern TAGS (BS/MS/
# direct, voice/data, slot) are Python-side metadata and come from the twin's
# table, so there is one copy of them; the C has the names, and the tests hold
# the two tables to each other.
import logging as _logging                                   # noqa: E402

from . import reference as _ref                               # noqa: E402

_dmr_log = _logging.getLogger("atkdsp.dmr_sync")

SYNC_DMR = 0
SYNC_P25 = 1
SYNC_INVERT = 1
SYNC_SYMBOLS = 24
SYNC_RATE = 4800.0
SYNC_CONFIRM_ERRORS = _ref.SYNC_CONFIRM_ERRORS


class _DmrSyncHit(C.Structure):
    _fields_ = [("start", C.c_longlong), ("frac", C.c_double), ("cfo_hz", C.c_double),
                ("score", C.c_float), ("errors", C.c_int), ("pattern", C.c_int),
                ("reserved", C.c_int)]


def _bind_dmr_sync(lib) -> None:
    P = C.POINTER
    cf = P(_Cf32)
    lib.atkdsp_dmr_sync_patterns.restype = C.c_int
    lib.atkdsp_dmr_sync_patterns.argtypes = [C.c_int]
    lib.atkdsp_dmr_sync_pattern.restype = C.c_int
    lib.atkdsp_dmr_sync_pattern.argtypes = [C.c_int, C.c_int, P(C.c_byte), C.c_size_t]
    lib.atkdsp_dmr_sync_pattern_name.restype = C.c_char_p
    lib.atkdsp_dmr_sync_pattern_name.argtypes = [C.c_int, C.c_int]
    lib.atkdsp_dmr_sync_create.restype = C.c_void_p
    lib.atkdsp_dmr_sync_create.argtypes = [C.c_int, C.c_double, C.c_int, C.c_uint,
                                           C.c_double, C.c_double, C.c_float, C.c_int,
                                           C.c_double, C.c_int]
    lib.atkdsp_dmr_sync_destroy.argtypes = [C.c_void_p]
    lib.atkdsp_dmr_sync_reset.argtypes = [C.c_void_p]
    lib.atkdsp_dmr_sync_info.restype = C.c_int
    lib.atkdsp_dmr_sync_info.argtypes = [C.c_void_p, P(C.c_int), P(C.c_int),
                                         P(C.c_double), P(C.c_int)]
    lib.atkdsp_dmr_sync_hits_max.restype = C.c_size_t
    lib.atkdsp_dmr_sync_hits_max.argtypes = [C.c_void_p, C.c_size_t]
    lib.atkdsp_dmr_sync_process.restype = C.c_ssize_t
    lib.atkdsp_dmr_sync_process.argtypes = [C.c_void_p, cf, C.c_size_t, C.c_size_t,
                                            P(_DmrSyncHit), cf, C.c_size_t]
    lib.atkdsp_dmr_sync_flush.restype = C.c_ssize_t
    lib.atkdsp_dmr_sync_flush.argtypes = [C.c_void_p, P(_DmrSyncHit), cf, C.c_size_t]
    lib.atkdsp_dmr_sync_reference.restype = C.c_int
    lib.atkdsp_dmr_sync_reference.argtypes = [C.c_void_p, C.c_int, cf, C.c_size_t]
    lib.atkdsp_dmr_sync_describe.restype = C.c_int
    lib.atkdsp_dmr_sync_describe.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t]


_BIND_EXTRA.append(_bind_dmr_sync)


def dmr_sync_patterns(sync_set=SYNC_DMR) -> int:
    """How many sync words the set has (DMR 9, P25 1)."""
    return _check(load().atkdsp_dmr_sync_patterns(_ref.sync_set_id(sync_set)),
                  "dmr_sync_patterns")


def dmr_sync_pattern(sync_set, idx) -> np.ndarray:
    """The 24 symbols (+-3, +-1) of one sync word, as the library holds them."""
    out = np.zeros(SYNC_SYMBOLS, dtype=np.int8)
    _check(load().atkdsp_dmr_sync_pattern(_ref.sync_set_id(sync_set), int(idx),
                                          out.ctypes.data_as(C.POINTER(C.c_byte)), out.size),
           "dmr_sync_pattern")
    return out


def dmr_sync_pattern_name(sync_set, idx) -> str:
    name = load().atkdsp_dmr_sync_pattern_name(_ref.sync_set_id(sync_set), int(idx))
    if name is None:
        raise AtkDspError(f"dmr_sync_pattern_name: no pattern {idx}")
    return name.decode()


class DmrSync:
    """Streaming DMR / P25 sync correlator over 1..64 coherent channels; see
    atkdsp.h §13. Same constructor, methods and hit dicts as the twin
    (atkdsp.reference.DmrSync), so ATK's fallback is a drop-in.

    Feed `process` blocks of shape (N,) for one channel or (n_channels, N),
    complex baseband centred on the channel at `sample_rate` (24 or 48 kHz).
    It returns the hits whose peaks are final; `flush` returns a pending one.
    Each hit is a dict: start, frac, cfo_hz, score, errors, confirmed,
    pattern, name, source, kind, slot, amps (complex64, one per channel;
    amps / amps[0] is the steering vector)."""

    def __init__(self, sample_rate: float, n_channels: int = 1, sync_set=SYNC_DMR,
                 patterns=None, cfo_span_hz: float = 1000.0, cfo_step_hz: float = 0.0,
                 threshold=None, guard: int = 0, alpha=None, invert: bool = False):
        self._lib = load()
        sid = _ref.sync_set_id(sync_set)
        if threshold is None:
            threshold = _ref.sync_default_threshold(n_channels)
        mask = _ref._sync_mask(sid, patterns)
        a = float("nan") if alpha is None else float(alpha)
        self._h = self._lib.atkdsp_dmr_sync_create(
            sid, float(sample_rate), int(n_channels), int(mask), float(cfo_span_hz),
            float(cfo_step_hz), float(threshold), int(guard), a,
            SYNC_INVERT if invert else 0)
        if not self._h:
            raise AtkDspError(
                f"dmr_sync_create(set={sid}, fs={sample_rate}, channels={n_channels}, "
                f"mask={mask:#x}, span={cfo_span_hz}, step={cfo_step_hz}, "
                f"threshold={threshold}, guard={guard}, alpha={alpha}) refused — "
                f"fs 9600..384000, channels 1..64, a pattern selected, span 0..5000, "
                f"<= 201 bins, threshold in (0, 1], guard <= 16 windows, alpha <= 1")
        win, ncfo, gd = C.c_int(), C.c_int(), C.c_int()
        stp = C.c_double()
        self._lib.atkdsp_dmr_sync_info(self._h, C.byref(win), C.byref(ncfo),
                                       C.byref(stp), C.byref(gd))
        self.sync_set = sid
        self.sample_rate = float(sample_rate)
        self.n_channels = int(n_channels)
        self.mask = mask
        self.active = [p for p in range(_ref.dmr_sync_patterns(sid)) if mask >> p & 1]
        self.alpha = _ref._sync_alpha(sid, alpha)
        self.invert = bool(invert)
        self.L = int(win.value)
        self.n_cfo = int(ncfo.value)
        self.kmax = (self.n_cfo - 1) // 2
        self.cfo_step = float(stp.value)
        self.cfo_span = float(cfo_span_hz)
        self.threshold = float(threshold)
        self.guard = int(gd.value)
        _dmr_log.info("dmr_sync created: %s", self.describe())

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.atkdsp_dmr_sync_destroy(h)

    def reset(self) -> None:
        self._lib.atkdsp_dmr_sync_reset(self._h)

    def hits_max(self, n_in: int) -> int:
        return int(self._lib.atkdsp_dmr_sync_hits_max(self._h, int(n_in)))

    def describe(self) -> str:
        buf = C.create_string_buffer(512)
        self._lib.atkdsp_dmr_sync_describe(self._h, buf, 512)
        return buf.value.decode()

    def reference(self, idx) -> np.ndarray:
        out = np.empty(self.L, dtype=np.complex64)
        _check(self._lib.atkdsp_dmr_sync_reference(self._h, int(idx), _cfp(out), out.size),
               "dmr_sync_reference")
        return out

    def _hits(self, buf, amps, n) -> list:
        hits = []
        for i in range(n):
            h = buf[i]
            d = {"start": int(h.start), "frac": float(h.frac), "cfo_hz": float(h.cfo_hz),
                 "score": float(h.score), "errors": int(h.errors),
                 "confirmed": int(h.errors) <= SYNC_CONFIRM_ERRORS, "pattern": int(h.pattern)}
            d.update(_ref.dmr_sync_pattern_info(self.sync_set, int(h.pattern)))
            d["amps"] = amps[i].copy()
            hits.append(d)
            _dmr_log.debug("dmr_sync hit: %s at %d%+.2f, score %.3f, %d errors, cfo %+.1f Hz",
                           d["name"], d["start"], d["frac"], d["score"], d["errors"], d["cfo_hz"])
        return hits

    def process(self, x) -> list:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[0] != self.n_channels:
            raise AtkDspError(f"dmr_sync: expected shape ({self.n_channels}, N), got {x.shape}")
        x = np.ascontiguousarray(x, dtype=np.complex64)
        n = x.shape[1]
        cap = self.hits_max(n)
        buf = (_DmrSyncHit * cap)()
        amps = np.zeros((cap, self.n_channels), dtype=np.complex64)
        got = _check(self._lib.atkdsp_dmr_sync_process(
            self._h, x.ctypes.data_as(C.POINTER(_Cf32)), n, n, buf, _cfp(amps), cap),
            "dmr_sync_process")
        return self._hits(buf, amps, got)

    def flush(self) -> list:
        buf = (_DmrSyncHit * 1)()
        amps = np.zeros((1, self.n_channels), dtype=np.complex64)
        got = _check(self._lib.atkdsp_dmr_sync_flush(self._h, buf, _cfp(amps), 1),
                     "dmr_sync_flush")
        return self._hits(buf, amps, got)


__all__ += ["SYNC_DMR", "SYNC_P25", "SYNC_INVERT", "SYNC_SYMBOLS", "SYNC_RATE",
            "SYNC_CONFIRM_ERRORS", "DmrSync", "dmr_sync_patterns", "dmr_sync_pattern",
            "dmr_sync_pattern_name"]


# ============================================================================
# THE BITSTREAM ENGINE (§14-17): bit primitives, scramblers, FEC, CRC.
# ----------------------------------------------------------------------------
# Same names, arguments and return shapes as the twins in reference.py §14-17,
# so `atkdsp.X` and `atkdsp.reference.X` are interchangeable and ATK can fall
# back to the twin when the DLL is absent. Logging (Bill's rule, 2026-10-10):
# every stateful decoder logs describe() at INFO when made and its counters at
# DEBUG every 1000 calls; every error code is logged at WARNING with the call.
# ============================================================================
_bit_log = _logging.getLogger("atkdsp.bitstream")

BITS_INVERTED = 1
BITS_REVERSED = 2
SYNC_MAXBITS = 128
SYNC_REPETITIVE = 1
SCRAM_ADD = 0
SCRAM_MULT = 1
CONV_TERM = 0
CONV_TAILBITE = 1
CONV_STREAM = 2
RS_DUAL_BASIS = 1
CRC_END = 0
CRC_SEARCH = 1
E_DECODE = -5
_ERRORS[-5] = "uncorrectable"

bm_mask = _ref.bm_mask
crc_spec = _ref.crc_spec
ascii_bits = _ref.ascii_bits
cyclic_generator_matrix = _ref.cyclic_generator_matrix
extend_with_parity = _ref.extend_with_parity
SMALL_CODES = _ref.SMALL_CODES


def _chk(rc, what, **args):
    """_check, logging the call's arguments at WARNING first."""
    if rc < 0:
        _bit_log.warning("%s returned %d (%s); args %s", what, rc,
                         _ERRORS.get(rc, "error"), args)
    return _check(rc, what)


def _u8(a, name="bits") -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.uint8)
    if a.ndim != 1:
        raise AtkDspError(f"{name} must be one-dimensional")
    return a


def _u8p(a):
    return a.ctypes.data_as(C.POINTER(C.c_uint8))


class _BitHit(C.Structure):
    _fields_ = [("pos", C.c_longlong), ("errors", C.c_int), ("variant", C.c_int)]


class _SyncCand(C.Structure):
    _fields_ = [("first", C.c_longlong), ("period", C.c_int), ("length", C.c_int),
                ("count", C.c_int), ("flags", C.c_int), ("expected", C.c_double),
                ("log10_pfa", C.c_double), ("pattern", C.c_uint8 * SYNC_MAXBITS)]


class _BitStats(C.Structure):
    _fields_ = [("n", C.c_uint64), ("ones", C.c_uint64), ("density", C.c_double),
                ("run0", C.c_uint64 * 33), ("run1", C.c_uint64 * 33),
                ("ngram", (C.c_uint64 * 256) * 8),
                ("chi2", C.c_double), ("chi2_p", C.c_double), ("maurer", C.c_double),
                ("maurer_p", C.c_double), ("apen", C.c_double), ("apen_p", C.c_double),
                ("block_entropy", C.c_double * 16), ("chi2_bits", C.c_int),
                ("maurer_L", C.c_int), ("apen_m", C.c_int), ("reserved", C.c_int)]


class _ScramCand(C.Structure):
    _fields_ = [("poly", C.c_uint64), ("seed", C.c_uint64), ("g", C.c_double),
                ("log10_pfa", C.c_double), ("g_data", C.c_double), ("degree", C.c_int),
                ("terms", C.c_int), ("seeded", C.c_int), ("block_bits", C.c_int)]


class _FecStats(C.Structure):
    _fields_ = [("calls", C.c_uint64), ("in_", C.c_uint64), ("out", C.c_uint64),
                ("corrected", C.c_uint64), ("failures", C.c_uint64), ("resets", C.c_uint64)]


class _CrcSpec(C.Structure):
    _fields_ = [("poly", C.c_uint64), ("init", C.c_uint64), ("xorout", C.c_uint64),
                ("check", C.c_uint64), ("width", C.c_int), ("refin", C.c_int),
                ("refout", C.c_int), ("reserved", C.c_int)]


class _CrcCand(C.Structure):
    _fields_ = [("spec", _CrcSpec), ("layout", C.c_int), ("header_bits", C.c_int),
                ("trailer_bits", C.c_int), ("init_known", C.c_int), ("frames_ok", C.c_int),
                ("catalog", C.c_int), ("log10_pfa", C.c_double), ("alternatives", C.c_int),
                ("reserved", C.c_int)]


def _bind_bitstream(lib) -> None:
    P = C.POINTER
    u8 = P(C.c_uint8)
    fp = P(C.c_float)
    sz = C.c_size_t
    ss = C.c_ssize_t
    vp = C.c_void_p
    cp = C.c_char_p
    sig = {
        "atkdsp_bits_autocorr": (C.c_int, [u8, sz, sz, fp]),
        "atkdsp_bits_xcorr": (C.c_int, [u8, sz, u8, sz, fp]),
        "atkdsp_bits_search": (ss, [u8, sz, u8, sz, C.c_int, C.c_uint, P(_BitHit), sz]),
        "atkdsp_bits_columns": (ss, [u8, sz, sz, sz, P(C.c_uint32), P(C.c_uint32), u8, u8]),
        "atkdsp_bits_width_scan": (C.c_int, [u8, sz, sz, sz, fp, fp]),
        "atkdsp_bits_sync_discover": (ss, [u8, sz, C.c_int, C.c_int, C.c_double,
                                           P(_SyncCand), sz]),
        "atkdsp_bits_stats": (C.c_int, [u8, sz, P(_BitStats)]),
        "atkdsp_bits_wht": (ss, [u8, sz, sz, P(C.c_uint32), C.c_int, P(C.c_int32)]),
        "atkdsp_bits_walsh": (ss, [u8, sz, sz, C.c_int, fp, fp]),
        "atkdsp_bits_cyclic": (ss, [u8, sz, sz, C.c_int, fp, fp]),
        "atkdsp_bm": (ss, [u8, sz, u8, sz]),
        "atkdsp_bm_profile": (ss, [u8, sz, P(C.c_int32)]),
        "atkdsp_descramble_add": (C.c_int, [u8, sz, C.c_uint64, P(C.c_uint64), u8]),
        "atkdsp_descramble_mult": (C.c_int, [u8, sz, C.c_uint64, P(C.c_uint64), u8]),
        "atkdsp_scramble_mult": (C.c_int, [u8, sz, C.c_uint64, P(C.c_uint64), u8]),
        "atkdsp_scrambler_search": (ss, [u8, sz, C.c_int, C.c_int, C.c_int, C.c_double,
                                         P(C.c_double), P(_ScramCand), sz]),
        "atkdsp_viterbi_create": (vp, [C.c_int, C.c_int, P(C.c_uint32), u8, C.c_int, C.c_int,
                                       C.c_int, sz]),
        "atkdsp_viterbi_destroy": (None, [vp]),
        "atkdsp_viterbi_reset": (None, [vp]),
        "atkdsp_viterbi_out_max": (sz, [vp, sz]),
        "atkdsp_viterbi_decode": (ss, [vp, fp, sz, u8, sz, fp]),
        "atkdsp_viterbi_flush": (ss, [vp, u8, sz]),
        "atkdsp_viterbi_describe": (C.c_int, [vp, cp, sz]),
        "atkdsp_viterbi_stats": (C.c_int, [vp, P(_FecStats)]),
        "atkdsp_conv_encode": (ss, [C.c_int, C.c_int, P(C.c_uint32), u8, C.c_int, C.c_int,
                                    u8, sz, u8, sz]),
        "atkdsp_rs_create": (vp, [C.c_int, C.c_uint32, C.c_int, C.c_int, C.c_int, C.c_int,
                                  C.c_int]),
        "atkdsp_rs_destroy": (None, [vp]),
        "atkdsp_rs_params": (C.c_int, [vp, P(C.c_int), P(C.c_int)]),
        "atkdsp_rs_encode": (C.c_int, [vp, P(C.c_uint16), P(C.c_uint16)]),
        "atkdsp_rs_decode": (C.c_int, [vp, P(C.c_uint16), P(C.c_int), C.c_int, P(C.c_int)]),
        "atkdsp_rs_describe": (C.c_int, [vp, cp, sz]),
        "atkdsp_rs_stats": (C.c_int, [vp, P(_FecStats)]),
        "atkdsp_bch_create": (vp, [C.c_int, C.c_int, C.c_int, C.c_uint32]),
        "atkdsp_bch_destroy": (None, [vp]),
        "atkdsp_bch_params": (C.c_int, [vp, P(C.c_int), P(C.c_int), P(C.c_int)]),
        "atkdsp_bch_generator": (C.c_int, [vp, u8, sz]),
        "atkdsp_bch_encode": (C.c_int, [vp, u8, u8]),
        "atkdsp_bch_decode": (C.c_int, [vp, u8, P(C.c_int)]),
        "atkdsp_bch_describe": (C.c_int, [vp, cp, sz]),
        "atkdsp_bch_stats": (C.c_int, [vp, P(_FecStats)]),
        "atkdsp_blk_create": (vp, [C.c_int, C.c_int, u8]),
        "atkdsp_blk_destroy": (None, [vp]),
        "atkdsp_blk_params": (C.c_int, [vp, P(C.c_int), P(C.c_int), P(C.c_int)]),
        "atkdsp_blk_encode": (C.c_int, [vp, u8, u8]),
        "atkdsp_blk_decode": (C.c_int, [vp, u8, u8]),
        "atkdsp_blk_describe": (C.c_int, [vp, cp, sz]),
        "atkdsp_blk_stats": (C.c_int, [vp, P(_FecStats)]),
        "atkdsp_crc": (C.c_int, [u8, sz, P(_CrcSpec), P(C.c_uint64)]),
        "atkdsp_crc_catalog_count": (C.c_int, []),
        "atkdsp_crc_catalog": (C.c_int, [C.c_int, P(_CrcSpec)]),
        "atkdsp_crc_catalog_name": (cp, [C.c_int]),
        "atkdsp_crc_recover": (ss, [u8, P(C.c_size_t), C.c_int, C.c_int, C.c_int, C.c_int,
                                    C.c_double, P(_CrcCand), sz]),
    }
    for name, (res, args) in sig.items():
        f = getattr(lib, name)
        f.restype = res
        f.argtypes = args


_BIND_EXTRA.append(_bind_bitstream)


# ---- §14 bit primitives -----------------------------------------------------------
def bits_autocorr(b, max_lag: int) -> np.ndarray:
    b = _u8(b)
    out = np.empty(int(max_lag) + 1, np.float32)
    _chk(load().atkdsp_bits_autocorr(_u8p(b), b.size, int(max_lag), _fp(out)),
         "bits_autocorr", n=b.size, max_lag=max_lag)
    return out


def bits_xcorr(a, p) -> np.ndarray:
    a, p = _u8(a, "a"), _u8(p, "p")
    if p.size < 1 or p.size > a.size:
        raise AtkDspError("bits_xcorr needs 1 <= len(p) <= len(a)")
    out = np.empty(a.size - p.size + 1, np.float32)
    _chk(load().atkdsp_bits_xcorr(_u8p(a), a.size, _u8p(p), p.size, _fp(out)),
         "bits_xcorr", na=a.size, nb=p.size)
    return out


def bits_search(b, pat, max_err: int = 0, flags: int = 0) -> list:
    """[(pos, errors, variant)] — see atkdsp.h §14."""
    lib = load()
    b, pat = _u8(b), _u8(pat, "pat")
    cap = 4096
    while True:
        buf = (_BitHit * cap)()
        n = lib.atkdsp_bits_search(_u8p(b), b.size, _u8p(pat), pat.size, int(max_err),
                                   int(flags), buf, cap)
        if n == -3:
            cap = int(lib.atkdsp_bits_search(_u8p(b), b.size, _u8p(pat), pat.size,
                                             int(max_err), int(flags), None, 0))
            continue
        _chk(n, "bits_search", n=b.size, plen=pat.size, max_err=max_err, flags=flags)
        return [(int(buf[i].pos), int(buf[i].errors), int(buf[i].variant)) for i in range(n)]


def bits_columns(b, width: int, offset: int = 0):
    b = _u8(b)
    rows = (b.size - offset) // width if offset < b.size else 0
    col = np.zeros(width, np.uint32)
    row = np.zeros(max(rows, 1), np.uint32)
    cpar = np.zeros(width, np.uint8)
    rpar = np.zeros(max(rows, 1), np.uint8)
    r = _chk(load().atkdsp_bits_columns(_u8p(b), b.size, int(width), int(offset),
                                        col.ctypes.data_as(C.POINTER(C.c_uint32)),
                                        row.ctypes.data_as(C.POINTER(C.c_uint32)),
                                        _u8p(cpar), _u8p(rpar)),
             "bits_columns", n=b.size, width=width, offset=offset)
    return col, row[:r], cpar, rpar[:r]


def bits_width_scan(b, w_lo: int, w_hi: int):
    b = _u8(b)
    z = np.empty(w_hi - w_lo + 1, np.float32)
    raw = np.empty_like(z)
    _chk(load().atkdsp_bits_width_scan(_u8p(b), b.size, int(w_lo), int(w_hi), _fp(z), _fp(raw)),
         "bits_width_scan", n=b.size, w_lo=w_lo, w_hi=w_hi)
    return z, raw


def bits_sync_discover(b, word_len: int = 16, min_count: int = 3, pfa: float = 1e-3,
                       cap: int = 64) -> list:
    b = _u8(b)
    buf = (_SyncCand * max(int(cap), 1))()
    n = _chk(load().atkdsp_bits_sync_discover(_u8p(b), b.size, int(word_len), int(min_count),
                                              float(pfa), buf, int(cap)),
             "bits_sync_discover", n=b.size, word_len=word_len, min_count=min_count)
    out = []
    for i in range(n):
        c = buf[i]
        out.append({"first": int(c.first), "period": int(c.period), "length": int(c.length),
                    "count": int(c.count), "flags": int(c.flags),
                    "expected": float(c.expected), "log10_pfa": float(c.log10_pfa),
                    "pattern": np.array(c.pattern[:c.length], np.uint8)})
    return out


def bits_stats(b) -> dict:
    b = _u8(b)
    s = _BitStats()
    _chk(load().atkdsp_bits_stats(_u8p(b), b.size, C.byref(s)), "bits_stats", n=b.size)
    return {"n": int(s.n), "ones": int(s.ones), "density": float(s.density),
            "run0": np.ctypeslib.as_array(s.run0).copy(),
            "run1": np.ctypeslib.as_array(s.run1).copy(),
            "ngram": np.ctypeslib.as_array(s.ngram).copy(),
            "chi2": float(s.chi2), "chi2_p": float(s.chi2_p), "chi2_bits": int(s.chi2_bits),
            "maurer": float(s.maurer), "maurer_p": float(s.maurer_p),
            "maurer_L": int(s.maurer_L), "apen": float(s.apen), "apen_p": float(s.apen_p),
            "apen_m": int(s.apen_m),
            "block_entropy": np.ctypeslib.as_array(s.block_entropy).copy()}


def bits_wht(b, width: int, cols):
    b = _u8(b)
    cols = np.ascontiguousarray(cols, np.uint32)
    spec = np.empty(1 << cols.size, np.int32)
    rows = _chk(load().atkdsp_bits_wht(_u8p(b), b.size, int(width),
                                       cols.ctypes.data_as(C.POINTER(C.c_uint32)), cols.size,
                                       spec.ctypes.data_as(C.POINTER(C.c_int32))),
                "bits_wht", n=b.size, width=width, ncols=cols.size)
    return spec, rows


def bits_walsh(b, size: int, navg: int = 0):
    b = _u8(b)
    seq = np.empty(size, np.float32)
    dy = np.empty(size, np.float32)
    k = _chk(load().atkdsp_bits_walsh(_u8p(b), b.size, int(size), int(navg), _fp(seq), _fp(dy)),
             "bits_walsh", n=b.size, size=size, navg=navg)
    return seq, dy, k


def bits_cyclic(b, size: int, navg: int = 0):
    b = _u8(b)
    tm = np.empty(size, np.float32)
    fq = np.empty(size // 2 + 1, np.float32)
    k = _chk(load().atkdsp_bits_cyclic(_u8p(b), b.size, int(size), int(navg), _fp(tm), _fp(fq)),
             "bits_cyclic", n=b.size, size=size, navg=navg)
    return tm, fq, k


# ---- §15 scramblers and linear complexity ----------------------------------------------
def bm(b) -> tuple:
    b = _u8(b)
    poly = np.zeros(b.size + 2, np.uint8)
    L = _chk(load().atkdsp_bm(_u8p(b), b.size, _u8p(poly), poly.size), "bm", n=b.size)
    return L, poly[:L + 1].copy()


def bm_profile(b) -> np.ndarray:
    b = _u8(b)
    out = np.zeros(b.size, np.int32)
    _chk(load().atkdsp_bm_profile(_u8p(b), b.size, out.ctypes.data_as(C.POINTER(C.c_int32))),
         "bm_profile", n=b.size)
    return out


def _scr(fn, what, b, poly, state):
    b = _u8(b)
    out = np.empty_like(b)
    st = C.c_uint64(int(state[0]))
    _chk(getattr(load(), fn)(_u8p(b), b.size, C.c_uint64(int(poly)), C.byref(st), _u8p(out)),
         what, n=b.size, poly=hex(int(poly)), state=hex(int(state[0])))
    state[0] = st.value
    return out


def descramble_add(b, poly, state) -> np.ndarray:
    return _scr("atkdsp_descramble_add", "descramble_add", b, poly, state)


def descramble_mult(b, poly, state) -> np.ndarray:
    return _scr("atkdsp_descramble_mult", "descramble_mult", b, poly, state)


def scramble_mult(b, poly, state) -> np.ndarray:
    return _scr("atkdsp_scramble_mult", "scramble_mult", b, poly, state)


def scrambler_search(b, kind: int = SCRAM_MULT, max_degree: int = 24, max_terms: int = 5,
                     pfa: float = 1e-3, cap: int = 16):
    b = _u8(b)
    buf = (_ScramCand * max(int(cap), 1))()
    gid = C.c_double()
    n = _chk(load().atkdsp_scrambler_search(_u8p(b), b.size, int(kind), int(max_degree),
                                            int(max_terms), float(pfa), C.byref(gid), buf,
                                            int(cap)),
             "scrambler_search", n=b.size, kind=kind, max_degree=max_degree,
             max_terms=max_terms)
    out = [{"poly": int(c.poly), "seed": int(c.seed), "g": float(c.g),
            "log10_pfa": float(c.log10_pfa), "g_data": float(c.g_data),
            "degree": int(c.degree), "terms": int(c.terms), "seeded": int(c.seeded),
            "block_bits": int(c.block_bits)} for c in buf[:n]]
    return out, float(gid.value)


# ---- §16 FEC -----------------------------------------------------------------------------
def _stats_dict(s) -> dict:
    return {"calls": s.calls, "in": s.in_, "out": s.out, "corrected": s.corrected,
            "failures": s.failures, "resets": s.resets}


def _punc_arg(n, puncture):
    if puncture is None:
        return None, 0
    p = (np.asarray(puncture, np.uint8).reshape(n, -1) != 0).astype(np.uint8)
    return np.ascontiguousarray(p), p.shape[1]


def conv_encode(K: int, polys, bits, mode: int = CONV_TERM, puncture=None) -> np.ndarray:
    polys_a = np.ascontiguousarray(polys, np.uint32)
    n = polys_a.size
    bits = _u8(bits)
    p, plen = _punc_arg(n, puncture)
    cap = n * (bits.size + K) + 8
    out = np.empty(cap, np.uint8)
    m = _chk(load().atkdsp_conv_encode(int(K), n, polys_a.ctypes.data_as(C.POINTER(C.c_uint32)),
                                       _u8p(p) if p is not None else None, plen, int(mode),
                                       _u8p(bits), bits.size, _u8p(out), cap),
             "conv_encode", K=K, n=n, mode=mode, nbits=bits.size)
    return out[:m].copy()


class _Handle:
    _kind = ""

    def _log_made(self):
        _bit_log.info("%s created: %s", self._kind, self.describe())
        self._ncalls = 0

    def _tick(self):
        self._ncalls += 1
        if self._ncalls % 1000 == 0:
            _bit_log.debug("%s stats: %s", self._kind, self.stats())

    def describe(self) -> str:
        buf = C.create_string_buffer(1024)
        getattr(self._lib, f"atkdsp_{self._kind}_describe")(self._h, buf, 1024)
        return buf.value.decode()

    def stats(self) -> dict:
        s = _FecStats()
        getattr(self._lib, f"atkdsp_{self._kind}_stats")(self._h, C.byref(s))
        return _stats_dict(s)

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            getattr(self._lib, f"atkdsp_{self._kind}_destroy")(h)


class Viterbi(_Handle):
    """General Viterbi decoder (atkdsp.h §16a); the twin is reference.Viterbi."""
    _kind = "viterbi"

    def __init__(self, K: int, polys, puncture=None, mode: int = CONV_TERM, depth: int = 0,
                 max_bits: int = 4096):
        self._lib = load()
        self.K = int(K)
        self.polys = [int(x) for x in polys]
        self.n = len(self.polys)
        self.mode = int(mode)
        pa = np.ascontiguousarray(self.polys, np.uint32)
        p, plen = _punc_arg(self.n, puncture)
        self._h = self._lib.atkdsp_viterbi_create(
            self.K, self.n, pa.ctypes.data_as(C.POINTER(C.c_uint32)),
            _u8p(p) if p is not None else None, plen, self.mode, int(depth), int(max_bits))
        if not self._h:
            raise AtkDspError(f"viterbi_create(K={K}, polys={[oct(x) for x in self.polys]}, "
                              f"mode={mode}, depth={depth}, max_bits={max_bits}) refused — "
                              "K 2..15, n 1..8, generators below 2^K, every puncture column "
                              "sending something, max_bits >= 1 for blocks")
        self.metric = float("nan")
        self.depth = int(depth) or ((12 if puncture is not None else 6) * self.K
                                    if self.mode == CONV_STREAM else 0)
        self._log_made()

    def reset(self) -> None:
        self._lib.atkdsp_viterbi_reset(self._h)

    def decode(self, llr) -> np.ndarray:
        llr = _f32(llr, "llr")
        cap = int(self._lib.atkdsp_viterbi_out_max(self._h, llr.size))
        out = np.empty(max(cap, 1), np.uint8)
        met = C.c_float()
        m = _chk(self._lib.atkdsp_viterbi_decode(self._h, _fp(llr), llr.size, _u8p(out), cap,
                                                 C.byref(met)),
                 "viterbi_decode", nllr=llr.size, describe=self.describe())
        self.metric = float(met.value)
        self._tick()
        return out[:m].copy()

    def flush(self) -> np.ndarray:
        out = np.empty(2 * self.depth + 16, np.uint8)
        m = _chk(self._lib.atkdsp_viterbi_flush(self._h, _u8p(out), out.size), "viterbi_flush")
        return out[:m].copy()


class RS(_Handle):
    """Reed-Solomon over GF(2^m) (atkdsp.h §16b); the twin is reference.RS."""
    _kind = "rs"

    def __init__(self, m, gfpoly, fcr, prim, nroots, pad=0, dual_basis=False):
        self._lib = load()
        self._h = self._lib.atkdsp_rs_create(int(m), int(gfpoly), int(fcr), int(prim),
                                             int(nroots), int(pad),
                                             RS_DUAL_BASIS if dual_basis else 0)
        if not self._h:
            raise AtkDspError(f"rs_create(m={m}, gfpoly={gfpoly:#x}, fcr={fcr}, prim={prim}, "
                              f"nroots={nroots}, pad={pad}, dual={dual_basis}) refused")
        n, k = C.c_int(), C.c_int()
        self._lib.atkdsp_rs_params(self._h, C.byref(n), C.byref(k))
        self.n, self.k, self.nroots = n.value, k.value, int(nroots)
        self._log_made()

    def encode(self, data) -> np.ndarray:
        d = np.ascontiguousarray(data, np.uint16)
        par = np.empty(self.nroots, np.uint16)
        _chk(self._lib.atkdsp_rs_encode(self._h, d.ctypes.data_as(C.POINTER(C.c_uint16)),
                                        par.ctypes.data_as(C.POINTER(C.c_uint16))), "rs_encode")
        return par

    def decode(self, sym, erasures=()):
        s = np.array(sym, np.uint16)
        er = np.ascontiguousarray(list(erasures), np.int32)
        pos = np.zeros(self.nroots + 1, np.int32)
        rc = self._lib.atkdsp_rs_decode(self._h, s.ctypes.data_as(C.POINTER(C.c_uint16)),
                                        er.ctypes.data_as(C.POINTER(C.c_int)) if er.size else None,
                                        er.size, pos.ctypes.data_as(C.POINTER(C.c_int)))
        self._tick()
        if rc == E_DECODE:
            return np.array(sym, np.uint16), E_DECODE, []
        _chk(rc, "rs_decode", n=s.size, erasures=list(er))
        return s, rc, [int(p) for p in pos[:rc]]


class BCH(_Handle):
    """Binary BCH (atkdsp.h §16c); the twin is reference.BCH."""
    _kind = "bch"

    def __init__(self, m, t, shorten=0, gfpoly=0):
        self._lib = load()
        self._h = self._lib.atkdsp_bch_create(int(m), int(t), int(shorten), int(gfpoly))
        if not self._h:
            raise AtkDspError(f"bch_create(m={m}, t={t}, shorten={shorten}, "
                              f"gfpoly={gfpoly:#x}) refused")
        n, k, tt = C.c_int(), C.c_int(), C.c_int()
        self._lib.atkdsp_bch_params(self._h, C.byref(n), C.byref(k), C.byref(tt))
        self.n, self.k, self.t = n.value, k.value, tt.value
        self._log_made()

    def generator(self) -> np.ndarray:
        g = np.empty(self.n - self.k + 1, np.uint8)
        self._lib.atkdsp_bch_generator(self._h, _u8p(g), g.size)
        return g

    def encode(self, data) -> np.ndarray:
        d = _u8(data, "data")
        out = np.empty(self.n, np.uint8)
        _chk(self._lib.atkdsp_bch_encode(self._h, _u8p(d), _u8p(out)), "bch_encode")
        return out

    def decode(self, code):
        c = np.array(code, np.uint8)
        pos = np.zeros(self.t + 1, np.int32)
        rc = self._lib.atkdsp_bch_decode(self._h, _u8p(c), pos.ctypes.data_as(C.POINTER(C.c_int)))
        self._tick()
        if rc == E_DECODE:
            return np.array(code, np.uint8), E_DECODE, []
        _chk(rc, "bch_decode", n=c.size)
        return c, rc, [int(p) for p in pos[:rc]]


class BlockCode(_Handle):
    """Any small linear code by syndrome table (atkdsp.h §16d); twin reference.BlockCode."""
    _kind = "blk"

    def __init__(self, G):
        self._lib = load()
        G = np.ascontiguousarray(np.asarray(G, np.uint8) & 1)
        k, n = G.shape
        self._h = self._lib.atkdsp_blk_create(int(n), int(k), _u8p(G.ravel()))
        if not self._h:
            raise AtkDspError(f"blk_create({n}, {k}) refused — n <= 32, n - k <= 20, rank k")
        nn, kk, dm = C.c_int(), C.c_int(), C.c_int()
        self._lib.atkdsp_blk_params(self._h, C.byref(nn), C.byref(kk), C.byref(dm))
        self.n, self.k, self.dmin = nn.value, kk.value, dm.value
        self._log_made()

    def encode(self, data) -> np.ndarray:
        d = _u8(data, "data")
        out = np.empty(self.n, np.uint8)
        _chk(self._lib.atkdsp_blk_encode(self._h, _u8p(d), _u8p(out)), "blk_encode")
        return out

    def decode(self, code):
        c = np.array(code, np.uint8)
        d = np.zeros(self.k, np.uint8)
        rc = self._lib.atkdsp_blk_decode(self._h, _u8p(c), _u8p(d))
        self._tick()
        if rc == E_DECODE:
            return np.array(code, np.uint8), None, E_DECODE
        _chk(rc, "blk_decode", n=c.size)
        return c, d, rc


# ---- §17 CRC ---------------------------------------------------------------------------
def _spec_c(spec) -> _CrcSpec:
    s = _CrcSpec()
    s.width, s.poly, s.init = int(spec["width"]), int(spec["poly"]), int(spec["init"])
    s.refin, s.refout, s.xorout = int(spec["refin"]), int(spec["refout"]), int(spec["xorout"])
    s.check = int(spec.get("check", 0))
    return s


def _spec_dict(s) -> dict:
    return {"width": s.width, "poly": s.poly, "init": s.init, "refin": s.refin,
            "refout": s.refout, "xorout": s.xorout, "check": s.check}


def crc(bits, spec) -> int:
    b = _u8(bits)
    v = C.c_uint64()
    _chk(load().atkdsp_crc(_u8p(b), b.size, C.byref(_spec_c(spec)), C.byref(v)), "crc",
         spec=spec)
    return int(v.value)


def crc_catalog() -> list:
    """[(name, spec dict)] as the library holds them."""
    lib = load()
    out = []
    for i in range(lib.atkdsp_crc_catalog_count()):
        s = _CrcSpec()
        lib.atkdsp_crc_catalog(i, C.byref(s))
        out.append((lib.atkdsp_crc_catalog_name(i).decode(), _spec_dict(s)))
    return out


def crc_recover(frames, width: int = 0, pos: int = CRC_END, max_header: int = 0,
                pfa: float = 1e-3, cap: int = 8) -> list:
    frames = [_u8(f, "frame") for f in frames]
    bits = np.ascontiguousarray(np.concatenate(frames) if frames else np.zeros(0, np.uint8))
    lens = np.ascontiguousarray([f.size for f in frames], np.uintp)
    buf = (_CrcCand * max(int(cap), 1))()
    n = _chk(load().atkdsp_crc_recover(_u8p(bits), lens.ctypes.data_as(C.POINTER(C.c_size_t)),
                                       len(frames), int(width), int(pos), int(max_header),
                                       float(pfa), buf, int(cap)),
             "crc_recover", nframes=len(frames), width=width, pos=pos, max_header=max_header)
    return [{"spec": _spec_dict(c.spec), "layout": c.layout, "header_bits": c.header_bits,
             "trailer_bits": c.trailer_bits, "init_known": c.init_known,
             "frames_ok": c.frames_ok, "catalog": c.catalog, "log10_pfa": c.log10_pfa,
             "alternatives": c.alternatives}
            for c in buf[:n]]


__all__ += ["BITS_INVERTED", "BITS_REVERSED", "SYNC_MAXBITS", "SYNC_REPETITIVE", "SCRAM_ADD",
            "SCRAM_MULT", "CONV_TERM", "CONV_TAILBITE", "CONV_STREAM", "RS_DUAL_BASIS",
            "CRC_END", "CRC_SEARCH", "E_DECODE", "bits_autocorr", "bits_xcorr", "bits_search",
            "bits_columns", "bits_width_scan", "bits_sync_discover", "bits_stats", "bits_wht",
            "bits_walsh", "bits_cyclic", "bm", "bm_profile", "bm_mask", "descramble_add",
            "descramble_mult", "scramble_mult", "scrambler_search", "conv_encode", "Viterbi",
            "RS", "BCH", "BlockCode", "SMALL_CODES", "cyclic_generator_matrix",
            "extend_with_parity", "crc", "crc_catalog", "crc_spec", "crc_recover", "ascii_bits"]
