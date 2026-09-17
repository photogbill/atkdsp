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
    "unpack", "Nco", "Fir", "Resampler", "Fft", "window", "power_db",
    "spectrum_reduce", "fm_demod", "am_demod", "db_to_pixels", "decimate_max",
    "median", "detect_channels", "stitch_max", "Channel",
]

#: The ABI this binding was written against. A library reporting anything
#: else is refused by :func:`load`.
ABI_VERSION = 1

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


__all__ += ["design_lowpass", "Ddc"]
