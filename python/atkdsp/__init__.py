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
import math
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
#: 16 adds naming the code (§18): rank scan, block / conv / RS / interleaver id.
#: 17 adds the detection track (§19-21): CFAR + burst tracker, the burst
#:    descriptor word, the ring buffer; ATKDSP_E_OVERRUN and ATKDSP_E_EMPTY.
#: 18 adds phase 3 (§22-24): pulse runs (Pulses), the cyclic probes (Cyclic,
#:    CpProbe), the analog radio IDs (ToneId); the CFAR config's rho2 (the
#:    window's bin correlation) and block_code_id's fixed_bits.
ABI_VERSION = 26

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


# ============================================================================
# §18 NAMING THE CODE — rank scan, block / convolutional / RS / interleaver
# recognition. Same names and returns as reference.py §18 (dicts), so ATK's
# fallback is a drop-in. Each result carries its evidence (log10_pfa,
# deficiency, decode -> re-encode agreement); "no code found" is a result.
# ============================================================================
_id_log = _logging.getLogger("atkdsp.coderec")


class _BlockId(C.Structure):
    _fields_ = [("n", C.c_int), ("k", C.c_int), ("offset", C.c_int), ("deficiency", C.c_int),
                ("cyclic", C.c_int), ("extended", C.c_int), ("dmin", C.c_int),
                ("rows_used", C.c_int), ("robust", C.c_int), ("shortened", C.c_int),
                ("fixed_bits", C.c_int), ("reserved", C.c_int),
                ("log10_pfa", C.c_double), ("ber_est", C.c_double), ("agreement", C.c_double),
                ("valid_words", C.c_double), ("name", C.c_char * 48)]


class _ConvId(C.Structure):
    _fields_ = [("n", C.c_int), ("k", C.c_int), ("K", C.c_int), ("phase", C.c_int),
                ("polys", C.c_uint32 * 8), ("robust", C.c_int), ("rows_used", C.c_int),
                ("density", C.c_double), ("log10_pfa", C.c_double), ("ber_est", C.c_double),
                ("agreement", C.c_double), ("name", C.c_char * 64)]


class _RsId(C.Structure):
    _fields_ = [("m", C.c_int), ("n", C.c_int), ("k", C.c_int), ("nroots", C.c_int),
                ("fcr", C.c_int), ("prim", C.c_int), ("gfpoly", C.c_uint32),
                ("dual_basis", C.c_int), ("lsb_first", C.c_int), ("rows_used", C.c_int),
                ("log10_pfa", C.c_double), ("valid_words", C.c_double),
                ("agreement", C.c_double), ("name", C.c_char * 48)]


class _IlvId(C.Structure):
    _fields_ = [("period", C.c_int), ("offset", C.c_int), ("rows", C.c_int), ("cols", C.c_int),
                ("inner_width", C.c_int), ("inner_deficiency", C.c_int),
                ("log10_pfa", C.c_double)]


def _bind_coderec(lib) -> None:
    P = C.POINTER
    u8 = P(C.c_uint8)
    sz = C.c_size_t
    lib.atkdsp_gf2_rank_scan.restype = C.c_int
    lib.atkdsp_gf2_rank_scan.argtypes = [u8, sz, sz, sz, sz, sz, P(C.c_int16)]
    lib.atkdsp_block_code_id.restype = C.c_int
    lib.atkdsp_block_code_id.argtypes = [u8, sz, C.c_int, C.c_int, C.c_double, C.c_uint64,
                                         P(_BlockId), u8, sz, u8, sz]
    lib.atkdsp_conv_code_id.restype = C.c_int
    lib.atkdsp_conv_code_id.argtypes = [u8, sz, C.c_int, C.c_int, C.c_double, C.c_uint64,
                                        P(_ConvId)]
    lib.atkdsp_rs_code_id.restype = C.c_int
    lib.atkdsp_rs_code_id.argtypes = [u8, sz, sz, C.c_int, C.c_int, C.c_double, P(_RsId)]
    lib.atkdsp_interleaver_id.restype = C.c_int
    lib.atkdsp_interleaver_id.argtypes = [u8, sz, C.c_int, C.c_int, C.c_double, C.c_uint64,
                                          P(_IlvId)]


_BIND_EXTRA.append(_bind_coderec)


def _struct_dict(s) -> dict:
    out = {}
    for name, _t in s._fields_:
        v = getattr(s, name)
        if name == "name":
            v = v.decode(errors="replace")
        elif name == "polys":
            v = [int(x) for x in v]
        out[name] = v
    out.pop("reserved", None)
    return out


def gf2_rank_scan(b, w_lo: int, w_hi: int, n_offsets: int = 1, rows: int = 0) -> np.ndarray:
    """Deficiency heat map, int16 [width][offset] (-1 = not computed). See §18."""
    b = _u8(b)
    out = np.empty((w_hi - w_lo + 1, n_offsets), np.int16)
    _chk(load().atkdsp_gf2_rank_scan(_u8p(b), b.size, int(w_lo), int(w_hi), int(n_offsets),
                                     int(rows), out.ctypes.data_as(C.POINTER(C.c_int16))),
         "gf2_rank_scan", n=b.size, w_lo=w_lo, w_hi=w_hi, n_offsets=n_offsets)
    return out


def block_code_id(b, n_lo: int = 4, n_hi: int = 256, pfa: float = 1e-3, seed: int = 0) -> dict:
    """Name a block code (§18). The dict carries the atkdsp_blockid fields plus
    'found', 'G' (k x n uint8, canonical) and 'gpoly' (highest degree first)."""
    b = _u8(b)
    s = _BlockId()
    G = np.zeros(512 * 512, np.uint8)
    gp = np.zeros(513, np.uint8)
    rc = _chk(load().atkdsp_block_code_id(_u8p(b), b.size, int(n_lo), int(n_hi), float(pfa),
                                          int(seed), C.byref(s), _u8p(G), G.size, _u8p(gp), gp.size),
              "block_code_id", n=b.size, n_lo=n_lo, n_hi=n_hi)
    d = _struct_dict(s)
    d.pop("reserved", None)
    d["found"] = bool(rc)
    d["G"] = G[:s.k * s.n].reshape(s.k, s.n).copy() if rc else None
    nz = np.flatnonzero(gp)                     # g(0) = 1: the last coefficient is set
    d["gpoly"] = (gp[:nz[-1] + 1].copy() if rc and (s.cyclic or s.extended or s.shortened) and nz.size
                  else None)
    _id_log.info("block_code_id: %s (n=%d k=%d offset=%d, log10_pfa %.1f, agreement %s)",
                 d["name"], s.n, s.k, s.offset, s.log10_pfa, s.agreement)
    return d


def conv_code_id(b, max_n: int = 8, max_K: int = 15, pfa: float = 1e-3, seed: int = 0) -> dict:
    """Name a convolutional code (§18): rate k/n (n <= max_n), and for k = 1
    the constraint length and generators (octal, §16a convention, stream
    order) with Viterbi re-encode agreement. 'polys' is [] when k > 1."""
    b = _u8(b)
    s = _ConvId()
    rc = _chk(load().atkdsp_conv_code_id(_u8p(b), b.size, int(max_n), int(max_K), float(pfa),
                                         int(seed), C.byref(s)),
              "conv_code_id", n=b.size, max_n=max_n, max_K=max_K)
    d = _struct_dict(s)
    d["found"] = bool(rc)
    d["polys"] = d["polys"][:s.n] if rc and s.K else []     # k > 1: no generators
    _id_log.info("conv_code_id: %s (agreement %s)", d["name"], s.agreement)
    return d


def rs_code_id(b, offset: int = 0, m: int = 0, nsym: int = 0, pfa: float = 1e-3) -> dict:
    """Name a Reed-Solomon code (§18) on codeword-aligned bits from `offset`:
    m (0 = 3..8), field polynomial, bit order, dual basis, n, k, fcr, prim,
    with the §16b decoder's valid words and agreement."""
    b = _u8(b)
    s = _RsId()
    rc = _chk(load().atkdsp_rs_code_id(_u8p(b), b.size, int(offset), int(m), int(nsym),
                                       float(pfa), C.byref(s)),
              "rs_code_id", n=b.size, offset=offset, m=m, nsym=nsym)
    d = _struct_dict(s)
    d["found"] = bool(rc)
    _id_log.info("rs_code_id: %s (valid %s)", d["name"], s.valid_words)
    return d


def interleaver_id(b, n_lo: int = 8, n_hi: int = 512, pfa: float = 1e-3, seed: int = 0) -> dict:
    """A block interleaver's period, block start and rows x cols (written by
    rows, read by columns), from the inner code's reappearance (§18)."""
    b = _u8(b)
    s = _IlvId()
    rc = _chk(load().atkdsp_interleaver_id(_u8p(b), b.size, int(n_lo), int(n_hi), float(pfa),
                                           int(seed), C.byref(s)),
              "interleaver_id", n=b.size, n_lo=n_lo, n_hi=n_hi)
    d = _struct_dict(s)
    d["found"] = bool(rc)
    _id_log.info("interleaver_id: period %d rows %d cols %d", s.period, s.rows, s.cols)
    return d


__all__ += ["gf2_rank_scan", "block_code_id", "conv_code_id", "rs_code_id", "interleaver_id"]


# ============================================================================
# THE DETECTION TRACK (v0.16.0, ABI 17): §19 CFAR + burst tracker, §20 the
# burst descriptor word, §21 the ring. Logging as the bitstream kernels:
# describe() at INFO when made, counters at DEBUG every 1000 calls, every
# error code at WARNING with the call's arguments.
# ============================================================================
_det_log = _logging.getLogger("atkdsp.detect")
E_OVERRUN = -6
E_EMPTY = -7
_ERRORS[-6] = "overrun (the ring lapped this reader)"
_ERRORS[-7] = "nothing there"

CFAR_CA, CFAR_OS, CFAR_GO, CFAR_SO = 0, 1, 2, 3
CFAR_OPEN, CFAR_UPDATE, CFAR_CLOSE = 1, 2, 3
CFAR_T0_COARSE, CFAR_T1_COARSE, CFAR_EDGE, CFAR_MASK_ADJ, CFAR_FLUSHED = 1, 2, 4, 8, 16
CFAR_MODES = {"ca": 0, "os": 1, "go": 2, "so": 3}
cfar_defaults = _ref.cfar_defaults


def _dchk(rc, what, **args):
    if rc < 0:
        _det_log.warning("%s returned %d (%s); args %s", what, rc,
                         _ERRORS.get(rc, "error"), args)
    return _check(rc, what)


class _CfarConfig(C.Structure):
    _fields_ = [("mode", C.c_int32), ("n_ref", C.c_int32), ("n_guard", C.c_int32),
                ("k_frames", C.c_int32), ("pfa", C.c_double), ("looks", C.c_double),
                ("os_k", C.c_int32), ("open_m", C.c_int32), ("open_n", C.c_int32),
                ("close_k", C.c_int32), ("merge_gap", C.c_int32), ("min_bins", C.c_int32),
                ("max_tracks", C.c_int32), ("reserved", C.c_int32), ("rho2", C.c_double * 4)]


class _CfarEvent(C.Structure):
    _fields_ = [("t0", C.c_double), ("t1", C.c_double), ("fc_bin", C.c_double),
                ("type", C.c_int32), ("id", C.c_uint32), ("flags", C.c_uint32),
                ("lo_bin", C.c_int32), ("hi_bin", C.c_int32), ("n_hits", C.c_int32),
                ("level_db", C.c_float), ("noise_db", C.c_float), ("snr_db", C.c_float),
                ("band_snr_db", C.c_float)]


class _CfarCounters(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in
                ("frames", "warmup", "cells_tested", "cell_hits", "untestable",
                 "opened", "closed", "updates", "dropped", "resets", "tracks")]


def _bind_cfar(lib) -> None:
    P = C.POINTER
    vp, sz, i, d = C.c_void_p, C.c_size_t, C.c_int, C.c_double
    lib.atkdsp_cfar_defaults.argtypes = [P(_CfarConfig)]
    lib.atkdsp_cfar_defaults.restype = None
    lib.atkdsp_cfar_alpha.argtypes = [i, i, d, d, i, P(d), P(d), P(d)]
    lib.atkdsp_cfar_alpha.restype = i
    lib.atkdsp_cfar_n_eff.argtypes = [i, P(d)]
    lib.atkdsp_cfar_n_eff.restype = d
    lib.atkdsp_cfar_window_rho2.argtypes = [P(C.c_float), sz, P(d)]
    lib.atkdsp_cfar_window_rho2.restype = i
    lib.atkdsp_cfar_create.argtypes = [sz, P(_CfarConfig)]
    lib.atkdsp_cfar_create.restype = vp
    lib.atkdsp_cfar_destroy.argtypes = [vp]
    lib.atkdsp_cfar_destroy.restype = None
    lib.atkdsp_cfar_reset.argtypes = [vp]
    lib.atkdsp_cfar_reset.restype = None
    lib.atkdsp_cfar_set_mask.argtypes = [vp, P(C.c_uint8)]
    lib.atkdsp_cfar_set_mask.restype = i
    lib.atkdsp_cfar_events_max.argtypes = [vp, sz]
    lib.atkdsp_cfar_events_max.restype = sz
    lib.atkdsp_cfar_process.argtypes = [vp, P(C.c_float), sz, d, d, P(_CfarEvent), sz]
    lib.atkdsp_cfar_process.restype = C.c_ssize_t
    lib.atkdsp_cfar_flush.argtypes = [vp, P(_CfarEvent), sz]
    lib.atkdsp_cfar_flush.restype = C.c_ssize_t
    lib.atkdsp_cfar_last.argtypes = [vp, P(C.c_float), P(C.c_float), P(C.c_uint8)]
    lib.atkdsp_cfar_last.restype = i
    lib.atkdsp_cfar_info.argtypes = [vp, P(d), P(d), P(d), P(_CfarConfig)]
    lib.atkdsp_cfar_info.restype = i
    lib.atkdsp_cfar_describe.argtypes = [vp, C.c_char_p, sz]
    lib.atkdsp_cfar_describe.restype = i
    lib.atkdsp_cfar_stats.argtypes = [vp, P(_CfarCounters)]
    lib.atkdsp_cfar_stats.restype = i


_BIND_EXTRA.append(_bind_cfar)


def _rho2_arg(rho2):
    """A (c_double * 4) for the C, or None for independent cells."""
    if rho2 is None:
        return None
    vals = [float(v) for v in rho2]
    if len(vals) != 4:
        raise AtkDspError("rho2 needs four correlations (bins 1..4 apart)")
    return (C.c_double * 4)(*vals)


def cfar_alpha(mode, n_ref: int, pfa: float, looks: float = 1.0, os_k: int = 0,
               rho2=None) -> tuple:
    """(alpha, E[T]) — the CFAR threshold multiplier solved for `pfa` per cell
    (atkdsp.h §19), with the window's bin correlation `rho2` (four values, or
    None for independent cells); the twin is reference.cfar_alpha."""
    mode = CFAR_MODES.get(mode, mode) if isinstance(mode, str) else int(mode)
    a, et = C.c_double(), C.c_double()
    _dchk(load().atkdsp_cfar_alpha(int(mode), int(n_ref), float(pfa), float(looks), int(os_k),
                                    _rho2_arg(rho2), C.byref(a), C.byref(et)),
          "cfar_alpha", mode=mode, n_ref=n_ref, pfa=pfa, looks=looks, os_k=os_k, rho2=rho2)
    return a.value, et.value


def cfar_n_eff(n_ref: int, rho2=None) -> float:
    """Equivalent independent cells among 2 n_ref correlated ones (atkdsp.h §19)."""
    return float(load().atkdsp_cfar_n_eff(int(n_ref), _rho2_arg(rho2)))


def cfar_window_rho2(window) -> tuple:
    """The power correlation of bins 1..4 apart through `window` (atkdsp.h §19):
    pass it as a Cfar's rho2 so the stated Pfa holds on windowed frames."""
    w = _f32(window, "window")
    out = (C.c_double * 4)()
    _check(load().atkdsp_cfar_window_rho2(_fp(w), w.size, out), "cfar_window_rho2")
    return tuple(float(v) for v in out)


def _cfar_event_dict(e) -> dict:
    return {"type": int(e.type), "id": int(e.id), "flags": int(e.flags), "lo": int(e.lo_bin),
            "hi": int(e.hi_bin), "n_hits": int(e.n_hits), "t0": float(e.t0),
            "t1": float(e.t1), "fc_bin": float(e.fc_bin), "level_db": float(e.level_db),
            "noise_db": float(e.noise_db), "snr_db": float(e.snr_db),
            "band_snr_db": float(e.band_snr_db)}


class Cfar:
    """CFAR across frequency + the burst tracker (atkdsp.h §19). Options are
    the atkdsp_cfar_config fields as keywords (mode may be 'ca'/'os'/'go'/
    'so'). process() returns OPEN / UPDATE / CLOSE events as dicts with the
    twin's keys (reference.Cfar). The twin is the fallback ATK uses when the
    library is absent."""

    _kind = "cfar"

    def __init__(self, nbins: int, **cfg):
        self._lib = load()
        c = _ref.cfar_defaults()
        for key, v in cfg.items():
            if key not in c:
                raise AtkDspError(f"unknown CFAR option {key!r}; known: {sorted(c)}")
            c[key] = CFAR_MODES.get(v, v) if key == "mode" and isinstance(v, str) else v
        for key in c:
            if key == "rho2":
                c[key] = (tuple(float(v) for v in c[key]) if _ref._rho2_ok(c[key])
                          else c[key])
            else:
                c[key] = float(c[key]) if key in ("pfa", "looks") else int(c[key])
        why = _ref._cfar_check(int(nbins), c)
        if why:
            raise AtkDspError(f"cfar_create({nbins} bins, {c}) refused — {why}")
        self.nbins = int(nbins)
        k = _CfarConfig()
        for key, v in c.items():
            if key == "rho2":
                k.rho2 = (C.c_double * 4)(*v)
            else:
                setattr(k, key, v)
        self._h = self._lib.atkdsp_cfar_create(self.nbins, C.byref(k))
        if not self._h:
            raise AtkDspError(f"cfar_create({nbins} bins, {c}) refused — the threshold "
                              "could not be solved for that Pfa (use more reference cells)")
        self._buf = None
        self._ncalls = 0
        a, ac, et, rc = C.c_double(), C.c_double(), C.c_double(), _CfarConfig()
        self._lib.atkdsp_cfar_info(self._h, C.byref(a), C.byref(ac), C.byref(et), C.byref(rc))
        self.alpha, self.alpha_ca, self.et = a.value, ac.value, et.value
        self.cfg = {n: getattr(rc, n) for n, _t in _CfarConfig._fields_ if n != "reserved"}
        self.cfg["rho2"] = tuple(float(v) for v in rc.rho2)
        self.n_eff = cfar_n_eff(self.cfg["n_ref"], self.cfg["rho2"])
        _det_log.info("cfar created: %s", self.describe())

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.atkdsp_cfar_destroy(h)

    def describe(self) -> str:
        buf = C.create_string_buffer(1024)
        self._lib.atkdsp_cfar_describe(self._h, buf, 1024)
        return buf.value.decode()

    def stats(self) -> dict:
        s = _CfarCounters()
        self._lib.atkdsp_cfar_stats(self._h, C.byref(s))
        return {n: int(getattr(s, n)) for n, _t in s._fields_}

    def reset(self) -> None:
        self._lib.atkdsp_cfar_reset(self._h)

    def set_mask(self, mask) -> None:
        if mask is None:
            self._lib.atkdsp_cfar_set_mask(self._h, None)
            return
        m = np.ascontiguousarray(np.asarray(mask).ravel() != 0, dtype=np.uint8)
        if m.size != self.nbins:
            raise AtkDspError(f"mask has {m.size} entries; the CFAR has {self.nbins} bins")
        _dchk(self._lib.atkdsp_cfar_set_mask(self._h, _u8p(m)), "cfar_set_mask")

    def events_max(self, nframes: int) -> int:
        return int(self._lib.atkdsp_cfar_events_max(self._h, int(nframes)))

    def process(self, frames_db, frame_t0: float, frame_dt: float) -> list:
        f = np.ascontiguousarray(frames_db, dtype=np.float32)
        if f.ndim == 1:
            f = f.reshape(1, -1)
        if f.ndim != 2 or f.shape[1] != self.nbins:
            raise AtkDspError(f"frames must be (nframes, {self.nbins}); got {f.shape}")
        nf = f.shape[0]
        cap = self.events_max(nf)
        if self._buf is None or len(self._buf) < cap:
            self._buf = (_CfarEvent * max(cap, 1))()
        n = _dchk(self._lib.atkdsp_cfar_process(self._h, _fp(f), nf, float(frame_t0),
                                                float(frame_dt), self._buf, cap),
                  "cfar_process", nframes=nf, frame_t0=frame_t0, frame_dt=frame_dt)
        self._ncalls += 1
        if self._ncalls % 1000 == 0:
            _det_log.debug("cfar stats: %s", self.stats())
        return [_cfar_event_dict(self._buf[j]) for j in range(n)]

    def flush(self) -> list:
        cap = int(self.cfg["max_tracks"])
        buf = (_CfarEvent * cap)()
        n = _dchk(self._lib.atkdsp_cfar_flush(self._h, buf, cap), "cfar_flush")
        return [_cfar_event_dict(buf[j]) for j in range(n)]

    def last(self):
        """(averaged power dB, threshold dB (NaN untested), hits) of the last
        tested frame, or None before one."""
        a = np.empty(self.nbins, np.float32)
        t = np.empty(self.nbins, np.float32)
        h = np.empty(self.nbins, np.uint8)
        rc = self._lib.atkdsp_cfar_last(self._h, _fp(a), _fp(t), _u8p(h))
        if rc == E_EMPTY:
            return None
        _dchk(rc, "cfar_last")
        return a, t, h


__all__ += ["E_OVERRUN", "E_EMPTY", "CFAR_CA", "CFAR_OS", "CFAR_GO", "CFAR_SO", "CFAR_OPEN",
            "CFAR_UPDATE", "CFAR_CLOSE", "CFAR_T0_COARSE", "CFAR_T1_COARSE", "CFAR_EDGE",
            "CFAR_MASK_ADJ", "CFAR_FLUSHED", "cfar_alpha", "Cfar"]


# ---- §20 the burst descriptor word ------------------------------------------------
BDW_TRUNC_START, BDW_TRUNC_END, BDW_NO_NOISE, BDW_LOW_SNR = 1, 2, 4, 8
BDW_RISE_UNRESOLVED, BDW_FALL_UNRESOLVED, BDW_BW26_FLOOR, BDW_SHORT = 16, 32, 64, 128
BDW_GAUSSIAN = 256
BDW_FLAG_NAMES = {1: "truncated start", 2: "truncated end", 4: "no noise outside",
                  8: "low SNR", 16: "rise unresolved", 32: "fall unresolved",
                  64: "-26 dB width below the noise floor", 128: "too short for drift",
                  256: "Gaussian-like envelope"}


class _Bdw(C.Structure):
    _fields_ = [("t0", C.c_double), ("t1", C.c_double), ("duration_s", C.c_double),
                ("fc_hz", C.c_double), ("drift_hz_s", C.c_double), ("obw99_hz", C.c_double),
                ("bw26_hz", C.c_double), ("rise_s", C.c_double), ("fall_s", C.c_double),
                ("reserved_d", C.c_double), ("peak_dbfs", C.c_float), ("mean_dbfs", C.c_float),
                ("noise_dbfs", C.c_float), ("snr_db", C.c_float), ("mod_hint", C.c_int32),
                ("flags", C.c_uint32), ("nfft", C.c_int32), ("window", C.c_int32)]


def _bind_burst(lib) -> None:
    lib.atkdsp_burst_measure.argtypes = [C.POINTER(_Cf32), C.c_size_t, C.c_double, C.c_size_t,
                                         C.c_size_t, C.c_float, C.c_int, C.POINTER(_Bdw)]
    lib.atkdsp_burst_measure.restype = C.c_int


_BIND_EXTRA.append(_bind_burst)


def burst_measure(iq, fs: float, i0: int, i1: int, thresh_db: float = 0.0,
                  nfft: int = 0):
    """One burst's descriptor word (atkdsp.h §20) from channelised I/Q and a
    coarse [i0, i1): a dict with the twin's keys (reference.burst_measure),
    or None when there is no burst there (ATKDSP_E_EMPTY)."""
    x = _cf32(iq, "iq")
    o = _Bdw()
    rc = load().atkdsp_burst_measure(_cfp(x), x.size, float(fs), int(i0), int(i1),
                                     float(thresh_db), int(nfft), C.byref(o))
    if rc == E_EMPTY:
        _det_log.debug("burst_measure: nothing at [%d, %d)", i0, i1)
        return None
    _dchk(rc, "burst_measure", n=x.size, fs=fs, i0=i0, i1=i1, thresh_db=thresh_db, nfft=nfft)
    d = _struct_dict(o)
    d.pop("reserved_d", None)
    return d


__all__ += ["BDW_TRUNC_START", "BDW_TRUNC_END", "BDW_NO_NOISE", "BDW_LOW_SNR",
            "BDW_RISE_UNRESOLVED", "BDW_FALL_UNRESOLVED", "BDW_BW26_FLOOR", "BDW_SHORT",
            "BDW_GAUSSIAN", "BDW_FLAG_NAMES", "burst_measure"]


# ---- §21 the ring ------------------------------------------------------------------
RING_MAX_CONSUMERS = 16
RING_RETUNE, RING_OVERFLOW, RING_CAL = 1, 2, 4
ring_bytes_twin = _ref.ring_bytes


class _RingMeta(C.Structure):
    _fields_ = [("seq", C.c_uint64), ("first_sample", C.c_uint64), ("t_utc_ns", C.c_int64),
                ("fc_hz", C.c_double), ("fs_hz", C.c_double), ("gain_db", C.c_float),
                ("format", C.c_int32), ("flags", C.c_uint32), ("nbytes", C.c_uint32)]


class _RingCounters(C.Structure):
    _fields_ = [("committed", C.c_uint64), ("oldest", C.c_uint64), ("block_bytes", C.c_uint64),
                ("nblocks", C.c_uint64), ("total_bytes", C.c_uint64), ("torn", C.c_uint64),
                ("max_consumers", C.c_uint32), ("active", C.c_uint32),
                ("active_mask", C.c_uint32), ("reserved", C.c_uint32),
                ("lost", C.c_uint64 * 16), ("read", C.c_uint64 * 16), ("lag", C.c_uint64 * 16)]


def _bind_ring(lib) -> None:
    P, vp, sz = C.POINTER, C.c_void_p, C.c_size_t
    lib.atkdsp_ring_bytes.argtypes = [sz, sz, C.c_int]
    lib.atkdsp_ring_bytes.restype = sz
    lib.atkdsp_ring_init.argtypes = [vp, sz, sz, sz, C.c_int]
    lib.atkdsp_ring_init.restype = vp
    lib.atkdsp_ring_attach.argtypes = [vp, sz]
    lib.atkdsp_ring_attach.restype = vp
    lib.atkdsp_ring_claim.argtypes = [vp]
    lib.atkdsp_ring_claim.restype = vp
    lib.atkdsp_ring_commit.argtypes = [vp, P(_RingMeta)]
    lib.atkdsp_ring_commit.restype = C.c_int
    lib.atkdsp_ring_open.argtypes = [vp, C.c_int]
    lib.atkdsp_ring_open.restype = C.c_int
    lib.atkdsp_ring_close.argtypes = [vp, C.c_int]
    lib.atkdsp_ring_close.restype = C.c_int
    lib.atkdsp_ring_read.argtypes = [vp, C.c_int, vp, sz, P(_RingMeta), P(C.c_uint64)]
    lib.atkdsp_ring_read.restype = C.c_ssize_t
    lib.atkdsp_ring_peek.argtypes = [vp, C.c_int, P(_RingMeta), P(C.c_uint64)]
    lib.atkdsp_ring_peek.restype = vp
    lib.atkdsp_ring_release.argtypes = [vp, C.c_int]
    lib.atkdsp_ring_release.restype = C.c_int
    lib.atkdsp_ring_copy_range.argtypes = [vp, C.c_uint64, C.c_uint64, vp, sz, P(_RingMeta), sz]
    lib.atkdsp_ring_copy_range.restype = C.c_ssize_t
    lib.atkdsp_ring_stats.argtypes = [vp, P(_RingCounters)]
    lib.atkdsp_ring_stats.restype = C.c_int
    lib.atkdsp_ring_describe.argtypes = [vp, C.c_char_p, sz]
    lib.atkdsp_ring_describe.restype = C.c_int


_BIND_EXTRA.append(_bind_ring)


def _meta_dict(m) -> dict:
    return {"seq": int(m.seq), "first_sample": int(m.first_sample), "t_utc_ns": int(m.t_utc_ns),
            "fc_hz": float(m.fc_hz), "fs_hz": float(m.fs_hz), "gain_db": float(m.gain_db),
            "format": int(m.format), "flags": int(m.flags), "nbytes": int(m.nbytes)}


def ring_bytes(block_bytes: int, nblocks: int, max_consumers: int = 4) -> int:
    """Bytes of memory a ring of this shape needs (atkdsp.h §21); 0 if impossible."""
    return int(load().atkdsp_ring_bytes(int(block_bytes), int(nblocks), int(max_consumers)))


class Ring:
    """The atkdsp ring (atkdsp.h §21) in memory this object owns: a private
    numpy buffer, or — with `name` — a named shared-memory mapping another
    process can attach to with Ring.attach(name). Windows backs a named
    mapping with the paging file and Linux with /dev/shm: RAM, never a file
    in a user folder. The twin is reference.Ring (single process)."""

    def __init__(self, block_bytes: int, nblocks: int, max_consumers: int = 4,
                 name: str | None = None, _attach: bool = False):
        self._lib = load()
        self._shm = None
        self._mbuf = None
        self.name = name
        if _attach:
            self._shm = _shm_open(name)
            size = self._shm.size
            self._addr = C.addressof(C.c_char.from_buffer(self._shm.buf))
            self._h = self._lib.atkdsp_ring_attach(self._addr, size)
            if not self._h:
                self._shm.close()
                raise AtkDspError(f"ring_attach({name!r}): not an atkdsp ring of this "
                                  "version (magic, version or sizes wrong)")
        else:
            size = self._lib.atkdsp_ring_bytes(int(block_bytes), int(nblocks), int(max_consumers))
            if not size:
                raise AtkDspError(f"ring_bytes({block_bytes}, {nblocks}, {max_consumers}) "
                                  "refused — block_bytes >= 1, nblocks >= 2, consumers 1..16")
            if name:
                self._shm = _quiet_shm_class()(name=name, create=True, size=size)
                self._addr = C.addressof(C.c_char.from_buffer(self._shm.buf))
            else:
                self._mbuf = np.zeros(size + 64, np.uint8)
                base = self._mbuf.ctypes.data
                self._addr = base + (-base) % 64
            self._h = self._lib.atkdsp_ring_init(self._addr, size, int(block_bytes),
                                                 int(nblocks), int(max_consumers))
            if not self._h:
                raise AtkDspError("ring_init refused the memory (alignment or size)")
        self.size = size
        st = self.stats()
        self.block_bytes, self.nblocks = st["block_bytes"], st["nblocks"]
        self.max_consumers = st["max_consumers"]
        self._rbuf = np.empty(self.block_bytes, np.uint8)
        self._claimed = False
        _det_log.info("ring %s: %s", "attached" if _attach else "made", self.describe())

    def _view(self, p: int, n: int) -> np.ndarray:
        """n bytes of the ring at address p, as a numpy view that HOLDS the
        memory it points into (the private buffer, or the shared mapping), so
        a view kept past close() or past this object keeps the bytes alive
        instead of pointing at freed memory."""
        off = int(p) - int(self._addr)
        if self._shm is not None:
            base = self._addr - C.addressof(C.c_char.from_buffer(self._shm.buf))
            return np.frombuffer(self._shm.buf, np.uint8, count=n, offset=base + off)
        start = (self._addr - self._mbuf.ctypes.data) + off
        return self._mbuf[start:start + n]

    @classmethod
    def attach(cls, name: str) -> "Ring":
        """Another process's named ring."""
        return cls(0, 0, 0, name=name, _attach=True)

    def close(self, unlink: bool = False) -> None:
        """Let go of the memory. The ring is unusable after this; a view handed
        out earlier keeps its bytes (the mapping is released with the last
        view)."""
        self._h = None
        if self._shm is not None:
            shm, self._shm = self._shm, None
            try:
                shm.close()
            except BufferError:      # a view still holds it; released with the view
                pass
            if unlink:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass

    def _need(self):
        if not self._h:
            raise AtkDspError("this ring has been closed")
        return self._h

    # -- producer -------------------------------------------------------------
    def claim(self) -> np.ndarray:
        """The next block to fill, in place (block_bytes, uint8)."""
        p = self._lib.atkdsp_ring_claim(self._need())
        self._claimed = True
        return self._view(p, self.block_bytes)

    def commit(self, nbytes: int, **meta) -> None:
        m = _RingMeta()
        for k, v in meta.items():
            if k in ("seq", "nbytes") or not hasattr(m, k):
                raise AtkDspError(f"unknown ring meta {k!r}")
            setattr(m, k, v)
        m.nbytes = int(nbytes)
        _dchk(self._lib.atkdsp_ring_commit(self._need(), C.byref(m)), "ring_commit",
              nbytes=nbytes, claimed=self._claimed)
        self._claimed = False

    def write(self, data, **meta) -> None:
        """claim + copy + commit."""
        b = np.frombuffer(memoryview(data).cast("B"), np.uint8)
        if b.size > self.block_bytes:
            raise AtkDspError(f"{b.size} bytes is more than one block ({self.block_bytes})")
        blk = self.claim()
        blk[:b.size] = b
        self.commit(b.size, **meta)

    # -- consumers ------------------------------------------------------------
    def open(self, from_oldest: bool = False) -> int:
        cid = self._lib.atkdsp_ring_open(self._need(), int(bool(from_oldest)))
        if cid < 0:
            _dchk(cid, "ring_open", from_oldest=from_oldest)
        return cid

    def close_consumer(self, cid: int) -> None:
        self._lib.atkdsp_ring_close(self._need(), int(cid))

    def read(self, cid: int):
        """(bytes, meta, lost) of the next block, or (None, None, lost)."""
        m, lost = _RingMeta(), C.c_uint64()
        n = self._lib.atkdsp_ring_read(self._need(), int(cid), self._rbuf.ctypes.data,
                                       self._rbuf.size, C.byref(m), C.byref(lost))
        if n < 0:
            _dchk(n, "ring_read", cid=cid)
        if n == 0:
            return None, None, int(lost.value)
        return self._rbuf[:n].copy(), _meta_dict(m), int(lost.value)

    def peek(self, cid: int):
        """Zero-copy: (view, meta, lost); call release(cid) when done."""
        m, lost = _RingMeta(), C.c_uint64()
        p = self._lib.atkdsp_ring_peek(self._need(), int(cid), C.byref(m), C.byref(lost))
        if not p:
            return None, None, int(lost.value)
        return self._view(p, int(m.nbytes)), _meta_dict(m), int(lost.value)

    def release(self, cid: int) -> bool:
        """True when the peeked block survived; False when it was overwritten
        while held (discard whatever was computed from it)."""
        rc = self._lib.atkdsp_ring_release(self._need(), int(cid))
        if rc == E_OVERRUN:
            _det_log.debug("ring_release(%d): the block was overwritten while held", cid)
            return False
        _dchk(rc, "ring_release", cid=cid)
        return True

    def copy_range(self, seq0: int, seq1: int):
        """The blocks seq0 <= seq < seq1 still held: (bytes, [meta])."""
        st = self.stats()
        lo, hi = max(int(seq0), st["oldest"]), min(int(seq1), st["committed"])
        nmax = max(hi - lo, 0)
        out = np.empty(max(nmax * self.block_bytes, 1), np.uint8)
        metas = (_RingMeta * max(nmax, 1))()
        n = self._lib.atkdsp_ring_copy_range(self._need(), int(seq0), int(seq1),
                                             out.ctypes.data, out.size, metas, nmax)
        if n < 0:
            _dchk(n, "ring_copy_range", seq0=seq0, seq1=seq1)
        ms = [_meta_dict(metas[i]) for i in range(n)]
        return out[:sum(m["nbytes"] for m in ms)].copy(), ms

    def stats(self) -> dict:
        s = _RingCounters()
        self._lib.atkdsp_ring_stats(self._need(), C.byref(s))
        k = int(s.max_consumers)
        return {"committed": int(s.committed), "oldest": int(s.oldest),
                "block_bytes": int(s.block_bytes), "nblocks": int(s.nblocks),
                "total_bytes": int(s.total_bytes), "max_consumers": k,
                "active": int(s.active), "torn": int(s.torn),
                "lost": [int(s.lost[i]) for i in range(k)],
                "read": [int(s.read[i]) for i in range(k)],
                "lag": [int(s.lag[i]) for i in range(k)]}

    def describe(self) -> str:
        buf = C.create_string_buffer(512)
        self._lib.atkdsp_ring_describe(self._need(), buf, 512)
        return buf.value.decode()


_QUIET_SHM = None


def _quiet_shm_class():
    """SharedMemory whose finaliser does not complain when a view handed out
    by Ring still holds the mapping (the view keeps it alive on purpose; the
    mapping goes with the last view)."""
    global _QUIET_SHM
    if _QUIET_SHM is None:
        from multiprocessing import shared_memory

        class _QuietSharedMemory(shared_memory.SharedMemory):
            def __del__(self):
                try:
                    self.close()
                except BufferError:
                    pass
        _QUIET_SHM = _QuietSharedMemory
    return _QUIET_SHM


def _shm_open(name):
    """Attach to an existing named mapping without this process's resource
    tracker unlinking it at exit (it belongs to its creator)."""
    cls = _quiet_shm_class()
    try:
        return cls(name=name, create=False, track=False)
    except TypeError:                      # Python < 3.13 has no track=
        shm = cls(name=name, create=False)
        if sys.platform != "win32":
            try:
                from multiprocessing import resource_tracker
                resource_tracker.unregister(shm._name, "shared_memory")
            except Exception:              # noqa: BLE001 — best effort
                pass
        return shm


__all__ += ["RING_MAX_CONSUMERS", "RING_RETUNE", "RING_OVERFLOW", "RING_CAL", "ring_bytes",
            "Ring"]


# ============================================================================
# PHASE 3 (v0.17.0, ABI 18) — §22 pulse runs, §23 cyclic probes, §24 tone IDs
# ============================================================================
_p3_log = _logging.getLogger("atkdsp.phase3")
PDW_TRUNCATED, PDW_NO_FREQ, PDW_EDGE_START, PDW_FLUSHED = 1, 2, 4, 8


class _PulsesConfig(C.Structure):
    _fields_ = [("fs", C.c_double), ("floor", C.c_double), ("thresh_db", C.c_double),
                ("hyst_db", C.c_double), ("min_samples", C.c_int32), ("max_samples", C.c_int32),
                ("reserved", C.c_int32 * 2)]


class _Pdw(C.Structure):
    _fields_ = [("toa", C.c_double), ("pw", C.c_double), ("freq_hz", C.c_double),
                ("chirp_hz", C.c_double), ("slope_hz_s", C.c_double), ("pa_db", C.c_float),
                ("peak_db", C.c_float), ("energy", C.c_float), ("flags", C.c_uint32),
                ("start", C.c_uint64), ("end", C.c_uint64), ("n", C.c_int32),
                ("reserved", C.c_int32)]


class _PulsesCounters(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in
                ("samples", "pulses", "dropped", "truncated", "short_runs", "open", "resets")]


def _bind_pulses(lib) -> None:
    P = C.POINTER
    vp, sz, i, d = C.c_void_p, C.c_size_t, C.c_int, C.c_double
    lib.atkdsp_pulses_defaults.argtypes = [P(_PulsesConfig)]
    lib.atkdsp_pulses_defaults.restype = None
    lib.atkdsp_pulses_create.argtypes = [P(_PulsesConfig), d]
    lib.atkdsp_pulses_create.restype = vp
    lib.atkdsp_pulses_destroy.argtypes = [vp]
    lib.atkdsp_pulses_destroy.restype = None
    lib.atkdsp_pulses_reset.argtypes = [vp, d]
    lib.atkdsp_pulses_reset.restype = None
    lib.atkdsp_pulses_set_floor.argtypes = [vp, d]
    lib.atkdsp_pulses_set_floor.restype = i
    lib.atkdsp_pulses_max.argtypes = [vp, sz]
    lib.atkdsp_pulses_max.restype = sz
    lib.atkdsp_pulses_process.argtypes = [vp, P(_Cf32), sz, P(_Pdw), sz]
    lib.atkdsp_pulses_process.restype = C.c_ssize_t
    lib.atkdsp_pulses_flush.argtypes = [vp, P(_Pdw), sz]
    lib.atkdsp_pulses_flush.restype = C.c_ssize_t
    lib.atkdsp_pulses_describe.argtypes = [vp, C.c_char_p, sz]
    lib.atkdsp_pulses_describe.restype = i
    lib.atkdsp_pulses_stats.argtypes = [vp, P(_PulsesCounters)]
    lib.atkdsp_pulses_stats.restype = i
    lib.atkdsp_mag_median.argtypes = [P(_Cf32), sz, P(d)]
    lib.atkdsp_mag_median.restype = i


_BIND_EXTRA.append(_bind_pulses)


def _pdw_dict(w) -> dict:
    return {"toa": float(w.toa), "pw": float(w.pw), "freq_hz": float(w.freq_hz),
            "chirp_hz": float(w.chirp_hz), "slope_hz_s": float(w.slope_hz_s),
            "pa_db": float(w.pa_db), "peak_db": float(w.peak_db), "energy": float(w.energy),
            "flags": int(w.flags), "start": int(w.start), "end": int(w.end), "n": int(w.n)}


def mag_median(x) -> float:
    """The median of |x| — pulse.py's noise floor (atkdsp.h §22)."""
    a = _cf32(x)
    out = C.c_double()
    _check(load().atkdsp_mag_median(_cfp(a), a.size, C.byref(out)), "mag_median")
    return out.value


class Pulses:
    """Live pulse runs -> PDWs (atkdsp.h §22). Options are the
    atkdsp_pulses_config fields as keywords; process() returns one dict per
    closed pulse with the twin's keys (reference.Pulses); flush() closes an
    open one at the end of a stream. The twin is ATK's fallback."""

    _kind = "pulses"

    def __init__(self, t0: float = 0.0, **cfg):
        self._lib = load()
        c = _ref.pulses_defaults()
        for key, v in cfg.items():
            if key not in c:
                raise AtkDspError(f"unknown pulses option {key!r}; known: {sorted(c)}")
            c[key] = v
        for key in c:
            c[key] = int(c[key]) if key in ("min_samples", "max_samples") else float(c[key])
        why = _ref._pulses_check(c)
        if why:
            raise AtkDspError(f"pulses_create({c}) refused — {why}")
        k = _PulsesConfig()
        for key, v in c.items():
            setattr(k, key, v)
        self._h = self._lib.atkdsp_pulses_create(C.byref(k), float(t0))
        if not self._h:
            raise AtkDspError(f"pulses_create({c}) refused")
        self.cfg = dict(c)
        self._buf = None
        self._ncalls = 0
        _p3_log.info("pulses created: %s", self.describe())

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.atkdsp_pulses_destroy(h)

    def reset(self, t0: float = 0.0) -> None:
        self._lib.atkdsp_pulses_reset(self._h, float(t0))

    def set_floor(self, floor: float) -> None:
        _check(self._lib.atkdsp_pulses_set_floor(self._h, float(floor)), "pulses_set_floor")
        self.cfg["floor"] = float(floor)

    def max(self, n: int) -> int:
        return int(self._lib.atkdsp_pulses_max(self._h, int(n)))

    def process(self, x, cap: int | None = None) -> list:
        a = _cf32(x)
        if cap is None:
            cap = self.max(a.size)
        if self._buf is None or len(self._buf) < cap:
            self._buf = (_Pdw * int(cap))()
        got = _check(self._lib.atkdsp_pulses_process(self._h, _cfp(a), a.size, self._buf, int(cap)),
                     "pulses_process")
        self._ncalls += 1
        if self._ncalls % 1000 == 0:
            _p3_log.debug("pulses: %s", self.stats())
        return [_pdw_dict(self._buf[i]) for i in range(got)]

    def flush(self, cap: int = 1) -> list:
        buf = (_Pdw * max(1, int(cap)))()
        got = _check(self._lib.atkdsp_pulses_flush(self._h, buf, int(cap)), "pulses_flush")
        return [_pdw_dict(buf[i]) for i in range(got)]

    def describe(self) -> str:
        buf = C.create_string_buffer(512)
        self._lib.atkdsp_pulses_describe(self._h, buf, 512)
        return buf.value.decode()

    def stats(self) -> dict:
        s = _PulsesCounters()
        self._lib.atkdsp_pulses_stats(self._h, C.byref(s))
        return {n: int(getattr(s, n)) for n, _t in s._fields_}


def detect_pdws(x, fs: float, floor: float | None = None, **cfg) -> list:
    """One call over a whole record with its own median floor unless one is
    given: pulse.detect_pulses' words, natively (the twin: reference.detect_pdws)."""
    a = _cf32(x)
    if floor is None:
        floor = mag_median(a)
    p = Pulses(0.0, fs=fs, floor=floor, **cfg)
    out = p.process(a, cap=p.max(a.size))
    out += p.flush()
    return out


__all__ += ["PDW_TRUNCATED", "PDW_NO_FREQ", "PDW_EDGE_START", "PDW_FLUSHED", "mag_median",
            "Pulses", "detect_pdws"]


# -- §23 cyclic probes ---------------------------------------------------------
NL_ENV2, NL_POW2, NL_POW4, NL_POW8, NL_DM, NL_FDIFF = 0, 1, 2, 3, 4, 5
NL_NAMES = {"env2": 0, "pow2": 1, "pow4": 2, "pow8": 3, "dm": 4, "fdiff": 5}
CYC_LINE, CYC_WARMUP = 1, 2
CP_DETECT, CP_LINE, CP_WARMUP = 1, 2, 4


class _CycCand(C.Structure):
    _fields_ = [("rate_hz", C.c_double), ("nonlin", C.c_int32), ("hard_limit", C.c_int32)]


class _CycResult(C.Structure):
    _fields_ = [("level", C.c_double), ("coherence", C.c_double), ("line_db", C.c_double),
                ("delta_hz", C.c_double), ("level_aligned", C.c_double),
                ("coherence_aligned", C.c_double), ("level_thr", C.c_double),
                ("coh_thr", C.c_double), ("nseg", C.c_int32), ("flags", C.c_uint32)]


class _CpCand(C.Structure):
    _fields_ = [("lag", C.c_int32), ("cp", C.c_int32), ("period", C.c_int32), ("reserved", C.c_int32)]


class _CpResult(C.Structure):
    _fields_ = [("score", C.c_double), ("score_thr", C.c_double), ("npairs", C.c_double),
                ("line_level", C.c_double), ("line_coherence", C.c_double),
                ("line_coh_thr", C.c_double), ("line_delta_hz", C.c_double),
                ("nseg", C.c_int32), ("flags", C.c_uint32)]


class _CyclicCounters(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in ("samples", "segments", "resets", "segment_fill")]


def _bind_cyclic(lib) -> None:
    P = C.POINTER
    vp, sz, i, d = C.c_void_p, C.c_size_t, C.c_int, C.c_double
    lib.atkdsp_cyclic_thresholds.argtypes = [i, d, P(d), P(d)]
    lib.atkdsp_cyclic_thresholds.restype = None
    lib.atkdsp_cyclic_create.argtypes = [d, P(_CycCand), i, i, i, d]
    lib.atkdsp_cyclic_create.restype = vp
    lib.atkdsp_cyclic_destroy.argtypes = [vp]
    lib.atkdsp_cyclic_destroy.restype = None
    lib.atkdsp_cyclic_reset.argtypes = [vp]
    lib.atkdsp_cyclic_reset.restype = None
    lib.atkdsp_cyclic_process.argtypes = [vp, P(_Cf32), sz]
    lib.atkdsp_cyclic_process.restype = i
    lib.atkdsp_cyclic_read.argtypes = [vp, P(_CycResult)]
    lib.atkdsp_cyclic_read.restype = i
    lib.atkdsp_cyclic_describe.argtypes = [vp, C.c_char_p, sz]
    lib.atkdsp_cyclic_describe.restype = i
    lib.atkdsp_cyclic_stats.argtypes = [vp, P(_CyclicCounters)]
    lib.atkdsp_cyclic_stats.restype = i
    lib.atkdsp_cp_probe_create.argtypes = [d, P(_CpCand), i, i, i, d]
    lib.atkdsp_cp_probe_create.restype = vp
    lib.atkdsp_cp_probe_destroy.argtypes = [vp]
    lib.atkdsp_cp_probe_destroy.restype = None
    lib.atkdsp_cp_probe_reset.argtypes = [vp]
    lib.atkdsp_cp_probe_reset.restype = None
    lib.atkdsp_cp_probe_process.argtypes = [vp, P(_Cf32), sz]
    lib.atkdsp_cp_probe_process.restype = i
    lib.atkdsp_cp_probe_read.argtypes = [vp, P(_CpResult)]
    lib.atkdsp_cp_probe_read.restype = i
    lib.atkdsp_cp_probe_describe.argtypes = [vp, C.c_char_p, sz]
    lib.atkdsp_cp_probe_describe.restype = i
    lib.atkdsp_cp_probe_stats.argtypes = [vp, P(_CyclicCounters)]
    lib.atkdsp_cp_probe_stats.restype = i


_BIND_EXTRA.append(_bind_cyclic)


def cyclic_thresholds(nseg: int, pfa_each: float) -> tuple:
    """(level_thr, coh_thr) for nseg segments at pfa_each (atkdsp.h §23)."""
    a, b = C.c_double(), C.c_double()
    load().atkdsp_cyclic_thresholds(int(nseg), float(pfa_each), C.byref(a), C.byref(b))
    return a.value, b.value


def _nl_code(v):
    return NL_NAMES[v.lower()] if isinstance(v, str) else int(v)


class Cyclic:
    """A line at each candidate symbol rate after its nonlinearity (atkdsp.h
    §23). cands: (rate_hz, nonlin[, hard_limit]) tuples, nonlin a code or
    'env2' / 'pow2' / 'pow4' / 'pow8' / 'dm'. read() returns one dict per
    candidate with the twin's keys (reference.Cyclic)."""

    def __init__(self, fs: float, cands, seg_len: int, nseg: int, pfa: float = 1e-3):
        self._lib = load()
        cl = [(float(c[0]), _nl_code(c[1]), int(c[2]) if len(c) > 2 else 0) for c in cands]
        # the twin's checks give the reason the C would refuse with NULL
        try:
            _ref.Cyclic(fs, cl, seg_len, nseg, pfa)
        except ValueError as e:
            raise AtkDspError(f"cyclic_create refused — {e}") from None
        arr = (_CycCand * len(cl))(*[_CycCand(r, nl, hl) for r, nl, hl in cl])
        self._h = self._lib.atkdsp_cyclic_create(float(fs), arr, len(cl), int(seg_len), int(nseg),
                                                 float(pfa))
        if not self._h:
            raise AtkDspError("cyclic_create refused")
        self.n = len(cl)
        self.cands = cl
        self._res = (_CycResult * self.n)()
        _p3_log.info("cyclic created: %s", self.describe())

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.atkdsp_cyclic_destroy(h)

    def reset(self) -> None:
        self._lib.atkdsp_cyclic_reset(self._h)

    def process(self, x) -> None:
        a = _cf32(x)
        _check(self._lib.atkdsp_cyclic_process(self._h, _cfp(a), a.size), "cyclic_process")

    def read(self) -> list:
        _check(self._lib.atkdsp_cyclic_read(self._h, self._res), "cyclic_read")
        return [{n: (int(getattr(r, n)) if n in ("nseg", "flags") else float(getattr(r, n)))
                 for n, _t in _CycResult._fields_} for r in self._res]

    def describe(self) -> str:
        buf = C.create_string_buffer(2048)
        self._lib.atkdsp_cyclic_describe(self._h, buf, 2048)
        return buf.value.decode()

    def stats(self) -> dict:
        s = _CyclicCounters()
        self._lib.atkdsp_cyclic_stats(self._h, C.byref(s))
        return {n: int(getattr(s, n)) for n, _t in s._fields_}


class CpProbe:
    """OFDM cyclic-prefix probe (atkdsp.h §23). cands: (lag, cp[, period])."""

    def __init__(self, fs: float, cands, seg_len: int, nseg: int, pfa: float = 1e-3):
        self._lib = load()
        cl = [(int(c[0]), int(c[1]), int(c[2]) if len(c) > 2 else 0) for c in cands]
        try:
            _ref.CpProbe(fs, cl, seg_len, nseg, pfa)
        except ValueError as e:
            raise AtkDspError(f"cp_probe_create refused — {e}") from None
        arr = (_CpCand * len(cl))(*[_CpCand(a, b, p, 0) for a, b, p in cl])
        self._h = self._lib.atkdsp_cp_probe_create(float(fs), arr, len(cl), int(seg_len), int(nseg),
                                                   float(pfa))
        if not self._h:
            raise AtkDspError("cp_probe_create refused")
        self.n = len(cl)
        self.cands = cl
        self._res = (_CpResult * self.n)()
        _p3_log.info("cp_probe created: %s", self.describe())

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.atkdsp_cp_probe_destroy(h)

    def reset(self) -> None:
        self._lib.atkdsp_cp_probe_reset(self._h)

    def process(self, x) -> None:
        a = _cf32(x)
        _check(self._lib.atkdsp_cp_probe_process(self._h, _cfp(a), a.size), "cp_probe_process")

    def read(self) -> list:
        _check(self._lib.atkdsp_cp_probe_read(self._h, self._res), "cp_probe_read")
        return [{n: (int(getattr(r, n)) if n in ("nseg", "flags") else float(getattr(r, n)))
                 for n, _t in _CpResult._fields_} for r in self._res]

    def describe(self) -> str:
        buf = C.create_string_buffer(2048)
        self._lib.atkdsp_cp_probe_describe(self._h, buf, 2048)
        return buf.value.decode()

    def stats(self) -> dict:
        s = _CyclicCounters()
        self._lib.atkdsp_cp_probe_stats(self._h, C.byref(s))
        return {n: int(getattr(s, n)) for n, _t in s._fields_}


__all__ += ["NL_ENV2", "NL_POW2", "NL_POW4", "NL_POW8", "NL_DM", "NL_FDIFF", "CYC_LINE", "CYC_WARMUP",
            "CP_DETECT", "CP_LINE", "CP_WARMUP", "cyclic_thresholds", "Cyclic", "CpProbe"]


# -- §24 analog radio IDs ------------------------------------------------------
TONE_CTCSS, TONE_DCS, TONE_DTMF, TONE_TONE = 1, 2, 3, 4
TONE_START, TONE_END = 1, 2
TONE_F_REVERSE_BURST, TONE_F_TURNOFF, TONE_F_NONSTANDARD, TONE_F_CORRECTED, TONE_F_FLUSHED = 1, 2, 4, 8, 16
TONE_EN_CTCSS, TONE_EN_DCS, TONE_EN_DTMF, TONE_EN_TONE, TONE_EN_ALL = 1, 2, 4, 8, 15
TONE_KINDS = {1: "CTCSS", 2: "DCS", 3: "DTMF", 4: "TONE"}


class _ToneConfig(C.Structure):
    _fields_ = [("fs", C.c_double), ("enable", C.c_int32), ("dcs_max_errors", C.c_int32),
                ("ctcss_min_snr_db", C.c_double), ("dtmf_min_dbfs", C.c_double),
                ("tone_min_dbfs", C.c_double), ("tone_min_snr_db", C.c_double),
                ("reserved", C.c_int32 * 2)]


class _ToneEvent(C.Structure):
    _fields_ = [("t0", C.c_double), ("t1", C.c_double), ("freq_hz", C.c_double),
                ("freq2_hz", C.c_double), ("level_db", C.c_float), ("snr_db", C.c_float),
                ("kind", C.c_int32), ("type", C.c_int32), ("code", C.c_int32), ("alias", C.c_int32),
                ("flags", C.c_uint32), ("reserved", C.c_uint32), ("text", C.c_char * 16)]


class _ToneCounters(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in
                ("samples", "events", "dropped", "ctcss", "dcs", "dtmf", "tones", "resets")]


def _bind_tone_id(lib) -> None:
    P = C.POINTER
    vp, sz, i, d = C.c_void_p, C.c_size_t, C.c_int, C.c_double
    lib.atkdsp_tone_id_defaults.argtypes = [P(_ToneConfig)]
    lib.atkdsp_tone_id_defaults.restype = None
    lib.atkdsp_tone_id_create.argtypes = [P(_ToneConfig), d]
    lib.atkdsp_tone_id_create.restype = vp
    lib.atkdsp_tone_id_destroy.argtypes = [vp]
    lib.atkdsp_tone_id_destroy.restype = None
    lib.atkdsp_tone_id_reset.argtypes = [vp, d]
    lib.atkdsp_tone_id_reset.restype = None
    lib.atkdsp_tone_id_events_max.argtypes = [vp, sz]
    lib.atkdsp_tone_id_events_max.restype = sz
    lib.atkdsp_tone_id_process.argtypes = [vp, P(C.c_float), sz, P(_ToneEvent), sz]
    lib.atkdsp_tone_id_process.restype = C.c_ssize_t
    lib.atkdsp_tone_id_flush.argtypes = [vp, P(_ToneEvent), sz]
    lib.atkdsp_tone_id_flush.restype = C.c_ssize_t
    lib.atkdsp_tone_id_describe.argtypes = [vp, C.c_char_p, sz]
    lib.atkdsp_tone_id_describe.restype = i
    lib.atkdsp_tone_id_stats.argtypes = [vp, P(_ToneCounters)]
    lib.atkdsp_tone_id_stats.restype = i
    lib.atkdsp_ctcss_tone.argtypes = [i, P(d)]
    lib.atkdsp_ctcss_tone.restype = i
    lib.atkdsp_dcs_code.argtypes = [i, P(i)]
    lib.atkdsp_dcs_code.restype = i
    lib.atkdsp_dcs_encode.argtypes = [i]
    lib.atkdsp_dcs_encode.restype = C.c_uint32
    lib.atkdsp_dcs_decode.argtypes = [C.c_uint32, i, P(i), P(i)]
    lib.atkdsp_dcs_decode.restype = i


_BIND_EXTRA.append(_bind_tone_id)


def dcs_encode(octal: int) -> int:
    """The 23-bit Golay word a DCS code is sent as (bit 0 first on the air)."""
    return int(load().atkdsp_dcs_encode(int(octal)))


def dcs_decode(word: int, max_errors: int = 1) -> tuple:
    """(corrections, code, alias): the normal and inverted readings of a 23-bit
    word, or (-1, -1, -1) when it is not a standard code within max_errors."""
    c, a = C.c_int(), C.c_int()
    rc = load().atkdsp_dcs_decode(int(word) & 0x7FFFFF, int(max_errors), C.byref(c), C.byref(a))
    return (int(rc), int(c.value), int(a.value)) if rc >= 0 else (-1, -1, -1)


def _tone_event_dict(e) -> dict:
    return {"kind": int(e.kind), "type": int(e.type), "t0": float(e.t0), "t1": float(e.t1),
            "freq_hz": float(e.freq_hz), "freq2_hz": float(e.freq2_hz), "level_db": float(e.level_db),
            "snr_db": float(e.snr_db), "code": int(e.code), "alias": int(e.alias),
            "flags": int(e.flags), "text": e.text.decode(errors="replace")}


class ToneId:
    """Analog radio IDs on demodulated audio (atkdsp.h §24): CTCSS, DCS, DTMF
    and tracked tones, as START / END event dicts (reference.ToneId's keys)."""

    _kind = "tone_id"

    def __init__(self, t0: float = 0.0, **cfg):
        self._lib = load()
        c = _ref.tone_id_defaults()
        for key, v in cfg.items():
            if key not in c:
                raise AtkDspError(f"unknown tone_id option {key!r}; known: {sorted(c)}")
            c[key] = v
        for key in c:
            c[key] = int(c[key]) if key in ("enable", "dcs_max_errors") else float(c[key])
        why = _ref._tone_id_check(c)
        if why:
            raise AtkDspError(f"tone_id_create({c}) refused — {why}")
        k = _ToneConfig()
        for key, v in c.items():
            setattr(k, key, v)
        self._h = self._lib.atkdsp_tone_id_create(C.byref(k), float(t0))
        if not self._h:
            raise AtkDspError(f"tone_id_create({c}) refused")
        self.cfg = dict(c)
        self._buf = None
        _p3_log.info("tone_id created: %s", self.describe())

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.atkdsp_tone_id_destroy(h)

    def reset(self, t0: float = 0.0) -> None:
        self._lib.atkdsp_tone_id_reset(self._h, float(t0))

    def events_max(self, n: int) -> int:
        return int(self._lib.atkdsp_tone_id_events_max(self._h, int(n)))

    def process(self, x, cap: int | None = None) -> list:
        a = _f32(x, "audio")
        if cap is None:
            cap = self.events_max(a.size)
        if self._buf is None or len(self._buf) < cap:
            self._buf = (_ToneEvent * int(cap))()
        got = _check(self._lib.atkdsp_tone_id_process(self._h, _fp(a), a.size, self._buf, int(cap)),
                     "tone_id_process")
        return [_tone_event_dict(self._buf[i]) for i in range(got)]

    def flush(self) -> list:
        buf = (_ToneEvent * 8)()
        got = _check(self._lib.atkdsp_tone_id_flush(self._h, buf, 8), "tone_id_flush")
        return [_tone_event_dict(buf[i]) for i in range(got)]

    def describe(self) -> str:
        buf = C.create_string_buffer(1024)
        self._lib.atkdsp_tone_id_describe(self._h, buf, 1024)
        return buf.value.decode()

    def stats(self) -> dict:
        s = _ToneCounters()
        self._lib.atkdsp_tone_id_stats(self._h, C.byref(s))
        return {n: int(getattr(s, n)) for n, _t in s._fields_}


__all__ += ["TONE_CTCSS", "TONE_DCS", "TONE_DTMF", "TONE_TONE", "TONE_START", "TONE_END",
            "TONE_F_REVERSE_BURST", "TONE_F_TURNOFF", "TONE_F_NONSTANDARD", "TONE_F_CORRECTED",
            "TONE_F_FLUSHED", "TONE_EN_CTCSS", "TONE_EN_DCS", "TONE_EN_DTMF", "TONE_EN_TONE",
            "TONE_EN_ALL", "TONE_KINDS", "dcs_encode", "dcs_decode", "ToneId"]


# ============================================================================
# §25-27 spread spectrum bindings (src/dsss_det.c, dsss_code.c, dsss_acq.c,
# css.c, xspec.c, dsss_codes.c). Twins: reference.py §25-27.
# ============================================================================
def _struct_dict(s):
    """A ctypes result struct -> dict with the twin's keys; int arrays -> lists."""
    out = {}
    for n, t in s._fields_:
        v = getattr(s, n)
        if isinstance(v, C.Array):
            out[n] = [int(x) for x in v]
        elif t in (C.c_int32, C.c_uint32, C.c_int, C.c_uint64, C.c_int64):
            out[n] = int(v)
        elif isinstance(v, bytes):
            out[n] = v.decode(errors="replace")
        else:
            out[n] = float(v)
    return out


class _MethodInfo(C.Structure):
    _fields_ = [("name", C.c_char_p), ("authors", C.c_char_p), ("year", C.c_char_p),
                ("venue", C.c_char_p), ("country", C.c_char_p)]


class _DsssDetCfg(C.Structure):
    _fields_ = [("fs", C.c_double), ("win", C.c_int32), ("max_lag", C.c_int32),
                ("nwin_min", C.c_int32), ("handles", C.c_int32), ("pfa", C.c_double),
                ("cav_lags", C.c_int32), ("min_peaks", C.c_int32), ("period_min", C.c_int32),
                ("reserved", C.c_int32 * 4)]


class _DsssDetResult(C.Structure):
    _fields_ = [("period", C.c_double), ("period_s", C.c_double), ("peak_z", C.c_double),
                ("peak_ratio", C.c_double), ("floor_mean", C.c_double), ("floor_std", C.c_double),
                ("fluct_log10_pfa", C.c_double), ("cav_t", C.c_double), ("cav_thr", C.c_double),
                ("cav_log10_pfa", C.c_double), ("tc_acf", C.c_double), ("ps_acf", C.c_double),
                ("llr", C.c_double), ("npeaks", C.c_int32), ("nwin", C.c_int32),
                ("best_lag", C.c_int32), ("flags", C.c_uint32), ("method", C.c_int32 * 4)]


class _DsssCounters(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in ("samples", "windows", "resets", "window_fill")]


class _ScanLine(C.Structure):
    _fields_ = [("freq_hz", C.c_double), ("level", C.c_double), ("coherence", C.c_double),
                ("level_thr", C.c_double), ("coh_thr", C.c_double), ("log10_pfa", C.c_double),
                ("bin", C.c_int32), ("flags", C.c_uint32)]


class _DsssCodeCfg(C.Structure):
    _fields_ = [("fs", C.c_double), ("rc", C.c_double), ("period_s", C.c_double),
                ("chip_mod", C.c_int32), ("vote_passes", C.c_int32), ("poly_max_degree", C.c_int32),
                ("no_cfo", C.c_int32), ("pfa", C.c_double), ("no_decim", C.c_int32), ("reserved", C.c_int32)]


class _DsssCodeResult(C.Structure):
    _fields_ = [("L", C.c_int32), ("nperiods", C.c_int32), ("periods_needed", C.c_int32),
                ("start", C.c_int32), ("cfo_hz", C.c_double), ("lambda1", C.c_double),
                ("lambda2", C.c_double), ("noise_var", C.c_double), ("ident_margin", C.c_double),
                ("snr_db", C.c_double), ("vote_min", C.c_double), ("vote_mean", C.c_double),
                ("poly_method", C.c_int32), ("poly_degree", C.c_int32), ("poly", C.c_uint64),
                ("poly_z", C.c_double), ("lc", C.c_int32), ("lc_plateau", C.c_int32),
                ("flags", C.c_uint32), ("method", C.c_int32 * 8), ("name", C.c_char * 48)]


class _DsssAcqCfg(C.Structure):
    _fields_ = [("fs", C.c_double), ("rc", C.c_double), ("L", C.c_int32), ("ncoh", C.c_int32),
                ("nnoncoh", C.c_int32), ("combiner", C.c_int32), ("dop_lo", C.c_double),
                ("dop_hi", C.c_double), ("dop_step", C.c_double), ("pfa", C.c_double),
                ("reserved", C.c_int32 * 2)]


class _DsssAcq(C.Structure):
    _fields_ = [("code_phase", C.c_double), ("doppler_hz", C.c_double), ("gamma", C.c_double),
                ("gamma_thr", C.c_double), ("log10_pfa", C.c_double), ("peak_ratio", C.c_double),
                ("cn0_dbhz", C.c_double), ("ncells", C.c_int32), ("ndop", C.c_int32),
                ("flags", C.c_uint32), ("method", C.c_int32 * 4)]


class _DsssTrkCfg(C.Structure):
    _fields_ = [("fs", C.c_double), ("rc", C.c_double), ("L", C.c_int32), ("reserved0", C.c_int32),
                ("code_phase", C.c_double), ("doppler_hz", C.c_double), ("dll_bw_hz", C.c_double),
                ("pll_bw_hz", C.c_double), ("spacing", C.c_double), ("quality_spacing", C.c_double),
                ("cn0_periods", C.c_int32), ("reserved1", C.c_int32), ("power_window_s", C.c_double),
                ("distortion_pfa", C.c_double), ("power_high_db", C.c_double),
                ("reserved", C.c_int32 * 2)]


class _DsssTrkState(C.Structure):
    _fields_ = [("code_phase", C.c_double), ("doppler_hz", C.c_double), ("cn0_m2m4", C.c_double),
                ("cn0_nwpr", C.c_double), ("distortion", C.c_double), ("distortion_log10_pfa", C.c_double),
                ("power_db", C.c_double), ("power_start_db", C.c_double), ("lock", C.c_double),
                ("periods", C.c_uint64), ("flags", C.c_uint32), ("method", C.c_int32 * 4)]


class _CssCand(C.Structure):
    _fields_ = [("sf", C.c_int32), ("reserved", C.c_int32), ("bw_hz", C.c_double)]


class _CssResult(C.Structure):
    _fields_ = [("level", C.c_double), ("level_thr", C.c_double), ("offset_hz", C.c_double),
                ("bin", C.c_int32), ("run", C.c_int32), ("nwin", C.c_int32), ("flags", C.c_uint32)]


class _XspecResult(C.Structure):
    _fields_ = [("max_coh", C.c_double), ("coh_thr", C.c_double), ("log10_pfa", C.c_double),
                ("max_bin", C.c_int32), ("band_lo", C.c_int32), ("band_hi", C.c_int32),
                ("band_bins", C.c_int32), ("band_coh_mean", C.c_double), ("tdoa_s", C.c_double),
                ("tdoa_err_s", C.c_double), ("phase0", C.c_double), ("slope_resid", C.c_double),
                ("navg", C.c_int32), ("flags", C.c_uint32), ("method", C.c_int32 * 2)]


class _Align(C.Structure):
    _fields_ = [("lag", C.c_double), ("lag_err", C.c_double), ("rate_ppm", C.c_double),
                ("lo_offset_hz", C.c_double), ("peak_ratio", C.c_double), ("snr_db", C.c_double),
                ("nblocks", C.c_int32), ("flags", C.c_uint32), ("method", C.c_int32 * 2)]


class _LatticeIn(C.Structure):
    _fields_ = [("rc_acf_hz", C.c_double), ("rc_line_hz", C.c_double), ("rc_nominal_hz", C.c_double),
                ("period_s", C.c_double), ("fc_hz", C.c_double), ("cfo_hz", C.c_double),
                ("rc_tol", C.c_double), ("ppm_tol", C.c_double), ("rx_ppm", C.c_double)]


class _LatticeOut(C.Structure):
    _fields_ = [("rc_hz", C.c_double), ("rc_spread", C.c_double), ("L", C.c_double),
                ("L_frac", C.c_double), ("ppm_carrier", C.c_double), ("ppm_chip", C.c_double),
                ("ppm_diff", C.c_double), ("L_int", C.c_int32), ("family", C.c_int32),
                ("flags", C.c_uint32), ("method", C.c_int32)]


def _bind_dsss(lib) -> None:
    P = C.POINTER
    vp, sz, i, d, pd = C.c_void_p, C.c_size_t, C.c_int, C.c_double, C.POINTER(C.c_double)
    i8, u16, pt = C.POINTER(C.c_int8), C.POINTER(C.c_uint16), C.c_long
    lib.atkdsp_method.argtypes = [i]; lib.atkdsp_method.restype = P(_MethodInfo)
    # detector
    lib.atkdsp_dsss_det_defaults.argtypes = [P(_DsssDetCfg), d, i, i]; lib.atkdsp_dsss_det_defaults.restype = None
    lib.atkdsp_dsss_det_create.argtypes = [P(_DsssDetCfg)]; lib.atkdsp_dsss_det_create.restype = vp
    lib.atkdsp_dsss_det_destroy.argtypes = [vp]; lib.atkdsp_dsss_det_destroy.restype = None
    lib.atkdsp_dsss_det_reset.argtypes = [vp]; lib.atkdsp_dsss_det_reset.restype = None
    lib.atkdsp_dsss_det_process.argtypes = [vp, P(_Cf32), sz]; lib.atkdsp_dsss_det_process.restype = i
    lib.atkdsp_dsss_det_read.argtypes = [vp, P(_DsssDetResult)]; lib.atkdsp_dsss_det_read.restype = i
    lib.atkdsp_dsss_det_rho.argtypes = [vp, pd, sz]; lib.atkdsp_dsss_det_rho.restype = pt
    lib.atkdsp_dsss_det_describe.argtypes = [vp, C.c_char_p, sz]; lib.atkdsp_dsss_det_describe.restype = i
    lib.atkdsp_dsss_det_stats.argtypes = [vp, P(_DsssCounters)]; lib.atkdsp_dsss_det_stats.restype = i
    lib.atkdsp_cav_threshold.argtypes = [i, d, d]; lib.atkdsp_cav_threshold.restype = d
    # scan
    lib.atkdsp_cyclic_scan_create.argtypes = [d, i, i, i, i, d, d, d]; lib.atkdsp_cyclic_scan_create.restype = vp
    lib.atkdsp_cyclic_scan_destroy.argtypes = [vp]; lib.atkdsp_cyclic_scan_destroy.restype = None
    lib.atkdsp_cyclic_scan_reset.argtypes = [vp]; lib.atkdsp_cyclic_scan_reset.restype = None
    lib.atkdsp_cyclic_scan_process.argtypes = [vp, P(_Cf32), sz]; lib.atkdsp_cyclic_scan_process.restype = i
    lib.atkdsp_cyclic_scan_read.argtypes = [vp, P(_ScanLine), sz]; lib.atkdsp_cyclic_scan_read.restype = pt
    lib.atkdsp_cyclic_scan_level.argtypes = [vp, P(C.c_float), sz]; lib.atkdsp_cyclic_scan_level.restype = pt
    lib.atkdsp_cyclic_scan_nbins.argtypes = [vp]; lib.atkdsp_cyclic_scan_nbins.restype = i
    lib.atkdsp_cyclic_scan_describe.argtypes = [vp, C.c_char_p, sz]; lib.atkdsp_cyclic_scan_describe.restype = i
    lib.atkdsp_cyclic_scan_stats.argtypes = [vp, P(_CyclicCounters)]; lib.atkdsp_cyclic_scan_stats.restype = i
    # code + codes
    lib.atkdsp_dsss_code_defaults.argtypes = [P(_DsssCodeCfg), d, d, d]; lib.atkdsp_dsss_code_defaults.restype = None
    lib.atkdsp_dsss_code.argtypes = [P(_Cf32), sz, P(_DsssCodeCfg), i8, sz, P(_DsssCodeResult)]
    lib.atkdsp_dsss_code.restype = pt
    lib.atkdsp_parity_z.argtypes = [i8, i, C.c_uint64, i]; lib.atkdsp_parity_z.restype = d
    lib.atkdsp_primitive_polys.argtypes = [i, P(C.c_uint64), sz]; lib.atkdsp_primitive_polys.restype = pt
    lib.atkdsp_gps_ca.argtypes = [i, i8, sz]; lib.atkdsp_gps_ca.restype = pt
    lib.atkdsp_barker.argtypes = [i, i8, sz]; lib.atkdsp_barker.restype = pt
    lib.atkdsp_ieee802154_chips.argtypes = [i, i8, sz]; lib.atkdsp_ieee802154_chips.restype = pt
    # acquire / track
    lib.atkdsp_acq_threshold.argtypes = [i, d, d]; lib.atkdsp_acq_threshold.restype = d
    lib.atkdsp_dsss_acq_defaults.argtypes = [P(_DsssAcqCfg), d, d, i]; lib.atkdsp_dsss_acq_defaults.restype = None
    lib.atkdsp_dsss_acquire.argtypes = [P(_Cf32), sz, i8, P(_DsssAcqCfg), P(_DsssAcq)]
    lib.atkdsp_dsss_acquire.restype = i
    lib.atkdsp_dsss_trk_defaults.argtypes = [P(_DsssTrkCfg), d, d, i, d, d]; lib.atkdsp_dsss_trk_defaults.restype = None
    lib.atkdsp_dsss_trk_create.argtypes = [i8, P(_DsssTrkCfg)]; lib.atkdsp_dsss_trk_create.restype = vp
    lib.atkdsp_dsss_trk_destroy.argtypes = [vp]; lib.atkdsp_dsss_trk_destroy.restype = None
    lib.atkdsp_dsss_trk_process.argtypes = [vp, P(_Cf32), sz, P(_Cf32), sz]; lib.atkdsp_dsss_trk_process.restype = pt
    lib.atkdsp_dsss_trk_read.argtypes = [vp, P(_DsssTrkState)]; lib.atkdsp_dsss_trk_read.restype = i
    lib.atkdsp_dsss_trk_describe.argtypes = [vp, C.c_char_p, sz]; lib.atkdsp_dsss_trk_describe.restype = i
    lib.atkdsp_dsss_trk_stats.argtypes = [vp, P(_DsssCounters)]; lib.atkdsp_dsss_trk_stats.restype = i
    # css
    lib.atkdsp_css_probe_create.argtypes = [d, P(_CssCand), i, i, d]; lib.atkdsp_css_probe_create.restype = vp
    lib.atkdsp_css_probe_destroy.argtypes = [vp]; lib.atkdsp_css_probe_destroy.restype = None
    lib.atkdsp_css_probe_reset.argtypes = [vp]; lib.atkdsp_css_probe_reset.restype = None
    lib.atkdsp_css_probe_process.argtypes = [vp, P(_Cf32), sz]; lib.atkdsp_css_probe_process.restype = i
    lib.atkdsp_css_probe_read.argtypes = [vp, P(_CssResult)]; lib.atkdsp_css_probe_read.restype = i
    lib.atkdsp_css_probe_describe.argtypes = [vp, C.c_char_p, sz]; lib.atkdsp_css_probe_describe.restype = i
    lib.atkdsp_css_probe_stats.argtypes = [vp, P(_CyclicCounters)]; lib.atkdsp_css_probe_stats.restype = i
    lib.atkdsp_css_demod.argtypes = [P(_Cf32), sz, d, i, d, d, d, u16, sz]; lib.atkdsp_css_demod.restype = pt
    # xspec / align / lattice
    lib.atkdsp_msc_threshold.argtypes = [i, d, d]; lib.atkdsp_msc_threshold.restype = d
    lib.atkdsp_xspec_create.argtypes = [d, i, i, d]; lib.atkdsp_xspec_create.restype = vp
    lib.atkdsp_xspec_destroy.argtypes = [vp]; lib.atkdsp_xspec_destroy.restype = None
    lib.atkdsp_xspec_reset.argtypes = [vp]; lib.atkdsp_xspec_reset.restype = None
    lib.atkdsp_xspec_process.argtypes = [vp, P(_Cf32), P(_Cf32), sz]; lib.atkdsp_xspec_process.restype = i
    lib.atkdsp_xspec_baseline.argtypes = [vp, P(_Cf32), P(_Cf32), sz]; lib.atkdsp_xspec_baseline.restype = i
    lib.atkdsp_xspec_read.argtypes = [vp, P(_XspecResult)]; lib.atkdsp_xspec_read.restype = i
    lib.atkdsp_xspec_bins.argtypes = [vp, P(C.c_float), P(C.c_float)]; lib.atkdsp_xspec_bins.restype = i
    lib.atkdsp_xspec_describe.argtypes = [vp, C.c_char_p, sz]; lib.atkdsp_xspec_describe.restype = i
    lib.atkdsp_xspec_stats.argtypes = [vp, P(_CyclicCounters)]; lib.atkdsp_xspec_stats.restype = i
    lib.atkdsp_align_estimate.argtypes = [P(_Cf32), P(_Cf32), sz, d, i, i, P(_Align), P(C.c_float), sz]
    lib.atkdsp_align_estimate.restype = i
    lib.atkdsp_align_apply.argtypes = [P(_Cf32), sz, d, P(_Align), P(C.c_float), i, P(_Cf32)]
    lib.atkdsp_align_apply.restype = i
    lib.atkdsp_lattice.argtypes = [P(_LatticeIn), P(_LatticeOut)]; lib.atkdsp_lattice.restype = i


_BIND_EXTRA.append(_bind_dsss)


def _bind_pfb(lib) -> None:
    P = C.POINTER
    vp, sz, u, d, i64 = C.c_void_p, C.c_size_t, C.c_uint, C.c_double, C.c_int64
    pcf, ppcf = P(_Cf32), P(P(_Cf32))
    lib.atkdsp_pfb_create.argtypes = [d, u, u, u, d, u]; lib.atkdsp_pfb_create.restype = vp
    lib.atkdsp_pfb_destroy.argtypes = [vp]; lib.atkdsp_pfb_destroy.restype = None
    lib.atkdsp_pfb_reset.argtypes = [vp]; lib.atkdsp_pfb_reset.restype = None
    lib.atkdsp_pfb_nchan.argtypes = [vp]; lib.atkdsp_pfb_nchan.restype = u
    lib.atkdsp_pfb_out_rate.argtypes = [vp]; lib.atkdsp_pfb_out_rate.restype = d
    lib.atkdsp_pfb_out_hops.argtypes = [vp, sz]; lib.atkdsp_pfb_out_hops.restype = sz
    lib.atkdsp_pfb_process.argtypes = [vp, pcf, sz, pcf, sz]; lib.atkdsp_pfb_process.restype = C.c_ssize_t
    lib.atkdsp_pfb_process_n.argtypes = [vp, ppcf, u, sz, pcf, sz]; lib.atkdsp_pfb_process_n.restype = C.c_ssize_t
    lib.atkdsp_pfb_block_origin.argtypes = [vp, i64]; lib.atkdsp_pfb_block_origin.restype = i64
    lib.atkdsp_pfb_group_delay.argtypes = [vp]; lib.atkdsp_pfb_group_delay.restype = d
    lib.atkdsp_pfb_set_cal.argtypes = [vp, pcf, u]; lib.atkdsp_pfb_set_cal.restype = C.c_int
    lib.atkdsp_pfb_describe.argtypes = [vp, C.c_char_p, sz]; lib.atkdsp_pfb_describe.restype = C.c_int
    lib.atkdsp_pfb_cov_accumulate.argtypes = [pcf, u, u, sz, P(C.c_uint8), pcf, P(C.c_uint32)]
    lib.atkdsp_pfb_cov_accumulate.restype = None
    lib.atkdsp_pfb_cov_reset.argtypes = [pcf, P(C.c_uint32), u, u, P(C.c_uint8)]
    lib.atkdsp_pfb_cov_reset.restype = None
    lib.atkdsp_pfb_channel_power.argtypes = [pcf, u, u, sz, P(C.c_float), P(C.c_float)]
    lib.atkdsp_pfb_channel_power.restype = None


_BIND_EXTRA.append(_bind_pfb)


def _bind_osx(lib) -> None:
    P = C.POINTER
    vp, sz, u, d, i = C.c_void_p, C.c_size_t, C.c_uint, C.c_double, C.c_int
    pcf, ppcf, psz = P(_Cf32), P(P(_Cf32)), P(C.c_size_t)
    lib.atkdsp_osx_create.argtypes = [d, sz, d, u]; lib.atkdsp_osx_create.restype = vp
    lib.atkdsp_osx_destroy.argtypes = [vp]; lib.atkdsp_osx_destroy.restype = None
    lib.atkdsp_osx_reset.argtypes = [vp]; lib.atkdsp_osx_reset.restype = None
    lib.atkdsp_osx_add.argtypes = [vp, d, d, d]; lib.atkdsp_osx_add.restype = i
    lib.atkdsp_osx_retune.argtypes = [vp, i, d]; lib.atkdsp_osx_retune.restype = i
    lib.atkdsp_osx_remove.argtypes = [vp, i]; lib.atkdsp_osx_remove.restype = i
    lib.atkdsp_osx_channels.argtypes = [vp]; lib.atkdsp_osx_channels.restype = i
    lib.atkdsp_osx_chan_rate.argtypes = [vp, i]; lib.atkdsp_osx_chan_rate.restype = d
    lib.atkdsp_osx_out_max.argtypes = [vp, i, sz]; lib.atkdsp_osx_out_max.restype = sz
    lib.atkdsp_osx_process.argtypes = [vp, pcf, sz, ppcf, psz, sz]; lib.atkdsp_osx_process.restype = C.c_ssize_t
    lib.atkdsp_osx_process_n.argtypes = [vp, ppcf, u, sz, ppcf, psz, sz]; lib.atkdsp_osx_process_n.restype = C.c_ssize_t
    lib.atkdsp_osx_describe.argtypes = [vp, C.c_char_p, sz]; lib.atkdsp_osx_describe.restype = i


_BIND_EXTRA.append(_bind_osx)


class _UndertowCfg(C.Structure):
    _fields_ = [("fs", C.c_double), ("nscales", C.c_int32), ("nchan0", C.c_int32), ("win", C.c_int32),
                ("max_lag", C.c_int32), ("period_min", C.c_int32), ("acf_lags", C.c_int32),
                ("seg_len", C.c_int32), ("taps_per_poly", C.c_int32), ("atten_db", C.c_double),
                ("pfa", C.c_double), ("max_block", C.c_int32), ("reserved", C.c_int32 * 3)]


class _UndertowCell(C.Structure):
    _fields_ = [("fc_hz", C.c_double), ("bw_hz", C.c_double), ("fluct_log10_pfa", C.c_double),
                ("period_s", C.c_double), ("acf_log10_pfa", C.c_double), ("acf_z", C.c_double),
                ("cyc_log10_pfa", C.c_double), ("cyc_alpha_hz", C.c_double), ("cyc_z", C.c_double),
                ("rc_hz", C.c_double), ("L", C.c_double), ("log10_pfa", C.c_double),
                ("scale", C.c_int32), ("chan", C.c_int32), ("nwin", C.c_int32), ("nseen", C.c_int32),
                ("flags", C.c_uint32), ("method", C.c_int32 * 4), ("reserved", C.c_int32)]


def _bind_undertow(lib) -> None:
    P = C.POINTER
    vp, sz, i, d = C.c_void_p, C.c_size_t, C.c_int, C.c_double
    lib.atkdsp_undertow_defaults.argtypes = [P(_UndertowCfg), d, i, i]; lib.atkdsp_undertow_defaults.restype = None
    lib.atkdsp_undertow_create.argtypes = [P(_UndertowCfg)]; lib.atkdsp_undertow_create.restype = vp
    lib.atkdsp_undertow_destroy.argtypes = [vp]; lib.atkdsp_undertow_destroy.restype = None
    lib.atkdsp_undertow_reset.argtypes = [vp]; lib.atkdsp_undertow_reset.restype = None
    lib.atkdsp_undertow_process.argtypes = [vp, P(_Cf32), sz]; lib.atkdsp_undertow_process.restype = i
    lib.atkdsp_undertow_ncells.argtypes = [vp]; lib.atkdsp_undertow_ncells.restype = i
    lib.atkdsp_undertow_cells.argtypes = [vp, P(_UndertowCell), sz]; lib.atkdsp_undertow_cells.restype = C.c_ssize_t
    lib.atkdsp_undertow_read.argtypes = [vp, P(_UndertowCell), sz]; lib.atkdsp_undertow_read.restype = C.c_ssize_t
    lib.atkdsp_undertow_describe.argtypes = [vp, C.c_char_p, sz]; lib.atkdsp_undertow_describe.restype = i
    lib.atkdsp_t_log10_sf2.argtypes = [d, d]; lib.atkdsp_t_log10_sf2.restype = d


_BIND_EXTRA.append(_bind_undertow)


def method(method_id: int):
    """(name, authors, year, country) or None — the twin of atkdsp_method()."""
    p = load().atkdsp_method(int(method_id))
    if not p:
        return None
    m = p.contents
    return (m.name.decode(), m.authors.decode(), m.year.decode(), m.country.decode())


DSSS_H_FLUCT, DSSS_H_CAV, DSSS_H_ACF = 1, 2, 4
DSSS_FLUCT, DSSS_CAV, DSSS_PERIOD, DSSS_WARMUP, DSSS_ACF, DSSS_DETECT = 1, 2, 4, 8, 16, 32
SCAN_LINE, SCAN_HARMONIC, SCAN_WARMUP = 1, 2, 4
CHIP_BPSK, CHIP_QPSK, CHIP_OQPSK = 0, 1, 2
CODE_IDENTIFIABLE, CODE_CFO, CODE_POLY, CODE_NAMED, CODE_Q, CODE_REFUSED, CODE_DECIM, CODE_REGEN = 1, 2, 4, 8, 16, 32, 64, 128
POLY_NONE, POLY_BM, POLY_PARITY = 0, 1, 2
ACQ_NPDI, ACQ_MBAPDI, ACQ_DETECT = 0, 1, 1
TRK_LOCK, TRK_VALID_CN0, TRK_DISTORTED, TRK_POWER_HIGH = 1, 2, 4, 8
CSS_DETECT, CSS_PREAMBLE, CSS_WARMUP = 1, 2, 4
XS_DETECT, XS_WARMUP, XS_TDOA = 1, 2, 4
ALIGN_FOUND = 1
LAT_AGREE_RC, LAT_INTEGER_L, LAT_FAMILY, LAT_ONE_CLOCK, LAT_CONSISTENT = 1, 2, 4, 8, 16


def _i8(code):
    a = np.ascontiguousarray(code, np.int8)
    return a.ctypes.data_as(C.POINTER(C.c_int8)), a


def gps_ca(prn: int) -> np.ndarray:
    buf = (C.c_int8 * 1023)()
    _check(load().atkdsp_gps_ca(int(prn), buf, 1023), "gps_ca")
    return np.ctypeslib.as_array(buf).copy()


def barker(length: int) -> np.ndarray:
    buf = (C.c_int8 * 13)()
    n = _check(load().atkdsp_barker(int(length), buf, 13), "barker")
    return np.ctypeslib.as_array(buf)[:n].copy()


def ieee802154_chips(symbol: int) -> np.ndarray:
    buf = (C.c_int8 * 32)()
    _check(load().atkdsp_ieee802154_chips(int(symbol), buf, 32), "ieee802154_chips")
    return np.ctypeslib.as_array(buf).copy()


def primitive_polys(degree: int) -> list:
    lib = load()
    n = lib.atkdsp_primitive_polys(int(degree), None, 0)
    if n < 0:
        raise AtkDspError("primitive_polys: degree 2..16")
    buf = (C.c_uint64 * n)()
    lib.atkdsp_primitive_polys(int(degree), buf, n)
    return [int(x) for x in buf]


def parity_z(chips, poly: int, degree: int) -> float:
    p, a = _i8(chips)
    return float(load().atkdsp_parity_z(p, a.size, C.c_uint64(int(poly)), int(degree)))


def parity_z_frob(chips, poly: int, degree: int):
    """(Z, N) over the Frobenius stack — binding of atkdsp_parity_z_frob."""
    lib = load()
    if not getattr(lib, "_frob_bound", False):
        lib.atkdsp_parity_z_frob.argtypes = [C.POINTER(C.c_int8), C.c_int, C.c_uint64, C.c_int, C.POINTER(C.c_int32)]
        lib.atkdsp_parity_z_frob.restype = C.c_double
        lib._frob_bound = True
    p, a = _i8(chips)
    n = C.c_int32(0)
    z = lib.atkdsp_parity_z_frob(p, a.size, C.c_uint64(int(poly)), int(degree), C.byref(n))
    return float(z), int(n.value)


def cav_threshold(lags: int, ns: float, pfa: float) -> float:
    return float(load().atkdsp_cav_threshold(int(lags), float(ns), float(pfa)))


def acq_threshold(dwells: int, ncells: float, pfa: float) -> float:
    return float(load().atkdsp_acq_threshold(int(dwells), float(ncells), float(pfa)))


def msc_threshold(navg: int, nbins: float, pfa: float) -> float:
    return float(load().atkdsp_msc_threshold(int(navg), float(nbins), float(pfa)))


def dsss_det_defaults(fs: float, win: int, max_lag: int) -> dict:
    return _ref.dsss_det_defaults(fs, win, max_lag)


class DsssDet:
    """Twin-checked binding of atkdsp_dsss_det (atkdsp.h §25)."""

    def __init__(self, cfg: dict):
        self._lib = load()
        try:
            _ref.DsssDet(cfg)
        except ValueError as e:
            raise AtkDspError(f"dsss_det_create refused — {e}") from None
        c = _DsssDetCfg()
        self._lib.atkdsp_dsss_det_defaults(C.byref(c), float(cfg["fs"]), int(cfg["win"]), int(cfg["max_lag"]))
        for k in ("nwin_min", "handles", "cav_lags", "min_peaks", "period_min"):
            if k in cfg:
                setattr(c, k, int(cfg[k]))
        if "pfa" in cfg:
            c.pfa = float(cfg["pfa"])
        self._h = self._lib.atkdsp_dsss_det_create(C.byref(c))
        if not self._h:
            raise AtkDspError("dsss_det_create refused")
        self._max_lag = int(cfg["max_lag"])

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.atkdsp_dsss_det_destroy(h)

    def reset(self):
        self._lib.atkdsp_dsss_det_reset(self._h)

    def process(self, x):
        a = _cf32(x)
        _check(self._lib.atkdsp_dsss_det_process(self._h, _cfp(a), a.size), "dsss_det_process")

    def read(self) -> dict:
        r = _DsssDetResult()
        _check(self._lib.atkdsp_dsss_det_read(self._h, C.byref(r)), "dsss_det_read")
        return _struct_dict(r)

    def lags(self, k: int = 3) -> list:
        """The k strongest interior fluctuation peaks (atkdsp_dsss_det_lags, §30):
        candidate periods for psync even when the handle has not fired."""
        k = int(k)
        if k < 1:
            return []
        buf = (_DsssLag * k)()
        n = _check(self._lib.atkdsp_dsss_det_lags(self._h, buf, k), "dsss_det_lags")
        return [dict(lag=int(buf[i].lag), period=float(buf[i].period), ratio=float(buf[i].ratio),
                     log10_pfa=float(buf[i].log10_pfa)) for i in range(n)]

    def rho(self) -> np.ndarray:
        buf = (C.c_double * (self._max_lag + 1))()
        n = _check(self._lib.atkdsp_dsss_det_rho(self._h, buf, self._max_lag + 1), "dsss_det_rho")
        return np.ctypeslib.as_array(buf)[:n].copy()

    def describe(self) -> str:
        buf = C.create_string_buffer(1024)
        self._lib.atkdsp_dsss_det_describe(self._h, buf, 1024)
        return buf.value.decode()

    def stats(self) -> dict:
        s = _DsssCounters()
        self._lib.atkdsp_dsss_det_stats(self._h, C.byref(s))
        return {n: int(getattr(s, n)) for n, _t in s._fields_}


class CyclicScan:
    """Twin-checked binding of atkdsp_cyclic_scan (atkdsp.h §25)."""

    def __init__(self, fs, nonlin, hard_limit, seg_len, nseg, f_lo, f_hi, pfa):
        self._lib = load()
        nl = _nl_code(nonlin)
        try:
            _ref.CyclicScan(fs, nl, hard_limit, seg_len, nseg, f_lo, f_hi, pfa)
        except ValueError as e:
            raise AtkDspError(f"cyclic_scan_create refused — {e}") from None
        self._h = self._lib.atkdsp_cyclic_scan_create(float(fs), int(nl), int(hard_limit),
                                                      int(seg_len), int(nseg), float(f_lo),
                                                      float(f_hi), float(pfa))
        if not self._h:
            raise AtkDspError("cyclic_scan_create refused")
        self._nb = self._lib.atkdsp_cyclic_scan_nbins(self._h)

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.atkdsp_cyclic_scan_destroy(h)

    def reset(self):
        self._lib.atkdsp_cyclic_scan_reset(self._h)

    def process(self, x):
        a = _cf32(x)
        _check(self._lib.atkdsp_cyclic_scan_process(self._h, _cfp(a), a.size), "cyclic_scan_process")

    def read(self, cap: int = 64) -> list:
        buf = (_ScanLine * int(cap))()
        n = _check(self._lib.atkdsp_cyclic_scan_read(self._h, buf, int(cap)), "cyclic_scan_read")
        return [_struct_dict(buf[i]) for i in range(n)]

    def level(self) -> np.ndarray:
        buf = (C.c_float * self._nb)()
        n = _check(self._lib.atkdsp_cyclic_scan_level(self._h, buf, self._nb), "cyclic_scan_level")
        return np.ctypeslib.as_array(buf)[:n].copy()

    def nbins(self) -> int:
        return int(self._nb)

    def describe(self) -> str:
        buf = C.create_string_buffer(1024)
        self._lib.atkdsp_cyclic_scan_describe(self._h, buf, 1024)
        return buf.value.decode()

    def stats(self) -> dict:
        s = _CyclicCounters()
        self._lib.atkdsp_cyclic_scan_stats(self._h, C.byref(s))
        return {n: int(getattr(s, n)) for n, _t in s._fields_}


def dsss_code_defaults(fs, rc, period_s) -> dict:
    return _ref.dsss_code_defaults(fs, rc, period_s)


def dsss_code(iq, cfg: dict):
    """(chips, result) — binding of atkdsp_dsss_code."""
    lib = load()
    a = _cf32(iq)
    c = _DsssCodeCfg()
    lib.atkdsp_dsss_code_defaults(C.byref(c), float(cfg["fs"]), float(cfg["rc"]), float(cfg["period_s"]))
    for k in ("chip_mod", "vote_passes", "poly_max_degree", "no_cfo", "no_decim"):
        if k in cfg:
            setattr(c, k, int(cfg[k]))
    if "pfa" in cfg:
        c.pfa = float(cfg["pfa"])
    two = int(cfg.get("chip_mod", 0)) != 0
    sps = c.fs / c.rc
    L = int(round(c.period_s * c.fs / sps))
    cap = L * (2 if two else 1) + 2
    buf = (C.c_int8 * cap)()
    r = _DsssCodeResult()
    n = lib.atkdsp_dsss_code(_cfp(a), a.size, C.byref(c), buf, cap, C.byref(r))
    if n < 0:
        _check(n, "dsss_code")
    res = _struct_dict(r)
    chips = np.ctypeslib.as_array(buf)[: (r.L * (2 if (r.flags & CODE_Q) else 1)) if n else 0].copy()
    return chips, res


def dsss_acq_defaults(fs, rc, L) -> dict:
    return _ref.dsss_acq_defaults(fs, rc, L)


def dsss_acquire(iq, code, cfg: dict) -> dict:
    lib = load()
    a = _cf32(iq)
    p, ca = _i8(code)
    c = _DsssAcqCfg()
    lib.atkdsp_dsss_acq_defaults(C.byref(c), float(cfg["fs"]), float(cfg["rc"]), int(cfg["L"]))
    for k in ("ncoh", "nnoncoh", "combiner"):
        if k in cfg:
            setattr(c, k, int(cfg[k]))
    for k in ("dop_lo", "dop_hi", "dop_step", "pfa"):
        if k in cfg:
            setattr(c, k, float(cfg[k]))
    out = _DsssAcq()
    _check(lib.atkdsp_dsss_acquire(_cfp(a), a.size, p, C.byref(c), C.byref(out)), "dsss_acquire")
    return _struct_dict(out)


def dsss_trk_defaults(fs, rc, L, code_phase, doppler_hz) -> dict:
    return _ref.dsss_trk_defaults(fs, rc, L, code_phase, doppler_hz)


class DsssTrk:
    """Binding of atkdsp_dsss_trk (atkdsp.h §25)."""

    def __init__(self, code, cfg: dict):
        self._lib = load()
        p, self._code = _i8(code)
        c = _DsssTrkCfg()
        self._lib.atkdsp_dsss_trk_defaults(C.byref(c), float(cfg["fs"]), float(cfg["rc"]), int(cfg["L"]),
                                           float(cfg["code_phase"]), float(cfg["doppler_hz"]))
        for k in ("dll_bw_hz", "pll_bw_hz", "spacing", "quality_spacing", "power_window_s",
                  "distortion_pfa", "power_high_db"):
            if k in cfg:
                setattr(c, k, float(cfg[k]))
        if "cn0_periods" in cfg:
            c.cn0_periods = int(cfg["cn0_periods"])
        self._h = self._lib.atkdsp_dsss_trk_create(p, C.byref(c))
        if not self._h:
            raise AtkDspError("dsss_trk_create refused")

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.atkdsp_dsss_trk_destroy(h)

    def process(self, x, cap: int | None = None) -> np.ndarray:
        a = _cf32(x)
        if cap is None:
            cap = a.size  # at most one prompt per sample
        buf = (_Cf32 * int(cap))()
        n = _check(self._lib.atkdsp_dsss_trk_process(self._h, _cfp(a), a.size, buf, int(cap)),
                   "dsss_trk_process")
        return np.array([complex(buf[i].re, buf[i].im) for i in range(n)], np.complex64)

    def read(self) -> dict:
        st = _DsssTrkState()
        _check(self._lib.atkdsp_dsss_trk_read(self._h, C.byref(st)), "dsss_trk_read")
        return _struct_dict(st)

    def describe(self) -> str:
        buf = C.create_string_buffer(512)
        self._lib.atkdsp_dsss_trk_describe(self._h, buf, 512)
        return buf.value.decode()

    def stats(self) -> dict:
        s = _DsssCounters()
        self._lib.atkdsp_dsss_trk_stats(self._h, C.byref(s))
        return {n: int(getattr(s, n)) for n, _t in s._fields_}


class CssProbe:
    """Binding of atkdsp_css_probe (atkdsp.h §26)."""

    def __init__(self, fs, cands, preamble_min, pfa):
        self._lib = load()
        cl = [(int(s), float(b)) for (s, b) in cands]
        try:
            _ref.CssProbe(fs, cl, preamble_min, pfa)
        except ValueError as e:
            raise AtkDspError(f"css_probe_create refused — {e}") from None
        arr = (_CssCand * len(cl))(*[_CssCand(s, 0, b) for s, b in cl])
        self._h = self._lib.atkdsp_css_probe_create(float(fs), arr, len(cl), int(preamble_min), float(pfa))
        if not self._h:
            raise AtkDspError("css_probe_create refused")
        self.n = len(cl)

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.atkdsp_css_probe_destroy(h)

    def reset(self):
        self._lib.atkdsp_css_probe_reset(self._h)

    def process(self, x):
        a = _cf32(x)
        _check(self._lib.atkdsp_css_probe_process(self._h, _cfp(a), a.size), "css_probe_process")

    def read(self) -> list:
        buf = (_CssResult * self.n)()
        _check(self._lib.atkdsp_css_probe_read(self._h, buf), "css_probe_read")
        return [_struct_dict(buf[i]) for i in range(self.n)]

    def describe(self) -> str:
        buf = C.create_string_buffer(512)
        self._lib.atkdsp_css_probe_describe(self._h, buf, 512)
        return buf.value.decode()

    def stats(self) -> dict:
        s = _CyclicCounters()
        self._lib.atkdsp_css_probe_stats(self._h, C.byref(s))
        return {n: int(getattr(s, n)) for n, _t in s._fields_}


def css_demod(iq, fs, sf, bw_hz, offset_hz, t0, cap: int = 1 << 20) -> np.ndarray:
    lib = load()
    a = _cf32(iq)
    buf = (C.c_uint16 * int(cap))()
    n = _check(lib.atkdsp_css_demod(_cfp(a), a.size, float(fs), int(sf), float(bw_hz), float(offset_hz),
                                    float(t0), buf, int(cap)), "css_demod")
    return np.ctypeslib.as_array(buf)[:n].copy()


class Xspec:
    """Binding of atkdsp_xspec (atkdsp.h §27)."""

    def __init__(self, fs, nfft, navg, pfa):
        self._lib = load()
        try:
            _ref.Xspec(fs, nfft, navg, pfa)
        except ValueError as e:
            raise AtkDspError(f"xspec_create refused — {e}") from None
        self._h = self._lib.atkdsp_xspec_create(float(fs), int(nfft), int(navg), float(pfa))
        if not self._h:
            raise AtkDspError("xspec_create refused")
        self._nfft = int(nfft)

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.atkdsp_xspec_destroy(h)

    def reset(self):
        self._lib.atkdsp_xspec_reset(self._h)

    def process(self, x1, x2):
        a, b = _cf32(x1), _cf32(x2)
        _check(self._lib.atkdsp_xspec_process(self._h, _cfp(a), _cfp(b), min(a.size, b.size)),
               "xspec_process")

    def baseline(self, x1, x2):
        a, b = _cf32(x1), _cf32(x2)
        _check(self._lib.atkdsp_xspec_baseline(self._h, _cfp(a), _cfp(b), min(a.size, b.size)),
               "xspec_baseline")

    def read(self) -> dict:
        r = _XspecResult()
        _check(self._lib.atkdsp_xspec_read(self._h, C.byref(r)), "xspec_read")
        return _struct_dict(r)

    def bins(self):
        coh = (C.c_float * self._nfft)()
        ph = (C.c_float * self._nfft)()
        _check(self._lib.atkdsp_xspec_bins(self._h, coh, ph), "xspec_bins")
        return np.ctypeslib.as_array(coh).copy(), np.ctypeslib.as_array(ph).copy()

    def describe(self) -> str:
        buf = C.create_string_buffer(512)
        self._lib.atkdsp_xspec_describe(self._h, buf, 512)
        return buf.value.decode()

    def stats(self) -> dict:
        s = _CyclicCounters()
        self._lib.atkdsp_xspec_stats(self._h, C.byref(s))
        return {n: int(getattr(s, n)) for n, _t in s._fields_}


def align_estimate(x1, x2, fs, max_lag, block: int = 0):
    """(result, phase) — binding of atkdsp_align_estimate."""
    lib = load()
    a, b = _cf32(x1), _cf32(x2)
    n = min(a.size, b.size)
    out = _Align()
    phase = None
    pp = None
    if block and block > 0:
        nb = (n + block - 1) // block
        phase = (C.c_float * nb)()
        pp = phase
    _check(lib.atkdsp_align_estimate(_cfp(a), _cfp(b), n, float(fs), int(max_lag), int(block),
                                     C.byref(out), pp, len(phase) if phase else 0), "align_estimate")
    ph = np.ctypeslib.as_array(phase)[: out.nblocks].copy() if phase else None
    return _struct_dict(out), ph


def align_apply(x2, fs, a: dict, phase=None, block: int = 0) -> np.ndarray:
    lib = load()
    arr = _cf32(x2)
    out = (_Cf32 * arr.size)()
    st = _Align(lag=a["lag"], lag_err=a.get("lag_err", 0.0), rate_ppm=a["rate_ppm"],
                lo_offset_hz=a["lo_offset_hz"], peak_ratio=a.get("peak_ratio", 0.0),
                snr_db=a.get("snr_db", 0.0), nblocks=a.get("nblocks", 0), flags=a.get("flags", 0))
    pp = None
    if phase is not None and block > 0:
        pa = np.ascontiguousarray(phase, np.float32)
        pp = pa.ctypes.data_as(C.POINTER(C.c_float))
    _check(lib.atkdsp_align_apply(_cfp(arr), arr.size, float(fs), C.byref(st), pp, int(block), out),
           "align_apply")
    return np.array([complex(out[i].re, out[i].im) for i in range(arr.size)], np.complex64)


def lattice(in_: dict) -> dict:
    lib = load()
    li = _LatticeIn()
    for k in ("rc_acf_hz", "rc_line_hz", "rc_nominal_hz", "period_s", "fc_hz", "cfo_hz",
              "rc_tol", "ppm_tol", "rx_ppm"):
        if k in in_:
            setattr(li, k, float(in_[k]))
    out = _LatticeOut()
    _check(lib.atkdsp_lattice(C.byref(li), C.byref(out)), "lattice")
    return _struct_dict(out)


__all__ += ["method", "gps_ca", "barker", "ieee802154_chips", "primitive_polys", "parity_z",
            "cav_threshold", "acq_threshold", "msc_threshold", "dsss_det_defaults", "DsssDet",
            "CyclicScan", "dsss_code_defaults", "dsss_code", "dsss_acq_defaults", "dsss_acquire",
            "dsss_trk_defaults", "DsssTrk", "CssProbe", "css_demod", "Xspec", "align_estimate",
            "align_apply", "lattice", "DSSS_H_FLUCT", "DSSS_H_CAV", "DSSS_H_ACF", "DSSS_FLUCT",
            "DSSS_CAV", "DSSS_PERIOD", "DSSS_WARMUP", "DSSS_ACF", "DSSS_DETECT", "SCAN_LINE",
            "SCAN_HARMONIC", "CHIP_BPSK", "CHIP_QPSK", "CHIP_OQPSK", "CODE_IDENTIFIABLE", "CODE_CFO",
            "CODE_POLY", "CODE_NAMED", "CODE_Q", "CODE_REFUSED", "POLY_BM", "POLY_PARITY",
            "ACQ_NPDI", "ACQ_MBAPDI", "ACQ_DETECT", "TRK_LOCK", "TRK_VALID_CN0", "TRK_DISTORTED",
            "CSS_DETECT", "CSS_PREAMBLE", "XS_DETECT", "XS_TDOA", "ALIGN_FOUND", "LAT_AGREE_RC",
            "LAT_INTEGER_L", "LAT_FAMILY", "LAT_ONE_CLOCK", "LAT_CONSISTENT"]


# -- §28 the channelizer ------------------------------------------------------
class Pfb:
    """Oversampled polyphase filterbank (atkdsp_pfb). nchan channels on a
    uniform grid spaced fs/nchan, `oversample`x (output rate = oversample *
    spacing). process_n runs N element streams on one block clock, the DF path.
    Twin: atkdsp.reference.Pfb."""

    def __init__(self, fs, nchan, oversample=2, taps_per_poly=12, atten_db=80.0, max_nel=1):
        self._lib = load()
        self._p = self._lib.atkdsp_pfb_create(float(fs), int(nchan), int(oversample),
                                              int(taps_per_poly), float(atten_db), int(max_nel))
        if not self._p:
            raise AtkDspError("pfb_create: check fs>0, nchan>=2, nchan%%oversample==0")
        self.nchan = int(nchan)
        self.max_nel = int(max_nel)

    def __del__(self):
        p = getattr(self, "_p", None)
        if p:
            self._lib.atkdsp_pfb_destroy(p)
            self._p = None

    def reset(self):
        self._lib.atkdsp_pfb_reset(self._p)

    def out_rate(self):
        return self._lib.atkdsp_pfb_out_rate(self._p)

    def group_delay(self):
        return self._lib.atkdsp_pfb_group_delay(self._p)

    def block_origin(self, hop_index):
        return int(self._lib.atkdsp_pfb_block_origin(self._p, int(hop_index)))

    def set_cal(self, table, nel):
        if table is None:
            _check(self._lib.atkdsp_pfb_set_cal(self._p, None, 0), "pfb_set_cal")
            return
        t = np.ascontiguousarray(np.asarray(table, np.complex64)).reshape(int(nel), self.nchan)
        _check(self._lib.atkdsp_pfb_set_cal(self._p, _cfp(t), int(nel)), "pfb_set_cal")

    def describe(self):
        buf = C.create_string_buffer(256)
        n = self._lib.atkdsp_pfb_describe(self._p, buf, 256)
        return buf.value.decode() if n > 0 else ""

    def process_n(self, ins, nel):
        nel = int(nel)
        arrs = [np.ascontiguousarray(np.asarray(ins[e]).ravel(), np.complex64) for e in range(nel)]
        n = arrs[0].size
        for a in arrs:
            if a.size != n:
                raise AtkDspError("pfb: elements must be equal length")
        m = self._lib.atkdsp_pfb_out_hops(self._p, n)
        out = np.zeros(nel * self.nchan * m, np.complex64)
        inarr = (C.POINTER(_Cf32) * nel)(*[_cfp(a) for a in arrs])
        got = self._lib.atkdsp_pfb_process_n(self._p, inarr, nel, n, _cfp(out), out.size)
        _check(got if got >= 0 else int(got), "pfb_process_n")
        return out[: nel * self.nchan * got].reshape(nel, self.nchan, got)

    def process(self, x):
        return self.process_n([x], 1)[0]


def pfb_cov_accumulate(out, nel, nchan, gate, R, counts):
    """W1 (atkdsp_pfb_cov_accumulate): R[k] += sum_i y_k y_k^H over gated
    channels; counts[k] += m. out is (nel,nchan,m) complex64; R is
    (nchan,nel,nel) complex64; counts is (nchan,) uint32. R and counts are
    updated in place (pass contiguous arrays). Twin: reference.pfb_cov_accumulate."""
    lib = load()
    o = np.ascontiguousarray(out, np.complex64)
    nel = int(nel); nchan = int(nchan); m = o.shape[2] if o.ndim == 3 else 0
    if not (R.flags["C_CONTIGUOUS"] and R.dtype == np.complex64):
        raise AtkDspError("pfb_cov_accumulate: R must be contiguous complex64")
    if not (counts.flags["C_CONTIGUOUS"] and counts.dtype == np.uint32):
        raise AtkDspError("pfb_cov_accumulate: counts must be contiguous uint32")
    gp = None
    if gate is not None:
        g = np.ascontiguousarray(gate, np.uint8)
        gp = g.ctypes.data_as(C.POINTER(C.c_uint8))
    lib.atkdsp_pfb_cov_accumulate(_cfp(o.reshape(-1)), nel, nchan, m, gp,
                                  _cfp(R.reshape(-1)), counts.ctypes.data_as(C.POINTER(C.c_uint32)))


def pfb_cov_reset(R, counts, nel, nchan, gate=None):
    lib = load()
    gp = None
    if gate is not None:
        g = np.ascontiguousarray(gate, np.uint8)
        gp = g.ctypes.data_as(C.POINTER(C.c_uint8))
    lib.atkdsp_pfb_cov_reset(_cfp(R.reshape(-1)), counts.ctypes.data_as(C.POINTER(C.c_uint32)),
                             int(nel), int(nchan), gp)


def pfb_channel_power(out, nel, nchan):
    """W2 (atkdsp_pfb_channel_power): returns (pwr[nel,nchan], mean_pwr[nchan])
    float32. Twin: reference.pfb_channel_power."""
    lib = load()
    o = np.ascontiguousarray(out, np.complex64)
    nel = int(nel); nchan = int(nchan); m = o.shape[2] if o.ndim == 3 else 0
    pwr = np.zeros(nel * nchan, np.float32)
    meanp = np.zeros(nchan, np.float32)
    lib.atkdsp_pfb_channel_power(_cfp(o.reshape(-1)), nel, nchan, m,
                                 pwr.ctypes.data_as(C.POINTER(C.c_float)),
                                 meanp.ctypes.data_as(C.POINTER(C.c_float)))
    return pwr.reshape(nel, nchan), meanp


__all__ += ["Pfb", "pfb_cov_accumulate", "pfb_cov_reset", "pfb_channel_power"]


class Osx:
    """Overlap-save extractor (atkdsp_osx): arbitrary channels at arbitrary
    centres, each landed on its own out_rate. add/retune/remove between calls
    without disturbing the others; process_n runs N elements phase-coherently.
    Twin: atkdsp.reference.Osx."""

    def __init__(self, fs, nfft=4096, atten_db=80.0, max_nel=1):
        self._lib = load()
        self._p = self._lib.atkdsp_osx_create(float(fs), int(nfft), float(atten_db), int(max_nel))
        if not self._p:
            raise AtkDspError("osx_create: check fs>0, nfft a power of two >= 64")
        self.max_nel = int(max_nel)

    def __del__(self):
        p = getattr(self, "_p", None)
        if p:
            self._lib.atkdsp_osx_destroy(p); self._p = None

    def reset(self):
        self._lib.atkdsp_osx_reset(self._p)

    def add(self, offset_hz, bw_hz, out_rate):
        ch = self._lib.atkdsp_osx_add(self._p, float(offset_hz), float(bw_hz), float(out_rate))
        return _check(ch if ch >= 0 else int(ch), "osx_add")

    def retune(self, ch, offset_hz):
        _check(self._lib.atkdsp_osx_retune(self._p, int(ch), float(offset_hz)), "osx_retune")

    def remove(self, ch):
        _check(self._lib.atkdsp_osx_remove(self._p, int(ch)), "osx_remove")

    def channels(self):
        return int(self._lib.atkdsp_osx_channels(self._p))

    def chan_rate(self, ch):
        return float(self._lib.atkdsp_osx_chan_rate(self._p, int(ch)))

    def out_max(self, ch, n):
        return int(self._lib.atkdsp_osx_out_max(self._p, int(ch), int(n)))

    def describe(self):
        buf = C.create_string_buffer(256)
        n = self._lib.atkdsp_osx_describe(self._p, buf, 256)
        return buf.value.decode() if n > 0 else ""

    def process_n(self, ins, nel):
        nel = int(nel)
        arrs = [np.ascontiguousarray(np.asarray(ins[e]).ravel(), np.complex64) for e in range(nel)]
        n = arrs[0].size
        ns = self.channels()
        active = [self.chan_rate(ci) > 0 for ci in range(ns)]
        cap_each = max([self.out_max(ci, n) for ci in range(ns) if active[ci]], default=0)
        bufs = [[(np.zeros(cap_each, np.complex64) if active[ci] else None) for ci in range(ns)]
                for _ in range(nel)]
        ptrs = (C.POINTER(_Cf32) * (nel * ns))()
        for e in range(nel):
            for ci in range(ns):
                ptrs[e * ns + ci] = _cfp(bufs[e][ci]) if active[ci] else None
        counts = (C.c_size_t * ns)() if ns else (C.c_size_t * 1)()
        inarr = (C.POINTER(_Cf32) * nel)(*[_cfp(a) for a in arrs])
        got = self._lib.atkdsp_osx_process_n(self._p, inarr, nel, n, ptrs, counts, cap_each)
        _check(got if got >= 0 else int(got), "osx_process_n")
        cnts = [int(counts[ci]) for ci in range(ns)]
        outs = [None] * ns
        for ci in range(ns):
            if active[ci]:
                m = cnts[ci]
                outs[ci] = (np.array([bufs[e][ci][:m] for e in range(nel)], np.complex64)
                            if m else np.zeros((nel, 0), np.complex64))
        return outs, cnts

    def process(self, x):
        outs, cnts = self.process_n([x], 1)
        return [o[0] if o is not None else None for o in outs], cnts


__all__ += ["Osx"]


# -- §29 Undertow: the scale-space survey --------------------------------------
UT_FLUCT, UT_ACF, UT_CYC, UT_PERIOD, UT_CONSISTENT, UT_WARMUP = 1, 2, 4, 8, 16, 32


def undertow_defaults(fs, nscales=4, lmax=64) -> dict:
    """The survey config (a dict, the twin's keys) for codes up to lmax chips."""
    lib = load()
    c = _UndertowCfg()
    lib.atkdsp_undertow_defaults(C.byref(c), float(fs), int(nscales), int(lmax))
    out = _struct_dict(c)
    out.pop("reserved", None)
    return out


def t_log10_sf2(t, nu) -> float:
    """log10 P(|T| > t), Student's t with nu dof (atkdsp_t_log10_sf2)."""
    return float(load().atkdsp_t_log10_sf2(float(t), float(nu)))


class Undertow:
    """The scale-space survey (atkdsp_undertow). cfg: undertow_defaults(...)
    with any field changed. Twin: atkdsp.reference.Undertow."""

    def __init__(self, cfg: dict):
        self._lib = load()
        c = _UndertowCfg()
        self._lib.atkdsp_undertow_defaults(C.byref(c), float(cfg["fs"]), int(cfg.get("nscales", 4)), 64)
        for k, v in cfg.items():
            if k in ("reserved",):
                continue
            if not hasattr(c, k):
                raise AtkDspError(f"undertow: unknown config key {k!r}")
            setattr(c, k, type(getattr(c, k))(v))
        self._p = self._lib.atkdsp_undertow_create(C.byref(c))
        if not self._p:
            raise AtkDspError("undertow_create refused the config (see atkdsp.h §29)")
        self.ncells = int(self._lib.atkdsp_undertow_ncells(self._p))

    def __del__(self):
        p = getattr(self, "_p", None)
        if p:
            self._lib.atkdsp_undertow_destroy(p); self._p = None

    def reset(self):
        self._lib.atkdsp_undertow_reset(self._p)

    def process(self, x):
        a = np.ascontiguousarray(np.asarray(x).ravel(), np.complex64)
        _check(self._lib.atkdsp_undertow_process(self._p, _cfp(a), a.size), "undertow_process")

    def cells(self) -> list:
        buf = (_UndertowCell * self.ncells)()
        n = self._lib.atkdsp_undertow_cells(self._p, buf, self.ncells)
        _check(n if n >= 0 else int(n), "undertow_cells")
        return [self._d(buf[i]) for i in range(n)]

    def read(self) -> list:
        buf = (_UndertowCell * self.ncells)()
        n = self._lib.atkdsp_undertow_read(self._p, buf, self.ncells)
        _check(n if n >= 0 else int(n), "undertow_read")
        return [self._d(buf[i]) for i in range(n)]

    @staticmethod
    def _d(c):
        d = _struct_dict(c)
        d.pop("reserved", None)
        return d

    def candidates(self, max_log10: float = -1.0, per_cell: int = 3) -> list:
        """Every cell's per_cell strongest fluctuation lags with a CELL-level
        tail <= max_log10 (atkdsp_undertow_candidates, §30) — psync's seeds."""
        n = self._lib.atkdsp_undertow_candidates(self._p, float(max_log10), int(per_cell), None, 0)
        _check(n if n >= 0 else int(n), "undertow_candidates")
        if n == 0:
            return []
        buf = (_UndertowCand * n)()
        n = self._lib.atkdsp_undertow_candidates(self._p, float(max_log10), int(per_cell), buf, n)
        _check(n if n >= 0 else int(n), "undertow_candidates")
        out = []
        for i in range(n):
            d = _struct_dict(buf[i]); d.pop("reserved", None); out.append(d)
        return out

    def describe(self) -> str:
        buf = C.create_string_buffer(256)
        n = self._lib.atkdsp_undertow_describe(self._p, buf, 256)
        return buf.value.decode() if n > 0 else ""


__all__ += ["Undertow", "undertow_defaults", "t_log10_sf2", "UT_FLUCT", "UT_ACF", "UT_CYC",
            "UT_PERIOD", "UT_CONSISTENT", "UT_WARMUP"]


# ---- §30 Undertow pieces 2, 4, 6: psync, the bootstrap, the hit blob ----------
class _DsssLag(C.Structure):
    _fields_ = [("period", C.c_double), ("ratio", C.c_double), ("log10_pfa", C.c_double),
                ("lag", C.c_int32), ("reserved", C.c_int32)]


class _UndertowCand(C.Structure):
    _fields_ = [("fc_hz", C.c_double), ("bw_hz", C.c_double), ("fs_cell", C.c_double), ("period_s", C.c_double),
                ("ratio", C.c_double), ("log10_pfa", C.c_double), ("scale", C.c_int32), ("chan", C.c_int32),
                ("lag", C.c_int32), ("reserved", C.c_int32)]


class _PsyncCfg(C.Structure):
    _fields_ = [("fs", C.c_double), ("period_s", C.c_double), ("tol", C.c_double), ("rate_max", C.c_double),
                ("pfa", C.c_double), ("esc_pfa", C.c_double), ("n0", C.c_int32), ("n0_max", C.c_int32),
                ("nref", C.c_int32), ("max_periods", C.c_int32), ("reserved", C.c_int32 * 4)]


class _PsyncResult(C.Structure):
    _fields_ = [(k, C.c_double) for k in ("period_s", "period_samples", "rate", "lambda1", "lambda2", "ref_mean",
                                          "ref_sd", "z", "log10_pfa", "snr_db", "sigma2", "dominance",
                                          "window_offset")] + \
               [(k, C.c_int32) for k in ("nperiods", "n0_used", "levels", "evals", "window")] + \
               [("flags", C.c_uint32), ("method", C.c_int32 * 4)]


class _BootstrapCfg(C.Structure):
    _fields_ = [("ps", _PsyncCfg), ("bw_hz", C.c_double), ("rc_hz", C.c_double), ("chip_mod", C.c_int32),
                ("vote_passes", C.c_int32), ("poly_max_degree", C.c_int32), ("reserved0", C.c_int32),
                ("code_pfa", C.c_double), ("reserved", C.c_int32 * 4)]


class _BootstrapResult(C.Structure):
    _fields_ = [("ps", _PsyncResult), ("code", _DsssCodeResult), ("cfo_hz", C.c_double), ("rc_hz", C.c_double),
                ("fs_code", C.c_double), ("line_ratio", C.c_double), ("L", C.c_int32),
                ("support_start", C.c_int32), ("flags", C.c_uint32), ("method", C.c_int32 * 4),
                ("harmonic", C.c_int32)]


class _UndertowCtx(C.Structure):
    _fields_ = [("t_unix", C.c_double), ("rf_center_hz", C.c_double), ("fs_survey", C.c_double),
                ("source", C.c_char * 40)]


def _bind_psync(lib) -> None:
    P = C.POINTER
    vp, sz, i, d, ss = C.c_void_p, C.c_size_t, C.c_int, C.c_double, C.c_ssize_t
    lib.atkdsp_dsss_det_lags.argtypes = [vp, P(_DsssLag), sz]; lib.atkdsp_dsss_det_lags.restype = ss
    lib.atkdsp_undertow_candidates.argtypes = [vp, d, i, P(_UndertowCand), sz]
    lib.atkdsp_undertow_candidates.restype = ss
    lib.atkdsp_psync_defaults.argtypes = [P(_PsyncCfg), d, d]; lib.atkdsp_psync_defaults.restype = None
    lib.atkdsp_psync.argtypes = [P(_Cf32), sz, P(_PsyncCfg), P(_Cf32), sz, P(_PsyncResult)]
    lib.atkdsp_psync.restype = i
    lib.atkdsp_bootstrap_defaults.argtypes = [P(_BootstrapCfg), d, d, d]; lib.atkdsp_bootstrap_defaults.restype = None
    lib.atkdsp_bootstrap.argtypes = [P(_Cf32), sz, P(_BootstrapCfg), P(C.c_int8), sz, P(_BootstrapResult)]
    lib.atkdsp_bootstrap.restype = ss
    lib.atkdsp_undertow_hit_export.argtypes = [P(_UndertowCtx), P(_UndertowCell), P(_BootstrapResult),
                                               P(C.c_int8), sz, P(C.c_uint8), sz]
    lib.atkdsp_undertow_hit_export.restype = ss
    lib.atkdsp_undertow_hit_import.argtypes = [P(C.c_uint8), sz, P(_UndertowCtx), P(_UndertowCell),
                                               P(_BootstrapResult), P(C.c_int), P(C.c_int8), sz, P(sz)]
    lib.atkdsp_undertow_hit_import.restype = i


_BIND_EXTRA.append(_bind_psync)

PS_DETECT, PS_RATE, PS_EDGE, PS_BUDGET, PS_MULTI, PS_RECUT = 1, 2, 4, 8, 16, 32
BS_DETECT, BS_CODE, BS_REFUSED, BS_RC_GIVEN, BS_HARMONIC = 1, 2, 4, 8, 16
UT_BLOB_VERSION = 1


def psync_defaults(fs: float, period_s: float) -> dict:
    c = _PsyncCfg()
    load().atkdsp_psync_defaults(C.byref(c), float(fs), float(period_s))
    d = _struct_dict(c); d.pop("reserved", None)
    return d


def _fill(c, cfg, skip=()):
    for k, v in cfg.items():
        if k in skip or k == "reserved":
            continue
        if not hasattr(c, k):
            raise AtkDspError(f"unknown config key {k!r}")
        setattr(c, k, type(getattr(c, k))(v))


def _ps_dict(r) -> dict:
    return _struct_dict(r)


def psync(y, cfg: dict):
    """(result, wave) — binding of atkdsp_psync (§30). Twin: reference.psync."""
    lib = load()
    a = _cf32(y)
    c = _PsyncCfg()
    lib.atkdsp_psync_defaults(C.byref(c), float(cfg["fs"]), float(cfg["period_s"]))
    _fill(c, cfg)
    W = 2 * int(math.floor(c.period_s * c.fs + 0.5))
    wave = np.zeros(max(W, 1), np.complex64)
    r = _PsyncResult()
    _check(lib.atkdsp_psync(_cfp(a), a.size, C.byref(c), _cfp(wave), wave.size, C.byref(r)), "psync")
    return _ps_dict(r), wave[:r.window].copy()


def bootstrap_defaults(fs: float, period_s: float, bw_hz: float) -> dict:
    c = _BootstrapCfg()
    load().atkdsp_bootstrap_defaults(C.byref(c), float(fs), float(period_s), float(bw_hz))
    d = _struct_dict(c.ps); d.pop("reserved", None)
    for k in ("bw_hz", "rc_hz", "chip_mod", "vote_passes", "poly_max_degree", "code_pfa"):
        d[k] = getattr(c, k)
    return d


def _bs_dict(r) -> dict:
    d = {k: _struct_dict(getattr(r, k)) for k in ("ps", "code")}
    for n, _t in r._fields_:
        if n in ("ps", "code", "reserved"):
            continue
        v = getattr(r, n)
        d[n] = [int(x) for x in v] if isinstance(v, C.Array) else (int(v) if isinstance(v, int) else float(v))
    return d


def bootstrap(y, cfg: dict):
    """(chips, result) — binding of atkdsp_bootstrap (§30). Twin: reference.bootstrap."""
    lib = load()
    a = _cf32(y)
    c = _BootstrapCfg()
    lib.atkdsp_bootstrap_defaults(C.byref(c), float(cfg["fs"]), float(cfg["period_s"]), float(cfg.get("bw_hz", 0.0)))
    _fill(c.ps, {k: v for k, v in cfg.items() if hasattr(c.ps, k)})
    _fill(c, {k: v for k, v in cfg.items() if not hasattr(c.ps, k)})
    cap = 2 * int(math.floor(c.ps.period_s * c.ps.fs + 0.5)) + 2
    buf = (C.c_int8 * cap)()
    r = _BootstrapResult()
    n = lib.atkdsp_bootstrap(_cfp(a), a.size, C.byref(c), buf, cap, C.byref(r))
    _check(n if n >= 0 else int(n), "bootstrap")
    return np.ctypeslib.as_array(buf)[:n].copy(), _bs_dict(r)


def _cell_struct(hit: dict) -> "_UndertowCell":
    h = _UndertowCell()
    for n, _t in h._fields_:
        if n == "reserved" or n not in hit:
            continue
        v = hit[n]
        if n == "method":
            for i, m in enumerate(list(v)[:4]):
                h.method[i] = int(m)
        else:
            setattr(h, n, type(getattr(h, n))(v))
    return h


def _bs_struct(bs: dict) -> "_BootstrapResult":
    r = _BootstrapResult()
    for part in ("ps", "code"):
        st = getattr(r, part)
        for n, _t in st._fields_:
            if n not in (bs.get(part) or {}):
                continue
            v = bs[part][n]
            if isinstance(getattr(st, n), C.Array):
                for i, m in enumerate(list(v)[:len(getattr(st, n))]):
                    getattr(st, n)[i] = int(m)
            elif isinstance(getattr(st, n), bytes):
                setattr(st, n, str(v).encode()[:47])
            else:
                setattr(st, n, type(getattr(st, n))(v))
    for n, _t in r._fields_:
        if n in ("ps", "code", "reserved") or n not in bs:
            continue
        if n == "method":
            for i, m in enumerate(list(bs[n])[:4]):
                r.method[i] = int(m)
        else:
            setattr(r, n, type(getattr(r, n))(bs[n]))
    return r


def hit_export(ctx: dict, hit: dict, bs: dict | None = None, chips=None) -> bytes:
    """The versioned hit blob (atkdsp_undertow_hit_export, §30). Twin: reference.hit_export."""
    lib = load()
    cx = _UndertowCtx(float(ctx.get("t_unix", 0.0)), float(ctx.get("rf_center_hz", 0.0)),
                      float(ctx.get("fs_survey", 0.0)), str(ctx.get("source", "")).encode()[:39])
    h = _cell_struct(hit)
    b = _bs_struct(bs) if bs is not None else None
    ch = np.zeros(0, np.int8) if chips is None else np.ascontiguousarray(chips, np.int8).ravel()
    chp = ch.ctypes.data_as(C.POINTER(C.c_int8)) if ch.size else None
    bp = C.byref(b) if b is not None else None
    need = lib.atkdsp_undertow_hit_export(C.byref(cx), C.byref(h), bp, chp, ch.size, None, 0)
    _check(need if need >= 0 else int(need), "hit_export")
    out = (C.c_uint8 * need)()
    n = lib.atkdsp_undertow_hit_export(C.byref(cx), C.byref(h), bp, chp, ch.size, out, need)
    _check(n if n >= 0 else int(n), "hit_export")
    return bytes(out)


def hit_import(blob: bytes):
    """(ctx, hit, bs or None, chips) — atkdsp_undertow_hit_import; AtkDspError on
    a wrong magic, version, length or CRC."""
    lib = load()
    raw = (C.c_uint8 * len(blob)).from_buffer_copy(bytes(blob))
    cx, h, b = _UndertowCtx(), _UndertowCell(), _BootstrapResult()
    has, nch = C.c_int(0), C.c_size_t(0)
    cap = max(1, len(blob))
    chips = (C.c_int8 * cap)()
    rc = lib.atkdsp_undertow_hit_import(raw, len(blob), C.byref(cx), C.byref(h), C.byref(b), C.byref(has),
                                        chips, cap, C.byref(nch))
    _check(rc, "hit_import")
    ctx = dict(t_unix=cx.t_unix, rf_center_hz=cx.rf_center_hz, fs_survey=cx.fs_survey,
               source=cx.source.decode(errors="replace"))
    hit = _struct_dict(h); hit.pop("reserved", None)
    return ctx, hit, (_bs_dict(b) if has.value else None), np.ctypeslib.as_array(chips)[:nch.value].copy()


__all__ += ["psync_defaults", "psync", "bootstrap_defaults", "bootstrap", "hit_export", "hit_import",
            "PS_DETECT", "PS_RATE", "PS_EDGE", "PS_BUDGET", "PS_MULTI", "PS_RECUT",
            "BS_DETECT", "BS_CODE", "BS_REFUSED", "BS_RC_GIVEN", "BS_HARMONIC", "UT_BLOB_VERSION",
            "parity_z_frob", "CODE_DECIM", "CODE_REGEN"]



# ---- §31 Undertow piece 5: the full lag scan ------------------------------------
class _LagscanCfg(C.Structure):
    _fields_ = [("fs", C.c_double), ("pmin", C.c_int32), ("pmax", C.c_int32), ("nlags", C.c_int32),
                ("lag_min", C.c_int32), ("lag_frac", C.c_double), ("nmax", C.c_int32),
                ("reserved0", C.c_int32), ("pfa", C.c_double), ("trim", C.c_double),
                ("reserved", C.c_int32 * 4)]


class _LagscanRow(C.Structure):
    _fields_ = [(k, C.c_double) for k in ("period_samples", "period_s", "log10_pfa", "log10_pfa_raw",
                                          "stat", "shape", "scale", "excess")] + \
               [(k, C.c_int32) for k in ("base", "nfold", "harmonic_p", "harmonic_q", "family")] + \
               [("flags", C.c_uint32), ("method", C.c_int32 * 2)]


class _LagscanFamily(C.Structure):
    _fields_ = [(k, C.c_double) for k in ("period_samples", "period_s", "log10_pfa", "log10_step", "fold_snr")] + \
               [(k, C.c_int32) for k in ("seed_base", "seed_p", "seed_q", "members")] + \
               [("flags", C.c_uint32), ("method", C.c_int32 * 2), ("reserved", C.c_int32)]


def _bind_lagscan(lib) -> None:
    P = C.POINTER
    lib.atkdsp_lagscan_defaults.argtypes = [P(_LagscanCfg), C.c_double, C.c_int, C.c_int]
    lib.atkdsp_lagscan_defaults.restype = None
    lib.atkdsp_lagscan.argtypes = [P(_Cf32), C.c_size_t, P(_LagscanCfg), P(_LagscanRow), C.c_size_t,
                                   P(_LagscanFamily), C.c_size_t, P(C.c_size_t), P(C.c_double)]
    lib.atkdsp_lagscan.restype = C.c_ssize_t


_BIND_EXTRA.append(_bind_lagscan)

LAGSCAN_DETECT, LAGSCAN_HARMONIC, LAGSCAN_SHORT, LAGSCAN_FUNDAMENTAL = 1, 2, 4, 8
LAGSCAN_FAM_UP, LAGSCAN_FAM_DOWN, LAGSCAN_FAM_OUTSIDE = 1, 2, 4


def lagscan_defaults(fs: float, pmin: int = 16, pmax: int = 257) -> dict:
    c = _LagscanCfg()
    load().atkdsp_lagscan_defaults(C.byref(c), float(fs), int(pmin), int(pmax))
    d = _struct_dict(c)
    d.pop("reserved", None); d.pop("reserved0", None)
    return d


def lagscan(y, cfg: dict, fcap: int = 16):
    """(rows, families, ntrials) — binding of atkdsp_lagscan (§31): one row
    per base period, then up to fcap families, most significant first. Twin:
    reference.lagscan."""
    lib = load()
    a = _cf32(y)
    c = _LagscanCfg()
    lib.atkdsp_lagscan_defaults(C.byref(c), float(cfg["fs"]), int(cfg["pmin"]), int(cfg["pmax"]))
    _fill(c, {k: v for k, v in cfg.items() if k != "reserved0"})
    nrow = max(0, c.pmax - c.pmin)
    rows = (_LagscanRow * max(1, nrow))()
    fcap = max(0, int(fcap))
    fams = (_LagscanFamily * max(1, fcap))()
    nf = C.c_size_t(0)
    tot = C.c_double(0.0)
    n = lib.atkdsp_lagscan(_cfp(a), a.size, C.byref(c), rows, nrow, fams, fcap, C.byref(nf), C.byref(tot))
    _check(n if n >= 0 else int(n), "lagscan")
    out_f = []
    for i in range(nf.value):
        d = _struct_dict(fams[i])
        d.pop("reserved", None)
        out_f.append(d)
    return [_struct_dict(rows[i]) for i in range(n)], out_f, float(tot.value)


__all__ += ["lagscan_defaults", "lagscan", "LAGSCAN_DETECT", "LAGSCAN_HARMONIC", "LAGSCAN_SHORT",
            "LAGSCAN_FUNDAMENTAL", "LAGSCAN_FAM_UP", "LAGSCAN_FAM_DOWN", "LAGSCAN_FAM_OUTSIDE"]


# ============================================================================
# §32 — analog video: composite to pictures (docs/VIDEO_PLAN.md, §C6)
# ============================================================================
_vid_log = _logging.getLogger("atkdsp.video")


class _VideoConfig(C.Structure):
    _fields_ = [("fs", C.c_double), ("custom_line_hz", C.c_double),
                ("standard", C.c_int32), ("custom_lines", C.c_int32),
                ("polarity", C.c_int32), ("yc", C.c_int32), ("fields", C.c_int32),
                ("swap_fields", C.c_int32), ("width", C.c_int32), ("stack", C.c_int32),
                ("motion_aware", C.c_int32), ("reserved", C.c_int32 * 3)]


class _VideoMeta(C.Structure):
    _fields_ = [("sample0", C.c_int64), ("standard", C.c_int32), ("polarity", C.c_int32),
                ("progressive", C.c_int32), ("field_order", C.c_int32),
                ("width", C.c_int32), ("lines", C.c_int32),
                ("line_hz", C.c_double), ("lines_per_field", C.c_double),
                ("h_lock", C.c_float), ("snr_db", C.c_float),
                ("v_lock", C.c_int32), ("stacked", C.c_int32)]


_VIDEO_COUNTERS = ("samples_in", "lines_locked", "lines_coasted", "pulses_rejected", "fields",
                   "fields_rendered", "frames_out", "frames_dropped", "h_relocks", "v_relocks",
                   "polarity_flips")


class _VideoCounters(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in _VIDEO_COUNTERS]


def _bind_video(lib) -> None:
    P = C.POINTER
    vp, sz, i = C.c_void_p, C.c_size_t, C.c_int
    lib.atkdsp_video_defaults.argtypes = [P(_VideoConfig)]
    lib.atkdsp_video_defaults.restype = None
    lib.atkdsp_video_create.argtypes = [P(_VideoConfig)]
    lib.atkdsp_video_create.restype = vp
    lib.atkdsp_video_destroy.argtypes = [vp]
    lib.atkdsp_video_destroy.restype = None
    lib.atkdsp_video_reset.argtypes = [vp]
    lib.atkdsp_video_reset.restype = None
    lib.atkdsp_video_set.argtypes = [vp, P(_VideoConfig)]
    lib.atkdsp_video_set.restype = i
    lib.atkdsp_video_process.argtypes = [vp, P(C.c_float), sz]
    lib.atkdsp_video_process.restype = C.c_ssize_t
    lib.atkdsp_video_frame.argtypes = [vp, P(C.c_float), sz, P(_VideoMeta)]
    lib.atkdsp_video_frame.restype = i
    lib.atkdsp_video_describe.argtypes = [vp, C.c_char_p, sz]
    lib.atkdsp_video_describe.restype = i
    lib.atkdsp_video_stats.argtypes = [vp, P(_VideoCounters)]
    lib.atkdsp_video_stats.restype = i


_BIND_EXTRA.append(_bind_video)


def _video_cfg(cfg: dict, base: dict | None = None):
    c = dict(base) if base is not None else _ref.video_defaults()
    for key, v in cfg.items():
        if key not in c:
            raise AtkDspError(f"unknown video option {key!r}; known: {sorted(c)}")
        c[key] = v
    c = _ref._vid_norm(c)
    why = _ref._video_check(c)
    if why:
        raise AtkDspError(f"video refused — {why}")
    k = _VideoConfig()
    for key, v in c.items():
        setattr(k, key, v)
    return c, k


def _video_meta(m) -> dict:
    return {"sample0": int(m.sample0), "standard": int(m.standard), "polarity": int(m.polarity),
            "progressive": int(m.progressive), "field_order": int(m.field_order),
            "width": int(m.width), "lines": int(m.lines), "line_hz": float(m.line_hz),
            "lines_per_field": float(m.lines_per_field), "h_lock": float(m.h_lock),
            "v_lock": int(m.v_lock), "snr_db": float(m.snr_db), "stacked": int(m.stacked)}


class Video:
    """Analog video: composite samples in, luma frames out (atkdsp.h §32).
    Options are the atkdsp_video_config fields as keywords (the twin's
    reference.video_defaults() lists them). process() returns the number of
    frames waiting; frame() returns (pixels H x W float32, meta dict) or
    None. The twin, reference.Video, is ATK's fallback: same answers, file
    speed only."""

    _kind = "video"

    def __init__(self, **cfg):
        self._lib = load()
        self.cfg, k = _video_cfg(cfg)
        self._h = self._lib.atkdsp_video_create(C.byref(k))
        if not self._h:
            raise AtkDspError(f"video_create({self.cfg}) refused (out of memory?)")
        self._ncalls = 0
        self._relocks = (0, 0)
        _vid_log.info("video created: %s", self.describe())

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.atkdsp_video_destroy(h)

    def reset(self) -> None:
        self._lib.atkdsp_video_reset(self._h)
        _vid_log.info("video reset")

    def set(self, **cfg) -> None:
        """New settings from the next segment on (display settings keep lock;
        a new standard, custom timing or polarity re-acquires)."""
        c, k = _video_cfg(cfg, self.cfg)
        if c["fs"] != self.cfg["fs"]:
            raise AtkDspError("video set refused — fs cannot change (create a new one)")
        rc = self._lib.atkdsp_video_set(self._h, C.byref(k))
        if rc != 0:
            _vid_log.warning("video_set(%s) returned %d", c, rc)
        _check(rc, "video_set")
        self.cfg = c
        _vid_log.info("video settings: %s", c)

    def process(self, x) -> int:
        a = _f32(x)
        n = self._lib.atkdsp_video_process(self._h, _fp(a), a.size)
        if n < 0:
            _vid_log.warning("video_process(%d samples) returned %d", a.size, n)
        _check(n, "video_process")
        self._ncalls += 1
        st = self.stats()
        rel = (st["h_relocks"], st["v_relocks"])
        if rel != self._relocks:                       # a lock change: say what it is now
            self._relocks = rel
            _vid_log.info("video: %s", self.describe())
        if self._ncalls % 1000 == 0:
            _vid_log.debug("video stats: %s", st)
        return int(n)

    def frame(self):
        m = _VideoMeta()
        rc = self._lib.atkdsp_video_frame(self._h, None, 0, C.byref(m))
        if rc == 0:
            return None
        if rc != -3:                                    # ATKDSP_E_CAP is the expected answer
            _vid_log.warning("video_frame(peek) returned %d", rc)
            _check(rc, "video_frame")
        pix = np.empty((m.lines, m.width), dtype=np.float32)
        rc = self._lib.atkdsp_video_frame(self._h, _fp(pix.ravel()),
                                          pix.size, C.byref(m))
        if rc != 1:
            _vid_log.warning("video_frame returned %d", rc)
            _check(rc, "video_frame")
        return pix, _video_meta(m)

    def describe(self) -> str:
        buf = C.create_string_buffer(512)
        self._lib.atkdsp_video_describe(self._h, buf, 512)
        return buf.value.decode()

    def stats(self) -> dict:
        s = _VideoCounters()
        self._lib.atkdsp_video_stats(self._h, C.byref(s))
        return {n: int(getattr(s, n)) for n in _VIDEO_COUNTERS}


__all__ += ["Video"]
