"""Signal generation, containment, and CSV I/O.

The :class:`Signal` dataclass bundles a set of uniformly sampled amplitudes
with their time stamps and sampling rate, and knows how to round-trip itself
through a small, self-describing CSV format. The module-level generators
(``sine``, ``square``, ``mixed``, ...) all return :class:`Signal` instances so
the rest of the toolkit has a single, consistent representation to work with.

All waveforms are parameterised the way a DSP practitioner expects: by
frequency in Hz, amplitude, and phase in radians, sampled at ``sample_rate``
Hz for ``duration`` seconds.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .errors import InvalidParameterError, InvalidSignalError

__all__ = [
    "Signal",
    "time_axis",
    "sine",
    "cosine",
    "square",
    "sawtooth",
    "triangle",
    "mixed",
    "add_noise",
    "generate",
    "WAVEFORMS",
]

#: Names of the single-tone periodic waveforms this module can synthesize.
WAVEFORMS: tuple[str, ...] = ("sine", "cosine", "square", "sawtooth", "triangle")


@dataclass
class Signal:
    """A uniformly sampled, real (or complex) 1-D signal.

    Attributes
    ----------
    t:
        Sample time stamps in seconds (shape ``(N,)``).
    y:
        Sample values (shape ``(N,)``). Usually real, but complex is allowed so
        the container can also hold, e.g., an inverse-transform result.
    sample_rate:
        Sampling frequency in Hz.
    name:
        Optional human-readable label used in plots and console output.
    """

    t: NDArray[np.float64]
    y: NDArray[np.float64]
    sample_rate: float
    name: str = "signal"

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=np.float64)
        self.y = np.asarray(self.y)
        if self.t.ndim != 1 or self.y.ndim != 1:
            raise InvalidSignalError("Signal t and y must be one-dimensional")
        if self.t.shape != self.y.shape:
            raise InvalidSignalError(
                f"t and y must have the same length, got {self.t.shape} and "
                f"{self.y.shape}"
            )
        if self.t.size == 0:
            raise InvalidSignalError("Signal must contain at least one sample")
        if self.sample_rate <= 0:
            raise InvalidParameterError("sample_rate must be positive")

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def n(self) -> int:
        """Number of samples."""
        return int(self.t.size)

    @property
    def duration(self) -> float:
        """Total duration in seconds (``N / sample_rate``)."""
        return self.n / self.sample_rate

    @property
    def dt(self) -> float:
        """Sample spacing in seconds (``1 / sample_rate``)."""
        return 1.0 / self.sample_rate

    def to_csv(self, path: str | Path) -> Path:
        """Write the signal to ``path`` as a two-column ``t,y`` CSV.

        A ``# sample_rate=...`` comment is written first so the rate survives
        exactly, even if the time column suffers floating-point drift.
        """
        path = Path(path)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            fh.write(f"# FourierLab signal '{self.name}'\n")
            fh.write(f"# sample_rate={self.sample_rate!r}\n")
            writer = csv.writer(fh)
            writer.writerow(["t", "y"])
            for ti, yi in zip(self.t, self.y):
                writer.writerow([repr(float(ti)), repr(float(np.real(yi)))])
        return path

    @classmethod
    def from_csv(cls, path: str | Path) -> "Signal":
        """Load a signal written by :meth:`to_csv` (or any ``t,y`` CSV).

        The sampling rate is read from the ``# sample_rate=`` comment when
        present; otherwise it is inferred from the median spacing of the time
        column.
        """
        path = Path(path)
        if not path.exists():
            raise InvalidSignalError(f"signal file not found: {path}")

        sample_rate: float | None = None
        name = path.stem
        t_vals: list[float] = []
        y_vals: list[float] = []

        with path.open("r", newline="") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    if "sample_rate=" in line:
                        try:
                            sample_rate = float(line.split("sample_rate=")[1].strip())
                        except (IndexError, ValueError):
                            sample_rate = None
                    continue
                parts = [p.strip() for p in line.split(",")]
                # Skip a header row such as "t,y".
                try:
                    ti = float(parts[0])
                    yi = float(parts[1])
                except (ValueError, IndexError):
                    continue
                t_vals.append(ti)
                y_vals.append(yi)

        if not t_vals:
            raise InvalidSignalError(f"no numeric samples found in {path}")

        t = np.asarray(t_vals, dtype=np.float64)
        y = np.asarray(y_vals, dtype=np.float64)

        if sample_rate is None:
            if t.size >= 2:
                dt = float(np.median(np.diff(t)))
                if dt <= 0:
                    raise InvalidSignalError(
                        f"cannot infer sample rate from non-increasing time column "
                        f"in {path}"
                    )
                sample_rate = 1.0 / dt
            else:
                sample_rate = 1.0
        return cls(t=t, y=y, sample_rate=sample_rate, name=name)


def time_axis(sample_rate: float, duration: float) -> NDArray[np.float64]:
    """Return evenly spaced sample times for ``duration`` seconds at ``sample_rate`` Hz.

    Uses ``t = arange(N) / sample_rate`` so the spacing is exactly ``1/sample_rate``
    and the first sample sits at ``t = 0``.
    """
    if sample_rate <= 0:
        raise InvalidParameterError("sample_rate must be positive")
    if duration <= 0:
        raise InvalidParameterError("duration must be positive")
    n = int(round(duration * sample_rate))
    if n < 1:
        raise InvalidParameterError(
            "duration * sample_rate rounds to zero samples; increase either value"
        )
    return np.arange(n, dtype=np.float64) / sample_rate


def _theta(
    t: NDArray[np.float64], frequency: float, phase: float
) -> NDArray[np.float64]:
    """Instantaneous phase angle ``2*pi*f*t + phase`` for each sample time."""
    return 2.0 * np.pi * frequency * t + phase


def sine(
    frequency: float = 1.0,
    sample_rate: float = 256.0,
    duration: float = 1.0,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> Signal:
    """Generate ``A*sin(2*pi*f*t + phase)``."""
    t = time_axis(sample_rate, duration)
    y = amplitude * np.sin(_theta(t, frequency, phase))
    return Signal(t, y, sample_rate, name=f"sine {frequency:g} Hz")


def cosine(
    frequency: float = 1.0,
    sample_rate: float = 256.0,
    duration: float = 1.0,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> Signal:
    """Generate ``A*cos(2*pi*f*t + phase)``."""
    t = time_axis(sample_rate, duration)
    y = amplitude * np.cos(_theta(t, frequency, phase))
    return Signal(t, y, sample_rate, name=f"cosine {frequency:g} Hz")


def square(
    frequency: float = 1.0,
    sample_rate: float = 256.0,
    duration: float = 1.0,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> Signal:
    """Generate a ``+/-A`` square wave via the sign of a sine.

    The value at an exact zero crossing (``sign(0) == 0``) is nudged to ``+A``
    so the output is strictly two-valued.
    """
    t = time_axis(sample_rate, duration)
    s = np.sign(np.sin(_theta(t, frequency, phase)))
    s[s == 0] = 1.0
    y = amplitude * s
    return Signal(t, y, sample_rate, name=f"square {frequency:g} Hz")


def sawtooth(
    frequency: float = 1.0,
    sample_rate: float = 256.0,
    duration: float = 1.0,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> Signal:
    """Generate a rising sawtooth ramping linearly from ``-A`` to ``+A`` each period."""
    t = time_axis(sample_rate, duration)
    frac = np.mod(frequency * t + phase / (2.0 * np.pi), 1.0)
    y = amplitude * (2.0 * frac - 1.0)
    return Signal(t, y, sample_rate, name=f"sawtooth {frequency:g} Hz")


def triangle(
    frequency: float = 1.0,
    sample_rate: float = 256.0,
    duration: float = 1.0,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> Signal:
    """Generate a ``+/-A`` triangle wave.

    Built from ``(2/pi)*arcsin(sin(theta))``, which is a unit triangle wave with
    the same period and phase as ``sin(theta)`` but without any modulo edge cases.
    """
    t = time_axis(sample_rate, duration)
    y = amplitude * (2.0 / np.pi) * np.arcsin(np.sin(_theta(t, frequency, phase)))
    return Signal(t, y, sample_rate, name=f"triangle {frequency:g} Hz")


_GENERATORS = {
    "sine": sine,
    "cosine": cosine,
    "square": square,
    "sawtooth": sawtooth,
    "triangle": triangle,
}


def mixed(
    frequencies: Sequence[float],
    amplitudes: Sequence[float] | None = None,
    phases: Sequence[float] | None = None,
    sample_rate: float = 256.0,
    duration: float = 1.0,
    component: str = "sine",
) -> Signal:
    """Sum several single-tone waveforms into one signal.

    Parameters
    ----------
    frequencies:
        Component frequencies in Hz.
    amplitudes:
        Per-component amplitudes. Defaults to all ones.
    phases:
        Per-component phases in radians. Defaults to all zeros.
    component:
        Which waveform each component is (``"sine"``, ``"cosine"``, ...).
    """
    frequencies = list(frequencies)
    if not frequencies:
        raise InvalidParameterError("mixed signal needs at least one frequency")
    if component not in _GENERATORS:
        raise InvalidParameterError(
            f"unknown component waveform {component!r}; choose from {WAVEFORMS}"
        )

    if amplitudes is None:
        amplitudes = [1.0] * len(frequencies)
    if phases is None:
        phases = [0.0] * len(frequencies)
    if len(amplitudes) != len(frequencies):
        raise InvalidParameterError(
            f"got {len(frequencies)} frequencies but {len(amplitudes)} amplitudes"
        )
    if len(phases) != len(frequencies):
        raise InvalidParameterError(
            f"got {len(frequencies)} frequencies but {len(phases)} phases"
        )

    gen = _GENERATORS[component]
    t = time_axis(sample_rate, duration)
    y = np.zeros_like(t)
    for f, a, p in zip(frequencies, amplitudes, phases):
        y = y + gen(f, sample_rate, duration, a, p).y
    label = "+".join(f"{f:g}Hz" for f in frequencies)
    return Signal(t, y, sample_rate, name=f"mixed({label})")


def add_noise(
    signal: Signal, noise_level: float, seed: int | None = None
) -> Signal:
    """Return a copy of ``signal`` with additive white Gaussian noise.

    Parameters
    ----------
    noise_level:
        Standard deviation of the Gaussian noise, in signal units.
    seed:
        Seed for the random generator, for reproducible output.
    """
    if noise_level < 0:
        raise InvalidParameterError("noise_level must be non-negative")
    if noise_level == 0:
        return Signal(signal.t.copy(), signal.y.copy(), signal.sample_rate, signal.name)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_level, size=signal.n)
    return Signal(
        signal.t.copy(),
        np.real(signal.y) + noise,
        signal.sample_rate,
        name=f"{signal.name}+noise",
    )


def generate(
    kind: str,
    *,
    frequency: float = 5.0,
    sample_rate: float = 256.0,
    duration: float = 1.0,
    amplitude: float = 1.0,
    phase: float = 0.0,
    frequencies: Sequence[float] | None = None,
    amplitudes: Sequence[float] | None = None,
    phases: Sequence[float] | None = None,
    noise: float = 0.0,
    seed: int | None = None,
) -> Signal:
    """High-level dispatcher used by the CLI ``generate`` command.

    ``kind`` is either one of :data:`WAVEFORMS` (a single tone described by
    ``frequency``/``amplitude``/``phase``) or ``"mixed"`` (a sum described by
    the plural ``frequencies``/``amplitudes``/``phases``). Gaussian noise of
    standard deviation ``noise`` is added afterwards when ``noise > 0``.
    """
    if kind == "mixed":
        if not frequencies:
            raise InvalidParameterError(
                "kind='mixed' requires a non-empty 'frequencies' list"
            )
        base = mixed(
            frequencies=frequencies,
            amplitudes=amplitudes,
            phases=phases,
            sample_rate=sample_rate,
            duration=duration,
        )
    elif kind in _GENERATORS:
        base = _GENERATORS[kind](
            frequency, sample_rate, duration, amplitude, phase
        )
    else:
        raise InvalidParameterError(
            f"unknown signal kind {kind!r}; choose from {WAVEFORMS + ('mixed',)}"
        )

    if noise > 0:
        base = add_noise(base, noise, seed=seed)
    return base
