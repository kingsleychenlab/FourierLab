"""Small, dependency-light mathematical helpers shared across FourierLab.

Nothing in here touches :mod:`numpy.fft`; the transforms themselves live in
:mod:`fourierlab.dft` and :mod:`fourierlab.fft`. This module only holds the
supporting utilities: power-of-two checks, frequency-bin bookkeeping, spectrum
conversions, and error metrics.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "is_power_of_two",
    "next_power_of_two",
    "fft_freqs",
    "positive_freq_mask",
    "magnitude_spectrum",
    "phase_spectrum",
    "amplitude_spectrum",
    "dominant_frequencies",
    "mse",
    "max_error",
    "parseval",
    "ParsevalResult",
    "Peak",
]


def is_power_of_two(n: int) -> bool:
    """Return ``True`` iff ``n`` is a positive integer power of two.

    Uses the classic bit trick ``n & (n - 1) == 0``, which is only valid for
    ``n > 0``.
    """
    return isinstance(n, (int, np.integer)) and n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    """Return the smallest power of two that is >= ``n`` (and at least 1)."""
    if n < 1:
        return 1
    power = 1
    while power < n:
        power <<= 1
    return power


def fft_freqs(n: int, sample_rate: float = 1.0) -> NDArray[np.float64]:
    """Return the DFT sample frequencies, in Hz, in standard FFT bin order.

    The returned array matches ``numpy.fft.fftfreq(n, d=1/sample_rate)`` but is
    computed here directly so the package does not lean on NumPy's FFT module.

    Bins ``0 .. n//2 - 1`` hold non-negative frequencies; the remaining bins
    hold the negative frequencies, reflecting the conjugate symmetry of a real
    signal's spectrum.

    Parameters
    ----------
    n:
        Number of samples / DFT bins. Must be positive.
    sample_rate:
        Sampling frequency in Hz. With the default of 1.0 the frequencies are
        expressed in cycles-per-sample.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    val = sample_rate / n
    results = np.empty(n, dtype=np.float64)
    half = (n - 1) // 2 + 1
    results[:half] = np.arange(0, half)
    results[half:] = np.arange(-(n // 2), 0)
    return results * val


def positive_freq_mask(n: int) -> NDArray[np.bool_]:
    """Boolean mask selecting the non-negative-frequency half of a spectrum.

    Includes DC (bin 0) and, for even ``n``, the Nyquist bin (``n // 2``).
    """
    mask = np.zeros(n, dtype=bool)
    mask[: n // 2 + 1] = True
    return mask


def magnitude_spectrum(spectrum: ArrayLike) -> NDArray[np.float64]:
    """Return ``|X_k|`` for each complex coefficient in ``spectrum``."""
    return np.abs(np.asarray(spectrum, dtype=complex))


def phase_spectrum(
    spectrum: ArrayLike, magnitude_threshold: float = 1e-10
) -> NDArray[np.float64]:
    """Return the phase angle (radians) of each coefficient in ``spectrum``.

    Coefficients whose magnitude is below ``magnitude_threshold`` are reported
    as phase ``0``. Their "true" phase is dominated by floating-point noise, so
    zeroing it produces far more readable phase plots and console output.
    """
    spectrum = np.asarray(spectrum, dtype=complex)
    phase = np.angle(spectrum)
    if magnitude_threshold > 0:
        phase = np.where(np.abs(spectrum) < magnitude_threshold, 0.0, phase)
    return phase


def amplitude_spectrum(
    spectrum: ArrayLike, n: int | None = None
) -> NDArray[np.float64]:
    """Return the single-sided amplitude spectrum of a real signal.

    Given the full complex DFT of a length-``N`` real signal, this returns the
    amplitude of each cosine component for the non-negative frequencies,
    scaled so that a pure tone ``A*cos(2*pi*f*t)`` reads back with height
    ``A``. Concretely, with ``X`` the DFT:

    * DC term:      ``|X_0| / N``
    * middle bins:  ``2 * |X_k| / N``   (energy is split between +f and -f)
    * Nyquist bin:  ``|X_{N/2}| / N``   (only present, and not doubled, for even N)

    Parameters
    ----------
    spectrum:
        Full complex DFT coefficients (length ``N``).
    n:
        Length of the original signal. Defaults to ``len(spectrum)``.
    """
    spectrum = np.asarray(spectrum, dtype=complex)
    if n is None:
        n = spectrum.shape[0]
    half = n // 2 + 1
    amp = np.abs(spectrum[:half]) * (2.0 / n)
    amp[0] = np.abs(spectrum[0]) / n  # DC is not doubled
    if n % 2 == 0:
        amp[-1] = np.abs(spectrum[n // 2]) / n  # Nyquist is not doubled
    return amp


class Peak(NamedTuple):
    """A single dominant spectral component."""

    frequency: float
    amplitude: float


def dominant_frequencies(
    spectrum: ArrayLike,
    sample_rate: float = 1.0,
    count: int = 3,
    min_amplitude: float = 1e-6,
) -> list[Peak]:
    """Return the strongest positive-frequency components of a spectrum.

    Peaks are ranked by single-sided amplitude (see :func:`amplitude_spectrum`),
    so the reported amplitude is directly comparable to the amplitude of the
    sinusoid that produced it.

    Parameters
    ----------
    spectrum:
        Full complex DFT coefficients.
    sample_rate:
        Sampling frequency in Hz.
    count:
        Maximum number of peaks to return.
    min_amplitude:
        Components weaker than this are ignored (filters out numerical dust).
    """
    spectrum = np.asarray(spectrum, dtype=complex)
    n = spectrum.shape[0]
    amp = amplitude_spectrum(spectrum, n)
    freqs = fft_freqs(n, sample_rate)[: amp.shape[0]]

    order = np.argsort(amp)[::-1]
    peaks: list[Peak] = []
    for idx in order:
        if amp[idx] < min_amplitude:
            break
        peaks.append(Peak(frequency=float(freqs[idx]), amplitude=float(amp[idx])))
        if len(peaks) >= count:
            break
    return peaks


def mse(a: ArrayLike, b: ArrayLike) -> float:
    """Mean squared error between two equal-length signals.

    Works for real or complex input; ``|.|**2`` is used so the result is a
    non-negative real number.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean(np.abs(a - b) ** 2))


def max_error(a: ArrayLike, b: ArrayLike) -> float:
    """Maximum absolute (L-infinity) error between two equal-length signals."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


class ParsevalResult(NamedTuple):
    """Both sides of Parseval's identity plus their relative discrepancy."""

    time_energy: float       # sum |x_n|^2
    freq_energy: float       # (1/N) sum |X_k|^2
    relative_error: float    # |time - freq| / max(time, tiny)


def parseval(signal: ArrayLike, spectrum: ArrayLike) -> ParsevalResult:
    """Evaluate Parseval's theorem for a signal/spectrum pair.

    Parseval's theorem for the DFT states::

        sum_n |x_n|^2  ==  (1 / N) * sum_k |X_k|^2

    i.e. energy is conserved between the time and frequency domains. The
    returned :class:`ParsevalResult` reports both sides and the relative error,
    which should be at the level of floating-point round-off for a correct
    transform.
    """
    x = np.asarray(signal, dtype=complex)
    X = np.asarray(spectrum, dtype=complex)
    n = X.shape[0]
    time_energy = float(np.sum(np.abs(x) ** 2))
    freq_energy = float(np.sum(np.abs(X) ** 2) / n)
    denom = max(abs(time_energy), 1e-300)
    rel = abs(time_energy - freq_energy) / denom
    return ParsevalResult(time_energy, freq_energy, rel)
