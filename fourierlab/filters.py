"""Ideal frequency-domain filters built on the from-scratch transforms.

Each filter follows the same three-step recipe:

1. Transform the signal to the frequency domain (FFT when the length is a power
   of two, otherwise the direct DFT -- see
   :func:`fourierlab.fft.forward_transform`).
2. Zero out the DFT bins whose frequency falls in the reject band.
3. Transform back with the inverse to recover the filtered time-domain signal.

Because the reject decision is made on the *absolute* frequency ``|f|``, the
mask is automatically symmetric between each bin ``k`` and its conjugate partner
``N-k``. That symmetry is what guarantees the reconstructed signal is real (up
to round-off) for a real input. These are "brick-wall" ideal filters: perfectly
flat pass/stop bands and an infinitely sharp edge. They are ideal for teaching;
real-time DSP typically uses smoother filters to avoid ringing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .errors import InvalidCutoffError
from .fft import forward_transform, inverse_transform
from .math_utils import fft_freqs
from .signals import Signal

__all__ = [
    "FILTER_TYPES",
    "FilterResult",
    "frequency_mask",
    "apply_filter",
]

#: Supported filter type names.
FILTER_TYPES: tuple[str, ...] = ("lowpass", "highpass", "bandpass", "bandstop")

# Frequencies are compared with a tiny tolerance so a component sitting exactly
# on a bin edge is classified deterministically.
_TOL = 1e-9


class _Band(NamedTuple):
    cutoff: float | None
    low: float | None
    high: float | None


def _validate(
    filter_type: str,
    cutoff: float | None,
    low: float | None,
    high: float | None,
    nyquist: float,
) -> _Band:
    """Validate filter parameters and normalise them into a :class:`_Band`."""
    if filter_type not in FILTER_TYPES:
        raise InvalidCutoffError(
            f"unknown filter type {filter_type!r}; choose from {FILTER_TYPES}"
        )

    if filter_type in ("lowpass", "highpass"):
        if cutoff is None:
            raise InvalidCutoffError(f"{filter_type} requires a --cutoff value")
        if cutoff <= 0:
            raise InvalidCutoffError(
                f"cutoff must be positive, got {cutoff}"
            )
        if cutoff > nyquist + _TOL:
            raise InvalidCutoffError(
                f"cutoff {cutoff} Hz exceeds the Nyquist frequency "
                f"{nyquist:g} Hz; no frequencies above Nyquist exist to filter"
            )
        return _Band(cutoff=float(cutoff), low=None, high=None)

    # bandpass / bandstop
    if low is None or high is None:
        raise InvalidCutoffError(
            f"{filter_type} requires both --low and --high band edges"
        )
    if low < 0 or high <= 0:
        raise InvalidCutoffError("band edges must be positive")
    if low >= high:
        raise InvalidCutoffError(
            f"low edge ({low}) must be strictly less than high edge ({high})"
        )
    if high > nyquist + _TOL:
        raise InvalidCutoffError(
            f"high edge {high} Hz exceeds the Nyquist frequency {nyquist:g} Hz"
        )
    return _Band(cutoff=None, low=float(low), high=float(high))


def frequency_mask(
    freqs: NDArray[np.float64],
    filter_type: str,
    band: _Band,
) -> NDArray[np.bool_]:
    """Return a boolean *keep* mask over DFT bins for the given filter.

    ``freqs`` are the (signed) bin frequencies; the decision is made on their
    absolute value so the mask is conjugate-symmetric.
    """
    absf = np.abs(freqs)
    if filter_type == "lowpass":
        return absf <= band.cutoff + _TOL
    if filter_type == "highpass":
        return absf >= band.cutoff - _TOL
    if filter_type == "bandpass":
        return (absf >= band.low - _TOL) & (absf <= band.high + _TOL)
    # bandstop
    return (absf < band.low - _TOL) | (absf > band.high + _TOL)


@dataclass
class FilterResult:
    """Everything produced by applying a filter, ready for reporting/plotting."""

    original: Signal
    filtered: Signal
    filter_type: str
    band: _Band
    keep_mask: NDArray[np.bool_]
    freqs: NDArray[np.float64]
    spectrum: NDArray[np.complex128]
    filtered_spectrum: NDArray[np.complex128]

    @property
    def removed_count(self) -> int:
        """Number of DFT bins that were zeroed out (rejected)."""
        return int(np.count_nonzero(~self.keep_mask))

    @property
    def kept_count(self) -> int:
        """Number of DFT bins that were kept."""
        return int(np.count_nonzero(self.keep_mask))


def apply_filter(
    signal: Signal,
    filter_type: str,
    cutoff: float | None = None,
    low: float | None = None,
    high: float | None = None,
) -> FilterResult:
    """Apply an ideal frequency-domain filter to ``signal``.

    Parameters
    ----------
    signal:
        Input :class:`~fourierlab.signals.Signal`.
    filter_type:
        One of :data:`FILTER_TYPES`.
    cutoff:
        Cutoff frequency in Hz for ``lowpass`` / ``highpass``.
    low, high:
        Band edges in Hz for ``bandpass`` / ``bandstop``.

    Returns
    -------
    FilterResult
        The filtered signal (same length and sample rate as the input) plus the
        spectra and mask used, for inspection or plotting.

    Raises
    ------
    InvalidCutoffError
        If the filter type or cutoff/band parameters are invalid.
    """
    nyquist = signal.sample_rate / 2.0
    band = _validate(filter_type, cutoff, low, high, nyquist)

    y = np.real(signal.y).astype(float)
    spectrum = forward_transform(y)
    freqs = fft_freqs(signal.n, signal.sample_rate)

    keep_mask = frequency_mask(freqs, filter_type, band)
    filtered_spectrum = spectrum * keep_mask
    filtered_y = np.real(inverse_transform(filtered_spectrum))

    filtered_signal = Signal(
        t=signal.t.copy(),
        y=filtered_y,
        sample_rate=signal.sample_rate,
        name=f"{signal.name} [{filter_type}]",
    )
    return FilterResult(
        original=signal,
        filtered=filtered_signal,
        filter_type=filter_type,
        band=band,
        keep_mask=keep_mask,
        freqs=freqs,
        spectrum=spectrum,
        filtered_spectrum=filtered_spectrum,
    )
