"""FourierLab: a from-scratch computational toolkit for Fourier analysis.

FourierLab implements the Discrete Fourier Transform and the Cooley-Tukey Fast
Fourier Transform directly from their definitions (no ``numpy.fft`` in the core),
along with real Fourier series, signal generation, spectrum analysis, and
frequency-domain filtering. It is designed for students of Fourier analysis,
numerical methods, and signal processing.

Typical use::

    import numpy as np
    from fourierlab import fft, ifft, dft, idft
    from fourierlab.signals import sine
    from fourierlab.series import approximate

    x = sine(frequency=5, sample_rate=128, duration=1).y
    X = fft(x)                 # forward transform
    x_back = ifft(X).real      # exact reconstruction

The command-line interface exposes the same functionality; see
``python -m fourierlab --help``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .dft import dft, idft
from .fft import fft, forward_transform, ifft, inverse_transform
from .filters import apply_filter
from .math_utils import (
    amplitude_spectrum,
    dominant_frequencies,
    fft_freqs,
    is_power_of_two,
    magnitude_spectrum,
    parseval,
    phase_spectrum,
)
from .series import FourierSeries, approximate
from .signals import Signal, generate

__all__ = [
    "__version__",
    # transforms
    "dft",
    "idft",
    "fft",
    "ifft",
    "forward_transform",
    "inverse_transform",
    # spectra
    "magnitude_spectrum",
    "phase_spectrum",
    "amplitude_spectrum",
    "dominant_frequencies",
    "fft_freqs",
    "is_power_of_two",
    "parseval",
    # series
    "FourierSeries",
    "approximate",
    # signals & filters
    "Signal",
    "generate",
    "apply_filter",
]
