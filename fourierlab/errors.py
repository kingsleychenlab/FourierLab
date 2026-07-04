"""Custom exception hierarchy for FourierLab.

All exceptions raised deliberately by the library derive from
:class:`FourierLabError`, so callers (including the CLI) can catch the whole
family with a single ``except FourierLabError`` and print a clean message
instead of a traceback.
"""

from __future__ import annotations


class FourierLabError(Exception):
    """Base class for every error raised on purpose by FourierLab."""


class InvalidSignalError(FourierLabError):
    """Raised when an input signal is empty, malformed, or the wrong shape."""


class NotPowerOfTwoError(FourierLabError):
    """Raised when a radix-2 FFT receives a length that is not a power of two."""

    def __init__(self, length: int) -> None:
        self.length = length
        super().__init__(
            f"radix-2 FFT requires the input length to be a power of two, "
            f"but got N={length}. Pad or truncate the signal to a length such "
            f"as {_next_power_of_two_hint(length)}, or use the DFT instead."
        )


class InvalidCutoffError(FourierLabError):
    """Raised when a filter is given a nonsensical cutoff frequency."""


class InvalidFunctionError(FourierLabError):
    """Raised when an unknown waveform / function name is requested."""


class InvalidParameterError(FourierLabError):
    """Raised when a numeric parameter (duration, sample rate, ...) is invalid."""


def _next_power_of_two_hint(n: int) -> int:
    """Return the next power of two >= ``n`` (>= 1), used only for messages."""
    if n < 1:
        return 1
    power = 1
    while power < n:
        power <<= 1
    return power
