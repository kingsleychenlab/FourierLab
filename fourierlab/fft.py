"""Recursive Cooley-Tukey FFT, implemented from scratch.

The radix-2 decimation-in-time FFT rewrites a length-``N`` DFT (``N`` a power of
two) in terms of two length-``N/2`` DFTs, one over the even-indexed samples and
one over the odd-indexed samples:

.. math::

    X_k       &= E_k + W_N^{\\,k}\\, O_k \\\\
    X_{k+N/2} &= E_k - W_N^{\\,k}\\, O_k,
    \\qquad W_N^{\\,k} = e^{-2\\pi i k / N}

where :math:`E` is the DFT of the even samples and :math:`O` the DFT of the odd
samples. Applying this recursively gives the familiar :math:`O(N \\log N)` cost,
versus :math:`O(N^2)` for the direct DFT in :mod:`fourierlab.dft`.

Implementation note
-------------------
The recursion bottoms out at :data:`_BASE_CASE_SIZE` samples and finishes with a
direct DFT of that small block, rather than recursing all the way down to
single samples. This is a standard, mathematically identical optimisation: it
removes the great majority of Python-level recursive calls, which is what
otherwise makes a pure-Python FFT lose to NumPy's vectorised matrix DFT for
moderate ``N``. Set ``_BASE_CASE_SIZE = 1`` to recover the pure textbook
recursion ``X_k = E_k + W_N^k O_k`` with a scalar base case.
"""

from __future__ import annotations

from time import perf_counter
from typing import NamedTuple, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .dft import dft, dft_matrix, idft
from .errors import InvalidSignalError, NotPowerOfTwoError
from .math_utils import is_power_of_two

__all__ = [
    "fft",
    "ifft",
    "forward_transform",
    "inverse_transform",
    "benchmark",
    "TimingResult",
]

#: Recursion stops at or below this size and finishes with a direct DFT.
_BASE_CASE_SIZE = 32


def _as_1d(x: ArrayLike) -> NDArray[np.complex128]:
    """Validate and coerce ``x`` into a 1-D complex array."""
    arr = np.asarray(x, dtype=complex)
    if arr.ndim != 1:
        raise InvalidSignalError(
            f"expected a one-dimensional signal, got shape {arr.shape}"
        )
    if arr.size == 0:
        raise InvalidSignalError("signal must contain at least one sample")
    return arr


def _fft_recursive(x: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Core radix-2 recursion; assumes ``len(x)`` is a power of two."""
    n = x.shape[0]
    if n <= _BASE_CASE_SIZE:
        # Small block: a direct DFT is both correct and, at this size, faster
        # than continuing to recurse in Python.
        return dft_matrix(n) @ x

    even = _fft_recursive(x[0::2])
    odd = _fft_recursive(x[1::2])

    # Twiddle factors W_N^k = exp(-2j*pi*k/N) for k = 0 .. N/2 - 1.
    twiddle = np.exp(-2j * np.pi * np.arange(n // 2) / n)
    scaled_odd = twiddle * odd

    # Butterfly combine: the two halves of the output.
    return np.concatenate([even + scaled_odd, even - scaled_odd])


def fft(x: ArrayLike) -> NDArray[np.complex128]:
    """Compute the FFT of a power-of-two-length signal.

    Produces exactly the same coefficients as :func:`fourierlab.dft.dft`, but in
    :math:`O(N \\log N)` time.

    Parameters
    ----------
    x:
        Input signal (1-D, real or complex). ``len(x)`` must be a power of two.

    Raises
    ------
    NotPowerOfTwoError
        If the length of ``x`` is not a power of two.
    """
    arr = _as_1d(x)
    n = arr.shape[0]
    if not is_power_of_two(n):
        raise NotPowerOfTwoError(n)
    return _fft_recursive(arr)


def ifft(spectrum: ArrayLike) -> NDArray[np.complex128]:
    """Compute the inverse FFT of a power-of-two-length spectrum.

    Uses the conjugation identity, which reuses the forward transform::

        ifft(X) = conj( fft( conj(X) ) ) / N

    so no separate inverse butterfly is needed. ``ifft(fft(x))`` returns ``x``
    up to floating-point round-off.

    Raises
    ------
    NotPowerOfTwoError
        If the length of ``spectrum`` is not a power of two.
    """
    arr = _as_1d(spectrum)
    n = arr.shape[0]
    if not is_power_of_two(n):
        raise NotPowerOfTwoError(n)
    return np.conjugate(_fft_recursive(np.conjugate(arr))) / n


def forward_transform(x: ArrayLike) -> NDArray[np.complex128]:
    """Forward transform that works for *any* length.

    Uses the fast radix-2 :func:`fft` when the length is a power of two, and
    falls back to the direct :func:`fourierlab.dft.dft` otherwise. Both return
    identical coefficients, so downstream code (e.g. the filters) does not have
    to care which path was taken. The fallback is :math:`O(N^2)`; pad signals to
    a power of two if that matters for your data size.
    """
    arr = _as_1d(x)
    if is_power_of_two(arr.shape[0]):
        return _fft_recursive(arr)
    return dft(arr)


def inverse_transform(spectrum: ArrayLike) -> NDArray[np.complex128]:
    """Inverse counterpart of :func:`forward_transform`, for any length."""
    arr = _as_1d(spectrum)
    if is_power_of_two(arr.shape[0]):
        return ifft(arr)
    return idft(arr)


class TimingResult(NamedTuple):
    """One row of a DFT-vs-FFT timing comparison."""

    size: int
    dft_seconds: float
    fft_seconds: float
    speedup: float  # dft_seconds / fft_seconds


def benchmark(
    sizes: Sequence[int], repeats: int = 1, seed: int = 0
) -> list[TimingResult]:
    """Time the direct DFT against the FFT for each power-of-two size.

    For each ``N`` a random real signal is transformed ``repeats`` times with
    both methods and the average wall-clock time recorded. This is what powers
    the ``compare`` CLI command and demonstrates :math:`O(N^2)` versus
    :math:`O(N \\log N)` scaling.

    Parameters
    ----------
    sizes:
        Transform sizes to test. Each must be a power of two.
    repeats:
        How many times to run each transform (averaged) for a steadier reading.
    seed:
        Seed for the random test signals, for reproducibility.

    Raises
    ------
    NotPowerOfTwoError
        If any requested size is not a power of two.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    rng = np.random.default_rng(seed)
    results: list[TimingResult] = []
    for n in sizes:
        if not is_power_of_two(n):
            raise NotPowerOfTwoError(n)
        x = rng.standard_normal(n)

        start = perf_counter()
        for _ in range(repeats):
            dft(x)
        dft_seconds = (perf_counter() - start) / repeats

        start = perf_counter()
        for _ in range(repeats):
            fft(x)
        fft_seconds = (perf_counter() - start) / repeats

        speedup = dft_seconds / fft_seconds if fft_seconds > 0 else float("inf")
        results.append(TimingResult(int(n), dft_seconds, fft_seconds, speedup))
    return results
