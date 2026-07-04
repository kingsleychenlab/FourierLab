"""The Discrete Fourier Transform, implemented directly from its definition.

This module deliberately does **not** use :mod:`numpy.fft`. NumPy is used only
to store the data and to evaluate the complex exponentials in a vectorized
way; the transform itself is the textbook sum

.. math::

    X_k = \\sum_{n=0}^{N-1} x_n \\, e^{-2\\pi i k n / N},
    \\qquad k = 0, 1, \\dots, N-1

and its inverse

.. math::

    x_n = \\frac{1}{N} \\sum_{k=0}^{N-1} X_k \\, e^{+2\\pi i k n / N}.

We form the :math:`N \\times N` matrix ``W`` with entries
``W[k, n] = exp(-2j*pi*k*n/N)`` and compute ``X = W @ x``. Building that matrix
is :math:`O(N^2)` in both time and memory, which is exactly the cost the FFT
exists to avoid -- see :mod:`fourierlab.fft`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import InvalidSignalError

__all__ = ["dft_matrix", "dft", "idft"]


def _as_1d(x: ArrayLike, *, dtype: type) -> NDArray:
    """Validate and coerce ``x`` into a 1-D array of the requested dtype."""
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim != 1:
        raise InvalidSignalError(
            f"expected a one-dimensional signal, got shape {arr.shape}"
        )
    if arr.size == 0:
        raise InvalidSignalError("signal must contain at least one sample")
    return arr


def dft_matrix(n: int, *, inverse: bool = False) -> NDArray[np.complex128]:
    """Return the ``n x n`` DFT (or inverse-DFT) matrix.

    The forward matrix has entries ``exp(-2j*pi*k*m/n)``. The inverse matrix
    uses ``+`` in the exponent and is scaled by ``1/n`` so that
    ``idft_matrix @ dft_matrix == I``.

    Parameters
    ----------
    n:
        Transform size.
    inverse:
        If ``True``, return the inverse-DFT matrix instead of the forward one.
    """
    if n < 1:
        raise InvalidSignalError("DFT size n must be >= 1")
    k = np.arange(n).reshape((n, 1))  # row index  (output frequency bins)
    m = np.arange(n).reshape((1, n))  # column index (input sample indices)
    sign = 1.0 if inverse else -1.0
    w = np.exp(sign * 2j * np.pi * k * m / n)
    if inverse:
        w = w / n
    return w


def dft(x: ArrayLike) -> NDArray[np.complex128]:
    """Compute the Discrete Fourier Transform of ``x`` from the definition.

    Accepts real or complex input of any length ``N`` and returns the ``N``
    complex Fourier coefficients ``X_0 .. X_{N-1}``. Coefficient ``X_k``
    measures how much of the complex sinusoid ``exp(+2j*pi*k*n/N)`` is present
    in the signal.

    Parameters
    ----------
    x:
        Input signal (1-D, real or complex).

    Returns
    -------
    numpy.ndarray
        Complex spectrum of length ``N``.
    """
    arr = _as_1d(x, dtype=complex)
    n = arr.shape[0]
    return dft_matrix(n) @ arr


def idft(spectrum: ArrayLike) -> NDArray[np.complex128]:
    """Compute the inverse DFT, reconstructing the signal from its spectrum.

    ``idft(dft(x))`` returns ``x`` up to floating-point round-off. For a
    spectrum that came from a real signal the imaginary part of the result is
    numerical noise; callers that know their signal is real may take
    ``.real``.

    Parameters
    ----------
    spectrum:
        Complex Fourier coefficients (1-D).

    Returns
    -------
    numpy.ndarray
        Reconstructed complex signal of length ``N``.
    """
    arr = _as_1d(spectrum, dtype=complex)
    n = arr.shape[0]
    return dft_matrix(n, inverse=True) @ arr
