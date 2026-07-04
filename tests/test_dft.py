"""Tests for the from-scratch Discrete Fourier Transform.

The central correctness claim is that our hand-rolled DFT agrees with NumPy's
FFT (used purely as an independent oracle), that the inverse undoes the forward
transform, and that Parseval's energy identity holds.
"""

from __future__ import annotations

import numpy as np
import pytest

from fourierlab.dft import dft, dft_matrix, idft
from fourierlab.errors import InvalidSignalError
from fourierlab.math_utils import amplitude_spectrum, fft_freqs, parseval

RNG = np.random.default_rng(1234)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 15, 16, 33, 64])
def test_dft_matches_numpy_real(n: int) -> None:
    x = RNG.standard_normal(n)
    assert np.allclose(dft(x), np.fft.fft(x), atol=1e-9)


@pytest.mark.parametrize("n", [2, 3, 7, 16, 31])
def test_dft_matches_numpy_complex(n: int) -> None:
    x = RNG.standard_normal(n) + 1j * RNG.standard_normal(n)
    assert np.allclose(dft(x), np.fft.fft(x), atol=1e-9)


@pytest.mark.parametrize("n", [1, 2, 4, 7, 16, 31, 64])
def test_idft_reconstructs_signal(n: int) -> None:
    x = RNG.standard_normal(n) + 1j * RNG.standard_normal(n)
    reconstructed = idft(dft(x))
    assert np.allclose(reconstructed, x, atol=1e-9)


def test_idft_of_real_signal_is_real() -> None:
    x = RNG.standard_normal(32)
    reconstructed = idft(dft(x))
    assert np.max(np.abs(reconstructed.imag)) < 1e-9
    assert np.allclose(reconstructed.real, x, atol=1e-9)


def test_known_small_transform() -> None:
    # DFT of [1, 0, -1, 0] is [0, 2, 0, 2].
    x = np.array([1.0, 0.0, -1.0, 0.0])
    assert np.allclose(dft(x), [0, 2, 0, 2], atol=1e-12)


def test_magnitude_spectrum_of_pure_cosine() -> None:
    # A cosine with an integer number of cycles has its energy in exactly two
    # conjugate bins, and its single-sided amplitude equals its amplitude.
    n = 64
    cycles = 8
    amplitude = 1.7
    t = np.arange(n)
    x = amplitude * np.cos(2 * np.pi * cycles * t / n)
    spectrum = dft(x)

    amp = amplitude_spectrum(spectrum, n)
    freqs = fft_freqs(n, sample_rate=float(n))  # 1 sample/sec -> freq == cycle index
    peak_bin = int(np.argmax(amp))
    assert peak_bin == cycles
    assert amp[peak_bin] == pytest.approx(amplitude, abs=1e-9)
    # Everything away from the peak is essentially zero.
    others = np.delete(amp, peak_bin)
    assert np.max(np.abs(others)) < 1e-9


@pytest.mark.parametrize("n", [4, 8, 16, 31, 64])
def test_parseval_theorem(n: int) -> None:
    x = RNG.standard_normal(n) + 1j * RNG.standard_normal(n)
    result = parseval(x, dft(x))
    assert result.relative_error < 1e-12
    assert result.time_energy == pytest.approx(result.freq_energy, rel=1e-9)


def test_dft_matches_fft_bridge() -> None:
    from fourierlab.fft import fft

    x = RNG.standard_normal(128)
    assert np.allclose(dft(x), fft(x), atol=1e-9)


def test_dft_matrix_inverse_identity() -> None:
    n = 12
    forward = dft_matrix(n)
    inverse = dft_matrix(n, inverse=True)
    assert np.allclose(inverse @ forward, np.eye(n), atol=1e-12)


def test_empty_signal_raises() -> None:
    with pytest.raises(InvalidSignalError):
        dft([])


def test_two_dimensional_input_raises() -> None:
    with pytest.raises(InvalidSignalError):
        dft(np.ones((4, 4)))
