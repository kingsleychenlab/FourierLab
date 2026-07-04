"""Tests for the from-scratch recursive Cooley-Tukey FFT.

Covers agreement with NumPy and with our own DFT, inverse reconstruction, the
power-of-two guard, and the headline performance claim that the FFT beats the
direct DFT for reasonably large inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from fourierlab.dft import dft
from fourierlab.errors import NotPowerOfTwoError
from fourierlab.fft import (
    benchmark,
    fft,
    forward_transform,
    ifft,
    inverse_transform,
)
from fourierlab.math_utils import parseval

RNG = np.random.default_rng(2024)

POWERS_OF_TWO = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


@pytest.mark.parametrize("n", POWERS_OF_TWO)
def test_fft_matches_numpy_real(n: int) -> None:
    x = RNG.standard_normal(n)
    assert np.allclose(fft(x), np.fft.fft(x), atol=1e-9)


@pytest.mark.parametrize("n", [2, 8, 64, 256])
def test_fft_matches_numpy_complex(n: int) -> None:
    x = RNG.standard_normal(n) + 1j * RNG.standard_normal(n)
    assert np.allclose(fft(x), np.fft.fft(x), atol=1e-9)


@pytest.mark.parametrize("n", POWERS_OF_TWO)
def test_fft_matches_dft(n: int) -> None:
    x = RNG.standard_normal(n)
    assert np.allclose(fft(x), dft(x), atol=1e-9)


@pytest.mark.parametrize("n", POWERS_OF_TWO)
def test_ifft_reconstructs(n: int) -> None:
    x = RNG.standard_normal(n) + 1j * RNG.standard_normal(n)
    assert np.allclose(ifft(fft(x)), x, atol=1e-9)


def test_ifft_of_real_signal_is_real() -> None:
    x = RNG.standard_normal(128)
    reconstructed = ifft(fft(x))
    assert np.max(np.abs(reconstructed.imag)) < 1e-9
    assert np.allclose(reconstructed.real, x, atol=1e-9)


@pytest.mark.parametrize("n", [3, 5, 6, 7, 12, 24, 100, 129])
def test_fft_rejects_non_power_of_two(n: int) -> None:
    x = RNG.standard_normal(n)
    with pytest.raises(NotPowerOfTwoError) as excinfo:
        fft(x)
    # The message should be actionable: mention power of two and the bad length.
    msg = str(excinfo.value)
    assert "power of two" in msg
    assert str(n) in msg


def test_ifft_rejects_non_power_of_two() -> None:
    with pytest.raises(NotPowerOfTwoError):
        ifft(RNG.standard_normal(6))


@pytest.mark.parametrize("n", [12, 24, 48, 100])
def test_forward_transform_handles_any_length(n: int) -> None:
    # The dispatcher falls back to the DFT for non-power-of-two lengths.
    x = RNG.standard_normal(n)
    assert np.allclose(forward_transform(x), np.fft.fft(x), atol=1e-9)
    assert np.allclose(inverse_transform(forward_transform(x)), x, atol=1e-9)


@pytest.mark.parametrize("n", [8, 64, 256])
def test_parseval_holds_for_fft(n: int) -> None:
    x = RNG.standard_normal(n)
    assert parseval(x, fft(x)).relative_error < 1e-12


def test_fft_is_faster_than_dft_for_large_input() -> None:
    # At N=2048 the O(N log N) FFT should comfortably beat the O(N^2) DFT.
    # The measured margin is tens of times; we only require a clear win.
    (result,) = benchmark([2048], repeats=2)
    assert result.fft_seconds < result.dft_seconds
    assert result.speedup > 3.0


def test_benchmark_rejects_non_power_of_two() -> None:
    with pytest.raises(NotPowerOfTwoError):
        benchmark([48])
