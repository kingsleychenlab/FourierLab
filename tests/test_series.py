"""Tests for the real Fourier-series module.

Verifies that numerically estimated coefficients agree with the known closed
forms, that approximations improve as more harmonics are added, that a couple of
harmonics suffice for a pure sinusoid, and that the Gibbs overshoot behaves as
theory predicts.
"""

from __future__ import annotations

import numpy as np
import pytest

from fourierlab.errors import InvalidFunctionError, InvalidParameterError
from fourierlab.series import (
    analytical_coefficients,
    approximate,
    canonical_function,
    coefficients_from_samples,
    fourier_coefficients,
    gibbs_demo,
    gibbs_overshoot,
)


@pytest.mark.parametrize("name", ["sine", "cosine", "square", "sawtooth", "triangle"])
def test_numerical_matches_analytical(name: str) -> None:
    n = 20
    numeric = fourier_coefficients(canonical_function(name), n)
    exact = analytical_coefficients(name, n)
    # Discontinuous waveforms carry a tiny O(1/num_samples) quadrature error;
    # 1e-3 is far tighter than the O(1) gap a wrong formula would produce.
    assert np.allclose(numeric.an, exact.an, atol=1e-3)
    assert np.allclose(numeric.bn, exact.bn, atol=1e-3)
    assert numeric.a0 == pytest.approx(exact.a0, abs=1e-3)


def _mse_sequence(name: str, term_counts, method: str = "numerical"):
    return [approximate(name, n, method=method).mse for n in term_counts]


def test_square_wave_improves_with_more_terms() -> None:
    errors = _mse_sequence("square", [3, 8, 20, 50])
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))


def test_triangle_wave_improves_with_more_terms() -> None:
    errors = _mse_sequence("triangle", [2, 5, 12, 30])
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))


def test_sawtooth_improves_with_more_terms() -> None:
    errors = _mse_sequence("sawtooth", [3, 8, 20, 50])
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))


def test_sine_reconstructed_with_few_terms() -> None:
    approx = approximate("sine", 3)
    assert approx.max_error < 1e-6


def test_cosine_reconstructed_with_few_terms() -> None:
    approx = approximate("cosine", 3)
    assert approx.max_error < 1e-6


def test_gibbs_overshoot_magnitude() -> None:
    # The square-wave partial sum peaks near (2/pi) Si(pi) ~= 1.1790 -- an
    # overshoot of ~9% of the unit half-jump.
    peak = gibbs_overshoot(80)
    assert 1.16 < peak < 1.20


def test_gibbs_overshoot_persists_as_terms_grow() -> None:
    # The overshoot does NOT vanish with more terms; it converges to ~1.179.
    peaks = [gibbs_overshoot(n) for n in [20, 50, 100, 200]]
    assert all(p > 1.16 for p in peaks)
    # And it is settling toward the theoretical constant rather than shrinking.
    assert peaks[-1] == pytest.approx(1.1790, abs=5e-3)


def test_gibbs_demo_structure() -> None:
    demo = gibbs_demo((5, 25))
    assert set(demo["reconstructions"].keys()) == {5, 25}
    assert demo["x"].shape == demo["exact"].shape
    assert all(v > 1.0 for v in demo["overshoots"].values())


def test_analytical_series_reconstructs_sine_exactly() -> None:
    approx = approximate("sine", 5, method="analytical")
    assert approx.max_error < 1e-12


def test_coefficients_from_samples_recovers_cosine() -> None:
    # Sample a pure cosine over one period and recover a_1 == 1.
    m = 512
    x = -np.pi + np.arange(m) * (2 * np.pi / m)
    samples = np.cos(x)
    series = coefficients_from_samples(samples, num_terms=5)
    assert series.an[0] == pytest.approx(1.0, abs=1e-6)
    assert np.max(np.abs(series.bn)) < 1e-9


def test_unknown_function_raises() -> None:
    with pytest.raises(InvalidFunctionError):
        approximate("triangular", 5)


def test_zero_terms_raises() -> None:
    with pytest.raises(InvalidParameterError):
        analytical_coefficients("square", 0)


def test_analytical_method_rejects_custom_callable() -> None:
    with pytest.raises(InvalidFunctionError):
        approximate(np.sin, 5, method="analytical")
