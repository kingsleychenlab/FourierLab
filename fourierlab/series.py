r"""Real Fourier series: coefficients, reconstruction, and Gibbs phenomenon.

A :math:`2\pi`-periodic (more generally, ``period``-periodic) function can be
written as a sum of harmonically related sines and cosines,

.. math::

    f(x) \;\approx\; \frac{a_0}{2}
        + \sum_{n=1}^{N} \big[\, a_n \cos(n\omega x) + b_n \sin(n\omega x) \,\big],
    \qquad \omega = \frac{2\pi}{\text{period}},

with coefficients (shown here for the canonical period :math:`2\pi`,
:math:`\omega = 1`)

.. math::

    a_n = \frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\cos(nx)\,dx,
    \qquad
    b_n = \frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\sin(nx)\,dx .

``a_0 / 2`` is simply the average value of ``f`` over one period.

This module estimates those integrals numerically (so it works for *any*
sampled function), and also carries the exact closed-form coefficients for the
classic square / sawtooth / triangle / sine / cosine waves so the two can be
cross-checked. For functions with jump discontinuities the partial sums exhibit
the **Gibbs phenomenon**: a persistent ~9%-of-the-jump overshoot that narrows
but never vanishes as ``N`` grows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import InvalidFunctionError, InvalidParameterError
from .math_utils import max_error, mse

__all__ = [
    "FourierSeries",
    "SeriesApproximation",
    "CANONICAL_FUNCTIONS",
    "canonical_function",
    "fourier_coefficients",
    "coefficients_from_samples",
    "analytical_coefficients",
    "approximate",
    "gibbs_overshoot",
    "gibbs_demo",
]

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
# Canonical, unit-amplitude, period-2*pi reference waveforms.
# --------------------------------------------------------------------------- #
def _wrap(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wrap ``x`` into the half-open interval ``(-pi, pi]``."""
    return np.mod(x + np.pi, TWO_PI) - np.pi


def _f_sine(x: ArrayLike) -> NDArray[np.float64]:
    return np.sin(np.asarray(x, dtype=float))


def _f_cosine(x: ArrayLike) -> NDArray[np.float64]:
    return np.cos(np.asarray(x, dtype=float))


def _f_square(x: ArrayLike) -> NDArray[np.float64]:
    """Odd square wave: +1 on (0, pi), -1 on (-pi, 0)."""
    s = np.sign(np.sin(np.asarray(x, dtype=float)))
    s[s == 0] = 1.0
    return s


def _f_sawtooth(x: ArrayLike) -> NDArray[np.float64]:
    """Rising sawtooth ``x/pi`` on (-pi, pi), ranging over [-1, 1]."""
    return _wrap(np.asarray(x, dtype=float)) / np.pi


def _f_triangle(x: ArrayLike) -> NDArray[np.float64]:
    """Even triangle wave, +1 at x=0 down to -1 at x=+/-pi."""
    return 1.0 - 2.0 * np.abs(_wrap(np.asarray(x, dtype=float))) / np.pi


#: Canonical unit waveforms on period ``2*pi`` keyed by name.
CANONICAL_FUNCTIONS: dict[str, Callable[[ArrayLike], NDArray[np.float64]]] = {
    "sine": _f_sine,
    "cosine": _f_cosine,
    "square": _f_square,
    "sawtooth": _f_sawtooth,
    "triangle": _f_triangle,
}


def canonical_function(name: str) -> Callable[[ArrayLike], NDArray[np.float64]]:
    """Return the canonical period-2*pi waveform callable named ``name``."""
    try:
        return CANONICAL_FUNCTIONS[name]
    except KeyError:
        raise InvalidFunctionError(
            f"unknown function {name!r}; choose from "
            f"{tuple(CANONICAL_FUNCTIONS)}"
        ) from None


# --------------------------------------------------------------------------- #
# The series object.
# --------------------------------------------------------------------------- #
@dataclass
class FourierSeries:
    """Real Fourier-series coefficients and the machinery to evaluate them.

    Attributes
    ----------
    a0:
        The ``a_0`` coefficient (note: the constant term of the series is
        ``a0 / 2``).
    an, bn:
        Cosine and sine coefficients for harmonics ``n = 1 .. N`` (length ``N``).
    period:
        The period of the function, used to set ``omega = 2*pi/period``.
    """

    a0: float
    an: NDArray[np.float64]
    bn: NDArray[np.float64]
    period: float = TWO_PI

    def __post_init__(self) -> None:
        self.an = np.asarray(self.an, dtype=float)
        self.bn = np.asarray(self.bn, dtype=float)
        if self.an.shape != self.bn.shape:
            raise InvalidParameterError("an and bn must have the same length")
        if self.period <= 0:
            raise InvalidParameterError("period must be positive")

    @property
    def num_terms(self) -> int:
        """Number of harmonics ``N`` retained (excluding the constant term)."""
        return int(self.an.shape[0])

    @property
    def omega(self) -> float:
        """Angular fundamental frequency ``2*pi / period``."""
        return TWO_PI / self.period

    def evaluate(self, x: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the truncated series ``S_N(x)`` at points ``x``."""
        x = np.asarray(x, dtype=float)
        result = np.full_like(x, self.a0 / 2.0)
        w = self.omega
        for n in range(1, self.num_terms + 1):
            result += self.an[n - 1] * np.cos(n * w * x)
            result += self.bn[n - 1] * np.sin(n * w * x)
        return result


@dataclass
class SeriesApproximation:
    """The result of approximating a function by a truncated Fourier series."""

    name: str
    num_terms: int
    x: NDArray[np.float64]
    original: NDArray[np.float64]
    approximation: NDArray[np.float64]
    series: FourierSeries
    mse: float
    max_error: float


# --------------------------------------------------------------------------- #
# Coefficient estimation.
# --------------------------------------------------------------------------- #
def fourier_coefficients(
    func: Callable[[ArrayLike], ArrayLike],
    num_terms: int,
    period: float = TWO_PI,
    num_samples: int = 4096,
) -> FourierSeries:
    r"""Estimate Fourier coefficients of ``func`` by numerical integration.

    The integrals for ``a_n`` and ``b_n`` are evaluated with the midpoint rule
    on ``num_samples`` points spanning one period. For smooth periodic
    integrands this rule is extremely accurate; a single jump discontinuity
    contributes only :math:`O(1/\text{num\_samples})` error.

    Parameters
    ----------
    func:
        A vectorised callable evaluating the target function at an array of x.
    num_terms:
        Number of harmonics ``N`` to compute.
    period:
        Period of the function.
    num_samples:
        Number of quadrature points across one period.
    """
    if num_terms < 1:
        raise InvalidParameterError("num_terms must be >= 1")
    if num_samples < 2 * num_terms:
        # Need enough samples to resolve the highest requested harmonic.
        num_samples = 2 * num_terms

    half = period / 2.0
    dx = period / num_samples
    # Midpoints: avoids landing exactly on a symmetric discontinuity.
    x = -half + (np.arange(num_samples) + 0.5) * dx
    fx = np.asarray(func(x), dtype=float)
    w = TWO_PI / period

    a0 = (2.0 / period) * np.sum(fx) * dx
    an = np.empty(num_terms)
    bn = np.empty(num_terms)
    for n in range(1, num_terms + 1):
        an[n - 1] = (2.0 / period) * np.sum(fx * np.cos(n * w * x)) * dx
        bn[n - 1] = (2.0 / period) * np.sum(fx * np.sin(n * w * x)) * dx
    return FourierSeries(a0=float(a0), an=an, bn=bn, period=period)


def coefficients_from_samples(
    samples: ArrayLike, num_terms: int, period: float = TWO_PI
) -> FourierSeries:
    """Estimate Fourier coefficients from uniformly spaced samples of one period.

    ``samples`` are assumed to be ``M`` equally spaced values covering exactly
    one period (the point at the far end of the period is *not* repeated). This
    is the tool to use for custom, measured periodic data.
    """
    fx = np.asarray(samples, dtype=float)
    if fx.ndim != 1 or fx.size < 2:
        raise InvalidParameterError("samples must be a 1-D array of length >= 2")
    if num_terms < 1:
        raise InvalidParameterError("num_terms must be >= 1")

    m = fx.shape[0]
    half = period / 2.0
    dx = period / m
    x = -half + np.arange(m) * dx
    w = TWO_PI / period

    a0 = (2.0 / m) * np.sum(fx)
    an = np.empty(num_terms)
    bn = np.empty(num_terms)
    for n in range(1, num_terms + 1):
        an[n - 1] = (2.0 / m) * np.sum(fx * np.cos(n * w * x))
        bn[n - 1] = (2.0 / m) * np.sum(fx * np.sin(n * w * x))
    return FourierSeries(a0=float(a0), an=an, bn=bn, period=period)


def analytical_coefficients(
    name: str, num_terms: int, period: float = TWO_PI
) -> FourierSeries:
    r"""Return the exact closed-form Fourier coefficients of a canonical waveform.

    Known series for the unit-amplitude, period-2*pi waveforms:

    * ``sine``     : :math:`b_1 = 1`.
    * ``cosine``   : :math:`a_1 = 1`.
    * ``square``   : :math:`b_n = \tfrac{4}{n\pi}` for odd ``n``.
    * ``sawtooth`` : :math:`b_n = \tfrac{2(-1)^{n+1}}{n\pi}`.
    * ``triangle`` : :math:`a_n = \tfrac{8}{\pi^2 n^2}` for odd ``n``.

    The coefficient *values* are the same for any period (they describe the
    normalised shape); only the reconstruction frequency ``omega`` changes.
    """
    if num_terms < 1:
        raise InvalidParameterError("num_terms must be >= 1")
    if name not in CANONICAL_FUNCTIONS:
        raise InvalidFunctionError(
            f"no analytical series for {name!r}; choose from "
            f"{tuple(CANONICAL_FUNCTIONS)}"
        )

    an = np.zeros(num_terms)
    bn = np.zeros(num_terms)
    for n in range(1, num_terms + 1):
        if name == "sine":
            if n == 1:
                bn[0] = 1.0
        elif name == "cosine":
            if n == 1:
                an[0] = 1.0
        elif name == "square":
            if n % 2 == 1:
                bn[n - 1] = 4.0 / (n * np.pi)
        elif name == "sawtooth":
            bn[n - 1] = 2.0 * ((-1.0) ** (n + 1)) / (n * np.pi)
        elif name == "triangle":
            if n % 2 == 1:
                an[n - 1] = 8.0 / (np.pi**2 * n**2)
    return FourierSeries(a0=0.0, an=an, bn=bn, period=period)


def approximate(
    function: str | Callable[[ArrayLike], ArrayLike],
    num_terms: int,
    period: float = TWO_PI,
    method: str = "numerical",
    num_points: int = 2000,
    num_samples: int = 4096,
) -> SeriesApproximation:
    """Approximate a function by an ``num_terms``-harmonic Fourier series.

    Parameters
    ----------
    function:
        Either the name of a canonical waveform (see
        :data:`CANONICAL_FUNCTIONS`) or a custom vectorised callable.
    num_terms:
        Number of harmonics to keep.
    period:
        Period of the function.
    method:
        ``"numerical"`` (default) estimates coefficients by integration and
        works for any function; ``"analytical"`` uses the exact closed forms and
        is only available for the canonical named waveforms.
    num_points:
        Number of points on which to render the comparison.
    num_samples:
        Quadrature resolution for the numerical method.

    Returns
    -------
    SeriesApproximation
        Bundle of the sample grid, the true and approximated values, the
        coefficient set, and the MSE / max-error metrics.
    """
    if isinstance(function, str):
        name = function
        func = canonical_function(name)
    else:
        name = getattr(function, "__name__", "custom")
        func = function

    if method == "analytical":
        if not isinstance(function, str):
            raise InvalidFunctionError(
                "analytical coefficients are only available for named waveforms"
            )
        series = analytical_coefficients(name, num_terms, period)
    elif method == "numerical":
        series = fourier_coefficients(func, num_terms, period, num_samples)
    else:
        raise InvalidParameterError(
            f"unknown method {method!r}; use 'numerical' or 'analytical'"
        )

    half = period / 2.0
    x = np.linspace(-half, half, num_points)
    original = np.asarray(func(x), dtype=float)
    approximation = series.evaluate(x)
    return SeriesApproximation(
        name=name,
        num_terms=num_terms,
        x=x,
        original=original,
        approximation=approximation,
        series=series,
        mse=mse(original, approximation),
        max_error=max_error(original, approximation),
    )


# --------------------------------------------------------------------------- #
# Gibbs phenomenon.
# --------------------------------------------------------------------------- #
def gibbs_overshoot(num_terms: int, num_points: int = 8000) -> float:
    """Return the peak value of the ``num_terms``-term square-wave partial sum.

    For the unit square wave the maximum of the partial sum converges to
    :math:`\\tfrac{2}{\\pi}\\,\\mathrm{Si}(\\pi) \\approx 1.1790` -- an overshoot
    of about 9% of the jump (which is 2). The peak height does *not* shrink as
    ``num_terms`` grows; only the width of the overshoot does. A dense grid is
    used so the ever-narrowing peak near ``x = 0`` is resolved.
    """
    series = analytical_coefficients("square", num_terms)
    # Sample just to the right of the discontinuity at x = 0, where the first
    # (largest) overshoot lobe lives.
    x = np.linspace(0.0, np.pi / 2.0, num_points)
    values = series.evaluate(x)
    return float(np.max(values))


def gibbs_demo(
    term_counts: tuple[int, ...] = (5, 15, 45),
    num_points: int = 4000,
) -> dict[str, object]:
    """Produce data illustrating the Gibbs phenomenon for a square wave.

    Returns a dictionary with the shared ``x`` grid, the exact square wave, a
    mapping ``{N: partial_sum}`` for each requested term count, and the peak
    overshoot for each ``N``.
    """
    x = np.linspace(-np.pi, np.pi, num_points)
    exact = _f_square(x)
    reconstructions: dict[int, NDArray[np.float64]] = {}
    overshoots: dict[int, float] = {}
    for n in term_counts:
        series = analytical_coefficients("square", n)
        reconstructions[n] = series.evaluate(x)
        overshoots[n] = gibbs_overshoot(n)
    return {
        "x": x,
        "exact": exact,
        "reconstructions": reconstructions,
        "overshoots": overshoots,
    }
