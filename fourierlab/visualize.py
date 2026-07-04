"""Matplotlib plotting for signals, spectra, series, filters, and benchmarks.

The non-interactive ``Agg`` backend is selected on import so every command can
render and save a PNG from a headless terminal without a display server. Each
``plot_*`` function builds a labelled, grid-lined figure and, if given a
``save`` path, writes it there and returns the :class:`~matplotlib.figure.Figure`.

Only matplotlib is used -- no seaborn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import ArrayLike, NDArray  # noqa: E402

from .fft import TimingResult  # noqa: E402
from .filters import FilterResult  # noqa: E402
from .math_utils import (  # noqa: E402
    amplitude_spectrum,
    fft_freqs,
    phase_spectrum,
)
from .series import SeriesApproximation  # noqa: E402
from .signals import Signal  # noqa: E402

__all__ = [
    "plot_signal",
    "plot_reconstruction",
    "plot_series",
    "plot_spectrum",
    "plot_filter",
    "plot_comparison",
    "plot_gibbs",
    "save_figure",
]

_FIGSIZE = (9.0, 5.0)


# Deterministic, non-identifying PNG metadata. Overriding the defaults keeps
# saved figures reproducible and free of any environment-derived information
# (matplotlib version strings, timestamps, etc.).
_PNG_METADATA = {"Software": "FourierLab"}


def save_figure(fig: "plt.Figure", save: str | Path | None) -> "plt.Figure":
    """Tidy layout and, if ``save`` is given, write the figure to disk."""
    fig.tight_layout()
    if save is not None:
        path = Path(save)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        metadata = _PNG_METADATA if path.suffix.lower() == ".png" else None
        fig.savefig(path, dpi=120, bbox_inches="tight", metadata=metadata)
    return fig


def plot_signal(
    signal: Signal,
    title: str | None = None,
    save: str | Path | None = None,
) -> "plt.Figure":
    """Plot a single time-domain signal."""
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(signal.t, np.real(signal.y), lw=1.3, color="C0")
    ax.set_title(title or f"Signal: {signal.name}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("amplitude")
    ax.grid(True, alpha=0.3)
    return save_figure(fig, save)


def plot_reconstruction(
    t: ArrayLike,
    original: ArrayLike,
    reconstructed: ArrayLike,
    title: str = "Original vs reconstructed",
    xlabel: str = "time [s]",
    save: str | Path | None = None,
) -> "plt.Figure":
    """Overlay an original signal and its reconstruction, with the residual below."""
    t = np.asarray(t)
    original = np.real(np.asarray(original))
    reconstructed = np.real(np.asarray(reconstructed))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9.0, 6.0), sharex=True, height_ratios=[3, 1]
    )
    ax_top.plot(t, original, lw=2.0, color="C0", label="original", alpha=0.8)
    ax_top.plot(
        t, reconstructed, lw=1.2, color="C3", ls="--", label="reconstructed"
    )
    ax_top.set_title(title)
    ax_top.set_ylabel("amplitude")
    ax_top.legend(loc="upper right")
    ax_top.grid(True, alpha=0.3)

    ax_bot.plot(t, original - reconstructed, lw=1.0, color="C2")
    ax_bot.set_ylabel("residual")
    ax_bot.set_xlabel(xlabel)
    ax_bot.grid(True, alpha=0.3)
    return save_figure(fig, save)


def plot_series(
    approx: SeriesApproximation, save: str | Path | None = None
) -> "plt.Figure":
    """Plot a Fourier-series approximation over the target function."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9.0, 6.0), sharex=True, height_ratios=[3, 1]
    )
    ax_top.plot(approx.x, approx.original, lw=2.2, color="C0", label="f(x)", alpha=0.8)
    ax_top.plot(
        approx.x,
        approx.approximation,
        lw=1.3,
        color="C3",
        label=f"{approx.num_terms}-term series",
    )
    ax_top.set_title(
        f"Fourier series of '{approx.name}'  "
        f"(N={approx.num_terms}, MSE={approx.mse:.2e}, "
        f"max err={approx.max_error:.2e})"
    )
    ax_top.set_ylabel("amplitude")
    ax_top.legend(loc="upper right")
    ax_top.grid(True, alpha=0.3)

    ax_bot.plot(
        approx.x, approx.original - approx.approximation, lw=1.0, color="C2"
    )
    ax_bot.set_ylabel("error")
    ax_bot.set_xlabel("x")
    ax_bot.grid(True, alpha=0.3)
    return save_figure(fig, save)


def plot_spectrum(
    spectrum: ArrayLike,
    sample_rate: float,
    title: str = "Frequency spectrum",
    save: str | Path | None = None,
    max_freq: float | None = None,
) -> "plt.Figure":
    """Plot the single-sided magnitude (amplitude) and phase spectra.

    Parameters
    ----------
    spectrum:
        Full complex DFT coefficients.
    sample_rate:
        Sampling frequency in Hz (sets the frequency axis).
    max_freq:
        Optional upper limit for the frequency axis (defaults to Nyquist).
    """
    spectrum = np.asarray(spectrum, dtype=complex)
    n = spectrum.shape[0]
    half = n // 2 + 1
    freqs = fft_freqs(n, sample_rate)[:half]
    amplitude = amplitude_spectrum(spectrum, n)
    phase = phase_spectrum(spectrum)[:half]

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(9.0, 6.5), sharex=True)
    ax_mag.plot(freqs, amplitude, lw=1.4, color="C0")
    ax_mag.set_title(title)
    ax_mag.set_ylabel("amplitude")
    ax_mag.grid(True, alpha=0.3)

    ax_phase.plot(freqs, phase, lw=1.0, color="C4")
    ax_phase.set_ylabel("phase [rad]")
    ax_phase.set_xlabel("frequency [Hz]")
    ax_phase.grid(True, alpha=0.3)

    upper = max_freq if max_freq is not None else sample_rate / 2.0
    ax_mag.set_xlim(0, upper)
    return save_figure(fig, save)


def plot_filter(
    result: FilterResult, save: str | Path | None = None
) -> "plt.Figure":
    """Plot the before/after of a filter in both time and frequency domains."""
    orig = result.original
    filt = result.filtered
    n = orig.n
    half = n // 2 + 1
    freqs = fft_freqs(n, orig.sample_rate)[:half]
    amp_before = amplitude_spectrum(result.spectrum, n)
    amp_after = amplitude_spectrum(result.filtered_spectrum, n)

    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(9.0, 6.5))
    ax_time.plot(
        orig.t, np.real(orig.y), lw=1.0, color="C0", alpha=0.55, label="original"
    )
    ax_time.plot(
        filt.t, np.real(filt.y), lw=1.4, color="C3", label="filtered"
    )
    band = result.band
    if band.cutoff is not None:
        edge = f"cutoff = {band.cutoff:g} Hz"
    else:
        edge = f"band = {band.low:g}-{band.high:g} Hz"
    ax_time.set_title(f"{result.filter_type} filter  ({edge})")
    ax_time.set_xlabel("time [s]")
    ax_time.set_ylabel("amplitude")
    ax_time.legend(loc="upper right")
    ax_time.grid(True, alpha=0.3)

    ax_freq.plot(freqs, amp_before, lw=1.2, color="C0", alpha=0.55, label="before")
    ax_freq.plot(freqs, amp_after, lw=1.4, color="C3", label="after")
    ax_freq.set_xlabel("frequency [Hz]")
    ax_freq.set_ylabel("amplitude")
    ax_freq.set_xlim(0, orig.sample_rate / 2.0)
    ax_freq.legend(loc="upper right")
    ax_freq.grid(True, alpha=0.3)
    return save_figure(fig, save)


def plot_comparison(
    results: Sequence[TimingResult], save: str | Path | None = None
) -> "plt.Figure":
    """Plot DFT vs FFT runtimes on log-log axes with complexity guide lines."""
    sizes = np.array([r.size for r in results], dtype=float)
    dft_t = np.array([r.dft_seconds for r in results], dtype=float)
    fft_t = np.array([r.fft_seconds for r in results], dtype=float)

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.loglog(sizes, dft_t, "o-", color="C3", label="DFT (direct)")
    ax.loglog(sizes, fft_t, "s-", color="C0", label="FFT (Cooley-Tukey)")

    # Reference complexity curves, anchored to the DFT/FFT first data points.
    n2 = sizes**2
    nlogn = sizes * np.log2(sizes)
    ax.loglog(
        sizes, dft_t[0] * n2 / n2[0], ":", color="C3", alpha=0.5,
        label=r"$O(N^2)$ guide",
    )
    ax.loglog(
        sizes, fft_t[0] * nlogn / nlogn[0], ":", color="C0", alpha=0.5,
        label=r"$O(N \log N)$ guide",
    )

    ax.set_title("DFT vs FFT runtime scaling")
    ax.set_xlabel("signal length N")
    ax.set_ylabel("time per transform [s]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")
    return save_figure(fig, save)


def plot_gibbs(
    demo: dict, save: str | Path | None = None
) -> "plt.Figure":
    """Plot the Gibbs phenomenon: square-wave partial sums at several term counts."""
    x = np.asarray(demo["x"])
    exact = np.asarray(demo["exact"])
    reconstructions: dict[int, NDArray[np.float64]] = demo["reconstructions"]
    overshoots: dict[int, float] = demo["overshoots"]

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(x, exact, color="k", lw=1.6, alpha=0.6, label="square wave")
    cmap = ["C0", "C1", "C2", "C3", "C4", "C5"]
    for i, (n, y) in enumerate(sorted(reconstructions.items())):
        ax.plot(
            x, y, lw=1.1, color=cmap[i % len(cmap)],
            label=f"N={n}  (peak {overshoots[n]:.3f})",
        )
    ax.axhline(1.0, color="grey", ls=":", lw=0.8)
    ax.set_title("Gibbs phenomenon: ~9% overshoot persists as N grows")
    ax.set_xlabel("x")
    ax.set_ylabel("amplitude")
    ax.set_xlim(-np.pi, np.pi)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    return save_figure(fig, save)
