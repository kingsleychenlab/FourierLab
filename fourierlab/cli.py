"""Command-line interface for FourierLab.

Built on the standard library's :mod:`argparse` (no extra dependency). Each
sub-command is a thin adapter: it parses arguments, calls into the math modules,
and prints a compact, aligned report. All deliberate errors derive from
:class:`~fourierlab.errors.FourierLabError` and are caught in :func:`main`, so
the user sees a clean ``error: ...`` line instead of a traceback.

Run ``python -m fourierlab --help`` for the full command list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from . import __version__
from .dft import dft, idft
from .errors import FourierLabError, InvalidParameterError
from .fft import benchmark, fft, ifft, forward_transform
from .filters import FILTER_TYPES, apply_filter
from .math_utils import (
    amplitude_spectrum,
    dominant_frequencies,
    fft_freqs,
    is_power_of_two,
    magnitude_spectrum,
    max_error,
    parseval,
    phase_spectrum,
)
from .series import CANONICAL_FUNCTIONS, approximate, gibbs_demo
from .signals import WAVEFORMS, Signal, generate

# --------------------------------------------------------------------------- #
# Small output helpers.
# --------------------------------------------------------------------------- #
_KEY_WIDTH = 18


def _header(title: str) -> None:
    print(f"\nFourierLab :: {title}\n")


def _kv(key: str, value: object) -> None:
    print(f"{key:<{_KEY_WIDTH}} = {value}")


def _fmt_complex(z: complex, prec: int = 4) -> str:
    """Format a complex number as ``+a.aaaa+b.bbbbj``."""
    return f"{z.real:+.{prec}f}{z.imag:+.{prec}f}j"


def _preview(arr: np.ndarray, count: int = 6, prec: int = 3) -> str:
    """Return a short ``[a, b, ..., y, z]`` preview of an array."""
    arr = np.asarray(arr)
    real = not np.iscomplexobj(arr)

    def fmt(v: object) -> str:
        return f"{float(np.real(v)):.{prec}f}" if real else _fmt_complex(complex(v), prec)

    if arr.size <= count:
        return "[" + ", ".join(fmt(v) for v in arr) + "]"
    head = ", ".join(fmt(v) for v in arr[: count // 2])
    tail = ", ".join(fmt(v) for v in arr[-count // 2 :])
    return f"[{head}, ..., {tail}]  (n={arr.size})"


def _parse_number_list(text: str, *, allow_complex: bool = False) -> np.ndarray:
    """Parse a comma-separated list of numbers into an array.

    With ``allow_complex`` the tokens may be complex literals such as ``1+2j``.
    """
    tokens = [tok.strip() for tok in text.replace(" ", "").split(",") if tok.strip()]
    if not tokens:
        raise InvalidParameterError("empty number list")
    try:
        if allow_complex:
            values = [complex(tok) for tok in tokens]
            arr = np.array(values, dtype=complex)
            if np.allclose(arr.imag, 0.0):
                arr = arr.real.astype(float)
            return arr
        return np.array([float(tok) for tok in tokens], dtype=float)
    except ValueError as exc:
        raise InvalidParameterError(f"could not parse number list {text!r}: {exc}")


def _print_coefficient_table(
    spectrum: np.ndarray, sample_rate: float, max_rows: int = 16
) -> None:
    """Print a per-bin table of DFT coefficients (truncated for long spectra)."""
    n = spectrum.shape[0]
    freqs = fft_freqs(n, sample_rate)
    mag = magnitude_spectrum(spectrum)
    phase = phase_spectrum(spectrum)

    print(f"{'k':>4} {'freq[Hz]':>11} {'real':>11} {'imag':>11} "
          f"{'|X_k|':>10} {'phase[rad]':>11}")
    if n <= max_rows:
        rows = range(n)
    else:
        rows = list(range(max_rows // 2)) + [-1] + list(range(n - max_rows // 2, n))
    for k in rows:
        if k == -1:
            print(f"{'...':>4}")
            continue
        z = spectrum[k]
        print(f"{k:>4} {freqs[k]:>11.4g} {z.real:>11.4f} {z.imag:>11.4f} "
              f"{mag[k]:>10.4f} {phase[k]:>11.4f}")


def _print_dominant(spectrum: np.ndarray, sample_rate: float, count: int = 5) -> None:
    peaks = dominant_frequencies(spectrum, sample_rate, count=count)
    if not peaks:
        print("dominant frequencies:  (none above noise floor)")
        return
    print("dominant frequencies:")
    for peak in peaks:
        print(f"  {peak.frequency:8.3f} Hz   amplitude {peak.amplitude:.4f}")


# --------------------------------------------------------------------------- #
# Command handlers. Each returns a process exit code (0 == success).
# --------------------------------------------------------------------------- #
def cmd_series(args: argparse.Namespace) -> int:
    approx = approximate(
        args.function,
        num_terms=args.terms,
        method=args.method,
        num_points=args.points,
    )
    _header("Fourier Series")
    _kv("function", approx.name)
    _kv("terms (N)", approx.num_terms)
    _kv("method", args.method)
    _kv("mean sq. error", f"{approx.mse:.6e}")
    _kv("max error", f"{approx.max_error:.6e}")
    if approx.name == "square":
        _kv("note", "square wave shows Gibbs overshoot near its jumps")
    if args.save:
        from . import visualize

        visualize.plot_series(approx, save=args.save)
        _kv("saved plot", args.save)
    return 0


def cmd_gibbs(args: argparse.Namespace) -> int:
    term_counts = tuple(args.terms) if args.terms else (5, 15, 45)
    demo = gibbs_demo(term_counts=term_counts)
    _header("Gibbs Phenomenon")
    _kv("waveform", "square")
    _kv("term counts", ", ".join(map(str, term_counts)))
    print("\npeak overshoot (theory -> 1.1790):")
    for n in term_counts:
        print(f"  N={n:<4d} peak = {demo['overshoots'][n]:.4f}")
    if args.save:
        from . import visualize

        visualize.plot_gibbs(demo, save=args.save)
        _kv("saved plot", args.save)
    return 0


def _spectrum_report(signal: Signal, spectrum: np.ndarray, method: str) -> None:
    _kv("samples", signal.n)
    _kv("sample rate", f"{signal.sample_rate:g} Hz")
    _kv("method", method)
    _kv("freq resolution", f"{signal.sample_rate / signal.n:g} Hz/bin")
    _kv("Nyquist", f"{signal.sample_rate / 2:g} Hz")
    print()
    _print_dominant(spectrum, signal.sample_rate)


def cmd_dft(args: argparse.Namespace) -> int:
    x = _parse_number_list(args.signal, allow_complex=True)
    spectrum = dft(x)
    reconstruction = idft(spectrum)
    _header("DFT")
    _kv("input length", x.shape[0])
    _kv("input signal", _preview(x))
    _kv("sample rate", f"{args.sample_rate:g} Hz")
    _kv("complex input", "yes" if np.iscomplexobj(x) else "no")
    print()
    _print_coefficient_table(spectrum, args.sample_rate)
    print()
    _kv("magnitude", _preview(magnitude_spectrum(spectrum)))
    _kv("phase [rad]", _preview(phase_spectrum(spectrum)))
    _kv("reconstruction", _preview(reconstruction))
    _kv("reconstruct err", f"{max_error(x, reconstruction):.3e}")
    pv = parseval(x, spectrum)
    _kv("Parseval rel.err", f"{pv.relative_error:.3e}")
    print()
    _print_dominant(spectrum, args.sample_rate)
    if args.save:
        from . import visualize

        visualize.plot_spectrum(spectrum, args.sample_rate, save=args.save)
        _kv("saved plot", args.save)
    return 0


def cmd_idft(args: argparse.Namespace) -> int:
    X = _parse_number_list(args.spectrum, allow_complex=True)
    reconstruction = idft(X)
    _header("Inverse DFT")
    _kv("input length", X.shape[0])
    _kv("input spectrum", _preview(X))
    _kv("reconstruction", _preview(reconstruction))
    _kv("max |imag|", f"{np.max(np.abs(reconstruction.imag)):.3e}")
    return 0


def cmd_fft(args: argparse.Namespace) -> int:
    x = _parse_number_list(args.signal, allow_complex=True)
    n = x.shape[0]
    spectrum = fft(x)  # raises NotPowerOfTwoError with a helpful message
    reconstruction = ifft(spectrum)
    reference = np.fft.fft(x)  # oracle, only used to *report* agreement
    _header("FFT")
    _kv("input length", n)
    _kv("sample rate", f"{args.sample_rate:g} Hz")
    _kv("method", "recursive Cooley-Tukey FFT")
    _kv("power of two", "yes" if is_power_of_two(n) else "no")
    _kv("max NumPy error", f"{max_error(spectrum, reference):.3e}")
    _kv("reconstruct err", f"{max_error(x, reconstruction):.3e}")
    print()
    _print_coefficient_table(spectrum, args.sample_rate)
    print()
    _print_dominant(spectrum, args.sample_rate)
    if args.save:
        from . import visualize

        visualize.plot_spectrum(spectrum, args.sample_rate, save=args.save)
        _kv("saved plot", args.save)
    return 0


def cmd_ifft(args: argparse.Namespace) -> int:
    X = _parse_number_list(args.spectrum, allow_complex=True)
    reconstruction = ifft(X)
    _header("Inverse FFT")
    _kv("input length", X.shape[0])
    _kv("input spectrum", _preview(X))
    _kv("reconstruction", _preview(reconstruction))
    _kv("max |imag|", f"{np.max(np.abs(reconstruction.imag)):.3e}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    results = benchmark(args.sizes, repeats=args.repeats)
    _header("Compare (DFT vs FFT)")
    print(f"{'N':>7} {'DFT [ms]':>12} {'FFT [ms]':>12} {'speedup':>10}")
    for r in results:
        print(f"{r.size:>7} {r.dft_seconds * 1e3:>12.3f} "
              f"{r.fft_seconds * 1e3:>12.3f} {r.speedup:>9.1f}x")
    print("\nDFT cost grows ~O(N^2); FFT cost grows ~O(N log N).")
    if args.save:
        from . import visualize

        visualize.plot_comparison(results, save=args.save)
        _kv("saved plot", args.save)
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    freqs = _parse_number_list(args.freqs).tolist() if args.freqs else None
    amps = _parse_number_list(args.amps).tolist() if args.amps else None
    phases = _parse_number_list(args.phases).tolist() if args.phases else None

    signal = generate(
        args.kind,
        frequency=args.frequency,
        sample_rate=args.sample_rate,
        duration=args.duration,
        amplitude=args.amplitude,
        phase=args.phase,
        frequencies=freqs,
        amplitudes=amps,
        phases=phases,
        noise=args.noise,
        seed=args.seed,
    )
    _header("Generate")
    _kv("kind", args.kind)
    if args.kind == "mixed":
        _kv("frequencies", freqs)
        _kv("amplitudes", amps if amps is not None else "all 1.0")
    else:
        _kv("frequency", f"{args.frequency:g} Hz")
        _kv("amplitude", args.amplitude)
    _kv("sample rate", f"{args.sample_rate:g} Hz")
    _kv("duration", f"{args.duration:g} s")
    _kv("samples", signal.n)
    _kv("noise std", args.noise)
    if args.noise > 0:
        _kv("seed", args.seed)
    _kv("preview", _preview(signal.y))
    if args.save:
        path = signal.to_csv(args.save)
        _kv("saved", path)
    if args.plot:
        from . import visualize

        visualize.plot_signal(signal, save=args.plot)
        _kv("saved plot", args.plot)
    return 0


def cmd_spectrum(args: argparse.Namespace) -> int:
    signal = Signal.from_csv(args.input)
    y = np.real(signal.y)
    if args.method == "dft":
        spectrum = dft(y)
        method = "direct DFT"
    elif args.method == "fft":
        spectrum = fft(y)
        method = "recursive FFT"
    else:  # auto
        spectrum = forward_transform(y)
        method = "FFT" if is_power_of_two(signal.n) else "DFT (non-power-of-two)"
    _header("Spectrum")
    _kv("input", args.input)
    _spectrum_report(signal, spectrum, method)
    if args.save:
        from . import visualize

        visualize.plot_spectrum(
            spectrum, signal.sample_rate, title=f"Spectrum of {signal.name}",
            save=args.save,
        )
        _kv("saved plot", args.save)
    return 0


def cmd_filter(args: argparse.Namespace) -> int:
    signal = Signal.from_csv(args.input)
    result = apply_filter(
        signal,
        args.type,
        cutoff=args.cutoff,
        low=args.low,
        high=args.high,
    )
    _header("Filter")
    _kv("input", args.input)
    _kv("filter type", args.type)
    if args.type in ("lowpass", "highpass"):
        _kv("cutoff", f"{args.cutoff:g} Hz")
    else:
        _kv("band", f"{args.low:g} - {args.high:g} Hz")
    _kv("samples", signal.n)
    _kv("sample rate", f"{signal.sample_rate:g} Hz")
    _kv("bins removed", f"{result.removed_count} of {signal.n}")
    _kv("original", _preview(result.original.y))
    _kv("filtered", _preview(result.filtered.y))
    if args.export:
        path = result.filtered.to_csv(args.export)
        _kv("exported", path)
    if args.save:
        from . import visualize

        visualize.plot_filter(result, save=args.save)
        _kv("saved plot", args.save)
    return 0


# --------------------------------------------------------------------------- #
# Parser construction.
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fourierlab",
        description="FourierLab: a from-scratch Fourier / DFT / FFT toolkit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"FourierLab {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    # series
    p = sub.add_parser("series", help="approximate a periodic function by a Fourier series")
    p.add_argument("--function", "-f", default="square", choices=sorted(CANONICAL_FUNCTIONS),
                   help="waveform to approximate")
    p.add_argument("--terms", "-n", type=int, default=15, help="number of harmonics")
    p.add_argument("--method", default="numerical", choices=["numerical", "analytical"],
                   help="coefficient estimation method")
    p.add_argument("--points", type=int, default=2000, help="points used to render the curve")
    p.add_argument("--save", metavar="FILE.png", help="save a comparison plot")
    p.set_defaults(func=cmd_series)

    # gibbs
    p = sub.add_parser("gibbs", help="demonstrate the Gibbs phenomenon for a square wave")
    p.add_argument("--terms", "-n", type=int, nargs="+", help="term counts to overlay")
    p.add_argument("--save", metavar="FILE.png", help="save the Gibbs plot")
    p.set_defaults(func=cmd_gibbs)

    # dft
    p = sub.add_parser("dft", help="Discrete Fourier Transform of a signal")
    p.add_argument("--signal", "-s", required=True, help='comma-separated samples, e.g. "1,0,-1,0"')
    p.add_argument("--sample-rate", type=float, default=1.0, help="sampling rate in Hz")
    p.add_argument("--save", metavar="FILE.png", help="save the spectrum plot")
    p.set_defaults(func=cmd_dft)

    # idft
    p = sub.add_parser("idft", help="inverse DFT of a spectrum")
    p.add_argument("--spectrum", "-s", required=True, help='comma-separated coefficients (may be complex)')
    p.set_defaults(func=cmd_idft)

    # fft
    p = sub.add_parser("fft", help="Fast Fourier Transform (power-of-two length)")
    p.add_argument("--signal", "-s", required=True, help='comma-separated samples, e.g. "1,0,-1,0"')
    p.add_argument("--sample-rate", type=float, default=1.0, help="sampling rate in Hz")
    p.add_argument("--save", metavar="FILE.png", help="save the spectrum plot")
    p.set_defaults(func=cmd_fft)

    # ifft
    p = sub.add_parser("ifft", help="inverse FFT of a spectrum (power-of-two length)")
    p.add_argument("--spectrum", "-s", required=True, help="comma-separated coefficients (may be complex)")
    p.set_defaults(func=cmd_ifft)

    # compare
    p = sub.add_parser("compare", help="benchmark DFT vs FFT across signal sizes")
    p.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024],
                   help="power-of-two sizes to test")
    p.add_argument("--repeats", type=int, default=3, help="timing repeats per size (averaged)")
    p.add_argument("--save", metavar="FILE.png", help="save the runtime plot")
    p.set_defaults(func=cmd_compare)

    # generate
    p = sub.add_parser("generate", help="synthesize a signal and optionally save it as CSV")
    p.add_argument("--kind", "-k", default="sine", choices=sorted(WAVEFORMS) + ["mixed"],
                   help="waveform kind")
    p.add_argument("--frequency", type=float, default=5.0, help="frequency [Hz] (single-tone kinds)")
    p.add_argument("--amplitude", type=float, default=1.0, help="amplitude (single-tone kinds)")
    p.add_argument("--phase", type=float, default=0.0, help="phase [rad] (single-tone kinds)")
    p.add_argument("--freqs", help='comma list of frequencies for kind=mixed, e.g. "5,20,50"')
    p.add_argument("--amps", help='comma list of amplitudes for kind=mixed, e.g. "1,0.5,0.2"')
    p.add_argument("--phases", help="comma list of phases [rad] for kind=mixed")
    p.add_argument("--sample-rate", type=float, default=256.0, help="sampling rate [Hz]")
    p.add_argument("--duration", type=float, default=1.0, help="duration [s]")
    p.add_argument("--noise", type=float, default=0.0, help="additive Gaussian noise std")
    p.add_argument("--seed", type=int, default=0, help="random seed for the noise")
    p.add_argument("--save", metavar="FILE.csv", help="save the signal as CSV")
    p.add_argument("--plot", metavar="FILE.png", help="save a plot of the signal")
    p.set_defaults(func=cmd_generate)

    # spectrum
    p = sub.add_parser("spectrum", help="frequency spectrum of a signal loaded from CSV")
    p.add_argument("--input", "-i", required=True, help="input CSV signal file")
    p.add_argument("--method", default="auto", choices=["auto", "dft", "fft"],
                   help="transform to use")
    p.add_argument("--save", metavar="FILE.png", help="save the spectrum plot")
    p.set_defaults(func=cmd_spectrum)

    # filter
    p = sub.add_parser("filter", help="apply a frequency-domain filter to a CSV signal")
    p.add_argument("--input", "-i", required=True, help="input CSV signal file")
    p.add_argument("--type", "-t", required=True, choices=list(FILTER_TYPES),
                   help="filter type")
    p.add_argument("--cutoff", type=float, help="cutoff [Hz] for lowpass/highpass")
    p.add_argument("--low", type=float, help="lower band edge [Hz] for bandpass/bandstop")
    p.add_argument("--high", type=float, help="upper band edge [Hz] for bandpass/bandstop")
    p.add_argument("--save", metavar="FILE.png", help="save the before/after plot")
    p.add_argument("--export", metavar="FILE.csv", help="export the filtered signal as CSV")
    p.set_defaults(func=cmd_filter)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FourierLabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: file not found: {exc.filename}", file=sys.stderr)
        return 1
    except BrokenPipeError:  # pragma: no cover - e.g. piping into head
        return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
