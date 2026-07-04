"""Tests for signal generation and CSV round-tripping."""

from __future__ import annotations

import numpy as np
import pytest

from fourierlab.errors import InvalidParameterError, InvalidSignalError
from fourierlab.signals import (
    Signal,
    add_noise,
    cosine,
    generate,
    mixed,
    sawtooth,
    sine,
    square,
    triangle,
)


def test_time_axis_and_rate() -> None:
    s = sine(frequency=5.0, sample_rate=100.0, duration=2.0)
    assert s.n == 200
    assert s.sample_rate == 100.0
    assert s.duration == pytest.approx(2.0)
    assert s.dt == pytest.approx(0.01)
    # First sample sits at t = 0, spacing exactly 1/fs.
    assert s.t[0] == 0.0
    assert np.allclose(np.diff(s.t), 0.01)


def test_sine_amplitude_and_shape() -> None:
    s = sine(frequency=1.0, sample_rate=1000.0, duration=1.0, amplitude=2.5)
    assert np.max(s.y) == pytest.approx(2.5, abs=1e-2)
    assert np.min(s.y) == pytest.approx(-2.5, abs=1e-2)


def test_cosine_starts_at_amplitude() -> None:
    s = cosine(frequency=3.0, sample_rate=256.0, duration=1.0, amplitude=1.0)
    assert s.y[0] == pytest.approx(1.0)


@pytest.mark.parametrize("gen", [square])
def test_square_is_two_valued(gen) -> None:
    s = gen(frequency=4.0, sample_rate=256.0, duration=1.0, amplitude=1.0)
    assert set(np.unique(np.round(s.y, 6))).issubset({-1.0, 1.0})


@pytest.mark.parametrize("gen", [sine, cosine, square, sawtooth, triangle])
def test_waveforms_stay_within_amplitude(gen) -> None:
    amp = 1.3
    s = gen(frequency=5.0, sample_rate=512.0, duration=1.0, amplitude=amp)
    assert np.max(np.abs(s.y)) <= amp + 1e-9


def test_triangle_peaks_at_amplitude() -> None:
    s = triangle(frequency=2.0, sample_rate=2048.0, duration=1.0, amplitude=1.0)
    assert np.max(s.y) == pytest.approx(1.0, abs=1e-2)
    assert np.min(s.y) == pytest.approx(-1.0, abs=1e-2)


def test_mixed_is_superposition() -> None:
    fs, dur = 256.0, 1.0
    a = sine(5.0, fs, dur, amplitude=1.0)
    b = sine(20.0, fs, dur, amplitude=0.5)
    combined = mixed([5.0, 20.0], [1.0, 0.5], sample_rate=fs, duration=dur)
    assert np.allclose(combined.y, a.y + b.y, atol=1e-12)


def test_mixed_length_mismatch_raises() -> None:
    with pytest.raises(InvalidParameterError):
        mixed([5.0, 20.0], [1.0], sample_rate=256.0, duration=1.0)


def test_noise_is_reproducible_with_seed() -> None:
    base = sine(5.0, 256.0, 1.0)
    n1 = add_noise(base, 0.2, seed=42)
    n2 = add_noise(base, 0.2, seed=42)
    n3 = add_noise(base, 0.2, seed=7)
    assert np.allclose(n1.y, n2.y)
    assert not np.allclose(n1.y, n3.y)


def test_noise_increases_variance() -> None:
    base = sine(5.0, 512.0, 1.0)
    noisy = add_noise(base, 0.5, seed=0)
    assert np.var(noisy.y) > np.var(base.y)


def test_generate_dispatch_single_and_mixed() -> None:
    s1 = generate("sine", frequency=10.0, sample_rate=128.0, duration=1.0)
    assert s1.n == 128
    s2 = generate(
        "mixed", frequencies=[5.0, 20.0], amplitudes=[1.0, 0.5],
        sample_rate=128.0, duration=1.0,
    )
    assert s2.n == 128


def test_generate_unknown_kind_raises() -> None:
    with pytest.raises(InvalidParameterError):
        generate("wobble", frequency=5.0)


def test_generate_mixed_without_freqs_raises() -> None:
    with pytest.raises(InvalidParameterError):
        generate("mixed")


def test_csv_roundtrip(tmp_path) -> None:
    original = mixed([5.0, 20.0], [1.0, 0.5], sample_rate=200.0, duration=1.0)
    path = tmp_path / "sig.csv"
    original.to_csv(path)
    assert path.exists()

    loaded = Signal.from_csv(path)
    assert loaded.n == original.n
    assert loaded.sample_rate == pytest.approx(original.sample_rate)
    assert np.allclose(loaded.y, np.real(original.y), atol=1e-9)
    assert np.allclose(loaded.t, original.t, atol=1e-9)


def test_from_csv_infers_rate_without_comment(tmp_path) -> None:
    path = tmp_path / "plain.csv"
    path.write_text("t,y\n0.0,0.0\n0.5,1.0\n1.0,0.0\n1.5,-1.0\n")
    loaded = Signal.from_csv(path)
    assert loaded.n == 4
    assert loaded.sample_rate == pytest.approx(2.0)  # dt = 0.5 -> 2 Hz


def test_from_csv_missing_file_raises(tmp_path) -> None:
    with pytest.raises(InvalidSignalError):
        Signal.from_csv(tmp_path / "does_not_exist.csv")


def test_invalid_signal_construction_raises() -> None:
    with pytest.raises(InvalidSignalError):
        Signal(t=np.array([0.0, 1.0]), y=np.array([1.0]), sample_rate=1.0)
    with pytest.raises(InvalidSignalError):
        Signal(t=np.array([]), y=np.array([]), sample_rate=1.0)


def test_invalid_duration_raises() -> None:
    with pytest.raises(InvalidParameterError):
        sine(frequency=5.0, sample_rate=100.0, duration=0.0)
