"""Tests for the frequency-domain filters.

A three-tone test signal (5, 20, 50 Hz) is filtered and the recovered amplitude
at each tone is checked, confirming that each filter passes and rejects exactly
the right components while preserving length and realness.
"""

from __future__ import annotations

import numpy as np
import pytest

from fourierlab.errors import InvalidCutoffError
from fourierlab.fft import forward_transform
from fourierlab.filters import apply_filter, frequency_mask
from fourierlab.math_utils import amplitude_spectrum, fft_freqs
from fourierlab.signals import Signal, mixed, sine

# 5 Hz, 20 Hz, 50 Hz tones. fs = 256, dur = 2 -> 512 samples (power of two).
TONES = (5.0, 20.0, 50.0)
AMPS = (1.0, 0.5, 0.25)


@pytest.fixture()
def three_tone() -> Signal:
    return mixed(TONES, AMPS, sample_rate=256.0, duration=2.0)


def amplitude_at(signal: Signal, freq: float) -> float:
    """Recover the single-sided amplitude of ``signal`` at frequency ``freq``."""
    spectrum = forward_transform(np.real(signal.y))
    amp = amplitude_spectrum(spectrum, signal.n)
    freqs = fft_freqs(signal.n, signal.sample_rate)[: amp.shape[0]]
    return float(amp[np.argmin(np.abs(freqs - freq))])


def test_lowpass_removes_high_frequencies(three_tone: Signal) -> None:
    result = apply_filter(three_tone, "lowpass", cutoff=30.0)
    out = result.filtered
    assert amplitude_at(out, 5.0) == pytest.approx(1.0, abs=1e-6)
    assert amplitude_at(out, 20.0) == pytest.approx(0.5, abs=1e-6)
    assert amplitude_at(out, 50.0) < 1e-9


def test_highpass_removes_low_frequencies(three_tone: Signal) -> None:
    result = apply_filter(three_tone, "highpass", cutoff=30.0)
    out = result.filtered
    assert amplitude_at(out, 5.0) < 1e-9
    assert amplitude_at(out, 20.0) < 1e-9
    assert amplitude_at(out, 50.0) == pytest.approx(0.25, abs=1e-6)


def test_bandpass_keeps_only_the_band(three_tone: Signal) -> None:
    result = apply_filter(three_tone, "bandpass", low=10.0, high=30.0)
    out = result.filtered
    assert amplitude_at(out, 5.0) < 1e-9
    assert amplitude_at(out, 20.0) == pytest.approx(0.5, abs=1e-6)
    assert amplitude_at(out, 50.0) < 1e-9


def test_bandstop_removes_only_the_band(three_tone: Signal) -> None:
    result = apply_filter(three_tone, "bandstop", low=10.0, high=30.0)
    out = result.filtered
    assert amplitude_at(out, 5.0) == pytest.approx(1.0, abs=1e-6)
    assert amplitude_at(out, 20.0) < 1e-9
    assert amplitude_at(out, 50.0) == pytest.approx(0.25, abs=1e-6)


def test_filtered_length_matches_original(three_tone: Signal) -> None:
    result = apply_filter(three_tone, "lowpass", cutoff=30.0)
    assert result.filtered.n == three_tone.n
    assert result.filtered.sample_rate == three_tone.sample_rate
    assert np.allclose(result.filtered.t, three_tone.t)


def test_filtered_signal_is_real(three_tone: Signal) -> None:
    result = apply_filter(three_tone, "bandpass", low=10.0, high=30.0)
    assert not np.iscomplexobj(result.filtered.y)
    assert np.all(np.isfinite(result.filtered.y))


def test_filter_on_non_power_of_two_length() -> None:
    # 300 samples is not a power of two; the DFT fallback must still work.
    signal = mixed(TONES, AMPS, sample_rate=150.0, duration=2.0)
    assert signal.n == 300
    result = apply_filter(signal, "lowpass", cutoff=30.0)
    assert result.filtered.n == 300
    assert amplitude_at(result.filtered, 50.0) < 1e-9


def test_removed_count_is_reported(three_tone: Signal) -> None:
    result = apply_filter(three_tone, "lowpass", cutoff=30.0)
    assert result.removed_count > 0
    assert result.removed_count + result.kept_count == three_tone.n


def test_lowpass_of_single_tone_is_identity() -> None:
    signal = sine(frequency=5.0, sample_rate=128.0, duration=1.0)
    result = apply_filter(signal, "lowpass", cutoff=20.0)
    assert np.allclose(result.filtered.y, signal.y, atol=1e-9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"filter_type": "lowpass", "cutoff": -5.0},
        {"filter_type": "lowpass", "cutoff": 0.0},
        {"filter_type": "highpass", "cutoff": 10_000.0},  # above Nyquist
        {"filter_type": "lowpass"},                        # missing cutoff
        {"filter_type": "bandpass", "low": 30.0, "high": 10.0},  # low >= high
        {"filter_type": "bandpass", "low": 10.0},          # missing high
        {"filter_type": "notch", "cutoff": 10.0},          # unknown type
    ],
)
def test_invalid_filter_parameters_raise(three_tone: Signal, kwargs: dict) -> None:
    with pytest.raises(InvalidCutoffError):
        apply_filter(three_tone, **kwargs)


def test_frequency_mask_is_conjugate_symmetric() -> None:
    # A mask built on |f| must be symmetric so the output stays real.
    from fourierlab.filters import _Band

    freqs = fft_freqs(16, sample_rate=16.0)
    mask = frequency_mask(freqs, "lowpass", _Band(cutoff=4.0, low=None, high=None))
    # bin k and bin N-k carry conjugate frequencies and must share a decision.
    for k in range(1, 8):
        assert mask[k] == mask[16 - k]
