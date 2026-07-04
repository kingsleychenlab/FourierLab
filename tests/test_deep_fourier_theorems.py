import numpy as np
from fourierlab import fft, ifft, dft, idft


ATOL = 1e-9
RTOL = 1e-9


def circular_convolution(x, h):
    n = len(x)
    y = np.zeros(n, dtype=complex)
    for i in range(n):
        for j in range(n):
            y[i] += x[j] * h[(i - j) % n]
    return y


def test_complex_exponential_maps_to_single_frequency_bin():
    for n in [8, 16, 32, 64, 128]:
        for k0 in [1, 3, n // 4]:
            t = np.arange(n)
            x = np.exp(2j * np.pi * k0 * t / n)
            X = fft(x)

            expected = np.zeros(n, dtype=complex)
            expected[k0] = n

            assert np.allclose(X, expected, atol=ATOL, rtol=RTOL)


def test_impulse_transforms_to_flat_spectrum():
    for n in [8, 16, 32, 64, 128]:
        x = np.zeros(n, dtype=complex)
        x[0] = 1

        assert np.allclose(fft(x), np.ones(n), atol=ATOL, rtol=RTOL)


def test_shifted_impulse_has_phase_ramp():
    for n in [8, 16, 32, 64, 128]:
        shift = n // 4
        x = np.zeros(n, dtype=complex)
        x[shift] = 1

        k = np.arange(n)
        expected = np.exp(-2j * np.pi * k * shift / n)

        assert np.allclose(fft(x), expected, atol=ATOL, rtol=RTOL)


def test_time_shift_theorem():
    rng = np.random.default_rng(2026)

    for n in [16, 32, 64, 128]:
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        shift = 5

        X = fft(x)
        Y = fft(np.roll(x, shift))

        k = np.arange(n)
        expected = X * np.exp(-2j * np.pi * k * shift / n)

        assert np.allclose(Y, expected, atol=ATOL, rtol=RTOL)


def test_frequency_shift_modulation_theorem():
    rng = np.random.default_rng(2027)

    for n in [16, 32, 64, 128]:
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        r = 3

        X = fft(x)
        t = np.arange(n)

        modulated = x * np.exp(2j * np.pi * r * t / n)
        Y = fft(modulated)

        assert np.allclose(Y, np.roll(X, r), atol=ATOL, rtol=RTOL)


def test_circular_convolution_theorem():
    rng = np.random.default_rng(2028)

    for n in [8, 16, 32, 64]:
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        h = rng.normal(size=n) + 1j * rng.normal(size=n)

        direct = circular_convolution(x, h)
        via_fft = ifft(fft(x) * fft(h))

        assert np.allclose(via_fft, direct, atol=ATOL, rtol=RTOL)


def test_parseval_energy_preservation_complex():
    rng = np.random.default_rng(2029)

    for n in [8, 16, 32, 64, 128, 256, 512]:
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        X = fft(x)

        time_energy = np.sum(np.abs(x) ** 2)
        freq_energy = np.sum(np.abs(X) ** 2) / n

        assert np.isclose(time_energy, freq_energy, atol=ATOL, rtol=RTOL)


def test_linearity_theorem():
    rng = np.random.default_rng(2030)

    for n in [16, 32, 64, 128]:
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        y = rng.normal(size=n) + 1j * rng.normal(size=n)

        a = -2.5 + 0.7j
        b = 3.2 - 1.1j

        assert np.allclose(
            fft(a * x + b * y),
            a * fft(x) + b * fft(y),
            atol=ATOL,
            rtol=RTOL,
        )


def test_hermitian_symmetry_for_real_signals():
    rng = np.random.default_rng(2031)

    for n in [16, 32, 64, 128]:
        x = rng.normal(size=n)
        X = fft(x)

        for k in range(1, n):
            assert np.allclose(X[k], np.conj(X[-k]), atol=ATOL, rtol=RTOL)


def test_spectral_differentiation_periodic_function():
    for n in [64, 128, 256, 512]:
        t = 2 * np.pi * np.arange(n) / n

        f = np.sin(3 * t) + 0.5 * np.cos(5 * t)
        exact = 3 * np.cos(3 * t) - 2.5 * np.sin(5 * t)

        F = fft(f)
        freqs = np.fft.fftfreq(n, d=1 / n)

        derivative = ifft(1j * freqs * F).real

        assert np.allclose(derivative, exact, atol=1e-8, rtol=1e-8)


def test_large_dynamic_range_complex_signal():
    rng = np.random.default_rng(2032)

    for n in [32, 64, 128, 256]:
        magnitudes = 10 ** rng.uniform(-8, 8, size=n)
        phases = rng.uniform(0, 2 * np.pi, size=n)

        x = magnitudes * np.exp(1j * phases)

        assert np.allclose(fft(x), np.fft.fft(x), atol=1e-5, rtol=1e-9)


def test_dft_non_power_of_two_lengths():
    rng = np.random.default_rng(2033)

    for n in [7, 11, 13, 17, 29, 31]:
        x = rng.normal(size=n) + 1j * rng.normal(size=n)

        X = dft(x)

        assert np.allclose(X, np.fft.fft(x), atol=ATOL, rtol=RTOL)
        assert np.allclose(idft(X), x, atol=ATOL, rtol=RTOL)
