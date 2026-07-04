import numpy as np
from fourierlab import fft, ifft, dft, idft


def test_fft_matches_numpy_complex():
    rng = np.random.default_rng(42)
    for n in [2, 4, 8, 16, 32, 64, 128, 256]:
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        assert np.allclose(fft(x), np.fft.fft(x), atol=1e-9)


def test_dft_matches_numpy_complex():
    rng = np.random.default_rng(123)
    for n in [2, 3, 4, 5, 8, 12, 16, 31]:
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        assert np.allclose(dft(x), np.fft.fft(x), atol=1e-9)


def test_inverse_transforms_reconstruct_complex_signal():
    rng = np.random.default_rng(7)
    for n in [4, 8, 16, 32, 64, 128]:
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        assert np.allclose(ifft(fft(x)), x, atol=1e-9)

    for n in [3, 5, 9, 17, 31]:
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        assert np.allclose(idft(dft(x)), x, atol=1e-9)


def test_parseval_theorem_complex_signal():
    rng = np.random.default_rng(2026)
    for n in [8, 16, 32, 64, 128, 256]:
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        X = fft(x)

        time_energy = np.sum(np.abs(x) ** 2)
        freq_energy = np.sum(np.abs(X) ** 2) / n

        assert np.isclose(time_energy, freq_energy, atol=1e-9)


def test_dft_fft_consistency_complex():
    rng = np.random.default_rng(99)
    for n in [4, 8, 16, 32, 64, 128]:
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        assert np.allclose(dft(x), fft(x), atol=1e-9)
