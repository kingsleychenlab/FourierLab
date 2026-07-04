# FourierLab Examples

Every image in this folder was produced by the FourierLab CLI itself — the exact
commands are listed below, so you can regenerate them and tweak the parameters.
All commands assume you are in the repository root with the package installed
(`pip install -e .`).

---

## 1. Fourier series of a square wave

![Fourier series of a square wave](fourier_series_square.png)

```bash
python -m fourierlab series --function square --terms 25 --save examples/fourier_series_square.png
```

A 25-harmonic partial sum tracks the square wave well on the flat sections but
**overshoots near the jumps** — the Gibbs phenomenon. The lower panel shows the
pointwise error, which spikes at the two discontinuities (`x = 0` and `x = ±π`).

Try `--function triangle` or `--function sawtooth`, and vary `--terms`.

---

## 2. The Gibbs phenomenon

![Gibbs phenomenon](gibbs_phenomenon.png)

```bash
python -m fourierlab gibbs --terms 5 15 45 120 --save examples/gibbs_phenomenon.png
```

Overlaying partial sums with increasing term counts makes the key point visible:
the overshoot near a discontinuity **narrows** as `N` grows but its **height does
not shrink** — it converges to about `1.179` (≈9% of the unit jump).

---

## 3. Frequency spectrum

![Frequency spectrum](spectrum.png)

```bash
python -m fourierlab generate --kind mixed --freqs 5,20,50 --amps 1,0.5,0.2 \
    --noise 0.1 --save data/sample_signal.csv
python -m fourierlab spectrum --input data/sample_signal.csv --save examples/spectrum.png
```

The single-sided amplitude spectrum recovers the three planted tones at 5, 20,
and 50 Hz with roughly their true amplitudes (1.0, 0.5, 0.2); the added noise
forms a low floor across the band. The lower panel is the phase spectrum.

---

## 4. Low-pass filtering

![Low-pass filter](filter_lowpass.png)

```bash
python -m fourierlab filter --input data/sample_signal.csv --type lowpass \
    --cutoff 20 --save examples/filter_lowpass.png --export filtered.csv
```

Everything above 20 Hz is removed in the frequency domain, so the 50 Hz tone and
most of the noise disappear while the 5 and 20 Hz tones survive. Top: signal
before/after in time; bottom: amplitude spectrum before/after.

Swap in `--type highpass`, `--type bandpass --low 10 --high 30`, or
`--type bandstop --low 10 --high 30`.

---

## 5. DFT vs FFT runtime

![DFT vs FFT scaling](dft_vs_fft.png)

```bash
python -m fourierlab compare --sizes 32 64 128 256 512 1024 2048 --save examples/dft_vs_fft.png
```

On log-log axes the direct DFT climbs with slope ≈ 2 (`O(N²)`) while the FFT
climbs with slope ≈ 1 (`O(N log N)`), tracking the dotted complexity guides. The
gap widens with `N`: this is *why* the FFT matters.

---

## 6. A synthesized mixed signal

![Mixed signal](signal_mixed.png)

```bash
python -m fourierlab generate --kind mixed --freqs 5,20,50 --amps 1,0.5,0.2 \
    --noise 0.15 --seed 1 --plot examples/signal_mixed.png
```

The time-domain view of a three-tone signal with additive Gaussian noise — the
kind of input the spectrum and filter commands consume. The `--seed` flag makes
the noise reproducible.
