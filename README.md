# FourierLab

**A computational mathematics toolkit for Fourier series, discrete Fourier
transforms, fast Fourier transforms, signal reconstruction, spectrum analysis,
and frequency-domain filtering.**

FourierLab implements the **DFT and FFT from scratch** (no `numpy.fft` in the
core), validates its results against NumPy, demonstrates the **Gibbs
phenomenon**, compares **O(N²) vs O(N log N)** algorithms head to head, and
exports clean visualizations for signals and spectra. It is built to be *read*
as much as *run* — a learning tool for students of Fourier analysis, numerical
methods, and signal processing.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-163%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Dependencies](https://img.shields.io/badge/deps-numpy%20%2B%20matplotlib-lightgrey)

```bash
python -m fourierlab fft --signal "1,0,-1,0"
python -m fourierlab series --function square --terms 25 --save square.png
python -m fourierlab compare --sizes 32 64 128 256 512 1024
```

If you have never met the Fourier transform before, start at
[The big idea](#the-big-idea). If you just want to run things, jump to
[Installation](#installation) and the [5-minute tour](#a-5-minute-tour).

---

## Table of contents

- [The big idea](#the-big-idea)
- [Why Fourier analysis matters](#why-fourier-analysis-matters)
- [Installation](#installation)
- [A 5-minute tour](#a-5-minute-tour)
- [Command reference](#command-reference)
  - [`generate`](#generate--make-a-signal) · [`spectrum`](#spectrum--see-the-frequencies) · [`filter`](#filter--keep-or-remove-frequencies) · [`dft` / `fft`](#dft--fft--transform-an-explicit-signal) · [`series`](#series--approximate-a-waveform) · [`gibbs`](#gibbs--the-overshoot-near-a-jump) · [`compare`](#compare--dft-vs-fft-speed)
- [The mathematics explained](#the-mathematics-explained)
  - [Signals, sampling & the Nyquist limit](#1-signals-sampling--the-nyquist-limit)
  - [The Fourier series](#2-the-fourier-series)
  - [The DFT (worked by hand)](#3-the-dft-worked-by-hand)
  - [The FFT (worked by hand)](#4-the-fft-worked-by-hand)
  - [Magnitude & phase spectra](#5-magnitude--phase-spectra)
  - [Frequency-domain filtering](#6-frequency-domain-filtering)
  - [The Gibbs phenomenon](#7-the-gibbs-phenomenon)
- [Glossary](#glossary)
- [Example gallery](#example-gallery)
- [Using FourierLab as a library](#using-fourierlab-as-a-library)
- [How the code is organized](#how-the-code-is-organized)
- [Testing](#testing)
- [Project structure](#project-structure)
- [Limitations](#limitations)
- [Future work](#future-work)
- [License](#license)

---

## The big idea

Shine white light through a **prism** and it fans out into a rainbow — the prism
separates the light into the pure colours (frequencies) it was secretly made of.

The Fourier transform is a mathematical prism. Give it a signal that changes
over **time** — a sound, a heartbeat, a voltage — and it tells you which pure
**frequencies** that signal is built from and how strong each one is.

We call these two ways of looking at the same signal the **time domain** and the
**frequency domain**:

```
   TIME DOMAIN                          FREQUENCY DOMAIN
   (what you measure)                   (what it's made of)

   amplitude                            strength
      ^   /\      /\                        ^
      |  /  \    /  \        Fourier        |     |
      | /    \  /    \      ────────►       |     |        |
      |/      \/      \                     |     |        |
      +----------------> time               +-----+--------+----> frequency
                                                 5 Hz     20 Hz
```

Everything in FourierLab is a variation on moving between these two pictures:

- **Fourier series** — build a periodic signal by *adding up* sine and cosine waves.
- **DFT / FFT** — *analyse* a sampled signal into its frequencies (and back again).
- **Spectrum** — *plot* the strength of each frequency.
- **Filtering** — *edit* a signal by turning specific frequencies up or down.

The rest of this README explains each of these from scratch, with worked
examples you can reproduce.

---

## Why Fourier analysis matters

The idea that "any signal is a sum of simple oscillations" turns out to power an
enormous amount of technology:

- **Audio & music** — equalizers, pitch detection, MP3/AAC compression, noise removal.
- **Images & video** — JPEG and MPEG both compress using frequency-domain transforms.
- **Communications** — Wi-Fi, 4G/5G, and DSL split data across frequency bands.
- **Science & engineering** — vibration analysis, spectroscopy, MRI, radar, sonar.
- **Numerical methods** — fast convolution, solving differential equations, interpolation.

The **Fast Fourier Transform** — which turns an `O(N²)` computation into
`O(N log N)` — is regularly named one of the most important algorithms ever
written. FourierLab lets you read the ~40 lines that *are* the FFT, run them, and
watch them beat the slow DFT on a plot.

---

## Installation

FourierLab targets **Python 3.10+** and depends only on **NumPy** and
**Matplotlib** (plus **pytest** for the test suite).

```bash
# 1. Clone
git clone https://github.com/yourname/FourierLab.git
cd FourierLab

# 2. (Recommended) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate

# 3. Install the package (editable install for development)
pip install -e .                   # or: pip install -e ".[test]" for tests
```

Verify it works:

```bash
python -m fourierlab --help
python -m fourierlab --version
```

You can invoke the tool as `python -m fourierlab ...` or via the installed
console script `fourierlab ...`. Every plotting command saves a PNG headlessly,
so it works over SSH or in CI with no display attached.

---

## A 5-minute tour

Run these four commands in order. Together they cover generation, analysis,
filtering, and the FFT's whole reason for existing.

```bash
# 1. Synthesize a noisy three-tone signal (5, 20, 50 Hz) and save it as CSV.
python -m fourierlab generate --kind mixed --freqs 5,20,50 --amps 1,0.5,0.2 \
    --noise 0.1 --save data/sample_signal.csv

# 2. Look at WHICH frequencies are in it.
python -m fourierlab spectrum --input data/sample_signal.csv --save spectrum.png

# 3. Low-pass filter it: keep <= 20 Hz, so the 50 Hz tone and most noise vanish.
python -m fourierlab filter --input data/sample_signal.csv --type lowpass \
    --cutoff 20 --save filtered.png --export filtered.csv

# 4. Watch the FFT crush the DFT as the signal grows.
python -m fourierlab compare --sizes 32 64 128 256 512 1024 --save speed.png
```

Step 2 prints something like this — and the numbers are worth understanding:

```text
FourierLab :: Spectrum

input              = data/sample_signal.csv
samples            = 256              ← number of data points (N)
sample rate        = 256 Hz          ← how many samples per second
method             = FFT             ← N is a power of two, so the fast path is used
freq resolution    = 1 Hz/bin        ← smallest frequency difference we can resolve (fs / N)
Nyquist            = 128 Hz          ← highest frequency the data can represent (fs / 2)

dominant frequencies:
     5.000 Hz   amplitude 0.9980     ← recovered the planted 5 Hz tone (amp ≈ 1.0)
    20.000 Hz   amplitude 0.5049     ← recovered the 20 Hz tone (amp ≈ 0.5)
    50.000 Hz   amplitude 0.1946     ← recovered the 50 Hz tone (amp ≈ 0.2)
    33.000 Hz   amplitude 0.0301     ← noise: a small, broadband floor
```

The transform found exactly the three tones we planted, with roughly their true
amplitudes, and reported the added noise as a low floor. That is Fourier
analysis doing its job.

---

## Command reference

| Command | What it does |
|---|---|
| [`generate`](#generate--make-a-signal) | Synthesize sine/cosine/square/sawtooth/triangle/mixed signals (± noise) |
| [`spectrum`](#spectrum--see-the-frequencies) | Magnitude + phase spectrum of a signal from CSV |
| [`filter`](#filter--keep-or-remove-frequencies) | Low / high / band-pass / band-stop filtering |
| [`dft`](#dft--fft--transform-an-explicit-signal) / [`idft`](#dft--fft--transform-an-explicit-signal) | Discrete Fourier Transform and its inverse (from scratch) |
| [`fft`](#dft--fft--transform-an-explicit-signal) / [`ifft`](#dft--fft--transform-an-explicit-signal) | Fast Fourier Transform and its inverse (from scratch) |
| [`series`](#series--approximate-a-waveform) | Approximate a periodic function by an N-term Fourier series |
| [`gibbs`](#gibbs--the-overshoot-near-a-jump) | Demonstrate the Gibbs overshoot near a discontinuity |
| [`compare`](#compare--dft-vs-fft-speed) | Benchmark DFT vs FFT across signal sizes |

Every command supports `--help`. Every plotting command supports `--save FILE.png`.

### `generate` — make a signal

Creates a signal you can then analyse or filter. Single-tone kinds use
`--frequency/--amplitude/--phase`; `--kind mixed` uses the comma-separated
`--freqs/--amps/--phases`.

```bash
# One 10 Hz sine, 100 samples/second, 1 second long
python -m fourierlab generate --kind sine --frequency 10 --sample-rate 100 \
    --duration 1 --save data/sine.csv

# A sum of three tones plus reproducible Gaussian noise
python -m fourierlab generate --kind mixed --freqs 5,20,50 --amps 1,0.5,0.2 \
    --noise 0.1 --seed 0 --save data/sample_signal.csv --plot signal.png
```

Key options: `--kind {sine,cosine,square,sawtooth,triangle,mixed}`,
`--sample-rate` (Hz), `--duration` (s), `--noise` (noise standard deviation),
`--seed` (makes the noise reproducible), `--save FILE.csv`, `--plot FILE.png`.

### `spectrum` — see the frequencies

Loads a CSV signal and shows its single-sided **amplitude** and **phase** spectra,
listing the dominant tones. Uses the FFT when the length is a power of two,
otherwise the DFT.

```bash
python -m fourierlab spectrum --input data/sample_signal.csv --save spectrum.png
```

### `filter` — keep or remove frequencies

Transforms the signal, zeros the unwanted frequency bins, and transforms back.

```bash
# Keep only frequencies at or below 20 Hz
python -m fourierlab filter -i data/sample_signal.csv -t lowpass --cutoff 20 \
    --save low.png --export low.csv

# Keep only the 10–30 Hz band
python -m fourierlab filter -i data/sample_signal.csv -t bandpass --low 10 --high 30 \
    --save band.png --export band.csv
```

`--type` is one of `lowpass`, `highpass`, `bandpass`, `bandstop`. Low/high-pass
take `--cutoff`; band-pass/stop take `--low` and `--high`. The report tells you
how many frequency bins were removed and previews the signal before and after.

### `dft` / `fft` — transform an explicit signal

For learning and quick experiments you can transform a signal typed right on the
command line. Values may be real or complex (`"1+1j,0,-1,0"`).

```bash
python -m fourierlab dft --signal "1,0,-1,0"
python -m fourierlab fft --signal "1,0,-1,0" --sample-rate 4
python -m fourierlab idft --spectrum "0,2,0,2"     # inverse DFT
python -m fourierlab ifft --spectrum "0,2,0,2"     # inverse FFT
```

The `dft` report — annotated:

```text
FourierLab :: DFT

input length       = 4
input signal       = [1.000, 0.000, -1.000, 0.000]
sample rate        = 1 Hz
complex input      = no

   k    freq[Hz]        real        imag      |X_k|  phase[rad]   ← one row per frequency bin
   0           0      0.0000      0.0000     0.0000      0.0000   ← k=0 is DC (the average)
   1        0.25      2.0000      0.0000     2.0000      0.0000   ← energy here: 1 cycle per window
   2        -0.5      0.0000     -0.0000     0.0000      0.0000   ← k=2 is the Nyquist bin
   3       -0.25      2.0000      0.0000     2.0000      0.0000   ← conjugate partner of bin 1

reconstruct err    = 1.369e-16      ← idft(dft(x)) returns x to machine precision
Parseval rel.err   = 0.000e+00      ← energy is conserved (see the maths section)

dominant frequencies:
     0.250 Hz   amplitude 1.0000    ← the signal is a unit cosine at fs/4
```

`fft` additionally reports `max NumPy error` — the largest difference between our
from-scratch FFT and `numpy.fft.fft`, typically ~1e-15, i.e. they agree to
machine precision. `fft` requires a **power-of-two** length and gives a clear
error otherwise.

### `series` — approximate a waveform

Approximates a canonical periodic function by an `N`-harmonic Fourier series and
reports how good the fit is.

```bash
python -m fourierlab series --function square --terms 25 --save square.png
python -m fourierlab series --function triangle --terms 10 --method analytical
```

`--function` is one of `sine, cosine, square, sawtooth, triangle`; `--terms` sets
`N`; `--method {numerical,analytical}` chooses numerically-estimated coefficients
(works for any function) or the exact closed forms. The report gives the
**mean squared error** and **max error** of the approximation.

### `gibbs` — the overshoot near a jump

```bash
python -m fourierlab gibbs --terms 5 15 45 120 --save gibbs.png
```

Overlays square-wave partial sums at several term counts to show that the
overshoot near the discontinuity **narrows but never shrinks** (it converges to
≈ 1.179).

### `compare` — DFT vs FFT speed

```bash
python -m fourierlab compare --sizes 32 64 128 256 512 1024 2048 --save speed.png
```

Times both transforms at each size and prints the speed-up. The saved plot is
log-log, so the DFT's `O(N²)` slope (≈ 2) and the FFT's `O(N log N)` slope (≈ 1)
are visible as straight lines of different steepness.

### Signal CSV format

Signals are stored as a simple, self-describing two-column CSV:

```text
# FourierLab signal 'mixed(5Hz+20Hz+50Hz)'
# sample_rate=256.0
t,y
0.0,0.012573
0.00390625,0.533207
...
```

The `sample_rate` comment is authoritative; if it is missing (e.g. for
hand-written data) the rate is inferred from the spacing of the `t` column. Any
plain `t,y` CSV will load.

---

## The mathematics explained

Each section below follows the same shape: **intuition → formula → a worked
example you can reproduce → try it**.

### 1. Signals, sampling & the Nyquist limit

A **signal** here is a list of `N` numbers measured at a steady rate — the
**sample rate** `fs`, in samples per second (Hz). Sample `n` was taken at time
`t = n / fs`.

Sampling has a hard limit. A signal sampled at rate `fs` can only faithfully
represent frequencies **below half the sample rate**, the **Nyquist frequency**
`fs / 2`. A frequency above Nyquist does not vanish — it **aliases**, folding
down and disguising itself as a lower frequency (the same effect that makes wagon
wheels appear to spin backwards on film).

> **Try it:** sample a 90 Hz tone at only 100 Hz (Nyquist = 50 Hz) and the
> spectrum reports a phantom 10 Hz tone — that is aliasing.
>
> ```bash
> python -m fourierlab generate --kind sine --frequency 90 --sample-rate 100 \
>     --duration 1 --save aliased.csv
> python -m fourierlab spectrum --input aliased.csv
> ```

This is why `spectrum` only reports up to `fs/2`, and why the filters reject a
cutoff above Nyquist.

### 2. The Fourier series

**Intuition.** Any repeating (periodic) shape can be built by stacking sine and
cosine waves of the right heights. A few waves give a rough outline; more waves
sharpen the detail.

**Formula.** In real form, for a function with period `T` (and
$\omega = 2\pi / T$):

$$
f(x) \;\approx\; \frac{a_0}{2} + \sum_{n=1}^{N}
\Big[\, a_n \cos(n\omega x) + b_n \sin(n\omega x) \,\Big]
$$

The coefficients (shown for the standard period $T = 2\pi$, so $\omega = 1$) are

$$
a_n = \frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\cos(nx)\,dx,
\qquad
b_n = \frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\sin(nx)\,dx .
$$

$a_0/2$ is just the average value of $f$ over one period. Each $a_n$ / $b_n$ asks
"how much does $f$ look like a cosine / sine at frequency $n$?".

**Worked example — the first square-wave coefficient.** For the square wave
$f(x) = +1$ on $(0,\pi)$ and $-1$ on $(-\pi,0)$:

$$
b_1 = \frac{1}{\pi}\left[\int_{-\pi}^{0} (-1)\sin x\,dx + \int_{0}^{\pi} (+1)\sin x\,dx\right]
    = \frac{1}{\pi}\,(2 + 2) = \frac{4}{\pi} \approx 1.273 .
$$

Continuing gives the classic result — only odd harmonics survive:

$$
\text{square}(x) = \frac{4}{\pi}\left(\sin x + \tfrac{1}{3}\sin 3x + \tfrac{1}{5}\sin 5x + \cdots\right).
$$

| Waveform | Nonzero coefficients (period $2\pi$, unit amplitude) |
|---|---|
| sine     | $b_1 = 1$ |
| cosine   | $a_1 = 1$ |
| square   | $b_n = \dfrac{4}{n\pi}$ for odd $n$ |
| sawtooth | $b_n = \dfrac{2(-1)^{n+1}}{n\pi}$ |
| triangle | $a_n = \dfrac{8}{\pi^2 n^2}$ for odd $n$ |

FourierLab estimates these integrals numerically (so it works for *any* sampled
function) **and** carries the exact closed forms above, and its tests confirm the
two agree.

> **Try it:** `python -m fourierlab series --function square --terms 25 --save square.png`

### 3. The DFT (worked by hand)

**Intuition.** The Discrete Fourier Transform takes `N` samples and returns `N`
complex numbers `X_0 … X_{N-1}`, one per **frequency bin**. Each `X_k` says how
much of frequency-`k` is in the signal (its size) and how that wave is shifted
(its angle).

**Formula.**

$$
X_k = \sum_{n=0}^{N-1} x_n \, e^{-2\pi i k n / N},
\qquad
x_n = \frac{1}{N}\sum_{k=0}^{N-1} X_k \, e^{+2\pi i k n / N}\ \text{(inverse).}
$$

**Worked example — `x = [1, 0, -1, 0]`, `N = 4`.** Here
$e^{-2\pi i/4} = -i$, so $X_k = \sum_n x_n(-i)^{kn}$:

$$
\begin{aligned}
X_0 &= 1 + 0 - 1 + 0 = 0 \\
X_1 &= 1\cdot 1 + 0 + (-1)(-i)^2 + 0 = 1 + 1 = 2 \\
X_2 &= 1 + 0 + (-1)(-i)^4 + 0 = 1 - 1 = 0 \\
X_3 &= 1 + 0 + (-1)(-i)^6 + 0 = 1 + 1 = 2
\end{aligned}
\qquad\Rightarrow\qquad X = [\,0,\ 2,\ 0,\ 2\,].
$$

The energy sits in bins 1 and 3 (a conjugate pair), meaning the signal is a
single cosine completing **one cycle per 4 samples**. Indeed
$\cos(2\pi\cdot 1\cdot n/4) = [1, 0, -1, 0]$ — the transform recovered exactly
the wave the samples came from.

FourierLab computes this by building the matrix $W_{kn} = e^{-2\pi i kn/N}$ and
multiplying: $X = Wx$. That is `O(N²)` work — which is what the FFT fixes.

> **Try it:** `python -m fourierlab dft --signal "1,0,-1,0"` — you should see
> `[0, 2, 0, 2]`, and `idft` turns it straight back.

**Parseval's theorem** — a correctness check the tests use — says energy is the
same in both domains:

$$
\sum_{n=0}^{N-1} |x_n|^2 \;=\; \frac{1}{N}\sum_{k=0}^{N-1} |X_k|^2 .
$$

### 4. The FFT (worked by hand)

**Intuition.** The Fast Fourier Transform computes the *exact same* `X_k` as the
DFT, but cleverly. It splits the signal into its **even-indexed** and
**odd-indexed** samples, transforms each half, and stitches them together —
recursively. Reusing the shared work collapses the cost from `O(N²)` to
`O(N log N)`.

**Formula (radix-2 Cooley–Tukey).** With $E$ = DFT of the even samples, $O$ = DFT
of the odd samples, and twiddle factor $W_N^{\,k} = e^{-2\pi i k/N}$:

$$
X_k = E_k + W_N^{\,k}\,O_k, \qquad
X_{k+N/2} = E_k - W_N^{\,k}\,O_k .
$$

**Worked example — same `x = [1, 0, -1, 0]`.** Split into even indices
`[x₀, x₂] = [1, -1]` and odd indices `[x₁, x₃] = [0, 0]`.

- $E = \mathrm{DFT}([1,-1]) = [\,1+(-1),\ 1-(-1)\,] = [0, 2]$
- $O = \mathrm{DFT}([0,0]) = [0, 0]$
- Twiddles: $W_4^0 = 1,\ W_4^1 = -i$.

$$
\begin{aligned}
X_0 &= E_0 + W_4^0 O_0 = 0 + 0 = 0 \\
X_1 &= E_1 + W_4^1 O_1 = 2 + 0 = 2 \\
X_2 &= E_0 - W_4^0 O_0 = 0 - 0 = 0 \\
X_3 &= E_1 - W_4^1 O_1 = 2 - 0 = 2
\end{aligned}
\qquad\Rightarrow\qquad X = [\,0, 2, 0, 2\,] \checkmark
$$

Identical to the DFT, as it must be. The **inverse FFT** reuses the forward
transform via $\text{ifft}(X) = \tfrac{1}{N}\,\overline{\text{fft}(\overline{X})}$.

**Why it's faster.** The recursion has $\log_2 N$ levels, each doing `O(N)` work:

| N | DFT operations (`~N²`) | FFT operations (`~N log₂N`) | speed-up |
|---:|---:|---:|---:|
| 32 | 1,024 | 160 | 6× |
| 1,024 | ~1,000,000 | ~10,000 | ~100× |
| 1,048,576 | ~10¹² | ~2×10⁷ | ~50,000× |

The recursion needs a **power-of-two** length; FourierLab raises a clear error
otherwise (and its general dispatcher falls back to the DFT for other lengths).

> **Try it:** `python -m fourierlab compare --sizes 32 64 128 256 512 1024 2048 --save speed.png`

### 5. Magnitude & phase spectra

Each coefficient is a complex number, which you can read as a size and an angle:
$X_k = |X_k|\,e^{i\phi_k}$.

- The **magnitude spectrum** $|X_k|$ is the *strength* of each frequency.
  FourierLab also reports a **single-sided amplitude spectrum**, rescaled so that
  a pure tone of amplitude $A$ reads back with height $A$ (that is why the tour's
  5 Hz tone showed `amplitude ≈ 1.0`).
- The **phase spectrum** $\phi_k = \arg(X_k)$ is the *shift* of each frequency —
  whether it starts like a sine, a cosine, or somewhere in between. Phase for
  near-zero magnitudes is meaningless numerical noise, so FourierLab zeroes it for
  readability.

> **Try it:** `python -m fourierlab spectrum --input data/sample_signal.csv --save spectrum.png`
> — the top panel is magnitude, the bottom is phase.

### 6. Frequency-domain filtering

**Intuition.** To remove a hum or isolate a band, it is often easiest to *edit the
recipe*: transform to the frequency domain, turn the unwanted frequencies down to
zero, and transform back.

$$
x \;\xrightarrow{\ \text{FFT}\ }\; X \;\xrightarrow{\ \text{zero unwanted bins}\ }\; X' \;\xrightarrow{\ \text{IFFT}\ }\; x' .
$$

FourierLab provides four **ideal ("brick-wall")** filters, decided on the
absolute frequency $|f|$:

| Type | Keeps | Removes |
|---|---|---|
| `lowpass`  | $|f| \le$ cutoff | everything higher |
| `highpass` | $|f| \ge$ cutoff | everything lower (incl. DC) |
| `bandpass` | low $\le |f| \le$ high | outside the band |
| `bandstop` | outside the band | low $\le |f| \le$ high |

Deciding on $|f|$ keeps the mask symmetric between each bin and its conjugate
partner, which guarantees the filtered signal stays **real**.

> **Try it:** low-pass the three-tone signal at 20 Hz and the 50 Hz tone
> disappears while 5 and 20 Hz survive:
> `python -m fourierlab filter -i data/sample_signal.csv -t lowpass --cutoff 20 --save low.png`

### 7. The Gibbs phenomenon

**Intuition.** Sines and cosines are smooth, so a *finite* sum of them cannot make
a perfectly vertical jump. Near a discontinuity the partial sum **overshoots**,
and adding more terms only makes the overshoot **thinner**, never shorter.

For the unit square wave the peak converges to a specific constant:

$$
\frac{2}{\pi}\int_0^{\pi}\frac{\sin t}{t}\,dt = \frac{2}{\pi}\,\text{Si}(\pi) \approx 1.1790,
$$

an overshoot of about **9% of the jump** that persists forever. This is not a bug
in the approximation — it is a theorem about it.

> **Try it:** `python -m fourierlab gibbs --terms 5 15 45 120` — the reported peak
> hugs 1.179 no matter how many terms you add.

---

## Glossary

| Term | Meaning |
|---|---|
| **Sample rate** ($f_s$) | Samples measured per second (Hz). |
| **DC component** | The zero-frequency term — the signal's average value (bin 0). |
| **Bin** | One DFT output slot, i.e. one discrete frequency. Bin $k$ ↔ frequency $k\,f_s/N$. |
| **Frequency resolution** | Spacing between bins, $f_s/N$. Longer signals resolve finer detail. |
| **Nyquist frequency** | $f_s/2$: the highest frequency the samples can represent. |
| **Aliasing** | A frequency above Nyquist masquerading as a lower one. |
| **Magnitude / amplitude** | How strong a frequency component is, $|X_k|$. |
| **Phase** | How a frequency component is shifted, $\arg(X_k)$. |
| **Twiddle factor** | The complex weight $W_N^k = e^{-2\pi i k/N}$ used to combine FFT halves. |
| **Spectral leakage** | Smearing of energy across bins when a tone doesn't fit a whole number of cycles in the window. |
| **Parseval's theorem** | Total energy is equal in the time and frequency domains. |

---

## Example gallery

See [`examples/`](examples/README.md) for the exact command behind each image.

| | |
|---|---|
| **Fourier series (square, N=25)** | **Gibbs phenomenon** |
| ![series](examples/fourier_series_square.png) | ![gibbs](examples/gibbs_phenomenon.png) |
| **Frequency spectrum** | **Low-pass filter** |
| ![spectrum](examples/spectrum.png) | ![filter](examples/filter_lowpass.png) |
| **DFT vs FFT runtime** | **Synthesized mixed signal** |
| ![compare](examples/dft_vs_fft.png) | ![signal](examples/signal_mixed.png) |

---

## Using FourierLab as a library

Everything the CLI does is available programmatically, with type hints and
docstrings throughout.

```python
import numpy as np
from fourierlab import fft, ifft, dft, idft, apply_filter
from fourierlab.signals import mixed
from fourierlab.series import approximate
from fourierlab.math_utils import dominant_frequencies

# --- transforms ---------------------------------------------------------
x = np.array([1.0, 0.0, -1.0, 0.0])
X = dft(x)                          # from-scratch DFT  ->  [0, 2, 0, 2]
assert np.allclose(ifft(fft(x)), x)         # exact round-trip

# --- spectrum analysis --------------------------------------------------
sig = mixed([5, 20, 50], [1.0, 0.5, 0.2], sample_rate=256, duration=2)
peaks = dominant_frequencies(fft(sig.y), sig.sample_rate, count=3)
for p in peaks:
    print(f"{p.frequency:5.1f} Hz  amplitude {p.amplitude:.2f}")
# 5.0 Hz  amplitude 1.00
# 20.0 Hz  amplitude 0.50
# 50.0 Hz  amplitude 0.20

# --- Fourier series -----------------------------------------------------
approx = approximate("square", num_terms=25)
print(approx.mse, approx.max_error)         # fit quality metrics

# --- filtering ----------------------------------------------------------
result = apply_filter(sig, "lowpass", cutoff=30)
clean = result.filtered.y                    # 50 Hz tone removed
```

---

## How the code is organized

FourierLab keeps a strict separation between the maths, file I/O, plotting, and
the CLI, so each piece can be read and tested on its own. The data flows like
this:

```
              signals.py                     dft.py / fft.py
   generate ──► Signal ──► CSV ──► spectrum ──► forward transform ──► math_utils.py
   (waveforms) (t, y, fs)        filter    ◄── inverse transform ◄──  (bins, |X|, phase,
                                  │                                    dominant freqs)
                                  ▼
                              filters.py  ── zero bins, invert ──► filtered Signal
                                  │
   series.py (Fourier series, Gibbs)         visualize.py (matplotlib → PNG)
                                  \______________  cli.py  ______________/
                                        (argparse + printed reports)
```

| Module | Responsibility |
|---|---|
| `dft.py` | DFT / inverse DFT, straight from the definition (`X = Wx`). |
| `fft.py` | Recursive Cooley–Tukey FFT / IFFT, a length-agnostic dispatcher, and the DFT-vs-FFT benchmark. |
| `series.py` | Real Fourier series: numerical **and** analytical coefficients, reconstruction, Gibbs. |
| `filters.py` | The four ideal frequency-domain filters. |
| `signals.py` | The `Signal` container, waveform generators, and CSV I/O. |
| `math_utils.py` | Frequency bins, magnitude/phase/amplitude spectra, error metrics, Parseval. |
| `visualize.py` | All matplotlib plotting (headless `Agg` backend). |
| `cli.py` | The `argparse` command-line interface and its formatted reports. |
| `errors.py` | A small typed exception hierarchy so the CLI prints clean messages. |

The core transforms never call `numpy.fft`; NumPy is used only for array storage
and vectorized arithmetic, and — in the tests — as an independent oracle.

---

## Testing

The suite has **163 deterministic tests** (random seeds fixed everywhere) and
runs in about a second.

```bash
pip install -e ".[test]"
python -m pytest              # run everything
python -m pytest -v           # verbose, one line per test
python -m pytest tests/test_fft.py    # a single module
```

What the tests verify:

- **DFT** — matches `numpy.fft` within tolerance; the inverse reconstructs the
  input; real *and* complex signals; correct magnitude for pure tones; Parseval's
  energy identity holds.
- **FFT** — matches both NumPy and our own DFT; the inverse reconstructs the
  input; rejects non-power-of-two lengths with a useful error; is measurably
  faster than the DFT for large inputs.
- **Fourier series** — numerical coefficients match the analytical formulas; the
  square/triangle/sawtooth approximations improve with more terms; a sinusoid is
  recovered with a couple of terms; the Gibbs overshoot peaks near 1.179 and
  persists as `N` grows.
- **Filters** — low-pass removes highs, high-pass removes lows, band-pass keeps
  the band, band-stop removes it; length and realness are preserved; invalid
  cutoffs raise clear errors.
- **CLI** — core commands run and print sensible output; invalid input yields a
  clean `error:` message (not a traceback); saved files are actually created.

---

## Project structure

```text
FourierLab/
├── fourierlab/
│   ├── __init__.py       # public API surface
│   ├── __main__.py       # enables `python -m fourierlab`
│   ├── cli.py            # argparse CLI + polished console reports
│   ├── signals.py        # Signal container, waveform generators, CSV I/O
│   ├── series.py         # real Fourier series, coefficients, Gibbs
│   ├── dft.py            # DFT / inverse DFT, from the definition
│   ├── fft.py            # recursive Cooley-Tukey FFT / IFFT + benchmarking
│   ├── filters.py        # low/high/band-pass/band-stop filters
│   ├── visualize.py      # matplotlib plotting (Agg, headless-safe)
│   ├── math_utils.py     # frequency bins, spectra, error metrics, Parseval
│   └── errors.py         # typed exception hierarchy
├── tests/                # pytest suite (dft, fft, series, filters, signals, cli)
├── data/
│   └── sample_signal.csv # a ready-to-use three-tone noisy signal
├── examples/             # generated plots + the commands that made them
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

---

## Limitations

- **Radix-2 FFT only.** The fast path requires power-of-two lengths. Other
  lengths transparently fall back to the `O(N²)` DFT rather than a mixed-radix or
  Bluestein FFT, so very large non-power-of-two signals are slow.
- **Ideal (brick-wall) filters.** The filters zero frequency bins with an
  infinitely sharp edge. This is perfect for teaching but causes time-domain
  ringing; production DSP uses smoother FIR/IIR filters (e.g. Butterworth).
- **Educational performance.** The transforms are written for clarity; a
  pure-Python recursive FFT is far slower than FFTW or `numpy.fft`. FourierLab is
  for understanding, not high-throughput production pipelines.
- **1-D, uniformly sampled, single-channel.** No 2-D transforms, non-uniform
  sampling, or multi-channel audio in the core.
- **No anti-aliasing on synthesis.** Generated square/sawtooth waves contain ideal
  harmonics above Nyquist and can alias if sampled coarsely.

---

## Future work

Planned once the core is solid:

- Window functions (Hann, Hamming, Blackman) for spectral-leakage control
- Spectrogram / short-time Fourier transform
- The convolution theorem as a runnable demo
- 2-D FFT and a basic image-frequency demo
- WAV audio loading and spectrum analysis
- Fourier epicycle ("drawing with circles") animation
- Symbolic Fourier series via SymPy
- Mixed-radix / Bluestein FFT for arbitrary lengths

---

## License

Released under the **MIT License**. See [`LICENSE`](LICENSE) for details.
