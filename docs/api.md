# API reference

Import everything from the package root:

```python
from dpd import OFDM, nmse_db, PA, GMP, ila, shift, MLP, Adam, Feat, Surrogate, GMPNNDPD
from dpd.signals import wifi20, make_signal, WIFI_MODES, MASK_HE20, MASK_LEGACY
```

Conventions: complex baseband `np.ndarray` (complex128), one sample per element. `rng` is always a
`numpy.random.Generator`. Delays `m > 0` mean the past (`z[n−m]`); `m < 0` means look-ahead.

---

## `dpd.signals`

### `class OFDM(nfft=1024, n_act=200, ncp=72, qam=64, fs=61.44e6, tones=None, adj_offset=None, mask=None, evm_spec_db=None, eq="gain", name="lte12", filt_taps=401, filt_guard=None)`

CP-OFDM generator and metrics. The defaults give the original 12 MHz signal: `n_act` contiguous
subcarriers around DC, with DC nulled. `nfft` and `ncp` are given at the DPD rate `fs`, so an oversampled
signal uses `nfft = nfft_base · osr`.

| Argument | Meaning |
|---|---|
| `tones` | signed subcarrier indices to load (overrides `n_act`) |
| `adj_offset` | adjacent-channel centre offset (default `1.1·bw`) |
| `mask` | spectral mask `[(offset_Hz, dBr), …]`, linear in dB between points, last level held beyond |
| `evm_spec_db` | EVM requirement, stored for reporting |
| `eq` | EVM equalisation: `"gain"` (one complex gain) or `"subcarrier"` (per-tone LS, like a receiver with an ideal channel estimate) |
| `filt_taps`, `filt_guard` | TX channel filter length, and cutoff = `bw/2 + filt_guard` (default cutoff `0.52·bw`) |

| Attribute | Meaning |
|---|---|
| `tones`, `idx` | signed subcarrier indices and the FFT bins they map to |
| `bw` | occupied bandwidth in Hz: `n_act·Δf`, or the tone span when `filt_guard` is set |
| `df` | subcarrier spacing |
| `sym_len` | `nfft + ncp` samples |

| Method | Returns | Notes |
|---|---|---|
| `generate(n_sym, rng, papr_db=8.0, cfr_iters=5)` | `(x, ref)` | Unit-rms waveform after the channel filter and optional clip-and-filter CFR (`papr_db=None` disables CFR); `ref = demod(x)` |
| `demod(y)` | `(n_sym, n_act)` complex | FFT demodulation per symbol (CP removed) |
| `evm(y, ref)` | float, % | RMS EVM after equalisation (see `eq`) |
| `evm_db(pct)` | float, dB | static: `20·log10(pct/100)` |
| `psd(y, nperseg=None)` | `(f, P)` | Two-sided Welch PSD, Blackman-Harris, fftshifted. Default 4096 bins, or about 50 kHz bins (about 100 kHz ENBW, the 802.11 mask RBW) when a mask is set |
| `acpr(y)` | `(lower, upper)` dBc | Adjacent channels of width `bw` centred at ±`adj_offset` |
| `mask_limit(f)` | dBr | mask level at frequency `f` |
| `mask_margin(y)` | dB | worst-case `mask − PSD(dBr)` over \|f\| ≥ first breakpoint; PSD relative to the in-band peak. > 0 means pass |
| `describe()` | str | one-line summary |

### `wifi20(mode="he", osr=6, gi_us=0.8) -> OFDM`

IEEE 802.11 20 MHz OFDM data symbols at `fs = 20 MHz · osr`, with guard interval `gi_us`. All
loaded tones carry QAM; pilots are treated as data, and there is no preamble or packet structure.
ACLR is measured at ±20 MHz, and EVM uses per-subcarrier equalisation.

| `mode` | FFT @ 20 MS/s | Tones | QAM | EVM requirement (top MCS) | Mask |
|---|---:|---|---:|---:|---|
| `legacy` (11a/g) | 64 | ±1…±26 (52) | 64 | −25 dB | `MASK_LEGACY` |
| `ht` (11n) | 64 | ±1…±28 (56) | 64 | −28 dB | `MASK_LEGACY` |
| `vht` (11ac) | 64 | ±1…±28 (56) | 256 | −32 dB | `MASK_LEGACY` |
| `he` (11ax, RU242) | 256 | ±2…±122 (242) | 1024 | −35 dB | `MASK_HE20` |
| `eht` (11be, RU242) | 256 | ±2…±122 (242) | 4096 | −38 dB | `MASK_HE20` |

`MASK_LEGACY` = 0 / −20 / −28 / −40 dBr at 9 / 11 / 20 / 30 MHz. `MASK_HE20` = 0 / −20 / −28 / −40 dBr
at 9.75 / 10.5 / 20 / 30 MHz.

### `make_signal(kind="lte12", wifi_mode="he", osr=6) -> OFDM`

`"lte12"` gives `OFDM()`; `"wifi20"` gives `wifi20(wifi_mode, osr)`.

### `nmse_db(y, ref) -> float`

`10·log10(Σ|y−ref|² / Σ|ref|²)`. No gain or delay alignment is applied.

---

## `dpd.pa`

### `class PA(kind="doherty", h1=..., h2=..., kappa=0.10, tau=80.0, ...)`

Behavioural Wiener-Hammerstein PA: `x → H1 → static NL → (1 − κ·LPF_τ{|u|²}) → H2`. `H1` and `H2`
are normalised to unity DC gain, so the small-signal gain is 1.

| Parameter | Default | Meaning |
|---|---|---|
| `kind` | `"doherty"` | `"doherty"` or `"rapp"` static nonlinearity |
| `h1`, `h2` | 3- and 4-tap complex | linear memory before and after the NL |
| `kappa`, `tau` | 0.10, 80 | thermal droop depth and 1-pole time constant (samples) |
| `asat`, `p` | 1.0, 1.1 | Rapp saturation level and smoothness |
| `am_pm_deg`, `beta` | 12, 1.5 | Rapp AM/PM: φ(A) = α A²/(1+βA²), with φ(1) = `am_pm_deg` |
| `v_th` | 0.45 | Doherty peaking turn-on amplitude |
| `c_sat` | 0.5 | carrier and peaking saturation level |
| `pk_gain` | 2.2 | peaking amplifier gain above threshold |
| `pk_phase_deg` | −25 | peaking path phase relative to the carrier |
| `c_pm_deg` | 6 | carrier AM/PM |
| `knee` | 25 | softplus sharpness of the peaking turn-on |

| Method | Returns |
|---|---|
| `__call__(x, snr_db=None, rng=None)` | PA output. If `snr_db` is given, complex AWGN at that SNR is added (a capture). |
| `static(A)` | complex gain G(A) of the static NL, for AM/AM and AM/PM plots: `abs(A*G)`, `angle(G)` |

---

## `dpd.gmp`

### `shift(z, m) -> ndarray`

`z[n−m]` with zero fill; `m < 0` gives a lead. Its adjoint is `shift(·, −m)`.

### `class GMP(Ka=7, La=4, Kb=5, Lb=3, Mb=2, Kc=5, Lc=3, Mc=2)`

| Member | Meaning |
|---|---|
| `terms` | list of `(k, l, m)`: term `x[n−l]·|x[n−l−m]|^(k−1)`. `m=0` aligned, `m>0` lagging, `m<0` leading |
| `P` | number of coefficients (99 by default) |
| `basis(x)` | `(N, P)` complex basis matrix Φ |
| `ls(Phi, t, lam=1e-7)` | static method: ridge LS solution `c`, with ridge scaled by `tr(ΦᴴΦ)/P` |
| `complexity()` | rough real-op count per sample |

Apply a GMP DPD with `z = gmp.basis(x) @ c`.

### `ila(pa, gmp, x, G=1.0, iters=4, snr_db=None, rng=None, lam=1e-7) -> (c, history)`

Indirect learning. Starts from an identity predistorter. Each iteration predistorts, captures
`pa(z, snr_db, rng)`, and refits the post-inverse `Φ(y/G)·c ≈ z`. Returns the final coefficients and a
list of the coefficients after each iteration.

---

## `dpd.nn`

### `class MLP(sizes, rng, out_scale=1.0)`

Fully connected network, tanh hidden layers, linear output. `sizes = [D_in, h1, …, D_out]`.
`out_scale` multiplies the last layer's initial weights (use a small value for residual-style initialisation).

| Method | Notes |
|---|---|
| `forward(X)` | `(N, D_in) → (N, D_out)`; caches activations for `backward` |
| `backward(g)` | `g` is dL/d(output). Returns `(grads, g_in)`; `grads` is ordered as `params` |
| `params` | `[W0, W1, …, b0, b1, …]` (references, updated in place by `Adam`) |
| `macs()` | `(multiply-accumulates, tanh count)` per sample |

### `class Adam(params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-12)`

Adam over a list of arrays, updated in place. Complex arrays get separate second moments for the real and
imaginary parts. `step(grads, lr_scale=1.0)` applies one update; `lr_scale` is used for schedules.

### `cosine(it, n, floor=0.03) -> float`

Cosine decay factor from 1 down to `floor` over `n` iterations.

### `class Feat(amp_delays, prod_delays, lpf_taus=(), lpf_len=None, eps=1e-6)`

Differentiable, phase-invariant feature extractor. Columns, in order:
`|z[n−m]|, |z[n−m]|²` for each `m` in `amp_delays`; `Re, Im{z[n−m]·z*[n]}` for each `m` in `prod_delays`;
`LPF_τ{|z|²}` for each `τ` in `lpf_taus`. With `lpf_len=None` (default) the LPF is an exact one-pole IIR,
`s[n] = a·s[n−1] + (1−a)·q[n]` with `a = e^{−1/τ}` and zero initial state, which matches a hardware
one-pole. Its adjoint is the same filter run backwards in time. With an integer `lpf_len` it is that
filter's `lpf_len`-tap FIR truncation, which is what the committed 12 MHz results used (320 taps).

| Method | Notes |
|---|---|
| `fit_scale(z)` | sets the fixed standardisation (`mu`, `scale`); call once on representative data |
| `forward(z)` | `(N, D)` standardised features |
| `backward(gF)` | dL/dF → dL/dz (complex, using the convention in [Theory §7](theory.md#7-complex-gradients)) |
| `D` | number of features |

### `lagmat(z, L) -> (N, L+1)`

Tapped delay line `[z[n], z[n−1], …, z[n−L]]`.

### `class Surrogate(rng, L=5, amp=range(0, 6), prod=range(1, 6), lpf_taus=(30, 150), hidden=(64, 64), lpf_len=None)`

Differentiable PA model `S(z) = Σ_l h_l z[n−l] + z[n]·w(Feat(z))`.

| Method | Notes |
|---|---|
| `fit(Z, Y, rng, iters=6000, batch=4096, lr=2e-3, init=True, log=None)` | `Z` and `Y` are lists of capture input and output arrays. `init=True` re-initialises the FIR by LS (use `False` to fine-tune). Feature scaling is fitted on the first call only. `log`, if a list, collects training NMSE (dB) every 500 iterations. |
| `forward(z)` / `__call__(z)` | prediction; caches for `backward` |
| `backward(gy)` | dL/dy → dL/dz (the surrogate itself stays frozen) |

### `class GMPNNDPD(gmp, c0, x_train, rng, use_nn=True, amp=range(-2, 5), prod=(-2, -1, 1, 2, 3, 4), lpf_taus=(30, 150), hidden=(24, 24), lpf_len=None)`

The predistorter `z = GMP(x) + x[n]·w(Feat(x))`. With `use_nn=False` it is a pure GMP trained by DLA.
The GMP coefficients are stored as `d` in the QR-orthonormalised basis of `x_train`. `c0` is the initial
GMP solution (typically from `ila`).

| Member | Notes |
|---|---|
| `c` | property: effective GMP coefficients `R⁻¹d`, usable with `gmp.basis` |
| `__call__(x)` | apply the DPD to any waveform |
| `train_dla(sur, x, G=1.0, iters=3000, win=6144, pre=400, lr_gmp=3e-4, lr_nn=1e-3, rng=None, log=None)` | direct learning through frozen `sur`. Uses random contiguous windows of `win` samples, with the first `pre` masked from the loss. `log` collects window NMSE (dB) every 100 iterations. |
| `complexity()` | `(real ops, tanh)` per sample |
| `mlp`, `feat` | the residual network and its feature extractor (`feat.mu`, `feat.scale` are needed for export) |

---

## Scripts

| Script | Purpose | Key options |
|---|---|---|
| `run_demo.py` | full benchmark, writes `results/` | `--signal {lte12,wifi20}`, `--wifi-mode {legacy,ht,vht,he,eht}`, `--osr 6`, `--cfr PAPR_dB`, `--lpf-len {auto,iir,N}`, `--pa {doherty,rapp}`, `--rms` (default 0.30 lte12 / 0.27 wifi20), `--snr 50`, `--rounds 3`, `--quick`, `--out DIR` |
| `plot_results.py [DIR]` | figure `DIR/dpd_results.png` | |
| `examples/tutorial.py` | guided run, writes `tutorial_out/dpd_nn.npz` | |
| `examples/drive_sweep.py` | no-DPD vs GMP-ILA sweep | |
| `tests/test_gradcheck.py` | finite-difference gradient check | |

### Output files (`results/`)

| File | Contents |
|---|---|
| `results.json` | config (incl. signal, fs, LPF form and time constants, DLA window), metrics per DPD (NMSE, ACPR, EVM % and dB, mask margin for Wi-Fi, complexity), surrogate NMSE (round 1 and final), training logs, runtime |
| `waveforms.npz` | test input `x`, each DPD's output (`GMP-ILA`, `GMPpNN-DLA`, …) and the matching PA outputs `y_…` |
| `dpd_params.npz` | `gmp_terms`, `c_ila`, `c_gmp_dla`, `c_nn_gmp`, `feat_mu`, `feat_scale`, `W0..W2`, `b0..b2` |
