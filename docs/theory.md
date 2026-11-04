# Theory

All signals are complex baseband sequences. The PA is normalised to unit small-signal gain,
so the linearisation target is simply `y[n] = G·x[n]` with `G = 1`.

## 1. The GMP model

The Generalized Memory Polynomial (Morgan et al., *IEEE Trans. Signal Processing*, 2006) is

```
y[n] =  Σ_{k=1..Ka} Σ_{l=0..La}               a_kl  · x[n−l] |x[n−l]|^(k−1)        aligned
     +  Σ_{k=2..Kb} Σ_{l=0..Lb} Σ_{m=1..Mb}   b_klm · x[n−l] |x[n−l−m]|^(k−1)      lagging envelope
     +  Σ_{k=2..Kc} Σ_{l=0..Lc} Σ_{m=1..Mc}   c_klm · x[n−l] |x[n−l+m]|^(k−1)      leading envelope
```

It is linear in its coefficients, so for a data matrix `Φ` (one column per term) any fit is a
least-squares problem `min ‖Φc − t‖²`. Both odd and even `k` are used; even-order envelope terms
improve the fit for real PAs noticeably.

Default orders (`GMP()`): `Ka=7, La=4` (35 terms), `Kb=Kc=5, Lb=Lc=3, Mb=Mc=2` (32 + 32 terms),
**99 complex coefficients**. Each term is stored as a tuple `(k, l, m)`: `m=0` aligned, `m>0` lagging,
`m<0` leading.

Least squares uses a scaled ridge term, `(ΦᴴΦ + λ·tr(ΦᴴΦ)/P · I) c = Φᴴt` with `λ = 1e−7`, because
high-order GMP columns are strongly correlated (the condition number of `ΦᴴΦ` is large).

## 2. Indirect learning (ILA): the baseline

ILA fits a *post-inverse* and copies it in front of the PA:

1. Predistort: `z = Φ(x) c`.
2. Capture: `y = PA(z) + noise`.
3. Fit the post-inverse from PA output back to PA input: `c ← argmin ‖Φ(y/G) c − z‖²`.
4. Repeat (4 iterations here).

ILA is cheap and needs no PA model, but it has two well-known weaknesses:

* **Noise in the regressor.** The basis is built from the noisy capture `y`, so LS is biased
  (errors-in-variables); the coefficients shrink toward a smoother solution.
* **Post-inverse ≠ pre-inverse.** For a nonlinear system with memory, the best post-inverse within a
  finite model class is not the best pre-inverse.

Both are why direct learning beats ILA even with the *same* GMP model (see [Results](results.md)).

## 3. Direct learning (DLA) through a learned PA surrogate

Direct learning optimises the quantity we actually care about:

```
min_θ  E ‖ PA(DPD_θ(x)) − G·x ‖²
```

The real PA cannot be differentiated, so a differentiable model `S ≈ PA` is fitted first and the
loss is minimised through it:

```
L(θ) = Σ_n | S(DPD_θ(x))[n] − G·x[n] |²  /  Σ_n |G·x[n]|²
```

Gradients flow `L → S output → S input z → DPD parameters θ` with `S` frozen. The surrogate's
accuracy therefore roughly bounds the achievable linearity: the DPD ends up near the surrogate's
error level on the region it is driven over. This is why the flow is iterative (section 6).

## 4. The ML DPD: GMP + phase-equivariant NN residual

```
z[n] = Σ_q c_q φ_q(x)[n]   +   x[n] · w( F(x)[n] )
       └──── GMP ───────┘     └─── NN residual ──┘
```

`w` is a real-valued MLP whose two outputs form a complex gain `w = w_re + j·w_im`.

### Features `F(x)[n]`

| Feature | Delays | Purpose |
|---|---|---|
| `|x[n−m]|`, `|x[n−m]|²` | m = −2 … 4 | envelope history (with 2-sample look-ahead) |
| `Re/Im{ x[n−m] · x*[n] }` | m ∈ {−2, −1, 1, 2, 3, 4} | phase of neighbours *relative to* x[n] |
| `LPF_τ{|x|²}[n]` | τ = 30, 150 samples | slow envelope power, for thermal/trapping memory |

That is 28 features. Each is standardised with a fixed mean and scale computed once on the training waveform.

### Why it is phase-equivariant

Rotate the input by any constant phase, `x → x·e^{jα}`. Every feature above is unchanged:
magnitudes trivially, and `x[n−m]·x*[n]` because the rotations cancel. So `w` is unchanged, and the
output becomes `x[n]e^{jα}·w`, i.e. it rotates with the input. The GMP is equivariant too. Therefore
the whole DPD satisfies `DPD(x·e^{jα}) = DPD(x)·e^{jα}` **by construction**.

I/Q-input networks (RVTDNN, ARVTDNN) must learn this symmetry from data, which wastes capacity
and generalises worse. Here the network only has to learn a function of amplitudes and *relative*
phases.

### Why a residual on GMP

* The DPD is initialised from the ILA solution, with the last NN layer scaled to 0.01, so training
  starts from a working predistorter instead of from scratch.
* The GMP captures the bulk of the inverse cheaply; the NN spends its capacity only on what a
  polynomial fits badly: the Doherty peaking knee, the AM/PM bump, and long-term memory.

### MLP

`28 → 24 → 24 → 2`, tanh hidden units, linear output, 1,346 parameters.

## 5. The PA surrogate

The surrogate uses the same equivariant structure, with a linear FIR in place of the GMP:

```
S(z)[n] = Σ_{l=0..5} h_l z[n−l]  +  z[n] · w_s( F_s(z)[n] )
```

Features use causal delays only (a physical PA cannot see the future): `|z|`, `|z|²` for m = 0…5,
relative-phase products for m = 1…5, and the two LPF features. MLP: `24 → 64 → 64 → 2`.

Training: the FIR taps are initialised by linear least squares. Then FIR and MLP are trained
jointly with Adam (minibatch 4096 samples, cosine learning-rate decay) on NMSE against the
noisy captures. Features of the captures are precomputed, since the PA inputs are fixed data.

## 6. Training flow

```
round 1:  captures = { x, 1.15·x, ILA-predistorted x }        (cover the DPD's drive range)
          fit surrogate S
          DLA:  GMP-only and GMP+NN, both initialised from ILA
round r:  capture PA with each current DPD output, add to captures
          fine-tune S on all captures (lower LR, no re-init)
          fine-tune both DPDs (lower LR)
```

Recapturing puts training data exactly where the DPD now drives the PA. The surrogate's test
NMSE improved from −46.5 dB to −49.9 dB over three rounds, and the DPD followed it.

### DLA minibatching with memory

The DPD and surrogate both have memory, including the 320-tap LPF kernels. Each DLA step
therefore takes a random **contiguous** window of 6,144 samples, runs the full chain on it, and
masks the first 400 samples out of the loss so every scored sample sees its full history.

### Conditioning the GMP for gradient descent

GMP columns differ in scale by orders of magnitude and are highly correlated, so plain gradient
steps on `c` converge very slowly. The DPD instead trains `d` in an orthonormalised basis:

```
Φ(x_train) = Q R   (economic QR),   c = R⁻¹ d,   z_GMP = (Φ R⁻¹) d
```

On the training waveform `Φ R⁻¹` is orthonormal (scaled), so the problem in `d` is well conditioned.
At run time the effective `c = R⁻¹ d` is used with the ordinary basis.

## 7. Complex gradients

For a real loss `L` and complex variable `z`, the code uses

```
g_z = ∂L/∂Re(z) + j·∂L/∂Im(z)
```

Under this convention, `dL = Re(g_z* · dz)`, and the rules used throughout `dpd/nn.py` are:

| Forward | Backward |
|---|---|
| `y = a·w` | `g_a += g_y · w*`,  `g_w += g_y · a*` |
| `y = Σ_l h_l z[n−l]` | `g_z[n] += Σ_l h_l* g_y[n+l]`, `g_h_l = Σ_n z*[n−l] g_y[n]` |
| `p = a · b*` | `g_a += g_p · b`,  `g_b += g_p* · a` |
| `r = |z|` | `g_z += g_r · z/|z|` |
| `q = |z|²` | `g_z += 2 g_q · z` |
| `s = k ⊛ q` (causal FIR) | `g_q = k ⋆ g_s` (correlation = time-reversed convolution) |
| `L = Σ|e|²` | `g_e = 2e` |

Complex parameters are optimised with Adam treating real and imaginary parts as separate
parameters (separate second-moment estimates). `tests/test_gradcheck.py` checks `dL/dz` through the
surrogate and `dL/dW` for the DPD network against central finite differences; relative errors are
about 1e−7 and 1e−8.

## 8. Metrics

| Metric | Definition |
|---|---|
| NMSE | `10 log10( Σ|y − x|² / Σ|x|² )` with the fixed target gain (no best-fit gain or delay) |
| ACPR | Welch PSD (Blackman-Harris, 4096 bins). Measurement BW = occupied 12 MHz; adjacent channels centred at ±13.2 MHz (10 % guard). Worst side reported. |
| EVM | Per-subcarrier RMS error after a single complex-gain LS fit, against the subcarrier values the transmitted (CFR'd, filtered) signal actually carries |

All metrics use the *clean* PA output on the *test* waveform. The 50 dB noise is only in the
captures used for identification.
