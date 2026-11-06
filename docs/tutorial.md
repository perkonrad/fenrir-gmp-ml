# Tutorial

This walks through [`examples/tutorial.py`](../examples/tutorial.py) one step at a time. The
script is a shorter version of `run_demo.py` (fewer iterations, one refinement round) and runs in
about 5 minutes on 2 CPU cores. The outputs shown are from an actual run with seed 42.

```bash
pip install numpy scipy matplotlib
python examples/tutorial.py
```

## Step 1: generate the test signal

```python
sig = OFDM()                         # 12 MHz, 64-QAM, 200/1024 subcarriers, fs = 61.44 MHz
x_tr, _ = sig.generate(32, rng)      # training waveform
x_te, _ = sig.generate(24, rng)      # independent test waveform
RMS = 0.30
x_tr, x_te = RMS * x_tr, RMS * x_te
ref_te = sig.demod(x_te)             # subcarrier values for EVM
```

`generate` builds contiguous CP-OFDM, applies a 401-tap Kaiser channel filter, and runs 5
iterations of clip-and-filter crest-factor reduction to about 8.4 dB PAPR. The result has unit rms. The
oversampling ratio is about 5×, enough room for the DPD's spectral regrowth.

**Always train and test on different waveforms.** A DPD scored on its own training data looks
several dB better than it is.

`RMS` sets the drive level. With the default Doherty model, 0.30 puts the signal peaks (about 0.79)
well into the peaking-amplifier region, where the PA shows gain expansion and an AM/PM swing. No-DPD
ACPR is about −28 dBc.

```
[1] signal: 35072 train / 26304 test samples, PAPR 8.4 dB, BW 12 MHz
```

## Step 2: the PA

```python
pa = PA("doherty")
measure = lambda z: pa(z, snr_db=50, rng=rng)
```

The model is `H1 → static NL → thermal droop → H2`. The Doherty static NL is a class-AB carrier
amplifier that saturates near half voltage, plus a class-C peaking amplifier that turns on at
|u| ≈ 0.45 with a −25° relative phase. `measure` represents a bench capture with 50 dB SNR;
scoring always uses the clean `pa(z)`.

```
[2] PA without DPD
    No DPD       NMSE  -17.3 dB   ACPR  -27.6 dBc   EVM 11.64 %
```

## Step 3: GMP baseline with indirect learning

```python
gmp = GMP()                                          # 99 coefficients
c_ila, _ = ila(pa, gmp, x_tr, iters=4, snr_db=50, rng=rng)
z = gmp.basis(x_te) @ c_ila                          # apply the DPD
```

Each ILA iteration predistorts, captures, and refits the post-inverse by ridge LS. It converges in
2 to 3 iterations; the 4th changes little.

```
[3] GMP-ILA (99 coefficients)
    GMP-ILA      NMSE  -39.4 dB   ACPR  -45.1 dBc   EVM  0.60 %
```

This is a solid conventional result: 17 dB of ACPR improvement.

## Step 4: learn a differentiable PA model

```python
z_ila = gmp.basis(x_tr) @ c_ila
caps_z = [x_tr, 1.15 * x_tr, z_ila]
caps_y = [measure(z) for z in caps_z]
sur = Surrogate(np.random.default_rng(1))
sur.fit(caps_z, caps_y, rng, iters=4000)
```

The three captures span the range the DPD will drive: the raw signal, an overdriven copy, and the
ILA-predistorted signal (which has expanded peaks). The surrogate is FIR + equivariant NN
(see [Theory §5](theory.md#5-the-pa-surrogate)).

```
[4] surrogate NMSE vs true PA on test drive: -43.9 dB
```

This number matters: it caps what direct learning can reach. The full benchmark trains longer
(8,000 iterations) and reaches −46.5 dB in round 1.

## Step 5: direct learning

```python
dpd_gmp = GMPNNDPD(gmp, c_ila, x_tr, rng2, use_nn=False)   # GMP only
dpd_nn  = GMPNNDPD(gmp, c_ila, x_tr, rng3, use_nn=True)    # GMP + NN residual
dpd_gmp.train_dla(sur, x_tr, iters=1200, rng=rng)
dpd_nn.train_dla(sur, x_tr, iters=2500, rng=rng)
```

Both start from the ILA coefficients. `train_dla` minimises `‖S(DPD(x)) − x‖²` by backpropagating
through the frozen surrogate on random 6,144-sample windows. The GMP is trained in a
QR-orthonormalised basis; the NN output starts near zero.

```
[5] direct learning, round 1
    GMP-DLA      NMSE  -41.6 dB   ACPR  -47.3 dBc   EVM  0.45 %
    GMP+NN-DLA   NMSE  -45.2 dB   ACPR  -50.2 dBc   EVM  0.27 %
```

Two separate effects show up:

* **GMP-ILA → GMP-DLA** (same model): +2.2 dB NMSE. This is the benefit of optimising the actual
  pre-inverse instead of a noisy post-inverse.
* **GMP-DLA → GMP+NN-DLA**: another +3.6 dB, from the NN residual.

## Step 6: recapture and refine

```python
for d in (dpd_gmp, dpd_nn):
    z = d(x_tr)
    caps_z.append(z); caps_y.append(measure(z))
sur.fit(caps_z, caps_y, rng, iters=2000, lr=7e-4, init=False)
dpd_gmp.train_dla(sur, x_tr, iters=800,  lr_gmp=1e-4, rng=rng)
dpd_nn.train_dla(sur, x_tr,  iters=1500, lr_gmp=1e-4, lr_nn=4e-4, rng=rng)
```

The new captures are exactly where the DPDs now drive the PA, so the surrogate becomes more
accurate in the region that matters. Then both DPDs are fine-tuned at a lower learning rate.
`init=False` keeps the surrogate's existing weights.

```
[6] after recapture + fine-tune
    GMP-DLA      NMSE  -42.3 dB   ACPR  -48.7 dBc   EVM  0.43 %
    GMP+NN-DLA   NMSE  -47.2 dB   ACPR  -53.0 dBc   EVM  0.23 %
```

The NN variant gains 2 dB from one refinement; the GMP-only one gains less, because it is
limited by its model class rather than by the surrogate. With three rounds and longer training,
`run_demo.py` reaches −51.5 dB / −58.2 dBc (see [Results](results.md)).

## Step 7: export

```python
np.savez("tutorial_out/dpd_nn.npz", gmp_terms=..., c=dpd_nn.c, feat_mu=..., feat_scale=...,
         W0=..., W1=..., W2=..., b0=..., b1=..., b2=...)
```

```
[7] exported tutorial_out/dpd_nn.npz  (~1922 real ops + 48 tanh per sample)
```

This file is everything needed to run the DPD elsewhere: GMP term list and coefficients,
feature normalisation, and MLP weights. See [Hardware notes](hardware.md).

To apply it to a new signal in Python, use the object directly: `z = dpd_nn(x_new)`.

## Exercises

1. **Smooth PA.** Change to `PA("rapp")`. GMP-ILA alone reaches about −50 dB; the NN adds much less.
   The NN pays off on Doherty-like nonlinearities and long-term memory.
2. **Drive level.** Try `RMS = 0.22` and `0.34` and compare the GMP vs GMP+NN gap at each.
3. **Remove long-term memory.** Use `PA("doherty", kappa=0)`, or construct `GMPNNDPD(..., lpf_taus=())`
   on the original PA, to see how much the LPF features contribute.
4. **Noisier bench.** Set `snr_db=40` in `measure` (and in the `ila` call). Noise enters ILA's
   regressor directly, while DLA only sees it through the surrogate fit; compare how each degrades.
5. **Smaller network.** Try `hidden=(12, 12)` to explore the complexity/performance trade-off.
6. **Lower-order GMP.** Try `GMP(Ka=5, La=2, Kb=3, Lb=2, Mb=1, Kc=3, Lc=2, Mc=1)`; the NN residual can
   compensate for part of the reduced polynomial.
7. **Wi-Fi.** Replace step 1 with `from dpd.signals import wifi20; sig = wifi20("he")` and
   `sig.generate(n, rng, papr_db=None)`, use `PA("doherty", tau=80 * sig.fs / 61.44e6)` and `RMS = 0.27`,
   and score with `sig.mask_margin(y)` and `sig.evm_db(sig.evm(y, ref_te))`. Scale the LPF time constants
   by `sig.fs / 61.44e6` too (`lpf_taus=` on `Surrogate` and `GMPNNDPD`). `run_demo.py --signal wifi20` does all of this.
   For the IIR features, also pass a longer warm-up to `train_dla` (`pre` ≈ 5× the slowest τ).
