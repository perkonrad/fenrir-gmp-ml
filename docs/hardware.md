# Hardware notes

How the trained GMP+NN DPD maps onto an FPGA or ASIC datapath, and what changes when you
move from this simulation to a real bench.

## Datapath

```
x[n] ──┬── delay line (n+2 … n−4) ──┬── GMP basis ── 99 complex MACs ──────────────┐
       │                            │                                              (+)── z[n]
       │                            └── features (28) ── MLP 28→24→24→2 ── w ── × x[n] ──┘
       └── |x|² ── two 1-pole LPFs ───┘
```

The 2-sample look-ahead (GMP leading terms and NN features) is 2 samples of pipeline latency.

## Operation counts per sample

| Block | Count |
|---|---|
| \|x\|, \|x\|² | 1 sqrt (or CORDIC), 2 mult |
| GMP basis | 99 complex × real products; the \|x\|^k powers are shared across delays |
| GMP accumulate | 99 complex MACs = 396 real mults |
| Relative-phase features | 6 complex products `x[n−m]·x*[n]` = 24 real mults |
| LPF features | 2 one-pole IIRs |
| Standardisation | 28 multiply-adds; these fold into the first MLP layer offline |
| MLP | 28·24 + 24·24 + 24·2 = **1,296 MACs**, 48 tanh |
| Output | 1 complex multiply `x[n]·w` |

At 61.44 MS/s the NN is about 80 GMAC/s and the GMP about 24 G real mult/s. With DSP slices clocked at
491.52 MHz (8 clocks per sample), that is roughly 160 slices for the MLP and 50 for the GMP
accumulation, plus basis generation, before any resource sharing or reduced precision.

## Fixed-point mapping

* **Fold the standardisation.** `F = (raw − mu)·scale` is affine, so absorb it into `W0` and `b0`
  offline: `W0' = diag(scale)·W0`, `b0' = b0 − (mu·scale)·W0`. The network then takes raw features.
* **tanh.** A 64 to 256 entry LUT with linear interpolation, or a piecewise-linear approximation. Hidden
  activations are bounded in (−1, 1), which makes fixed-point formats easy.
* **Output gain `w`.** It is a small correction around 0. Its full-scale range can be read from the
  training data, which allows a narrow word length.
* **Quantisation-aware training.** Reuse `train_dla` with fake-quantised weights and activations in
  the forward pass (straight-through estimator in the backward pass). Because training is direct
  learning through the surrogate, quantisation error is optimised against end-to-end linearity, not
  against a weight-matching proxy.
* **GMP coefficient range.** High-order GMP coefficients can be large with alternating signs.
  Check their dynamic range after training, or reduce `Ka`, if word length is tight.

## LPF features: match the trained kernel

Since the Wi-Fi update, `Feat` defaults to an **exact one-pole IIR** (`lpf_len=None`). That is what
the 20 MHz Wi-Fi results use, and it maps directly onto a hardware one-pole with nothing more to do. The rest of this
section applies to the committed **12 MHz** results, which were trained with the older FIR form.

The 12 MHz model uses a **320-tap truncation** of the 1-pole impulse response `(1−a)·aⁿ` with `a = e^{−1/τ}`.
For τ = 30 the truncated tail is negligible. For τ = 150 it is about 12 % (`e^{−320/150}`), so a plain
one-pole IIR does **not** match what the network was trained on. Options:

1. Implement the truncated kernel exactly with a one-pole recursion and one delay tap:
   `s[n] = a·s[n−1] + (1−a)·(q[n] − a³²⁰·q[n−320])`.
2. Retrain with `lpf_len` large enough (about 8τ) that the truncation is negligible, then use a plain IIR.

## Moving to a real bench

Replace `pa(z, snr_db, rng)` with a capture function that:

1. Transmits `z` and captures the PA output through the observation receiver.
2. Aligns delay (fractional, via cross-correlation and interpolation), phase and gain against `z`.
3. Returns the aligned output at the same sample rate.

Then use the same flow: ILA initialisation, surrogate fit on several captures, DLA, and recapture rounds.

Practical points:

* **Capture SNR and alignment set the floor.** The surrogate cannot model below the capture
  noise. Averaging repeated captures of the same waveform raises the effective SNR.
* **Capture bandwidth.** The observation path needs about 5× the signal bandwidth, as in this simulation.
* **Cover the drive range.** Include overdriven or predistorted captures, as `run_demo.py` does. A
  surrogate trained only on the raw signal extrapolates badly where the DPD expands peaks.
* **Drift.** Thermal and bias drift change the PA slowly. Periodic recapture and short surrogate
  and DPD fine-tunes (round 2+ style, low learning rate) track it without retraining from scratch.
