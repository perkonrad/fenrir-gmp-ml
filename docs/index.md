# fenrir-gmp-ml documentation

Digital predistortion (DPD) for RF power amplifiers: a Generalized Memory Polynomial (GMP)
baseline and a machine-learning variant that adds a phase-equivariant neural-network
residual to the GMP and trains both by direct learning through a learned PA model.

Everything is NumPy/SciPy. Backpropagation is written by hand (and gradient-checked), so each
operation has an obvious fixed-point / RTL counterpart.

## Pages

| Page | What it covers |
|---|---|
| [Theory](theory.md) | GMP, ILA, direct learning, the NN residual, the surrogate, the complex-gradient convention |
| [Tutorial](tutorial.md) | Step-by-step walk-through of `examples/tutorial.py`, with its real output |
| [Results](results.md) | Full benchmark, round-by-round convergence, drive sweep, PA-model comparison |
| [API reference](api.md) | Every class and function in `dpd/` |
| [Hardware notes](hardware.md) | Op counts, fixed-point mapping, bench integration |

## Headline result

Doherty-style PA, 12 MHz 64-QAM OFDM, 50 dB capture SNR, scored on an unseen test waveform
through the true PA:

| DPD | NMSE (dB) | ACPR worst (dBc) | EVM (%) |
|---|---:|---:|---:|
| None | −17.4 | −27.6 | 11.62 |
| GMP, indirect learning | −39.9 | −45.4 | 0.56 |
| GMP, direct learning | −43.4 | −50.3 | 0.39 |
| **GMP + NN, direct learning** | **−51.5** | **−58.2** | **0.14** |

## Quick start

```bash
pip install numpy scipy matplotlib
python tests/test_gradcheck.py        # verifies all hand-written gradients (seconds)
python examples/tutorial.py           # guided run, ~5 min on 2 cores
python run_demo.py                    # full benchmark, ~15 min  (--quick: ~4 min)
python plot_results.py results        # regenerate results/dpd_results.png
```

## Repository layout

```
dpd/
  signals.py      OFDM generator with CFR; NMSE, ACPR, EVM, PSD
  pa.py           Wiener-Hammerstein PA: Doherty or Rapp static NL + thermal droop
  gmp.py          GMP basis, ridge least squares, ILA loop, shift()
  nn.py           MLP, complex Adam, equivariant features, PA surrogate, GMP+NN DPD + DLA trainer
examples/
  tutorial.py     the tutorial script
tests/
  test_gradcheck.py
run_demo.py       full benchmark -> results/
plot_results.py   figure from results/
results/          results.json, waveforms.npz, dpd_params.npz, dpd_results.png
docs/             this documentation
```
