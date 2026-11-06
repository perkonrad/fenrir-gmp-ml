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
| [20 MHz Wi-Fi](wifi20.md) | 802.11ax HE20 (1024-QAM, 120 MS/s, no CFR): setup, spectral mask and EVM against the spec, results |
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

### 20 MHz Wi-Fi (802.11ax HE20, 1024-QAM, no CFR, 120 MS/s, rms 0.27)

| DPD | NMSE (dB) | ACLR worst (dBc) | EVM (dB, requirement −35) | Mask margin (dB) |
|---|---:|---:|---:|---:|
| None | −18.8 | −28.0 | −22.5 | −2.0 (fail) |
| GMP, indirect learning | −39.6 | −44.7 | −45.1 | +7.6 |
| GMP, direct learning | −39.8 | −44.8 | −44.5 | +7.2 |
| **GMP + NN, direct learning** | **−50.5** | **−56.9** | **−57.7** | **+17.4** |

## Quick start

```bash
pip install numpy scipy matplotlib
python tests/test_gradcheck.py        # verifies all hand-written gradients (seconds)
python examples/tutorial.py           # guided run, ~5 min on 2 cores
python run_demo.py                    # full 12 MHz benchmark, ~15 min  (--quick: ~4 min)
python run_demo.py --signal wifi20    # 802.11ax HE20 benchmark, ~17 min -> results_wifi20/
python plot_results.py results        # regenerate results/dpd_results.png
```

## Repository layout

```
dpd/
  signals.py      OFDM generator (LTE-like 12 MHz, 802.11 20 MHz tone plans), CFR; NMSE, ACPR, EVM, PSD, mask
  pa.py           Wiener-Hammerstein PA: Doherty or Rapp static NL + thermal droop
  gmp.py          GMP basis, ridge least squares, ILA loop, shift()
  nn.py           MLP, complex Adam, equivariant features, PA surrogate, GMP+NN DPD + DLA trainer
examples/
  tutorial.py     the tutorial script
tests/
  test_gradcheck.py
run_demo.py       full benchmark -> results/
plot_results.py   figure from results/
results/          12 MHz: results.json, waveforms.npz, dpd_params.npz, dpd_results.png
results_wifi20/   Wi-Fi HE20: same files
docs/             this documentation
```
