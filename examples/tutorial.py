"""Tutorial: build, train and evaluate a GMP and a GMP+NN DPD step by step.

Run from the repository root:   python examples/tutorial.py
Takes ~3-4 minutes on 2 CPU cores (iteration counts are reduced vs run_demo.py).
Each step matches a section of docs/tutorial.md.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dpd import OFDM, PA, GMP, ila, nmse_db, Surrogate, GMPNNDPD

rng = np.random.default_rng(42)

# ---------------------------------------------------------------- 1. signal
sig = OFDM()                                   # 12 MHz, 64-QAM, fs = 61.44 MHz
x_tr, _ = sig.generate(32, rng)                # training waveform (unit rms)
x_te, _ = sig.generate(24, rng)                # independent test waveform
RMS = 0.30                                     # drive level into the PA
x_tr, x_te = RMS * x_tr, RMS * x_te
ref_te = sig.demod(x_te)                       # transmitted subcarrier values for EVM
print(f"[1] signal: {len(x_tr)} train / {len(x_te)} test samples, "
      f"PAPR {20*np.log10(np.abs(x_te).max()/np.sqrt(np.mean(np.abs(x_te)**2))):.1f} dB, BW {sig.bw/1e6:.0f} MHz")

# ---------------------------------------------------------------- 2. PA
pa = PA("doherty")
measure = lambda z: pa(z, snr_db=50, rng=rng)  # 'bench capture' with receiver noise


def report(name, z):
    y = pa(z)                                  # clean output for scoring
    print(f"    {name:<12} NMSE {nmse_db(y, x_te):6.1f} dB   ACPR {max(sig.acpr(y)):6.1f} dBc   "
          f"EVM {sig.evm(y, ref_te):5.2f} %")


print("[2] PA without DPD")
report("No DPD", x_te)

# ---------------------------------------------------------------- 3. GMP + ILA
gmp = GMP()                                    # 99 coefficients
c_ila, _ = ila(pa, gmp, x_tr, iters=4, snr_db=50, rng=rng)
print(f"[3] GMP-ILA ({gmp.P} coefficients)")
report("GMP-ILA", gmp.basis(x_te) @ c_ila)

# ---------------------------------------------------------------- 4. NN PA surrogate
z_ila = gmp.basis(x_tr) @ c_ila
caps_z = [x_tr, 1.15 * x_tr, z_ila]            # drive the PA over the range the DPD will use
caps_y = [measure(z) for z in caps_z]
sur = Surrogate(np.random.default_rng(1), lpf_len=320)   # FIR form, as in the docs' output
sur.fit(caps_z, caps_y, rng, iters=4000)
z_chk = gmp.basis(x_te) @ c_ila
print(f"[4] surrogate NMSE vs true PA on test drive: {nmse_db(sur(z_chk), pa(z_chk)):.1f} dB")

# ---------------------------------------------------------------- 5. direct learning
dpd_gmp = GMPNNDPD(gmp, c_ila, x_tr, np.random.default_rng(2), use_nn=False, lpf_len=320)
dpd_nn = GMPNNDPD(gmp, c_ila, x_tr, np.random.default_rng(3), use_nn=True, lpf_len=320)
dpd_gmp.train_dla(sur, x_tr, iters=1200, rng=rng)
dpd_nn.train_dla(sur, x_tr, iters=2500, rng=rng)
print("[5] direct learning, round 1")
report("GMP-DLA", dpd_gmp(x_te))
report("GMP+NN-DLA", dpd_nn(x_te))

# ---------------------------------------------------------------- 6. refinement round
for d in (dpd_gmp, dpd_nn):
    z = d(x_tr)
    caps_z.append(z)
    caps_y.append(measure(z))
sur.fit(caps_z, caps_y, rng, iters=2000, lr=7e-4, init=False)
dpd_gmp.train_dla(sur, x_tr, iters=800, lr_gmp=1e-4, rng=rng)
dpd_nn.train_dla(sur, x_tr, iters=1500, lr_gmp=1e-4, lr_nn=4e-4, rng=rng)
print("[6] after recapture + fine-tune")
report("GMP-DLA", dpd_gmp(x_te))
report("GMP+NN-DLA", dpd_nn(x_te))

# ---------------------------------------------------------------- 7. export
os.makedirs("tutorial_out", exist_ok=True)
np.savez("tutorial_out/dpd_nn.npz", gmp_terms=np.array(gmp.terms), c=dpd_nn.c,
         feat_mu=dpd_nn.feat.mu, feat_scale=dpd_nn.feat.scale,
         **{f"W{i}": W for i, W in enumerate(dpd_nn.mlp.W)}, **{f"b{i}": b for i, b in enumerate(dpd_nn.mlp.b)})
ops, nt = dpd_nn.complexity()
print(f"[7] exported tutorial_out/dpd_nn.npz  (~{ops} real ops + {nt} tanh per sample)")
