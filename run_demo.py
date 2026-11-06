"""End-to-end benchmark:  no DPD  |  GMP-ILA  |  GMP-DLA  |  GMP+NN-DLA (ML variant).

Flow
 1. Train / test OFDM (independent seeds): 12 MHz LTE-like with CFR (--signal lte12) or
    IEEE 802.11 20 MHz (--signal wifi20, HE/EHT/VHT/HT/legacy tone plans, no CFR by default).  PA = Doherty-like WH model + thermal droop,
    'measured' with 50 dB SNR receiver noise.
 2. GMP-ILA baseline (4 iterations, ridge LS).
 3. Fit NN PA surrogate on captures: raw x, 1.15x overdrive, ILA-predistorted x.
 4. Direct learning through the frozen surrogate, both for pure GMP and for GMP + NN residual,
    initialised from the ILA solution.
 5. Rounds 2..R: capture PA with the new DPD outputs, fine-tune the surrogate, fine-tune the DPDs.
 6. Evaluate everything on the *test* signal through the *true* PA.
"""
import json, time, argparse
import numpy as np
from dpd import PA, GMP, ila, nmse_db, Surrogate, GMPNNDPD
from dpd.signals import make_signal

ap = argparse.ArgumentParser()
ap.add_argument("--signal", default="lte12", choices=["lte12", "wifi20"])
ap.add_argument("--wifi-mode", default="he", choices=["legacy", "ht", "vht", "he", "eht"])
ap.add_argument("--osr", type=int, default=6, help="Wi-Fi oversampling: fs = 20 MHz * osr")
ap.add_argument("--cfr", type=float, default=None, help="CFR target PAPR dB (default 8 for lte12, off for wifi20)")
ap.add_argument("--lpf-len", default="auto",
                help="LPF feature form: 'iir' (exact one-pole) or an FIR length; auto = 320 for lte12 "
                     "(reproduces the committed 12 MHz results), iir for wifi20")
ap.add_argument("--pa", default="doherty", choices=["doherty", "rapp"])
ap.add_argument("--rms", type=float, default=None, help="drive rms (default 0.30 lte12, 0.27 wifi20)")
ap.add_argument("--snr", type=float, default=50.0)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default="results")
args = ap.parse_args()

rng = np.random.default_rng(7)
sig = make_signal(args.signal, args.wifi_mode, args.osr)
if args.rms is None:
    args.rms = 0.30 if args.signal == "lte12" else 0.27
ts = sig.fs / 61.44e6                     # time scale vs the 61.44 MS/s reference design
lpf_len = {"auto": 320 if args.signal == "lte12" else None, "iir": None}.get(args.lpf_len, None)
if args.lpf_len not in ("auto", "iir"):
    lpf_len = int(args.lpf_len)
taus = (30 * ts, 150 * ts)                # same physical LPF time constants (0.49 / 2.44 us)
cfr = args.cfr if args.cfr is not None else (8.0 if args.signal == "lte12" else None)
pa, gmp = PA(args.pa, tau=80 * ts), GMP()  # thermal tau = 1.30 us at any fs
# DLA window: FIR features are exact after their length (400 at 61.44 MS/s); an IIR state started
# at zero needs ~5 tau of warm-up. The scored part of the window is kept at 5744 * ts samples.
pre = int(400 * ts) if lpf_len is not None else int(5 * max(taus))
win = int(5744 * ts) + pre
scale = 0.35 if args.quick else 1.0
it = lambda n: max(200, int(n * scale))
n_tr, n_te = int(round(52608 * ts / sig.sym_len)), int(round(43840 * ts / sig.sym_len))
print(sig.describe(), f"| CFR {cfr} | LPF {'IIR' if lpf_len is None else lpf_len} taus {np.round(taus, 1)} "
      f"| DLA win {win} pre {pre} | {n_tr}/{n_te} symbols", flush=True)

x_tr, _ = sig.generate(n_tr, rng, papr_db=cfr)
x_te, ref_te = sig.generate(n_te, rng, papr_db=cfr)
x_tr, x_te = args.rms * x_tr, args.rms * x_te
ref_te = sig.demod(x_te)
meas = lambda z: pa(z, args.snr, rng)
T0 = time.time()
log = lambda s: print(f"[{time.time() - T0:6.1f}s] {s}", flush=True)

def metrics(z):
    y = pa(z)                                   # clean PA output for evaluation
    lo, hi = sig.acpr(y)
    ev = sig.evm(y, ref_te)
    r = dict(nmse=nmse_db(y, x_te), acpr_l=lo, acpr_h=hi, acpr=max(lo, hi),
             evm=ev, evm_db=float(sig.evm_db(ev)), peak_in=float(np.abs(z).max()))
    if sig.mask is not None:
        r["mask_margin"] = sig.mask_margin(y)
    return r

res, Z = {}, {}
Z["No DPD"] = x_te
res["No DPD"] = metrics(x_te)
log(f"No DPD      {res['No DPD']}")

# ---------------------------------------------------------------- GMP ILA
c_ila, _ = ila(pa, gmp, x_tr, iters=4, snr_db=args.snr, rng=rng)
Z["GMP-ILA"] = gmp.basis(x_te) @ c_ila
res["GMP-ILA"] = metrics(Z["GMP-ILA"])
log(f"GMP-ILA     {res['GMP-ILA']}")

# ---------------------------------------------------------------- surrogate, round 1
z_ila_tr = gmp.basis(x_tr) @ c_ila
caps_z = [x_tr, 1.15 * x_tr, z_ila_tr]
caps_y = [meas(z) for z in caps_z]
sur = Surrogate(np.random.default_rng(1), lpf_taus=taus, lpf_len=lpf_len)
sur_log = []
sur.fit(caps_z, caps_y, rng, iters=it(8000), log=sur_log)
sur_nmse = nmse_db(sur(Z["GMP-ILA"]), pa(Z["GMP-ILA"]))
log(f"surrogate NMSE vs true PA (test, ILA drive): {sur_nmse:.1f} dB")

# ---------------------------------------------------------------- DLA round 1
dla_logs = {}
dpd_g = GMPNNDPD(gmp, c_ila, x_tr, np.random.default_rng(2), use_nn=False, lpf_taus=taus, lpf_len=lpf_len)
dpd_n = GMPNNDPD(gmp, c_ila, x_tr, np.random.default_rng(3), use_nn=True, lpf_taus=taus, lpf_len=lpf_len)
for name, d, n in (("GMP-DLA", dpd_g, 2500), ("GMP+NN-DLA", dpd_n, 5000)):
    dla_logs[name] = []
    d.train_dla(sur, x_tr, iters=it(n), win=win, pre=pre, rng=rng, log=dla_logs[name])
    Z[name] = d(x_te)
    res[name] = metrics(Z[name])
    log(f"{name:11s} r1 {res[name]}")
sur_nmse2 = sur_nmse

# ---------------------------------------------------------------- rounds 2..R: recapture + fine-tune
for r in range(2, args.rounds + 1):
    for d in (dpd_g, dpd_n):
        z = d(x_tr)
        caps_z.append(z)
        caps_y.append(meas(z))
    sur.fit(caps_z, caps_y, rng, iters=it(4000), lr=7e-4, init=False, log=sur_log)
    sur_nmse2 = nmse_db(sur(dpd_n(x_te)), pa(dpd_n(x_te)))
    log(f"surrogate NMSE r{r} (test, NN-DPD drive): {sur_nmse2:.1f} dB")
    for name, d, n in (("GMP-DLA", dpd_g, 1500), ("GMP+NN-DLA", dpd_n, 3000)):
        d.train_dla(sur, x_tr, iters=it(n), win=win, pre=pre, lr_gmp=1e-4, lr_nn=4e-4, rng=rng,
                    log=dla_logs[name])
        Z[name] = d(x_te)
        res[name] = metrics(Z[name])
        log(f"{name:11s} r{r} {res[name]}")

# ---------------------------------------------------------------- complexity
cx = {"GMP-ILA": dict(coeffs=gmp.P, real_ops=gmp.complexity(), tanh=0),
      "GMP-DLA": dict(coeffs=gmp.P, real_ops=gmp.complexity(), tanh=0)}
ops, nt = dpd_n.complexity()
nn_params = sum(p.size for p in dpd_n.mlp.params)
cx["GMP+NN-DLA"] = dict(coeffs=gmp.P, nn_params=int(nn_params), real_ops=int(ops), tanh=int(nt))
for k in cx:
    res[k]["complexity"] = cx[k]

cfg = dict(vars(args), signal_desc=sig.describe(), fs=sig.fs, bw=sig.bw, cfr_used=cfr,
           lpf_used="iir" if lpf_len is None else lpf_len, lpf_taus=list(taus), win=win, pre=pre,
           n_sym=[n_tr, n_te], evm_spec_db=sig.evm_spec_db, mask=sig.mask)
out = dict(config=cfg, sur_nmse_r1=sur_nmse, sur_nmse_final=sur_nmse2, results=res,
           dla_logs=dla_logs, sur_log=sur_log, runtime_s=time.time() - T0)
import os
os.makedirs(args.out, exist_ok=True)
json.dump(out, open(f"{args.out}/results.json", "w"), indent=2)
np.savez_compressed(f"{args.out}/waveforms.npz", x=x_te, **{k.replace("+", "p"): v for k, v in Z.items()},
                    **{"y_" + k.replace("+", "p"): pa(v) for k, v in Z.items()})
np.savez(f"{args.out}/dpd_params.npz", gmp_terms=np.array(gmp.terms), c_ila=c_ila, c_gmp_dla=dpd_g.c,
         c_nn_gmp=dpd_n.c, feat_mu=dpd_n.feat.mu, feat_scale=dpd_n.feat.scale,
         **{f"W{i}": W for i, W in enumerate(dpd_n.mlp.W)}, **{f"b{i}": b for i, b in enumerate(dpd_n.mlp.b)})

print("\n{:<12}{:>9}{:>10}{:>9}{:>9}{:>9}{:>10}".format("", "NMSE dB", "ACPR dBc", "EVM %", "EVM dB",
                                                       "mask dB", "peak|z|"))
for k, r in res.items():
    print("{:<12}{:>9.1f}{:>10.1f}{:>9.2f}{:>9.1f}{:>9}{:>10.2f}".format(
        k, r["nmse"], r["acpr"], r["evm"], r["evm_db"],
        f"{r['mask_margin']:+.1f}" if "mask_margin" in r else "-", r["peak_in"]))
