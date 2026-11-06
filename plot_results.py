"""Plots for run_demo.py output (light theme): PSD (+ 802.11 mask for Wi-Fi), AM/AM, AM/PM,
DLA training curves.   python plot_results.py [results_dir]"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dpd.signals import make_signal

out = sys.argv[1] if len(sys.argv) > 1 else "results"
R = json.load(open(f"{out}/results.json"))
W = np.load(f"{out}/waveforms.npz")
C = R["config"]
sig = make_signal(C.get("signal", "lte12"), C.get("wifi_mode", "he"), C.get("osr", 6))
wifi = sig.mask is not None
x = W["x"]

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"No DPD": "#8a8984", "GMP-ILA": "#2a78d6", "GMP-DLA": "#eb6834", "GMP+NN-DLA": "#1baf7a"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})

fig, ax = plt.subplots(2, 2, figsize=(12, 8.2))
# PSD
a = ax[0, 0]
for k in COL:
    y = W["y_" + k.replace("+", "p")]
    f, P = sig.psd(y)
    r = R["results"][k]
    ref = P[np.abs(f) <= sig.bw / 2].max()
    lab = f"{k}  (ACPR {r['acpr']:.1f} dBc" + (f", mask {r['mask_margin']:+.1f} dB)" if wifi else ")")
    a.plot(f / 1e6, 10 * np.log10(P / ref), color=COL[k], lw=1.4 if k != "No DPD" else 1.2, label=lab)
off = sig.adj_offset
for s in (-1, 1):
    a.axvspan((s * off - sig.bw / 2) / 1e6, (s * off + sig.bw / 2) / 1e6, color="#f0efec", zorder=0)
xl = 30 if not wifi else 50
if wifi:
    fm = np.linspace(-xl, xl, 2001) * 1e6
    a.plot(fm / 1e6, sig.mask_limit(fm), color=INK, lw=1.0, ls="--", label="802.11 20 MHz mask")
a.set(xlim=(-xl, xl), ylim=(-90 if not wifi else -118, 5), xlabel="Frequency (MHz)",
      ylabel="PSD (dBr)" if wifi else "PSD (dB, normalised)",
      title="PA output spectrum (shaded = adjacent channels)")
a.legend(loc="lower center", fontsize=8)

# AM/AM and AM/PM (normalised to linear target)
idx = np.random.default_rng(0).choice(len(x), 6000, replace=False)
for k in COL:
    y = W["y_" + k.replace("+", "p")][idx]
    ax[0, 1].scatter(np.abs(x[idx]), np.abs(y), s=1.5, color=COL[k], alpha=0.35, lw=0, rasterized=True)
    ax[1, 0].scatter(np.abs(x[idx]), np.degrees(np.angle(y * np.conj(x[idx]))), s=1.5, color=COL[k],
                     alpha=0.35, lw=0, rasterized=True)
m = np.abs(x).max()
ax[0, 1].plot([0, m], [0, m], color=INK, lw=0.8, ls="--")
ax[0, 1].set(xlabel="|x| (DPD input)", ylabel="|y| (PA output)", title="AM/AM, linearised output")
ax[1, 0].set(xlabel="|x| (DPD input)", ylabel="∠y − ∠x (deg)", title="AM/PM, linearised output", ylim=(-25, 25))
for k in COL:
    ax[0, 1].scatter([], [], color=COL[k], s=12, label=k)
ax[0, 1].legend(loc="upper left", markerscale=1.5)

# DLA learning curves (training-window NMSE through the surrogate)
a = ax[1, 1]
for k in ("GMP-DLA", "GMP+NN-DLA"):
    L = R["dla_logs"][k]
    a.plot(np.arange(len(L)) * 100, L, color=COL[k], lw=1.6, label=k)
a.axhline(R["results"]["GMP-ILA"]["nmse"], color=COL["GMP-ILA"], lw=1, ls="--", label="GMP-ILA (true PA, test)")
a.set(xlabel="DLA iteration (all rounds)", ylabel="NMSE via surrogate (dB)", title="Direct-learning convergence")
a.legend()

desc = (f"802.11 {C.get('wifi_mode', 'he').upper()} 20 MHz, {sig.qam}-QAM, {sig.fs/1e6:.0f} MS/s" if wifi
        else f"{sig.bw/1e6:.0f} MHz OFDM {sig.qam}-QAM")
fig.suptitle(f"GMP vs ML-augmented GMP DPD — {C['pa']} PA, {desc}, rms {C['rms']:.2f}, "
             f"{C['snr']:.0f} dB capture SNR", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(f"{out}/dpd_results.png", dpi=150)
print("saved", f"{out}/dpd_results.png")
