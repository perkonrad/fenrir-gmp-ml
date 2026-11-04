"""No-DPD vs GMP-ILA over drive level, for both PA models (fast: ~30 s).

python examples/drive_sweep.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dpd import OFDM, PA, GMP, ila, nmse_db

rng = np.random.default_rng(1)
sig, gmp = OFDM(), GMP()
x_tr0, _ = sig.generate(40, rng)
x_te0, _ = sig.generate(40, rng)
print(f"{'PA':<8}{'rms':>5} | {'no DPD NMSE':>12}{'ACPR':>8} | {'GMP-ILA NMSE':>13}{'ACPR':>8}{'EVM %':>7}")
for kind in ("rapp", "doherty"):
    pa = PA(kind)
    for rms in (0.22, 0.26, 0.30, 0.34):
        x_tr, x_te = rms * x_tr0, rms * x_te0
        ref = sig.demod(x_te)
        y0 = pa(x_te)
        c, _ = ila(pa, gmp, x_tr, iters=4, snr_db=50, rng=rng)
        y1 = pa(gmp.basis(x_te) @ c)
        print(f"{kind:<8}{rms:>5.2f} | {nmse_db(y0, x_te):>12.1f}{max(sig.acpr(y0)):>8.1f} | "
              f"{nmse_db(y1, x_te):>13.1f}{max(sig.acpr(y1)):>8.1f}{sig.evm(y1, ref):>7.2f}")
