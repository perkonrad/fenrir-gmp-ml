"""Finite-difference check of the hand-written complex backprop (surrogate + DPD chain)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dpd import Surrogate, GMP, GMPNNDPD


def loss_and_grad(sur, z, t):
    e = sur.forward(z) - t
    return np.sum(np.abs(e) ** 2), sur.backward(2 * e)


def main():
    rng = np.random.default_rng(0)
    N = 700
    z = 0.4 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    t = 0.4 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    sur = Surrogate(rng, hidden=(16, 16))
    sur.feat.fit_scale(z)
    sur.h = rng.standard_normal(sur.L + 1) * 0.3 + 0j
    sur.mlp = type(sur.mlp)([sur.feat.D, 16, 16, 2], rng, out_scale=1.0)
    L0, g = loss_and_grad(sur, z, t)
    h = 1e-6
    errs = []
    for n in rng.integers(0, N, 12):
        for d in (1, 1j):
            zp = z.copy(); zp[n] += h * d
            zm = z.copy(); zm[n] -= h * d
            num = (loss_and_grad(sur, zp, t)[0] - loss_and_grad(sur, zm, t)[0]) / (2 * h)
            ana = g[n].real if d == 1 else g[n].imag
            errs.append(abs(num - ana) / (abs(num) + 1e-9))
    print(f"surrogate dL/dz   max rel err = {max(errs):.2e}")
    assert max(errs) < 1e-4

    # DPD parameter gradient through the frozen surrogate (one DLA step's math)
    x = 0.3 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    gmp = GMP(Ka=3, La=1, Kb=2, Lb=1, Mb=1, Kc=2, Lc=1, Mc=1)
    c0 = np.zeros(gmp.P, complex); c0[0] = 1
    dpd = GMPNNDPD(gmp, c0, x, rng, hidden=(8, 8))
    dpd.mlp = type(dpd.mlp)([dpd.feat.D, 8, 8, 2], rng, out_scale=1.0)

    def L(dpd):
        e = sur.forward(dpd(x)) - t
        return np.sum(np.abs(e) ** 2), e

    _, e = L(dpd)
    F = dpd.feat.forward(x)
    dpd.mlp.forward(F)
    gz = sur.backward(2 * e)
    gw = gz * x.conj()
    gp, _ = dpd.mlp.backward(np.stack([gw.real, gw.imag], 1))
    W = dpd.mlp.W[0]
    errs = []
    for (i, j) in [(0, 0), (3, 2), (5, 7)]:
        W[i, j] += h; Lp = L(dpd)[0]
        W[i, j] -= 2 * h; Lm = L(dpd)[0]
        W[i, j] += h
        num = (Lp - Lm) / (2 * h)
        errs.append(abs(num - gp[0][i, j]) / (abs(num) + 1e-9))
    print(f"DPD dL/dW         max rel err = {max(errs):.2e}")
    assert max(errs) < 1e-4
    print("gradcheck OK")


if __name__ == "__main__":
    main()
