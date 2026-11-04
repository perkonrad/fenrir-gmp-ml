"""Generalized Memory Polynomial (Morgan et al., 2006) basis + LS / ILA."""
import numpy as np


def shift(z, m):
    """z[n-m] with zero fill (m<0 = lead)."""
    if m == 0:
        return z
    out = np.zeros_like(z)
    if m > 0:
        out[m:] = z[:-m]
    else:
        out[:m] = z[-m:]
    return out


class GMP:
    """y[n] = sum_{k,l} a_kl x[n-l]|x[n-l]|^(k-1)                       (aligned)
            + sum_{k,l,m} b_klm x[n-l]|x[n-l-m]|^(k-1)                 (lagging env)
            + sum_{k,l,m} c_klm x[n-l]|x[n-l+m]|^(k-1)                 (leading env)
    All k (odd and even) are used."""

    def __init__(self, Ka=7, La=4, Kb=5, Lb=3, Mb=2, Kc=5, Lc=3, Mc=2):
        self.terms = [(k, l, 0) for k in range(1, Ka + 1) for l in range(La + 1)]
        self.terms += [(k, l, m) for k in range(2, Kb + 1) for l in range(Lb + 1) for m in range(1, Mb + 1)]
        self.terms += [(k, l, -m) for k in range(2, Kc + 1) for l in range(Lc + 1) for m in range(1, Mc + 1)]

    @property
    def P(self):
        return len(self.terms)

    def basis(self, x):
        a = np.abs(x)
        return np.stack([shift(x, l) * shift(a, l + m) ** (k - 1) for k, l, m in self.terms], axis=1)

    @staticmethod
    def ls(Phi, t, lam=1e-7):
        A = Phi.conj().T @ Phi
        A += lam * np.trace(A).real / A.shape[0] * np.eye(A.shape[0])
        return np.linalg.solve(A, Phi.conj().T @ t)

    def complexity(self, K_max=7):
        """Rough real-op count per sample (basis gen + complex MACs)."""
        # |x| (sqrt) once, powers up to K_max-1 shared, products per term, complex MAC per coeff
        basis = 2 * self.P          # complex * real per term
        mac = 4 * self.P            # real mults per complex MAC
        return basis + mac


def ila(pa, gmp, x, G=1.0, iters=4, snr_db=None, rng=None, lam=1e-7):
    """Indirect learning: fit post-inverse on y/G -> z, copy as predistorter."""
    Phi_x = gmp.basis(x)
    c = np.zeros(gmp.P, complex)
    c[gmp.terms.index((1, 0, 0))] = 1.0
    hist = []
    for _ in range(iters):
        z = Phi_x @ c
        y = pa(z, snr_db, rng)
        c = gmp.ls(gmp.basis(y / G), z, lam)
        hist.append(c.copy())
    return c, hist
