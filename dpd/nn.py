"""Minimal NumPy deep-learning pieces with hand-written backprop:
MLP, Adam (complex-aware), phase-equivariant feature extractor, NN PA surrogate,
and the GMP + NN residual DPD trained by direct learning (DLA).

Gradient convention for a real loss L and complex variable z:
    g_z = dL/dRe(z) + j dL/dIm(z)
so that for y = a*w:  g_a = g_y*conj(w),  g_w = g_y*conj(a).
"""
import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.signal import fftconvolve

from .gmp import shift


# ----------------------------------------------------------------------------- MLP / Adam
class MLP:
    def __init__(self, sizes, rng, out_scale=1.0):
        self.W, self.b = [], []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            W = rng.standard_normal((a, b)) * np.sqrt(1.0 / a)
            if i == len(sizes) - 2:
                W *= out_scale
            self.W.append(W)
            self.b.append(np.zeros(b))

    @property
    def params(self):
        return self.W + self.b

    def forward(self, X):
        hs, h = [X], X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.tanh(h)
            hs.append(h)
        self._hs = hs
        return h

    def backward(self, g):
        hs, n = self._hs, len(self.W)
        gW, gb = [None] * n, [None] * n
        for i in reversed(range(n)):
            if i < n - 1:
                g = g * (1 - hs[i + 1] ** 2)
            gW[i] = hs[i].T @ g
            gb[i] = g.sum(0)
            g = g @ self.W[i].T
        return gW + gb, g

    def macs(self):
        return sum(W.size for W in self.W), sum(W.shape[1] for W in self.W[:-1])  # MACs, tanh count


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-12):
        self.p, self.lr, self.b1, self.b2, self.eps, self.t = params, lr, b1, b2, eps, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]   # complex v stores Re^2 + j Im^2

    def step(self, grads, lr_scale=1.0):
        self.t += 1
        b1, b2 = self.b1, self.b2
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        lr = self.lr * lr_scale
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            m *= b1
            m += (1 - b1) * g
            if np.iscomplexobj(p):
                v *= b2
                v += (1 - b2) * (g.real ** 2 + 1j * g.imag ** 2)
                mh, vh = m / c1, v / c2
                p -= lr * (mh.real / (np.sqrt(vh.real) + self.eps) + 1j * mh.imag / (np.sqrt(vh.imag) + self.eps))
            else:
                v *= b2
                v += (1 - b2) * g * g
                p -= lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


def cosine(it, n, floor=0.03):
    return floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * it / n))


# ----------------------------------------------------------------------------- features
class Feat:
    """Phase-invariant features of a complex sequence z (differentiable w.r.t. z):
        |z[n-m]|, |z[n-m]|^2                      m in amp_delays
        Re/Im{ z[n-m] conj(z[n]) }                m in prod_delays   (relative phase)
        LPF_tau{|z|^2}[n]                         tau in lpf_taus    (long-term memory)
    Rotating z by a constant phase leaves every feature unchanged, so a network that
    outputs a complex gain w applied as z[n]*w is exactly phase-equivariant."""

    def __init__(self, amp_delays, prod_delays, lpf_taus=(), lpf_len=320, eps=1e-6):
        self.amp, self.prod, self.eps = list(amp_delays), list(prod_delays), eps
        n = np.arange(lpf_len)
        self.kern = [(1 - np.exp(-1 / t)) * np.exp(-n / t) for t in lpf_taus]
        self.scale = None
        self.D = 2 * len(self.amp) + 2 * len(self.prod) + len(self.kern)

    def _raw(self, z):
        q = (z * z.conj()).real
        a = np.sqrt(q + self.eps)
        cols = []
        for m in self.amp:
            cols += [shift(a, m), shift(q, m)]
        for m in self.prod:
            p = shift(z, m) * z.conj()
            cols += [p.real, p.imag]
        for k in self.kern:
            cols.append(fftconvolve(q, k)[: len(z)])
        return np.stack(cols, 1), a

    def fit_scale(self, z):
        F, _ = self._raw(z)
        self.mu = F.mean(0)
        self.scale = 1.0 / (F.std(0) + 1e-12)

    def forward(self, z):
        F, a = self._raw(z)
        self._z, self._a = z, a
        return (F - self.mu) * self.scale

    def backward(self, gF):
        z, a = self._z, self._a
        gF = gF * self.scale
        gq = np.zeros(len(z))   # grad wrt |z|^2
        ga = np.zeros(len(z))   # grad wrt |z|
        g = np.zeros(len(z), complex)
        i = 0
        for m in self.amp:
            ga += shift(gF[:, i], -m)
            gq += shift(gF[:, i + 1], -m)
            i += 2
        for m in self.prod:
            gp = gF[:, i] + 1j * gF[:, i + 1]
            g += shift(gp * z, -m)                 # d/d z[n-m]
            g += gp.conj() * shift(z, m)           # d/d z[n] (conjugated factor)
            i += 2
        for k in self.kern:
            gs = gF[:, i]
            gq += fftconvolve(gs[::-1], k)[: len(z)][::-1]   # adjoint of causal FIR
            i += 1
        return g + ga * z / a + 2 * gq * z


def lagmat(z, L):
    return np.stack([shift(z, l) for l in range(L + 1)], 1)


# ----------------------------------------------------------------------------- PA surrogate
class Surrogate:
    """Differentiable PA model:  y = sum_l h_l z[n-l]  +  z[n] * w(Feat(z))."""

    def __init__(self, rng, L=5, amp=range(0, 6), prod=range(1, 6), lpf_taus=(30, 150), hidden=(64, 64)):
        self.L = L
        self.feat = Feat(amp, prod, lpf_taus)
        self.mlp = MLP([self.feat.D, *hidden, 2], rng, out_scale=0.1)
        self.h = np.zeros(L + 1, complex)

    def fit(self, Z, Y, rng, iters=6000, batch=4096, lr=2e-3, init=True, log=None):
        if self.feat.scale is None:
            self.feat.fit_scale(np.concatenate(Z))
        F = np.vstack([self.feat.forward(z) for z in Z])
        Zl = np.vstack([lagmat(z, self.L) for z in Z])
        Yc = np.concatenate(Y)
        P = np.mean(np.abs(Yc) ** 2)
        if init:
            self.h = np.linalg.lstsq(Zl, Yc, rcond=None)[0]
        opt = Adam([self.h, *self.mlp.params], lr)
        N = len(Yc)
        for it in range(iters):
            idx = rng.integers(0, N, batch)
            zl, z0 = Zl[idx], Zl[idx, 0]
            wo = self.mlp.forward(F[idx])
            w = wo[:, 0] + 1j * wo[:, 1]
            e = zl @ self.h + z0 * w - Yc[idx]
            gy = 2 * e / (batch * P)
            gw = gy * z0.conj()
            gp, _ = self.mlp.backward(np.stack([gw.real, gw.imag], 1))
            opt.step([zl.conj().T @ gy, *gp], cosine(it, iters))
            if log is not None and it % 500 == 0:
                log.append(10 * np.log10(np.mean(np.abs(e) ** 2) / P))
        return self

    def forward(self, z):
        F = self.feat.forward(z)
        wo = self.mlp.forward(F)
        self._w = wo[:, 0] + 1j * wo[:, 1]
        self._z = z
        y = sum(self.h[l] * shift(z, l) for l in range(self.L + 1))
        return y + z * self._w

    def backward(self, gy):
        z, w = self._z, self._w
        gz = sum(shift(self.h[l].conj() * gy, -l) for l in range(self.L + 1))
        gz = gz + gy * w.conj()
        gw = gy * z.conj()
        _, gF = self.mlp.backward(np.stack([gw.real, gw.imag], 1))
        return gz + self.feat.backward(gF)

    def __call__(self, z):
        return self.forward(z)


# ----------------------------------------------------------------------------- DPD
class GMPNNDPD:
    """z = GMP(x) + x[n] * w(Feat(x)).  use_nn=False gives a pure GMP trained by DLA.

    The GMP coefficients are trained in an orthonormalised basis (Phi = Q R, c = R^-1 d)
    so first-order optimisation is well conditioned; the effective c is used at run time."""

    def __init__(self, gmp, c0, x_train, rng, use_nn=True, amp=range(-2, 5), prod=(-2, -1, 1, 2, 3, 4),
                 lpf_taus=(30, 150), hidden=(24, 24)):
        self.gmp, self.use_nn = gmp, use_nn
        Phi = gmp.basis(x_train)
        nrm = np.sqrt(len(x_train))
        Q, R = qr(Phi, mode="economic")
        self.R = R / nrm                                  # Phi = (Q*nrm) @ (R/nrm)
        self.d = self.R @ c0
        self.feat = Feat(amp, prod, lpf_taus)
        self.feat.fit_scale(x_train)
        self.mlp = MLP([self.feat.D, *hidden, 2], rng, out_scale=0.01)

    @property
    def c(self):
        return solve_triangular(self.R, self.d)

    def __call__(self, x):
        z = self.gmp.basis(x) @ self.c
        if self.use_nn:
            wo = self.mlp.forward(self.feat.forward(x))
            z = z + x * (wo[:, 0] + 1j * wo[:, 1])
        return z

    def train_dla(self, sur, x, G=1.0, iters=3000, win=6144, pre=400, lr_gmp=3e-4, lr_nn=1e-3,
                  rng=None, log=None):
        """Direct learning: minimise |S(DPD(x)) - G x|^2 by backprop through frozen surrogate S."""
        Qn = self.gmp.basis(x) @ np.linalg.inv(self.R)          # orthonormal-ish basis on x
        F = self.feat.forward(x) if self.use_nn else None
        t = G * x
        P = np.mean(np.abs(t) ** 2)
        opt_g = Adam([self.d], lr_gmp)
        opt_n = Adam(self.mlp.params, lr_nn) if self.use_nn else None
        for it in range(iters):
            s = rng.integers(0, len(x) - win)
            sl = slice(s, s + win)
            q, xs = Qn[sl], x[sl]
            z = q @ self.d
            if self.use_nn:
                wo = self.mlp.forward(F[sl])
                z = z + xs * (wo[:, 0] + 1j * wo[:, 1])
            e = sur.forward(z) - t[sl]
            e[:pre] = 0
            gy = 2 * e / ((win - pre) * P)
            gz = sur.backward(gy)
            lr = cosine(it, iters)
            opt_g.step([q.conj().T @ gz], lr)
            if self.use_nn:
                gw = gz * xs.conj()
                gp, _ = self.mlp.backward(np.stack([gw.real, gw.imag], 1))
                opt_n.step(gp, lr)
            if log is not None and it % 100 == 0:
                log.append(10 * np.log10(np.sum(np.abs(e) ** 2) / ((win - pre) * P)))
        return self

    def complexity(self):
        ops = self.gmp.complexity()
        if self.use_nn:
            macs, nt = self.mlp.macs()
            ops += macs + 4 * len(self.feat.prod) + 2 * len(self.feat.kern) + 4   # features + x*w
            return ops, nt
        return ops, 0
