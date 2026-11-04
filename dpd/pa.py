"""Behavioural PA models (baseband, unit small-signal gain so the linear target is y = x).

Structure (both):  x -> H1 -> static NL -> thermal/trap gain droop -> H2 -> y
  * H1/H2  : short complex FIRs (unity DC gain) -> frequency-dependent memory
  * droop  : gain *= 1 - kappa * LPF_tau{|u|^2}   (slow long-term memory)

Static NL:
  'rapp'    : Rapp AM/AM + saturating AM/PM (smooth class-AB-like)
  'doherty' : carrier (class-AB, saturates at ~half voltage) + peaking (class-C,
              turns on above a threshold with its own phase) -> AM/AM knee and an
              AM/PM bump at peaking turn-on.  Much harder for low-order polynomials.
"""
import numpy as np
from scipy.signal import lfilter


def rapp(A, asat, p):
    return A * (1 + (A / asat) ** (2 * p)) ** (-1 / (2 * p))


def softplus(v, s):
    return np.logaddexp(0, s * v) / s


class PA:
    def __init__(self, kind="doherty",
                 h1=(1.0, -0.18 + 0.10j, 0.06 - 0.02j),
                 h2=(1.0, 0.28 - 0.12j, -0.07j, 0.03 + 0.02j),
                 kappa=0.10, tau=80.0,
                 # rapp params
                 asat=1.0, p=1.1, am_pm_deg=12.0, beta=1.5,
                 # doherty params
                 v_th=0.45, c_sat=0.5, pk_gain=2.2, pk_phase_deg=-25.0, c_pm_deg=6.0, knee=25.0):
        self.kind = kind
        self.h1 = np.asarray(h1, complex) / np.sum(h1)
        self.h2 = np.asarray(h2, complex) / np.sum(h2)
        self.kappa, self.a = kappa, np.exp(-1.0 / tau)
        self.asat, self.p, self.beta = asat, p, beta
        self.alpha = np.deg2rad(am_pm_deg) * (1 + beta)        # phi(|u|=1) = am_pm_deg
        self.v_th, self.c_sat, self.pk_gain, self.knee = v_th, c_sat, pk_gain, knee
        self.pk_ph, self.c_pm = np.deg2rad(pk_phase_deg), np.deg2rad(c_pm_deg)

    def static(self, A):
        """Complex gain G(A) so that v = u * G(|u|)."""
        if self.kind == "rapp":
            g = rapp(A, self.asat, self.p) / np.maximum(A, 1e-12)
            return g * np.exp(1j * self.alpha * A ** 2 / (1 + self.beta * A ** 2))
        c = rapp(A, self.c_sat, 2.0) * np.exp(1j * self.c_pm * (A / self.c_sat) ** 2 / (1 + (A / self.c_sat) ** 2))
        pk = rapp(self.pk_gain * softplus(A - self.v_th, self.knee), self.c_sat, 2.0) * np.exp(1j * self.pk_ph)
        Ac = np.maximum(A, 1e-12)
        return (c + pk) / Ac

    def __call__(self, x, snr_db=None, rng=None):
        u = lfilter(self.h1, 1, x)
        A = np.abs(u)
        s = lfilter([1 - self.a], [1, -self.a], A ** 2)
        v = u * self.static(A) * (1 - self.kappa * s)
        y = lfilter(self.h2, 1, v)
        if snr_db is not None:
            pn = np.mean(np.abs(y) ** 2) * 10 ** (-snr_db / 10)
            y = y + np.sqrt(pn / 2) * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))
        return y
