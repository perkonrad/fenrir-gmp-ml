"""Test signal (CFR'd OFDM) and linearity metrics: NMSE, ACPR, EVM."""
import numpy as np
from scipy.signal import welch, firwin, fftconvolve


class OFDM:
    """Contiguous CP-OFDM, QAM on n_act subcarriers around DC (DC nulled)."""

    def __init__(self, nfft=1024, n_act=200, ncp=72, qam=64, fs=61.44e6):
        self.nfft, self.n_act, self.ncp, self.qam, self.fs = nfft, n_act, ncp, qam, fs
        h = n_act // 2
        self.idx = np.r_[1:h + 1, nfft - h:nfft]
        self.bw = n_act * fs / nfft
        m = int(np.sqrt(qam))
        self.levels = 2 * np.arange(m) - m + 1
        self.qnorm = np.sqrt(2 * (qam - 1) / 3)

    @property
    def sym_len(self):
        return self.nfft + self.ncp

    def generate(self, n_sym, rng, papr_db=8.0, cfr_iters=5):
        L = self.levels
        syms = (rng.choice(L, (n_sym, self.n_act)) + 1j * rng.choice(L, (n_sym, self.n_act))) / self.qnorm
        X = np.zeros((n_sym, self.nfft), complex)
        X[:, self.idx] = syms
        t = np.fft.ifft(X, axis=1) * np.sqrt(self.nfft)
        x = np.concatenate([t[:, -self.ncp:], t], axis=1).ravel()
        # channel filter (kills CP-boundary splatter) + iterative clip-and-filter CFR
        h = firwin(401, 0.52 * self.bw, fs=self.fs, window=("kaiser", 9.0))
        filt = lambda s: fftconvolve(s, h, mode="same")
        x = filt(x)
        for _ in range(cfr_iters):
            thr = np.sqrt(np.mean(np.abs(x) ** 2)) * 10 ** (papr_db / 20)
            a = np.abs(x)
            x = filt(np.where(a > thr, x * thr / np.maximum(a, 1e-12), x))
        x /= np.sqrt(np.mean(np.abs(x) ** 2))
        return x, self.demod(x)          # reference = what the transmitted signal carries

    def demod(self, y):
        n_sym = len(y) // self.sym_len
        Y = y[: n_sym * self.sym_len].reshape(n_sym, self.sym_len)[:, self.ncp:]
        return np.fft.fft(Y, axis=1)[:, self.idx] / np.sqrt(self.nfft)

    def evm(self, y, ref):
        """RMS EVM (%) after a single complex-gain LS fit."""
        S = self.demod(y)
        g =np.vdot(S.ravel(), ref.ravel()) / np.vdot(S.ravel(), S.ravel())
        e = g * S - ref
        return 100 * np.sqrt(np.mean(np.abs(e) ** 2) / np.mean(np.abs(ref) ** 2))

    def psd(self, y, nperseg=4096):
        f, P = welch(y, fs=self.fs, nperseg=nperseg, return_onesided=False,
                     window="blackmanharris", detrend=False)
        return np.fft.fftshift(f), np.fft.fftshift(P)

    def acpr(self, y):
        """(lower, upper) ACPR in dBc. Measurement BW = occupied BW; adjacent channels centred
        at +-1.1*BW (10 % guard, LTE-like channel raster)."""
        f, P = self.psd(y)
        bw, off = self.bw, 1.1 * self.bw
        ch = np.sum(P[np.abs(f) <= bw / 2])
        lo = np.sum(P[np.abs(f + off) <= bw / 2])
        hi = np.sum(P[np.abs(f - off) <= bw / 2])
        return 10 * np.log10(lo / ch), 10 * np.log10(hi / ch)


def nmse_db(y, ref):
    return 10 * np.log10(np.sum(np.abs(y - ref) ** 2) / np.sum(np.abs(ref) ** 2))
