"""Test signals (CP-OFDM: LTE-like 12 MHz and IEEE 802.11 20 MHz) and linearity metrics:
NMSE, ACPR/ACLR, EVM, spectral-mask margin."""
import numpy as np
from scipy.signal import welch, firwin, fftconvolve


class OFDM:
    """CP-OFDM with QAM on an arbitrary set of subcarriers.

    Defaults reproduce the original 12 MHz test signal: 200 contiguous subcarriers around DC
    (DC nulled), nfft 1024 at 61.44 MHz, adjacent channels at +-1.1*BW.

    Arguments
      nfft      FFT size at the *DPD* sample rate fs (for oversampled signals this includes
                the oversampling: nfft = nfft_base * osr)
      tones     signed subcarrier indices to load (default: +-1..+-n_act/2)
      ncp       cyclic-prefix length in samples at fs
      qam       constellation order (square QAM)
      fs        sample rate (Hz)
      adj_offset  adjacent-channel centre offset in Hz (default 1.1*bw)
      mask      optional transmit spectral mask [(offset_Hz, dBr), ...] (linear-in-dB interpolation)
      evm_spec_db optional EVM requirement in dB, reported alongside measured EVM
      eq        'gain' (single complex gain) or 'subcarrier' (per-subcarrier LS, like a receiver
                using an ideal channel estimate) for EVM
      filt_taps, filt_guard  TX channel filter length and cutoff guard (Hz beyond bw/2)
    """

    def __init__(self, nfft=1024, n_act=200, ncp=72, qam=64, fs=61.44e6, tones=None,
                 adj_offset=None, mask=None, evm_spec_db=None, eq="gain", name="lte12",
                 filt_taps=401, filt_guard=None):
        self.nfft, self.ncp, self.qam, self.fs, self.name = nfft, ncp, qam, fs, name
        if tones is None:
            h = n_act // 2
            tones = np.r_[1:h + 1, -h:0]
        self.tones = np.asarray(tones)
        self.idx = np.mod(self.tones, nfft)
        self.n_act = len(self.tones)
        df = fs / nfft
        self.df = df
        self.bw = (self.tones.max() - self.tones.min() + 1) * df if filt_guard is not None else self.n_act * df
        self.adj_offset = 1.1 * self.bw if adj_offset is None else adj_offset
        self.mask, self.evm_spec_db, self.eq = mask, evm_spec_db, eq
        self.filt_taps = filt_taps
        self.filt_cut = 0.52 * self.bw if filt_guard is None else self.bw / 2 + filt_guard
        m = int(round(np.sqrt(qam)))
        self.levels = 2 * np.arange(m) - m + 1
        self.qnorm = np.sqrt(2 * (qam - 1) / 3)

    @property
    def sym_len(self):
        return self.nfft + self.ncp

    def describe(self):
        return (f"{self.name}: {self.n_act} tones x {self.df / 1e3:.2f} kHz, {self.qam}-QAM, "
                f"BW {self.bw / 1e6:.2f} MHz, fs {self.fs / 1e6:.2f} MS/s")

    def generate(self, n_sym, rng, papr_db=8.0, cfr_iters=5):
        """Unit-rms waveform: OFDM -> TX channel filter -> optional clip-and-filter CFR
        (papr_db=None disables CFR). Returns (x, ref) with ref = demod(x)."""
        L = self.levels
        syms = (rng.choice(L, (n_sym, self.n_act)) + 1j * rng.choice(L, (n_sym, self.n_act))) / self.qnorm
        X = np.zeros((n_sym, self.nfft), complex)
        X[:, self.idx] = syms
        t = np.fft.ifft(X, axis=1) * np.sqrt(self.nfft)
        x = np.concatenate([t[:, -self.ncp:], t], axis=1).ravel()
        h = firwin(self.filt_taps, self.filt_cut, fs=self.fs, window=("kaiser", 9.0))
        filt = lambda s: fftconvolve(s, h, mode="same")
        x = filt(x)
        if papr_db is not None:
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
        """RMS EVM (%) after equalisation ('gain': one complex gain; 'subcarrier': per-tone LS)."""
        S = self.demod(y)
        if self.eq == "subcarrier":
            g = np.sum(S.conj() * ref, 0) / np.sum(np.abs(S) ** 2, 0)
        else:
            g = np.vdot(S.ravel(), ref.ravel()) / np.vdot(S.ravel(), S.ravel())
        e = g * S - ref
        return 100 * np.sqrt(np.mean(np.abs(e) ** 2) / np.mean(np.abs(ref) ** 2))

    @staticmethod
    def evm_db(evm_pct):
        return 20 * np.log10(evm_pct / 100)

    def psd(self, y, nperseg=None):
        """Two-sided Welch PSD. Default resolution: 4096 bins (LTE) or ~50 kHz bins
        (Blackman-Harris ENBW ~100 kHz, i.e. the 802.11 mask RBW) when a mask is set."""
        if nperseg is None:
            nperseg = 4096 if self.mask is None else int(round(self.fs / 50e3))
        f, P = welch(y, fs=self.fs, nperseg=nperseg, return_onesided=False,
                     window="blackmanharris", detrend=False)
        return np.fft.fftshift(f), np.fft.fftshift(P)

    def acpr(self, y):
        """(lower, upper) adjacent-channel power ratio in dBc. Measurement BW = occupied BW,
        adjacent channels centred at +-adj_offset (1.1*BW for LTE-like, 20 MHz for Wi-Fi)."""
        f, P = self.psd(y)
        bw, off = self.bw, self.adj_offset
        ch = np.sum(P[np.abs(f) <= bw / 2])
        lo = np.sum(P[np.abs(f + off) <= bw / 2])
        hi = np.sum(P[np.abs(f - off) <= bw / 2])
        return 10 * np.log10(lo / ch), 10 * np.log10(hi / ch)

    def mask_limit(self, f):
        fo, lv = np.array(self.mask).T
        return np.interp(np.abs(f), fo, lv, left=0.0, right=lv[-1])

    def mask_margin(self, y):
        """Worst-case margin (dB, >0 = pass) of the PSD in dBr (relative to the peak in-band
        PSD) against the spectral mask, over |f| >= first mask breakpoint."""
        f, P = self.psd(y)
        dbr = 10 * np.log10(P / P[np.abs(f) <= self.bw / 2].max())
        sel = (np.abs(f) >= self.mask[0][0]) & (np.abs(f) <= self.fs / 2 * 0.98)
        return float(np.min(self.mask_limit(f[sel]) - dbr[sel]))


# ----------------------------------------------------------------------------- 802.11 20 MHz
# 20 MHz transmit spectral masks, (offset Hz, dBr), linear-in-dB between points
MASK_LEGACY = [(9e6, 0.0), (11e6, -20.0), (20e6, -28.0), (30e6, -40.0)]       # 11a/g, 11n/ac 20 MHz (5 GHz)
MASK_HE20 = [(9.75e6, 0.0), (10.5e6, -20.0), (20e6, -28.0), (30e6, -40.0)]     # 11ax / 11be 20 MHz

WIFI_MODES = {
    #        nfft@20MS/s  tones                               QAM   EVM req (dB, top MCS)  mask
    "legacy": (64,  np.r_[-26:0, 1:27],                        64,  -25.0,  MASK_LEGACY),   # 11a/g 54 Mb/s
    "ht":     (64,  np.r_[-28:0, 1:29],                        64,  -28.0,  MASK_LEGACY),   # 11n MCS7
    "vht":    (64,  np.r_[-28:0, 1:29],                        256, -32.0,  MASK_LEGACY),   # 11ac MCS9
    "he":     (256, np.r_[-122:-1, 2:123],                     1024, -35.0, MASK_HE20),     # 11ax MCS11, RU242
    "eht":    (256, np.r_[-122:-1, 2:123],                     4096, -38.0, MASK_HE20),     # 11be MCS13, RU242
}


def wifi20(mode="he", osr=6, gi_us=0.8):
    """IEEE 802.11 20 MHz OFDM data symbols at fs = 20 MHz * osr.
    All loaded tones carry QAM (pilots treated as data); no preamble/packet structure."""
    nb, tones, qam, evm_req, mask = WIFI_MODES[mode]
    fs = 20e6 * osr
    ncp = int(round(gi_us * 1e-6 * fs))
    return OFDM(nfft=nb * osr, ncp=ncp, qam=qam, fs=fs, tones=tones, adj_offset=20e6, mask=mask,
                evm_spec_db=evm_req, eq="subcarrier", name=f"wifi20-{mode}",
                filt_taps=int(2 * round(0.86e-5 * fs / 2) + 1), filt_guard=0.5e6)


def make_signal(kind="lte12", wifi_mode="he", osr=6):
    if kind == "lte12":
        return OFDM()
    if kind == "wifi20":
        return wifi20(wifi_mode, osr)
    raise ValueError(kind)


def nmse_db(y, ref):
    return 10 * np.log10(np.sum(np.abs(y - ref) ** 2) / np.sum(np.abs(ref) ** 2))
