from abc import ABC, abstractmethod
import numpy as np
from scipy import ndimage, signal


class Window(ABC):
    """
    Abstract base class for applying window functions to signals.
    """

    @abstractmethod
    def apply(self, x, fs, ref=None):
        """

        Parameters
        ----------
        x: np.ndarray
            The signal to be windowed.
        fs: float
            Sample rate [Hz].
        ref: np.ndarray, optional
                The reference signal used to find the peak (e.g., the Hammer).
                If None, the peak is found on `x` itself.

        Returns
        -------
        np.ndarray
            The windowed signal.
        """
        pass


class Tukey(Window):
    """
    Apply a Tukey window centered on the impact peak.
    """

    def __init__(self, pre_ms=5.0, post_ms=5.0, taper_ms=2.0):
        """
        Parameters
        ----------
        pre_ms : float
            Gate extent before impact peak [ms].
        post_ms : float
            Gate extent after impact peak [ms].
        taper_ms : float
            Length of cosine taper at each gate edge [ms]. 0 = rectangular.

        """

        self.pre_ms = pre_ms
        self.post_ms = post_ms
        self.taper_ms = taper_ms

    def apply(self, x, fs, ref=None):
        pre_samples = int(self.pre_ms * fs / 1000)
        post_samples = int(self.post_ms * fs / 1000)
        taper_samples = int(self.taper_ms * fs / 1000)

        x = x.copy()
        for i in range(x.shape[0]):
            peak = int(np.argmax(np.abs(x[i, :])))
            lo = max(0, peak - pre_samples)
            hi = min(x.shape[1], peak + post_samples)

            gate_len = hi - lo
            alpha = (
                2 * taper_samples / gate_len
                if (taper_samples > 0 and gate_len > 0)
                else 0.0
            )  # taper_ms is the length of the taper at each edge, so the total tapered length is 2*taper_ms.
            win = signal.windows.tukey(gate_len, alpha=alpha, sym=True)

            windowed = np.zeros(x.shape[1])
            windowed[lo:hi] = x[i, lo:hi] * win

            x[i, :] = windowed

        return x


class Exponential(Window):
    """
    Apply a one-sided exponential window to a response signal, starting at
    the impact peak. Optionally zeros the pre-trigger region.
    """

    def __init__(self, alpha, pre_ms=1.0, zero_pretrigger=False):
        """
        Parameters
        ----------
        alpha : float
            Exponential decay rate [1/s]. Added damping: sigma_id = sigma_true + alpha.
        pre_ms : float
            Pre-trigger data to keep before the impact [ms].
        zero_pretrigger : bool
            If True, zero samples before peak - pre_ms.
        """
        self.alpha = alpha
        self.pre_ms = pre_ms
        self.zero_pretrigger = zero_pretrigger

    @classmethod
    def from_tap_tests(
        cls,
        data,
        fs,
        target_x=0.01,
        pre_ms=1.0,
        tail_frac=0.05,
        smooth_ms=20.0,
        zero_pretrigger=False,
    ):
        """Size alpha from data and return a ready-to-use Exponential window.

        Parameters
        ----------
        data : np.ndarray | list[np.ndarray]
            Signal array(s) of shape (n_reps, n_channels, n_samples). Channel 0
            is the force; channels 1: are responses. Mask out bad reps before
            passing. Pass a list to size one alpha across several sets.
        fs : float
            Sample rate [Hz].
        target_x : float
            Desired linear fraction at block end (default 0.01 = -40 dB).
        pre_ms : float
            Pre-trigger region to keep [ms] — passed to both the decay
            measurement and the constructed window.
        tail_frac : float
            Fraction of each trimmed block used to estimate residual level.
        smooth_ms : float
            Envelope smoothing window [ms].
        zero_pretrigger : bool
            Forwarded to ``Exponential.__init__``.
        """
        alpha = cls._find_alpha(data, fs, target_x, pre_ms, tail_frac, smooth_ms)
        return cls(alpha=alpha, pre_ms=pre_ms, zero_pretrigger=zero_pretrigger)

    @staticmethod
    def _find_alpha(
        data, fs, target_x=0.01, pre_ms=1.0, tail_frac=0.05, smooth_ms=20.0
    ):
        """Size alpha [1/s] to the slowest-decaying response channel."""
        if isinstance(data, np.ndarray):
            data = [data]

        pre_samples = int(pre_ms * fs / 1000)
        smooth_samples = max(1, int(smooth_ms * fs / 1000))

        worst_decay_db, worst_T_s = -np.inf, None
        for arr in data:
            n_reps, n_channels = arr.shape[0], arr.shape[1]

            for ch_idx in range(1, n_channels):
                for rep_idx in range(n_reps):
                    peak = int(np.argmax(np.abs(arr[rep_idx, 0, :])))
                    response = arr[rep_idx, ch_idx, max(0, peak - pre_samples) :]
                    decay_db, duration_s = Exponential._response_decay(
                        response, fs, smooth_samples, tail_frac
                    )
                    if decay_db is not None and decay_db > worst_decay_db:
                        worst_decay_db, worst_T_s = decay_db, duration_s

        if worst_T_s is None:
            raise ValueError("No valid response data to size alpha from.")

        worst_x = 10.0 ** (worst_decay_db / 20.0)
        return -np.log(target_x / worst_x) / worst_T_s

    @staticmethod
    def _response_decay(response, fs, smooth_samples, tail_frac):
        """Envelope end level relative to peak [dB] and block duration [s]."""
        if response.size < 2 * smooth_samples:
            return None, None

        envelope = ndimage.uniform_filter1d(
            np.abs(signal.hilbert(response)), smooth_samples, mode="nearest"
        )
        peak_level = envelope.max()
        if peak_level == 0:
            return None, None

        tail_samples = max(1, int(tail_frac * envelope.size))
        end_level = np.median(envelope[-tail_samples:])
        decay_db = 20.0 * np.log10(end_level / peak_level)
        return decay_db, response.size / fs

    def apply(self, x, fs, ref=None):
        pre_samples = int(self.pre_ms * fs / 1000)
        tau_samples = fs / self.alpha  # alpha [1/s] -> tau [samples]

        src = ref if ref is not None else x
        x = x.copy()
        for i in range(x.shape[0]):
            peak = int(np.argmax(np.abs(src[i, :])))
            start = max(0, peak - pre_samples)
            n_tail = x.shape[1] - start

            if self.zero_pretrigger:
                x[i, :start] = 0.0

            win = signal.windows.exponential(
                n_tail, center=0, tau=tau_samples, sym=False
            )
            x[i, start:] = x[i, start:] * win

        return x


class CompositeWindow(Window):
    """
    Combines multiple window functions in sequence.
    """

    def __init__(self, windows):
        """
        Initialize with a list of window functions.

        Parameters
        ----------
        windows : list
            List of window functions to apply in sequence
        """
        self.windows = windows

    def apply(self, x, fs, ref=None):
        result = x
        for window in self.windows:
            result = window.apply(result, fs, ref=ref)
        return result
