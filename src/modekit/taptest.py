import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TapTest:
    """
    A given tap-test.

    Attributes
    ----------
    sample_rate : float
        Sampling rate in Hz.
    channels : list[str]
        Channel names (tap-suffix stripped).
    data : np.ndarray
        Signal array of shape (n_reps, n_channels, n_samples).
    bad_reps : np.ndarray
        Boolean mask of shape (n_reps,); True marks a rep to drop from
        analysis. Defaults to all-False (every rep kept).
    """

    sample_rate: float
    channels: list[str]
    data: np.ndarray  # (n_reps, n_channels, n_samples)
    bad_reps: np.ndarray | None = None  # bool (n_reps,), True = drop

    def __post_init__(self):
        if self.bad_reps is None:
            self.bad_reps = np.zeros(self.data.shape[0], dtype=bool)

    @property
    def good(self) -> np.ndarray:
        """Data with bad reps removed: (n_good, n_channels, n_samples)."""
        return self.data[~self.bad_reps]

    def flag(self, *reps: int) -> None:
        """Mark one or more reps (0-based) as bad."""
        self.bad_reps[list(reps)] = True

    def load_channel(self, name: str, good: bool = False) -> np.ndarray:
        """
        Return taps for one channel.

        Parameters
        ----------
        name : str
            Channel name (as listed in ``channels``).
        good : bool
            If True, drop reps flagged in ``bad_reps``.

        Returns
        -------
        np.ndarray, shape (n_reps, n_samples)
        """
        data = self.good if good else self.data
        return data[:, self.channels.index(name), :]


class DataLoader:

    def __init__(self, data_dir):
        """
        Initializes the DataLoader with the folder of the tap-test CSVs.

        Parameters
        ----------
        data_dir : str or pathlib.Path
            Folder holding one tab-separated CSV per tap-test set (see
            :meth:`load_data`).

        Raises
        ------
        FileNotFoundError
            If the data directory does not exist.
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist.")

    def load_data(self) -> dict[str, TapTest]:
        """
        Loads all CSV tap-tests from data_dir into a dictionary.

        Each CSV file is one tap-test set, tab-separated: the sample rate
        [Hz] as the second field of the first line, the column names
        (``<channel>_tap_<rep>``) on the second, units on the third, then the
        samples. Columns repeat the same channels across the reps, so data is
        reshaped into (n_reps, n_channels, n_samples).

        Returns
        -------
        dict[str, TapTest]
            Keyed by tap-test name (CSV stem).
        """
        tap_tests = {}
        for csv_path in sorted(self.data_dir.glob("*.csv")):
            with open(csv_path) as f:
                lines = f.readlines()

            sample_rate = float(lines[0].split("\t")[1])
            all_columns = lines[1].strip().split("\t")
            # units row is lines[2], data starts at line 3
            raw = np.loadtxt(csv_path, delimiter="\t", skiprows=3)

            # Infer channel names from tap_1 columns (strip the _tap_N suffix)
            tap1_cols = [c for c in all_columns if c.endswith("_tap_1")]
            channels = [c.removesuffix("_tap_1") for c in tap1_cols]
            n_channels = len(channels)
            n_reps = len(all_columns) // n_channels

            assert len(all_columns) == n_channels * n_reps, (
                f"{csv_path.name}: column count {len(all_columns)} is not "
                f"divisible by n_channels={n_channels}"
            )

            # (n_samples, n_reps * n_channels) -> (n_samples, n_reps, n_channels)
            # -> (n_reps, n_channels, n_samples)
            data = raw.reshape(len(raw), n_reps, n_channels).transpose(1, 2, 0)

            # Trim trailing-zero padding
            tap_lengths = [
                (
                    np.where(np.any(data[t, :, :] != 0, axis=0))[0][-1] + 1
                    if np.any(data[t, :, :] != 0)
                    else data.shape[2]
                )
                for t in range(n_reps)
            ]
            data = data[:, :, : min(tap_lengths)]

            tap_tests[csv_path.stem] = TapTest(
                sample_rate=sample_rate,
                channels=channels,
                data=data,
            )

        return tap_tests


# --- data-quality gate ---
def _is_clipping(x, clipped_samples):
    s = np.sort(np.abs(x))[::-1]
    return int(np.sum(s == s[0])) >= clipped_samples


def _is_double_impact(
    x,
    fs,
    min_fraction=0.15,
    min_delay_ms=2.0,
    pre_ms=5.0,
    post_ms=300.0,
):
    """Real-cepstrum detector for double impacts.

    See: at the end of http://scholar.lib.vt.edu/ejournals/MODAL/ijaema_v7n2/trethewey/trethewey.pdf
    """
    # Window around the impact to maximize SNR
    pk = int(np.argmax(np.abs(x)))
    start = max(0, pk - int(pre_ms * 1e-3 * fs))
    stop = min(len(x), pk + int(post_ms * 1e-3 * fs))
    seg = x[start:stop]
    n = len(seg)

    # Real cepstrum: IFFT( log|FFT(seg)| ).
    X = np.fft.rfft(seg)
    peak_mag = np.max(np.abs(X))
    if peak_mag == 0:
        return False
    log_mag = np.log(np.abs(X) + peak_mag * 1e-8)  # scale-relative floor avoids log(0)
    cepstrum = np.real(np.fft.irfft(log_mag, n=n))

    # Quefrency index k corresponds to delay k/fs seconds. The cepstrum is
    # symmetric, so search only positive quefrencies up to half the window.
    min_q = max(1, int(min_delay_ms * 1e-3 * fs))
    max_q = n // 2

    peak = np.max(np.abs(cepstrum[min_q:max_q]))
    est_fraction = 2.0 * peak  # peak = a/2  ->  a = 2*peak
    return est_fraction > min_fraction


def find_bad_reps(
    x,
    fs,
    signal="i",
    clipped_samples=3,
    min_fraction=0.15,
    min_delay_ms=2.0,
    pre_ms=5.0,
    post_ms=300.0,
    print_summary=True,
    name=None,
):
    """Check signals for clipping and double impacts.
    Always confirm flagged reps by inspecting the time-domain signals first.

    Intended as a pre-processing gate: run this before constructing
    :class:`~modekit.freq_models.EmaModel`, then drop the flagged reps.

    Parameters
    ----------
    x : array-like  shape (n_reps, C, n_t)
        Signals to inspect, one channel per column.
    fs : float
        Sample rate [Hz].
    signal : {'i', 'o'}
        ``'i'`` (input): checks clipping **and** double impact; used for
        force/excitation channels.
        ``'o'`` (output): checks clipping only; used for response channels.
    clipped_samples : int
        Minimum number of samples at the peak amplitude to flag as saturated.
    min_fraction : float
        Smallest second-hit amplitude fraction (relative to the main hit) that
        triggers a flag.
    min_delay_ms : float
        Excludes the low-quefrency content of the main hit.
    pre_ms, post_ms : float
        Impact-region window before/after the main peak [ms].
    print_summary : bool
        If True, print a summary of flagged reps.
    name : str, optional
        Label printed in the summary header.

    Returns
    -------
    bad_mask : np.ndarray bool, shape (n_reps, C)
        True where a (rep, channel) pair is flagged.

    Examples
    --------
    >>> # hammer: (n_reps, 1, n_t) inputs, accel: (n_reps, Q, n_t) outputs
    >>> bad_f = find_bad_reps(hammer, fs, signal='i')
    >>> bad_r = find_bad_reps(accel, fs, signal='o')
    >>> bad = bad_f.any(axis=1) | bad_r.any(axis=1)
    >>> model = EmaModel(hammer[~bad], accel[~bad], fs)
    """
    if signal not in ("i", "o"):
        raise ValueError(f"signal must be 'i' or 'o', got {signal!r}")
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[:, None, :]
    n_reps, C, _ = x.shape
    bad_mask = np.zeros((n_reps, C), dtype=bool)
    lines = []

    for r in range(n_reps):
        for c in range(C):
            sig = x[r, c]
            clipping = _is_clipping(sig, clipped_samples)
            double_impact = signal == "i" and _is_double_impact(
                sig, fs, min_fraction, min_delay_ms, pre_ms, post_ms
            )
            if clipping or double_impact:
                bad_mask[r, c] = True
                reasons = ", ".join(
                    [
                        *(["clipping"] if clipping else []),
                        *(["double impact"] if double_impact else []),
                    ]
                )
                lines.append(f"  rep {r+1}, channel {c}: {reasons}")

    if print_summary and lines:
        header = name if name is not None else "Set"
        print(f"{header}:\n" + "\n".join(lines))

    return bad_mask
