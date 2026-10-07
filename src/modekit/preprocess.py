"""Time-domain signal preprocessing ahead of spectral estimation.

Plain functions on ``(..., nt)`` arrays—time on the last axis—so the
``(n_reps, channels, nt)`` layout passes through unchanged.
"""

import numpy as np
from scipy import signal


def detrend_data(data, type="linear"):
    """Remove the mean or a linear trend along the time (last) axis.

    Parameters
    ----------
    data : array-like  (..., nt)
    type : {'linear', 'constant'}
        'linear' subtracts a least-squares line, 'constant' the mean.

    Returns
    -------
    ndarray, same shape as ``data``
    """
    return signal.detrend(np.asarray(data), axis=-1, type=type)


def decimate_data(data, fs, q, **kwargs):
    """Anti-aliased downsampling by an integer factor along time.

    Wraps :func:`scipy.signal.decimate` (zero-phase by default).
    For ``q > 13`` apply in stages (e.g. two calls) as scipy recommends.

    Parameters
    ----------
    data : array-like  (..., nt)
    fs : float  sample rate [Hz]
    q : int  downsampling factor
    **kwargs : forwarded to :func:`scipy.signal.decimate`.

    Returns
    -------
    data_dec : ndarray  (..., ceil(nt / q))
    fs_dec : float  the new sample rate ``fs / q``
    """
    data_dec = signal.decimate(np.asarray(data), q, axis=-1, **kwargs)
    return data_dec, fs / q


def filter_data(data, fs, Wn, order=8, btype="highpass"):
    """Zero-phase Butterworth filter along the time (last) axis.

    Parameters
    ----------
    data : array-like  (..., nt)
    fs : float  sample rate [Hz]
    Wn : float or (float, float)  cutoff frequency/-ies [Hz]
    order : int  Butterworth order. Default 8.
    btype : {'highpass', 'lowpass', 'bandpass', 'bandstop'}

    Returns
    -------
    ndarray, same shape as ``data``
    """
    sos = signal.butter(order, Wn, btype=btype, output="sos", fs=fs)
    return signal.sosfiltfilt(sos, np.asarray(data), axis=-1)
