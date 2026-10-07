"""
Tests for ``EmaModel`` / ``OmaModel``, focused on the output-only (OMA) path.

Synthetic data: white noise filtered through two second-order digital
resonators; channel 0 is dominated by the 40 Hz mode, channel 1 by the
90 Hz mode.
"""

import numpy as np
import pytest

from modekit.freq_models import EmaModel, OmaModel
from modekit.criteria import mac
from modekit.plscf import lambd_to_fn, lambd_to_zeta, window_shift
from modekit.algorithms import LSFD, pLSCF

FS = 512.0
F_TRUE = np.array([40.0, 90.0])
ZETA = 0.02
N_T = 2**15
FFT_ARGS = {"nperseg": 1024}
SHAPES = np.array([[1.0, 0.1], [0.1, 1.0]])  # columns: the shape of each mode


def _resonator(x, fn, zeta, fs):
    """Filter ``x`` through a single second-order digital resonator."""
    wn = 2 * np.pi * fn
    r = np.exp(-zeta * wn / fs)
    wd = wn * np.sqrt(1 - zeta**2) / fs
    a1, a2 = 2 * r * np.cos(wd), -(r**2)
    y = np.zeros_like(x)
    for n in range(2, x.size):
        y[n] = a1 * y[n - 1] + a2 * y[n - 2] + x[n]
    return y


def _synth_outputs(n_reps=2):
    """Responses (n_reps, 2, nt); each channel dominated by one mode."""
    rng = np.random.default_rng(0)
    Y = np.empty((n_reps, 2, N_T))
    for rep in range(n_reps):
        x = rng.standard_normal(N_T)
        modes = np.stack([_resonator(x, fn, ZETA, FS) for fn in F_TRUE])
        Y[rep] = SHAPES @ modes
    return Y


def _rel_err(fitted, measured, band):
    """Largest deviation over ``band``, relative to the data there."""
    fitted, measured = np.asarray(fitted)[..., band], np.asarray(measured)[..., band]
    return np.abs(fitted - measured).max() / np.abs(measured).max()


@pytest.fixture(scope="module")
def oma_model():
    return OmaModel(_synth_outputs(), FS, fft_args=FFT_ARGS)


# =============================================================================
# construction
# =============================================================================


def test_oma_model_builds(oma_model):
    m = oma_model
    assert m.S_yy.shape[:2] == (2, 2)
    assert m.S_yy.shape[-1] == m.freqs.shape[0]
    assert not hasattr(m, "get_frf")  # input estimators live on EmaModel only


def test_non_3d_signals_raise():
    Y = _synth_outputs(n_reps=1)  # (1, 2, nt)
    with pytest.raises(ValueError, match="Y must be 3D"):
        OmaModel(Y[0], FS, fft_args=FFT_ARGS)
    with pytest.raises(ValueError, match="X must be 3D"):
        EmaModel(Y[0], Y, FS, exc_type="continuous", fft_args=FFT_ARGS)


def test_oma_matches_ema_output_spectra():
    """The same Y must yield the same S_yy whichever class computes it."""
    Y = _synth_outputs(n_reps=1)
    X = np.random.default_rng(1).standard_normal((1, 1, N_T))
    ema = EmaModel(X, Y, FS, exc_type="continuous", fft_args=FFT_ARGS)
    oma = OmaModel(Y, FS, fft_args=FFT_ARGS)
    assert np.allclose(oma.S_yy, ema.S_yy)
    freqs, H = ema.get_frf()  # the EMA path works alongside
    assert H.shape == (2, 1, freqs.shape[0])


def test_vmap_batches_frf_over_stacked_tests():
    """The documented batch pattern: jax.vmap over the constructor + get_frf."""
    import jax

    Y = _synth_outputs(n_reps=1)
    X = np.random.default_rng(2).standard_normal((1, 1, N_T))
    Xb = np.stack([X, 1.1 * X])  # (2, n_reps, P, nt)
    Yb = np.stack([Y, 1.1 * Y])

    freqs, H = jax.vmap(
        lambda X, Y: EmaModel(X, Y, FS, "continuous", FFT_ARGS).get_frf()
    )(Xb, Yb)
    assert H.shape == (2, 2, 1, freqs.shape[-1])  # (batch, Q, P, nf)
    # scaling both signals equally leaves the FRF unchanged
    assert np.allclose(H[0], H[1], rtol=1e-10)


# =============================================================================
# output-only estimators
# =============================================================================


def test_oma_psd_is_outputs_only(oma_model):
    freqs, s_yy = oma_model.get_psd()
    assert s_yy.shape == (2, freqs.shape[0])


def test_output_psd_peaks_at_resonances(oma_model):
    freqs, s_yy = oma_model.get_psd()
    freqs = np.asarray(freqs)
    for ch, fn in enumerate(F_TRUE):
        peak = freqs[np.argmax(np.asarray(s_yy[ch]))]
        assert abs(peak - fn) < 3.0, (ch, peak)


def test_transmissibility_shapes(oma_model):
    freqs, Tr = oma_model.get_transmissibility()
    assert Tr.shape == (2, 2, freqs.shape[0])


def test_get_output_spectra_default_is_full_syy(oma_model):
    freqs, S = oma_model.get_output_spectra()
    assert np.allclose(S, oma_model.S_yy)
    assert freqs.shape[0] == S.shape[-1]


def test_get_output_spectra_ref_subset(oma_model):
    _, S = oma_model.get_output_spectra(ref_dofs=[1])
    assert S.shape == (2, 1, oma_model.freqs.shape[0])
    assert np.allclose(S[:, 0], oma_model.S_yy[:, 1])


# =============================================================================
# OMA bridge: output spectra -> pLSCF
# =============================================================================


def test_output_spectra_feed_plscf(oma_model):
    """End-to-end OMA: get_output_spectra -> pLSCF recovers the resonances.

    A full (periodogram) spectrum holds each mode twice (pole + mirror), so
    the model order is doubled relative to an FRF fit of the same modes.
    """
    freqs, S = oma_model.get_output_spectra()
    est = pLSCF(freqs, fs=FS, ordmax=16, spectrum="sd_per")
    poles = est.fit(S)

    # the lowest order at which both modes are stable; a fixed order near the top of
    # the chart loses the 90 Hz pole on some platforms
    Lambd, _, _, _ = poles.mpe(F_TRUE, rtol=0.05, on_missing="raise")
    Fn = lambd_to_fn(Lambd)
    assert np.allclose(Fn, F_TRUE, rtol=2e-2), Fn


# =============================================================================
# correlogram estimator
# =============================================================================


@pytest.fixture(scope="module")
def cor_model():
    return OmaModel(_synth_outputs(), FS, estimator="cor", fft_args={"n_lags": 512})


def test_correlogram_model_shapes_and_ref_subset(cor_model):
    freqs, S = cor_model.get_output_spectra()
    assert freqs.shape[0] == 513  # nfft defaults to 2 * n_lags, one-sided
    assert S.shape == (2, 2, 513)
    assert np.iscomplexobj(S)

    _, S0 = cor_model.get_output_spectra(ref_dofs=[0])
    assert S0.shape == (2, 1, 513)
    assert np.allclose(S0[:, 0], S[:, 0])


def test_correlogram_validates_its_arguments():
    Y = _synth_outputs(n_reps=1)
    with pytest.raises(ValueError, match="n_lags"):
        OmaModel(Y, FS, estimator="cor")
    with pytest.raises(ValueError, match="estimator"):
        OmaModel(Y, FS, estimator="blackman")


def test_correlogram_psd_peaks_at_resonances(cor_model):
    """get_psd reconstructs the full spectrum (S+ + S+^H) for the diagonals."""
    freqs, s_yy = cor_model.get_psd()
    freqs = np.asarray(freqs)
    for ch, fn in enumerate(F_TRUE):
        peak = freqs[np.argmax(np.asarray(s_yy[ch]))]
        assert abs(peak - fn) < 3.0, (ch, peak)


def test_correlogram_spectra_feed_plscf_sd_cor(cor_model):
    """End-to-end OMA on half spectra: half the model order of the full
    spectrum, and the exponential-window correction gives damping back."""
    freqs, S = cor_model.get_output_spectra(ref_dofs=[0])
    est = pLSCF(freqs, fs=FS, ordmax=8, spectrum="sd_cor", alpha=cor_model.window_rate)
    poles = est.fit(S)

    Lambd, _, _, _ = poles.mpe(F_TRUE, order_in=5, rtol=0.05, on_missing="raise")
    Fn, Zeta = lambd_to_fn(Lambd), lambd_to_zeta(Lambd)
    assert np.allclose(Fn, F_TRUE, rtol=2e-2), Fn
    assert np.allclose(Zeta, ZETA, atol=0.01), Zeta


# =============================================================================
# OMA bridge, second step: output spectra -> LSFD
# =============================================================================


def test_lsfd_on_half_spectra_recovers_the_mode_shapes(cor_model):
    """The mixing that built the data must come back out of the modal constants.

    The half spectra still carry the exponential window, so ``LSFD`` needs the
    same window rate as ``pLSCF``: it puts that damping back into the poles
    before fitting, and the fit is far worse without it.
    """
    freqs, S = cor_model.get_output_spectra(ref_dofs=[0])
    est = pLSCF(freqs, fs=FS, ordmax=8, spectrum="sd_cor", alpha=cor_model.window_rate)
    lambd, _, _, _ = est.fit(S).mpe(F_TRUE, order_in=5, rtol=0.05)

    band = (freqs > 10.0) & (freqs < 200.0)
    lsfd = LSFD(
        freqs, fs=FS, spectrum="sd_cor", alpha=cor_model.window_rate, band=(10, 200)
    )
    model = lsfd.fit(S, lambd)

    macs = np.diag(np.asarray(mac(model.mode_shapes(), SHAPES)))
    assert np.all(macs > 0.99), macs
    err = _rel_err(lsfd.synthesize(model), S, band)
    assert err < 0.05, err

    # the same fit with the window damping left out of the poles
    shift = window_shift("sd_cor", cor_model.window_rate)
    unwindowed = lsfd.fit(S, np.asarray(lambd) + shift)
    assert _rel_err(lsfd.synthesize(unwindowed), S, band) > 10 * err


def test_lsfd_on_full_spectra_fits_the_mirrored_modes(oma_model):
    """A full spectrum holds each mode twice, so LSFD fits a mirrored pair too.

    The mirrored half transposes the constants, which the operational reference
    factors cannot describe: it keeps its own constant per reference even when
    the modes themselves are tied to the factors.
    """
    freqs, S = oma_model.get_output_spectra()
    est = pLSCF(freqs, fs=FS, ordmax=16, spectrum="sd_per")
    Lambd, _, Lr, _ = est.fit(S).mpe(F_TRUE, rtol=0.05)  # lowest order with both modes stable

    lsfd = LSFD(freqs, fs=FS, spectrum="sd_per", band=(10, 200))
    model = lsfd.fit(S, Lambd, part_factors=Lr)

    assert model.A_mirror is not None
    macs = np.diag(np.asarray(mac(model.mode_shapes(), SHAPES)))
    assert np.all(macs > 0.99), macs
    band = (freqs > 10.0) & (freqs < 200.0)
    assert _rel_err(lsfd.synthesize(model), S, band) < 0.15
