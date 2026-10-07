"""
Round-trip verification of the pLSCF math in ``modekit.plscf``.

Calls ``fit`` / ``get_poles`` / ``mpe`` (and the lower-level ``rmfd_to_ss`` /
``ss_to_modal_params``) directly on an analytic receptance FRF built from a
known mass-normalised modal model (M = I), and checks the identified natural
frequencies, damping ratios and mode shapes match the ground truth.

Deliberately bypasses ``criteria.py`` and ``algorithms.py`` (not yet wired
up to the current ``plscf`` API) — everything here is exercised through
``plscf`` alone.
"""

import numpy as np
import pytest

from modekit import plscf

# --- Ground-truth modal model ---
F_TRUE = np.array([50.0, 150.0, 300.0])  # natural frequencies [Hz]
ZETA_TRUE = np.array([0.010, 0.015, 0.020])  # damping ratios
W_TRUE = 2 * np.pi * F_TRUE

FS = 2000.0  # sampling frequency [Hz]; full-band FRF spans 0..FS/2 = 0..1000 Hz
DT = 1.0 / FS
N_OUT = 3  # Q, output/response locations
N_IN = 2  # P, inputs/references
ORDMAX = 20
ORDER = 18  # fixed, well-resolved order for deterministic checks
COL = ORDER - 1  # column j of the pole arrays holds model order j+1


def _mode_shapes(seed=0):
    """Random orthonormal (mass-normalised, M=I) mode-shape matrix, (Q, n_modes)."""
    rng = np.random.default_rng(seed)
    V, _ = np.linalg.qr(rng.standard_normal((N_OUT, N_OUT)))
    return V  # columns are the mode shapes


def _synth_frf(V, f_max=FS / 2, nf=2048):
    """Analytic receptance FRF for the modal model, shape (Q, P, N)."""
    freq = np.linspace(0.0, f_max, nf)
    w = 2 * np.pi * freq
    # H[o, i, f] = sum_r V[o,r] V[i,r] / (w_r^2 - w^2 + 2j zeta_r w_r w)
    denom = (
        W_TRUE[:, None] ** 2
        - w[None, :] ** 2
        + 2j * ZETA_TRUE[:, None] * W_TRUE[:, None] * w[None, :]
    )  # (n_modes, N)
    H = np.einsum("or,ir,rf->oif", V, V[:N_IN], 1.0 / denom)  # (Q, P, N)
    return freq, H


def _mac(a, b) -> float:
    """Scale/phase-invariant MAC between two complex mode-shape vectors."""
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    num = np.abs(np.vdot(a, b)) ** 2
    den = np.vdot(a, a).real * np.vdot(b, b).real
    return float((num / den).real)


def _fit_and_get_poles(freq, H, ordmax=ORDMAX):
    Ad, Bn = plscf.fit(H, freq, DT, ordmax)
    Phis, Lambds, Lrs = plscf.get_poles(Ad, Bn, DT, spectrum="frf_shaker")
    return Ad, Bn, Lambds, Phis, Lrs


def _run(freq=None, H=None):
    V = _mode_shapes()
    if freq is None or H is None:
        freq, H = _synth_frf(V)
    _, _, Lambds, Phis, Lrs = _fit_and_get_poles(freq, H)
    Lambd, Phi, _, order_out = plscf.mpe(
        F_TRUE, Lambds, Phis, Lrs, order_in=COL, rtol=0.1
    )
    return V, plscf.lambd_to_fn(Lambd), plscf.lambd_to_zeta(Lambd), Phi, order_out


# =============================================================================
# get_poles: pole-array shapes and dtypes
# =============================================================================


def test_pole_array_shapes():
    V = _mode_shapes()
    freq, H = _synth_frf(V)
    _, _, Lambds, Phis, Lrs = _fit_and_get_poles(freq, H)
    Fns, Zetas = plscf.lambd_to_fn(Lambds), plscf.lambd_to_zeta(Lambds)

    assert Fns.shape == Zetas.shape == Lambds.shape
    assert Fns.shape[1] == ORDMAX  # one column per fitted order
    assert Phis.shape[:2] == Fns.shape
    assert Phis.shape[2] == N_OUT
    assert Lrs.shape[:2] == Fns.shape
    assert Lrs.shape[2] == N_IN  # one reference factor per input
    assert np.iscomplexobj(Phis)
    assert np.iscomplexobj(Lrs)
    assert np.iscomplexobj(Lambds)


# =============================================================================
# fit + get_poles + mpe: pole recovery against ground truth
# =============================================================================


def test_frequencies_recovered():
    _, Fn, _, _, _ = _run()
    assert np.allclose(Fn, F_TRUE, rtol=2e-3), Fn


def test_damping_recovered():
    _, _, Zeta, _, _ = _run()
    assert np.allclose(Zeta, ZETA_TRUE, atol=2e-3), Zeta


def test_mode_shapes_recovered():
    V, _, _, Phi, _ = _run()  # Phi: (Q, n_modes), aligned with F_TRUE order
    macs = np.array([_mac(Phi[:, k], V[:, k]) for k in range(len(F_TRUE))])
    assert np.all(macs > 0.99), macs


def test_bandlimited_frequencies_recovered():
    """fit/get_poles/mpe should work when the FRF doesn't span the full Nyquist band."""
    V = _mode_shapes()
    freq_full, H_full = _synth_frf(V)
    mask = freq_full <= 400.0
    freq_sub, H_sub = freq_full[mask], H_full[:, :, mask]

    _, _, Lambds, Phis, Lrs = _fit_and_get_poles(freq_sub, H_sub)
    Lambd, Phi, _, _ = plscf.mpe(F_TRUE, Lambds, Phis, Lrs, order_in=COL, rtol=0.1)
    Fn, Zeta = plscf.lambd_to_fn(Lambd), plscf.lambd_to_zeta(Lambd)

    assert np.allclose(Fn, F_TRUE, rtol=2e-3), Fn
    assert np.allclose(Zeta, ZETA_TRUE, atol=2e-3), Zeta
    macs = np.array([_mac(Phi[:, k], V[:, k]) for k in range(len(F_TRUE))])
    assert np.all(macs > 0.99), macs


# =============================================================================
# rmfd_to_ss / ss_to_modal_params: direct low-level shape check
# =============================================================================


def test_rmfd_to_ss_and_ss_to_modal_params_shapes():
    V = _mode_shapes()
    freq, H = _synth_frf(V)
    order = 5
    Ad, Bn = plscf.fit(H, freq, DT, order)

    A, C = plscf.rmfd_to_ss(Ad[order], Bn[order])
    n_poles = (order + 1) * N_IN
    assert A.shape == (n_poles, n_poles)
    assert C.shape == (N_OUT, n_poles)

    phi, lambd, lr = plscf.ss_to_modal_params(A, C, DT, N_IN, spectrum="frf_shaker")
    assert lambd.shape == (n_poles,)
    assert phi.shape == (n_poles, N_OUT)
    assert lr.shape == (n_poles, N_IN)
    assert plscf.lambd_to_fn(lambd).shape == (n_poles,)
    assert plscf.lambd_to_zeta(lambd).shape == (n_poles,)


def test_rmfd_response_reproduces_the_fitted_frf():
    """B(z) A(z)^-1 of a well-resolved order must synthesise the data it fitted."""
    V = _mode_shapes()
    freq, H = _synth_frf(V)
    Ad, Bn = plscf.fit(H, freq, DT, ORDER)
    H_fit = np.asarray(plscf.rmfd_response(Ad[ORDER], Bn[ORDER], freq, DT))
    assert H_fit.shape == H.shape
    rel = np.linalg.norm(H_fit - H) / np.linalg.norm(H)
    assert rel < 1e-3, rel


def test_fit_zero_weights_drop_lines_like_slicing():
    """Zero-weighting the lines outside a band equals fitting the band's slice."""
    V = _mode_shapes()
    freq, H = _synth_frf(V)
    band = (freq >= 30.0) & (freq <= 80.0)
    order = 4
    Ad_w, Bn_w = plscf.fit(H, freq, DT, order, weight=band.astype(float))
    Ad_s, Bn_s = plscf.fit(H[:, :, band], freq[band], DT, order)
    # the coefficients are not unique (surplus poles cancel against zeros), the
    # curve they describe over the band is
    for r in (2, order):
        H_w = np.asarray(plscf.rmfd_response(Ad_w[r], Bn_w[r], freq[band], DT))
        H_s = np.asarray(plscf.rmfd_response(Ad_s[r], Bn_s[r], freq[band], DT))
        rel = np.linalg.norm(H_w - H_s) / np.linalg.norm(H_s)
        assert rel < 1e-4, (r, rel)  # surplus poles leave the solve ill-conditioned
    rel = np.linalg.norm(H_s - H[:, :, band]) / np.linalg.norm(H[:, :, band])
    assert rel < 1e-3, rel  # and at full order it is the data


# =============================================================================
# mpe: error handling and selection modes
# =============================================================================


def test_mpe_raises_when_no_admissible_pole():
    V = _mode_shapes()
    freq, H = _synth_frf(V)
    _, _, Lambds, Phis, Lrs = _fit_and_get_poles(freq, H)

    # a frequency far outside the modal band has no matching pole
    with pytest.raises(ValueError):
        plscf.mpe(np.array([9999.0]), Lambds, Phis, Lrs, order_in=COL, rtol=1e-3)


def test_mpe_on_missing_nan_returns_nan_without_raising():
    V = _mode_shapes()
    freq, H = _synth_frf(V)
    _, _, Lambds, Phis, Lrs = _fit_and_get_poles(freq, H)

    f_ref = np.array([F_TRUE[0], 9999.0])
    with pytest.warns(RuntimeWarning):
        Lambd, Phi, Lr, order_out = plscf.mpe(
            f_ref, Lambds, Phis, Lrs, order_in=COL, rtol=1e-3, on_missing="nan"
        )

    assert np.isclose(plscf.lambd_to_fn(Lambd)[0], F_TRUE[0], rtol=2e-3)
    assert order_out[0] >= 0
    assert np.isnan(Lambd[1])
    assert np.all(np.isnan(Phi[:, 1]))
    assert np.all(np.isnan(Lr[:, 1]))
    assert order_out[1] == -1


def test_mpe_find_min_resolves_lowest_order():
    V = _mode_shapes()
    freq, H = _synth_frf(V)
    _, _, Lambds, Phis, Lrs = _fit_and_get_poles(freq, H)

    # Exercise 'find_min' in isolation: mark every pole "stable" so the only
    # real constraint is each reference frequency resolving to a distinct pole.
    stab_label = np.ones(Lambds.shape, dtype=int)
    Lambd, _, _, order_out = plscf.mpe(
        F_TRUE,
        Lambds,
        Phis,
        Lrs,
        order_in="find_min",
        stab_label=stab_label,
        rtol=0.1,
    )
    Fn = plscf.lambd_to_fn(Lambd)

    assert np.allclose(Fn, F_TRUE, rtol=5e-2), Fn
    assert np.unique(order_out).size == 1
    assert (order_out >= 0).all()


# =============================================================================
# complex_to_real_mode: real-mode extraction (Ahmadian, Gladwell & Ismail)
# =============================================================================


def test_complex_to_real_mode_real_input_is_returned():
    """A real (proportionally damped) mode comes back real, up to sign/scale."""
    rng = np.random.default_rng(1)
    phi = rng.standard_normal(6)
    out = np.asarray(plscf.complex_to_real_mode(phi + 0j))
    assert np.isrealobj(out)
    assert np.abs(out.imag).max() == 0.0
    assert _mac(out, phi) > 1 - 1e-9


def test_complex_to_real_mode_global_phase_invariant():
    """phi * e^{i theta} is still real once rotated back: MAC with the real mode ~ 1."""
    rng = np.random.default_rng(2)
    phi = rng.standard_normal(6)
    out = np.asarray(plscf.complex_to_real_mode(phi * np.exp(1j * 0.7)))
    assert _mac(out, phi) > 1 - 1e-9


def test_complex_to_real_mode_scale_equivariance():
    """Scaling the complex input scales the real mode identically (keeps residue scale)."""
    rng = np.random.default_rng(3)
    phi = rng.standard_normal(6) + 0.15j * rng.standard_normal(6)
    out = np.asarray(plscf.complex_to_real_mode(phi))
    out_scaled = np.asarray(plscf.complex_to_real_mode(3.7 * phi))
    assert np.allclose(out_scaled, 3.7 * out, atol=1e-6)


def test_complex_to_real_mode_matches_optimal_rotation():
    """Result equals the paper's Real(phi * e^{i theta*}) maximising the real-part norm."""
    rng = np.random.default_rng(4)
    phi = rng.standard_normal(6) + 0.3j * rng.standard_normal(6)
    thetas = np.linspace(0.0, np.pi, 200001)
    norms = np.linalg.norm(np.real(phi[:, None] * np.exp(1j * thetas)), axis=0)
    ref = np.real(phi * np.exp(1j * thetas[np.argmax(norms)]))
    if ref[np.argmax(np.abs(ref))] < 0:  # match the sign convention
        ref = -ref
    out = np.asarray(plscf.complex_to_real_mode(phi))
    assert np.allclose(out, ref, atol=1e-4), np.abs(out - ref).max()


def test_complex_to_real_mode_matrix_and_nan_passthrough():
    """(Q, M) modal matrix: all-NaN columns pass through, others stay finite and real."""
    rng = np.random.default_rng(5)
    Phi = rng.standard_normal((6, 3)) + 0.1j * rng.standard_normal((6, 3))
    Phi[:, 1] = np.nan
    R = np.asarray(plscf.complex_to_real_mode(Phi))
    assert R.shape == (6, 3)
    assert np.isrealobj(R)
    assert np.all(np.isnan(R[:, 1]))
    assert np.all(np.isfinite(R[:, 0])) and np.all(np.isfinite(R[:, 2]))


def test_complex_to_real_mode_sign_convention():
    """The largest-magnitude entry is made positive for a deterministic sign."""
    rng = np.random.default_rng(6)
    phi = rng.standard_normal(6) + 0.2j * rng.standard_normal(6)
    out = np.asarray(plscf.complex_to_real_mode(phi))
    assert out[np.argmax(np.abs(out))] > 0


if __name__ == "__main__":
    V, Fn, Zeta, Phi, order_out = _run()
    macs = np.array([_mac(Phi[:, k], V[:, k]) for k in range(len(F_TRUE))])

    print("true freq [Hz]:", F_TRUE)
    print("ided freq [Hz]:", np.round(Fn, 4))
    print("true zeta     :", ZETA_TRUE)
    print("ided zeta     :", np.round(Zeta, 5))
    print("mode-shape MAC:", np.round(macs, 5))
    print("order_out     :", order_out)
