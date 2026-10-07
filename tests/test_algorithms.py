"""
End-to-end tests for the ``pLSCF`` driver in ``modekit.algorithms``.

Exercises the full chain — ``fit`` → ``get_poles`` → hard criteria → soft
criteria → ``mpe`` — against the same synthetic ground-truth model used by
``test_plscf``, so a break anywhere in the wiring between ``plscf`` and
``criteria`` surfaces here.
"""

import jax
import numpy as np
import pytest

from test_plscf import (
    F_TRUE,
    FS,
    N_IN,
    ORDER,
    W_TRUE,
    ZETA_TRUE,
    _mac,
    _mode_shapes,
    _synth_frf,
)
from modekit.plscf import lambd_to_fn, lambd_to_zeta
from modekit.algorithms import (
    LSFD,
    PoleArrays,
    _residual_powers,
    pLSCF,
)

ORDMAX = 20
COL = ORDER - 1  # column j of the pole arrays holds model order j + 1


@pytest.fixture(scope="module")
def fitted():
    """A pLSCF estimator and its ``fit(frf)`` poles on the synthetic FRF.

    ``pLSCF`` is an immutable ``eqx.Module`` estimator: the FRF is passed to
    ``fit`` (not the constructor), ``fit`` returns a ``PoleArrays`` whose
    ``mpe``/``select`` pick modes; ``plot_stab`` takes the poles back.
    """
    V = _mode_shapes()
    freq, H = _synth_frf(V)
    algo = pLSCF(freq, fs=FS, ordmax=ORDMAX, spectrum="frf_shaker")
    poles = algo.fit(H)
    return V, algo, poles


# =============================================================================
# pLSCF is an immutable eqx.Module
# =============================================================================


def test_plscf_is_a_frozen_module():
    import equinox as eqx

    V = _mode_shapes()
    freq, _ = _synth_frf(V)
    algo = pLSCF(freq, fs=FS, ordmax=6, spectrum="frf_shaker")
    assert isinstance(algo, eqx.Module)
    assert algo.dt == 1.0 / FS  # derived property, nothing stale stored
    with pytest.raises(Exception):  # noqa: B017 — frozen: any mutation is rejected
        algo.ordmax = 7


# =============================================================================
# fit()
# =============================================================================


def test_fit_returns_populated_poles(fitted):
    _, _, r = fitted
    assert isinstance(r, PoleArrays)
    assert r.Fn.shape == r.Zeta.shape == r.stab_label.shape
    assert r.Phi.shape[:2] == r.Fn.shape
    assert r.Phi.shape[2] == 3  # Q outputs
    assert r.Lr.shape[:2] == r.Fn.shape
    assert r.Lr.shape[2] == N_IN  # P references
    assert r.Fn.shape[1] == ORDMAX  # one column per fitted order


def test_fit_labels_some_poles_stable(fitted):
    _, _, r = fitted
    assert r.stab_label.max() == 1


def test_hard_criteria_leave_a_common_nan_pattern(fitted):
    """Every screen is applied to all the pole arrays, so NaNs must align."""
    _, _, r = fitted
    dead = np.isnan(r.Fn)
    assert np.array_equal(np.isnan(r.Zeta), dead)
    assert np.array_equal(np.isnan(r.Lambd), dead)
    assert np.array_equal(np.isnan(r.Phi).all(axis=-1), dead)
    assert np.array_equal(np.isnan(r.Lr).all(axis=-1), dead)


# =============================================================================
# mpe()
# =============================================================================


def test_mpe_recovers_ground_truth(fitted):
    V, _, poles = fitted
    Lambd, Phi, _, order_out = poles.mpe(F_TRUE, order_in=COL, rtol=0.1)
    Fn, Zeta = lambd_to_fn(Lambd), lambd_to_zeta(Lambd)

    assert np.allclose(Fn, F_TRUE, rtol=2e-3), Fn
    assert np.allclose(Zeta, ZETA_TRUE, atol=2e-3), Zeta
    macs = np.array([_mac(Phi[:, k], V[:, k]) for k in range(len(F_TRUE))])
    assert np.all(macs > 0.99), macs
    assert (order_out == COL).all()


def test_mpe_find_min_recovers_ground_truth(fitted):
    """'find_min' must reach the true modes using this fit's stability labels."""
    V, _, poles = fitted
    Lambd, Phi, _, _ = poles.mpe(F_TRUE, order_in="find_min", rtol=0.1)
    Fn = lambd_to_fn(Lambd)

    assert np.allclose(Fn, F_TRUE, rtol=5e-2), Fn
    macs = np.array([_mac(Phi[:, k], V[:, k]) for k in range(len(F_TRUE))])
    assert np.all(macs > 0.99), macs


def test_mpe_on_missing_nan_keeps_alignment(fitted):
    _, _, poles = fitted
    f_ref = np.array([F_TRUE[0], 9999.0])
    with pytest.warns(RuntimeWarning):
        Lambd, _, _, order_out = poles.mpe(
            f_ref, order_in=COL, rtol=1e-3, on_missing="nan"
        )
    Fn = np.asarray(lambd_to_fn(Lambd))

    assert np.isclose(Fn[0], F_TRUE[0], rtol=2e-3)
    assert np.isnan(Fn[1])
    assert order_out.tolist() == [COL, -1]


def test_mpe_phi_ref_gate_accepts_correct_shapes(fitted):
    """Passing the true shapes as the MAC reference must not lose any mode."""
    V, _, poles = fitted
    Lambd, _, _, _ = poles.mpe(F_TRUE, order_in=COL, rtol=0.1, phi_ref=V, mac_lim=0.9)
    Fn = lambd_to_fn(Lambd)
    assert np.allclose(Fn, F_TRUE, rtol=2e-3), Fn


# =============================================================================
# Batched pipeline — jax.vmap over the estimator's methods (no batch functions)
# =============================================================================

BATCH_ORDMAX = 8  # keep the batched fixtures cheap


def _frf_stack(B=3, scales=None):
    """B segments of the same model, scaled so they are not bitwise identical."""
    V = _mode_shapes()
    freq, H = _synth_frf(V)
    scales = np.linspace(1.0, 1.2, B) if scales is None else scales
    return V, freq, np.stack([s * H for s in scales])


def _estimator(freq):
    return pLSCF(freq, fs=FS, ordmax=BATCH_ORDMAX, spectrum="frf_shaker")


def test_vmap_fit_matches_a_python_loop():
    """The whole point: jax.vmap(est.fit) must agree with per-segment est.fit."""
    _, freq, stack = _frf_stack()
    est = _estimator(freq)
    batched = jax.vmap(est.fit)(stack)  # only frf maps; freqs shared via `est`

    assert isinstance(batched, PoleArrays)
    assert batched.Fn.shape[0] == stack.shape[0]

    for b in range(stack.shape[0]):
        one = est.fit(stack[b])
        for field in ("Fn", "Zeta", "Lambd", "stab_label"):
            got, want = getattr(batched, field)[b], getattr(one, field)
            assert np.array_equal(np.isnan(got), np.isnan(want)), field
            assert np.allclose(got, want, equal_nan=True, rtol=1e-5), field


def test_vmap_fit_shapes_carry_the_batch_axis():
    _, freq, stack = _frf_stack(B=4)
    poles = jax.vmap(_estimator(freq).fit)(stack)
    B, Q = stack.shape[0], stack.shape[1]
    assert poles.Fn.shape == (B, poles.Fn.shape[1], BATCH_ORDMAX)
    assert poles.Phi.shape == (B, poles.Fn.shape[1], BATCH_ORDMAX, Q)
    assert poles.stab_label.shape == poles.Fn.shape


def test_vmap_select_recovers_ground_truth_for_every_segment():
    V, freq, stack = _frf_stack(B=3)
    poles = jax.vmap(_estimator(freq).fit)(stack)
    # batch mode selection is jax.vmap over the pure PoleArrays.select
    Lambd, Phi, Lr, cols, ok = jax.vmap(lambda p: p.select(F_TRUE, rtol=0.1))(poles)
    Fn, Zeta = lambd_to_fn(Lambd), lambd_to_zeta(Lambd)

    B = stack.shape[0]
    assert Lambd.shape == Fn.shape == (B, len(F_TRUE))
    assert Phi.shape == (B, stack.shape[1], len(F_TRUE))
    assert Lr.shape == (B, stack.shape[2], len(F_TRUE))
    assert np.asarray(ok).all(), ok
    # scaling the FRF cannot move the poles, so every segment sees the same modes
    assert np.allclose(Fn, F_TRUE[None, :], rtol=5e-3), Fn
    assert np.allclose(Zeta, ZETA_TRUE[None, :], atol=3e-3), Zeta
    for b in range(B):
        macs = [_mac(np.asarray(Phi)[b, :, k], V[:, k]) for k in range(len(F_TRUE))]
        assert np.all(np.array(macs) > 0.99), (b, macs)


def test_fit_is_jittable():
    """jit needs the same purity vmap does, so it is a cheap extra guard.

    Use ``eqx.filter_jit`` (as ``freq_models`` does): plain ``jax.jit`` of a bound
    eqx.Module method tries to hash ``self``, whose ``freqs`` array is unhashable.
    """
    import equinox as eqx

    _, freq, stack = _frf_stack(B=1)
    est = _estimator(freq)
    poles = eqx.filter_jit(est.fit)(stack[0])
    eager = est.fit(stack[0])
    assert np.allclose(poles.Fn, eager.Fn, equal_nan=True, rtol=1e-5)


# =============================================================================
# LSFD — modal constants once the poles are fixed
# =============================================================================

LAM_TRUE = -ZETA_TRUE * W_TRUE + 1j * W_TRUE * np.sqrt(1 - ZETA_TRUE**2)


def _true_residues(V):
    """Modal constants of the synthetic receptance FRF, (Q, P, M).

    ``1 / ((jw - lam)(jw - conj(lam)))`` splits into ``A / (jw - lam)`` plus its
    conjugate term, with ``A = 1 / (lam - conj(lam))``.
    """
    return np.einsum("or,ir->oir", V, V[:N_IN]) / (2j * LAM_TRUE.imag)


def _receptance_lsfd(freq, **kwargs):
    return LSFD(freq, fs=FS, spectrum="frf_shaker", quantity="displacement", **kwargs)


@pytest.fixture(scope="module")
def lsfd_fitted(fitted):
    """LSFD on the synthetic FRF, reading the poles pLSCF identified."""
    V, _, poles = fitted
    freq, H = _synth_frf(V)
    Lambd, _, Lr, _ = poles.mpe(F_TRUE, order_in=COL, rtol=0.1)
    lsfd = _receptance_lsfd(freq)
    return V, freq, H, lsfd, lsfd.fit(H, Lambd), Lr


def test_residual_powers_follow_the_table():
    """Peeters & Van der Auweraer (IOMAC 2005), table 1."""
    quantities = ["displacement", "velocity", "acceleration"]
    assert [_residual_powers("frf", q) for q in quantities] == [
        (-2, 0),
        (-1, 1),
        (0, 2),
    ]
    assert [_residual_powers("full", q) for q in quantities] == [
        (-4, 0),
        (-2, 2),
        (0, 4),
    ]
    assert [_residual_powers("half", q) for q in quantities] == [(-1, 1)] * 3


@pytest.mark.parametrize("constrained", [False, True])
def test_lsfd_recovers_the_analytic_residues(constrained):
    """With the exact poles the fit is a plain projection, so it must be exact.

    The residues are rank one, ``A_i = v_i g_i^T`` with the participation
    factors ``g_i = V[:P, i]``, so constraining the fit with them cannot change
    the answer — only the number of unknowns it takes to get there.
    """
    V = _mode_shapes()
    freq, H = _synth_frf(V)
    part_factors = V[:N_IN] if constrained else None
    model = _receptance_lsfd(freq).fit(H, LAM_TRUE, part_factors=part_factors)

    A_true = _true_residues(V)
    tiny = 1e-9 * np.abs(A_true).max()
    assert np.allclose(model.A, A_true, rtol=1e-6, atol=tiny)
    # every mode of this FRF is inside the band, so both residuals vanish
    assert np.abs(np.asarray(model.LR)).max() < tiny
    assert np.abs(np.asarray(model.UR)).max() < tiny


def test_lsfd_participation_factors_make_the_constants_rank_one(lsfd_fitted):
    """The pLSCF reference factors buy the rank-one structure without an SVD."""
    _, _, H, lsfd, free, Lr = lsfd_fitted
    tied = lsfd.fit(H, free.Lambd, part_factors=Lr)

    s = np.linalg.svd(np.asarray(tied.A).transpose(2, 0, 1), compute_uv=False)
    assert np.all(s[:, 1] < 1e-12 * s[:, 0]), s  # exactly rank one, by construction
    # fitting each constant on its own lands in the same place
    assert np.allclose(tied.A, free.A, rtol=1e-3, atol=1e-9 * np.abs(free.A).max())


def test_lsfd_synthesis_matches_the_frf(lsfd_fitted):
    _, _, H, lsfd, model, _ = lsfd_fitted
    H_hat = np.asarray(lsfd.synthesize(model))

    line = np.isfinite(H_hat).all(axis=(0, 1))
    assert line.sum() == H.shape[-1] - 1  # only w = 0 is out of the model's reach
    assert np.abs(H_hat[:, :, line] - H[:, :, line]).max() < 1e-6 * np.abs(H).max()


def test_lsfd_mode_shapes_and_driving_point_scaling(lsfd_fitted):
    V, _, _, _, model, _ = lsfd_fitted
    phi = np.asarray(model.mode_shapes())
    macs = np.array([_mac(phi[:, k], V[:, k]) for k in range(len(F_TRUE))])
    assert np.all(macs > 0.99), macs

    # output 0 is also reference 0, so it is a driving point: A_i[0, 0] = phi_i[0]^2
    phi_dp = np.asarray(model.mode_shapes(driving_point=(0, 0)))
    assert np.allclose(phi_dp[0] ** 2, np.asarray(model.A)[0, 0], rtol=1e-6)


def test_lsfd_band_narrows_the_fit_not_the_synthesis(lsfd_fitted):
    _, freq, H, _, model, _ = lsfd_fitted
    lsfd = _receptance_lsfd(freq, band=(40.0, 350.0))
    H_hat = np.asarray(lsfd.synthesize(lsfd.fit(H, model.Lambd)))

    line = np.isfinite(H_hat).all(axis=(0, 1))
    assert line.sum() == H.shape[-1] - 1  # the whole vector is synthesised
    assert np.abs(H_hat[:, :, line] - H[:, :, line]).max() < 1e-4 * np.abs(H).max()


def test_lsfd_skips_nan_poles(lsfd_fitted):
    """A mode that mpe could not find (on_missing='nan') drops out of the fit."""
    _, _, H, lsfd, _, _ = lsfd_fitted
    lam = np.asarray(LAM_TRUE).copy()
    lam[1] = np.nan
    model = lsfd.fit(H, lam)

    A = np.asarray(model.A)
    assert np.isnan(A[:, :, 1]).all()
    assert np.isfinite(A[:, :, [0, 2]]).all()
    assert np.isfinite(np.asarray(lsfd.synthesize(model))[:, :, 1:]).all()


def test_lsfd_is_jittable_and_vmappable(lsfd_fitted):
    import equinox as eqx

    _, _, H, lsfd, model, _ = lsfd_fitted
    assert np.allclose(eqx.filter_jit(lsfd.fit)(H, model.Lambd).A, model.A)

    batched = jax.vmap(lsfd.fit, in_axes=(0, None))(np.stack([H, 1.1 * H]), model.Lambd)
    assert batched.A.shape == (2, *model.A.shape)
    # the modal constants are linear in the data
    assert np.allclose(batched.A[1], 1.1 * np.asarray(batched.A[0]))


def test_lsfd_rejects_unknown_flags():
    freq = np.linspace(0.0, 100.0, 8)
    with pytest.raises(ValueError, match="spectrum"):
        LSFD(freq, fs=FS, spectrum="psd_cor")
    with pytest.raises(ValueError, match="quantity"):
        LSFD(freq, fs=FS, quantity="strain")
