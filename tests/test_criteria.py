"""
Tests for ``modekit.criteria``.

The vectorised MAC / MPC / MPD / stabilisation implementations are checked
against literal ports of the original loop-based versions (``_ref_*`` below), so
the refactor is verified to be behaviour-preserving rather than merely
self-consistent. Separate tests cover the JAX-input regressions (``get_poles``
returns ``jax.Array``, which the previous code could not consume) and the
per-order scoping of :func:`hc_conj`.
"""

import numpy as np
import pytest

from modekit import criteria

RNG = np.random.default_rng(0)


# =============================================================================
# Reference implementations — the pre-refactor code, kept for comparison
# =============================================================================


def _ref_mac(phi_X, phi_A):
    if phi_X.ndim == 1:
        phi_X = phi_X[:, np.newaxis]
    if phi_A.ndim == 1:
        phi_A = phi_A[:, np.newaxis]
    MAC = np.abs(np.conj(phi_X).T @ phi_A) ** 2
    MAC = MAC.astype(complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(phi_X.shape[1]):
            for j in range(phi_A.shape[1]):
                MAC[i, j] = MAC[i, j] / (
                    np.conj(phi_X[:, i])
                    @ phi_X[:, i]
                    * np.conj(phi_A[:, j])
                    @ phi_A[:, j]
                )
    if MAC.shape == (1, 1):
        MAC = MAC[0, 0]
    return MAC.real


def _ref_mpc(phi):
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            S = np.cov(phi.real, phi.imag)
            lambd = np.linalg.eigvals(S)
            return (lambd[0] - lambd[1]) ** 2 / (lambd[0] + lambd[1]) ** 2
    except Exception:
        return np.nan


def _ref_mpd(phi):
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            _, _, VT = np.linalg.svd(np.c_[phi.real, phi.imag])
            V = VT.T
            w = np.abs(phi)
            num = phi.real * V[1, 1] - phi.imag * V[0, 1]
            den = np.sqrt(V[0, 1] ** 2 + V[1, 1] ** 2) * np.abs(phi)
            return np.sum(w * np.arccos(np.abs(num / den))) / np.sum(w)
    except Exception:
        return np.nan


def _ref_sc_apply(Fn, Zeta, Phi, ordmin, ordmax, step, err_fn, err_zeta, err_phi):
    import math

    Lab = np.zeros(Fn.shape, dtype="int")
    for o in range(math.ceil(ordmin / step), math.floor(ordmax / step) + 1):
        f_n = Fn[:, o].reshape(-1, 1)
        zeta_n = Zeta[:, o].reshape(-1, 1)
        phi_n = Phi[:, o, :]
        f_n1 = Fn[:, o - 1].reshape(-1, 1)
        zeta_n1 = Zeta[:, o - 1].reshape(-1, 1)
        phi_n1 = Phi[:, o - 1, :]
        if o == 0:
            continue
        for i in range(len(f_n)):
            try:
                idx = np.nanargmin(np.abs(f_n1 - f_n[i]))
                with np.errstate(divide="ignore", invalid="ignore"):
                    cond1 = np.abs(f_n[i] - f_n1[idx]) / f_n[i]
                    cond2 = np.abs(zeta_n[i] - zeta_n1[idx]) / zeta_n[i]
                cond3 = 1 - _ref_mac(phi_n[i, :], phi_n1[idx, :])
                if cond1 < err_fn and cond2 < err_zeta and cond3 < err_phi:
                    Lab[i, o] = 1
                else:
                    Lab[i, o] = 0
            except ValueError:
                continue
    return Lab


# =============================================================================
# Fixtures
# =============================================================================


def _random_shapes(K=6, p=5, Q=4, nan_frac=0.2):
    """Pole arrays with a realistic sprinkling of NaN padding."""
    Fn = RNG.uniform(10.0, 500.0, (K, p))
    Zeta = RNG.uniform(0.001, 0.08, (K, p))
    Phi = RNG.standard_normal((K, p, Q)) + 1j * RNG.standard_normal((K, p, Q))
    dead = RNG.random((K, p)) < nan_frac
    Fn[dead] = np.nan
    Zeta[dead] = np.nan
    Phi[dead] = np.nan
    return Fn, Zeta, Phi


# =============================================================================
# mac
# =============================================================================


def test_mac_matches_reference_matrix():
    phi_X = RNG.standard_normal((7, 3)) + 1j * RNG.standard_normal((7, 3))
    phi_A = RNG.standard_normal((7, 4)) + 1j * RNG.standard_normal((7, 4))
    assert np.allclose(criteria.mac(phi_X, phi_A), _ref_mac(phi_X, phi_A))


def test_mac_matches_reference_vectors():
    a = RNG.standard_normal(7) + 1j * RNG.standard_normal(7)
    b = RNG.standard_normal(7) + 1j * RNG.standard_normal(7)
    got = criteria.mac(a, b)
    assert np.ndim(got) == 0
    assert np.isclose(got, _ref_mac(a, b))


def test_mac_self_is_one_and_scale_phase_invariant():
    a = RNG.standard_normal(7) + 1j * RNG.standard_normal(7)
    assert np.isclose(criteria.mac(a, a), 1.0)
    assert np.isclose(criteria.mac(a, 3.7 * np.exp(1.1j) * a), 1.0)


def test_mac_orthogonal_shapes_are_zero():
    V, _ = np.linalg.qr(RNG.standard_normal((5, 5)))
    M = criteria.mac(V, V)
    assert np.allclose(M, np.eye(5), atol=1e-12)


def test_mac_rejects_mismatched_locations():
    with pytest.raises(ValueError):
        criteria.mac(RNG.standard_normal((5, 2)), RNG.standard_normal((6, 2)))


def test_mac_paired_is_the_diagonal_of_mac_per_batch_element():
    n_seg, Q, M = 3, 6, 4
    Phi = RNG.standard_normal((n_seg, Q, M)) + 1j * RNG.standard_normal((n_seg, Q, M))
    phi_ref = RNG.standard_normal((Q, M)) + 1j * RNG.standard_normal((Q, M))
    got = np.asarray(criteria.mac_paired(Phi, phi_ref))
    assert got.shape == (n_seg, M)
    for i in range(n_seg):
        full = np.asarray(criteria.mac(Phi[i], phi_ref))  # (M, M)
        assert np.allclose(got[i], np.diag(full))
    # unbatched (Q, M) against (Q, M) -> (M,), one per column
    assert np.allclose(criteria.mac_paired(phi_ref, phi_ref), np.ones(M))
    # a NaN shape gives a NaN MAC rather than an error
    Phi[1, :, 2] = np.nan
    assert np.isnan(np.asarray(criteria.mac_paired(Phi, phi_ref))[1, 2])


# =============================================================================
# mpc / mpd
# =============================================================================


def test_mpc_matches_reference():
    # (tr^2 - 4 det) cancels heavily for a strongly complex mode, so the JAX
    # float32 result trails the float64 reference by ~1e-5 relative; that is far
    # below any sensible mpc_lim. Enable jax_enable_x64 to tighten it.
    for _ in range(20):
        phi = RNG.standard_normal(6) + 1j * RNG.standard_normal(6)
        assert np.isclose(criteria.mpc(phi), _ref_mpc(phi), rtol=1e-4)


def test_mpd_matches_reference():
    for _ in range(20):
        phi = RNG.standard_normal(6) + 1j * RNG.standard_normal(6)
        assert np.isclose(criteria.mpd(phi), _ref_mpd(phi), rtol=1e-4)


def test_mpc_real_mode_is_one_mpd_is_zero():
    phi = (RNG.standard_normal(8) + 0j) * np.exp(0.7j)  # perfectly collinear
    assert np.isclose(criteria.mpc(phi), 1.0, atol=1e-10)
    # arccos is vertical at 1, so machine-eps error in the ratio surfaces as
    # ~sqrt(2 * eps) ≈ 1e-8 rad here; 1e-10 would be tighter than float64 allows
    assert np.isclose(criteria.mpd(phi), 0.0, atol=1e-6)


def test_mpc_mpd_vectorised_match_per_vector_calls():
    _, _, Phi = _random_shapes(nan_frac=0.0)
    K, p, _ = Phi.shape
    got_c, got_d = criteria.mpc(Phi), criteria.mpd(Phi)
    assert got_c.shape == got_d.shape == (K, p)
    for k in range(K):
        for j in range(p):
            assert np.isclose(got_c[k, j], criteria.mpc(Phi[k, j]))
            assert np.isclose(got_d[k, j], criteria.mpd(Phi[k, j]))


def test_mpd_nan_rows_give_nan_not_linalg_error():
    """NaN padding used to reach LAPACK; it must be masked out instead."""
    _, _, Phi = _random_shapes(nan_frac=0.4)
    out = criteria.mpd(Phi)
    padded = ~np.isfinite(Phi).all(axis=-1)
    assert np.isnan(out[padded]).all()
    assert np.isfinite(out[~padded]).all()


# =============================================================================
# hc_* hard criteria
# =============================================================================


def test_hc_conj_keeps_pairs_and_drops_singletons():
    lam = np.array([[1 + 2j], [1 - 2j], [3 + 4j]])  # third has no partner
    mask = criteria.hc_conj(lam)
    assert mask[:, 0].tolist() == [True, True, False]


def test_hc_conj_is_scoped_per_order():
    """A conjugate at a *different* model order must not validate a pole."""
    lam = np.array([[1 + 2j, 1 - 2j]])  # pair split across two columns
    mask = criteria.hc_conj(lam)
    assert not mask.any()


def test_hc_conj_tolerates_float32_rounding():
    lam64 = np.array([[1.0 + 2.0j], [1.0 - 2.0j]])
    lam32 = lam64.astype(np.complex64) + np.array([[0j], [1e-7j]])
    mask = criteria.hc_conj(lam32)
    assert mask.all()


def test_hc_damp_range_and_mask():
    Zeta = np.array([[-0.01, 0.02, 0.5, np.nan]])
    mask = criteria.hc_damp(Zeta, 0.1)
    assert mask.tolist() == [[False, True, False, False]]


def test_mpc_mpd_limit_masks_reject_padded_poles():
    """NaN MPC/MPD (padded poles) must fail the hard-criteria comparisons."""
    _, _, Phi = _random_shapes(nan_frac=0.3)
    padded = ~np.isfinite(Phi).all(axis=-1)
    assert not np.asarray(criteria.mpc(Phi) >= 0.7)[padded].any()
    assert not np.asarray(criteria.mpd(Phi) <= 0.3)[padded].any()


# =============================================================================
# sc_apply
# =============================================================================


def test_sc_apply_matches_reference():
    Fn, Zeta, Phi = _random_shapes(K=8, p=6, Q=5, nan_frac=0.25)
    errs = (0.01, 0.05, 0.02)
    assert np.array_equal(
        criteria.sc_apply(Fn, Zeta, Phi, 0, *errs),
        _ref_sc_apply(Fn, Zeta, Phi, 0, 5, 1, *errs),  # pyOMA-style ordmax/step
    )


def test_sc_apply_matches_reference_loose_tolerances():
    """Loose tolerances label many poles stable, exercising the True branch."""
    Fn, Zeta, Phi = _random_shapes(K=8, p=6, Q=5, nan_frac=0.1)
    errs = (10.0, 10.0, 10.0)
    assert np.array_equal(
        criteria.sc_apply(Fn, Zeta, Phi, 0, *errs),
        _ref_sc_apply(Fn, Zeta, Phi, 0, 5, 1, *errs),
    )


def test_sc_apply_labels_a_genuinely_stable_pole():
    """A pole repeated identically across orders must come out stable."""
    K, p, Q = 3, 5, 4
    Fn = np.tile(np.array([50.0, 150.0, 300.0])[:, None], (1, p))
    Zeta = np.tile(np.array([0.01, 0.015, 0.02])[:, None], (1, p))
    shape = RNG.standard_normal((K, Q)) + 1j * RNG.standard_normal((K, Q))
    Phi = np.tile(shape[:, None, :], (1, p, 1))
    lab = criteria.sc_apply(Fn, Zeta, Phi, 0, 0.01, 0.05, 0.02)
    assert lab[:, 0].tolist() == [0, 0, 0]  # column 0 has no predecessor
    assert (lab[:, 1:] == 1).all()


def test_sc_apply_ordmin_beyond_last_column_gives_all_zeros():
    Fn, Zeta, Phi = _random_shapes(K=4, p=4, Q=3, nan_frac=0.0)
    lab = criteria.sc_apply(Fn, Zeta, Phi, 99, 0.01, 0.05, 0.02)
    assert lab.shape == Fn.shape
    assert (np.asarray(lab) == 0).all()


# =============================================================================
# JAX interop — NumPy in, jax.Array out, and traceable under jit/vmap
# =============================================================================


def test_criteria_accept_numpy_and_jax_and_return_jax():
    import jax
    import jax.numpy as jnp

    Fn, Zeta, Phi = _random_shapes(nan_frac=0.2)
    Lambd = -Zeta * Fn + 1j * Fn

    for cast in (np.asarray, jnp.asarray):  # both input flavours must work
        cFn, cZeta, cPhi, cLambd = map(cast, (Fn, Zeta, Phi, Lambd))
        m_conj = criteria.hc_conj(cLambd)
        m_damp = criteria.hc_damp(cZeta, 0.1)
        m_mpc = criteria.mpc(cPhi) >= 0.7
        m_mpd = criteria.mpd(cPhi) <= 0.3
        lab = criteria.sc_apply(cFn, cZeta, cPhi, 0, 0.01, 0.05, 0.02)

        for got in (m_conj, m_damp, m_mpc, m_mpd, lab):
            assert isinstance(got, jax.Array)
            assert got.shape == Fn.shape


def test_criteria_are_vmappable():
    """The whole screening chain must trace under vmap over a batch axis."""
    import jax
    import jax.numpy as jnp

    B = 3
    stack = [_random_shapes(K=5, p=4, Q=3, nan_frac=0.1) for _ in range(B)]
    Fn = jnp.asarray(np.stack([s[0] for s in stack]))
    Zeta = jnp.asarray(np.stack([s[1] for s in stack]))
    Phi = jnp.asarray(np.stack([s[2] for s in stack]))

    def screen(fn, zeta, phi):
        keep = criteria.hc_damp(zeta, 0.1)
        keep &= (criteria.mpc(phi) >= 0.7) & (criteria.mpd(phi) <= 0.3)
        fn = jnp.where(keep, fn, jnp.nan)
        zeta = jnp.where(keep, zeta, jnp.nan)
        phi = jnp.where(keep[:, :, None], phi, jnp.nan)
        return criteria.sc_apply(fn, zeta, phi, 0, 0.01, 0.05, 0.02)

    batched = jax.vmap(screen)(Fn, Zeta, Phi)
    assert batched.shape == (B, 5, 4)
    # each batch element must equal the same call made on its own
    for b in range(B):
        assert np.array_equal(batched[b], screen(Fn[b], Zeta[b], Phi[b]))


def test_criteria_chain_on_real_plscf_output():
    """End-to-end: every criterion consumes get_poles output without crashing."""
    from test_plscf import DT, _mode_shapes, _synth_frf

    from modekit import plscf

    V = _mode_shapes()
    freq, H = _synth_frf(V)
    Ad, Bn = plscf.fit(H, freq, DT, 8)
    Phis, Lambds, _ = plscf.get_poles(Ad, Bn, DT, spectrum="frf_shaker")
    Fns = plscf.lambd_to_fn(Lambds)
    Zetas = plscf.lambd_to_zeta(Lambds)

    keep = np.asarray(
        criteria.hc_conj(Lambds)
        & criteria.hc_damp(Zetas, 0.1)
        & (criteria.mpc(Phis) >= 0.7)
    )
    Fns = np.where(keep, Fns, np.nan)
    Zetas = np.where(keep, Zetas, np.nan)
    Phis = np.where(keep[:, :, None], Phis, np.nan)

    stab_label = criteria.sc_apply(Fns, Zetas, Phis, 0, 0.01, 0.05, 0.02)
    assert stab_label.shape == Fns.shape
    # the three true modes should survive and stabilise somewhere
    assert stab_label.max() == 1
