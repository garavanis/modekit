"""
Tests for the band-wise MPE pipeline in ``modekit.bandmpe``.

Runs ``band_mpe`` / ``full_band_mpe`` / ``plscf_local`` against the
synthetic ground-truth FRF of ``test_plscf`` and checks mode recovery, the eqx
cache round-trip, and the cross-pass pole merge.
"""

import numpy as np
import pytest

from test_plscf import F_TRUE, FS, ZETA_TRUE, _mac, _mode_shapes, _synth_frf
from modekit.algorithms import LSFD, ModalModel, PoleArrays, pLSCF
from modekit.bandmpe import (
    BandFit,
    BandMPE,
    LSFDFit,
    LocalModes,
    band_mpe,
    combine_lambd,
    full_band_mpe,
    lsfd_batch,
    plscf_local,
)
from modekit.clustering import ClusterResult
from modekit.plscf import fn_zeta_to_lambd

# 50 & 150 Hz land in band 1, 300 Hz in band 2; fs_fake ~ 2.5 * f_hi
BANDS = [(20.0, 200.0, 500.0), (200.0, 400.0, 1000.0)]
SETTINGS = dict(spectrum="frf_shaker", ordmax=24, ordmin=2, min_pts=6, progress=False)


@pytest.fixture(scope="module")
def frf():
    V = _mode_shapes()
    freq, H = _synth_frf(V)
    return V, freq, H


@pytest.fixture(scope="module")
def banded(frf):
    _, freq, H = frf
    return band_mpe(freq, H, None, BANDS, **SETTINGS)


def _fn(lambd):
    return np.sort(np.abs(np.asarray(lambd)) / (2 * np.pi))


# =============================================================================
# band_mpe
# =============================================================================


def test_band_mpe_recovers_every_mode(banded):
    assert isinstance(banded, BandMPE)
    Lambd, conf, _ = banded  # unpacks like BandFit
    fn = _fn(Lambd)
    assert fn.size >= len(F_TRUE)
    assert conf.shape == (fn.size,)
    for f in F_TRUE:
        assert np.abs(fn - f).min() < 0.01 * f, (f, fn)


def test_band_mpe_band_data_unpacks_like_tuples(banded):
    _, _, band_data = banded
    assert len(band_data) == len(BANDS)
    for bf, (flo, fhi, fs_fake) in zip(band_data, BANDS):
        assert isinstance(bf, BandFit)
        got_lo, got_hi, est, poles, res = bf  # the plot loops rely on this
        assert (got_lo, got_hi) == (flo, fhi)
        assert isinstance(est, pLSCF) and est.fs == fs_fake
        assert isinstance(poles, PoleArrays)
        assert isinstance(res, ClusterResult)
        f = np.asarray(est.freqs)
        assert f.min() >= flo and f.max() <= fhi


def test_band_mpe_cache_roundtrip(frf, tmp_path, capsys):
    _, freq, H = frf
    L1, c1, bd1 = band_mpe(
        freq, H, None, BANDS, cache="t", cache_dir=tmp_path, **SETTINGS
    )
    assert len(list(tmp_path.glob("t-*.eqx"))) == 1

    L2, c2, bd2 = band_mpe(
        freq, H, None, BANDS, cache="t", cache_dir=tmp_path, **SETTINGS
    )
    assert "loaded cached fit" in capsys.readouterr().out
    assert len(list(tmp_path.glob("t-*.eqx"))) == 1

    np.testing.assert_allclose(np.asarray(L2), np.asarray(L1))
    assert np.array_equal(c2, c1)
    for b1, b2 in zip(bd1, bd2):
        np.testing.assert_allclose(np.asarray(b2.est.freqs), np.asarray(b1.est.freqs))
        np.testing.assert_allclose(
            np.asarray(b2.poles.Fn), np.asarray(b1.poles.Fn), equal_nan=True
        )
        assert np.array_equal(
            np.asarray(b2.poles.stab_label), np.asarray(b1.poles.stab_label)
        )
        np.testing.assert_allclose(np.asarray(b2.res.Lambd), np.asarray(b1.res.Lambd))
        assert np.array_equal(np.asarray(b2.res.labels), np.asarray(b1.res.labels))


def test_band_mpe_changed_settings_change_the_cache_key(frf, tmp_path):
    _, freq, H = frf
    band_mpe(freq, H, None, BANDS, cache="t", cache_dir=tmp_path, **SETTINGS)
    band_mpe(freq, H, None, BANDS, cache="t", cache_dir=tmp_path, eps=0.03, **SETTINGS)
    assert len(list(tmp_path.glob("t-*.eqx"))) == 2


# =============================================================================
# combine_lambd
# =============================================================================


def test_combine_lambd_keeps_the_better_supported_coincident_pole():
    la = np.array([fn_zeta_to_lambd(50.0, 0.010)])
    lb = np.array([fn_zeta_to_lambd(50.2, 0.012)])
    lam, cnf = combine_lambd(la, [10], lb, [20])
    assert lam.shape == (1,)
    assert cnf.tolist() == [20]
    np.testing.assert_allclose(np.asarray(lam), lb)  # b wins on confidence

    lam, cnf = combine_lambd(la, [30], lb, [20])
    np.testing.assert_allclose(np.asarray(lam), la)  # a wins on confidence
    assert cnf.tolist() == [30]


def test_combine_lambd_concatenates_disjoint_sets_sorted():
    la = np.array([fn_zeta_to_lambd(150.0, 0.01)])
    lb = np.array([fn_zeta_to_lambd(50.0, 0.01), fn_zeta_to_lambd(300.0, 0.02)])
    lam, cnf = combine_lambd(la, [5], lb, [7, 9])
    np.testing.assert_allclose(_fn(lam), [50.0, 150.0, 300.0])
    assert cnf.tolist() == [7, 5, 9]


def test_combine_lambd_empty_side_passes_through():
    lb = np.array([fn_zeta_to_lambd(50.0, 0.01)])
    lam, cnf = combine_lambd(np.array([]), np.array([]), lb, [4])
    np.testing.assert_allclose(np.asarray(lam), lb)
    assert cnf.tolist() == [4]


def test_combine_lambd_carries_the_survivors_factors():
    la = np.array([fn_zeta_to_lambd(150.0, 0.01), fn_zeta_to_lambd(50.0, 0.010)])
    lb = np.array([fn_zeta_to_lambd(50.2, 0.012), fn_zeta_to_lambd(300.0, 0.02)])
    lr_a = np.array([[1.0, 2.0], [10.0, 20.0]])  # (P=2, 2), column per pole
    lr_b = np.array([[3.0, 4.0], [30.0, 40.0]])
    lam, cnf, lr = combine_lambd(la, [5, 10], lb, [20, 9], lr_a=lr_a, lr_b=lr_b)
    np.testing.assert_allclose(_fn(lam), [50.2, 150.0, 300.0], rtol=1e-6)
    # 50 Hz: b's pole wins (20 > 10) and brings its column; others keep theirs
    np.testing.assert_allclose(np.asarray(lr), [[3.0, 1.0, 4.0], [30.0, 10.0, 40.0]])
    with pytest.raises(ValueError, match="both"):
        combine_lambd(la, [5, 10], lb, [20, 9], lr_a=lr_a)


def test_band_mpe_lr_follows_its_pole_order(banded):
    Lambd, _, band_data = banded
    Lr = np.asarray(banded.Lr)
    assert Lr.shape == (2, Lambd.shape[0])  # (P, M) on the FRF's two inputs
    # each column is the factor the band fit found with that pole, in concatenation order
    cols = np.concatenate(
        [np.asarray(bf.res.Lr)[:, np.asarray((bf.res.Fn >= bf.flo) & (bf.res.Fn <= bf.fhi))]
         for bf in band_data],
        axis=1,
    )
    np.testing.assert_allclose(Lr, cols)


# =============================================================================
# full_band_mpe: always with fixed reference factors
# =============================================================================


def _check_rank_one_and_true_shapes(model, V):
    fn = np.asarray(model.Fn)
    good = np.isfinite(fn)
    assert good.sum() >= len(F_TRUE)
    # fixed factors make every residue matrix rank one by construction
    for k in np.where(good)[0]:
        sv = np.linalg.svd(np.asarray(model.A)[:, :, k], compute_uv=False)
        assert sv[1] / sv[0] < 1e-8, (k, sv)
    shapes = np.asarray(model.mode_shapes())
    for f, v in zip(F_TRUE, V.T):
        k = np.abs(np.where(good, fn, np.inf) - f).argmin()
        assert abs(fn[k] - f) < 0.01 * f
        assert _mac(shapes[:, k], v) > 0.99, (f, _mac(shapes[:, k], v))


def test_full_mpe_two_passes_global_refs_yield_one_model(frf):
    """Passes on one input set: the global fit's own factors carry through the merge."""
    V, freq, H = frf
    # second pass: the same system seen through a rescaled FRF
    fit, band_data = full_band_mpe(
        [(freq, H), (freq, 1.2 * H)],
        None,
        FS,
        BANDS,
        quantity="displacement",
        refs="global",
        **SETTINGS,
    )
    assert isinstance(fit, LSFDFit)
    est, model = fit
    assert isinstance(est, LSFD) and isinstance(model, ModalModel)
    assert fit.refs is None  # no local models needed
    # the estimator carries the fit's settings, so plots need not rebuild it
    assert est.quantity == "displacement"
    assert est.band == (BANDS[0][0], BANDS[-1][1])
    assert len(band_data) == 2 and all(len(bd) == len(BANDS) for bd in band_data)

    fn = np.asarray(model.Fn)
    fn = np.sort(fn[np.isfinite(fn)])
    # the merge must not double-count modes both passes found
    close = np.abs(fn[:, None] - np.asarray(F_TRUE)[None, :]) < 0.01 * np.asarray(
        F_TRUE
    )
    assert close.sum(axis=0).max() == 1, fn
    _check_rank_one_and_true_shapes(model, V)


def test_full_mpe_local_refs_yield_rank_one_residues_and_true_shapes(frf):
    """Factors re-estimated on the LSFD data by local models (the default)."""
    V, freq, H = frf
    fit, _ = full_band_mpe(
        [(freq, H)],
        None,
        FS,
        BANDS,
        quantity="displacement",
        local_kw=dict(order=2, deltaf=5.0, rtol=0.05),
        **SETTINGS,
    )
    est, model = fit  # still unpacks as a pair
    assert isinstance(est, LSFD)
    assert isinstance(fit.refs, LocalModes) and fit.refs.Lr.shape[0] == 1
    _check_rank_one_and_true_shapes(model, V)
    # the local models' own shapes agree with the residue-fit shapes here
    for k in range(len(F_TRUE)):
        phi_loc = np.asarray(fit.refs.Phi[0])[:, k]
        assert _mac(phi_loc, V[:, np.abs(F_TRUE - float(fit.refs.f_ref[k])).argmin()]) > 0.99


def test_full_mpe_f_min_drops_the_poles_below_it_only(frf):
    """The floor cuts the merged pole set before the residue stage; nothing else moves."""
    _, freq, H = frf
    kw = dict(quantity="displacement", local_kw=dict(order=2, deltaf=5.0, rtol=0.05), **SETTINGS)
    fit0, _ = full_band_mpe([(freq, H)], None, FS, BANDS, **kw)
    fn0 = np.sort(np.asarray(fit0.model.Fn))
    f_min = 0.5 * (fn0[0] + fn0[1])   # between the first two poles found
    fit, _ = full_band_mpe([(freq, H)], None, FS, BANDS, f_min=f_min, **kw)
    assert np.allclose(np.sort(np.asarray(fit.model.Fn)), fn0[1:])
    assert fit.refs.Lr.shape[2] == fn0.size - 1 and fit.refs.f_ref.size == fn0.size - 1   # Lr: (n_rec, P, M)
    # a floor under every pole changes nothing, with either source of factors
    same, _ = full_band_mpe([(freq, H)], None, FS, BANDS, f_min=0.5 * fn0[0], **kw)
    assert np.allclose(np.sort(np.asarray(same.model.Fn)), fn0)
    g0, _ = full_band_mpe([(freq, H)], None, FS, BANDS, quantity="displacement", refs="global", **SETTINGS)
    g, _ = full_band_mpe([(freq, H)], None, FS, BANDS, quantity="displacement", refs="global",
                         f_min=f_min, **SETTINGS)
    assert np.allclose(np.sort(np.asarray(g.model.Fn)), np.sort(np.asarray(g0.model.Fn))[1:])


def test_full_mpe_global_refs_reject_mixed_reference_sets(frf):
    _, freq, H = frf
    with pytest.raises(ValueError, match="one reference set"):
        full_band_mpe(
            [(freq, H), (freq, H[:, :1])],
            None,
            FS,
            BANDS,
            quantity="displacement",
            refs="global",
            **SETTINGS,
        )
    with pytest.raises(ValueError, match="refs must be"):
        full_band_mpe([(freq, H)], None, FS, BANDS, refs="free", **SETTINGS)


# =============================================================================
# plscf_local: narrow-band fixed-order fits at f_ref
# =============================================================================

# +-5 Hz at 50 Hz (~20 lines of the 0.49 Hz grid), +-5 % above
LOCAL_KW = dict(spectrum="frf_shaker", order=2, deltaf=5.0, rtol=0.05)


@pytest.fixture(scope="module")
def stack(frf):
    _, freq, H = frf
    return freq, np.stack([H, 1.05 * H, 1.1 * H])  # (3, Q, P, N)


def test_plscf_local_estimates_every_segment(frf, stack):
    V, _, _ = frf
    freq, S = stack
    sm = plscf_local(freq, S, F_TRUE, None, phi_ref=V, **LOCAL_KW)

    assert isinstance(sm, LocalModes)
    n_seg, Q, P, _ = S.shape
    M = len(F_TRUE)
    assert sm.Lambd.shape == sm.ok.shape == sm.conf.shape == sm.mac.shape == (n_seg, M)
    assert sm.Phi.shape == (n_seg, Q, M)
    assert sm.Lr.shape == (n_seg, P, M)
    np.testing.assert_allclose(np.asarray(sm.f_ref), F_TRUE)

    # every (segment, mode) gets an in-band estimate, and a clean one
    assert np.asarray(sm.ok).all(), sm.ok
    assert np.allclose(np.asarray(sm.Fn), F_TRUE[None, :], rtol=5e-3), sm.Fn
    assert np.allclose(np.asarray(sm.Zeta), ZETA_TRUE[None, :], atol=3e-3), sm.Zeta
    assert (np.asarray(sm.conf) > 0.99).all(), sm.conf
    assert (np.asarray(sm.mac) > 0.99).all(), sm.mac
    for b in range(n_seg):
        macs = [_mac(np.asarray(sm.Phi)[b, :, k], V[:, k]) for k in range(M)]
        assert np.all(np.array(macs) > 0.99), (b, macs)


def test_plscf_local_without_phi_ref_has_no_mac(stack):
    freq, S = stack
    sm = plscf_local(freq, S, F_TRUE, None, **LOCAL_KW)
    assert sm.mac is None
    assert sm.conf.shape == sm.ok.shape


def test_plscf_local_grades_a_noise_band_low(stack):
    """A band holding no mode fits worse than a resonant one and reads it in conf."""
    freq, S = stack
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(S.shape) + 1j * rng.standard_normal(S.shape)
    sm_mode = plscf_local(freq, S, F_TRUE[:1], None, **LOCAL_KW)
    sm_noise = plscf_local(freq, noise, F_TRUE[:1], None, **LOCAL_KW)
    assert (np.asarray(sm_noise.conf) < np.asarray(sm_mode.conf)).all()
    assert (np.asarray(sm_noise.conf) < 0.9).all(), sm_noise.conf


# =============================================================================
# lsfd_batch: residue fit per record at known poles
# =============================================================================


def test_lsfd_batch_recovers_shapes_per_record(frf, stack):
    V, _, _ = frf
    freq, S = stack
    n_seg, Q, P, _ = S.shape
    M = len(F_TRUE)
    lam = np.tile(np.asarray(fn_zeta_to_lambd(F_TRUE, ZETA_TRUE)), (n_seg, 1))  # (n_seg, M)
    # factors held fixed from a local fit on one record, as the full-signal stage does
    Lr = np.asarray(plscf_local(freq, S[:1], F_TRUE, None, **LOCAL_KW).Lr[0])  # (P, M)
    band = (BANDS[0][0], BANDS[-1][1])

    model, Phi = lsfd_batch(freq, S, lam, Lr, None, FS, spectrum="frf_shaker", band=band)
    assert isinstance(model, ModalModel) and model.A.shape == (n_seg, Q, P, M)
    assert Phi.shape == (n_seg, Q, M) and np.iscomplexobj(np.asarray(Phi))
    for b in range(n_seg):
        macs = [_mac(np.asarray(Phi)[b, :, k], V[:, k]) for k in range(M)]
        assert np.all(np.array(macs) > 0.99), (b, macs)

    # a NaN pole sits out its record: NaN shape there, the other modes untouched
    lam_gap = lam.copy()
    lam_gap[1, 0] = np.nan
    _, Phi_gap = lsfd_batch(freq, S, lam_gap, Lr, None, FS, spectrum="frf_shaker", band=band)
    assert np.isnan(np.asarray(Phi_gap)[1, :, 0]).all()
    assert np.isfinite(np.asarray(Phi_gap)[1, :, 1:]).all()
    assert _mac(np.asarray(Phi_gap)[1, :, 1], V[:, 1]) > 0.99

    # unconstrained residues (no factors) recover the shapes as well
    _, Phi_free = lsfd_batch(freq, S, lam, None, None, FS, spectrum="frf_shaker", band=band)
    assert Phi_free.shape == (n_seg, Q, M)
    assert _mac(np.asarray(Phi_free)[0, :, 0], V[:, 0]) > 0.99


def test_lsfd_batch_batch_size_paths_agree(stack):
    """0 (one vmap), None (one record at a time) and a chunk size with a remainder
    give the same shapes, for shared and per-record factors alike."""
    freq, S = stack
    n_seg = S.shape[0]  # 3: batch_size=2 exercises one scan chunk plus a remainder
    lam = np.tile(np.asarray(fn_zeta_to_lambd(F_TRUE, ZETA_TRUE)), (n_seg, 1))
    Lr = np.asarray(plscf_local(freq, S[:1], F_TRUE, None, **LOCAL_KW).Lr[0])
    kw = dict(spectrum="frf_shaker", band=(BANDS[0][0], BANDS[-1][1]))

    def same(A, B):  # up to the unit phase a batched vs unbatched SVD may pick
        A, B = np.asarray(A), np.asarray(B)
        np.testing.assert_allclose(np.abs(A), np.abs(B), rtol=1e-6, atol=1e-9)
        for b in range(A.shape[0]):
            for k in range(A.shape[2]):
                assert _mac(A[b, :, k], B[b, :, k]) > 1 - 1e-8, (b, k)

    _, Phi0 = lsfd_batch(freq, S, lam, Lr, None, FS, **kw)
    _, Phi1 = lsfd_batch(freq, S, lam, Lr, None, FS, batch_size=None, **kw)
    _, Phi2 = lsfd_batch(freq, S, lam, Lr, None, FS, batch_size=2, **kw)
    same(Phi1, Phi0)
    same(Phi2, Phi0)
    Lr3 = np.repeat(Lr[None], n_seg, axis=0)  # (n_seg, P, M): the per-record path
    _, Phi3 = lsfd_batch(freq, S, lam, Lr3, None, FS, batch_size=2, **kw)
    same(Phi3, Phi0)


def test_plscf_local_batch_size_does_not_change_the_result(stack):
    freq, S = stack
    sm_a = plscf_local(freq, S, F_TRUE, None, phi_ref=None, **LOCAL_KW)
    sm_b = plscf_local(freq, S, F_TRUE, None, batch_size=2, **LOCAL_KW)
    for field in ("Lambd", "Phi", "Lr", "ok", "conf"):
        np.testing.assert_allclose(
            np.asarray(getattr(sm_b, field)),
            np.asarray(getattr(sm_a, field)),
            rtol=1e-10,
            err_msg=field,
        )


def test_plscf_local_sorts_f_ref_and_phi_ref(frf, stack):
    V, _, _ = frf
    freq, S = stack
    perm = [1, 2, 0]
    sm = plscf_local(
        freq, S, np.asarray(F_TRUE)[perm], None, phi_ref=V[:, perm], **LOCAL_KW
    )
    np.testing.assert_allclose(np.asarray(sm.f_ref), np.sort(F_TRUE))
    assert (np.asarray(sm.mac) > 0.99).all()  # phi_ref followed the sort


def test_plscf_local_rejects_too_narrow_bands(stack):
    freq, S = stack
    with pytest.raises(ValueError, match="fewer than"):
        plscf_local(
            freq, S, F_TRUE, None, spectrum="frf_shaker", deltaf=0.5, rtol=0.0
        )


# =============================================================================
# LSFDFit.restrict
# =============================================================================


def _rank_one_fit(Q=6, P=6, M=4, seed=0, with_refs=True):
    """A fit with rank-one residues phi (x) lr, so the shapes are known exactly."""
    rng = np.random.default_rng(seed)
    fn = np.array([12.0, 30.0, 55.0, 80.0])[:M]
    lam = np.asarray(fn_zeta_to_lambd(fn, np.full(M, 0.01)))
    cx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    phi, lr = cx(Q, M), cx(P, M)
    model = ModalModel(
        lam, phi[:, None, :] * lr[None, :, :], np.zeros((Q, P), complex), np.zeros((Q, P), complex)
    )
    refs = None
    if with_refs:
        refs = LocalModes(fn, lam[None], phi[None], lr[None], np.ones((1, M), bool), np.ones((1, M)), None)
    return LSFDFit(None, model, refs), phi, lr


def test_lsfdfit_restrict_cuts_channels_and_keeps_poles():
    from modekit import criteria

    fit, phi, lr = _rank_one_fit()
    keep, ref_keep = [0, 1, 2, 3, 5], [0, 2, 3, 4, 5]
    red = fit.restrict(keep, ref_keep)
    assert np.allclose(red.model.Lambd, fit.model.Lambd)
    assert red.model.A.shape == (5, 5, 4) and red.model.LR.shape == (5, 5) and red.model.UR.shape == (5, 5)
    assert np.allclose(red.refs.Phi[0], phi[keep]) and np.allclose(red.refs.Lr[0], lr[ref_keep])
    # rank-one residues: the cut model's shapes are the reference shapes on the kept rows
    mac = np.asarray(criteria.mac_paired(red.model.mode_shapes(), phi[keep]))
    assert np.allclose(mac, 1.0)
    # the estimator is passed through and refs=None (refs='global') survives
    fit_g, _, _ = _rank_one_fit(with_refs=False)
    assert fit_g.restrict(keep, ref_keep).refs is None


def test_lsfdfit_restrict_with_every_channel_is_the_identity():
    fit, _, _ = _rank_one_fit()
    same = fit.restrict(list(range(6)), list(range(6)))
    assert np.allclose(same.model.A, fit.model.A)
    assert np.allclose(same.refs.Phi, fit.refs.Phi) and np.allclose(same.refs.Lr, fit.refs.Lr)
