"""
Tests for DBSCAN pole clustering in ``modekit.clustering``.

Three layers, cheapest first:

* the pure kernels — ``modal_distance`` (Eq. 1) and ``dbscan`` (the label
  propagation), the latter cross-checked against scikit-learn when available;
* ``DBSCAN.fit`` on a directly-built synthetic stabilisation diagram (fast, no
  fit) and on real ``pLSCF`` output from the ``test_plscf`` ground-truth model;
* the cluster plots and their ``pLSCF`` wrappers (Agg smoke tests).
"""

import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from test_plscf import (  # noqa: E402
    F_TRUE,
    FS,
    N_IN,
    _mac,
    _mode_shapes,
    _synth_frf,
)
from modekit import clustering, plscf  # noqa: E402
from modekit.algorithms import LSFD, PoleArrays, pLSCF  # noqa: E402

plt.rcParams["text.usetex"] = False  # tests must not require a LaTeX install


# =============================================================================
# Synthetic stabilisation diagram (built directly, no expensive fit)
# =============================================================================
DIAG_F = np.array([15.0, 35.0, 55.0])  # three well-separated modes [Hz]
DIAG_Q, DIAG_P = 4, 2


def _diagram(n_orders=30, spurious=True, conjugates=False, seed=0):
    """A ``PoleArrays`` with ``DIAG_F`` present over many orders, plus noise.

    Each mode gets a fixed complex shape with tiny per-order jitter, so its poles
    sit at modal distance ~0 from each other and far from everything else. With
    ``spurious`` a few random poles per order are added (they must fall out as
    noise); with ``conjugates`` each physical pole is duplicated at ``conj`` (to
    exercise the positive-imaginary de-duplication).
    """
    rng = np.random.default_rng(seed)
    K = 4 * (len(DIAG_F) + 4)
    Q, P, p = DIAG_Q, DIAG_P, n_orders
    shapes = rng.normal(size=(Q, len(DIAG_F))) + 1j * rng.normal(size=(Q, len(DIAG_F)))
    lr = rng.normal(size=(P, len(DIAG_F))) + 1j * rng.normal(size=(P, len(DIAG_F)))

    Lambd = np.full((K, p), np.nan, dtype=complex)
    Phi = np.full((K, p, Q), np.nan, dtype=complex)
    Lr = np.full((K, p, P), np.nan, dtype=complex)
    stab = np.zeros((K, p), dtype=int)

    def place(row, c, lam, phi, lrv):
        Lambd[row, c], Phi[row, c], Lr[row, c], stab[row, c] = lam, phi, lrv, 1

    for c in range(4, p):  # modes appear from order 5 onward
        row = 0
        for m in range(len(DIAG_F)):
            f = DIAG_F[m] + rng.normal(0, 0.01)
            lam = complex(plscf.fn_zeta_to_lambd(f, 0.02 + rng.normal(0, 1e-4)))
            phi = shapes[:, m] + rng.normal(0, 0.02, Q)
            place(row, c, lam, phi, lr[:, m])
            row += 1
            if conjugates:  # the (Im<0) conjugate, same frequency
                place(row, c, np.conj(lam), np.conj(phi), np.conj(lr[:, m]))
                row += 1
        if spurious:
            for _ in range(int(rng.integers(1, 4))):
                lam = complex(plscf.fn_zeta_to_lambd(rng.uniform(5, 60), rng.uniform(0.01, 0.08)))
                place(row, c, lam, rng.normal(size=Q) + 1j * rng.normal(size=Q),
                      rng.normal(size=P) + 1j * rng.normal(size=P))
                row += 1

    return PoleArrays(
        jnp.asarray(Lambd), jnp.asarray(Phi), jnp.asarray(Lr), jnp.asarray(stab)
    )


# =============================================================================
# modal_distance: the composite metric of Eq. (1)
# =============================================================================
def test_modal_distance_matches_the_formula():
    rng = np.random.default_rng(1)
    Q, n = 5, 12
    Fn = rng.uniform(5, 60, size=n)
    Phi = rng.normal(size=(Q, n)) + 1j * rng.normal(size=(Q, n))
    D = np.asarray(clustering.modal_distance(Fn, Phi))

    ref = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            df = abs(Fn[i] - Fn[j]) / max(Fn[i], Fn[j])
            ref[i, j] = df + (1 - _mac(Phi[:, i], Phi[:, j]))

    assert np.allclose(D, ref, atol=1e-9)
    assert np.allclose(np.diag(D), 0.0, atol=1e-12)  # a pole is 0 from itself
    assert np.allclose(D, D.T)  # symmetric


# =============================================================================
# dbscan: label propagation on a precomputed distance matrix
# =============================================================================
def test_dbscan_finds_dense_clusters_and_noise():
    # two tight blobs (mutual distance ~0) far apart, plus one lone outlier
    n = 7
    D = np.full((n, n), 10.0)
    A, B = [0, 1, 2], [3, 4, 5]  # 6 is the outlier
    for grp in (A, B):
        for i in grp:
            for j in grp:
                D[i, j] = 0.001
    np.fill_diagonal(D, 0.0)

    labels = np.asarray(clustering.dbscan(jnp.asarray(D), 0.01, 3))
    assert labels[6] == -1  # outlier is noise
    assert labels[A[0]] == labels[A[1]] == labels[A[2]]  # blob A is one cluster
    assert labels[B[0]] == labels[B[1]] == labels[B[2]]  # blob B is one cluster
    assert labels[A[0]] != labels[B[0]]  # the two blobs are distinct


# =============================================================================
# DBSCAN.fit: mode recovery on the synthetic diagram
# =============================================================================
def test_fit_recovers_the_synthetic_modes():
    res = clustering.DBSCAN().fit(_diagram())  # eps=0.005, min_pts=25, medoid
    assert res.n_clusters == len(DIAG_F)
    assert np.allclose(np.sort(np.asarray(res.Fn)), DIAG_F, atol=0.2)
    assert res.Phi.shape == (DIAG_Q, len(DIAG_F))
    assert res.Lr.shape == (DIAG_P, len(DIAG_F))


def test_fit_orders_clusters_by_ascending_frequency():
    res = clustering.DBSCAN().fit(_diagram())
    labels, Fn_fl = np.asarray(res.labels), np.asarray(res.Fn_fl)
    means = [Fn_fl[labels == m].mean() for m in range(res.n_clusters)]
    assert np.all(np.diff(means) > 0)
    assert np.all(np.diff(np.asarray(res.Fn)) > 0)


def test_fit_marks_spurious_poles_as_noise():
    res = clustering.DBSCAN().fit(_diagram(spurious=True))
    assert int((np.asarray(res.labels) == -1).sum()) > 0


def test_fit_representatives_are_lsfd_ready():
    res = clustering.DBSCAN().fit(_diagram())
    # one pole per mode, positive damped frequency (what LSFD.fit expects)
    assert np.all(np.imag(np.asarray(res.Lambd)) > 0)
    assert res.Lambd.shape[0] == res.n_clusters
    assert res.rep_idx.shape[0] == res.n_clusters


@pytest.mark.parametrize("select", ["medoid", "median", "mean"])
def test_fit_select_modes_all_recover_the_modes(select):
    res = clustering.DBSCAN(select=select).fit(_diagram())
    assert res.n_clusters == len(DIAG_F)
    assert np.allclose(np.sort(np.asarray(res.Fn)), DIAG_F, atol=0.3)


def test_fit_min_cluster_size_drops_small_clusters():
    poles = _diagram()
    full = clustering.DBSCAN().fit(poles)
    # every real cluster here spans ~26 orders; an impossibly large floor drops all
    pruned = clustering.DBSCAN(min_cluster_size=10_000).fit(poles)
    assert full.n_clusters == len(DIAG_F)
    assert pruned.n_clusters == 0


def test_fit_deduplicates_conjugate_pairs():
    """Both conjugate branches are stable, but only one pole per order survives."""
    res = clustering.DBSCAN(min_pts=10).fit(_diagram(conjugates=True, spurious=False))
    assert res.n_clusters == len(DIAG_F)  # not 2x
    assert np.all(np.imag(np.asarray(res.Lambd)) > 0)
    labels, order_fl = np.asarray(res.labels), np.asarray(res.order_fl)
    for m in range(res.n_clusters):
        orders = order_fl[labels == m]
        assert len(orders) == len(np.unique(orders))  # at most one pole per order


def test_fit_empty_diagram_returns_no_clusters():
    empty = PoleArrays(
        jnp.full((5, 4), jnp.nan, dtype=complex),
        jnp.full((5, 4, DIAG_Q), jnp.nan, dtype=complex),
        jnp.full((5, 4, DIAG_P), jnp.nan, dtype=complex),
        jnp.zeros((5, 4), dtype=int),
    )
    res = clustering.DBSCAN().fit(empty)
    assert res.n_clusters == 0
    assert res.Phi.shape == (DIAG_Q, 0)
    assert np.all(np.asarray(res.labels) == -1)


def test_cluster_result_sizes_and_counts_agree():
    res = clustering.DBSCAN().fit(_diagram())
    assert res.sizes.sum() == int((np.asarray(res.labels) != -1).sum())
    assert len(res.sizes) == res.n_clusters


def test_dbscan_rejects_unknown_select():
    with pytest.raises(ValueError, match="select"):
        clustering.DBSCAN(select="centroid")


# =============================================================================
# k_distance: the eps diagnostic
# =============================================================================
def test_k_distance_is_sorted_descending():
    poles = _diagram()
    kd = clustering.k_distance(poles, min_pts=25)
    n_stable = int((np.imag(np.asarray(poles.Lambd)) > 0).sum())  # all synth are Im>0
    assert kd.shape == (n_stable,)
    assert np.all(np.diff(kd) <= 1e-12)  # descending


def test_k_distance_separates_dense_modes_from_sparse_noise():
    kd = clustering.k_distance(_diagram(spurious=True), min_pts=25)
    # the many mode poles sit at small k-distance; the few spurious form a high head
    assert np.median(kd) < 0.05
    assert kd.max() > 5 * np.median(kd)


def test_k_distance_empty_when_too_few_poles():
    empty = PoleArrays(
        jnp.full((5, 4), jnp.nan, dtype=complex),
        jnp.full((5, 4, DIAG_Q), jnp.nan, dtype=complex),
        jnp.full((5, 4, DIAG_P), jnp.nan, dtype=complex),
        jnp.zeros((5, 4), dtype=int),
    )
    assert clustering.k_distance(empty).size == 0


# =============================================================================
# End-to-end: real pLSCF output from the test_plscf ground-truth model
# =============================================================================
@pytest.fixture(scope="module")
def fitted_clusters():
    """Cluster the poles pLSCF identifies on the analytic 3-mode FRF."""
    V = _mode_shapes()
    freq, H = _synth_frf(V)
    est = pLSCF(freq, fs=FS, ordmax=24, ordmin=2, spectrum="frf_shaker")
    poles = est.fit(H)
    # ordmax=24 caps the per-mode pole count, so lower min_pts to form clusters
    res = est.cluster(poles, min_pts=8)
    return V, freq, H, est, poles, res


def test_end_to_end_recovers_ground_truth(fitted_clusters):
    V, _, _, _, _, res = fitted_clusters
    assert res.n_clusters == len(F_TRUE)
    assert np.allclose(np.sort(np.asarray(res.Fn)), F_TRUE, rtol=2e-3)


def test_end_to_end_representative_shapes_match(fitted_clusters):
    V, _, _, _, _, res = fitted_clusters
    Phi = np.asarray(res.Phi)  # (Q, M), ascending frequency == F_TRUE order
    macs = [_mac(Phi[:, k], V[:, k]) for k in range(len(F_TRUE))]
    assert np.all(np.array(macs) > 0.99), macs


def test_end_to_end_one_pole_per_order(fitted_clusters):
    """Real fits carry conjugate pairs; the dedup must leave one pole per order."""
    _, _, _, _, _, res = fitted_clusters
    labels, order_fl = np.asarray(res.labels), np.asarray(res.order_fl)
    for m in range(res.n_clusters):
        orders = order_fl[labels == m]
        assert len(orders) == len(np.unique(orders))


def test_end_to_end_representatives_feed_lsfd(fitted_clusters):
    _, freq, H, _, _, res = fitted_clusters
    lsfd = LSFD(freq, fs=FS, spectrum="frf_shaker", quantity="displacement")
    model = lsfd.fit(H, res.Lambd, part_factors=res.Lr)
    assert model.A.shape == (H.shape[0], H.shape[1], res.n_clusters)
    assert np.isfinite(np.asarray(model.A)).all()


# =============================================================================
# Cluster plots (Agg smoke tests)
# =============================================================================
@pytest.fixture(scope="module")
def plot_case():
    poles = _diagram()
    res = clustering.DBSCAN().fit(poles)
    est = pLSCF(np.linspace(0, 64, 256), fs=128.0, ordmax=30, ordmin=2, spectrum="frf_shaker")
    return res, est


def test_plot_stab_cluster_returns_a_figure(plot_case):
    from modekit import plots

    res, _ = plot_case
    fig, ax = plots.plot_stab_cluster(
        np.asarray(res.Fn_fl),
        np.asarray(res.order_fl),
        np.asarray(res.labels),
        ordmax=30,
        plot_noise=True,
    )
    assert isinstance(fig, plt.Figure)
    assert ax.lines  # coloured pole markers + median lines
    plt.close(fig)


def test_plot_cluster_freqs_zetas_returns_a_figure(plot_case):
    from modekit import plots

    res, _ = plot_case
    fig, ax = plots.plot_cluster_freqs_zetas(
        np.asarray(res.Fn_fl),
        np.asarray(res.Zeta_fl),
        np.asarray(res.labels),
        plot_noise=True,
    )
    assert isinstance(fig, plt.Figure)
    assert len(ax.lines) >= res.n_clusters  # one marker series per cluster
    plt.close(fig)


def test_plscf_cluster_plot_wrappers_return_figures(plot_case):
    res, est = plot_case
    fig1, _ = est.stab_cluster_plot(res)
    fig2, _ = est.cluster_freqs_zetas_plot(res, plot_noise=True)
    assert isinstance(fig1, plt.Figure) and isinstance(fig2, plt.Figure)
    plt.close(fig1)
    plt.close(fig2)


def test_plot_k_distance_returns_a_figure():
    from modekit import plots

    kd = clustering.k_distance(_diagram(), min_pts=25)
    fig, ax = plots.plot_k_distance(kd, eps=clustering.DEFAULT_EPS, k=25)
    assert isinstance(fig, plt.Figure)
    assert ax.lines  # the k-distance curve + the eps line
    plt.close(fig)


def test_plscf_k_distance_plot_wrapper():
    poles = _diagram()
    est = pLSCF(np.linspace(0, 64, 256), fs=128.0, ordmax=30, ordmin=2, spectrum="frf_shaker")
    fig, ax = est.k_distance_plot(poles, min_pts=25)
    assert isinstance(fig, plt.Figure)
    assert ax.lines
    plt.close(fig)


def test_stab_cluster_plot_rejects_mmif_for_oma(plot_case):
    res, _ = plot_case
    est_oma = pLSCF(np.linspace(0, 64, 256), fs=128.0, ordmax=30, spectrum="sd_per")
    with pytest.raises(ValueError, match="mmif"):
        est_oma.stab_cluster_plot(res, data=np.zeros((2, 2, 256)), indicator="mmif")
