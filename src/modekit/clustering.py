"""
Density-based clustering of pLSCF stabilisation-diagram poles.

A stabilisation diagram scatters, across model orders, many poles per physical
mode. Clustering groups those poles into modes: each dense group is one mode,
and the sparse poles fall out as noise. The grouping runs DBSCAN on the
composite *modal distance* between two poles ``p_i`` and ``p_j``

    d(p_i, p_j) = |f_i - f_j| / max(f_i, f_j) + (1 - MAC(phi_i, phi_j))     (1)

i.e. a normalised natural-frequency gap plus a mode-shape (MAC) gap, following
Boroschek & Bilbao. Two poles of the same mode at neighbouring orders sit at ``d ~ 0``;
poles of different modes sit far apart, so a small reachability radius ``eps``
isolates each mode.

Shape notation follows ``plscf``: ``Q`` outputs, ``P`` references, ``K`` poles
per order, ``p`` model orders, ``n`` flattened stable poles, ``M`` modes found.

The distance matrix (:func:`modal_distance`) and the DBSCAN label propagation
(:func:`dbscan`) are pure and traceable (safe under ``jax.jit``). The flattening
and cluster post-processing in :meth:`DBSCAN.fit` are eager, like
:func:`plscf.mpe`, because the pole and cluster counts are data-dependent.
"""

from typing import NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from modekit import criteria, plscf

# Reachability radius and minimum object count of Boroschek & Bilbao (Eq. 1).
DEFAULT_EPS = 0.005
DEFAULT_MIN_PTS = 25


@jax.jit
def modal_distance(Fn: jax.Array, Phi: jax.Array) -> jax.Array:
    """
    Composite modal distance matrix between poles, Eq. (1).

    Pure and traceable (safe under ``jax.jit``).

    Parameters
    ----------
    Fn : array_like
        Natural frequencies [Hz], shape ``(n,)``.
    Phi : array_like
        Complex mode shapes, shape ``(Q, n)`` with one mode per column.

    Returns
    -------
    jax.Array
        Symmetric distance matrix ``(n, n)`` with a zero diagonal; entry
        ``(i, j)`` is the normalised frequency gap plus ``1 - MAC``.
    """
    Fn = jnp.asarray(Fn, dtype=float)
    Phi = jnp.asarray(Phi)

    fmax = jnp.maximum(Fn[:, None], Fn[None, :])
    fmax = jnp.where(fmax > 0, fmax, 1.0)  # guard the (spurious) zero-frequency pole
    d_f = jnp.abs(Fn[:, None] - Fn[None, :]) / fmax

    d_mac = 1.0 - criteria.mac(Phi, Phi)  # (n, n), MAC of every pole pair

    D = d_f + d_mac
    D = 0.5 * (D + D.T)  # kill rounding asymmetry from the MAC
    return jnp.clip(D, 0.0, None)


def dbscan(D: jax.Array, eps: jax.Array, min_pts: jax.Array) -> jax.Array:
    """
    DBSCAN on a precomputed distance matrix, by matrix label propagation.

    Pure and traceable. A point is *core* when at least ``min_pts`` points
    (itself included) lie within ``eps``; connected core points share a label,
    non-core points within ``eps`` of a core point join its cluster, and the
    rest are noise. Because every core point pulls in its whole ``eps``
    neighbourhood, each returned cluster already holds at least ``min_pts``
    poles.

    Parameters
    ----------
    D : array_like
        Pairwise distance matrix, shape ``(n, n)``.
    eps : float
        Neighbourhood radius (the reachability distance).
    min_pts : int
        Minimum neighbourhood count for a core point (self included).

    Returns
    -------
    jax.Array
        Raw integer labels, shape ``(n,)``: each cluster is tagged by one of its
        member indices (not yet sequential), ``-1`` marks noise. Re-labelled by
        :meth:`DBSCAN.fit`.
    """
    D = jnp.asarray(D)
    n = D.shape[0]

    A = D <= eps  # eps-neighbourhood adjacency; the diagonal (self) is included
    degrees = jnp.sum(A, axis=1)
    core = degrees >= min_pts
    A_core = A & core[:, None] & core[None, :]

    # connected components of the core graph, by propagating the largest index
    def cond(state):
        _, changed = state
        return changed

    def body(state):
        labels, _ = state
        new = jnp.max(jnp.where(A_core, labels[None, :], -1), axis=1)
        new = jnp.where(core, new, labels)  # only core points move
        return new, jnp.any(new != labels)

    labels, _ = jax.lax.while_loop(cond, body, (jnp.arange(n), True))

    # border points: non-core, but within eps of a core point -> its cluster
    A_border = A & (~core[:, None]) & core[None, :]
    border = jnp.max(jnp.where(A_border, labels[None, :], -1), axis=1)
    is_border = jnp.any(A_border, axis=1)

    return jnp.where(core, labels, jnp.where(is_border, border, -1))


_dbscan = jax.jit(dbscan)


# =============================================================================
# Flatten + eps diagnostic
# =============================================================================
class _Flat(NamedTuple):
    """Stable, finite, positive-imaginary poles flattened to 1-D (``n`` entries)."""

    Fn: np.ndarray  # (n,)
    Zeta: np.ndarray  # (n,)
    order: np.ndarray  # (n,), model order of each pole
    Phi: np.ndarray  # (n, Q)
    Lr: np.ndarray  # (n, P)
    Lambd: np.ndarray  # (n,)


def _flatten_stable(poles) -> _Flat:
    """
    Flatten the poles that take part in clustering (shared by fit/k_distance).

    Keeps stable poles with a finite frequency and mode shape, one per conjugate
    pair (``Im > 0``, so ``min_pts`` counts model orders). Column ``c`` is order
    ``c + 1``.
    """
    Fn = np.asarray(poles.Fn, dtype=float)
    Zeta = np.asarray(poles.Zeta, dtype=float)
    Phi = np.asarray(poles.Phi)
    Lr = np.asarray(poles.Lr)
    Lambd = np.asarray(poles.Lambd)
    stab = np.asarray(poles.stab_label)
    _, p = Fn.shape

    orders = np.broadcast_to(np.arange(1, p + 1), Fn.shape)
    keep = (
        (stab == 1)
        & np.isfinite(Fn)
        & np.isfinite(Phi).all(axis=2)
        & (np.imag(Lambd) > 0)
    )
    return _Flat(
        Fn[keep], Zeta[keep], orders[keep].astype(int), Phi[keep], Lr[keep], Lambd[keep]
    )


def k_distance(poles, min_pts: int = DEFAULT_MIN_PTS) -> np.ndarray:
    """
    Sorted k-distance curve for choosing / validating the DBSCAN ``eps``.

    For every clustered pole this is the modal distance to its ``min_pts``-th
    nearest neighbour, returned in descending order. The low plateau on the
    right is the dense poles inside modes; the sharp rise on the left is the
    sparse / noise poles. A good ``eps`` sits at the knee between them.

    Parameters
    ----------
    poles : PoleArrays
        Screened poles, flattened exactly as :meth:`DBSCAN.fit` flattens them.
    min_pts : int, optional
        Neighbour rank ``k``, i.e. the DBSCAN ``min_pts``. Default ``25``.

    Returns
    -------
    numpy.ndarray
        The ``min_pts``-NN modal distance of each pole, sorted descending, shape
        ``(n,)``.
    """
    fl = _flatten_stable(poles)
    n = fl.Fn.shape[0]
    if n < 2:
        return np.zeros(0)
    D = np.asarray(modal_distance(fl.Fn, fl.Phi.T))  # (n, n)
    kcol = int(np.clip(min_pts - 1, 1, n - 1))  # min_pts-th smallest incl. self
    kdist = np.sort(D, axis=1)[:, kcol]
    return np.sort(kdist)[::-1]


# =============================================================================
# Result container
# =============================================================================
class ClusterResult(eqx.Module):
    """
    Modes recovered from a stabilisation diagram, one per surviving cluster.

    ``Lambd``/``Phi``/``Lr`` are the representative modes (ascending frequency)
    and feed :class:`~modekit.algorithms.LSFD` directly. The
    ``*_fl`` arrays are the flattened stable poles that were clustered, kept for
    the cluster plots and for inspection.
    """

    Lambd: jax.Array  # representative complex poles, (M,)
    Phi: jax.Array  # representative mode shapes, (Q, M)
    Lr: jax.Array  # representative reference factors, (P, M)
    Lambd_fl: jax.Array  # clustered complex poles, (n,)
    order_fl: jax.Array  # model order of each clustered pole, (n,)
    labels: jax.Array  # cluster id per pole (0..M-1), -1 = noise, (n,)
    rep_idx: jax.Array  # index of each cluster's representative pole, (M,)

    @property
    def Fn(self) -> jax.Array:
        """Representative natural frequencies [Hz], (M,)."""
        return plscf.lambd_to_fn(self.Lambd)

    @property
    def Zeta(self) -> jax.Array:
        """Representative damping ratios, (M,)."""
        return plscf.lambd_to_zeta(self.Lambd)

    @property
    def Fn_fl(self) -> jax.Array:
        """Clustered pole frequencies [Hz], (n,)."""
        return plscf.lambd_to_fn(self.Lambd_fl)

    @property
    def Zeta_fl(self) -> jax.Array:
        """Clustered pole damping ratios, (n,)."""
        return plscf.lambd_to_zeta(self.Lambd_fl)

    @property
    def n_clusters(self) -> int:
        """Number of modes (clusters) kept."""
        return int(self.Lambd.shape[0])

    @property
    def sizes(self) -> np.ndarray:
        """Pole count of each cluster, ordered like ``Fn``."""
        labels = np.asarray(self.labels)
        return np.array([int((labels == m).sum()) for m in range(self.n_clusters)])


# =============================================================================
# Estimator
# =============================================================================
class DBSCAN(eqx.Module):
    """
    DBSCAN clustering of pLSCF poles on the modal distance of Eq. (1).

    Eager (not traceable): the stable-pole and cluster counts are data-dependent.
    """

    eps: float = eqx.field(static=True)
    min_pts: int = eqx.field(static=True)
    select: str = eqx.field(static=True)
    min_cluster_size: Optional[int] = eqx.field(static=True)

    def __init__(
        self,
        eps: float = DEFAULT_EPS,
        min_pts: int = DEFAULT_MIN_PTS,
        select: str = "density",
        min_cluster_size: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        eps : float, optional
            Reachability radius on the modal distance. Default ``0.005``.
        min_pts : int, optional
            Minimum neighbourhood count for a core pole. Default ``25``.
        select : {'medoid', 'median', 'mean', 'density'}, optional
            How each cluster's representative mode is read off:

            - ``'medoid'`` (default): the pole with least total distance to the
              rest of its cluster.
            - ``'median'``: the pole whose frequency is nearest the cluster
              median frequency.
            - ``'mean'``: cluster-mean frequency and damping, with the mode
              shapes phase-aligned and averaged.
            - ``'density'``: the densest pole by iterative Core-Distance peeling
              (Boroschek & Bilbao 2019, Sec. 3.3).
        min_cluster_size : int, optional
            Drop clusters with fewer than this many poles (assigning them to
            noise). ``None`` (default) keeps every DBSCAN cluster.
        """
        if select not in ("medoid", "median", "mean", "density"):
            raise ValueError(
                f"select must be 'medoid', 'median', 'mean' or 'density', "
                f"got {select!r}"
            )
        self.eps = float(eps)
        self.min_pts = int(min_pts)
        self.select = select
        self.min_cluster_size = (
            None if min_cluster_size is None else int(min_cluster_size)
        )

    def fit(self, poles) -> ClusterResult:
        """
        Cluster the stable poles of ``poles`` into modes.

        Parameters
        ----------
        poles : PoleArrays
            Screened poles from :meth:`~modekit.algorithms.pLSCF.fit`.
            Only poles labelled stable, with a finite frequency and mode shape,
            and one per conjugate pair (positive damped frequency) take part.

        Returns
        -------
        ClusterResult
        """
        # flatten the stable, finite, positive-imaginary poles (shared with
        # k_distance); Q, P survive an empty flatten as the trailing axes.
        fl = _flatten_stable(poles)
        Fn_fl, Zeta_fl, order_fl = fl.Fn, fl.Zeta, fl.order
        Phi_fl, Lr_fl, Lambd_fl = fl.Phi, fl.Lr, fl.Lambd
        Q, P = Phi_fl.shape[1], Lr_fl.shape[1]
        n = Fn_fl.shape[0]

        if n < self.min_pts:  # too few poles to form even one cluster
            return self._result([], Q, P, Lambd_fl, order_fl, np.full(n, -1))

        # ---- distances + DBSCAN (pure/jit), then eager post-processing ----
        D = np.asarray(modal_distance(Fn_fl, Phi_fl.T))  # (n, n)
        raw = np.asarray(_dbscan(D, self.eps, self.min_pts))  # (n,)

        clusters = [np.where(raw == lab)[0] for lab in np.unique(raw) if lab != -1]
        if self.min_cluster_size is not None:
            clusters = [c for c in clusters if c.size >= self.min_cluster_size]

        # ---- representative pole per cluster, ordered by ascending frequency ----
        reps = [
            self._representative(c, D, Fn_fl, Zeta_fl, Phi_fl, Lr_fl, Lambd_fl)
            for c in clusters
        ]
        order = np.argsort([r["fn"] for r in reps]) if reps else np.array([], int)

        labels = np.full(n, -1)
        cluster_bundle = []
        for new_lab, k in enumerate(order):
            labels[clusters[k]] = new_lab
            cluster_bundle.append(reps[k])

        return self._result(cluster_bundle, Q, P, Lambd_fl, order_fl, labels)

    # ---- internals ----------------------------------------------------------
    def _representative(self, ind, D, Fn_fl, Zeta_fl, Phi_fl, Lr_fl, Lambd_fl) -> dict:
        """Reduce one cluster (row indices ``ind``) to a representative mode."""
        # medoid: least total distance to the rest of the cluster
        medoid = ind[np.argmin(D[np.ix_(ind, ind)].sum(axis=1))]

        if self.select == "mean":
            fn = float(Fn_fl[ind].mean())
            lambd = complex(plscf.fn_zeta_to_lambd(fn, Zeta_fl[ind].mean()))
            phi = _mean_shape(Phi_fl[ind], Phi_fl[medoid])
            lr = Lr_fl[ind].mean(axis=0)
            rep = medoid  # nearest real pole, for the plot marker only
        else:
            if self.select == "median":
                med_f = np.median(Fn_fl[ind])
                rep = ind[np.argmin(np.abs(Fn_fl[ind] - med_f))]
            elif self.select == "density":
                rep = _densest(ind, D)
            else:  # medoid
                rep = medoid
            fn = float(Fn_fl[rep])
            lambd = complex(Lambd_fl[rep])
            phi = Phi_fl[rep]
            lr = Lr_fl[rep]

        return {"fn": fn, "lambd": lambd, "phi": phi, "lr": lr, "rep": int(rep)}

    def _result(self, bundle, Q, P, Lambd_fl, order_fl, labels) -> ClusterResult:
        """Assemble a :class:`ClusterResult` from per-cluster representatives."""
        if bundle:
            Lambd = np.array([b["lambd"] for b in bundle], dtype=complex)
            Phi = np.stack([b["phi"] for b in bundle], axis=1)  # (Q, M)
            Lr = np.stack([b["lr"] for b in bundle], axis=1)  # (P, M)
            rep_idx = np.array([b["rep"] for b in bundle], dtype=int)
        else:
            Lambd = np.zeros(0, dtype=complex)
            Phi = np.zeros((Q, 0), dtype=complex)
            Lr = np.zeros((P, 0), dtype=complex)
            rep_idx = np.zeros(0, dtype=int)

        return ClusterResult(
            Lambd=jnp.asarray(Lambd),
            Phi=jnp.asarray(Phi),
            Lr=jnp.asarray(Lr),
            Lambd_fl=jnp.asarray(Lambd_fl),
            order_fl=jnp.asarray(order_fl),
            labels=jnp.asarray(labels),
            rep_idx=jnp.asarray(rep_idx),
        )


def _mean_shape(Phi_ind: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Phase-aligned mean of a cluster's complex mode shapes.

    Each shape ``(m, Q)`` is rescaled by the complex factor that best matches it
    to ``ref`` before averaging, then the mean is normalised to unit max-abs
    entry (as :func:`plscf.ss_to_modal_params` normalises single shapes).
    """
    num = Phi_ind.conj() @ ref  # phi_k^H ref
    den = np.einsum("kq,kq->k", Phi_ind.conj(), Phi_ind)  # phi_k^H phi_k
    aligned = (num / den)[:, None] * Phi_ind  # (m, Q)
    mean = aligned.mean(axis=0)
    return mean / mean[np.argmax(np.abs(mean))]


def _densest(ind: np.ndarray, D: np.ndarray) -> int:
    """
    Densest pole of a cluster by iterative Core-Distance peeling.

    Boroschek & Bilbao (2019), Sec. 3.3.

    Parameters
    ----------
    ind : np.ndarray
        Row indices of the cluster's poles into ``D``.
    D : np.ndarray
        The full pairwise modal-distance matrix (zero diagonal).

    Returns
    -------
    int
        Index into ``D`` of the selected representative pole.
    """
    ind = np.asarray(ind)
    while ind.size > 1:
        m = ind.size
        # core distance: k-th nearest neighbour within the subset
        k = min((m + 1) // 2, m - 1)
        cd = np.sort(D[np.ix_(ind, ind)], axis=1)[:, k]
        keep = cd <= cd.mean()  # drop poles above the mean core distance
        ind = ind[keep] if int(keep.sum()) < m else np.delete(ind, int(np.argmax(cd)))
    return int(ind[0])
