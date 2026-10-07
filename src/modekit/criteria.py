"""
Stabilisation / validation helpers for modal identification.

Shape notation follows ``plscf``: ``K`` poles per order, ``p`` model orders,
``Q`` response locations, so the pole arrays are ``(K, p)`` and mode shapes are
``(K, p, Q)``.

All functions are pure and traceable (safe under ``jax.jit``/``jax.vmap``).
NumPy input is accepted; outputs are ``jax.Array``.

https://community.sw.siemens.com/s/article/Simcenter-Testlab-Modal-Validation
"""

import jax
import jax.numpy as jnp


def _mac_elementwise(a: jax.Array, b: jax.Array) -> jax.Array:
    """
    MAC between mode-shape vectors along the last axis; leading axes broadcast.

    Parameters
    ----------
    a, b : jax.Array
        Complex mode shapes, shape ``(..., Q)``.

    Returns
    -------
    jax.Array
        MAC values, shape ``(...)``; NaN for NaN or zero mode shapes.
    """
    cross = jnp.einsum("...q,...q->...", a.conj(), b)
    norm_a = jnp.einsum("...q,...q->...", a.conj(), a).real
    norm_b = jnp.einsum("...q,...q->...", b.conj(), b).real
    return jnp.abs(cross) ** 2 / (norm_a * norm_b)


def mac(phi_X, phi_A) -> jax.Array:
    """
    Modal Assurance Criterion (MAC) between two sets of mode shapes.

    Parameters
    ----------
    phi_X, phi_A : array_like
        Mode shapes, shape ``(Q, n)``, or a single ``(Q,)`` vector.

    Returns
    -------
    jax.Array
        MAC matrix ``(n_X, n_A)``, or a scalar when both inputs hold one shape.
    """
    phi_X = jnp.asarray(phi_X)
    phi_A = jnp.asarray(phi_A)

    if phi_X.ndim == 1:
        phi_X = phi_X[:, None]
    if phi_A.ndim == 1:
        phi_A = phi_A[:, None]

    if phi_X.ndim > 2 or phi_A.ndim > 2:
        raise ValueError(
            f"Mode shape matrices must have 1 or 2 dimensions "
            f"(phi_X: {phi_X.ndim}, phi_A: {phi_A.ndim})"
        )
    if phi_X.shape[0] != phi_A.shape[0]:
        raise ValueError(
            f"Mode shapes must have the same first dimension "
            f"(phi_X: {phi_X.shape[0]}, phi_A: {phi_A.shape[0]})"
        )

    # broadcast (n_X, 1, Q) against (1, n_A, Q) -> (n_X, n_A)
    out = _mac_elementwise(phi_X.T[:, None, :], phi_A.T[None, :, :])
    return out[0, 0] if out.shape == (1, 1) else out


def mac_paired(phi_X, phi_A) -> jax.Array:
    """
    MAC between corresponding mode shapes of two sets.

    Where :func:`mac` crosses every shape of one set with every shape of the
    other, this pairs the ``m``-th column of ``phi_X`` with the ``m``-th column
    of ``phi_A`` — e.g. a batch of tracked shapes against their references.

    Parameters
    ----------
    phi_X, phi_A : array_like
        Mode shapes ``(..., Q, M)``. Leading axes broadcast, so a batch
        ``(n_seg, Q, M)`` pairs with a single reference ``(Q, M)``.

    Returns
    -------
    jax.Array
        MAC values ``(..., M)``; NaN for NaN or zero mode shapes.
    """
    X = jnp.moveaxis(jnp.asarray(phi_X), -2, -1)  # (..., M, Q)
    A = jnp.moveaxis(jnp.asarray(phi_A), -2, -1)
    return _mac_elementwise(X, A)


def mpc(phi) -> jax.Array:
    """
    Modal Phase Collinearity (MPC) of complex mode shapes, in [0, 1].

    1 → real/collinear (normal) mode; lower → more complex mode. Vectorised over
    any leading axes: ``(Q,)`` gives a scalar, ``(K, p, Q)`` gives ``(K, p)``.

    Parameters
    ----------
    phi : array_like
        Complex mode shapes, shape ``(..., Q)``.

    Returns
    -------
    jax.Array
        MPC values, shape ``(...)``; NaN where the mode shape is NaN or zero.
    """
    phi = jnp.asarray(phi)
    x = phi.real - phi.real.mean(axis=-1, keepdims=True)
    y = phi.imag - phi.imag.mean(axis=-1, keepdims=True)

    # MPC = (l0 - l1)^2 / (l0 + l1)^2 of the (Re, Im) scatter matrix,
    # computed as (tr^2 - 4*det) / tr^2 to skip the eigensolve
    sxx = jnp.einsum("...q,...q->...", x, x)
    syy = jnp.einsum("...q,...q->...", y, y)
    sxy = jnp.einsum("...q,...q->...", x, y)
    tr = sxx + syy
    det = sxx * syy - sxy**2

    out = (tr**2 - 4 * det) / tr**2
    # clip rounding error at the limits; NaN passes through
    return jnp.clip(out, 0.0, 1.0)


def mpd(phi) -> jax.Array:
    """
    Mean Phase Deviation (MPD) [rad] of complex mode shapes.

    The amplitude-weighted scatter of the phases about the best-fit line through
    the complex plane; 0 → perfectly real mode. Vectorised like :func:`mpc`.

    Parameters
    ----------
    phi : array_like
        Complex mode shapes, shape ``(..., Q)``.

    Returns
    -------
    jax.Array
        MPD values, shape ``(...)``; NaN where the mode shape is NaN or zero.
    """
    phi = jnp.asarray(phi)
    flat = phi.reshape(-1, phi.shape[-1])  # (B, Q)

    # the SVD rejects non-finite input: zero-fill NaN shapes, mask back after
    finite = jnp.isfinite(flat).all(axis=-1)  # (B,)
    safe = jnp.where(finite[:, None], flat, 0.0)

    # second right-singular vector: normal to the best-fit line in (Re, Im)
    _, _, Vh = jnp.linalg.svd(jnp.stack([safe.real, safe.imag], axis=-1))
    v = Vh[:, 1, :]  # (B, 2) = V[:, 1]

    w = jnp.abs(safe)  # (B, Q), amplitude weights
    num = safe.real * v[:, 1:2] - safe.imag * v[:, 0:1]
    den = jnp.sqrt(v[:, 0:1] ** 2 + v[:, 1:2] ** 2) * w
    # clip keeps arccos real when |num/den| overshoots 1 by rounding
    angle = jnp.arccos(jnp.clip(jnp.abs(num / den), 0.0, 1.0))
    out = (w * angle).sum(axis=-1) / w.sum(axis=-1)

    return jnp.where(finite, out, jnp.nan).reshape(phi.shape[:-1])


def hc_conj(Lambd, rtol: float = 1e-5, atol: float = 1e-8) -> jax.Array:
    """
    Boolean keep-mask of poles whose complex conjugate is also present at the
    same model order.

    Matching is per column, via :func:`jax.numpy.isclose`.

    Parameters
    ----------
    Lambd : array_like
        Complex poles, shape ``(K, p)``.
    rtol, atol : float, optional
        Tolerances passed to :func:`jax.numpy.isclose`.

    Returns
    -------
    jax.Array
        Boolean mask, shape ``(K, p)``; False for NaN or inf poles.
    """
    Lambd = jnp.asarray(Lambd)
    # (K, K, p) pairwise check per order; NaN never compares close, so
    # padded poles are dropped
    close = jnp.isclose(
        Lambd[:, None, :], Lambd.conj()[None, :, :], rtol=rtol, atol=atol
    )
    return close.any(axis=1) & jnp.isfinite(Lambd)


def hc_damp(Zeta, max_damp: float) -> jax.Array:
    """
    Boolean keep-mask of damping ratios in ``(0, max_damp)``.

    Parameters
    ----------
    Zeta : array_like
        Damping ratios, shape ``(K, p)``.
    max_damp : float
        Upper damping limit.

    Returns
    -------
    jax.Array
        Boolean mask, shape ``(K, p)``; NaN-padded poles give False.
    """
    Zeta = jnp.asarray(Zeta)
    return (Zeta > 0) & (Zeta < max_damp)  # NaN-padded poles compare False


def sc_apply(Fn, Zeta, Phi, ordmin: int, err_fn, err_zeta, err_phi) -> jax.Array:
    """
    Label poles stable (1) / unstable (0) by comparing each order with the previous.

    For every pole the nearest-in-frequency pole one order below is found; the
    pole is stable when the relative frequency error is < ``err_fn``, the
    relative damping error < ``err_zeta`` and ``1 - MAC`` < ``err_phi``.

    Parameters
    ----------
    Fn, Zeta : array_like
        Frequencies / damping ratios, shape ``(K, p)``.
    Phi : array_like
        Mode shapes, shape ``(K, p, Q)``.
    ordmin : int
        Lowest column to score; static (it drives Python-level slicing).
        Columns below it and column 0 keep the 0 label.
    err_fn, err_zeta, err_phi : float
        Relative tolerances on frequency, damping and ``1 - MAC``.

    Returns
    -------
    stab_label : jax.Array
        Integer labels, shape ``(K, p)``; 1 = stable, 0 = unstable.
    """
    Fn = jnp.asarray(Fn)
    Zeta = jnp.asarray(Zeta)
    Phi = jnp.asarray(Phi)
    p = Fn.shape[1]

    # nearest pole one order below, for every pole
    cur, prev = Fn[:, 1:], Fn[:, :-1]  # (K, p-1)
    dist = jnp.abs(cur[:, None, :] - prev[None, :, :])  # (K, K, p-1)
    # NaN distances -> inf so padded poles are never the nearest; a NaN pole
    # fails the tolerance checks below and stays labelled 0
    dist = jnp.where(jnp.isnan(dist), jnp.inf, dist)
    idx = dist.argmin(axis=1)  # (K, p-1)

    cols = jnp.arange(p - 1)
    f_prev = prev[idx, cols]
    z_prev = Zeta[:, :-1][idx, cols]
    phi_prev = Phi[:, :-1][idx, cols, :]  # (K, p-1, Q)

    # a zero reference fn/zeta gives 0/0; the NaN fails the < checks below
    cond_fn = jnp.abs(cur - f_prev) / cur
    cond_zeta = jnp.abs(Zeta[:, 1:] - z_prev) / Zeta[:, 1:]
    cond_phi = 1 - _mac_elementwise(Phi[:, 1:], phi_prev)
    stable = (cond_fn < err_fn) & (cond_zeta < err_zeta) & (cond_phi < err_phi)

    # column c is scored by stable[:, c-1]
    lo = max(int(ordmin), 1)
    stab_label = jnp.zeros(Fn.shape, dtype=int)
    return stab_label.at[:, lo:].set(stable[:, lo - 1 :].astype(int))
