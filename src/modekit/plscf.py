"""
Math for poly-reference Least-Squares Complex Frequency (pLSCF / "PolyMAX").

Shape notation follows : ``Q`` outputs, ``P`` references, ``N``
frequency lines, ``K`` poles per order, ``p`` model orders, ``M`` requested modes.

"""

import warnings
from functools import partial
from typing import Optional, Tuple, List, Union
import jax
import jax.numpy as jnp
from tqdm import trange
import numpy as np

from modekit.criteria import _mac_elementwise


def _pad_orders(seqs: List[jax.Array]) -> jax.Array:
    """Stack ragged per-order arrays along a new order axis (axis 1), NaN-padded."""
    max_len = max((s.shape[0] for s in seqs), default=0)
    cols = [
        jnp.concatenate(
            [s, jnp.full((max_len - s.shape[0], *s.shape[1:]), jnp.nan, dtype=s.dtype)]
        )
        for s in seqs
    ]
    return jnp.stack(cols, axis=1)


def fit(
    data: jax.Array,
    freqs: jax.Array,
    dt: float,
    p: int,
    progress: bool = False,
    weight: Optional[jax.Array] = None,
) -> Tuple[dict[int, jax.Array], dict[int, jax.Array]]:
    """
    Fit the right matrix fraction description to the data for orders 1...p.

    Pure and traceable: safe under ``jax.jit`` and ``jax.vmap``.

    Parameters
    ----------
    data : jax.Array
        FRF or output-spectra matrix, shape (Q, P, N)
    freqs : jax.Array
        Frequency vector [Hz]
    dt : float
        Sampling interval.
    p : int
        Maximum model order.
    progress : bool, optional
        Show a per-order progress bar. Leave False under ``jit``/``vmap``, where
        the loop is unrolled at trace time and the bar is meaningless.
    weight : jax.Array, optional
        Least-squares weight per frequency line, shape ``(N,)`` or ``(Q, N)``.
        A zero drops that line from the fit, so a fixed-length window can be
        narrowed under ``vmap`` without changing shapes. Default: all ones.

    Returns
    -------
    tuple of list of jax.Array
        - Ad : denominator polynomial coefficients per order.
        - Bn : numerator polynomial coefficients per order.
    """
    Q, P, N = data.shape

    weight = (
        jnp.ones((Q, N))
        if weight is None
        else jnp.broadcast_to(jnp.asarray(weight, dtype=float), (Q, N))
    )

    omegas = 2 * jnp.pi * freqs  # (N,)
    Omegas = jnp.exp(1j * omegas[:, None] * dt) ** jnp.arange(0, p + 1)  # (N, p+1)

    X = weight[:, :, None] * Omegas[None, :, :]  # (Q, N, p+1)
    Y = (-data.mT[:, :, None, :] * X[:, :, :, None]).reshape(
        Q, N, -1
    )  # (Q, N, P*(p+1))
    Xh = X.conj().mT  # (Q, p+1,    N)
    Yh = Y.conj().mT  # (Q, P*(p+1), N)
    Ro = (Xh @ X).real  # (Q, p+1, p+1)
    So = (Xh @ Y).real  # (Q, p+1, P*(p+1))
    To = (Yh @ Y).real  # (Q, P*(p+1), P*(p+1))

    Ad = {}
    Bn = {}
    orders = trange(1, p + 1) if progress else range(1, p + 1)
    for r in orders:
        Rr = Ro[:, : r + 1, : r + 1]  # (Q, r+1, r+1)
        Sr = So[:, : r + 1, : P * (r + 1)]  # (Q, r+1, P*(r+1))
        Tr = To[:, : P * (r + 1), : P * (r + 1)]  # (Q, P*(r+1), P*(r+1))
        Mr = 2 * (Tr - Sr.mT @ jnp.linalg.solve(Rr, Sr)).sum(0)  # (P*(r+1), P*(r+1))

        # highest-order coefficient constrained to identity
        alpha = jnp.concatenate(
            [
                jnp.linalg.solve(-Mr[: r * P, : r * P], Mr[: r * P, r * P :]),
                jnp.eye(P),
            ]
        )  # (P*(r+1), P)

        Ad[r] = alpha.reshape(-1, P, P)  # (r+1, P, P)
        beta = jnp.linalg.solve(-Rr, Sr @ alpha)  # (Q, r+1, P)
        Bn[r] = beta.swapaxes(0, 1)  # (r+1, Q, P)

    return Ad, Bn


def get_poles(
    Ad: dict[int, jax.Array],
    Bn: dict[int, jax.Array],
    dt: float,
    spectrum: str = "frf_tap",
    alpha: Optional[float] = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Extract mode shapes, complex poles and reference factors for every order.

    Parameters
    ----------
    Ad, Bn : dict of {int: jax.Array}
        Denominator / numerator polynomial coefficients from :func:`fit`,
        keyed by model order.
    dt : float
        Sampling interval.
    spectrum : str, optional
        ``"frf_tap"``/``"frf_shaker"`` (EMA) or ``"sd_cor"``/``"sd_per"`` (OMA).
    alpha : float, optional
        Exponential-window decay rate [1/s], used by the ``"frf_tap"`` and
        ``"sd_cor"`` corrections (see :func:`window_shift`).

    Returns
    -------
    tuple of jax.Array
        Phis (mode shapes, ``(K, p, Q)``), Lambds (complex poles, ``(K, p)``) and
        Lrs (participation / operational reference factors, ``(K, p, P)``).
        Columns are ordered by model order and NaN-padded to the largest order's
        pole count. Frequencies and damping ratios follow from
        :func:`lambd_to_fn` / :func:`lambd_to_zeta`.
    """
    Phis = []
    Lambds = []
    Lrs = []
    for r in sorted(Ad):  # iterate by model order
        A, C = rmfd_to_ss(Ad[r], Bn[r])

        phi, lambd, lr = ss_to_modal_params(A, C, dt, Ad[r].shape[1], spectrum, alpha)
        Phis.append(phi)
        Lambds.append(lambd)
        Lrs.append(lr)
    # Pad the ragged per-order results into dense, order-indexed arrays.
    return _pad_orders(Phis), _pad_orders(Lambds), _pad_orders(Lrs)


@jax.jit
def rmfd_to_ss(Ad_r: jax.Array, Bn_r: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Build the companion-form state-space ``(A, C)`` from one RMFD order.

    Parameters
    ----------
    Ad_r : jax.Array
        Denominator coefficients, shape ``(r, P, P)``.
    Bn_r : jax.Array
        Numerator coefficients, shape ``(r, Q, P)``.

    Returns
    -------
    tuple of jax.Array
        - A : companion state matrix, shape ``(r*P, r*P)``.
        - C : output matrix, shape ``(Q, r*P)``.
    """
    r, Q, P = Bn_r.shape
    Ad_last = Ad_r[-1]  # (P, P)
    Bn_last = Bn_r[-1]  # (Q, P)
    Ad_rev = Ad_r[:-1][::-1]  # (r-1, P, P)
    Bn_rev = Bn_r[:-1][::-1]  # (r-1, Q, P)
    prod = jnp.linalg.solve(Ad_last, Ad_rev)  # (r-1, P, P)

    A_top = (-prod).swapaxes(0, 1).reshape(P, (r - 1) * P)  # (P, (r-1)*P)
    A = jnp.zeros((r * P, r * P)).at[P:, :-P].set(jnp.eye((r - 1) * P))
    A = A.at[:P, : (r - 1) * P].set(A_top)

    C_coefs = Bn_rev - Bn_last[None] @ prod  # (r-1, Q, P)
    C_blocks = C_coefs.swapaxes(0, 1).reshape(Q, (r - 1) * P)  # (Q, (r-1)*P)
    C = jnp.zeros((Q, r * P)).at[:, : (r - 1) * P].set(C_blocks)
    return A, C


def rmfd_response(
    Ad_r: jax.Array, Bn_r: jax.Array, freqs: jax.Array, dt: float
) -> jax.Array:
    """
    Synthesise the response ``B(z) A(z)^-1`` of one RMFD order on ``freqs``.

    The curve :func:`fit` matched to the data at that order, for goodness-of-fit
    checks; ``z = exp(2j pi f dt)`` as in :func:`fit`.

    Parameters
    ----------
    Ad_r, Bn_r : jax.Array
        Denominator ``(r+1, P, P)`` and numerator ``(r+1, Q, P)`` coefficients
        of one order from :func:`fit`.
    freqs : jax.Array
        Frequency vector [Hz], shape ``(N,)``.
    dt : float
        Sampling interval the fit used.

    Returns
    -------
    jax.Array
        Modelled FRF / spectra, shape ``(Q, P, N)``.
    """
    z = jnp.exp(2j * jnp.pi * jnp.asarray(freqs) * dt)  # (N,)
    zk = z[:, None] ** jnp.arange(Ad_r.shape[0])  # (N, r+1)
    A = jnp.einsum("nk,kij->nij", zk, Ad_r)  # (N, P, P)
    B = jnp.einsum("nk,kqp->nqp", zk, Bn_r)  # (N, Q, P)
    resp = jnp.linalg.solve(A.mT, B.mT).mT  # B A^-1 via A^-T B^T, (N, Q, P)
    return jnp.moveaxis(resp, 0, -1)


def lambd_to_fn(Lambd: jax.Array) -> jax.Array:
    """Natural frequencies [Hz] from complex poles; inf (zero poles) -> NaN."""
    fn = jnp.abs(Lambd) / (2 * jnp.pi)
    return jnp.where(jnp.isinf(fn), jnp.nan, fn)


def lambd_to_zeta(Lambd: jax.Array) -> jax.Array:
    """Damping ratios from complex poles."""
    return -jnp.real(Lambd) / jnp.abs(Lambd)


def fn_zeta_to_lambd(Fn: jax.Array, Zeta: jax.Array) -> jax.Array:
    """Complex poles from natural frequencies [Hz] and damping ratios."""
    wn = 2 * jnp.pi * jnp.asarray(Fn)
    zeta = jnp.asarray(Zeta)
    return wn * (-zeta + 1j * jnp.sqrt(1 - zeta**2))


def complex_to_real_mode(Phi: jax.Array) -> jax.Array:
    """
    Real (normal) mode shapes most correlated with complex ones.

    Ahmadian, Gladwell & Ismail, "Extracting Real Modes from Complex Measured
    Modes".

    Parameters
    ----------
    Phi : jax.Array, shape (Q,) or (Q, M)
        Complex mode shape, or modal matrix with the ``M`` modes as columns.
        All-NaN columns (poles that sat out the fit) pass through as NaN.

    Returns
    -------
    jax.Array
        Real mode shape(s), same shape as ``Phi``. Kept at the input's
        amplitude, so a scaled complex mode gives an equally scaled real one
        (scale does not affect MAC). The largest-magnitude entry is made
        positive to fix the sign.
    """
    Phi = jnp.asarray(Phi)
    single = Phi.ndim == 1
    if single:
        Phi = Phi[:, None]

    def one(phi):  # phi : (Q,) complex, one mode
        ok = jnp.all(jnp.isfinite(phi))
        phi = jnp.where(ok, phi, 0.0)  # keep eigh clear of NaN
        U = jnp.outer(phi.real, phi.real) + jnp.outer(phi.imag, phi.imag)
        vals, vecs = jnp.linalg.eigh(U)  # ascending eigenvalues
        # leading eigenpair: unit direction scaled by the paper's optimal real-
        # part norm sqrt(lambda_max), so the amplitude tracks the complex input.
        r = vecs[:, -1] * jnp.sqrt(jnp.maximum(vals[-1], 0.0))
        r = r * jnp.sign(r[jnp.argmax(jnp.abs(r))])  # deterministic sign
        return jnp.where(ok, r, jnp.nan)

    R = jax.vmap(one, in_axes=1, out_axes=1)(Phi)
    return R[:, 0] if single else R


def window_shift(spectrum: str, alpha: Optional[float] = None) -> float:
    """
    Damping [rad/s] added by an exponential window applied before the fit.

    Poles fitted to windowed data sit ``shift`` further into the left half
    plane: add it to recover the physical poles, subtract it to describe the
    windowed data again (as :class:`~modekit.algorithms.LSFD` does).

    Parameters
    ----------
    spectrum : str
        ``"sd_cor"`` (correlogram window) and ``"frf_tap"`` (tap-test
        window) carry a window; anything else gives no shift.
    alpha : float, optional
        Window decay rate [1/s]:
        ``windows.Exponential.alpha`` (tap test) or
        ``OmaModel.window_rate`` (correlogram). Required for ``"sd_cor"``,
        whose window is always applied; optional for ``"frf_tap"``, where
        ``None`` means an unwindowed tap test.
    """
    if spectrum not in ("sd_cor", "frf_tap"):
        return 0.0
    if alpha is not None:
        return alpha
    if spectrum == "sd_cor":  # the 'cor' half-spectra are always windowed
        raise ValueError(
            "spectrum='sd_cor' requires the window rate alpha [1/s] "
            "(OmaModel.window_rate)."
        )
    return 0.0  # unwindowed tap test


@partial(jax.jit, static_argnames=("n_ref", "spectrum", "alpha"))
def ss_to_modal_params(
    A: jax.Array,
    C: jax.Array,
    dt: float,
    n_ref: int,
    spectrum: str = "frf_tap",
    alpha: Optional[float] = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Convert state-space ``(A, C)`` to mode shapes, poles and reference factors.

    Parameters
    ----------
    A, C : jax.Array
        State / output matrices.
    dt : float
        Sampling interval.
    n_ref : int
        Number of references ``P``, i.e. the block size of the companion form.
    spectrum : str, optional
        Selects the window correction: ``"sd_cor"`` and ``"frf_tap"`` use
        ``alpha``; ``"frf_shaker"``/``"sd_per"`` apply none.
    alpha : float, optional
        Exponential-window decay rate [1/s] (see :func:`window_shift`);
        required for ``"sd_cor"``, optional for ``"frf_tap"``.

    Returns
    -------
    tuple of jax.Array
        phi (complex mode shapes, ``(K, Q)``), lambd (complex poles, ``(K,)``)
        and lr (participation / operational reference factors, ``(K, P)``).
        Frequencies and damping ratios follow from :func:`lambd_to_fn` /
        :func:`lambd_to_zeta`.
    """
    # Undo the artificial damping added by a pre-processing exponential window.
    shift = window_shift(spectrum, alpha)

    lambd_z, psi = jnp.linalg.eig(A)
    # log(0) and 0/0 are expected for spurious/zero poles; they become nan downstream
    lambd_s = jnp.log(lambd_z) / dt + shift

    # mask unstable poles (Re > 0) to nan, on both poles and eigenvectors
    unstable = jnp.real(lambd_s) > 0
    lambd = jnp.where(unstable, jnp.nan, lambd_s)
    Psi = jnp.where(unstable[None, :], jnp.nan, psi)

    def normalise(V):  # unit max-abs entry per column, then one mode per row
        peak = jnp.argmax(jnp.abs(V), axis=0)
        return (V / jnp.take_along_axis(V, peak[None, :], axis=0)).T

    phi = normalise(C @ Psi)  # complex mode shapes, (K, Q)

    left = jnp.linalg.solve(psi, jnp.eye(A.shape[0])[:, :n_ref])  # (K, P)
    lr = normalise(jnp.where(unstable[:, None], jnp.nan, left).T)  # (K, P)
    return phi, lambd, lr


# -----------------------------------------------------------------------------
# Pole selection. The kernels below are pure and traceable (safe under jit/vmap);
# `mpe` wraps them with the validation, errors and warnings that tracing forbids.
# -----------------------------------------------------------------------------
def _n_resolvable(best: jax.Array, found: jax.Array) -> jax.Array:
    """
    Per-order count of reference frequencies that each match a different pole.

    Parameters
    ----------
    best, found : jax.Array
        Nearest-pole row indices and match flags, both shape ``(M, p)``.

    Returns
    -------
    jax.Array
        Counts, shape ``(p,)``.
    """
    shared = (
        (best[:, None, :] == best[None, :, :]) & found[:, None, :] & found[None, :, :]
    )  # (M, M, p)
    earlier = jnp.tril(jnp.ones(shared.shape[:2], dtype=bool), -1)[:, :, None]
    # a pole matched by several references counts once, at the lowest index
    return (found & ~(shared & earlier).any(axis=1)).sum(axis=0)


def mpe_kernel(
    f_ref: jax.Array,
    Lambd_poles: jax.Array,
    Phi_poles: jax.Array,
    Lr_poles: jax.Array,
    stab_label: Optional[jax.Array] = None,
    cols: Optional[jax.Array] = None,
    deltaf: float = 0.05,
    rtol: float = 5e-2,
    phi_ref: Optional[jax.Array] = None,
    mac_lim: float = 0.85,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Pure, traceable core of :func:`mpe`, safe under ``jax.jit``/``jax.vmap``.
    Reports failures through the ``ok`` mask instead of raising.

    Parameters
    ----------
    f_ref : jax.Array
        Reference frequencies, shape ``(M,)``.
    Lambd_poles, Phi_poles, Lr_poles : jax.Array
        Pole arrays, shapes ``(K, p)``, ``(K, p, Q)``, ``(K, p, P)``.
    stab_label : jax.Array, optional
        Stability labels ``(K, p)``; 1 = stable. None keeps every pole.
    cols : jax.Array, optional
        Column to read per mode, shape ``(M,)``. None selects the lowest column
        that matches the most reference frequencies ('find_min').
    deltaf, rtol : float
        Absolute [Hz] and relative half-width of the search window.
    phi_ref : jax.Array, optional
        Reference mode shapes ``(Q, M)``; enables the MAC gate.
    mac_lim : float
        Minimum MAC against ``phi_ref``.

    Returns
    -------
    tuple of jax.Array
        Lambd ``(M,)``, Phi ``(Q, M)``, Lr ``(P, M)``, cols ``(M,)``, ok ``(M,)``.
    """
    M = f_ref.shape[0]
    Fn_poles = lambd_to_fn(Lambd_poles)  # the search window is in frequency

    keep = (
        jnp.ones(Fn_poles.shape, dtype=bool) if stab_label is None else stab_label == 1
    )
    if phi_ref is None:
        mac_ok = jnp.ones((M, *Fn_poles.shape), dtype=bool)
    else:
        # MAC of every pole mode shape (K, p, Q) against every reference (M, Q)
        mac = _mac_elementwise(Phi_poles[None], phi_ref.T[:, None, None, :])
        mac_ok = mac >= mac_lim  # (M, K, p)

    # nearest valid pole per reference frequency, at every order
    tol = jnp.maximum(deltaf, rtol * jnp.abs(f_ref))  # (M,)
    dist = jnp.abs(Fn_poles[None, :, :] - f_ref[:, None, None])  # (M, K, p)
    # NaN-padded poles compare False, so they are never picked
    valid = (dist <= tol[:, None, None]) & keep[None, :, :] & mac_ok
    dist = jnp.where(valid, dist, jnp.inf)
    best = dist.argmin(axis=1)  # (M, p)
    found = jnp.isfinite(dist.min(axis=1))  # (M, p)

    if cols is None:  # 'find_min'
        n_ok = _n_resolvable(best, found)  # (p,)
        cols = jnp.full(M, jnp.argmax(n_ok == n_ok.max()))

    modes = jnp.arange(M)
    rows = best[modes, cols]  # (M,)
    hit = found[modes, cols]  # (M,)
    Lambd = Lambd_poles[rows, cols]
    Phi = Phi_poles[rows, cols, :].T  # (Q, M)
    Lr = Lr_poles[rows, cols, :].T  # (P, M)

    # a pole matched by two references goes to the closer one; ties to the lower index
    gap = jnp.abs(Fn_poles[rows, cols] - f_ref)
    same = (
        (rows[:, None] == rows[None, :])
        & (cols[:, None] == cols[None, :])
        & hit[:, None]
        & hit[None, :]
    )  # (M, M)
    closer = (gap[None, :] < gap[:, None]) | (
        (gap[None, :] == gap[:, None]) & (modes[None, :] < modes[:, None])
    )
    ok = hit & ~(same & closer).any(axis=1)
    return Lambd, Phi, Lr, cols, ok


def mpe(
    f_ref: np.ndarray,
    Lambd_poles: np.ndarray,
    Phi_poles: np.ndarray,
    Lr_poles: np.ndarray,
    order_in: Union[int, List[int], str] = "find_min",
    stab_label: Optional[np.ndarray] = None,
    deltaf: float = 0.05,
    rtol: float = 5e-2,
    phi_ref: Optional[np.ndarray] = None,
    mac_lim: float = 0.85,
    on_missing: str = "raise",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract modal parameters near ``f_ref`` from the pole arrays.

    Eager wrapper around :func:`mpe_kernel`: validates the arguments and turns
    the kernel's ``ok`` mask into an exception or NaN-filled output. Use
    :func:`mpe_kernel` directly under ``jit``/``vmap``.

    Parameters
    ----------
    f_ref : array_like
        Approximate natural frequencies to extract, length ``M``.
    Lambd_poles, Phi_poles, Lr_poles : array_like
        Pole arrays, shapes ``(K, p)``, ``(K, p, Q)``, ``(K, p, P)``, from
        :func:`get_poles`.
    order_in : int, list of int, or 'find_min'
        Column to read from; a list gives one column per mode; 'find_min' takes
        the lowest column that matches every mode (needs ``stab_label``).
    stab_label : array_like, optional
        Stability labels (1 = stable) from ``sc_apply``; required for 'find_min'.
    deltaf, rtol : float, optional
        Absolute [Hz] and relative half-width of the search window.
        Defaults 0.05 and 5e-2.
    phi_ref : array_like, optional
        Reference mode shapes ``(Q, M)``; enables the MAC gate. Default None.
    mac_lim : float, optional
        Minimum MAC against ``phi_ref``. Default 0.85.
    on_missing : {'raise', 'nan'}, optional
        When no pole is found for a reference frequency: ``'raise'`` (default)
        raises an error; ``'nan'`` warns and returns NaN and ``order_out = -1``
        for that mode.

    Returns
    -------
    tuple of numpy.ndarray
        Lambd ``(M,)``, Phi ``(Q, M)``, Lr ``(P, M)``, order_out ``(M,)``.
    """
    if on_missing not in ("raise", "nan"):
        raise ValueError('on_missing must be "raise" or "nan"')

    Lambd_poles = jnp.asarray(Lambd_poles, dtype=complex)
    Phi_poles = jnp.asarray(Phi_poles, dtype=complex)
    Lr_poles = jnp.asarray(Lr_poles, dtype=complex)
    f_ref = jnp.asarray(f_ref, dtype=float).reshape(-1)
    M = f_ref.shape[0]
    p = Lambd_poles.shape[1]

    if isinstance(order_in, str):
        if order_in != "find_min":
            raise ValueError('order_in must be an int, a list of int, or "find_min"')
        if stab_label is None:
            raise ValueError(
                "When order_in='find_min', one must also provide the stab_label "
                "list for the poles"
            )
        cols = None
    elif isinstance(order_in, (int, np.integer)):
        cols = jnp.full(M, int(order_in))
    elif isinstance(order_in, (list, tuple, np.ndarray, jax.Array)):
        cols = jnp.asarray(order_in, dtype=int).reshape(-1)
        if cols.shape[0] != M:
            raise ValueError(
                f"order_in has {cols.shape[0]} entries but {M} frequencies "
                "were requested"
            )
    else:
        raise ValueError('order_in must be an int, a list of int, or "find_min"')

    if cols is not None and bool(((cols < 0) | (cols >= p)).any()):
        raise ValueError(f"order_in out of range: the pole arrays have {p} columns")

    if phi_ref is not None:
        phi_ref = jnp.asarray(phi_ref, dtype=complex)
        phi_ref = phi_ref[:, None] if phi_ref.ndim == 1 else phi_ref
        if phi_ref.shape != (Phi_poles.shape[2], M):
            raise ValueError(
                f"phi_ref must have shape (Q, M) = "
                f"({Phi_poles.shape[2]}, {M}), got {phi_ref.shape}"
            )

    Lambd, Phi, Lr, cols, ok = mpe_kernel(
        f_ref,
        Lambd_poles,
        Phi_poles,
        Lr_poles,
        stab_label=None if stab_label is None else jnp.asarray(stab_label),
        cols=cols,
        deltaf=deltaf,
        rtol=rtol,
        phi_ref=phi_ref,
        mac_lim=mac_lim,
    )
    Lambd, Phi, Lr = np.asarray(Lambd), np.asarray(Phi), np.asarray(Lr)
    cols, ok = np.asarray(cols), np.asarray(ok)
    f_ref = np.asarray(f_ref)

    if not ok.all():
        detail = (
            f"no matching pole found for {f_ref[~ok].tolist()} Hz "
            "(narrow deltaf/rtol, or pass phi_ref)"
        )
        if on_missing == "raise":
            raise ValueError(
                f"{detail}. Pass on_missing='nan' to return NaN for these instead"
            )
        warnings.warn(f"{detail}; returning NaN", RuntimeWarning, stacklevel=2)
        Lambd = np.where(ok, Lambd, np.nan)
        Phi = np.where(ok[None, :], Phi, np.nan)
        Lr = np.where(ok[None, :], Lr, np.nan)
        cols = np.where(ok, cols, -1)

    return Lambd, Phi, Lr, cols
