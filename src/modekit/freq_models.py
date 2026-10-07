"""Frequency-domain analysis of test data.

Two independent model classes, one per test setup:

- :class:`EmaModel` — input-output data (measured excitation): FRF,
  coherence, reciprocity, input/output PSDs.
- :class:`OmaModel` — output-only data (ambient excitation): output
  spectra for identification, transmissibility, output PSDs,
  autocorrelations.

Both cache their spectra once at init and are ``eqx.Module`` pytrees, so
they batch over independent tests with ``jax.vmap``, e.g.
``eqx.filter_jit(jax.vmap(lambda X, Y: EmaModel(X, Y, fs).get_frf()))(Xb, Yb)``.
"""

import math
import warnings
from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx


def _as_3d(name, A):
    A = jnp.asarray(A)
    if A.ndim != 3:
        raise ValueError(f"{name} must be 3D (n_reps, channels, nt); got {A.shape}")
    return A


def _csd_matrix(X, Y, fs, csd_kwargs):
    """Cross-spectral density matrix, averaged over repetitions.

    Parameters
    ----------
    X : jax.Array  shape (n_reps, P, nt)
    Y : jax.Array  shape (n_reps, Q, nt)
    fs : float
    csd_kwargs : dict  forwarded to jax.scipy.signal.csd

    Returns
    -------
    Freqs : (N,)
    S_yx : (Q, P, N)  S_yx[q, p, f] = mean_reps( csd(X_p, Y_q) )
    """

    def pair(X_p, Y_q):
        freqs, s_yx = jax.scipy.signal.csd(X_p, Y_q, fs, **csd_kwargs)
        return freqs, s_yx

    vmapped = jax.vmap(
        jax.vmap(
            jax.vmap(pair, in_axes=(0, None)),  # over P channels
            in_axes=(None, 0),  # over Q channels
        ),
        in_axes=(0, 0),  # over reps
    )

    Freqs, S_yx = vmapped(X, Y)  # (n_reps, Q, P, N)
    return Freqs[0, 0, 0], S_yx.mean(axis=0)


def _correlation_matrix(Y, n_lags):
    """Biased positive-lag correlation matrix, averaged over repetitions.

    Computed via zero-padded FFT (circular correlation made linear).

    Parameters
    ----------
    Y : jax.Array  shape (n_reps, Q, nt)
    n_lags : int  number of positive lags kept, << nt

    Returns
    -------
    R : (Q, Q, n_lags + 1)  R[q, r, i] at lag i samples
    """
    nt = Y.shape[-1]
    F = jnp.fft.rfft(Y, n=nt + n_lags, axis=-1)  # (n_reps, Q, N)
    C = F[:, :, None, :] * jnp.conj(F[:, None, :, :])  # (n_reps, Q, Q, N)
    R = jnp.fft.irfft(C, n=nt + n_lags, axis=-1)[..., : n_lags + 1] / nt
    return R.mean(axis=0)  # (Q, Q, n_lags + 1)


def _half_spectra_matrix(Y, fs, n_lags, nfft=None):
    """Exponential-windowed half-spectra S+ from positive-lag correlations.
    (Peeters & Van der Auweraer 2005, eqs. 8-10).

    Returns
    -------
    freqs : (nfft // 2 + 1,)
    S : (Q, Q, nfft // 2 + 1)  complex half-spectra.
    """
    nfft = 2 * n_lags if nfft is None else nfft
    R = _correlation_matrix(Y, n_lags)

    # exponential window reaching 1% at the last kept lag
    tau = -n_lags / jnp.log(0.01)  # samples
    w = jnp.exp(-jnp.arange(n_lags + 1) / tau)
    c = (R * w).at[..., 0].multiply(0.5)  # w0 R0 / 2

    S = jnp.fft.rfft(c, n=nfft, axis=-1)  # (Q, Q, nfft // 2 + 1)
    return jnp.fft.rfftfreq(nfft, 1.0 / fs), S


def _real_diag(M):
    """Real diagonal of a (Q, Q, N) matrix stacked along the last axis, as (Q, N)."""
    return jax.vmap(lambda m: jnp.real(jnp.diag(m)), in_axes=-1, out_axes=-1)(M)


class EmaModel(eqx.Module):
    """Input-output (EMA) frequency-domain analysis: P inputs, Q outputs."""

    X: jax.Array
    Y: jax.Array
    fs: float
    exc_type: str = eqx.field(static=True)
    # Cached CSDs computed once at init, reused by all estimators.
    freqs: jax.Array  # N
    S_xx: jax.Array  # PPN
    S_yy: jax.Array  # QQN
    S_yx: jax.Array  # QPN

    def __init__(self, X, Y, fs, exc_type="transient", fft_args=None):
        """
        Parameters
        ----------
        X : array-like  shape (n_reps, P, nt)  input signals
        Y : array-like  shape (n_reps, Q, nt)  output signals
        fs : float  sample rate [Hz]
        exc_type : {'transient', 'continuous'}
        fft_args : dict, forwarded to jax.scipy.signal.csd.
        """
        self.X = _as_3d("X", X)
        self.Y = _as_3d("Y", Y)
        self.fs = fs
        self.exc_type = exc_type

        nt = self.Y.shape[-1]
        if exc_type == "transient":
            nfft = None
            if fft_args is not None:
                nfft = fft_args.get("nfft")
                ignored = [k for k in fft_args if k != "nfft"]
                if ignored:
                    warnings.warn(
                        f"For exc_type='transient', only 'nfft' is accepted in "
                        f"fft_args; ignoring: {ignored}.",
                        UserWarning,
                        stacklevel=2,
                    )
            # Signals are pre-windowed by the caller; boxcar = no extra window.
            csd_kwargs = {
                "window": "boxcar",
                "nperseg": nt,
                "noverlap": 0,
                "scaling": "spectrum",
                "detrend": False,
                "return_onesided": True,
            }
            if nfft is not None:
                csd_kwargs["nfft"] = nfft
        else:
            csd_kwargs = fft_args or {}

        self.freqs, self.S_yy = _csd_matrix(self.Y, self.Y, self.fs, csd_kwargs)
        _, self.S_yx = _csd_matrix(self.X, self.Y, self.fs, csd_kwargs)
        _, self.S_xx = _csd_matrix(self.X, self.X, self.fs, csd_kwargs)

    def get_frf(self, method="H1", rcond=1e-15, kappa=1.0):
        """
        Estimate the frequency response function.

        H1/H2 are true MIMO (full matrix inversion over P inputs). Hv is element-wise
        (valid for independent SIMO tests, not correlated simultaneous inputs).

        Parameters
        ----------
        method : {'H1', 'H2', 'Hv'}
        rcond  : float, regularisation threshold for pinv (H1/H2)
        kappa  : float or (N,) array
            Ratio of output-to-input noise spectra (Hv only).
            kappa=1 → H_T (total least-squares, equal noise power).
            kappa→0 → H2; kappa→∞ → H1.
        """
        if method == "H1":
            inv_S_xx = jax.vmap(
                lambda s: jnp.linalg.pinv(s, rcond), in_axes=-1, out_axes=-1
            )(
                self.S_xx
            )  # with out_axes=-1 we get PPN and not NPP (since vmap defaults to stacking the results in the leading axis)
            H = jax.vmap(jnp.matmul, in_axes=(-1, -1), out_axes=-1)(
                self.S_yx, inv_S_xx
            )  # QPN

        elif method == "H2":
            S_xy = jnp.conj(jnp.swapaxes(self.S_yx, 0, 1))  # PQN
            inv_S_xy = jax.vmap(
                lambda s: jnp.linalg.pinv(s, rcond), in_axes=-1, out_axes=-1
            )(
                S_xy
            )  # QPN
            H = jax.vmap(jnp.matmul, in_axes=(-1, -1), out_axes=-1)(
                self.S_yy, inv_S_xy
            )  # QPN

        elif method == "Hv":  # for SISO/SIMO only, Kihong & Hammond p. 293, eq. 9.67

            def hv_single_freq(s_yx_n, s_xx_n, s_yy_n, kappa_n):
                # s_yx_n: QP  s_xx_n: PP  s_yy_n: QQ  kappa_n: scalar
                sxx = jnp.real(jnp.diag(s_xx_n))  # P
                syy = jnp.real(jnp.diag(s_yy_n))  # Q
                disc = jnp.sqrt(
                    (kappa_n * sxx[None, :] - syy[:, None]) ** 2
                    + 4 * kappa_n * jnp.abs(s_yx_n) ** 2
                )
                return (syy[:, None] - kappa_n * sxx[None, :] + disc) / (
                    2 * s_yx_n
                )  # QP

            nf = self.S_yx.shape[-1]
            kappa_arr = jnp.broadcast_to(jnp.asarray(kappa), (nf,))
            H = jax.vmap(hv_single_freq, in_axes=(-1, -1, -1, 0), out_axes=-1)(
                self.S_yx, self.S_xx, self.S_yy, kappa_arr
            )  # QPN

        else:
            raise ValueError(f"Unknown method {method!r}. Choose 'H1', 'H2', or 'Hv'.")

        return self.freqs, H

    def get_coherence(self, method="ordinary", rcond=1e-15):
        """
        Estimate coherence.

        Parameters
        ----------
        method : {'ordinary', 'multiple'}
            'ordinary' : element-wise γ²[q,p,f] = |S_yx|² / (S_xx[p,p] · S_yy[q,q]),
                         shape (Q, P, N).
                         Warning: in correlated MISO/MIMO tests, pairwise ordinary coherence can be biased.
                         Correlation between inputs leaks into each pair estimate, inflating γ²
                         for inputs that contribute little.
                         Use 'multiple' for an overall data-quality assessment in that case.
            'multiple' : MISO/MIMO γ²_m[q,f] = S_yx[q,:] @ S_xx⁻¹ @ S_yx[q,:]ᴴ / S_yy[q,q],
                         shape (Q, N). Accounts for all inputs jointly; unbiased under
                         correlated excitation.
        rcond : float, regularisation threshold for pinv (multiple only).
        """
        sxx_diag = _real_diag(self.S_xx)  # PN
        syy_diag = _real_diag(self.S_yy)  # QN

        if method == "ordinary":
            gamma = jnp.abs(self.S_yx) ** 2 / (
                sxx_diag[None, :, :] * syy_diag[:, None, :]
            )  # QPN

        elif method == "multiple":
            S_xy = jnp.conj(jnp.swapaxes(self.S_yx, 0, 1))  # PQN
            inv_S_xx = jax.vmap(
                lambda s: jnp.linalg.pinv(s, rcond), in_axes=-1, out_axes=-1
            )(
                self.S_xx
            )  # PPN

            def _mgamma_single_freq(s_yx_n, inv_s_xx_n, s_xy_n, syy_diag_n):
                # s_yx_n: QP  inv_s_xx_n: PP  s_xy_n: PQ  syy_diag_n: Q
                temp = s_yx_n @ inv_s_xx_n  # QP
                num = jnp.real(jnp.sum(temp * s_xy_n.T, axis=-1))  # Q
                return num / syy_diag_n

            gamma = jax.vmap(
                _mgamma_single_freq, in_axes=(-1, -1, -1, -1), out_axes=-1
            )(
                self.S_yx, inv_S_xx, S_xy, syy_diag
            )  # QN

        else:
            raise ValueError(
                f"Unknown method {method!r}. Choose 'ordinary' or 'multiple'."
            )

        return self.freqs, gamma

    def get_psd(self):
        """Return power spectra of inputs and outputs.

        Returns
        -------
        freqs : (N,)
        S_xx_diag : (P, N)  PSD of each input channel
        S_yy_diag : (Q, N)  PSD of each output channel
        """
        return self.freqs, _real_diag(self.S_xx), _real_diag(self.S_yy)

    def check_reciprocity(self, idx_qp, idx_pq, method="H1", rcond=1e-15, kappa=1.0):
        """Extract a reciprocal FRF pair for visual comparison.
        Both indices are drawn from the same test (MIMO);
        for reciprocity across separate SISO/SIMO tests use a standalone function.

        By Maxwell-Betti reciprocity, H[q, p] == H[p, q] for a linear structure.
        The caller identifies the two index pairs; this method extracts and returns
        them for plotting.

        Parameters
        ----------
        idx_qp : (int, int)  (output, input) index of the first FRF
        idx_pq : (int, int)  (output, input) index of the reciprocal FRF
        method, rcond, kappa : forwarded to get_frf

        Returns
        -------
        freqs : (N,)
        h_qp  : (N,)  complex FRF at idx_qp
        h_pq  : (N,)  complex FRF at idx_pq
        """
        freqs, H = self.get_frf(method=method, rcond=rcond, kappa=kappa)
        h_qp = H[idx_qp[0], idx_qp[1], :]
        h_pq = H[idx_pq[0], idx_pq[1], :]
        return freqs, h_qp, h_pq


class OmaModel(eqx.Module):
    """Output-only (OMA) frequency-domain analysis: Q outputs, no inputs."""

    Y: jax.Array
    fs: float
    estimator: str = eqx.field(static=True)
    n_lags: Optional[int] = eqx.field(static=True)  # correlogram only
    freqs: jax.Array  # N
    S_yy: jax.Array  # QQN; half-spectra S+ for estimator='cor'

    def __init__(self, Y, fs, fft_args=None, estimator="per"):
        """
        Parameters
        ----------
        Y : array-like  shape (n_reps, Q, nt)  output signals
        fs : float  sample rate [Hz]
        fft_args : dict
            'per': forwarded to jax.scipy.signal.csd.
            'cor': ``{"n_lags": int, "nfft": int (optional,
            default 2 * n_lags)}``.
        estimator : {'per', 'cor'}
            How the cached spectra are estimated. 'per' gives full Welch
            (periodogram) spectra. 'cor' gives exponential-windowed
            correlogram half-spectra S+; each mode then appears without
            its mirror, at a known extra damping.
        """
        self.Y = _as_3d("Y", Y)
        self.fs = fs
        self.estimator = estimator

        if estimator == "per":
            self.n_lags = None
            self.freqs, self.S_yy = _csd_matrix(self.Y, self.Y, self.fs, fft_args or {})
        elif estimator == "cor":
            args = fft_args or {}
            if "n_lags" not in args:
                raise ValueError(
                    "estimator='cor' needs fft_args={'n_lags': ...} "
                    "(number of positive correlation lags kept, << nt)."
                )
            self.n_lags = int(args["n_lags"])
            self.freqs, self.S_yy = _half_spectra_matrix(
                self.Y, self.fs, self.n_lags, args.get("nfft")
            )
        else:
            raise ValueError(f"Unknown estimator {estimator!r}. Choose 'per' or 'cor'.")

    @property
    def window_rate(self):
        """Decay rate [1/s] of the 1% exponential window behind the ``'cor'``
        half-spectra: the artificial damping every pole carries.

        Pass it as ``alpha`` to :class:`~modekit.algorithms.pLSCF` /
        :class:`~modekit.algorithms.LSFD` with ``spectrum="sd_cor"``.
        """
        if self.n_lags is None:
            raise ValueError(
                "window_rate is defined by the 'cor' exponential window; "
                "this model uses estimator='per'."
            )
        return math.log(100.0) * self.fs / self.n_lags

    def _syy_full(self):
        """Full output spectra: S_yy = S+ + S+^H for a correlogram model
        (Peeters & Van der Auweraer 2005, eq. 11); the cached S_yy otherwise."""
        if self.estimator == "cor":
            return self.S_yy + jnp.conj(jnp.swapaxes(self.S_yy, 0, 1))
        return self.S_yy

    def get_output_spectra(self, ref_dofs=None):
        """Cached output spectra against reference channels, for OMA identification.

        The result feeds :meth:`modekit.algorithms.pLSCF.fit`,
        with the reference outputs in the role the inputs play in EMA.
        Which pLSCF ``spectrum`` flag applies follows the estimator chosen at
        construction:

        - ``estimator='per'`` → full spectra → ``spectrum="sd_per"``
          (each mode appears twice, pole + mirror: double the model order).
        - ``estimator='cor'`` → half-spectra → ``spectrum="sd_cor"``
          with ``alpha=model.window_rate`` (pLSCF's correction removes
          exactly the artificial damping of the 1% exponential window
          applied here).

        Parameters
        ----------
        ref_dofs : sequence of int, optional
            Output channels used as references. None keeps all outputs.

        Returns
        -------
        freqs : (N,)
        S : (Q, R, N)
        """
        if ref_dofs is None:
            return self.freqs, self.S_yy
        return self.freqs, self.S_yy[:, jnp.asarray(ref_dofs), :]

    def get_transmissibility(self, ref_dofs=None, rcond=1e-15):
        """
        Estimate transmissibility functions from the output spectra.

        Parameters
        ----------
        ref_dofs : None or sequence of int
            Indices of the reference output DOFs (denominators).
            None  → Single-reference H1 element-wise estimator:
                    Tr[o, ref, f] = S_yy[o, ref, f] / S_yy[ref, ref, f],
                    shape (Q, Q, N). Valid if the excitation is physically a single source.
                    If there are multiple sources, this estimator is biased everywhere except
                    at the system poles.
            tuple → Poly-reference matrix estimator:
                    Tr = S_y_all_yref @ inv(S_yref_yref),
                    shape (Q, R, N).
        rcond : float
            Regularisation threshold for pinv.

        Returns
        -------
        freqs : (N,)
        Tr    : (Q, Q, N) if ref_dofs is None, else (Q, R, N)

        Notes
        -----
        Single-reference transmissibility is a pure structural property because
        the single input force cancels out (Tr_ij = H_i / H_j).

        In Poly-reference testing, this breaks down (the "Transmissibility Paradox"). Because the
        responses are a linear combination of multiple inputs (e.g., Y_i = H_i1·F_1 + H_i2·F_2),
        the forces no longer cancel. The transmissibility matrix becomes dependent
        on the spatial distribution of the excitation [1].

        Even so, at system resonances the mode shape (φ) dominates and transmissibility
        converges to φ_i / φ_j. Curves measured under different loading conditions
        therefore intersect exactly at the natural frequencies. Thus, one can use the single-reference formula
        to build Tr for load A, and then again for load B and find the intersections. The postprocessing of the
        poly-reference approach downstream needs more digging :)

        Literature:
            [1] Devriendt and Guillaume: The use of transmissibility measurements in
                output-only modal analysis, 2006.
        """
        S_yy = self._syy_full()
        if ref_dofs is None:
            # divide every column of S_yy by its diagonal (auto-spectrum of that ref).
            syy_diag = _real_diag(S_yy)  # QN
            Tr = S_yy / syy_diag[None, :, :]  # QQN
        else:
            ref_arr = jnp.array(ref_dofs)
            S_cross = S_yy[:, ref_arr, :]  # QRN
            S_ref = S_cross[ref_arr, :, :]  # RRN
            inv_S_ref = jax.vmap(
                lambda s: jnp.linalg.pinv(s, rcond), in_axes=-1, out_axes=-1
            )(
                S_ref
            )  # RRN
            Tr = jax.vmap(jnp.matmul, in_axes=(-1, -1), out_axes=-1)(
                S_cross, inv_S_ref
            )  # QRN

        return self.freqs, Tr

    def get_psd(self):
        """Return power spectra of the outputs.

        Returns
        -------
        freqs : (N,)
        S_yy_diag : (Q, N)  PSD of each output channel
        """
        return self.freqs, _real_diag(self._syy_full())

    def get_ac(self, n_lags=None, normalize=False):
        """Return positive-lag autocorrelation functions of the outputs.

        Reuses the biased correlation estimator behind the correlogram
        half-spectra (:func:`_correlation_matrix`), so for
        ``estimator='cor'`` these are exactly the (unwindowed)
        correlations the identification spectra are built from.

        Parameters
        ----------
        n_lags : int, optional
            Number of positive lags kept, << nt. Defaults to the
            model's ``n_lags`` (set with ``estimator='cor'``); required
            for ``estimator='per'``, which caches no correlations.
        normalize : bool, optional
            Scale each channel by its zero-lag value, so R(0) = 1.

        Returns
        -------
        lags : (n_lags + 1,)  lag times [s]
        R : (Q, n_lags + 1)  autocorrelation of each output channel
        """
        if n_lags is None:
            n_lags = self.n_lags
        if n_lags is None:
            raise ValueError(
                "estimator='per' caches no correlations; pass n_lags "
                "(number of positive lags kept, << nt)."
            )
        n_lags = int(n_lags)
        R = _real_diag(_correlation_matrix(self.Y, n_lags))  # (Q, n_lags + 1)
        if normalize:
            R = R / R[:, :1]
        return jnp.arange(n_lags + 1) / self.fs, R


# --- reciprocity ---
def check_reciprocity_cross(
    model_qp, idx_qp, model_pq, idx_pq, method="H1", rcond=1e-15, kappa=1.0
):
    """Check reciprocity across two separate EmaModel instances.

    Use this when H_qp and H_pq come from different tests (e.g., two independent
    SISO hammer runs with swapped excitation/response DOFs).

    Parameters
    ----------
    model_qp : EmaModel  model containing H_qp
    idx_qp   : (int, int)          (output, input) index into model_qp
    model_pq : EmaModel  model containing H_pq
    idx_pq   : (int, int)          (output, input) index into model_pq
    method, rcond, kappa : forwarded to get_frf

    Returns
    -------
    freqs : (N,)  frequency axis from model_qp
    h_qp  : (N,)  complex FRF from model_qp at idx_qp
    h_pq  : (N,)  complex FRF from model_pq at idx_pq
    """
    if model_qp.fs != model_pq.fs:
        warnings.warn(
            f"Sample rates differ ({model_qp.fs} vs {model_pq.fs}); "
            "reciprocity comparison may be invalid.",
            UserWarning,
            stacklevel=2,
        )
    freqs, H_qp = model_qp.get_frf(method=method, rcond=rcond, kappa=kappa)
    _, H_pq = model_pq.get_frf(method=method, rcond=rcond, kappa=kappa)
    return freqs, H_qp[idx_qp[0], idx_qp[1], :], H_pq[idx_pq[0], idx_pq[1], :]
