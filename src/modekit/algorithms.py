"""
Shape notation follows ``plscf``: ``Q`` outputs, ``P`` references, ``N``
frequency lines, ``K`` poles per order, ``p`` model orders, ``M`` requested modes.
"""

from typing import List, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from modekit import clustering, criteria, plots, plscf

DEFAULT_SC = {
    "err_fn": 0.01,
    "err_zeta": 0.05,
    "err_phi": 0.02,
}  # Avitabile, p.208-9, could also be {0.05, 0.05, 0.05}
DEFAULT_HC = {
    "zeta_max": 0.1,
    "mpc_lim": 0.7,
    "mpd_lim": 0.3,
}  # could also be {0.1, 0.5, 0.5}


class PoleArrays(eqx.Module):
    """
    Screened poles for one segment.
    """

    Lambd: jax.Array  # complex poles, (K, p)
    Phi: jax.Array  # (K, p, Q)
    Lr: jax.Array  # participation / reference factors, (K, p, P)
    stab_label: jax.Array  # (K, p), 1 = stable

    @property
    def Fn(self) -> jax.Array:
        """Natural frequencies [Hz], (K, p)."""
        return plscf.lambd_to_fn(self.Lambd)

    @property
    def Zeta(self) -> jax.Array:
        """Damping ratios, (K, p)."""
        return plscf.lambd_to_zeta(self.Lambd)

    def select(
        self,
        f_ref,
        cols=None,
        deltaf: float = 0.05,
        rtol: float = 5e-2,
        phi_ref=None,
        mac_lim: float = 0.85,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Pure, traceable mode selection — see :func:`plscf.mpe_kernel`.

        Returns ``(Lambd, Phi, Lr, cols, ok)``; ``ok`` is False where no pole
        matched. Batch with ``jax.vmap(lambda p: p.select(f_ref))(poles)``.
        """
        return plscf.mpe_kernel(
            jnp.asarray(f_ref, dtype=float),
            self.Lambd,
            self.Phi,
            self.Lr,
            stab_label=self.stab_label,
            cols=None if cols is None else jnp.asarray(cols, dtype=int),
            deltaf=deltaf,
            rtol=rtol,
            phi_ref=None if phi_ref is None else jnp.asarray(phi_ref, dtype=complex),
            mac_lim=mac_lim,
        )

    def mpe(
        self,
        f_ref,
        order_in: Union[int, List[int], str] = "find_min",
        deltaf: float = 0.05,
        rtol: float = 5e-2,
        phi_ref=None,
        mac_lim: float = 0.85,
        on_missing: str = "raise",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract the modes nearest ``f_ref`` — eager counterpart of :meth:`select`.

        Pass-through to :func:`plscf.mpe`, using the fit's stability labels.

        Parameters
        ----------
        f_ref : array_like
            Approximate natural frequencies to extract, length ``M``.
        order_in : int, list of int, or 'find_min', optional
            Column of the pole arrays to read from. Default 'find_min'.
        deltaf, rtol : float, optional
            Absolute [Hz] and relative half-width of the search window.
        phi_ref : array_like, optional
            Reference mode shapes ``(Q, M)``, enabling the MAC gate.
        mac_lim : float, optional
            Minimum MAC against ``phi_ref``. Default 0.85.
        on_missing : {'raise', 'nan'}, optional
            What to do when no pole is found for a reference frequency.

        Returns
        -------
        tuple of numpy.ndarray
            Lambd ``(M,)``, Phi ``(Q, M)``, Lr ``(P, M)``, order_out ``(M,)``.
            ``Lambd`` and ``Lr`` feed :meth:`LSFD.fit`.
        """
        return plscf.mpe(
            f_ref,
            self.Lambd,
            self.Phi,
            self.Lr,
            order_in=order_in,
            stab_label=self.stab_label,
            deltaf=deltaf,
            rtol=rtol,
            phi_ref=phi_ref,
            mac_lim=mac_lim,
            on_missing=on_missing,
        )


class pLSCF(eqx.Module):
    """
    Poly-reference Least-Squares Complex Frequency (pLSCF / PolyMAX) for EMA/OMA.
    """

    freqs: jax.Array
    fs: float = eqx.field(static=True)
    ordmax: int = eqx.field(static=True)
    ordmin: int = eqx.field(static=True)
    spectrum: str = eqx.field(static=True)
    alpha: Optional[float] = eqx.field(static=True)
    sc: dict
    hc: dict

    def __init__(
        self,
        freqs,
        fs: float,
        ordmax: int,
        ordmin: int = 0,
        spectrum: str = "frf_tap",
        alpha: Optional[float] = None,
        sc: Optional[dict] = None,
        hc: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        freqs : array_like
            Frequency vector [Hz], length ``N``.
        fs : float
            Sampling frequency [Hz] of the acquisition.
        ordmax : int
            Highest model order to fit; orders ``1...ordmax`` are all fitted.
        ordmin : int, optional
            Lowest order the stabilisation chart scores. Default 0.
        spectrum : str, optional
            Exponential-window correction, see :func:`plscf.get_poles`;
            ``frf_*`` for EMA, ``psd_*`` for OMA.
        alpha : float, optional
            Exponential-window decay rate [1/s]
            ``spectrum="frf_tap"``: the tap-test window rate;
            ``spectrum="sd_cor"``: the correlogram
            window rate (``OmaModel.window_rate``), required.
        sc : dict, optional
            Soft criteria — ``err_fn``, ``err_zeta``, ``err_phi``.
        hc : dict, optional
            Hard criteria — ``zeta_max``, and optionally ``mpc_lim`` / ``mpd_lim``
            (set either to None to skip that screen).
        """
        self.freqs = jnp.asarray(freqs)
        self.fs = float(fs)
        self.ordmax = int(ordmax)
        self.ordmin = int(ordmin)
        self.spectrum = spectrum
        self.alpha = None if alpha is None else float(alpha)
        self.sc = dict(DEFAULT_SC if sc is None else sc)
        self.hc = dict(DEFAULT_HC if hc is None else hc)

    @property
    def dt(self) -> float:
        """Sampling interval [s]."""
        return 1.0 / self.fs

    def fit(self, data, progress: bool = False) -> PoleArrays:
        """
        Fit every model order, screen the poles, and label them stable/unstable.

        Pure and traceable. Batch over segments with
        ``jax.vmap(est.fit)(data_stack)``.

        Parameters
        ----------
        data : array_like
            FRF (EMA) or output spectra from ``OmaModel.get_output_spectra``
            (OMA), shape ``(Q, P, N)``; ``P`` counts inputs (EMA) or
            reference outputs (OMA).
        progress : bool, optional
            Per-order progress bar. Leave False under ``jit``/``vmap``.

        Returns
        -------
        PoleArrays
        """
        Ad, Bn = plscf.fit(
            data,
            self.freqs,
            self.dt,
            self.ordmax,
            progress=progress,
        )
        Phis, Lambds, Lrs = plscf.get_poles(
            Ad, Bn, self.dt, spectrum=self.spectrum, alpha=self.alpha
        )

        # ---- Hard criteria ----
        keep = criteria.hc_conj(Lambds)
        keep &= criteria.hc_damp(plscf.lambd_to_zeta(Lambds), self.hc["zeta_max"])
        if self.hc.get("mpc_lim") is not None:
            keep &= criteria.mpc(Phis) >= self.hc["mpc_lim"]
        if self.hc.get("mpd_lim") is not None:
            keep &= criteria.mpd(Phis) <= self.hc["mpd_lim"]
        Lambds = jnp.where(keep, Lambds, jnp.nan)
        Phis = jnp.where(keep[:, :, None], Phis, jnp.nan)
        Lrs = jnp.where(keep[:, :, None], Lrs, jnp.nan)

        # ---- Soft criteria (stabilisation) ----
        stab_label = criteria.sc_apply(
            plscf.lambd_to_fn(Lambds),
            plscf.lambd_to_zeta(Lambds),
            Phis,
            self.ordmin,
            self.sc["err_fn"],
            self.sc["err_zeta"],
            self.sc["err_phi"],
        )
        return PoleArrays(Lambds, Phis, Lrs, stab_label)

    def stab_plot(
        self,
        poles: PoleArrays,
        data=None,
        indicator: Optional[str] = "cmif",
        y_label: Optional[str] = None,
        xlim=None,
        hide_poles=True,
        indicator_color: str = "tab:blue",
        stable_color: str = "#4caf4a",
        unstable_color: str = "black",
        figsize=None,
        save_fig_name: Optional[str] = None,
    ):
        """
        Plot the stabilisation chart of fitted poles.

        Parameters
        ----------
        poles : PoleArrays
            Screened poles from :meth:`fit`.
        data : array_like, optional
            The FRF / output spectra ``(Q, P, N)`` the poles were fitted to.
            Given, its mode-identification tool is overlaid on the left y-axis.
        indicator : {"cmif", "mmif", "sum"} or None, optional
            Which tool to overlay when ``data`` is passed. Default ``"cmif"``.
            ``"mmif"`` assumes FRF data and is rejected for OMA spectra.
        y_label : str, optional
            Label for the indicator (left) y-axis. Defaults to ``indicator``
            upper-cased (e.g. ``"CMIF"``).
        xlim : tuple(float, float), optional
            Frequency-axis limits. Default None.
        hide_poles : bool, optional
            If True show only stable poles. Default True.
        indicator_color : str, optional
            Colour of the overlaid indicator curves. Default ``"tab:blue"``.
        stable_color : str, optional
            Colour for stable poles. Default ``"#4caf4a"``.
        unstable_color : str, optional
            Colour for unstable poles. Default ``"black"``.
        figsize : tuple, optional
            Figure size. Default None.
        save_fig_name : str, optional
            If given, the figure is saved to this path.

        Raises
        ------
        ValueError
            If ``indicator="mmif"`` is requested for an OMA (``sd_*``) spectrum.
        """
        if (
            indicator is not None
            and indicator.lower() == "mmif"
            and not self.spectrum.startswith("frf")
        ):
            raise ValueError(
                f"indicator='mmif' assumes FRF data (its dip needs the real part "
                f"to vanish at resonance) and is undefined for the OMA spectrum "
                f"{self.spectrum!r}; use 'cmif' or 'sum'."
            )
        return plots.plot_stab(
            np.asarray(poles.Fn),
            np.asarray(poles.stab_label),
            ordmax=self.ordmax,
            ordmin=self.ordmin,
            freqs=np.asarray(self.freqs),
            data=None if data is None else np.asarray(data),
            indicator=indicator,
            y_label=y_label,
            indicator_color=indicator_color,
            stable_color=stable_color,
            unstable_color=unstable_color,
            xlim=xlim,
            hide_poles=hide_poles,
            figsize=figsize,
            save_fig_name=save_fig_name,
        )

    def indicator_plot(
        self,
        data,
        indicator: str = "cmif",
        y_label: Optional[str] = None,
        xlim=None,
        indicator_color: str = "tab:blue",
        figsize=None,
        save_fig_name: Optional[str] = None,
    ):
        """
        Plot a mode-indicator function (CMIF / MMIF / SUM) of the data.

        Parameters
        ----------
        data : array_like
            The FRF / output spectra ``(Q, P, N)`` to reduce to an indicator.
        indicator : {"cmif", "mmif", "sum"}, optional
            Which tool to draw. Default ``"cmif"``. ``"mmif"`` assumes FRF data
            and is rejected for OMA spectra.
        y_label : str, optional
            Label for the y-axis. Defaults to ``indicator`` upper-cased.
        xlim : tuple(float, float), optional
            Frequency-axis limits. Default None.
        indicator_color : str, optional
            Colour of the indicator curves. Default ``"tab:blue"``.
        figsize : tuple, optional
            Figure size. Default None.
        save_fig_name : str, optional
            If given, the figure is saved to this path.

        Raises
        ------
        ValueError
            If ``indicator="mmif"`` is requested for an OMA (``sd_*``) spectrum.
        """
        if indicator.lower() == "mmif" and not self.spectrum.startswith("frf"):
            raise ValueError(
                f"indicator='mmif' assumes FRF data (its dip needs the real part "
                f"to vanish at resonance) and is undefined for the OMA spectrum "
                f"{self.spectrum!r}; use 'cmif' or 'sum'."
            )
        return plots.plot_indicator(
            np.asarray(self.freqs),
            np.asarray(data),
            indicator=indicator,
            y_label=y_label,
            indicator_color=indicator_color,
            xlim=xlim,
            figsize=figsize,
            save_fig_name=save_fig_name,
        )

    def freqs_zetas_plot(
        self,
        poles: PoleArrays,
        xlim=None,
        hide_poles=True,
        stable_color: str = "#4caf4a",
        unstable_color: str = "black",
        sel_fn=None,
        sel_zeta=None,
        figsize=None,
        save_fig_name: Optional[str] = None,
    ):
        """
        Plot the frequency-damping cluster chart of fitted poles.

        Parameters
        ----------
        poles : PoleArrays
            Screened poles from :meth:`fit`.
        xlim : tuple(float, float), optional
            Frequency-axis limits. Default None.
        hide_poles : bool, optional
            If True show only stable poles. Default True.
        stable_color : str, optional
            Colour for stable poles. Default ``"#4caf4a"``.
        unstable_color : str, optional
            Colour for unstable poles. Default ``"black"``.
        sel_fn, sel_zeta : array_like, optional
            Frequencies [Hz] and damping ratios of the picked modes.
            Default None (no overlay).
        figsize : tuple, optional
            Figure size. Default None.
        save_fig_name : str, optional
            If given, the figure is saved to this path.
        """
        return plots.plot_freqs_zetas(
            np.asarray(poles.Fn),
            np.asarray(poles.Zeta),
            np.asarray(poles.stab_label),
            stable_color=stable_color,
            unstable_color=unstable_color,
            xlim=xlim,
            hide_poles=hide_poles,
            sel_fn=None if sel_fn is None else np.asarray(sel_fn),
            sel_zeta=None if sel_zeta is None else np.asarray(sel_zeta),
            figsize=figsize,
            save_fig_name=save_fig_name,
        )

    def cluster(
        self,
        poles: PoleArrays,
        eps: float = clustering.DEFAULT_EPS,
        min_pts: int = clustering.DEFAULT_MIN_PTS,
        select: str = "density",
        min_cluster_size: Optional[int] = None,
    ) -> clustering.ClusterResult:
        """
        Group the stable poles into modes with DBSCAN.

        Convenience wrapper around
        :class:`~modekit.clustering.DBSCAN`, which clusters the poles
        on the composite modal distance (normalised frequency gap + ``1 - MAC``):
        each dense column of poles is one physical mode, and the sparse poles are
        left as noise.

        Parameters
        ----------
        poles : PoleArrays
            Screened poles from :meth:`fit`.
        eps : float, optional
            Reachability radius on the modal distance. Default ``0.005``.
        min_pts : int, optional
            Fewest orders a mode must persist over, self included. Default ``25``.
        select : {'medoid', 'median', 'mean', 'density'}, optional
            How each cluster's representative mode is read off. Default 'density'.
        min_cluster_size : int, optional
            Drop clusters with fewer than this many poles. Default None.

        Returns
        -------
        clustering.ClusterResult
            Representative modes (``Lambd``, ``Phi``, ``Lr``) that feed
            :meth:`LSFD.fit`, plus the clustered poles for the cluster plots.
        """
        return clustering.DBSCAN(
            eps=eps,
            min_pts=min_pts,
            select=select,
            min_cluster_size=min_cluster_size,
        ).fit(poles)

    def stab_cluster_plot(
        self,
        result: clustering.ClusterResult,
        data=None,
        indicator: Optional[str] = "cmif",
        y_label: Optional[str] = None,
        xlim=None,
        plot_noise: bool = False,
        show_medians: bool = True,
        indicator_color: str = "tab:blue",
        cmap_name: Optional[str] = None,
        figsize=None,
        save_fig_name: Optional[str] = None,
    ):
        """
        Plot the stabilisation chart with the poles coloured by cluster.

        Cluster counterpart of :meth:`stab_plot`.

        Parameters
        ----------
        result : clustering.ClusterResult
            Clustering from :meth:`cluster`.
        data : array_like, optional
            The FRF / output spectra ``(Q, P, N)`` the poles were fitted to.
            Given, its mode-identification tool is overlaid on the left y-axis.
        indicator : {"cmif", "mmif", "sum"} or None, optional
            Which tool to overlay when ``data`` is passed. Default ``"cmif"``.
            ``"mmif"`` assumes FRF data and is rejected for OMA spectra.
        y_label : str, optional
            Label for the indicator (left) y-axis. Defaults to ``indicator``
            upper-cased.
        xlim : tuple(float, float), optional
            Frequency-axis limits. Default None.
        plot_noise : bool, optional
            Draw the noise poles in grey. Default False.
        show_medians : bool, optional
            Dashed vertical line at each cluster's median frequency. Default True.
        indicator_color : str, optional
            Colour of the overlaid indicator curves. Default ``"tab:blue"``.
        cmap_name : str, optional
            Qualitative colormap for the clusters. Default ``"tab10"``/``"tab20"``.
        figsize : tuple, optional
            Figure size. Default None.
        save_fig_name : str, optional
            If given, the figure is saved to this path.

        Raises
        ------
        ValueError
            If ``indicator="mmif"`` is requested for an OMA (``sd_*``) spectrum.
        """
        if (
            indicator is not None
            and indicator.lower() == "mmif"
            and not self.spectrum.startswith("frf")
        ):
            raise ValueError(
                f"indicator='mmif' assumes FRF data (its dip needs the real part "
                f"to vanish at resonance) and is undefined for the OMA spectrum "
                f"{self.spectrum!r}; use 'cmif' or 'sum'."
            )
        return plots.plot_stab_cluster(
            np.asarray(result.Fn_fl),
            np.asarray(result.order_fl),
            np.asarray(result.labels),
            ordmax=self.ordmax,
            ordmin=self.ordmin,
            freqs=np.asarray(self.freqs),
            data=None if data is None else np.asarray(data),
            indicator=indicator,
            y_label=y_label,
            indicator_color=indicator_color,
            cmap_name=cmap_name,
            plot_noise=plot_noise,
            show_medians=show_medians,
            xlim=xlim,
            figsize=figsize,
            save_fig_name=save_fig_name,
        )

    def cluster_freqs_zetas_plot(
        self,
        result: clustering.ClusterResult,
        xlim=None,
        plot_noise: bool = False,
        cmap_name: Optional[str] = None,
        show_selected: bool = True,
        figsize=None,
        save_fig_name: Optional[str] = None,
    ):
        """
        Plot the frequency-damping chart with the poles coloured by cluster.

        Cluster counterpart of :meth:`freqs_zetas_plot`.

        Parameters
        ----------
        result : clustering.ClusterResult
            Clustering from :meth:`cluster`.
        xlim : tuple(float, float), optional
            Frequency-axis limits. Default None.
        plot_noise : bool, optional
            Draw the noise poles in grey. Default False.
        cmap_name : str, optional
            Qualitative colormap for the clusters. Default ``"tab10"``/``"tab20"``.
        show_selected : bool, optional
            Mark each cluster's picked representative with a red cross.
            Default True.
        figsize : tuple, optional
            Figure size. Default None.
        save_fig_name : str, optional
            If given, the figure is saved to this path.
        """
        return plots.plot_cluster_freqs_zetas(
            np.asarray(result.Fn_fl),
            np.asarray(result.Zeta_fl),
            np.asarray(result.labels),
            cmap_name=cmap_name,
            plot_noise=plot_noise,
            xlim=xlim,
            sel_fn=np.asarray(result.Fn) if show_selected else None,
            sel_zeta=np.asarray(result.Zeta) if show_selected else None,
            figsize=figsize,
            save_fig_name=save_fig_name,
        )

    def k_distance_plot(
        self,
        poles: PoleArrays,
        min_pts: int = clustering.DEFAULT_MIN_PTS,
        eps: Optional[float] = clustering.DEFAULT_EPS,
        line_color: str = "tab:blue",
        eps_color: str = "tab:red",
        figsize=None,
        save_fig_name: Optional[str] = None,
    ):
        """
        Plot the k-distance elbow to choose / validate the DBSCAN ``eps``.

        The curve's knee is a good reachability radius; the drawn ``eps`` line
        (default ``0.005``) should meet it. Use it when tuning ``eps`` on new
        data — see :func:`~modekit.clustering.k_distance`.

        Parameters
        ----------
        poles : PoleArrays
            Screened poles from :meth:`fit`.
        min_pts : int, optional
            Neighbour rank ``k`` (the DBSCAN ``min_pts``). Default ``25``.
        eps : float, optional
            Reachability radius to mark. Default ``0.005``; pass None to omit.
        line_color : str, optional
            Colour of the k-distance curve. Default ``"tab:blue"``.
        eps_color : str, optional
            Colour of the ``eps`` line. Default ``"tab:red"``.
        figsize : tuple, optional
            Figure size. Default None.
        save_fig_name : str, optional
            If given, the figure is saved to this path.
        """
        return plots.plot_k_distance(
            clustering.k_distance(poles, min_pts=min_pts),
            eps=eps,
            k=min_pts,
            line_color=line_color,
            eps_color=eps_color,
            figsize=figsize,
            save_fig_name=save_fig_name,
        )


# =============================================================================
# Second step: modal constants from fixed poles (LSFD)
# =============================================================================

_QUANTITY_ORDER = {"displacement": 0, "velocity": 1, "acceleration": 2}
_SPECTRUM_KIND = {
    "frf_tap": "frf",
    "frf_shaker": "frf",
    "sd_cor": "half",
    "sd_per": "full",
}


def _residual_powers(kind: str, quantity: str) -> Tuple[int, int]:
    """
    Powers of ``jw`` carrying the lower and upper residuals.

    Peeters & Van der Auweraer (IOMAC 2005), table 1
    """
    d = _QUANTITY_ORDER[quantity]
    if kind == "frf":
        return d - 2, d
    if kind == "full":
        return 2 * d - 4, 2 * d
    return -1, 1


def _normalize_shapes(shapes: jax.Array, kind: str) -> jax.Array:
    """
    Rescale each mode (column of ``(Q, M)``) to a chosen unit norm.

    ``"l2"`` gives unit Euclidean length; ``"max"`` sets the largest-magnitude
    DOF to 1 (carrying its sign / phase onto the whole column). All-NaN columns
    (poles that sat out the fit) pass through unchanged.
    """
    kind = kind.lower()
    if kind in ("l2", "unit"):
        scale = jnp.linalg.norm(shapes, axis=0, keepdims=True)  # (1, M)
    elif kind == "max":
        idx = jnp.argmax(jnp.abs(shapes), axis=0)  # (M,) dominant DOF per mode
        scale = jnp.take_along_axis(shapes, idx[None, :], axis=0)  # (1, M)
    else:
        raise ValueError(f"Unknown normalize {kind!r}; choose 'l2' or 'max'.")
    # leave zero / NaN columns be (NaN fails the > 0 test and maps to 1.0)
    scale = jnp.where(jnp.abs(scale) > 0, scale, 1.0)
    return shapes / scale


class ModalModel(eqx.Module):
    """
    Modal constants of one pole set, as fitted by :class:`LSFD`.
    """

    Lambd: jax.Array  # complex poles, (M,)
    A: jax.Array  # modal constants (residues), (Q, P, M)
    LR: jax.Array  # lower residual, (Q, P)
    UR: jax.Array  # upper residual, (Q, P)
    A_mirror: Optional[jax.Array] = None  # (Q, P, M), full spectra only

    @property
    def Fn(self) -> jax.Array:
        """Natural frequencies [Hz], (M,)."""
        return plscf.lambd_to_fn(self.Lambd)

    @property
    def Zeta(self) -> jax.Array:
        """Damping ratios, (M,)."""
        return plscf.lambd_to_zeta(self.Lambd)

    def mode_shapes(self, driving_point=None, real=False, normalize=None) -> jax.Array:
        """
        Mode shapes ``(Q, M)`` read off the residues.

        The constants of one mode are the outer product of its mode shape and
        its participation (or operational reference) factors.

        Parameters
        ----------
        driving_point : (int, int), optional
            ``(output, reference)`` index of a driving-point measurement, where
            the response and the reference sit on the same DOF. Yields
            unity-modal-A (mass-normalised) shapes. Only meaningful for FRFs —
            output spectra carry no absolute scale. Without it the shapes keep
            the leading singular value of each residue matrix as their scale.
        real : bool, optional
            If True, return the real (normal) mode most correlated with each
            complex shape, via :func:`plscf.complex_to_real_mode`. Default
            False.
        normalize : {"l2", "max"}, optional
            Rescale each returned mode: ``"l2"`` to unit Euclidean norm,
            ``"max"`` so the largest-magnitude DOF is 1. None (default)
            leaves the native scale untouched.
        """
        if driving_point is None:
            U, S, _ = jnp.linalg.svd(self.A.transpose(2, 0, 1), full_matrices=False)
            shapes = (U[:, :, 0] * S[:, :1]).T
        else:
            q, p = driving_point
            shapes = self.A[:, p, :] / jnp.sqrt(self.A[q, p, :])
        if real:
            shapes = plscf.complex_to_real_mode(shapes)
        if normalize is not None:
            shapes = _normalize_shapes(shapes, normalize)
        return shapes


class LSFD(eqx.Module):
    """
    Least-Squares Frequency-Domain estimation of modal constants and residuals.

    Pure and traceable: batch over segments with ``jax.vmap(lsfd.fit)``.
    """

    freqs: jax.Array
    fs: float = eqx.field(static=True)
    spectrum: str = eqx.field(static=True)
    quantity: str = eqx.field(static=True)
    alpha: Optional[float] = eqx.field(static=True)
    lower_res: bool = eqx.field(static=True)
    upper_res: bool = eqx.field(static=True)
    band: Optional[Tuple[float, float]] = eqx.field(static=True)

    def __init__(
        self,
        freqs,
        fs: float,
        spectrum: str = "frf_tap",
        quantity: str = "acceleration",
        alpha: Optional[float] = None,
        lower_res: bool = True,
        upper_res: bool = True,
        band: Optional[Tuple[float, float]] = None,
    ):
        """
        Parameters
        ----------
        freqs : array_like
            Frequency vector [Hz], length ``N``.
        fs : float
            Sampling frequency [Hz] of the acquisition.
        spectrum : str, optional
            The flag given to :class:`pLSCF`: ``"frf_tap"``/``"frf_shaker"``
            (FRFs), ``"sd_cor"`` (half spectra) or ``"sd_per"`` (full spectra).
        quantity : {'acceleration', 'velocity', 'displacement'}, optional
            Measured response quantity. Ignored for half spectra, whose
            residuals are the same whatever was measured.
        alpha : float, optional
            Exponential-window decay rate [1/s], as in :class:`pLSCF`.
        lower_res, upper_res : bool, optional
            Fit the out-of-band residual terms. Default True.
        band : (float, float), optional
            Frequency range [Hz] the fit reads. None uses every line.
        """
        if spectrum not in _SPECTRUM_KIND:
            raise ValueError(
                f"Unknown spectrum {spectrum!r}. Choose from "
                f"{sorted(_SPECTRUM_KIND)}."
            )
        if quantity not in _QUANTITY_ORDER:
            raise ValueError(
                f"Unknown quantity {quantity!r}. Choose from "
                f"{sorted(_QUANTITY_ORDER)}."
            )
        self.freqs = jnp.asarray(freqs)
        self.fs = float(fs)
        self.spectrum = spectrum
        self.quantity = quantity
        self.alpha = None if alpha is None else float(alpha)
        self.lower_res = bool(lower_res)
        self.upper_res = bool(upper_res)
        self.band = None if band is None else (float(band[0]), float(band[1]))

    @property
    def dt(self) -> float:
        """Sampling interval [s]."""
        return 1.0 / self.fs

    @property
    def kind(self) -> str:
        """Data the residuals are written for: 'frf', 'half' or 'full'."""
        return _SPECTRUM_KIND[self.spectrum]

    def fit(self, data, Lambd, part_factors=None) -> ModalModel:
        """
        Fit the modal constants and residuals of ``Lambd`` to ``data``.

        Parameters
        ----------
        data : array_like
            The same FRF or output spectra the poles were fitted to, ``(Q, P, N)``.
        Lambd : array_like
            Complex poles of the ``M`` modes to fit, one per mode (no conjugates).
        part_factors : array_like, optional
            Participation (EMA) or operational reference (OMA) factors ``(P, M)``.

        Returns
        -------
        ModalModel
        """
        data = jnp.asarray(data)
        Lambd = jnp.asarray(Lambd).reshape(-1)
        if part_factors is not None:
            part_factors = jnp.asarray(part_factors, dtype=complex).reshape(
                -1, Lambd.shape[0]
            )
        Q, P, N = data.shape

        Psi, lines = self._basis(Lambd, part_factors, P)  # (P*N, C), (N,)
        if self.band is not None:
            lines = lines & (self.freqs >= self.band[0]) & (self.freqs <= self.band[1])
        rows = jnp.broadcast_to(lines, (P, N)).reshape(-1)  # (P*N,)
        Psi = jnp.where(rows[:, None], Psi, 0.0)
        D = jnp.where(rows[:, None], data.transpose(1, 2, 0).reshape(P * N, Q), 0.0)

        X = jnp.concatenate([Psi.real, Psi.imag])  # (2PN, C)
        Y = jnp.concatenate([D.real, D.imag])  # (2PN, Q)
        scale = jnp.linalg.norm(X, axis=0)  # columns span many decades
        scale = jnp.where(scale > 0, scale, 1.0)
        theta = jnp.linalg.lstsq(X / scale, Y)[0] / scale[:, None]  # (C, Q)
        return self._unpack(theta, Lambd, part_factors, Q, P)

    def synthesize(self, model: ModalModel) -> jax.Array:
        """
        Rebuild the data ``(Q, P, N)`` from a fitted model, to compare with the
        measurement. NaN on frequency lines the model cannot reach (``jw = 0``
        under a negative residual power).
        """
        (u, w), mirror, (r_lo, r_up), lines = self._terms(model.Lambd)

        def modal(A, x, y):  # NaN poles sat out of the fit, so we leave them out here
            A = jnp.where(jnp.isfinite(A), A, 0.0)
            return jnp.einsum("qpi,ni->qpn", A, x) + jnp.einsum(
                "qpi,ni->qpn", A.conj(), y
            )

        D = modal(model.A, u, w)
        if mirror is not None and model.A_mirror is not None:
            D = D + modal(model.A_mirror, *mirror)
        if self.lower_res:
            D = D + model.LR[:, :, None] * r_lo
        if self.upper_res:
            D = D + model.UR[:, :, None] * r_up
        return jnp.where(lines, D, jnp.nan)

    def synthesis_plot(
        self,
        model: ModalModel,
        measured,
        pairs=None,
        phase=False,
        xlim=None,
        yscale="log",
        y_labels=None,
        measured_color="black",
        synth_color="tab:red",
        figsize=None,
        save_fig_name: Optional[str] = None,
    ):
        """
        Overlay the measured data against this model's synthesis.

        Synthesizes ``model`` on ``freqs`` and delegates to
        :func:`plots.plot_synthesis`.

        Parameters
        ----------
        model : ModalModel
            Fitted modal model from :meth:`fit`.
        measured : array_like
            The FRF / output spectra ``(Q, P, N)`` the model was fitted to.
        pairs : sequence of (int, int), optional
            ``(output, reference)`` indices to draw. Defaults to every pair.
        phase : bool, optional
            If True, add a phase panel under each magnitude panel. Default False.
        xlim : tuple(float, float), optional
            Frequency-axis limits. Default None.
        yscale : str, optional
            Matplotlib scale of the magnitude axis. Default ``"log"``.
        y_labels : sequence of str, optional
            Y-axis label of each pair's magnitude subplot, one per pair.
            Default None: no labels.
        measured_color, synth_color : str, optional
            Colours of the measured and synthesized curves.
        figsize : tuple, optional
            Figure size. Default None.
        save_fig_name : str, optional
            If given, the figure is saved to this path.
        """
        return plots.plot_synthesis(
            np.asarray(self.freqs),
            np.asarray(measured),
            np.asarray(self.synthesize(model)),
            pairs=pairs,
            phase=phase,
            xlim=xlim,
            yscale=yscale,
            y_labels=y_labels,
            measured_color=measured_color,
            synth_color=synth_color,
            figsize=figsize,
            save_fig_name=save_fig_name,
        )

    # ---- model matrix -------------------------------------------------------
    def _terms(self, Lambd):
        """
        What the model is built from, on ``freqs``: the pole pair ``(u, w)``, its
        mirror for a full spectrum, the two residual shapes, and the lines it can
        reach, ``(N,)``.
        """
        s = 2j * jnp.pi * self.freqs[:, None]  # (N, 1)
        lam = Lambd[None, :] - plscf.window_shift(
            self.spectrum, self.alpha
        )  # (1, M), as seen through the pre-processing window
        ok = jnp.isfinite(lam)
        lam = jnp.where(ok, lam, 0.0)

        def pole_pair(z):  # a pole and its conjugate, (N, M) each
            return (
                jnp.where(ok, 1.0 / (z - lam), 0.0),
                jnp.where(ok, 1.0 / (z - lam.conj()), 0.0),
            )

        lo, up = _residual_powers(self.kind, self.quantity)
        res = (s[:, 0] ** lo, s[:, 0] ** up)
        lines = jnp.ones(self.freqs.shape, dtype=bool)
        if self.lower_res:
            lines = lines & jnp.isfinite(res[0])
        if self.upper_res:
            lines = lines & jnp.isfinite(res[1])
        mirror = pole_pair(-s) if self.kind == "full" else None
        return pole_pair(s), mirror, res, lines

    def _basis(self, Lambd, part_factors, P) -> Tuple[jax.Array, jax.Array]:
        """
        Columns of the model, ``(P*N, C)``, with the lines it can reach, ``(N,)``.
        """
        (u, w), mirror, (r_lo, r_up), lines = self._terms(Lambd)
        N, M = u.shape
        per_ref = jnp.broadcast_to(jnp.eye(P), (M, P, P))  # one unknown per reference

        def pole_cols(W, x, y):
            """Columns of a pole pair whose unknowns are weighted by ``(M, ., P)``."""
            a = jnp.einsum("ikp,ni->pnik", W, x)
            b = jnp.einsum("ikp,ni->pnik", W.conj(), y)
            return [(a + b).reshape(P * N, -1), (1j * (a - b)).reshape(P * N, -1)]

        weights = per_ref if part_factors is None else part_factors.T[:, None]
        cols = pole_cols(weights, u, w)
        if mirror is not None:
            # a mirrored mode transposes the constants, which the reference
            # factors cannot describe, so it keeps one unknown per reference
            cols += pole_cols(per_ref, *mirror)
        for fitted, r in ((self.lower_res, r_lo), (self.upper_res, r_up)):
            if fitted:
                R = (jnp.eye(P)[:, None, :] * r[None, :, None]).reshape(P * N, P)
                cols += [R, 1j * R]
        return jnp.concatenate(cols, axis=1), lines

    def _unpack(self, theta, Lambd, part_factors, Q, P) -> ModalModel:
        """Read the fitted real coefficients back as complex arrays."""
        M = Lambd.shape[0]
        n_ref = P if part_factors is None else 1  # unknowns per mode and output

        def block(i, n):  # n complex unknowns, real parts first
            return theta[i : i + n] + 1j * theta[i + n : i + 2 * n]

        coef = block(0, M * n_ref).reshape(M, n_ref, Q)
        if part_factors is None:
            A = coef.transpose(2, 1, 0)  # (Q, P, M)
        else:
            A = jnp.einsum("iq,pi->qpi", coef[:, 0, :], part_factors)
        i = 2 * M * n_ref
        A_mirror = None
        if self.kind == "full":
            A_mirror = block(i, M * P).reshape(M, P, Q).transpose(2, 1, 0)
            i += 2 * M * P
        LR = block(i, P).T if self.lower_res else jnp.zeros_like(A[:, :, 0])
        i += 2 * P if self.lower_res else 0
        UR = block(i, P).T if self.upper_res else jnp.zeros_like(A[:, :, 0])

        bad = ~jnp.isfinite(Lambd)
        A = jnp.where(bad, jnp.nan, A)
        if A_mirror is not None:
            A_mirror = jnp.where(bad, jnp.nan, A_mirror)
        return ModalModel(Lambd, A, LR, UR, A_mirror)
