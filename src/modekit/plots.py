"""
Figures of modal tests and their identification.

Measured signals and signal ensembles with their confidence bands,
stabilisation charts and their clusters, mode-indicator functions,
frequency-damping and k-distance charts, measured spectra against a fitted
model's synthesis, MAC matrices, mode complexity and 3D mode-shape wireframes.
The result classes of :mod:`modekit.algorithms` draw through these.

Importing this module changes no global matplotlib setting. A ``save_fig_name``
is the path the figure is written to.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from scipy.linalg import eigh

from modekit import criteria


def _series_colors(cmap, n, span=(0.0, 1.0)):
    """
    ``n`` distinct colours from a colormap.
    """
    if isinstance(cmap, ListedColormap) and cmap.N <= 20:
        return [cmap(i) for i in range(n)]
    return list(cmap(np.linspace(span[0], span[1], n)))


def _legend_above(fig, handles, labels, group=1):
    """
    Draw a figure legend centered above the axes, wrapping onto extra rows when a
    single row would be wider than the figure.

    Returns
    -------
    float
        Fraction of the figure height the legend occupies, so the caller can
        reserve top margin, e.g. ``tight_layout(rect=[0, 0, 1, 1 - h])``.
    """
    group = max(1, group)
    n_blocks = max(1, len(handles) // group)

    def _place(ncol):
        return fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=ncol,
            frameon=False,
        )

    try:
        fig.canvas.draw()  # realize a renderer so the legend can be measured
        renderer = fig.canvas.get_renderer()
        fig_ext = fig.get_window_extent()
    except Exception:  # backend without an Agg renderer: fall back to a cap
        ncol = group * min(n_blocks, 3)
        _place(ncol)
        return 0.04 * np.ceil(len(handles) / ncol)

    leg = None
    for blocks in range(n_blocks, 0, -1):  # widest first, shrink until it fits
        if leg is not None:
            leg.remove()
        leg = _place(blocks * group)
        if leg.get_window_extent(renderer).width <= 0.98 * fig_ext.width:
            break
    return leg.get_window_extent(renderer).height / fig_ext.height


def plot_signal(
    x,
    signal,
    cmap_name="Dark2",
    cmap_span=(0.0, 1.0),
    figsize=None,
    xlim=None,
    ylim=None,
    yscale="linear",
    x_label=None,
    y_labels=None,
    save_fig_name=None,
):
    """
    Plot signals on stacked subplots, one subplot per channel.

    Silent channels are skipped.

    Parameters
    ----------
    x : array
        Independent-variable vector of shape ``(nx,)``, e.g. time or frequency.
    signal : array or dict
        Either a single ``(n_ch, nx)`` array, or a mapping
        ``{label: (n_ch, nx) array}``. When a dict is given, every series is
        overlaid on the same subplot per channel and identified through a legend.
    cmap_name : str, optional
        Name of colormap to use. Default is 'Dark2'.
    cmap_span : (float, float), optional
        Sub-range of ``[0, 1]`` over which a continuous colormap is sampled.
    figsize : tuple, optional
        Figure size. Defaults to ``(10, 2 * n_active)``.
    xlim : tuple, optional
        ``(xmin, xmax)`` for the x-axis. Defaults to the span of ``x``.
    ylim : tuple, optional
        ``(ymin, ymax)`` applied to every subplot. Defaults to autoscale.
    yscale : str, optional
        Matplotlib y-axis scale, e.g. ``"linear"`` or ``"log"``.
    x_label : str, optional
        Label for the (shared) x-axis. Default None: no label.
    y_labels : sequence of str, optional
        Y-axis label of each channel's subplot, one per channel (row of
        ``signal``). Default None: no labels.
    save_fig_name : str, optional
        If provided, the figure will be saved with this name.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray of matplotlib.axes.Axes
    """

    series = signal if isinstance(signal, dict) else {None: signal}
    arrays = list(series.values())
    n_ch = arrays[0].shape[0]
    if y_labels is not None and len(y_labels) != n_ch:
        raise ValueError(f"y_labels has {len(y_labels)} entries for {n_ch} channels.")

    active = [i for i in range(n_ch) if any(np.any(arr[i, :] != 0.0) for arr in arrays)]
    n = len(active)
    if n == 0:
        raise ValueError("All channels are silent — nothing to plot.")

    if figsize is None:
        figsize = (10, 2 * n)

    cmap = plt.get_cmap(cmap_name)
    n_series = len(series)
    colors = _series_colors(cmap, n if n_series == 1 else n_series, span=cmap_span)

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True, squeeze=False)
    axes = axes.ravel()

    for k, ch in enumerate(active):  # subplot k shows channel ch
        for s, (label, arr) in enumerate(series.items()):
            color = colors[k] if n_series == 1 else colors[s]
            axes[k].plot(
                x,
                arr[ch, :],
                color=color,
                linewidth=1.0,
                label=None if label is None else str(label),
            )
        axes[k].set_xlim(xlim if xlim is not None else (x[0], x[-1]))
        axes[k].set_yscale(yscale)
        if ylim is not None:
            axes[k].set_ylim(ylim)
        axes[k].xaxis.set_minor_locator(AutoMinorLocator())
        if yscale == "linear":
            axes[k].yaxis.set_minor_locator(AutoMinorLocator())
        axes[k].tick_params(which="both", direction="in", top=True, right=True)
        if y_labels is not None:
            axes[k].set_ylabel(
                y_labels[ch]
            )  # by channel, so a silent one takes its label along
        axes[k].grid(True, linestyle="--", alpha=0.7)

    if x_label is not None:
        axes[-1].set_xlabel(x_label)

    if n_series > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        h = _legend_above(fig, handles, labels, group=len(handles) // n_series)
        plt.tight_layout(rect=[0, 0, 1, 1 - h - 0.01])
    else:
        plt.tight_layout()

    if save_fig_name:
        save_path = Path(save_fig_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_signal_ci(
    x,
    signal,
    cmap_name="Dark2",
    cmap_span=(0.0, 1.0),
    figsize=None,
    xlim=None,
    yscale="linear",
    x_label=None,
    y_labels=None,
    ci=0.95,
    max_samples=50,
    linestyle="dashdot",
    marker=None,
    simple_legend=False,
    save_fig_name=None,
):
    """
    Plot ensemble signals on stacked subplots, one subplot per channel.

    Each series is rendered as (i) up to ``max_samples`` thin realizations in
    the background, (ii) the central ``ci`` empirical credible interval as a
    shaded band, and (iii) the ensemble mean on top.

    Parameters
    ----------
    x : array
        Independent-variable vector of shape ``(nx,)``, e.g. time or frequency.
    signal : array or dict
        A ``(P, n_ch, nx)`` ensemble, or ``{label: (P, n_ch, nx) array}`` to
        overlay several series on each channel's subplot, with a legend. NaNs are
        ignored, so realizations of unequal length can be padded with NaN; a
        series is drawn only through the ``x`` where it has samples.
    cmap_name : str or Colormap, optional
        Colormap name (or a ``Colormap`` object) to colour the series. Default
        is 'Dark2'.
    cmap_span : (float, float), optional
        Sub-range of ``[0, 1]`` over which a continuous colormap is sampled.
    figsize : tuple, optional
        Figure size. Defaults to ``(10, 2 * n_active)``.
    xlim : tuple, optional
        ``(xmin, xmax)`` for the x-axis. Defaults to the span of ``x``.
    yscale : str, optional
        Matplotlib y-axis scale, e.g. ``"linear"`` or ``"log"``.
    x_label : str, optional
        Label of the x-axis. Default None: no label.
    y_labels : sequence of str, optional
        Y-axis label of each channel's subplot, one per channel (second axis
        of ``signal``). Default None: no labels.
    ci : float, optional
        Central probability mass in [0, 1] for the empirical credible band.
        Defaults to 0.95.
    max_samples : int, optional
        Maximum number of individual realizations to overlay per series.
    linestyle : str, optional
        Line style of the ensemble mean. Default ``"dashdot"``.
    marker : str, optional
        Marker drawn on the mean line at every ``x`` sample, e.g. ``"o"``;
        None draws none. Default None.
    simple_legend : bool, optional
        If True, the legend carries one entry per series.
    save_fig_name : str, optional
        If provided, the figure will be saved with this name.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray of matplotlib.axes.Axes
    """

    x = np.asarray(x, dtype=float)
    series = signal if isinstance(signal, dict) else {None: signal}
    arrays = list(series.values())
    n_ch = arrays[0].shape[1]
    if y_labels is not None and len(y_labels) != n_ch:
        raise ValueError(f"y_labels has {len(y_labels)} entries for {n_ch} channels.")

    active = [
        i for i in range(n_ch) if any(np.any(arr[:, i, :] != 0.0) for arr in arrays)
    ]
    n = len(active)
    if n == 0:
        raise ValueError("All channels are silent — nothing to plot.")

    if figsize is None:
        figsize = (10, 2 * n)

    cmap = plt.get_cmap(cmap_name)
    n_series = len(series)
    colors = _series_colors(cmap, n if n_series == 1 else n_series, span=cmap_span)

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True, squeeze=False)
    axes = axes.ravel()

    mean_lines = []
    for k, ch in enumerate(active):  # subplot k shows channel ch
        for s, (label, arr) in enumerate(series.items()):
            color = colors[k] if n_series == 1 else colors[s]
            # NaN-aware (ensembles of unequal size come NaN-padded); keep only the x
            # where this series has samples, so the lines join them (markers show them)
            has = np.isfinite(arr[:, ch, :]).any(axis=0)
            if not has.any():
                continue
            xs, ens = x[has], arr[:, ch, :][:, has]
            mean = np.nanmean(ens, axis=0)
            q_lo = 0.5 * (1.0 - ci) * 100.0
            q_hi = 100.0 - q_lo
            lo = np.nanpercentile(ens, q_lo, axis=0)
            hi = np.nanpercentile(ens, q_hi, axis=0)
            band_label = rf"{ci * 100:g}\% CI"

            n_show = min(max_samples, ens.shape[0])
            for i in range(n_show):
                axes[k].plot(
                    xs, ens[i], color=color, alpha=0.15, linewidth=0.7, zorder=1
                )

            prefix = "" if label is None else f"{label}: "
            axes[k].fill_between(
                xs,
                lo,
                hi,
                color=color,
                alpha=0.2,
                label=f"{prefix}{band_label}",
                zorder=2,
            )
            (mean_line,) = axes[k].plot(
                xs,
                mean,
                color=color,
                linewidth=1.5,
                linestyle=linestyle,
                marker=marker,
                markersize=5,
                label=f"{prefix}mean",
                zorder=3,
            )
            if k == 0:
                mean_lines.append(mean_line)

        axes[k].set_xlim(xlim if xlim is not None else (x[0], x[-1]))
        axes[k].set_yscale(yscale)
        axes[k].xaxis.set_minor_locator(AutoMinorLocator())
        if yscale == "linear":
            axes[k].yaxis.set_minor_locator(AutoMinorLocator())
        axes[k].tick_params(which="both", direction="in", top=True, right=True)
        if y_labels is not None:
            axes[k].set_ylabel(
                y_labels[ch]
            )  # by channel, so a silent one takes its label along
        axes[k].grid(True, linestyle="--", alpha=0.7)

    if x_label is not None:
        axes[-1].set_xlabel(x_label)

    if simple_legend and n_series > 1:
        labels = [str(lbl) for lbl in series]
        h = _legend_above(fig, mean_lines, labels, group=1)
        plt.tight_layout(rect=[0, 0, 1, 1 - h - 0.01])
    else:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            # each series contributes a CI band + a mean entry — keep the pair together
            h = _legend_above(fig, handles, labels, group=len(handles) // n_series)
            plt.tight_layout(rect=[0, 0, 1, 1 - h - 0.01])
        else:
            plt.tight_layout()

    if save_fig_name:
        save_path = Path(save_fig_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def _mode_indicator(data, kind="cmif"):
    """
    Mode-identification tool from an FRF / spectral matrix.

    Shape notation follows ``plscf``: ``Q`` outputs, ``P`` references, ``N``
    frequency lines.

    Parameters
    ----------
    data : ndarray, shape (Q, P, N)
        FRF (EMA) or output-spectra (OMA) matrix.
    kind : {"cmif", "mmif", "sum"}, optional
        - ``"cmif"``: singular values per line (Eq. 5.18); FDD, EMA and OMA.
        - ``"sum"``: ``sum |data|`` over all pairs (& 5.4.1); EMA and OMA.
        - ``"mmif"``: generalized eigenvalues of
          ``(H_R^T H_R, H_R^T H_R + H_I^T H_I)`` per line (Eq. 5.17); EMA only.

    Returns
    -------
    ndarray, shape (n_curves, N)
        One row per indicator curve (``min(Q, P)`` for CMIF, ``P`` for MMIF,
        ``1`` for SUM), ordered so that row 0 is the primary curve.
    """
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"data must have shape (Q, P, N), got {data.shape}.")
    _, P, N = data.shape
    Dn = np.moveaxis(data, -1, 0)  # (N, Q, P), one matrix per frequency line

    kind = kind.lower()
    if kind == "sum":
        return np.abs(Dn).sum(axis=(1, 2))[None, :]  # (1, N)
    if kind == "cmif":
        sv = np.linalg.svd(Dn, compute_uv=False)  # (N, min(Q, P)), descending
        return sv.T  # (min(Q, P), N), primary (largest) first
    if kind == "mmif":
        # EMA only: H_R / H_I are the FRF real / imaginary parts (Eq. 5.17).
        Hr, Hi = Dn.real, Dn.imag
        A = np.matmul(Hr.transpose(0, 2, 1), Hr)  # (N, P, P)
        B = A + np.matmul(Hi.transpose(0, 2, 1), Hi)  # (N, P, P)
        jitter = 1e-12 * np.eye(P)  # keep B positive-definite for eigh
        mmif = np.array(
            [np.sort(eigh(A[n], B[n] + jitter, eigvals_only=True)) for n in range(N)]
        )  # (N, P), primary (smallest) first
        return mmif.T  # (P, N)
    raise ValueError(f"Unknown indicator {kind!r}; choose 'cmif', 'mmif' or 'sum'.")


def plot_indicator(
    freqs,
    data,
    indicator="cmif",
    y_label=None,
    indicator_color="tab:blue",
    figsize=None,
    xlim=None,
    legend=True,
    save_fig_name=None,
):
    """
    Mode-indicator function (CMIF / MMIF / SUM) of an FRF / spectral matrix.

    Parameters
    ----------
    freqs : ndarray, shape (N,)
        Frequency vector [Hz].
    data : ndarray, shape (Q, P, N)
        FRF / output-spectra matrix, passed to :func:`_mode_indicator`.
    indicator : {"cmif", "mmif", "sum"}, optional
        Which mode-identification tool to draw. Default ``"cmif"``.
    y_label : str, optional
        Label for the y-axis. Defaults to ``indicator`` upper-cased.
    indicator_color : str, optional
        Single colour for the indicator curves; the leading curve reads boldest.
        Default ``"tab:blue"``.
    figsize : tuple, optional
        Figure size. Defaults to ``(10, 6)``.
    xlim : tuple(float, float), optional
        Frequency-axis limits. Defaults to the span of ``freqs``. The y-axis is
        fitted to the curves inside these limits, not to the whole record.
    legend : bool, optional
        Name the curves in a legend above the axes: the singular values
        :math:`\\sigma_1, \\sigma_2, \\ldots` of the CMIF (largest first), the
        eigenvalues :math:`\\lambda_i` of the MMIF, or the single SUM curve.
        Default True.
    save_fig_name : str, optional
        If given, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    freqs = np.asarray(freqs)
    curves = _mode_indicator(data, indicator)
    kind = indicator.lower()

    if figsize is None:
        figsize = (10, 6)
    fig, ax = plt.subplots(figsize=figsize)

    n = len(curves)
    symbol = {"cmif": r"\sigma", "mmif": r"\lambda"}.get(kind)
    for i, curve in enumerate(curves):
        alpha = 1.0 - 0.6 * i / max(n - 1, 1)
        label = rf"${symbol}_{{{i + 1}}}$" if symbol else r"$\sum |H|$"
        ax.plot(
            freqs, curve, color=indicator_color, alpha=alpha, linewidth=1.0, label=label
        )
    # MMIF lives in [0, 1] and dips to a hard floor, so it reads linearly;
    # CMIF/SUM span decades and stay logarithmic.
    log_y = kind != "mmif"
    ax.set_yscale("log" if log_y else "linear")
    ax.set_ylabel(y_label if y_label is not None else indicator.upper())

    ax.set_xlabel("Frequency [Hz]")
    xlim = tuple(xlim) if xlim is not None else (freqs[0], freqs[-1])
    ax.set_xlim(xlim)
    # y-limits from what is shown: the record outside xlim (the decades the lower
    # singular values fall through near 0 Hz, say) must not set the scale
    shown = curves[:, (freqs >= xlim[0]) & (freqs <= xlim[1])]
    shown = shown[np.isfinite(shown) & ((shown > 0) if log_y else True)]
    if shown.size:
        lo, hi = float(shown.min()), float(shown.max())
        if log_y:
            pad = (
                (hi / lo) ** 0.05 if hi > lo else 2.0
            )  # ~5 % of the decades shown, each side
            ax.set_ylim(lo / pad, hi * pad)
        else:
            pad = 0.05 * (hi - lo) or 0.05
            ax.set_ylim(lo - pad, hi + pad)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)

    if legend:
        h = _legend_above(fig, *ax.get_legend_handles_labels())
        plt.tight_layout(rect=[0, 0, 1, 1 - h - 0.01])
    else:
        plt.tight_layout()

    if save_fig_name:
        save_path = Path(save_fig_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_stab(
    Fn,
    stab_label,
    ordmax,
    ordmin=0,
    freqs=None,
    data=None,
    indicator="cmif",
    y_label=None,
    indicator_color="tab:blue",
    stable_color="#4caf4a",
    unstable_color="black",
    figsize=None,
    xlim=None,
    hide_poles=True,
    save_fig_name=None,
):
    """
    Stabilisation chart: fitted poles over a mode-identification tool.

    Shape notation follows ``plscf``: ``Q`` outputs, ``P`` references, ``N``
    frequency lines, ``K`` poles per order, ``p`` model orders.

    Parameters
    ----------
    Fn : ndarray, shape (K, p)
        Pole frequencies [Hz] per pole and order; NaN where screened out.
    stab_label : ndarray, shape (K, p)
        Stability labels, 1 = stable, 0 = unstable.
    ordmax : int
        Maximum model order (upper limit of the right y-axis).
    ordmin : int, optional
        Minimum model order (lower limit of the right y-axis). Default 0.
    freqs : ndarray, shape (N,), optional
        Frequency vector [Hz] for the overlaid indicator. Required to overlay.
    data : ndarray, shape (Q, P, N), optional
        FRF / output-spectra matrix the poles were fitted to; passed to
        :func:`_mode_indicator`. Required to overlay.
    indicator : {"cmif", "mmif", "sum"} or None, optional
        Which mode-identification tool to overlay. None (or missing
        ``freqs``/``data``) draws the poles alone. Default ``"cmif"``.
    y_label : str, optional
        Label for the indicator (left) y-axis. Defaults to ``indicator``
        upper-cased (e.g. ``"CMIF"``).
    indicator_color : str, optional
        Single colour for the indicator curves; the leading curve reads boldest.
        Default ``"tab:blue"``.
    stable_color : str, optional
        Colour for stable poles. Default ``"#4caf4a"``.
    unstable_color : str, optional
        Colour for unstable poles. Default ``"black"``.
    figsize : tuple, optional
        Figure size. Defaults to ``(10, 6)``.
    xlim : tuple(float, float), optional
        Frequency-axis limits. Defaults to the span of ``freqs``, else autoscale.
    hide_poles : bool, optional
        If True show only stable poles. Default True.
    save_fig_name : str, optional
        If given, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
        The pole (model-order) axis.
    """
    Fn = np.asarray(Fn)
    stab_label = np.asarray(stab_label)

    if figsize is None:
        figsize = (10, 6)
    fig, ax = plt.subplots(figsize=figsize)

    overlay = indicator is not None and freqs is not None and data is not None

    # ---- indicator on the left axis, poles on the right ---------------------
    if overlay:
        freqs = np.asarray(freqs)
        curves = _mode_indicator(data, indicator)
        # One hue for the whole family
        n = len(curves)
        for i, curve in enumerate(curves):
            alpha = 1.0 - 0.6 * i / max(n - 1, 1)
            ax.plot(
                freqs,
                curve,
                color=indicator_color,
                alpha=alpha,
                linewidth=1.0,
                zorder=1,
            )
        # MMIF lives in [0, 1] and dips to a hard floor, so it reads linearly;
        # CMIF/SUM span decades and stay logarithmic.
        ax.set_yscale("linear" if indicator.lower() == "mmif" else "log")
        ax.set_ylabel(y_label if y_label is not None else indicator.upper())
        ax_pole = ax.twinx()  # drawn on top; poles overlay the curves
    else:
        ax_pole = ax

    # ---- poles, coloured by stability ----
    orders = np.broadcast_to(np.arange(1, Fn.shape[1] + 1), Fn.shape)  # (K, p)
    freq_flat = Fn.ravel()
    order_flat = orders.ravel()
    stable = (stab_label == 1).ravel()

    if not hide_poles:
        ax_pole.plot(
            freq_flat[~stable],
            order_flat[~stable],
            ".",
            color=unstable_color,
            markersize=3,
            label="unstable pole",
        )
    ax_pole.plot(
        freq_flat[stable],
        order_flat[stable],
        ".",
        color=stable_color,
        markersize=3,
        label="stable pole",
    )
    ax_pole.set_ylabel("Model order")
    ax_pole.set_ylim(ordmin, ordmax + 1)

    ax.set_xlabel("Frequency [Hz]")
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    elif overlay:
        ax.set_xlim(freqs[0], freqs[-1])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True)
    ax_pole.tick_params(which="both", direction="in")

    ax_pole.legend(loc="upper right", frameon=False, markerscale=3)
    plt.tight_layout()

    if save_fig_name:
        save_path = Path(save_fig_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax_pole


def plot_freqs_zetas(
    Fn,
    Zeta,
    stab_label,
    stable_color="#4caf4a",
    unstable_color="black",
    figsize=None,
    xlim=None,
    hide_poles=True,
    sel_fn=None,
    sel_zeta=None,
    sel_color="red",
    save_fig_name=None,
):
    """
    Frequency-damping cluster chart: damping vs frequency, coloured by stability.

    Shape notation follows ``plscf``: ``K`` poles per order, ``p`` model orders.

    Parameters
    ----------
    Fn : ndarray, shape (K, p)
        Pole frequencies [Hz] per pole and order; NaN where screened out.
    Zeta : ndarray, shape (K, p)
        Damping ratios per pole and order; same shape as ``Fn``.
    stab_label : ndarray, shape (K, p)
        Stability labels, 1 = stable, 0 = unstable.
    stable_color : str, optional
        Colour for stable poles. Default ``"#4caf4a"``.
    unstable_color : str, optional
        Colour for unstable poles. Default ``"black"``.
    figsize : tuple, optional
        Figure size. Defaults to ``(10, 6)``.
    xlim : tuple(float, float), optional
        Frequency-axis limits. Default None (autoscale).
    hide_poles : bool, optional
        If True show only stable poles. Default True.
    sel_fn, sel_zeta : array_like, optional
        Frequencies [Hz] and damping ratios of the picked modes.
        Default None.
    sel_color : str, optional
        Colour of the picked-mode overlay. Default ``"red"``.
    save_fig_name : str, optional
        If given, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    Fn = np.asarray(Fn)
    Zeta = np.asarray(Zeta)
    stab_label = np.asarray(stab_label)

    if figsize is None:
        figsize = (10, 6)
    fig, ax = plt.subplots(figsize=figsize)

    freq_flat = Fn.ravel()
    zeta_flat = Zeta.ravel()
    stable = (stab_label == 1).ravel()

    if not hide_poles:
        ax.plot(
            freq_flat[~stable],
            zeta_flat[~stable],
            ".",
            color=unstable_color,
            markersize=3,
            label="unstable pole",
        )
    ax.plot(
        freq_flat[stable],
        zeta_flat[stable],
        ".",
        color=stable_color,
        markersize=3,
        label="stable pole",
    )

    if sel_fn is not None and sel_zeta is not None:
        sf, sz = np.asarray(sel_fn).ravel(), np.asarray(sel_zeta).ravel()
        ok = np.isfinite(sf) & np.isfinite(sz)
        ax.plot(
            sf[ok],
            sz[ok],
            linestyle="none",
            marker="x",
            color=sel_color,
            markersize=8,
            markeredgewidth=1.2,
            label="selected mode",
            zorder=5,
        )

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Damping ratio")
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)

    ax.legend(loc="upper right", frameon=False, markerscale=3)
    plt.tight_layout()

    if save_fig_name:
        save_path = Path(save_fig_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def _cluster_colors(labels, cmap_name=None):
    """
    One colour per cluster id, keyed by label; noise (``-1``) is left out.

    Cluster ids are taken in ascending order so the colours track the
    frequency-sorted labels produced by :class:`clustering.DBSCAN`.
    """
    ids = sorted(int(v) for v in np.unique(labels) if v != -1)
    if cmap_name is None:
        cmap_name = "tab10" if len(ids) <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name)
    ncol = getattr(cmap, "N", 10)
    return {lab: cmap(i % ncol) for i, lab in enumerate(ids)}


def plot_stab_cluster(
    Fn_fl,
    order_fl,
    labels,
    ordmax,
    ordmin=0,
    freqs=None,
    data=None,
    indicator="cmif",
    y_label=None,
    indicator_color="tab:blue",
    cmap_name=None,
    plot_noise=False,
    noise_color="0.6",
    show_medians=True,
    figsize=None,
    xlim=None,
    save_fig_name=None,
):
    """
    Stabilisation chart with the poles coloured by cluster.

    Companion to :func:`plot_stab`: instead of stable/unstable, each pole is
    tinted by the DBSCAN cluster it belongs to, so the vertical columns of a
    physical mode read as one colour. A dashed line marks each cluster's median
    frequency.

    Shape notation follows ``plscf``: ``Q`` outputs, ``P`` references, ``N``
    frequency lines, ``n`` clustered poles.

    Parameters
    ----------
    Fn_fl : ndarray, shape (n,)
        Frequencies [Hz] of the clustered poles (``ClusterResult.Fn_fl``).
    order_fl : ndarray, shape (n,)
        Model order of each pole (``ClusterResult.order_fl``).
    labels : ndarray, shape (n,)
        Cluster id of each pole, ``-1`` for noise (``ClusterResult.labels``).
    ordmax : int
        Maximum model order (upper limit of the order axis).
    ordmin : int, optional
        Minimum model order (lower limit of the order axis). Default 0.
    freqs : ndarray, shape (N,), optional
        Frequency vector [Hz] for the overlaid indicator. Required to overlay.
    data : ndarray, shape (Q, P, N), optional
        FRF / output-spectra matrix, passed to :func:`_mode_indicator`. Required
        to overlay.
    indicator : {"cmif", "mmif", "sum"} or None, optional
        Mode-identification tool to overlay. None (or missing ``freqs``/``data``)
        draws the poles alone. Default ``"cmif"``.
    y_label : str, optional
        Label for the indicator (left) y-axis. Defaults to ``indicator``
        upper-cased.
    indicator_color : str, optional
        Colour for the indicator curves. Default ``"tab:blue"``.
    cmap_name : str, optional
        Qualitative colormap for the clusters. Defaults to ``"tab10"`` (up to 10
        clusters) or ``"tab20"``.
    plot_noise : bool, optional
        Draw the noise poles (label ``-1``) in ``noise_color``. Default False.
    noise_color : str, optional
        Colour of the noise poles. Default light grey.
    show_medians : bool, optional
        Draw a dashed vertical line at each cluster's median frequency. Default
        True.
    figsize : tuple, optional
        Figure size. Defaults to ``(10, 6)``.
    xlim : tuple(float, float), optional
        Frequency-axis limits. Defaults to the span of ``freqs``, else autoscale.
    save_fig_name : str, optional
        If given, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
        The pole (model-order) axis.
    """
    Fn_fl = np.asarray(Fn_fl)
    order_fl = np.asarray(order_fl)
    labels = np.asarray(labels)

    if figsize is None:
        figsize = (10, 6)
    fig, ax = plt.subplots(figsize=figsize)

    overlay = indicator is not None and freqs is not None and data is not None

    # ---- indicator on the left axis, poles on the right --------------------
    if overlay:
        freqs = np.asarray(freqs)
        curves = _mode_indicator(data, indicator)
        n = len(curves)
        for i, curve in enumerate(curves):
            alpha = 1.0 - 0.6 * i / max(n - 1, 1)
            ax.plot(
                freqs,
                curve,
                color=indicator_color,
                alpha=alpha,
                linewidth=1.0,
                zorder=1,
            )
        ax.set_yscale("linear" if indicator.lower() == "mmif" else "log")
        ax.set_ylabel(y_label if y_label is not None else indicator.upper())
        ax_pole = ax.twinx()  # drawn on top; poles overlay the curves
    else:
        ax_pole = ax

    # ---- poles, coloured by cluster ----
    colors = _cluster_colors(labels, cmap_name)

    if plot_noise:
        noise = labels == -1
        ax_pole.plot(
            Fn_fl[noise],
            order_fl[noise],
            "o",
            color=noise_color,
            markersize=3,
            alpha=0.4,
            markeredgewidth=0.0,
            label="noise",
            zorder=2,
        )
    for lab, color in colors.items():
        mask = labels == lab
        ax_pole.plot(
            Fn_fl[mask],
            order_fl[mask],
            "o",
            color=color,
            markersize=4,
            markeredgewidth=0.0,
            label=f"cluster {lab + 1}",
            zorder=3,
        )
        if show_medians:
            ax_pole.axvline(
                np.median(Fn_fl[mask]),
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
                zorder=2,
            )
    ax_pole.set_ylabel("Model order")
    ax_pole.set_ylim(ordmin, ordmax + 1)

    ax.set_xlabel("Frequency [Hz]")
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    elif overlay:
        ax.set_xlim(freqs[0], freqs[-1])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True)
    ax_pole.tick_params(which="both", direction="in")

    ax_pole.legend(
        loc="center left",
        bbox_to_anchor=(1.06 if overlay else 1.02, 0.5),
        frameon=False,
        markerscale=1.5,
        ncol=1 if len(colors) <= 14 else 2,
        title="Clusters",
    )
    plt.tight_layout()

    if save_fig_name:
        save_path = Path(save_fig_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax_pole


def plot_cluster_freqs_zetas(
    Fn_fl,
    Zeta_fl,
    labels,
    cmap_name=None,
    plot_noise=False,
    noise_color="0.6",
    sel_fn=None,
    sel_zeta=None,
    sel_color="red",
    figsize=None,
    xlim=None,
    save_fig_name=None,
):
    """
    Frequency-damping chart with the poles coloured by cluster.

    Companion to :func:`plot_freqs_zetas`: damping vs frequency, each pole tinted
    by its DBSCAN cluster so a physical mode reads as a tight same-colour blob.

    Shape notation follows ``plscf``: ``n`` clustered poles.

    Parameters
    ----------
    Fn_fl : ndarray, shape (n,)
        Frequencies [Hz] of the clustered poles (``ClusterResult.Fn_fl``).
    Zeta_fl : ndarray, shape (n,)
        Damping ratios of the clustered poles (``ClusterResult.Zeta_fl``).
    labels : ndarray, shape (n,)
        Cluster id of each pole, ``-1`` for noise (``ClusterResult.labels``).
    cmap_name : str, optional
        Qualitative colormap for the clusters. Defaults to ``"tab10"``/``"tab20"``.
    plot_noise : bool, optional
        Draw the noise poles (label ``-1``) in ``noise_color``. Default False.
    noise_color : str, optional
        Colour of the noise poles. Default light grey.
    sel_fn, sel_zeta : array_like, optional
        Frequencies [Hz] and damping ratios of the picked modes.
        Default None.
    sel_color : str, optional
        Colour of the picked-mode overlay. Default ``"red"``.
    figsize : tuple, optional
        Figure size. Defaults to ``(10, 6)``.
    xlim : tuple(float, float), optional
        Frequency-axis limits. Default None (autoscale).
    save_fig_name : str, optional
        If given, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    Fn_fl = np.asarray(Fn_fl)
    Zeta_fl = np.asarray(Zeta_fl)
    labels = np.asarray(labels)

    if figsize is None:
        figsize = (10, 6)
    fig, ax = plt.subplots(figsize=figsize)

    colors = _cluster_colors(labels, cmap_name)

    if plot_noise:
        noise = labels == -1
        ax.plot(
            Fn_fl[noise],
            Zeta_fl[noise],
            "o",
            color=noise_color,
            markersize=3,
            alpha=0.4,
            markeredgewidth=0.0,
            label="noise",
        )
    for lab, color in colors.items():
        mask = labels == lab
        ax.plot(
            Fn_fl[mask],
            Zeta_fl[mask],
            "o",
            color=color,
            markersize=4,
            markeredgewidth=0.0,
            label=f"cluster {lab + 1}",
        )

    if sel_fn is not None and sel_zeta is not None:
        sf, sz = np.asarray(sel_fn).ravel(), np.asarray(sel_zeta).ravel()
        ok = np.isfinite(sf) & np.isfinite(sz)
        ax.plot(
            sf[ok],
            sz[ok],
            linestyle="none",
            marker="x",
            color=sel_color,
            markersize=8,
            markeredgewidth=1.2,
            label="selected mode",
            zorder=5,
        )

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Damping ratio")
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        markerscale=1.5,
        ncol=1 if len(colors) <= 14 else 2,
        title="Clusters",
    )
    plt.tight_layout()

    if save_fig_name:
        save_path = Path(save_fig_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_k_distance(
    kdist,
    eps=None,
    k=None,
    line_color="tab:blue",
    eps_color="tab:red",
    figsize=None,
    save_fig_name=None,
):
    """
    k-distance (kNN) elbow for choosing the DBSCAN reachability radius ``eps``.

    Each pole's modal distance to its ``k``-th nearest neighbour, sorted
    descending (from :func:`modekit.clustering.k_distance`). The low
    plateau on the right is the dense poles inside modes; the sharp rise on the
    left is the sparse / noise poles. A good ``eps`` sits at the knee between
    them — the drawn ``eps`` line should meet the bend.

    Parameters
    ----------
    kdist : ndarray, shape (n,)
        Sorted-descending k-distance curve from ``clustering.k_distance``.
    eps : float, optional
        Reachability radius to mark with a horizontal line. Default None.
    k : int, optional
        Neighbour rank behind ``kdist`` (the DBSCAN ``min_pts``); labels the
        y-axis only.
    line_color : str, optional
        Colour of the k-distance curve. Default ``"tab:blue"``.
    eps_color : str, optional
        Colour of the ``eps`` line. Default ``"tab:red"``.
    figsize : tuple, optional
        Figure size. Defaults to ``(10, 6)``.
    save_fig_name : str, optional
        If given, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    kdist = np.asarray(kdist)

    if figsize is None:
        figsize = (10, 6)
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(np.arange(kdist.size), kdist, color=line_color, linewidth=1.5)
    if eps is not None:
        ax.axhline(
            eps,
            color=eps_color,
            linestyle="--",
            linewidth=1.2,
            label=rf"$\epsilon = {eps:g}$",
        )
        ax.legend(loc="upper right", frameon=False)

    ax.set_xlabel(r"Poles (sorted by descending $k$-distance)")
    ax.set_ylabel(rf"${k}$-NN distance" if k is not None else r"$k$-NN distance")
    ax.set_xlim(0, max(kdist.size - 1, 1))
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    plt.tight_layout()

    if save_fig_name:
        save_path = Path(save_fig_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_synthesis(
    freqs,
    measured,
    synthesized,
    pairs=None,
    phase=False,
    xlim=None,
    yscale="log",
    y_labels=None,
    measured_color="black",
    synth_color="tab:red",
    figsize=None,
    save_fig_name=None,
):
    """
    Overlay measured spectra against a fitted model's synthesis.

    Shape notation follows ``plscf``: ``Q`` outputs, ``P`` references, ``N``
    frequency lines.

    Parameters
    ----------
    freqs : ndarray, shape (N,)
        Frequency vector [Hz].
    measured : ndarray, shape (Q, P, N)
        Measured FRF / output-spectra matrix the model was fitted to.
    synthesized : ndarray, shape (Q, P, N)
        Reconstruction from :meth:`LSFD.synthesize`; NaN lines are skipped.
    pairs : sequence of (int, int), optional
        ``(output, reference)`` indices to draw, one subplot each. Defaults to
        every pair, row-major.
    phase : bool, optional
        If True, pair a phase panel under each magnitude panel
        (Avitabile-style), labelled ``"Phase [deg]"``. Default False.
    xlim : tuple(float, float), optional
        Frequency-axis limits. Defaults to the span of ``freqs``.
    yscale : str, optional
        Matplotlib scale of the magnitude axis. Default ``"log"``.
    y_labels : sequence of str, optional
        Y-axis label of each pair's magnitude subplot, one per pair (same
        length as ``pairs``). Default None: no labels.
    measured_color : str, optional
        Colour of the measured curve. Default ``"black"``.
    synth_color : str, optional
        Colour of the synthesized curve. Default ``"tab:red"``.
    figsize : tuple, optional
        Figure size. Defaults to ``(10, 2 * n_pairs)`` (``2.5`` with ``phase``).
    save_fig_name : str, optional
        If given, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray of matplotlib.axes.Axes
        The magnitude axes; with ``phase`` the phase axis of pair ``k`` is
        reachable as ``axes[k].phase_ax``.
    """
    freqs = np.asarray(freqs)
    measured = np.asarray(measured)
    synthesized = np.asarray(synthesized)
    if measured.shape != synthesized.shape:
        raise ValueError(
            f"measured {measured.shape} and synthesized {synthesized.shape} "
            f"must have the same shape."
        )
    Q, P, _ = measured.shape

    if pairs is None:
        pairs = [(q, p) for q in range(Q) for p in range(P)]
    n = len(pairs)
    if y_labels is not None and len(y_labels) != n:
        raise ValueError(f"y_labels has {len(y_labels)} entries for {n} pairs.")

    # With phase, each pair is a tall magnitude panel over a short phase panel.
    rows_per_pair = 2 if phase else 1
    if figsize is None:
        figsize = (10, (2.5 if phase else 2) * n)
    fig, axes = plt.subplots(
        rows_per_pair * n,
        1,
        figsize=figsize,
        sharex=True,
        squeeze=False,
        gridspec_kw={"height_ratios": [3, 1] * n if phase else None},
    )
    axes = axes.ravel()

    mag_axes = []
    for k, (q, p) in enumerate(pairs):
        ax_mag = axes[rows_per_pair * k]
        mag_axes.append(ax_mag)
        ax_mag.plot(
            freqs,
            np.abs(measured[q, p]),
            color=measured_color,
            linewidth=1.0,
            label="measured",
        )
        ax_mag.plot(
            freqs,
            np.abs(synthesized[q, p]),
            color=synth_color,
            linewidth=1.0,
            linestyle="--",
            label="synthesized",
        )
        ax_mag.set_yscale(yscale)
        if yscale == "linear":
            ax_mag.yaxis.set_minor_locator(AutoMinorLocator())
        ax_mag.tick_params(which="both", direction="in", top=True, right=True)
        if y_labels is not None:
            ax_mag.set_ylabel(y_labels[k])
        ax_mag.grid(True, linestyle="--", alpha=0.7)

        if phase:
            ax_ph = axes[rows_per_pair * k + 1]
            ax_mag.phase_ax = ax_ph
            ax_ph.plot(
                freqs,
                np.degrees(np.angle(measured[q, p])),
                color=measured_color,
                linewidth=1.0,
            )
            ax_ph.plot(
                freqs,
                np.degrees(np.angle(synthesized[q, p])),
                color=synth_color,
                linewidth=1.0,
                linestyle="--",
            )
            ax_ph.set_ylim(-180, 180)
            ax_ph.set_yticks([-180, -90, 0, 90, 180])
            ax_ph.tick_params(which="both", direction="in", top=True, right=True)
            ax_ph.set_ylabel("Phase [deg]")
            ax_ph.grid(True, linestyle="--", alpha=0.7)

    for ax in axes:
        ax.set_xlim(xlim if xlim is not None else (freqs[0], freqs[-1]))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    axes[-1].set_xlabel("Frequency [Hz]")

    handles, labels = mag_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_fig_name:
        save_path = Path(save_fig_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_mac_matrix(
    phi_X,
    phi_A,
    cmap_name="viridis",
    x_label="Mode (A)",
    y_label="Mode (X)",
    ticks_X=None,
    ticks_A=None,
    annotate=True,
    figsize=None,
    save_fig_name=None,
):
    """
    Modal Assurance Criterion (MAC) matrix between two mode-shape sets, as a heatmap.

    Shape notation follows ``plscf``: ``Q`` response locations, ``M`` modes.
    Entry ``(i, j)`` is the MAC between mode ``i`` of ``phi_X`` (rows, y-axis)
    and mode ``j`` of ``phi_A`` (columns, x-axis).

    Parameters
    ----------
    phi_X : ndarray, shape (Q,) or (Q, M_X)
        First mode-shape set, modes as columns.
    phi_A : ndarray, shape (Q,) or (Q, M_A)
        Second mode-shape set, modes as columns.
    cmap_name : str, optional
        Colormap for the heatmap. Default ``"viridis"``.
    x_label, y_label : str, optional
        Axis labels for the ``phi_A`` / ``phi_X`` mode indices.
    ticks_X, ticks_A : sequence, optional
        Tick labels of the ``phi_X`` rows / ``phi_A`` columns, e.g. the modes'
        frequencies. Default: 1-based mode indices.
    annotate : bool, optional
        Write each MAC value in its cell. Default True.
    figsize : tuple, optional
        Figure size. Defaults to ``(8, 6)``.
    save_fig_name : str, optional
        If given, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    # atleast_2d keeps the single-mode case (a scalar) plottable.
    mac = np.atleast_2d(np.asarray(criteria.mac(phi_X, phi_A)))
    n_X, n_A = mac.shape

    if figsize is None:
        figsize = (8, 6)
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(mac, cmap=cmap_name, vmin=0.0, vmax=1.0, aspect="auto")
    fig.colorbar(im, ax=ax, label="MAC")

    ax.set_xticks(np.arange(n_A))
    ax.set_xticklabels(
        np.arange(1, n_A + 1) if ticks_A is None else list(ticks_A), rotation=90
    )
    ax.set_yticks(np.arange(n_X))
    ax.set_yticklabels(np.arange(1, n_X + 1) if ticks_X is None else list(ticks_X))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(which="both", direction="in")

    if annotate:
        for i in range(n_X):
            for j in range(n_A):
                val = mac[i, j]
                if not np.isfinite(val):
                    continue
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white" if val < 0.5 else "black",
                    fontsize=8,
                )

    plt.tight_layout()

    if save_fig_name:
        save_path = Path(save_fig_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_mode_complexity(
    phi,
    align=True,
    arrow_color="tab:blue",
    figsize=None,
    save_fig_name=None,
):
    """
    Polar plot of a complex mode shape's phase scatter (its "complexity").

    Each DOF is an arrow from the origin at angle = phase, length = magnitude.

    Shape notation follows ``plscf``: ``Q`` response locations.

    Parameters
    ----------
    phi : ndarray, shape (Q,)
        Complex mode shape, one entry per DOF. Non-finite entries are skipped.
    align : bool, optional
        Remove the arbitrary global phase. Default True.
    arrow_color : str, optional
        Colour of the per-DOF arrows. Default ``"tab:blue"``.
    figsize : tuple, optional
        Figure size. Defaults to ``(6, 6)``.
    save_fig_name : str, optional
        If given, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
        The polar axes.
    """
    phi = np.asarray(phi, dtype=complex).ravel()
    phi = phi[np.isfinite(phi)]
    if align:
        # Ahmadian step 1: the phase that maximises the real part; keep the imag.
        phi = phi * np.exp(1j * -0.5 * np.angle(np.sum(phi**2)))
    angles = np.angle(phi)
    magnitudes = np.abs(phi)
    rmax = magnitudes.max() * 1.1 if magnitudes.size and magnitudes.max() > 0 else 1.1

    if figsize is None:
        figsize = (6, 6)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=figsize)
    ax.set_theta_zero_location("E")  # 0 deg at East (positive real axis)
    ax.set_theta_direction(1)  # counter-clockwise
    ax.set_rmax(rmax)
    ax.set_yticklabels([])
    ax.grid(True, linestyle="--", alpha=0.5)

    for angle, magnitude in zip(angles, magnitudes):
        ax.annotate(
            "",
            xy=(angle, magnitude),
            xytext=(angle, 0.0),
            arrowprops=dict(
                color=arrow_color,
                arrowstyle="-|>",
                linewidth=1.5,
                mutation_scale=20,
            ),
        )

    # the real axis (0 deg / 180 deg): a normal mode lies entirely on this line
    for a in (0.0, np.pi):
        ax.plot(
            [a, a], [0.0, rmax], color="black", linestyle="--", linewidth=1, alpha=0.6
        )

    plt.tight_layout()

    if save_fig_name:
        save_path = Path(save_fig_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def _style_3d_axes(ax):
    """Publication styling for a 3D axes"""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor("0.85")
        axis.set_major_locator(MaxNLocator(nbins=5))  # uncrowded, round majors
        axis.set_minor_locator(AutoMinorLocator(2))  # one minor tick per interval
        try:  # private but stable
            axis._axinfo["grid"].update(color="0.85", linestyle="--", linewidth=0.6)
        except Exception:
            pass
    ax.grid(True)
    ax.tick_params(which="both", direction="in", colors="0.2", labelcolor="black")


def plot_mode_wireframe(
    undeformed,
    deformed,
    sensor_points=None,
    sensor_base=None,
    arrows=True,
    zero_z=True,
    undeformed_color="0.6",
    deformed_color="#2b2d42",
    sensor_color="#2b2d42",
    arrow_pos_color="#e63946",
    arrow_neg_color="#457b9d",
    elev=22,
    azim=-60,
    show_axes=True,
    figsize=None,
    save_fig_name=None,
):
    """
    3D wireframe of a structure's undeformed skeleton and a deformed mode shape.

    Geometry-agnostic: the caller supplies the polylines.

    Parameters
    ----------
    undeformed, deformed : sequence of ndarray, each shape (n_i, 3)
        Matching lists of ``(x, y, z)`` polylines for the reference and deformed
        skeleton. A static member simply has ``deformed[i] == undeformed[i]``.
    sensor_points : ndarray, shape (S, 3), optional
        Deformed coordinates of the measured nodes.
    sensor_base : ndarray, shape (S, 3), optional
        Undeformed sensor coordinates. When omitted, markers fall back
        to ``sensor_points`` and no arrows are drawn.
    arrows : bool, optional
        Draw a displacement arrow at each sensor from ``sensor_base`` to
        ``sensor_points``. Needs both. Default True.
    zero_z : bool, optional
        Shift the vertical axis so the structure's lowest point reads ``0`` mm.
        Default True.
    undeformed_color, deformed_color : str, optional
        Colours of the reference (light grey) and deformed (dark neutral) skeletons.
    sensor_color : str, optional
        Edge colour of the sensor markers. Default ``"#2b2d42"``.
    arrow_pos_color, arrow_neg_color : str, optional
        Arrow colours for displacement. Defaults warm red / steel blue.
    elev, azim : float, optional
        3D view angles [deg]. Defaults 22 / -60 (isometric).
    show_axes : bool, optional
        Draw the styled axes. If False, a clean skeleton-only view. Default True.
    figsize : tuple, optional
        Figure size. Defaults to ``(7, 6)``.
    save_fig_name : str, optional
        If given, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
    """
    undeformed = [np.asarray(u, dtype=float) for u in undeformed]
    deformed = [np.asarray(d, dtype=float) for d in deformed]
    base = None if sensor_base is None else np.asarray(sensor_base, dtype=float)
    tip = None if sensor_points is None else np.asarray(sensor_points, dtype=float)

    # re-zero the vertical axis to the structure's lowest point, so z starts at 0
    if zero_z:
        zmin = min(u[:, 2].min() for u in undeformed)
        if base is not None:
            zmin = min(zmin, float(base[:, 2].min()))
        shift = np.array([0.0, 0.0, zmin])
        undeformed = [u - shift for u in undeformed]
        deformed = [d - shift for d in deformed]
        base = None if base is None else base - shift
        tip = None if tip is None else tip - shift

    if figsize is None:
        figsize = (7, 6)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    # honour the explicit zorder
    ax.computed_zorder = False

    # reference skeleton; deformed skeleton on top
    for u in undeformed:
        ax.plot(
            u[:, 0],
            u[:, 1],
            u[:, 2],
            color=undeformed_color,
            linewidth=1.0,
            linestyle="--",
            zorder=1,
        )
    for d in deformed:
        ax.plot(
            d[:, 0], d[:, 1], d[:, 2], color=deformed_color, linewidth=1.6, zorder=3
        )

    # displacement arrows
    if arrows and base is not None and tip is not None:
        vec = tip - base
        dominant = np.argmax(np.abs(vec), axis=1)
        sign = np.sign(vec[np.arange(len(vec)), dominant])
        for keep, color in ((sign >= 0, arrow_pos_color), (sign < 0, arrow_neg_color)):
            if keep.any():
                ax.quiver(
                    base[keep, 0],
                    base[keep, 1],
                    base[keep, 2],
                    vec[keep, 0],
                    vec[keep, 1],
                    vec[keep, 2],
                    color=color,
                    linewidth=0.7,
                    arrow_length_ratio=0.2,
                    normalize=False,
                    zorder=4,
                )

    # measured nodes
    marker = base if base is not None else tip
    if marker is not None:
        ax.scatter(
            marker[:, 0],
            marker[:, 1],
            marker[:, 2],
            facecolors="white",
            edgecolors=sensor_color,
            s=22,
            linewidths=0.9,
            depthshade=False,
            zorder=6,
        )

    # true proportions and axes limits
    pts = np.vstack(undeformed + deformed + [m for m in (base, tip) if m is not None])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    pad = (
        span - (hi - lo)
    ) / 2  # a flat structure: its flat axis gets the unit span, centred
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
    ax.set_box_aspect(tuple(span))
    ax.view_init(elev=elev, azim=azim)

    if show_axes:
        _style_3d_axes(ax)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")
    else:
        ax.set_axis_off()

    plt.tight_layout()

    if save_fig_name:
        save_path = Path(save_fig_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax
