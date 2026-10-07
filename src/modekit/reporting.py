"""
Tables of identification results.

Builds the recurring report tables of modal identification as tidy
:class:`pandas.DataFrame` objects; :mod:`modekit.export` renders and
writes them.

- :func:`mode_table` — the modes of one or more fits, one row per (test, mode)
- :func:`tracking_table` — per-mode statistics of one condition's segment tracking
- :func:`tracking_summary` — one row per tracked condition
- :func:`match_table` — modes matched across structures by frequency ratio
- :func:`scale_table` — pairwise frequency scale between the structures of a match

Every table records how :mod:`modekit.export` renders its columns in
``df.attrs["formats"]``: fractions (``in_band``, ``mac_ok``, ...) and damping
ratios are stored as plain ratios and rendered as percentages.
"""

from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from modekit import criteria, matching

# A fit is anything exposing ``Fn`` / ``Zeta`` (``ModalModel``), a wrapper with
# a ``model`` attribute (``LSFDFit``), or a plain ``(Fn, Zeta)`` pair.
FitLike = Union[object, Tuple[Sequence[float], Sequence[float]]]

# the rendering of the tracking statistics (export.to_text); each table attaches
# the formats of the columns it has in attrs["formats"]
_TRACKING_FORMATS = {
    "f_ref": "{:.1f}", "in_band": "{:.0%}", "fn_med": "{:.1f}", "fn_iqr": "{:.2f}",
    "zeta_med": "{:.2%}", "conf": "{:.2f}", "mac": "{:.2f}", "mac_ok": "{:.0%}",
    "sample_mac": "{:.2f}", "sample_mac_ok": "{:.0%}",
}


# =============================================================================
# input adapters
# =============================================================================


def _unwrap(fit):
    """An ``LSFDFit`` (``est, model, refs``) reports through its model."""
    return fit.model if hasattr(fit, "model") and not hasattr(fit, "Fn") else fit


def _fn_zeta(fit: FitLike) -> Tuple[np.ndarray, np.ndarray]:
    """``(Fn [Hz], Zeta [-])`` of a fit, finite modes only, ascending frequency."""
    if isinstance(fit, (tuple, list)):
        fn, zeta = fit
    else:
        fit = _unwrap(fit)
        fn, zeta = fit.Fn, fit.Zeta
    fn = np.asarray(fn, dtype=float).ravel()
    zeta = np.asarray(zeta, dtype=float).ravel()
    ok = np.isfinite(fn)
    fn, zeta = fn[ok], zeta[ok]
    order = np.argsort(fn)
    return fn[order], zeta[order]


def _fn_phi(fit) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    ``(Fn [Hz], Phi (Q, M))`` of a fit, in the fit's own mode order.

    A model gives its real, max-normalised shapes; a ``(fn, phi)`` pair is
    used as given (``phi`` may be None).
    """
    if isinstance(fit, (tuple, list)):
        fn, phi = fit
        phi = None if phi is None else np.asarray(phi)
    else:
        fit = _unwrap(fit)
        fn = fit.Fn
        phi = np.asarray(fit.mode_shapes(real=True, normalize="max"))
    return np.asarray(fn, dtype=float).ravel(), phi


def _tracking_arrays(seg, phi=None, phi_ref=None) -> dict:
    """
    The per-(segment, mode) arrays a tracking report is built from.

    ``seg`` is a :class:`~modekit.bandmpe.LocalModes` or any object with
    ``Fn`` / ``Zeta`` ``(n_seg, M)`` — a modal-store condition, say. ``ok``,
    ``conf``, ``mac`` and ``f_ref`` are read when present (``Fn_ref`` also
    counts as ``f_ref``); without ``ok``, finite ``Fn`` entries are in band.
    The sample MAC pairs ``phi`` ``(n_seg, Q, M)`` with ``phi_ref`` ``(Q, M)``;
    when neither is given, an object carrying both ``Phi`` and ``Phi_ref`` (a
    store condition) supplies them.
    """
    Fn = np.asarray(seg.Fn, dtype=float)
    Zeta = np.asarray(seg.Zeta, dtype=float)
    if Fn.ndim != 2:
        raise ValueError(f"Fn must be (n_seg, M), got shape {Fn.shape}")
    ok = getattr(seg, "ok", None)
    ok = np.isfinite(Fn) if ok is None else np.asarray(ok, dtype=bool)
    f_ref = getattr(seg, "f_ref", None)
    if f_ref is None:
        f_ref = getattr(seg, "Fn_ref", None)
    conf = getattr(seg, "conf", None)
    mac = getattr(seg, "mac", None)

    if phi is None and phi_ref is None and hasattr(seg, "Phi_ref"):
        phi, phi_ref = seg.Phi, seg.Phi_ref
    if (phi is None) != (phi_ref is None):
        raise ValueError("pass phi and phi_ref together")
    sample_mac = (
        None
        if phi is None
        else np.asarray(criteria.mac_paired(phi, phi_ref), dtype=float)
    )
    return {
        "Fn": Fn,
        "Zeta": Zeta,
        "ok": ok,
        "f_ref": None if f_ref is None else np.asarray(f_ref, dtype=float).ravel(),
        "conf": None if conf is None else np.asarray(conf, dtype=float),
        "mac": None if mac is None else np.asarray(mac, dtype=float),
        "sample_mac": sample_mac,
    }


def _masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The finite entries of ``values`` where ``mask`` holds."""
    v = values[mask]
    return v[np.isfinite(v)]


def _mean(values, mask) -> float:
    v = _masked(values, mask)
    return float(v.mean()) if v.size else np.nan


def _share_above(values, mask, lim) -> float:
    v = _masked(values, mask)
    return float((v > lim).mean()) if v.size else np.nan


# =============================================================================
# tables
# =============================================================================


def mode_table(models: Mapping[str, FitLike]) -> pd.DataFrame:
    """
    Tidy table of identified modes, one row per ``(test, mode)``.

    Parameters
    ----------
    models : mapping {str: fit}
        ``{test_id: ModalModel}``, ``{test_id: LSFDFit}`` or
        ``{test_id: (Fn, Zeta)}``.

    Returns
    -------
    pandas.DataFrame
        Columns ``test``, ``mode`` (1-based, ascending frequency), ``fn_hz``,
        ``zeta`` (ratio) and ``zeta_pct``.
    """
    rows = []
    for test, fit in models.items():
        fn, zeta = _fn_zeta(fit)
        for i, (f, z) in enumerate(zip(fn, zeta), start=1):
            rows.append(
                {"test": test, "mode": i, "fn_hz": f, "zeta": z, "zeta_pct": z * 100.0}
            )
    df = pd.DataFrame(rows, columns=["test", "mode", "fn_hz", "zeta", "zeta_pct"])
    df.attrs["formats"] = {"fn_hz": "{:.2f}", "zeta": "{:.4f}", "zeta_pct": "{:.2f}"}
    return df


def tracking_table(seg, phi=None, phi_ref=None, *, mac_lim: float = 0.8) -> pd.DataFrame:
    """
    Per-mode statistics of one condition's segment tracking.

    Every statistic is taken over the in-band samples of the mode (``ok``);
    ``in_band`` is their share of all segments.

    Parameters
    ----------
    seg : LocalModes or store condition
        The tracking, from :func:`~modekit.bandmpe.plscf_local`, or any
        object with ``Fn`` / ``Zeta`` ``(n_seg, M)`` such as a condition of the
        modal store (its finite entries are the in-band samples).
    phi : array_like, optional
        The samples' mode shapes ``(n_seg, Q, M)``, e.g. from
        :func:`~modekit.bandmpe.lsfd_batch`.
    phi_ref : array_like, optional
        Reference shapes ``(Q, M)`` the samples are MAC-graded against. Given
        with ``phi``; a store condition supplies both when neither is passed.
    mac_lim : float, optional
        MAC threshold of the ``*_ok`` shares. Default 0.8.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``mode`` (aligned with the columns of ``Fn``). Columns
        ``f_ref`` (the reference frequency, else the in-band median),
        ``in_band``, ``fn_med``, ``fn_iqr`` [Hz], ``zeta_med`` (ratio), then
        ``conf``, ``mac`` and ``mac_ok`` when ``seg`` grades its estimates, and
        ``sample_mac`` / ``sample_mac_ok`` when shapes are available.
    """
    a = _tracking_arrays(seg, phi, phi_ref)
    Fn, Zeta, ok = a["Fn"], a["Zeta"], a["ok"]
    M = Fn.shape[1]
    rows = []
    for k in range(M):
        m = ok[:, k]
        fn_k = _masked(Fn[:, k], m)
        q = np.percentile(fn_k, [25, 50, 75]) if fn_k.size else np.full(3, np.nan)
        row = {
            "f_ref": a["f_ref"][k] if a["f_ref"] is not None else q[1],
            "in_band": float(m.mean()),
            "fn_med": q[1],
            "fn_iqr": q[2] - q[0],
            "zeta_med": float(np.median(z)) if (z := _masked(Zeta[:, k], m)).size else np.nan,
        }
        if a["conf"] is not None:
            row["conf"] = _mean(a["conf"][:, k], m)
        if a["mac"] is not None:
            row["mac"] = _mean(a["mac"][:, k], m)
            row["mac_ok"] = _share_above(a["mac"][:, k], m, mac_lim)
        if a["sample_mac"] is not None:
            row["sample_mac"] = _mean(a["sample_mac"][:, k], m)
            row["sample_mac_ok"] = _share_above(a["sample_mac"][:, k], m, mac_lim)
        rows.append(row)
    df = pd.DataFrame(rows, index=pd.RangeIndex(M, name="mode"))
    df.attrs["formats"] = {c: f for c, f in _TRACKING_FORMATS.items() if c in df.columns}
    return df


def tracking_summary(seg, phi=None, phi_ref=None, *, mac_lim: float = 0.8) -> pd.DataFrame:
    """
    One row per tracked condition: size, in-band share and mean grades.

    The means and shares pool every in-band ``(segment, mode)`` sample of the
    condition (so modes with more in-band samples weigh more).

    Parameters
    ----------
    seg : LocalModes or mapping {label: LocalModes}
        One condition's tracking, or several keyed by a label. Tuple labels
        such as ``(wind, structure)`` become a ``MultiIndex``. Any object with
        ``Fn`` / ``Zeta`` ``(n_seg, M)`` serves, as for :func:`tracking_table`.
    phi, phi_ref : array_like or mapping, optional
        The samples' shapes ``(n_seg, Q, M)`` and their references ``(Q, M)``,
        as for :func:`tracking_table`. With a mapping ``seg``, each may be a
        mapping with the same labels, or one array shared by every condition.
    mac_lim : float, optional
        MAC threshold of the ``*_ok`` shares. Default 0.8.

    Returns
    -------
    pandas.DataFrame
        Columns ``n_seg``, ``n_modes``, ``in_band``, then ``conf``, ``mac``,
        ``mac_ok`` and ``sample_mac``, ``sample_mac_ok`` where available.
    """
    if isinstance(seg, Mapping):
        items = list(seg.items())
    else:
        items = [(0, seg)]

    def pick(x, key):
        return x[key] if isinstance(x, Mapping) else x

    rows, keys = [], []
    for key, s in items:
        a = _tracking_arrays(s, pick(phi, key), pick(phi_ref, key))
        ok = a["ok"]
        row = {
            "n_seg": ok.shape[0],
            "n_modes": ok.shape[1],
            "in_band": float(ok.mean()),
        }
        if a["conf"] is not None:
            row["conf"] = _mean(a["conf"], ok)
        if a["mac"] is not None:
            row["mac"] = _mean(a["mac"], ok)
            row["mac_ok"] = _share_above(a["mac"], ok, mac_lim)
        if a["sample_mac"] is not None:
            row["sample_mac"] = _mean(a["sample_mac"], ok)
            row["sample_mac_ok"] = _share_above(a["sample_mac"], ok, mac_lim)
        rows.append(row)
        keys.append(key)

    if keys and all(isinstance(k, tuple) for k in keys):
        index = pd.MultiIndex.from_tuples(keys)
    else:
        index = pd.Index(keys)
    order = [
        "n_seg", "n_modes", "in_band", "conf", "mac", "mac_ok", "sample_mac", "sample_mac_ok"
    ]
    df = pd.DataFrame(rows, index=index)
    df = df[[c for c in order if c in df.columns]]
    df.attrs["formats"] = {c: f for c, f in _TRACKING_FORMATS.items() if c in df.columns}
    return df


_MATCH_FIXED = ("ratio", "mac", "flag")  # non-structure columns of a match table


def match_table(
    models: Mapping[str, object],
    *,
    rtol: float = 0.05,
    mac_lo: float = 0.6,
    mac_weight: Optional[float] = None,
    catalogue: Optional[Mapping[str, np.ndarray]] = None,
) -> pd.DataFrame:
    """
    Match the modes of several structures by frequency ratio, or label them
    against a catalogue.

    Wraps :func:`~modekit.matching.match_modes`: modes are grouped into
    slots by ``f_n / f_1``, which cancels the material scale of structures
    sharing a geometry; the MAC breaks ties and flags slots whose shapes
    disagree. A structure may miss a slot. With ``catalogue`` the rows are
    fixed instead, and each structure's modes are assigned to its column
    (:func:`~modekit.matching.assign_catalogue`).

    Parameters
    ----------
    models : mapping {structure: fit}
        ``{name: ModalModel}`` or ``{name: LSFDFit}`` (real, max-normalised
        shapes read off the model), or ``{name: (fn, phi)}`` with shapes
        ``(Q, M)`` used as given (``phi=None`` everywhere skips the MAC). The
        names become the table's columns.
    rtol : float, optional
        Relative tolerance on the ratio; with a catalogue, on the frequency.
        Default 0.05.
    mac_lo : float, optional
        Slots whose worst pairwise MAC is below this are flagged ``"?"``.
        Default 0.6.
    mac_weight : float, optional
        MAC tie-break weight, see :func:`~modekit.matching.match_modes`.
    catalogue : mapping {structure: array_like}, optional
        Nominal frequencies per structure, ``(n_row,)``, NaN where the mode is
        absent; same keys as ``models``. Frequencies no row claims are kept in
        ``attrs["unassigned"]``.

    Returns
    -------
    pandas.DataFrame
        One row per ``slot``: ``ratio`` (``f / f1``; with a catalogue, the
        row's nominal), one frequency column [Hz] per structure (NaN where
        absent), ``mac`` (worst pairwise MAC, NaN with fewer than two
        structures) and ``flag``.
    """
    fn, phi = {}, {}
    for name, fit in models.items():
        fn[name], phi[name] = _fn_phi(fit)
    have_phi = [p is not None for p in phi.values()]
    if any(have_phi) and not all(have_phi):
        raise ValueError("give mode shapes for every structure, or for none")
    shapes = phi if all(have_phi) else None

    if catalogue is None:
        m = matching.match_modes(fn, shapes, rtol=rtol, mac_weight=mac_weight)
    else:
        m = matching.assign_catalogue(fn, catalogue, rtol=rtol)
    n_slot = m.ratio.size
    df = pd.DataFrame({"ratio": m.ratio}, index=pd.RangeIndex(n_slot, name="slot"))
    for j, s in enumerate(m.names):
        idx = m.index[:, j]
        col = np.full(n_slot, np.nan)
        col[idx >= 0] = fn[s][idx[idx >= 0]]
        df[s] = col
    mac = matching.slot_mac(m, shapes) if shapes is not None else np.full(n_slot, np.nan)
    df["mac"] = mac
    df["flag"] = np.where(np.isfinite(mac) & (mac < mac_lo), "?", "")
    df.attrs["formats"] = {"ratio": "{:.2f}", **{s: "{:.1f}" for s in m.names}, "mac": "{:.2f}"}
    if catalogue is not None:
        df.attrs["unassigned"] = {
            s: fn[s][[j for j in np.flatnonzero(np.isfinite(fn[s])) if j not in set(m.index[:, k])]]
            for k, s in enumerate(m.names)
        }
    return df


def scale_table(match: pd.DataFrame, structs: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Frequency scale between each pair of structures of a match table.

    For structures sharing a geometry the ratio ``f_a / f_b`` of matched modes
    is the material scale ``sqrt(E_a rho_b / (E_b rho_a))`` and should be the
    same for every mode; its spread is a check on the matching.

    Parameters
    ----------
    match : pandas.DataFrame
        A :func:`match_table`.
    structs : sequence of str, optional
        The structure columns to pair, in order. Default: every column but
        ``ratio``, ``mac`` and ``flag``.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(a, b)``: ``scale`` (the median of ``f_a / f_b`` over the
        jointly resolved slots), ``spread`` (their standard deviation) and
        ``n`` (their number).
    """
    cols = (
        [c for c in match.columns if c not in _MATCH_FIXED]
        if structs is None
        else list(structs)
    )
    rows, keys = [], []
    for k, a in enumerate(cols):
        for b in cols[k + 1 :]:
            fa, fb = match[a].to_numpy(float), match[b].to_numpy(float)
            both = np.isfinite(fa) & np.isfinite(fb)
            r = fa[both] / fb[both]
            rows.append(
                {
                    "scale": float(np.median(r)) if r.size else np.nan,
                    "spread": float(r.std()) if r.size else np.nan,
                    "n": int(both.sum()),
                }
            )
            keys.append((a, b))
    df = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(keys, names=["a", "b"]),
        columns=["scale", "spread", "n"],
    )
    df.attrs["formats"] = {"scale": "{:.3f}", "spread": "{:.3f}"}
    return df
