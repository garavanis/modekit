"""
Band-wise modal parameter estimation on top of :mod:`modekit.algorithms`.

Poles are curve-fitted in small frequency bands, clustered into modes per band,
merged across bands and passes, and finally handed to a single full-band
:class:`~modekit.algorithms.LSFD` residue fit.

Serves both EMA FRFs and OMA output spectra. :func:`plscf_local` then fits
local pLSCF models around the discovered modes on batches of segments (one
jitted ``jax.lax.map``), and :func:`lsfd_batch` refits the residues per segment
at those tracked poles, so the segments' mode shapes are the same kind of
estimate as the full-signal reference.

Fits are cached as equinox files (JSON hyperparams header + serialised leaves)
through :mod:`modekit.serialisation`.

Shape notation follows ``plscf``: ``Q`` outputs, ``P`` references, ``N``
frequency lines, ``M`` modes, plus ``n_seg`` segments here.
"""

import dataclasses
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from modekit import clustering, plscf
from modekit.criteria import _mac_elementwise
from modekit.algorithms import (
    DEFAULT_HC,
    DEFAULT_SC,
    LSFD,
    ModalModel,
    PoleArrays,
    pLSCF,
)
from modekit.serialisation import cache_key, load_model, save_model


class BandFit(eqx.Module):
    """
    One band's fit: the estimator, its screened poles and their clustering.

    Iterates as the tuple ``(flo, fhi, est, poles, res)``, so the plot loops
    unpack it like::

        for flo, fhi, est, poles, res in band_data:
            est.stab_plot(poles, ...)
    """

    flo: float = eqx.field(static=True)
    fhi: float = eqx.field(static=True)
    est: pLSCF
    poles: PoleArrays
    res: clustering.ClusterResult

    def __iter__(self):
        return iter((self.flo, self.fhi, self.est, self.poles, self.res))


class LocalModes(eqx.Module):
    """
    Modal parameters of the local pLSCF models over a batch of records,
    aligned with ``f_ref``. The records are the segments of a condition, or a
    single full record (batch of one) when :func:`local_factors` re-estimates
    known poles on the LSFD data.

    :func:`plscf_local` returns an estimate wherever a local model produced a
    stable pole (``Lambd`` is NaN only where none existed); ``ok`` says whether
    that pole landed inside the model's band, and ``conf`` / ``mac`` grade it.
    """

    f_ref: jax.Array  # reference frequencies [Hz], (M,)
    Lambd: jax.Array  # complex poles, (n_seg, M)
    Phi: jax.Array  # (n_seg, Q, M)
    Lr: jax.Array  # (n_seg, P, M)
    ok: jax.Array  # (n_seg, M) bool
    conf: Optional[jax.Array] = None  # local-fit goodness of fit in [0, 1], (n_seg, M)
    mac: Optional[jax.Array] = None  # MAC against phi_ref, (n_seg, M); None without it

    @property
    def Fn(self) -> jax.Array:
        """Natural frequencies [Hz], (n_seg, M)."""
        return plscf.lambd_to_fn(self.Lambd)

    @property
    def Zeta(self) -> jax.Array:
        """Damping ratios, (n_seg, M)."""
        return plscf.lambd_to_zeta(self.Lambd)


class LSFDFit(eqx.Module):
    """
    The residue stage: the LSFD estimator and the modal model it fitted.

    Iterates as the tuple ``(est, model)``, so the synthesis plot reads::

        est, model = fit
        est.synthesis_plot(model, data, ...)
    """

    est: LSFD
    model: ModalModel
    # local pLSCF models that gave the reference factors; None with refs='global'
    refs: Optional[LocalModes] = None

    def __iter__(self):
        return iter((self.est, self.model))

    def restrict(self, outputs, refs) -> "LSFDFit":
        """
        The fit on a subset of its channels, the poles untouched.

        For a record that carries fewer channels than the reference fit (a dead
        accelerometer), so its segments can be tracked and stored against the
        same modes: the residues, residuals, local shapes and reference factors
        are cut to the kept channels. A rank-one residue stays rank one, so
        ``model.mode_shapes()`` gives the reference shapes on the kept outputs.

        Parameters
        ----------
        outputs : sequence of int
            Output rows to keep, in the fit's own numbering.
        refs : sequence of int
            Reference columns to keep, positions in the fit's reference set.

        Returns
        -------
        LSFDFit
            Same estimator, cut model and (when present) cut local models.
        """
        ko, kr = jnp.asarray(outputs), jnp.asarray(refs)

        def sub(a):
            return jnp.take(jnp.take(a, ko, 0), kr, 1)

        m = self.model
        model = ModalModel(
            Lambd=m.Lambd,
            A=sub(m.A),
            LR=sub(m.LR),
            UR=sub(m.UR),
            A_mirror=None if m.A_mirror is None else sub(m.A_mirror),
        )
        r = self.refs
        if r is not None:
            r = LocalModes(
                r.f_ref,
                r.Lambd,
                jnp.take(r.Phi, ko, 1),
                jnp.take(r.Lr, kr, 1),
                r.ok,
                r.conf,
                r.mac,
            )
        return LSFDFit(self.est, model, r)


class BandMPE(eqx.Module):
    """
    One :func:`band_mpe` result: the merged modes and the per-band fits.

    Iterates as the tuple ``(Lambd, conf, bands)``, so a pass unpacks like::

        Lambd, conf, bands = band_mpe(...)
    """

    Lambd: jax.Array  # combined physical poles, (M,)
    conf: jax.Array  # poles per cluster, (M,)
    bands: List[BandFit]

    def __iter__(self):
        return iter((self.Lambd, self.conf, self.bands))

    @property
    def Lr(self) -> jax.Array:
        """Reference factors ``(P, M)`` of the combined poles, in the same order."""
        return jnp.asarray(_gather(self.bands, "Lr"))


# =============================================================================
# eqx cache plumbing (serialisation.save_model / load_model)
# =============================================================================


def _spec(module) -> dict:
    """``{field: [shape, dtype]}`` of an eqx.Module's array leaves."""
    out = {}
    for f in dataclasses.fields(module):
        v = getattr(module, f.name)
        if eqx.is_array(v):
            out[f.name] = [list(v.shape), str(v.dtype)]
    return out


def _zeros(spec) -> dict:
    """Placeholder arrays for a :func:`_spec` dict, to be deserialised into."""
    return {
        k: jnp.zeros(tuple(shape), dtype=np.dtype(d)) for k, (shape, d) in spec.items()
    }


def _make_band_mpe(*, spectrum, alpha, ordmax, ordmin, sc, hc, bands, Lambd, conf):
    """Skeleton builder for :func:`load_model`: a :class:`BandMPE` shell.

    The pLSCF constructors are cheap (no fitting runs), so they rebuild the
    static fields and the ``sc``/``hc`` leaves directly; every array leaf is a
    zeros placeholder that ``tree_deserialise_leaves`` overwrites.
    """
    fits = []
    for b in bands:
        est = pLSCF(
            freqs=_zeros(b["est"])["freqs"],
            fs=b["fs_fake"],
            ordmax=ordmax,
            ordmin=ordmin,
            spectrum=spectrum,
            alpha=alpha,
            sc=sc,
            hc=hc,
        )
        fits.append(
            BandFit(
                b["flo"],
                b["fhi"],
                est,
                PoleArrays(**_zeros(b["poles"])),
                clustering.ClusterResult(**_zeros(b["res"])),
            )
        )
    (shape, d), (cshape, cd) = Lambd, conf
    return BandMPE(
        jnp.zeros(tuple(shape), np.dtype(d)),
        jnp.zeros(tuple(cshape), np.dtype(cd)),
        fits,
    )


def _cache_file(cache, cache_dir, key) -> Path:
    if cache_dir is None:
        raise ValueError("cache_dir is required when cache is set")
    return Path(cache_dir) / f"{cache}-{key}.eqx"


def _data_fingerprint(*arrays) -> str:
    return hashlib.md5(b"".join(np.asarray(a).tobytes() for a in arrays)).hexdigest()


# =============================================================================
# Band-wise pLSCF
# =============================================================================


def band_mpe(
    freqs,
    data,
    alpha: Optional[float],
    bands,
    *,
    spectrum: str = "frf_tap",
    ordmax: int = 100,
    ordmin: int = 5,
    sc: Optional[dict] = None,
    hc: Optional[dict] = None,
    eps: float = 0.02,
    min_pts: int = 8,
    select: str = "density",
    cache: Optional[str] = None,
    cache_dir=None,
    refit: bool = False,
    progress: bool = True,
) -> BandMPE:
    """
    Band-wise pLSCF: fit each sub-band (fake-dt normalised), cluster its poles,
    and combine the modes.

    Serves both the EMA FRFs and the OMA output spectra.

    With ``cache`` set, the result is stored as an equinox file under
    ``cache_dir`` (via :func:`modekit.serialisation.save_model`).

    Parameters
    ----------
    freqs : array_like
        Frequency axis, shape ``(N,)``.
    data : array_like
        FRF or output spectrum matrix, shape ``(Q, P, N)``.
    alpha : float
        Exponential-window decay rate (EMA) or correlogram window rate (OMA)
        [1/s].
    bands : list of tuple
        ``(f_lo, f_hi, fs_fake)`` per band, ``fs_fake ~ 2.5 * f_hi``.
    spectrum : {'frf_tap', 'frf_shaker', 'sd_cor', 'sd_per'}, optional
        Spectrum kind passed to :class:`pLSCF`. Default 'frf_tap'.
    ordmax : int, optional
        Maximum model order. Default 100.
    ordmin : int, optional
        Lowest order the stabilisation chart scores. Default 5.
    sc, hc : dict, optional
        Soft/hard stabilisation criteria. Default ``DEFAULT_SC`` / ``DEFAULT_HC``.
    eps : float, optional
        DBSCAN modal-distance radius. Default 0.02.
    min_pts : int, optional
        DBSCAN core-point threshold, also the minimum cluster size. Default 8.
    select : str, optional
        Cluster-representative rule, see :meth:`pLSCF.cluster`. Default
        'density'.
    cache : str, optional
        Cache identity for this fit. Default None (no caching).
    cache_dir : path-like, optional
        Directory of the cache files; required when ``cache`` is set.
    refit : bool, optional
        Ignore an existing cache file and fit anew. Default False.
    progress : bool, optional
        Per-order progress bar of each band fit. Default True.

    Returns
    -------
    BandMPE
        ``Lambd`` — combined physical poles, shape ``(M,)``; ``conf`` — poles
        per cluster for each mode, driving the merge in :func:`combine_lambd`;
        ``bands`` — per-band :class:`BandFit` for the diagrams.
    """
    freqs = jnp.asarray(freqs)
    data = jnp.asarray(data)
    sc = dict(DEFAULT_SC if sc is None else sc)
    hc = dict(DEFAULT_HC if hc is None else hc)
    bands = [(float(lo), float(hi), float(fsf)) for lo, hi, fsf in bands]

    cache_file = None
    if cache is not None:
        key = cache_key(
            spectrum,
            bands,
            ordmax,
            ordmin,
            sorted(sc.items()),
            sorted(hc.items()),
            eps,
            min_pts,
            select,
            None if alpha is None else float(alpha),
            _data_fingerprint(freqs, data),
        )
        cache_file = _cache_file(cache, cache_dir, key)
        if cache_file.exists() and not refit:
            try:
                out = load_model(cache_file, _make_band_mpe)
                print(f"{cache}: loaded cached fit ({cache_file.name})")
                return out
            except Exception as exc:  # e.g. the classes changed since the dump
                print(f"{cache}: unreadable cache ({exc!r}), refitting")

    band_data = []
    for flo, fhi, fs_fake in bands:
        sel = (freqs >= flo) & (freqs <= fhi)
        est = pLSCF(
            freqs=freqs[sel],
            fs=fs_fake,
            ordmax=ordmax,
            ordmin=ordmin,
            spectrum=spectrum,
            alpha=alpha,
            sc=sc,
            hc=hc,
        )
        poles = est.fit(data[:, :, sel], progress=progress)
        res = est.cluster(
            poles, eps=eps, min_pts=min_pts, select=select, min_cluster_size=min_pts
        )
        band_data.append(BandFit(flo, fhi, est, poles, res))
    out = BandMPE(
        jnp.asarray(_gather(band_data, "Lambd")),
        jnp.asarray(_gather(band_data, "sizes").astype(int)),  # poles per cluster
        band_data,
    )

    if cache_file is not None:
        save_model(
            cache_file,
            out,
            {
                "spectrum": spectrum,
                "alpha": None if alpha is None else float(alpha),
                "ordmax": ordmax,
                "ordmin": ordmin,
                "sc": sc,
                "hc": hc,
                "bands": [
                    {
                        "flo": bf.flo,
                        "fhi": bf.fhi,
                        "fs_fake": bf.est.fs,
                        "est": _spec(bf.est),
                        "poles": _spec(bf.poles),
                        "res": _spec(bf.res),
                    }
                    for bf in band_data
                ],
                "Lambd": [list(out.Lambd.shape), str(out.Lambd.dtype)],
                "conf": [list(out.conf.shape), str(out.conf.dtype)],
            },
        )
        print(f"{cache}: fitted and cached ({cache_file.name})")
    return out


def _gather(band_data, name: str) -> np.ndarray:
    """Collect the per-pole attribute ``name`` (e.g. ``"Lambd"``, ``"Lr"``) of the
    cluster results of all bands into one array along the pole axis."""
    out = []
    for flo, fhi, *_, res in band_data:
        mask = np.asarray((res.Fn >= flo) & (res.Fn <= fhi))
        out.append(np.asarray(getattr(res, name))[..., mask])
    return np.concatenate(out, axis=-1)


def combine_lambd(
    lambd_a,
    conf_a,
    lambd_b,
    conf_b,
    deltaf: float = 0.5,
    rtol: float = 0.02,
    lr_a=None,
    lr_b=None,
):
    """
    Combination of two pole sets; a mode found by both passes is
    counted once.

    For a coincident pair the survivor is the pole with the larger
    stabilisation-cluster confidence.

    Parameters
    ----------
    lambd_a, lambd_b : array_like
        Complex pole sets to merge.
    conf_a, conf_b : array_like
        Cluster confidence per pole, aligned with ``lambd_a``/``lambd_b``.
    deltaf, rtol : float, optional
        Absolute [Hz] and relative match tolerance; a pair is coincident within
        ``max(deltaf, rtol * f)``. Defaults 0.5 and 0.02.
    lr_a, lr_b : array_like, optional
        Reference factors ``(P, .)`` aligned with the poles, on one common
        reference set; given, the survivors' factors are returned too.

    Returns
    -------
    Lambd : jax.Array
        Merged poles, frequency-sorted.
    conf : np.ndarray
        Confidence of each surviving pole.
    Lr : jax.Array
        Factors of the surviving poles ``(P, M)``; only with ``lr_a``/``lr_b``.
    """
    if (lr_a is None) != (lr_b is None):
        raise ValueError("pass reference factors for both pole sets or neither")
    la, lb = np.asarray(lambd_a), np.asarray(lambd_b)
    ca, cb = np.asarray(conf_a), np.asarray(conf_b)

    def out(keep_a, keep_b):
        lam = np.concatenate([la[keep_a], lb[keep_b]])
        conf = np.concatenate([ca[keep_a], cb[keep_b]])
        order = np.argsort(np.abs(lam))
        if lr_a is None:
            return jnp.asarray(lam[order]), conf[order]
        lr = np.concatenate(
            [np.asarray(lr_a)[:, keep_a], np.asarray(lr_b)[:, keep_b]], axis=1
        )
        return jnp.asarray(lam[order]), conf[order], jnp.asarray(lr[:, order])

    if la.size == 0 or lb.size == 0:
        return out(np.ones(la.size, bool), np.ones(lb.size, bool))

    fa, fb = np.asarray(plscf.lambd_to_fn(la)), np.asarray(plscf.lambd_to_fn(lb))
    d = np.abs(fb[:, None] - fa[None, :])  # |f_b - f_a|
    j = d.argmin(axis=1)  # nearest a-pole per b-pole
    matched = d[np.arange(fb.size), j] <= np.maximum(deltaf, rtol * fb)

    keep_a = np.ones(la.size, bool)
    keep_b = ~matched  # unmatched b-poles kept as new
    for ai in np.unique(j[matched]):  # one survivor per shared mode
        grp = np.where(matched & (j == ai))[0]
        cnf = np.concatenate([[ca[ai]], cb[grp]])  # a-pole vs its matched b-poles
        win = int(np.argmax(cnf))  # 0 -> a-pole, k>0 -> grp[k-1]
        if win:  # a b-pole is better supported
            keep_a[ai] = False
            keep_b[grp[win - 1]] = True
    return out(keep_a, keep_b)


def local_factors(
    freqs, data, Lambd, alpha: Optional[float], *, spectrum="sd_cor", **local_kw
):
    """
    Reference factors of known poles on ``data``'s reference set, ``(P, M)``.

    One local pLSCF model per pole re-estimates the pole on ``data`` and reads off its
    factors. Needed when the poles were found under other reference sets (the
    merged passes).

    Parameters
    ----------
    freqs, data : array_like
        Frequency axis ``(N,)`` and spectra ``(Q, P, N)`` of the LSFD stage.
    Lambd : array_like
        Complex poles ``(M,)`` to re-estimate.
    alpha : float
        Window decay rate [1/s], as for :func:`plscf_local`.
    spectrum : str, optional
        Spectrum kind. Default 'sd_cor'.
    **local_kw
        Settings of the local models (``order``, ``deltaf``, ``rtol``, ...).

    Returns
    -------
    Lambd : jax.Array
        The poles, NaN where no local pole was found.
    Lr : np.ndarray
        Reference factors ``(P, M)``, zero columns for NaN poles.
    loc : LocalModes
        The local models.
    """
    Lambd = jnp.asarray(Lambd)
    loc = plscf_local(
        freqs,
        jnp.asarray(data)[None],
        plscf.lambd_to_fn(Lambd),
        alpha,
        spectrum=spectrum,
        **local_kw,
    )
    ok = np.asarray(loc.ok[0])
    if not ok.all():
        fn = np.asarray(plscf.lambd_to_fn(Lambd))
        print(
            f"local_factors: no local pole in band for {np.round(fn[~ok], 2)} Hz; "
            "those modes sit out the residue fit"
        )
    Lambd = jnp.where(jnp.asarray(ok), Lambd, jnp.nan)
    Lr = np.where(ok[None, :], np.asarray(loc.Lr[0]), 0.0)
    return Lambd, Lr, loc


def full_band_mpe(
    passes,
    alpha: Optional[float],
    fs: float,
    bands,
    *,
    spectrum: str = "frf_tap",
    lsfd_data=None,
    quantity: str = "acceleration",
    ordmax: int = 100,
    ordmin: int = 5,
    sc: Optional[dict] = None,
    hc: Optional[dict] = None,
    eps: float = 0.02,
    min_pts: int = 8,
    select: str = "density",
    deltaf: float = 0.5,
    rtol: float = 0.02,
    lsfd_band: Optional[Tuple[float, float]] = None,
    f_min: Optional[float] = None,
    refs: str = "local",
    local_kw: Optional[dict] = None,
    cache=None,
    cache_dir=None,
    refit: bool = False,
    progress: bool = True,
):
    """
    Full band MPE procedure for one measurement set: band-wise pLSCF per pass, poles
    merged across passes, then a residue (LSFD) fit with fixed reference factors.

    Parameters
    ----------
    passes : list of tuple
        ``(freqs, data)`` spectra fitted independently.
    alpha : float
        Decay rate.
    fs : float
        Physical sampling rate for the LSFD stage.
    bands : list of tuple
        ``(f_lo, f_hi, fs_fake)`` per band, forwarded to :func:`band_mpe`.
    spectrum : str, optional
        Spectrum kind. Default 'frf_tap'.
    lsfd_data : tuple, optional
        ``(freqs, data)`` for the residue fit. Defaults to the first pass.
    quantity : str, optional
        Measured response quantity for LSFD; ignored for half-spectra.
        Default 'acceleration'.
    ordmax, ordmin, sc, hc, eps, min_pts, select, progress : optional
        Passed through to :func:`band_mpe`.
    deltaf, rtol : float, optional
        Match tolerances of the cross-pass merge, see :func:`combine_lambd`.
    lsfd_band : tuple, optional
        ``(f_lo, f_hi)`` for the residue fit. Defaults to the span the
        sub-bands cover.
    f_min : float, optional
        Frequency floor [Hz]: merged poles below it are discarded before the
        residue stage, so they take no residue and never reach the model.
        A guard against a spurious pole under the first mode that the
        band edge cannot exclude without cutting into that mode's peak.
        The band fits themselves are untouched (the stabilisation charts
        still show the pole). Default None (no floor).
    refs : {'local', 'global'}, optional
        Source of the reference factors. ``'global'``: the factors each pole
        already carries from the global band-wise fit that found it.
        ``'local'`` (default): a local pLSCF model per merged
        pole on the LSFD data.
    local_kw : dict, optional
        Settings of those local models (``order``, ``deltaf``, ``rtol``, ...),
        see :func:`plscf_local`. Only with ``refs='local'``.
    cache : str or list of str, optional
        Cache identity per pass, forwarded to :func:`band_mpe`.
        Default None (no caching).
    cache_dir : path-like, optional
        Directory of the cache files; required when ``cache`` is set.
    refit : bool, optional
        Ignore existing cache files and fit anew. Default False.

    Returns
    -------
    fit : LSFDFit
        The residue-stage estimator and its :class:`ModalModel`; the merged
        physical poles are ``fit.model.Lambd``.
    band_data : list of list of BandFit
        Per-pass :func:`band_mpe` band data.
    """
    if refs not in ("local", "global"):
        raise ValueError("refs must be 'local' or 'global'")
    lf, lD = passes[0] if lsfd_data is None else lsfd_data
    if refs == "global":
        P = np.asarray(lD).shape[1]
        if any(np.asarray(D).shape[1] != P for _, D in passes):
            raise ValueError(
                "refs='global' needs every pass and the LSFD data on one reference "
                f"set (P={P}); with different reference subsets use refs='local'"
            )

    keys = (
        [cache] * len(passes)
        if cache is None or isinstance(cache, str)
        else list(cache)
    )
    Lambd = conf = Lr = None
    band_data = []
    for (f, D), key in zip(passes, keys):
        bm = band_mpe(
            f,
            D,
            alpha,
            bands,
            spectrum=spectrum,
            ordmax=ordmax,
            ordmin=ordmin,
            sc=sc,
            hc=hc,
            eps=eps,
            min_pts=min_pts,
            select=select,
            cache=key,
            cache_dir=cache_dir,
            refit=refit,
            progress=progress,
        )
        L, c, bd = bm
        band_data.append(bd)
        if Lambd is None:
            Lambd, conf = L, c
            Lr = bm.Lr if refs == "global" else None
        elif refs == "global":  # the poles' own factors follow the survivors
            Lambd, conf, Lr = combine_lambd(
                Lambd, conf, L, c, deltaf, rtol, lr_a=Lr, lr_b=bm.Lr
            )
        else:
            Lambd, conf = combine_lambd(Lambd, conf, L, c, deltaf, rtol)

    if f_min is not None:
        fn = np.asarray(plscf.lambd_to_fn(Lambd))
        keep = fn >= f_min
        if not keep.all():
            print(
                f"full_band_mpe: dropped {int((~keep).sum())} merged pole(s) below "
                f"f_min={f_min:g} Hz at {np.round(fn[~keep], 2)}"
            )
        Lambd, conf = jnp.asarray(Lambd)[keep], np.asarray(conf)[keep]
        if Lr is not None:
            Lr = jnp.asarray(Lr)[:, keep]

    # default the LSFD band to the span the sub-bands cover
    if lsfd_band is None:
        lsfd_band = (bands[0][0], bands[-1][1])

    lsfd = LSFD(
        freqs=lf,
        fs=fs,
        spectrum=spectrum,
        quantity=quantity,
        alpha=alpha,
        band=lsfd_band,
    )
    loc = None
    if refs == "local":  # the merge kept no factors: re-estimate on the LSFD data
        Lambd, Lr, loc = local_factors(
            lf, lD, Lambd, alpha, spectrum=spectrum, **(local_kw or {})
        )
    model = lsfd.fit(lD, Lambd, part_factors=Lr)
    return LSFDFit(lsfd, model, loc), band_data


# =============================================================================
# local pLSCF models
# =============================================================================


@eqx.filter_jit
def _plscf_local_map(
    S,
    freqs,
    starts,
    weights,
    dts,
    f_ref,
    f_lo,
    f_hi,
    phi_ref,
    order,
    zeta_max,
    spectrum,
    alpha,
    batch_size,
):
    """One fixed-order pLSCF per (record, mode) on the mode's band; ``vmap`` over
    the modes (equal-length windows, zero weights outside each band), ``lax.map``
    over the records.
    """
    n_loc = weights.shape[1]
    P = S.shape[2]
    idx = starts[:, None] + jnp.arange(n_loc)  # (M, N_loc)
    f_loc = freqs[idx]  # (M, N_loc)

    def one_mode(Sm, fm, wm, dt, fr, flo, fhi, pr):  # Sm : (Q, P, N_loc)
        Ad, Bn = plscf.fit(Sm, fm, dt, order, weight=wm)
        A, C = plscf.rmfd_to_ss(Ad[order], Bn[order])
        phi, lambd, lr = plscf.ss_to_modal_params(A, C, dt, P, spectrum, alpha)

        # nearest positive-frequency stable pole to f_ref; (order * P) candidates
        fn = plscf.lambd_to_fn(lambd)
        cand = jnp.isfinite(lambd) & (lambd.imag > 0)
        if zeta_max is not None:
            cand &= plscf.lambd_to_zeta(lambd) < zeta_max
        dist = jnp.where(cand, jnp.abs(fn - fr), jnp.inf)
        k = dist.argmin()
        found = jnp.isfinite(dist[k])
        ok = found & (fn[k] >= flo) & (fn[k] <= fhi)
        lam = jnp.where(found, lambd[k], jnp.nan)
        ph = jnp.where(found, phi[k], jnp.nan)
        lrk = jnp.where(found, lr[k], jnp.nan)

        # goodness of fit of the local model over the weighted band (NMSE).
        Sm_hat = plscf.rmfd_response(Ad[order], Bn[order], fm, dt)
        w = wm[None, None, :]
        err = (w * jnp.abs(Sm - Sm_hat) ** 2).sum()
        tot = (w * jnp.abs(Sm) ** 2).sum()
        conf = jnp.clip(1.0 - err / tot, 0.0, 1.0)
        mac = jnp.array(jnp.nan) if pr is None else _mac_elementwise(ph, pr)
        return lam, ph, lrk, ok, conf, mac

    pr_axis = None if phi_ref is None else 0

    def one_seg(Sb):  # (Q, P, N)
        S_loc = jnp.moveaxis(Sb[:, :, idx], 2, 0)  # (M, Q, P, N_loc)
        return jax.vmap(one_mode, in_axes=(0, 0, 0, 0, 0, 0, 0, pr_axis))(
            S_loc, f_loc, weights, dts, f_ref, f_lo, f_hi, phi_ref
        )

    return jax.lax.map(one_seg, S, batch_size=batch_size)


def plscf_local(
    freqs,
    S,
    f_ref,
    alpha: Optional[float],
    *,
    spectrum: str = "sd_cor",
    order: int = 2,
    deltaf: float = 1.0,
    rtol: float = 0.02,
    fs_factor: float = 2.5,
    zeta_max: Optional[float] = 0.1,
    phi_ref=None,
    batch_size: Optional[int] = 0,
) -> LocalModes:
    """
    Local pLSCF models of known modes over a batch of records.

    Per record and ``f_ref`` mode, one fixed-order pLSCF on a narrow band around
    the mode; the stable pole nearest ``f_ref`` is the estimate, graded by
    ``conf`` (fit quality over the band) and ``mac`` (against ``phi_ref``).

    Parameters
    ----------
    freqs : array_like
        Frequency axis shared by the segments, shape ``(N,)``.
    S : array_like
        Segment spectra, shape ``(n_seg, Q, P, N)``.
    f_ref : array_like
        Reference (global) natural frequencies [Hz]; sorted internally.
    alpha : float
        Exponential-window / correlogram decay rate [1/s].
    spectrum : str, optional
        Spectrum kind, see :class:`pLSCF`. Default 'sd_cor'.
    order : int, optional
        Order of every local model. Default 2.
    deltaf, rtol : float, optional
        Absolute [Hz] and relative half-width of each mode's band,
        ``max(deltaf, rtol * f_ref)``. Defaults 1.0 and 0.02.
    fs_factor : float, optional
        Fake sampling rate per band as a multiple of its upper edge (> 2).
        Default 2.5.
    zeta_max : float, optional
        Candidate poles must have damping below this; None keeps every stable
        pole. Default 0.1.
    phi_ref : array_like, optional
        Reference mode shapes ``(Q, M)`` aligned with the sorted ``f_ref``;
        enables the ``mac`` grading.
    batch_size : int, optional
        Segments modelled simultaneously per ``lax.map`` step; None does one
        at a time, 0 all at once (a plain ``vmap``). Default 0.

    Returns
    -------
    LocalModes
        ``Lambd`` ``(n_seg, M)``, ``Phi`` ``(n_seg, Q, M)``, ``Lr`` ``(n_seg, P, M)``,
        ``ok`` ``(n_seg, M)`` (a stable pole was found inside the band), ``conf``
        ``(n_seg, M)`` and ``mac`` ``(n_seg, M)`` (None without ``phi_ref``),
        aligned with the sorted ``f_ref``. ``Lambd`` is NaN only where no stable
        pole existed.
    """
    freqs = jnp.asarray(freqs)
    S = jnp.asarray(S)
    if fs_factor <= 2:
        raise ValueError("fs_factor must exceed 2 (Nyquist of the fake rate)")

    order_idx = np.argsort(np.asarray(f_ref, dtype=float).ravel())
    f_ref = np.asarray(f_ref, dtype=float).ravel()[order_idx]
    phi_ref = None if phi_ref is None else np.asarray(phi_ref)[:, order_idx]

    # per-mode band and its lines; every mode reads N_loc lines, zero-weighted
    # outside its own band
    f = np.asarray(freqs)
    half = np.maximum(deltaf, rtol * f_ref)
    f_lo, f_hi = f_ref - half, f_ref + half
    i_lo = np.searchsorted(f, f_lo, side="left")
    i_hi = np.searchsorted(f, f_hi, side="right")
    n_lines = i_hi - i_lo
    if (n_lines < order + 2).any():
        bad = f_ref[n_lines < order + 2]
        raise ValueError(
            f"fewer than {order + 2} lines in the band of f_ref={bad} Hz; widen "
            "deltaf/rtol"
        )
    n_loc = int(n_lines.max())
    starts = np.minimum(i_lo, f.size - n_loc)
    win = starts[:, None] + np.arange(n_loc)  # (M, N_loc)
    weights = ((f[win] >= f_lo[:, None]) & (f[win] <= f_hi[:, None])).astype(float)
    dts = 1.0 / (fs_factor * f_hi)

    Lambd, Phi, Lr, ok, conf, mac = _plscf_local_map(
        S,
        freqs,
        jnp.asarray(starts),
        jnp.asarray(weights),
        jnp.asarray(dts),
        jnp.asarray(f_ref),
        jnp.asarray(f_lo),
        jnp.asarray(f_hi),
        None if phi_ref is None else jnp.asarray(phi_ref.T, dtype=complex),
        int(order),
        None if zeta_max is None else float(zeta_max),
        spectrum,
        None if alpha is None else float(alpha),
        batch_size,
    )
    return LocalModes(
        f_ref=jnp.asarray(f_ref),
        Lambd=Lambd,
        Phi=jnp.moveaxis(Phi, 1, 2),  # (n_seg, M, Q) -> (n_seg, Q, M)
        Lr=jnp.moveaxis(Lr, 1, 2),
        ok=ok,
        conf=conf,
        mac=None if phi_ref is None else mac,
    )


# =============================================================================
# LSFD per segment
# =============================================================================


def lsfd_batch(
    freqs,
    S,
    Lambd,
    Lr,
    alpha: Optional[float],
    fs: float,
    *,
    spectrum: str = "sd_cor",
    quantity: str = "acceleration",
    band: Optional[Tuple[float, float]] = None,
    batch_size: Optional[int] = 0,
):
    """
    LSFD residue fit of a batch of records at known per-record poles.

    One :class:`LSFD` fit per record (segment) with its poles and the reference
    factors held fixed, so the mode shapes are the same
    kind of estimate as the full-signal reference shapes.

    Parameters
    ----------
    freqs : array_like
        Frequency axis shared by the records, ``(N,)``.
    S : array_like
        Record spectra ``(n_seg, Q, P, N)``.
    Lambd : array_like
        Per-record poles ``(n_seg, M)``, NaN where a record has no pole for a
        mode.
    Lr : array_like or None
        Reference factors held fixed: ``(P, M)`` shared by every record (the
        full-signal factors) or ``(n_seg, P, M)`` per record. None fits the
        residues unconstrained.
    alpha : float
        Window decay rate [1/s] of the records.
    fs : float
        Physical sampling rate [Hz].
    spectrum, quantity, band : optional
        As for :class:`LSFD`.
    batch_size : int, optional
        Records fitted simultaneously per ``lax.map`` step; None does one at a
        time, 0 all at once (a plain ``vmap``). Default 0.

    Returns
    -------
    model : ModalModel
        The residue fits, batched over records (leading axis ``n_seg``).
    Phi : jax.Array
        Their mode shapes ``(n_seg, Q, M)``, complex, native scale.
    """
    lsfd = LSFD(
        freqs=jnp.asarray(freqs),
        fs=fs,
        spectrum=spectrum,
        quantity=quantity,
        alpha=alpha,
        band=band,
    )
    S = jnp.asarray(S)
    Lambd = jnp.asarray(Lambd)
    if Lr is not None:
        Lr = jnp.asarray(Lr, dtype=complex)
    per_record = Lr is not None and Lr.ndim == 3

    def fit(D, lam, lr):
        return lsfd.fit(D, lam, part_factors=lr)

    # lax.map maps the leading axis of every input, so shared factors are closed over
    if per_record:
        model = jax.lax.map(lambda x: fit(*x), (S, Lambd, Lr), batch_size=batch_size)
    else:
        model = jax.lax.map(
            lambda x: fit(x[0], x[1], Lr), (S, Lambd), batch_size=batch_size
        )
    # a NaN pole came out with NaN residues: keep the SVD clean, blank its shape
    clean = eqx.tree_at(lambda m: m.A, model, jnp.nan_to_num(model.A))
    Phi = jax.lax.map(lambda m: m.mode_shapes(), clean, batch_size=batch_size)
    Phi = jnp.where(jnp.isfinite(Lambd)[:, None, :], Phi, jnp.nan)
    return model, Phi
