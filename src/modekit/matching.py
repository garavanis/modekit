"""
Matching modes across structures that share a geometry.

For structures of one geometry but different materials (say, brass, steel and
aluminium copies of one design) the ratio ``f_n / f_1`` cancels the material scale
``sqrt(E / rho)``, leaving a geometry-only fingerprint of each mode.
:func:`match_modes` groups the modes of several structures into slots by that
ratio. The MAC only breaks ties: with a few single-axis sensors the higher
modes alias spatially. A structure may resolve fewer modes than another, or
modes the others lack.

The matching is a diagnostic. Once the pairing is decided (by eye, or from an
FE model) it becomes a catalogue of nominal frequencies, one row per mode,
and :func:`assign_catalogue` labels any condition's modes against it.
:func:`load_catalogue` reads such a catalogue from a CSV file.
"""

import io
import warnings
from pathlib import Path
from typing import (Dict, Hashable, List, Mapping, NamedTuple, Optional, Sequence, Tuple,
                    TypeVar)

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from modekit import criteria

_BIG = 1e6  # cost of a pair outside the ratio gate (never survives)
_K = TypeVar("_K", bound=Hashable)  # column label of assign_catalogue


class ModeMatch(NamedTuple):
    """
    Modes of several structures matched into slots by frequency ratio.

    One slot is one physical mode of the shared geometry. ``index[k, j]`` is
    the position of slot ``k``'s mode in the ``fn`` array of structure
    ``names[j]``, or ``-1`` when that structure did not resolve it.
    """

    names: Tuple[Hashable, ...]  # structures (or conditions), column order of ``index``
    ratio: np.ndarray  # (n_slot,) target f / f1 of each slot, ascending
    index: np.ndarray  # (n_slot, n_struct) int, -1 where absent

    @property
    def present(self) -> np.ndarray:
        """Boolean ``(n_slot, n_struct)``: which structures resolved each slot."""
        return self.index >= 0


class _Pairwise:
    """Lazily computed MAC matrices between the modes of two structures."""

    def __init__(self, phi: Optional[Mapping[str, np.ndarray]]):
        self._phi = phi
        self._cache: Dict[Tuple[str, str], np.ndarray] = {}

    def __call__(self, a: str, b: str) -> Optional[np.ndarray]:
        """MAC matrix ``(M_a, M_b)``, or None when no shapes were given."""
        if self._phi is None:
            return None
        if (a, b) not in self._cache:
            self._cache[(a, b)] = np.atleast_2d(
                np.asarray(criteria.mac(self._phi[a], self._phi[b]), dtype=float)
            )
        return self._cache[(a, b)]


def _assign(
    slots: List[dict],
    ref: str,
    b: str,
    cand: List[int],
    ratio: Mapping[str, np.ndarray],
    rtol: float,
    mac_weight: float,
    pair_mac: _Pairwise,
) -> Dict[int, int]:
    """
    Optimal ratio-gated assignment of ``b``'s modes ``cand`` to fixed ``slots``.

    Each slot carries its target ratio under ``"r"`` and, under key ``ref``,
    the index of the ref structure's mode behind it (for the MAC tie-break).
    Returns ``{slot index: b mode index}`` for the surviving pairs.
    """
    if not slots or not cand:
        return {}
    mac = pair_mac(ref, b)
    cost = np.full((len(slots), len(cand)), _BIG)
    for k, slot in enumerate(slots):
        for c, j in enumerate(cand):
            d = abs(ratio[b][j] / slot["r"] - 1)
            if d <= rtol:
                tie = 0.0 if mac is None else mac_weight * (1 - mac[slot[ref], j])
                cost[k, c] = d + tie
    rows, cols = linear_sum_assignment(cost)
    return {k: cand[c] for k, c in zip(rows, cols) if cost[k, c] < _BIG}


def _match_into(
    slots: List[dict],
    ref: str,
    cands: Dict[str, List[int]],
    ratio: Mapping[str, np.ndarray],
    rtol: float,
    mac_weight: float,
    pair_mac: _Pairwise,
) -> Dict[str, List[int]]:
    """
    Match each structure's candidate modes into ``slots`` (targets fixed from
    ``ref``); returns ``{structure: unmatched mode indices}``.
    """
    left = {}
    for b, cand in cands.items():
        got = _assign(slots, ref, b, cand, ratio, rtol, mac_weight, pair_mac)
        for k, j in got.items():
            slots[k][b] = j
        left[b] = [j for j in cand if j not in got.values()]
    return left


def match_modes(
    fn: Mapping[str, np.ndarray],
    phi: Optional[Mapping[str, np.ndarray]] = None,
    *,
    rtol: float = 0.05,
    mac_weight: Optional[float] = None,
) -> ModeMatch:
    """
    Group the modes of several structures into slots by frequency ratio.

    The structure with the most modes sets the slots, one per mode at its
    ratio ``f_n / f_1``. The other structures' modes are paired with those
    slots by closest ratio, within ``rtol``, at most one mode per slot; the
    MAC only decides near-ties. Modes left over open new slots, and the
    remaining structures are paired with them the same way.

    Parameters
    ----------
    fn : mapping {structure: array_like}
        Natural frequencies [Hz] per structure, ``(M_s,)``, any order;
        non-finite entries are ignored. Ratios are taken to the lowest
        frequency, so every structure must have resolved the same first mode.
    phi : mapping {structure: array_like}, optional
        Mode shapes ``(Q, M_s)`` aligned with ``fn``, real or complex, for the
        MAC tie-break. None disables it.
    rtol : float, optional
        Relative tolerance on the ratio. Default 0.05.
    mac_weight : float, optional
        Weight of ``(1 - MAC)`` in the cost. Default ``rtol / 100``, so the
        ratio dominates.

    Returns
    -------
    ModeMatch
        ``names`` in the order of ``fn``, slots sorted by ``ratio``, and the
        per-structure mode ``index`` (-1 where a structure lacks the slot).
    """
    names = tuple(fn)
    if phi is not None and set(phi) != set(names):
        raise ValueError("phi must have exactly the keys of fn")
    w_mac = rtol * 1e-2 if mac_weight is None else float(mac_weight)

    f = {s: np.asarray(fn[s], dtype=float).ravel() for s in names}
    cands = {s: [int(j) for j in np.flatnonzero(np.isfinite(f[s]))] for s in names}
    if any(not c for c in cands.values()):
        empty = [s for s, c in cands.items() if not c]
        raise ValueError(f"no finite frequencies for {empty}")
    ratio = {s: f[s] / f[s][cands[s]].min() for s in names}
    phi_np = None if phi is None else {s: np.asarray(phi[s]) for s in names}
    pair_mac = _Pairwise(phi_np)

    # spine = the structure with the most modes; its ratios seed the slots
    spine = max(names, key=lambda s: len(cands[s]))
    slots = [{"r": ratio[spine][j], spine: j} for j in cands[spine]]
    left = _match_into(
        slots,
        spine,
        {b: cands[b] for b in names if b != spine},
        ratio,
        rtol,
        w_mac,
        pair_mac,
    )

    # leftovers: sub-spine with the most, then the rest into it, repeat
    while any(left.values()):
        sub = max(left, key=lambda s: len(left[s]))
        extra = [{"r": ratio[sub][j], sub: j} for j in left.pop(sub)]
        left = _match_into(extra, sub, left, ratio, rtol, w_mac, pair_mac)
        slots += extra
    slots.sort(key=lambda sl: sl["r"])

    index = np.full((len(slots), len(names)), -1, dtype=int)
    for k, sl in enumerate(slots):
        for j, s in enumerate(names):
            if s in sl:
                index[k, j] = sl[s]
    return ModeMatch(names, np.array([sl["r"] for sl in slots]), index)


def slot_mac(match: ModeMatch, phi: Mapping[str, np.ndarray]) -> np.ndarray:
    """
    Worst pairwise MAC of each slot among the structures that resolved it.

    Parameters
    ----------
    match : ModeMatch
        A :func:`match_modes` result.
    phi : mapping {structure: array_like}
        Mode shapes ``(Q, M_s)`` aligned with the ``fn`` given to
        :func:`match_modes`.

    Returns
    -------
    numpy.ndarray
        ``(n_slot,)``, the smallest MAC over every pair of present structures;
        NaN where fewer than two structures resolved the slot.
    """
    pair_mac = _Pairwise({s: np.asarray(phi[s]) for s in match.names})
    out = np.full(match.index.shape[0], np.nan)
    for k, row in enumerate(match.index):
        present = [(s, j) for s, j in zip(match.names, row) if j >= 0]
        macs = [
            pair_mac(a, b)[i, j]
            for n, (a, i) in enumerate(present)
            for b, j in present[n + 1 :]
        ]
        if macs:
            out[k] = min(macs)
    return out


def assign_catalogue(
    fn: Mapping[_K, np.ndarray],
    catalogue: Mapping[_K, np.ndarray],
    *,
    rtol: float = 0.03,
) -> ModeMatch:
    """
    Label modes against a catalogue of nominal frequencies.

    The catalogue has one row per physical mode and one column per structure
    or condition, NaN where a mode is absent. In each column, modes and rows
    are paired at most one to one, so that the frequencies sit as close to
    their nominals as possible overall; a mode more than ``rtol`` from a
    row's nominal cannot take that row. Pairing them all at once, rather
    than nearest first, keeps two close modes in order when both drift.

    Parameters
    ----------
    fn : mapping {column: array_like}
        Natural frequencies [Hz] per column, ``(M,)``, any order; non-finite
        entries are ignored. Keys can be any hashable label.
    catalogue : mapping {column: array_like}
        Nominal frequencies [Hz] per column, ``(n_row,)``, NaN where the mode
        is absent. Same keys as ``fn``.
    rtol : float, optional
        Relative frequency gate. Default 0.03.

    Returns
    -------
    ModeMatch
        One slot per catalogue row: ``index[k, j]`` is the position in
        ``fn[names[j]]`` of the mode on row ``k``, -1 if none (modes no row
        takes are left out). ``ratio`` is each row's nominal ``f / f1``,
        averaged over the columns that have it.
    """
    names = tuple(fn)
    if set(catalogue) != set(names):
        raise ValueError("catalogue must have exactly the keys of fn")
    cat = np.column_stack(
        [np.asarray(catalogue[s], dtype=float).ravel() for s in names]
    )  # (n_row, n_struct)
    gate = np.log1p(rtol)
    index = np.full(cat.shape, -1, dtype=int)
    for j, s in enumerate(names):
        f = np.asarray(fn[s], dtype=float).ravel()
        rows = np.flatnonzero(np.isfinite(cat[:, j]))
        modes = np.flatnonzero(np.isfinite(f))
        if rows.size == 0 or modes.size == 0:
            continue
        d = np.abs(np.log(f[modes][None, :] / cat[rows, j][:, None]))
        cost = np.where(d <= gate, d, _BIG)
        for k, m in zip(*linear_sum_assignment(cost)):
            if cost[k, m] < _BIG:
                index[rows[k], j] = modes[m]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a row no column carries: NaN
        ratio = np.nanmean(cat / np.nanmin(cat, axis=0), axis=1)
    return ModeMatch(names, ratio, index)


class Catalogue(NamedTuple):
    """
    Nominal frequencies of a set of modes, read from a CSV by :func:`load_catalogue`.

    Each catalogue row is one physical mode. ``table`` holds the file's lines,
    ``keys`` the names of its key columns (e.g. config, material, temp).
    """

    table: pd.DataFrame
    keys: Tuple[str, ...]

    @property
    def n_row(self) -> int:
        """Number of catalogue rows, the length of every :meth:`nominal` column."""
        return int(self.table["row"].max())

    def nominal(self, *key) -> Optional[np.ndarray]:
        """
        The nominal frequency [Hz] of every row for one key, ``(n_row,)``.

        A blank key cell means "any value". When several lines fit, the one
        with more keys filled in wins. With ``keys = ("material", "temp")``::

            material,temp,row,f_nom
            steel,,1,11.5        <- steel at any temperature
            steel,30,1,10.9      <- steel at 30: overrides the line above

        ``nominal("steel", 30)`` is ``[10.9]`` and ``nominal("steel", 10)`` is
        ``[11.5]``. A row no line fits is NaN. Entry ``k`` is row ``k + 1``.

        Parameters
        ----------
        *key
            One value per key column, in the order of ``keys``. Values are
            compared as text (``30`` matches ``30``); None matches blank
            cells only.

        Returns
        -------
        numpy.ndarray or None
            None when no line fits the key at all.

        Raises
        ------
        ValueError
            If two lines with the same number of keys filled in give one row
            different values.
        """
        if len(key) != len(self.keys):
            raise ValueError(f"expected one value per key {self.keys}, got {key}")
        t = self.table
        hit = np.ones(len(t), dtype=bool)
        for name, value in zip(self.keys, key):
            col = t[name]
            match = col.isna().to_numpy()
            if value is not None:
                match = match | (col == str(value)).to_numpy(dtype=bool, na_value=False)
            hit &= match
        if not hit.any():
            return None
        sub = t[hit]
        spec = sub[list(self.keys)].notna().sum(axis=1)
        out = np.full(self.n_row, np.nan)
        for row, g in sub.groupby("row"):
            top = g.loc[spec[g.index] == spec[g.index].max(), "f_nom"]
            if top.nunique() > 1:
                raise ValueError(f"row {row} of {key}: lines equally specific disagree ({top.tolist()} Hz)")
            out[row - 1] = top.iloc[0]
        return out


def load_catalogue(path, keys: Sequence[str]) -> Catalogue:
    """
    Read a catalogue of nominal frequencies from a CSV file.

    One line per nominal: the key columns, ``row`` (1-based; the same row is
    the same mode everywhere) and ``f_nom`` [Hz]. A blank key cell means "any
    value" (see :meth:`Catalogue.nominal`). Other columns, such as ``note``,
    are kept but not used. Lines starting with ``#`` are comments.

    Parameters
    ----------
    path : str or pathlib.Path
        The CSV file (UTF-8).
    keys : sequence of str
        The key columns, in the order :meth:`Catalogue.nominal` takes them.

    Returns
    -------
    Catalogue

    Raises
    ------
    ValueError
        If a column is missing, a ``row`` is not a positive integer, an
        ``f_nom`` is not a positive number, or two lines repeat the same keys
        and row.
    """
    keys = tuple(keys)
    text = Path(path).read_text(encoding="utf-8")
    body = "".join(ln for ln in text.splitlines(keepends=True) if not ln.lstrip().startswith("#"))
    t = pd.read_csv(io.StringIO(body), dtype={k: str for k in keys})
    missing = [c for c in (*keys, "row", "f_nom") if c not in t.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    row = pd.to_numeric(t["row"], errors="coerce")
    if row.isna().any() or (row < 1).any() or (row % 1 != 0).any():
        raise ValueError(f"{path}: row must be a positive integer")
    f_nom = pd.to_numeric(t["f_nom"], errors="coerce")
    if not (np.isfinite(f_nom) & (f_nom > 0)).all():
        raise ValueError(f"{path}: f_nom must be a positive number [Hz]")
    t["row"], t["f_nom"] = row.astype(int), f_nom.astype(float)
    dup = t.duplicated(subset=[*keys, "row"], keep=False)
    if dup.any():
        raise ValueError(f"{path}: repeated lines {t.loc[dup, [*keys, 'row']].values.tolist()}")
    return Catalogue(t, keys)
