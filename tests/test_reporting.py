"""
Tests for ``modekit.reporting``.

The tables are checked against direct NumPy computations on synthetic
``LocalModes`` / ``ModalModel`` objects and on plain array pairs.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_float_dtype

from modekit import reporting as rp
from modekit.algorithms import ModalModel
from modekit.bandmpe import LocalModes, LSFDFit
from modekit.plscf import fn_zeta_to_lambd

RNG = np.random.default_rng(0)
RATIOS = np.array([1.0, 2.5, 2.64, 6.1, 7.8])


def _model(fn, zeta, Q=6, P=2, seed=0):
    """A rank-one-residue ``ModalModel`` (real, well separated shapes)."""
    rng = np.random.default_rng(seed)
    fn = np.asarray(fn, float)
    M = fn.size
    lam = np.asarray(fn_zeta_to_lambd(fn, np.asarray(zeta, float)))
    V, _ = np.linalg.qr(rng.standard_normal((Q, Q)))
    phi = V[:, :M].astype(complex)
    lr = rng.standard_normal((P, M)) + 1j * rng.standard_normal((P, M))
    return ModalModel(
        Lambd=lam,
        A=phi[:, None, :] * lr[None, :, :],
        LR=np.zeros((Q, P), complex),
        UR=np.zeros((Q, P), complex),
    )


def _tracking(fn, n_seg=8, Q=6, P=2, seed=0):
    """A synthetic ``LocalModes`` plus sample shapes and their reference."""
    rng = np.random.default_rng(seed)
    fn = np.asarray(fn, float)
    M = fn.size
    Fn = fn * (1 + 0.01 * rng.standard_normal((n_seg, M)))
    Zeta = 0.01 * (1 + 0.2 * rng.random((n_seg, M)))
    lam = np.asarray(fn_zeta_to_lambd(Fn, Zeta))
    ok = rng.random((n_seg, M)) > 0.25
    ok[0] = True  # every mode has at least one in-band sample
    seg = LocalModes(
        f_ref=fn,
        Lambd=lam,
        Phi=rng.standard_normal((n_seg, Q, M)) + 0j,
        Lr=rng.standard_normal((n_seg, P, M)) + 0j,
        ok=ok,
        conf=rng.random((n_seg, M)),
        mac=rng.random((n_seg, M)),
    )
    phi_ref = rng.standard_normal((Q, M)) + 1j * rng.standard_normal((Q, M))
    phi = rng.standard_normal((n_seg, Q, M)) + 1j * rng.standard_normal((n_seg, Q, M))
    return seg, phi, phi_ref


def _sample_mac(phi, phi_ref):
    num = np.abs(np.einsum("nqm,qm->nm", np.conj(phi), phi_ref)) ** 2
    return num / (np.sum(np.abs(phi) ** 2, axis=1) * np.sum(np.abs(phi_ref) ** 2, axis=0))


# =============================================================================
# mode_table
# =============================================================================


def test_mode_table_accepts_pairs_models_and_fits():
    fn, zeta = [30.0, 10.0, np.nan, 20.0], [0.03, 0.01, 0.0, 0.02]
    model = _model([30.0, 10.0, 20.0], [0.03, 0.01, 0.02])
    df = rp.mode_table({"pair": (fn, zeta), "model": model, "fit": LSFDFit(None, model)})
    assert list(df.columns) == ["test", "mode", "fn_hz", "zeta", "zeta_pct"]
    for test in ("pair", "model", "fit"):
        sub = df[df.test == test]
        assert list(sub["mode"]) == [1, 2, 3]  # NaN dropped, sorted ascending
        assert np.allclose(sub.fn_hz, [10.0, 20.0, 30.0])
        assert np.allclose(sub.zeta, [0.01, 0.02, 0.03])
        assert np.allclose(sub.zeta_pct, [1.0, 2.0, 3.0])


# =============================================================================
# tracking_table / tracking_summary
# =============================================================================


def test_tracking_table_matches_numpy_over_in_band_samples():
    fn = [10.0, 25.0, 50.0]
    seg, phi, phi_ref = _tracking(fn)
    df = rp.tracking_table(seg, phi, phi_ref, mac_lim=0.5)
    assert df.index.name == "mode" and list(df.index) == [0, 1, 2]
    assert list(df.columns) == [
        "f_ref", "in_band", "fn_med", "fn_iqr", "zeta_med",
        "conf", "mac", "mac_ok", "sample_mac", "sample_mac_ok",
    ]
    Fn, Z, ok = np.asarray(seg.Fn), np.asarray(seg.Zeta), np.asarray(seg.ok)
    conf, mac = np.asarray(seg.conf), np.asarray(seg.mac)
    smac = _sample_mac(phi, phi_ref)
    for k in range(3):
        m = ok[:, k]
        q = np.percentile(Fn[m, k], [25, 50, 75])
        assert df.f_ref[k] == fn[k]
        assert np.isclose(df.in_band[k], m.mean())
        assert np.isclose(df.fn_med[k], q[1]) and np.isclose(df.fn_iqr[k], q[2] - q[0])
        assert np.isclose(df.zeta_med[k], np.median(Z[m, k]))
        assert np.isclose(df.conf[k], conf[m, k].mean())
        assert np.isclose(df.mac[k], mac[m, k].mean())
        assert np.isclose(df.mac_ok[k], (mac[m, k] > 0.5).mean())
        assert np.isclose(df.sample_mac[k], smac[m, k].mean())
        assert np.isclose(df.sample_mac_ok[k], (smac[m, k] > 0.5).mean())


def test_tracking_table_columns_follow_what_is_available():
    seg, phi, phi_ref = _tracking([10.0, 25.0])
    plain = rp.tracking_table(seg)
    assert "sample_mac" not in plain.columns and "mac" in plain.columns
    with pytest.raises(ValueError):
        rp.tracking_table(seg, phi)  # phi without phi_ref

    # a bare Fn / Zeta object (a modal-store condition): finite == in band,
    # f_ref from Fn_ref, shapes picked up from Phi / Phi_ref
    class Cond:
        Fn = np.where(np.asarray(seg.ok), np.asarray(seg.Fn), np.nan)
        Zeta = np.where(np.asarray(seg.ok), np.asarray(seg.Zeta), np.nan)
        Fn_ref = np.asarray(seg.f_ref)
        Phi = phi
        Phi_ref = phi_ref

    store = rp.tracking_table(Cond())
    assert list(store.columns) == [
        "f_ref", "in_band", "fn_med", "fn_iqr", "zeta_med", "sample_mac", "sample_mac_ok"
    ]
    full = rp.tracking_table(seg, phi, phi_ref)
    for c in store.columns:
        assert np.allclose(store[c], full[c])

    # without any reference frequency the in-band median stands in
    class Bare:
        Fn = Cond.Fn
        Zeta = Cond.Zeta

    bare = rp.tracking_table(Bare())
    assert np.allclose(bare.f_ref, bare.fn_med)


def test_tracking_table_handles_a_mode_with_no_in_band_sample():
    seg, _, _ = _tracking([10.0, 25.0])
    ok = np.asarray(seg.ok).copy()
    ok[:, 1] = False
    seg = LocalModes(seg.f_ref, seg.Lambd, seg.Phi, seg.Lr, ok, seg.conf, seg.mac)
    df = rp.tracking_table(seg)
    assert df.in_band[1] == 0
    assert df[["fn_med", "fn_iqr", "zeta_med", "conf", "mac", "mac_ok"]].iloc[1].isna().all()
    assert np.isfinite(df.iloc[0]).all()


def test_tracking_summary_pools_every_in_band_sample():
    seg, phi, phi_ref = _tracking([10.0, 25.0, 50.0])
    df = rp.tracking_summary(seg, phi, phi_ref, mac_lim=0.5)
    assert df.shape == (1, 8)
    ok = np.asarray(seg.ok)
    row = df.iloc[0]
    assert row.n_seg == ok.shape[0] and row.n_modes == ok.shape[1]
    assert np.isclose(row.in_band, ok.mean())
    assert np.isclose(row.conf, np.asarray(seg.conf)[ok].mean())
    assert np.isclose(row.mac, np.asarray(seg.mac)[ok].mean())
    assert np.isclose(row.mac_ok, (np.asarray(seg.mac)[ok] > 0.5).mean())
    smac = _sample_mac(phi, phi_ref)
    assert np.isclose(row.sample_mac, smac[ok].mean())
    assert np.isclose(row.sample_mac_ok, (smac[ok] > 0.5).mean())


def test_tracking_summary_over_conditions_with_tuple_labels():
    a, phi_a, ref_a = _tracking([10.0, 25.0], n_seg=5, seed=1)
    b, phi_b, ref_b = _tracking([12.0, 30.0, 61.0], n_seg=7, seed=2)
    segs = {(20, "brs"): a, (20, "stl"): b}
    df = rp.tracking_summary(
        segs, phi={(20, "brs"): phi_a, (20, "stl"): phi_b},
        phi_ref={(20, "brs"): ref_a, (20, "stl"): ref_b},
    )
    assert isinstance(df.index, pd.MultiIndex)
    assert list(df.index) == [(20, "brs"), (20, "stl")]
    assert list(df.n_seg) == [5, 7] and list(df.n_modes) == [2, 3]
    one = rp.tracking_summary(b, phi_b, ref_b)
    assert np.allclose(df.loc[(20, "stl")].to_numpy(float), one.iloc[0].to_numpy(float))

    # one shared reference array, and no shapes at all
    df2 = rp.tracking_summary({"x": a, "y": a}, phi=phi_a, phi_ref=ref_a)
    assert np.allclose(df2.loc["x"], df2.loc["y"])
    df3 = rp.tracking_summary({"x": a, "y": b})
    assert "sample_mac" not in df3.columns and "mac_ok" in df3.columns


# =============================================================================
# match_table / scale_table
# =============================================================================


def test_match_table_from_models_and_pairs():
    fn_b, fn_s = 8.0 * RATIOS, 11.7 * RATIOS
    zeta = np.full(RATIOS.size, 0.01)
    brass, steel = _model(fn_b, zeta), _model(fn_s, zeta)  # same seed -> same shapes
    df = rp.match_table({"brass": brass, "steel": LSFDFit(None, steel)}, rtol=0.05)
    assert df.index.name == "slot"
    assert list(df.columns) == ["ratio", "brass", "steel", "mac", "flag"]
    assert np.allclose(df.ratio, RATIOS)
    assert np.allclose(df.brass, fn_b) and np.allclose(df.steel, fn_s)
    assert np.allclose(df.mac, 1.0) and (df.flag == "").all()
    assert df.attrs["formats"] == {"ratio": "{:.2f}", "brass": "{:.1f}", "steel": "{:.1f}", "mac": "{:.2f}"}

    # pairs, one structure missing a mode, shapes disagreeing on another
    phi = np.asarray(brass.mode_shapes(real=True, normalize="max"))
    phi_alu = phi.copy()
    phi_alu[:, 3] = phi[:, 4]  # slot 6.1: aluminium carries the wrong shape
    alu = (11.3 * np.delete(RATIOS, 1), np.delete(phi_alu, 1, axis=1))
    df = rp.match_table({"brass": (fn_b, phi), "aluminium": alu}, mac_lo=0.6)
    assert df.shape[0] == RATIOS.size
    assert np.isnan(df.aluminium[1]) and np.isnan(df.mac[1]) and df.flag[1] == ""
    assert df.mac[3] < 0.6 and df.flag[3] == "?"
    assert (df.flag.drop([1, 3]) == "").all()

    # no shapes anywhere: ratio-only matching, no MAC
    df0 = rp.match_table({"brass": (fn_b, None), "steel": (fn_s, None)})
    assert df0.mac.isna().all() and (df0.flag == "").all()
    with pytest.raises(ValueError):
        rp.match_table({"brass": (fn_b, phi), "steel": (fn_s, None)})


def test_scale_table_pairs_the_structure_columns():
    fn_b, fn_s, fn_a = 8.0 * RATIOS, 11.7 * RATIOS, 11.5 * RATIOS
    fn_a = np.delete(fn_a, 2)
    df = rp.match_table(
        {"brass": (fn_b, None), "steel": (fn_s, None), "aluminium": (fn_a, None)}
    )
    sc = rp.scale_table(df)
    assert list(sc.index) == [("brass", "steel"), ("brass", "aluminium"), ("steel", "aluminium")]
    assert list(sc.columns) == ["scale", "spread", "n"]
    assert np.isclose(sc.loc[("brass", "steel"), "scale"], 8.0 / 11.7)
    assert np.isclose(sc.loc[("brass", "aluminium"), "scale"], 8.0 / 11.5)
    assert np.allclose(sc.spread, 0.0)
    assert list(sc.n) == [RATIOS.size, RATIOS.size - 1, RATIOS.size - 1]
    only = rp.scale_table(df, structs=["steel", "brass"])
    assert list(only.index) == [("steel", "brass")]
    assert np.isclose(only.scale.iloc[0], 11.7 / 8.0)


def test_match_table_with_a_catalogue_fixes_the_rows():
    cat = {"brass": [7.9, 20.5, np.nan, 355.0], "steel": [11.5, 29.4, 30.8, 521.7]}
    models = {"brass": (np.array([7.9, 20.4, 351.1, 46.5]), None),
              "steel": (np.array([11.5, 29.0, 30.3, 516.5]), None)}
    df = rp.match_table(models, catalogue=cat, rtol=0.03)
    assert list(df.columns) == ["ratio", "brass", "steel", "mac", "flag"] and len(df) == 4
    assert df["brass"].tolist()[:2] == [7.9, 20.4] and np.isnan(df["brass"][2]) and df["brass"][3] == 351.1
    assert df["steel"].tolist() == [11.5, 29.0, 30.3, 516.5]
    assert df["mac"].isna().all() and (df["flag"] == "").all()
    assert df.attrs["unassigned"]["brass"].tolist() == [46.5] and df.attrs["unassigned"]["steel"].size == 0
    # the scale table reads the same rows
    sc = rp.scale_table(df)
    assert sc.loc[("brass", "steel"), "n"] == 3


def test_every_table_records_the_formats_of_its_float_columns():
    """The renderer knows no column by name: each table carries its own formats."""
    seg, phi, phi_ref = _tracking([10.0, 25.0])
    match = rp.match_table({"a": (8.0 * RATIOS, None), "b": (11.7 * RATIOS, None)})
    tables = [
        rp.mode_table({"t": _model([10.0, 20.0], [0.01, 0.02])}),
        rp.tracking_table(seg, phi, phi_ref),
        rp.tracking_table(seg),                      # no shapes: no sample_mac columns
        rp.tracking_summary({"x": seg, "y": seg}),
        match,
        rp.scale_table(match),
    ]
    for df in tables:
        floats = {c for c in df.columns if is_float_dtype(df[c])}
        assert floats and set(df.attrs["formats"]) == floats
