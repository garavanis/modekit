"""
Tests for ``modekit.matching``.

Synthetic structures of one geometry scaled by a material factor: the slots
must line up by ratio, missing and extra modes must not be forced into a
bijection, and the MAC must only break ratio ties.
"""

import numpy as np
import pytest

from modekit.matching import assign_catalogue, load_catalogue, match_modes, slot_mac

RNG = np.random.default_rng(0)
RATIOS = np.array([1.0, 2.5, 2.64, 6.1, 7.8, 10.1])  # geometry fingerprint


def _shapes(Q=6, M=None, seed=0):
    rng = np.random.default_rng(seed)
    V, _ = np.linalg.qr(rng.standard_normal((Q, Q)))  # orthogonal columns
    return V[:, : (Q if M is None else M)]


def test_scaled_copies_match_slot_for_slot():
    fn = {"a": 8.0 * RATIOS, "b": 11.5 * RATIOS, "c": 11.3 * RATIOS}
    m = match_modes(fn, rtol=0.05)
    assert m.names == ("a", "b", "c")
    assert m.ratio.shape == (RATIOS.size,)
    assert np.allclose(m.ratio, RATIOS)
    assert np.array_equal(m.index, np.tile(np.arange(RATIOS.size)[:, None], (1, 3)))
    assert m.present.all()


def test_ratio_is_taken_to_the_lowest_finite_frequency_in_any_order():
    perm = RNG.permutation(RATIOS.size)
    fn = {"a": 8.0 * RATIOS, "b": (11.5 * RATIOS)[perm]}
    m = match_modes(fn)
    # b's modes are found at their permuted positions
    assert np.array_equal(m.index[:, 1], np.argsort(perm))
    assert np.array_equal(m.index[:, 0], np.arange(RATIOS.size))


def test_missing_and_extra_modes_open_or_leave_slots():
    # b lacks the 2.64 mode; c has an extra mode at ratio 4 nobody else sees
    fn = {
        "a": 8.0 * RATIOS,
        "b": 11.5 * np.delete(RATIOS, 2),
        "c": 11.3 * np.sort(np.append(RATIOS, 4.0)),
    }
    m = match_modes(fn)
    assert m.ratio.size == RATIOS.size + 1
    slot_264 = int(np.argmin(np.abs(m.ratio - 2.64)))
    assert m.index[slot_264, 1] == -1  # b absent there
    slot_4 = int(np.argmin(np.abs(m.ratio - 4.0)))
    assert m.index[slot_4, 0] == -1 and m.index[slot_4, 1] == -1
    assert m.index[slot_4, 2] >= 0
    # every finite mode of every structure sits in exactly one slot
    for j, s in enumerate(m.names):
        got = np.sort(m.index[m.index[:, j] >= 0, j])
        assert np.array_equal(got, np.arange(fn[s].size))
    assert np.all(np.diff(m.ratio) > 0)


def test_non_finite_frequencies_are_ignored():
    fn_a = 8.0 * RATIOS
    fn_a_padded = np.append(fn_a, [np.nan, np.nan])
    m = match_modes({"a": fn_a_padded, "b": 11.5 * RATIOS})
    assert m.ratio.size == RATIOS.size
    assert (m.index[:, 0] < RATIOS.size).all()
    with pytest.raises(ValueError):
        match_modes({"a": fn_a, "b": np.full(3, np.nan)})


def test_mac_breaks_a_ratio_tie():
    # a has one mode at ratio 3; b has two candidates equidistant in ratio
    # (2.97 and 3.03), only one of which shares a's shape
    phi_a = _shapes(M=2)  # columns: mode 1, mode at ratio 3
    fn_a = np.array([10.0, 30.0])
    fn_b = np.array([10.0, 29.7, 30.3])
    other = _shapes(seed=1)[:, 3]
    for same_first in (True, False):
        cols = [phi_a[:, 1], other] if same_first else [other, phi_a[:, 1]]
        phi_b = np.column_stack([phi_a[:, 0], *cols])
        m = match_modes({"a": fn_a, "b": fn_b}, {"a": phi_a, "b": phi_b}, rtol=0.05)
        # b (more modes) is the spine; a's second mode must land on the slot
        # of b's like-shaped candidate, not merely the nearer-in-ratio one
        slot = int(np.flatnonzero(m.index[:, 0] == 1)[0])
        want = 1 if same_first else 2
        assert m.index[slot, 1] == want
        assert m.ratio.size == 3
    # without shapes the assignment is still ratio-gated and one-to-one
    m0 = match_modes({"a": fn_a, "b": fn_b})
    assert m0.ratio.size == 3
    assert (np.sort(m0.index[m0.index[:, 1] >= 0, 1]) == np.arange(3)).all()


def test_gate_is_relative_and_respected():
    fn = {"a": np.array([10.0, 20.0]), "b": np.array([10.0, 22.0])}
    tight = match_modes(fn, rtol=0.05)  # 22/20 = 1.10 outside the gate
    assert tight.ratio.size == 3
    loose = match_modes(fn, rtol=0.15)
    assert loose.ratio.size == 2
    assert loose.present.all()


def test_phi_keys_must_match_fn():
    with pytest.raises(ValueError):
        match_modes({"a": RATIOS, "b": RATIOS}, {"a": _shapes(), "c": _shapes()})


def test_slot_mac_is_the_worst_pair_and_nan_for_singletons():
    phi = _shapes(M=3)
    fn = {"a": 8.0 * RATIOS[:3], "b": 11.5 * RATIOS[:3], "c": 11.3 * RATIOS[:2]}
    phis = {
        "a": phi,
        "b": np.column_stack([phi[:, 0], 0.6 * phi[:, 1] + 0.8 * phi[:, 2], phi[:, 2]]),
        "c": phi[:, :2],
    }
    m = match_modes(fn, phis)
    mac = slot_mac(m, phis)
    assert mac.shape == (3,)
    assert np.isclose(mac[0], 1.0)
    # slot 1: a-b MAC = 0.36, a-c = 1, b-c = 0.36 -> worst 0.36
    assert np.isclose(mac[1], 0.36)
    # slot 2 is resolved by a and b only, both with the same shape
    assert m.index[2, 2] == -1 and np.isclose(mac[2], 1.0)
    # a slot held by a single structure has no pair to compare
    fn_single = {"a": 8.0 * RATIOS[:2], "b": 11.5 * RATIOS[:1]}
    m1 = match_modes(fn_single, {"a": phi[:, :2], "b": phi[:, :1]})
    assert np.isnan(slot_mac(m1, {"a": phi[:, :2], "b": phi[:, :1]})[1])


# =============================================================================
# assign_catalogue: labelling against a fixed pairing
# =============================================================================


def test_assign_catalogue_keeps_the_order_under_a_common_drift():
    """A warm condition: every mode ~1 % lower. Brass's close pair (355, 359.6) sits at
    (351.1, 356.4): nearest-first would put 356.4 on the 355 row; the optimal
    assignment keeps the order. Steel's 29.4/30.8 pair likewise."""
    cat = {"brs": [7.9, 20.5, np.nan, 355.0, 359.6], "stl": [11.5, 29.4, 30.8, 521.7, 532.2]}
    fn = {"brs": [351.1, 7.9, 356.4, 20.4], "stl": [11.5, 30.3, 29.0, 516.5, 526.8]}
    m = assign_catalogue(fn, cat, rtol=0.03)
    assert m.names == ("brs", "stl") and m.index.shape == (5, 2)
    assert m.index[:, 0].tolist() == [1, 3, -1, 0, 2]     # -1: brass has no 30.8 row
    assert m.index[:, 1].tolist() == [0, 2, 1, 3, 4]
    # the row's nominal f/f1, averaged over the structures that carry it
    assert m.ratio[0] == pytest.approx(1.0) and m.ratio[2] == pytest.approx(30.8 / 11.5)
    assert m.ratio[1] == pytest.approx(0.5 * (20.5 / 7.9 + 29.4 / 11.5))


def test_assign_catalogue_gate_extras_and_keys():
    cat = {"a": [10.0, 20.0, 30.0]}
    fn = {"a": [10.1, 25.0, np.nan, 30.2, 40.0]}       # 25 and 40 belong to no row
    m = assign_catalogue(fn, cat, rtol=0.03)
    assert m.index[:, 0].tolist() == [0, -1, 3]
    assert assign_catalogue(fn, cat, rtol=0.3).index[:, 0].tolist() == [0, 1, 3]   # 25 now reaches the 20 row
    with pytest.raises(ValueError, match="keys"):
        assign_catalogue(fn, {"b": [10.0]})


# =============================================================================
# load_catalogue: a catalogue from a CSV file
# =============================================================================
CAT_CSV = """\
# a comment line
struct,cond,row,f_nom,note
a,,1,10.0,
a,,2,20.0,
a,,3,30.0,
a,hot,2,19.5,softer when hot
b,,1,15.0,
b,,3,45.0,
"""


def _csv(tmp_path, text):
    path = tmp_path / "cat.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_catalogue_defaults_overrides_and_absent_rows(tmp_path):
    cat = load_catalogue(_csv(tmp_path, CAT_CSV), keys=("struct", "cond"))
    assert cat.keys == ("struct", "cond") and cat.n_row == 3
    assert cat.nominal("a", "cold").tolist() == [10.0, 20.0, 30.0]   # the defaults
    assert cat.nominal("a", "hot").tolist() == [10.0, 19.5, 30.0]    # the hot line overrides row 2 only
    assert cat.nominal("a", None).tolist() == [10.0, 20.0, 30.0]     # None matches blank cells only
    b = cat.nominal("b", "hot")
    assert b[0] == 15.0 and np.isnan(b[1]) and b[2] == 45.0          # b lacks row 2
    assert cat.nominal("c", "hot") is None                            # not catalogued
    assert cat.table["note"].iloc[3] == "softer when hot"
    with pytest.raises(ValueError, match="key"):
        cat.nominal("a")


def test_load_catalogue_compares_keys_as_text(tmp_path):
    cat = load_catalogue(_csv(tmp_path, "s,temp,row,f_nom\na,,1,10.0\na,-15,1,10.4\n"), keys=("s", "temp"))
    assert cat.nominal("a", -15)[0] == 10.4 and cat.nominal("a", 30)[0] == 10.0


@pytest.mark.parametrize(
    "text, match",
    [
        ("s,row\na,1\n", "missing"),
        ("s,row,f_nom\na,0,10.0\n", "row"),
        ("s,row,f_nom\na,1.5,10.0\n", "row"),
        ("s,row,f_nom\na,1,-3.0\n", "f_nom"),
        ("s,row,f_nom\na,1,\n", "f_nom"),
        ("s,row,f_nom\na,1,10.0\na,1,10.2\n", "repeated"),
    ],
)
def test_load_catalogue_rejects_a_malformed_file(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        load_catalogue(_csv(tmp_path, text), keys=("s",))


def test_catalogue_lines_equally_specific_must_agree(tmp_path):
    cat = load_catalogue(_csv(tmp_path, "s,c,row,f_nom\na,,1,10.0\n,x,1,11.0\n"), keys=("s", "c"))
    assert cat.nominal("a", "y")[0] == 10.0 and cat.nominal("b", "x")[0] == 11.0
    with pytest.raises(ValueError, match="equally specific"):
        cat.nominal("a", "x")      # both lines fill one key
