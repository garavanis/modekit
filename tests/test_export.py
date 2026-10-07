"""
Tests for ``modekit.export``: the text renderer and the file exporter are
checked for their formats and outputs, on the tables of ``modekit.reporting``.
"""

import numpy as np
import pandas as pd
import pytest

from test_reporting import _tracking
from modekit import export
from modekit import reporting as rp


# =============================================================================
# to_text / save_table
# =============================================================================


def test_to_text_uses_the_tables_formats_and_renders_missing_cells():
    df = pd.DataFrame(
        {"in_band": [0.995, np.nan], "mac": [0.8456, 0.5], "brass": [7.94, 304.31]},
        index=pd.Index(["w20 brs", "w25 brs"]),
    )
    df.attrs["formats"] = {"in_band": "{:.0%}", "mac": "{:.2f}", "brass": "{:.1f}"}
    txt = export.to_text(df)
    lines = txt.splitlines()
    assert lines[0].split() == ["in_band", "mac", "brass"]
    assert lines[1].split() == ["w20", "brs", "100%", "0.85", "7.9"]
    assert lines[2].split() == ["w25", "brs", "-", "0.50", "304.3"]
    # overrides win, callables work, the index can be dropped
    txt = export.to_text(df, {"mac": lambda v: f"<{v:.1f}>"}, index=False, na_rep="n/a")
    assert txt.splitlines()[1].split() == ["100%", "<0.8>", "7.9"]
    assert "n/a" in txt
    # a frame without formats: floats get a general format whatever their names, ints untouched
    txt = export.to_text(pd.DataFrame({"in_band": [0.995], "x": [1234.5678], "n": [3]}))
    assert txt.splitlines()[1].split()[1:] == ["0.995", "1235", "3"]


def test_save_table_writes_every_format(tmp_path):
    seg, phi, phi_ref = _tracking([10.0, 25.0])
    df = rp.tracking_table(seg, phi, phi_ref)
    paths = export.save_table(df, tmp_path / "out" / "tracked", fmt=["csv", "xlsx", "html"])
    assert [p.rsplit(".", 1)[1] for p in paths] == ["csv", "xlsx", "html"]
    assert paths[0] == str(tmp_path / "out" / "tracked.csv")  # the folder is created
    back = pd.read_csv(paths[0], index_col="mode")  # the named index is written
    assert list(back.index) == [0, 1]
    assert np.allclose(back.in_band, df.in_band, atol=1e-4)
    html = open(paths[2], encoding="utf-8").read()
    assert "<title>tracked</title>" in html
    assert f"{df.in_band[0]:.0%}" in html  # rendered like the text
    assert pd.read_excel(paths[1], index_col=0).shape == df.shape

    # a bare RangeIndex is not written; a label index is
    p = export.save_table(rp.mode_table({"t": ([10.0], [0.01])}), tmp_path / "modes.txt")[0]
    assert p.endswith("modes.csv")  # an extension given is replaced by the format's
    assert list(pd.read_csv(p).columns) == ["test", "mode", "fn_hz", "zeta", "zeta_pct"]
    summ = rp.tracking_summary({"w20 brs": seg})
    p = export.save_table(summ, tmp_path / "summary")[0]
    assert pd.read_csv(p, index_col=0).index[0] == "w20 brs"
    with pytest.raises(ValueError):
        export.save_table(df, tmp_path / "x", fmt="pdf")
