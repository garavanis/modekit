"""
Tests for ``modekit.dataprep.taptest``: reading a folder of tap-test CSVs.
"""

import numpy as np
import pytest

from modekit.dataprep import taptest

# two reps of (Hammer, Acc); the second rep is zero-padded after two samples
CSV = (
    "Sample rate (Hz)\t1.0E+3\n"
    "Hammer_tap_1\tAcc_tap_1\tHammer_tap_2\tAcc_tap_2\n"
    "N\tms^-2\tN\tms^-2\n"
    "1.0\t0.1\t2.0\t0.2\n"
    "3.0\t0.3\t4.0\t0.4\n"
    "5.0\t0.5\t0.0\t0.0\n"
    "0.0\t0.0\t0.0\t0.0\n"
)


def test_load_data_reads_every_csv_of_the_folder_given(tmp_path):
    (tmp_path / "set_a.csv").write_text(CSV)
    taps = taptest.DataLoader(tmp_path).load_data()
    assert list(taps) == ["set_a"]
    tap = taps["set_a"]
    assert tap.sample_rate == 1000.0 and tap.channels == ["Hammer", "Acc"]
    # (reps, channels, samples), cut to the shortest rep's non-zero length
    assert tap.data.shape == (2, 2, 2)
    np.testing.assert_array_equal(tap.load_channel("Acc"), [[0.1, 0.3], [0.2, 0.4]])


def test_a_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        taptest.DataLoader(tmp_path / "missing")
