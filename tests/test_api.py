"""
The package namespace: ``import modekit`` exposes the main names and the
modules, and they are the modules' own objects.
"""

import modekit


def test_top_level_names_are_the_modules_objects():
    assert modekit.EmaModel is modekit.freq_models.EmaModel
    assert modekit.pLSCF is modekit.algorithms.pLSCF
    assert modekit.full_band_mpe is modekit.bandmpe.full_band_mpe
    assert modekit.mode_table is modekit.reporting.mode_table
    assert modekit.save_table is modekit.export.save_table
    assert modekit.load_model is modekit.serialisation.load_model
    assert modekit.taptest.DataLoader and modekit.windows.Tukey and modekit.preprocess.detrend_data
    assert all(hasattr(modekit, name) for name in modekit.__all__)
