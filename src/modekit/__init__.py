"""modekit: modal analysis tools in Python, built on JAX.

The main names are available here (``import modekit as mk``); the modules hold
the rest.
"""

from importlib import metadata as _metadata

import jax

# the identification needs double precision; set before any module builds arrays
jax.config.update("jax_enable_x64", True)

try:
    __version__ = _metadata.version("modekit")
except _metadata.PackageNotFoundError:  # a source checkout that was never installed
    __version__ = "0+unknown"

from modekit import (  # noqa: E402
    algorithms,
    bandmpe,
    clustering,
    criteria,
    datasets,
    export,
    freq_models,
    matching,
    plots,
    plscf,
    preprocess,
    reporting,
    serialisation,
    taptest,
    windows,
)
from modekit.algorithms import LSFD, ModalModel, PoleArrays, pLSCF  # noqa: E402
from modekit.bandmpe import (  # noqa: E402
    BandFit,
    BandMPE,
    LocalModes,
    LSFDFit,
    band_mpe,
    combine_lambd,
    full_band_mpe,
    lsfd_batch,
    plscf_local,
)
from modekit.export import save_table, to_text  # noqa: E402
from modekit.freq_models import EmaModel, OmaModel  # noqa: E402
from modekit.matching import Catalogue, assign_catalogue, load_catalogue, match_modes  # noqa: E402
from modekit.reporting import (  # noqa: E402
    match_table,
    mode_table,
    scale_table,
    tracking_summary,
    tracking_table,
)
from modekit.serialisation import cache_key, load_model, save_model  # noqa: E402

__all__ = [
    # spectra
    "EmaModel",
    "OmaModel",
    # estimators
    "pLSCF",
    "PoleArrays",
    "LSFD",
    "ModalModel",
    # the band-wise procedure and the tracking
    "band_mpe",
    "combine_lambd",
    "full_band_mpe",
    "plscf_local",
    "lsfd_batch",
    "BandFit",
    "BandMPE",
    "LSFDFit",
    "LocalModes",
    # matching and catalogues
    "match_modes",
    "assign_catalogue",
    "load_catalogue",
    "Catalogue",
    # tables
    "mode_table",
    "tracking_table",
    "tracking_summary",
    "match_table",
    "scale_table",
    "to_text",
    "save_table",
    # models on disk
    "save_model",
    "load_model",
    "cache_key",
    # modules
    "algorithms",
    "bandmpe",
    "clustering",
    "criteria",
    "datasets",
    "export",
    "freq_models",
    "matching",
    "plots",
    "plscf",
    "preprocess",
    "reporting",
    "serialisation",
    "taptest",
    "windows",
]
