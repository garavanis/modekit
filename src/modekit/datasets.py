"""
Example data, downloaded on first use.

The example notebooks read their measurements from the companion repository
`modekit-data <https://github.com/garavanis/modekit-data>`_. :func:`fetch`
downloads the files a notebook needs into a local folder once; later calls find
them there.
"""

from pathlib import Path
from typing import Iterable, Optional
from urllib.request import urlretrieve

# raw files of the modekit-data repository, branch main; point it elsewhere for
# a mirror
DATA_URL = "https://raw.githubusercontent.com/garavanis/modekit-data/main"


def fetch(folder: str, files: Iterable[str], cache_dir, url: Optional[str] = None) -> Path:
    """
    Download the files of a dataset that are not in ``cache_dir`` yet.

    Parameters
    ----------
    folder : str
        Dataset folder in the data repository, e.g. ``"heartspace"``.
    files : iterable of str
        File names inside that folder.
    cache_dir : path-like
        Local folder for the downloads; the files land in ``cache_dir / folder``,
        which is created.
    url : str, optional
        Base URL of the data repository. Default :data:`DATA_URL`.

    Returns
    -------
    pathlib.Path
        ``cache_dir / folder``, holding every requested file.
    """
    base = DATA_URL if url is None else url
    out = Path(cache_dir) / folder
    out.mkdir(parents=True, exist_ok=True)
    for name in files:
        target = out / name
        if target.exists():
            continue
        print(f"downloading {folder}/{name}")
        part = target.with_name(target.name + ".part")
        urlretrieve(f"{base}/{folder}/{name}", part)
        part.replace(target)  # an interrupted download leaves no half file behind
    return out
