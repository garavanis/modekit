"""
Rendering and export of tables.

:func:`to_text` renders any :class:`pandas.DataFrame` as fixed-width text with
per-column formats, for ``print``; :func:`save_table` writes one as CSV, Excel
or HTML files at the path given. Neither knows any column by name: the
formats come from the table's own ``attrs["formats"]`` (every table of
:mod:`modekit.reporting` carries them, and any frame can) and from
per-call overrides; other float columns print with four significant digits.
"""

from pathlib import Path
from typing import Callable, List, Mapping, Optional, Sequence, Union

import pandas as pd
from pandas.api.types import is_float_dtype

_EXT = {"csv": ".csv", "excel": ".xlsx", "xlsx": ".xlsx", "html": ".html"}

_FLOAT_DEFAULT = "{:.4g}"  # float columns with no format of their own


# =============================================================================
# text
# =============================================================================


def _formatter(spec: Union[str, Callable], na_rep: str) -> Callable:
    """Cell formatter from a ``str.format`` template or a callable; NaN-safe."""
    f = spec if callable(spec) else spec.format
    return lambda v: na_rep if pd.isna(v) else f(v)


def _formatters(df: pd.DataFrame, formats: Optional[Mapping], na_rep: str) -> dict:
    """Per-column formatters: the table's own, then the overrides."""
    fmts = {**df.attrs.get("formats", {}), **(formats or {})}
    out = {}
    for col in df.columns:
        spec = fmts.get(col)
        if spec is None:
            if not is_float_dtype(df[col]):
                continue
            spec = _FLOAT_DEFAULT
        out[col] = _formatter(spec, na_rep)
    return out


def to_text(
    df: pd.DataFrame,
    formats: Optional[Mapping[str, Union[str, Callable]]] = None,
    *,
    index: bool = True,
    na_rep: str = "-",
) -> str:
    """
    Fixed-width text rendering of a table, for ``print``.

    Parameters
    ----------
    df : pandas.DataFrame
        Any table, e.g. of :mod:`modekit.reporting`.
    formats : mapping {column: format}, optional
        Overrides by column name: a ``str.format`` template (``"{:.1f}"``,
        ``"{:.0%}"``) or a callable. Defaults come from the table's own
        ``attrs["formats"]``; other float columns use ``{:.4g}``.
    index : bool, optional
        Print the index. Default True.
    na_rep : str, optional
        Rendering of missing cells. Default ``"-"``.

    Returns
    -------
    str
    """
    return df.to_string(
        formatters=_formatters(df, formats, na_rep), na_rep=na_rep, index=index
    )


# =============================================================================
# files
# =============================================================================


# CSS for the standalone HTML export
_HTML_CSS = """\
  body { font-family: system-ui, "Segoe UI", Arial, sans-serif; margin: 24px; color: #222; }
  h1 { font-size: 1.15rem; font-weight: 600; margin: 0 0 12px; }
  table.summary { border-collapse: collapse; font-variant-numeric: tabular-nums; }
  table.summary th, table.summary td { padding: 6px 12px; border: 1px solid #ddd; text-align: right; }
  table.summary th { background: #f2f2f2; font-weight: 600; }
  table.summary tbody tr:nth-child(odd) { background: #fafafa; }
  table.summary tbody tr:hover { background: #eef4ff; }
"""


def _write_excel(df: pd.DataFrame, path: Path, index: bool) -> None:
    """Write ``df`` to ``path``, auto-sizing each column to its widest cell."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=index, sheet_name="summary")
        ws = writer.sheets["summary"]
        for col in ws.columns:
            widest = max(
                (len(str(c.value)) for c in col if c.value is not None), default=0
            )
            letter = next(
                (c.column_letter for c in col if getattr(c, "column_letter", None)),
                None,
            )
            if letter is not None:
                ws.column_dimensions[letter].width = min(widest + 2, 50)


def _write_html(
    df: pd.DataFrame,
    path: Path,
    index: bool,
    title: str,
    float_format: str,
    formatters: dict,
) -> None:
    """Write ``df`` to ``path`` as a self-contained, lightly-styled HTML page."""

    def fmt(v):
        return float_format % v

    # na_rep renders any missing cell as an em-dash
    table = df.to_html(
        index=index,
        border=0,
        classes="summary",
        float_format=fmt,
        formatters=formatters,
        na_rep="—",
    )
    html = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{title}</title>\n<style>\n{_HTML_CSS}</style>\n"
        f"</head>\n<body>\n<h1>{title}</h1>\n{table}\n</body>\n</html>\n"
    )
    path.write_text(html, encoding="utf-8")


def save_table(
    df: pd.DataFrame,
    path: Union[str, Path],
    fmt: Union[str, Sequence[str]] = "csv",
    float_format: str = "%.4f",
    title: Optional[str] = None,
    formats: Optional[Mapping[str, Union[str, Callable]]] = None,
) -> List[str]:
    """
    Write a table as one or more files: ``path`` plus each format's extension.

    CSV and Excel keep the raw values; the HTML page renders the columns as
    :func:`to_text` does (percentages and all).

    Parameters
    ----------
    df : pandas.DataFrame
        Any table, e.g. of :mod:`modekit.reporting`.
    path : str or pathlib.Path
        Folder and file stem of the files written: an extension given is
        dropped, each format adds its own. A relative path is relative to the
        current working directory. The folder is created.
    fmt : str or sequence of str, optional
        Any of ``"csv"``, ``"excel"`` / ``"xlsx"``, ``"html"``. A sequence
        writes every requested format. Default ``"csv"``.
    float_format : str, optional
        printf-style float format for CSV, and for the HTML columns without a
        format of their own (Excel keeps full precision).
    title : str, optional
        Heading / page title for the HTML export. Defaults to the file's stem.
    formats : mapping {column: format}, optional
        Per-column format overrides for the HTML export, as for
        :func:`to_text`.

    Returns
    -------
    list of str
        The paths written, one per format.
    """
    fmts = [fmt] if isinstance(fmt, str) else list(fmt)
    unknown = sorted({f for f in fmts if f.lower() not in _EXT})
    if unknown:
        raise ValueError(
            f"Unknown format(s) {unknown}; choose from {sorted(set(_EXT))}."
        )

    base = Path(path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    # a bare RangeIndex carries no data; anything else (labels, a named
    # index, a MultiIndex) is written
    write_index = not (isinstance(df.index, pd.RangeIndex) and df.index.name is None)
    title = title if title is not None else base.name

    paths = []
    for f in fmts:
        f = f.lower()
        path = base.with_suffix(_EXT[f])
        if f == "csv":
            df.to_csv(path, index=write_index, float_format=float_format)
        elif f in ("excel", "xlsx"):
            _write_excel(df, path, write_index)
        else:  # html
            _write_html(
                df, path, write_index, title, float_format, _formatters(df, formats, "—")
            )
        paths.append(str(path))
    return paths
