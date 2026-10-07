"""
The figure styles: two sheets a notebook opts into by package name; importing
modekit changes no matplotlib setting.
"""

import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def test_both_style_sheets_resolve_by_package_name():
    with plt.style.context("modekit.latex"):  # latex.mplstyle inside the package
        assert matplotlib.rcParams["text.usetex"] is True
    with plt.style.context("modekit.mathtext"):  # Computer Modern maths, no LaTeX needed
        assert matplotlib.rcParams["text.usetex"] is False
        assert matplotlib.rcParams["mathtext.fontset"] == "cm"
        fig, ax = plt.subplots()
        ax.set_xlabel(r"$f_{n,1}$ [Hz]")
        fig.canvas.draw()  # renders the maths with matplotlib's own fonts
        plt.close(fig)
    assert matplotlib.rcParams["text.usetex"] is False  # the contexts put everything back


def test_importing_modekit_leaves_matplotlib_alone():
    """``import modekit`` (which imports every module) sets no LaTeX, font or colormap:
    a notebook opts in with plt.style.use. Run in a fresh interpreter, unaffected by
    this one."""
    code = (
        "import matplotlib; matplotlib.use('Agg'); "
        "keys = [k for k in matplotlib.rcParams if k.startswith(('text.', 'font.', 'mathtext.'))]; "
        "before = {k: matplotlib.rcParams[k] for k in keys}; "
        "import modekit; "
        "print([k for k in keys if matplotlib.rcParams[k] != before[k]])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=os.environ)
    assert out.stdout.strip() == "[]"
