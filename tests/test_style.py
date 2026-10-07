"""
The figure style: a style sheet a notebook opts into; importing modekit changes
no matplotlib setting.
"""

import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def test_the_style_sheet_turns_on_latex_text():
    with plt.style.context("modekit.latex"):  # by name: latex.mplstyle inside the modekit package
        rc = dict(matplotlib.rcParams)
    assert rc["text.usetex"] is True and rc["font.family"] == ["sans-serif"]
    assert rc["font.sans-serif"] == ["Computer Modern Sans Serif"]  # a name LaTeX mode knows
    assert matplotlib.rcParams["text.usetex"] is False  # the context put it back


def test_imports_leave_matplotlib_alone():
    """Importing the figure modules sets no LaTeX or font: a notebook opts in with
    plt.style.use("modekit.latex"). Run in a fresh interpreter, unaffected by this one."""
    code = (
        "import matplotlib; matplotlib.use('Agg'); "
        "keys = [k for k in matplotlib.rcParams if k.startswith(('text.', 'font.', 'mathtext.'))]; "
        "before = {k: matplotlib.rcParams[k] for k in keys}; "
        "import modekit, modekit.plots, modekit.algorithms; "
        "print([k for k in keys if matplotlib.rcParams[k] != before[k]])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=os.environ)
    assert out.stdout.strip() == "[]"
