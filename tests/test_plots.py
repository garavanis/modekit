"""
Smoke tests for the modal figures: the plot methods of the identification
results, the signal plots, the 3D mode-shape renderer, the synthesis overlay, the
MAC matrix and the mode complexity.
"""

import warnings
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from test_plscf import F_TRUE, FS, ORDER, ORDMAX, _mode_shapes, _synth_frf  # noqa: E402
from modekit import plots  # noqa: E402
from modekit.algorithms import LSFD, pLSCF  # noqa: E402



class Fit(NamedTuple):
    H: np.ndarray        # the synthetic FRF
    algo: pLSCF
    poles: object        # its PoleArrays
    lsfd: LSFD
    model: object        # the LSFD ModalModel at the identified modes


@pytest.fixture(scope="module")
def fit():
    """pLSCF and LSFD on the synthetic FRF of ``test_plscf``."""
    freq, H = _synth_frf(_mode_shapes())
    algo = pLSCF(freq, fs=FS, ordmax=ORDMAX, spectrum="frf_shaker")
    poles = algo.fit(H)
    lsfd = LSFD(freq, fs=FS, spectrum="frf_shaker", quantity="displacement")
    Lambd = poles.mpe(F_TRUE, order_in=ORDER - 1, rtol=0.1)[0]
    return Fit(H, algo, poles, lsfd, lsfd.fit(H, Lambd))


# the result-class plot methods no other test draws (test_clustering draws the cluster ones)
METHODS = {
    "stab_plot": lambda f: f.algo.stab_plot(f.poles),
    "indicator_plot": lambda f: f.algo.indicator_plot(f.H),
    "freqs_zetas_plot": lambda f: f.algo.freqs_zetas_plot(f.poles),
    "synthesis_plot": lambda f: f.lsfd.synthesis_plot(f.model, f.H),
}


@pytest.mark.parametrize("method", METHODS)
def test_result_plot_methods_draw_through_the_plots_module(fit, method):
    out = METHODS[method](fit)
    fig = out[0] if isinstance(out, tuple) else out
    assert isinstance(fig, plt.Figure) and fig.axes
    plt.close(fig)


def test_plot_signal_labels_one_panel_per_live_channel():
    x = np.arange(4.0)
    signal = np.ones((3, 4))
    signal[1] = 0.0  # a silent channel: no panel, and its label goes with it
    fig, axes = plots.plot_signal(x, signal, y_labels=["a", "b", "c"])
    assert [ax.get_ylabel() for ax in axes] == ["a", "c"]
    plt.close(fig)


def test_plot_signal_ci_labels_one_panel_per_live_channel():
    x = np.arange(4.0)
    signal = np.ones((5, 3, 4))
    signal[:, 1] = 0.0  # a silent channel: no panel, and its label goes with it
    fig, axes = plots.plot_signal_ci(x, signal, y_labels=["a", "b", "c"])
    assert [ax.get_ylabel() for ax in axes] == ["a", "c"]
    plt.close(fig)
    fig, axes = plots.plot_signal_ci(x, signal)  # no labels by default
    assert [ax.get_ylabel() for ax in axes] == ["", ""]
    plt.close(fig)


def test_plot_mode_wireframe_draws_a_structure_spanning_three_axes():
    # a T of two members spanning x, y and z: the bar's tips move up, the slanted stem stays put
    undeformed = [np.array([[-1.0, 0, 1], [0, 0, 1], [1, 0, 1]]), np.array([[0.0, -0.5, 0], [0, 0, 1]])]
    deformed = [undeformed[0] + [[0, 0, 0.2], [0, 0, 0], [0, 0, 0.2]], undeformed[1]]
    base = np.array([[-1.0, 0, 1], [1, 0, 1]])
    fig, ax = plots.plot_mode_wireframe(
        undeformed, deformed, sensor_points=base + [0, 0, 0.2], sensor_base=base, show_axes=False
    )
    assert ax.name == "3d"
    assert ax.lines and ax.collections  # the skeleton polylines; the sensors and arrows
    # each axis spans exactly the drawn points
    assert ax.get_xlim() == (-1.0, 1.0) and ax.get_ylim() == (-0.5, 0.0) and ax.get_zlim() == (0.0, 1.2)
    plt.close(fig)


def test_plot_mode_wireframe_draws_a_flat_structure_without_warning():
    # every point at y = 0: the y axis gets a unit-wide range around it, the others their own
    undeformed = [np.array([[-1.0, 0, 1], [1, 0, 1]]), np.array([[0.0, 0, 0], [0, 0, 1]])]
    deformed = [undeformed[0] + [0, 0, 0.2], undeformed[1]]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig, ax = plots.plot_mode_wireframe(undeformed, deformed, zero_z=False)
    assert ax.get_ylim() == (-0.5, 0.5)
    assert ax.get_xlim() == (-1.0, 1.0) and ax.get_zlim() == (0.0, 1.2)
    plt.close(fig)


def test_plot_synthesis_labels_the_magnitude_panels():
    x = np.arange(4.0)
    H = np.ones((2, 1, 4), dtype=complex)
    fig, axes = plots.plot_synthesis(x, H, H, phase=True, y_labels=["q1", "q2"])
    assert [ax.get_ylabel() for ax in axes[::2]] == ["q1", "q2"]  # magnitude, phase, ...
    assert axes[0].phase_ax.get_ylabel() == "Phase [deg]"
    plt.close(fig)


def test_plot_mac_matrix_shows_the_mac_of_the_two_sets():
    phi, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((6, 3)))  # orthonormal shapes
    fig, ax = plots.plot_mac_matrix(phi, phi[:, :2])
    np.testing.assert_allclose(ax.images[0].get_array(), np.eye(3, 2), atol=1e-12)  # (X, A)
    assert len(ax.texts) == 6  # one value per cell
    plt.close(fig)


def test_plot_mode_complexity_draws_one_arrow_per_finite_dof():
    phi = np.array([1.0, 0.5 + 0.1j, -0.8, np.nan])  # the NaN DOF is left out
    fig, ax = plots.plot_mode_complexity(phi)
    assert ax.name == "polar"
    assert len(ax.texts) == 3
    plt.close(fig)


def test_save_fig_name_is_a_path_and_its_folder_is_created(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a relative name lands in the working directory
    fig, _ = plots.plot_mode_complexity(np.array([1.0, 0.5j]), save_fig_name="figs/c.png")
    plt.close(fig)
    assert (tmp_path / "figs" / "c.png").exists()


def test_plot_samples_labels_one_per_feature():
    x = [20.0, 30.0]
    samples = np.random.default_rng(0).normal(10.0, 1.0, size=(8, 3, 2))
    samples[:, 1, 0] = np.nan  # a feature missing at one point is still labelled once
    fig, ax = plots.plot_samples(x, samples, labels=["r1", "r4", "r7"])
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["r1", "r4", "r7"]
    plt.close(fig)
    fig, ax = plots.plot_samples(x, samples)  # no labels: no legend, no axis labels
    assert ax.get_legend() is None and ax.get_xlabel() == "" and ax.get_ylabel() == ""
    plt.close(fig)
    with pytest.raises(ValueError, match="labels"):
        plots.plot_samples(x, samples, labels=["r1"])


def test_plot_pdfs_labels_one_panel_per_feature():
    samples = np.random.default_rng(0).normal(size=(30, 2))
    fig, axes = plots.plot_pdfs(samples, y_labels=["row 4", "row 7"])
    assert [ax.get_ylabel() for ax in axes] == ["row 4", "row 7"]
    plt.close(fig)
    with pytest.raises(ValueError, match="y_labels"):
        plots.plot_pdfs(samples, y_labels=["only one"])
