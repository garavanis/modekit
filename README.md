# modekit

Modal analysis in JAX: natural frequencies, damping ratios and mode shapes
from tap tests (EMA) or from output-only records (OMA).

- **Spectra**: FRFs, coherence and power spectra of tap tests (`EmaModel`);
  correlations and half spectra of output-only records (`OmaModel`).
- **Poles**: band-wise pLSCF (PolyMAX) with stabilisation diagrams; the stable
  poles are clustered into modes and merged across bands and reference sets.
- **Mode shapes**: an LSFD residue fit at the identified poles.
- **Tracking**: small local pLSCF models follow known modes through the
  segments of a long record.
- **Also**: windows and preprocessing, mode matching and catalogues, tables
  (CSV, Excel, HTML), figures, and fits saved to disk.

Built on [JAX](https://github.com/jax-ml/jax) and
[Equinox](https://github.com/patrick-kidger/equinox). Importing `modekit`
switches JAX to 64-bit precision.

## Install

```bash
uv add git+https://github.com/garavanis/modekit
# or
pip install git+https://github.com/garavanis/modekit
```

`export.save_table(..., fmt="excel")` also needs `openpyxl`: install
`modekit[excel]`.

## Quick start

A hammer test: `x` is the force, `(n_reps, 1, nt)`, `y` the accelerations,
`(n_reps, Q, nt)`, both sampled at `fs` Hz.

```python
from modekit.freq_models import EmaModel
from modekit.bandmpe import full_band_mpe

ema = EmaModel(X=x, Y=y, fs=fs, exc_type="transient")
freqs, H = ema.get_frf(method="H1")                       # FRFs, (Q, 1, N)

# one pLSCF per band, at a fake sampling rate about 2.5x the band's upper edge;
# the bands' modes combined; then one LSFD residue fit over 2-560 Hz
bands = [(2, 120, 300.0), (120, 320, 800.0), (320, 560, 1400.0)]
alpha = None                                              # decay rate of the exponential window, if one was applied
fit, band_data = full_band_mpe([(freqs, H)], alpha, fs, bands, spectrum="frf_tap",
                               quantity="acceleration", refs="global")
lsfd, model = fit
model.Fn, model.Zeta                                      # natural frequencies [Hz], damping ratios
phi = model.mode_shapes(real=True, normalize="l2")        # mode shapes, (Q, M)

# diagnostics: a stabilisation chart per band, and the synthesis against the data
for flo, fhi, est, poles, res in band_data[0]:
    est.stab_plot(poles, data=H[:, :, (freqs >= flo) & (freqs <= fhi)], indicator="cmif", xlim=(flo, fhi))
lsfd.synthesis_plot(model, H, xlim=lsfd.band)
```

For output-only data the steps are the same, with `OmaModel` in place of
`EmaModel`, its half spectra in place of the FRFs, and `spectrum="sd_cor"`.
The example notebook goes through both cases in full.

## Example

`examples/garteur.ipynb` walks through tap tests on the GARTEUR structure:
EMA of the hammer tests (force and exponential windows, FRFs, band-wise pLSCF,
LSFD) and OMA of the fan-excited records. On its first run it downloads the
measurements it needs (about 450 MB) from
[modekit-data](https://github.com/garavanis/modekit-data) into
`examples/data/`; tables and figures go to `examples/outputs/` and
`examples/figures/`.

The figures use LaTeX text (`plt.style.use("modekit.latex")`). Without a LaTeX
installation, remove that line.

## Modules

| Module | Contents |
| --- | --- |
| `freq_models` | `EmaModel`, `OmaModel`: spectra, FRFs, coherence, correlations |
| `algorithms` | `pLSCF`, `LSFD`, `ModalModel`, and their plot methods |
| `bandmpe` | band-wise pLSCF, pass merging, local models, batched LSFD |
| `plscf`, `clustering`, `criteria` | the pLSCF core, DBSCAN on modal distance, MAC / MPC / MPD and stabilisation criteria |
| `matching` | modes matched across structures; catalogues of nominal frequencies |
| `reporting`, `export` | report tables; text, CSV, Excel and HTML output |
| `plots` | the figures |
| `serialisation` | saving and loading Equinox models |
| `dataprep` | tap-test loading, windows, detrending, decimation, filtering |
| `datasets` | the example data download |

## Tests

```bash
uv run pytest
```

## References

Many parts of `modekit` were inspired by pyOMA2:

- Pasca, D. P., Margoni, D. F. (2025). pyOMA2: A Python module for conducting
  operational modal analysis. *Journal of Open Source Software*, 10(115), 7656.
  https://doi.org/10.21105/joss.07656

Methods and practice:

- Peeters, B., Van der Auweraer, H., Guillaume, P., Leuridan, J. (2004). The
  PolyMAX frequency-domain method: a new standard for modal parameter
  estimation? *Shock and Vibration*, 11(3–4), 395–409.
- Peeters, B., Van der Auweraer, H. (2005). PolyMAX: a revolution in
  operational modal analysis. *Proceedings of the 1st International Operational
  Modal Analysis Conference (IOMAC)*, Copenhagen.
- Ester, M., Kriegel, H.-P., Sander, J., Xu, X. (1996). A density-based
  algorithm for discovering clusters in large spatial databases with noise.
  *Proceedings of the 2nd International Conference on Knowledge Discovery and
  Data Mining (KDD-96)*, 226–231. AAAI Press.
- Boroschek, R. L., Bilbao, J. A. (2019). Interpretation of stabilization
  diagrams using density-based clustering algorithm. *Engineering Structures*,
  178, 245–257. https://doi.org/10.1016/j.engstruct.2018.09.091
- Shin, K., Hammond, J. K. (2008). *Fundamentals of Signal Processing for
  Sound and Vibration Engineers*. Wiley.
- Farrar, C. R., Worden, K. (2013). *Structural Health Monitoring: A Machine
  Learning Perspective*, Appendix A: Signal Processing for SHM. Wiley.
- Avitabile, P. (2018). *Modal Testing: A Practitioner's Guide*. Wiley.

## License

MIT, see [LICENSE](LICENSE).
