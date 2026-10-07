# modekit

Modal parameter estimation in Python, for experimental (EMA) and operational
(OMA) modal analysis.

- **Measurement models**: FRFs, coherence and power spectra of tap tests
  (`EmaModel`); correlations and half-spectra of output-only records
  (`OmaModel`).
- **Band-wise pLSCF (PolyMAX)**: one fit per frequency band on a fake sampling
  rate, stabilisation diagrams, stable poles clustered into modes (DBSCAN),
  modes merged across bands and reference passes.
- **LSFD**: residue fits at the identified poles, with the participation
  factors held fixed, for the mode shapes.
- **Tracking**: local low-order pLSCF models follow known modes over the
  segments of long records.
- **Around the fits**: signal preparation and windows, mode matching and
  catalogues, report tables (CSV, Excel, HTML), figures, and fits cached on
  disk as Equinox files.

Built on [JAX](https://github.com/jax-ml/jax) and
[Equinox](https://github.com/patrick-kidger/equinox). Importing `modekit`
switches JAX to 64-bit precision, which the identification needs.

## Install

```bash
uv add git+https://github.com/garavanis/modekit
# or
pip install git+https://github.com/garavanis/modekit
```

`export.save_table(..., fmt="excel")` also needs `openpyxl`: install
`modekit[excel]`.

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

Many parts of modekit were inspired by pyOMA2:

- Pasca, D. P., Margoni, D. F. (2025). pyOMA2: A Python module for conducting
  operational modal analysis. *Journal of Open Source Software*, 10(115), 7656.
  https://doi.org/10.21105/joss.07656

Methods and practice:

- Peeters, B., Van der Auweraer, H. (2005). PolyMAX: a revolution in
  operational modal analysis. *Proceedings of the 1st International Operational
  Modal Analysis Conference (IOMAC)*, Copenhagen.
- Avitabile, P. (2018). *Modal Testing: A Practitioner's Guide*. Wiley.

## License

MIT, see [LICENSE](LICENSE).
