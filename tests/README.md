# Tests

## How to run

```bash
pip install -e ".[test]"
pytest
```

The tests marked `slow` (long Monte Carlo simulations, validated against
known critical temperatures) are skipped by default; run them with
`pytest --run-slow`. The CI always does.

The tests of the graphical interface (`test_gui.py`, and one test in
`test_logging.py`) need a display: without one they are skipped. On a
headless machine, run them with `xvfb-run pytest`, as the CI does.

The tests that start Python subprocesses (command line tools, imports in a
fresh interpreter) run the code of the repository, not another installed
copy of spectrojotometer (see `_helpers.repo_env`).

## Organization

Helpers and data:

- `conftest.py` — fixtures with the paths of the data (`examples_dir`,
  `fixtures_dir`); `bond_length` / `all_bond_lengths`, which recompute the
  actual distance of a bond from the Cartesian positions and the Bravais
  lattice of a loaded model, instead of trusting the value declared in the
  input file; `assert_models_equivalent`, used to compare the two CIF
  readers.
- `_helpers.py` — `ROOT`, the root of the repository, and `repo_env()`,
  the environment for subprocesses.
- `fixtures/` — small synthetic geometries whose positions and bond
  distances can be checked by hand (unlike the files in `examples/`, which
  are real data, not necessarily self-consistent).

Reading models (`spectrojotometer/model_io/`):

- `test_cif_loading.py` — the hand-written CIF reader: atoms, species,
  symmetry expansion and `primitive_cell=True`, bonds.
- `test_cif_loading_pymatgen.py` — the same checks on the pymatgen-based
  reader, on its own, so that a bug shared by both readers is not missed.
- `test_cif_parser_equivalence.py` — both readers, file by file: same
  atoms, cell and bonds.
- `test_struct_loading.py` — WIEN2k `.struct` files.
- `test_dispatch.py` — `magnetic_model_from_file` picks the reader by the
  file extension (case insensitive).
- `test_symmetry_helpers.py` — low-level functions: `parse_symmetry`,
  `centering_letter_from_symbol`, `expand_symmetries_with_centering`,
  `normalize_bond`, `pack_offset` / `unpack_offset`.
- `test_geometry_consistency.py` — invariants: saving with `save_cif` and
  reading back reproduces the geometry; the conventional and primitive
  cells of a structure have the same atomic density.

The model (`spectrojotometer/magnetic_model.py`):

- `test_magnetic_model_init.py` — the constructor: defaults, values per
  site, bonds, deprecated arguments; the English names of
  `coefficient_matrix(centered=)` and `remove_bond_by_name`.
- `test_magnetic_model.py` — configurations, `optimize_independent_set`,
  `find_optimal_configurations`, `save_cif` (dimensions and angles of the
  cell), and bond names that do not depend on the hash seed.
- `test_compatibility_region.py` — the region of couplings compatible
  with the energies: exact box, Monte Carlo and ellipse estimates, their
  invariance under a change of energy units, `bound_inequalities`.
- `test_error_bounds.py` — `error_bounds` / `cost`, the largest error of
  each coupling per unit error in the energies, and the choice of
  configurations based on it (including a comparison with the exhaustive
  search on a small model, and the 5% rule for the number of
  configurations).
- `test_model_mapping.py` — `site_map` and `map_configurations`.
- `test_generate_bonds.py` — `generate_bonds` finds every bond within the
  range, against a brute-force search: ranges longer than the cell, sites
  outside the cell, oblique cells, chains and rings, offsets that cannot be
  stored.
- `test_cif_bond_symmetry.py` — CIF bonds declared through symmetry codes
  (`_geom_bond_site_symmetry`), with both readers, against all the pairs of
  sites at each distance found by brute force (a spinel and a hexagonal
  structure); cell offsets, bonds between a site and its own image.
- `test_spin_configurations.py` — reading spin configurations: the errors
  and their line numbers, padding, comments; the file reader, visualbond's
  main window and the output of `optimize-configurations`.

The Monte Carlo simulation (`spectrojotometer/monte_carlo/`):

- `test_monte_carlo.py` — the engine, Wolff, parallel tempering, the
  detection of frustration, the analysis of the results.
- `test_monte_carlo_integration.py` — the simulation describes the fitted
  model: the lattice built by the adapter (every bond, also across cells,
  at the distance of its coupling, on a real model), the same sign
  convention as the fit (energies equal for every configuration), the
  values stored by `compute_couplings`, Wolff and Metropolis against the
  exact energy of a small antiferromagnetic ring, the warning about
  disagreeing replicas; the couplings editor and the panel, including
  the scrolling of its controls in a short window.
- `test_monte_carlo_wolff.py` — a Wolff sweep flips about N spins (a fixed
  number of clusters, estimated at each temperature), and the algorithm
  samples the Boltzmann distribution: energy, specific heat and staggered
  magnetization against the exact enumeration of a 4x4 antiferromagnet.
- `test_mc_gui.py` — the adapter and the Monte Carlo panel.

Interfaces and packaging:

- `test_cli.py` — the commands declared in `pyproject.toml` exist and
  answer to `--help`; each one is also run on the examples.
- `test_gui.py` — visualbond's main window and its import window, driven
  programmatically.
- `test_logging.py` — the library reports through `logging` and prints
  nothing by itself; the log panel of the GUI; the verbosity of the
  commands.
- `test_imports.py` — every module imports in a fresh interpreter, where
  nothing has been imported before (this caught a missing
  `import importlib.util` that only failed in regular installations).

## Formerly known issue

A bond declared with `_geom_bond_site_symmetry_2` pointing at a
non-identity symmetry operator (e.g. `"2_555"`) was silently dropped by
both CIF readers: the image of the operator was looked up under the
label of the previous one. The fixture `synthetic_bond_via_symmetry.cif`
reproduced it, and `test_cif_parser_equivalence.py` documented it as
`xfail`; it is fixed (the ends of a bond are found by their position),
and `test_cif_bond_symmetry.py` checks such bonds against all the pairs
of sites, found by brute force.
