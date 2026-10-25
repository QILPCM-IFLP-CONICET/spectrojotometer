"""
The ZnCr2O4 examples: the Cr sublattice of the cubic spinel, with its
couplings (CIF), and three cells relaxed by DFT (WIEN2k .struct): cubic,
orthorhombic, and tetragonal (half the volume, rotated 45 degrees).
"""

import logging
import re
import subprocess
import sys

import numpy as np
import pytest

from spectrojotometer.magnetic_model import _geometry_copy, equivalent_model, site_maps
from spectrojotometer.model_io import magnetic_model_from_file
from spectrojotometer.tools import unpack_offset

from ._helpers import repo_env
from .conftest import EXAMPLES_DIR, PROJECT_ROOT

RELAXED = {
    "FERRO": (16, "position"),
    "ORTOFERRO": (16, "position"),
    "TETRAFERRO": (8, "structure"),
}


def length(model, bond):
    i, j, tag = bond
    positions = np.asarray(model.site_properties["coord_atomos"], dtype=float)
    lattice = np.asarray(model.lattice_properties["bravais_vectors"], dtype=float)
    return float(np.linalg.norm(positions[j] + unpack_offset(tag) @ lattice - positions[i]))


def canonical(bonds):
    result = set()
    for i, j, tag in bonds:
        shift = tuple(int(n) for n in unpack_offset(tag))
        if i > j:
            i, j, shift = j, i, tuple(-n for n in shift)
        result.add((i, j, shift))
    return result


@pytest.fixture(scope="module")
def cubic():
    return magnetic_model_from_file(str(EXAMPLES_DIR / "ZnCr2O4_cubica.cif"))


def test_the_cubic_model(cubic, caplog):
    with caplog.at_level(logging.WARNING, logger="spectrojotometer"):
        model = magnetic_model_from_file(str(EXAMPLES_DIR / "ZnCr2O4_cubica.cif"))
    assert "old convention" not in caplog.text  # standard bond codes
    assert model.num_sites == 16
    assert {name: len(c["bonds"]) for name, c in cubic.bonds.items()} == {
        "J1": 48, "J2": 96, "J3a": 48, "J3b": 48,
    }
    # Every bond has the declared length.
    for coupling in cubic.bonds.values():
        for bond in coupling["bonds"]:
            assert length(cubic, bond) == pytest.approx(float(coupling["distance"]), abs=1e-3)
    # J1: the nearest neighbours, as generate_bonds finds them.
    nearest = _geometry_copy(cubic)
    nearest.generate_bonds(0.02, [[0, 3.2]])
    assert canonical(nearest.bonds["J1"]["bonds"]) == canonical(cubic.bonds["J1"]["bonds"])


@pytest.mark.parametrize("name", sorted(RELAXED))
def test_the_relaxed_cells_align_onto_the_cubic_one(cubic, name):
    num_sites, method = RELAXED[name]
    relaxed = magnetic_model_from_file(str(EXAMPLES_DIR / f"ZnCr2O4_{name}.struct"))
    assert relaxed.num_sites == num_sites
    assert set(relaxed.site_properties["magnetic_species"]) == {"Cr"}
    assert site_maps(relaxed, cubic)[2] == method
    aligned = equivalent_model(relaxed, cubic, align=True)
    shift = np.linalg.norm(
        np.asarray(aligned.site_properties["coord_atomos"])
        - np.asarray(cubic.site_properties["coord_atomos"]),
        axis=1,
    )
    assert shift.max() < 0.02
    # With the cubic model as reference, generate_bonds gives its J1.
    aligned.generate_bonds(0.02, [[0, 3.2]])
    assert list(aligned.bonds) == ["J1"]
    assert canonical(aligned.bonds["J1"]["bonds"]) == canonical(cubic.bonds["J1"]["bonds"])
    # The lengths of the relaxed cell, in its own cell (no alignment):
    # the same multiset as through the aligned model.
    relaxed.generate_bonds(1e-4, [[0, 3.2]])
    own = sorted(length(relaxed, b) for c in relaxed.bonds.values() for b in c["bonds"])
    via = sorted(length(aligned, b) for b in cubic.bonds["J1"]["bonds"])
    copies = 16 // num_sites
    np.testing.assert_allclose(sorted(own * copies), via, atol=2e-5)


def plane(cubic, bond):
    i, j, tag = bond
    positions = np.asarray(cubic.site_properties["coord_atomos"], dtype=float)
    lattice = np.asarray(cubic.lattice_properties["bravais_vectors"], dtype=float)
    vector = positions[j] + unpack_offset(tag) @ lattice - positions[i]
    return "".join(axis for axis, x in zip("xyz", vector) if abs(x) > 0.1)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("FERRO", {"xy": 0.0, "xz": 0.0, "yz": 0.0}),
        ("ORTOFERRO", {"xy": -5.46, "xz": -1.44, "yz": 1.38}),
        ("TETRAFERRO", {"xy": -0.04, "xz": -2.49, "yz": -2.49}),
    ],
)
def test_the_j1_length_changes(cubic, name, expected):
    """Mean change of the J1 bonds in each plane, in mÅ."""
    relaxed = magnetic_model_from_file(str(EXAMPLES_DIR / f"ZnCr2O4_{name}.struct"))
    aligned = equivalent_model(relaxed, cubic, align=True)
    changes = {}
    for bond in cubic.bonds["J1"]["bonds"]:
        changes.setdefault(plane(cubic, bond), []).append(
            1000 * (length(aligned, bond) - length(cubic, bond))
        )
    assert {key: len(values) for key, values in changes.items()} == {"xy": 16, "xz": 16, "yz": 16}
    for key, value in expected.items():
        assert np.mean(changes[key]) == pytest.approx(value, abs=0.02)


def test_the_readme_snippet():
    """The code of "Comparing distorted phases" in the README prints what
    the README says."""
    text = (PROJECT_ROOT / "README.md").read_text()
    section = text.split("Comparing distorted phases\n", 1)[1]
    code = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    printed = re.search(r"prints\n\n((?:    .*\n)+)", section).group(1)
    expected = "\n".join(line[4:] for line in printed.rstrip("\n").split("\n"))
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=repo_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected
