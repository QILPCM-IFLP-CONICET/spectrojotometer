"""
generate_bonds lists every bond that contributes to the energy of the
cell, once: all the pairs of sites closer than the largest range, in any
cell, whatever the positions (inside the cell or not) and the shape of
the cell. Checked against a brute-force search.
"""

import itertools

import numpy as np
import pytest

from spectrojotometer.magnetic_model import MagneticModel
from spectrojotometer.model_io import magnetic_model_from_file
from spectrojotometer.tools import unpack_offset

from .conftest import EXAMPLES_DIR
from .test_cif_bond_symmetry import HEXAGONAL


def canonical(i, j, offset):
    offset = tuple(int(x) for x in offset)
    offset += (0,) * (3 - len(offset))  # with fewer than three Bravais vectors
    if i > j or (i == j and next((x for x in offset if x), 0) < 0):
        i, j, offset = j, i, tuple(-x for x in offset)
    return i, j, offset


def found(model):
    """The bonds of each coupling, by distance, as sets."""
    out = {}
    for info in model.bonds.values():
        bonds = [
            canonical(i, j, (0,) * 3 if tag in (".", None) else unpack_offset(tag))
            for i, j, tag in info["bonds"]
        ]
        assert len(bonds) == len(set(bonds))  # each bond once
        out[round(info["distance"], 2)] = set(bonds)
    return out


def brute_force(model, rmax, reach=4):
    pos = np.array(model.site_properties["coord_atomos"], dtype=float)
    lat = np.array(model.lattice_properties["bravais_vectors"], dtype=float)
    out = {}
    for i, j in itertools.product(range(len(pos)), repeat=2):
        for n in itertools.product(range(-reach, reach + 1), repeat=len(lat)):
            d = np.linalg.norm(pos[j] + np.array(n) @ lat - pos[i])
            if 0.01 < d < rmax:
                n3 = tuple(n) + (0,) * (3 - len(n))
                out.setdefault(round(round(d / 0.02) * 0.02, 2), set()).add(canonical(i, j, n3))
    return out


def test_hexagonal_structure_read_from_a_cif(tmp_path):
    """Its second site is stored at (-1/3, -2/3, 0.6): with a search in
    cells -1..1 around it, 2 of the 6 bonds between layers were missing."""
    path = tmp_path / "hexagonal.cif"
    path.write_text(HEXAGONAL)
    model = magnetic_model_from_file(str(path))
    model.generate_bonds(0.02, [[0, 6.0]])
    assert found(model) == brute_force(model, 6.0)
    assert sorted(len(b) for b in found(model).values()) == [6, 6]


def test_a_site_outside_the_cell():
    inside = MagneticModel(
        [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])],
        [np.array([2.0, 0.0, 0.0]), np.array([0.0, 9.0, 0.0]), np.array([0.0, 0.0, 9.0])],
    )
    outside = MagneticModel(
        [np.array([0.0, 0.0, 0.0]), np.array([1.0 + 3 * 2.0, 0.0, 0.0])],
        [np.array([2.0, 0.0, 0.0]), np.array([0.0, 9.0, 0.0]), np.array([0.0, 0.0, 9.0])],
    )
    for model in (inside, outside):
        model.generate_bonds(0.02, [[0, 2.5]])
        assert {d: len(b) for d, b in found(model).items()} == {1.0: 2, 2.0: 2}
        assert found(model) == brute_force(model, 2.5, reach=5)


def test_an_oblique_cell():
    """A cell sheared by 2.4 a: the nearest neighbours along b are two
    cells away in a, beyond a search in cells -1..1."""
    a, b, c = np.array([1.0, 0, 0]), np.array([2.4, 1.0, 0]), np.array([0, 0, 5.0])
    model = MagneticModel([np.zeros(3)], [a, b, c])
    model.generate_bonds(0.02, [[0, 1.5]])
    assert found(model) == brute_force(model, 1.5)
    # a, and b - 2a (length 1.077), b - 3a (1.166)
    assert sorted(found(model)) == [1.0, 1.08, 1.16]


@pytest.mark.parametrize("rmax", [3.5, 6.2])
def test_complete_on_the_spinel(rmax):
    model = magnetic_model_from_file(str(EXAMPLES_DIR / "cromita_ortogonal.cif"))
    model.generate_bonds(0.02, [[0, rmax]])
    assert found(model) == brute_force(model, rmax, reach=2)


def test_a_chain_and_a_ring():
    """With one Bravais vector, or none."""
    chain = MagneticModel([np.array([0.0, 0, 0]), np.array([0.7, 0.3, 0])], [np.array([1.5, 0, 0])])
    chain.generate_bonds(0.02, [[0, 2.0]])
    assert found(chain) == brute_force(chain, 2.0)
    ring = MagneticModel([np.array([np.cos(t), np.sin(t), 0.0]) for t in np.arange(6) * np.pi / 3], [])
    ring.generate_bonds(0.02, [[0, 1.5]])
    assert {d: len(b) for d, b in found(ring).items()} == {1.0: 6}


def test_offsets_that_cannot_be_stored_are_an_error():
    """A cell offset beyond -5..4 was stored wrapped (5 as -5)."""
    model = MagneticModel([np.zeros(3)], [np.array([1.0, 0, 0]), np.array([0, 9.0, 0]), np.array([0, 0, 9.0])])
    with pytest.raises(ValueError, match="cell offset"):
        model.generate_bonds(0.02, [[0, 5.5]])


def test_generating_twice_does_not_duplicate_bonds():
    """The second call finds the same bonds, which the couplings already
    have: they are not added again (they were, doubling the coefficients)."""
    model = magnetic_model_from_file(str(EXAMPLES_DIR / "example1.cif"))
    model.generate_bonds(0.02, [[0, 4]])
    first = {name: list(c["bonds"]) for name, c in model.bonds.items()}
    model.generate_bonds(0.02, [[0, 4]])
    assert {name: c["bonds"] for name, c in model.bonds.items()} == first
