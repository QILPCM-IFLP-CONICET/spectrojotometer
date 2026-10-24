"""
Bonds declared in a CIF with _geom_bond_site_symmetry codes (`n_klm`: the
image of an atom under the n-th symmetry operator of the file, moved by
(k-5, l-5, m-5) cells), read by both CIF readers and checked against all
the pairs of sites at each distance, found by brute force.
"""

import itertools
import logging

import numpy as np
import pytest

from spectrojotometer.model_io import magnetic_model_from_file
from spectrojotometer.model_io.cif import (
    find_atom_offset_by_symmetry,
    magnetic_model_from_cif,
    normalize_bond,
    parse_symmetry,
)
from spectrojotometer.model_io.model_io_pymatgen import magnetic_model_from_cif_pymatgen
from spectrojotometer.tools import unpack_offset

from .conftest import EXAMPLES_DIR

READERS = [magnetic_model_from_cif, magnetic_model_from_cif_pymatgen]

HEXAGONAL_OPS = ["x, y, z", "-y, x-y, z", "-x+y, -x, z", "-x, -y, z+1/2", "y, -x+y, z+1/2", "x-y, x, z+1/2"]

HEXAGONAL = """data_hexagonal
_cell_length_a   5.0
_cell_length_b   5.0
_cell_length_c   8.0
_cell_angle_alpha   90.0
_cell_angle_beta    90.0
_cell_angle_gamma   120.0
loop_
_space_group_symop_operation_xyz
""" + "".join(f"'{op}'\n" for op in HEXAGONAL_OPS) + """loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Cr1 Cr 0.33333333 0.66666667 0.1
"""


def canonical(i, j, offset):
    """A bond in a canonical form, independent of the library."""
    offset = tuple(int(x) for x in offset)
    if i > j or (i == j and next((x for x in offset if x), 0) < 0):
        i, j, offset = j, i, tuple(-x for x in offset)
    return i, j, offset


def offset_of(tag):
    if tag in (None, ".", ""):
        return (0, 0, 0)
    return tuple(unpack_offset(tag)) if isinstance(tag, str) else tuple(tag)


def shells(model, rmax):
    """All the bonds shorter than rmax, by brute force, by distance."""
    pos = np.array(model.site_properties["coord_atomos"], dtype=float)
    lat = np.array(model.lattice_properties["bravais_vectors"], dtype=float)
    out = {}
    for i, j in itertools.product(range(len(pos)), repeat=2):
        for off in itertools.product(range(-2, 3), repeat=3):
            d = np.linalg.norm(pos[j] + np.array(off) @ lat - pos[i])
            if 0.01 < d < rmax:
                out.setdefault(round(d, 3), set()).add(canonical(i, j, off))
    return [out[d] for d in sorted(out)], sorted(out)


def declare(model, sets, distances, n_asym, ops):
    lat = np.array(model.lattice_properties["bravais_vectors"], dtype=float)
    frac = np.linalg.solve(lat.T, np.array(model.site_properties["coord_atomos"], dtype=float).T).T
    symops = [parse_symmetry(op) for op in ops]
    rows = []
    for k, (bonds, d) in enumerate(zip(sets, distances)):
        for i, j, off in sorted(bonds):
            for a, b, o in ((i, j, np.array(off)), (j, i, -np.array(off))):
                if a >= n_asym:
                    continue
                partner = frac[b] + o
                code = next(
                    (f"Cr{c + 1}", f"{n + 1}_{cell[0] + 5}{cell[1] + 5}{cell[2] + 5}")
                    for n, (R, t) in enumerate(symops)
                    for c in range(n_asym)
                    for cell in [np.rint(partner - (R @ frac[c] + t)).astype(int)]
                    if np.allclose(partner - (R @ frac[c] + t), cell, atol=1e-6)
                )
                rows.append(f"Cr{a + 1} {code[0]} {d:.4f} J{k + 1} {code[1]}")
                break
    return (
        "loop_\n_geom_bond_atom_site_label_1\n_geom_bond_atom_site_label_2\n"
        "_geom_bond_distance\n_geom_bond_label\n_geom_bond_site_symmetry_2\n"
        + "\n".join(rows)
        + "\n"
    )


@pytest.mark.parametrize("reader", READERS, ids=["hand-written", "pymatgen"])
@pytest.mark.parametrize(
    "structure",
    ["spinel (centring translations)", "hexagonal (rotations, screw axis)"],
)
def test_bonds_declared_by_symmetry_codes(tmp_path, reader, structure):
    """Every bond, once, in its coupling. Before: missing bonds, bonds
    counted twice, and bonds that do not exist (the image of a non-identity
    operator was looked up under another label, cell offsets were
    truncated, images out of the home cell were dropped)."""
    if structure.startswith("spinel"):
        text = (EXAMPLES_DIR / "cromita_ortogonal.cif").read_text()
        ops = ["x, y, z", "x, y+1/2, z+1/2", "x+1/2, y, z+1/2", "x+1/2, y+1/2, z"]
        n_asym, rmax = 4, 6.2
    else:
        text, ops, n_asym, rmax = HEXAGONAL, HEXAGONAL_OPS, 1, 6.0
    path = tmp_path / "model.cif"
    path.write_text(text)
    sets, distances = shells(magnetic_model_from_file(str(path)), rmax)
    path.write_text(text.rstrip() + "\n" + declare(
        magnetic_model_from_file(str(path)), sets, distances, n_asym, ops
    ))
    model = reader(str(path))
    assert list(model.bonds) == [f"J{k + 1}" for k in range(len(sets))]
    for k, expected in enumerate(sets):
        got = {canonical(i, j, offset_of(tag)) for i, j, tag in model.bonds[f"J{k + 1}"]["bonds"]}
        assert len(model.bonds[f"J{k + 1}"]["bonds"]) == len(got)  # no duplicates
        assert got == expected, f"J{k + 1}"


def test_cell_offsets_are_rounded():
    """They were truncated: -0.9999999 gave 0 instead of -1."""
    syms = [parse_symmetry(op) for op in HEXAGONAL_OPS]
    base = np.array([1 / 3, 2 / 3, 0.1])
    positions = []
    for R, t in syms:
        p = R @ base + t
        if all(np.linalg.norm((p - q + 0.5) % 1 - 0.5) > 1e-3 for q in positions):
            positions.append(p)
    for (R, t), shift in itertools.product(syms, itertools.product((-1, 0, 1), repeat=3)):
        for p in positions:
            q = np.asarray(p) + np.array(shift)
            j, offset = find_atom_offset_by_symmetry(q, (R, t), positions)
            np.testing.assert_allclose(positions[j] + offset, R @ q + t, atol=1e-9)


def test_a_bond_to_the_image_of_a_site_is_normalized():
    """(i, i, +n) and (i, i, -n) are the same bond."""
    assert normalize_bond(2, 2, [-1, 0, 0]) == normalize_bond(2, 2, [1, 0, 0])
    assert normalize_bond(2, 2, [0, -1, 1]) == normalize_bond(2, 2, [0, 1, -1])
    assert normalize_bond(3, 1, [1, 0, 0]) == normalize_bond(1, 3, [-1, 0, 0])


@pytest.mark.parametrize("reader", READERS, ids=["hand-written", "pymatgen"])
def test_an_operator_that_is_not_in_the_file_is_reported(tmp_path, reader, caplog):
    """It was skipped silently (only the primitive-cell reader warned)."""
    path = tmp_path / "p1.cif"
    path.write_text(
        HEXAGONAL.replace("".join(f"'{op}'\n" for op in HEXAGONAL_OPS[1:]), "")
        + "loop_\n_geom_bond_atom_site_label_1\n_geom_bond_atom_site_label_2\n"
        "_geom_bond_distance\n_geom_bond_label\n_geom_bond_site_symmetry_2\n"
        "Cr1 Cr1 5.0 J1 1_655\nCr1 Cr1 8.0 J2 2_556\n"
    )
    with caplog.at_level(logging.WARNING, logger="spectrojotometer"):
        model = reader(str(path))
    assert "'2_556' refers to operator 2, and the file has 1" in caplog.text
    assert [len(model.bonds[k]["bonds"]) for k in model.bonds] == [1]
