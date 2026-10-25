"""
Cell shifts of the bonds in CIF files: save_cif writes the standard
_geom_bond_site_symmetry_2 code (n_klm, each digit the shift + 5), and
the readers also read the files written in the old convention of
spectrojotometer (each digit the shift: 0 = none, 1 = +1, 9 = -1),
detected by the declared bond lengths, with a warning.

Before, save_cif wrote the old codes and the readers took them as
standard ones: the bonds with a cell shift came back with shifts of 4 or
5 cells, or were lost.
"""

import logging
import pathlib

import numpy as np
import pytest

from spectrojotometer.magnetic_model import MagneticModel, _bond_offset, equivalent_model
from spectrojotometer.model_io import (
    magnetic_model_from_cif,
    magnetic_model_from_cif_pymatgen,
)
from spectrojotometer.model_io.cif import (
    _with_standard_symmetry_codes,
    standard_symmetry_code,
)

from .conftest import EXAMPLES_DIR

READERS = [magnetic_model_from_cif, magnetic_model_from_cif_pymatgen]
LOGGER = "spectrojotometer"


def bond_table(model):
    """Each bond as (coupling, i, j, shift, length), with i <= j (the
    readers normalize the bonds)."""
    positions = np.asarray(model.site_properties["coord_atomos"], dtype=float)
    lattice = np.asarray(model.lattice_properties["bravais_vectors"], dtype=float)
    rows = []
    for name, coupling in model.bonds.items():
        for i, j, tag in coupling["bonds"]:
            shift = _bond_offset(tag)
            if i > j or (i == j and next((n for n in shift if n), 0) < 0):
                i, j, shift = j, i, -shift
            length = np.linalg.norm(positions[j] + shift @ lattice - positions[i])
            rows.append((name, i, j, tuple(int(n) for n in shift), round(float(length), 4)))
    return sorted(rows)


def cubic():
    model = MagneticModel(
        [np.zeros(3)], [2.0 * np.eye(3)[k] for k in range(3)], magnetic_species=["Cu"]
    )
    model.generate_bonds(0.02, [[0, 2.5]])
    return model


def spinel():
    model = magnetic_model_from_cif(str(EXAMPLES_DIR / "cromita_ortogonal.cif"))
    model.generate_bonds(0.02, [[0, 6.0]])
    return model


def supercell():
    """A model whose bonds are not normalized (built by equivalent_model)."""
    lattice = [np.array([3.0, 0, 0]), np.array([0, 4.0, 0]), np.array([0, 0, 4.5])]
    cell = MagneticModel(
        [np.zeros(3), np.array([1.2, 0, 0])], lattice, magnetic_species=["Cu", "Cu"]
    )
    cell.generate_bonds(0.02, [[0, 4.6]])
    double = MagneticModel(
        [np.array([x, 0, 0]) for x in (0, 1.2, 3, 4.2)],
        [2 * lattice[0], lattice[1], lattice[2]],
        magnetic_species=["Cu"] * 4,
    )
    return equivalent_model(cell, double)


MODELS = {"cubic": cubic, "spinel": spinel, "supercell": supercell}


def bond_codes(path):
    lines = pathlib.Path(path).read_text().split("# Bonds")[1].strip().split("\n")[6:]
    return [line.split()[-1] for line in lines if line.strip()]


def test_save_cif_writes_standard_codes(tmp_path):
    cubic().save_cif(str(tmp_path / "m.cif"))
    assert bond_codes(tmp_path / "m.cif") == ["1_655", "1_565", "1_556"]


def test_save_cif_refers_to_the_identity_operator(tmp_path):
    """With the symmetries of the model, n is the identity among them."""
    model = cubic()
    model.lattice_properties["symmetries"] = [
        (-np.eye(3), np.zeros(3)),
        (np.eye(3), np.zeros(3)),
    ]
    model.save_cif(str(tmp_path / "m.cif"))
    assert bond_codes(tmp_path / "m.cif") == ["2_655", "2_565", "2_556"]


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("name", sorted(MODELS))
def test_round_trip(name, reader, tmp_path, caplog):
    model = MODELS[name]()
    assert any(_bond_offset(t).any() for c in model.bonds.values() for _, _, t in c["bonds"])
    model.save_cif(str(tmp_path / "m.cif"))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        back = reader(str(tmp_path / "m.cif"))
    assert bond_table(back) == bond_table(model)
    assert "old convention" not in caplog.text


def write_old_convention(model, path):
    """The CIF that save_cif wrote before: the same, with the old codes
    (the conversion is its own inverse)."""
    model.save_cif(str(path))
    head, bonds = path.read_text().split("# Bonds")
    lines = bonds.split("\n")
    for k, line in enumerate(lines):
        fields = line.split("\t")
        if len(fields) == 5:
            fields[-1] = standard_symmetry_code(fields[-1])
            lines[k] = "\t".join(fields)
    path.write_text(head + "# Bonds" + "\n".join(lines))


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("name", sorted(MODELS))
def test_files_in_the_old_convention_are_read(name, reader, tmp_path, caplog):
    model = MODELS[name]()
    write_old_convention(model, tmp_path / "old.cif")
    assert "1_100" in (tmp_path / "old.cif").read_text() or "1_900" in (
        tmp_path / "old.cif"
    ).read_text()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        back = reader(str(tmp_path / "old.cif"))
    assert bond_table(back) == bond_table(model)
    assert "old convention" in caplog.text
    # Saved again, in the standard convention.
    caplog.clear()
    back.save_cif(str(tmp_path / "new.cif"))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        again = reader(str(tmp_path / "new.cif"))
    assert bond_table(again) == bond_table(model)
    assert "old convention" not in caplog.text


def test_standard_symmetry_code():
    assert standard_symmetry_code("1_100") == "1_655"
    assert standard_symmetry_code("1_909") == "1_454"
    assert standard_symmetry_code("1_655") == "1_100"  # its own inverse
    assert standard_symmetry_code("2_000") == "2_555"
    assert standard_symmetry_code(".") == "."
    assert standard_symmetry_code("3") == "3"


LABELS = [
    "_geom_bond_atom_site_label_1",
    "_geom_bond_atom_site_label_2",
    "_geom_bond_distance",
    "_geom_bond_label",
    "_geom_bond_site_symmetry_2",
]


def test_without_geometry_the_size_of_the_shifts_decides(caplog):
    old = [["A", "B", "1.0", "J1", "1_100"], ["A", "B", "1.0", "J1", "1_090"]]
    standard = [["A", "B", "1.0", "J1", "1_655"], ["A", "B", "1.0", "J1", "2_555"]]
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _with_standard_symmetry_codes(LABELS, standard) is standard
        assert "old convention" not in caplog.text
        converted = _with_standard_symmetry_codes(LABELS, old)
    assert [row[-1] for row in converted] == ["1_655", "1_545"]
    assert "old convention" in caplog.text
    # Without cell shifts there is nothing to decide.
    plain = [["A", "B", "1.0", "J1", "."]]
    assert _with_standard_symmetry_codes(LABELS, plain) is plain


def test_the_declared_lengths_decide_over_the_size_of_the_shifts():
    """A code whose standard reading has the larger shifts (1_900: +4, -5,
    -5 cells; as an old code, -1 along a), but the declared length is
    that of the standard reading: it is kept."""
    rows = [["A", "A", "8.0", "J1", "1_900"]]

    def length(row):
        return 8.0 if row[-1] == "1_900" else 2.0

    assert _with_standard_symmetry_codes(LABELS, rows, length) is rows
    # Without lengths, the smaller shifts would win.
    assert _with_standard_symmetry_codes(LABELS, rows)[0][-1] == "1_455"
